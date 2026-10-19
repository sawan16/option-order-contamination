"""Staged experiments. Usage: python run.py <stage> <out_dir> [--fake]
Stages: audit1, audit2 (public models on public benchmarks, published orders)
        power_mmlu, power_arc (simulated contamination by fine-tuning; options re-shuffled by us)
Every unit of work is cached as one .npz in out_dir, so a stopped run resumes where it left off."""
import sys, os, json, time, gc, traceback
import numpy as np, torch
import core

STAGE, OUT = sys.argv[1], sys.argv[2]; FAKE = "--fake" in sys.argv
os.makedirs(OUT, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
log = lambda *a: print(*a, flush=True)

AUDIT_BENCH = {"mmlu": (500, None), "arc": (500, None), "obqa": (500, None), "mmlupro": (300, 29), "supergpqa": (300, 29)}
STAGES = {
    "audit1": {"models": ["EleutherAI/pythia-410m-deduped", "EleutherAI/pythia-1.4b-deduped", "HuggingFaceTB/SmolLM2-360M",
                          "Qwen/Qwen2.5-0.5B", "Qwen/Qwen3-0.6B-Base", "TinyLlama/TinyLlama_v1.1"]},
    "audit2": {"models": ["EleutherAI/pythia-2.8b-deduped", "HuggingFaceTB/SmolLM2-1.7B", "Qwen/Qwen2.5-1.5B",
                          "Qwen/Qwen3-1.7B-Base", "allenai/OLMo-2-0425-1B"]},
    "power_mmlu": {"bench": "mmlu", "models": ["HuggingFaceTB/SmolLM2-360M", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen3-0.6B-Base"]},
    "power_arc": {"bench": "arc", "models": ["HuggingFaceTB/SmolLM2-360M", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen3-0.6B-Base"]},
}
POWER_ITEMS = 600  # first half fine-tuned on (contaminated), second half held out (clean)
POWER_CONFIGS = [("base", 0), ("full", 1), ("full", 3), ("answer", 1), ("answer", 3), ("answer", 5)]
LR, BS_TRAIN = 2e-5, 4
if FAKE:
    AUDIT_BENCH = {"mmlu": (20, None), "mmlupro": (10, 5)}
    STAGES = {k: {**v, "models": ["fake"]} for k, v in STAGES.items()}
    POWER_ITEMS, POWER_CONFIGS, LR = 40, [("base", 0), ("full", 2), ("answer", 2)], 1e-3


# ---------------------------------------------------------------- data (published order kept unless shuffle=True)
def load_bench(name, n, seed=0, shuffle=False):
    if FAKE:
        from fake import fake_items
        return fake_items(n, 10 if name in ("mmlupro", "supergpqa") else 4, seed)
    from datasets import load_dataset
    rows = []
    if name == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split="test")
        rows = [(r["question"], list(r["choices"]), int(r["answer"])) for r in ds]
    elif name in ("arc", "obqa"):
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test") if name == "arc" else load_dataset("allenai/openbookqa", "main", split="test")
        for r in ds:
            labs, txt = list(r["choices"]["label"]), list(r["choices"]["text"])
            if len(txt) == 4 and r["answerKey"] in labs:
                rows.append((r["question"] if name == "arc" else r["question_stem"], txt, labs.index(r["answerKey"])))
    elif name == "mmlupro":
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        rows = [(r["question"], list(r["options"]), int(r["answer_index"])) for r in ds if len(r["options"]) == 10]
    elif name == "supergpqa":
        ds = load_dataset("m-a-p/SuperGPQA", split="train")
        for r in ds:
            opts = list(r["options"])
            if r.get("answer_letter") and len(opts) >= 4:
                a = "ABCDEFGHIJ".index(str(r["answer_letter"]).strip()[0])
            elif r.get("answer") in opts:
                a = opts.index(r["answer"])
            else:
                continue
            if len(opts) >= 4:
                rows.append((r["question"], opts, a))
    rng = np.random.default_rng(seed)
    pick = rng.permutation(len(rows))[:n]
    items = []
    for i in pick:
        q, o, a = rows[int(i)]
        if shuffle:
            p = rng.permutation(len(o)); o = [o[j] for j in p]; a = int(list(p).index(a))
        items.append({"q": q, "opts": o, "ans": a})
    return items


def load_model(name, dtype):
    if FAKE:
        from fake import fake_model
        return fake_model()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(dev).eval()
    return m, tok


def budget(model):
    """Token budget per scoring batch so that fp32 log-sum-exp over the vocabulary stays near 1.2 GB."""
    V = model.get_output_embeddings().weight.shape[0]
    return int(max(256, min(8192, 3.0e8 / V)))


def save_scores(path, S, items, extra=None):
    np.savez(path, scores=np.array(S["scores"]), full_idx=np.array(S["full_idx"]), anch_idx=np.array(S["anch_idx"]),
             ans=np.array([it["ans"] for it in items]), nopt=np.array([len(it["opts"]) for it in items]),
             M=-1 if S["M"] is None else S["M"], **(extra or {}))


def score_safe(m, tok, items, M, seed=1):
    S = core.score_bench(m, tok, items, dev, np.random.default_rng(seed), M=M, token_budget=budget(m), log=log)
    if not np.isfinite(np.array(S["scores"])).all():
        log("    non-finite fp16 scores -> rescoring in fp32"); m = m.float()
        S = core.score_bench(m, tok, items, dev, np.random.default_rng(seed), M=M, token_budget=budget(m) // 2, log=log)
    return S, m


def free():
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- audit stages
def run_audit(cfg):
    bench_items = {}
    for b, (n, M) in AUDIT_BENCH.items():
        try:
            bench_items[b] = load_bench(b, n); log(f"{b}: {len(bench_items[b])} items")
            json.dump(bench_items[b], open(f"{OUT}/items_{b}.json", "w"))
        except Exception as e:
            log(f"SKIPPING benchmark {b}: could not load ({e!r})")
    for name in cfg["models"]:
        short = name.split("/")[-1]
        todo = [b for b in bench_items if not os.path.exists(f"{OUT}/audit__{short}__{b}.npz")]
        if not todo:
            log("done:", short); continue
        try:
            m, tok = load_model(name, torch.float16 if dev == "cuda" else torch.float32)
        except Exception as e:
            log("FAILED to load", name, repr(e)); continue
        for b in todo:
            log(f"audit {short} / {b}")
            try:
                S, m = score_safe(m, tok, bench_items[b], AUDIT_BENCH[b][1])
                save_scores(f"{OUT}/audit__{short}__{b}.npz", S, bench_items[b])
            except Exception:
                log(traceback.format_exc())
            free()
        del m, tok; free()


# ---------------------------------------------------------------- power stages
def run_power(cfg):
    items = load_bench(cfg["bench"], POWER_ITEMS, seed=0, shuffle=True)
    half = len(items) // 2
    log(f"{cfg['bench']}: {len(items)} items ({half} contaminated, {len(items)-half} clean), options re-shuffled")
    for name in cfg["models"]:
        short = name.split("/")[-1]
        for mode, ep in POWER_CONFIGS:
            path = f"{OUT}/power__{short}__{cfg['bench']}__{mode}{ep}.npz"
            if os.path.exists(path):
                continue
            log(f"power {short} / {mode} x{ep}")
            try:
                if mode == "base":
                    m, tok = load_model(name, torch.float16 if dev == "cuda" else torch.float32)
                else:
                    m, tok = load_model(name, torch.float32)
                    t = time.time()
                    m = core.finetune(m, tok, items[:half], dev, mode=mode, epochs=ep, lr=LR, bs=BS_TRAIN, log=log)
                    log(f"    fine-tuned in {time.time()-t:.0f}s")
                    if dev == "cuda":
                        m = m.half()
                S, m = score_safe(m, tok, items, None)
                texts = ["".join(core.fmt(it["q"], it["opts"], it["ans"])) for it in items]
                mk = core.mink_pp(m, tok, texts, dev, bs=1 if not FAKE else 4)
                save_scores(path, S, items, {"mink": mk, "half": half})
            except Exception:
                log(traceback.format_exc())
            try:
                del m, tok
            except NameError:
                pass
            free()


# ---------------------------------------------------------------- quick summary of whatever is finished
def summarize():
    rng = np.random.default_rng(7); summ = {}
    for f in sorted(os.listdir(OUT)):
        if not f.endswith(".npz"):
            continue
        z = np.load(f"{OUT}/{f}", allow_pickle=True)
        items = [{"opts": list(range(int(n))), "ans": int(a)} for a, n in zip(z["ans"], z["nopt"])]
        M = None if int(z["M"]) < 0 else int(z["M"])
        S = {"scores": list(z["scores"]), "full_idx": list(z["full_idx"]), "anch_idx": list(z["anch_idx"])}
        t = core.tests_from_scores(S, items, M, rng)
        if f.startswith("audit"):
            summ[f[:-4]] = core.set_tests(t)
        else:
            h = int(z["half"]); row = {}
            for k, d in t.items():
                if not d:
                    continue
                rc = d["r"][:h] if d.get("r") is not None else None; r0 = d["r"][h:] if d.get("r") is not None else None
                row[k] = {"clean": core.set_tests({k: {**d, "u": d["u"][h:], "r": r0}})[k],
                          "contaminated": core.set_tests({k: {**d, "u": d["u"][:h], "r": rc}})[k],
                          "power": core.power_split(d["u"][:h], d["u"][h:], Ns=(25, 50, 100) if not FAKE else (10,), B=1000,
                                                    mk_c=z["mink"][:h], mk_0=z["mink"][h:], r_c=rc, r_0=r0, K=d.get("K"))}
            from sklearn.metrics import roc_auc_score
            y = np.r_[np.ones(h), np.zeros(len(z["mink"]) - h)]
            row["auroc"] = {"minkpp": float(roc_auc_score(y, z["mink"]))}
            for k in ["order", "slot", "anch"]:
                if t.get(k):
                    row["auroc"][k] = float(roc_auc_score(y, -t[k]["u"]))
            summ[f[:-4]] = row
    json.dump(summ, open(f"{OUT}/summary_{STAGE}.json", "w"), indent=1, default=float)
    for k, v in summ.items():
        if k.startswith("audit"):
            log(f"{k:55s} " + " | ".join(f"{t}: mean_u {d['mean_u']:.3f} p_fisher {d['p_fisher']:.2g}" for t, d in v.items()))
        else:
            log(f"{k:55s} AUROC {v['auroc']} | " + " | ".join(
                f"{t}: power@{list(v[t]['power'])[-1]} {v[t]['power'][list(v[t]['power'])[-1]]['contaminated']}" for t in ["order", "slot", "anch"] if t in v))


if __name__ == "__main__":
    t0 = time.time()
    (run_audit if STAGE.startswith("audit") else run_power)(STAGES[STAGE])
    summarize()
    log(f"stage {STAGE} finished in {(time.time()-t0)/60:.0f} min")
