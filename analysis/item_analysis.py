"""Which option structures drive the published-order preference (Fig. 6)."""
import json, re, os
import numpy as np
from common import core, ITEMS, AUDIT, OUT
items = json.load(open(ITEMS)); dirs = [AUDIT]; out = os.path.join(OUT, "tables", "item_table.tex")
MODELS = ["pythia-410m-deduped","pythia-1.4b-deduped","pythia-2.8b-deduped","TinyLlama_v1.1","SmolLM2-360M","SmolLM2-1.7B",
          "Qwen2.5-0.5B","Qwen2.5-1.5B","OLMo-2-0425-1B","Qwen3-0.6B-Base","Qwen3-1.7B-Base"]
def num(o):
    m = re.fullmatch(r"\s*[-+$]?\s*([-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?)\s*(%|[a-zA-Z/^\s\d]*)?\s*", str(o))
    if not m: return None
    try: return float(m.group(1).replace(",", ""))
    except ValueError: return None
def feats(opts):
    f = {}
    v = [num(o) for o in opts]
    f["numeric, sorted"] = all(x is not None for x in v) and (v == sorted(v) or v == sorted(v, reverse=True))
    f["numeric, unsorted"] = all(x is not None for x in v) and not f["numeric, sorted"]
    low = [str(o).strip().lower() for o in opts]
    f["fixed template (e.g. True/False, Wrong/Not wrong pairs)"] = all(re.fullmatch(r"(true|false|wrong|not wrong)(,\s*(true|false|wrong|not wrong))*", x) for x in low)
    f["roman-numeral combinations (I only, II and III, ...)"] = sum(bool(re.search(r"\b(i|ii|iii|iv)\b", x)) for x in low) >= 3 and not f["fixed template (e.g. True/False, Wrong/Not wrong pairs)"]
    f["'all/none/both of the above' option"] = any(re.search(r"(all|none|both|neither) of (the )?(above|these)", x) for x in low)
    L = [len(x) for x in low]
    f["lengths monotone"] = (L == sorted(L) or L == sorted(L, reverse=True)) and len(set(L)) == len(L)
    return f
res = {}
for b in ["mmlu", "arc", "obqa"]:
    U = []
    for m in MODELS:
        p = [f"{d}/audit__{m}__{b}.npz" for d in dirs if os.path.exists(f"{d}/audit__{m}__{b}.npz")][0]
        z = np.load(p); its = [{"opts": list(range(int(n))), "ans": int(a)} for a, n in zip(z["ans"], z["nopt"])]
        t = core.tests_from_scores({"scores": list(z["scores"]), "full_idx": list(z["full_idx"]), "anch_idx": list(z["anch_idx"])}, its, None, np.random.default_rng(0))
        U.append(t["order"]["u"])
    U = np.array(U)  # models x items
    F = [feats(x["opts"]) for x in items[b]]
    names = list(F[0])
    any_struct = np.array([any(f[k] for k in names if k not in ("numeric, unsorted",)) for f in F])
    rows = {}
    for k in names + ["any structured option set", "none of these structures"]:
        mask = any_struct if k == "any structured option set" else (~any_struct if k == "none of these structures" else np.array([f[k] for f in F]))
        if mask.sum() == 0: rows[k] = None; continue
        u = U[:, mask]
        # per-model exact Fisher p on the subset, then report how many models reject at 0.05/11
        ps = [core.p_fisher(uu) for uu in u]
        rows[k] = {"n": int(mask.sum()), "share": float(mask.mean()), "mean_u": float(u.mean()), "top1": float((u < 1 / 24).mean()),
                   "models_reject": int(sum(p < 0.05 / 11 for p in ps)), "median_p": float(np.median(ps))}
    res[b] = rows
json.dump(res, open(out.replace(".tex", ".json"), "w"), indent=1)
for b, rows in res.items():
    print("\n==", b)
    for k, r in rows.items():
        if r: print(f"  {k:60s} n={r['n']:3d} ({r['share']:.0%}) mean_u {r['mean_u']:.3f} top1 {r['top1']:.3f} models rejecting {r['models_reject']}/11 median p {r['median_p']:.2g}")
