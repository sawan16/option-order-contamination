"""Exact permutation tests for multiple-choice benchmark contamination (paper version).

For an item with n options and published order identity, every permutation p (p[j] = published index shown at slot j)
is scored by the total log-likelihood of the formatted item including the answer letter.
Three exact sub-tests (published order uniform on S_n, model independent of it):
  * order  : rank of the published order among all permutations (or identity + M random ones)      K = n! or M+1
  * anch   : rank among permutations that keep the correct answer in its published slot            K = (n-1)! or M+1
  * slot   : rank of the published answer slot among the n slots, each slot scored by the logsumexp
             over all permutations placing the correct answer there (exact enumeration only)         K = n
Under the null, slot and anch ranks are independent, so their Fisher combination is also exact.
"""
import itertools, math, random, time
import numpy as np
import torch
from scipy import stats, special

LET = "ABCDEFGHIJKLMNOP"


def fmt(q, opts, ans_pos):
    s = f"Question: {q.strip()}\n" + "".join(f"{LET[i]}. {str(o).strip()}\n" for i, o in enumerate(opts))
    return s + "Answer:", f" {LET[ans_pos]}"


def perm_plan(n, ans, rng, M=None):
    """Permutations to score and index lists for the three tests. Identity is index 0."""
    ident = tuple(range(n))
    if M is None:  # exact enumeration
        perms = [ident] + [p for p in itertools.permutations(range(n)) if p != ident]
        anch = [i for i, p in enumerate(perms) if p[ans] == ans]
        slots = [[i for i, p in enumerate(perms) if p.index(ans) == s] for s in range(n)]
        return perms, list(range(len(perms))), anch, slots
    perms = [ident] + [tuple(int(x) for x in rng.permutation(n)) for _ in range(M)]
    others = [i for i in range(n) if i != ans]
    anch = [0]
    for _ in range(M):
        o = list(rng.permutation(others)); p = [ans if j == ans else int(o.pop()) for j in range(n)]
        perms.append(tuple(p)); anch.append(len(perms) - 1)
    return perms, list(range(M + 1)), anch, None


def item_texts(item, perms):
    q, opts, a = item["q"], item["opts"], item["ans"]
    out = []
    for p in perms:
        pr, an = fmt(q, [opts[i] for i in p], p.index(a))
        out.append(pr + an)
    return out


@torch.no_grad()
def seq_logprob(model, tok, texts, dev, token_budget=2048, maxlen=1024):
    """Total log-probability of each text; dynamic batches of about token_budget tokens, sorted by length."""
    lens = [len(t) for t in texts]
    order = np.argsort(lens)[::-1]          # longest first: the first batch shows any memory problem at once
    res = np.zeros(len(texts)); i = 0
    while i < len(order):
        est = lens[order[i]] // 3 + 8       # conservative token estimate (about 3 characters per token)
        idx = order[i:i + max(1, token_budget // est)]
        enc = tok([texts[k] for k in idx], return_tensors="pt", padding=True, truncation=True, max_length=maxlen).to(dev)
        logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits[:, :-1]
        tgt = enc["input_ids"][:, 1:]
        lse = torch.logsumexp(logits.float(), -1)
        tl = logits.gather(-1, tgt[..., None])[..., 0].float()
        res[idx] = ((tl - lse) * enc["attention_mask"][:, 1:].float()).sum(1).cpu().numpy()
        del logits, lse, tl
        i += len(idx)
    return res


@torch.no_grad()
def mink_pp(model, tok, texts, dev, k=0.2, bs=2, maxlen=1024):
    """Min-K%++ (Zhang et al. 2024): mean of the lowest k fraction of standardized token log-probs. Higher = more member-like."""
    out = np.zeros(len(texts))
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=maxlen).to(dev)
        logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits[:, :-1].float()
        lsm = torch.log_softmax(logits, -1); del logits
        p = lsm.exp()
        mu = (p * lsm).sum(-1); sig = ((p * lsm * lsm).sum(-1) - mu ** 2).clamp_min(1e-8).sqrt(); del p
        tl = lsm.gather(-1, enc["input_ids"][:, 1:, None])[..., 0]; del lsm
        z = ((tl - mu) / sig).cpu().numpy(); m = enc["attention_mask"][:, 1:].cpu().numpy().astype(bool)
        for b in range(z.shape[0]):
            v = np.sort(z[b][m[b]]); n = max(1, int(len(v) * k)); out[i + b] = v[:n].mean()
    return out


def rank_u(scores_pub, scores_other, rng):
    """Randomized rank of the published score among {published} ∪ others. Returns (u in (0,1), integer rank, K)."""
    gt = (scores_other > scores_pub[:, None] + 1e-9).sum(1)
    tie = (np.abs(scores_other - scores_pub[:, None]) <= 1e-9).sum(1)
    r = gt + np.floor(rng.random(len(gt)) * (tie + 1)).astype(int) + 1
    K = scores_other.shape[1] + 1
    return (r - rng.random(len(gt))) / K, r, K


def score_bench(model, tok, items, dev, rng, M=None, token_budget=2048, log=print):
    """Score all permutations; return per-item dicts of u and r for order / anch / slot tests plus raw scores."""
    plans, texts, spans = [], [], []
    for it in items:
        perms, full, anch, slots = perm_plan(len(it["opts"]), it["ans"], rng, M)
        plans.append((perms, full, anch))
        spans.append((len(texts), len(texts) + len(perms)))
        texts += item_texts(it, perms)
    t = time.time()
    lp = seq_logprob(model, tok, texts, dev, token_budget)
    log(f"    {len(texts)} sequences in {time.time()-t:.0f}s")
    out = {"scores": [], "full_idx": [], "anch_idx": [], "M": M}
    for (perms, full, anch), (a, b) in zip(plans, spans):
        out["scores"].append(lp[a:b]); out["full_idx"].append(full); out["anch_idx"].append(anch)
    return out


def tests_from_scores(S, items, M, rng):
    """Compute u for order, anch, slot, and Fisher(slot, anch) from saved scores."""
    res = {k: [] for k in ["order", "anch", "slot", "slot_anch"]}
    Ks = {}
    order_pub, order_oth, anch_pub, anch_oth, slot_pub, slot_oth = [], [], [], [], [], []
    for s, it, fi, ai in zip(S["scores"], items, S["full_idx"], S["anch_idx"]):
        order_pub.append(s[0]); order_oth.append(s[[i for i in fi if i != 0]])
        anch_pub.append(s[0]); anch_oth.append(s[[i for i in ai if i != 0]])
        if M is None:
            n, a = len(it["opts"]), it["ans"]
            perms = [tuple(range(n))] + [p for p in itertools.permutations(range(n)) if p != tuple(range(n))]
            agg = np.array([special.logsumexp(s[[i for i, p in enumerate(perms) if p.index(a) == sl]]) for sl in range(n)])
            slot_pub.append(agg[a]); slot_oth.append(np.delete(agg, a))
    for name, pub, oth in [("order", order_pub, order_oth), ("anch", anch_pub, anch_oth), ("slot", slot_pub, slot_oth)]:
        if not pub:
            continue
        u, r, K = rank_u(np.array(pub), np.array(oth), rng)
        res[name] = {"u": u, "r": r, "K": K}
    if M is None:
        f = -2 * (np.log(res["slot"]["u"]) + np.log(res["anch"]["u"]))
        res["slot_anch"] = {"u": stats.chi2.sf(f, 4), "r": None, "K": None}  # per-item exact p-value, uniform under H0
    else:
        res.pop("slot"); res.pop("slot_anch")
    return res


# ---------------------------------------------------------------- set-level exact p-values
def p_meanrank_exact(r, K):
    N = len(r); conv = np.array([1.0]); pmf = np.ones(K) / K
    for _ in range(N):
        conv = np.convolve(conv, pmf)
    return float(conv[: int(r.sum()) - N + 1].sum())


def p_fisher(u):
    u = np.clip(u, 1e-300, 1)
    return float(stats.chi2.sf(-2 * np.log(u).sum(), 2 * len(u)))


def p_top1(r, K):
    return float(stats.binom.sf(np.sum(r == 1) - 1, len(r), 1.0 / K))


def set_tests(t):
    """Exact set-level p-values for one benchmark (all items) and each sub-test."""
    out = {}
    for name, d in t.items():
        if not d:
            continue
        u = d["u"]
        row = {"n": int(len(u)), "mean_u": float(np.mean(u)), "p_fisher": p_fisher(u),
               "p_ks_low": float(stats.kstest(u, "uniform", alternative="greater").pvalue)}
        if d.get("r") is not None:
            row.update({"K": int(d["K"]), "top1_rate": float(np.mean(d["r"] == 1)), "p_top1": p_top1(d["r"], d["K"]),
                        "p_meanrank": float(stats.norm.cdf((d["r"].mean() - (d["K"] + 1) / 2) / math.sqrt((d["K"] ** 2 - 1) / 12 / len(u))))})
        out[name] = row
    return out


def power_split(u_c, u_0, Ns=(25, 50, 100, 200), B=2000, seed=0, alpha=0.05, mk_c=None, mk_0=None, r_c=None, r_0=None, K=None):
    """Power on contaminated items, size on clean items, from disjoint resampling (with replacement) of each pool.
    Min-K%++ baseline uses the clean pool as its reference (an advantage the exact tests do not need)."""
    rng = np.random.default_rng(seed); out = {}
    for N in Ns:
        row = {"contaminated": {}, "clean": {}}
        for lab, uu, rr in [("contaminated", u_c, r_c), ("clean", u_0, r_0)]:
            ix = rng.integers(0, len(uu), (B, N))
            row[lab]["fisher"] = float(np.mean([p_fisher(uu[i]) < alpha for i in ix]))
            if rr is not None:
                row[lab]["top1"] = float(np.mean([p_top1(rr[i], K) < alpha for i in ix]))
        if mk_c is not None:
            ref = mk_0[rng.integers(0, len(mk_0), (B, N))].mean(1); thr = np.quantile(ref, 1 - alpha)
            row["contaminated"]["minkpp_ref"] = float(np.mean(mk_c[rng.integers(0, len(mk_c), (B, N))].mean(1) > thr))
        out[str(N)] = row
    return out


# ---------------------------------------------------------------- contamination simulation
def finetune(model, tok, items, dev, mode="full", epochs=1, lr=2e-5, bs=4, seed=0, log=print):
    """mode='full': loss on all tokens (pretraining-style). mode='answer': loss on the answer only (instruction-tuning style)."""
    torch.manual_seed(seed); rnd = random.Random(seed)
    model.train(); model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    data = [fmt(it["q"], it["opts"], it["ans"]) for it in items]
    for ep in range(epochs):
        rnd.shuffle(data)
        for i in range(0, len(data), bs):
            batch = data[i:i + bs]
            enc = tok([p + a for p, a in batch], return_tensors="pt", padding=True, truncation=True, max_length=1024).to(dev)
            labels = enc["input_ids"].clone(); labels[enc["attention_mask"] == 0] = -100
            if mode == "answer":
                for b, (p, a) in enumerate(batch):
                    labels[b, :len(tok(p)["input_ids"])] = -100
            loss = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels).loss
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
        log(f"    epoch {ep+1}: last loss {loss.item():.3f}")
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    del opt
    return model
