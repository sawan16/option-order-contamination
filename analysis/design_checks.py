"""Model-free checks of assumption (D) (Table 3 of the paper)."""
import os, json, numpy as np
from scipy import stats
from common import ITEMS, OUT, num
it = json.load(open(ITEMS)); rows = []; res = {}
for b, name in [("mmlu", "MMLU"), ("arc", "ARC-Challenge"), ("obqa", "OpenBookQA")]:
    ns = nt = ls = lt = 0; pos = []
    for x in it[b]:
        v = [num(o) for o in x["opts"]]
        if all(y is not None for y in v) and len(set(v)) == 4:
            nt += 1; ns += v == sorted(v) or v == sorted(v, reverse=True)
        L = [len(str(o)) for o in x["opts"]]
        if len(set(L)) == 4:
            lt += 1; ls += L == sorted(L) or L == sorted(L, reverse=True)
        pos.append(x["ans"])
    pn = stats.binomtest(ns, nt, 1 / 12, alternative="greater").pvalue if nt else 1.0
    pl = stats.binomtest(ls, lt, 1 / 12, alternative="greater").pvalue
    pc = stats.chisquare(np.bincount(pos, minlength=4)).pvalue
    res[b] = {"numeric_sorted": [ns, nt, pn], "lengths_monotone": [ls, lt, pl], "answer_slot_chi2_p": pc}
    print(f"{name}: numeric sorted {ns}/{nt} p={pn:.2g} | lengths monotone {ls}/{lt}={ls/lt:.3f} p={pl:.2g} | answer-slot chi2 p={pc:.2g}")
json.dump(res, open(os.path.join(OUT, "tables", "design_checks.json"), "w"), indent=1)
