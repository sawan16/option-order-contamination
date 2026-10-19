"""Table: audit p-values and leak rates (Table 2 of the paper)."""
import os, numpy as np
from common import audit_summary, MODELS, BENCH, OUT
S = audit_summary()
def fp(p):
    if p < 1e-3:
        e = int(np.floor(np.log10(p))); s = f"${p/10**e:.0f}{{\\times}}10^{{{e}}}$"
    else:
        s = f"{p:.2f}" if p >= 0.01 else f"{p:.3f}"
    return f"\\textbf{{{s}}}" if p < 0.05 / 55 else s
rows = [f"{name} & {yr} & " + " & ".join(f"{fp(S[f'audit__{k}__{b}']['order']['p_fisher'])} ({S[f'audit__{k}__{b}']['order']['top1_rate']:.2f})" for b, _ in BENCH) + "\\\\"
        for k, name, yr in MODELS]
tab = ("\\begin{tabular}{llccccc}\n\\toprule\nModel & Released & MMLU & ARC-C & OBQA & MMLU-Pro & SuperGPQA\\\\\n\\midrule\n" + "\n".join(rows)
       + "\n\\midrule\n & & " + " & ".join(["(chance 0.04)"] * 3 + ["(chance 0.03)"] * 2) + "\\\\\n\\bottomrule\n\\end{tabular}\n")
open(os.path.join(OUT, "tables", "audit_table.tex"), "w").write(tab)
for t in ["order", "anch", "slot"]:
    print(t, "models rejecting (Bonferroni over 55):",
          {b: sum(S[f'audit__{k}__{b}'][t]['p_fisher'] < 0.05 / 55 for k, _, _ in MODELS) for b, _ in BENCH if t in S[f'audit__pythia-410m-deduped__{b}']})
