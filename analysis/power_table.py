"""Power table (Table 4 of the paper)."""
import os, json
from common import power_summary, OUT
out = os.path.join(OUT, "tables", "power_table.tex")
S = {**power_summary("mmlu"), **power_summary("arc")}
MOD = [("SmolLM2-360M", "SmolLM2-360M"), ("Qwen2.5-0.5B", "Qwen2.5-0.5B"), ("Qwen3-0.6B-Base", "Qwen3-0.6B")]
CFG = [("base0", "none"), ("full1", "all tokens, 1 ep."), ("full3", "all tokens, 3 ep."), ("answer1", "answer only, 1 ep."), ("answer3", "answer only, 3 ep."), ("answer5", "answer only, 5 ep.")]
BEN = [b for b in ["mmlu", "arc"] if any(f"__{b}__" in k for k in S)]
def f2(x): return f"{x:.2f}"
rows = []
for b in BEN:
    rows.append(f"\\multicolumn{{9}}{{l}}{{\\textit{{{ {'mmlu':'MMLU','arc':'ARC-Challenge'}[b] }}}}}\\\\")
    for mk, mname in MOD:
        first = True
        for ck, cname in CFG:
            k = f"power__{mk}__{b}__{ck}"
            if k not in S: continue
            v = S[k]; N = "50"
            pw = lambda t, s: v[t]["power"][N]["contaminated"][s]
            cells = [f2(pw("order", "top1")), f2(pw("order", "fisher")), f2(pw("slot", "fisher")), f2(pw("anch", "fisher")), f2(pw("order", "minkpp_ref")), f2(v["auroc"]["minkpp"])]
            rows.append(f"{mname if first else ''} & {cname} & " + " & ".join(cells) + f" & {v['slot']['clean']['p_fisher']:.2f}\\\\")
            first = False
        rows.append("\\addlinespace")
tab = ("\\begin{tabular}{llccccccc}\n\\toprule\n & & \\multicolumn{5}{c}{Power at $N=50$ items ($\\alpha=0.05$)} & & Clean\\\\\n\\cmidrule(lr){3-7}\n"
       "Model & Exposure & Top-1~\\cite{ni2025training} & Order & Slot & Within-slot & Min-K\\%++$^\\dagger$ & AUROC Min-K\\%++ & $p_{\\text{slot}}$\\\\\n\\midrule\n"
       + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")
open(out, "w").write(tab); print("ok", len(rows))
