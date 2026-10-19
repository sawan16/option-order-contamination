"""All data figures of the paper (written to outputs/figures)."""
import sys, os, json, re, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from common import core, AUDIT, POWER, ITEMS, OUT as _OUT
OUT = os.path.join(_OUT, "figures")
AUD = [AUDIT]
PW = POWER
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C = {"slot": "#2a78d6", "order": "#eb6834", "anch": "#1baf7a", "top1": "#eda100", "mink": "#e87ba4"}
MK = {"slot": "o", "order": "s", "anch": "^", "top1": "D", "mink": "v"}
LS = {"slot": "-", "order": "--", "anch": "-.", "top1": ":", "mink": (0, (5, 1, 1, 1))}
LAB = {"slot": "Answer slot (ours)", "order": "Order (ours)", "anch": "Within-slot (ours)", "top1": "Top-1 rule (Ni et al.)", "mink": "Min-K%++ (clean ref.)"}
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42, "lines.linewidth": 1.6, "lines.markersize": 4.5,
                     "figure.facecolor": "white", "axes.facecolor": "white"})
MODELS = [("pythia-410m-deduped", "Pythia-410M"), ("pythia-1.4b-deduped", "Pythia-1.4B"), ("pythia-2.8b-deduped", "Pythia-2.8B"),
          ("TinyLlama_v1.1", "TinyLlama-1.1B"), ("SmolLM2-360M", "SmolLM2-360M"), ("SmolLM2-1.7B", "SmolLM2-1.7B"),
          ("Qwen2.5-0.5B", "Qwen2.5-0.5B"), ("Qwen2.5-1.5B", "Qwen2.5-1.5B"), ("OLMo-2-0425-1B", "OLMo-2-1B"),
          ("Qwen3-0.6B-Base", "Qwen3-0.6B"), ("Qwen3-1.7B-Base", "Qwen3-1.7B")]
BEN = [("mmlu", "MMLU"), ("arc", "ARC-C"), ("obqa", "OBQA"), ("mmlupro", "MMLU-Pro"), ("supergpqa", "SuperGPQA")]
PM = [("SmolLM2-360M", "SmolLM2-360M"), ("Qwen2.5-0.5B", "Qwen2.5-0.5B"), ("Qwen3-0.6B-Base", "Qwen3-0.6B")]


def tests(path):
    z = np.load(path)
    its = [{"opts": list(range(int(n))), "ans": int(a)} for a, n in zip(z["ans"], z["nopt"])]
    M = None if int(z["M"]) < 0 else int(z["M"])
    t = core.tests_from_scores({"scores": list(z["scores"]), "full_idx": list(z["full_idx"]), "anch_idx": list(z["anch_idx"])}, its, M, np.random.default_rng(0))
    return t, z


def aud_path(m, b):
    return [p for d in AUD for p in [f"{d}/audit__{m}__{b}.npz"] if os.path.exists(p)][0]


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/{name}.png", dpi=200, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- Fig: audit heatmap (-log10 p, order test)
S = {}
for f in [f"{AUDIT}/summary_audit1.json", f"{AUDIT}/summary_audit2.json"]:
    S.update(json.load(open(f)))
P = np.array([[S[f"audit__{m}__{b}"]["order"]["p_fisher"] for b, _ in BEN] for m, _ in MODELS])
L = -np.log10(np.clip(P, 1e-300, 1))
fig, ax = plt.subplots(figsize=(3.4, 3.3))
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("blues", ["#f4f7fc", "#a9c8ef", "#2a78d6", "#123f78"])
im = ax.imshow(np.minimum(L, 40), cmap=cmap, vmin=0, vmax=40, aspect="auto")
thr = -np.log10(0.05 / 55)
for i in range(L.shape[0]):
    for j in range(L.shape[1]):
        txt = f"{L[i, j]:.0f}" if L[i, j] >= 1 else f"{L[i, j]:.1f}"
        ax.text(j, i, txt + ("*" if L[i, j] > thr else ""), ha="center", va="center", fontsize=6.5, color="white" if L[i, j] > 20 else INK)
ax.set_xticks(range(len(BEN))); ax.set_xticklabels([b for _, b in BEN], rotation=30, ha="right")
ax.set_yticks(range(len(MODELS))); ax.set_yticklabels([m for _, m in MODELS])
ax.grid(False); [s.set_visible(False) for s in ax.spines.values()]
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03); cb.set_label(r"$-\log_{10} p$ (capped at 40)", color=INK2); cb.outline.set_visible(False)
save(fig, "fig_audit_heatmap")

# ---------------------------------------------------------------- Fig: ECDF of per-item u
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
ax = axes[0]; x = np.linspace(0, 1, 201)
bcol = {"mmlu": "#eb6834", "arc": "#e87ba4", "obqa": "#2a78d6", "mmlupro": "#1baf7a", "supergpqa": "#6250d6"}
bls = {"mmlu": "-", "arc": "--", "obqa": "-", "mmlupro": "-.", "supergpqa": ":"}
for b, name in BEN:
    u = np.concatenate([tests(aud_path(m, b))[0]["order"]["u"] for m, _ in MODELS])
    ax.plot(x, [np.mean(u <= t) for t in x], color=bcol[b], ls=bls[b], label=name)
ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, label="Uniform (null)")
ax.set_xlim(0, 0.5); ax.set_ylim(0, 0.65); ax.set_xlabel("rank statistic $u$ of the published order"); ax.set_ylabel("share of items with $U \\leq u$")
ax.set_title("(a) Audit: published orders, 11 models pooled", fontsize=8, color=INK); ax.legend(fontsize=6.5, loc="upper left")
ax = axes[1]
clean, cont = [], {"full1": [], "answer3": []}
for b, d in PW.items():
    for m, _ in PM:
        for c in ["base0", "full1", "full3", "answer1", "answer3", "answer5"]:
            t, z = tests(f"{d}/power__{m}__{b}__{c}.npz"); h = int(z["half"])
            clean.append(t["slot"]["u"][h:])
            if c in cont: cont[c].append(t["slot"]["u"][:h])
for lab, u, col, ls in [("clean items (36 runs)", np.concatenate(clean), "#2a78d6", "-"),
                        ("all tokens, 1 epoch", np.concatenate(cont["full1"]), "#eb6834", "--"),
                        ("answer only, 3 epochs", np.concatenate(cont["answer3"]), "#1baf7a", "-.")]:
    ax.plot(x, [np.mean(u <= t) for t in x], color=col, ls=ls, label=lab)
ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, label="Uniform (null)")
ax.set_xlim(0, 0.5); ax.set_ylim(0, 0.65); ax.set_xlabel("rank statistic $u$ (answer-slot test)")
ax.set_title("(b) Randomized release: clean vs contaminated", fontsize=8, color=INK); ax.legend(fontsize=6.5, loc="upper left")
fig.tight_layout(); save(fig, "fig_ecdf")

# ---------------------------------------------------------------- Fig: power vs exposure (answer-only), N=50
fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.9), sharey=True)
xs = ["none", "1", "3", "5"]; cfg = ["base0", "answer1", "answer3", "answer5"]
for r, (b, d) in enumerate(PW.items()):
    Sb = json.load(open(glob.glob(f"{d}/summary_*.json")[0]))
    for cidx, (m, mn) in enumerate(PM):
        ax = axes[r, cidx]
        for key, t, s in [("slot", "slot", "fisher"), ("order", "order", "fisher"), ("anch", "anch", "fisher"), ("top1", "order", "top1"), ("mink", "order", "minkpp_ref")]:
            y = [Sb[f"power__{m}__{b}__{c}"][t]["power"]["50"]["contaminated"][s] for c in cfg]
            ax.plot(xs, y, color=C[key], marker=MK[key], ls=LS[key], label=LAB[key])
        ax.axhline(0.05, color=INK2, lw=0.7, ls=":")
        ax.set_ylim(-0.03, 1.05); ax.set_title(f"{mn}, {'MMLU' if b == 'mmlu' else 'ARC-C'}", fontsize=8, color=INK)
        if r == 1: ax.set_xlabel("epochs of answer-only training")
        if cidx == 0: ax.set_ylabel("power at $N=50$")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=5, fontsize=6.5, bbox_to_anchor=(0.5, 1.03))
fig.tight_layout(rect=(0, 0, 1, 0.95)); save(fig, "fig_power")

# ---------------------------------------------------------------- Fig: model-free design checks
it = json.load(open(ITEMS))
def num(o):
    m = re.fullmatch(r"\s*[-+$]?\s*([-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?)\s*(%|[a-zA-Z/^\s\d]*)?\s*", str(o))
    if not m: return None
    try: return float(m.group(1).replace(",", ""))
    except ValueError: return None
vals = {}
for b in ["mmlu", "arc", "obqa"]:
    ns = nt = ls_ = lt = 0
    for x_ in it[b]:
        v = [num(o) for o in x_["opts"]]
        if all(y is not None for y in v) and len(set(v)) == 4:
            nt += 1; ns += v == sorted(v) or v == sorted(v, reverse=True)
        Ls = [len(str(o)) for o in x_["opts"]]
        if len(set(Ls)) == 4:
            lt += 1; ls_ += Ls == sorted(Ls) or Ls == sorted(Ls, reverse=True)
    vals[b] = (ns / nt if nt else 0, nt, ls_ / lt, lt)
fig, ax = plt.subplots(figsize=(3.4, 2.2))
w = 0.36; xb = np.arange(3)
b1 = ax.bar(xb - w / 2 - 0.01, [vals[b][0] for b in vals], w, color="#2a78d6", label="numerical options sorted")
b2 = ax.bar(xb + w / 2 + 0.01, [vals[b][2] for b in vals], w, color="#eb6834", label="option lengths monotone")
for bars, k, n in [(b1, 0, 1), (b2, 2, 3)]:
    for bb, b in zip(bars, vals):
        ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 0.02, f"{vals[b][k]:.0%}\n(n={vals[b][n]})", ha="center", fontsize=6, color=INK2)
ax.axhline(1 / 12, color=INK, lw=0.9, ls="--", label="chance under random order (8.3%)")
ax.set_xticks(xb); ax.set_xticklabels(["MMLU", "ARC-C", "OBQA"]); ax.set_ylim(0, 1.22); ax.set_xlim(-0.65, 2.55); ax.set_ylabel("share of option sets")
ax.legend(fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2); ax.set_ylim(0, 1.12); save(fig, "fig_design")

# ---------------------------------------------------------------- Fig: MMLU option-structure breakdown
ia = json.load(open(os.path.join(_OUT, "tables", "item_table.json")))["mmlu"]
cats = [("numeric, sorted", "sorted numbers"), ("fixed template (e.g. True/False, Wrong/Not wrong pairs)", "fixed template"),
        ("roman-numeral combinations (I only, II and III, ...)", "roman-numeral sets"), ("'all/none/both of the above' option", "'... of the above'"),
        ("lengths monotone", "monotone lengths"), ("numeric, unsorted", "unsorted numbers"), ("none of these structures", "no structure")]
fig, ax = plt.subplots(figsize=(3.4, 2.4))
yv = [ia[k]["top1"] for k, _ in cats]; ns = [ia[k]["n"] for k, _ in cats]
bars = ax.barh(range(len(cats)), yv, color=["#2a78d6"] * 5 + ["#a9c8ef"] * 2, height=0.6)
for i, (v, n) in enumerate(zip(yv, ns)):
    ax.text(max(v, 0.09) + 0.01, i, f"{v:.0%} (n={n})", va="center", fontsize=6.5, color=INK2)
ax.axvline(1 / 24, color=INK, lw=0.9, ls="--"); ax.text(1 / 24 + 0.008, -0.75, "chance 4.2%", fontsize=6.5, color=INK, va="bottom")
ax.set_yticks(range(len(cats))); ax.set_yticklabels([c for _, c in cats]); ax.set_ylim(len(cats) - 0.5, -1.0)
ax.set_xlim(0, 0.8); ax.set_xlabel("share of model-item pairs with published order ranked first"); ax.grid(axis="y", visible=False)
save(fig, "fig_structure")

# ---------------------------------------------------------------- Fig: predicted vs observed power (Proposition 3)
from scipy.stats import norm, chi2
pts = []
for b, d in PW.items():
    Sb = json.load(open(glob.glob(f"{d}/summary_*.json")[0]))
    for m, _ in PM:
        for c in ["full1", "answer1", "answer3", "answer5"]:
            t, z = tests(f"{d}/power__{m}__{b}__{c}.npz"); h = int(z["half"])
            for key in ["slot", "order"]:
                X = -2 * np.log(t[key]["u"][:h]); mu, sd = X.mean(), X.std(ddof=1)
                for N in [25, 50, 100]:
                    crit = chi2.ppf(0.95, 2 * N)
                    pred = 1 - norm.cdf((crit - N * mu) / (sd * np.sqrt(N)))
                    obs = Sb[f"power__{m}__{b}__{c}"][key]["power"][str(N)]["contaminated"]["fisher"]
                    pts.append((pred, obs, key))
pts = np.array([(a, o) for a, o, _ in pts])
fig, ax = plt.subplots(figsize=(3.0, 2.8))
ax.scatter(pts[:, 0], pts[:, 1], s=14, color="#2a78d6", edgecolor="white", linewidth=0.6)
ax.plot([0, 1], [0, 1], color=INK2, lw=0.8)
ax.set_xlabel("power predicted by Proposition 3"); ax.set_ylabel("observed power (resampling)")
mae = np.mean(np.abs(pts[:, 0] - pts[:, 1])); ax.text(0.97, 0.05, f"mean abs. error {mae:.3f}\n{len(pts)} settings", ha="right", fontsize=7, color=INK2, transform=ax.transAxes)
save(fig, "fig_prop3")
json.dump({"prop3_mae": float(mae), "prop3_n": int(len(pts)), "design": vals}, open(f"{OUT}/fig_stats.json", "w"), indent=1)
print("figures done; prop3 MAE", round(mae, 4), "n", len(pts))
