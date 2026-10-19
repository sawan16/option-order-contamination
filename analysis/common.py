"""Shared paths and helpers for the analysis scripts."""
import os, sys, json, re
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "code"))
import core  # noqa: E402

RES = os.path.join(ROOT, "results")
AUDIT = os.path.join(RES, "audit")
POWER = {"mmlu": os.path.join(RES, "power_mmlu"), "arc": os.path.join(RES, "power_arc")}
ITEMS = os.path.join(RES, "items", "audit_items.json")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)

MODELS = [("pythia-410m-deduped", "Pythia-410M", "2023 (2020 data)"), ("pythia-1.4b-deduped", "Pythia-1.4B", "2023 (2020 data)"),
          ("pythia-2.8b-deduped", "Pythia-2.8B", "2023 (2020 data)"), ("TinyLlama_v1.1", "TinyLlama-1.1B", "2024"),
          ("SmolLM2-360M", "SmolLM2-360M", "2024"), ("SmolLM2-1.7B", "SmolLM2-1.7B", "2024"),
          ("Qwen2.5-0.5B", "Qwen2.5-0.5B", "2024"), ("Qwen2.5-1.5B", "Qwen2.5-1.5B", "2024"),
          ("OLMo-2-0425-1B", "OLMo-2-1B", "2025"), ("Qwen3-0.6B-Base", "Qwen3-0.6B", "2025"), ("Qwen3-1.7B-Base", "Qwen3-1.7B", "2025")]
BENCH = [("mmlu", "MMLU"), ("arc", "ARC-C"), ("obqa", "OBQA"), ("mmlupro", "MMLU-Pro"), ("supergpqa", "SuperGPQA")]
POWER_MODELS = [("SmolLM2-360M", "SmolLM2-360M"), ("Qwen2.5-0.5B", "Qwen2.5-0.5B"), ("Qwen3-0.6B-Base", "Qwen3-0.6B")]


def audit_summary():
    S = {}
    for f in ["summary_audit1.json", "summary_audit2.json"]:
        S.update(json.load(open(os.path.join(AUDIT, f))))
    return S


def power_summary(bench):
    return json.load(open(os.path.join(POWER[bench], f"summary_power_{bench}.json")))


def tests(path, seed=0):
    """Recompute the per-item test statistics from a saved score file."""
    z = np.load(path)
    its = [{"opts": list(range(int(n))), "ans": int(a)} for a, n in zip(z["ans"], z["nopt"])]
    M = None if int(z["M"]) < 0 else int(z["M"])
    t = core.tests_from_scores({"scores": list(z["scores"]), "full_idx": list(z["full_idx"]), "anch_idx": list(z["anch_idx"])},
                               its, M, np.random.default_rng(seed))
    return t, z


def num(o):
    m = re.fullmatch(r"\s*[-+$]?\s*([-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?)\s*(%|[a-zA-Z/^\s\d]*)?\s*", str(o))
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None
