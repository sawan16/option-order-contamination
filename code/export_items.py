"""Re-create the exact audit items (same seed as run.py) with their texts.
Usage: python export_items.py [out.json]   (needs the `datasets` package and internet access)"""
import json, sys
import numpy as np
from datasets import load_dataset

def pick(rows, n):
    idx = np.random.default_rng(0).permutation(len(rows))[:n]
    return [{"q": rows[int(i)][0], "opts": rows[int(i)][1], "ans": rows[int(i)][2]} for i in idx]

out = {}
ds = load_dataset("cais/mmlu", "all", split="test")
out["mmlu"] = pick([(r["question"], list(r["choices"]), int(r["answer"])) for r in ds], 500)
for name in ["arc", "obqa"]:
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test") if name == "arc" else load_dataset("allenai/openbookqa", "main", split="test")
    rows = []
    for r in ds:
        labs, txt = list(r["choices"]["label"]), list(r["choices"]["text"])
        if len(txt) == 4 and r["answerKey"] in labs:
            rows.append((r["question"] if name == "arc" else r["question_stem"], txt, labs.index(r["answerKey"])))
    out[name] = pick(rows, 500)
path = sys.argv[1] if len(sys.argv) > 1 else "audit_items.json"
json.dump(out, open(path, "w"))
print({k: len(v) for k, v in out.items()}, "->", path)
