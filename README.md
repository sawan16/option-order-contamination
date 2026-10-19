# Exact option-order tests for multiple-choice benchmark contamination

Code, item-level results and analysis scripts for the paper:

> **Natural Option Orders Confound Contamination Detection in Multiple-Choice Benchmarks: Exact Permutation Tests and a Release Protocol**
> Sawan Rai, SRM University-AP (submitted to *IEEE Access*)

Option-order detectors flag contamination when a language model prefers a multiple-choice item's published option order over re-orderings of the same options. This repository implements three **exact** randomization tests of that preference:
- the **order** test, which ranks the published order among all orderings;
- the **answer-slot** test;
- the **within-slot** test.

It also contains everything needed to reproduce the audit of 11 public models on 5 benchmarks and the simulated-contamination experiments.

## Repository layout

| Path | Contents |
|---|---|
| `code/core.py` | The tests: permutation plans, scoring, randomized ranks, exact Fisher and binomial p-values, contamination simulation by fine-tuning |
| `code/run.py` | Experiment stages: `audit1`, `audit2` (public models, published orders) and `power_mmlu`, `power_arc` (fine-tuning on shuffled items) |
| `code/contamination_paper_experiments.ipynb` | The same stages as a notebook for one T4 GPU (Kaggle or Colab) |
| `code/export_items.py` | Re-creates the exact audited items with their texts |
| `results/audit/` | Scores of every ordering for 11 models × 5 benchmarks (`.npz`) and summaries |
| `results/power_mmlu/`, `results/power_arc/` | Scores and Min-K%++ values for 3 models × 6 exposure settings per benchmark |
| `results/items/audit_items.json` | Texts of the 1,500 audited MMLU, ARC-Challenge and OpenBookQA items |
| `analysis/` | Scripts that produce every table, figure and number in the paper |
| `outputs/` | The tables and figures as produced by `analysis/make_all.sh` |

## Reproduce the paper from the saved results (CPU, about 2 minutes)

```bash
pip install -r requirements.txt
bash analysis/make_all.sh
```

This writes `outputs/tables/*.tex` and `outputs/figures/*.pdf`, and prints:
- the audit counts (models rejecting per benchmark);
- the model-free design checks;
- the option-structure analysis;
- the accuracy of the sample-size proposition (mean absolute error 0.020 over 144 settings).

## Re-run the experiments (GPU)

Open `code/contamination_paper_experiments.ipynb` on Kaggle or Colab with one T4 GPU and Internet on. Set `STAGE` to `audit1`, `audit2`, `power_mmlu` or `power_arc`, then run all cells. Each finished unit is cached, so an interrupted stage resumes where it stopped.

The stages take about 3–8 GPU-hours each on a T4. Alternatively:

```bash
cd code
python run.py audit1 ../results_new/audit
```

## Auditing your own model or benchmark

```python
import numpy as np, core
S = core.score_bench(model, tok, items, "cuda", np.random.default_rng(1))   # items: [{"q":..., "opts":[...], "ans": int}]
t = core.tests_from_scores(S, items, None, np.random.default_rng(0))
print(core.set_tests(t))   # exact Fisher p-values and top-1 rates for the order, slot and within-slot tests
```

**Read the result only after a design check.** The p-values are exact only if the benchmark's published option order is random given the content. Run the model-free checks in `analysis/design_checks.py` and include a negative-control model trained before the benchmark's release. On MMLU and ARC-Challenge this assumption fails, so a rejection there is not evidence of contamination.

## Licences

- **Code** (`code/`, `analysis/`): MIT licence; see `LICENSE`.
- **Results** (`results/audit`, `results/power_*`, `outputs/`): CC BY 4.0.
- **Benchmark item texts** in `results/items/audit_items.json`: copied from MMLU, ARC-Challenge and OpenBookQA, and they remain under the original licences of those datasets.

## Citation

See `CITATION.cff`. A BibTeX entry will be added on publication.
