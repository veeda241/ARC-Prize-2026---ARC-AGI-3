# ARC Prize 2026 — ARC-AGI-3

Agent that plays the hidden, interactive ARC-AGI-3 games using the TAAF harness plus a
served Qwen3.8-27B-FP8 model on Kaggle GPUs.

## Where this actually stands

| Stack | Public 25 games | Hidden leaderboard |
|-------|-----------------|--------------------|
| **Duck v12 boost (primary, uncapped)** — [`notebooks/kaggle_duck_v12_boost.ipynb`](notebooks/kaggle_duck_v12_boost.ipynb) | **3.79** | not yet run |
| Duck v12 base (1-pass, as submitted) | — | **~2.23** |
| Capped TAAF (`target_325`, actions limited to 55) | 3.28 by construction | **1.76** |

Scores are the official 0–100% RHAE scale. Numbers above are measured, not projected.

For calibration, the strongest system on this benchmark anywhere in the world is about
**62%**, and the 100% grand prize remains unclaimed. A double-digit leaderboard score is a
good outcome for an open-weight submission; anything near 90% is not currently reachable.

## Primary path

[`notebooks/kaggle_duck_v12_boost.ipynb`](notebooks/kaggle_duck_v12_boost.ipynb)
(mirrored in [`kaggle_submit_duck_v12/`](kaggle_submit_duck_v12/) for `kaggle kernels push`).
Regenerate it from source with `python scripts/build_duck_v12_boost.py`.

It deliberately **does not cap actions**. Capping was tried and cost real score: 1.76 vs 2.23.
The old `target_325` notebook and [`configs/target_325.json`](configs/target_325.json) are kept
only as that negative result — do not submit them.

Full procedure, attached datasets/models, and risk notes:
**[docs/RUNBOOK.md](docs/RUNBOOK.md)**

## Repo layout

```
notebooks/                     canonical notebooks (regenerable via scripts/build_*.py)
kaggle_submit_duck_v12/        push-ready notebook + kernel-metadata.json  <- PRIMARY
kaggle_submit/                 archived capped TAAF submit
kaggle_submit_calibrated/      archived heuristic/Unsloth agent submit
agents/, agent/                budgeted agent used by the archived paths
configs/target_325.json        archived calibration knobs
scripts/eval_rhae.py           offline RHAE scorer over recorded per-game results
data/                          competition files + linux wheelhouse
lb-9-arc3-duck-v12-*.ipynb     pulled upstream base notebook (input to the builder)
```
