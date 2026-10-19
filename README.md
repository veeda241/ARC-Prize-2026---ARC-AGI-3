# ARC Prize 2026 — ARC-AGI-3

Agent that plays the hidden, interactive ARC-AGI-3 games using the TAAF harness plus a
served Qwen3.8-27B-FP8 model on Kaggle GPUs.

## Where this actually stands

Verified against the live Kaggle leaderboard and this account's submission log on 2026-10-19.

| Leaderboard position | Team | Score |
|----------------------|------|-------|
| 1 | Tufa Labs | **68.91** |
| 2 | Yi-Chia Chen | **67.49** |
| 3 | Team Turducken | 52.82 |
| 20 | (cut) | ~38 |

| This account's submissions (public score) | Date | Score |
|-------------------------------------------|------|-------|
| Path A TAAF, capped at 2 levels / 55 actions | 2026-08-17 | **1.76** ← still the best here |
| Uncapped TAAF v3 | 2026-08-18 | 1.10 |
| Calibrated heuristic MyAgent | 2026-08-19 | 0.01 |
| Duck v12 Boost | 2026-08-20 | 1.13 |

Two conclusions that reverse earlier notes in this repo:

1. **Uncapping did not help.** 1.76 capped beat 1.10 and 1.13 uncapped. The "2.23 vs 1.76"
   figures inside the notebooks are not supported by the submission log.
2. **The gap to the top is roughly 40–60 points, and it is closable.** The leading solutions
   are open: Tufa Labs' [Duck harness](https://github.com/Tufalabs/duck-harness) plus a newer
   model (Qwen3.8-Flash-Next, AutoRound INT4, with a speculative-decoding draft model) and
   4-pass scoring. This repo forked an August snapshot of that lineage
   (`foysalemonshanto/lb-9-...`, last run 2026-08-18) and has not tracked it since.

See **[docs/RUNBOOK.md](docs/RUNBOOK.md)** for the upgrade path and the measured local baselines.

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
