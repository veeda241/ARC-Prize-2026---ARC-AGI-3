# ARC Prize 2026 — ARC-AGI-3

Agent that plays the hidden, interactive ARC-AGI-3 games on Kaggle GPUs. Current target is the
Tufa Labs Duck harness serving Qwen3.8-Flash-Next (AutoRound INT4 + MTP draft head); the older
TAAF / Qwen3.8-27B-FP8 paths in this repo are archived.

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
| Milestone 2 Flash-Next stack, py3.12 venv | 2026-10-19 | **30.55** ← new best, ~rank 510 |
| Path A TAAF, capped at 2 levels / 55 actions | 2026-08-17 | 1.76 |
| Uncapped TAAF v3 | 2026-08-18 | 1.10 |
| Calibrated heuristic MyAgent | 2026-08-19 | 0.01 |
| Duck v12 Boost | 2026-08-20 | 1.13 |

Three conclusions that reverse earlier notes in this repo:

1. **Uncapping did not help** — on the old stack. 1.76 capped beat 1.10 and 1.13 uncapped, and
   the "2.23 vs 1.76" figures inside those notebooks are not supported by the submission log.
2. **The gap to the top was closable, and it is now closed by ~half.** The leading solutions
   are open: Tufa Labs' [Duck harness](https://github.com/Tufalabs/duck-harness) plus a newer
   model (Qwen3.8-Flash-Next, AutoRound INT4, with a speculative-decoding draft model). This
   repo had forked an August snapshot of that lineage (`foysalemonshanto/lb-9-...`, last run
   2026-08-18) and never tracked it forward.
3. **Porting the leaders' stack worked: 1.76 → 30.55 on the hidden set.** Two measurements from
   2026-10-19, both in this repo:
   - demo set (25 public games, best of 4 passes, `kaggle_submit_milestone2/` v2): **59.82**
     mean RHAE, against **61.89** for the author's own published run on the same games
     ([`artifacts/milestone2_demo_run.json`](artifacts/milestone2_demo_run.json));
   - hidden set (submission ref `57402922`, where the notebook forces `n_passes = 1`):
     **30.55**, roughly rank 510.

   The 29-point gap between those two is the cost of one shot per game instead of keeping the
   best of four — that is the single biggest remaining lever, not the model. One local patch
   was required to run any of it: Kaggle's image moved to Python 3.13 while the SGLang
   wheelhouse is cp312-only (`scripts/patch_milestone2_python312.py`).

See **[docs/RUNBOOK.md](docs/RUNBOOK.md)** for the upgrade path and the measured local baselines.

## Repo layout

```
notebooks/                     canonical notebooks (regenerable via scripts/build_*.py)
kaggle_submit_milestone2/      push-ready copy of the top public kernel   <- PRIMARY TARGET
kaggle_submit_duck_v12/        push-ready notebook + kernel-metadata.json (scored 1.13)
kaggle_submit/                 archived capped TAAF submit
kaggle_submit_calibrated/      archived heuristic/Unsloth agent submit
agents/, agent/                budgeted agent used by the archived paths
configs/target_325.json        archived calibration knobs
scripts/eval_rhae.py           offline RHAE scorer over recorded per-game results
data/                          competition files + linux wheelhouse
lb-9-arc3-duck-v12-*.ipynb     pulled upstream base notebook (input to the builder)
```
