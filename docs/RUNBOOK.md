# ARC-AGI-3 Runbook — ship the best stack you have

## 1. Score reality

| Item | Value |
|------|-------|
| Official metric | RHAE averaged over games, scaled **0–100%** |
| Your submitted best (Duck v12 base, 1 pass) | **~2.23** hidden LB |
| Your offline public 25-game mean (Duck v12 boost) | **3.79** |
| Best public/community open-weight submissions | low single digits |
| Strongest system on the benchmark | **~62%** (frontier closed model) |
| Humans | define the baseline; 100% by construction |

Implication for planning: the next realistic step is converting 3.79 offline into a
**better hidden-LB number than 2.23**, i.e. single-digit improvement. Hidden-set tasks are
unseen by design, so offline public mean is only a weak proxy. Nobody should plan around a
double-digit jump, and 90% is beyond the state of the art.

## 2. Primary path — Duck v12 boost (uncapped)

Canonical file: [`../notebooks/kaggle_duck_v12_boost.ipynb`](../notebooks/kaggle_duck_v12_boost.ipynb)
Push directory: [`../kaggle_submit_duck_v12/`](../kaggle_submit_duck_v12/)
Regenerate: `python scripts/build_duck_v12_boost.py` (reads `lb-9-arc3-duck-v12-with-qwen-3-8-27b.ipynb`)

### Attachments (must match `kernel-metadata.json`)

| Type | Reference |
|------|-----------|
| Dataset | `jakobbrggen/taaf-kaggle-source-anim-20260807-anim` |
| Dataset | `driessmit1/arc3-vllm-h100-wheelhouse-v3` |
| Model | `foysalemonshanto/qwen3-8-27b-fp8-repacked-v1` → PyTorch → `hf-fp8` → Version 1 |
| Competition | `arc-prize-2026-arc-agi-3` |
| Accelerator | NVIDIA RTX Pro 6000 |
| Internet | **off** |

### What the boost hook changes (notebook cell 8)

Applied on **both** debug and real reruns:

```
solver.concurrency            = 28
solver.max_runtime_s_per_game = 7920.0
solver.max_actions_per_game   = None      # explicitly uncapped
solver.analyzer_timeout       = 1800.0    # was timing out on vLLM reads
solver.hard_noop_guard        = True
solver.animation_awareness    = True
```

Applied **only when `KAGGLE_IS_COMPETITION_RERUN` is false** (i.e. offline/debug):

- replaces `bm.games` with the 25 known public game ids,
- sets `bm.n_passes = 2`, `bm.game_weights = None`.

### Known gap — verify before believing any second-pass gain

The 2-pass trick lives inside the `if not true_submission:` branch, so **the scored rerun
probably still runs 1 pass** (`bm.n_passes` comes from `benchmark_initial.pkl`). The offline
3.79 therefore is not an estimate of the rerun. Two consequences:

1. Do not claim a 2-pass improvement until you have seen `n_passes` in a real rerun log.
2. Forcing `bm.n_passes = 2` on the rerun is **not** a safe free win: the hidden set can be
   larger than 25 games and `7920 s` per game against a 9 h wall can overrun, and an overrun
   scores zero. If you want to try it, gate it on the actual game count and total budget,
   and burn a throwaway submission to validate.

Rerun-side behavior worth knowing (cell 9): `ONLY_RESET_LEVELS=true`,
`TAAF_MINIMAL_DIAGNOSTICS=1` (so no `diagnostics.html` on the scored run), gateway polled at
`http://gateway:8001/` for up to 600 s before the run aborts.

## 3. Archived path — capped TAAF `target_325`

[`../notebooks/kaggle_submit_target_325.ipynb`](../notebooks/kaggle_submit_target_325.ipynb),
[`../configs/target_325.json`](../configs/target_325.json),
[`../agents/calibrated_agent.py`](../agents/calibrated_agent.py).

Kept as a **negative result**: `MAX_ACTIONS_PER_GAME=55`, `MAX_LEVELS_PER_GAME=2`,
`STRICT_PREFIXES=["sc25"]` produced hidden LB **1.76** vs the uncapped stack's 2.23. Budget
caps cut score; the earlier instruction in this file to land inside a 3.0–3.5 band was
self-imposed and should not be reinstated. Do not submit this path.

## 4. Local scoring

```bash
python scripts/eval_rhae.py --results artifacts/my_run.json --env-dir data/environment_files
```

`--calibrate-taaf` only simulates cap effects on the archived path; it is not a predictor for
the primary notebook. `artifacts/example_results.json` and
`artifacts/path_a_calibration.json` belong to that archived experiment.

## 5. Submission checklist

- [ ] `kernel-metadata.json` in `kaggle_submit_duck_v12/` lists all four attachments above
- [ ] Model version still `1` and `hf-fp8` variation exists (path is hard-coded in cell 3)
- [ ] Notebook commits under the 9 h limit with internet disabled
- [ ] You read the rerun log and recorded the real `n_passes` value
- [ ] Submit through the Kaggle UI on a committed version, then note the LB delta vs 2.23

## 6. Commands (run these yourself — they consume GPU quota and use a submission slot)

From the repo root, in PowerShell. Each line is standalone; no continuations.

```powershell
cd C:\hackathon\Gemini_CLI\ARC-Prize-2026---ARC-AGI-3
```

Upload the primary notebook as a new kernel version (attach the datasets/model/competition
in the Kaggle UI first, or trust `kaggle_submit_duck_v12/kernel-metadata.json`):

```powershell
kaggle kernels push -p .\kaggle_submit_duck_v12
```

Watch it, then submit to the competition when the status is COMPLETE:

```powershell
python .\scripts\kaggle_wait_and_submit.py
```

Or submit manually once the run finishes:

```powershell
kaggle competitions submissions -c arc-prize-2026-arc-agi-3 -k vyassenthilkumar/arc3-duck-v12-boost -m "uncapped duck v12"
```

Check the result against the previous best:

```powershell
kaggle competitions leaderboard -c arc-prize-2026-arc-agi-3 -s
```

Before pushing, confirm in the Kaggle UI that the accelerator is **NvidiaRtxPro6000**,
internet is **off**, and all four attachments resolve (cell 4 of the notebook prints an
existence audit — read it, and note the printed `n_passes`).

## 7. What not to do

- Do not add action/level caps back.
- Do not report offline 3.79 as a leaderboard score.
- Do not aim at a target *band*; aim at the highest hidden LB the stack can reach.
