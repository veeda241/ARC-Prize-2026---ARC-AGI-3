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

## 7. Local no-GPU evaluation (`scripts/local_search_agent.py`)

The 25 public games run **on this laptop** without a GPU, a served model, or Kaggle.
`data/arc_agi_3_wheels` ships `arcengine` and `arc_agi` as pure-`py3-none-any` wheels, and
this machine has Python 3.12 (`arcengine` requires >=3.12; the default 3.11 cannot take them):

```powershell
python3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install data\arc_agi_3_wheels\arcengine-0.9.3-py3-none-any.whl data\arc_agi_3_wheels\arc_agi-0.9.8-py3-none-any.whl
.\.venv312\Scripts\python.exe scripts\local_search_agent.py --policy novelty --budget 10000
python scripts\eval_rhae.py --results artifacts\local_novelty_10000.json --env-dir data\environment_files
```

Throughput is ~1,400 env steps/second, so a 25-game x 10,000-step sweep costs about 3 minutes.
This is the fastest iteration loop in the repo — use it before spending a Kaggle run.

### Measured: search-only baselines are worse than the LLM stack

Same 25 public games, same scorer, mean RHAE on the 0–100 scale. Upper rows are single runs;
the three lower policies are the same 25 games run 5 times with different RNG seeds.

| Agent | Steps/game | Mean RHAE | Games reaching level 1 | Past level 1 |
|-------|-----------|-----------|------------------------|--------------|
| random actions | 2,000 | 0.38 | 2 | 0 |
| state-hash novelty | 2,000 | 0.49 | 5 | 0 |
| state-hash novelty | 10,000 | 0.70 | 7 | 0 |
| **random**, 5 seeds | 2,000 | **0.60** (0.49–0.71) | 6–7 | 0 |
| **novelty**, 5 seeds | 2,000 | **0.55** (0.38–0.73) | 5–7 | 0 |
| **frontier prior**, 5 seeds | 2,000 | **0.41** (0.31–0.55) | 5 | 0 |
| Qwen3.8-27B TAAF/Duck stack | — | **3.79** | — | — |

Three honest takeaways:

1. **No search policy ever cleared level 1**, in any seed, at any budget. The served-LLM stack
   is ~4–6x ahead of the best search run, so exploration does not substitute for the model.
2. **Seed variance is bigger than the policy differences.** Random alone swings 0.49–0.71. Any
   A/B test in this harness needs several seeds; single-run deltas under ~0.15 are noise.
3. **The frontier-prior policy is a negative result.** `--policy prior` learns each action's
   screen displacement and walks toward the nearest never-seen salient pixel — and it loses to
   plain random (0.41 vs 0.60, all five seeds lower). The likely cause is the perception layer:
   estimating the player as "centre of changed pixels" is polluted by animation, explosions and
   particles, so the learned action model points the wrong way. Treat the idea as unproven, not
   as validated.

### Ensembling: measured, and it is sublinear

`--seeds N` re-runs every game and keeps the best result per game (the same rule Kaggle's
scorecard applies across passes):

| Policy | Single run | Best of 5 | Gain |
|--------|-----------|-----------|------|
| random | 0.60 | **0.86** | +43% |
| novelty | 0.55 | **0.82** | +51% |
| frontier prior | 0.41 | **0.70** | +71% |

5x compute buys roughly 1.4–1.7x score, and restarting beats lengthening one run
(novelty: 5 seeds at 2k = 0.83 vs one run at 10k = 0.70). Sample-keep-the-best is real but
modest — it does not close the 4x gap to the LLM stack.

Reproduce with:

```powershell
.\.venv312\Scripts\python.exe scripts\local_search_agent.py --policy prior --budget 2000 --seeds 5
```

### The lever that search *does* expose: action efficiency

RHAE rewards finishing a level in fewer actions than the human `baseline_actions`, and
`metadata.json` carries those per-level baselines locally. The solved levels show the spread:

| Game | Agent actions | Human baseline |
|------|---------------|----------------|
| r11l | 7 | 22 |
| sp80 | 25 | 39 |
| cd82 | 25 | 55 |
| m0r0 | 94 | 30 |

Exploration is undirected: it can stumble into a level cheaply (r11l beats the human by 3x) and
wastefully (m0r0 at 3x the human). Since the score is efficiency-weighted, narrowing that
variance is worth more than adding raw steps.

## 8. What not to do

- Do not add action/level caps back.
- Do not report offline 3.79 as a leaderboard score.
- Do not aim at a target *band*; aim at the highest hidden LB the stack can reach.
- Do not import ARC-AGI-2 recipes wholesale. ARC-AGI-2 is static input/output grid pairs, so
  "write a Python transform, run it against the demo pairs, feed the traceback back until every
  demo matches" has a correctness test to converge on. ARC-AGI-3 has no demo pairs — only frame
  deltas, `GAME_OVER`, and `levels_completed` — so that self-debugging loop has nothing to check
  itself against. The widely-cited ~67–70% test-time-training / code-generation results are
  ARC-AGI-2 (ARC Prize 2025) numbers, not ARC-AGI-3.
