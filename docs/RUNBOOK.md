# ARC-AGI-3 Runbook — ship the best stack you have

## 1. Score reality

Verified 2026-10-19 with `kaggle competitions leaderboard --show` and
`kaggle competitions submissions` (not from notebook markdown).

| Item | Value |
|------|-------|
| Official metric | RHAE averaged over games, scaled **0–100%** |
| Leaderboard #1 (Tufa Labs) | **68.91** |
| Leaderboard #2 (Yi-Chia Chen) | **67.49** |
| Leaderboard rank 20 | ~**38** |
| **Best score this account has achieved** | **1.76** (capped Path A, 2026-08-17) |
| Uncapped TAAF v3 | 1.10 (2026-08-18) |
| Duck v12 Boost | 1.13 (2026-08-20) |
| Heuristic MyAgent | 0.01 (2026-08-19) |
| Local search-only baselines (this repo) | 0.41–0.86 |

Corrections to what this file used to claim:

- **Uncapping did not raise the score.** The capped run (1.76) beat both uncapped runs
  (1.10, 1.13). Do not treat "remove the caps" as a free win; it was measured and it lost.
- The `~2.23` and `3.79` numbers embedded in the notebooks are not in the submission log and
  should be treated as unverified self-reporting, not as results.
- Scores of 40–68 on this competition are real and are held by open, readable public kernels.
  The gap to them is a harness-and-model gap, not a ceiling.

## 1b. What the leaders actually run

Pulled and inspected 2026-10-19 from `dfranzen/arc-agi-3-milestone-2-solution` (949 votes,
the most-upvoted public kernel in the competition):

- Base: **Tufa Labs Duck harness** — <https://github.com/Tufalabs/duck-harness>, pinned at
  commit `7652836`, applied as an in-notebook patch rather than a forked bundle.
- Model: **Qwen3.8-Flash-Next**, Intel AutoRound **INT4** quantised, asserted in-code
  (`quant_method == "auto-round"`, `bits == 4`).
- **Speculative decoding** via a separate draft model
  (`dfranzen/albucino-qwen3-8-flash-next-drafter`, `mtp-int4-g32`) — this is how they buy
  throughput inside the 9 h wall, rather than by cutting actions.
- **`n_passes = 4`** over the 25 demo games, with per-game runtime scaled to the competition.
- Explicit score-formula engineering: the notebook reasons about `min(1, (human/used)^2)`
  per level, the 0.25 endgame floor, and that level 1 is worth only `1/N` of a game — i.e.
  budget is allocated across levels by score value, which is the opposite of a global cap.
- Write-up: <https://www.kaggle.com/writeups/dfranzen/arc-agi-3-milestone-2>

It is committed at `kaggle_submit_milestone2/arc-agi-3-milestone-2-solution.ipynb` (see §1c);
pull a fresh copy any time with:

```powershell
kaggle kernels pull dfranzen/arc-agi-3-milestone-2-solution -p .\upstream
```

Other high-vote public kernels in the same lineage, worth comparing:
`jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner`,
`wuliao0/duck-qwen3-8-anim-base`, `chiakazirim/duck-qwen3-8-tuned`,
`keithtyser/duck-qwen3-8-flash-next-nvfp4-mtp`, `scottlegrand/taaf-flashnext-sheetu12b-0922`.
List them yourself with
`kaggle kernels list --competition arc-prize-2026-arc-agi-3 --sort-by voteCount`.

## 1c. Push-ready copy in this repo — `kaggle_submit_milestone2/`

Verbatim pull of `dfranzen/arc-agi-3-milestone-2-solution` (23 cells) plus
`kernel-metadata.json` with every attachment it hard-codes. Nothing in the notebook is edited.

| Attachment | Value | Verified 2026-10-19 |
|---|---|---|
| model | `dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1` | public, v1 READY, **181 GB** |
| model | `dfranzen/albucino-qwen3-8-flash-next-drafter/Transformers/default/1` | public, v1 READY, 4.2 GB |
| dataset | `dfranzen/pennyroyal-v253` | exists (SGLang RTX-Pro-6000 fork: lockfiles + `hot_tokens_64k.pt`) |
| dataset | `dfranzen/taaf-kaggle-source-bundle-copy` | exists (`benchmark_initial.pkl`, `deploy_target.pkl`) |
| competition | `arc-prize-2026-arc-agi-3` | supplies `arc_agi_3_wheels` |

All remote URLs in the notebook are `127.0.0.1` (the local vLLM/SGLang server) or markdown
links, and there is no `pip install`/`git clone`, so `enable_internet: false` is safe.

**This has never been run from this account.** Expect it to need: the RTX PRO 6000 accelerator
selected (the notebook's own markdown says to pick it manually), ~185 GB of attached model
storage, and a runtime close to the 9 h ceiling. Model-source strings use the documented
`<owner>/<slug>/<Framework>/<instance>/<version>` form with `Transformers` capitalised (the API
reports `MODEL_FRAMEWORK_TRANSFORMERS`; the mount path stays lowercase, which is exactly what
the notebook hard-codes). Still unverified: whether the harness-patch cell assumes any
author-local state, and whether the run fits the wall clock.

## 2. Existing path in this repo — Duck v12 boost (scored 1.13)

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

## 3. Best-scoring path so far — capped TAAF `target_325` (1.76)

[`../notebooks/kaggle_submit_target_325.ipynb`](../notebooks/kaggle_submit_target_325.ipynb),
[`../configs/target_325.json`](../configs/target_325.json),
[`../agents/calibrated_agent.py`](../agents/calibrated_agent.py).

Correction: an earlier version of this file called this path a **negative result** and claimed
the uncapped stack scored 2.23 against its 1.76. The submission log says the opposite — the
capped run is this account's **best score, 1.76**, and the uncapped runs scored 1.10 and 1.13.
The 2.23 figure appears only inside notebook markdown and was never verified.

Do not read this as "caps are good" either. Both numbers are single-digit and the leaderboard
top is 68.91, so capping versus uncapping is a fight over noise between two submissions that
are both roughly 40 points behind the leaders for structural reasons (see §1b). Keep this path
only as the fallback that reliably completes; the upgrade in §1b is the real target.

## 4. Local scoring

```bash
python scripts/eval_rhae.py --results artifacts/my_run.json --env-dir data/environment_files
```

`--calibrate-taaf` only simulates cap effects on the archived path; it is not a predictor for
the primary notebook. `artifacts/example_results.json` and
`artifacts/path_a_calibration.json` belong to that archived experiment.

## 5. Submission checklist

- [ ] `kernel-metadata.json` in `kaggle_submit_duck_v12/` lists all four attachments above
- [ ] For `kaggle_submit_milestone2/`: all five §1c attachments resolve, and the first minutes of
      the log show the model directories found before the server launches
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

For the §1c target instead (this is the one worth the GPU hours):

```powershell
kaggle kernels push -p .\kaggle_submit_milestone2
```

Check that Kaggle accepted the attachments before waiting nine hours — push it, then immediately:

```powershell
kaggle kernels status vyassenthilkumar/arc3-milestone2-flashnext
```

If the run errors within minutes, it is almost certainly an unresolved mount; read the log with
`kaggle kernels output vyassenthilkumar/arc3-milestone2-flashnext`.

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

- Do not report a public/offline mean as a leaderboard score. Public 25-game means and hidden
  LB numbers have diverged by 1.5x+ in this repo's own history.
- Do not aim at a target *band*; aim at the highest hidden LB the stack can reach.
- Do not trust a score quoted in notebook markdown. Verify with
  `kaggle competitions submissions -c arc-prize-2026-arc-agi-3` and
  `kaggle competitions leaderboard -c arc-prize-2026-arc-agi-3 --show` before writing a claim
  into this file. Two earlier claims here — "uncapping is a free win", and "67–70% belongs to
  ARC-AGI-2" — were both checked against the live leaderboard on 2026-10-19 and were wrong.
- Do not treat caps versus uncaps as the decisive lever. Both are single-digit; the leaders are
  40–60 points away because of harness, model and budget allocation, not because of one flag.
- ARC-AGI-3 differs from ARC-AGI-2 in format: it is interactive (frame deltas, `GAME_OVER`,
  `levels_completed`) rather than static input/output grid pairs, so a "write a transform, test
  it on the demos, feed the traceback back" self-debugging loop has no demo set to converge on.
  Search-only policies were measured here and did not work (§7) — but that is a result about
  *this* search implementation, not proof the interactive format defeats code-generation agents;
  the leaderboard leaders pair an LLM harness with score-aware budget allocation.

