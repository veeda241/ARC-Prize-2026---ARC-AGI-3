# ARC-AGI-3 Runbook — ship the best stack you have

## 1. Score reality

Verified 2026-10-20 with `kaggle competitions leaderboard --show` and
`kaggle competitions submissions` (not from notebook markdown).

| Item | Value |
|------|-------|
| Official metric | RHAE averaged over games, scaled **0–100%** |
| Leaderboard #1 (Tufa Labs) | **68.91** |
| Leaderboard #2 (Yi-Chia Chen) | **67.49** |
| Leaderboard rank 20 | ~**38** |
| **Best score this account has achieved** | **30.55** (Milestone 2 Flash-Next, 2026-10-19, ~rank 510) |
| Previous best, archived TAAF/Duck paths | 1.76 (2026-08-17), 1.13, 1.10, 0.01 |
| Same stack on the 25 public games, best of 4 passes | **59.82** (author's own run: 61.89) |
| Local search-only baselines (this repo) | 0.41–0.86 |

Corrections to what this file used to claim:

- **Uncapping did not raise the score** — on the old stack. The capped run (1.76) beat both
  uncapped runs (1.10, 1.13). Do not treat "remove the caps" as a free win; it was measured and
  it lost. The new stack runs with `max_actions_per_game = None` and scores 30.55, but that
  changes harness, model and caps together, so it says nothing about the flag on its own.
- The `~2.23` and `3.79` numbers embedded in the notebooks are not in the submission log and
  should be treated as unverified self-reporting, not as results.
- Scores of 40–68 on this competition are real and are held by open, readable public kernels.
  The gap to them is a harness-and-model gap, not a ceiling — and porting that harness moved
  this account from 1.76 to 30.55 in one run (§1f).

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
`kernel-metadata.json` with every attachment it hard-codes. One line of the launcher cell is
changed for the Python-3.13 image — see §1d, and `scripts/patch_milestone2_python312.py` is
that diff in reproducible form. Everything else is byte-for-byte upstream.

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
the notebook hard-codes). §1d confirmed the patch cell applies and all five mounts resolve. Still unverified: whether
`machine_shape` in metadata really selects the RTX PRO 6000 without touching the UI, and
whether the run fits the 9 h wall clock.

## 1d. Attempt 1, 2026-10-19: failed in 17 s and why

Version 1 of `vyassenthilkumar/arc3-milestone2-flashnext-duck` errored before playing a game.
The log (`kaggle kernels output <slug> -p .\tmp`) shows the good news first: the harness patch
applied against the bundle, the `taaf_kaggle_sources.pth` was written, and all five §1c
attachments resolved — so the metadata and the 181 GB model mount are fine.

The failure is a Python ABI mismatch, not a Kaggle config problem:

```
$ uv venv --python /usr/bin/python3 /tmp/sgl-intel/venv
error: No solution found when resolving dependencies
  cause: aiohttp==3.14.3 has no wheels with a matching Python ABI tag (e.g., `cp313`)
hint: You require CPython 3.13, but we only found wheels for aiohttp (v3.14.3) with ABI tag cp312
```

Kaggle's notebook image has moved to **Python 3.13**, but `dfranzen/pennyroyal-v253` (SGLang
fork v2.5.3) ships **cp312-only** wheels. The author's run predates the image change.

Fix applied in this repo: `scripts/patch_milestone2_python312.py` rewrites the one venv line in
cell 6 to search for a 3.12 interpreter (`/usr/bin/python3.12`, `/usr/local/bin/python3.12`,
`python3.12` on PATH), prints an `ls /usr/bin` audit, and raises an explicit error if none
exists. Re-apply after any re-pull with:

```powershell
python3.12 .\scripts\patch_milestone2_python312.py .\kaggle_submit_milestone2\arc-agi-3-milestone-2-solution.ipynb
```

**Unverified:** whether the Kaggle image still ships a 3.12 interpreter at all. If version 2
fails with "No CPython 3.12 on this image", the options are a cp313 SGLang bundle from another
kernel in this competition's lineage, or accepting that this particular public stack can no
longer be reproduced from its own wheelhouse.

- A plain (non-rerun) kernel run is **not scored and not self-reporting**: the notebook writes a
  hard-coded one-row `submission.parquet` (`[["1_0","1",True,1]]`) whenever
  `KAGGLE_IS_COMPETITION_RERUN` is unset. The real per-game results only exist in
  `.per_game_cards.pkl` and `diagnostics.html` under `/kaggle/working`, so after any run pull
  the outputs and decode them with `scripts/parse_milestone2_cards.py` (§1e).

Two facts read out of the notebook that matter for scoring:

- In a real competition rerun (`KAGGLE_IS_COMPETITION_RERUN`), the script **overrides
  `bm.n_passes = 1`** and sets `concurrency = 120`, `max_runtime_s_per_game = 532*60`. The
  4-pass configuration only applies to the offline demo run, so do not expect 4x sampling on
  the hidden set.
- `bm.solver.max_actions_per_game = None` in the customization hook — the leaders run uncapped,
  which is the opposite of the archived `target_325` path in §3.

## 1e. Reference measurement: the author's own decoded run

`dfranzen` publishes `.per_game_cards.pkl` as kernel output, and it records one card per game
in the form `won|gave_up:<actions>:<levels_completed>:<game_RHAE_0-100>` — one segment per
pass. Decoding his 4-pass run on the 25 **public** games
(`python3.12 scripts/parse_milestone2_cards.py <file>`, pulled 2026-10-19):

| Metric | Value |
|---|---|
| mean best-pass RHAE over 25 public games | **61.89** |
| games scoring 100.0 | 9 |
| games solving ≥1 level | 24 / 25 |
| games fully won in at least one pass | 11 |
| worst game | `sk48` at 0.00 (4 passes, 0 levels) |
| typical actions/game on a won game | 100–300 |

This is the number to compare our own runs against, because it is computed on the same 25
games by the same scorer. It is **not** a leaderboard score: the hidden rerun plays ~110 games
and the notebook forces `bm.n_passes = 1` when `KAGGLE_IS_COMPETITION_RERUN` is set.

It also puts §7's local search numbers in perspective: best-of-5 ensembled novelty search on
those same 25 games reached **0.86**, never clearing level 1. The served-LLM harness is ~70x
that on the identical set. The gap is the model and its action proposals, not the search
strategy, which is why the local search loop is no longer where effort should go.

Pass-to-pass spread inside one game is wide (`cd82`: 58.6 → 99.4; `m0r0`: 0 → 100), so a
single-pass hidden run is a one-shot draw from that distribution. Expect our submission to land
below the 61.89 best-of-4 figure, and treat any demo-set mean from a 1-pass run as noisy.

## 1f. Our reproduction of it — first real GPU measurement in this repo

Version 2 of `vyassenthilkumar/arc3-milestone2-flashnext-duck` ran the patched notebook
end-to-end on the RTX Pro 6000 on 2026-10-19: **COMPLETE** after ~8.4 h, ~9 GPU hours burned.

| Metric | Ours | `dfranzen` reference |
|---|---|---|
| mean best-pass RHAE, 25 public games | **59.82** | 61.89 |
| games at 100.0 | 9 | 9 |
| games solving ≥1 level | 24 / 25 | 24 / 25 |
| games won in some pass | 11 | 11 |

Per-game deltas (`artifacts/milestone2_demo_run.json` holds both runs at pass granularity):
`tn36` 10.7 vs his 100.0 and `g50t` 0.0 vs 15.6 are our losses; `sc25` 96.9 vs 27.9 and
`ls20` 65.9 vs 37.9 are our wins. The aggregate matches to within one game's swing, which is
the expected noise level for best-of-4 sampling, and confirms the cp312 wheelhouse plus a 3.12
venv is a faithful reproduction of the leader stack.

For scale on the same 25 public games: this repo's local search-only ensembling reached **0.86**,
and the archived Duck/TAAF offline mean this repo used to quote was **3.79** (never verified on
the leaderboard). The demo-set reproduction is ~34x the account's then-best hidden score of 1.76
— but a demo mean is not a leaderboard score, since an offline run writes a dummy
`submission.parquet` (§1d) and plays only the public games.

**Hidden-set rerun, submitted 2026-10-19 16:43 UTC → public 30.55** (ref `57402922`, team row
"Hendric Nazarius", ~rank 510). That is 17x this repo's previous best of 1.76, and 29 points
below the 59.82 the same code produced on the 25 public games. The difference is structural, not
bad luck: the rerun forces `bm.n_passes = 1`, so the hidden set gets one draw per game where the
demo set kept the best of four, and the demo cards show how wide that distribution is
(`tn36` 10.7 vs the author's 100.0; `sc25` 96.9 vs his 27.9).

Two operational facts measured by that submission:

- A rerun took **9 h 14 min** from submit to score (16:43 UTC → 01:57 UTC). Budget for that
  latency; the CLI only reports `PENDING` the whole time, and `kaggle kernels status` keeps
  reporting the *interactive* session, so there is no progress signal until the score appears.
- The rerun did **not** draw on the account's GPU quota: 9.00 h used / 21.00 remaining before it
  and unchanged after. Only interactive runs cost session hours, so a submission is free in
  quota terms and costs only one of the daily (team-shared) slots.

Where that leaves the score: raising `n_passes` is not available on the hidden set — the
notebook's own submission branch sets it to 1, and the 9 h wall is what forces that choice.

## 1g. Pass anatomy: what the 30.55 is actually made of

`artifacts/milestone2_demo_run.json` holds 100 passes (25 games x 4) for both runs, so the
single-draw distribution is measurable rather than guessed.

| Quantity | Ours | Author's reference |
|---|---|---|
| mean **best-of-4** (what a demo run shows) | 59.82 | 61.89 |
| mean **expected single draw** (what a rerun scores) | **45.85** | 46.49 |
| loss to the pass lottery | 13.97 | 15.40 |
| mean per-game spread across its 4 passes | 32.2 | 35.7 |

- **The private set is 110 games, 55 scored to the public leaderboard and 55 to the private one**
  (competition `Evaluation`/`data-description` pages). Our visible 30.55 is therefore the public
  half, and comparing it to the 45.85 public-development expectation mixes two different sets.
- **The reference kernel's author is at 38.85** on the same hidden set with code I diffed as
  identical (same 498,799-char harness patch, same analyzer env). So ~8 points of our deficit is
  single-draw noise, not a porting error.
- **Nothing newer is public to port**: `Tufalabs/duck-harness` is frozen at `7652836` (2026-07-01).
  The 68.91 / 67.49 rows are the harness authors' private work. The 106-vote "Try merging it" (V6)
  kernel is byte-for-byte the same patch and env as ours, and differs only in offline config —
  which is where the proxy below comes from.
- **The loss is completion, not efficiency.** Of our 100 passes: **70 ended with levels
  unfinished**, only **11 lost points to action cost**, 19 were clean 100s. On `ar25` we spend
  ~300 actions where the human baseline totals 648, i.e. already past the `(human/used)^2` cap.
  So cutting actions buys nothing on games we already finish; the score lives in levels reached.
- **A hard wall, not the lottery**: ~8 of 25 public games stall at level 1-3 in *every* pass —
  `sk48` 0/8, `bp35` 1/9, `g50t` 0/7, `sp80` 1/6, `lf52` 2/10. Identical across passes means the
  solver is blind to those levels, and they contribute ~2 points of the 45.85.
- **Re-roll arithmetic** (Monte-Carlo over the real pass data, 4000 sims): best-of-2 = 52.7,
  best-of-3 = 55.6, best-of-5 = 58.1 on public. Scaled to our observed hidden/public ratio that is
  ~**37 hidden after 3 submissions** — free points, but it does not reach 50. Reaching 50 needs
  the wall broken, not the lottery harvested.

## 1h. Fast proxy: a hidden-like measurement for ~1 h of GPU and no slot

`scripts/patch_milestone2_fast_proxy.py` rewrites only the notebook's **offline** branch to
`bm.n_passes = 1` and `bm.solver.max_runtime_s_per_game = 25*60` (from 4 passes at ~8 h each,
which is why §1f took 8.4 h). One draw per game, 25 games at concurrency 25, so the run is
model-load plus ~25 minutes. The `if TRUE_SUBMISSION:` branch is untouched, so a patched kernel
still submits exactly what ref `57402922` submitted. `--restore` puts the file back byte-identically.

Why it matters: the offline branch already sets `bm.solver.save_request_logs = True`, and the
harness then writes `<game_id>_requests.jsonl` per game (one JSON object per model call:
`messages`, `tools`, `event` request/response, `finish_reason`, `analysis_step`, `action`, `usage`)
plus `.per_game_cards.pkl`. Downloading the kernel output and running
`scripts/analyze_proxy_traces.py <dir>` sorts games by proxy score and reports, per stalled game,
how many model calls it spent, how many distinct actions it ever tried, its longest identical-action
streak (a no-op loop), truncation/`finish_reason` counts, and the last assistant turn with its tool
calls and tool results. That distinguishes "the model never found the right action" from "it looped
on a no-op" from "it timed out mid-level" — which have different fixes.

It costs ~1 h of interactive GPU quota (21.00 h remain, pool refreshes 2026-10-24) and **no**
submission slot: an offline run writes a dummy `submission.parquet` (§1d) and is never submitted
unless we run `kaggle competitions submit` against it.

**Version 4 (2026-10-20) played nothing, and the reason was not the proxy config.** The notebook
gates the benchmark on `SERVER_STARTUP_TIMEOUT = 12 * 60` measured from *notebook start*, while the
185 GB model precache alone took **2157 s**. So the log's `DEADLINE at 721s from notebook start;
releasing the benchmark with the server still loading` was followed by `ConnectionError` on port
8001 for every game, `analyzer failed 11 times in a row; giving up`, and a summary of
`mean score: 0.00` with **0 actions and 0 tokens** on all 25 games. Patching `n_passes` and the
per-game runtime was never the problem: the startup gate is fixed and independent of them, and §1f
only worked because that machine had the model warm. The proxy patch now raises the gate to 60 min
as well (`startup` anchor); with a cold cache expect ~40 min of load before the first token, so a
proxy run is ~1.5-2 h, not 1 h.

A run that scores 0.00 everywhere is distinguishable from a genuinely weak config by exactly those
two fields — `actions=0, tokens=0` means it never talked to the model, whereas stalled games still
show hundreds of actions. Check that before drawing any conclusion from a proxy mean.

## 1i. Three proposed optimizations, checked against the harness

Checked because each was pitched as a large score multiplier. Two are already in the served
stack, one rests on a premise we measured the other way.

**1. "Retain reasoning history — 6x fewer repetitive tokens, 3x score (13.3 -> 38.3)."**
Already on, three ways: the kernel sets `ARC3_REASONING_HISTORY_KEY = reasoning_content` (the
harness defaults it to `""`, i.e. off, so this is ours/the author's change, not a default),
`LOCAL_ANALYZER_ENABLE_THINKING = true`, and the server is launched with
`--default-chat-template-kwargs {"preserve_thinking":true}`. `ARC3_PRESERVE_THINKING` defaults to
`1` in the harness, so it is on even where unset. There is no Responses API to migrate to: the
analyzer talks to `/v1/chat/completions` on a local SGLang server, and the harness's reasoning
arms (`ARC3_STUB_DEAD_REASONING`, `ARC3_REASONING_EFFORT_LADDER`, `ARC3_SUMMARY_*`) are the
equivalent lever. Expect ~0 points, not 3x.

**2. "Safe-exit prompt for hard games, because `(human/used)^2` punishes wasted actions."**
The arithmetic is inverted. A level left unfinished scores 0 no matter how few actions were spent,
so exiting early can only lose points, never gain them. Of 100 recorded passes 70 ended with levels
unfinished and only 11 lost points to inefficiency, and on ar25 we already spend ~300 actions where
the human baseline is 648, i.e. past the cap anyway. The scarce resource is wall-clock per game,
which is a scheduler concern (`ARC3_PRIORITY_*`, partly set), not a politeness concern.

**3. "Replace the raw grid with a `(0,0)[Blue] ...` coordinate overlay."**
The premise is false: the harness never sends the raw matrix. `prompts.py` states "The raw numeric
grid is intentionally not exposed. Use `current_frame.segmentation` as your primary view", and the
tool already returns objects with `boundary` `[row, col]`, `pixels`, cross-frame position-invariant
hashes and an adjacency list, plus `current_frame.ascii` for small regions and a multimodal grid
image (`ARC3_DIFF_IMAGE`, `ARC3_GAMEOVER_DIFF_IMAGE` are set). Adding a per-cell text overlay would
reintroduce the token cost the segmentation view exists to avoid.

## 1j. The free search space: 85 harness knobs we never set

The kernel's `setup_env` assigns 36 entries plus ~7 in the priority-scheduling update; counting the
`ARC3_*` / `LOCAL_ANALYZER_*` names that appear in the harness source and the embedded patch gives
~130, so **~85 run at their default**, and the defaults are mostly off
(counted by grep, not audited one by one — the point is the order of magnitude):

```
ARC3_NOOP_REPEAT_GUARD  ARC3_REPEAT_STATE_GUARD  ARC3_DEATH_REPEAT_GUARD  ARC3_DEATH_LEDGER
ARC3_DEATH_GUARD_LOOKAHEAD  ARC3_NOOP_GUARD_BORDER  ARC3_PREFER_TOOL_CALLS
ARC3_STUB_DEAD_REASONING  ARC3_OPENER_COMMIT_HINT  ARC3_STEP_VERIFICATION_HINT
ARC3_AUTO_FRAME_DIFF(_BUDGET)  ARC3_LEVEL_INVENTORY  ARC3_RESUME_DIFF_IMAGE
ARC3_REPORT_GAMEPLAY_CHANGED  ARC3_BOARD_DUMP_GUARD  ARC3_CONTEXT_DRAIN_FLOOR
ARC3_WM_*  ARC3_SUMMARY_*  ARC3_PRUNE_*  extra ARC3_PRIORITY_*
LOCAL_ANALYZER_SEED  LOCAL_ANALYZER_TOOL_CHOICE  MULTIMODAL_UPSCALE/RESAMPLE
```

This is the only part of the public stack that has not been tried: the upstream repo is frozen at
`7652836` (2026-07-01), the author's kernel and `jingshuifu/arc3-m2-solution` (2026-10-11) carry a
byte-identical 498,799-char patch with the same env set, so there is no newer code to port. The
proxy run (§1h) is the cheap way to test them: `--arm` writes them into `setup_env`, one flag per
knob, and `--restore` returns the notebook byte-identical to the unpatched config state
(verified against commit `d0bf2b7`, which is the last commit whose notebook has no proxy edits).

Set `LOCAL_ANALYZER_SEED` (harness default `-1`) whenever comparing two arm sets, otherwise a
better-looking number may be the same sampling lottery that moved one game between 10.7 and 100.0
across passes of identical code.

## 1k. The scored branch is the only thing a submission can test

The two branches of the config cell are disjoint: `if TRUE_SUBMISSION:` runs on the hidden rerun and
`else:` runs in every interactive session. A patch that only touches `else:` (the §1h proxy, the
arms in `setup_env` are visible in both but their effect is measured offline) decides **nothing**
about what Kaggle scores — the rerun ignores it. Conversely a scored-branch change cannot be tested
in the proxy at all, because the proxy is by definition the other branch.

`--scored-concurrency N` therefore rewrites only the `TRUE_SUBMISSION` pair, and version 8 is the
first submission this repo makes on that evidence:

```
bm.solver.concurrency = 28                    # was 120, "game count should be 110"
bm.solver.max_runtime_s_per_game = 7920       # was 532*60 = 31920
```

7920 s is derived, not copied: the session is 32400 s, ~720 s of it goes to model load, and a game
that never finishes holds its slot until its cap, so the worst case is `ceil(110 / 28) = 4` waves and
`(32400 - 720) // 4 = 7920`. It lands on the same number ranks 3/7/8 publish, which is reassuring
about the arithmetic rather than evidence the value ports — they run vLLM + FP8, we run SGLang +
W4A16 181 GB.

Why 120 is suspect: it is one stream per game, so no game is ever *unserved* (the failure §1h hit at
derived-25), but every one of the 110 games competes for the same GPU for the whole session. The
author's own comment `# game count should be 110` admits the number was never tuned for this set.
Trading guaranteed-shallow coverage of 110 games for deep coverage of ~28 at a time is the bet, and
the scoring formula rewards it: a game earns credit per level completed, and level 2+ is unreachable
at ~250 actions per game.

The risk in the other direction is real and is what the number must be read against: with 28 slots
and 110 games, the tail games only get played if earlier ones finish before their cap. If the
submission comes back far below 30.55 with several games at 0 levels, the tail was starved and 28
was too aggressive, not too conservative.

**Verdict (ref `57508879`, scored 2026-10-22): 24.65.** That is 5.9 below the 30.55 the same code
scores at concurrency 120, and larger than the pure-noise roll we measured (26.74, same config). So
the starved-tail branch of the prediction is the one that happened: on this set, breadth beats depth,
and 120 streams — one per game, however shallow each one stays — is the right end of that axis. The
axis is now closed: any `concurrency < 110` gives up coverage of games that all contribute to the
mean, so do not spend another slot between 28 and 120. Version 9 reverts the scored branch to
`120` / `532*60` exactly and is the config every later slot should start from.

Two changes ride along with it, both defensive rather than experimental: the health gate stays at the
60-minute `SERVER_STARTUP_TIMEOUT` (with the author's 12 minutes the version-4 session released the
benchmark against a port nothing was listening on, because the 185 GB precache alone took 36 min, and
all 25 games came back `actions=0, tokens=0` for a mean of 0.00 — §1h), and `n_passes` stays 1.
No `setup_env` arms are set, so the analyzer sees exactly the configuration that scored 30.55.

## 1l. What is actually left, in expected-value order

The two measured negative results bound the search: the concurrency axis is closed (§1k, 24.65 at
28 vs 30.55 at 120) and the leaders' analyzer env is closed (§1j's v6 probe, served-subset
14.03 → 11.62). What remains, cheapest first:

1. **Re-roll the baseline.** The leaderboard keeps the team's best, so each roll of ref `57402922`'s
   config is upside-only. Observed rolls of (nearly) identical code: 30.55, 26.74, and the noise is
   roughly ±3. With ~9 slots left before 2026-11-02 this is worth a few points, not twenty.
2. **Raise throughput per stream, not the number of streams.** At 120 streams every game is served
   but each one gets ~1/110 of the GPU, which is why level 2+ stays out of reach. The only lever that
   changes that product is a different serving stack on the same card. Ported — see §1m.
3. **Semantic harness arms** (`ARC3_DEATH_LEDGER`, `ARC3_REPEAT_STATE_GUARD`,
   `ARC3_LEVEL_INVENTORY`, `ARC3_STEP_VERIFICATION_HINT`, `EXPOSE_RESET=on`). They can be proxied for
   free-ish, but the proxy runs at offline concurrency and only measures *semantics* — v6 showed a
   proxy win/loss does not transfer to the scored branch's throughput regime. Treat a proxy gain here
   as permission to spend one slot, not as a prediction.

## 1m. The throughput port — `kaggle_submit_milestone2_vllm/`

Item 2 above is a serving-stack replacement, so it is a second kernel, not a patch of the first:
[`kaggle_submit_milestone2_vllm/`](../kaggle_submit_milestone2_vllm/) stages
`vyassenthilkumar/arc3-milestone2-vllm-port` from the generic TAAF launcher
`lordhansolo/built-on-tufa-labs-duck-harness-milestone-2` (10 cells, 15.7k chars — it contains no
model code at all; everything lives in its attachments).

What the attachments actually are, verified against the Kaggle API on 2026-10-23:

- `lordhansolo/taaf-kaggle-source` — the harness bundle (`benchmark_initial.pkl`, `deploy_target.pkl`,
  `setup_commands.json`, `preamble.txt`, `inference.json`).
- `lordhansolo/vllm-main-e975732-arc3` — 7,934,814,402 B. **A patched vLLM build shipped as a data
  set**, applied as an overlay tar into `/tmp/vllm-image-runtime`, because internet is off and the
  upstream image `vllm/vllm-openai:nightly-e9757321527…` cannot be pulled. Commit `e97573215`,
  version `0.29.1rc1.dev573+ge97573215`, patched in the KV-cache coordinator and the mamba/GDN
  hybrid model-runner paths.
- `lordhansolo/qwen3-8-flash-next-mixed-nvfp4-fp8/PyTorch/hf-mixed-mtp-nvfp4/1` — 180,166,090,115 B
  on disk, status READY. **Not** 180 GB in VRAM: the files are `carry-model-bf16-*.safetensors` plus
  one `ct-experts-layerNN.safetensors` per MoE layer, and setup runs a shard-prefetch step that
  reports into `vllm-shard-prefetch.json`. Experts stream from disk, the rest stays resident — that
  is how a bigger checkpoint earns the same card.

Server block (from the bundle's `inference.json`): port 1234, `max_model_len 147072`,
`kv_cache_dtype fp8_e4m3`, `max_num_seqs 14`, `max_num_batched_tokens 2048`, `mtp_tokens 3` with
`configs/draft_vocab_32k.json`, `enable_prefix_caching true`, `tool_call_parser qwen3_coder`,
`reasoning_parser qwen3`, `preserve_thinking true`, `reasoning_effort xhigh`,
`gpu_memory_utilization 0.98`, plus a restart watchdog.

Solver block (from `preamble.txt`): `n_passes=1`, `concurrency=14`,
`max_runtime_s_per_game=3918.0`, `analyzer_timeout=900.0`, 25 offline games.

Do the wave arithmetic before believing any scored run of it: `ceil(110/14) = 8` waves × 3,918 s =
31,344 s against a 32,400 s session, which leaves ~1,050 s for pip, the overlay and the model mount.
The final wave is at risk of the hard stop; it only bites if enough games actually run to the cap,
since a finished game releases its slot early.

Staging is reproducible and touches exactly one cell — the upstream's empty "Inline customization
hook" (cell 8), which is where its authors intended per-run edits:

```powershell
kaggle kernels pull lordhansolo/built-on-tufa-labs-duck-harness-milestone-2 -p .\upstream ; python .\scripts\build_vllm_port.py ; kaggle kernels push -p .\kaggle_submit_milestone2_vllm
```

`enable_internet` stays **off**; the model and runtime arrive through `model_sources` /
`dataset_sources`, and `machine_shape` is the same `NvidiaRtxPro6000`.

**Feasibility gate (v1, 2026-10-23).** The hook trims offline play to 4 games × 600 s behind
`if not run_as_submission:` — cell 7 defines that flag and cell 9 never lets a competition rerun
enter the branch, so the hidden set keeps the bundle's own 14/3918 config. Read the log for, in
order: the overlay applying, `vllm-openai-server.ready`, the bundled Qwen smoke test, then actions
and turn latency per game. Two questions decide the next slot: does the per-stream turn rate beat
our SGLang 120-stream number, and if so whether to keep `concurrency = 14` (the leader's own scored
config) or widen it — at `max_num_seqs 14` a 110-wide fan-out only queues behind the server.

**Never submit the feasibility version.** Its offline branch writes a dummy one-row
`submission.parquet` (`row_id=1_0`, `score=1`); that scores ~0 and burns the day's slot.

**v2 measured on the real stack (2026-10-23, `artifacts/vllm_port2_full/`).** The `/usr/bin/python3.12`
fork works — the port printed `setup interpreter -> /usr/bin/python3.12` at t=26 s, the overlay
installed, the server became ready, and 6 trimmed games were played against it. Numbers, all read
out of that run's logs:

| Quantity | Measured |
|---|---|
| Boot, session start → first game | **582 s** (overlay extraction 159 s, weights 295 s, engine profile+KV+warmup 67 s, graphs 17 s) |
| Model resident | 71.94 GiB weights + 0.81 GiB activation + 0.46 GiB graphs, of 94.43 GiB free |
| KV pool | **1,417,100 tokens** = 19.69 GiB at `fp8_e4m3` |
| KV pool ceiling | `--kv-cache-memory=21,939,971,584` (20.43 GiB) is the most the card can give it: **+3.8 %**, so `gpu_memory_utilization` (already 0.98) is not a lever |
| Max concurrency at full 147,072-token requests | 9.64× |
| Occupancy with 6 streams | 35-40 % of the pool → **~90 k tokens resident per stream** → ~15 streams is the physical wall, which is why the bundle pins `max_num_seqs 14` |
| Generation throughput | 345-651 tok/s aggregate at 6 streams (~570 mean), prefix-cache hit 93-96 % |
| Request latency at 6 streams | 273 requests, median **11.4 s**, p90 38.9 s, max 185.6 s, 1.8 requests per turn → engine capacity ≈ **0.31 requests/s** |
| Cost of play, per action | **33,747 prompt tokens** (93 % cache-hit, so ~2.4 k actually prefilled) and **1,172 completion tokens** — the run is prefill/step-budget bound, not decode bound |
| Score on 6 public games × 900 s | mean 4.97, levels 1.0-3.0 of 6-10 |

Three consequences for the schedule, the last two of which correct earlier notes here:

- The wave arithmetic is not the same argument as the leader's. Raising the *harness* fan-out does
  not raise the *server's* concurrency — vLLM still admits `max_num_seqs` requests and the rest
  queue. It only makes every game eligible from the first second, which is what our own SGLang data
  says wins (120 streams = 30.55, 28 streams = 24.65), so the scored branch fans out to 110.
- `VLLM_MAX_NUM_SEQS` cannot be pushed to the fan-out. The captured graph ladder is
  `[1,2,4,8,16,24,32,40,44,48,52,56]` = `max_num_seqs × (1 + mtp_tokens)` topped at 56, and the KV
  pool only holds 15 streams at the observed 90 k tokens each. Widening the server needs the
  per-request context cut first, not the other way round.
- The "1,166 tokens per action" number that `summary.txt` prints is **completion tokens only**
  (468,961 / 402). The real per-action cost is ~29× that, and it lands on the engine's
  token-per-step budget (`VLLM_MAX_NUM_BATCHED_TOKENS = 2048`). So the quantity that caps the score
  is requests/s, and the levers that can raise it are a bigger step budget or a smaller context —
  not more streams, and not a faster decoder.

**v3 failed to prove anything, and the reason is a schedule bug (`artifacts/vllm_port3/`).** The
run's `summary.txt` is 1 s long, zero actions, every game `state=cancelled`. The log gives it away:
`deploy.kaggle: soft_end_time = 2026-10-23 05:02:14` while the server only became ready at
t=**1,974.5 s**. Two facts collide:

- The model mount is not a fixed cost. v2 booted in 582 s on a warm node; v3 paid 1,975 s on a cold
  one. Budget the *cold* number.
- Cell 7 computes `soft_end = _soft_end_time(target.max_runtime_s, run_as_submission)`, and
  `_soft_end_time` returns `None` under a rerun. Offline, it returns the deploy target's own
  interactive budget (~2,000 s), so `taaf/benchmark.py:169` installs `_cancel_at(soft_end, ...)` and
  cancels every game the instant the run starts. A validation run on a cold node therefore looks
  identical to a working stack that never got to play.

**Current port version to submit: 4** — `--scored-concurrency 110 --scored-load-s 3600
--offline-games 25 --offline-cap-s 600`. The scored branch fans out 110 streams × 28,800 s against
the cold-boot budget, keeps the stock server config, and re-arms a session soft end at
`t0 + 31,200 s`; the offline branch rehearses all 25 bundled games at 25-deep queueing with a soft
end at `t0 + 10,800 s`, so a cold mount no longer fakes a failure.

```powershell
kaggle competitions submit arc-prize-2026-arc-agi-3 -k vyassenthilkumar/arc3-milestone2-vllm-port -v 4 -f submission.parquet -m "vLLM port: 110 streams x 28800s one-wave, cold-boot-safe"
```

Run it from this repo root: `-f submission.parquet` must resolve to a file that exists locally, and a
missing one fails the CLI before the API is called. The committed content is irrelevant to a code
competition -- the notebook writes its own columns -- so `submission.parquet` at the root is a 1-row
placeholder, gitignored because it is a submit prop and not data.

Rebuilding with the flags above produces a file that is not byte-identical to the pushed version 4:
the stager gained `import re` and an `--env` block (both no-ops when no `--env` is passed), plus an
extra `port: offline schedule = …` print in the `else:` branch. Verified against `kaggle kernels pull`
of the pushed notebook, the only fork-cell difference is that unused `import re` line and the scored
branch is character-for-character what version 4 runs — so version 4 stays the version to submit, and
pushing the local file would only create a version 5 that scores the same thing while costing another
42-minute rehearsal.

Version 4 is its own rehearsal: push it, read the offline log for turn latency at queue depth 25,
and only then let the scored branch run. The scored rerun executes the `if run_as_submission:`
branch only, so validating the offline branch does not change what gets scored — but it does expose
whether a deep client queue starves a game past `analyzer_timeout = 900 s`, which is the one way
110-stream fan-out can fail on a server that still admits 14 sequences.

**v4 rehearsal result (2026-10-23, `artifacts/vllm_port4/`, 42 min wall, 25 games × 600 s at depth
25).** It played, so the schedule is sound. Read out of that run:

| Quantity | Measured at depth 25 |
|---|---|
| Fork + interpreter | `port: child interpreter verified 3.12` at **t=9.8 s** — the pin survives container start |
| Cold boot, session start → server ready | **1,865 s** (image runtime at 754.9 s, engine KV + warmup to 1,864.9 s). v2's 582 s was a warm node |
| Hook actually applied | cell 10 printed `soft_end_time = 2026-10-23 08:26:16` = t0 + 10,800 s (the re-armed value), and play ran 605 s against the 600 s cap |
| Server occupancy | `Running: 14`, `Waiting: up to 23` — the client queue keeps every admitted slot full, KV usage median 28.6 %, max 43.5 %, **zero preemptions** |
| Aggregate decode | **967 tok/s** median (max 1,121) vs 538 at 6 streams — fanning out wide is what fills the 14 slots |
| Request latency | median **30.4 s**, p90 58.0, p99 116.1, max **158.7 s** |
| Request errors | 25, all `Read timed out` with `timeout =` the game's *remaining budget* (103, 91, 90, 75, 61 … 2 s) — end-of-cap bookkeeping, not queue starvation |
| Cost per action | 896 completion tokens (96 % of them reasoning) and **12,579 newly-prefilled** prompt tokens (88 % of the prompt was cache-hit) |

Two decisions this closes, both by arithmetic rather than taste:

- **110-deep is safe.** Extrapolating the queue 25 → 110 multiplies the wait by 4.4: median ~134 s,
  worst observed 158.7 s → ~700 s, inside the 900 s ceiling with ~1.3× margin. Each game will still
  execute its first request per turn, because `LOCAL_ANALYZER_YIELD_SECONDS` is only checked *between*
  requests (`tool_agent.py:1484`), and the per-request timeout is already `min(900, budget left)` — a
  game at the end of its budget times out early by design instead of hanging.
- **Do not widen the server, and do not cut the context.** At depth 25 the engine served 1.01
  actions/s, which over the scored session's 28,800 s across 110 games is **~264 actions per game** —
  already sitting on the measured score-vs-actions peak (~300 → 75.6, 405-655 → 51.6). Raising
  `max_num_seqs` to 28 with `LOCAL_ANALYZER_TARGET_CONTEXT=40960`, or trimming prefill to buy more
  actions, pushes past the peak, where the extra actions cost efficiency on levels already reached and
  buy no new levels. `min(human/agent, 1)²` is capped at 1.0, so throughput only pays when it clears
  levels — and clearing levels is the capability wall this stack does not move.

**So the remaining slots are: submit version 4 on Oct 24, then re-roll the same version on Oct 25 and
Oct 26.** A re-roll of an already-submitted kernel version is upside-only (the leaderboard keeps the
team's best row) and is worth more than an unproven arm, given the two bullets above.

What "re-roll, don't experiment" is worth in absolute terms, measured off the public leaderboard at
full pagination on 2026-10-23 (1,543 scored teams): 4 teams are above 50 (68.91, 67.77, 52.82, one
more in 50-55), 9 above 40, 48 above 35, and our current best row of 30.55 sits at rank 531 -- 625
teams are packed into 30-35 and 871 into 25-30. So 50 is not a scheduling target, it is the top 0.3 %
of the field and a different capability tier than this port reaches; the honest win condition for the
three remaining slots is to convert 30.55 into the best roll of version 4 the field can see, which the
spread of our own identical-code rolls (30.55 / 26.74 / 24.65) says is worth several points.

The fourth draw arrived the same day: ref `57547459` -- the 120-stream baseline config, re-rolled with
nothing changed -- scored **28.18** at 17:47 UTC. Excluding the deliberate `concurrency 28` experiment
that lost at 24.65, the same code has now rolled 30.55, 26.74 and 28.18, i.e. mean 28.5 and spread ±2.
That is the true size of a re-roll: version 4's *mean*, not its luck, is what has to clear the field, so
if version 4 lands near the baseline's 28 rather than the 35-40 the throughput arithmetic predicts, the
remaining slots are worth more spent on a different arm than on more draws.

The audit-interval patch (`--audit-interval-s`) is deliberately **off** in version 4. The audit timer
is wall-clock (`time.monotonic() - _strategy_audit_started_at`, `inference/framework/solver.py:244`)
and a one-wave game spends most of its life queued behind the server's 14 slots, so a 300 s cap would
stamp `STRATEGY_AUDIT_PROMPT` onto nearly every turn. At the default 25 % of a 28,800 s cap each game
gets ~4 audits, which is the audit density the author's own 38.85 run had per game.

### Version 5 is staged locally as the one experiment arm worth a slot

Because the baseline's own re-roll has now landed at 28.18, more draws of a config that merely matches
it cannot reach 50, so the working tree holds a second candidate: version 4 plus **one** variable, the
analyzer's sampling temperature.

```powershell
python .\scripts\build_vllm_port.py --scored-concurrency 110 --scored-load-s 3600 --offline-games 25 --offline-cap-s 600 --env LOCAL_ANALYZER_TEMPERATURE=0.9
```

`git diff` of the staged notebook against the committed version 4 is 22 added lines and **zero deleted
ones**: the schedule, the soft ends and the fork's server-override logic are character-identical, and the
only new behaviour is the `setup_env` rewrite block. Verified against `upstream/lord_bundle/setup_commands
.json` before pushing anything: exactly one line matches, `'LOCAL_ANALYZER_TEMPERATURE': '0.6',` →
`'0.9',`, and the patched 72,822-char setup heredoc still parses as Python. Temperature is the arm to
spend a slot on because it costs *nothing* in the throughput arithmetic -- same token budget, same KV
occupancy, same 28,800 s wave -- and it targets the one failure the traces actually show: single-action
monotony (tn36 pressed one action 73 times in a row) while the audit prompt explicitly forbids abandoning
a plan. The schedule arms (`max_num_seqs`, target context) and the audit cadence are all closed above.

**Kaggle still had version 4 as its latest push until 2026-10-24 17:07 UTC**, when version 4 was finally
submitted by hand (ref `57601341`). The two automations scheduled for that day at 03:40 and 13:30 UTC
never produced anything: the allowance had not rolled over at 03:40, the submit task exhausted its 4-hour
retry window, and the push task -- gated on a submission row that therefore did not exist -- skipped
itself. Both then deleted themselves as `deleteAfterRun` one-shots, which is why the schedule below moved
to the afternoon UTC and widened the slot retries to 8 hours.

| Slot | Action | Decided by |
|---|---|---|
| Oct 24 17:07 UTC | `-v 4` submitted (ref `57601341`) | first read of the port's real hidden-set mean, due ~02:00-04:00 UTC Oct 25 |
| Oct 25 04:00 UTC | `kaggle kernels push` (no submit) | version 5's ~42-min offline rehearsal must show `port: analyzer env -> …'0.6' -> '0.9'` and nonzero actions; it waits out the version-4 rerun rather than overlapping it |
| Oct 25 09:00 UTC | `-v 4` if version 4 scored ≥ 35, else `-v 5` | ≥ 35 means the port beats the baseline's best roll, so another draw of it is the better bet; below that a draw of 28.5-mean code is not, and the slot goes to the experiment |
| Oct 26 10:00 UTC | the untried arm if version 5 never scored and rehearsed clean, else a re-roll of the higher-scoring arm | last slot before the entry deadline; nothing is submitted if the port scored below 20 |

### Which flag moves which knob

The bundle ships exactly **one** setup command, a 72,864-char heredoc that defines every constant and
then writes `taaf_setup_env.json`. Reading it is the only way to know which lever is real:

| Wants to change | Flag | Why |
|---|---|---|
| `VLLM_MAX_NUM_SEQS = 14`, `VLLM_MAX_NUM_BATCHED_TOKENS = 2048`, `VLLM_MTP_TOKENS = 3`, `ANALYZER_CONTEXT_WINDOW = 127488` | `--server NAME=VALUE` | unquoted integer assignments; the fork rewrites the line and raises if the name is absent |
| `LOCAL_ANALYZER_TARGET_CONTEXT`, `_TOOL_OUTPUT_TOKENS`, `_TEMPERATURE`, `_YIELD_SECONDS`, … | `--env NAME=VALUE` | they are *quoted dict values* in the `setup_env` block near the end of the heredoc |
| anything else the child processes read | `--arm NAME=VALUE` | exported into the notebook's `os.environ` before setup |

`--arm` does **not** work for the analyzer knobs: cell 5 re-merges `taaf_setup_env.json` over
`os.environ` after the setup command runs (`env.update(_load_setup_env())`), and the heredoc rewrites
that file, so the bundle's value always wins. That is the trap the `--env` flag exists to avoid.
`VLLM_GPU_MEMORY_UTILIZATION = '0.98'` is quoted, so `--server` will refuse it — which is correct,
since the profiler already shows the KV pool has +3.8 % left to take.

Two runtime facts that bound any deep fan-out, both read out of the bundle:

- Each request's timeout is `min(analyzer_timeout, per-game budget left, soft-deadline left)`
  (`_HarnessGameSession.request_timeout_seconds`, `inference/framework/solver.py:358`), so at
  `analyzer_timeout = 900` a queue that makes a single request wait longer than 900 s produces
  `Read timed out` + `analyzer_retry` for every game at once. At the measured ~0.31 requests/s and
  110 streams the expected wait is ~355 s, i.e. ~2.5× headroom — real but not huge, and it is what
  the 25-deep rehearsal measures directly.
- The cooperative yield is safe: `LOCAL_ANALYZER_YIELD_SECONDS = 150` is only checked *between*
  requests inside a turn (`tool_agent.py:1484`), with a budget of
  `150 × min(1 + consecutive_yielded_turns, 3)`, so a queued game still executes the actions of its
  first request instead of discarding a generation.

## 1n. Daily loop — what it is now, and why it changed on 2026-10-25

The loop was written on Oct 24 to author one new single-variable **port** arm per day. Two scored port
draws killed that design, so on Oct 25 it was rewritten to draw the proven arm instead of editing it.
The record below is the reason.

| Date (UTC) | Kernel / version | Arm | Submitted ref | Score |
|---|---|---|---|---|
| 2026-10-24 | port v4 | 110 streams × 28,800 s, stock server, temperature 0.6 | `57601341` | **5.68** |
| 2026-10-25 | port v5 | v4 + `LOCAL_ANALYZER_TEMPERATURE 0.9` | `57621235` | **5.34** |
| 2026-10-25 | duck v10 | `analyzer_timeout 900 → 1800` on the 30.55 config | staged, Oct 26 slot | — |

**The port is closed, and it was never one variable.** Its `kernel-metadata.json` differs from the
baseline's in three attachments at once: harness `lordhansolo/taaf-kaggle-source` instead of
`pennyroyal-v253`, model `qwen3-8-flash-next-mixed-nvfp4-fp8` instead of
`intel-qwen3.8-flash-next-w4a16-autoround` + the `albucino` MTP drafter, and the patched vLLM engine
instead of the bundle's. So 5.68 vs 30.55 says the *combination* is far weaker in our hands, and the
temperature arm at 5.34 confirms the level is the stack, not the knob — a 0.3 temperature change cannot
move 0.34 points on a 110-game mean, so both numbers are the same draw from the same weak distribution.
Every remaining queued port lever (`TOOL_OUTPUT_TOKENS`, `YIELD_SECONDS`, `TOP_P`, `MULTIMODAL_UPSCALE`,
`--audit-interval-s`) is therefore worthless: they are all tuning on a stack that is 5× below the one
that scores. **Never spend another slot, push or GPU hour on `arc3-milestone2-vllm-port`.**

What replaced it, in expected-value order, is the arithmetic the loop now encodes: the scored branch of
`arc3-milestone2-flashnext-duck` is one fixed configuration (concurrency 120, `max_runtime_s_per_game`
532×60, `n_passes` 1) and its draws so far are 30.55, 28.18, 26.74 — while the notebook's own author drew
**38.85 with byte-identical code**. That pair of facts is the whole strategy: the spread of a single draw
is several points wide, the leaderboard keeps the team's best row, so re-drawing the proven config is
upside-only, and every attempt to beat it by editing it has lost (24.65 at concurrency 28, 5.68 on the
port). Eight slots remain to the 2026-11-02 final deadline.

## 1o. The `analyzer_timeout` arm, and why it was withdrawn

Version 10 of `arc3-milestone2-flashnext-duck`, pushed 2026-10-25 22:35 UTC, changed exactly one
character of the notebook JSON (`9` → `18` in `bm.solver.analyzer_timeout = 900.0`, cell 16, which the
`TRUE_SUBMISSION` block does not override). Verified that way deliberately: a whole-file text replace,
then a prefix/suffix scan proving the only difference from HEAD is that one token.

The motivation was the queue: the scored run creates 110 game tasks, and if 110 requests really queued
against a 14-sequence engine, the measured latency curve at queue depth 25 (median 30.4 s, p90 58.0,
max 158.7 s) extrapolates to ~700 s — a thin margin against a 900 s per-request cap.

**It is a no-op, and the reason is the thing the port was missing.** The baseline does not run 110 games
against the server. Its `setup_env` arms `ARC3_MAX_ACTIVE_STREAMS: 10` under `USE_PRIORITY_SCHEDULING`,
and the patch implements that as a process-wide priority gate: ten games hold a slot, and a slot frees
only when a game's context is first trimmed, which the author's own comment puts 30-50 turns in — the
staggering is deliberate, so "a prefix is reused for as long as it lasts, instead of fourteen games each
progressing at a fourteenth of the rate". At ten active streams the measured max request latency on this
engine was 185.6 s, so a 900 s timeout never binds and raising it is invisible to the score. The repo
copy is back to `900.0`; version 10 stays on Kaggle as an unsubmitted dead version so the slot is not
spent rediscovering this twice.

The gate is also the cleanest explanation of the port's 5.68: it ran 110 games flat against 14 slots with
no priority rotation, so every game advanced at ~1/110 of the engine and almost nothing cleared a level,
while the baseline gives ten games at a time real throughput and rotates them by score, pace and tail.

## 1p. Rank chase — the measured curve, and what a draw is worth

Two numbers, both read from the platform on 2026-10-25 rather than reasoned about.

**The rank gradient at our score is enormous.** Downloading the whole public leaderboard
(`kaggle competitions leaderboard … -d`; 4,833 teams; top score now **71.48**, Yi-Chia Chen @ Fields
Model, 41 submissions) gives:

| Score | Public rank |
|---|---|
| 30.55 — our best row | 581 |
| 31 | 468 |
| 33 | 160 |
| 35 | 55 |
| 38.85 | 16 |
| 40 | 13 |
| 50 | 6 |

One point near 31 is worth ~100 places; three points ~400. Rank, not score, is what makes drawing
rational here.

**The config's spread is wider than our own three samples implied, and the teams at this level run
exactly our code.** Searching the competition's kernels by most-recently-run and diffing the two
candidates cell-by-cell against ours:

- `yukailiu/arc-agi-3-champion-solution` (44 votes, run 2026-10-25) reports **31.46**. Its scored branch is
  `concurrency 120`, `max_runtime_s_per_game 532*60`, `analyzer_timeout 900`,
  `USE_PRIORITY_SCHEDULING = True`, and its 54-key `setup_env` dict matches ours key-for-key and
  value-for-value. There is nothing in it to copy.
- `rajrajak99/cprs-arc-agi-3-apex-sota-lb-31-54` claims 31.54 in its title but runs `concurrency 28` with
  `max_runtime_s_per_game = 180`. 180 seconds per game cannot produce a 31 across 110 games, so the title
  is marketing; discarded.

That is five independent draws of byte-identical code: **26.74, 28.18, 30.55, 31.46, 38.85** — mean 31.2,
spread ~4.6, with the author's 38.85 sitting at rank 16. Nothing published beats this config, which is the
strongest available argument for spending the last eight slots on draws: every edit we have made has lost
(24.65 at concurrency 28; 5.68 and 5.34 on the port) while the same code has drawn 38.85. Expected best of
eight draws from a mean-31.2 / spread-4.6 distribution is roughly 37, i.e. rank ~70-90, and because the
leaderboard keeps the maximum, the downside of any individual draw is zero.

### Rank chase log

| Date (UTC) | Ref | Arm drawn | Score | Team best | Rank |
|---|---|---|---|---|---|
| 2026-10-25 07:21 | `57621235` | port v5, temperature 0.9 — the last port draw | 5.34 | 30.55 | 581 |
| 2026-10-26 10:00 | scheduled | v9, proven config | — | — | — |

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

## 3. Archived path — capped TAAF `target_325` (1.76, superseded by §1f)

[`../notebooks/kaggle_submit_target_325.ipynb`](../notebooks/kaggle_submit_target_325.ipynb),
[`../configs/target_325.json`](../configs/target_325.json),
[`../agents/calibrated_agent.py`](../agents/calibrated_agent.py).

Correction: an earlier version of this file called this path a **negative result** and claimed
the uncapped stack scored 2.23 against its 1.76. The submission log says the opposite — the
capped run scored **1.76** and the uncapped runs 1.10 and 1.13, so 1.76 was this account's best
until the Milestone 2 port reached **30.55** on 2026-10-19 (§1f).
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
- [ ] Submit through the Kaggle UI on a committed version, then note the LB delta vs **30.55** (§1f)

## 6. Commands (interactive runs cost GPU quota; a submission costs only a daily team slot)

From the repo root, in PowerShell. Each line is standalone; no continuations.

```powershell
cd C:\hackathon\Gemini_CLI\ARC-Prize-2026---ARC-AGI-3
```

Upload the primary notebook as a new kernel version (attach the datasets/model/competition
in the Kaggle UI first, or trust `kaggle_submit_duck_v12/kernel-metadata.json`):

```powershell
kaggle kernels push -p .\kaggle_submit_duck_v12
```

For the §1c target instead (this is the one worth the GPU hours). Note the **slug is derived
from the notebook `title`, not the metadata `id`**, so it is
`arc3-milestone2-flashnext-duck`:

```powershell
kaggle kernels push -p .\kaggle_submit_milestone2
```

```powershell
kaggle kernels status vyassenthilkumar/arc3-milestone2-flashnext-duck
```

### Fast proxy (§1h) — patch, run, pull the traces back

```powershell
python .\scripts\patch_milestone2_fast_proxy.py ; kaggle kernels push -p .\kaggle_submit_milestone2
```

Then poll with `kaggle kernels status`. `kaggle_wait_and_submit.py` polls and submits the version
**you pass as argv[2]** — this competition rejects a version-less submit with "Code competition
submissions require both the output file name and the version number", so the script now refuses to
run without it. Submitting a proxy-patched notebook is safe: the `TRUE_SUBMISSION` branch is
untouched, so the scored rerun is the same code path as ref `57402922`, which is why the 2026-10-21
re-roll (v4) was filed against a proxy-mode kernel.

```powershell
kaggle kernels output vyassenthilkumar/arc3-milestone2-flashnext-duck -p .\artifacts\proxy_run --file-pattern "(_requests\.jsonl|\.per_game_cards\.pkl|run_config\.json)$"
```

```powershell
python .\scripts\analyze_proxy_traces.py .\artifacts\proxy_run --games sk48 bp35 g50t
```

Restore the 4-pass demo config before any submission you want scored like §1f (`--restore` is
byte-exact), or leave it patched — the submission branch is untouched either way.

To A/B a knob set (§1j) in the same ~1 h, add `--arm` flags before the push; re-running with a new
set replaces the previous one:

```powershell
python .\scripts\patch_milestone2_fast_proxy.py --arm ARC3_NOOP_REPEAT_GUARD=1 --arm ARC3_DEATH_LEDGER=1 --arm LOCAL_ANALYZER_SEED=7 ; kaggle kernels push -p .\kaggle_submit_milestone2
```

To change what the hidden rerun actually executes (§1k), add `--scored-concurrency`. Combine it with
a short offline branch, because `kaggle kernels push` **also starts a session** and a code submission
is rejected until that version has produced output:

```powershell
python .\scripts\patch_milestone2_fast_proxy.py --scored-concurrency 28 --concurrency 10 --runtime-minutes 5 ; kaggle kernels push -p .\kaggle_submit_milestone2
```

```powershell
kaggle competitions submit arc-prize-2026-arc-agi-3 -k vyassenthilkumar/arc3-milestone2-flashnext-duck -v 8 -f submission.parquet -m "scored branch: concurrency 28, per-game cap 7920s"
```

The rejection reads `Submission not allowed: Notebook is still running. Did not find provided
Notebook Output File.` (HTTP 400) and costs no slot, so `kaggle_wait_and_submit.py` retries the
submit for `SUBMIT_RETRIES` rounds instead of trusting the COMPLETE token alone. Run the push from the
directory holding `submission.parquet`, or set `SUBMISSION_PARQUET` to its absolute path — the file's
content is ignored by a code competition, it only has to exist locally for `-f`.

Poll it, then submit automatically when the status is COMPLETE:

```powershell
python .\scripts\kaggle_wait_and_submit.py vyassenthilkumar/arc3-milestone2-flashnext-duck 8 "scored branch: concurrency 28"
```

Or submit by hand once the run finishes — the verb is `submit`, **not** `submissions`
(`submissions` takes no `-k`/`-m` and errors with "unrecognized arguments"):

```powershell
kaggle competitions submit arc-prize-2026-arc-agi-3 -k vyassenthilkumar/arc3-milestone2-flashnext-duck -v 2 -f submission.parquet -m "milestone2 flashnext"
```

Same form for the fallback Duck v12 kernel (its real slug ends in `-2pass`):

```powershell
kaggle competitions submit arc-prize-2026-arc-agi-3 -k vyassenthilkumar/arc3-duck-v12-boost-2pass -v 1 -f submission.parquet -m "duck v12 resubmit"

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

