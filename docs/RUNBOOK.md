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

**What a medal actually costs, read off Kaggle's own rule rather than guessed.** The progression table
(`kaggle.com/progression/competitions`, verified 2026-10-25) gives, for a competition of **1,000+ teams**,
Bronze = top 10 %, Silver = top 5 %, Gold = top 10 + 0.2 %, with percentages rounded down. At the **4,908**
scored teams on the board as of 2026-10-27 14:42 UTC that is:

| Medal | Boundary | Public score that reaches it |
|---|---|---|
| Bronze | top 490 | **31.00** |
| Silver | top 245 | **32.31** |
| Gold | top 19 | 38.22 |

Our best row 30.55 is rank 606 = **top 12.35 %** — 0.45 points outside bronze, 1.76 outside silver. That is
the operative target, and it reframes the whole exercise: silver is not a 50-score problem (50 is rank 5,
which is *harder* than a silver medal) but a gap of under two points against a config whose known draws
already reach 38.85. These boundaries move fast — 4,833 teams on Oct 25, 4,867 on Oct 26, 4,894 by 05:01 UTC
on Oct 27, 4,908 by that afternoon — so the field is growing by ~40 teams a day and dragging the lines up
under us. Re-read them from a fresh download every day rather than trusting this table.

**The honest odds, re-read after every draw.** Six byte-identical draws are on the record (25.51, 26.74,
28.18, 30.55, 31.46, 38.85 — **four of them ours**; 31.46 is `yukailiu` and 38.85 is the author, same program): mean **30.21**, sd **4.79**. Exactly one (38.85) clears the silver line — a
**17 % raw hit rate** — while the normal fit gives z = +0.43 and **33 %**. With n = 6 the two are the same
sample read two ways, so the defensible statement is **17–33 % per draw**, and over the seven remaining daily
draws the chance of at least one silver-climbing hit is **1 − (1 − p)^7 = 72–94 %**. Bronze (31.00) is much
closer to a given: 33–44 % per draw ⇒ **94–98 %** over seven. Earlier drafts quoted only the normal fit, and
an n = 5 version of this section said 20–41 % — both overstate it slightly now that a low draw has landed.
Treat the range, never the upper number, as the plan's expectation. **(Superseded 2026-10-27: this pool mixes
202-draw and 80-draw maxima in with our own single draws — see "The unbiased version of that estimate" below
for the low-n population, and the Oct 29 chase-log entry for the current figures: P(silver) ≈ 7–24 % and
P(bronze) ≈ 14–45 % over the four draws that remain after today.)**

**"Can't we just submit the leader's exact code?" — checked 2026-10-26 21:59 UTC, and the answer is no twice
over.** (i) It is not obtainable: `kaggle kernels list --user tufalabs` returns *Not found*, so the rank-2 team
(68.91) publishes nothing; the only public Tufa artifact is Jeroen Cottaar's copy of the **harness**
(`taaf-duck-harness-kaggle`, last run 2026-06-30; `…june-30-milestone-winner`, 2026-07-01) — the milestone-1
scaffold, three months stale, not the model/data config that scores today. (ii) Identical code demonstrably
does **not** produce an identical score, and we are the control group: our scored branch is byte-identical to
the author's live kernel and that one config has drawn 25.51, 26.74, 28.18, 30.55, 31.46 **and** 38.85 — a
13-point spread on one program. A leaderboard score is a draw from a distribution, not a property of the code.
(iii) The top of this board is also a *budget* artifact. From the same download, every team above 45 has
dozens-to-hundreds of submissions while the field mean is 12.29 (median 1.31, because the LB keeps each
team's max): Yi-Chia Chen 71.48 in 42, Tufa Labs 68.91 in 172, Team Turducken 56.02 in 211, the last dance
50.69 in 92, mtg 50.37 in 80, NVARC3 48.84 in 74, Grigoriy Zatravkin & co 47.19 in 106. We have **11** and the
allowance is 1/day with 6 days left, so 17 is the ceiling. Even if a leader's config were handed to us whole,
seven more draws land near *its* mean, not near their max — which is why the plan stays on drawing the proven
config rather than chasing someone else's.

**"Then can we use the code of a team that is above us?" — answered by joining the board to the kernel list
(2026-10-26 22:20 UTC).** Method, re-runnable by the daily guard: `kaggle kernels list --competition
arc-prize-2026-arc-agi-3 --sort-by voteCount --page-size 100` walked to exhaustion (**1,139** public kernels,
**542** authors — note `--page-size 200` silently caps near 100 and `-v` dies on a cp1252 console, so set
`PYTHONIOENCODING=utf-8`), then intersected with `TeamMemberUserNames` from a fresh leaderboard download
(the `ref` prefix *is* the username). Result: **69 of the 598 teams above us publish at least one kernel, and
25 of the 244 above the silver line do.** Forking is permitted — they are public kernels — so the only
question is what the code actually is. Pulling the 14 highest-scoring ones and reading their scored cells
sorts them into three buckets, none of which is a lever:

1. **Not the scoring program.** `cpmpml/notebook582dba8536` (rank 6, 48.84) carries 1,270 code characters;
   `gyuyoungpark09/arc-agi-3-bono` (rank 34, 36.54) 1,218 *and* `enable_internet: True`, which cannot run
   under a code-comp rerun at all; the rank-2 "Tufa Labs" artifact
   `jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner` is 12,347 chars of milestone-1 scaffold
   against our 537,712 — same story for `gruvlg/duck-baseline-v3` (rank 35) and
   `biohack44/1-07-great-team-tufa-labs` (rank 47). A published wrapper says nothing about a ~500k-char
   scored notebook.
2. **The program we already run.** `dfranzen/arc-agi-3-milestone-2-solution` (rank 16, 38.85),
   `shiiin9/affectify-arc-31-54-in-a-single-sub` (rank 15, 38.98) and `lalalia9/arc3-franzen-v19` (rank 46,
   35.79) all attach our exact pair `dfranzen/pennyroyal-v253` + `taaf-kaggle-source-bundle-copy` and carry
   the same 10 / 120 concurrency shape. Their rows are draws of *our* config — the same 25.51 → 38.85
   distribution. Copying them changes nothing.
3. **A different *system*, and not a knob.** Rank 3 Team Turducken (56.02) publishes
   `keithtyser/duck-qwen3-8-flash-next-nvfp4-mtp`: `concurrency 28`, `max_runtime_s_per_game = 7920`, plus its
   own public `keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1` + `duck-qwen38-nvfp4-mtp-vllm-smoke-v1`
   datasets (multi-GB image layers). The same 28/7920 fingerprint recurs at rank 8 (46.82) and rank 22
   (37.97). Both halves of it have already been drawn on our account, separately, and both lost: 28/7920 on
   our stack scored **24.65** and a patched-vLLM + foreign-model port scored **5.68 / 5.34**. Rank 9 (45.41)
   swaps the model outright (`cmechevalier/face-of-agi-qwen36-35b-fp8-weights`), rank 27 runs a Gemma-31B.

**The specific candidate asked about on 2026-10-26 — `romantamrazov/the-agi-boys-i-am-better` (rank 51, 35.35)
— is our own kernel, minus a safety fix.** Full cell-by-cell diff against our working copy: 23 cells each, and
only four differ. Cell 0 is prose (−438). Cell 16 differs **entirely inside `else:`** — `bm.n_passes 4 → 1`,
their `demo_excluded_games = ['bp35','cd82','cn04','dc22','g50t','ka59','lf52','ls20','m0r0','s5i5','sk48',
'sp80',…]` (which is our measured capability wall, independently confirmed by the leader line-up: sk48, bp35,
g50t, sp80, lf52), and our `# FAST_PROXY` lines; `if TRUE_SUBMISSION:` is identical, and the exclusion list is
`[] if TRUE_SUBMISSION else […]`, so it cannot touch a scored run. Cells 2–15 and 17–22 match in length. The
**one scored-path difference is `SERVER_STARTUP_TIMEOUT`: theirs `12 * 60`, ours `60 * 60`** — and the 12-minute
gate is precisely what zeroed our v4 offline run when the model precache took 2,157 s, with cold boots
measured at 1,865 s since. Adopting their notebook would therefore reintroduce a boot hazard into a session
that has no other guard, for zero expected gain. Their 54 `setup_env` keys, all `bm.*` scored values and both
`*_PROMPT` constants are identical to ours; so are `dfranzen/pennyroyal-v253`,
`dfranzen/taaf-kaggle-source-bundle-copy` and both `dfranzen` model sources.

The same scan over the nine other above-silver lineage publishers (shiiin9 38.98, dfranzen 38.85, lalalia9
35.79, vickyfym 33.94, yongjiehishi 33.78, tantan0327 33.55, jingshuifu 33.22, weitaoguo 32.61, heliosli 32.60,
dennis6886 32.59, samrishb 32.42, greenalgeria 32.38) found **zero** `setup_env` changes and **zero** new
prompt constants anywhere on the board: every published delta lives in the demo branch. What actually
separates those rows from ours is the number of draws — 3, 6, 8, 10, 14, 20, 29, 38, 80, 90, 99, 111, 202
submissions versus our 11 — and their `n_passes`/`max_runtime_s_per_game` differences do not reach the rerun.

**So: no published kernel raises our mean, and gold is not a 35 problem.** Gold here is rank ≤ 19 = **38.20**;
35.35 would be rank 51 = top 1.05 % = **silver**, comfortably. Our own draws of the byte-identical config are
**25.51, 26.74, 28.18, 30.55** — four, none above the silver line — and the 31.46 and 38.85 in the six-sample
pool quoted above are *other accounts* (yukailiu, the author) running the same program, which is what proves
the tail is real rather than ours by right. Keep the slots on draws of our kernel, which is the same program
as theirs plus the boot guard.

### The unbiased version of that estimate — and it is much worse (2026-10-27 00:20 UTC)

The six-sample pool above is contaminated: 38.85 is a **202-submission** account's maximum and 38.98 an
80-submission one, so quoting them as if they were single draws of ours inflated every hit rate built on
them. The clean sample is the accounts that ran the identical program and **had few draws to spend**. Method:
hash the 498,799-char harness cell and the `if TRUE_SUBMISSION:` block of every publishing account whose team
has ≤8 submissions and a score ≥15, and keep only exact matches — **23 accounts match both hashes**
(harness `6fdcd1ae7445c0cf`, scored block `68839f77eb237f82`). Their best rows: 24.21, 24.46, 25.31, 26.73,
26.80, 26.88, 27.12, 27.22, 27.79, 27.80, 28.21, 28.52, 28.96, 29.31, 29.34, 30.65, 30.75, 31.27, 31.58,
32.16, 32.59, 33.22, 33.94 — **mean 28.91, median 28.52, max 33.94, and zero of 23 reach 35.**

Our own draws (24.29, 25.51, 26.74, 26.94, 28.18, 30.55 — six, as of the Oct 28 08:51 draw) sit inside that
distribution; 30.55 is its **65th percentile** (15 of 23 accounts are at or below it — an earlier draft said
80th, which was wrong), and our six-draw mean 27.04 against the population's 28.91 is **z = −1.68** (the five
draws before it gave 27.58 and z = −1.08), still inside sampling noise at n = 6 but no longer comfortably so.
The config is not underperforming for us — it performs for us as it does for
everyone with our number of attempts. Read the odds off the low-n population rather than the high-n maxima:

| Line | Of those 23 accounts | Inverted to a per-draw p | P(at least one in the 4 draws left) |
|---|---|---|---|
| Bronze 31.00 | 6 / 23 = 26 % | 1 − (1 − 0.26)^(1/k) = **3.7 – 14 %** | **≈ 14 – 45 %** |
| Silver 32.31 | 3 / 23 = 13 % | 1 − (1 − 0.13)^(1/k) = **1.7 – 6.7 %** | **≈ 7 – 24 %** |
| 35.00 | 0 / 23 | not observed | — |
| Gold 38.22 | 0 / 23 | not observed | — |

The "4 draws left" column is the Oct 29 05:00 state (Oct 30, 31, Nov 1, Nov 2); it was 6 draws / 10–34 % on Oct
27 and 5 draws / 8–29 % on Oct 28, so it shrinks by one factor of (1 − p) every day regardless of what we draw.
The account-level counts in column 2 were measured against the 31.00 / 32.31 lines, and today's re-read lines
are **higher** (bronze 31.15, silver 32.44 at 4,960 teams), so these per-draw rates are, if anything, slightly
optimistic.

The range is the number of real draws hidden inside each account's ≤8 submissions (k = 2 ⇒ high end, k = 8 ⇒
low end). Note this also **corrects the 2–13 % per-draw band used in the first version of this table**: 13 %
is the *account-level* rate, and an account's best row over several draws clears a line more often than one
draw does, so treating 13 % as a per-draw probability was the same contamination in a smaller hat.
**Retract the 72–94 % figure in the odds paragraph above**, and the 15–40 % revision with it — both were built
before that inversion. Honest statement now: silver is roughly one chance in five, maybe one in three at the
optimistic end; bronze is nearer one in two. Either way the binding constraint is the one-submission-per-day
allowance, not the code, which is why every remaining slot goes to a draw.

**Which also says the only accounts worth copying are the ones that beat the lottery — and none of them
publish.** Board-wide, exactly two teams reached ≥35 with ≤17 submissions: `_LT_zyc` 36.00 in 10 (`zyczyc`,
whose only kernel is a 2016 digit-recognizer notebook) and AquaUrsa Research 39.69 in 17
(`aquaursaresearch`, one unrelated AIMO/vLLM kernel); Majkel1337 at 45.03 in 28 publishes nothing. Nobody on
the board reached ≥38 with ≤20 submissions and nobody ≥40 with ≤25. The 35–45 band is private code plus
draws; the public band is 25–31, and we already run it byte-for-byte.

**Slot policy, decided by the user on 2026-10-26 after seeing that table: all seven remaining draws go to
v9, and no slot is spent on a code arm.** Asked to choose between holding every day for the proven config and
buying information with one exploration draw, the answer was the draws. The arithmetic says so: at p(silver)
2–13 % per draw, a diverted slot costs 2–13 % of a medal, while the only measurable candidate left — the
rank-3 Turducken stack at `keithtyser/duck-qwen3-8-flash-next-nvfp4-mtp` — has already lost *both* of its
halves on this account (24.65 for concurrency 28 / 7920 s, 5.68 and 5.34 for patched-vLLM + foreign model),
so it is a rebuild with a two-failure prior rather than an arm. Nothing in the runner automations may
deviate from this; a future lever gets written here with its measurement and waits for the human.
All three scheduled tasks (`ecfa8d23` 05:00, `13c22073` 15:00, `a396679e` 21:00 UTC) now carry this policy
plus the corrected boundaries (bronze ≤ 486 = 31.00, silver ≤ 243 = 32.29, gold ≤ 19 = 38.20, rank 596 on a
4,869-team board) and the retraction of the 72–94 % figure in their instructions, so a run cannot act on the
old numbers even though it inherits no context from this conversation.

**Nothing else is required to win a medal — checked in the rules, not assumed.** The competition's own
rules text (read from the live rules page 2026-10-25) attaches obligations only to *winners* and *prize*
eligibility: the open-source licence (CC-BY 4.0, plus open-source system/model/weights) and the detailed
methodology write-up apply to the teams that take the $850k prize pool, and the milestone prizes required
published notebooks on dates that have already passed. A progression medal is a function of final rank
alone, so our **private** kernel `arc3-milestone2-flashnext-duck` is fully eligible and must not be
published before the deadline just because a comment says "open source it". (One caveat in the same text: a
participant *removed* from the leaderboard for a rules breach also forfeits points and medals — which is why
nothing here bypasses the one-submission-per-day allowance or the code-integrity rules.)

Caveat that cannot be removed: medals are awarded on the **final** leaderboard, which scores the *other*
55 games, so these score boundaries are estimates read off the public half. The percentile rule is exact;
the score that lands at a percentile on the private half is not knowable from here. The field can also grow
until the 2026-10-26 entry deadline, which tightens every boundary in the table.

**The threat to this plan is not the score, it is losing slots silently — and the machine slept.** Two of
the last three days burned their allowance without a draw: Oct 24 (the allowance had not rolled over at the
scheduled hour and the retry window expired) and Oct 25 (`CHAT_SESSION_AGENT_RUNTIME_UNAVAILABLE` after
35 ms — a runtime failure, not task logic). Measured on this box, `powercfg /q SCHEME_CURRENT SUB_SLEEP
STANDBYIDLE` reported **AC index 0x258 = 600 s**, i.e. the machine slept ten minutes after going idle, and a
sleeping machine runs no automations. **Fixed on 2026-10-25 23:20 UTC with the user's approval**:
`powercfg /change standby-timeout-ac 0` — verified `Current AC Power Setting Index: 0x00000000`, and
`HIBERNATEIDLE` on AC was already `0x00000000`. Revert with `powercfg /change standby-timeout-ac 10`.
Battery (DC) behaviour is untouched at 600 s, so unplugging still sleeps normally.

Coverage is therefore three attempts per day, each idempotent (step 1 is "if a submission row is dated
today, report and stop"): `ecfa8d23` at 05:00 UTC, `13c22073` at 15:00 UTC and `a396679e` at 21:00 UTC, the
last of which also writes a "slot NOT used and why" row so a lost day is visible instead of inferred. (The
one-shot `1748cb44` at 10:00 then 00:10 UTC is gone: it deleted itself as a `deleteAfterRun` job without
leaving a submission row, so the Oct 26 draw had to be taken by hand.) **Remaining fragility, stated
plainly:** all three are *local* automations and only fire while the Qoder app is running, and Qoder had no
startup entry in `HKCU\...\Run`, `HKLM\...\Run`, or either Startup folder (checked 2026-10-25; those folders
do hold `Ollama.lnk`, `Hermes_Gateway.vbs`, `JARVIS.cmd`, `start_assistant.bat`).

**Fixed on 2026-10-26 11:30 UTC with the user's approval ("that fine push it now").** There is no registered
Qoder App setting for this — `manage_app_setting` searches for "startup" and "login" both returned zero
items — so the supported Windows mechanism was used:
`%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\Qoder.lnk` →
`C:\Users\Vyas S\AppData\Local\Programs\Qoder\Qoder.exe` (working dir `...\Programs\Qoder`), created through
`WScript.Shell` and read back to confirm the target resolves and the exe exists. It is absent from
`Explorer\StartupApproved\StartupFolder`, which is the key that records *disabled* state, so it starts
enabled; that key shows six of the eight pre-existing entries already pruned by the user (Ollama, JARVIS,
Hermes_Gateway, Wispr Flow, OneNote, start_assistant). Undo with
`Remove-Item "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup\Qoder.lnk"`.
**What this does not solve:** startup-folder items run at *logon*, so if the box reboots and sits at the lock
screen with nobody signing in, Qoder still never starts and that day's draw is lost. Nothing in this plan
survives an unattended reboot; the three-attempt spread and the chase log are what make the loss visible
rather than silent.

**Rejected on purpose, so no future run gets creative on the entry deadline (2026-10-26):** adding a second
entry via a Kaggle Notes *organization* would double the daily allowance, i.e. double the draws. It is not
being tried — one team/one entry is the shape these rules assume, the rules text says a participant removed
from the leaderboard for a breach also forfeits points **and medals**, and losing the medal we can already
almost see to gain eight more coin flips is a bad trade. Team settings, entries and merges are off-limits to
every automation in this file.

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

**One row per UTC day.** If a row for today already exists, update it — do not append a second one. Three
automations look at this file daily (05:00, 15:00, 21:00 UTC) and the day's slot can only be spent once.

| Date (UTC) | Ref | Arm drawn | Score | Team best | Rank |
|---|---|---|---|---|---|
| 2026-10-25 07:21 | `57621235` | port v5, temperature 0.9 — the last port draw | 5.34 | 30.55 | 581 |
| 2026-10-26 11:27 | `57664497` | v9 proven config, submitted by hand after the 00:10 one-shot vanished | **25.51** | 30.55 | 595 |
| 2026-10-27 05:00 | `57689693` | v9 proven config, 05:00 draw (first of three attempts, slot used) | **26.94** | 30.55 | 612 |
| 2026-10-28 08:51 | `57731198` | v9 proven config — taken by hand after the 05:00 automation's session died | **24.29** | 30.55 | 647 |
| 2026-10-29 05:00 | `57759315` | v9 proven config, 05:00 draw (slot used, no race this time) | **27.65** | 30.55 | 672 |

**2026-10-29 ~17:54 UTC — the revert-to-v9 contingency in the three draw crons is arithmetically wrong, and is replaced.**

All three daily tasks (`ecfa8d23`, `13c22073`, `a396679e`) currently carry this line: *"If both the v11 and v12
draws land below 30.0, the mean did not move — revert to `-v 9`."* Followed literally on Nov 1 or Nov 2 it
would spend our last two slots on the arm we have just shown is ~5 points worse, so it is corrected here, in
the block they are instructed to read first.

The rule confuses **one noisy observation** with **the mean having moved**. Work it through with the numbers
§1p just established:

| quantity | value | where it comes from |
|---|---|---|
| sd of one public-half draw | ~2.0 | our seven v9 draws, sd 1.9–2.17 |
| prior mean, v11 (SLM) | ~32.0 | `lalalia9` max 35.79 over 22 draws ⇒ mu = 35.79 − 1.91σ ≈ 32.0 at σ=2 |
| prior uncertainty, v11 | ±1.5 | σ of the *conversion* (σ=1.5 ⇒ 32.9; σ=2.5 ⇒ 31.0) plus the risk their node differs |
| prior mean, v12 (SLM+PACE) | ~33.5 | v11's 32.0 + PACE's ~1.5 (jingshuifu 33.22 over 6 ⇒ mu ≈ 30.7 vs ~29.2 for plain upstream) |
| prior uncertainty, v12 | ±2.0 | the SLM+PACE **combination** has never been published — additive is an assumption |
| known mean, v9 | 27.12 | seven direct draws, not an inference |

Bayes with a prior sd of 1.5 against an observation sd of 2.0 weights the prior 0.64 and the single draw 0.36.
So a v11 draw of **29.5** leaves a posterior mean of 0.64·32 + 0.36·29.5 = **31.1**; a draw of **25** still
leaves **29.5**; it takes roughly **20** before v11's posterior falls to v9's known 27.12. For v12 the prior
weight is 0.5, so a draw of 27 leaves **30.3** and it takes about **21**. Both arms therefore beat a known
27.12 across the entire plausible range of a bad-luck afternoon, and "below 30.0" is not a cliff at all — it
is the region where the prior is *right most of the time* (v11's own mean of 32 still clears only ~40 % of
draws over the 32.54 line).

**Corrected contingency, which supersedes that line wherever it appears:** on Nov 1 and Nov 2 keep drawing the
arm with the higher posterior mean, computed as above — in practice that means `-v 9` returns **only if both**
v11 and v12 rows land below ~21, or either row is 0.00 / its guard failed (a genuine broken-arm signal, not a
lottery signal). Anything in the 20s is one draw of a distribution, and the whole point of the sweep is that
the distribution moved up.

**Nov 1 tie-break, stated so nobody has to improvise:** "higher public row" is the rule, with one asymmetry —
the v12 prior is ~1.5 above v11's, so when v12's row is *within* ~1.5 of v11's, take `-v 12`; take `-v 11` only
when v12 fell more than ~1.5 below it. That is the 0.5/0.5-vs-0.64/0.36 weighting above, not a preference.

**What the two judged rows are actually worth (estimates with the assumptions named, not a promise).** Final
standing uses at most two selected rows scored on the private half. If the standing takes the better of the
two and the private half has the same sd as the public half, then at mean 31–32 against today's silver line
32.54 each row clears with p ≈ 0.31–0.41 and two rows give ≈ 0.52–0.65 — versus the 0 of 7 that v9 produced
at mean 27.12 (p ≈ 0.05 per row). The three honest caveats: we cannot see the private half, per-row sd on 55
games may exceed 2.0, and if Kaggle's standing uses something other than the max of the selected rows the
number is lower. Even so this is the first time in this competition that our *plan* has a coin-flip shot at
silver rather than a lottery ticket.

**2026-10-29 ~17:50 UTC — a 13-kernel forensic sweep: v11 is a byte-for-byte reproduction of the best published config, and the final four slots are scheduled.**
(The three draw crons point at this block as "the ~18:05 block"; that label was written from the rehearsal's expected
completion time rather than `date -u`. Same block — the clock said 17:49 when it was committed.)

Thirteen public kernels were pulled to temp and diffed cell-by-cell against ours (their refs, hashes and diffs are
reproducible from `C:\Users\Vyas S\AppData\Local\Temp\kernelforge`). The headline:

*(a) The harness patch is public property, and everyone runs it.* Cell 2 — the 498,799-char `git diff` heredoc — is
**byte-identical (`030f0b8f33`) in every Milestone-2 fork examined**: jingshuifu, tantan0327, heliosli, dennis6886,
weitaoguo, lalalia9, all three vickyfym kernels. Cells 6, 8, 10, 14, 18, 20, 22 are identical everywhere too. The
only scored-code divergence anywhere is in cells 4 (setup_env), 12 (serve/venv) and 16/17 (the `bm` hook). One fork
diverges: `sujanmajhisuzan/arc-agi-3-m2-top-submission` carries a **second** 138 KB base64+gzip patch and ~15 extra
scored env flags (`ARC3_UND`, `ARC3_DEATH_LEDGER`, `ARC3_WORLD_MODEL_SIM`, …) and scores 31.93 in 11 — heavy,
unverified, not adopted.

*(b) `lalalia9/arc3-franzen-v19` — 35.79 in 22 submissions, rank 52 — **is our v11**.* Its env cell differs from ours
in three lines: `SERVER_STARTUP_TIMEOUT` (offline-only), `ARC3_PRIORITY_PACE '0'` (our v11 is '0' too; only v12 is
'1'), and a no-op `os.environ.update({})` + `print('VARIANT franzen-v19')`. Its hook cell differs from ours only in
the comment block and in the `else:` demo branch (`n_passes`, `demo_excluded_games` = 15 games, `25*60`) — every one
of those is overridden by `if TRUE_SUBMISSION:`. Its serve cell is upstream's. So **v11 reproduces, on the scored
branch, the highest-placing published M2 configuration we can actually build**, which is a far stronger claim than
the patch authors' "+2.8 on the leaderboard" that motivated it.

*(c) The max-vs-mean trap, and the arithmetic that fixes it.* A delegated read of this evidence concluded "PACE='0'
configs score 33.55/93 and 32.60/13, so revert PACE" — that compares *maxima over different numbers of draws*, which
is not a mean. Correcting it with extreme-value arithmetic, `E[max_N] = mu + sigma * z_N` with
`z_N = Phi^-1((N-0.375)/(N+0.25))` and `sigma = 2.0` (our own seven byte-identical v9 draws have sd 1.9):

| config | observed max @ N draws | z_N | implied per-draw mean |
|---|---|---|---|
| plain upstream scored branch (tantan0327) | 33.55 @ 93 | 2.47 | ~28.6 |
| plain upstream (weitaoguo / heliosli / dennis6886) | 32.61 @ 17, 32.60 @ 13, 32.59 @ 11 | 1.80 / 1.67 / 1.59 | ~29.0 – 29.4 |
| **our v9, measured directly** | mean of 7 draws | — | **27.12 ± 0.7** (consistent) |
| SLM + PACE='0' (lalalia9) | 35.79 @ 22 | 1.91 | **~32.0** |
| SLM + `analyzer_timeout 3600` (vickyfym) | 33.94 @ 3 | 0.87 | **~32.2** |
| no SLM + PACE='1' (jingshuifu) | 33.22 @ 6 | 1.28 | **~30.7** |

Reading: the SLM patch is worth about **+3** over the plain config and PACE about **+1.7**, both positive, and the two
estimates are single-team observations with ±1.5 error bars — not measured. **Nobody has published SLM + PACE
together**, so v12 is an untested interaction of two individually-positive levers. That is why the four remaining
slots are split rather than committed to the stack.

*(d) Two findings are insurance, not upside, and are deliberately NOT taken in the final week.*
`weitaoguo/affectify-arc-dfranzen-form-d1` replaces `TRUE_SUBMISSION = env(KAGGLE_IS_COMPETITION_RERUN)` with
`_ENV_FLAG or _arc_gateway_reachable('http://gateway:8001/', 900, 15)` and records that on a **2026-10-22 scored
rerun the env var was absent, the notebook took the rehearsal branch, and Kaggle returned 0.00**. And
`dennis6886/arc3-d-prime-public-base` (32.59 @ 11) attaches `destinyfire1/arc3-cpython312-runtime`, concatenates
7 `cpython-3.12.14-part-*.bin`, asserts sha256 `3297691ae3…` and builds the venv from that, whereas our cell 12
probes `/usr/bin/python3.12` and **raises `RuntimeError` if absent** — a latent hard-fail of a whole scored rerun.
Both are real 0-score paths; both changes touch the interpreter/bootstrap of a config with 14 successful reruns
behind it, and neither can be validated without burning GPU and a slot. Left alone, recorded here.

*(e) The top of the board is not an M2 fork.* `rellik13/sirikilohit` (37.97 @ 63, rank 25) publishes
`arc-agi-3-duck-18-1gc-submit`: 50 cells, no `TRUE_SUBMISSION`/`setup_env` lineage, datasets
`keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1` + `keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1` +
`sirikilohit/sglang-penny-build-qwen`, model `woochangsim/qwen38-flash-next-w4a16-autoround-4c67bf6`, and
`concurrency = 28` / `max_runtime_s_per_game = 7920.0`. Not copyable into our attachment lineage, and that
per-game shape is exactly the concurrency that scored **24.65** for us (§1l).

*(f) Board re-read 2026-10-29 17:28:36 UTC:* 4,972 rows, **4,969 ranked**; rank 248 = **32.54** (silver line moved
+0.01), rank 249 = 32.53, rank 250 = 32.52; our team row reads rank **676**, 30.55, 14 submissions.

**DRAW SCHEDULE for the last four slots (this supersedes every earlier ordering):**

| Day | Draw | Why |
|---|---|---|
| Oct 30 | duck **`-v 11`** | the reproduced 35.79@22 config; highest-confirmed mean we hold, guard already passed at 16:00 UTC |
| Oct 31 | duck **`-v 12`** if tonight's PACE guard passes, else `-v 11` | tests the SLM×PACE interaction on a live slot; falls back without losing the day |
| Nov 1 | re-roll whichever of v11/v12 has the higher public row | the winner gets two draws; re-rolls are upside-only on the public board |
| Nov 2 | same winner, **only** if the 05:00 attempt can finalise by ~23:00 UTC | a scored run needs ~10 h; a PENDING row on the deadline day is worth nothing |

Contingency unchanged: if both the v11 and v12 draws land below 30.0, the mean did not move and the policy reverts to
`-v 9`. The NVFP4 arm stays retired (§ block above). No scored slot goes to an untested scheduler (D′) or to the
`analyzer_timeout 3600` value — §1o already documents why that arm is a **no-op** (at 10 gated streams the measured
max request latency was 185.6 s, so a 900 s cap never binds and 3600 s is invisible to the score). That also means
vickyfym's ~32.2 and lalalia9's ~32.0 are two *independent* readings of the same SLM arm, which is the single most
encouraging number in this table.


**2026-10-29 ~17:45 UTC — the NVFP4 gate passes and is still retired, because a second account gave us a cheaper, single-value arm.**

*(a) v4 verdict, read from `kaggle kernels output vyassenthilkumar/arc3-nvfp4-lord-recipe` (COMPLETE at 17:13 UTC).*
The gate set at 16:50 was "the vLLM server actually answered and games took actions on a card we can name in the
log", and it is met: `NVIDIA RTX PRO 6000 Blackwell Server Edition` appears in the session log, the un-pinned image
took the card, 14 games ran their demo cap (`rehearsal wave: 14 games` at t=819 s), and they took **539 actions /
17 levels completed** for `mean score: 6.09, median 3.57` at ~626 s per game (ar25 4 levels/27.78, vc33 3/21.43,
lp85 2/8.33; bp35, m0r0, sc25 0). So the arm is mechanically alive.

*(b) Retiring it anyway, on the schedule rather than the stack.* Its scored branch is **14 concurrency x 3,918 s =
8 fixed waves**, so a game gets exactly one 65-minute turn and no priority rotation: no second attempt on a stalled
game, and any boot overrun eats the *tail* waves. We measured that exact failure mode on the duck — the
`concurrency 28` draw scored **24.65** against **30.55** at 120, because the tail games never got scheduled — and
this session's own boot spread for a comparable image pull is 582 s (warm) to 1,975 s (cold) against a
32,400 - 30,800 = **1,600 s margin**. It also ships neither of the two arms we now have evidence for (no SLM patch,
no PACE) and drags a 180.166 GB NVFP4 checkpoint. One draw of an untested scored regime is worth less than one draw
of the proven regime with a positive-evidence value changed. Task #16/#18 close here; **no NVFP4 submission.**

*(c) The cheaper arm, found by diffing a 33.22 account against our own notebook.*
`jingshuifu/arc3-m2-solution-v3` (33.22 in **6** submissions, rank 165 at the Oct 29 read) was pulled and compared
cell-by-cell against our `kaggle_submit_milestone2/arc-agi-3-milestone-2-solution.ipynb`. All 23 cells are
identical except 4, 12 and 16, and **cell 2 — the 498,799-char harness patch — is byte-identical (`030f0b8f`)**.
Attachments match ours exactly (same 2 datasets, same 2 model instances, same `NvidiaRtxPro6000`), only reordered.
Cell 16's `if TRUE_SUBMISSION:` block is character-for-character ours (`save_request_logs = False`, `n_passes = 1`,
`concurrency = 120`, `max_runtime_s_per_game = 532*60`); their differences are confined to the `else:` demo branch.
Cell 4's diff is two hunks, one of which is `SERVER_STARTUP_TIMEOUT` — an offline-only constant. The other is a
**scored-branch env value**:

```
-    'ARC3_PRIORITY_PACE': '1',  # jingshuifu v3: enable pace penalty   (theirs, 33.22 @ 6 draws)
+    'ARC3_PRIORITY_PACE': '0',                                          (ours, 30.55 max over 7 draws)
```

What it does, read from our own identical harness cell: on every `level_transition` the agent logs
`log(tokens_spent_on_that_level / reference_tokens_for_that_level)`, EMA'd at 0.6/0.4, and the priority of a game is
divided by `pace = exp(clip(confidence * log_cost_ratio))` with `confidence = completed/(completed+3)`, clamped to
[0.5, 2.0] (`ProgressPace`, `priority_value`, `_update_summarized_knowledge_from_step_summary`). So a level that
costs more tokens than the empirical reference for that level number gets *deprioritised* — the gate's 10 streams
go to games that are progressing cheaply. The reference table it needs is **not** a missing-file risk: the patch
cell itself creates `ARC3-Inference/inference/agent/priority_pace_reference.json` (`fallback_tokens` 45,925;
per-level 27,011 / 51,073 / 48,932 / 56,352 / 55,797 / 67,264 / 40,699 / 22,696; "Kaplan-Meier restricted mean token
cost, capped at 120000 tokens", fit on 1,417 reached-level episodes), which is why the bundle dataset does not
contain it and why the file is absent from `kaggle datasets files` yet present at `Path(__file__)`.

*(d) v12 is built and pushed.* Kernel `vyassenthilkumar/arc3-milestone2-flashnext-duck` **version 12** = v11 + that
one character — verified as a single-token diff against HEAD (common prefix 526,367, common suffix 54,637, removed
`0`, added `1`; 23 cells parse) before `kernels push`, which printed `Kernel version 12 successfully pushed`.
It is running the offline 25-game FAST_PROXY branch as a guard. **Guard criteria for drawing it** (PACE prints
nothing on its own, so the check is absence of the failure it could cause): status COMPLETE, the log still contains
`SOLVED_LEVEL_MEMORY_PATCH on`, the run reaches the benchmark stage and prints `mean score:` **> 0** — a nonzero
mean requires a completed level, which is the only path that reads the pace reference file — and no
`FileNotFoundError`/`Traceback` around `priority_pace_reference` or `_pace_reference_tokens`.

*(e) Honest caveat.* Their notebook pins `docker_image: gcr.io/kaggle-private-byod/python@sha256:57e612b4...` while
ours takes the default image, so "PACE explains 33.22 vs 30.55" is not established — the image is a second,
untestable difference and the gap is inside a 7-draw spread of 24.29-30.55. What *is* established: a one-value,
zero-throughput-cost scored-branch flag that a medal-line account runs and we do not, on a code path we have read
and can prove is exercised. Stacking it on the SLM patch (v12) is deliberate: the two levers touch different
subsystems (prompt memory vs stream admission) and share no state.


**2026-10-29 ~17:20 UTC — the guard run's zero pins are explained, and the explanation says v11 is probably live.**

Reading the SLM code properly: it **captures** on every turn (`_slm_before_turn` is hung off `analyze`) but
**injects** only inside `_note_history_evicted` / `_trim_messages_for_context` — on purpose, because rewriting the
system prompt invalidates the shared prefix and the 94.3 % cache hit rate. So the arm pays off only on games that
actually drain, and tonight's rehearsal printed `SOLVED_LEVEL_MEMORY pinned` zero times. That looked like a dead arm.
The 25 downloaded `*_p0_requests.jsonl` traces say otherwise, because the offline branch is not the scored branch:
cell 16 sets `max_runtime_s_per_game = 5*60` "FAST_PROXY: sized for the demo set, not the hidden rerun", so every
game there was killed at 300 s.

Measured from those traces (real numbers, not a model of them): median growth **2,516 prompt tokens per turn**, first
request at **5,479** tokens, best game **21 turns → 55,807** tokens — i.e. the 300-second cap, not the context cap, is
what stopped it. The drain the patch waits for fires at ~118 k, which is **45 turns** in, and each later cycle
(118 k→59 k) is **23 turns**. Throughput under our `ARC3_MAX_ACTIVE_STREAMS=10` gate is 25 games × 25 turns / 300 s =
**2.08 turns/s aggregate**; spread over the scored 110 games that is 0.0189 turns/s each, so a game that lives the
whole 32,400 s session sees ~**614 turns ≈ 26 drain cycles**, and one that only gets a fifth of the session still sees
~**5**. So in the scored regime the pin should fire repeatedly, and tonight's zero is fully accounted for by the
5-minute demo cap.

**What this is and is not.** It is arithmetic on offline traces, with two soft spots: it assumes scored games live
long (most die early — a game that ends at 30 turns never drains, but also never re-derives anything), and it assumes
the aggregate turn rate transfers from 25 games to 110 (the 10-stream gate is what makes that roughly true). It is
**not** a measurement of a scored run, and it does not raise the +2.8 claim's status: the only real evidence stays
`vickyfym` 33.94 in 3 draws and `lalalia9` 35.79 in 22 on our exact stack plus this patch. Consequence for the plan:
the ~2.5 h **drain probe** (task #17) is demoted from "needed" to "confirmation" — the arithmetic already says the
arm is not inert, and tonight's remaining 7.1 h of quota is better left whole than spent proving a number the traces
give us.


**2026-10-29 ~16:55 UTC — the Nov 2 browser step is a taste decision, not a cliff.** Every automation here has been
treating "select two Final Submissions in the web UI" as must-do-or-lose-everything, on the strength of the rules
line *A "Final Submission" is the Submission selected by the user, **or automatically selected by Kaggle in the
event not selected by the user**, that is/are used for final placement*. That sentence never says *which* rows Kaggle
picks. Two Kaggle forum answers found by search (`general/432319`: "If none are selected then the 2 best public LB
are used for the final selection for private LB scores"; `general/6384`: "If you don't pick the total number
available we pick the highest public submissions") both say **best-public**, not "your last submission" — which also
means the common fear that a forgotten selection leaves you judged on a throwaway final roll is wrong.

**Honest status of that claim: forum answers, not documentation, and the rules text itself is silent.** Treat it as
"the downside of forgetting is small", not as proof. What it changes here: the 18:00 UTC selection automation stays
scheduled (its output — the two refs, plus for each one the arm, the log evidence and which games went deep — is
exactly what a *deliberate* choice needs, because judging is on the private half and public-best is only a proxy for
it), but nobody should panic if the browser step is missed: the fallback is our two highest public rows, which we
would very likely have chosen anyway.

**Rehearsal state while this is written:** `vyassenthilkumar/arc3-nvfp4-lord-recipe` v4 is `RUNNING` and the GPU
counter is climbing (22.39 → 22.46 h over ~10 min), which is the proof it needed — the un-pinned session owns a card,
where all three pinned-image attempts charged nothing. Still to see from its log: the layer extraction finishing,
`Image runtime ready at /tmp/vllm-image-runtime: vLLM …, torch …`, the vLLM server answering, and whether a level
past 1 falls out of a 3,918-second game.


**2026-10-29 ~16:50 UTC — the NVFP4 arm's no-GPU failure is the pinned docker image, not concurrency.**

The 1-cell probe `vyassenthilkumar/arc3-gpu-probe` (their exact pinned image + `NvidiaRtxPro6000`, no attachments,
nothing else on the card) came back `COMPLETE` in ~2 min printing `python 3.12.13 (main, Mar 4 2026)`,
`KAGGLE_GPU_TYPE= None KAGGLE_GPU_COUNT= None` and `rc 12 out "NVIDIA-SMI couldn't find libnvidia-ml.so library in
your system…"`. So the pin **is** honoured (that build is 3.12; our un-pinned sessions run 3.13 — the v11 rehearsal
log writes to `/usr/local/lib/python3.13/dist-packages`) and the card is **not**: the session was scheduled
CPU-only. Control from the same log set: the un-pinned duck kernel on the same machine shape prints
`GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition capability: (12, 0)`. I pulled the author's own
`kernel-metadata.json` first to rule out a copy mistake — same digest, same shape, same three sources, and their
kernel's last run was `COMPLETE` on **2026-10-01 18:38 UTC**. Whatever changed on Kaggle's side since then, today
that pin costs our account the GPU.

The pin is not load-bearing for *their* recipe, though, which is why the arm is worth one more try. Their
`setup_commands.json` (850 lines) does not install into the container's python: it unpacks the dataset's six
`layer-NN-<sha>.tar.gz.blob` files into `IMAGE_RUNTIME_ROOT = /tmp/vllm-image-runtime` and then derives
everything *manifest-relative* — `IMAGE_RUNTIME_ROOT / manifest['dist_packages']`,
`cuda_home = IMAGE_RUNTIME_ROOT / manifest['nvcc']`, and `PATH`/`LD_LIBRARY_PATH`/`PYTHONPATH` prepended from that
root (`'Image runtime ready at …: vLLM {…}, torch {…}'`). The only host-facing requirement is
`assert_expected_cuda_gpu`. So **v3 was pushed with `docker_image` removed** — the one config we know gets a card —
and **v4 adds a rehearsal-only trim** in the customization cell, guarded by Kaggle's own rerun flag so a scored
run keeps the full 110-game list and their native 14 × 3,918 s schedule:

```python
if not true_submission:
    bm.games = bm.games[:14]
```

One wave of 14 games at 3,918 s costs ~65 min of GPU plus setup instead of ~8.6 h, and 3,918 s is their real
per-game budget, so the rehearsal shows level depth under the scored schedule rather than a crippled version of it.
Quota re-read after all of this: **22.31 h used / 7.69 h remaining, refresh 2026-10-31T00:00 UTC** — three failed
pinned sessions plus the probe charged 0.04 h in total, which independently confirms the (5) finding that a session
with no GPU is free and that scored reruns are not the quota consumer.

**Oct 31 gate, restated.** Draw the NVFP4 arm only if v4 reaches a state where the vLLM server actually answered
and games took actions on a card we can name in the log; otherwise Oct 31 draws duck `-v 11` and the arm is retired.
**Resolved 2026-10-29 ~17:45 UTC (block at the top of this log): the condition was met, and the arm is retired anyway**
on its 14 x 3,918 s fixed-wave scored schedule. So the two sentences below that hand days to "the NVFP4 arm" are dead —
Oct 30 onward draws the duck: `-v 12` (PACE) if its guard passes, else `-v 11`, else `-v 9`.
**Correction, same evening: the scored rerun's log is not reachable, so that gate was not executable.**
`kaggle competitions submissions arc-prize-2026-arc-agi-3 --csv` returns exactly
`ref,fileName,date,description,status,publicScore,privateScore` — no notebook id, no session handle, no output URL.
The rerun happens on Kaggle's side and we get only its score. So nothing on Oct 30 will let us count
`SOLVED_LEVEL_MEMORY pinned` in a scored log; a draw below 30 would not even be evidence that the patch was inert,
because v9's own seven draws span 24.29–30.55.

The drain question **is** answerable, but only interactively, and it is the most informative experiment left for our
primary arm. Planned as the **drain probe** (after v4 releases the card, ~2.5 h, paid from the 7.12 h quota that is
left, and it costs no submission slot): push a duck version whose *offline* branch runs the **scored** schedule —
532×60 s per game, `analyzer_timeout 900`, `n_passes 1` — trimmed to one wave of games, long enough that the
conversation can reach the ~118 k→59 k drain instead of dying at 25 min like tonight's guard run. Then count the
`SOLVED_LEVEL_MEMORY pinned` prints. If the counter is zero even with scored-length games, the +2.8 claim does not
transfer to our configuration, the memory arm is a v9 re-roll, and Oct 31/Nov 1 belong to the NVFP4 arm. The scored
branch of that push stays byte-identical, so `-v 11` remains exactly what the daily crons draw.


**2026-10-29 ~16:45 UTC — v11 passes its guard, and a second, bigger arm is now in rehearsal.**

(1) **v11 guard = PASS.** `arc3-milestone2-flashnext-duck` v11's interactive rehearsal reached
`KernelWorkerStatus.COMPLETE` in ~36 min with **zero** `Traceback`/`AttributeError` in the session log, and the
marker line `SOLVED_LEVEL_MEMORY_PATCH on` is printed at log record 729, immediately before `benchmark.solver:`
(record 730) — i.e. the four `ToolAgent` monkeypatches were installed *before* the run started, and the scored
config printed unchanged after them. The ask path also fires for real: `Rule for level` appears in 10 of the
downloaded per-game `*_p0_requests.jsonl` traces, so cleared levels are being captured. What did **not** happen
offline is the write: `SOLVED LEVELS` appears in 0 traces and `SOLVED_LEVEL_MEMORY pinned` 0 times, because the
block is only rewritten inside `_note_history_evicted`/`_trim_messages_for_context` and a 25-min offline game
never drains the conversation (9-33 actions, 11-21 k tokens per game, far below the 118 k→59 k drain). So the
rehearsal proves **install safety, not effect** — the effect can only be measured by a scored draw. Offline mean
was 2.12/2.27, which is not a leaderboard number and proves nothing either way. Oct 30 05:00 therefore draws
`-v 11`.

(2) **Where the real levers are: Tufa Labs ships their harness as a *dataset*, not as notebook code.** Their own
readable notebook (`gruvlg/duck-baseline-v3` — the TAAF/Tufa Labs submission notebook, 17-24 KB) has an **empty**
customization cell, so nothing about their approach is in notebook source. But `kaggle datasets list -s taaf
--sort-by updated` returns modified bundles, and two of them are the same thing: `lordhansolo/taaf-kaggle-source`
(656 KB, Apache-2.0, 3,631 downloads) and `francescosabbarese97/taaf-kaggle-source-bundle` — both `git_status.txt`
at **`ca1bd02`, branch `solution-improve-qwen38-flash-next`, "Let the agent act without full certainty"**, i.e. the
harness authors' own improvement branch for the model we serve, whereas our attachment
`dfranzen/taaf-kaggle-source-bundle-copy` is the older `aa69123` (branch `add-kaggle-share-flag`). Diff: 87 files
vs 73, `prompts.py` 13,574→26,504 chars (adds `STRATEGY_AUDIT_PROMPT`, drops `TOOL_CALL_FORMAT_GUIDANCE`),
`solver.py` 49,315→69,332, `tool_agent.py` refactored into 10 new modules, plus `configs/draft_vocab_32k.json`,
`draft_vocab_48k.json` and `inference.local.json`. The `ONLY_RESET_LEVELS` fix in their `game_api.py` is **not** a
lever — our bundle already sets it at `game_api.py:214/221`.

(3) **The arm is `lordhansolo`'s whole published stack, not a patch.** `lordhansolo/built-on-tufa-labs-duck-harness-milestone-2`
(72 votes; author = **rank 31 at 37.07 in 106 submissions**, read from today's board) attaches three public inputs:
that `ca1bd02` bundle, `lordhansolo/vllm-main-e975732-arc3` (7.93 GB patched vLLM: `VLLM_GDN_DECODE_KERNEL=triton`,
`VLLM_ARC3_GDN_RECOVERSSM=1`, `--mamba-ssm-cache-dtype bfloat16`, `max_num_seqs 14` ⇒ decode step 14×(1+3)=56,
MTP `{'method':'mtp','num_speculative_tokens':3}` with `use_local_argmax_reduction`, `--kv-cache-metrics`, a
watchdog that restarts the server and a GPU shard prefetcher), and model
`lordhansolo/qwen3-8-flash-next-mixed-nvfp4-fp8/PyTorch/hf-mixed-mtp-nvfp4/1` — **180.2 GB over 108 files with
per-layer `ct-experts-layerNN.safetensors`**, i.e. the same Qwen3.8-Flash-Next MoE scale as our
`dfranzen/intel-qwen3.8-flash-next-w4a16-autoround` (181.2 GB over 51 files + 4.2 GB `albucino` drafter) but
quantised **mixed NVFP4+FP8** (Blackwell-native) instead of W4A16 AutoRound, served by vLLM instead of SGLang
pennyroyal. Their schedule: `VLLM_MAX_MODEL_LEN 147,072`, `ANALYZER_CONTEXT_WINDOW 127,488`,
`LOCAL_ANALYZER_TARGET_CONTEXT 81,536`, `gpu_memory_utilization 0.98`, **concurrency 14 × 3,918 s per game**
(110/14 × 3,918 = 30,800 s inside the 32,400 s wall) — a flat one-wave schedule rather than our 120-queued /
10-active-gate shape. Their analyzer env is temp 0.6, top_p 0.95, top_k 20, `MAX_OUTPUT 0`, `TOOL_OUTPUT_TOKENS 1024`,
`YIELD_SECONDS 150`, reasoning `xhigh`, `MULTIMODAL_CONTEXT=current_grid`, `MULTIMODAL_UPSCALE=4`.
Copied verbatim into `kaggle_submit_nvfp4_lord/` (git `46d1618`), pushed as the private kernel
`vyassenthilkumar/arc3-nvfp4-lord-recipe` with our own metadata (same pinned docker image, same machine shape).
**Honest caveat, from their own notebook text:** the 37.07 row came from notebook v293 with bundle **v297** and the
older runtime `arc3-vllm-e975732-gdn-recoverssm v2`; the currently published pairing we copied is newer and they
write "*should be faster and have a larger KV cache pool than the runtime used in the highest-scoring run.
Unfortunately, no run with it has beaten that score so far*". And 37.07 is a **best-of-106**, so it says nothing
directly about their per-draw mean — which is exactly what one or two of our draws would get.

(4) **Why v1/v2 of that rehearsal died — and the rule it teaches.** Both ended `KernelWorkerStatus.ERROR` after
~5 min with `AssertionError: nvidia-smi failed:` inside `assert_expected_cuda_gpu`, while the attachments resolved
fine (`/kaggle/input/datasets/lordhansolo/vllm-main-e975732-arc3`,
`/kaggle/input/models/lordhansolo/qwen3-8-flash-next-mixed-nvfp4-fp8/pytorch/hf-mixed-mtp-nvfp4/1`). My first
diagnosis was **concurrency** — the duck v11 rehearsal was still live when v1 ran, so Kaggle handed the second
session no card. **That was wrong**: v2 ran with the card free and failed identically, and `kaggle quota` never
moved for either (a session with no GPU cannot charge GPU time). See the ~17:00 entry for the real cause.

(5) **GPU quota, measured — and it does not threaten the draws.** `kaggle quota`: GPU **22.27 h used, 7.73 h
remaining, refreshes 2026-10-31T00:00 UTC**. Six ~10-hour scored reruns ran inside this window while only 22 h
were charged, so **competition reruns do not consume the notebook GPU quota; interactive rehearsals do**. Therefore
rehearsing the NVFP4 arm (~1-3 h) cannot starve the Oct 30/31 draws, and the quota resets two days before the
deadline anyway.

**Slot plan for the four remaining days, in EV order.** Oct 30 05:00 → duck `-v 11` (SLM, guard passed). Oct 31 →
the NVFP4 arm if its rehearsal reaches COMPLETE having played games on a card (v4 = un-pinned image, one 14-game
wave); otherwise duck `-v 11` again. Nov 1 → whichever of the two arms produced the better row on Oct 30/31. Nov 2 → the
same arm once more, leaving the last draw inside the day so the 18:00 UTC selection automation can still name two
refs. Keeping two arms also diversifies the ≤2 rows that the Private Leaderboard actually judges. Do **not** add the
D′ scheduler (task #15) on top of the NVFP4 arm: their `ca1bd02` bundle is not the tree D′ was written against.

**2026-10-29 ~15:50 UTC — the "nothing to copy" premise is FALSE, and the slot policy changes.** For four days
every automation here has repeated one sentence: *nobody at ≥35 with few submissions publishes a scoring
kernel, so there is no lever to copy*. It was never checked against the low-submission slice — only against
the top-14-by-score. It is now checked, and it is wrong. `kaggle kernels list --competition
arc-prize-2026-arc-agi-3 --user <name>` for the 26 highest-scoring ≤8-submission medal-club teams returns two
publishers, and both are forks of **our exact stack**:

| Team (live board read 15:46 UTC) | Score | Subs | Rank | What its published notebook adds |
|---|---|---|---|---|
| `vickyfym` | **33.94** | **3** | 106 | 6 kernels; `arc-agi-3-lalalia-final` = dfranzen M2 + solved-level memory |
| `jingshuifu` | 33.22 | 6 | 165 | `arc3-m2-solution-v3` = our notebook, 3 lines different (`ARC3_PRIORITY_PACE '0'→'1'`) |
| `lalalia9` (the memory patch's author) | 35.79 | 22 | 52 | the same `SOLVED_LEVEL_MEMORY_PATCH` |
| `rellik13` / `sirikilohit` (M85, the idea's origin) | 37.97 | 63 | 25 | the memory block, CC-BY 4.0 |
| `weitaoguo` (D′ scheduler author) | 32.61 | 17 | 237 | a replacement slot-priority formula |

**All five attachments are identical to ours** (`dfranzen/intel-qwen3.8-flash-next-w4a16-autoround`,
`dfranzen/albucino-qwen3.8-flash-next-drafter`, `pennyroyal-v253`, `taaf-kaggle-source-bundle-copy`,
competition 133468, `accelerator: nvidiaRtxPro6000`), and their harness patch cell is **hash-identical to ours
(498,799 chars)** — so these are not different models or different engines, they are our program plus a
text/scheduler change. That is the first mechanism found this week which shifts the *mean* rather than the max.

**The mechanism (solved-level memory, `SOLVED_LEVEL_MEMORY_PATCH`).** dfranzen keeps one rolling conversation
per game and drains it from ~118k to ~59k tokens when the window fills. After the first drain on a new level
the agent has lost how it solved the previous levels and re-derives mechanics it already knew. The patch
records, per cleared level, the level number, the actions of the winning attempt, the last 30 actions before it
cleared and the tail of the reasoning in the clearing turn; it then asks the model for one `Rule for level N:`
line and pins the whole block at the **END** of the system prompt. It rewrites the block only when a context
trim is already forcing the server to re-read the conversation, so the shared prefix — and the 94.3 % cache hit
rate — survive. Implementation is 4 monkeypatches on `ToolAgent` (`analyze`, `_build_user_prompt`,
`_note_history_evicted`, `_trim_messages_for_context`), every one wrapped in try/except so "any error leaves
the turn unchanged". Verified against *our* copy of the bundle before believing it: all four methods exist on
`ToolAgent` (patch lines 5513, 6878, 7024, 7126), and `HistoryEntry.action`, `HistoryEntry.frame`,
`Frame.level`, `self._system_prompt`, `self._history_messages`, `self._last_step_summary` and
`inference.agent.runtime_state.load_runtime_state` are all present. Their cell also independently confirms two
arms we had already killed: 12 streams instead of 10 and compacted per-turn prompts "looked neutral on public
games but cost ~7-8 points on the hidden games".

**What was built.** `kaggle_submit_milestone2/arc-agi-3-milestone-2-solution.ipynb` cell 16 = the patch
prepended verbatim above our untouched scored block; git `4b68c33`; pushed as **kernel version 11**. Diff-vs-v9
is exactly one cell and one delta — harness patch, server CFG (CTX 139,264 / mem 0.96 / MAXREQ 10 / fp8 KV),
`ARC3_MAX_ACTIVE_STREAMS=10`, and the `if TRUE_SUBMISSION:` block (concurrency 120, `532*60`, `n_passes 1`,
`analyzer_timeout 900`) are byte-identical. Deliberately NOT copied from vickyfym: `analyzer_timeout 3600`
(no-op — measured max request latency 185.6 s at 10 streams), `ARC3_HTTP_RETRY_INITIAL_SECONDS 3600` (a 1-hour
backoff inside a 9-hour budget), and their 24-game `demo_excluded_games` (interactive branch only).

**The honest size of the bet, and it is the first one this week worth a slot.** The patch's own authors credit
it with **+2.8 on the leaderboard**, which is the number to use — not 33.94. Our seven byte-identical v9 draws
have mean 27.12, sd 2.74, so at a +2.8 mean shift a single draw's p(≥ silver 32.53) goes from ~2.4 % to
~17 % and p(≥ bronze 31.22) from ~10 % to ~32 %. Caveats that are NOT resolved: we cannot attribute
vickyfym's 33.94 row to any specific published version (Kaggle does not expose other teams' kernel versions),
their own notebook text says identical code ranges 27.62–31.47 across draws, and the medal is judged on the
private half from ≤2 selected rows. **New policy, contingent on tonight's rehearsal:** Oct 30 and Oct 31 draw
`-v 11` (SLM); Nov 1–2 stay on whichever arm has actually landed, with v9 as the fallback if the SLM draws show
no mean shift. The D′ scheduler arm (`arc-agi-3-lalalia-dprime-m85tuned`, untested merge, its author's own cell
says "expected range roughly 33-40") is staged as knowledge, not as a slot: it replaces `priority_value` and
`ProgressPace` and removes the unstarted-game queue-jump by overriding `_PriorityGate.acquire`, and every symbol
it needs is present in our bundle, so it can be armed in minutes if the two SLM draws come in clean and low.
A v9 draw can never be beaten by an arm that fails to boot, so **`-v 11` goes into the board only after
`KernelWorkerStatus.COMPLETE` on the interactive rehearsal proves the patch installed without raising.**

**2026-10-29 05:00 UTC — the day's draw taken by the first attempt, and the lines have moved again.**
Step 1 found no row dated today (newest `57731198`, Oct 28, COMPLETE **24.29**), `submission-limits` read
*Submissions today: 0, Lifetime: 13, Remaining today: 1*, and `arc3-milestone2-flashnext-duck` read `COMPLETE`,
so the `-v 9` submit fired and returned *0 submissions remaining today* → ref **`57759315`**, PENDING at
05:00:39. No 400: this time nothing else was racing for the slot. Limits after: *Submissions today: 1,
Lifetime: 14, Remaining today: 0* (the board snapshot still shows `SubmissionCount` 13 for our row because the
PENDING row is not in it).

Fresh full-board download at **05:00:55 UTC**: 4,963 score rows, of which **4,960** carry a `Rank` ≥ 1 — the
same three `Rank` 0 demo teams (Stochastic Goose 0.25, Random Agent 0.12, Just Explore 0.06) — and the two
counting methods give identical rank and identical lines. Field **4,960** (4,942 last night 21:00, +18 in 8 h),
so **bronze rank ≤ 496 = 31.15**, **silver rank ≤ 248 = 32.44**, **gold rank ≤ 19 = 38.84**. Our best row 30.55
is **rank 652** (651 teams strictly above it, board's own `Rank` agrees, three tied at 30.55) = **top 13.15 %**
— the 647 → 652 slide is field growth, our score frozen. Gap now **0.60 to bronze, 1.89 to silver**, the widest
of the week; the bronze line has gone 31.00 → 31.02 → 31.06 → 31.11 → 31.13 → **31.15** across six consecutive
fresh reads. Two other things moved: the **gold line jumped 38.22 → 38.84** because rank 19 is now
Nhan Duc Nguyen at 38.84 in 27 submissions (rank 20 is 38.22), and the leader's submission count is 45 with the
score still 75.56.

Worth recording because it bears on the odds, not because it is a lever: **three teams now sit one to four
places off the silver line with almost no draws** — rank 245 Morten Middelfart 32.49 in **4** submissions, rank
246 atharva 32.48 in **4**, rank 247 Dheeraj Sharma 32.47 in **3**. That is the upper tail of this config's
distribution arriving on a handful of attempts (the 23 byte-identical low-draw accounts topped out at 33.94),
which is the honest case that one draw can still do it — and none of those three publishes a scoring kernel,
so there is nothing to copy, only to draw.

**With today's slot spent, four daily draws remain (Oct 30, 31, Nov 1, Nov 2)**, so §1p's inverted per-draw
rates (p(silver) 1.7–6.7 %, p(bronze) 3.7–14 %) give **P(silver) ≈ 7–24 %** and **P(bronze) ≈ 14–45 %** for the
rest of the run — down from 8–29 % / 17–53 % last night. Six scored byte-identical draws stand on the record
(24.29, 25.51, 26.74, 26.94, 28.18, 30.55; mean 27.04, z ≈ −1.68 against the 23-account population) and the
15:00 / 21:00 guards fill `57759315`'s score into the row above when it lands (~14:00–14:30 UTC on this
cadence). Slot policy unchanged: v9 daily, no arms.

**2026-10-29 15:00:38 UTC — the 15:00 guard filled the score in: `57759315` = 27.65, and it cleared neither
line.** Step 1 found a row dated today and `submission-limits` read *Submissions today: 1, Lifetime: 14,
Remaining today: 0*, so **nothing was submitted and no kernel was touched**. The rerun took ~10 h. **The team's
best row is still 30.55 (`57402922`, Oct 19) — today produced no improvement**, and this draw on its own sits at
rank 1471 = top 29.6 %, i.e. 3.57 below the bronze line and 4.88 below silver.

Fresh download: **4,969** scored teams (4,972 rows less the same three `Rank` 0 demo teams), so the percentile
brackets did not change (top 496 / 248 / 19) yet **every line still rose**: bronze 31.15 → **31.22**, silver
32.44 → **32.53**, gold 38.84 → **38.98**. Our 30.55 is **rank 672** (671 strictly above, three tied at 30.55,
the board's own `Rank` agrees) = **top 13.52 %**. The 652 → 672 slide in ten hours is **not** plain field
growth: the field net grew 9 teams while the count strictly above our row grew 20, so teams crossed *over* us,
and the fact that all three lines rose while their brackets (496 / 248 / 19) stayed fixed says the pack
thickened just under each cutoff rather than at the top of the board. The gap widened to **0.67 to bronze
and 1.98 to silver**, the widest of the week. Seven byte-identical draws now stand on the record (24.29, 25.51,
26.74, 26.94, 27.65, 28.18, 30.55; mean 27.12, best 30.55) and **0 of 7** reach either line. Four daily draws
remain (Oct 30, 31, Nov 1, Nov 2), so the band two paragraphs above — **P(silver) ≈ 7–24 %, P(bronze) ≈
14–45 %** — is still the standing figure, and ≈ **3–13 %** once the ≤2-selected-rows-on-the-private-half rule
is applied. Leader unchanged at 75.56 (45 submissions). Slot policy unchanged: v9 daily, no arms.

**2026-10-28 ~08:55 UTC — a second guard on the same day, arriving 85 s after the draw, and what its 400 means.**
This attempt passed step 1 honestly: its first `submissions -v` at 08:50:30 showed nothing newer than
`57689693`, `submission-limits` read *Submissions today: 0, Remaining today: 1*, and
`arc3-milestone2-flashnext-duck` read `KernelWorkerStatus.COMPLETE`. It then issued the mandated `-v 9` submit
and got **`400 Client Error: Bad Request for url:
https://api.kaggle.com/v1/competitions.CompetitionApiService/CreateCodeSubmission`**. Re-reading after the
error explains it: exactly one row dated 2026-10-28 exists (`57731198`, PENDING) and `submission-limits` now
reads *Submissions today: 1, Lifetime: 13, Remaining today: 0* — i.e. the interactive draw logged above had
consumed the allowance ~80 s earlier, and this was the day's second draw being refused. Not a closed window
(the daily allowance exists and is now merely spent) and not a bad file. **No retry was issued.**

**Operational lesson: a 400 on `CreateCodeSubmission` must be resolved by re-reading `submissions -v` before
anything else**, because the same generic message also covers "you already submitted today". And because the
guard is read-then-submit rather than atomic, two sessions scheduled near each other can both pass step 1 in
the same minute; the loser gets a 400 and the day still has exactly one row, which is the safe failure mode.
The fix is scheduling, not code — if the interactive/hand path and a cron entry can both be live, one of them
should not fire in the 05:00–09:00 window.

Boundaries re-read from a fresh download at **08:52:46 UTC**, 2026-10-28 — the field is now **4,930** scored
teams (4,908 yesterday 14:42 → +22), so **bronze rank ≤ 493 = 31.06**, **silver rank ≤ 246 = 32.34**, **gold
rank ≤ 19 = 38.22**. Both medal lines moved up overnight even though our score did not: yesterday's bronze
line 31.00 now sits at ranks 503–510 (outside the top 493) and yesterday's silver line 32.31 at ranks 249–250
(outside the top 246), which is the clearest possible demonstration of why these numbers are re-read daily and
never carried forward. Our best row 30.55 is **rank 625** (624 teams strictly above it, three tied at 30.55,
and the board's own `Rank` column says 625) = **top 12.68 %**, so the gap is now **0.51 to bronze and 1.79 to
silver** — both wider than yesterday (0.45 / 1.76) purely through field growth. 122 teams lie between 30.55
and 31.00; 376 between 30.55 and 32.34. Top of board 75.56, Yi-Chia Chen @ Fields Model, 44 submissions.

**2026-10-28 15:00:47 UTC — the 15:00 guard: slot already spent, and the lines moved again inside the same
day.** Step 1 found exactly one row dated today (`57731198`) and `submission-limits` read *Submissions today:
1, Lifetime: 13, Remaining today: 0*, so **nothing was submitted and no kernel was touched**. Fresh full-board
download, **4,941** scored teams (+11 in the 6 h since the 08:52 read), which re-cut the lines:
**bronze rank ≤ 494 = 31.11**, **silver rank ≤ 247 = 32.41**, **gold rank ≤ 19 = 38.22**. Our best row 30.55 is
**rank 637** (636 teams strictly above it; the board's own `Rank` agrees) = **top 12.89 %**, so the gap is now
**0.56 to bronze and 1.86 to silver** — both wider than this morning's 0.51 / 1.79 while our score is
unchanged, and the 08:52 lines (31.06 / 32.34) are already stale. `57731198` was still
`SubmissionStatus.PENDING` at 6.2 h into a ~9 h rerun, so **today's draw has no verdict yet** — when it lands
it clears bronze only at ≥ 31.11 and silver only at ≥ 32.41, and the 21:00 guard writes the result into the row
above.

**2026-10-28 21:00 UTC — evening guard: slot already spent, and the day's verdict is 24.29, our worst draw.**
Step 1 found exactly one row dated today (`57731198`, 08:51:09), now `SubmissionStatus.COMPLETE` at **24.29**,
and `submission-limits` agrees — *Submissions today: 1, Lifetime: 13, Remaining today: 0* — so nothing was
submitted and no kernel was touched. Lifetime 13 equals the board's own `SubmissionCount` for our row
(TeamId 15515494), so every submission to date is still ours.
Fresh full-board download at 21:00:37 UTC: **4,942** ranked teams (4,945 score rows minus the same 3 `Rank` 0
demo rows), up 29 in 24 h. Lines re-cut again: **bronze rank ≤ 494 = 31.13**, **silver rank ≤ 247 = 32.43**,
**gold rank ≤ 19 = 38.22**. Our best row 30.55 is **rank 647** (646 teams strictly above it; the board's own
`Rank` agrees) = **top 13.09 %**, so the gap is **0.58 to bronze and 1.88 to silver** — wider again than the
15:00 read (0.56 / 1.86) and the morning's (0.51 / 1.79), with our score frozen throughout. The bronze line has
risen 31.00 → 31.02 → 31.06 → 31.11 → 31.13 over the 30 h since Oct 27 14:42 (each step a fresh download); that
is the whole story of this week — the target moves away faster than an average draw can approach it. Top of
board still 75.56 (Yi-Chia Chen @ Fields Model, 44 subs).
**Today's draw cleared neither line**: 24.29 sits at **rank 1949 = top 39.44 %**, 6.84 short of bronze. It is
our lowest of the config, and because the leaderboard keeps the max it cost nothing — the best row stays 30.55.

**Six byte-identical draws of ours are now on the record — 24.29, 25.51, 26.74, 26.94, 28.18, 30.55 — mean
27.04, sd 2.17, and 0 of 6 reach either line.** Against the 23 byte-identical low-draw accounts (mean 28.91,
sd 2.74) the six-draw mean is now **z = −1.68**, not the −1.08 that five draws gave. That is still inside
ordinary sampling noise at n = 6 (one-sided ≈ 5 %), and our best row remains that population's ~65th
percentile, so it is not evidence the config is broken for us — but the margin has clearly narrowed and the
honest read is that six draws have produced nothing above 30.55. **After today, five daily draws remain
(Oct 29, 30, 31, Nov 1, Nov 2)**, so §1p's inverted per-draw rates give **P(silver) ≈ 8–29 %** and
**P(bronze) ≈ 17–53 %** for the rest of the run — the same window the 15:00 entry computed, because today's slot
is already the one that reduced it from six. Both are public-half
figures over *candidate* draws; medals judge at most **two selected rows on the private half**, which is why
the ~3–13 % figure remains the number to quote for the medal itself. Slot policy unchanged: v9 daily, no arms.

**2026-10-28 08:51 UTC — the day's draw was taken by hand, three minutes ahead of the cron that was also
trying.** The 05:00 task (`ecfa8d23`) first fired late, at ~07:32 UTC, and died in **3,934 ms** with `lastError
{"type":"session-failed"}` (`consecutiveFailures: 1`) — no step 1, no submission, no RUNBOOK edit, so at 08:45
the repo looked exactly like a day whose slot had been deliberately skipped. `submission-limits` still said
*Remaining today: 1*, so the slot was checked, the kernel confirmed `KernelWorkerStatus.COMPLETE`, and `-v 9`
was submitted by hand at **08:51:09** → ref **`57731198`** (PENDING as of 08:54). Qoder then re-ran the same
cron at ~08:50–08:52 and it got the 400 documented in the block above — the automated path and the hand path
were both live inside one minute, and the hand submit won by ~80 s. Score expected in ~9 h (≈17:45–18:30 UTC);
the 15:00 and 21:00 attempts will find this row dated today and spend nothing.

**The lesson is about the automation design, not the config.** Three attempts at three different times was
built to survive a *rejected submit* (slot not yet rolled over, kernel still running). It does not reliably
survive a *session that cannot start*: a `session-failed` run leaves no submission row, no RUNBOOK edit and no
push — the only trace is `lastRunStatus`/`lastError` inside the cron store, which nothing reads unless a human
or an agent opens the day. Whether Qoder re-fires such a job is not something this repo can observe or
control (today it did, ~80 min later, and today that saved nothing only because a live session was open).
Practical rule for the five days left: **if today's row is missing from the chase log when the file is next
read, run `submission-limits` before assuming anything was drawn** — an empty slot is recoverable only until
23:59 UTC, and a failed cron is silent in git.

**2026-10-27 05:00 UTC — the slot was spent on the proven config and the draw landed at 26.94.** Step 1 found
no row dated today (newest was Oct 26 11:27), `submission-limits` said *Submissions today: 0, Remaining
today: 1*, and `arc3-milestone2-flashnext-duck` was `COMPLETE`, so `-v 9` was submitted as the morning draw →
ref **`57689693`**, now `SubmissionStatus.COMPLETE` at **26.94**. The 15:00 and 21:00 attempts will find this
row dated today, stop at step 1 and spend nothing (`Remaining today: 0` confirmed at 14:41 UTC).

Boundaries re-read from a fresh download at 14:42 UTC (not from the Oct 26 numbers): the field is now
**4,908** scored teams (4,869 on Oct 26 21:59 → 4,894 at 05:01 → 4,908 by afternoon, +39 in a day), so
**bronze rank ≤ 490 = 31.00**, **silver rank ≤ 245 = 32.31**, **gold rank ≤ 19 = 38.22**. Our best row 30.55
is **rank 606** (board's own `Rank`) = **top 12.35 %** — the 9-place slide since this morning is field growth,
not a score change; our row is untouched. The gap is now **0.45 to bronze and 1.76 to silver**, and silver has
moved *away* twice as fast as we could have closed it with an average draw. Top of board: 75.56, Yi-Chia Chen
@ Fields Model, 43 submissions.

Independently re-read at **15:00:43 UTC** by the 15:00 attempt, which found the day's slot already spent and
submitted nothing: the field is unchanged at **4,908**, so the three lines above still hold exactly (490 /
245 / 19 = 31.00 / 32.31 / 38.22, our row still rank 606). **Verdict for the day: the 26.94 draw cleared
neither line** — 4.06 short of bronze, and that row on its own sits at rank 1543 (top 31.4 %) — while the
team's best row stays 30.55 at rank 606, so the draw cost nothing. `submission-limits` 15:00 UTC: *Submissions
today: 1, Lifetime: 12, Remaining today: 0*.

**2026-10-27 21:00 UTC — evening guard: slot already used, nothing submitted, and the field moved again.**
Step 1 found `57689693` dated today (`COMPLETE`, **26.94**), and `submission-limits` agrees — *Submissions
today: 1, Lifetime: 12, Remaining today: 0* — so this attempt fired no submit. The 05:00 task landed the draw
first, which is the three-attempt design working, not a failure.
Fresh download at 21:00:47 UTC, counted from `Rank > 0` rows (4,916 score rows minus the same 3 unranked demo
teams): the field is **4,913**, up 5 ranked teams from the afternoon read, and the boundaries have ticked again
— **bronze ≤ 491 = 31.02**, **silver ≤ 245 = 32.34**, **gold ≤ 19 = 38.22**. Our best row 30.55 is
**rank 612** — the board's own `Rank` for "Hendric Nazarius", and independently the count of strictly-higher
scores plus one — i.e. **top 12.46 %**, with the 6-place slide from 606 being pure field growth. Gaps are now
**0.47 to bronze and 1.79 to silver**. (The 3 rank-0 rows do not change any boundary: floor(4,913 × 0.1) and
floor(4,916 × 0.1) are both 491, and both give 245 for 5 %.) Top of board is now **75.56**, Yi-Chia Chen
@ Fields Model.
The day's draw also gets a rank of its own, from a second download at 21:02 UTC (same field, 4,913): 26.94 sits
at **rank 1559 = top 31.73 %**, so it cleared neither line, and the team's best row is unaffected. **Six draws
remain (Oct 28 → Nov 2)** and the selection step in the paragraph below is still the outstanding must-do — this
guard cannot perform it, because selecting Final Submissions is browser-only.

**2026-10-27 ~15:00 UTC — read the rules on final judging, and the "the LB keeps our best row" assumption
does NOT carry into the medal.** From the live rules page (browser, since WebFetch returns only the JS
shell): §2 SUBMISSION LIMITS — *"a. You may submit a maximum of 1 (1) Submissions per day. **b. You may
select up to two (2) Final Submissions for judging.**"*; §3.18(c) — *"A 'Final Submission' is the Submission
selected by the user, **or automatically selected by Kaggle in the event not selected by the user**, that
is/are used for **final placement** on the competition leaderboard"*; §8 — *"The **Private** Leaderboard
determines the final standing in the competition."* Three consequences:
1. The best-row rule we have been leaning on is a **public-LB display** behaviour. What medals are made of
   is the **private half of at most two selected runs** — and we have never observed a private score, so all
   of the 10–34 % band arithmetic in this entry is a *public-half* proxy for a private-half draw.
2. **Action required before 2026-11-02 23:59 UTC:** select our two highest-public rows as Final Submissions.
   The installed CLI cannot do it — `kaggle competitions --help` lists
   `{list,files,download,submit,submissions,leaderboard,team-submissions,submission-limits,episodes,replay,logs,
   pages,hosts,data,settings,solution,launch,init,create,topics,topic-messages}` and none of them selects a
   final submission — so it is a browser action on the team's logged-in session (today the connected browser
   is signed **out**, so nobody can do it silently). `kaggle competitions team-submissions 15515494` returns
   exactly one row today (`57402922`, 30.55), which is the public-LB row, not proof of selection state.
   Selecting the two best-public rows is safe under both possible judge configs, and leaving it to
   "automatically selected by Kaggle" is the one thing we cannot verify after the fact.
3. It also means the honest medal arithmetic is a **max over ~2 judged rows, not over the seven draws**. With
   per-draw p(silver) ≈ 1.7–6.7 %, two judged rows give ≈ **3–13 %**, not 10–34 % — the extra draws are still
   worth taking because each one is another *candidate* for the pair, and because a strong public half is the
   only observable signal that its private half is strong too, but their marginal value is sub-linear. Say
   this rather than quoting 10–34 % as a medal probability. Draw policy is unchanged; the **selection step is
   the new must-do**.

Five byte-identical draws of ours are now on the record — **25.51, 26.74, 26.94, 28.18, 30.55** — mean
**27.58**, median 26.94, and **0 of 5 reach either medal line**. That is *not* evidence the config is broken
for us: against the 23 byte-identical low-draw accounts (mean 28.91, sd 2.74) our five-draw mean sits at
**z = −1.08**, i.e. about one standard error of ordinary sampling noise, and our best row is the **65th**
percentile of that population (not the 80th claimed earlier — corrected below). With today's slot spent there
are **six daily draws left** (Oct 28 → Nov 2). Converting the population's account-level rates back to a
per-draw rate — 3/23 clears silver and 6/23 clears bronze, but those are maxima over 2–8 draws, so
per-draw p(silver) ≈ **1.7–6.7 %** and p(bronze) ≈ **3.7–14 %** — over six draws that is
**P(silver) ≈ 10–34 %** and **P(bronze) ≈ 20–60 %**. This supersedes the 15–40 % / 22–63 % figures, which were
computed for seven draws and used the wider 2–13 % band. Slot policy is unchanged: v9 every day, no arms —
a diverted slot still costs the whole per-draw figure, and there is no arm left whose prior is positive.

**2026-10-26 21:00 UTC — the day's draw came in low, and it cost us nothing.** `-v 9` scored **25.51**, a
below-average draw of unchanged code; the team's best row stays 30.55, so the only movement in rank (581 →
595) is **14 more teams now sitting above us**, while the field itself grew by 31 to **4,867**. That growth is
why the medal boundaries are now **bronze ≤ 486 = 30.99, silver ≤ 243 = 32.29, gold ≤ 19 = 38.20** — all
re-read from the downloaded board, and all slightly further away than yesterday's numbers. The 21:00 evening
guard found `57664497` already COMPLETE for today and `Remaining today: 0`, so it submitted nothing; rank 595
is the board's own `Rank` value for our row (team "Hendric Nazarius", score 30.55).

Six draws of the identical config are now on the record: **25.51, 26.74, 28.18, 30.55, 31.46, 38.85** —
mean **30.21**, sd **4.79** (was mean 31.16 / sd 4.69 on five). Read the hit rate two ways, as always: raw
count gives **1/6 = 17 %** per draw clearing 32.29, the normal fit gives **33 %**. Over the seven slots left
(Oct 27 → Nov 2) that is **P(silver) ≈ 72–94 %** and **P(bronze) ≈ 94–98 %**. The plan does not change on a
bad draw — the distribution is what we are betting on, and 25.51 is inside it.

**Correction, 2026-10-27 00:20 UTC — that 72–94 % is wrong and the numbers above it are the contaminated
pool.** 31.46 and 38.85 are not our draws: they are the best rows of a 6-submission and a 202-submission
account running the same program, i.e. maxima, not samples of our per-draw distribution. Measured against 23
accounts whose harness cell and scored block hash byte-identically to ours and who had ≤8 submissions, the
distribution is median **28.52**, max **33.94**, **0 of 23 at 35 or above**, 3 of 23 at or above the silver
line and 6 of 23 at or above bronze. Corrected: **P(silver) ≈ 15–40 %**, **P(bronze) ≈ 22–63 %** over the
seven remaining draws. The plan (draw the same config daily) is unchanged; the expectation is not.

**2026-10-26: the author's live kernel was re-diffed against ours, and there is no hidden fix in it.** Pulled
`dfranzen/arc-agi-3-milestone-2-solution` (23 cells, 540,581 source chars vs our 541,746) and diffed
cell-by-cell. Exactly three cells differ, and none of them is on the scored path:
- **cell 16** — the differences are all inside `else:`, the interactive branch (our `FAST_PROXY` pins:
  `n_passes = 1`, `concurrency = 10`, `max_runtime_s_per_game = 5*60`). The `if TRUE_SUBMISSION:` block is
  byte-identical: `concurrency 120`, `max_runtime_s_per_game 532*60`, `n_passes = 1`.
- **cell 4** — `SERVER_STARTUP_TIMEOUT`, ours `60*60` vs theirs `12*60`. Ours only waits *longer* for the
  server, so it cannot be what costs us points.
- **cell 12** — our cp312 venv patch (the image kernel is 3.13 and the Pennyroyal wheelhouse is cp312-only);
  without it the run does not build at all.

**Conclusion, and it closes the "just fix the code" question with evidence:** their 38.85 and our 30.55 are
draws of the same scored configuration, so no edit to our copy is known to raise the mean. Every change we
have actually submitted has lost a slot (24.65 at concurrency 28; 5.68 and 5.34 on the port), which is why
the live allowance goes to a draw of `-v 9` — the one move that can only ever raise the team's best row.

**2026-10-26 coverage finding.** The 00:10 UTC one-shot (`1748cb44`) deleted itself as a `deleteAfterRun` job
without leaving a submission row and without leaving a readable session, so the day's draw would have been
lost had it not been taken by hand at 11:27. A one-shot local task is therefore *not* evidence the slot was
used — only `kaggle competitions submissions` is. A fourth daily attempt (`0 5 * * *` UTC) was added so each
day now has 05:00 / 15:00 / 21:00 idempotent attempts instead of two.

**The team entry that actually holds the allowance (read from the downloaded board, 2026-10-25 23:10 UTC).**
Our row on the public leaderboard is **TeamId 15515494, team name "Hendric Nazarius", members
`hendricnazarius, marcbenjamessamuel, vyassenthilkumar`**, Score 30.55, Rank 581, SubmissionCount **10** —
exactly the lifetime count `submission-limits` reports, so all ten submissions to date are ours and no
teammate has spent the shared allowance yet. That could change while the user is away, and it is the one way
a day's draw disappears legitimately: the allowance is per *team*, so a teammate's submit on a Monday makes
every automation here stop at step 1 for that day. The field has meanwhile grown to **4,867 scored teams**
(4,833 on Oct 25, 4,836 on Oct 26 morning) and the boundaries have moved with it — bronze ≤ 486 = 30.99,
silver ≤ 243 = 32.29, gold ≤ 19 = 38.20 — which is why rank, not score, is the number to re-read daily.

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

