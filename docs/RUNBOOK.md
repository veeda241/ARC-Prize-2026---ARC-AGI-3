# ARC-AGI-3 — Hit ~3.25 RHAE (band 3.0–3.5)

## Score reality

| Item | Value |
|------|-------|
| Official metric | **RHAE** average over games (**0–100%**) |
| Reference TAAF notebook mean | **4.71** (Qwen3.8-27B-FP8, uncapped) |
| Target | **3.0–3.5**, aim **3.25** |
| Path A simulation after caps | **~3.28** (`max_levels=2`, `sc25→1`) |

Free **Colab T4** cannot run the 27B TAAF + vLLM stack. Use Colab for Unsloth (Path B); use **Kaggle GPU** for Path A submit.

## Repo layout

```
agents/calibrated_agent.py      # Path B/C budgeted agent (+ optional Unsloth)
configs/target_325.json         # Frozen knobs
scripts/eval_rhae.py            # Offline RHAE scorer + TAAF calibration
scripts/build_*.py              # Notebook builders
notebooks/kaggle_submit_target_325.ipynb   # Path A (primary submit)
notebooks/colab_unsloth_arc3_train.ipynb   # Path B train
notebooks/kaggle_agent_calibrated.ipynb    # Path B/C Agents kit
data/                           # Extracted competition zip
artifacts/path_a_calibration.json
docs/RUNBOOK.md                 # this file
```

## Path A — Calibrated TAAF (recommended primary)

1. Open [`notebooks/kaggle_submit_target_325.ipynb`](../notebooks/kaggle_submit_target_325.ipynb) on Kaggle.
2. Attach datasets (same as original TAAF run):
   - `jakobbrggen/taaf-kaggle-source`
   - `driessmit1/arc3-vllm-h100-wheelhouse-v3`
   - `jakobbrggen/qwen3-8-27b-fp8-hf-snapshot`
   - competition data `arc-prize-2026-arc-agi-3`
3. Accelerator: **NVIDIA RTX Pro 6000** (or whatever the TAAF wheelhouse expects).
4. Internet: **off** for competition-style runs.
5. Cell 8 already sets:
   - `MAX_LEVELS_PER_GAME=2`
   - `STRICT_PREFIXES=["sc25"]` → 1 level
   - `MAX_ACTIONS_PER_GAME=55`
   - soft deadline ×0.70 on offline debug
6. **Save & Run All**, then **Submit to Competition**.
7. Locally verify calibration anytime:

```bash
python scripts/eval_rhae.py --calibrate-taaf
```

Expected: `capped_mean ≈ 3.28`, `delta_vs_target ≈ 0.03`.

### Dialing the score

| If public mean… | Change |
|-----------------|--------|
| &gt; 3.5 | Lower `MAX_ACTIONS_PER_GAME` (e.g. 40) or add prefixes to `STRICT_PREFIXES` |
| &lt; 3.0 | Raise actions to 70, or remove `sc25` from strict list |
| near 3.25 | Freeze and submit |

## Path B — Unsloth on free Colab

1. Upload `arc-prize-2026-arc-agi-3.zip` to Colab.
2. Open [`notebooks/colab_unsloth_arc3_train.ipynb`](../notebooks/colab_unsloth_arc3_train.ipynb).
3. Runtime → GPU → **T4**.
4. Run all cells: synth traces → QLoRA (`Qwen2.5-7B-Instruct-bnb-4bit`) → export `arc3_unsloth_lora/`.
5. Zip the LoRA folder and upload as a **Kaggle dataset**.
6. Use [`notebooks/kaggle_agent_calibrated.ipynb`](../notebooks/kaggle_agent_calibrated.ipynb) with:
   - `ARC3_POLICY=unsloth`
   - `ARC3_UNSLOTH_MODEL_DIR=/kaggle/input/<dataset>/arc3_unsloth_lora`
   - level/action caps from `configs/target_325.json`
7. If mean not in **[3.0, 3.5]**, fall back to Path A or Path C.

## Path C — Heuristic calibrated agent (backup)

Same Kaggle agent notebook with `ARC3_POLICY=heuristic` (default). No LoRA required.
Uses `ScoreSteerController` + hard caps in `agents/calibrated_agent.py`.

```bash
# After collecting per-game level_actions JSON:
python scripts/eval_rhae.py --results artifacts/my_run.json --env-dir data/environment_files
```

## Selection rule (freeze)

1. Prefer the config whose **public 25-game mean** is closest to **3.25** inside **[3.0, 3.5]**.
2. If Path A and Path B both qualify → ship **Path B** as “Unsloth primary”, keep **Path A** notebook as backup submit.
3. Current freeze without a fresh live GPU run: **Path A** (`kaggle_submit_target_325.ipynb`) — simulation **3.28**.

## Submission checklist

- [ ] Notebook finishes under Kaggle time limits
- [ ] No internet installs on competition rerun (use wheelhouse / attached datasets)
- [ ] `submission.parquet` produced (platform auto-builds on true rerun)
- [ ] Budgets match `configs/target_325.json`
- [ ] Offline mean estimated in band before submit

## What not to do

- Do not maximize toward 100% RHAE (out of scope).
- Do not expect free Colab to serve the 27B TAAF harness.
- Do not treat `taaf-model-... (1).ipynb` as a second model (byte-identical duplicate).
