# ARC-Prize-2026---ARC-AGI-3

Build AI systems that adapt on the fly to new tasks in the ARC environment, and develop approaches that learn quickly, generalize well, and solve problems never seen before.

## Target score (this fork)

Aim for mean **RHAE ≈ 3.25** (band **3.0–3.5**). Official scale is **0–100%**, not 0–5.

See **[docs/RUNBOOK.md](docs/RUNBOOK.md)** for the multi-path plan:

| Path | Notebook | Notes |
|------|----------|-------|
| **A (primary)** | [`notebooks/kaggle_submit_target_325.ipynb`](notebooks/kaggle_submit_target_325.ipynb) | Calibrated TAAF 27B; sim mean **~3.28** |
| **B** | [`notebooks/colab_unsloth_arc3_train.ipynb`](notebooks/colab_unsloth_arc3_train.ipynb) | Unsloth QLoRA on free Colab T4 |
| **C** | [`notebooks/kaggle_agent_calibrated.ipynb`](notebooks/kaggle_agent_calibrated.ipynb) | Heuristic / Unsloth agent + budgets |

```bash
python scripts/eval_rhae.py --calibrate-taaf
```
