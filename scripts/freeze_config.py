#!/usr/bin/env python3
"""Write frozen Path-A policy artifacts for the ~3.25 RHAE target."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_rhae import calibrate_taaf  # noqa: E402

OUT = ROOT / "artifacts" / "frozen_target_325.json"
summary = calibrate_taaf(target=3.25, default_max_levels=2, strict_prefixes=["sc25"])
frozen = {
    "selected_path": "A",
    "reason": (
        "Path A TAAF calibration lands at ~3.28 on the public reference run without "
        "requiring Colab GPU training. Path B/C notebooks are ready as alternates."
    ),
    "submit_notebook": "notebooks/kaggle_submit_target_325.ipynb",
    "backup_notebooks": [
        "notebooks/kaggle_agent_calibrated.ipynb",
        "notebooks/colab_unsloth_arc3_train.ipynb",
    ],
    "config": json.loads((ROOT / "configs" / "target_325.json").read_text(encoding="utf-8")),
    "calibration": summary,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
print(json.dumps({"wrote": str(OUT), "mean": summary["mean"], "path": "A"}, indent=2))
