"""Build the Kaggle submission notebook from agent/my_agent.py.

Follows the official ARC-AGI-3 Kaggle starter:
  1. Offline pip install from competition wheels
  2. Write MyAgent to /tmp (not /kaggle/working, so it is not an output file)
  3. On competition rerun: wait for gateway, register agent, run main.py
  4. On Save & Run All: write a dummy submission.parquet
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
AGENT_SRC = ROOT / "agent" / "my_agent.py"
OUT = ROOT / "notebooks" / "kaggle_submit.ipynb"
OUT_ALIAS = ROOT / "notebooks" / "kaggle_agent_calibrated.ipynb"
PKG = ROOT / "kaggle_submit_calibrated"
KAGGLE_USER = "vyassenthilkumar"
KERNEL_SLUG = "arc3-calibrated-submit"

ACCELERATOR = "t4"
_ACCELERATORS = {
    "cpu": {"name": "none", "gpu": False},
    "t4": {"name": "nvidiaTeslaT4", "gpu": True},
    "p100": {"name": "nvidiaTeslaP100", "gpu": True},
    "rtx6000": {"name": "nvidiaRtx6000", "gpu": True},
}


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {"trusted": True},
        "outputs": [],
        "execution_count": None,
        "source": source,
    }


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def build() -> dict:
    if not AGENT_SRC.exists():
        raise SystemExit(f"Could not find {AGENT_SRC}")
    agent_body = AGENT_SRC.read_text(encoding="utf-8")
    if not agent_body.endswith("\n"):
        agent_body += "\n"

    install_cell = code_cell(
        "!pip install --no-index --find-links \\\n"
        "    /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels \\\n"
        "    arc-agi python-dotenv\n"
    )
    write_agent_cell = code_cell("%%writefile /tmp/my_agent.py\n" + agent_body)

    run_cell = code_cell(
        dedent(
            """\
            import os

            os.environ.setdefault("ARC3_TARGET_MEAN_RHAE", "3.25")
            os.environ.setdefault("ARC3_MAX_LEVELS_PER_GAME", "2")
            os.environ.setdefault("ARC3_MAX_ACTIONS_PER_GAME", "55")
            os.environ.setdefault("ARC3_STRICT_PREFIXES", "sc25")
            os.environ.setdefault("ARC3_STRICT_MAX_LEVELS", "1")
            os.environ.setdefault("ARC3_SEED", "7")

            if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
                # Wait for the gateway sidecar to be ready.
                !curl --fail --retry 999 --retry-all-errors --retry-delay 5 \\
                      --retry-max-time 600 http://gateway:8001/api/games

                # Copy the framework into a writable location.
                !cp -r /kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents \\
                       /kaggle/working/ARC-AGI-3-Agents

                # Drop our agent in as a framework template.
                !cp /tmp/my_agent.py \\
                    /kaggle/working/ARC-AGI-3-Agents/agents/templates/my_agent.py

                # Slim registry: upstream __init__.py imports langgraph/smolagents.
                with open("/kaggle/working/ARC-AGI-3-Agents/agents/__init__.py", "w") as f:
                    f.write(
                        "from typing import Type\\n"
                        "from dotenv import load_dotenv\\n"
                        "from .agent import Agent, Playback\\n"
                        "from .swarm import Swarm\\n"
                        "from .templates.random_agent import Random\\n"
                        "from .templates.my_agent import MyAgent\\n"
                        "\\n"
                        "load_dotenv()\\n"
                        "\\n"
                        "AVAILABLE_AGENTS: dict[str, Type[Agent]] = {\\n"
                        "    'random': Random,\\n"
                        "    'myagent': MyAgent,\\n"
                        "}\\n"
                    )

                with open("/kaggle/working/ARC-AGI-3-Agents/.env", "w") as f:
                    f.write(
                        "SCHEME=http\\n"
                        "HOST=gateway\\n"
                        "PORT=8001\\n"
                        "ARC_API_KEY=test-key-123\\n"
                        "ARC_BASE_URL=http://gateway:8001/\\n"
                        "OPERATION_MODE=online\\n"
                        "ENVIRONMENTS_DIR=\\n"
                        "RECORDINGS_DIR=/kaggle/working/server_recording\\n"
                    )

                # Gateway records actions and emits submission.parquet.
                !cd /kaggle/working/ARC-AGI-3-Agents && \\
                    MPLBACKEND=agg \\
                    python main.py --agent myagent
            """
        )
    )

    dummy_cell = code_cell(
        dedent(
            """\
            import os

            if not os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
                import pandas as pd

                submission = pd.DataFrame(
                    data=[["1_0", "1", True, 1]],
                    columns=["row_id", "game_id", "end_of_game", "score"],
                )
                submission.to_parquet("/kaggle/working/submission.parquet", index=False)
                submission.head()
            """
        )
    )

    accel = _ACCELERATORS[ACCELERATOR]
    return {
        "metadata": {
            "kernelspec": {
                "language": "python",
                "display_name": "Python 3",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "mimetype": "text/x-python",
                "file_extension": ".py",
                "pygments_lexer": "ipython3",
            },
            "kaggle": {
                "accelerator": accel["name"],
                "isInternetEnabled": False,
                "isGpuEnabled": accel["gpu"],
                "language": "python",
                "sourceType": "notebook",
            },
        },
        "nbformat_minor": 4,
        "nbformat": 4,
        "cells": [
            markdown_cell(
                "# ARC Prize 2026 — ARC-AGI-3 calibrated submit\n\n"
                "Built from `agent/my_agent.py` via `scripts/build_agent_notebook.py`.\n\n"
                "Attach **only** the competition data `arc-prize-2026-arc-agi-3`. "
                "Internet **off**. After Save & Run All succeeds, click "
                "**Submit to Competition** and choose `submission.parquet`.\n"
            ),
            install_cell,
            write_agent_cell,
            run_cell,
            dummy_cell,
        ],
    }


def main() -> None:
    nb = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(nb, indent=1, ensure_ascii=False)
    OUT.write_text(text, encoding="utf-8")
    OUT_ALIAS.write_text(text, encoding="utf-8")

    PKG.mkdir(parents=True, exist_ok=True)
    (PKG / "kaggle_submit.ipynb").write_text(text, encoding="utf-8")
    meta = {
        "id": f"{KAGGLE_USER}/{KERNEL_SLUG}",
        "title": "ARC3 Calibrated Submit",
        "code_file": "kaggle_submit.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": _ACCELERATORS[ACCELERATOR]["gpu"],
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "kernel_sources": [],
        "model_sources": [],
    }
    (PKG / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    print(f"wrote {PKG / 'kaggle_submit.ipynb'}")
    print(f"wrote {PKG / 'kernel-metadata.json'}")


if __name__ == "__main__":
    main()
