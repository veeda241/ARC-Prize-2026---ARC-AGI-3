#!/usr/bin/env python3
"""Stage the vLLM + NVFP4/FP8 port as its own Kaggle kernel.

The upstream notebook (lordhansolo's generic TAAF launcher) is left byte-identical except for
the empty "Inline customization hook" cell, which is the only place the harness authors
intended per-run edits. Keeping the edit inside that cell means the port can be re-staged from
a fresh pull at any time.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "upstream" / "built-on-tufa-labs-duck-harness-milestone-2.ipynb"
OUT_DIR = ROOT / "kaggle_submit_milestone2_vllm"
NOTEBOOK = OUT_DIR / "arc3-milestone2-vllm-port.ipynb"

OWNER = "vyassenthilkumar"
SLUG = "arc3-milestone2-vllm-port"
# Kaggle derives the kernel slug from the TITLE, not from the metadata "id", so the two must
# agree word-for-word or pushes land on a kernel nobody is polling.
TITLE = "ARC3 Milestone2 Vllm Port"

HOOK = """# Inline customization hook.
# Make one-off changes to `bm`, `bm.games`, or `bm.solver` here before the run starts.
if not run_as_submission:
    # Feasibility trim: the interactive session only has to prove that the vLLM runtime
    # overlay, the NVFP4/FP8 checkpoint and the TAAF analyzer come up and take actions.
    # The scored rerun never enters this branch, so the hidden set keeps the bundle's own
    # config (concurrency=14, max_runtime_s_per_game=3918, n_passes=1).
    bm.label = f"{bm.label}-feasibility"
    bm.n_passes = 1
    bm.solver.concurrency = 4
    bm.solver.max_runtime_s_per_game = 600.0
    bm.games = list(bm.games)[:4]
"""


def staged_notebook() -> dict:
    nb = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = nb["cells"]
    hook_index = next(
        i
        for i, cell in enumerate(cells)
        if cell["cell_type"] == "code"
        and "".join(cell["source"]).lstrip().startswith("# Inline customization hook.")
    )
    lines = [f"{line}\n" for line in HOOK.rstrip("\n").split("\n")]
    cells[hook_index]["source"] = lines
    cells[hook_index]["outputs"] = []
    cells[hook_index]["execution_count"] = None
    return nb


def metadata() -> dict:
    return {
        "id": f"{OWNER}/{SLUG}",
        "title": TITLE,
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "machine_shape": "NvidiaRtxPro6000",
        "dataset_sources": [
            "lordhansolo/taaf-kaggle-source",
            "lordhansolo/vllm-main-e975732-arc3",
        ],
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "kernel_sources": [],
        "model_sources": [
            "lordhansolo/qwen3-8-flash-next-mixed-nvfp4-fp8/PyTorch/hf-mixed-mtp-nvfp4/1"
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    nb = staged_notebook()
    text = json.dumps(nb, ensure_ascii=False, indent=1)
    json.loads(text)  # the push endpoint rejects a malformed notebook
    if not args.dry_run:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        NOTEBOOK.write_text(text, encoding="utf-8")
        (OUT_DIR / "kernel-metadata.json").write_text(
            json.dumps(metadata(), indent=2) + "\n", encoding="utf-8"
        )
    print(f"{NOTEBOOK}: {len(text)} bytes, hook cell replaced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
