"""Build a competition notebook from Duck v12 (LB ~2.23) with 2-pass boost settings."""
from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "lb-9-arc3-duck-v12-with-qwen-3-8-27b.ipynb"
OUT = ROOT / "notebooks" / "kaggle_duck_v12_boost.ipynb"
PKG = ROOT / "kaggle_submit_duck_v12"

MARKDOWN = """# ARC-AGI-3 Duck v12 boost — Qwen3.8-27B-FP8

Base: `lb-9-arc3-duck-v12-with-qwen-3-8-27b.ipynb`

| Run | Score |
|-----|--------|
| This notebook public/offline 25 games | **3.79** mean |
| Same stack hidden LB | **~2.23** |
| Our old capped TAAF | 1.76 |
| Target | as high as this stack can go (aim 3+, stretch 4.5) |

**Changes vs the 2.23 submit:**
- Keep Duck animation TAAF + repacked Qwen3.8 model (do not cap actions)
- Apply `concurrency=28` and `max_runtime_s_per_game=7920` on **competition** too
- **2 passes** when there are ≤28 games (scorecard uses max across passes)
- Raise analyzer timeout to 1800s (this log had many vLLM read timeouts)

**Attach**
- Dataset: `jakobbrggen/taaf-kaggle-source-anim-20260807-anim`
- Dataset: `driessmit1/arc3-vllm-h100-wheelhouse-v3`
- Model: `foysalemonshanto/qwen3-8-27b-fp8-repacked-v1` → PyTorch → `hf-fp8` → Version 1
- Competition: `arc-prize-2026-arc-agi-3`

GPU: **RTX Pro 6000**. Internet **off**.
"""

HOOK = r'''# Duck v12 boost hook.
# Public 25-game mean of this notebook: 3.79. Hidden LB of 1-pass Duck v12: ~2.23.
# Do NOT cap actions (that dropped our other submit from 1.76 toward 1.10).

print("Benchmark analyzer model:", os.environ.get("INFERENCE_ANALYZER_MODEL"))
print("Qwen3.8 model path:", os.environ.get("TAAF_QWEN_MODEL_PATH"))

Q38_P1_PUBLIC_GAME_IDS = [
    "ar25-0c556536",
    "bp35-0a0ad940",
    "cd82-fb555c5d",
    "cn04-2fe56bfb",
    "dc22-fdcac232",
    "ft09-0d8bbf25",
    "g50t-5849a774",
    "ka59-38d34dbb",
    "lf52-271a04aa",
    "lp85-305b61c3",
    "ls20-9607627b",
    "m0r0-492f87ba",
    "r11l-495a7899",
    "re86-8af5384d",
    "s5i5-18d95033",
    "sb26-7fbdac44",
    "sc25-635fd71a",
    "sk48-d8078629",
    "sp80-589a99af",
    "su15-1944f8ab",
    "tn36-ef4dde99",
    "tr87-cd924810",
    "tu93-0768757b",
    "vc33-5430563c",
    "wa30-ee6fef47",
]


def _apply_boost_solver_settings(label: str) -> None:
    solver = getattr(bm, "solver", None)
    if solver is None:
        print("boost: no solver on bm")
        return
    if hasattr(solver, "concurrency"):
        solver.concurrency = 28
    if hasattr(solver, "max_runtime_s_per_game"):
        solver.max_runtime_s_per_game = 7920.0
    if hasattr(solver, "max_actions_per_game"):
        solver.max_actions_per_game = None
    if hasattr(solver, "analyzer_timeout"):
        solver.analyzer_timeout = 1800.0
    if hasattr(solver, "hard_noop_guard"):
        solver.hard_noop_guard = True
    if hasattr(solver, "animation_awareness"):
        solver.animation_awareness = True
    print(
        f"boost[{label}]: concurrency={getattr(solver, 'concurrency', None)} "
        f"max_runtime_s={getattr(solver, 'max_runtime_s_per_game', None)} "
        f"max_actions={getattr(solver, 'max_actions_per_game', None)} "
        f"analyzer_timeout={getattr(solver, 'analyzer_timeout', None)}"
    )


if not true_submission:
    if len(Q38_P1_PUBLIC_GAME_IDS) != 25 or len(set(Q38_P1_PUBLIC_GAME_IDS)) != 25:
        raise RuntimeError("Q38 P1 public game list must contain exactly 25 unique games.")
    if not bm.games:
        raise RuntimeError("benchmark_initial.pkl contains no template public game.")

    import taaf.game_api

    template_game = bm.games[0]
    arcade_spec = getattr(template_game, "arcade_spec", None)
    if arcade_spec is None:
        arcade_spec = getattr(template_game, "_arcade_spec", None)
    if arcade_spec is None:
        raise RuntimeError(
            "Could not recover the public ArcadeSpec from benchmark_initial.pkl; "
            "cannot construct the 25-game Q38 P1 evaluation set."
        )

    bm.games = [
        taaf.game_api.GameAPI(env_name=game_id, arcade_spec=arcade_spec)
        for game_id in Q38_P1_PUBLIC_GAME_IDS
    ]
    bm.n_passes = 2
    bm.game_weights = None
    bm.label = f"{getattr(bm, 'label', 'duck')}-25g-2pass-boost"
    print(
        f"Public evaluation override: {len(bm.games)} games x {bm.n_passes} passes"
    )

_apply_boost_solver_settings("hook")
bm._boost_n_passes = 2
'''


def main() -> None:
    src = json.loads(SRC.read_text(encoding="utf-8"))
    nb = copy.deepcopy(src)
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None
        c.get("metadata", {}).pop("papermill", None)

    nb["cells"][0]["source"] = [MARKDOWN]
    nb["cells"][8]["source"] = [HOOK]

    run_src = "".join(nb["cells"][9]["source"])
    old = """        bm.games = _competition_games()
        bm.n_passes = 1
        bm.game_weights = None
"""
    new = """        bm.games = _competition_games()
        n_games = len(bm.games)
        # 2 passes if the hidden set is small enough to finish under the wall clock.
        # Scorecard keeps the max score across passes.
        bm.n_passes = 2 if n_games <= 28 else 1
        bm.game_weights = None
        if hasattr(bm.solver, "concurrency"):
            bm.solver.concurrency = 28
        if hasattr(bm.solver, "max_runtime_s_per_game"):
            bm.solver.max_runtime_s_per_game = 7920.0
        if hasattr(bm.solver, "max_actions_per_game"):
            bm.solver.max_actions_per_game = None
        if hasattr(bm.solver, "analyzer_timeout"):
            bm.solver.analyzer_timeout = 1800.0
        print(
            f"boost[competition]: games={n_games} n_passes={bm.n_passes} "
            f"concurrency={getattr(bm.solver, 'concurrency', None)}"
        )
"""
    if old not in run_src:
        raise SystemExit("Could not patch competition n_passes block in cell 9")
    nb["cells"][9]["source"] = [run_src.replace(old, new, 1)]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")

    PKG.mkdir(parents=True, exist_ok=True)
    (PKG / "kaggle_duck_v12_boost.ipynb").write_text(
        json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    meta = {
        "id": "vyassenthilkumar/arc3-duck-v12-boost",
        "title": "ARC3 Duck v12 Boost 2pass",
        "code_file": "kaggle_duck_v12_boost.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "machine_shape": "NvidiaRtxPro6000",
        "dataset_sources": [
            "jakobbrggen/taaf-kaggle-source-anim-20260807-anim",
            "driessmit1/arc3-vllm-h100-wheelhouse-v3",
        ],
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "kernel_sources": [],
        "model_sources": [
            "foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1"
        ],
    }
    (PKG / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    print(f"wrote {PKG}")


if __name__ == "__main__":
    main()
