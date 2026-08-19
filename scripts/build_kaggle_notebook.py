"""Build notebooks/kaggle_submit_target_325.ipynb from the TAAF source notebook."""
from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "taaf-model-20260815-q38-p1.ipynb"
OUT = ROOT / "notebooks" / "kaggle_submit_target_325.ipynb"

MARKDOWN = """# TAAF ARC-AGI-3 — v4 efficient full solver (aim 3.0–3.5)

Same Qwen 27B TAAF stack as the 4.71 **public-debug** run. Hidden-set results so far:

| Version | Policy | Public LB |
|---------|--------|-----------|
| v1 | hard caps: 2 levels, 55 actions, `sc25`→1 | **1.76** |
| v3 | fully uncapped | **1.10** |

Uncapping **lost** score: the agent burned wall-clock on games that never completed, so later games starved. Hard public-set caps also cannot hit 3+ because hidden games are different IDs.

**v4 policy (this notebook):**
- Keep the original TAAF solver (no public-game-only prefix rules)
- Allow up to **4 levels** so later-level RHAE weight can count
- **80 actions** per game (between v1=55 and unlimited)
- **Give up after 32 actions with no new level** (stuck detection)
- Fair **per-game time slice** so every competition game gets a turn

Attach: `jakobbrggen/taaf-kaggle-source`, `driessmit1/arc3-vllm-h100-wheelhouse-v3`, `jakobbrggen/qwen3-8-27b-fp8-hf-snapshot`.
GPU: **RTX Pro 6000**. Internet **off**.
"""

HOOK = r'''# v4: original TAAF solver + stuck/time budgets (hidden LB target 3.0-3.5)
# Evidence: v1 caps -> 1.76; v3 uncapped -> 1.10 (time starvation).

import os
import time as _time

MAX_ACTIONS_PER_GAME = 80
MAX_LEVELS_PER_GAME = 4
STUCK_NO_PROGRESS_ACTIONS = 32
PER_GAME_TIME_S = 720  # 12 min; 25 games ~= 5h, leaves headroom for vLLM

os.environ.setdefault("ARC3_HARD_NOOP_GUARD", "True")
os.environ.setdefault("ARC3_ANIMATION_AWARENESS", "True")
os.environ["ARC3_MAX_ACTIONS_PER_GAME"] = str(MAX_ACTIONS_PER_GAME)
os.environ["ARC3_MAX_LEVELS_PER_GAME"] = str(MAX_LEVELS_PER_GAME)
os.environ["ARC3_STUCK_NO_PROGRESS"] = str(STUCK_NO_PROGRESS_ACTIONS)

try:
    bm.label = f"{getattr(bm, 'label', 'taaf')}-v4-efficient"
except Exception:
    pass

solver = getattr(bm, "solver", None)


def _set_if_present(obj, name, value):
    if obj is None or not hasattr(obj, name):
        return False
    try:
        setattr(obj, name, value)
        print(f"v4: set {type(obj).__name__}.{name} = {value}")
        return True
    except Exception as exc:
        print(f"v4: could not set {name}: {exc!r}")
        return False


for obj in (solver, bm):
    _set_if_present(obj, "max_actions", MAX_ACTIONS_PER_GAME)
    _set_if_present(obj, "action_limit", MAX_ACTIONS_PER_GAME)
    _set_if_present(obj, "max_actions_per_game", MAX_ACTIONS_PER_GAME)
    _set_if_present(obj, "max_levels", MAX_LEVELS_PER_GAME)
    _set_if_present(obj, "max_level", MAX_LEVELS_PER_GAME)
    _set_if_present(obj, "level_limit", MAX_LEVELS_PER_GAME)
    _set_if_present(obj, "max_game_time_s", PER_GAME_TIME_S)
    _set_if_present(obj, "game_timeout_s", PER_GAME_TIME_S)

if solver is not None:
    print("v4 solver type:", type(solver).__name__, "attrs:", sorted(a for a in dir(solver) if not a.startswith("_"))[:40])

    orig_is_done = getattr(solver, "is_done", None)
    orig_choose = getattr(solver, "choose_action", None)

    def _levels(frame):
        return int(getattr(frame, "levels_completed", 0) or 0)

    if callable(orig_is_done):
        _best = {"lv": 0, "at": 0, "t0": _time.time()}

        def _is_done(frames, latest_frame, *args, **kwargs):
            actions = int(getattr(solver, "action_counter", 0) or 0)
            lv = _levels(latest_frame)
            if lv > _best["lv"]:
                _best["lv"] = lv
                _best["at"] = actions
            if lv >= MAX_LEVELS_PER_GAME:
                return True
            if actions >= MAX_ACTIONS_PER_GAME:
                return True
            if actions - _best["at"] >= STUCK_NO_PROGRESS_ACTIONS and actions > 0:
                print(f"v4: stuck give-up actions={actions} levels={lv}")
                return True
            if _time.time() - _best["t0"] >= PER_GAME_TIME_S:
                print(f"v4: time give-up after {PER_GAME_TIME_S}s levels={lv}")
                return True
            try:
                return bool(orig_is_done(frames, latest_frame, *args, **kwargs))
            except TypeError:
                return bool(orig_is_done())

        try:
            solver.is_done = _is_done
            print("v4: wrapped solver.is_done")
        except Exception as exc:
            print(f"v4: is_done wrap skipped: {exc!r}")

    if callable(orig_choose):
        _state = {"best_lv": 0, "best_at": 0, "n": 0, "t0": _time.time()}

        def _choose(frames, latest_frame, *args, **kwargs):
            _state["n"] += 1
            lv = _levels(latest_frame)
            if lv > _state["best_lv"]:
                _state["best_lv"] = lv
                _state["best_at"] = _state["n"]
            return orig_choose(frames, latest_frame, *args, **kwargs)

        try:
            solver.choose_action = _choose
            print("v4: wrapped solver.choose_action")
        except Exception as exc:
            print(f"v4: choose wrap skipped: {exc!r}")

print(
    f"v4: max_actions={MAX_ACTIONS_PER_GAME} max_levels={MAX_LEVELS_PER_GAME} "
    f"stuck={STUCK_NO_PROGRESS_ACTIONS} per_game_s={PER_GAME_TIME_S}"
)
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

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
