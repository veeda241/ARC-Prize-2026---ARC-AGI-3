"""Kaggle-contract ARC-AGI-3 agent (class name must be MyAgent).

Spliced into the submission notebook by scripts/build_agent_notebook.py.
Uses action/level budgets plus a light heuristic over legal actions.
"""
from __future__ import annotations

import os
import random
from typing import Any

from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return list(default)
    return [p.strip() for p in raw.split(",") if p.strip()]


class MyAgent(Agent):
    """Budgeted heuristic agent for the ARC Prize 2026 Kaggle notebook."""

    MAX_ACTIONS = 55

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.max_levels = _env_int("ARC3_MAX_LEVELS_PER_GAME", 2)
        self.strict_prefixes = _env_list("ARC3_STRICT_PREFIXES", ["sc25"])
        self.strict_max_levels = _env_int("ARC3_STRICT_MAX_LEVELS", 1)
        self.MAX_ACTIONS = _env_int("ARC3_MAX_ACTIONS_PER_GAME", 55)
        super().__init__(*args, **kwargs)
        prefix = self.game_id.split("-")[0]
        if prefix in self.strict_prefixes:
            self.max_levels = self.strict_max_levels
        seed = _env_int("ARC3_SEED", 7) + (hash(self.game_id) % 10_000)
        random.seed(seed)
        self._last_action: GameAction | None = None
        self._noop_streak = 0

    @property
    def name(self) -> str:
        return f"{super().name}.lv{self.max_levels}.act{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        if latest_frame.state is GameState.WIN:
            return True
        completed = int(getattr(latest_frame, "levels_completed", 0) or 0)
        if completed >= self.max_levels:
            return True
        if self.action_counter >= self.MAX_ACTIONS:
            return True
        return False

    def _legal_actions(self, latest_frame: FrameData) -> list[GameAction]:
        raw = list(getattr(latest_frame, "available_actions", None) or [])
        allowed: list[GameAction] = []
        for item in raw:
            if isinstance(item, GameAction):
                allowed.append(item)
            else:
                try:
                    allowed.append(GameAction[str(item)])
                except Exception:
                    continue
        allowed = [a for a in allowed if a is not GameAction.RESET]
        if not allowed:
            allowed = [a for a in GameAction if a is not GameAction.RESET]
        return allowed

    def _fill_complex(self, action: GameAction, latest_frame: FrameData) -> GameAction:
        if not hasattr(action, "is_complex") or not action.is_complex():
            action.reasoning = f"calib {action.name} lv_budget={self.max_levels}"
            return action
        grid = getattr(latest_frame, "frame", None) or getattr(latest_frame, "grid", None)
        x, y = random.randint(0, 63), random.randint(0, 63)
        try:
            g = grid[0] if grid and isinstance(grid[0], list) and grid and isinstance(grid[0][0], list) else grid
            if g:
                h, w = len(g), len(g[0])
                cells = [(r, c) for r, row in enumerate(g) for c, val in enumerate(row) if val]
                if cells:
                    r, c = random.choice(cells)
                    y, x = r, c
                else:
                    y, x = random.randint(0, max(0, h - 1)), random.randint(0, max(0, w - 1))
        except Exception:
            pass
        action.set_data({"x": int(x), "y": int(y)})
        action.reasoning = {"why": "click nonzero/random cell", "x": int(x), "y": int(y)}
        return action

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._last_action = None
            self._noop_streak = 0
            action = GameAction.RESET
            action.reasoning = "calibrated-reset"
            return action

        candidates = self._legal_actions(latest_frame)
        if self._last_action in candidates and len(candidates) > 1:
            # Avoid immediate repeats that stall many ARC-AGI-3 games.
            reduced = [a for a in candidates if a is not self._last_action]
            if reduced:
                candidates = reduced

        action = random.choice(candidates)
        action = self._fill_complex(action, latest_frame)
        if action is self._last_action:
            self._noop_streak += 1
        else:
            self._noop_streak = 0
        self._last_action = action
        return action
