"""Play the 25 public ARC-AGI-3 games locally (no LLM, no GPU) and emit RHAE results.

Policies are pure search/exploration baselines. Output matches scripts/eval_rhae.py:
    {"games": {"<game_id>": {"levels_completed": n, "level_actions": [...]}}}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

ROOT = Path(__file__).resolve().parents[1]
ENV_DIR = ROOT / "data" / "environment_files"
SIMPLE_ACTIONS = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4, GameAction.ACTION5, GameAction.ACTION7]
CLICK_ACTION = GameAction.ACTION6


def frame_grid(fd: Any) -> np.ndarray | None:
    if fd is None:
        return None
    frames = getattr(fd, "frame", None)
    if frames is None or len(frames) == 0:
        return None
    return np.asarray(frames[-1], dtype=np.int8)


def grid_key(grid: np.ndarray | None) -> str:
    if grid is None:
        return "-"
    return hashlib.sha1(np.ascontiguousarray(grid).tobytes()).hexdigest()[:16]


def click_targets(grid: np.ndarray, limit: int = 12) -> list[tuple[int, int]]:
    """Salient-pixel prior: rare colors are usually objects, doors, keys, or exits."""
    flat = grid.ravel()
    counts = np.bincount(flat[flat >= 0].astype(np.int64), minlength=16)
    background = int(counts.argmax())
    scored = []
    for color, total in enumerate(counts):
        if color == background or total == 0:
            continue
        ys, xs = np.where(grid == color)
        rarity = total / max(1, int(grid.size))
        scored.append((rarity, [(int(x), int(y)) for x, y in zip(xs, ys)]))
    scored.sort(key=lambda item: item[0])
    picks: list[tuple[int, int]] = []
    for _, pixels in scored[:4]:
        picks.extend(pixels[: max(1, limit // 4)])
    return picks[:limit] or [(32, 32)]


class Explorer:
    def __init__(self, arcade: Arcade, game_id: str, rng: random.Random, policy: str) -> None:
        self.env = arcade.make(game_id)
        self.rng = rng
        self.policy = policy
        self.seen: set[str] = set()
        self.preference: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        allowed = {int(a.value) for a in (self.env.action_space or [])}
        self.options = [a for a in SIMPLE_ACTIONS + [CLICK_ACTION] if not allowed or a.value in allowed]

    def choose(self, grid: np.ndarray | None) -> tuple[GameAction, dict[str, Any] | None]:
        key = grid_key(grid)
        options = self.options
        if self.policy == "random":
            return self.rng.choice(options), None

        tried = self.preference[key]
        untried = [a for a in options if tried[a.value][0] == 0]
        if untried:
            action = self.rng.choice(untried)
        else:
            # Exploit the action that most often opened new territory at this state.
            action = max(
                options,
                key=lambda a: (tried[a.value][1] / max(1, tried[a.value][0]), -tried[a.value][0]),
            )
        if action is CLICK_ACTION and grid is not None:
            x, y = self.rng.choice(click_targets(grid))
            return action, {"x": x, "y": y}
        return action, None

    def run(self, budget: int, stuck_limit: int) -> dict[str, Any]:
        fd = self.env.reset()
        grid = frame_grid(fd)
        self.seen.add(grid_key(grid))
        level_steps = 0
        level_actions: list[int] = []
        stale = 0
        steps = 0
        while steps < budget:
            if fd is not None and fd.state is GameState.GAME_OVER:
                fd = self.env.step(GameAction.RESET)
                steps += 1
                level_steps = 0
                continue
            action, data = self.choose(grid)
            prev_key = grid_key(grid)
            prev_levels = int(fd.levels_completed) if fd else 0
            fd = self.env.step(action, data)
            steps += 1
            level_steps += 1
            new_grid = frame_grid(fd)
            new_key = grid_key(new_grid)
            changed = new_key != prev_key
            if changed:
                stale = 0
                if new_key not in self.seen:
                    self.preference[prev_key][int(action.value)][1] += 1
                    self.seen.add(new_key)
            else:
                stale += 1
            self.preference[prev_key][int(action.value)][0] += 1

            if fd is not None and int(fd.levels_completed) > prev_levels:
                level_actions.append(level_steps)
                level_steps = 0
            grid = new_grid if new_grid is not None else grid
            if stale >= stuck_limit:
                self.env.step(GameAction.RESET)
                level_steps = 0
                stale = 0
                fd = self.env.reset()
                refreshed = frame_grid(fd)
                if refreshed is not None:
                    grid = refreshed
        levels = int(fd.levels_completed) if fd else 0
        return {"levels_completed": levels, "level_actions": level_actions, "steps": steps, "distinct_states": len(self.seen)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["random", "novelty"], default="novelty")
    ap.add_argument("--budget", type=int, default=400, help="max env steps per game")
    ap.add_argument("--stuck-limit", type=int, default=60)
    ap.add_argument("--games", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    arcade = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(ENV_DIR),
        recordings_dir=str(ROOT / "artifacts" / "recordings"),
    )
    game_ids = args.games or [e.game_id for e in arcade.get_environments()]
    results: dict[str, Any] = {}
    for gid in game_ids:
        rng = random.Random(1234)
        try:
            play = Explorer(arcade, gid, rng, args.policy).run(args.budget, args.stuck_limit)
        except Exception as exc:  # a broken game must not kill the sweep
            print(f"{gid}: FAILED {exc!r}")
            play = {"levels_completed": 0, "level_actions": [], "steps": 0, "distinct_states": 0}
        results[gid] = play
        print(f"{gid}: levels={play['levels_completed']} steps={play['steps']} states={play['distinct_states']}")

    payload = {
        "games": {gid: {"levels_completed": r["levels_completed"], "level_actions": r["level_actions"]} for gid, r in results.items()}
    }
    out = args.out or ROOT / "artifacts" / f"local_{args.policy}_{args.budget}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    total_levels = sum(r["levels_completed"] for r in results.values())
    print(f"\npolicy={args.policy} budget={args.budget} games={len(results)} levels_completed={total_levels}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
