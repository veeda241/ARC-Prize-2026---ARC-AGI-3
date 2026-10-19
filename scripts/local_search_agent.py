"""Play the 25 public ARC-AGI-3 games locally (no LLM, no GPU) and score the run.

Policies are pure exploration/search baselines:
  random  - uniform over the game's action space
  novelty - prefer actions that opened unseen states at the current state
  prior   - learns each action's screen displacement, then walks toward the
            nearest never-seen salient pixel (frontier targeting)

RHAE comes from scripts/eval_rhae.py against the per-level human `baseline_actions`
in each metadata.json. With --seeds > 1 the run also reports the max-across-seeds
score, which is the "sample many candidates and keep the best" test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

ROOT = Path(__file__).resolve().parents[1]
ENV_DIR = ROOT / "data" / "environment_files"
sys.path.insert(0, str(ROOT / "scripts"))
from eval_rhae import game_rhae, load_baselines  # noqa: E402

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


def salient_pixels(grid: np.ndarray, limit: int = 48) -> list[tuple[int, int]]:
    """Rare colors are usually objects, doors, keys or exits rather than terrain."""
    counts = np.bincount(grid.ravel()[grid.ravel() >= 0].astype(np.int64), minlength=16)
    background = int(counts.argmax())
    ranked = sorted(
        ((int(total), color) for color, total in enumerate(counts) if color != background and total > 0)
    )
    picks: list[tuple[int, int]] = []
    for _, color in ranked[:4]:
        ys, xs = np.where(grid == color)
        picks.extend((int(x), int(y)) for x, y in zip(xs, ys))
    if not picks:
        return [(32, 32)]
    step = max(1, len(picks) // limit)
    return picks[::step][:limit]


def changed_centre(prev: np.ndarray, cur: np.ndarray) -> tuple[float, float] | None:
    """Estimate the player as the centre of what moved between two frames."""
    if prev.shape != cur.shape:
        return None
    ys, xs = np.where(prev != cur)
    if len(xs) == 0 or len(xs) > 0.3 * cur.size:
        return None
    return float(xs.mean()), float(ys.mean())


class Explorer:
    def __init__(self, arcade: Arcade, game_id: str, rng: random.Random, policy: str) -> None:
        self.env = arcade.make(game_id)
        self.rng = rng
        self.policy = policy
        self.seen: set[str] = set()
        self.preference: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        self.displacement: dict[int, list[tuple[float, float]]] = defaultdict(list)
        self.occupancy = np.zeros((64, 64), dtype=np.int32)
        self.centre: tuple[float, float] | None = None
        allowed = {int(a.value) for a in (self.env.action_space or [])}
        self.options = [a for a in SIMPLE_ACTIONS + [CLICK_ACTION] if not allowed or a.value in allowed]

    def choose(self, grid: np.ndarray | None) -> tuple[GameAction, dict[str, Any] | None]:
        options = self.options
        if self.policy == "random":
            action = self.rng.choice(options)
        elif self.policy == "prior" and grid is not None:
            action = self.choose_prior(grid)
        else:
            tried = self.preference[grid_key(grid)]
            untried = [a for a in options if tried[a.value][0] == 0]
            action = self.rng.choice(untried) if untried else max(
                options, key=lambda a: (tried[a.value][1] / max(1, tried[a.value][0]), -tried[a.value][0])
            )
        if action is CLICK_ACTION and grid is not None:
            x, y = self.rng.choice(salient_pixels(grid, limit=12))
            return action, {"x": x, "y": y}
        return action, None

    def choose_prior(self, grid: np.ndarray) -> GameAction:
        """Walk toward the nearest pixel this agent has never stood near."""
        moves = [a for a in self.options if a is not CLICK_ACTION]
        if self.rng.random() < 0.06 and CLICK_ACTION in self.options:
            return CLICK_ACTION
        if len(moves) < 2:
            return self.rng.choice(self.options)
        unlearned = [a for a in moves if len(self.displacement[a.value]) < 3]
        if unlearned or self.centre is None:
            return self.rng.choice(unlearned or moves)

        cx, cy = self.centre
        target = self.frontier_target(grid, cx, cy)
        if target is None:
            return self.rng.choice(moves)
        dx, dy = target[0] - cx, target[1] - cy
        norm = max(1e-6, float(np.hypot(dx, dy)))
        scored = []
        for a in moves:
            mean = np.array(self.displacement[a.value], dtype=float).mean(axis=0)
            align = (mean[0] * dx + mean[1] * dy) / (norm * max(1e-6, float(np.hypot(*mean))))
            scored.append((-align, a))
        # Tie-break on the action value: GameAction is not orderable.
        return min(scored, key=lambda item: (item[0], item[1].value))[1]

    def frontier_target(self, grid: np.ndarray, cx: float, cy: float) -> tuple[int, int] | None:
        unvisited = self.occupancy == 0
        candidates = [
            (x, y)
            for x, y in salient_pixels(grid)
            if 0 <= x < 64 and 0 <= y < 64 and unvisited[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2].any()
        ]
        if candidates:
            return min(candidates, key=lambda p: abs(p[0] - cx) + abs(p[1] - cy))
        free = np.argwhere(unvisited)
        if len(free) == 0:
            return None
        dists = np.abs(free[:, 1] - cx) + np.abs(free[:, 0] - cy)
        row = free[int(dists.argmin())]
        return int(row[1]), int(row[0])

    def observe(self, action: GameAction, prev: np.ndarray | None, cur: np.ndarray | None) -> None:
        if prev is None or cur is None or action is CLICK_ACTION:
            return
        centre = changed_centre(prev, cur)
        if centre is None:
            return
        if self.centre is not None:
            self.displacement[int(action.value)].append((centre[0] - self.centre[0], centre[1] - self.centre[1]))
        self.centre = centre
        x, y = int(centre[0]), int(centre[1])
        self.occupancy[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2] += 1

    def run(self, budget: int, stuck_limit: int) -> dict[str, Any]:
        fd = self.env.reset()
        grid = frame_grid(fd)
        prev_grid = grid
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
                self.centre = None
                continue
            action, data = self.choose(grid)
            prev_key = grid_key(grid)
            prev_levels = int(fd.levels_completed) if fd else 0
            fd = self.env.step(action, data)
            steps += 1
            level_steps += 1
            new_grid = frame_grid(fd)
            new_key = grid_key(new_grid)
            if self.policy == "prior":
                self.observe(action, prev_grid, new_grid)
            if new_grid is not None:
                prev_grid = new_grid
                grid = new_grid

            if new_key != prev_key:
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
                self.centre = None
                self.occupancy[:] = 0
            if stale >= stuck_limit:
                self.env.step(GameAction.RESET)
                level_steps = 0
                stale = 0
                self.centre = None
                fd = self.env.reset()
                refreshed = frame_grid(fd)
                if refreshed is not None:
                    grid = refreshed
                    prev_grid = refreshed
        levels = int(fd.levels_completed) if fd else 0
        return {"levels_completed": levels, "level_actions": level_actions, "steps": steps, "distinct_states": len(self.seen)}


def mean_rhae(games: dict[str, dict[str, Any]], baselines: dict[str, dict[str, Any]]) -> float:
    scores = [
        game_rhae(baselines[gid]["baseline_actions"], run["level_actions"], run["levels_completed"])["score"]
        for gid, run in games.items()
        if gid in baselines
    ]
    return sum(scores) / len(scores) if scores else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["random", "novelty", "prior"], default="prior")
    ap.add_argument("--budget", type=int, default=2000, help="max env steps per game, per seed")
    ap.add_argument("--stuck-limit", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=1, help="independent runs; also reports best-of-seeds")
    ap.add_argument("--games", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    arcade = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(ENV_DIR),
        recordings_dir=str(ROOT / "artifacts" / "recordings"),
    )
    game_ids = args.games or [e.game_id for e in arcade.get_environments()]
    baselines = load_baselines(ENV_DIR)

    per_seed: list[float] = []
    combined: dict[str, dict[str, Any]] = {}
    for seed in range(args.seeds):
        games: dict[str, dict[str, Any]] = {}
        for gid in game_ids:
            rng = random.Random(1000 + seed)
            try:
                games[gid] = Explorer(arcade, gid, rng, args.policy).run(args.budget, args.stuck_limit)
            except Exception as exc:  # one broken game must not kill the sweep
                print(f"{gid}: FAILED {exc!r}")
                games[gid] = {"levels_completed": 0, "level_actions": [], "steps": 0, "distinct_states": 0}
        mean = mean_rhae(games, baselines)
        per_seed.append(mean)
        solved = sum(1 for g in games.values() if g["levels_completed"] >= 1)
        print(f"  seed {seed}: mean RHAE {mean:.3f} | games reaching level 1: {solved}/{len(games)}")
        for gid, run in games.items():
            best = combined.get(gid)
            if best is None or (run["levels_completed"], -min(run["level_actions"] or [10**9])) > (
                best["levels_completed"],
                -min(best["level_actions"] or [10**9]),
            ):
                combined[gid] = run

    ensemble = mean_rhae(combined, baselines)
    print(f"\npolicy={args.policy} budget={args.budget} seeds={args.seeds} games={len(game_ids)}")
    print(f"single-run mean   = {sum(per_seed) / len(per_seed):.3f}  (range {min(per_seed):.3f}-{max(per_seed):.3f})")
    print(f"best-of-{args.seeds} mean  = {ensemble:.3f}")

    out = args.out or ROOT / "artifacts" / f"local_{args.policy}_{args.budget}_s{args.seeds}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"games": {gid: {"levels_completed": r["levels_completed"], "level_actions": r["level_actions"]} for gid, r in combined.items()}}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
