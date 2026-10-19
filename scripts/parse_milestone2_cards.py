#!/usr/bin/env python3
"""Read a Milestone 2 run's `.per_game_cards.pkl` and report RHAE per game.

The harness writes one card per game as
`<n>|<game_id>|<state>|<outcome>:<actions>:<levels>:<score>|...` - one trailing segment per
pass. `score` is the harness's own per-game RHAE on the 0-100 scale, so a mean over the 25
public games is directly comparable between runs (and is NOT a leaderboard number, which is
played on hidden games with `n_passes` forced to 1).
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any


def parse_card(summary: str) -> dict[str, Any]:
    parts = summary.split("|")
    passes = []
    for seg in parts[3:]:
        fields = seg.split(":")
        if len(fields) != 4:
            continue
        outcome, actions, levels, score = fields
        passes.append(
            {
                "outcome": outcome,
                "actions": int(actions),
                "levels": int(levels),
                "score": float(score),
            }
        )
    return {
        "game_id": parts[1],
        "state": parts[2],
        "passes": passes,
        "best": max((p["score"] for p in passes), default=0.0),
        "levels_completed": max((p["levels"] for p in passes), default=0),
    }


def load_cards(path: Path) -> dict[str, tuple[str, str]]:
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    if not isinstance(raw, dict):
        raise SystemExit(f"{path} is a {type(raw).__name__}, expected dict[game_id, (summary, html)]")
    return raw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cards", type=Path, nargs="+", help=".per_game_cards.pkl file(s)")
    args = ap.parse_args()

    runs = []
    for path in args.cards:
        games = [parse_card(summary[0]) for summary in load_cards(path).values()]
        games.sort(key=lambda g: -g["best"])
        mean = sum(g["best"] for g in games) / max(1, len(games))
        level_games = sum(1 for g in games if g["levels_completed"] > 0)
        won = sum(1 for g in games if any(p["outcome"] == "won" for p in g["passes"]))
        runs.append((path, games, mean, level_games, won))

    for path, games, mean, level_games, won in runs:
        print(f"\n=== {path.name}: {len(games)} games, mean best-pass RHAE {mean:.2f}")
        print(f"    games with >=1 level solved: {level_games}, games won in some pass: {won}")
        for g in games:
            detail = " ".join(f"{p['outcome'][:3]}:{p['actions']}a/{p['levels']}l/{p['score']:.1f}" for p in g["passes"])
            print(f"    {g['game_id']:<18} best={g['best']:6.2f}  {detail}")

    if len(runs) > 1:
        print("\n=== per-game deltas (later runs minus first) ===")
        base = {g["game_id"]: g["best"] for g in runs[0][1]}
        for path, games, _, _, _ in runs[1:]:
            diffs = [g["best"] - base[g["game_id"]] for g in games if g["game_id"] in base]
            print(f"    {path.name}: mean {sum(diffs) / max(1, len(diffs)):+.2f} over {len(diffs)} shared games")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
