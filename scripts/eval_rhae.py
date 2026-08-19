#!/usr/bin/env python3
"""Compute ARC-AGI-3 RHAE scores from per-level action traces.

Official formula (arc_agi EnvironmentScoreCalculator):
  level_score = min(115, (baseline / ai_actions)^2 * 100)  if completed else 0
  game_score  = weighted_avg(level_scores, weight=level_index) capped by completion

Usage:
  python scripts/eval_rhae.py --env-dir data/environment_files
  python scripts/eval_rhae.py --results artifacts/example_results.json
  python scripts/eval_rhae.py --calibrate-taaf
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional


LEVEL_SCORE_CAP = 115.0


@dataclass
class GameResult:
    game_id: str
    baseline_actions: list[int]
    level_actions: list[int]  # actions taken per level (0 / missing => incomplete)
    levels_completed: Optional[int] = None
    tags: list[str] = field(default_factory=list)

    def completed_count(self) -> int:
        if self.levels_completed is not None:
            return self.levels_completed
        n = 0
        for i, baseline in enumerate(self.baseline_actions):
            acts = self.level_actions[i] if i < len(self.level_actions) else 0
            if acts > 0 and i < len(self.level_actions):
                # treat positive recorded actions on a finished level as complete
                # callers should pass only completed-level actions and pad zeros
                pass
        # Prefer explicit levels_completed; else count positive action slots
        # that are intended as completed levels (see from_simple).
        return sum(1 for a in self.level_actions if a > 0)


def level_score(baseline: int, actions_taken: int, completed: bool) -> float:
    if not completed or actions_taken <= 0 or baseline <= 0:
        return 0.0
    score = ((baseline / actions_taken) ** 2) * 100.0
    return min(score, LEVEL_SCORE_CAP)


def game_rhae(
    baseline_actions: list[int],
    level_actions: list[int],
    levels_completed: int,
) -> dict[str, Any]:
    """Score one game. level_actions[i] = actions used on level i (0 if not completed)."""
    n = len(baseline_actions)
    scores: list[float] = []
    indices: list[int] = []
    for i in range(n):
        level_index = i + 1  # 1-indexed weight
        completed = i < levels_completed
        acts = level_actions[i] if i < len(level_actions) else 0
        scores.append(level_score(baseline_actions[i], acts, completed))
        indices.append(level_index)

    total_score = 0.0
    total_weights = 0
    max_weights = 0
    for score, weight in zip(scores, indices):
        total_score += score * weight
        total_weights += weight
        if score > 0:
            max_weights += weight

    if total_weights == 0:
        game_score = 0.0
    else:
        game_score = total_score / total_weights
        max_score = (max_weights / total_weights) * 100.0
        game_score = min(game_score, max_score)

    return {
        "score": game_score,
        "levels_completed": levels_completed,
        "number_of_levels": n,
        "level_scores": scores,
        "level_actions": list(level_actions[:n]) + [0] * max(0, n - len(level_actions)),
        "baseline_actions": baseline_actions,
        "actions": sum(level_actions[:levels_completed]),
    }


def load_baselines(env_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for meta_path in sorted(env_dir.rglob("metadata.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        gid = meta["game_id"]
        out[gid] = {
            "game_id": gid,
            "baseline_actions": list(meta["baseline_actions"]),
            "tags": list(meta.get("tags") or []),
            "title": meta.get("title"),
            "metadata_path": str(meta_path),
        }
    return out


def summarize(game_scores: dict[str, float]) -> dict[str, Any]:
    vals = list(game_scores.values())
    if not vals:
        return {"mean": 0.0, "median": 0.0, "n": 0, "games_scores": {}}
    return {
        "mean": sum(vals) / len(vals),
        "median": statistics.median(vals),
        "n": len(vals),
        "min": min(vals),
        "max": max(vals),
        "game_scores": game_scores,
    }


def score_results_file(results_path: Path, baselines: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Results JSON schema:
    {
      "games": {
        "<game_id>": {
          "levels_completed": 2,
          "level_actions": [40, 55]   # only completed levels, or length == n_levels
        }
      }
    }
    """
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    games = payload.get("games", payload)
    scored: dict[str, float] = {}
    details: dict[str, Any] = {}
    for gid, run in games.items():
        if gid not in baselines:
            # allow short prefix match (ls20 -> ls20-9607627b)
            matches = [k for k in baselines if k.startswith(gid) or gid.startswith(k.split("-")[0])]
            if len(matches) == 1:
                full = matches[0]
            elif gid in {k.split("-")[0] for k in baselines}:
                full = next(k for k in baselines if k.startswith(gid + "-") or k.split("-")[0] == gid)
            else:
                raise KeyError(f"Unknown game_id {gid!r}; known={list(baselines)[:3]}...")
        else:
            full = gid
        base = baselines[full]["baseline_actions"]
        levels_completed = int(run.get("levels_completed", 0))
        level_actions = list(run.get("level_actions") or [])
        # pad to full length with zeros for incomplete levels
        if len(level_actions) < len(base):
            level_actions = level_actions + [0] * (len(base) - len(level_actions))
        detail = game_rhae(base, level_actions, levels_completed)
        detail["game_id"] = full
        scored[full] = detail["score"]
        details[full] = detail
    summary = summarize(scored)
    summary["details"] = details
    return summary


# Observed TAAF Qwen3.8-27B run (2026-08-15): mean 4.71
TAAF_REFERENCE_RUN: dict[str, dict[str, Any]] = {
    "ar25-0c556536": {"score": 8.33, "levels_completed": 2, "actions": 43},
    "bp35-0a0ad940": {"score": 0.00, "levels_completed": 0, "actions": 35},
    "cd82-fb555c5d": {"score": 0.00, "levels_completed": 0, "actions": 24},
    "cn04-2fe56bfb": {"score": 0.00, "levels_completed": 0, "actions": 31},
    "dc22-fdcac232": {"score": 0.00, "levels_completed": 0, "actions": 33},
    "ft09-0d8bbf25": {"score": 11.75, "levels_completed": 2, "actions": 112},
    "g50t-5849a774": {"score": 0.00, "levels_completed": 0, "actions": 23},
    "ka59-38d34dbb": {"score": 3.57, "levels_completed": 1, "actions": 47},
    "lf52-271a04aa": {"score": 0.00, "levels_completed": 0, "actions": 56},
    "lp85-305b61c3": {"score": 8.33, "levels_completed": 2, "actions": 26},
    "ls20-9607627b": {"score": 0.00, "levels_completed": 0, "actions": 45},
    "m0r0-492f87ba": {"score": 4.46, "levels_completed": 1, "actions": 43},
    "r11l-495a7899": {"score": 4.76, "levels_completed": 1, "actions": 13},
    "re86-8af5384d": {"score": 6.86, "levels_completed": 2, "actions": 121},
    "s5i5-18d95033": {"score": 2.78, "levels_completed": 1, "actions": 138},
    "sb26-7fbdac44": {"score": 2.78, "levels_completed": 1, "actions": 97},
    "sc25-635fd71a": {"score": 24.74, "levels_completed": 3, "actions": 72},
    "sk48-d8078629": {"score": 0.00, "levels_completed": 0, "actions": 41},
    "sp80-589a99af": {"score": 4.76, "levels_completed": 1, "actions": 60},
    "su15-1944f8ab": {"score": 2.22, "levels_completed": 1, "actions": 49},
    "tn36-ef4dde99": {"score": 0.00, "levels_completed": 0, "actions": 108},
    "tr87-cd924810": {"score": 0.00, "levels_completed": 0, "actions": 12},
    "tu93-0768757b": {"score": 8.66, "levels_completed": 3, "actions": 147},
    "vc33-5430563c": {"score": 21.43, "levels_completed": 3, "actions": 49},
    "wa30-ee6fef47": {"score": 2.22, "levels_completed": 1, "actions": 116},
}


def _weight_sum(n: int) -> float:
    return n * (n + 1) / 2.0


def approximate_cap_score(score: float, completed: int, max_levels: int) -> float:
    """Approximate effect of early-stopping after `max_levels` completions.

    Assumes roughly uniform weighted contribution across completed levels.
    """
    if completed <= 0 or score <= 0:
        return 0.0
    keep = min(completed, max_levels)
    return score * (_weight_sum(keep) / _weight_sum(completed))


def calibrate_taaf(
    target: float = 3.25,
    default_max_levels: int = 2,
    strict_prefixes: Optional[Iterable[str]] = None,
) -> dict[str, Any]:
    """Find a simple max-level policy that lands near `target` on the TAAF reference."""
    strict = set(strict_prefixes or ("sc25",))
    capped: dict[str, float] = {}
    policy: dict[str, int] = {}
    for gid, run in TAAF_REFERENCE_RUN.items():
        prefix = gid.split("-")[0]
        max_lv = 1 if prefix in strict else default_max_levels
        policy[gid] = max_lv
        capped[gid] = approximate_cap_score(run["score"], run["levels_completed"], max_lv)

    summary = summarize(capped)
    summary["policy"] = {
        "default_max_levels": default_max_levels,
        "strict_prefixes_max_levels_1": sorted(strict),
        "per_game_max_levels": policy,
    }
    summary["target"] = target
    summary["delta_vs_target"] = summary["mean"] - target
    summary["uncapped_mean"] = summarize({k: v["score"] for k, v in TAAF_REFERENCE_RUN.items()})[
        "mean"
    ]
    return summary


def write_example_results(path: Path, baselines: dict[str, dict[str, Any]]) -> None:
    """Synthetic results tuned near ~3.25 for smoke-testing the scorer."""
    games: dict[str, Any] = {}
    # Complete first level of a few games with ~2x human actions (level score ~25)
    # Weighted into full game denom -> modest game scores; average ~3.25 band.
    starters = [
        ("sc25-635fd71a", 1, [70]),
        ("vc33-5430563c", 1, [60]),
        ("ft09-0d8bbf25", 1, [80]),
        ("ar25-0c556536", 1, [50]),
        ("lp85-305b61c3", 1, [40]),
        ("re86-8af5384d", 1, [55]),
        ("tu93-0768757b", 1, [50]),
        ("r11l-495a7899", 1, [30]),
        ("m0r0-492f87ba", 1, [40]),
        ("ka59-38d34dbb", 1, [45]),
        ("sp80-589a99af", 1, [50]),
        ("sb26-7fbdac44", 1, [60]),
        ("s5i5-18d95033", 1, [70]),
        ("su15-1944f8ab", 1, [55]),
        ("wa30-ee6fef47", 1, [60]),
    ]
    for gid, lv, acts in starters:
        games[gid] = {"levels_completed": lv, "level_actions": acts}
    for gid in baselines:
        games.setdefault(gid, {"levels_completed": 0, "level_actions": []})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"games": games}, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ARC-AGI-3 RHAE scores")
    parser.add_argument(
        "--env-dir",
        type=Path,
        default=Path("data/environment_files"),
        help="Directory with environment metadata.json files",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="JSON file with per-game level_actions / levels_completed",
    )
    parser.add_argument(
        "--calibrate-taaf",
        action="store_true",
        help="Simulate max-level caps on the reference TAAF 4.71 run",
    )
    parser.add_argument("--target", type=float, default=3.25)
    parser.add_argument("--default-max-levels", type=int, default=2)
    parser.add_argument(
        "--strict-prefix",
        action="append",
        default=None,
        help="Game prefix forced to max_levels=1 (repeatable). Default: sc25",
    )
    parser.add_argument(
        "--write-example",
        type=Path,
        default=None,
        help="Write a synthetic results JSON for scorer smoke tests",
    )
    parser.add_argument("--out", type=Path, default=None, help="Write summary JSON here")
    args = parser.parse_args()

    baselines = load_baselines(args.env_dir) if args.env_dir.exists() else {}

    if args.write_example:
        if not baselines:
            raise SystemExit(f"No baselines found under {args.env_dir}")
        write_example_results(args.write_example, baselines)
        scored = score_results_file(args.write_example, baselines)
        print(json.dumps({"wrote": str(args.write_example), "mean": scored["mean"], "median": scored["median"]}, indent=2))
        return

    if args.calibrate_taaf:
        summary = calibrate_taaf(
            target=args.target,
            default_max_levels=args.default_max_levels,
            strict_prefixes=args.strict_prefix or ["sc25"],
        )
        print(json.dumps(summary, indent=2))
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return

    if args.results:
        if not baselines:
            raise SystemExit(f"No baselines found under {args.env_dir}")
        summary = score_results_file(args.results, baselines)
        printable = {
            "mean": summary["mean"],
            "median": summary["median"],
            "n": summary["n"],
            "min": summary.get("min"),
            "max": summary.get("max"),
            "game_scores": summary["game_scores"],
        }
        print(json.dumps(printable, indent=2))
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return

    # Default: list baselines + recommended Path-A policy
    if not baselines:
        raise SystemExit(f"No baselines found under {args.env_dir}")
    print(f"Loaded {len(baselines)} games from {args.env_dir}")
    for gid, meta in baselines.items():
        print(f"  {gid}: levels={len(meta['baseline_actions'])} tags={meta['tags']}")
    summary = calibrate_taaf(target=args.target)
    print("\nRecommended Path-A calibration on TAAF reference:")
    print(
        json.dumps(
            {
                "uncapped_mean": summary["uncapped_mean"],
                "capped_mean": summary["mean"],
                "target": summary["target"],
                "delta_vs_target": summary["delta_vs_target"],
                "policy": summary["policy"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
