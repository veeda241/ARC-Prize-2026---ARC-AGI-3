#!/usr/bin/env python3
"""Run ARC-AGI-3 locally on Windows (offline envs from data/environment_files).

This does NOT run the TAAF 27B Kaggle notebook — that needs Kaggle GPU.
Use this for smoke tests of calibrated_agent + RHAE scoring on your PC.

Examples:
  python scripts/run_local.py --game ls20 --agent random
  python scripts/run_local.py --game sc25 --agent calibrated
  python scripts/run_local.py --all --max-games 3
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENTS_DIR = ROOT / "data" / "ARC-AGI-3-Agents"
ENV_DIR = ROOT / "data" / "environment_files"
CALIB_AGENT = ROOT / "agents" / "calibrated_agent.py"


def setup_env() -> None:
    os.environ.setdefault("OPERATION_MODE", "offline")
    os.environ.setdefault("ENVIRONMENTS_DIR", str(ENV_DIR))
    os.environ.setdefault("RECORDINGS_DIR", str(ROOT / "artifacts" / "recordings"))
    os.environ.setdefault("ARC_API_KEY", "local-dev")
    os.environ.setdefault("ARC_BASE_URL", "http://localhost:8001/")
    os.environ.setdefault("ARC3_MAX_LEVELS_PER_GAME", "2")
    os.environ.setdefault("ARC3_MAX_ACTIONS_PER_GAME", "55")
    os.environ.setdefault("ARC3_STRICT_PREFIXES", "sc25")
    os.environ.setdefault("ARC3_POLICY", "heuristic")

    Path(os.environ["RECORDINGS_DIR"]).mkdir(parents=True, exist_ok=True)


def discover_games(prefix: str | None = None) -> list[str]:
    games: list[str] = []
    for meta_path in sorted(ENV_DIR.rglob("metadata.json")):
        gid = json.loads(meta_path.read_text(encoding="utf-8"))["game_id"]
        short = gid.split("-")[0]
        if prefix is None or short.startswith(prefix) or gid.startswith(prefix):
            games.append(gid)
    return games


def register_calibrated_agent() -> None:
    dst = AGENTS_DIR / "agents" / "templates" / "calibrated_agent.py"
    if CALIB_AGENT.is_file():
        dst.write_text(CALIB_AGENT.read_text(encoding="utf-8"), encoding="utf-8")


def _load_agent_modules():
    """Import agent modules without pulling optional LangGraph templates."""
    import importlib.util
    import types

    agents_root = AGENTS_DIR / "agents"
    pkg_name = "arc3_agents"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(agents_root)]  # type: ignore[attr-defined]
        sys.modules[pkg_name] = pkg
        tpl = types.ModuleType(f"{pkg_name}.templates")
        tpl.__path__ = [str(agents_root / "templates")]  # type: ignore[attr-defined]
        sys.modules[f"{pkg_name}.templates"] = tpl

    def load(rel: str):
        full = f"{pkg_name}.{rel}"
        if full in sys.modules:
            return sys.modules[full]
        path = agents_root.joinpath(*rel.split(".")).with_suffix(".py")
        spec = importlib.util.spec_from_file_location(full, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        module.__package__ = full.rpartition(".")[0]
        sys.modules[full] = module
        spec.loader.exec_module(module)
        return module

    load("recorder")
    load("tracing")
    agent_mod = load("agent")
    random_mod = load("templates.random_agent")
    return agent_mod.Agent, random_mod.Random


def run_games(agent: str, game_ids: list[str]) -> dict:
    from arc_agi import Arcade

    Agent, Random = _load_agent_modules()

    agent_cls: type[Agent]
    if agent == "calibrated":
        register_calibrated_agent()
        import importlib.util

        path = AGENTS_DIR / "agents" / "templates" / "calibrated_agent.py"
        mod_name = "arc3_calibrated_agent"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        agent_cls = mod.build_calibrated_agent_class(Agent)
    elif agent == "random":
        agent_cls = Random
    else:
        raise SystemExit(f"Unknown agent {agent!r}")

    arcade = Arcade()
    card_id = arcade.open_scorecard(tags=["local-run"])
    results: list[dict] = []

    for gid in game_ids:
        env = arcade.make(gid, scorecard_id=card_id)
        runner = agent_cls(
            card_id=card_id,
            game_id=gid,
            agent_name=agent,
            ROOT_URL="http://localhost:8001/",
            record=True,
            arc_env=env,
            tags=["local-run"],
        )
        runner.main()
        results.append(
            {
                "game_id": gid,
                "levels_completed": runner.levels_completed,
                "actions": runner.action_counter,
            }
        )
        runner.cleanup()

    scorecard = arcade.close_scorecard(card_id)
    payload = scorecard.model_dump() if scorecard else {"games": results}
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ARC-AGI-3 agents locally (offline)")
    parser.add_argument("--game", default="ls20", help="Game prefix or full id (default: ls20)")
    parser.add_argument("--agent", default="calibrated", choices=["calibrated", "random"])
    parser.add_argument("--all", action="store_true", help="Run all 25 public games")
    parser.add_argument("--max-games", type=int, default=1, help="Cap number of games when --all")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if not ENV_DIR.exists():
        raise SystemExit(f"Missing {ENV_DIR}. Extract arc-prize-2026-arc-agi-3.zip first.")

    setup_env()
    sys.path.insert(0, str(AGENTS_DIR))
    sys.path.insert(0, str(ROOT / "agents"))

    if args.all:
        games = discover_games()[: args.max_games]
    else:
        games = discover_games(args.game)
        if not games:
            raise SystemExit(f"No game matching {args.game!r}")

    print(f"Local run: agent={args.agent} games={games}")
    result = run_games(args.agent, games)

    out = ROOT / "artifacts" / "local_run_scorecard.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote scorecard -> {out}")


if __name__ == "__main__":
    main()
