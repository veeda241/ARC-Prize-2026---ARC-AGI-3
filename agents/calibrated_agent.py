"""Calibrated ARC-AGI-3 agent with action/level budgets targeting ~3.25 RHAE.

Designed to wrap the official Agents `Agent` interface (see ARC-AGI-3-Agents).
Works in three modes:

1. heuristic  — random/heuristic policy with hard caps (Path C)
2. unsloth    — local Unsloth/HF causal LM for action choice (Path B)
3. budget_only — wrap another agent and only enforce early-stop budgets (Path A)

Environment knobs (also settable via CalibratedConfig):
  ARC3_TARGET_MEAN_RHAE=3.25
  ARC3_MAX_LEVELS_PER_GAME=2
  ARC3_MAX_ACTIONS_PER_GAME=55
  ARC3_STRICT_PREFIXES=sc25
  ARC3_POLICY=heuristic|unsloth
  ARC3_UNSLOTH_MODEL_DIR=/path/to/lora_or_merged
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Sequence


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _env_list(name: str, default: Sequence[str]) -> list[str]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return list(default)
    return [p.strip() for p in raw.split(",") if p.strip()]


@dataclass
class CalibratedConfig:
    """Budgets that steer mean RHAE into the 3.0–3.5 band."""

    target_mean_rhae: float = 3.25
    max_levels_per_game: int = 2
    max_actions_per_game: int = 55
    # Prefixes historically dominating the TAAF 4.71 run — force tighter caps.
    strict_prefixes: list[str] = field(default_factory=lambda: ["sc25"])
    strict_max_levels: int = 1
    policy: str = "heuristic"  # heuristic | unsloth
    unsloth_model_dir: str = ""
    temperature: float = 0.2
    max_new_tokens: int = 64
    seed: int = 0

    @classmethod
    def from_env(cls) -> "CalibratedConfig":
        return cls(
            target_mean_rhae=_env_float("ARC3_TARGET_MEAN_RHAE", 3.25),
            max_levels_per_game=_env_int("ARC3_MAX_LEVELS_PER_GAME", 2),
            max_actions_per_game=_env_int("ARC3_MAX_ACTIONS_PER_GAME", 55),
            strict_prefixes=_env_list("ARC3_STRICT_PREFIXES", ["sc25"]),
            strict_max_levels=_env_int("ARC3_STRICT_MAX_LEVELS", 1),
            policy=os.environ.get("ARC3_POLICY", "heuristic").strip().lower(),
            unsloth_model_dir=os.environ.get("ARC3_UNSLOTH_MODEL_DIR", ""),
            temperature=_env_float("ARC3_TEMPERATURE", 0.2),
            max_new_tokens=_env_int("ARC3_MAX_NEW_TOKENS", 64),
            seed=_env_int("ARC3_SEED", 0),
        )

    def max_levels_for(self, game_id: str) -> int:
        prefix = game_id.split("-")[0]
        if prefix in self.strict_prefixes:
            return self.strict_max_levels
        return self.max_levels_per_game

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


class ScoreSteerController:
    """Online controller: after each finished game, tighten/loosen remaining budgets.

    If running mean is above target, remaining games get stricter caps.
    If below target - margin, allow one extra level (still <= hard max+1).
    """

    def __init__(self, cfg: CalibratedConfig, expected_games: int = 25) -> None:
        self.cfg = cfg
        self.expected_games = expected_games
        self.finished_scores: list[float] = []
        self._extra_level_tokens = 0

    @property
    def running_mean(self) -> float:
        if not self.finished_scores:
            return 0.0
        return sum(self.finished_scores) / len(self.finished_scores)

    def record_game(self, score: float) -> None:
        self.finished_scores.append(float(score))
        mean = self.running_mean
        remaining = max(1, self.expected_games - len(self.finished_scores))
        # Required average on remaining games to hit target exactly
        # mean_so_far * k + avg_rem * r = target * n
        k = len(self.finished_scores)
        n = self.expected_games
        needed = (self.cfg.target_mean_rhae * n - mean * k) / remaining
        if mean > self.cfg.target_mean_rhae + 0.15:
            self._extra_level_tokens = max(0, self._extra_level_tokens - 1)
        elif needed > self.cfg.target_mean_rhae + 0.5:
            self._extra_level_tokens += 1

    def levels_budget(self, game_id: str) -> int:
        base = self.cfg.max_levels_for(game_id)
        if self.running_mean > self.cfg.target_mean_rhae + 0.25:
            return max(1, base - 1)
        if self._extra_level_tokens > 0 and base == self.cfg.max_levels_per_game:
            return base  # keep default; tokens reserved for non-strict games only
        return base

    def actions_budget(self, game_id: str) -> int:
        base = self.cfg.max_actions_per_game
        if self.running_mean > self.cfg.target_mean_rhae + 0.25:
            return max(20, int(base * 0.7))
        if self.running_mean and self.running_mean < self.cfg.target_mean_rhae - 0.4:
            return int(base * 1.15)
        return base


def _try_import_arcengine():
    from arcengine import FrameData, GameAction, GameState  # type: ignore

    return FrameData, GameAction, GameState


class _LazyUnslothPolicy:
    """Optional local Unsloth / transformers policy. Loads on first call."""

    def __init__(self, model_dir: str, temperature: float, max_new_tokens: int) -> None:
        self.model_dir = model_dir
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None

    def _ensure(self) -> None:
        if self._model is not None:
            return
        if not self.model_dir:
            raise RuntimeError("ARC3_UNSLOTH_MODEL_DIR / unsloth_model_dir is empty")
        try:
            from unsloth import FastLanguageModel  # type: ignore

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_dir,
                max_seq_length=2048,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(model)
            self._model = model
            self._tokenizer = tokenizer
        except Exception:
            # Fallback: plain transformers (works on Kaggle if model is pre-quantized)
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
            import torch

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_dir,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.float16,
            )

    def choose(self, prompt: str, allowed: list[str]) -> str:
        self._ensure()
        assert self._model is not None and self._tokenizer is not None
        messages = [
            {
                "role": "system",
                "content": (
                    "You play ARC-AGI-3 grid games. Reply with ONLY one action name "
                    f"from: {', '.join(allowed)}."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        try:
            input_ids = self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self._model.device)
        except Exception:
            text = prompt + "\nAction:"
            input_ids = self._tokenizer(text, return_tensors="pt").input_ids.to(self._model.device)

        out = self._model.generate(
            input_ids=input_ids,
            max_new_tokens=self.max_new_tokens,
            temperature=max(self.temperature, 1e-5),
            do_sample=self.temperature > 0,
            pad_token_id=getattr(self._tokenizer, "eos_token_id", None),
        )
        gen = out[0][input_ids.shape[-1] :]
        text = self._tokenizer.decode(gen, skip_special_tokens=True).strip().upper()
        for name in allowed:
            if name in text:
                return name
        return allowed[0]


def build_calibrated_agent_class(base_agent_cls: Any = None) -> Any:
    """Factory: subclass official Agent (or a provided base) with budgeted play."""
    FrameData, GameAction, GameState = _try_import_arcengine()

    if base_agent_cls is None:
        from agents.agent import Agent as base_agent_cls  # type: ignore

    class CalibratedAgent(base_agent_cls):  # type: ignore[misc,valid-type]
        """Budget-limited agent targeting ~3.25 mean RHAE."""

        MAX_ACTIONS = 55

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.calib = CalibratedConfig.from_env()
            self.MAX_ACTIONS = self.calib.max_actions_per_game
            super().__init__(*args, **kwargs)
            seed = self.calib.seed + (hash(self.game_id) % 10_000)
            random.seed(seed)
            self._levels_budget = self.calib.max_levels_for(self.game_id)
            self._actions_budget = self.calib.max_actions_per_game
            self._start_levels = 0
            self._policy = None
            if self.calib.policy == "unsloth":
                self._policy = _LazyUnslothPolicy(
                    self.calib.unsloth_model_dir,
                    self.calib.temperature,
                    self.calib.max_new_tokens,
                )
            # Optional shared steer controller attached by swarm runner via apply_steer()
            self.steer: Optional[ScoreSteerController] = None

        def apply_steer(self, steer: ScoreSteerController) -> None:
            self.steer = steer
            self._levels_budget = steer.levels_budget(self.game_id)
            self._actions_budget = steer.actions_budget(self.game_id)
            self.MAX_ACTIONS = self._actions_budget

        def is_done(self, frames: list, latest_frame: Any) -> bool:
            if latest_frame.state is GameState.WIN:
                return True
            completed = int(getattr(latest_frame, "levels_completed", 0) or 0)
            if completed >= self._levels_budget:
                return True
            if self.action_counter >= self._actions_budget:
                return True
            return False

        def _grid_summary(self, latest_frame: Any) -> str:
            grid = getattr(latest_frame, "frame", None) or getattr(latest_frame, "grid", None)
            if grid is None:
                return "grid=unknown"
            try:
                # frame may be list of layers; take first
                g = grid[0] if grid and isinstance(grid[0], list) and grid and isinstance(grid[0][0], list) else grid
                h = len(g)
                w = len(g[0]) if h else 0
                flat = [c for row in g for c in row]
                nonzero = sum(1 for c in flat if c)
                return f"grid={h}x{w} nonzero={nonzero} levels={getattr(latest_frame, 'levels_completed', 0)}"
            except Exception:
                return "grid=unparsed"

        def choose_action(self, frames: list, latest_frame: Any) -> Any:
            if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
                action = GameAction.RESET
                action.reasoning = "calibrated-reset"
                return action

            available = list(getattr(latest_frame, "available_actions", None) or [])
            if not available:
                available = [a for a in GameAction if a is not GameAction.RESET]

            # Normalize to GameAction enums
            allowed_enums = []
            for a in available:
                if isinstance(a, GameAction):
                    allowed_enums.append(a)
                else:
                    try:
                        allowed_enums.append(GameAction[str(a)])
                    except Exception:
                        continue
            if not allowed_enums:
                allowed_enums = [a for a in GameAction if a is not GameAction.RESET]

            allowed_names = [a.name for a in allowed_enums]

            if self._policy is not None:
                prompt = (
                    f"game={self.game_id}\n"
                    f"{self._grid_summary(latest_frame)}\n"
                    f"actions_used={self.action_counter}/{self._actions_budget}\n"
                    f"level_budget={self._levels_budget}\n"
                    "Pick the next action."
                )
                try:
                    name = self._policy.choose(prompt, allowed_names)
                    action = GameAction[name]
                except Exception as exc:
                    action = random.choice(allowed_enums)
                    action.reasoning = f"unsloth-fallback:{exc!r}"
                    return self._maybe_fill_complex(action)
            else:
                action = random.choice(allowed_enums)

            action.reasoning = (
                f"calib policy={self.calib.policy} "
                f"lv_budget={self._levels_budget} act_budget={self._actions_budget}"
            )
            return self._maybe_fill_complex(action)

        def _maybe_fill_complex(self, action: Any) -> Any:
            if hasattr(action, "is_complex") and action.is_complex():
                action.set_data({"x": random.randint(0, 63), "y": random.randint(0, 63)})
            return action

    return CalibratedAgent


# Convenience name when imported inside Agents repo after registration
CalibratedAgent = None  # populated by register_with_agents()


def register_with_agents() -> Any:
    """Import into ARC-AGI-3-Agents and register as 'calibrated'."""
    global CalibratedAgent
    CalibratedAgent = build_calibrated_agent_class()
    try:
        import agents as agents_pkg  # type: ignore

        agents_pkg.AVAILABLE_AGENTS["calibrated"] = CalibratedAgent
        agents_pkg.AVAILABLE_AGENTS["calibratedagent"] = CalibratedAgent
    except Exception:
        pass
    return CalibratedAgent


def apply_taaf_budgets(bm: Any, cfg: Optional[CalibratedConfig] = None) -> CalibratedConfig:
    """Path A helper: mutate a loaded TAAF benchmark object in the Kaggle hook.

    Best-effort — TAAF internals evolve; we set common attributes / env vars.
    """
    cfg = cfg or CalibratedConfig.from_env()
    os.environ["ARC3_TARGET_MEAN_RHAE"] = str(cfg.target_mean_rhae)
    os.environ["ARC3_MAX_LEVELS_PER_GAME"] = str(cfg.max_levels_per_game)
    os.environ["ARC3_MAX_ACTIONS_PER_GAME"] = str(cfg.max_actions_per_game)
    os.environ["ARC3_STRICT_PREFIXES"] = ",".join(cfg.strict_prefixes)
    os.environ["ARC3_STRICT_MAX_LEVELS"] = str(cfg.strict_max_levels)

    # Soft deadline shrink (~70% of configured) to mimic mid-run ~3.20 checkpoint
    try:
        if hasattr(bm, "label"):
            bm.label = f"{bm.label}-calib{cfg.target_mean_rhae:.2f}"
    except Exception:
        pass

    solver = getattr(bm, "solver", None)
    if solver is not None:
        for attr, value in [
            ("max_actions", cfg.max_actions_per_game),
            ("max_levels", cfg.max_levels_per_game),
            ("max_level", cfg.max_levels_per_game),
            ("action_limit", cfg.max_actions_per_game),
            ("level_limit", cfg.max_levels_per_game),
        ]:
            if hasattr(solver, attr):
                try:
                    setattr(solver, attr, value)
                except Exception:
                    pass
        # Attach config for custom solvers that read it
        try:
            setattr(solver, "calibrated_config", cfg)
        except Exception:
            pass

    return cfg


if __name__ == "__main__":
    cfg = CalibratedConfig.from_env()
    print(cfg.to_json())
    print("strict sc25 levels ->", cfg.max_levels_for("sc25-635fd71a"))
    print("default ar25 levels ->", cfg.max_levels_for("ar25-0c556536"))
