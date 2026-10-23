#!/usr/bin/env python3
"""Turn the offline branch of the milestone-2 kernel into a hidden-like fast proxy.

The demo branch plays 25 public games with 4 passes at ~8 h per game (8.4 h total).
This switches it to 1 pass at 25 min per game, which is how the hidden set is actually
scored, and finishes in about an hour of GPU. The `if TRUE_SUBMISSION:` branch is not
touched, so a rerun of a patched kernel behaves exactly like ref 57402922.

`--arm` additionally flips harness environment knobs that the kernel never sets (and
that therefore run at their default in the submission that scored 30.55), so a knob set
can be measured in ~1 h of proxy before it costs a submission slot.

Usage:
    python scripts/patch_milestone2_fast_proxy.py            # patch
    python scripts/patch_milestone2_fast_proxy.py --restore  # back to 4 passes, no arms
    python scripts/patch_milestone2_fast_proxy.py --arm ARC3_NOOP_REPEAT_GUARD=1 --arm ARC3_DEATH_LEDGER=1
    python scripts/patch_milestone2_fast_proxy.py --scored-concurrency 28  # only the hidden branch
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parent.parent / "kaggle_submit_milestone2" / "arc-agi-3-milestone-2-solution.ipynb"

# The config cell stores its source as one JSON string, so these fragments are matched
# against the raw file bytes: "\n" here is a backslash plus n, not a newline.
PASSES = r"else:\n    if USE_PRIORITY_SCHEDULING:\n"
PASSES_PROXY = (
    r"else:\n"
    r"    bm.n_passes = 1  # FAST_PROXY: one draw per game, like the hidden rerun\n"
    r"    if USE_PRIORITY_SCHEDULING:\n"
)
RUNTIME = r"        bm.solver.max_runtime_s_per_game = 532*60 * bm.solver.concurrency // 110\n"


def runtime_proxy(minutes: int) -> str:
    return (
        f"        bm.solver.max_runtime_s_per_game = {minutes}*60"
        "  # FAST_PROXY: sized for the demo set, not the hidden rerun\\n"
    )

CONCURRENCY = r"        bm.solver.concurrency = (25 - len(demo_excluded_games)) * bm.n_passes\n"


def concurrency_proxy(value: int) -> str:
    # One server, one GPU: the derived 25-way fan-out left 8 of 25 games with 0 actions and
    # 0 tokens in version 5 because they were never scheduled before the deadline.
    return f"        bm.solver.concurrency = {value}  # FAST_PROXY: fixed, not game-count derived\n"


# The health gate is a fixed 12 min measured from notebook start, and the 185 GB model
# precache alone took 2157 s (36 min) in version 4, so the benchmark was released against a
# port that nothing was listening on: all 25 games scored 0 with 0 actions and 0 tokens.
STARTUP = "SERVER_STARTUP_TIMEOUT = 12 * 60"
STARTUP_PROXY = "SERVER_STARTUP_TIMEOUT = 60 * 60  # FAST_PROXY: precache took 2157s on v4"

# The scored branch is the only part of the notebook that the hidden rerun executes, so a
# change here is the one kind of edit a submission slot can actually measure. The author's
# 120 streams one-request-per-game over the whole 110-game set on a single GPU.
GAMES = 110
SESSION_S = 32400
LOAD_S = 720
PRISTINE_SCORED = (
    r"        bm.solver.concurrency = 120  # game count should be 110\n"
    r"        bm.solver.max_runtime_s_per_game = 532*60\n"
)
SCORED_MARK = "# SCORED:"


def scored_block(value: int) -> str:
    # A game that never finishes holds its slot until its cap, so the worst case is
    # ceil(GAMES / concurrency) waves and every cap has to fit inside the session.
    waves = -(-GAMES // value)
    cap = (SESSION_S - LOAD_S) // waves
    return (
        f"        bm.solver.concurrency = {value}  {SCORED_MARK} one GPU cannot feed 120 streams\\n"
        f"        bm.solver.max_runtime_s_per_game = {cap}  {SCORED_MARK} {waves} waves of {value} games in {SESSION_S}s\\n"
    )


def set_scored_concurrency(text: str, value: int | None) -> str:
    """Rewrite the scored branch's concurrency/cap pair, or put the author's pair back."""
    sep = "\\n"
    lines = text.split(sep)

    def is_conc(line: str) -> bool:
        st = line.strip()
        return st.startswith("bm.solver.concurrency =") and (
            "game count should be 110" in st or SCORED_MARK in st
        )

    def is_runtime(line: str) -> bool:
        st = line.strip()
        if st.startswith("bm.solver.max_runtime_s_per_game ="):
            return SCORED_MARK in st or st == "bm.solver.max_runtime_s_per_game = 532*60"
        return False

    hits = [i for i, line in enumerate(lines) if is_conc(line)]
    if not hits:
        if value:
            raise SystemExit("abort: no scored-branch concurrency line found")
        return text
    if len(hits) > 1:
        raise SystemExit(f"abort: scored concurrency line matched {len(hits)} times")
    i = hits[0]
    if not is_runtime(lines[i + 1]):
        raise SystemExit(f"abort: line after scored concurrency is not the runtime cap: {lines[i + 1]!r}")
    if lines[i][: len(lines[i]) - len(lines[i].lstrip())] != "        ":
        raise SystemExit("abort: scored concurrency is not at the depth the block assumes")
    block = (scored_block(value) if value else PRISTINE_SCORED).removesuffix(sep)
    lines[i : i + 2] = block.split(sep)
    return sep.join(lines)

# Last entry of setup_env, followed by the closing brace. Arms are inserted here rather
# than in the `if USE_PRIORITY_SCHEDULING:` update so they are set in both branches.
ENV_TAIL = r"    'EXPOSE_UNDO': 'on',\n}"
ARMS_MARKER = "# FAST_ARMS"


def _is_proxy_concurrency(line: str) -> bool:
    st = line.strip()
    return st.startswith("bm.solver.concurrency =") and (
        "(25 - len(demo_excluded_games))" in st or "FAST_PROXY: fixed" in st
    )


def set_runtime(text: str, minutes: int | None) -> str:
    """Rewrite the offline branch's per-game cap, or restore the derived form."""
    sep = "\\n"
    lines = text.split(sep)

    def is_offline_runtime(line: str) -> bool:
        st = line.strip()
        if not st.startswith("bm.solver.max_runtime_s_per_game ="):
            return False
        return "concurrency // 110" in st or "FAST_PROXY" in st

    hits = [i for i, line in enumerate(lines) if is_offline_runtime(line)]
    if not hits:
        if minutes:
            raise SystemExit("abort: no offline-branch runtime line found")
        return text
    if len(hits) > 1:
        raise SystemExit(f"abort: offline runtime line matched {len(hits)} times")
    lines[hits[0]] = (
        runtime_proxy(minutes).removesuffix(sep) if minutes else RUNTIME.removesuffix(sep)
    )
    return sep.join(lines)


def set_concurrency(text: str, value: int | None) -> str:
    """Rewrite the offline branch's concurrency line, or restore the derived form."""
    sep = "\\n"
    derived = CONCURRENCY[: -len(sep)]
    lines = text.split(sep)
    hits = [i for i, line in enumerate(lines) if _is_proxy_concurrency(line)]
    if not hits:
        if value:
            raise SystemExit("abort: no offline-branch concurrency line found")
        return text
    if len(hits) > 1:
        raise SystemExit(f"abort: concurrency line matched {len(hits)} times")
    i = hits[0]
    indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
    lines[i] = (
        f"{indent}bm.solver.concurrency = {value}  # FAST_PROXY: fixed, not game-count derived"
        if value
        else derived
    )
    return sep.join(lines)


def parse_arm(spec: str) -> tuple[str, str]:
    key, sep, value = spec.partition("=")
    key = key.strip()
    if not sep or not key or not value.strip():
        raise SystemExit(f"abort: --arm wants KEY=VALUE, got {spec!r}")
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
        raise SystemExit(f"abort: {key!r} is not an environment variable name")
    if "'" in value or "\\" in value:
        raise SystemExit(f"abort: {value!r} cannot be embedded in the notebook source")
    return key, value.strip()


def arms_block(arms: list[tuple[str, str]]) -> str:
    lines = "".join(f"    {key!r}: {value!r},\\n" for key, value in arms)
    return ENV_TAIL[:-1] + f"    {ARMS_MARKER}\\n{lines}}}"


def strip_arms(text: str) -> str:
    # The notebook keeps each cell as one JSON string, so line breaks inside the source
    # are the two characters backslash + n, not real newlines.
    sep = "\\n"
    marker = "    " + ARMS_MARKER
    head, found, _ = text.partition(marker)
    if not found:
        return text
    if text.count(marker) > 1:
        raise SystemExit("abort: more than one arm block, refusing to guess")
    lines = text[len(head) + len(marker):].split(sep)
    # lines[0] is the rest of the marker comment; the block is the arm entries up to "}".
    body = lines[1:]
    for i, line in enumerate(body):
        if line.strip() != "}":
            if not re.fullmatch(r"    '[A-Z0-9_]+': '[^']*',", line):
                raise SystemExit(f"abort: unexpected arm line {line!r}")
            continue
        rebuilt = head + "}" + sep + sep.join(body[i + 1:])
        if rebuilt.count(marker):
            raise SystemExit("abort: arm block did not fully come out")
        return rebuilt
    raise SystemExit("abort: arm block is not closed by '}'")


def apply(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"abort: anchor for {label} matched {count} times, expected 1")
    return text.replace(old, new, 1)


def restore(text: str) -> str:
    """Undo every patch this script can make, offline branch and scored branch alike."""
    for new, old, label in (
        (PASSES_PROXY, PASSES, "passes"),
        (STARTUP_PROXY, STARTUP, "startup"),
    ):
        if new in text:
            text = apply(text, new, old, label)
    text = strip_arms(text)
    text = set_runtime(text, None)
    text = set_concurrency(text, None)
    return set_scored_concurrency(text, None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore", action="store_true", help="put the 4-pass demo config back, arms included")
    parser.add_argument("--arm", action="append", default=[], metavar="KEY=VALUE",
                        help="set a harness env knob the kernel never sets (repeatable)")
    parser.add_argument("--concurrency", type=int, metavar="N",
                        help="fixed offline concurrency instead of game-count derived")
    parser.add_argument("--runtime-minutes", type=int, default=25, metavar="N",
                        help="offline per-game cap, minutes (default 25)")
    parser.add_argument("--scored-concurrency", type=int, metavar="N",
                        help="concurrency for the TRUE_SUBMISSION branch, cap re-derived from the session")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    arms = [parse_arm(spec) for spec in args.arm]
    if len({key for key, _ in arms}) != len(arms):
        raise SystemExit("abort: duplicate --arm keys")
    if args.concurrency is not None and args.concurrency < 1:
        raise SystemExit("abort: --concurrency must be at least 1")
    if not 1 <= args.runtime_minutes <= 532:
        raise SystemExit("abort: --runtime-minutes must be between 1 and 532")
    scored = args.scored_concurrency
    if scored is not None and not 1 <= scored <= GAMES:
        raise SystemExit(f"abort: --scored-concurrency must be between 1 and {GAMES}")

    raw = NOTEBOOK.read_text(encoding="utf-8")
    text = raw

    if args.restore:
        if args.arm or args.concurrency or scored or args.runtime_minutes != 25:
            raise SystemExit("abort: --restore takes no other patch flags")
        text = restore(text)
    else:
        # Each pair is applied independently so a notebook patched by an older copy of this
        # script still picks up the anchors it missed.
        for old, new, label in (
            (PASSES, PASSES_PROXY, "passes"),
            (STARTUP, STARTUP_PROXY, "startup"),
        ):
            if new in text:
                continue
            text = apply(text, old, new, label)
        text = set_runtime(text, args.runtime_minutes)
        text = set_concurrency(text, args.concurrency)
        if arms:
            text = strip_arms(text)
            text = apply(text, ENV_TAIL, arms_block(arms), "arms")
        # Unconditional, so a proxy run can never silently inherit the scored branch of the
        # version before it: with no --scored-concurrency the pair goes back to the author's.
        text = set_scored_concurrency(text, scored)

    if text == raw:
        print("no change")
    elif args.dry_run:
        print(f"dry run: {len(raw)} -> {len(text)} chars")
    else:
        NOTEBOOK.write_text(text, encoding="utf-8", newline="")
        print(f"{'restored' if args.restore else 'patched'} {NOTEBOOK.name} ({len(raw)} -> {len(text)} chars)")

    for key, value in ([] if args.restore else arms):
        print(f"    armed {key}={value}")
    for line in text.split(r"\n"):
        st = line.strip()
        if st.startswith("bm.") and any(k in st for k in ("n_passes", "max_runtime_s_per_game", "save_request_logs", "concurrency")):
            print("   ", st[:120])
    return 0


if __name__ == "__main__":
    sys.exit(main())
