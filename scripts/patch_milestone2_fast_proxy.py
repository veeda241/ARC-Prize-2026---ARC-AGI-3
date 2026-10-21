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
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parent.parent / "kaggle_submit_milestone2" / "arc-agi-3-milestone-2-solution.ipynb"

# The config cell stores its source as one JSON string, so these fragments are matched
# against the raw file bytes: "\n" here is a backslash plus n, not a newline.
PASSES = (
    r"else:\n    if USE_PRIORITY_SCHEDULING:\n"
    r"        bm.solver.concurrency = (25 - len(demo_excluded_games)) * bm.n_passes\n"
)
PASSES_PROXY = (
    r"else:\n"
    r"    bm.n_passes = 1  # FAST_PROXY: one draw per game, like the hidden rerun\n"
    r"    if USE_PRIORITY_SCHEDULING:\n"
    r"        bm.solver.concurrency = (25 - len(demo_excluded_games)) * bm.n_passes\n"
)
RUNTIME = r"        bm.solver.max_runtime_s_per_game = 532*60 * bm.solver.concurrency // 110\n"
RUNTIME_PROXY = (
    r"        bm.solver.max_runtime_s_per_game = 25*60  # FAST_PROXY: ~1 h for the whole demo set\n"
)

# Last entry of setup_env, followed by the closing brace. Arms are inserted here rather
# than in the `if USE_PRIORITY_SCHEDULING:` update so they are set in both branches.
ENV_TAIL = r"    'EXPOSE_UNDO': 'on',\n}"
ARMS_MARKER = "# FAST_ARMS"


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore", action="store_true", help="put the 4-pass demo config back, arms included")
    parser.add_argument("--arm", action="append", default=[], metavar="KEY=VALUE",
                        help="set a harness env knob the kernel never sets (repeatable)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    arms = [parse_arm(spec) for spec in args.arm]
    if len({key for key, _ in arms}) != len(arms):
        raise SystemExit("abort: duplicate --arm keys")

    raw = NOTEBOOK.read_text(encoding="utf-8")
    text = raw

    if args.restore:
        if args.arm:
            raise SystemExit("abort: --restore takes no --arm")
        if "FAST_PROXY" in text:
            text = apply(text, PASSES_PROXY, PASSES, "passes")
            text = apply(text, RUNTIME_PROXY, RUNTIME, "runtime")
        text = strip_arms(text)
    else:
        if "FAST_PROXY" not in text:
            text = apply(text, PASSES, PASSES_PROXY, "passes")
            text = apply(text, RUNTIME, RUNTIME_PROXY, "runtime")
        elif not arms:
            print("already patched")
            return 0
        if arms:
            text = strip_arms(text)
            text = apply(text, ENV_TAIL, arms_block(arms), "arms")

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
