#!/usr/bin/env python3
"""Read the harness request logs from a fast-proxy run and explain stalled games.

The harness writes one JSON object per line to `<game_id>_requests.jsonl` in the run
directory (see ARC3-Inference/inference/agent/tool_agent.py):
    {"messages": [...], "tools": [...], "event": "request"|"response",
     "tool_choice": ..., "finish_reason": ..., "analysis_step": n, "action": n,
     "request_index_within_turn": n, "usage": {...}, "served_by": ...}
Response snapshots carry the *request* messages, so the model's answer to step N is
visible as an assistant message inside the step N+1 request.

Usage (after `kaggle kernels output ... -p artifacts/proxy_run`):
    python scripts/analyze_proxy_traces.py artifacts/proxy_run
    python scripts/analyze_proxy_traces.py artifacts/proxy_run --games sk48 bp35
The run output also carries `.per_game_cards.pkl`; when found, rows are sorted by that
proxy score so a legitimately cheap game does not look like a stall.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from parse_milestone2_cards import load_cards, parse_card


def load_snapshots(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(text_of(part.get("text", "") if isinstance(part, dict) else part) for part in content)
    return "" if content is None else str(content)


def tail_conversation(messages: list[dict[str, Any]], width: int) -> list[str]:
    lines: list[str] = []
    for message in reversed(messages):
        role = str(message.get("role", "?"))
        if role == "assistant" and not lines:
            body = text_of(message.get("content")).strip()
            calls = message.get("tool_calls") or []
            rendered = []
            for call in calls:
                function = (call.get("function") or {}) if isinstance(call, dict) else {}
                rendered.append(f"{function.get('name', 'tool')}({str(function.get('arguments', ''))[:width]})")
            lines.append(f"assistant: {body[:width] or '(no text)'}" + (f"  calls: {'; '.join(rendered)}" if rendered else ""))
        elif role == "tool" and len(lines) < 3:
            lines.append(f"tool result: {text_of(message.get('content')).strip()[:width]}")
        elif len(lines) >= 3:
            break
    return lines


def streak(actions: list[int]) -> int:
    best = run = 1
    for i in range(1, len(actions)):
        run = run + 1 if actions[i] == actions[i - 1] else 1
        best = max(best, run)
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", help="directory holding the downloaded kernel output")
    parser.add_argument("--games", nargs="*", default=None, help="game ids to print in full")
    parser.add_argument("--width", type=int, default=220, help="truncate quoted model output to this many chars")
    parser.add_argument("--cards", type=Path, default=None, help=".per_game_cards.pkl to join scores from (auto-discovered under root)")
    parser.add_argument("--top", type=int, default=12, help="how many rows in the summary table")
    args = parser.parse_args()

    root = Path(args.root)
    logs = sorted(root.rglob("*_requests.jsonl")) or sorted(root.rglob("requests.jsonl"))
    if not logs:
        print(f"no *_requests.jsonl under {root}. Run the proxy first, then download it:")
        print(r"  kaggle kernels output vyassenthilkumar/arc3-milestone2-flashnext-duck -p .\artifacts\proxy_run")
        return 1

    rows = []
    for path in logs:
        game = path.stem.replace("_requests", "")
        snapshots = load_snapshots(path)
        responses = [s for s in snapshots if s.get("event") == "response"]
        actions = [int(s["action"]) for s in snapshots if isinstance(s.get("action"), int)]
        steps = [int(s["analysis_step"]) for s in snapshots if isinstance(s.get("analysis_step"), int)]
        finish = Counter(str(s.get("finish_reason")) for s in responses if s.get("finish_reason") is not None)
        tokens = sum(int((s.get("usage") or {}).get("total_tokens", 0)) for s in responses)
        rows.append(
            {
                "game": game,
                "requests": len(snapshots),
                "steps": max(steps) if steps else 0,
                "distinct_actions": len(set(actions)),
                "streak": streak(actions) if len(actions) > 1 else 0,
                "finish": finish,
                "tokens": tokens,
                "last": tail_conversation(snapshots[-1].get("messages", []) if snapshots else [], args.width),
                "path": path,
            }
        )

    cards = {}
    found = sorted(root.rglob("*.per_game_cards.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    cards_path = args.cards or (found[0] if found else None)
    if cards_path and Path(cards_path).is_file():
        for summary in load_cards(Path(cards_path)).values():
            card = parse_card(summary[0])
            cards[card["game_id"]] = card
        print(f"joined scores from {Path(cards_path).name}")

    for r in rows:
        card = next((c for g, c in cards.items() if g.lower() in r["game"].lower()), None)
        r["score"] = max((p["score"] for p in card["passes"]), default=0.0) if card else None
        r["levels"] = card["levels_completed"] if card else None

    def cell(value: Any, width: int) -> str:
        text = "-" if value is None else (f"{value:.1f}" if isinstance(value, float) else str(value))
        return f"{text:>{width}}"

    rows.sort(key=lambda r: (r["score"] if r["score"] is not None else 999.0, r["distinct_actions"], -r["requests"]))
    has_scores = any(r["score"] is not None for r in rows)
    header = f"{'game':<16}" + (f"{'score':>7}{'lv':>4}" if has_scores else "")
    print(header + f"{'reqs':>6}{'steps':>7}{'acts':>6}{'streak':>8}{'tokens':>10}  finish_reason")
    for r in rows[: args.top]:
        line = f"{r['game']:<16}"
        if has_scores:
            line += cell(r["score"], 7) + cell(r["levels"], 4)
        line += f"{r['requests']:>6}{r['steps']:>7}{r['distinct_actions']:>6}{r['streak']:>8}{r['tokens']:>10}  {dict(r['finish'])}"
        print(line)
    order = "proxy score, then fewest distinct actions" if has_scores else "fewest distinct actions, then most requests"
    print(f"\n{len(rows)} games logged; sorted worst-first by {order}.")
    print("streak = longest run of the same action (a no-op loop); acts = distinct action numbers tried; lv = levels reached.")

    wanted = set(args.games or [])
    if wanted:
        for r in rows:
            if any(w.lower() in r["game"].lower() for w in wanted):
                print(f"\n=== {r['game']} ({r['path'].name}) ===")
                print(f"  requests {r['requests']}  steps {r['steps']}  distinct actions {r['distinct_actions']}  streak {r['streak']}")
                print(f"  finish_reason {dict(r['finish'])}  tokens {r['tokens']}")
                for line in r["last"]:
                    print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
