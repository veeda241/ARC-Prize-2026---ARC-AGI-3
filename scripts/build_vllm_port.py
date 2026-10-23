#!/usr/bin/env python3
"""Stage the vLLM + NVFP4/FP8 port as its own Kaggle kernel.

Upstream is lordhansolo's generic TAAF launcher, which carries no model code: the harness, the
patched vLLM build and the checkpoint all arrive as attachments. Two things are added to it, and
both are needed because the attachments are read-only and their server budget was baked at
dataset-build time:

  * a *fork cell*, inserted before the launcher runs the bundle's setup commands, which copies the
    bundle into the working directory, rewrites the baked vLLM constants and exports the arms, then
    rebinds BUNDLE_DIR so everything downstream reads the patched copy;
  * the upstream's empty "Inline customization hook" cell, split into the scored branch (what the
    hidden rerun executes) and the offline branch (what every interactive session executes, so a
    test never costs a full demo run).

Re-run after pulling a fresh upstream:
    kaggle kernels pull lordhansolo/built-on-tufa-labs-duck-harness-milestone-2 -p ./upstream
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "upstream" / "built-on-tufa-labs-duck-harness-milestone-2.ipynb"
OUT_DIR = ROOT / "kaggle_submit_milestone2_vllm"
NOTEBOOK = OUT_DIR / "arc3-milestone2-vllm-port.ipynb"

OWNER = "vyassenthilkumar"
SLUG = "arc3-milestone2-vllm-port"
# Kaggle derives the kernel slug from the TITLE, not from the metadata "id", so the two must
# agree word-for-word or pushes land on a kernel nobody is polling.
TITLE = "ARC3 Milestone2 Vllm Port"

GAMES = 110
SESSION_S = 32400

# Default is the leader's own stock budget, so a run measures the published configuration; pass
# --server NAME=VALUE to move off it.
DEFAULT_SERVER_OVERRIDES: dict[str, int] = {}

# The strategy audit is the harness's only stuck-breaker and its interval is a *fraction of the
# per-game runtime cap*, so a one-wave schedule (cap ~29,000 s) makes it fire about once per game.
# Capping the interval in absolute seconds is what keeps it firing on a long budget.
AUDIT_FILE = "src/ARC3-Inference/inference/framework/solver.py"
AUDIT_ANCHOR = (
    "initial_runtime_limit * 0.25 if initial_runtime_limit is not None "
    "and initial_runtime_limit > 0 else None"
)


def audit_replacement(seconds: int) -> str:
    return (
        f"min(initial_runtime_limit * 0.25, {float(seconds)!r}) if initial_runtime_limit is not None "
        f"and initial_runtime_limit > 0 else None"
    )



def fork_cell(
    server: dict[str, int],
    arms: dict[str, str],
    audit_s: int = 0,
    env_overrides: dict[str, str] | None = None,
) -> str:
    env_overrides = env_overrides or {}
    lines = [
        "# Fork of the read-only bundle, inserted by scripts/build_vllm_port.py.",
        "# The bundle's setup script bakes its vLLM server arguments at dataset-build time, so the",
        "# serving budget cannot be changed from the notebook unless the setup commands themselves",
        "# are edited -- which needs a writable copy, because /kaggle/input is read-only.",
        "import re",
        "import shutil",
        "",
        "_upstream_bundle = BUNDLE_DIR",
        "BUNDLE_DIR = WORKING_DIR / 'bundle'",
        "if not BUNDLE_DIR.exists():",
        "    shutil.copytree(_upstream_bundle, BUNDLE_DIR, symlinks=True)",
        "print(f'port: bundle forked from {_upstream_bundle} to {BUNDLE_DIR}', flush=True)",
        "",
        "# The overlay in the runtime dataset is a python3.12 dist-packages tree, but the launcher",
        "# runs its setup heredoc with the notebook interpreter and every child process inherits it",
        "# (watchdog, server, applier all use sys.executable). Kaggle moved the notebook image to",
        "# 3.13, so that chain now imports 3.12-built torch C extensions and dies at once with",
        "# 'Failed to load PyTorch C extensions'. Pin the heredoc to a 3.12 interpreter instead.",
        "_py312 = next((_candidate for _candidate in ('/usr/bin/python3.12',",
        "                                                  '/usr/local/bin/python3.12',",
        "                                                  shutil.which('python3.12') or '')",
        "                  if _candidate and Path(_candidate).exists()), None)",
        "if _py312 is None:",
        "    raise RuntimeError(",
        "        f'port: no CPython 3.12 on this image (running {sys.version.split()[0]}), so the '",
        "        'cp312 vLLM overlay in lordhansolo/vllm-main-e975732-arc3 cannot be loaded.'",
        "    )",
        "print(f'port: setup interpreter -> {_py312}', flush=True)",
        "",
        "# The notebook kernel itself stays 3.13 by design; only this child chain must be 3.12, so",
        "# verify the binary rather than trusting its name (a stale symlink costs nine hours).",
        "_probe = subprocess.run([_py312, '-c',",
        "                         'import sys; print(\"%d.%d\" % sys.version_info[:2])'],",
        "                        capture_output=True, text=True)",
        "if _probe.stdout.strip() != '3.12':",
        "    raise RuntimeError(",
        "        f'port: {_py312} reports {_probe.stdout.strip()!r}, not 3.12: {_probe.stderr[-300:]}'",
        "    )",
        "print('port: child interpreter verified 3.12', flush=True)",
        "",
        "_setup_path = BUNDLE_DIR / 'setup_commands.json'",
        "_commands = json.loads(_setup_path.read_text(encoding='utf-8'))",
        "_commands = [_c.replace('\"$PYTHON\" - <<', f'{_py312} - <<', 1)",
        "               if _c.lstrip().startswith('\"$PYTHON\" - <<') else _c for _c in _commands]",
        "_server_overrides = " + repr(server),
        "_changed = []",
        "for _name, _value in _server_overrides.items():",
        "    _anchor = f'{_name} = '",
        "    _hits = 0",
        "    for _index, _command in enumerate(_commands):",
        "        _cursor = 0",
        "        while True:",
        "            _start = _command.find(_anchor, _cursor)",
        "            if _start < 0:",
        "                break",
        "            _start += len(_anchor)",
        "            _end = _command.index(chr(10), _start)",
        "            _current = _command[_start:_end].strip()",
        "            if not _current.lstrip('-').isdigit():",
        "                _cursor = _end",
        "                continue",
        "            _cursor = _end",
        "            _hits += 1",
        "            if _current != str(_value):",
        "                _command = _command[:_start] + str(_value) + _command[_end:]",
        "                _commands[_index] = _command",
        "                _changed.append(f'{_name}: {_current} -> {_value}')",
        "            break",
        "    if not _hits:",
        "        raise RuntimeError(f'port: {_name} assignment not found in setup_commands.json')",
        "print('port: server overrides ->', _changed or 'none', flush=True)",
    ]
    if arms:
        lines += [
            "_arms = " + repr(arms),
            "os.environ.update(_arms)",
            "print('port: armed env ->', json.dumps(_arms, sort_keys=True), flush=True)",
        ]
    if env_overrides:
        lines += [
            "# The analyzer's sampling and context knobs are not exported by the launcher: they live",
            "# as string values in the `setup_env` dict that the setup heredoc writes into",
            # taaf_setup_env.json, and that file is re-merged over the notebook's environment after
            # every setup command, so os.environ arms (see --arm) are silently ignored for them.
            # They can only be changed at the source.
            "_env_overrides = " + repr(env_overrides),
            "_env_changed = []",
            "for _name, _value in _env_overrides.items():",
            "    _pattern = re.compile(",
            "        r\"^[ \\t]*'\" + re.escape(_name) + r\"':[ \\t]*.*?[ \\t]*,?[ \\t]*$\", re.MULTILINE",
            "    )",
            "    for _index, _command in enumerate(_commands):",
            "        _found = _pattern.search(_command)",
            "        if not _found:",
            "            continue",
            "        _line = f\"    '{_name}': {_value!r},\"",
            "        if _found.group(0).rstrip() != _line.rstrip():",
            "            _commands[_index] = _command[: _found.start()] + _line + _command[_found.end() :]",
            "            _env_changed.append(",
            "                f\"{_name}: {_found.group(0).split(':', 1)[1].strip().rstrip(',')} -> {_value}\"",
            "            )",
            "        break",
            "    if not any(_pattern.search(_c) for _c in _commands):",
            "        raise RuntimeError(f\"port: {_name} not found in setup_commands.json\")",
            "print(\"port: analyzer env ->\", _env_changed or \"none\", flush=True)",
        ]
    lines += [
        "_setup_path.write_text(json.dumps(_commands), encoding='utf-8')",
        "",
        "# Patch the harness source in the fork. This is why the fork exists as much as the",
        "# server budget is: the .pth the launcher writes points Python at BUNDLE_DIR/src, so a",
        "# patched file here is the file the run actually imports.",
    ]
    if audit_s:
        lines += [
            f"_audit = BUNDLE_DIR / {AUDIT_FILE!r}",
            "_audit_text = _audit.read_text(encoding='utf-8')",
            f"_audit_old = {AUDIT_ANCHOR!r}",
            f"_audit_new = {audit_replacement(audit_s)!r}",
            "if _audit_old not in _audit_text:",
            "    raise RuntimeError('port: strategy-audit anchor missing from this bundle')",
            "_audit.write_text(_audit_text.replace(_audit_old, _audit_new, 1), encoding='utf-8')",
            f"print(f'port: strategy audit interval capped at {audit_s}s', flush=True)",
            "",
        ]
    lines += [
        "os.environ['TAAF_KAGGLE_BUNDLE_DIR'] = str(BUNDLE_DIR)",
        "",
        "# Re-render the paths the launcher published before the fork existed.",
        "_published = json.loads(os.environ['TAAF_KAGGLE_INPUT_PATHS'])",
        "_published = {k: (str(BUNDLE_DIR) if v == str(_upstream_bundle) else v) for k, v in _published.items()}",
        "os.environ['TAAF_KAGGLE_INPUT_PATHS'] = json.dumps(_published, sort_keys=True)",
        "_write_setup_env_updates({'TAAF_KAGGLE_INPUT_PATHS': os.environ['TAAF_KAGGLE_INPUT_PATHS']})",
    ]
    return "\n".join(lines) + "\n"


def hook_cell(
    scored: tuple[int, int] | None,
    offline: tuple[int, int, int],
    scored_soft_s: int = 0,
    offline_soft_s: int = 0,
) -> str:
    lines = [
        "# Inline customization hook (rewritten by scripts/build_vllm_port.py).",
    ]
    if scored:
        concurrency, cap = scored
        lines += [
            "if run_as_submission:",
            "    # Scored branch: the only cell the hidden rerun executes.",
            f"    bm.solver.concurrency = {concurrency}",
            f"    bm.solver.max_runtime_s_per_game = {cap}.0",
            "    bm.n_passes = 1",
            "    print(f'port: scored schedule = {bm.solver.concurrency} streams x "
            "{bm.solver.max_runtime_s_per_game:.0f}s', flush=True)",
        ]
        if scored_soft_s:
            lines += [
                "    # The bundle disables its own soft deadline under a competition rerun, which leaves",
                "    # only Kaggle's hard stop. Re-arm it: the runner cancels the solver task gracefully,",
                "    # so results reach submission.parquet instead of dying mid-write with the session.",
                f"    soft_end = datetime.fromtimestamp(NOTEBOOK_START_EPOCH) + timedelta(seconds={scored_soft_s})",
                "    print(f'port: session soft end = {soft_end}', flush=True)",
            ]
        lines += ["else:"]
    games, cap, passes = offline
    indent = "    " if scored else ""
    lines += [
        f"{indent}# Offline branch: every interactive session. It is the rehearsal for the scored",
        f"{indent}# schedule -- same server, deep client queue, bounded wall time -- so a test costs",
        f"{indent}# model load plus one wave instead of a nine-hour run.",
        f"{indent}bm.label = bm.label + '-trimmed'",
        f"{indent}bm.n_passes = {passes}",
        f"{indent}bm.solver.concurrency = {games}",
        f"{indent}bm.solver.max_runtime_s_per_game = {cap}.0",
        f"{indent}bm.games = list(bm.games)[:{games}]",
        f"{indent}# The bundle prints its own pickled solver during setup, before this cell runs, so",
        f"{indent}# print the live one too: without this line a log cannot show whether the hook",
        f"{indent}# actually moved the schedule or the run used the baked 14 x 3918.",
        f"{indent}print(f'port: offline schedule = {{bm.solver.concurrency}} streams x "
        f"{{bm.solver.max_runtime_s_per_game:.0f}}s on {{len(bm.games)}} games', flush=True)",
    ]
    if offline_soft_s:
        lines += [
            f"{indent}# Cold nodes pay ~2000s to mount the model; the deploy target's own budget would",
            f"{indent}# cancel every game on arrival, so widen it or a validation run proves nothing.",
            f"{indent}soft_end = datetime.fromtimestamp(NOTEBOOK_START_EPOCH) + timedelta(seconds={offline_soft_s})",
        ]
    return "\n".join(lines) + "\n"


def cap_for(concurrency: int, load_s: int) -> int:
    waves = math.ceil(GAMES / concurrency)
    return (SESSION_S - load_s) // waves


def staged_notebook(args: argparse.Namespace) -> dict:
    nb = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = nb["cells"]

    def index_of(marker: str, cell_type: str = "code") -> int:
        return next(
            i
            for i, cell in enumerate(cells)
            if cell["cell_type"] == cell_type and marker in "".join(cell["source"])
        )

    hook = index_of("# Inline customization hook.")
    setup = index_of('_run_shell_commands("setup_commands.json"')
    if hook < setup:
        raise RuntimeError("expected the customization hook to follow the setup commands")

    server = dict(DEFAULT_SERVER_OVERRIDES)
    for item in args.server or []:
        name, _, value = item.partition("=")
        server[name.strip()] = int(value)
    arms = {}
    for item in args.arm or []:
        name, _, value = item.partition("=")
        arms[name.strip()] = value
    env_overrides = {}
    for item in args.env or []:
        name, _, value = item.partition("=")
        env_overrides[name.strip()] = value

    scored = None
    if args.scored_concurrency:
        scored = (
            args.scored_concurrency,
            args.scored_cap
            or cap_for(args.scored_concurrency, args.scored_load_s),
        )

    def as_lines(text: str) -> list[str]:
        return [f"{line}\n" for line in text.rstrip("\n").split("\n")]

    cells[hook]["source"] = as_lines(
        hook_cell(
            scored,
            (args.offline_games, args.offline_cap_s, args.offline_passes),
            args.scored_soft_end_s,
            args.offline_soft_end_s,
        )
    )
    cells[hook]["outputs"] = []
    cells[hook]["execution_count"] = None
    cells.insert(setup, {"cell_type": "code", "metadata": {}, "execution_count": None,
                         "outputs": [], "source": as_lines(
                             fork_cell(server, arms, args.audit_interval_s, env_overrides))})
    return nb


def metadata() -> dict:
    return {
        "id": f"{OWNER}/{SLUG}",
        "title": TITLE,
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "machine_shape": "NvidiaRtxPro6000",
        "dataset_sources": [
            "lordhansolo/taaf-kaggle-source",
            "lordhansolo/vllm-main-e975732-arc3",
        ],
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "kernel_sources": [],
        "model_sources": [
            "lordhansolo/qwen3-8-flash-next-mixed-nvfp4-fp8/PyTorch/hf-mixed-mtp-nvfp4/1"
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", action="append", help="override a baked constant, NAME=VALUE")
    ap.add_argument("--arm", action="append", help="export env NAME=VALUE before the server starts")
    ap.add_argument(
        "--env",
        action="append",
        metavar="NAME=VALUE",
        help="rewrite the quoted setup_env entry NAME (analyzer knobs the launcher re-merges last)",
    )
    ap.add_argument(
        "--audit-interval-s",
        type=int,
        default=0,
        help="cap the harness strategy-audit interval in seconds (0 leaves it at 25%% of the game cap)",
    )
    ap.add_argument("--scored-concurrency", type=int, default=0)
    ap.add_argument("--scored-cap", type=int, default=0, help="default: derived from --scored-load-s")
    ap.add_argument("--scored-load-s", type=int, default=1500, help="setup+serve time measured on Kaggle")
    ap.add_argument(
        "--scored-soft-end-s",
        type=int,
        default=31200,
        help="graceful session cancel for the scored branch; 0 keeps the bundle's none",
    )
    ap.add_argument(
        "--offline-soft-end-s",
        type=int,
        default=10800,
        help="session cancel for offline runs; must clear the cold model mount or every game cancels on arrival",
    )
    ap.add_argument("--offline-games", type=int, default=6)
    ap.add_argument("--offline-cap-s", type=int, default=900)
    ap.add_argument("--offline-passes", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.scored_concurrency and not 1 <= args.scored_concurrency <= GAMES:
        ap.error(f"--scored-concurrency must be within 1..{GAMES}")
    # A soft end at or past the hard stop is not a safety net, it is the same wall the run is
    # trying to stay inside of -- the parquet write needs the difference.
    for name in ("scored_soft_end_s", "offline_soft_end_s"):
        value = getattr(args, name)
        if value and value >= SESSION_S:
            ap.error(f"--{name.replace('_', '-')} must be below the {SESSION_S}s session limit")

    nb = staged_notebook(args)
    text = json.dumps(nb, ensure_ascii=False, indent=1)
    json.loads(text)  # the push endpoint rejects a malformed notebook
    if not args.dry_run:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        NOTEBOOK.write_text(text, encoding="utf-8")
        (OUT_DIR / "kernel-metadata.json").write_text(
            json.dumps(metadata(), indent=2) + "\n", encoding="utf-8"
        )
    scored_note = (
        f"scored {args.scored_concurrency} x {args.scored_cap or cap_for(args.scored_concurrency, args.scored_load_s)}s"
        if args.scored_concurrency
        else "scored branch untouched (bundle default 14 x 3918)"
    )
    print(f"{NOTEBOOK.name}: {len(text)} bytes, {len(nb['cells'])} cells, {scored_note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
