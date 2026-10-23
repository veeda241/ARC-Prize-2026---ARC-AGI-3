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

# The bundle ships one giant heredoc whose constants are comments-free assignments. Anchoring on
# the assignment text means a missing knob fails at staging time rather than as a silent
# server-default at 2048 tokens. Default is the leader's own stock budget, so a run measures the
# published configuration; pass --server NAME=VALUE to move off it.
DEFAULT_SERVER_OVERRIDES: dict[str, int] = {}


def fork_cell(server: dict[str, int], arms: dict[str, str]) -> str:
    lines = [
        "# Fork of the read-only bundle, inserted by scripts/build_vllm_port.py.",
        "# The bundle's setup script bakes its vLLM server arguments at dataset-build time, so the",
        "# serving budget cannot be changed from the notebook unless the setup commands themselves",
        "# are edited -- which needs a writable copy, because /kaggle/input is read-only.",
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
        "_setup_path = BUNDLE_DIR / 'setup_commands.json'",
        "_commands = json.loads(_setup_path.read_text(encoding='utf-8'))",
        "_commands = [_c.replace('\"$PYTHON\" - <<', f'{_py312} - <<', 1)",
        "               if _c.lstrip().startswith('\"$PYTHON\" - <<') else _c for _c in _commands]",
        "_server_overrides = " + repr(server),
        "_changed = []",
        "for _index, _command in enumerate(_commands):",
        "    for _name, _value in _server_overrides.items():",
        "        _anchor = f'{_name} = '",
        "        _found = False",
        "        _cursor = 0",
        "        while not _found:",
        "            _start = _command.find(_anchor, _cursor)",
        "            if _start < 0:",
        "                break",
        "            _start += len(_anchor)",
        "            _end = _command.index(chr(10), _start)",
        "            _current = _command[_start:_end].strip()",
        "            if _current.lstrip('-').isdigit():",
        "                _cursor = _end",
        "                if _current != str(_value):",
        "                    _command = _command[:_start] + str(_value) + _command[_end:]",
        "                    _changed.append(f'{_name}: {_current} -> {_value}')",
        "                _found = True",
        "            else:",
        "                _cursor = _end",
        "        if not _found:",
        "            raise RuntimeError(f'port: {_name} assignment not found in setup_commands.json')",
        "    _commands[_index] = _command",
        "print('port: server overrides ->', _changed or 'none', flush=True)",
    ]
    if arms:
        lines += [
            "_arms = " + repr(arms),
            "os.environ.update(_arms)",
            "print('port: armed env ->', json.dumps(_arms, sort_keys=True), flush=True)",
        ]
    lines += [
        "_setup_path.write_text(json.dumps(_commands), encoding='utf-8')",
        "",
        "os.environ['TAAF_KAGGLE_BUNDLE_DIR'] = str(BUNDLE_DIR)",
        "",
        "# Re-render the paths the launcher published before the fork existed.",
        "_published = json.loads(os.environ['TAAF_KAGGLE_INPUT_PATHS'])",
        "_published = {k: (str(BUNDLE_DIR) if v == str(_upstream_bundle) else v) for k, v in _published.items()}",
        "os.environ['TAAF_KAGGLE_INPUT_PATHS'] = json.dumps(_published, sort_keys=True)",
        "_write_setup_env_updates({'TAAF_KAGGLE_INPUT_PATHS': os.environ['TAAF_KAGGLE_INPUT_PATHS']})",
    ]
    return "\n".join(lines) + "\n"


def hook_cell(scored: tuple[int, int] | None, offline: tuple[int, int, int]) -> str:
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
            "else:",
        ]
    games, cap, passes = offline
    indent = "    " if scored else ""
    lines += [
        f"{indent}# Offline branch: every interactive session. Trimmed so a test costs model load plus",
        f"{indent}# a few games instead of a full demo run.",
        f"{indent}bm.label = bm.label + '-trimmed'",
        f"{indent}bm.n_passes = {passes}",
        f"{indent}bm.solver.concurrency = {games}",
        f"{indent}bm.solver.max_runtime_s_per_game = {cap}.0",
        f"{indent}bm.games = list(bm.games)[:{games}]",
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
        hook_cell(scored, (args.offline_games, args.offline_cap_s, args.offline_passes))
    )
    cells[hook]["outputs"] = []
    cells[hook]["execution_count"] = None
    cells.insert(setup, {"cell_type": "code", "metadata": {}, "execution_count": None,
                         "outputs": [], "source": as_lines(fork_cell(server, arms))})
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
    ap.add_argument("--scored-concurrency", type=int, default=0)
    ap.add_argument("--scored-cap", type=int, default=0, help="default: derived from --scored-load-s")
    ap.add_argument("--scored-load-s", type=int, default=1500, help="setup+serve time measured on Kaggle")
    ap.add_argument("--offline-games", type=int, default=6)
    ap.add_argument("--offline-cap-s", type=int, default=900)
    ap.add_argument("--offline-passes", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.scored_concurrency and not 1 <= args.scored_concurrency <= GAMES:
        ap.error(f"--scored-concurrency must be within 1..{GAMES}")

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
