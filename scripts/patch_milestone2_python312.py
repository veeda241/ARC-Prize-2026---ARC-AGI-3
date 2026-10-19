#!/usr/bin/env python3
"""Point the Milestone 2 SGLang venv at a Python 3.12 interpreter.

Upstream builds the venv from `sys.executable`. Kaggle's notebook image has moved to
Python 3.13, while the `dfranzen/pennyroyal-v253` wheelhouse ships cp312-only wheels, so
an unpatched copy fails at cell 6 with "aiohttp==3.14.3 has no wheels with a matching
Python ABI tag (cp313)". Nothing else in the notebook changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json

OLD = '        run(uv + ["venv", "--python", sys.executable, VENV], env=install_env)\n'

NEW = '''        # PATCH (this repo): the wheelhouse is cp312-only, the image kernel is 3.13.
        print("=== interpreter audit ===")
        run(["ls", "/usr/bin"], check=False)
        print(sys.version)
        py312 = next(
            (
                c for c in [
                    "/usr/bin/python3.12", "/usr/local/bin/python3.12",
                    shutil.which("python3.12") or "", "python3.12",
                ]
                if c and subprocess.run(
                    [c, "-c", "import sys; assert sys.version_info[:2] == (3, 12)"],
                    capture_output=True,
                ).returncode == 0
            ),
            None,
        )
        if py312 is None:
            raise RuntimeError(
                "No CPython 3.12 on this image, so the cp312 Pennyroyal wheelhouse cannot "
                f"be installed (kernel is {sys.version.split()[0]}). Either find a cp313 "
                "bundle or pin the notebook to a 3.12 image."
            )
        print(f"venv interpreter: {py312}")
        run(uv + ["venv", "--python", py312, VENV], env=install_env)
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook", type=Path)
    args = ap.parse_args()

    nb = json.loads(args.notebook.read_text(encoding="utf-8"))
    patched = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if OLD in source:
            source = source.replace(OLD, NEW)
            lines = source.splitlines(keepends=True)
            cell["source"] = lines if len(lines) > 1 else [source]
            patched += 1
    if not patched:
        text = args.notebook.read_text(encoding="utf-8")
        print("already patched" if "PATCH (this repo)" in text else "anchor line not found")
        return 0 if "PATCH (this repo)" in text else 1

    args.notebook.write_text(json.dumps(nb), encoding="utf-8")
    print(f"patched {patched} cell(s) in {args.notebook.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
