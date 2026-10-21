#!/usr/bin/env python3
"""Wait for the Kaggle kernel to finish, then submit it to ARC-AGI-3.

Submission text comes from KAGGLE_SUBMIT_MESSAGE so a detached run can be labelled without
passing an empty positional version argument (Start-Process rejects those).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time

# Kaggle derives the slug from the notebook TITLE, not the metadata "id", so these
# strings must match what `kaggle kernels list --mine` reports.
KERNEL = "vyassenthilkumar/arc3-milestone2-flashnext-duck"
COMP = "arc-prize-2026-arc-agi-3"
MESSAGE = os.environ.get(
    "KAGGLE_SUBMIT_MESSAGE", "Milestone2 flashnext + MTP drafter, RTX Pro 6000"
)
# Empty means "whatever version Kaggle has for the kernel now" — the CLI's -v is optional,
# and pinning it silently submits an old (often failed) version.
VERSION = ""
POLL_S = 90
# The run itself may use the full 9 h GPU-notebook limit, so the poll window must
# cover queue time plus a maximum-length run.
MAX_WAIT_S = 11 * 60 * 60


def run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    return out


def main() -> int:
    kernel = sys.argv[1] if len(sys.argv) > 1 else KERNEL
    version = sys.argv[2] if len(sys.argv) > 2 else VERSION
    message = sys.argv[3] if len(sys.argv) > 3 else MESSAGE
    start = time.time()
    while True:
        status = run(["kaggle", "kernels", "status", kernel])
        print(status.strip())
        # Only trust the status token. The CLI prints API exceptions to the same stream, and
        # a connection blip reads like "ConnectTimeoutError" - which contains "ERROR", so a
        # substring scan used to abort the wait and skip the submission entirely.
        match = re.search(r"KernelWorkerStatus\.([A-Z_]+)", status)
        if match:
            token = match.group(1)
            if token == "COMPLETE":
                break
            if token in {"ERROR", "CANCELLED", "FAILED"}:
                print(f"Kernel ended as {token}; not submitting.")
                return 1
        if time.time() - start > MAX_WAIT_S:
            print("Timed out waiting for kernel.")
            return 1
        time.sleep(POLL_S)

    submit = [
        "kaggle",
        "competitions",
        "submit",
        COMP,
        "-k",
        kernel,
        "-f",
        "submission.parquet",
        "-m",
        message,
    ]
    if version:
        submit += ["-v", str(version)]
    print("$", " ".join(submit))
    print(run(submit))
    print(run(["kaggle", "competitions", "submissions", "-c", COMP]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
