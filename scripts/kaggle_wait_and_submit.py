#!/usr/bin/env python3
"""Wait for the Kaggle kernel to finish, then submit it to ARC-AGI-3."""
from __future__ import annotations

import subprocess
import sys
import time

KERNEL = "vyassenthilkumar/arc3-duck-v12-boost"
COMP = "arc-prize-2026-arc-agi-3"
VERSION = "1"
MESSAGE = "Duck v12 boost, uncapped actions, RTX Pro 6000"
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
        upper = status.upper()
        if "COMPLETE" in upper:
            break
        if any(x in upper for x in ("ERROR", "CANCELLED", "FAILED")):
            print("Kernel did not complete successfully; not submitting.")
            return 1
        if time.time() - start > MAX_WAIT_S:
            print("Timed out waiting for kernel.")
            return 1
        time.sleep(POLL_S)

    submit = run(
        [
            "kaggle",
            "competitions",
            "submit",
            COMP,
            "-k",
            kernel,
            "-v",
            str(version),
            "-f",
            "submission.parquet",
            "-m",
            message,
        ]
    )
    print(submit)
    print(run(["kaggle", "competitions", "submissions", "-c", COMP]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
