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
# Empty means "omit -v", which this competition rejects: "Code competition submissions
# require both the output file name and the version number". Pass the version the push
# printed; it is only empty so that a stale caller fails loudly instead of silently
# submitting an old version.
VERSION = ""
POLL_S = 90
# Kaggle only accepts a code submission once that kernel version has produced an output, and
# a push starts a session immediately, so "COMPLETE" can still be a version behind. The submit
# itself is the authoritative check and a rejected submit costs no slot, so retry it.
SUBMIT_RETRY_S = 120
SUBMIT_RETRIES = 15
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
    if not version:
        print("No kernel version: this competition rejects a version-less submit. Pass it as argv[2].")
        return 1
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
        os.environ.get("SUBMISSION_PARQUET", "submission.parquet"),
        "-m",
        message,
    ]
    submit += ["-v", str(version)]
    for attempt in range(1, SUBMIT_RETRIES + 1):
        print("$", " ".join(submit))
        out = run(submit)
        print(out.strip())
        if "remaining" in out or "Successfully" in out:
            break
        print(f"attempt {attempt} of {SUBMIT_RETRIES} not accepted; waiting {SUBMIT_RETRY_S}s")
        time.sleep(SUBMIT_RETRY_S)
    print(run(["kaggle", "competitions", "submissions", "-c", COMP]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
