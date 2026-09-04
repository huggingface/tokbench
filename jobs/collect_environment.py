#!/usr/bin/env python3
"""Write the non-secret execution environment needed to interpret a run."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def command(*args: str) -> str | None:
    try:
        result = subprocess.run(args, check=False, capture_output=True, text=True)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: collect_environment.py OUTPUT.json")

    env_names = (
        "ACCELERATOR",
        "CPU_CORES",
        "JOB_ID",
        "MEMORY",
        "TOKBENCH_FEATURES",
        "TOKBENCH_ENGINES",
        "TOKBENCH_IMAGE",
        "TOKBENCH_INPUT_REVISION",
        "TOKBENCH_LATENCY",
        "TOKBENCH_LATENCY_BYTES",
        "TOKBENCH_LATENCY_SAMPLES",
        "TOKBENCH_MAX_THREADS",
        "TOKBENCH_MODELS",
        "TOKBENCH_NO_DECODE",
        "TOKBENCH_PINNED_CPUSET",
        "TOKBENCH_PIN_PHYSICAL_CORES",
        "TOKBENCH_PROFILE",
        "TOKBENCH_REPS",
        "TOKBENCH_RUNS",
        "TOKBENCH_SCALING",
        "TOKBENCH_SCALING_ORDER",
        "TOKBENCH_SOURCE_REVISION",
    )
    document = {
        "environment": {name: os.environ[name] for name in env_names if name in os.environ},
        "platform": platform.platform(),
        "python": platform.python_version(),
        "commands": {
            "cargo": command("cargo", "--version"),
            "cpu": command("lscpu", "--json"),
            "git": command("git", "rev-parse", "HEAD"),
            "kernel": command("uname", "-a"),
            "rustc": command("rustc", "--version", "--verbose"),
            "task_affinity": command("taskset", "-pc", str(os.getpid())),
        },
    }
    Path(sys.argv[1]).write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
