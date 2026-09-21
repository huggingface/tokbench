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


def mac_hardware() -> str | None:
    """Return useful Mac hardware fields without device identifiers."""
    raw = command("system_profiler", "SPHardwareDataType", "-json")
    if raw is None:
        return None
    try:
        hardware = json.loads(raw)["SPHardwareDataType"][0]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return None
    allowed = (
        "machine_name",
        "machine_model",
        "chip_type",
        "number_processors",
        "physical_memory",
    )
    return json.dumps(
        {key: hardware[key] for key in allowed if key in hardware},
        sort_keys=True,
    )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: collect_environment.py OUTPUT.json")

    env_names = (
        "ACCELERATOR",
        "CARGO_BUILD_JOBS",
        "CARGO_PROFILE_RELEASE_DEBUG",
        "CC",
        "CPU_CORES",
        "CXX",
        "JOB_ID",
        "MEMORY",
        "RUSTFLAGS",
        "TOKBENCH_FEATURES",
        "TOKBENCH_GIGATOKEN_REVISION",
        "TOKBENCH_RUST_TOOLCHAIN",
        "TOKBENCH_SKIP_BUILD",
        "TOKBENCH_ENGINES",
        "TOKBENCH_MEASURE",
        "TOKBENCH_COMPARE_TO",
        "TOKBENCH_CACHE_CAPACITY",
        "TOKBENCH_IMAGE",
        "TOKBENCH_INPUT_REVISION",
        "TOKBENCH_CORPORA_REVISION",
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
        "TOKBENCH_SCALING_MODE",
        "TOKBENCH_SOURCE_REVISION",
    )
    document = {
        "environment": {name: os.environ[name] for name in env_names if name in os.environ},
        "platform": platform.platform(),
        "python": platform.python_version(),
        "commands": {
            "cargo": command("cargo", "--version"),
            "cc": command(os.environ.get("CC", "cc"), "--version"),
            "cpu": command("lscpu", "--json") or mac_hardware(),
            "cxx": command(os.environ.get("CXX", "c++"), "--version"),
            "git": command("git", "rev-parse", "HEAD"),
            "kernel": command("uname", "-a"),
            "rustc": command("rustc", "--version", "--verbose"),
            "rustc_selected": (
                command(
                    "rustc",
                    f"+{os.environ['TOKBENCH_RUST_TOOLCHAIN']}",
                    "--version",
                    "--verbose",
                )
                if "TOKBENCH_RUST_TOOLCHAIN" in os.environ
                else None
            ),
            "task_affinity": command("taskset", "-pc", str(os.getpid())),
        },
    }
    Path(sys.argv[1]).write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
