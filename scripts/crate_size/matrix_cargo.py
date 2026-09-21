#!/usr/bin/env python3
"""Cargo shim used by cargo-matrix to record each linked executable size."""
from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
from pathlib import Path


def selected_features(args: list[str]) -> set[str]:
    selected: set[str] = set()
    for i, arg in enumerate(args):
        if arg in ("-F", "--features") and i + 1 < len(args):
            selected.update(args[i + 1].replace(" ", ",").split(","))
        elif arg.startswith("--features="):
            selected.update(arg.split("=", 1)[1].replace(" ", ",").split(","))
    selected.discard("")
    return selected


def main() -> None:
    args = sys.argv[1:]
    cargo = os.environ["MATRIX_REAL_CARGO"]
    toolchain = os.environ["MATRIX_TOOLCHAIN"]
    result = subprocess.run([cargo, f"+{toolchain}", *args])
    if result.returncode:
        raise SystemExit(result.returncode)
    if not args or args[0] != "build":
        return

    binary = Path(os.environ["MATRIX_BINARY"])
    subprocess.run([str(binary)], check=True, stdout=subprocess.DEVNULL)
    options = os.environ["MATRIX_FEATURE_OPTIONS"].split(",")
    selected = selected_features(args)
    enabled = [feature for feature in options if feature in selected]
    row = {
        "key": "+".join(enabled) or "bpe",
        "bytes": len(gzip.compress(binary.read_bytes(), compresslevel=9, mtime=0)),
    }
    with Path(os.environ["MATRIX_RESULTS"]).open("a") as output:
        output.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
