#!/usr/bin/env python3
"""Cargo shim used by cargo-matrix to record each linked executable size."""
import gzip
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    args = sys.argv[1:]
    result = subprocess.run([
        os.environ["MATRIX_REAL_CARGO"], f'+{os.environ["MATRIX_TOOLCHAIN"]}', *args
    ])
    if result.returncode:
        raise SystemExit(result.returncode)
    if not args or args[0] != "build":
        return
    selected = set()
    for i, arg in enumerate(args):
        if arg in ("-F", "--features") and i + 1 < len(args):
            selected.update(args[i + 1].replace(" ", ",").split(","))
        elif arg.startswith("--features="):
            selected.update(arg.split("=", 1)[1].replace(" ", ",").split(","))
    binary = Path(os.environ["MATRIX_BINARY"])
    subprocess.run([binary], check=True, stdout=subprocess.DEVNULL)
    enabled = [f for f in os.environ["MATRIX_FEATURE_OPTIONS"].split(",") if f in selected]
    row = {"key": "+".join(enabled) or "bpe",
           "bytes": len(gzip.compress(binary.read_bytes(), compresslevel=9, mtime=0))}
    with Path(os.environ["MATRIX_RESULTS"]).open("a") as output:
        output.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
