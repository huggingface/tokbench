#!/usr/bin/env python3
"""Prepare a disposable checkout for the pinned Gigatoken build."""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "engines" / "gigatoken" / "Cargo.toml"
LOCK_SOURCE = ROOT / "jobs" / "Cargo.gigatoken.lock"
LOCK_DESTINATION = ROOT / "Cargo.lock"
DEPENDENCY = (
    'gigatoken_rs = { package = "gigatoken", '
    'git = "https://github.com/marcelroed/gigatoken", '
    'rev = "34a1599f0c0ae7d7cd0d1c530e6522320158b360" }'
)


def main() -> None:
    text = MANIFEST.read_text()
    commented = f"# {DEPENDENCY}"
    if commented in text:
        MANIFEST.write_text(text.replace(commented, DEPENDENCY, 1))
    elif DEPENDENCY not in text:
        raise SystemExit(f"expected pinned Gigatoken dependency in {MANIFEST}")

    lock = LOCK_SOURCE.read_text()
    expected_source = (
        "git+https://github.com/marcelroed/gigatoken?"
        "rev=34a1599f0c0ae7d7cd0d1c530e6522320158b360#"
        "34a1599f0c0ae7d7cd0d1c530e6522320158b360"
    )
    if expected_source not in lock:
        raise SystemExit(f"{LOCK_SOURCE} does not pin the expected Gigatoken revision")
    shutil.copyfile(LOCK_SOURCE, LOCK_DESTINATION)


if __name__ == "__main__":
    main()
