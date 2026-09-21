#!/usr/bin/env python3
"""Measure the linked size of tokenizers v1 crate and feature selections."""
from __future__ import annotations

import argparse
import gzip
import itertools
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CURRENT_REVISION = "484be1d3351077fc5c0b81e876c94814c08ef3eb"
INITIAL_REVISION = "0e9bb9ca3bfebcc3892173a788d5fcda9e7bf2cd"
STABLE = "1.97.1"
NIGHTLY = "nightly-2026-08-05"
CRATES = ("serialize", "convert", "train")
FEATURES = ("unigram", "wordpiece", "wordlevel", "normalizers",
            "unicode-scripts", "parallelism")


def run(command: list[object], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run([str(part) for part in command], check=True, **kwargs)


def gz_size(path: Path) -> int:
    return len(gzip.compress(path.read_bytes(), compresslevel=9, mtime=0))


def host_triple() -> str:
    output = subprocess.check_output(["rustc", f"+{STABLE}", "-vV"], text=True)
    return next(line.split(":", 1)[1].strip() for line in output.splitlines()
                if line.startswith("host:"))


def platform_name() -> str:
    cpu = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        try:
            cpu = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True,
                stderr=subprocess.DEVNULL).strip()
        except subprocess.CalledProcessError:
            pass
    return f"{cpu}, {host_triple()}"


def selected_features(args: list[str]) -> set[str]:
    selected: set[str] = set()
    for index, arg in enumerate(args):
        if arg in ("-F", "--features") and index + 1 < len(args):
            selected.update(args[index + 1].replace(" ", ",").split(","))
        elif arg.startswith("--features="):
            selected.update(arg.split("=", 1)[1].replace(" ", ",").split(","))
    return selected


def matrix_cargo() -> None:
    """Act as Cargo for cargo-matrix and record the resulting executable."""
    args = sys.argv[1:]
    result = subprocess.run([
        os.environ["MATRIX_REAL_CARGO"], f'+{os.environ["MATRIX_TOOLCHAIN"]}', *args
    ])
    if result.returncode:
        raise SystemExit(result.returncode)
    if not args or args[0] != "build":
        return
    binary = Path(os.environ["MATRIX_BINARY"])
    run([binary], stdout=subprocess.DEVNULL)
    enabled = [name for name in FEATURES if f"crate-{name}" in selected_features(args)]
    row = {"key": "+".join(enabled) or "bpe", "bytes": gz_size(binary)}
    with Path(os.environ["MATRIX_RESULTS"]).open("a") as output:
        output.write(json.dumps(row) + "\n")


def download(probe: Path, revision: str) -> Path:
    archive = probe / "initial.tar.gz"
    urllib.request.urlretrieve(
        f"https://github.com/huggingface/tokenizers/archive/{revision}.tar.gz", archive)
    with tarfile.open(archive) as source:
        if sys.version_info >= (3, 12):
            source.extractall(probe, filter="data")
        else:
            source.extractall(probe)
    return probe / f"tokenizers-{revision}" / "tokenizers"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("crate_sizes.json"))
    args = parser.parse_args()

    matrix = shutil.which("cargo-matrix")
    if not matrix:
        raise SystemExit("install cargo-matrix 0.4.5 with: cargo install cargo-matrix --version 0.4.5 --locked")
    version_run = subprocess.run([matrix, "matrix", "--version"], text=True,
                                 capture_output=True)
    matrix_version = (version_run.stdout or version_run.stderr).strip()
    if matrix_version != "cargo-matrix 0.4.5":
        raise SystemExit(f"expected cargo-matrix 0.4.5, found {matrix_version!r}")

    with tempfile.TemporaryDirectory(prefix="tokbench-crate-size-") as raw:
        temporary = Path(raw)
        target = temporary / "target"
        env = {**os.environ, "CARGO_TARGET_DIR": str(target)}
        cargo = shutil.which("cargo") or "cargo"

        initial_root = download(temporary, INITIAL_REVISION)
        with (initial_root / "Cargo.toml").open("a") as manifest:
            manifest.write('\n[profile.minsize]\ninherits="release"\nopt-level="z"\n'
                           'codegen-units=1\npanic="abort"\nlto="fat"\nstrip=true\n')
        initial_target = temporary / "initial-target"
        initial_env = {**env, "CARGO_TARGET_DIR": str(initial_target)}
        run([cargo, f"+{STABLE}", "build", "--quiet", "--manifest-path",
             initial_root / "Cargo.toml", "--profile", "minsize", "-p", "tk-encode",
             "--example", "binsize_pipeline", "--locked", "--no-default-features"],
            env=initial_env)
        initial_binary = initial_target / "minsize/examples/binsize_pipeline"
        run([initial_binary, args.model.resolve(), "hello world"],
            stdout=subprocess.DEVNULL)
        baseline = gz_size(initial_binary)
        print(f"{'before crate split':36} {baseline:9,d} B")

        def build(profile: str, features: list[str], toolchain: str = STABLE,
                  extra: tuple[str, ...] = (), extra_env: dict[str, str] | None = None) -> Path:
            command = [cargo, f"+{toolchain}", "build", "--quiet", "--locked",
                       "--manifest-path", ROOT / "Cargo.toml", "--profile", profile,
                       "-p", "tokbench-binsize", "--bin", "tokenizers-crate-size",
                       "--no-default-features", "--features", ",".join(features), *extra]
            run(command, cwd=ROOT, env={**env, **(extra_env or {})})
            if "--target" in extra:
                return target / extra[extra.index("--target") + 1] / profile / "tokenizers-crate-size"
            return target / profile / "tokenizers-crate-size"

        configs = {}
        for enabled in itertools.product((False, True), repeat=len(CRATES)):
            selected = [name for name, on in zip(CRATES, enabled) if on]
            binary = build("minsize", ["crate-bpe", *[f"crate-{x}" for x in selected]])
            run([binary, args.model.resolve()], stdout=subprocess.DEVNULL)
            key = "+".join(selected) or "encode"
            configs[key] = gz_size(binary)
            print(f"{key:36} {configs[key]:9,d} B")

        def feature_matrix(profile: str, toolchain: str, binary: Path,
                           extra: tuple[str, ...] = (), flags: str | None = None):
            results = temporary / f"matrix-{profile}.jsonl"
            matrix_env = {**env, "TOKBENCH_MATRIX_CARGO": "1", "CARGO": str(Path(__file__).resolve()),
                "MATRIX_REAL_CARGO": cargo, "MATRIX_TOOLCHAIN": toolchain,
                "MATRIX_RESULTS": str(results), "MATRIX_BINARY": str(binary)}
            if flags:
                matrix_env["RUSTFLAGS"] = flags
            run([matrix, "matrix", "--channel", "crate-sizes", "--package",
                 "tokbench-binsize", "--manifest-path", ROOT / "Cargo.toml", "build",
                 "--quiet", "--profile", profile, "--locked", "--bin",
                 "tokenizers-crate-size", *extra], cwd=ROOT, env=matrix_env)
            rows = [json.loads(line) for line in results.read_text().splitlines()]
            values = {row["key"]: row["bytes"] for row in rows}
            if len(rows) != 64 or len(values) != 64:
                raise SystemExit(f"incomplete {profile} feature matrix")
            return values

        profiles = {"minsize": feature_matrix(
            "minsize", STABLE, target / "minsize/tokenizers-crate-size")}
        sysroot = subprocess.run(["rustc", f"+{NIGHTLY}", "--print", "sysroot"],
                                 text=True, capture_output=True)
        if sysroot.returncode or not (Path(sysroot.stdout.strip()) /
                                     "lib/rustlib/src/rust/library").is_dir():
            raise SystemExit(f"install {NIGHTLY} with rust-src to measure slimest")
        host = host_triple()
        profiles["slimest"] = feature_matrix(
            "slimest", NIGHTLY, target / host / "slimest/tokenizers-crate-size",
            ("-Z", "build-std=std,panic_abort", "--target", host),
            "-Zunstable-options -Cpanic=immediate-abort")

    result = {
        "unit": "gzipped executable bytes", "profile": "minsize, stripped, gzip -9",
        "platform": platform_name(),
        "rustc": subprocess.check_output(["rustc", f"+{STABLE}", "-V"], text=True).strip(),
        "tokenizers_revision": CURRENT_REVISION, "initial_revision": INITIAL_REVISION,
        "baseline": {"name": "tokenizers before the split", "bytes": baseline},
        "required": "tk-encode", "options": list(CRATES), "configs": configs,
        "feature_required": "bpe", "feature_options": list(FEATURES),
        "feature_profiles": profiles, "matrix_tool": matrix_version,
        "absolute_minimum": {"bytes": profiles["slimest"]["bpe"],
            "profile": "slimest, rebuilt std with panic_immediate_abort",
            "rustc": subprocess.check_output(["rustc", f"+{NIGHTLY}", "-V"],
                                               text=True).strip()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    if os.environ.get("TOKBENCH_MATRIX_CARGO") == "1":
        matrix_cargo()
    else:
        main()
