#!/usr/bin/env python3
"""Measure tokenizers v1's linked crate and feature configurations."""
from __future__ import annotations

import argparse
import gzip
import hashlib
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

HERE = Path(__file__).resolve().parent
RC0_SHA = "484be1d3351077fc5c0b81e876c94814c08ef3eb"
INITIAL_SHA = "0e9bb9ca3bfebcc3892173a788d5fcda9e7bf2cd"
STABLE = "1.97.1"
NIGHTLY = "nightly-2026-08-05"
CRATES = ("serialize", "convert", "train")
FEATURES = ("unigram", "wordpiece", "wordlevel", "normalizers",
            "unicode-scripts", "parallelism")


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=True, **kwargs)


def gz_size(path: Path) -> int:
    return len(gzip.compress(path.read_bytes(), compresslevel=9, mtime=0))


def download(probe: Path, revision: str, name: str) -> Path:
    archive = probe / f"{name}.tar.gz"
    urllib.request.urlretrieve(
        f"https://github.com/huggingface/tokenizers/archive/{revision}.tar.gz", archive)
    with tarfile.open(archive) as tar:
        if sys.version_info >= (3, 12):
            tar.extractall(probe, filter="data")
        else:
            tar.extractall(probe)
    return probe / f"tokenizers-{revision}" / "tokenizers"


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizers", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default="crate_sizes.json")
    args = parser.parse_args()

    matrix = shutil.which("cargo-matrix")
    if not matrix:
        raise SystemExit("install cargo-matrix 0.4.5 with: cargo install cargo-matrix --version 0.4.5 --locked")
    version = subprocess.run([matrix, "matrix", "--version"], text=True,
                             capture_output=True)
    matrix_version = (version.stdout or version.stderr).strip()
    if matrix_version != "cargo-matrix 0.4.5":
        raise SystemExit(f"expected cargo-matrix 0.4.5, found {matrix_version!r}")

    with tempfile.TemporaryDirectory(prefix="tokbench-crate-size-") as raw:
        probe = Path(raw)
        if args.tokenizers:
            checkout = args.tokenizers.resolve()
            root = checkout / "tokenizers"
            if not (root / "tk-encode").is_dir():
                root = checkout
                checkout = root.parent
            if not (root / "tk-encode").is_dir():
                raise SystemExit(f"no tokenizers Rust workspace under {args.tokenizers}")
            revision = subprocess.check_output(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
        else:
            root = download(probe, RC0_SHA, "rc0")
            revision = RC0_SHA
        initial_root = download(probe, INITIAL_SHA, "initial")

        manifest = (HERE / "Cargo.toml.in").read_text().replace(
            "__TOKENIZERS__", root.as_posix())
        (probe / "Cargo.toml").write_text(manifest)
        (probe / "src").mkdir()
        shutil.copy2(HERE / "main.rs", probe / "src/main.rs")
        shutil.copy2(HERE / "feature_main.rs", probe / "src/feature_main.rs")
        shim = probe / "matrix_cargo.py"
        shutil.copy2(HERE / "matrix_cargo.py", shim)
        shim.chmod(0o755)
        env = {**os.environ, "CARGO_HOME": str(probe / "cargo-home"),
               "CARGO_TARGET_DIR": str(probe / "target")}
        run(["cargo", f"+{STABLE}", "generate-lockfile"], cwd=probe, env=env)
        lock_sha = hashlib.sha256((probe / "Cargo.lock").read_bytes()).hexdigest()

        canonical = probe / "canonical.json"
        canonical.write_text('{"version":"2.0","added_tokens":[],"normalizer":null,'
            '"pre_tokenizer":null,"post_processor":null,"decoder":null,'
            '"model":{"type":"BPE","byte_level":false,'
            '"vocab":{"a":0,"b":1,"ab":2,"abab":3},'
            '"merges":[["a","b"],["ab","ab"]]},"padding":null}')

        def measure(features: tuple[str, ...]) -> int:
            command = ["cargo", f"+{STABLE}", "build", "--quiet", "--profile",
                "minsize", "--locked", "--no-default-features", "--bin",
                "tokenizers-crate-size-probe"]
            if features:
                command += ["--features", ",".join(features)]
            run(command, cwd=probe, env=env)
            binary = probe / "target/minsize/tokenizers-crate-size-probe"
            run([binary, canonical], stdout=subprocess.DEVNULL)
            size = gz_size(binary)
            print(f"{'+'.join(features) or 'encode':36} {size:9,d} B")
            return size

        with (initial_root / "Cargo.toml").open("a") as output:
            output.write('\n[profile.minsize]\ninherits="release"\nopt-level="z"\n'
                         'codegen-units=1\npanic="abort"\nlto="fat"\nstrip=true\n')
        initial_target = probe / "initial-target"
        initial_env = {**env, "CARGO_TARGET_DIR": str(initial_target)}
        run(["cargo", f"+{STABLE}", "build", "--quiet", "--manifest-path",
             initial_root / "Cargo.toml", "--profile", "minsize", "-p", "tk-encode",
             "--example", "binsize_pipeline", "--locked", "--no-default-features"],
            env=initial_env)
        initial_binary = initial_target / "minsize/examples/binsize_pipeline"
        run([initial_binary, args.model.resolve(), "hello world"], stdout=subprocess.DEVNULL)
        old = gz_size(initial_binary)

        configs = {}
        for enabled in itertools.product((False, True), repeat=len(CRATES)):
            selected = tuple(name for name, on in zip(CRATES, enabled) if on)
            configs["+".join(selected) or "encode"] = measure(selected)

        def feature_matrix(profile: str, toolchain: str, binary: Path,
                           extra: tuple[str, ...] = (), flags: str | None = None):
            results = probe / f"matrix-{profile}.jsonl"
            matrix_env = {**env, "CARGO": str(shim),
                "MATRIX_REAL_CARGO": shutil.which("cargo") or "cargo",
                "MATRIX_TOOLCHAIN": toolchain, "MATRIX_RESULTS": str(results),
                "MATRIX_FEATURE_OPTIONS": ",".join(FEATURES),
                "MATRIX_BINARY": str(binary)}
            if flags:
                matrix_env["RUSTFLAGS"] = flags
            run([matrix, "matrix", "--channel", "sizes", "--package",
                 "tokenizers-crate-size-probe", "--manifest-path", probe / "Cargo.toml",
                 "build", "--quiet", "--profile", profile, "--locked", "--bin",
                 "tokenizers-feature-size-probe", *extra], cwd=probe, env=matrix_env)
            rows = [json.loads(line) for line in results.read_text().splitlines()]
            values = {row["key"]: row["bytes"] for row in rows}
            if len(rows) != 64 or len(values) != 64:
                raise SystemExit(f"incomplete {profile} feature matrix")
            return values

        profiles = {"minsize": feature_matrix(
            "minsize", STABLE, probe / "target/minsize/tokenizers-feature-size-probe")}
        sysroot = subprocess.run(["rustc", f"+{NIGHTLY}", "--print", "sysroot"],
                                 text=True, capture_output=True)
        if sysroot.returncode or not (Path(sysroot.stdout.strip()) /
                                     "lib/rustlib/src/rust/library").is_dir():
            raise SystemExit(f"install {NIGHTLY} with rust-src to measure slimest")
        host = host_triple()
        profiles["slimest"] = feature_matrix(
            "slimest", NIGHTLY,
            probe / f"target/{host}/slimest/tokenizers-feature-size-probe",
            ("-Z", "build-std=std,panic_abort", "--target", host),
            "-Zunstable-options -Cpanic=immediate-abort")

    result = {
        "unit": "gzipped executable bytes", "profile": "minsize, stripped, gzip -9",
        "platform": platform_name(), "rustc": subprocess.check_output(
            ["rustc", f"+{STABLE}", "-V"], text=True).strip(),
        "tokenizers_revision": revision, "initial_revision": INITIAL_SHA,
        "cargo_lock_sha256": lock_sha,
        "baseline": {"name": "tokenizers before the split", "bytes": old},
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
    main()
