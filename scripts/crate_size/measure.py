#!/usr/bin/env python3
"""Measure every selectable tokenizers v1 crate configuration.

Each value is the gzip -9 size of the same executable after a `minsize` build.
The program calls every selected capability so dead-code elimination cannot turn
an unused dependency into a misleading zero-cost toggle.
"""
from __future__ import annotations

import argparse
import gzip
import itertools
import json
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
OPTIONS = ("serialize", "convert", "train")
FEATURE_OPTIONS = (
    "unigram", "wordpiece", "wordlevel", "normalizers",
    "unicode-scripts", "parallelism",
)
SLIMEST_TOOLCHAIN = "nightly-2026-08-05"
STABLE_TOOLCHAIN = "1.97.1"
LOCKFILE = HERE / "Cargo.lock"


def gz_size(path: Path) -> int:
    return len(gzip.compress(path.read_bytes(), compresslevel=9, mtime=0))


def platform_name() -> str:
    if platform.system() == "Darwin":
        try:
            cpu = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True, stderr=subprocess.DEVNULL,
            ).strip()
        except subprocess.CalledProcessError:
            hardware = subprocess.check_output(
                ["/usr/sbin/system_profiler", "SPHardwareDataType"],
                text=True, stderr=subprocess.DEVNULL,
            )
            cpu = next(
                line.split(":", 1)[1].strip()
                for line in hardware.splitlines()
                if line.strip().startswith(("Chip:", "Processor Name:"))
            )
    else:
        cpu = platform.processor() or platform.machine()
    host = next(
        line.split(":", 1)[1].strip()
        for line in subprocess.check_output(
            ["rustc", f"+{STABLE_TOOLCHAIN}", "-vV"], text=True
        ).splitlines()
        if line.startswith("host:")
    )
    return f"{cpu}, {host}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizers", type=Path,
                        help="optional tokenizers/ workspace at the pinned rc0 SHA")
    parser.add_argument("--model", type=Path, required=True,
                        help="canonical BPE tokenizer.json used by the executable")
    parser.add_argument("--output", type=Path,
                        default=Path("crate_sizes.json"))
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="tokenizers-size-") as raw:
        probe = Path(raw)
        def download_source(sha: str, name: str) -> Path:
            archive = probe / f"{name}.tar.gz"
            urllib.request.urlretrieve(
                f"https://github.com/huggingface/tokenizers/archive/{sha}.tar.gz",
                archive,
            )
            with tarfile.open(archive) as tar:
                if sys.version_info >= (3, 12):
                    tar.extractall(probe, filter="data")
                else:
                    tar.extractall(probe)
            return probe / f"tokenizers-{sha}" / "tokenizers"

        if args.tokenizers:
            checkout = args.tokenizers.resolve()
            root = checkout / "tokenizers"
            if not (root / "tk-encode").is_dir():
                root = checkout
                checkout = root.parent
            if not (root / "tk-encode").is_dir():
                raise SystemExit(f"no tokenizers Rust workspace under {args.tokenizers}")
            revision = subprocess.check_output(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
            ).strip()
        else:
            root = download_source(RC0_SHA, "rc0")
            revision = RC0_SHA
        initial_root = download_source(INITIAL_SHA, "initial")
        manifest = (HERE / "Cargo.toml.in").read_text().replace(
            "__TOKENIZERS__", root.as_posix()
        )
        (probe / "Cargo.toml").write_text(manifest)
        (probe / "src").mkdir()
        shutil.copy2(HERE / "main.rs", probe / "src/main.rs")
        shutil.copy2(HERE / "feature_main.rs", probe / "src/feature_main.rs")
        matrix_cargo = probe / "matrix_cargo.py"
        shutil.copy2(HERE / "matrix_cargo.py", matrix_cargo)
        matrix_cargo.chmod(0o755)
        env = {
            **__import__("os").environ,
            "CARGO_HOME": str(probe / "cargo-home"),
            "CARGO_TARGET_DIR": str(probe / "target"),
        }
        if LOCKFILE.exists():
            shutil.copy2(LOCKFILE, probe / "Cargo.lock")
            lock_check = subprocess.run(
                ["cargo", f"+{STABLE_TOOLCHAIN}", "metadata", "--locked"],
                cwd=probe, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            if lock_check.returncode:
                subprocess.run(
                    ["cargo", f"+{STABLE_TOOLCHAIN}", "generate-lockfile"],
                    cwd=probe, env=env, check=True,
                )
                shutil.copy2(probe / "Cargo.lock", LOCKFILE)
        else:
            subprocess.run(
                ["cargo", f"+{STABLE_TOOLCHAIN}", "generate-lockfile"],
                cwd=probe, env=env, check=True,
            )
            shutil.copy2(probe / "Cargo.lock", LOCKFILE)
        canonical_model = probe / "canonical.json"
        canonical_model.write_text(
            '{"version":"2.0","added_tokens":[],"normalizer":null,'
            '"pre_tokenizer":null,"post_processor":null,"decoder":null,'
            '"model":{"type":"BPE","byte_level":false,'
            '"vocab":{"a":0,"b":1,"ab":2,"abab":3},'
            '"merges":[["a","b"],["ab","ab"]]},"padding":null}'
        )
        def measure(name: str, features: tuple[str, ...]) -> int:
            cmd = ["cargo", f"+{STABLE_TOOLCHAIN}", "build", "--quiet", "--profile", "minsize",
                   "--locked", "--no-default-features", "--bin",
                   "tokenizers-crate-size-probe"]
            if features:
                cmd += ["--features", ",".join(features)]
            subprocess.run(cmd, cwd=probe, env=env, check=True)
            binary = probe / "target/minsize/tokenizers-crate-size-probe"
            subprocess.run([str(binary), str(canonical_model)], check=True,
                           stdout=subprocess.DEVNULL)
            size = gz_size(binary)
            print(f"{name:36} {size:9,d} B")
            return size

        with (initial_root / "Cargo.toml").open("a") as manifest_file:
            manifest_file.write(
                "\n[profile.minsize]\ninherits = \"release\"\nopt-level = \"z\"\n"
                "codegen-units = 1\npanic = \"abort\"\nlto = \"fat\"\nstrip = true\n"
            )
        initial_target = probe / "initial-target"
        initial_env = {**env, "CARGO_TARGET_DIR": str(initial_target)}
        subprocess.run(
            ["cargo", f"+{STABLE_TOOLCHAIN}", "build", "--quiet", "--manifest-path", str(initial_root / "Cargo.toml"),
             "--profile", "minsize", "-p", "tk-encode", "--example",
             "binsize_pipeline", "--locked", "--no-default-features"],
            env=initial_env, check=True,
        )
        initial_binary = initial_target / "minsize/examples/binsize_pipeline"
        subprocess.run([str(initial_binary), str(args.model.resolve()), "hello world"],
                       check=True, stdout=subprocess.DEVNULL)
        old = gz_size(initial_binary)
        print(f"{'before crate split':36} {old:9,d} B")
        configs = {}
        for enabled in itertools.product((False, True), repeat=len(OPTIONS)):
            features = tuple(name for name, on in zip(OPTIONS, enabled) if on)
            key = "+".join(features) or "encode"
            configs[key] = measure(key, features)

        matrix = shutil.which("cargo-matrix")
        if not matrix:
            raise SystemExit(
                "the feature-size probe needs cargo-matrix 0.4.5; run: "
                "cargo install cargo-matrix --version 0.4.5 --locked"
            )
        matrix_version_run = subprocess.run(
            [matrix, "matrix", "--version"], text=True, capture_output=True
        )
        matrix_version = (matrix_version_run.stdout or matrix_version_run.stderr).strip()
        if matrix_version != "cargo-matrix 0.4.5":
            raise SystemExit(
                f"expected cargo-matrix 0.4.5, found {matrix_version!r}"
            )
        def run_feature_matrix(
            profile: str,
            toolchain: str,
            binary: Path,
            extra_args: tuple[str, ...] = (),
            extra_env: dict[str, str] | None = None,
        ) -> dict[str, int]:
            matrix_results = probe / f"matrix-results-{profile}.jsonl"
            matrix_env = {
                **env,
                **(extra_env or {}),
                "CARGO": str(matrix_cargo),
                "MATRIX_REAL_CARGO": shutil.which("cargo") or "cargo",
                "MATRIX_TOOLCHAIN": toolchain,
                "MATRIX_RESULTS": str(matrix_results),
                "MATRIX_FEATURE_OPTIONS": ",".join(FEATURE_OPTIONS),
                "MATRIX_BINARY": str(binary),
            }
            subprocess.run(
                [matrix, "matrix", "--channel", "sizes", "--package",
                 "tokenizers-crate-size-probe", "--manifest-path", str(probe / "Cargo.toml"),
                 "build", "--quiet", "--profile", profile, "--locked", "--bin",
                 "tokenizers-feature-size-probe", *extra_args],
                cwd=probe, env=matrix_env, check=True,
            )
            rows = [json.loads(line) for line in matrix_results.read_text().splitlines()]
            configs = {row["key"]: row["bytes"] for row in rows}
            if len(rows) != 2 ** len(FEATURE_OPTIONS) or len(configs) != len(rows):
                raise SystemExit(
                    f"cargo-matrix did not produce the complete unique {profile} feature matrix"
                )
            for key, size in configs.items():
                print(f"{profile} features: {key:26} {size:9,d} B")
            return configs

        feature_profiles = {
            "minsize": run_feature_matrix(
                "minsize", STABLE_TOOLCHAIN,
                probe / "target/minsize/tokenizers-feature-size-probe",
            )
        }

        sysroot = subprocess.run(
            ["rustc", f"+{SLIMEST_TOOLCHAIN}", "--print", "sysroot"],
            text=True, capture_output=True,
        )
        if sysroot.returncode or not (
            Path(sysroot.stdout.strip()) / "lib/rustlib/src/rust/library"
        ).is_dir():
            raise SystemExit(
                "the slimest probe needs its pinned nightly and rust-src; run: "
                f"rustup toolchain install {SLIMEST_TOOLCHAIN} --profile minimal "
                "--component rust-src"
            )
        host = platform_name().rsplit(", ", 1)[1]
        slimest_binary = probe / f"target/{host}/slimest/tokenizers-feature-size-probe"
        feature_profiles["slimest"] = run_feature_matrix(
            "slimest", SLIMEST_TOOLCHAIN, slimest_binary,
            ("-Z", "build-std=std,panic_abort", "--target", host),
            {"RUSTFLAGS": "-Zunstable-options -Cpanic=immediate-abort"},
        )
        absolute_minimum = feature_profiles["slimest"]["bpe"]
        print(f"{'absolute minimum (slimest BPE)':36} {absolute_minimum:9,d} B")

    out = {
        "unit": "gzipped executable bytes",
        "profile": "minsize, stripped, gzip -9",
        "platform": platform_name(),
        "rustc": subprocess.check_output(
            ["rustc", f"+{STABLE_TOOLCHAIN}", "-V"], text=True
        ).strip(),
        "tokenizers_revision": revision,
        "initial_revision": INITIAL_SHA,
        "baseline": {"name": "tokenizers before the split", "bytes": old},
        "required": "tk-encode",
        "options": list(OPTIONS),
        "configs": configs,
        "feature_required": "bpe",
        "feature_options": list(FEATURE_OPTIONS),
        "feature_profiles": feature_profiles,
        "matrix_tool": matrix_version,
        "absolute_minimum": {
            "bytes": absolute_minimum,
            "profile": "slimest, rebuilt std with panic_immediate_abort",
            "rustc": subprocess.check_output(
                ["rustc", f"+{SLIMEST_TOOLCHAIN}", "-V"], text=True
            ).strip(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
