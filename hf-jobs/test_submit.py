#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from submit import (  # noqa: E402
    BLOG_V1_ENGINES,
    BLOG_V1_GIGATOKEN_LIBRARY_ENGINES,
    BLOG_V1_LIBRARY_ENGINES,
    BLOG_V1_MODELS,
    resolve_benchmark,
)
from publish_space import render_dockerfile  # noqa: E402


def args(**overrides):
    values = {
        "profile": "default",
        "models": None,
        "engines": None,
        "measure": None,
        "compare_to": None,
        "cache_capacity": None,
        "scaling_mode": "auto",
        "scaling": None,
        "max_threads": None,
        "no_decode": False,
        "pin_physical_cores": False,
        "latency": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class SubmitProfileTests(unittest.TestCase):
    def test_blog_profile_is_the_fixed_blog_matrix(self) -> None:
        config = resolve_benchmark(args(profile="blog-v1"))
        self.assertEqual(config["models"].split(","), list(BLOG_V1_MODELS))
        self.assertEqual(config["engines"], BLOG_V1_ENGINES)
        self.assertEqual(config["measure"], "")
        self.assertEqual(config["compare_to"], "")
        self.assertEqual(config["scaling"], "eng_Latn,cmn_Hani")
        self.assertEqual(config["max_threads"], "8")
        self.assertEqual(config["no_decode"], "0")
        self.assertEqual(config["pin_physical_cores"], "1")
        self.assertEqual(config["latency"], "eng_Latn")
        self.assertEqual(config["scaling_mode"], "auto")

    def test_blog_profile_rejects_matrix_overrides(self) -> None:
        with self.assertRaisesRegex(SystemExit, "fixes the measurement matrix"):
            resolve_benchmark(args(profile="blog-v1", models="gpt2"))
        with self.assertRaisesRegex(SystemExit, "--cache-capacity"):
            resolve_benchmark(args(profile="blog-v1", cache_capacity=8192))

    def test_blog_profile_accepts_scaling_mode_axis(self) -> None:
        config = resolve_benchmark(
            args(profile="blog-v1", scaling_mode="independent-instances")
        )
        self.assertEqual(config["scaling_mode"], "independent-instances")

    def test_library_profile_is_encode_only_with_one_comparator(self) -> None:
        config = resolve_benchmark(args(profile="blog-v1-libraries"))
        self.assertEqual(config["models"].split(","), list(BLOG_V1_MODELS))
        self.assertEqual(config["engines"], BLOG_V1_LIBRARY_ENGINES)
        self.assertEqual(config["measure"], "encode")
        self.assertEqual(config["compare_to"], "hf-tokenizers")
        self.assertEqual(config["scaling"], "")
        self.assertEqual(config["latency"], "")
        self.assertEqual(config["no_decode"], "1")
        self.assertEqual(config["max_threads"], "1")
        self.assertEqual(config["pin_physical_cores"], "0")

    def test_library_profile_rejects_scaling_mode(self) -> None:
        with self.assertRaisesRegex(SystemExit, "--scaling-mode"):
            resolve_benchmark(
                args(
                    profile="blog-v1-libraries",
                    scaling_mode="native-threads",
                )
            )

    def test_default_profile_propagates_pipeline_cache_capacity(self) -> None:
        config = resolve_benchmark(args(cache_capacity=0))
        self.assertEqual(config["cache_capacity"], "0")

    def test_default_profile_propagates_scaling_mode(self) -> None:
        config = resolve_benchmark(args(scaling_mode="independent-instances"))
        self.assertEqual(config["scaling_mode"], "independent-instances")

    def test_gigatoken_library_profile_adds_gigatoken(self) -> None:
        config = resolve_benchmark(args(profile="blog-v1-libraries-gigatoken"))
        self.assertEqual(config["engines"], BLOG_V1_GIGATOKEN_LIBRARY_ENGINES)
        self.assertEqual(config["measure"], "encode")
        self.assertEqual(config["compare_to"], "hf-tokenizers")
        self.assertEqual(config["max_threads"], "1")
        self.assertEqual(config["pin_physical_cores"], "0")

    def test_default_profile_retains_decode_and_custom_selection(self) -> None:
        config = resolve_benchmark(
            args(models="gpt2", scaling="eng_Latn", max_threads=4)
        )
        self.assertEqual(config["models"], "gpt2")
        self.assertEqual(config["scaling"], "eng_Latn")
        self.assertEqual(config["max_threads"], "4")
        self.assertEqual(config["no_decode"], "0")
        self.assertEqual(config["pin_physical_cores"], "0")
        self.assertEqual(config["latency"], "")

    def test_default_profile_can_pin_physical_cores(self) -> None:
        config = resolve_benchmark(args(pin_physical_cores=True))
        self.assertEqual(config["pin_physical_cores"], "1")


class PublishSpaceTests(unittest.TestCase):
    def test_recipe_pins_source_and_base_images(self) -> None:
        revision = "1" * 40
        recipe = render_dockerfile(revision)
        self.assertGreaterEqual(recipe.count(revision), 3)
        self.assertIn("rust:1.93.0-bookworm@sha256:", recipe)
        self.assertIn("ghcr.io/astral-sh/uv:0.8.15@sha256:", recipe)

    def test_gigatoken_recipe_uses_pinned_nightly_and_prebuilt_binary(self) -> None:
        recipe = render_dockerfile("1" * 40, gigatoken=True)
        self.assertIn("python3 hf-jobs/prepare_gigatoken.py", recipe)
        self.assertIn("nightly-2026-08-05", recipe)
        self.assertIn("-Z profile-rustflags", recipe)
        self.assertIn("--features rust-engines,gigatoken", recipe)
        self.assertIn("TOKBENCH_SKIP_BUILD=1", recipe)
        self.assertIn("CARGO_BUILD_JOBS=1", recipe)
        self.assertIn("CARGO_PROFILE_RELEASE_DEBUG=0", recipe)
        self.assertIn("-fuse-ld=lld", recipe)
        self.assertIn("lld python3-dev", recipe)
        self.assertIn(
            "TOKBENCH_GIGATOKEN_REVISION=34a1599f0c0ae7d7cd0d1c530e6522320158b360",
            recipe,
        )
        self.assertIn("python3-dev", recipe)


if __name__ == "__main__":
    unittest.main()
