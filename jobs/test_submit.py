#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from submit import BLOG_V1_ENGINES, BLOG_V1_MODELS, resolve_benchmark  # noqa: E402
from publish_space import render_dockerfile  # noqa: E402


def args(**overrides):
    values = {
        "profile": "default",
        "models": None,
        "engines": None,
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
        self.assertEqual(config["scaling"], "eng_Latn,cmn_Hani")
        self.assertEqual(config["max_threads"], "8")
        self.assertEqual(config["no_decode"], "0")
        self.assertEqual(config["pin_physical_cores"], "1")
        self.assertEqual(config["latency"], "eng_Latn")

    def test_blog_profile_rejects_matrix_overrides(self) -> None:
        with self.assertRaisesRegex(SystemExit, "fixes the model"):
            resolve_benchmark(args(profile="blog-v1", models="gpt2"))

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


if __name__ == "__main__":
    unittest.main()
