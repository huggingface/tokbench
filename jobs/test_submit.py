#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from submit import BLOG_V1_MODELS, resolve_benchmark  # noqa: E402


def args(**overrides):
    values = {
        "profile": "default",
        "models": None,
        "engines": None,
        "scaling": None,
        "max_threads": None,
        "no_decode": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class SubmitProfileTests(unittest.TestCase):
    def test_blog_profile_is_the_eight_model_encode_matrix(self) -> None:
        config = resolve_benchmark(args(profile="blog-v1"))
        self.assertEqual(config["models"].split(","), list(BLOG_V1_MODELS))
        self.assertEqual(config["scaling"], "eng_Latn,cmn_Hani")
        self.assertEqual(config["max_threads"], "8")
        self.assertEqual(config["no_decode"], "1")

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


if __name__ == "__main__":
    unittest.main()
