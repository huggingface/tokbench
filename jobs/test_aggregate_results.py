#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_results import IncompatibleReports, aggregate_reports  # noqa: E402


def report(
    one: float, eight: float, version: str = "1", decode_mbps: float = 100
) -> dict:
    metadata = {
        "file_size_bytes": 100,
        "total_characters": 100,
        "corpus": "eng_Latn",
        "model": "gpt2",
        "reps": 5,
        "warmup": True,
    }
    result = {
        "tokenizer_name": "pipeline",
        "total_tokens_produced": 25,
        "mean_execution_time_seconds": 1 / one,
        "engine_version": version,
        "engine_lang": "rust",
        "engine_class": "native",
        "also_computes": "",
        "internally_parallel": False,
        "load_ms": 10,
        "mbps": one,
        "ns_per_byte": 100 / one,
        "ids_hash": "same",
        "verified": True,
        "decode_mbps": decode_mbps,
        "decode_ns_per_token": 12_000 / decode_mbps,
        "decode_text_hash": "decoded-same",
        "decode_verified": True,
        "scaling": [
            {"threads": 1, "mbps": one, "efficiency_pct": 100},
            {
                "threads": 8,
                "mbps": eight,
                "efficiency_pct": eight / (8 * one) * 100,
            },
        ],
        "reused_text": False,
    }
    run = {"dataset_metadata": metadata, "results": [result]}
    return {"dataset_metadata": metadata, "results": [result], "runs": [run]}


class AggregateTests(unittest.TestCase):
    def write(self, directory: Path, name: str, value: dict) -> Path:
        path = directory / name
        path.write_text(json.dumps(value))
        return path

    def test_scaling_efficiency_is_paired_before_median(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            paths = [
                self.write(directory, "run-01.json", report(10, 80)),
                self.write(directory, "run-02.json", report(20, 144)),
            ]
            aggregate = aggregate_reports(paths)
            result = aggregate["runs"][0]["results"][0]
            self.assertEqual(result["scaling"][0]["mbps"], 15)
            self.assertEqual(result["scaling"][1]["mbps"], 112)
            self.assertEqual(result["scaling"][1]["efficiency_pct"], 95)
            self.assertEqual(result["decode_mbps"], 100)
            self.assertEqual(aggregate["aggregation"]["complete_reports"], 2)

    def test_decode_rates_use_the_median(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            paths = [
                self.write(directory, "run-01.json", report(10, 80, decode_mbps=80)),
                self.write(directory, "run-02.json", report(10, 80, decode_mbps=120)),
            ]
            result = aggregate_reports(paths)["runs"][0]["results"][0]
            self.assertEqual(result["decode_mbps"], 100)
            self.assertEqual(result["decode_ns_per_token"], 125)

    def test_identity_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            paths = [
                self.write(directory, "run-01.json", report(10, 80)),
                self.write(directory, "run-02.json", report(10, 80, version="2")),
            ]
            with self.assertRaisesRegex(IncompatibleReports, "engine_version"):
                aggregate_reports(paths)


if __name__ == "__main__":
    unittest.main()
