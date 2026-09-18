#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dash import prepare, verify_artifacts  # noqa: E402
from test_aggregate_results import report  # noqa: E402


class DashboardTests(unittest.TestCase):
    def test_prepare_directory_aggregates_all_runs(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            results = root / "results"
            stage = root / "stage"
            results.mkdir()
            (results / "run-01.json").write_text(json.dumps(report(10, 80)))
            (results / "run-02.json").write_text(json.dumps(report(20, 144)))

            output = prepare(results, stage, run=None)
            aggregate = json.loads(output.read_text())
            self.assertEqual(aggregate["aggregation"]["complete_reports"], 2)
            self.assertTrue((stage / "dashboard.html").is_file())

    def test_verify_artifacts_rejects_a_modified_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            artifact = directory / "run-01.json"
            artifact.write_text("original")
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            (directory / "artifact-sha256.txt").write_text(
                f"{digest}  ./run-01.json\n"
            )
            verify_artifacts(directory)
            artifact.write_text("modified")
            with self.assertRaisesRegex(SystemExit, "checksum mismatch"):
                verify_artifacts(directory)


if __name__ == "__main__":
    unittest.main()
