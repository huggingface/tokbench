#!/usr/bin/env python3
"""Summarize paired scaling efficiencies across complete tokbench reports."""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: summarize_scaling.py OUTPUT.json RUN.json [RUN.json ...]")

    paired = []
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for report_path in map(Path, sys.argv[2:]):
        report = json.loads(report_path.read_text())
        for run in report.get("runs", []):
            metadata = run["dataset_metadata"]
            for result in run.get("results", []):
                curve = result.get("scaling") or []
                points = {point["threads"]: point["mbps"] for point in curve}
                if 1 not in points or len(points) < 2:
                    continue
                top_threads = max(points)
                efficiency = 100.0 * points[top_threads] / (top_threads * points[1])
                row = {
                    "report": report_path.name,
                    "model": metadata["model"],
                    "corpus": metadata["corpus"],
                    "engine": result["tokenizer_name"],
                    "threads": top_threads,
                    "mbps_1": points[1],
                    "mbps_n": points[top_threads],
                    "efficiency_pct": efficiency,
                }
                paired.append(row)
                grouped[(result["tokenizer_name"], report_path.name)].append(efficiency)

    report_medians = [
        {"engine": engine, "report": report, "paired_cells": len(values),
         "median_efficiency_pct": statistics.median(values)}
        for (engine, report), values in sorted(grouped.items())
    ]
    medians_by_engine: dict[str, list[float]] = defaultdict(list)
    for row in report_medians:
        medians_by_engine[row["engine"]].append(row["median_efficiency_pct"])

    by_engine = {}
    for engine, values in sorted(medians_by_engine.items()):
        by_engine[engine] = {
            "complete_reports": len(values),
            "median_efficiency_pct": statistics.median(values),
            "min_efficiency_pct": min(values),
            "max_efficiency_pct": max(values),
        }

    output = {
        "method": "per-cell paired throughput_n / (n * throughput_1), then median per complete report",
        "by_engine": by_engine,
        "report_medians": report_medians,
        "pairs": paired,
    }
    Path(sys.argv[1]).write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
