#!/usr/bin/env python3
"""Aggregate complete tokbench reports without breaking paired measurements."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
from pathlib import Path
from typing import Any


NUMERIC_FIELDS = {
    "binary_delta_kb",
    "crate_size_kb",
    "decode_mbps",
    "decode_ns_per_token",
    "heap_encode_mb",
    "heap_load_mb",
    "load_ms",
    "mbps",
    "mean_execution_time_seconds",
    "ns_per_byte",
    "rss_delta_mb",
}


class IncompatibleReports(ValueError):
    """Raised when reports do not describe the same benchmark matrix."""


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def same(values: list[Any], where: str) -> Any:
    first = values[0]
    if any(value != first for value in values[1:]):
        raise IncompatibleReports(f"{where} differs across reports")
    return copy.deepcopy(first)


def optional_median(values: list[Any], where: str) -> float | None:
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise IncompatibleReports(f"{where} is missing from some reports")
    return median([float(value) for value in values])


def aggregate_breakdown(values: list[Any], where: str) -> dict[str, int] | None:
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise IncompatibleReports(f"{where} is missing from some reports")
    keys = set(values[0])
    if any(set(value) != keys for value in values[1:]):
        raise IncompatibleReports(f"{where} has different stages")
    return {key: round(median([value[key] for value in values])) for key in sorted(keys)}


def aggregate_scaling(
    values: list[Any], sources: list[str], where: str
) -> list[dict[str, Any]] | None:
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise IncompatibleReports(f"{where} is missing from some reports")

    curves = [{point["threads"]: point for point in curve} for curve in values]
    threads = sorted(curves[0])
    if not threads or threads[0] != 1:
        raise IncompatibleReports(f"{where} has no 1-thread baseline")
    if any(sorted(curve) != threads for curve in curves[1:]):
        raise IncompatibleReports(f"{where} has different thread counts")

    output = []
    for count in threads:
        rates = [float(curve[count]["mbps"]) for curve in curves]
        efficiencies = [
            100.0 * rate / (count * float(curve[1]["mbps"]))
            for rate, curve in zip(rates, curves)
        ]
        output.append(
            {
                "threads": count,
                "mbps": median(rates),
                # Pair each point with the 1-thread point from the same report
                # before taking the median. Do not divide pooled medians.
                "efficiency_pct": median(efficiencies),
                "aggregation_samples": [
                    {
                        "report": source,
                        "mbps": rate,
                        "efficiency_pct": efficiency,
                    }
                    for source, rate, efficiency in zip(sources, rates, efficiencies)
                ],
            }
        )
    return output


def aggregate_result(
    results: list[dict[str, Any]], sources: list[str], where: str
) -> dict[str, Any]:
    keys = set().union(*(result.keys() for result in results))
    output: dict[str, Any] = {}
    samples: dict[str, list[dict[str, Any]]] = {}
    for key in sorted(keys):
        values = [result.get(key) for result in results]
        field = f"{where}.{key}"
        if key in NUMERIC_FIELDS:
            value = optional_median(values, field)
            if value is not None:
                output[key] = value
                samples[key] = [
                    {"report": source, "value": raw}
                    for source, raw in zip(sources, values)
                ]
        elif key == "breakdown_nanoseconds":
            value = aggregate_breakdown(values, field)
            if value is not None:
                output[key] = value
        elif key == "scaling":
            value = aggregate_scaling(values, sources, field)
            if value is not None:
                output[key] = value
        else:
            value = same(values, field)
            if value is not None or key in results[0]:
                output[key] = value
    if samples:
        output["aggregation_samples"] = samples
    return output


def index_report(
    report: dict[str, Any], source: str
) -> tuple[list[tuple[str, str]], dict[tuple[str, str], dict[str, Any]]]:
    runs = report.get("runs")
    if not isinstance(runs, list) or not runs:
        raise IncompatibleReports(f"{source} has no runs")
    order = []
    indexed = {}
    for run in runs:
        metadata = run.get("dataset_metadata") or {}
        key = (metadata.get("model"), metadata.get("corpus"))
        if None in key or key in indexed:
            raise IncompatibleReports(f"{source} has an invalid or duplicate cell {key}")
        order.append(key)
        indexed[key] = run
    return order, indexed


def aggregate_reports(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        raise IncompatibleReports("no reports supplied")
    sources = [path.name for path in paths]
    documents = [json.loads(path.read_text()) for path in paths]
    indexed = [
        index_report(document, source)
        for document, source in zip(documents, sources)
    ]
    order = indexed[0][0]
    if any(item[0] != order for item in indexed[1:]):
        raise IncompatibleReports("model/corpus cells or their order differ across reports")

    aggregate_runs = []
    for key in order:
        runs = [item[1][key] for item in indexed]
        metadata = same(
            [run["dataset_metadata"] for run in runs], f"{key}.dataset_metadata"
        )
        result_orders = [
            [result.get("tokenizer_name") for result in run.get("results", [])]
            for run in runs
        ]
        names = same(result_orders, f"{key}.engine order")
        if any(name is None for name in names) or len(set(names)) != len(names):
            raise IncompatibleReports(f"{key} has an invalid or duplicate engine")
        by_run = [
            {result["tokenizer_name"]: result for result in run["results"]}
            for run in runs
        ]
        aggregate_runs.append(
            {
                "dataset_metadata": metadata,
                "results": [
                    aggregate_result(
                        [result_map[name] for result_map in by_run],
                        sources,
                        f"{key}.{name}",
                    )
                    for name in names
                ],
            }
        )

    first = aggregate_runs[0]
    return {
        "dataset_metadata": copy.deepcopy(first["dataset_metadata"]),
        "results": copy.deepcopy(first["results"]),
        "runs": aggregate_runs,
        "aggregation": {
            "statistic": "median",
            "complete_reports": len(paths),
            "sources": sources,
            "scaling_method": (
                "pair each thread count with the same report's 1-thread result, "
                "then take the median"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args()
    args.out.write_text(
        json.dumps(aggregate_reports(args.reports), indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote median of {len(args.reports)} complete reports to {args.out}")


if __name__ == "__main__":
    main()
