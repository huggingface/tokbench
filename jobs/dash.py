#!/usr/bin/env python3
"""Fetch, aggregate and serve tokbench results in the native dashboard."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import threading
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from aggregate_results import aggregate_reports


ROOT = Path(__file__).resolve().parent.parent


def safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "bucket"


def verify_artifacts(directory: Path) -> None:
    manifest = directory / "artifact-sha256.txt"
    if not manifest.exists():
        raise SystemExit(f"missing checksum manifest: {manifest}")
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        try:
            expected, relative = line.split(maxsplit=1)
        except ValueError as error:
            raise SystemExit(f"invalid checksum line: {line!r}") from error
        relative = relative.lstrip("*").removeprefix("./")
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise SystemExit(f"unsafe path in checksum manifest: {relative}")
        artifact = directory / path
        if not artifact.is_file():
            raise SystemExit(f"artifact listed in manifest is missing: {artifact}")
        actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"checksum mismatch for {artifact}")


def fetch_job(bucket: str, job_id: str, cache_root: Path) -> Path:
    if not job_id or "/" in job_id or job_id in {".", ".."}:
        raise SystemExit("JOB_ID must be one exact Job identifier")
    try:
        from huggingface_hub import sync_bucket
    except ImportError as error:
        raise SystemExit(
            "bucket fetching needs a current huggingface_hub: "
            "pip install -U huggingface_hub"
        ) from error

    base = bucket.removeprefix("hf://buckets/").rstrip("/")
    destination = cache_root / safe_component(base) / job_id
    destination.mkdir(parents=True, exist_ok=True)
    uri = f"hf://buckets/{base}/{job_id}"
    print(f"syncing {uri} -> {destination}")
    sync_bucket(uri, str(destination))
    verify_artifacts(destination)
    return destination


def select_run(directory: Path, run: str) -> Path:
    value = run.removeprefix("run-").removesuffix(".json")
    if not value.isdigit():
        raise SystemExit("RUN must be a number such as 1, 01, or run-01.json")
    path = directory / f"run-{int(value):02d}.json"
    if not path.is_file():
        raise SystemExit(f"run does not exist: {path}")
    return path


def prepare(results: Path, stage: Path, run: str | None) -> Path:
    stage.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "dashboard.html", stage / "dashboard.html")
    output = stage / "tokenizer_bench_results.json"

    if results.is_file():
        if run:
            raise SystemExit("RUN applies to a Job/results directory, not one JSON file")
        shutil.copy2(results, output)
        print(f"dashboard report: {results}")
        return output

    if not results.is_dir():
        raise SystemExit(f"results path does not exist: {results}")
    if run:
        selected = select_run(results, run)
        shutil.copy2(selected, output)
        print(f"dashboard report: {selected}")
        return output

    reports = sorted(results.glob("run-*.json"))
    if not reports:
        raise SystemExit(f"no run-*.json reports under {results}")
    output.write_text(
        json.dumps(aggregate_reports(reports), indent=2, sort_keys=True) + "\n"
    )
    print(f"dashboard report: median of {len(reports)} complete reports")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket")
    parser.add_argument("--job-id")
    parser.add_argument("--run")
    parser.add_argument("--results", type=Path)
    parser.add_argument(
        "--cache-dir", type=Path, default=ROOT / ".tokbench" / "jobs"
    )
    parser.add_argument(
        "--stage-dir", type=Path, default=ROOT / ".tokbench" / "dash"
    )
    parser.add_argument("--port", type=int, default=8712)
    parser.add_argument("--prepare-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-open", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if bool(args.bucket) != bool(args.job_id):
        raise SystemExit("BUCKET and JOB_ID must be provided together")
    if args.bucket and args.results:
        raise SystemExit("use either BUCKET + JOB_ID or RESULTS, not both")

    if args.bucket:
        results = fetch_job(args.bucket, args.job_id, args.cache_dir)
    elif args.results:
        results = args.results.resolve()
    else:
        results = ROOT / "tokenizer_bench_results.json"

    prepare(results, args.stage_dir, args.run)
    if args.prepare_only:
        return

    handler = partial(SimpleHTTPRequestHandler, directory=str(args.stage_dir))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://127.0.0.1:{args.port}/dashboard.html"
    print(f"serving {url} (Ctrl-C to stop)")
    if not args.no_open:
        threading.Timer(0.2, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
