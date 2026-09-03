#!/usr/bin/env python3
"""Submit an immutable tokbench image to Hugging Face Jobs."""

from __future__ import annotations

import argparse
import json
import os


BLOG_V1_MODELS = (
    "deepseek-v4",
    "glm-5.2",
    "gpt-oss",
    "gpt2",
    "llama-3",
    "minimax",
    "nemotron-3",
    "qwen2",
)
BLOG_V1_SCALING = "eng_Latn,cmn_Hani"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        required=True,
        help="Container image, required by default to use @sha256:...",
    )
    parser.add_argument(
        "--input-revision",
        required=True,
        help="Commit SHA of hf-internal-testing/tokenizers-test-data",
    )
    parser.add_argument(
        "--bucket",
        required=True,
        help="Writable HF Storage Bucket, for example huggingface/tokbench-results",
    )
    parser.add_argument("--namespace", default=None)
    parser.add_argument(
        "--profile",
        choices=("default", "blog-v1"),
        default="default",
        help="Pinned benchmark matrix; blog-v1 reproduces the blog's Section 01 inputs",
    )
    parser.add_argument("--flavor", default="cpu-performance")
    parser.add_argument("--timeout", default="6h")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--max-threads", type=int)
    parser.add_argument("--scaling", help="Comma-separated scaling corpora")
    parser.add_argument("--models", help="Comma-separated model allowlist")
    parser.add_argument("--engines", help="Comma-separated engine allowlist")
    parser.add_argument("--no-decode", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved non-secret Job configuration without submitting",
    )
    parser.add_argument("--allow-mutable-image", action="store_true")
    parser.add_argument(
        "--no-hf-token",
        action="store_true",
        help="Do not forward the local HF token as a Job secret",
    )
    return parser.parse_args()


def resolve_benchmark(args: argparse.Namespace) -> dict[str, str]:
    """Resolve defaults and reject overrides that would make a named profile ambiguous."""
    if args.profile == "blog-v1":
        overridden = [
            flag
            for flag, value in (
                ("--models", args.models),
                ("--engines", args.engines),
                ("--scaling", args.scaling),
            )
            if value is not None
        ]
        if overridden:
            raise SystemExit(
                "--profile blog-v1 fixes the model, engine and scaling matrix; "
                f"remove {', '.join(overridden)}"
            )
        if args.max_threads not in (None, 8):
            raise SystemExit("--profile blog-v1 requires --max-threads 8")
        return {
            "models": ",".join(BLOG_V1_MODELS),
            "engines": "",
            "scaling": BLOG_V1_SCALING,
            "max_threads": "8",
            "no_decode": "1",
        }

    max_threads = args.max_threads if args.max_threads is not None else 8
    return {
        "models": args.models or "",
        "engines": args.engines or "",
        "scaling": args.scaling or BLOG_V1_SCALING,
        "max_threads": str(max_threads),
        "no_decode": "1" if args.no_decode else "0",
    }


def main() -> None:
    args = parse_args()
    benchmark = resolve_benchmark(args)
    if args.runs < 1 or args.reps < 1 or int(benchmark["max_threads"]) < 1:
        raise SystemExit("--runs, --reps and --max-threads must be positive")
    if not args.allow_mutable_image and "@sha256:" not in args.image:
        raise SystemExit(
            "--image must use an immutable @sha256: digest "
            "(or pass --allow-mutable-image)"
        )
    if len(args.input_revision) != 40 or any(
        c not in "0123456789abcdefABCDEF" for c in args.input_revision
    ):
        raise SystemExit("--input-revision must be a full 40-character commit SHA")

    env = {
        "TOKBENCH_IMAGE": args.image,
        "TOKBENCH_INPUT_REVISION": args.input_revision,
        "TOKBENCH_RUNS": str(args.runs),
        "TOKBENCH_REPS": str(args.reps),
        "TOKBENCH_PROFILE": args.profile,
        "TOKBENCH_MAX_THREADS": benchmark["max_threads"],
        "TOKBENCH_SCALING": benchmark["scaling"],
        "TOKBENCH_SCALING_ORDER": "alternating-forward-reverse",
        "TOKBENCH_MODELS": benchmark["models"],
        "TOKBENCH_ENGINES": benchmark["engines"],
        "TOKBENCH_NO_DECODE": benchmark["no_decode"],
    }
    if args.dry_run:
        print(
            json.dumps(
                {
                    "image": args.image,
                    "command": ["bash", "jobs/run.sh"],
                    "flavor": args.flavor,
                    "namespace": args.namespace,
                    "timeout": args.timeout,
                    "environment": env,
                    "bucket": args.bucket,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    try:
        from huggingface_hub import Volume, get_token, run_job
    except ImportError as error:
        raise SystemExit(
            "submission needs a current huggingface_hub: "
            "pip install -U huggingface_hub"
        ) from error

    secrets = {}
    if not args.no_hf_token:
        token = os.environ.get("HF_TOKEN") or get_token()
        if not token:
            raise SystemExit(
                "no Hugging Face token found; log in, set HF_TOKEN, "
                "or pass --no-hf-token"
            )
        secrets["HF_TOKEN"] = token

    job = run_job(
        image=args.image,
        command=["bash", "jobs/run.sh"],
        flavor=args.flavor,
        namespace=args.namespace,
        timeout=args.timeout,
        env=env,
        secrets=secrets,
        volumes=[Volume(type="bucket", source=args.bucket, mount_path="/outputs")],
        name="tokbench-reproducible",
        labels={
            "project": "tokbench",
            "profile": args.profile,
            "input-revision": args.input_revision[:12],
        },
    )
    print(job.url)
    print(f"job id: {job.id}")


if __name__ == "__main__":
    main()
