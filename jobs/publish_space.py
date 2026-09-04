#!/usr/bin/env python3
"""Publish a commit-specific tokbench image as a Hugging Face Docker Space."""

from __future__ import annotations

import argparse
import subprocess
from io import BytesIO


UV_IMAGE = (
    "ghcr.io/astral-sh/uv:0.8.15@"
    "sha256:a5727064a0de127bdb7c9d3c1383f3a9ac307d9f2d8a391edc7896c54289ced0"
)
RUST_IMAGE = (
    "rust:1.93.0-bookworm@"
    "sha256:d0a4aa3ca2e1088ac0c81690914a0d810f2eee188197034edf366ed010a2b382"
)
HUB_VERSION = "1.29.0rc1"


def git(*args: str) -> str:
    result = subprocess.run(
        ("git", *args), check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def render_dockerfile(revision: str) -> str:
    return f"""FROM {UV_IMAGE} AS uv
FROM {RUST_IMAGE}

ARG HUGGINGFACE_HUB_VERSION={HUB_VERSION}
ENV TOKBENCH_SOURCE_REVISION={revision} \\
    HF_HUB_VERSION=${{HUGGINGFACE_HUB_VERSION}} \\
    HF=\"uvx --from huggingface_hub==${{HUGGINGFACE_HUB_VERSION}} hf\"

RUN apt-get update && apt-get install -y --no-install-recommends \\
      build-essential ca-certificates clang cmake git jq libssl-dev \\
      pkg-config python3 python3-venv util-linux \\
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /uvx /usr/local/bin/

WORKDIR /workspace
RUN git clone https://github.com/huggingface/tokbench.git \\
    && cd tokbench \\
    && git checkout --detach {revision} \\
    && test \"$(git rev-parse HEAD)\" = \"{revision}\"
WORKDIR /workspace/tokbench
RUN cargo build --locked --release -p tokbench --features rust-engines

# A Job overrides this command with jobs/run.sh. The server only keeps the
# backing Space healthy and does not execute benchmarks on Space hardware.
CMD [\"python3\", \"-m\", \"http.server\", \"7860\"]
"""


def render_readme(revision: str) -> str:
    short = revision[:7]
    return f"""---
title: tokbench Jobs {short}
emoji: 🧪
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
---

Private, commit-specific Docker image for reproducible tokbench Jobs.

Tokbench source: https://github.com/huggingface/tokbench/tree/{revision}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Space ID, for example user/tokbench-jobs-abc1234")
    parser.add_argument("--revision", help="Full tokbench commit SHA; defaults to the clean local HEAD")
    parser.add_argument("--public", action="store_true", help="Create a public Space instead of a private one")
    args = parser.parse_args()

    revision = args.revision or git("rev-parse", "HEAD")
    if len(revision) != 40 or any(c not in "0123456789abcdefABCDEF" for c in revision):
        raise SystemExit("--revision must be a full 40-character commit SHA")
    if args.revision is None and git("status", "--porcelain"):
        raise SystemExit("refusing to publish a dirty checkout; commit and push it first")

    try:
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError as error:
        raise SystemExit("install a current huggingface_hub first") from error

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="space",
        space_sdk="docker",
        private=not args.public,
        exist_ok=False,
    )
    info = api.create_commit(
        repo_id=args.repo_id,
        repo_type="space",
        commit_message=f"Build tokbench {revision[:12]}",
        operations=[
            CommitOperationAdd(
                path_in_repo="Dockerfile",
                path_or_fileobj=BytesIO(render_dockerfile(revision).encode()),
            ),
            CommitOperationAdd(
                path_in_repo="README.md",
                path_or_fileobj=BytesIO(render_readme(revision).encode()),
            ),
        ],
    )
    print(info.commit_url)
    print(f"Jobs image: hf.co/spaces/{args.repo_id}")


if __name__ == "__main__":
    main()
