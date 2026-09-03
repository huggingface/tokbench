ARG RUST_IMAGE=rust:1.89.0-bookworm
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.8.15

FROM ${UV_IMAGE} AS uv
FROM ${RUST_IMAGE}

ARG TOKBENCH_REVISION=unknown
ARG HUGGINGFACE_HUB_VERSION=1.29.0rc1

LABEL org.opencontainers.image.source="https://github.com/huggingface/tokbench" \
      org.opencontainers.image.revision="${TOKBENCH_REVISION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates clang cmake git jq libssl-dev \
      pkg-config python3 python3-venv util-linux \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

WORKDIR /workspace/tokbench
COPY . .

ENV TOKBENCH_SOURCE_REVISION=${TOKBENCH_REVISION} \
    HF_HUB_VERSION=${HUGGINGFACE_HUB_VERSION} \
    HF="uvx --from huggingface_hub==${HUGGINGFACE_HUB_VERSION} hf"

# Resolve and compile dependencies while building the immutable image. The
# benchmark itself never compiles code and therefore cannot measure a different
# dependency graph from the one identified by Cargo.lock.
RUN test "${TOKBENCH_REVISION}" != unknown \
    && cargo build --locked --release -p tokbench --features rust-engines

CMD ["bash", "jobs/run.sh"]
