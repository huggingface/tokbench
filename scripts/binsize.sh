#!/usr/bin/env bash
# Measure what each engine adds to a binary.
#
# For every engine: build the minimal `tokbench-binsize` program with only that
# engine enabled, run it (so a dead-stripped build cannot pass as a small one),
# strip it, and record the size. The reported number is the delta over a
# no-engine baseline build, which is the honest "cost of adding this library"
# figure — the absolute size would mostly be Rust runtime and libstd, identical
# for everyone.
#
# Writes binary_sizes.json, which the driver merges into the report.
#
# Usage: scripts/binsize.sh [engine ...]      (default: the pure-Rust set)
set -uo pipefail
cd "$(dirname "$0")/.."

ENGINES=("$@")
if [ ${#ENGINES[@]} -eq 0 ]; then
  ENGINES=(hf-tokenizers fastokens tokie tiktoken rust-gems-bpe)
fi

# A model directory is only needed so the binary can attempt a real load; any
# model works, and a failed load still exercises the linked code paths.
MODEL_DIR="${MODEL_DIR:-data/models/gpt2}"

strip_size() {
  local bin=$1
  cp "$bin" "$bin.stripped"
  strip "$bin.stripped" 2>/dev/null || strip -S "$bin.stripped" 2>/dev/null || true
  # GNU stat and BSD stat disagree on flags.
  stat -c %s "$bin.stripped" 2>/dev/null || stat -f %z "$bin.stripped"
}

echo "building baseline (no engine linked)…"
cargo build --release -p tokbench-binsize --no-default-features -q || exit 1
BIN=target/release/tokbench-binsize
"$BIN" "$MODEL_DIR" >/dev/null || true
BASE=$(strip_size "$BIN")
echo "  baseline: $((BASE / 1024)) kB"

total=${#ENGINES[@]}
i=0
{
  echo "{"
  sep=""
  for e in "${ENGINES[@]}"; do
    i=$((i + 1))
    printf '[%d/%d] %-16s ' "$i" "$total" "$e" >&2
    if ! cargo build --release -p tokbench-binsize \
        --no-default-features --features "$e" -q 2>/tmp/binsize-$e.err; then
      echo "BUILD FAILED (see /tmp/binsize-$e.err)" >&2
      continue
    fi
    # Prove the engine is really linked and callable.
    "$BIN" "$MODEL_DIR" >/dev/null 2>&1
    sz=$(strip_size "$BIN")
    delta=$((sz - BASE))
    printf 'total %6d kB   +%6d kB over baseline\n' "$((sz / 1024))" "$((delta / 1024))" >&2
    printf '%s  "%s": %s' "$sep" "$e" "$(awk "BEGIN{printf \"%.1f\", $delta/1024}")"
    sep=$',\n'
  done
  echo ""
  echo "}"
} > binary_sizes.json

echo "wrote binary_sizes.json" >&2
cat binary_sizes.json >&2
