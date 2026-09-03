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

fsize() { stat -c %s "$1" 2>/dev/null || stat -f %z "$1"; }

strip_size() {
  local bin=$1
  cp "$bin" "$bin.stripped"
  strip "$bin.stripped" 2>/dev/null || strip -S "$bin.stripped" 2>/dev/null || true
  fsize "$bin.stripped"
}

# Gzipped, because that is what actually ships: an npm/wheel/app bundle is
# transferred compressed, and tokenizers#2293 tracks this number specifically.
# Stripped-but-uncompressed flatters engines that carry large, highly
# compressible tables (Unicode data compresses ~5x; hashed vocabularies do not).
gz_size() {
  gzip -9 -c "$1.stripped" | wc -c | tr -d ' '
}

echo "building baseline (no engine linked)…"
cargo build --profile binsize -p tokbench-binsize --no-default-features -q || exit 1
BIN=target/binsize/tokbench-binsize
"$BIN" "$MODEL_DIR" >/dev/null || true
BASE=$(strip_size "$BIN")
BASE_GZ=$(gz_size "$BIN")
echo "  baseline: $((BASE / 1024)) kB stripped, $((BASE_GZ / 1024)) kB gzipped"

total=${#ENGINES[@]}
i=0
# Accumulate plain "<engine> <kB>" lines and convert at the end. Emitting JSON
# directly from printf inside the loop is fragile: one command substitution
# that yields an empty string silently shifts every remaining format argument
# and produces a valid-looking but wrong document.
TSV=$(mktemp)
{
  for e in "${ENGINES[@]}"; do
    i=$((i + 1))
    printf '[%d/%d] %-16s ' "$i" "$total" "$e" >&2
    if ! cargo build --profile binsize -p tokbench-binsize \
        --no-default-features --features "$e" -q 2>/tmp/binsize-$e.err; then
      echo "BUILD FAILED (see /tmp/binsize-$e.err)" >&2
      continue
    fi
    # Prove the engine is really linked and callable.
    "$BIN" "$MODEL_DIR" >/dev/null 2>&1
    sz=$(strip_size "$BIN")
    gz=$(gz_size "$BIN")
    delta=$((sz - BASE))
    dgz=$((gz - BASE_GZ))
    printf '+%6d kB stripped   +%6d kB gzipped\n' "$((delta / 1024))" "$((dgz / 1024))" >&2
    echo "$e $dgz" >> "$TSV"
  done
} >&2

python3 - "$TSV" <<'PY'
import json, sys
rows = {}
for line in open(sys.argv[1]):
    name, gz = line.split()
    # The driver's column is gzipped kB: that is what actually ships.
    rows[name] = round(int(gz) / 1024, 1)
json.dump(rows, open("binary_sizes.json", "w"), indent=2)
print("wrote binary_sizes.json:", json.dumps(rows), file=sys.stderr)
PY
rm -f "$TSV"
