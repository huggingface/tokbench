#!/usr/bin/env bash
#
# Vendor and build ONLY IREE's tokenizer, as a static library.
#
# Why this script exists at all
# -----------------------------
# iree-org/iree is a compiler AND a runtime; a full CMake configure+build is
# tens of minutes and gigabytes, and produces nothing this benchmark needs. But
# the tokenizer under runtime/src/iree/tokenizer turns out to be almost
# free-standing: it is plain C17, it depends only on iree/base (also plain C),
# and — verified, not assumed — every one of its translation units compiles
# with nothing but `-I runtime/src`. There are no generated headers, no
# flatbuffer codegen, no configure step in its dependency cone.
#
# So this script skips CMake entirely and drives `cc` directly. That is not a
# shortcut around a hard build; it is the whole build. It costs ~110 compiler
# invocations (a few seconds) instead of a full IREE configure, and it keeps
# the vendored surface auditable: you can read the file list below and know
# exactly what got linked into the benchmark.
#
# What gets fetched
# -----------------
# A blob-filtered, sparse clone of three directories at a PINNED commit:
#
#   runtime/src/iree/tokenizer  the tokenizer itself
#   runtime/src/iree/base       allocator, status, string_view, unicode tables
#   runtime/src/iree/schemas    cpu_data.h, included by base/internal/cpu.c
#
# That is ~11 MB instead of the ~1 GB a full clone costs.
#
# Usage:  bash engines/iree/scripts/vendor_iree.sh
# Result: engines/iree/vendor/lib/libiree_tokenizer.a  (+ vendor/COMMIT)
#
# Re-run it after changing IREE_COMMIT; it is idempotent otherwise.

set -euo pipefail

# Pinned so the number in the report refers to a specific IREE, and so a
# re-run months later measures the same code. Bump deliberately, never
# implicitly: there is no "latest" here.
IREE_COMMIT="83076f9236ba928aced834b7e0597d8d54b5f8bf"
IREE_REPO="https://github.com/iree-org/iree.git"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"          # engines/iree
VENDOR="$ROOT/vendor"
CHECKOUT="$VENDOR/iree"
SRC="$CHECKOUT/runtime/src"
LIBDIR="$VENDOR/lib"
OBJDIR="$VENDOR/obj"

CC_BIN="${CC:-cc}"
AR_BIN="${AR:-ar}"

# IREE's own default (runtime/src/iree/base/BUILD.bazel): the system allocator
# is malloc/free via iree_allocator_libc_ctl. `iree_allocator_system()` is only
# DECLARED when this macro is defined, so it is required, not optional.
DEFINES=(-DIREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_libc_ctl)

# -std=gnu17, not c17: iree/base/threading/processor.h uses `asm volatile`,
# which strict ISO mode rejects.
CFLAGS=(-O2 -std=gnu17 -fPIC -I "$SRC" "${DEFINES[@]}")

echo "==> vendoring IREE tokenizer @ ${IREE_COMMIT:0:12}"

# ---------------------------------------------------------------------------
# 1. Fetch (sparse, blob-filtered, pinned)
# ---------------------------------------------------------------------------
if [ -f "$VENDOR/COMMIT" ] && [ "$(cat "$VENDOR/COMMIT")" = "$IREE_COMMIT" ] \
   && [ -d "$SRC/iree/tokenizer" ]; then
  echo "    checkout already at pinned commit, skipping fetch"
else
  rm -rf "$CHECKOUT"
  mkdir -p "$CHECKOUT"
  echo "    cloning (blob:none, sparse) ..."
  # init+fetch rather than `git clone`, on purpose. A plain
  # `clone --filter=blob:none` still downloads the commit and tree objects for
  # IREE's entire history — ~55 MB and most of the wall clock. Fetching the one
  # pinned commit at depth 1 with blobs filtered brings only what the sparse
  # paths actually need: ~11 MB.
  git init --quiet "$CHECKOUT"
  git -C "$CHECKOUT" remote add origin "$IREE_REPO"
  git -C "$CHECKOUT" fetch --quiet --depth 1 --filter=blob:none origin "$IREE_COMMIT"
  git -C "$CHECKOUT" sparse-checkout init --cone
  git -C "$CHECKOUT" sparse-checkout set \
      runtime/src/iree/tokenizer \
      runtime/src/iree/base \
      runtime/src/iree/schemas
  git -C "$CHECKOUT" checkout --quiet FETCH_HEAD
  echo "    fetched $(du -sh "$CHECKOUT" | cut -f1)"
fi

# ---------------------------------------------------------------------------
# 2. Enumerate sources
# ---------------------------------------------------------------------------
# Everything under tokenizer/ and base/ except:
#   *_test.c/_fuzz.c/_benchmark.c  test-only
#   testing/ tooling/ tracing/     harnesses and the optional Tracy integration
#   allocator_mimalloc.c           needs mimalloc.h; we use the libc allocator
#
# base/ is taken wholesale rather than hand-pruned. It is small, and because
# the result is a static archive the linker pulls only the objects the
# tokenizer actually references — unused ones cost nothing in the final binary.
# (Written to a file and read back rather than `mapfile`, which needs bash 4;
# macOS still ships bash 3.2 as /bin/bash.)
SRCLIST="$VENDOR/sources.txt"
mkdir -p "$VENDOR"
find "$SRC/iree/tokenizer" "$SRC/iree/base" \
  -name '*.c' \
  ! -path '*/testing/*' \
  ! -path '*/tooling/*' \
  ! -path '*/tracing/*' \
  ! -path '*/testdata/*' \
  ! -name '*_test.c' \
  ! -name '*_fuzz.c' \
  ! -name '*_benchmark.c' \
  ! -name 'allocator_mimalloc.c' \
  | sort > "$SRCLIST"

N=$(wc -l < "$SRCLIST" | tr -d ' ')
if [ "$N" -eq 0 ]; then
  echo "!!  no sources found under $SRC — sparse checkout failed?" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 3. Compile
# ---------------------------------------------------------------------------
rm -rf "$OBJDIR" "$LIBDIR"
mkdir -p "$OBJDIR" "$LIBDIR"

echo "==> compiling $N translation units with $CC_BIN"
START=$(date +%s)
i=0
while IFS= read -r s; do
  i=$((i + 1))
  rel="${s#"$SRC"/}"
  obj="$OBJDIR/$(echo "$rel" | tr '/' '_').o"
  if ! "$CC_BIN" "${CFLAGS[@]}" -c "$s" -o "$obj" 2>"$OBJDIR/.err"; then
    echo "[$i/$N] FAILED $rel" >&2
    cat "$OBJDIR/.err" >&2
    exit 1
  fi
  # Progress with an ETA once a few units have been timed.
  NOW=$(date +%s)
  ELAPSED=$((NOW - START))
  if [ "$i" -ge 5 ] && [ "$ELAPSED" -gt 0 ]; then
    ETA=$(( (ELAPSED * (N - i)) / i ))
    printf '[%3d/%d] %-58s | %ds elapsed | eta ~%ds\n' "$i" "$N" "$rel" "$ELAPSED" "$ETA"
  else
    printf '[%3d/%d] %s\n' "$i" "$N" "$rel"
  fi
done < "$SRCLIST"
rm -f "$OBJDIR/.err"

# ---------------------------------------------------------------------------
# 4. Archive
# ---------------------------------------------------------------------------
"$AR_BIN" rcs "$LIBDIR/libiree_tokenizer.a" "$OBJDIR"/*.o
echo "$IREE_COMMIT" > "$VENDOR/COMMIT"

TOTAL=$(( $(date +%s) - START ))
SIZE=$(ls -l "$LIBDIR/libiree_tokenizer.a" | awk '{print $5}')
echo "==> built $LIBDIR/libiree_tokenizer.a ($SIZE bytes) in ${TOTAL}s"
echo "    now: cargo build --release -p tokbench --features iree"
