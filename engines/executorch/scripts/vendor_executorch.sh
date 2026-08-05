#!/usr/bin/env bash
#
# Build pytorch-labs/tokenizers (the C++ library behind ExecuTorch and
# torchchat) so `tokbench-executorch` has something to link against.
#
# Run it from anywhere:
#
#     engines/executorch/scripts/vendor_executorch.sh
#
# Everything lands in `engines/executorch/vendor/`, which is gitignored. If you
# never run this, the crate still compiles and the engine reports Unsupported —
# see build.rs and src/lib.rs.
#
# ---------------------------------------------------------------------------
# Why build from source instead of `pip install pytorch-tokenizers`
# ---------------------------------------------------------------------------
#
# The wheel exists (1.4.0, with macOS/Linux/Windows binaries) and it does ship
# the headers under `pytorch_tokenizers/include/`. It is still unusable here,
# and not for a stylistic reason: the only binary in the wheel is
#
#     pytorch_tokenizers/pytorch_tokenizers_cpp.cpython-3XX-<plat>.so
#
# which `file(1)` reports as a "Mach-O 64-bit bundle" on macOS. A bundle is a
# dlopen-only artifact — it has no install name and cannot be passed to the
# linker — so there is no way to link a C++ caller against it. There is no
# libtokenizers.a and no .dylib/.so in the wheel at all.
#
# Even if it were linkable, driving it through its pybind11 entry points would
# put a CPython call and the GIL inside the timed region, which is the
# `Class::Python` measurement, not the `Class::Cffi` one this engine claims.
# Building the C++ library gives an in-process, no-interpreter call — the fair
# comparison against the Rust engines.
#
# ---------------------------------------------------------------------------
# Why SUPPORT_REGEX_LOOKAHEAD=ON is mandatory, not an optimisation
# ---------------------------------------------------------------------------
#
# Upstream defaults it OFF, and OFF cannot tokenize GPT-2. The ByteLevel
# pre-tokenizer pattern in a HuggingFace tokenizer.json contains `\s+(?!\S)`.
# RE2 has no lookahead by design, so with the option OFF, `create_regex` fails
# with "RE2 doesn't support lookahead patterns. Link with `regex_lookahead` to
# enable support." (src/regex.cpp). ON compiles PCRE2 and a `regex_lookahead`
# archive that supplies the fallback.
#
# That archive registers itself with a *static initializer*:
#
#     static bool registered = register_override_fallback_regex(...);
#                                              // src/regex_lookahead.cpp
#
# Nothing references that symbol, so a linker is free to drop the object file
# from the static archive and silently give you the RE2-only behaviour back.
# Upstream guards this with `target_link_options_shared_lib()` (-force_load on
# Apple, --whole-archive elsewhere); our build.rs re-applies the same flag,
# because we link the archives ourselves rather than through CMake's export.
#
# You can confirm it took effect at runtime: a wired build prints
# "Registering override fallback regex" once, from the initializer.
#
# ---------------------------------------------------------------------------
# A note on TOKENIZERS_ENABLE_LOGGING, which we deliberately leave alone
# ---------------------------------------------------------------------------
#
# CMake defaults that option OFF for a Release build, but OFF does not actually
# disable logging: log.h opens with
#
#     #ifndef TK_LOG_ENABLED
#     #define TK_LOG_ENABLED 1
#     #endif
#
# so declining to define the macro leaves logging *on* via the header default.
# We do not paper over it with -DTK_LOG_ENABLED=0, because the benchmark's rule
# is to measure the library as an ordinary `cmake --build` produces it, not a
# configuration hand-tuned to flatter this row.
#
# It does not affect the measurement. Every TK_LOG site reachable from encode
# (bpe_tokenizer_base.cpp:242/255/344, hf_tokenizer.cpp:230) is an error path
# that does not fire on a run whose ids verify; the rest are load-time. The one
# line on stderr is printed once, at load, outside the timed region.

set -euo pipefail

# The pinned upstream commit. Bump this and the `REV` in build.rs/src/lib.rs
# together — the benchmark reports it, and a throughput number attached to
# "whatever main was that day" is not a result.
REV="9b96c3941d8a1bd9dfe8261ac066f35d272c1959"
REPO="https://github.com/pytorch-labs/tokenizers.git"

ENGINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="$ENGINE_DIR/vendor"
SRC="$VENDOR/src"
BUILD="$VENDOR/build"

JOBS="$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu || echo 4 )"

step() { printf '\n=== [%s/5] %s ===\n' "$1" "$2"; }

printf 'vendor_executorch.sh\n'
printf '  repo   %s\n' "$REPO"
printf '  rev    %s\n' "$REV"
printf '  into   %s\n' "$VENDOR"
printf '  jobs   %s\n' "$JOBS"

# --- 1. source ------------------------------------------------------------
step 1 "fetch source at pinned rev"
if [ -d "$SRC/.git" ]; then
  echo "reusing $SRC"
else
  mkdir -p "$VENDOR"
  git clone --filter=blob:none "$REPO" "$SRC"
fi
git -C "$SRC" fetch --depth 1 origin "$REV" 2>/dev/null || git -C "$SRC" fetch origin
git -C "$SRC" checkout --quiet --detach "$REV"
echo "HEAD = $(git -C "$SRC" rev-parse HEAD)"

# --- 2. submodules --------------------------------------------------------
# abseil-cpp, re2, sentencepiece, nlohmann/json and pcre2. Shallow first:
# nlohmann/json in particular has a very large history and a full clone of it
# dwarfs the actual build. Fall back to a full checkout on the odd server that
# refuses a by-SHA fetch.
step 2 "init submodules (shallow)"
git -C "$SRC" submodule update --init --recursive --depth 1 --jobs "$JOBS" \
  || git -C "$SRC" submodule update --init --recursive --jobs "$JOBS"

# --- 3. configure ---------------------------------------------------------
step 3 "cmake configure"
cmake -S "$SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DSUPPORT_REGEX_LOOKAHEAD=ON \
  -DTOKENIZERS_BUILD_TEST=OFF \
  -DTOKENIZERS_BUILD_TOOLS=OFF \
  -DTOKENIZERS_BUILD_PYTHON=OFF

# --- 4. build -------------------------------------------------------------
# No `cmake --install`: upstream adds sentencepiece with EXCLUDE_FROM_ALL, so
# libsentencepiece.a never reaches the install prefix even though libtokenizers
# has a PUBLIC link dependency on it. build.rs therefore consumes the build
# tree directly, which also keeps abseil's several dozen archives in one place.
step 4 "cmake build (this is the slow part: abseil + re2 + sentencepiece + pcre2 + tokenizers)"
cmake --build "$BUILD" --parallel "$JOBS" --target tokenizers regex_lookahead

# --- 5. verify ------------------------------------------------------------
step 5 "verify artifacts"
missing=0
for lib in libtokenizers.a libregex_lookahead.a; do
  if find "$BUILD" -name "$lib" | grep -q .; then
    echo "  ok      $lib"
  else
    echo "  MISSING $lib"
    missing=1
  fi
done
[ "$missing" -eq 0 ] || { echo "build did not produce the expected archives"; exit 1; }

count="$(find "$BUILD" -name '*.a' | wc -l | tr -d ' ')"
echo "  $count static archives under $BUILD"
echo
echo "done. now: cargo build -p tokbench-executorch"
