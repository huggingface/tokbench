#!/usr/bin/env bash
#
# vendor_blingfire.sh — obtain libblingfiretokdll (and, optionally, BlingFire's
# compiled .bin tokenizer models) for the tokbench `blingfire` engine.
#
# Why this script exists at all
# -----------------------------
# The `blingfire` Rust crate on crates.io binds only text_to_words /
# text_to_sentences, which return space-separated STRINGS, not token ids. There
# is nothing there to compare against a subword tokenizer. The comparable entry
# point is `TextToIds` in Microsoft's C library `libblingfiretokdll`, which no
# Rust crate binds — so tokbench links the C library directly and this script is
# how that library gets onto the machine. See ../src/lib.rs for the full
# argument.
#
# Two ways to get the library, and why both are here
# --------------------------------------------------
#   prebuilt  Microsoft commit built binaries straight into the BlingFire repo
#             (dist-pypi/blingfire/libblingfiretokdll.{so,dylib}); the PyPI
#             wheel `blingfire` ships the very same files (it is a
#             py3-none-any wheel — one wheel, one set of binaries, for every
#             platform). Downloading one file takes seconds.
#
#             The catch, and it is not a small one: those binaries are
#             **x86_64 only**. `lipo -archs libblingfiretokdll.dylib` prints
#             `x86_64` and nothing else — there is no arm64 slice, so on Apple
#             silicon the prebuilt path cannot produce a library the linker
#             will accept for an arm64 target.
#
#   source    CMake build from the pinned tag. Slower (a couple of minutes) but
#             produces a native library for whatever the host is. Only the
#             blingfiretokdll targets are built: they link `fsaClient` alone,
#             so the FSA compiler library and the ~20 command-line tools in
#             blingfiretools/ are never compiled.
#
#             The source path builds the STATIC archives as well as the shared
#             library, and build.rs prefers the static ones. That is not a
#             size preference, it is a runtime-correctness one: this crate is
#             an rlib, and `cargo:rustc-link-arg` (how you would inject an
#             -rpath) does not propagate from a library package to the
#             dependent binary. A dynamically linked libblingfiretokdll
#             therefore links fine and then fails to load at run time unless
#             the user exports DYLD_/LD_LIBRARY_PATH. Static archives make the
#             question disappear.
#
# The default mode is `auto`: fetch the prebuilt binary, check its architecture
# against the host, and fall back to a source build when they disagree. That
# way a Linux x86_64 CI runner takes the fast path and an arm64 laptop still
# ends up with a working library, without either having to know which it is.
#
# Nothing here is required to *build* the crate. build.rs probes for the
# library and, when it is absent, compiles a stub whose `build()` returns
# Unsupported with instructions. The workspace must keep building for people
# who never run this script.
#
# Usage
# -----
#   scripts/vendor_blingfire.sh                 # auto: prebuilt, else source
#   scripts/vendor_blingfire.sh --source        # force a CMake build
#   scripts/vendor_blingfire.sh --prebuilt      # force the download (may fail)
#   scripts/vendor_blingfire.sh --models-only   # just fetch/install .bin models
#   scripts/vendor_blingfire.sh --no-models     # library only
#   scripts/vendor_blingfire.sh --clean         # remove vendor/
#
#   BLINGFIRE_TAG=v0.1.8 scripts/vendor_blingfire.sh    # pin a different tag
#
# Output layout (all under engines/blingfire/vendor/, git-ignored)
# ----------------------------------------------------------------
#   vendor/lib/libblingfiretokdll.{so,dylib}   what build.rs links against
#   vendor/include/blingfiretokdll.h           the upstream header, for reading
#   vendor/models/*.bin                        BlingFire's compiled models
#   vendor/PROVENANCE                          how the library was obtained
#
set -euo pipefail

# --------------------------------------------------------------------------
# Paths and configuration
# --------------------------------------------------------------------------

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_DIR="$(dirname "$HERE")"                 # engines/blingfire
REPO_ROOT="$(cd "$ENGINE_DIR/../.." && pwd)"    # tokbench/
VENDOR="$ENGINE_DIR/vendor"

# Pinned so the number in the report refers to a known build. v0.1.8 is the
# latest tag and matches the version PyPI ships.
TAG="${BLINGFIRE_TAG:-v0.1.8}"
UPSTREAM="https://github.com/microsoft/BlingFire"
RAW="$UPSTREAM/raw/$TAG"

MODE="auto"
WANT_MODELS=1
WANT_LIB=1
JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

# Which compiled models to fetch. These are the ones that could plausibly
# correspond to a vocabulary in data/models/. See install_models() for why the
# mapping is so short.
MODELS="${BLINGFIRE_MODELS:-gpt2.bin}"

for arg in "$@"; do
  case "$arg" in
    --source)       MODE="source" ;;
    --prebuilt)     MODE="prebuilt" ;;
    --auto)         MODE="auto" ;;
    --models-only)  WANT_LIB=0 ;;
    --no-models)    WANT_MODELS=0 ;;
    --clean)        echo "removing $VENDOR"; rm -rf "$VENDOR"; exit 0 ;;
    -h|--help)      sed -n '2,70p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)              echo "unknown flag: $arg (try --help)" >&2; exit 2 ;;
  esac
done

say() { printf '  %s\n' "$*"; }
step() { printf '\n==> %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------------------
# Host detection
# --------------------------------------------------------------------------

HOST_OS="$(uname -s)"
HOST_ARCH="$(uname -m)"

case "$HOST_OS" in
  Darwin) LIBEXT="dylib" ;;
  Linux)  LIBEXT="so" ;;
  *)      die "unsupported host OS '$HOST_OS'. BlingFire builds on Windows too, but this script only handles macOS and Linux; build blingfiretokdll by hand and drop it in $VENDOR/lib/." ;;
esac

LIBNAME="libblingfiretokdll.$LIBEXT"

# Normalise the arch names the various tools print, so "arm64" (uname on macOS)
# and "aarch64" (uname on Linux, and what `file` says) compare equal.
norm_arch() {
  case "$1" in
    arm64|aarch64|ARM64) echo "aarch64" ;;
    x86_64|amd64|x86-64) echo "x86_64" ;;
    *) echo "$1" ;;
  esac
}

# Architecture of a built library, or empty when it cannot be determined.
lib_arch() {
  local f="$1"
  if [ "$HOST_OS" = "Darwin" ] && command -v lipo >/dev/null 2>&1; then
    norm_arch "$(lipo -archs "$f" 2>/dev/null | tr ' ' '\n' | head -1)"
  elif command -v file >/dev/null 2>&1; then
    case "$(file -b "$f")" in
      *aarch64*|*arm64*) echo "aarch64" ;;
      *x86-64*|*x86_64*) echo "x86_64" ;;
      *) echo "" ;;
    esac
  else
    echo ""
  fi
}

# A universal (fat) macOS binary is fine as long as it contains our slice.
lib_has_host_arch() {
  local f="$1" want; want="$(norm_arch "$HOST_ARCH")"
  if [ "$HOST_OS" = "Darwin" ] && command -v lipo >/dev/null 2>&1; then
    for a in $(lipo -archs "$f" 2>/dev/null); do
      [ "$(norm_arch "$a")" = "$want" ] && return 0
    done
    return 1
  fi
  [ "$(lib_arch "$f")" = "$want" ]
}

fetch() {  # fetch <url> <dest>
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL --retry 3 -o "$2" "$1"
  elif command -v wget >/dev/null 2>&1; then
    wget -q -O "$2" "$1"
  else
    die "need curl or wget"
  fi
}

mkdir -p "$VENDOR/lib" "$VENDOR/include" "$VENDOR/models"

# --------------------------------------------------------------------------
# Shared-library loadability (macOS)
# --------------------------------------------------------------------------
#
# Only matters for the dynamic fallback, i.e. when the static archives are not
# available. A dylib whose install name is a bare filename links fine and then
# fails to LOAD: dyld looks the name up in the standard directories, does not
# find it under engines/blingfire/vendor/lib, and aborts the process. Rewriting
# the id to an absolute path fixes that without anyone having to export
# DYLD_LIBRARY_PATH. No-op everywhere else.

fix_install_name() {
  [ "$HOST_OS" = "Darwin" ] || return 0
  command -v install_name_tool >/dev/null 2>&1 || return 0
  install_name_tool -id "$VENDOR/lib/$LIBNAME" "$VENDOR/lib/$LIBNAME" 2>/dev/null || true
  # Editing load commands invalidates the ad-hoc signature, and dyld refuses to
  # load a library whose signature no longer matches. Re-sign ad-hoc.
  if command -v codesign >/dev/null 2>&1; then
    codesign -f -s - "$VENDOR/lib/$LIBNAME" >/dev/null 2>&1 || true
  fi
  say "install_name -> $VENDOR/lib/$LIBNAME"
}

# --------------------------------------------------------------------------
# Library: prebuilt
# --------------------------------------------------------------------------

try_prebuilt() {
  step "prebuilt: $RAW/dist-pypi/blingfire/$LIBNAME"
  local tmp="$VENDOR/.$LIBNAME.download"
  if ! fetch "$RAW/dist-pypi/blingfire/$LIBNAME" "$tmp"; then
    rm -f "$tmp"
    say "download failed"
    return 1
  fi

  local got; got="$(lib_arch "$tmp")"
  say "downloaded $(wc -c < "$tmp" | tr -d ' ') bytes, arch=${got:-unknown}, host=$(norm_arch "$HOST_ARCH")"

  if [ -n "$got" ] && ! lib_has_host_arch "$tmp"; then
    rm -f "$tmp"
    say "architecture mismatch: Microsoft's prebuilt binaries are x86_64 only"
    return 1
  fi

  mv "$tmp" "$VENDOR/lib/$LIBNAME"
  fix_install_name
  # No static archives on this path — Microsoft ship only the shared library —
  # so build.rs will link dynamically. See fix_install_name() above for why
  # that still loads.
  PROVENANCE="prebuilt $UPSTREAM/blob/$TAG/dist-pypi/blingfire/$LIBNAME (shared only)"
  return 0
}

# --------------------------------------------------------------------------
# Library: CMake build from source
# --------------------------------------------------------------------------

build_from_source() {
  step "source build: $UPSTREAM @ $TAG"
  command -v cmake >/dev/null 2>&1 || die "cmake not found (brew install cmake / apt install cmake)"
  command -v git   >/dev/null 2>&1 || die "git not found"

  local src="$VENDOR/src" build="$VENDOR/build"

  if [ ! -d "$src/.git" ]; then
    rm -rf "$src"
    say "cloning (shallow, single tag)"
    git clone --quiet --depth 1 --branch "$TAG" "$UPSTREAM" "$src"
  else
    say "reusing clone at $src"
  fi

  # BlingFire's CMakeLists declares `cmake_minimum_required(VERSION 3.0)`.
  # CMake 4 removed compatibility with anything below 3.5 and refuses to
  # configure such a project outright; CMAKE_POLICY_VERSION_MINIMUM is the
  # supported escape hatch and is simply an unused variable on CMake 3.x.
  say "configuring"
  cmake -S "$src" -B "$build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        >"$VENDOR/cmake-configure.log" 2>&1 \
    || { tail -30 "$VENDOR/cmake-configure.log" >&2; die "cmake configure failed; full log at $VENDOR/cmake-configure.log"; }

  # ONLY these targets. They link fsaClient and nothing else, so the FSA
  # compiler library and every tool under blingfiretools/ stay uncompiled.
  #
  # blingfiretokdll        shared, for anyone who wants to dlopen/LD_PRELOAD it
  # blingfiretokdll_static + fsaClient   what build.rs actually prefers to link
  say "building blingfiretokdll (+ static archives) with $JOBS jobs"
  cmake --build "$build" \
        --target blingfiretokdll blingfiretokdll_static fsaClient \
        -j "$JOBS" \
        >"$VENDOR/cmake-build.log" 2>&1 \
    || { tail -40 "$VENDOR/cmake-build.log" >&2; die "cmake build failed; full log at $VENDOR/cmake-build.log"; }

  local out
  out="$(find "$build" -name "$LIBNAME" -type f | head -1)"
  [ -n "$out" ] || die "build reported success but $LIBNAME is not in $build"

  cp "$out" "$VENDOR/lib/$LIBNAME"
  say "installed lib/$LIBNAME ($(lib_arch "$VENDOR/lib/$LIBNAME"))"

  # Static archives. Two of them, because CMake does not merge a static
  # library's dependencies into it: blingfiretokdll_static holds the exported
  # C entry points and fsaClient holds the FSA runtime they call.
  local n=0 a
  for a in libblingfiretokdll_static.a libfsaClient.a; do
    out="$(find "$build" -name "$a" -type f | head -1)"
    if [ -n "$out" ]; then
      cp "$out" "$VENDOR/lib/$a"
      say "installed lib/$a ($(wc -c < "$VENDOR/lib/$a" | tr -d ' ') bytes)"
      n=$((n + 1))
    fi
  done
  [ "$n" = 2 ] || say "WARNING: only $n/2 static archives found; build.rs will fall back to dynamic linking"

  fix_install_name

  # The header is not used to generate bindings — src/lib.rs declares the three
  # functions by hand, which is less machinery than bindgen for a 3-function
  # ABI. It is kept so the declarations can be checked against the source of
  # truth without re-cloning.
  cp "$src/blingfiretools/blingfiretokdll/blingfiretokdll.h" "$VENDOR/include/" 2>/dev/null || true

  local sha; sha="$(git -C "$src" rev-parse HEAD)"
  PROVENANCE="source cmake $UPSTREAM @ $TAG ($sha), targets blingfiretokdll{,_static}+fsaClient, $(norm_arch "$HOST_ARCH")"
}

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
#
# BlingFire does not read tokenizer.json. It reads its own compiled FSA image
# (a .bin produced by its offline toolchain), which carries the vocabulary AND
# the id assignment. That makes the model mapping a correctness question, not a
# convenience one: pointing the engine at a .bin whose vocabulary differs from
# the tokenizer.json every other engine loaded would produce a fast row that is
# answering a different question, which the harness would (correctly) flag as
# a mismatch anyway.
#
# So the mapping below is deliberately tiny, and each omission is reasoned:
#
#   gpt2         -> gpt2.bin            GPT-2's own byte-level BPE, 50257 ids.
#                                       The one genuine correspondence.
#   bert-wiki    -> (none)              BlingFire's bert_base_tok.bin is
#                                       bert-base-uncased ([PAD]=0, [CLS]=101,
#                                       "the"=1996). data/models/bert-wiki is a
#                                       wiki-trained WordPiece with a different
#                                       id space ([UNK]=0, [CLS]=1, [SEP]=2,
#                                       [PAD]=3, "the"=7108). Same size, same
#                                       algorithm, different vocabulary.
#   albert       -> (none)              Unigram/30000. BlingFire ships
#                                       xlnet.bin (SentencePiece/32000), which
#                                       is a different model.
#   llama-2/-3, deepseek-v4,
#   mistral-nemo -> (none)              BlingFire ships no .bin for any of them.
#
# To add a mapping you must first establish that the .bin and the
# tokenizer.json are the same vocabulary. The harness will tell you: a wrong
# pairing shows up as `mismatch` against the hf-tokenizers reference.

install_models() {
  step "models: fetching [$MODELS] from $TAG"
  for m in $MODELS; do
    if [ -f "$VENDOR/models/$m" ]; then
      say "$m already present"
    else
      say "fetch $m"
      fetch "$RAW/dist-pypi/blingfire/$m" "$VENDOR/models/$m" \
        || die "could not fetch model '$m' — check the name against $UPSTREAM/tree/$TAG/dist-pypi/blingfire"
    fi
  done

  # Install into the per-model directories the driver hands to build(), under
  # the fixed name the engine looks for.
  step "installing into data/models/<name>/blingfire.bin"
  install_one gpt2 gpt2.bin
}

install_one() {  # install_one <tokbench model dir> <blingfire .bin>
  local model="$1" bin="$2"
  local dir="$REPO_ROOT/data/models/$model"
  if [ ! -d "$dir" ]; then
    say "skip $model: $dir does not exist (run \`make models\` first)"
    return 0
  fi
  if [ ! -f "$VENDOR/models/$bin" ]; then
    say "skip $model: $bin was not fetched"
    return 0
  fi
  cp "$VENDOR/models/$bin" "$dir/blingfire.bin"
  say "$model <- $bin ($(wc -c < "$dir/blingfire.bin" | tr -d ' ') bytes)"
}

# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------

PROVENANCE=""

if [ "$WANT_LIB" = 1 ]; then
  case "$MODE" in
    prebuilt)
      try_prebuilt || die "prebuilt library unusable on this host. Microsoft's binaries are x86_64 only; re-run with --source to build a native one."
      ;;
    source)
      build_from_source
      ;;
    auto)
      try_prebuilt || { say "falling back to a source build"; build_from_source; }
      ;;
  esac

  {
    echo "tag:        $TAG"
    echo "obtained:   $PROVENANCE"
    echo "host:       $HOST_OS $(norm_arch "$HOST_ARCH")"
    echo "library:    lib/$LIBNAME"
    echo "date:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$VENDOR/PROVENANCE"
fi

[ "$WANT_MODELS" = 1 ] && install_models

step "done"
say "library:  $VENDOR/lib/$LIBNAME"
[ -f "$VENDOR/PROVENANCE" ] && say "$(sed -n 2p "$VENDOR/PROVENANCE")"
cat <<EOF

Next:
  cargo run --release -p tokbench --features blingfire,hf-tokenizers -- \\
      --engine blingfire --engine hf-tokenizers

build.rs finds the library at engines/blingfire/vendor/lib automatically. To
link one from elsewhere instead, set BLINGFIRE_LIB_DIR to the directory
containing $LIBNAME.
EOF
