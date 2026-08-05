"""The Rust timing protocol, reproduced in Python.

Engines that only exist as Python libraries (minbpe, mistral-common) are NOT
embedded into the Rust process. They are benchmarked by running this harness,
which repeats `tokbench_core::measure` step for step, and reporting the result
back as JSON on stdout.

Why re-implement instead of embedding through pyo3
--------------------------------------------------
Embedding CPython would put the measurement in the same process and make it
look directly comparable to the native rows -- but it would not be. Every call
would carry a `str` -> Python object conversion and a GIL acquisition
attributable to the *binding*, not to the tokenizer. Running the library the way
its users actually run it, and timing the encode loop from inside Python, gives
the number a Python user would really observe. It is a different measurement
class (`subprocess`) and the dashboard shows it as one; it is never ranked
against in-process rows without that label.

What is excluded and what is not
--------------------------------
Interpreter start-up and imports happen before the timer and are reported as
`load_ms`, exactly like the native engines' construction cost. The timed region
contains the encode calls and nothing else. Python's own per-call overhead IS
included, because a Python user pays it.

To keep the comparison honest, everything below must stay byte-identical in
behaviour to `core/src/lib.rs`:

  * chunking on the same byte boundaries, snapped forward to char boundaries;
  * one untimed warm-up pass, which also captures the ids for verification;
  * `reps` timed passes, median (not mean -- one descheduled pass should not
    move the number);
  * FNV-1a over the ids as little-endian u32, so the hash can be compared
    against the Rust reference to prove the same work was done.
"""

import argparse
import json
import sys
import time
from pathlib import Path


def chunk(text: str, chunk_bytes: int, max_chunks: int) -> list[str]:
    """Mirror of `tokbench_core::chunk`.

    Splits on byte offsets (so both languages cut in the same places) and
    snaps forward off any UTF-8 continuation byte, so a chunk is never invalid
    text. Feeding two engines differently-cut inputs would make them disagree
    for a reason that has nothing to do with the tokenizers.
    """
    raw = text.encode("utf-8")
    out: list[str] = []
    start = 0
    while start < len(raw) and len(out) < max_chunks:
        end = min(start + chunk_bytes, len(raw))
        # 0b10xxxxxx is a continuation byte; advance until a lead byte.
        while end < len(raw) and (raw[end] & 0xC0) == 0x80:
            end += 1
        out.append(raw[start:end].decode("utf-8"))
        start = end
    return out


def ids_hash(ids) -> int:
    """FNV-1a over little-endian u32, identical to `tokbench_core::ids_hash`."""
    h = 0xCBF29CE484222325
    for i in ids:
        v = i & 0xFFFFFFFF
        for shift in (0, 8, 16, 24):
            h ^= (v >> shift) & 0xFF
            h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def median(xs: list[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def args_parser() -> argparse.ArgumentParser:
    """The CLI the Rust driver invokes. Shared so every runner agrees."""
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True, help="model artifact directory")
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--chunk-bytes", type=int, default=10 * 1024)
    p.add_argument("--max-chunks", type=int, default=100)
    p.add_argument("--no-warmup", action="store_true")
    return p


def unsupported(version: str, lang: str, why: str) -> None:
    """Report a cell this engine genuinely cannot run, and exit 0.

    A stated reason is a result. Exiting non-zero would look like a harness
    failure and hide the fact that the engine simply does not support the
    model.
    """
    json.dump(
        {"version": version, "lang": lang, "secs": 0.0, "tokens": 0, "bytes": 0,
         "ids_hash": 0, "load_ms": 0.0, "unsupported": why},
        sys.stdout,
    )
    sys.stdout.write("\n")
    sys.exit(0)


def run(name, version, lang, load, encode, args,
        also_computes="", internally_parallel=False):
    """Time `encode` over the corpus and print the report as JSON.

    `load()` is called first and timed separately into `load_ms`; `encode(text)`
    must return an iterable of int ids and do the minimum the library's public
    API allows. Reaching into private internals to skip work the public path
    performs is out of bounds -- see the fairness notes in core/src/lib.rs.
    """
    text = args.corpus.read_text(encoding="utf-8", errors="replace")
    chunks = chunk(text, args.chunk_bytes, args.max_chunks)
    nbytes = sum(len(c.encode("utf-8")) for c in chunks)

    t0 = time.perf_counter()
    state = load()
    load_ms = (time.perf_counter() - t0) * 1e3

    # Untimed pass: fills caches and captures the ids used for verification.
    # Doing this inside a timed pass would charge the engine for the harness's
    # own bookkeeping.
    all_ids: list[int] = []
    for c in chunks:
        all_ids.extend(encode(state, c))
    tokens = len(all_ids)
    h = ids_hash(all_ids)
    del all_ids

    samples = []
    for _ in range(args.reps):
        t = time.perf_counter()
        for c in chunks:
            encode(state, c)
        samples.append(time.perf_counter() - t)

    json.dump(
        {
            "version": version,
            "lang": lang,
            "secs": median(samples),
            "tokens": tokens,
            "bytes": nbytes,
            "ids_hash": h,
            "load_ms": load_ms,
            "also_computes": also_computes,
            "internally_parallel": internally_parallel,
        },
        sys.stdout,
    )
    sys.stdout.write("\n")


if __name__ == "__main__":
    # Self-check: the chunker must be lossless and the hash order-sensitive,
    # since both are load-bearing for cross-language comparability.
    t = "héllo wörld — 日本語 " * 500
    cs = chunk(t, 1024, 100)
    assert "".join(cs) == t[: len("".join(cs))], "chunking lost or reordered text"
    assert all(isinstance(c, str) for c in cs)
    assert ids_hash([1, 2, 3]) != ids_hash([3, 2, 1])
    # Known-answer: pins the constants so a refactor cannot silently diverge
    # from the Rust implementation.
    assert ids_hash([]) == 0xCBF29CE484222325
    print("harness self-check ok", file=sys.stderr)
