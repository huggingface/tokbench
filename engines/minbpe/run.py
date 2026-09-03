#!/usr/bin/env python3
"""minbpe (karpathy) — the pure-Python reference BPE.

Run by the Rust driver through the shared protocol in `python/harness.py`.

This engine is in the matrix as the *floor*, not as a competitor. minbpe is
deliberately simple, unoptimised, pure-Python teaching code, and its author
never claimed otherwise. Its value here is calibration: it shows what the
naive algorithm costs, which is the only way to tell how much of a fast
engine's win comes from algorithms rather than from engineering.

Two honest caveats the report should carry with any minbpe number:

* It is slow enough to dominate a run, and its cost is quadratic in
  pre-token length: `_encode_chunk` recounts every adjacent pair in the
  pre-token to pick each single merge, so a pre-token of length L costs O(L^2)
  dict lookups. Ordinary prose is safe because byte-level pre-tokens are short;
  a corpus with long unsplittable runs (base64 blobs, minified data) is not.
  See "How slow" below for measured rates and the recommended cap.
* It cannot read `tokenizer.json`. It loads its own `.model` format, or, via
  `GPT4Tokenizer`, replicates cl100k_base. Anything else is reported
  unsupported rather than approximated.

Getting it to run at all
------------------------
minbpe is not on PyPI and has no `setup.py`/`pyproject.toml`, so there is
nothing to `pip install`. It is a plain package directory, vendored here::

    git clone --depth 1 https://github.com/karpathy/minbpe engines/minbpe/src
    python3 -m venv engines/minbpe/.venv
    engines/minbpe/.venv/bin/pip install regex tiktoken   # minbpe's imports

Both paths are gitignored. This file puts `src/` on `sys.path` and re-executes
itself under `.venv/bin/python` when one exists, so the driver's fixed
`python3 engines/minbpe/run.py` invocation works without a global install and
without polluting the system interpreter.

The vocabulary, and why ids need translating
--------------------------------------------
`make_minbpe_model.py` converts a byte-level BPE `tokenizer.json` into
`data/models/<name>/minbpe.model`, so minbpe runs on the *same* vocabulary as
every other engine and its ids can be verified rather than taken on trust. It
refuses to write a model it cannot prove faithful; today only gpt2 converts,
and the refusal reason is reported for the rest.

One gap has to be closed at run time. minbpe's numbering is not a free
parameter: `_encode_chunk` starts from `list(text_bytes)`, so byte b is always
token b. GPT-2 numbers its byte tokens in byte-encoder order instead (`Ġ`, byte
32, is id 220). The two agree on all 50000 merge tokens and disagree on all 256
byte tokens, so minbpe's raw output is the right segmentation in the wrong
numbering. `minbpe.idmap.json` carries the permutation and the list
comprehension below restates the ids in the reference's numbering.

That translation runs *inside* the timed region, so minbpe is charged for it
and never credited. It is not always free: measured against the un-remapped
loop over the 200 KB fixtures it costs 0.01% on english.txt and 0.02% on
dense.txt, but 1.7% on code.txt and 4.0% on korean.txt. The pattern is that the
cost scales with *tokens emitted* while the work it rides on scales with
*merges applied*, so the overhead peaks exactly where gpt2's vocabulary covers
the script badly and output is token-dense. 4% is the honest worst case to
subtract if a reader wants minbpe's bare encode rate.

How slow, and what to cap
-------------------------
Measured single-threaded, gpt2 vocabulary, 10 KiB chunks, median of 5 reps over
the full 200 KB fixtures, on an M-series laptop:

    chinese 0.40 MB/s   english 0.81 MB/s   dense 0.91 MB/s
    code    1.37 MB/s   korean  2.10 MB/s

Slowest is chinese, not korean, which is the opposite of the intuition that CJK
is hard: gpt2's vocabulary barely covers Korean, so most pre-tokens exhaust the
merge table almost immediately and fall out of the loop. Chinese has enough
coverage to keep merging. minbpe's cost tracks *merges applied*, not bytes.

So the practical cap is milder than "minbpe is pure Python" suggests. At the
slowest measured rate a full 200 KB fixture is ~0.5 s per rep, and the harness
default `--max-chunks 100` (1 MiB) bounds a rep to ~2.5 s. That default is fine
for these fixtures. Use `--max-chunks 2` (~20 KiB, well under 0.1 s per rep)
when iterating, or if a corpus contains long unsplittable runs, where the
quadratic term above stops making bytes a useful predictor of time.
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _bootstrap() -> None:
    """Make `import minbpe` work under the driver's plain `python3`.

    The vendored checkout goes on `sys.path`; minbpe's own imports (`regex`,
    and `tiktoken` for `GPT4Tokenizer`) may still be missing, and the local
    venv exists to supply them.

    The interpreter is only swapped when the current one genuinely cannot
    import minbpe. The driver exposes `--python` so an interpreter can be
    chosen deliberately, and re-executing unconditionally would silently
    discard that choice -- the reported number would be for an interpreter the
    caller did not ask for, which is the kind of quiet substitution this
    benchmark exists to avoid.
    """
    src = HERE / "src"
    if src.is_dir():
        sys.path.insert(0, str(src))
    try:
        import minbpe  # noqa: F401
        return
    except ImportError:
        pass
    py = HERE / ".venv" / "bin" / "python"
    # The guard stops an exec loop if the venv interpreter cannot satisfy the
    # import either; the caller then gets the honest `unsupported` reason.
    if py.exists() and not os.environ.get("_TOKBENCH_MINBPE_VENV"):
        os.environ["_TOKBENCH_MINBPE_VENV"] = "1"
        os.execv(str(py), [str(py), str(Path(__file__).resolve()), *sys.argv[1:]])


_bootstrap()

import json  # noqa: E402

sys.path.insert(0, str(HERE.parents[1] / "python"))
import harness  # noqa: E402


def version() -> str:
    """Pin the vendored checkout, since minbpe publishes no version number."""
    head = HERE / "src" / ".git" / "HEAD"
    try:
        ref = head.read_text().strip()
        if ref.startswith("ref: "):
            ref = (HERE / "src" / ".git" / ref[5:]).read_text().strip()
        return f"minbpe @ karpathy/minbpe {ref[:7]}"
    except OSError:
        return "minbpe @ karpathy/minbpe (unversioned)"


def main() -> None:
    args = harness.args_parser().parse_args()
    ver = version()

    try:
        from minbpe import GPT4Tokenizer, RegexTokenizer
    except ImportError as e:
        harness.unsupported(
            ver, "python",
            f"minbpe not importable ({e}); it is not on PyPI -- clone it into "
            "engines/minbpe/src (see the module docstring)",
        )

    own = args.model / "minbpe.model"
    idmap_path = args.model / "minbpe.idmap.json"
    marker = args.model / "bpe_openai.txt"

    if own.exists() and idmap_path.exists():
        idmap = json.loads(idmap_path.read_text())
        if idmap["merges_identity"]:
            merge_ids = range(256, 256 + idmap["n_merges"])
        else:
            merge_ids = idmap["merge_ids"]
        table = list(idmap["byte_ids"]) + list(merge_ids)

        def load():
            # `Tokenizer.load()` assigns `self.pattern` but never recompiles
            # `self.compiled_pattern`, so a tokenizer built with the default
            # constructor would keep splitting on minbpe's GPT-4 pattern no
            # matter what the file says. Hand the pattern to the constructor,
            # which is the documented way to override it.
            t = RegexTokenizer(pattern=idmap["pattern"])
            t.load(str(own))
            return t, table

        # `encode_ordinary` rather than `encode`: the reference is called with
        # `add_special_tokens = false`, under which HF tokenises a literal
        # "<|endoftext|>" as ordinary text. `encode`'s default is "none_raise",
        # which would instead assert. For text containing no special token --
        # every fixture -- `encode` reaches `encode_ordinary` anyway, so this
        # is the same work with the matching semantics.
        def encode(state, text):
            tok, tbl = state
            return [tbl[i] for i in tok.encode_ordinary(text)]

    elif marker.exists() and marker.read_text().strip() == "cl100k_base":
        # GPT4Tokenizer reproduces cl100k_base exactly, so its ids are
        # verifiable against the reference for cl100k-equivalent models. It
        # already emits reference ids, so no translation is needed.
        def load():
            return GPT4Tokenizer(), None

        def encode(state, text):
            return state[0].encode_ordinary(text)

    else:
        why = "no minbpe.model artifact; run engines/minbpe/make_minbpe_model.py"
        if (args.model / "tokenizer.json").exists():
            # Say *why* the converter declined, so the cell is a finding rather
            # than a shrug.
            try:
                sys.path.insert(0, str(HERE))
                from make_minbpe_model import Unfaithful, convert
                try:
                    convert(args.model)
                    why = ("minbpe.model not generated yet; run "
                           "engines/minbpe/make_minbpe_model.py")
                except Unfaithful as e:
                    why = f"not representable in minbpe's .model format: {e}"
            except Exception:  # noqa: BLE001 - fall back to the generic reason
                pass
        harness.unsupported(ver, "python", why)

    harness.run(
        name="minbpe",
        version=ver,
        lang="python",
        load=load,
        encode=encode,
        args=args,
    )


if __name__ == "__main__":
    main()
