#!/usr/bin/env python3
"""minbpe (karpathy) — the pure-Python reference BPE.

Run by the Rust driver through the shared protocol in `python/harness.py`.

This engine is in the matrix as the *floor*, not as a competitor. minbpe is
deliberately simple, unoptimised, pure-Python teaching code, and its author
never claimed otherwise. Its value here is calibration: it shows what the
naive algorithm costs, which is the only way to tell how much of a fast
engine's win comes from algorithms rather than from engineering.

Two honest caveats the report should carry with any minbpe number:

* It is slow enough to dominate a run. `RegexTokenizer.encode` is O(text x
  merges) in interpreted Python, so a 1 MB corpus can take minutes where the
  Rust engines take milliseconds. Cap the corpus with `--max-chunks` when
  including it.
* It cannot read `tokenizer.json`. It loads its own `.model` format, or, via
  `GPT4Tokenizer`, replicates cl100k_base. Anything else is reported
  unsupported rather than approximated.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
import harness  # noqa: E402

VERSION = "minbpe @ karpathy/minbpe (unversioned)"


def main() -> None:
    args = harness.args_parser().parse_args()

    try:
        from minbpe import GPT4Tokenizer, RegexTokenizer
    except ImportError:
        harness.unsupported(
            VERSION, "python",
            "minbpe not installed (pip install git+https://github.com/karpathy/minbpe)",
        )

    own = args.model / "minbpe.model"
    marker = args.model / "bpe_openai.txt"

    if own.exists():
        def load():
            t = RegexTokenizer()
            t.load(str(own))
            return t
    elif marker.exists() and marker.read_text().strip() == "cl100k_base":
        # GPT4Tokenizer reproduces cl100k_base exactly, so its ids are
        # verifiable against the reference for cl100k-equivalent models.
        def load():
            return GPT4Tokenizer()
    else:
        harness.unsupported(
            VERSION, "python",
            "minbpe reads its own .model format; no minbpe.model artifact and "
            "the model is not cl100k_base-equivalent",
        )

    harness.run(
        name="minbpe",
        version=VERSION,
        lang="python",
        load=load,
        encode=lambda t, text: t.encode(text),
        args=args,
    )


if __name__ == "__main__":
    main()
