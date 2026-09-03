#!/usr/bin/env python3
"""mistral-common — Mistral's official tokenizer (Tekken / SentencePiece).

Run by the Rust driver through the shared protocol in `python/harness.py`.

mistral-common is the reference implementation for Mistral models, so its ids
are authoritative *for those models* — if it disagrees with the HF reference on
a Mistral vocabulary, that is a finding about the conversion, not necessarily a
bug in mistral-common. The report records the mismatch either way and does not
adjudicate; it just refuses to present the two speeds as comparable.

Encoding is requested with `add_bos=False, add_eos=False` to match the
`add_special_tokens = false` the reference engine is called with.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
import harness  # noqa: E402


def version() -> str:
    try:
        from importlib.metadata import version as v
        return f"mistral-common {v('mistral-common')}"
    except Exception:
        return "mistral-common (unknown version)"


def main() -> None:
    args = harness.args_parser().parse_args()
    ver = version()

    tekken = args.model / "tekken.json"
    if not tekken.exists():
        harness.unsupported(
            ver, "python",
            "no tekken.json (mistral-common reads Tekken/SentencePiece artifacts, "
            "not tokenizer.json)",
        )

    try:
        from mistral_common.tokens.tokenizers.tekken import Tekkenizer
    except ImportError:
        harness.unsupported(ver, "python", "mistral-common not installed (pip install mistral-common)")

    def load():
        return Tekkenizer.from_file(str(tekken))

    harness.run(
        name="mistral-common",
        version=ver,
        lang="python",
        load=load,
        # Positional: `Tekkenizer.encode(s, bos, eos)`. Both False, to match
        # the `add_special_tokens = false` the reference is called with.
        encode=lambda t, text: t.encode(text, False, False),
        args=args,
    )


if __name__ == "__main__":
    main()
