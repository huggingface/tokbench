#!/usr/bin/env python3
"""Derive each engine's required model artifacts from one `tokenizer.json`.

Engines disagree about what a "model" is: tiktoken wants a rank file and a
split regex, bpe-openai wants the name of a prebuilt encoding, SentencePiece
wants a protobuf. If each were downloaded independently there would be no
guarantee they encode the same vocabulary, and the whole comparison would
quietly become "different tokenizers on different vocabularies".

So everything is derived here from the single `tokenizer.json` the reference
engine loads. When a conversion is not sound, this script writes nothing and
the engine reports `Unsupported` — which is the honest outcome, and much better
than an artifact that is subtly wrong and produces a fast but incorrect result.

Usage: scripts/make_artifacts.py data/models
"""

import base64
import json
import sys
from pathlib import Path


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's byte<->unicode table, as used by every ByteLevel tokenizer.

    Byte-level vocabularies store tokens as printable unicode strings; undoing
    that mapping is what turns a vocab entry back into the raw bytes tiktoken
    ranks are keyed on.
    """
    bs = list(range(ord("!"), ord("~") + 1)) \
       + list(range(ord("¡"), ord("¬") + 1)) \
       + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


BYTE_DECODER = {c: b for b, c in bytes_to_unicode().items()}

# Pre-tokenizer regexes. A ByteLevel pre-tokenizer does not store its pattern
# in tokenizer.json -- it is implied -- so the GPT-2 pattern is spelled out.
GPT2_PATTERN = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"

# Vocabularies that are byte-identical to a prebuilt bpe-openai encoding.
# Keyed by vocab size AND a probe token, because size alone collides.
KNOWN_OPENAI = {
    100256: "cl100k_base",
    100277: "cl100k_base",
    199998: "o200k_base",
    200018: "o200k_base",
}


def find_bytelevel(node) -> bool:
    """True if a ByteLevel pre-tokenizer appears anywhere in the config."""
    if isinstance(node, dict):
        if node.get("type") == "ByteLevel":
            return True
        return any(find_bytelevel(v) for v in node.values())
    if isinstance(node, list):
        return any(find_bytelevel(v) for v in node)
    return False


def find_split_regexes(node, out: list[str]) -> None:
    """Collect every explicit `Split` regex in the pre-tokenizer tree."""
    if isinstance(node, dict):
        if node.get("type") == "Split":
            pat = node.get("pattern")
            if isinstance(pat, dict) and "Regex" in pat:
                out.append(pat["Regex"])
        for v in node.values():
            find_split_regexes(v, out)
    elif isinstance(node, list):
        for v in node:
            find_split_regexes(v, out)


def split_pattern(cfg) -> tuple[str | None, str]:
    """The single regex tiktoken should split on, or `None` with a reason.

    tiktoken takes exactly ONE pattern and applies it with a global find. A
    HuggingFace `Sequence` of several `Split` stages applies them in order,
    which is not equivalent to alternating them -- so when a model has more
    than one regex (deepseek-v4 has three), there is no honest single-pattern
    translation and this returns `None`. Writing a joined pattern anyway would
    produce different token ids while still looking like a successful
    conversion, which is the failure mode this whole repo exists to prevent.
    """
    regexes: list[str] = []
    find_split_regexes(cfg.get("pre_tokenizer"), regexes)
    # De-duplicate while preserving order; some configs repeat a stage.
    seen = set()
    regexes = [r for r in regexes if not (r in seen or seen.add(r))]

    if len(regexes) == 1:
        return regexes[0], "model's own Split regex"
    if not regexes:
        if find_bytelevel(cfg):
            # A bare ByteLevel pre-tokenizer does not store a pattern; the
            # GPT-2 pattern is what it implies.
            return GPT2_PATTERN, "implied GPT-2 ByteLevel pattern"
        return None, "no Split regex and no ByteLevel pre-tokenizer"
    return None, f"{len(regexes)} sequential Split regexes — no single-pattern equivalent"


def truncation_check(name: str, model: dict) -> str | None:
    """Reject a vocabulary that has been cut down.

    Never benchmark a truncated vocab. "Slim" test fixtures keep the config
    but drop most of the vocabulary, and the result is not a tokenizer: the
    merge table still references ids that no longer exist. Every number
    measured on one is meaningless, and engines crash on them in ways that
    say nothing about the engine — a truncated glm fixture made tokie 0.1.4
    panic with `index out of bounds: len is 2951 but the index is 27300`,
    which reads like a tokie bug and is not one.

    The tell is a merge whose operands index past the end of the vocabulary.
    """
    vocab = model.get("vocab")
    if not isinstance(vocab, dict) or not vocab:
        return None
    n = len(vocab)
    top = max(vocab.values())

    # A real vocabulary numbers its tokens 0..n-1. A "slim" fixture keeps a
    # sample of entries but their ORIGINAL ids, so the id space is sparse:
    # glm-5.2-slim has 2,951 entries whose ids run to 151,248. Any engine that
    # sizes a table by entry count and indexes it by id then reads out of
    # bounds — which is precisely how tokie panicked.
    #
    # A handful of reserved gaps is normal; two orders of magnitude is not.
    if top + 1 > n * 1.01:
        return (f"{n} vocabulary entries but ids run to {top} — sparse id space, "
                f"i.e. a sampled/truncated vocabulary, not a real one")
    return None


def convert(model_dir: Path) -> None:
    tj = model_dir / "tokenizer.json"
    if not tj.exists():
        print(f"  {model_dir.name}: no tokenizer.json, skipped")
        return

    cfg = json.loads(tj.read_text())
    model = cfg.get("model", {})
    vocab = model.get("vocab")
    # Older tokenizer.json files omit `model.type` entirely; the reference
    # implementation infers BPE from the presence of `merges`, so this must
    # too, or genuinely-convertible models would be skipped.
    mtype = model.get("type") or ("BPE" if "merges" in model else None)

    # Hard stop: a truncated vocabulary must never reach the benchmark.
    if (why := truncation_check(model_dir.name, model)) is not None:
        raise SystemExit(
            f"\nREFUSING to prepare {model_dir.name}: {why}.\n"
            f"Remove {model_dir} and use the real model. Numbers measured on a "
            f"truncated vocabulary are meaningless, and engines fail on them in "
            f"ways that misrepresent the engine.\n"
        )

    if mtype != "BPE" or not isinstance(vocab, dict):
        print(f"  {model_dir.name}: {mtype} — no tiktoken/bpe-openai artifacts (correct: "
              f"those engines do not support this model type)")
        return
    if not find_bytelevel(cfg):
        print(f"  {model_dir.name}: BPE but not ByteLevel — ranks would be wrong, skipped")
        return

    pattern, why = split_pattern(cfg)
    if pattern is None:
        print(f"  {model_dir.name}: no tiktoken artifacts — {why}")
        for stale in ("ranks.tiktoken", "pattern.txt"):
            (model_dir / stale).unlink(missing_ok=True)
        return

    # ranks.tiktoken: "<base64 of raw token bytes> <rank>" per line.
    lines = []
    bad = 0
    for token, rank in vocab.items():
        try:
            raw = bytes(BYTE_DECODER[ch] for ch in token)
        except KeyError:
            # A token containing characters outside the byte-level alphabet is
            # an added/special token; tiktoken handles those separately and
            # including them here would corrupt the ranks.
            bad += 1
            continue
        lines.append(f"{base64.b64encode(raw).decode()} {rank}")
    (model_dir / "ranks.tiktoken").write_text("\n".join(lines) + "\n")
    (model_dir / "pattern.txt").write_text(pattern + "\n")
    note = f" ({bad} non-byte-level tokens excluded)" if bad else ""
    print(f"  {model_dir.name}: ranks.tiktoken {len(lines)} entries{note}, pattern.txt [{why}]")

    enc = KNOWN_OPENAI.get(len(vocab))
    if enc:
        (model_dir / "bpe_openai.txt").write_text(enc + "\n")
        print(f"  {model_dir.name}: bpe_openai.txt -> {enc}")


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/models")
    dirs = sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []
    if not dirs:
        print(f"no model directories under {root}", file=sys.stderr)
        sys.exit(1)
    print(f"deriving engine artifacts for {len(dirs)} model(s):")
    for d in dirs:
        convert(d)


if __name__ == "__main__":
    main()
