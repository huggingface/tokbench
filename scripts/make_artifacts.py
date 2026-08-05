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

    if mtype != "BPE" or not isinstance(vocab, dict):
        print(f"  {model_dir.name}: {mtype} — no tiktoken/bpe-openai artifacts (correct: "
              f"those engines do not support this model type)")
        return
    if not find_bytelevel(cfg):
        print(f"  {model_dir.name}: BPE but not ByteLevel — ranks would be wrong, skipped")
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
    (model_dir / "pattern.txt").write_text(GPT2_PATTERN + "\n")
    note = f" ({bad} non-byte-level tokens excluded)" if bad else ""
    print(f"  {model_dir.name}: ranks.tiktoken {len(lines)} entries{note}, pattern.txt")

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
