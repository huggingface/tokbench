#!/usr/bin/env python3
"""Convert a byte-level BPE `tokenizer.json` into karpathy/minbpe's `.model`.

Why this exists
---------------
minbpe cannot read `tokenizer.json`. Without a converter the only minbpe cell we
could ever fill is `GPT4Tokenizer`, which hardcodes cl100k_base -- a vocabulary
no other row in the matrix uses. That would make the minbpe number a curiosity
instead of a measurement: you could not tell whether it was slow because the
algorithm is naive or because it was doing a different amount of work on a
different vocabulary.

Converting instead puts minbpe on the *same* vocabulary as every other engine,
so its ids can be verified against the reference and its time can be read as
the cost of the naive algorithm and nothing else. That is the whole point of
having minbpe in the matrix: it is the floor, the calibration row.

The `.model` format (from `minbpe/base.py`, `Tokenizer.save`/`load`)
-------------------------------------------------------------------
    minbpe v1
    <split pattern>
    <number of special tokens>
    <special> <id>          x number of special tokens
    <id1> <id2>             one line per merge, in merge order

`load()` is the binding constraint, not `save()`. It assigns merge ids itself::

    idx = 256
    for line in f:
        merges[(idx1, idx2)] = idx
        idx += 1

so a merge's id is fixed by its *line number*: merge k is always token 256 + k.
Leaf ids are equally fixed -- `_build_vocab` seeds `vocab[i] = bytes([i])` for
i in 0..255 and `_encode_chunk` starts from `list(text_bytes)`, so byte b is
always token b. minbpe's numbering is therefore not a free parameter, and a
vocabulary whose ids disagree with it cannot be expressed in the file. See
`minbpe.idmap.json` below for how the runner closes that gap.

What is checked before anything is written
------------------------------------------
A wrong `.model` is worse than no `.model`: it produces plausible-looking ids
that are quietly incorrect, and it would take a verification failure elsewhere
to notice. So every property minbpe's format silently assumes is asserted here,
and a violation refuses the conversion with the reason rather than emitting a
best-effort file.

  1. the model is BPE and carries `vocab` + `merges`;
  2. there is no normalizer (minbpe applies none);
  3. the pre-tokenizer is a *single* regex over byte-level text, and that regex
     survives a round trip through one line of a text file;
  4. all 256 byte-level alphabet tokens are present (a SentencePiece-style
     vocabulary has no such alphabet and cannot be represented at all);
  5. every merge operand is a byte token or an *earlier* merge's product, so
     each merge line is writable when it is reached;
  6. no two merges produce the same token. This is the check that rejects
     llama-3 and mistral-nemo. Their `merges` lists reach one vocabulary entry
     by several different pairs -- normal and correct for a HF BPE, where a
     merge maps to a vocabulary id -- but minbpe mints a *new* id per merge
     line, so the same token would be handed two different ids and every later
     merge keyed on it would miss. There is no faithful encoding of that;
  7. special tokens contain no whitespace (`load` splits the line on it) and do
     not collide with a merge id.

`minbpe.idmap.json`
-------------------
minbpe numbers byte b as token b. GPT-2 numbers the byte tokens in byte-encoder
order instead -- `Ġ` (byte 32) is id 220, `!` (byte 33) is id 0. The two agree
on all 50000 merge tokens and disagree on all 256 byte tokens, so minbpe's raw
output is the correct *segmentation* in the wrong *numbering*, and hashing it
would report a mismatch that is not a tokenization difference.

The sidecar carries the permutation so the runner can restate minbpe's ids in
the reference's numbering. The runner does that translation inside the timed
region -- minbpe is charged for it, never credited.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# The GPT-2 split regex, used when the pre-tokenizer is a bare `ByteLevel`
# (which applies this pattern internally when `use_regex` is not false).
GPT2_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Unfaithful(Exception):
    """A property minbpe's format requires does not hold. Refuse to write."""


def byte_to_unicode() -> dict[int, str]:
    """GPT-2's byte -> printable-char alphabet (`bytes_to_unicode`).

    Byte-level BPE vocabularies store tokens as text, so every byte needs a
    printable stand-in; this is the mapping that produces `Ġ` for space.
    """
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def split_pattern(tj: dict) -> str:
    """The one regex minbpe will be given, or refuse.

    minbpe's `encode_ordinary` is a single `re.findall`. Anything the reference
    expresses as a *cascade* of splits is a different pre-tokenization, not a
    slower one, so it is rejected rather than approximated.
    """
    pt = tj.get("pre_tokenizer")
    if pt is None:
        raise Unfaithful("no pre-tokenizer; minbpe always splits on a regex")

    if pt.get("type") == "ByteLevel":
        if pt.get("use_regex") is False:
            raise Unfaithful("ByteLevel with use_regex=false does not split at all")
        if pt.get("add_prefix_space"):
            raise Unfaithful("ByteLevel add_prefix_space=true; minbpe has no such step")
        return GPT2_PATTERN

    if pt.get("type") == "Sequence":
        subs = pt["pretokenizers"]
        splits = [s for s in subs if s.get("type") == "Split"]
        others = [s for s in subs if s.get("type") != "Split"]
        if len(splits) != 1:
            raise Unfaithful(
                f"pre-tokenizer is a cascade of {len(splits)} Split stages applied in "
                "sequence; minbpe applies exactly one regex and cannot compose them"
            )
        if any(s.get("type") != "ByteLevel" for s in others):
            raise Unfaithful(
                f"unsupported pre-tokenizer stages {[s.get('type') for s in others]}"
            )
        for s in others:
            if s.get("add_prefix_space"):
                raise Unfaithful("ByteLevel add_prefix_space=true; minbpe has no such step")
        sp = splits[0]
        if sp.get("behavior") != "Isolated" or sp.get("invert"):
            raise Unfaithful(
                f"Split behavior={sp.get('behavior')} invert={sp.get('invert')}; "
                "only Isolated/non-inverted matches minbpe's re.findall"
            )
        pat = sp["pattern"].get("Regex")
        if pat is None:
            raise Unfaithful("Split uses a literal/String pattern, not a Regex")
        return pat

    raise Unfaithful(f"pre-tokenizer type {pt.get('type')!r} is not a byte-level regex split")


def convert(model_dir: Path) -> tuple[str, dict]:
    """Return the `.model` text and the id-map sidecar, or raise `Unfaithful`."""
    tj = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8"))

    if tj.get("normalizer") is not None:
        raise Unfaithful(
            f"tokenizer has a {tj['normalizer'].get('type')} normalizer; minbpe "
            "normalizes nothing, so ids would diverge before the model even runs"
        )

    model = tj["model"]
    if "merges" not in model or "vocab" not in model:
        raise Unfaithful(
            f"model type {model.get('type')!r} has no merges; minbpe is byte-level BPE only"
        )
    for field, why in (("continuing_subword_prefix", "prefix"), ("end_of_word_suffix", "suffix")):
        if model.get(field):
            raise Unfaithful(f"model sets {field}={model[field]!r}; minbpe has no word {why}")
    if model.get("byte_fallback"):
        raise Unfaithful("model uses byte_fallback; minbpe's alphabet is already the 256 bytes")
    if model.get("dropout"):
        raise Unfaithful("model uses BPE dropout, which is stochastic")

    vocab: dict[str, int] = model["vocab"]
    merges = model["merges"]

    pattern = split_pattern(tj)
    if "\n" in pattern or "\r" in pattern:
        raise Unfaithful("split regex contains a newline; it cannot survive the one-line format")
    if pattern != pattern.strip():
        # `load()` strips the pattern line, so padding would be silently dropped.
        raise Unfaithful("split regex has leading/trailing whitespace that load() would strip")

    enc = byte_to_unicode()
    missing = [b for b in range(256) if enc[b] not in vocab]
    if missing:
        raise Unfaithful(
            f"{len(missing)} of the 256 byte-level alphabet tokens are absent; this is not "
            "a byte-level vocabulary (SentencePiece-style models cannot be represented)"
        )

    # minbpe id of every token we can name. Byte b is token b, by construction.
    minbpe_id: dict[str, int] = {enc[b]: b for b in range(256)}
    byte_ids = [vocab[enc[b]] for b in range(256)]  # reference id of byte b

    lines: list[str] = []
    merge_ref_ids: list[int] = []
    seen: set[str] = set()
    for k, mg in enumerate(merges):
        a, b = mg.split(" ") if isinstance(mg, str) else (mg[0], mg[1])
        product = a + b
        if a not in minbpe_id or b not in minbpe_id:
            missing_op = a if a not in minbpe_id else b
            raise Unfaithful(
                f"merge {k} ({mg!r}) uses operand {missing_op!r}, which is neither a byte "
                "token nor the product of an earlier merge; the merge list is not in "
                "dependency order and cannot be replayed"
            )
        if product in seen:
            raise Unfaithful(
                f"merge {k} ({mg!r}) re-derives {product!r}, already produced by an earlier "
                "merge. minbpe mints a new id per merge line, so this token would be given "
                "two different ids and every later merge keyed on it would miss. A HF BPE "
                "may legitimately reach one vocabulary entry by several pairs; minbpe's "
                "format cannot express that"
            )
        if product not in vocab:
            raise Unfaithful(f"merge {k} ({mg!r}) produces {product!r}, absent from the vocab")
        seen.add(product)
        lines.append(f"{minbpe_id[a]} {minbpe_id[b]}")
        minbpe_id[product] = 256 + k
        merge_ref_ids.append(vocab[product])

    # Special tokens: everything in the vocab that no merge and no byte produced.
    n_merges = len(lines)
    merge_id_range = range(256, 256 + n_merges)
    specials: dict[str, int] = {}
    for tok, tid in vocab.items():
        if tok in minbpe_id:
            continue
        if any(ch.isspace() for ch in tok):
            raise Unfaithful(
                f"special token {tok!r} contains whitespace; minbpe's load() splits the "
                "special-token line on whitespace and would mis-parse it"
            )
        if tid in merge_id_range:
            raise Unfaithful(
                f"special token {tok!r} has id {tid}, which collides with minbpe's merge "
                f"id range 256..{255 + n_merges}"
            )
        specials[tok] = tid

    body = ["minbpe v1", pattern, str(len(specials))]
    body += [f"{tok} {tid}" for tok, tid in specials.items()]
    body += lines
    text = "\n".join(body) + "\n"

    # The runner needs the reference numbering back. Merge ids usually already
    # agree (merge k *is* token 256+k in a canonically-trained vocabulary), so
    # only store the exception.
    identity_merges = merge_ref_ids == list(merge_id_range)
    idmap = {
        "source": "tokenizer.json",
        "pattern": pattern,
        "n_merges": n_merges,
        "vocab_size": len(vocab),
        "n_special": len(specials),
        # reference id of byte b, for b in 0..255
        "byte_ids": byte_ids,
        "merges_identity": identity_merges,
    }
    if not identity_merges:
        idmap["merge_ids"] = merge_ref_ids
    return text, idmap


def remap_table(idmap: dict) -> list[int]:
    """minbpe id -> reference id, as a flat list the runner can index."""
    if idmap["merges_identity"]:
        merge_ids = range(256, 256 + idmap["n_merges"])
    else:
        merge_ids = idmap["merge_ids"]
    return list(idmap["byte_ids"]) + list(merge_ids)


def verify(model_dir: Path, idmap: dict, chars: int) -> bool:
    """Encode real corpus text with both minbpe and the reference and compare.

    The validation in `convert` proves the *file* is well-formed; only this
    proves the ids are right. A converter that is merely plausible is exactly
    the thing this script is supposed to not ship.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    try:
        from minbpe import RegexTokenizer
    except ImportError:
        print("  verify: SKIPPED (minbpe not importable)", file=sys.stderr)
        return True
    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("  verify: SKIPPED (`tokenizers` not installed, no reference to compare against)",
              file=sys.stderr)
        return True

    # `Tokenizer.load()` sets `self.pattern` but never recompiles
    # `self.compiled_pattern`, so a loaded tokenizer would keep whatever the
    # constructor compiled. Pass the pattern to the constructor instead.
    tok = RegexTokenizer(pattern=idmap["pattern"])
    tok.load(str(model_dir / "minbpe.model"))
    ref = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    table = remap_table(idmap)

    fixtures = sorted((REPO / "data" / "fixtures").glob("*.txt"))
    ok = True
    for f in fixtures:
        sample = f.read_text(encoding="utf-8", errors="replace")[:chars]
        got = [table[i] for i in tok.encode(sample)]
        want = ref.encode(sample, add_special_tokens=False).ids
        if got == want:
            print(f"  verify {f.name:<12} ok   ({len(want)} ids)")
        else:
            ok = False
            first = next((i for i, (g, w) in enumerate(zip(got, want)) if g != w), min(len(got), len(want)))
            print(f"  verify {f.name:<12} MISMATCH  minbpe={len(got)} ref={len(want)} ids, "
                  f"first differs at {first}: {got[first:first + 6]} != {want[first:first + 6]}")
    return ok


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", type=Path, action="append", default=[],
                   help="a data/models/<name> directory; repeatable")
    p.add_argument("--all", action="store_true", help="every directory under data/models")
    p.add_argument("--verify", action="store_true",
                   help="check ids against the HF reference on the corpus fixtures")
    p.add_argument("--verify-chars", type=int, default=4000,
                   help="prefix of each fixture to verify (minbpe is slow; keep it small)")
    args = p.parse_args()

    dirs = list(args.model)
    if args.all or not dirs:
        dirs = sorted(d for d in (REPO / "data" / "models").iterdir()
                      if d.is_dir() and (d / "tokenizer.json").exists())

    failures = 0
    for d in dirs:
        print(f"{d.name}:")
        try:
            text, idmap = convert(d)
        except Unfaithful as e:
            print(f"  unsupported -- {e}")
            continue
        except Exception as e:  # noqa: BLE001 - report, do not emit a maybe-wrong model
            print(f"  ERROR {type(e).__name__}: {e}")
            failures += 1
            continue
        (d / "minbpe.model").write_text(text, encoding="utf-8")
        (d / "minbpe.idmap.json").write_text(json.dumps(idmap), encoding="utf-8")
        print(f"  wrote minbpe.model ({idmap['n_merges']} merges, {idmap['n_special']} special, "
              f"vocab {idmap['vocab_size']}, merge ids "
              f"{'match' if idmap['merges_identity'] else 'DO NOT match'} the reference)")
        if args.verify and not verify(d, idmap, args.verify_chars):
            failures += 1
            # A model that does not reproduce the reference must not be left on
            # disk for the runner to pick up.
            (d / "minbpe.model").unlink()
            (d / "minbpe.idmap.json").unlink()
            print("  removed minbpe.model: it did not reproduce the reference ids")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
