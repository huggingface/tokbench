#!/usr/bin/env python3
"""Build `data/models/<name>/model.gguf` for the llamacpp engine.

Why this script exists
----------------------
llama.cpp does not read `tokenizer.json`. It reads a vocabulary out of a GGUF:
a token list, a token-type list, optional merges, and a *name*
(`tokenizer.ggml.pre`) that selects one of the pre-tokenizer regexes compiled
into the C++. So the llamacpp engine cannot be pointed at the same artifact as
every other engine in the matrix; it needs its own file.

The obvious shortcut -- download `Meta-Llama-3-8B-Q4_K_M.gguf` off the Hub and
point the engine at it -- is exactly the thing that must not happen. It would
produce a fast, plausible, `verified`-looking row measuring a vocabulary no
other engine in the run was given. So this script derives the GGUF from
`data/models/<name>/tokenizer.json`, the same file the reference engine loads,
and then reads the result back and checks token-for-token that the two agree.
The comparison is only meaningful if the vocabulary is literally the same one.

How
---
`convert_hf_to_gguf.py` is llama.cpp's own converter and the only sane way to
produce the `tokenizer.ggml.*` keys -- in particular `tokenizer.ggml.pre`,
which it derives by hashing a probe tokenization and looking the digest up in a
table. Guessing that name by eyeballing the regex in `tokenizer.json` is how
you get a tokenizer that is subtly wrong on whitespace and right on everything
else. The converter is downloaded (with the matching `gguf-py`) at the tag of
the *installed* llama.cpp, so the writer and the reader are the same version.

The converter wants a HuggingFace model directory, and `data/models/<name>/`
holds only a tokenizer. So each model is staged into a temp dir with its real
`tokenizer.json` plus a synthetic minimal `config.json`. The synthetic hparams
(layer count, hidden size, ...) land in the GGUF and are never read: with
`--vocab-only` there are no tensors, and the engine loads with
`vocab_only = true`, which makes llama.cpp parse the `tokenizer.ggml.*` keys
and skip everything else. What must be real is the vocabulary, and that is
copied byte-for-byte from `tokenizer.json`.

Two details in that staging are load-bearing, both learned the hard way:

* `tokenizer_config.json` pins `tokenizer_class` to `PreTrainedTokenizerFast`.
  Without it, `AutoTokenizer` picks a class from `config.json`'s `model_type`
  and that class contributes *its own* default special tokens. Staging llama-3
  under a gpt2 `model_type` silently appended `<|endoftext|>` as id 128256 --
  a 128257th token that exists in no llama-3 vocabulary anywhere. The neutral
  class loads `tokenizer.json` and nothing else.
* `vocab_size` is computed from `tokenizer.json` rather than left to default.
  The converter asserts `max(id) < vocab_size` and defaults the bound to the
  count of *base* tokens, which is smaller than the highest id whenever a model
  has added tokens (llama-3: 128000 base, ids up to 128255).

Usage
-----
    python engines/llamacpp/scripts/make_gguf.py              # every known model
    python engines/llamacpp/scripts/make_gguf.py gpt2 llama-3
    python engines/llamacpp/scripts/make_gguf.py --force gpt2 # rebuild

Requires: python with `numpy`, `torch` and `sentencepiece` importable (the
converter imports them at module scope even for `--vocab-only`).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MODELS = REPO / "data" / "models"
CACHE = Path(os.environ.get("TOKBENCH_CACHE", Path.home() / ".cache" / "tokbench"))

# Synthetic configs, one per model in the matrix.
#
# `arch` picks the converter class, which picks the *vocab* path -- that is the
# only thing it decides that matters here:
#   GPT2LMHeadModel -> _set_vocab_gpt2, the generic BPE path (token list +
#                      merges + a `pre` name resolved by digest). Correct for
#                      every byte-level BPE tokenizer regardless of which model
#                      it came from, because llama.cpp's vocabulary loader
#                      reads `tokenizer.ggml.*` and never looks at
#                      `general.architecture`.
#   BertModel       -> the WordPiece path (`tokenizer.ggml.model = "bert"`,
#                      "##" continuations folded into phantom-space tokens).
#
# The hparams are placeholders and are documented as such above; they exist
# because the converter calls `set_gguf_parameters()` even for --vocab-only.
CONFIGS: dict[str, dict] = {
    "gpt2": {
        "architectures": ["GPT2LMHeadModel"],
        "model_type": "gpt2",
        "n_ctx": 1024, "n_positions": 1024, "n_embd": 768,
        "n_head": 12, "n_layer": 12, "layer_norm_epsilon": 1e-5,
    },
    "llama-3": {
        "architectures": ["GPT2LMHeadModel"],
        "model_type": "gpt2",
        "n_ctx": 8192, "n_positions": 8192, "n_embd": 4096,
        "n_head": 32, "n_layer": 32, "layer_norm_epsilon": 1e-5,
    },
    "deepseek-v4": {
        "architectures": ["GPT2LMHeadModel"],
        "model_type": "gpt2",
        "n_ctx": 4096, "n_positions": 4096, "n_embd": 4096,
        "n_head": 32, "n_layer": 30, "layer_norm_epsilon": 1e-6,
    },
    "mistral-nemo": {
        "architectures": ["GPT2LMHeadModel"],
        "model_type": "gpt2",
        "n_ctx": 8192, "n_positions": 8192, "n_embd": 5120,
        "n_head": 32, "n_layer": 40, "layer_norm_epsilon": 1e-5,
    },
    # llama-2 is the one model here whose tokenizer is SentencePiece, not
    # byte-level BPE, so it must NOT go down the gpt2 path: that would label it
    # `tokenizer.ggml.model = "gpt2"` and llama.cpp would run the wrong
    # tokenizer type over a correct vocabulary. LlamaForCausalLM reaches the
    # SentencePiece path instead.
    "llama-2": {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32,
        "num_key_value_heads": 32, "intermediate_size": 11008,
        "max_position_embeddings": 4096, "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
    },
    "bert-wiki": {
        "architectures": ["BertModel"],
        "model_type": "bert",
        "hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12,
        "intermediate_size": 3072, "max_position_embeddings": 512,
        "layer_norm_eps": 1e-12, "type_vocab_size": 2,
    },
}

# albert is deliberately absent: it is a SentencePiece Unigram model, and
# llama.cpp's converter reaches that vocabulary through `spiece.model`, which
# `data/models/albert/` does not contain. Adding a hand-rolled Unigram writer
# would be inventing a vocabulary rather than converting one. The engine
# reports Unsupported for it, which is the honest cell.


def log(msg: str) -> None:
    print(msg, flush=True)


def installed_tag() -> str:
    """Release tag of the llama.cpp this machine will *load* the GGUF with.

    Converting with a different version than you read with is how you get a
    file that loads but tokenizes differently, so the converter is pinned to
    the installed library, not to master.
    """
    if tag := os.environ.get("LLAMA_CPP_TAG"):
        return tag
    try:
        out = subprocess.run(
            ["pkg-config", "--modversion", "llama"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        if out.startswith("0.0."):
            return "b" + out[4:]
    except (OSError, subprocess.CalledProcessError):
        pass
    return "b9140"


def ensure_converter(tag: str) -> Path:
    """Fetch `convert_hf_to_gguf.py` + `gguf-py` at `tag` into the cache.

    The two must travel together: the standalone script tracks master and calls
    into `gguf` constants that a pip-installed `gguf` release may not have yet
    (observed: `MODEL_ARCH.GEMMA4` missing -> AttributeError at import). The
    script puts its sibling `gguf-py` on `sys.path` itself, so extracting both
    into one directory is all that is needed.
    """
    root = CACHE / f"llama.cpp-{tag}"
    script = root / "convert_hf_to_gguf.py"
    if script.is_file() and (root / "gguf-py").is_dir():
        return script

    CACHE.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/ggml-org/llama.cpp/archive/refs/tags/{tag}.tar.gz"
    tarball = CACHE / f"{tag}.tar.gz"
    log(f"fetch  llama.cpp {tag} converter ({url})")
    urllib.request.urlretrieve(url, tarball)
    with tarfile.open(tarball) as tf:
        wanted = [
            m for m in tf.getmembers()
            if m.name.endswith("/convert_hf_to_gguf.py")
            or f"/gguf-py/" in m.name
        ]
        tf.extractall(CACHE, members=wanted)
    tarball.unlink(missing_ok=True)
    if not script.is_file():
        raise SystemExit(f"converter not found in {tag} tarball")
    log(f"       converter ready at {script}")
    return script


def gguf_tokens(path: Path, gguf_py: Path) -> list[str]:
    """Read the token list back out of a GGUF, using the same gguf-py."""
    sys.path.insert(0, str(gguf_py))
    import gguf  # noqa: E402  (path has to be set up first)

    reader = gguf.GGUFReader(path, "r")
    field = reader.get_field("tokenizer.ggml.tokens")
    return [str(bytes(field.parts[i]), encoding="utf-8") for i in field.data]


def json_vocab(tokenizer_json: Path) -> dict[int, str]:
    """`tokenizer.json`'s base vocabulary as id -> token."""
    vocab = json.loads(tokenizer_json.read_text())["model"]["vocab"]
    if isinstance(vocab, dict):  # BPE / WordPiece
        return {i: tok for tok, i in vocab.items()}
    return {i: entry[0] for i, entry in enumerate(vocab)}  # Unigram


def vocab_size(tokenizer_json: Path) -> int:
    """Highest id in the file, plus one -- base tokens and added tokens both.

    This is the bound `get_vocab_base` asserts against, and getting it from the
    tokenizer rather than from a hand-written constant is what keeps the
    staged config honest when a model's added-token block moves.
    """
    doc = json.loads(tokenizer_json.read_text())
    ids = list(json_vocab(tokenizer_json))
    ids += [t["id"] for t in doc.get("added_tokens", [])]
    return max(ids) + 1


def phantom(token: str) -> str:
    """llama.cpp's WordPiece spelling of a token.

    `BertModel.set_vocab` rewrites the vocabulary on the way in: continuations
    lose their `##`, everything else gains a U+2581. Replaying that here keeps
    the verification exact for BERT instead of skipping it.
    """
    return token[2:] if token.startswith("##") else "▁" + token


def build(name: str, script: Path, force: bool) -> tuple[bool, str]:
    model_dir = MODELS / name
    tokenizer_json = model_dir / "tokenizer.json"
    out = model_dir / "model.gguf"

    if not tokenizer_json.is_file():
        return False, "no tokenizer.json"
    if out.is_file() and not force:
        return True, f"exists ({out.stat().st_size / 1e6:.1f} MB), --force to rebuild"
    cfg = CONFIGS.get(name)
    if cfg is None:
        return False, "no config template (see CONFIGS in this script)"

    with tempfile.TemporaryDirectory(prefix=f"tokbench-{name}-") as tmp:
        stage = Path(tmp)
        # The vocabulary comes from OUR tokenizer.json. Nothing else does.
        shutil.copy2(tokenizer_json, stage / "tokenizer.json")
        (stage / "config.json").write_text(
            json.dumps({**cfg, "vocab_size": vocab_size(tokenizer_json)})
        )
        # Neutral tokenizer class: load tokenizer.json, invent nothing.
        (stage / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"})
        )

        proc = subprocess.run(
            [sys.executable, str(script), "--vocab-only",
             "--outfile", str(out), str(stage)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            out.unlink(missing_ok=True)
            tail = [ln for ln in proc.stderr.strip().splitlines() if ln.strip()]
            return False, "converter failed: " + (tail[-1] if tail else "?")

    # Verify. The claim being checked is the one the whole row depends on:
    # every id in this model's tokenizer.json means the same token in the GGUF.
    # Ids above the base vocabulary (added/special tokens, and the [PAD*] holes
    # the converter fills gaps with) are counted but not compared -- the
    # converter legitimately re-normalizes those, and none of them can be
    # produced by `parse_special = false` tokenization anyway.
    try:
        got = gguf_tokens(out, script.parent / "gguf-py")
        want = json_vocab(tokenizer_json)
        if cfg["architectures"][0] == "BertModel":
            want = {i: phantom(t) for i, t in want.items()}
    except Exception as exc:  # noqa: BLE001 - report, do not mask
        return False, f"wrote {out.name} but could not verify it: {exc}"

    if len(got) <= max(want):
        out.unlink(missing_ok=True)
        return False, (f"VOCAB MISMATCH: gguf holds {len(got)} tokens but "
                       f"tokenizer.json uses ids up to {max(want)}")
    bad = [i for i, tok in want.items() if got[i] != tok]
    if bad:
        out.unlink(missing_ok=True)
        i = min(bad)
        return False, (f"VOCAB MISMATCH at {len(bad)} of {len(want)} base ids, "
                       f"first id {i}: gguf {got[i]!r} != tokenizer.json {want[i]!r}")

    return True, (f"{len(want)} base ids verified against tokenizer.json "
                  f"(+{len(got) - len(want)} added/unused), "
                  f"{out.stat().st_size / 1e6:.1f} MB")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", help="model names under data/models (default: all known)")
    ap.add_argument("--force", action="store_true", help="rebuild even if model.gguf exists")
    args = ap.parse_args()

    names = args.models or [n for n in CONFIGS if (MODELS / n / "tokenizer.json").is_file()]
    if not names:
        log(f"no models found under {MODELS}; run `make models` first")
        return 1

    tag = installed_tag()
    script = ensure_converter(tag)
    log(f"llama.cpp {tag} converter; {len(names)} model(s): {', '.join(names)}")

    ok = 0
    t0 = time.time()
    for i, name in enumerate(names, 1):
        t = time.time()
        good, msg = build(name, script, args.force)
        ok += good
        done = time.time() - t0
        eta = done / i * (len(names) - i)
        log(f"[{i}/{len(names)}] {name:<14} {'ok  ' if good else 'FAIL'} {msg}"
            f" | {time.time() - t:.1f}s | eta ~{eta:.0f}s")

    log(f"{ok}/{len(names)} model.gguf ready under {MODELS}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
