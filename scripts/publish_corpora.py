#!/usr/bin/env python3
"""Publish the tokbench corpora as a public HuggingFace dataset.

One repo, two layouts built from the same bytes:

  fixtures/<corpus>.txt          what `make fixtures` downloads. Byte-identical
                                 to what the benchmark chunks -- the driver
                                 slices at fixed 10 KiB offsets from byte 0, so
                                 a re-encoded file is a different measurement.
  data/<corpus>/train-*.parquet  one row per document, for `load_dataset`.

The two cannot drift: rows are `text.split(sep)` at the corpus's own boundary,
`sep.join(rows)` is the file again, and that is checked against the source
sha256 before anything uploads. `sep` rides along as a column, so the identity
holds without reading this script.

Corpora whose licence or provenance does not clearly permit redistribution are
in BLOCKED and are never uploaded. Resolve the reason, then move the entry.

  scripts/publish_corpora.py --repo-id <owner>/<name> [--dry-run]
"""

import argparse
import hashlib
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "data" / "fixtures"
CARD = Path(__file__).with_name("corpora_card.md")

# Row boundary per corpus. Blank line is the document separator everywhere
# except the added-token fixtures, which are one generated line per line.
SEP = "\n\n"
SEPS = {
    "added-special-dense": "\n",
    "added-special-sparse": "\n",
    "added-normalized-dense": "\n",
    "added-normalized-sparse": "\n",
}

# A corpus name is the .txt stem, the config name, and the `--corpus` argument.
# One name everywhere. Provenance and licence: see the card.
CORPORA = [
    "amharic", "arabic", "bengali", "chinese", "english", "georgian", "greek",
    "hebrew", "hindi", "japanese", "korean", "russian", "tamil", "thai",
    "agentic-swe", "math-latex",
    "added-special-dense", "added-special-sparse",
    "added-normalized-dense", "added-normalized-sparse",
    "chat-llama3", "chat-chatml", "chat-mistral", "chat-deepseek",
]

# Not published. Each reason is a fact to resolve, not a formality.
BLOCKED = {
    "agentic-traces": "provenance unrecorded upstream (synthetic agent traces, builder unknown)",
    "code-mixed": "verbatim redis source (RSALv2/SSPLv1/AGPLv3) and junit5 (EPL-2.0)",
    "multilingual-mix": "interleaves code-mixed and agentic-traces paragraphs verbatim",
    "agentic-tools": "SWE-bench_Lite declares no licence; patches include pylint (GPL-2.0)",
    "xnli": "facebook/xnli declares no licence",
    "code": "no recorded builder or source",
    "dense": "no recorded builder; not in the benchmark set",
    "code-polyglot": "empty file",
}


def human(n):
    return f"{n / 1024 / 1024:5.2f} MB"


def build(out, selected):
    import pyarrow as pa
    import pyarrow.parquet as pq

    (out / "fixtures").mkdir(parents=True, exist_ok=True)
    (out / ".gitattributes").write_text(
        "*.txt filter=lfs diff=lfs merge=lfs -text\n"
        "*.parquet filter=lfs diff=lfs merge=lfs -text\n"
    )

    total, t0, rows_all, bytes_all = len(selected), time.time(), 0, 0
    for i, name in enumerate(selected, 1):
        raw = (FIXTURES / f"{name}.txt").read_bytes()
        sep = SEPS.get(name, SEP)
        docs = raw.decode("utf-8").split(sep)
        table = pa.table(
            {
                "corpus": pa.array([name] * len(docs)),
                "index": pa.array(range(len(docs)), pa.int32()),
                "sep": pa.array([sep] * len(docs)),
                "text": pa.array(docs),
            }
        )
        if sep.join(table.column("text").to_pylist()).encode("utf-8") != raw:
            sys.exit(f"{name}: parquet round-trip is not byte-exact -- refusing to publish")

        shard = out / "data" / name
        shard.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, shard / "train-00000-of-00001.parquet", compression="zstd")
        shutil.copyfile(FIXTURES / f"{name}.txt", out / "fixtures" / f"{name}.txt")

        rows_all += len(docs)
        bytes_all += len(raw)
        elapsed = time.time() - t0
        eta = elapsed / i * (total - i)
        print(
            f"  [{i:2d}/{total}] {name:24} {human(len(raw))}  {len(docs):6d} rows  ok"
            f"  | elapsed {elapsed:4.1f}s | eta ~{eta:4.1f}s",
            flush=True,
        )
    return rows_all, bytes_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--dry-run", action="store_true", help="build and verify, upload nothing")
    ap.add_argument("--private", action="store_true", help="create the repo private")
    ap.add_argument("--out", type=Path, help="staging dir (default: a temp dir)")
    args = ap.parse_args()

    present = {p.stem for p in FIXTURES.glob("*.txt")}
    selected = [n for n in CORPORA if n in present]
    missing = [n for n in CORPORA if n not in present]
    unknown = present - set(CORPORA) - set(BLOCKED)

    print(f"corpora in {FIXTURES}: {len(present)} .txt")
    for name, why in BLOCKED.items():
        if name in present:
            print(f"  skip  {name:24} {why}")
    for name in missing:
        print(f"  skip  {name:24} not on disk (run `make fixtures`)")
    for name in sorted(unknown):
        print(f"  skip  {name:24} unknown corpus -- add it to CORPORA or BLOCKED")
    if not selected:
        sys.exit("nothing to publish")

    card = CARD.read_text()
    declared = set(re.findall(r"^  - config_name: (\S+)", card, re.M))
    want = set(CORPORA) | {"default"}  # CORPORA is a list; set() takes the names
    if declared != want:
        sys.exit(
            f"{CARD} configs are out of date\n"
            f"  missing from card: {sorted(want - declared)}\n"
            f"  stale in card:     {sorted(declared - want)}"
        )
    if missing and not args.dry_run:
        sys.exit(f"run `make fixtures` first -- the card promises {missing}")

    out = args.out or Path(tempfile.mkdtemp(prefix="tokbench-corpora-"))
    out.mkdir(parents=True, exist_ok=True)
    print(f"\nbuilding {len(selected)} configs into {out}")
    rows, size = build(out, selected)
    (out / "README.md").write_text(card)
    print(f"\n{len(selected)} configs, {rows} rows, {human(size)} of text; card -> README.md")

    if args.dry_run:
        print(f"dry run: nothing uploaded. Inspect {out}")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(out),
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message=f"publish {len(selected)} tokbench corpora",
        delete_patterns=["data/**", "fixtures/**"],
    )
    print(f"pushed to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
