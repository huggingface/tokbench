#!/usr/bin/env python3
"""Record the published package size of every engine.

"How big is this dependency" is a real selection criterion, and it is not the
same question as "how much does it add to my binary" (that is
`scripts/binsize.sh`). A crate can be a small download that pulls a large
dependency tree, or a large download that compiles to very little. Both
columns are reported; neither substitutes for the other.

Sizes come from each ecosystem's own registry, so they are the numbers a user
actually downloads:

  * crates.io -- the `.crate` tarball (gzipped source) for the exact pinned
    version, plus the same for any second crate an engine needs.
  * PyPI      -- the wheel for the newest release, falling back to the sdist.
  * npm       -- the packed tarball, plus `unpackedSize` when the registry
    reports it.

Versions are pinned here to match what the engines actually build against.
An unpinned size would drift away from the benchmark it is printed next to.

Writes package_sizes.json, which the driver merges into the report.
"""

import json
import sys
import urllib.error
import urllib.request

UA = {"User-Agent": "tokbench/0.1 (https://github.com/huggingface/tokbench)"}

# engine folder -> where its code comes from.
#   ("crates", [(name, version), ...])  sizes are summed
#   ("pypi",   package)
#   ("npm",    package)
#   ("source", note)  -- vendored C/C++, no published package to weigh
ENGINES = {
    "hf-tokenizers":  ("crates", [("tokenizers", "0.23.1")]),
    "fastokens":      ("crates", [("fastokens", "0.3.1")]),
    "kitoken":        ("crates", [("kitoken", "0.11.0")]),
    "splintr":        ("crates", [("splintr", "0.19.1")]),
    "tokie":          ("crates", [("tokie", "0.1.4")]),
    "tiktoken":       ("crates", [("tiktoken-rs", "0.12.0")]),
    "rust-gems-bpe":  ("crates", [("bpe", "0.2.1"), ("bpe-openai", "0.3.0")]),
    "wordchipper":    ("crates", [("wordchipper", "0.9.2")]),
    "sentencepiece":  ("crates", [("sentencepiece", "0.14.0")]),
    "blingfire":      ("crates", [("blingfire", "1.0.0")]),
    "llamacpp":       ("crates", [("llama-cpp-2", "0.1.154")]),
    "gigatoken":      ("pypi",   "gigatoken"),
    "executorch":     ("pypi",   "pytorch-tokenizers"),
    "mistral-common": ("pypi",   "mistral-common"),
    "ai-tokenizer":   ("npm",    "ai-tokenizer"),
    "pipeline":       ("source", "tokenizers#2279 poc/target-encode - unreleased branch"),
    "minbpe":         ("source", "karpathy/minbpe - git only, no published package"),
    "iree":           ("source", "iree-org/iree - C source, built from the IREE tree"),
}


def get_json(url: str):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        return json.load(r)


def crate_size(name: str, version: str) -> int:
    """Size of the published `.crate` tarball, in bytes."""
    data = get_json(f"https://crates.io/api/v1/crates/{name}/{version}")
    return int(data["version"]["crate_size"])


def pypi_size(name: str) -> tuple[int, str]:
    data = get_json(f"https://pypi.org/pypi/{name}/json")
    version = data["info"]["version"]
    urls = data["urls"]
    # Prefer a wheel: it is what `pip install` actually fetches.
    wheels = [u for u in urls if u["packagetype"] == "bdist_wheel"]
    chosen = wheels or urls
    if not chosen:
        raise ValueError("no distributions")
    # Wheels are per-platform; report the largest so the number is not an
    # accidental best case from a pure-Python stub.
    return max(int(u["size"]) for u in chosen), version


def npm_size(name: str) -> tuple[int, str, int | None]:
    data = get_json(f"https://registry.npmjs.org/{name}")
    version = data["dist-tags"]["latest"]
    dist = data["versions"][version]["dist"]
    return int(dist.get("fileCount", 0) and dist["unpackedSize"] or 0) or 0, version, dist.get(
        "unpackedSize"
    )


def main() -> None:
    out: dict[str, dict] = {}
    total = len(ENGINES)
    for i, (engine, spec) in enumerate(ENGINES.items(), 1):
        kind = spec[0]
        prefix = f"[{i}/{total}] {engine:<16}"
        try:
            if kind == "crates":
                parts = spec[1]
                size = sum(crate_size(n, v) for n, v in parts)
                ver = ", ".join(f"{n} {v}" for n, v in parts)
                out[engine] = {"kb": round(size / 1024, 1), "version": ver,
                               "registry": "crates.io"}
            elif kind == "pypi":
                size, ver = pypi_size(spec[1])
                out[engine] = {"kb": round(size / 1024, 1), "version": f"{spec[1]} {ver}",
                               "registry": "pypi"}
            elif kind == "npm":
                _, ver, unpacked = npm_size(spec[1])
                size = unpacked or 0
                out[engine] = {"kb": round(size / 1024, 1), "version": f"{spec[1]} {ver}",
                               "registry": "npm (unpacked)"}
            else:
                out[engine] = {"kb": None, "version": spec[1], "registry": "source"}
                print(f"{prefix} n/a  ({spec[1]})", file=sys.stderr)
                continue
            print(f"{prefix} {out[engine]['kb']:>9} kB  ({out[engine]['version']})",
                  file=sys.stderr)
        except (urllib.error.HTTPError, urllib.error.URLError, KeyError, ValueError) as e:
            out[engine] = {"kb": None, "version": None, "registry": kind, "error": str(e)}
            print(f"{prefix} ERROR {e}", file=sys.stderr)

    with open("package_sizes.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote package_sizes.json", file=sys.stderr)


if __name__ == "__main__":
    main()
