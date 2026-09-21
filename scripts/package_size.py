#!/usr/bin/env python3
"""Published package size per engine, from each registry, into
package_sizes.json. A different question from `scripts/binsize.sh`: a small
download can pull a large dependency tree, and the reverse. Versions are
pinned to what the engines build against.
"""

import json
import sys
import urllib.error
import urllib.request

UA = {"User-Agent": "tokbench/0.1 (https://github.com/huggingface/tokbench)"}

# engine folder -> where its code comes from.
#   ("crates", [(name, version), ...])  sizes are summed
#   ("pypi",   package)
#   ("source", note)  -- built from source, no published package to weigh
ENGINES = {
    "hf-tokenizers":  ("crates", [("tokenizers", "0.23.1")]),
    "fastokens":      ("crates", [("fastokens", "0.3.1")]),
    "kitoken":        ("crates", [("kitoken", "0.11.0")]),
    "tokie":          ("crates", [("tokie", "0.1.4")]),
    "tiktoken":       ("crates", [("tiktoken-rs", "0.12.0")]),
    "rust-gems-bpe":  ("crates", [("bpe", "0.2.1"), ("bpe-openai", "0.3.0")]),
    "wordchipper":    ("crates", [("wordchipper", "0.9.2")]),
    "sentencepiece":  ("crates", [("sentencepiece", "0.14.0")]),
    "llamacpp":       ("crates", [("llama-cpp-2", "0.1.154")]),
    "gigatoken":      ("pypi",   "gigatoken"),
    "executorch":     ("pypi",   "pytorch-tokenizers"),
    "pipeline":       ("source", "tokenizers 1.0.0-rc.0 @ 5c3727a9 - unreleased source"),
    "iree":           ("source", "iree-org/iree - C source, built from the IREE tree"),
}


def get_json(url: str):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        return json.load(r)


def crate_size(name: str, version: str) -> int:
    data = get_json(f"https://crates.io/api/v1/crates/{name}/{version}")
    return int(data["version"]["crate_size"])


def pypi_size(name: str) -> tuple[int, str]:
    data = get_json(f"https://pypi.org/pypi/{name}/json")
    version = data["info"]["version"]
    urls = data["urls"]
    # A wheel is what `pip install` fetches; the largest, since they are
    # per-platform and the smallest may be a pure-Python stub.
    chosen = [u for u in urls if u["packagetype"] == "bdist_wheel"] or urls
    if not chosen:
        raise ValueError("no distributions")
    return max(int(u["size"]) for u in chosen), version


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
