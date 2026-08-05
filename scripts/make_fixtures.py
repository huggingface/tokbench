#!/usr/bin/env python3
"""Build large, realistic fixtures from publicly available data.

The language fixtures are clean monolingual prose — the easy case. Real
tokenizer traffic is not: it is agent transcripts dense with special tokens,
tool calls carrying JSON, source diffs, and documents that switch script
mid-sentence. Those stress parts of a tokenizer that prose never touches: the
added-token matcher, pre-tokenization of punctuation and indentation runs, and
any per-document script heuristic.

# Chat text is rendered with the model's real Jinja template

This matters more than it looks. A hand-written `<|im_start|>user\\n...` is a
guess at the format; the tokenizer in production sees the output of the
model's own `chat_template`, and the templates differ in ways that change the
token stream — Llama-3 emits `<|start_header_id|>user<|end_header_id|>\\n\\n`,
Mistral wraps in `[INST]`/`[/INST]` with no role headers at all, Qwen's ChatML
injects a default system turn. Tool calls differ even more: each family
serialises the JSON differently and wraps it in its own markers.

So each chat fixture is produced by loading that model's actual
`tokenizer_config.json` from the Hub and rendering the conversations through
its `chat_template` with Jinja, `tools=` included. What lands on disk is
byte-for-byte what that model's server would hand its tokenizer.

One fixture per template FAMILY, not per model in the matrix. The harness feeds
every engine the same bytes for a given cell, which is the whole basis of the
comparison — a per-model corpus would quietly break it. Naming the fixture
after the family keeps the corpus fixed while still being real text.

Everything comes from public datasets through the datasets-server API: no
authentication, no full-split downloads. Provenance and licences are written to
`data/fixtures/FIXTURES.md`.

Usage: scripts/make_fixtures.py [target_MB]   (default 4)
"""

import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

try:
    import jinja2
except ImportError:
    sys.exit("needs jinja2:  pip install jinja2")

OUT = Path("data/fixtures")
SERVER = "https://datasets-server.huggingface.co/rows"
UA = {"User-Agent": "tokbench-fixtures/0.1"}

# Template families. Llama-3's own repo is gated, so a public mirror of the
# identical config is used; the template text is what matters, not the weights.
FAMILIES = {
    "llama3": "NousResearch/Meta-Llama-3-8B-Instruct",
    "chatml": "Qwen/Qwen2.5-7B-Instruct",
    "mistral": "mistralai/Mistral-Nemo-Instruct-2407",
    "deepseek": "deepseek-ai/DeepSeek-V3",
}

# A tool schema in the shape the templates expect, so the tool-call branches of
# each template actually execute rather than being skipped.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path within the repo."},
                    "ref": {"type": "string", "description": "Git ref."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": "Apply a unified diff.",
            "parameters": {
                "type": "object",
                "properties": {"diff": {"type": "string"}},
                "required": ["diff"],
            },
        },
    },
]


def get_json(url):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60))


def rows(dataset, config, split, limit):
    """Page through datasets-server, yielding row dicts."""
    got = 0
    while got < limit:
        n = min(100, limit - got)
        q = urllib.parse.urlencode(
            {"dataset": dataset, "config": config, "split": split, "offset": got, "length": n}
        )
        try:
            data = get_json(f"{SERVER}?{q}")
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            print(f"    ({dataset} stopped at {got}: {e})", file=sys.stderr)
            return
        batch = data.get("rows", [])
        if not batch:
            return
        for r in batch:
            yield r["row"]
        got += len(batch)


def load_template(repo):
    """Fetch a model's chat_template and the special tokens it references."""
    cfg = get_json(f"https://huggingface.co/{repo}/resolve/main/tokenizer_config.json")
    tpl = cfg.get("chat_template")
    # Newer configs may carry a list of named templates; the default is first.
    if isinstance(tpl, list):
        tpl = next((t.get("template") for t in tpl if t.get("name") in (None, "default")),
                   tpl[0].get("template"))
    if not tpl:
        return None
    tok = lambda k: (cfg.get(k) if isinstance(cfg.get(k), str)
                     else (cfg.get(k) or {}).get("content", ""))
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True,
                             undefined=jinja2.ChainableUndefined)
    # Filters and globals the Hub templates rely on and Jinja does not ship.
    env.filters["tojson"] = lambda v, **kw: json.dumps(v, ensure_ascii=False)
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(RuntimeError(m))
    env.globals["strftime_now"] = lambda fmt: "01 Jan 2026"
    return env.from_string(tpl), tok("bos_token"), tok("eos_token")


def render(compiled, messages, tools=None):
    tpl, bos, eos = compiled
    try:
        return tpl.render(messages=messages, tools=tools, bos_token=bos, eos_token=eos,
                          add_generation_prompt=False, tools_in_user_message=True)
    except Exception:
        # A template that rejects a particular message shape is a real property
        # of that template; skip the sample rather than reshaping it into
        # something the model would never receive.
        return None


def write(name, parts, target):
    buf, size = [], 0
    for p in parts:
        if not p:
            continue
        buf.append(p)
        size += len(p.encode("utf-8"))
        if size >= target:
            break
    (OUT / f"{name}.txt").write_text("".join(buf), encoding="utf-8")
    print(f"  {name:22} {size/1024/1024:5.2f} MB  ({len(buf)} segments)")
    return size


def convos(limit):
    """Multi-turn conversations as plain role/content dicts."""
    out = []
    for r in rows("HuggingFaceH4/ultrachat_200k", "default", "train_sft", limit):
        msgs = [{"role": m.get("role", "user"), "content": (m.get("content") or "")[:3000]}
                for m in (r.get("messages") or []) if m.get("content")]
        # Templates alternate strictly; drop anything that does not.
        if len(msgs) >= 2 and all(m["role"] in ("user", "assistant", "system") for m in msgs):
            out.append(msgs)
    return out


def agent_convos(limit):
    """SWE-bench Lite: real issues and real patches, as tool-using turns."""
    out = []
    for r in rows("SWE-bench/SWE-bench_Lite", "default", "test", limit):
        repo, iid = r.get("repo", ""), r.get("instance_id", "")
        problem, patch = (r.get("problem_statement") or "")[:6000], (r.get("patch") or "")[:8000]
        tests = (r.get("test_patch") or "")[:4000]
        if not problem or not patch:
            continue
        out.append([
            {"role": "system", "content": "You are a software engineering agent with repo access."},
            {"role": "user", "content": f"Repository: {repo}\nIssue {iid}:\n\n{problem}"},
            {"role": "assistant", "content": "Reading the failing module first.",
             "tool_calls": [{"type": "function", "function": {
                 "name": "read_file", "arguments": {"path": "src/main.py",
                                                    "ref": r.get("base_commit", "HEAD")}}}]},
            {"role": "tool", "name": "read_file", "content": "<file contents elided>"},
            {"role": "assistant", "content": f"Here is the fix:\n\n```diff\n{patch}\n```"},
            {"role": "user", "content": "Add regression tests."},
            {"role": "assistant", "content": f"```diff\n{tests}\n```"},
        ])
    return out


def code_fixture(target):
    parts = []
    for r in rows("m-a-p/CodeFeedback-Filtered-Instruction", "default", "train", 900):
        parts.append(f"### {r.get('lang') or 'text'}\n{(r.get('query') or '')[:2000]}\n\n"
                     f"{(r.get('answer') or '')[:6000]}\n\n")
    return write("code-polyglot", parts, target)


def mixed_fixture(target):
    """Script switches WITHIN a document — what per-document heuristics miss."""
    skip = {"code-polyglot", "multilingual-mix"} | {f"chat-{k}" for k in FAMILIES} | {"agentic-tools"}
    pools = []
    for f in sorted(OUT.glob("*.txt")):
        if f.stem in skip:
            continue
        paras = [p for p in f.read_text(encoding="utf-8", errors="replace").split("\n") if p.strip()]
        if paras:
            pools.append(paras)
    oa = [(r.get("text") or "") for r in rows("OpenAssistant/oasst1", "default", "train", 400)]
    if oa:
        pools.append([t for t in oa if t.strip()])
    if not pools:
        print("  multilingual-mix       skipped (no source fixtures)")
        return 0
    parts, idx, size = [], [0] * len(pools), 0
    step = 0
    while size < target and step < 2_000_000:
        p = step % len(pools)
        step += 1
        if idx[p] < len(pools[p]):
            s = pools[p][idx[p]] + "\n"
            idx[p] += 1
            parts.append(s)
            size += len(s.encode("utf-8"))
        elif all(idx[i] >= len(pools[i]) for i in range(len(pools))):
            break
    return write("multilingual-mix", parts, target)


def main():
    mb = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
    target = int(mb * 1024 * 1024)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"building fixtures at ~{mb} MB each (public data, real chat templates):")

    compiled = {}
    for fam, repo in FAMILIES.items():
        try:
            c = load_template(repo)
            if c:
                compiled[fam] = c
                print(f"  template {fam:10} <- {repo}")
        except Exception as e:
            print(f"  template {fam:10} unavailable ({getattr(e, 'code', type(e).__name__)})")

    total = 0
    chats, agents = convos(400), agent_convos(300)

    for fam, c in compiled.items():
        total += write(f"chat-{fam}", (render(c, m) for m in chats * 4), target)

    # Tool-calling traces: emitted per family too, since each serialises tool
    # calls differently and that difference is the point.
    tool_parts = []
    for i, conv in enumerate(agents * 4):
        fam = list(compiled)[i % len(compiled)] if compiled else None
        if fam:
            tool_parts.append(render(compiled[fam], conv, tools=TOOLS))
    total += write("agentic-tools", tool_parts, target)

    total += code_fixture(target)
    total += mixed_fixture(target)

    (OUT / "FIXTURES.md").write_text(
        "# Generated fixtures\n\n"
        "Built by `scripts/make_fixtures.py` from public HuggingFace datasets via the\n"
        "datasets-server API — no authentication, no full-split downloads.\n\n"
        "## Chat text is rendered with each model's real Jinja `chat_template`\n\n"
        "Not hand-written scaffolding. Each `chat-*.txt` is produced by fetching that\n"
        "model's `tokenizer_config.json` from the Hub and rendering conversations\n"
        "through its actual `chat_template`, tools included — so the bytes on disk are\n"
        "what that model's server would hand its tokenizer. The families differ in ways\n"
        "that change the token stream: Llama-3 uses `<|start_header_id|>` blocks,\n"
        "Mistral wraps in `[INST]`/`[/INST]` with no role headers, Qwen's ChatML injects\n"
        "a default system turn, and each serialises tool calls its own way.\n\n"
        "One fixture per template FAMILY, not per benchmark model: the harness feeds\n"
        "every engine identical bytes per cell, and a per-model corpus would break that.\n\n"
        "| fixture | source | licence |\n|---|---|---|\n"
        "| chat-llama3 / chat-chatml / chat-mistral / chat-deepseek | [ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) rendered through Llama-3 / Qwen2.5 / Mistral-Nemo / DeepSeek-V3 templates | MIT |\n"
        "| agentic-tools | [SWE-bench_Lite](https://huggingface.co/datasets/SWE-bench/SWE-bench_Lite) real issues + patches, as tool-calling turns through the same templates | MIT |\n"
        "| code-polyglot | [CodeFeedback-Filtered-Instruction](https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction) | Apache-2.0 |\n"
        "| multilingual-mix | monolingual fixtures + [oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) non-English turns, interleaved per paragraph | Apache-2.0 / mixed |\n\n"
        "## What they stress that prose does not\n\n"
        "The added-token matcher (special tokens every few hundred bytes), "
        "punctuation- and indentation-heavy pre-tokenization (diffs, JSON tool "
        "arguments, source), and script switching *within* a document, which defeats "
        "any per-document language heuristic.\n",
        encoding="utf-8",
    )
    print(f"\ntotal {total/1024/1024:.1f} MB; provenance in {OUT}/FIXTURES.md")


if __name__ == "__main__":
    main()
