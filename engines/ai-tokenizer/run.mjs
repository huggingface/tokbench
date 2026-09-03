#!/usr/bin/env node
// ai-tokenizer (npm) -- a JS/TS BPE tokenizer targeting the Vercel AI SDK.
//
// Run by the Rust driver. The protocol below is a line-for-line reproduction of
// `core/src/lib.rs` and `python/harness.py`: same byte-offset chunking snapped
// to char boundaries, one untimed warm-up pass that also captures the ids for
// verification, `reps` timed passes, median, and FNV-1a over little-endian u32.
// It is duplicated rather than shared because it is the only JS consumer, and a
// shared module for one caller would be indirection without a payer.
//
// Node start-up and module resolution happen before the timer and land in
// `load_ms`, so what is timed is the encode loop -- but this is still the
// `subprocess` class and the dashboard labels it as such. JIT warm-up is real
// and is exactly what the warm-up pass exists to absorb.
//
// ---------------------------------------------------------------------------
// Which vocabulary this engine loads, and why
// ---------------------------------------------------------------------------
//
// ai-tokenizer's public surface is `new Tokenizer(encoding)` -> `.encode(text)`
// returning `number[]`. There are two ways to get an `encoding`:
//
//   1. One of the four it ships: `cl100k_base`, `o200k_base`, `p50k_base`,
//      `claude`. NONE of them is any model in `data/models`. gpt2 is r50k
//      (50,257), which the package does not ship at all -- p50k_base is a
//      different vocabulary (50,281), not a rename. llama-3 (128,256) and
//      mistral-nemo (131,072) have no OpenAI equivalent either. Benchmarking
//      gpt2 against p50k_base would be measuring a different vocabulary and
//      publishing it under the wrong name, so that path is not taken.
//
//   2. The `Encoding` object the constructor is typed to accept
//      (`dist/index.d.ts`): `{name, pat_str, special_tokens, stringEncoder,
//      binaryEncoder, decoder}`. That is public API, not an internal, and it
//      is what this adapter builds -- from `ranks.tiktoken` + `pattern.txt`,
//      the SAME artifacts the native `tiktoken` engine consumes and which
//      `scripts/make_artifacts.py` derives from the SAME `tokenizer.json` the
//      reference engine loads.
//
// The point of (2) is that it is checkable, and it was checked: over gpt2 x all
// ten fixtures this adapter reproduces the reference `ids_hash` exactly, cell
// for cell. If the vocabulary build were wrong the hash would diverge and the
// driver would mark the cell mismatched rather than publish a bogus number.
//
// ai-tokenizer cannot read `tokenizer.json`, so models without those artifacts
// report `unsupported` with the reason instead of being silently substituted.
//
// Fairness notes specific to this path:
//
//   * Building the encoding parses a text rank file at start-up, which costs
//     more than importing one of the package's pre-baked encoding modules.
//     Both are load, both are untimed, both land in `load_ms`.
//   * `special_tokens` is left empty on purpose. With no special tokens
//     `Tokenizer.encode` goes straight down its `encodeOrdinary` path, which
//     is the same work the native tiktoken adapter measures via
//     `encode_ordinary`. It also avoids `encode`'s default `disallowedSpecial:
//     "all"`, which throws if the corpus happens to contain a special-token
//     string -- a corpus-dependent exception, not a tokenizer measurement.
//   * `decoder` is left empty because the encode path never reads it; only
//     `decode()` does, and `decode()` is not benchmarked. Populating it would
//     add load time and resident memory for nothing.
//
// One known limitation, in the library rather than in this adapter, disclosed
// because it bounds what the green cells prove: ai-tokenizer resolves a byte
// slice by decoding it with a plain `new TextDecoder("utf-8")`, which strips a
// leading U+FEFF. A slice whose bytes begin EF BB BF therefore resolves to the
// rank of the BOM-less token instead of its own. That can only be reached if
// the input text itself contains U+FEFF; none of `data/fixtures` does, which is
// why every cell verifies. Text with an embedded BOM would tokenize wrongly,
// and the id hash would say so rather than hide it.

import { readFileSync } from "node:fs";
import { existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

function parseArgs() {
  const a = process.argv.slice(2);
  const get = (k, d) => {
    const i = a.indexOf(`--${k}`);
    return i >= 0 ? a[i + 1] : d;
  };
  return {
    model: get("model"),
    corpus: get("corpus"),
    reps: parseInt(get("reps", "5"), 10),
    chunkBytes: parseInt(get("chunk-bytes", String(10 * 1024)), 10),
    maxChunks: parseInt(get("max-chunks", "100"), 10),
  };
}

// Mirror of tokbench_core::chunk -- cut on byte offsets, then walk forward off
// any UTF-8 continuation byte (0b10xxxxxx) so no chunk is invalid text.
function chunk(buf, chunkBytes, maxChunks) {
  const dec = new TextDecoder("utf-8");
  const out = [];
  let start = 0;
  while (start < buf.length && out.length < maxChunks) {
    let end = Math.min(start + chunkBytes, buf.length);
    while (end < buf.length && (buf[end] & 0xc0) === 0x80) end++;
    out.push(dec.decode(buf.subarray(start, end)));
    start = end;
  }
  return out;
}

// FNV-1a over little-endian u32, in BigInt so the 64-bit wrap matches Rust.
const FNV_OFFSET = 0xcbf29ce484222325n;
const FNV_PRIME = 0x100000001b3n;
const MASK64 = 0xffffffffffffffffn;
function idsHash(ids) {
  let h = FNV_OFFSET;
  for (const id of ids) {
    const v = id >>> 0;
    for (const shift of [0, 8, 16, 24]) {
      h ^= BigInt((v >>> shift) & 0xff);
      h = (h * FNV_PRIME) & MASK64;
    }
  }
  return h;
}

function median(xs) {
  xs.sort((a, b) => a - b);
  const n = xs.length;
  return n % 2 ? xs[n >> 1] : (xs[n / 2 - 1] + xs[n / 2]) / 2;
}

// The driver parses `ids_hash` as a u64. A JS Number cannot hold one: the gpt2
// english hash alone is ~4.1e17, well past 2^53, so `Number(hash)` would round
// and JSON.stringify would emit the rounded value as a plain integer -- parsing
// cleanly on the Rust side and silently failing verification. Emit the BigInt
// through a sentinel so the digits survive exactly.
function report(o) {
  const json = JSON.stringify(o, (_k, v) =>
    typeof v === "bigint" ? `@bigint:${v}@` : v,
  ).replace(/"@bigint:(\d+)@"/g, "$1");
  process.stdout.write(json + "\n");
}

function unsupported(why) {
  report({ version: VERSION, lang: "javascript", secs: 0, tokens: 0,
           bytes: 0, ids_hash: 0, load_ms: 0, unsupported: why });
  process.exit(0);
}

// Version is part of the result: a number without one is not a result. Read
// from the copy actually resolved next to this script, so what is reported is
// what ran.
const HERE = dirname(fileURLToPath(import.meta.url));
let VERSION = "ai-tokenizer";
try {
  const pkg = JSON.parse(
    readFileSync(join(HERE, "node_modules/ai-tokenizer/package.json"), "utf8"),
  );
  VERSION = `ai-tokenizer ${pkg.version}`;
} catch {}

const args = parseArgs();

// `make models` writes these two only for byte-level BPE models, and only
// after checking the conversion is faithful. Their absence is the honest gate:
// no rank file means no vocabulary this library can represent.
const ranksPath = join(args.model, "ranks.tiktoken");
const patternPath = join(args.model, "pattern.txt");
if (!existsSync(ranksPath) || !existsSync(patternPath)) {
  unsupported(
    "no ranks.tiktoken/pattern.txt: `make models` derives those only for " +
      "byte-level BPE, so Unigram (albert), WordPiece (bert-wiki) and " +
      "non-ByteLevel BPE models have none. ai-tokenizer cannot read " +
      "tokenizer.json, and none of the four encodings it ships (cl100k_base, " +
      "o200k_base, p50k_base, claude) is this vocabulary",
  );
}

const t0 = process.hrtime.bigint();
let encode;
try {
  const { Tokenizer } = await import("ai-tokenizer");
  if (typeof Tokenizer !== "function") {
    unsupported("ai-tokenizer exports no Tokenizer class; adapter needs updating");
  }

  // `fatal` makes the decoder throw on invalid UTF-8 instead of substituting
  // U+FFFD, which is how a token is classified: the library looks a piece up in
  // `stringEncoder` by decoded string first and falls back to a binary search
  // over `binaryEncoder`, so a token whose bytes are not valid UTF-8 (a partial
  // multi-byte sequence, of which byte-level vocabularies have a few hundred)
  // must live in the binary table. Substituting U+FFFD would file it under the
  // wrong key and lose it.
  //
  // `ignoreBOM: true` is not optional, despite reading like a nicety. It means
  // "do not treat a leading U+FEFF specially"; the DEFAULT is to silently strip
  // it. llama-3's vocabulary has 8 tokens whose bytes start with EF BB BF, and
  // 7 of them differ from another token only by that prefix -- so with the
  // default decoder `ef bb bf 0a` decodes to "\n" and overwrites the real "\n"
  // token, and every bare newline in the corpus comes out as id 62619 instead
  // of 198. That is precisely the silent corruption the id hash exists to
  // catch, and it did: llama-3 x {arabic, thai, dense} mismatched until this
  // flag was set. gpt2 and mistral-nemo have no such tokens, so they passed
  // either way -- a reminder that one green model does not verify an adapter.
  const strict = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true });
  const pat = readFileSync(patternPath, "utf8").trim();
  const ranksText = readFileSync(ranksPath, "utf8");

  // Null-prototype, and this matters: the library reads ranks with a bare
  // `stringEncoder[piece]`. On a normal object literal a piece spelled
  // "constructor", "toString" or "valueOf" would hit Object.prototype and
  // return a *function*, which is `!== undefined` and would be pushed as if it
  // were a token id. Byte-level vocabularies do contain those words.
  const stringEncoder = Object.create(null);
  const binaryEncoder = [];
  for (const line of ranksText.split("\n")) {
    if (!line) continue;
    const sp = line.indexOf(" ");
    const bytes = Buffer.from(line.slice(0, sp), "base64");
    const rank = Number(line.slice(sp + 1));
    let s;
    try {
      s = strict.decode(bytes);
    } catch {
      s = undefined;
    }
    if (s !== undefined) stringEncoder[s] = rank;
    else binaryEncoder.push([new Uint8Array(bytes), rank]);
  }
  // `binarySearchBytes` compares byte-wise to the shorter length and breaks
  // ties by length, so the table has to be sorted that exact way or lookups
  // miss.
  binaryEncoder.sort((a, b) => {
    const x = a[0];
    const y = b[0];
    const n = Math.min(x.length, y.length);
    for (let i = 0; i < n; i++) if (x[i] !== y[i]) return x[i] - y[i];
    return x.length - y.length;
  });

  const tok = new Tokenizer({
    name: args.model,
    pat_str: pat,
    special_tokens: {},
    stringEncoder,
    binaryEncoder,
    decoder: Object.create(null),
  });
  encode = (text) => tok.encode(text);
} catch (e) {
  unsupported(`ai-tokenizer failed to load this vocabulary: ${e.message}`);
}
const loadMs = Number(process.hrtime.bigint() - t0) / 1e6;

const raw = readFileSync(args.corpus);
const chunks = chunk(raw, args.chunkBytes, args.maxChunks);
const nbytes = chunks.reduce((n, c) => n + Buffer.byteLength(c, "utf8"), 0);

// Untimed warm-up: fills caches, lets the JIT settle, and captures the ids
// used to verify this engine did the same work as the reference.
const all = [];
for (const c of chunks) all.push(...encode(c));
const tokens = all.length;
const h = idsHash(all);
all.length = 0;

const samples = [];
for (let r = 0; r < args.reps; r++) {
  const t = process.hrtime.bigint();
  for (const c of chunks) encode(c);
  samples.push(Number(process.hrtime.bigint() - t) / 1e9);
}

report({
  version: VERSION,
  lang: "javascript",
  secs: median(samples),
  tokens,
  bytes: nbytes,
  ids_hash: BigInt.asUintN(64, h),
  load_ms: loadMs,
  also_computes: "",
  internally_parallel: false,
});
