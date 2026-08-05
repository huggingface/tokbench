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

import { readFileSync } from "node:fs";
import { existsSync } from "node:fs";
import { join } from "node:path";

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

function report(o) {
  process.stdout.write(JSON.stringify(o) + "\n");
}

function unsupported(why) {
  report({ version: "ai-tokenizer", lang: "javascript", secs: 0, tokens: 0,
           bytes: 0, ids_hash: 0, load_ms: 0, unsupported: why });
  process.exit(0);
}

const args = parseArgs();

// ai-tokenizer ships per-encoding modules; the benchmark needs to know WHICH
// encoding corresponds to this model rather than guess one. `make models`
// writes that marker for models where an equivalent encoding genuinely exists.
const markerPath = join(args.model, "bpe_openai.txt");
if (!existsSync(markerPath)) {
  unsupported(
    "no bpe_openai.txt: ai-tokenizer loads a named encoding, and this model has " +
      "no verified-equivalent one (it does not read tokenizer.json)"
  );
}
const encodingName = readFileSync(markerPath, "utf8").trim();

const t0 = process.hrtime.bigint();
let encode, version = "ai-tokenizer";
try {
  const mod = await import("ai-tokenizer");
  const encMod = await import(`ai-tokenizer/encoding/${encodingName}`);
  const ranks = encMod.default ?? encMod;
  // The package exposes encode(text, ranks); if a future version renames it,
  // fail loudly rather than silently benchmarking the wrong function.
  if (typeof mod.encode !== "function") {
    unsupported("ai-tokenizer exports no encode(); adapter needs updating");
  }
  encode = (text) => mod.encode(text, ranks);
  try {
    const pkg = await import("ai-tokenizer/package.json", { with: { type: "json" } });
    version = `ai-tokenizer ${(pkg.default ?? pkg).version}`;
  } catch {}
} catch (e) {
  unsupported(`ai-tokenizer not installed or encoding ${encodingName} missing: ${e.message}`);
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
  version,
  lang: "javascript",
  secs: median(samples),
  tokens,
  bytes: nbytes,
  ids_hash: Number(BigInt.asUintN(64, h)),
  load_ms: loadMs,
  also_computes: "",
  internally_parallel: false,
});
