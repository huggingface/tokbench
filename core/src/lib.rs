//! The fairness contract every engine in `engines/` is measured through.
//!
//! There is exactly ONE timing loop per direction (`measure` for encode,
//! `measure_decode` for decode) and exactly ONE trait (`Engine`). An engine
//! folder's only job is to turn a model directory into something that answers
//! "give me the token ids for this `&str`", and — where the library can —
//! "give me the text for these ids". Everything
//! else — warm-up, repetition count, which clock, how bytes are counted, how
//! the median is taken — lives here, so no engine can be measured on terms of
//! its own choosing.
//!
//! # What "fair" means here, concretely
//!
//! 1. **Same work, verified.** Throughput of an engine that produces different
//!    ids is meaningless. Every engine's id stream is hashed and compared
//!    against the reference (`engines/hf-tokenizers`, the released `tokenizers`
//!    crate). A cell whose hash differs is reported as `mismatch` and is NEVER
//!    presented as a speedup. Being fast at the wrong answer is not a win.
//! 2. **Same clock, same process.** Native engines are called in-process
//!    through `Engine::encode`, timed by the same `Instant`. No subprocess, no
//!    IPC, no interpreter start-up in the measured region. Engines that can
//!    only be reached out-of-process are tagged [`Class::Subprocess`] and are
//!    rendered in a separate block — never ranked against in-process numbers.
//! 3. **Load is not encode.** Vocabulary parsing, trie/automaton construction
//!    and mmap all happen before the timer starts, and are reported separately
//!    as `load_ms`. An engine that front-loads work into a slow build is not
//!    rewarded by the throughput column, but it does not get to hide it either.
//! 4. **Warm engine, unseen text.** The corpus is cut into `reps + 1` disjoint
//!    slices: one warms the engine, and every timed rep gets text it has never
//!    seen. That is the regime a real server is in — warm process, new
//!    document — and, critically, it is the only way to stop the benchmark
//!    measuring memoization of its own input. Warming on the *same* chunks the
//!    timed passes re-encode inflated a document-granularity cache by up to
//!    **256x** (gigatoken on llama-2/dense) while barely moving a
//!    word-granularity one, and nothing from outside distinguishes the two.
//!    `--no-warmup` times only the genuinely cold first pass.
//! 5. **One thread by default.** Several engines parallelise internally, which
//!    silently turns a throughput comparison into a core-count comparison. The
//!    headline is single-thread; engines that cannot be pinned to one thread
//!    disclose it via [`Info::internally_parallel`] and are marked in the
//!    report. The multi-thread sweep is a separate, explicitly labelled axis.
//! 6. **Disclose the extra work.** Some engines also compute byte offsets or
//!    attention masks on the same pass ([`Info::also_computes`]). That cost is
//!    real and is left in the number, but the report prints the disclosure next
//!    to it so a reader knows the comparison is not like-for-like.
//! 7. **Decode gets the same ids, from the reference.** Decode throughput is
//!    measured over the reference engine's id stream, never over each
//!    engine's own encode output — otherwise an engine that merges harder
//!    feeds itself fewer, longer tokens and posts a better token rate for
//!    strictly less work. Correctness is checked the same way as encode, one
//!    level along: the decoded *text* is hashed and compared against the
//!    reference's decoded text. It cannot be compared against the original
//!    corpus, because a lowercasing or accent-stripping normalizer makes
//!    `decode(encode(t)) != t` for a perfectly correct tokenizer. Engines
//!    whose library has no decode entry point report `unsupported` and are
//!    absent from the decode ranking rather than scored zero in it.
//!
//! # What is deliberately NOT normalised
//!
//! The adapter must use the library's ordinary public API — the one a user
//! would call. If that API forces an allocation, a `Vec<i32>`→`Vec<u32>`
//! conversion, or a `String` copy, that cost stays in the measurement, because
//! the user pays it too. Adapters may not reach into private internals to skip
//! work the public path performs.

pub mod mem;

use std::path::{Path, PathBuf};
use std::time::Instant;

/// Ids are compared, so they need one representation. `u32` covers every
/// vocabulary in the matrix; adapters convert from whatever their library
/// returns, and that conversion is timed (see the module docs).
pub type Ids = Vec<u32>;

/// How the engine is reached from the Rust driver. This is a fairness label,
/// not an implementation detail: only engines in the same class are ranked
/// against each other without a caveat.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Class {
    /// A Rust crate called directly. Timed in-process; nothing between the
    /// clock and the library.
    Native,
    /// A C/C++ library behind a C ABI, called through FFI in-process. Same
    /// clock as `Native`; the FFI cost (usually a few ns) is included.
    Cffi,
    /// A Python library driven through an embedded interpreter in-process.
    /// The GIL and the CPython call overhead are included and disclosed.
    Python,
    /// Reached by running another process. Start-up is excluded (the child
    /// times its own encode loop and reports back), but this is NOT the same
    /// measurement as the in-process classes and is reported separately.
    Subprocess,
}

impl Class {
    pub fn as_str(self) -> &'static str {
        match self {
            Class::Native => "native",
            Class::Cffi => "cffi",
            Class::Python => "python",
            Class::Subprocess => "subprocess",
        }
    }
}

/// Everything the report needs to describe an engine honestly.
#[derive(Clone, Debug)]
pub struct Info {
    /// Folder name under `engines/`.
    pub name: &'static str,
    /// Exact version benchmarked — a number without a version is not a result.
    pub version: &'static str,
    /// Implementation language of the engine itself, not of the adapter.
    pub lang: &'static str,
    pub class: Class,
    pub url: &'static str,
    /// Work this engine does on the same pass beyond producing ids (e.g.
    /// `"byte offsets"`). Printed next to the throughput so the reader knows
    /// when a comparison is not like-for-like. `""` means ids only.
    pub also_computes: &'static str,
    /// True when the engine spreads a single `encode` call across threads and
    /// the adapter could not pin it to one. Such a number is a throughput
    /// figure for the whole machine, not for one core.
    pub internally_parallel: bool,
}

/// A model, as the set of on-disk artifacts different engines need. All of
/// them are derived from the SAME upstream model so the comparison is valid;
/// `make models` produces them. An engine takes the artifact it supports and
/// returns [`Unsupported`] when none is present.
#[derive(Clone, Debug)]
pub struct Model {
    pub name: String,
    /// Directory holding this model's artifacts.
    pub dir: PathBuf,
}

impl Model {
    /// HuggingFace `tokenizer.json` — the canonical artifact and the one the
    /// reference engine loads.
    pub fn tokenizer_json(&self) -> PathBuf {
        self.dir.join("tokenizer.json")
    }
    /// An artifact a specific engine needs, e.g. `ranks.tiktoken`,
    /// `spiece.model`, `model.tkz`. Returns `None` when it was not generated.
    pub fn artifact(&self, file: &str) -> Option<PathBuf> {
        let p = self.dir.join(file);
        p.exists().then_some(p)
    }
}

/// Why an engine cannot run a given cell. A sparse matrix is expected and
/// honest: tiktoken has no Unigram, sentencepiece has no byte-level BPE.
/// Recording the reason is better than a blank.
#[derive(Debug)]
pub struct Unsupported(pub String);

impl std::fmt::Display for Unsupported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Shorthand for the `?`-heavy adapter code.
pub fn unsupported<T>(msg: impl Into<String>) -> Result<T, Unsupported> {
    Err(Unsupported(msg.into()))
}

/// Where the time went inside one `encode`, in nanoseconds.
///
/// Only engines whose public API exposes the stages separately can fill this
/// in. The rest return `None` from [`Engine::phases`] and are rendered as a
/// single opaque block — an honest "not instrumented" beats a fabricated
/// split, which would be indistinguishable from a real measurement in a chart.
#[derive(Clone, Copy, Debug, Default)]
pub struct Phases {
    /// Unicode normalisation only: NFC/NFD/NFKC, lowercasing, accent
    /// stripping, control-character cleanup.
    pub normalization_ns: u64,
    /// Pre-tokenisation: the regex/FSM split into pre-tokens (ByteLevel,
    /// Metaspace, Whitespace, the cl100k/o200k patterns).
    ///
    /// Kept separate from normalisation because it is routinely the single
    /// most expensive stage — on byte-level BPE it can outweigh the merge
    /// loop — and folding it into either neighbour hides exactly the
    /// bottleneck this benchmark exists to expose.
    pub pre_tokenization_ns: u64,
    /// The model loop itself: BPE merges, WordPiece longest-match, Unigram
    /// Viterbi — whatever turns a pre-token into ids.
    pub core_encoding_ns: u64,
    /// Post-processing: template application, BOS/EOS, type ids.
    pub post_processing_ns: u64,
}

impl Phases {
    pub fn total_ns(&self) -> u64 {
        self.normalization_ns
            + self.pre_tokenization_ns
            + self.core_encoding_ns
            + self.post_processing_ns
    }
    pub fn accumulate(&mut self, other: Phases) {
        self.normalization_ns += other.normalization_ns;
        self.pre_tokenization_ns += other.pre_tokenization_ns;
        self.core_encoding_ns += other.core_encoding_ns;
        self.post_processing_ns += other.post_processing_ns;
    }
}

/// The whole surface an engine folder implements.
pub trait Engine: Send {
    fn info(&self) -> Info;

    /// Encode `text`, appending the ids to `out`.
    ///
    /// `out` arrives cleared but with its capacity retained, matching a real
    /// encode loop that reuses a buffer. Adapters whose library returns an
    /// owned `Vec` just extend from it — that copy is part of the API's cost
    /// and is timed on purpose.
    fn encode(&mut self, text: &str, out: &mut Ids);

    /// Ask the engine to use `threads` threads for its OWN parallelism, and
    /// say whether it can.
    ///
    /// `false` -- the default -- means the library exposes no such control, so
    /// the only way to get a scaling curve out of it is to run several
    /// instances side by side. `true` means the engine owns its threading and
    /// [`measure_scaling`] must hand it a batch and stay out of the way.
    ///
    /// This is the whole reason the harness can no longer assume one shape of
    /// parallelism: re-implementing it outside an engine that already has a
    /// pool measures the harness, not the engine.
    fn set_threads(&mut self, _threads: usize) -> bool {
        false
    }

    /// Encode a whole batch through the library's own batch entry point,
    /// appending every document's ids to `out`.
    ///
    /// This is what an engine with an internal pool has to be called through --
    /// its `encode` is one document on one thread by construction, so timing
    /// that in a loop can never show what its pool does.
    ///
    /// The default loops [`Engine::encode`], which is exactly right for a
    /// single-threaded library: same work, same order, no batch API invented on
    /// its behalf. Ids are concatenated rather than kept per document because
    /// every caller here either hashes the whole stream or discards it.
    fn encode_batch(&mut self, texts: &[&str], out: &mut Ids) {
        for text in texts {
            self.encode(text, out);
        }
    }

    /// Does this engine have a batch entry point of its own?
    ///
    /// Separate from [`Engine::set_threads`] because the two are independent,
    /// and the combination decides how the engine may be measured:
    ///
    /// * batch + settable threads -- a full curve through its own pool.
    /// * batch + unsettable threads -- some libraries hardcode their width
    ///   (`available_parallelism()` and no knob). One point, at the width the
    ///   engine chose, reported as such. Not a curve, because there is no
    ///   baseline, and emphatically not harness threading.
    /// * no batch -- the harness may thread it, and says so.
    ///
    /// Returning false here while overriding `encode_batch` is the one
    /// combination that silently loses: the harness would thread an engine
    /// that had its own way of doing it.
    fn has_native_batch(&self) -> bool {
        false
    }

    /// Configure padding for subsequent [`Engine::encode_batch`] calls, and
    /// say whether the library can do it.
    ///
    /// The default accepts [`Padding::Off`] -- every tokenizer can produce
    /// ragged output -- and refuses [`Padding::Longest`], so a padded cell is
    /// reported unsupported instead of being quietly measured unpadded. The
    /// harness must never pad on an engine's behalf: doing so would charge
    /// every engine the same fill cost and hide the thing being compared,
    /// which is whether the library pads well.
    fn set_padding(&mut self, padding: Padding) -> bool {
        matches!(padding, Padding::Off)
    }

    /// Per-stage timing for one `encode` of `text`, when the library exposes
    /// its stages separately.
    ///
    /// Called on a SEPARATE pass from [`measure`], never inside the timed
    /// loop: instrumenting the stages adds clock reads that would inflate the
    /// headline throughput. The default returns `None`, which the report
    /// renders as "not instrumented" rather than guessing a breakdown.
    fn phases(&mut self, _text: &str) -> Option<Phases> {
        None
    }

    /// Decode `ids` back to text, appending to `out`.
    ///
    /// `out` arrives cleared with its capacity retained, mirroring
    /// [`Engine::encode`]: adapters whose library returns an owned `String`
    /// push from it, and that copy is timed on purpose.
    ///
    /// The ids handed in are always the REFERENCE engine's, never the
    /// engine's own — see [`measure_decode`] for why that is the only fair
    /// input.
    ///
    /// The default returns [`Unsupported`], so an engine has to opt in rather
    /// than be silently credited with a decode it never ran. Returning `Err`
    /// mid-run is also the correct move for an id this engine cannot map: a
    /// short `out` would otherwise hash as a cheap, fast decode.
    fn decode(&mut self, _ids: &[u32], _out: &mut String) -> Result<(), Unsupported> {
        unsupported("library exposes no decode entry point")
    }
}

/// Constructor half of an engine, kept separate from [`Engine`] so the driver
/// can build one without an instance in hand.
pub trait Build {
    /// Build from a model directory, or explain why this engine cannot.
    /// Called before the timer starts; do all vocabulary work here.
    fn build(model: &Model) -> Result<Box<dyn Engine>, Unsupported>
    where
        Self: Sized;

    /// The same engine with the library's own caches turned off, when the
    /// library can be asked for that.
    ///
    /// A build-time question, not a setter: a pretoken cache is sized and
    /// populated when the model is constructed, so there is nothing to toggle
    /// afterwards. The registry pairs this with [`Build::build`] as a separate
    /// `<engine>-no-cache` row, which is what makes a cache's contribution
    /// readable instead of inferred.
    ///
    /// The default refuses, and the refusal is the point. Most of these
    /// libraries expose no way to disable their caches, and a cache-free cell
    /// filled with a cached number is worse than an absent one: subtracting
    /// the two rows would report the cache as worth 0%. Engines that can do it
    /// say how; engines that cannot say why not.
    fn build_without_cache(_model: &Model) -> Result<Box<dyn Engine>, Unsupported>
    where
        Self: Sized,
    {
        Err(Unsupported(
            "this library exposes no way to disable its caches".into(),
        ))
    }
}

/// A deterministic hash of an id stream, used to check that two engines did
/// the same work. FNV-1a rather than `DefaultHasher` because the value is
/// written to JSON and compared across runs and machines, and `std`'s hasher
/// makes no cross-version stability promise.
pub fn ids_hash(ids: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &id in ids {
        for b in id.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// A deterministic hash of decoded text — the decode-side counterpart to
/// [`ids_hash`], and the same FNV-1a for the same reason.
///
/// Decode cannot be verified against the original corpus: a normalizer that
/// lowercases or strips accents makes `decode(encode(t)) != t` for a correct
/// tokenizer. So the oracle is the reference engine's decoded text, and this
/// hashes it.
pub fn text_hash(text: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in text.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// One engine × one corpus.
#[derive(Clone, Debug)]
pub struct Measure {
    pub mbps: f64,
    pub ns_per_byte: f64,
    /// Median over `reps` passes, seconds.
    pub secs: f64,
    /// Bytes in ONE timed slice — what `secs` refers to.
    pub bytes: usize,
    /// Tokens over the whole corpus (the verification pass), not one slice.
    pub tokens: usize,
    pub ids_hash: u64,
    /// True when the corpus could not supply `reps + 1` disjoint slices and
    /// some text had to be encoded more than once. Such a cell measures the
    /// engine's memoization as much as its speed; the report flags it rather
    /// than presenting it as a clean number.
    pub reused: bool,
}

/// One engine × one corpus, decode direction.
#[derive(Clone, Debug)]
pub struct DecodeMeasure {
    /// MB/s of text produced. Deliberately the *output* rate, so it sits on
    /// the same axis as encode's MB/s of text consumed.
    pub mbps: f64,
    /// The input-side rate. Decode is driven by token count, not byte count,
    /// so this is the number to compare when two engines emit text of
    /// different lengths from the same ids.
    pub ns_per_token: f64,
    /// Median over `reps` passes, seconds.
    pub secs: f64,
    /// Bytes of text produced from ONE slice.
    pub bytes: usize,
    /// Ids consumed from ONE slice.
    pub tokens: usize,
    pub text_hash: u64,
}

/// Call-by-call encode latency over distinct, fixed-size documents.
#[derive(Clone, Debug)]
pub struct LatencyMeasure {
    pub p50_us: f64,
    pub p99_us: f64,
    pub samples: usize,
    pub document_bytes: usize,
}

/// Split a corpus into fixed-size chunks on char boundaries.
///
/// ~10 kB is the regime where per-call overhead is amortised but a document
/// still fits in cache — the same size the upstream pipeline benchmark uses,
/// so numbers stay comparable. Chunking also stops one engine from winning on
/// document-level parallelism alone.
pub fn chunk(text: &str, chunk_bytes: usize, max_chunks: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut start = 0;
    while start < text.len() && out.len() < max_chunks {
        let mut end = (start + chunk_bytes).min(text.len());
        while end < text.len() && !text.is_char_boundary(end) {
            end += 1;
        }
        out.push(text[start..end].to_string());
        start = end;
    }
    out
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// Measure one encode call at a time over documents that are each used once.
///
/// `documents[0]` warms the engine. Every remaining document contributes one
/// latency sample, which prevents a document-level cache from turning this
/// into a lookup benchmark. `document_bytes` is the requested maximum; a
/// document can be a few bytes shorter when a UTF-8 boundary requires it.
pub fn measure_latency(
    engine: &mut dyn Engine,
    documents: &[String],
    document_bytes: usize,
) -> Option<LatencyMeasure> {
    if documents.len() < 2 || document_bytes == 0 {
        return None;
    }
    let mut out = Ids::with_capacity(document_bytes);
    engine.encode(&documents[0], &mut out);

    let mut samples = Vec::with_capacity(documents.len() - 1);
    for document in &documents[1..] {
        out.clear();
        let started = Instant::now();
        engine.encode(document, &mut out);
        let elapsed_us = started.elapsed().as_secs_f64() * 1e6;
        std::hint::black_box(&out);
        samples.push(elapsed_us);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let percentile = |pct: usize| {
        let rank = (pct * samples.len()).div_ceil(100).max(1);
        samples[rank - 1]
    };
    Some(LatencyMeasure {
        p50_us: percentile(50),
        p99_us: percentile(99),
        samples: samples.len(),
        document_bytes,
    })
}

/// THE timing loop. Every engine, every language, goes through this function
/// and no other, so the only thing that differs between two numbers is the
/// engine.
///
/// # No document is ever encoded twice inside a measurement
///
/// The corpus is cut into `reps + 1` **disjoint** slices: slice 0 warms, and
/// each timed rep gets its own, never-before-seen slice. This is not fussiness;
/// the earlier version warmed on a set of chunks and then timed *those same
/// chunks* `reps` times, which measures memoization of the benchmark's own
/// input rather than tokenization.
///
/// The distortion was enormous and engine-specific. An engine that caches at
/// word granularity is barely affected, because words genuinely repeat inside
/// one pass. An engine that caches at document granularity gets the whole
/// answer back for a hash lookup — and how coarsely an engine caches is
/// invisible from outside. Measured on gigatoken / llama-2, replay versus
/// first-pass: english 1.1x (words, honest), chinese 20x, dense **256x**. The
/// CJK corpora have no spaces, so the "word" was the entire 10 kB chunk and the
/// cache degenerated into "have I seen this exact chunk before" — always yes
/// under replay, essentially never yes in production.
///
/// Warming on *different* text is also the more faithful regime: a real server
/// has a warm engine and a document it has not seen.
///
/// When the corpus cannot supply `reps + 1` disjoint slices, the run does not
/// silently reuse text — it shrinks the slices, and if even that fails it sets
/// [`Measure::reused`] so the report can flag the cell.
pub fn measure(engine: &mut dyn Engine, chunks: &[String], reps: usize, warmup: bool) -> Measure {
    let mut out: Ids = Vec::with_capacity(64 * 1024);
    let mut all: Ids = Vec::new();
    let secs;
    let bytes;
    let mut reused = false;

    if warmup {
        // reps timed slices + 1 warm-up slice, all disjoint.
        let want = reps + 1;
        let per = chunks.len() / want;
        if per == 0 {
            // Not enough distinct text. Say so rather than quietly replaying.
            reused = true;
        }
        let per = per.max(1);
        let slice = |i: usize| -> &[String] {
            let start = (i * per) % chunks.len().max(1);
            let end = (start + per).min(chunks.len());
            &chunks[start..end]
        };

        // Warm-up on slice 0. Ids for verification come from the WHOLE corpus
        // (below), not from this slice, so every engine is still hashed over
        // identical text.
        for c in slice(0) {
            out.clear();
            engine.encode(c, &mut out);
        }

        // Slices hold different text and therefore different byte counts, so
        // the comparable quantity across reps is seconds PER BYTE, not seconds.
        let mut spb = Vec::with_capacity(reps);
        let mut timed_bytes = 0usize;
        for r in 0..reps {
            let s = slice(r + 1);
            let sb: usize = s.iter().map(|c| c.len()).sum();
            if sb == 0 {
                continue;
            }
            let t0 = Instant::now();
            for c in s {
                out.clear();
                engine.encode(c, &mut out);
                // Keep the optimiser from deleting the call.
                std::hint::black_box(&out);
            }
            spb.push(t0.elapsed().as_secs_f64() / sb as f64);
            timed_bytes = sb;
        }
        // Report the median rate scaled to one slice, so `secs` stays a
        // duration the reader can sanity-check against `bytes`.
        bytes = timed_bytes;
        secs = median(spb) * bytes as f64;

        // Verification pass over the entire corpus, untimed.
        for c in chunks {
            out.clear();
            engine.encode(c, &mut out);
            all.extend_from_slice(&out);
        }
    } else {
        bytes = chunks.iter().map(|c| c.len()).sum();
        // COLD. There is exactly one cold pass available per engine instance,
        // so `reps` cannot apply: a second pass is warm by definition, and
        // taking a median over "1 cold + n-1 warm" would report a warm number
        // under a cold label. This measures the first pass and nothing else.
        //
        // (An earlier version of this function ran the same untimed pass in
        // both branches and then timed `reps` passes regardless, which made
        // `--no-warmup` silently identical to the warm path. Engines whose
        // whole design is a pretoken cache — gigatoken, tokie, the pipeline —
        // were the ones it misreported, and by the largest margin.)
        let t0 = Instant::now();
        for c in chunks {
            out.clear();
            engine.encode(c, &mut out);
            std::hint::black_box(&out);
        }
        secs = t0.elapsed().as_secs_f64();

        // Ids are captured afterwards, on a now-warm pass. Same ids, and
        // keeping the `extend` out of the timed region means the cold number
        // is not inflated by the harness's own bookkeeping.
        for c in chunks {
            out.clear();
            engine.encode(c, &mut out);
            all.extend_from_slice(&out);
        }
    }

    let tokens = all.len();
    let hash = ids_hash(&all);
    drop(all);
    Measure {
        mbps: (bytes as f64 / (1024.0 * 1024.0)) / secs,
        ns_per_byte: secs * 1e9 / bytes as f64,
        secs,
        bytes,
        tokens,
        ids_hash: hash,
        reused,
    }
}

/// THE decode timing loop, and the only one. Same contract as [`measure`]:
/// one clock, one median, load excluded, warm.
///
/// **Every engine is fed the same ids — the reference engine's.** This is the
/// decode-side reading of fairness rule 1, and it is not optional. Letting
/// each engine decode its *own* encode output would hand a different input to
/// every engine: one that merges aggressively decodes fewer, longer tokens
/// and would post a higher token rate for doing strictly less work. Same
/// input, same work, then compare.
///
/// The untimed probe pass ahead of the timer does triple duty — it is the
/// warm pass, it captures the text hash for verification, and it is where an
/// engine without a decode entry point drops out before any number is
/// attributed to it.
pub fn measure_decode(
    engine: &mut dyn Engine,
    id_chunks: &[Ids],
    reps: usize,
) -> Result<DecodeMeasure, Unsupported> {
    let tokens: usize = id_chunks.iter().map(|c| c.len()).sum();
    if tokens == 0 {
        return unsupported("no reference ids to decode");
    }
    // ~4 bytes/token is the loose upper bound for UTF-8 text; the exact
    // capacity does not matter, only that the timed loop never grows it.
    let mut out = String::with_capacity(tokens * 4);

    let mut all = String::new();
    for c in id_chunks {
        out.clear();
        engine.decode(c, &mut out)?;
        all.push_str(&out);
    }
    let bytes = all.len();
    let hash = text_hash(&all);
    drop(all);
    if bytes == 0 {
        return unsupported("decode produced no text");
    }

    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = Instant::now();
        for c in id_chunks {
            out.clear();
            // The probe pass above already proved this engine can decode
            // these ids, so `?` here is a genuine mid-run failure.
            engine.decode(c, &mut out)?;
            // Keep the optimiser from deleting the call.
            std::hint::black_box(&out);
        }
        samples.push(t0.elapsed().as_secs_f64());
    }

    let secs = median(samples);
    Ok(DecodeMeasure {
        mbps: (bytes as f64 / (1024.0 * 1024.0)) / secs,
        ns_per_token: secs * 1e9 / tokens as f64,
        secs,
        bytes,
        tokens,
        text_hash: hash,
    })
}

/// Read a corpus file, returning `None` rather than panicking so a missing
/// fixture degrades to a skipped cell.
pub fn read_corpus(path: &Path) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

// ---------------------------------------------------------------------------
// Multi-thread scaling
// ---------------------------------------------------------------------------

/// One point on the scaling curve.
#[derive(Clone, Copy, Debug)]
pub struct ThreadPoint {
    pub threads: usize,
    pub mbps: f64,
    /// Throughput as a percentage of perfect linear scaling from the
    /// single-thread number: `mbps(n) / (n * mbps(1)) * 100`.
    ///
    /// 100% means adding a core added a core's worth of throughput. This is
    /// the number that actually matters for a serving deployment, and it is
    /// where tokenizers differ most: an engine holding a shared cache behind a
    /// lock can post an excellent single-thread figure and then scale at 15%,
    /// while a slower engine with thread-local state scales at 95% and wins on
    /// any real machine.
    pub efficiency_pct: f64,
}

/// `1, 2, 4, 8, ...` — powers of two up to the count of **performance** cores.
///
/// Two deliberate exclusions:
///
/// * **Efficiency cores.** On a hybrid CPU (Apple silicon, Intel P/E),
///   `available_parallelism` counts E-cores, which run the same code several
///   times slower. Scheduling onto them drags the aggregate down and shows up
///   as an efficiency collapse that says nothing about the tokenizer. On a
///   10P+4E machine the 14-thread point is not a scaling measurement, it is a
///   measurement of the E-cores.
/// * **The odd top value.** A trailing non-power-of-two point (say 10) makes
///   the curve's last segment a different width from the others, which reads
///   as a slope change on a log axis when nothing changed.
pub fn thread_counts() -> Vec<usize> {
    let max = perf_cores();
    let mut v = Vec::new();
    let mut n = 1;
    while n <= max {
        v.push(n);
        n *= 2;
    }
    if v.is_empty() {
        v.push(1);
    }
    v
}

/// Performance-core count, falling back to total parallelism where the
/// distinction is unavailable.
fn perf_cores() -> usize {
    #[cfg(target_os = "macos")]
    {
        // `hw.perflevel0` is the performance cluster; absent on non-hybrid Macs.
        if let Ok(out) = std::process::Command::new("sysctl")
            .args(["-n", "hw.perflevel0.logicalcpu"])
            .output()
        {
            if let Ok(n) = String::from_utf8_lossy(&out.stdout).trim().parse::<usize>() {
                if n > 0 {
                    return n;
                }
            }
        }
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Whether a batch is padded to the length of its longest member.
///
/// A first-class axis rather than a detail, because it is a large and uneven
/// cost: padding is a fill, often a second pass over the batch, and sometimes
/// a different output layout entirely. An engine that skips it looks fast for
/// a reason that has nothing to do with tokenizing, and a serving deployment
/// that needs rectangular tensors cannot use the unpadded number.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Padding {
    /// Ragged output: every document keeps its own length.
    Off,
    /// Every document padded to the longest in the batch.
    Longest,
}

impl Padding {
    pub fn as_str(self) -> &'static str {
        match self {
            Padding::Off => "off",
            Padding::Longest => "longest",
        }
    }
}

/// How a scaling curve was produced.
///
/// Not cosmetic, and not a detail a reader may skip: the two variants answer
/// different questions and must never share a column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalingKind {
    /// One engine, told to use `n` threads, handed one batch. What a caller
    /// actually gets from the library's own batch API.
    Internal,
    /// `n` independent single-thread engines fed by a shared cursor. What a
    /// caller gets by sharding requests, and the only curve obtainable from a
    /// library with no threading of its own.
    External,
}

impl ScalingKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ScalingKind::Internal => "internal",
            ScalingKind::External => "external",
        }
    }
}

/// A scaling curve and, inseparably, the conditions it was obtained under.
#[derive(Clone, Debug)]
pub struct Scaling {
    pub kind: ScalingKind,
    /// Which padding mode was in force. Two curves for the same engine are
    /// only comparable at equal padding.
    pub padding: Padding,
    /// One point per thread count the engine agreed to run at. Thread counts
    /// it refused are absent rather than approximated -- see
    /// [`Engine::set_threads`].
    pub points: Vec<ThreadPoint>,
}

impl Scaling {
    fn empty(kind: ScalingKind, padding: Padding) -> Self {
        Scaling {
            kind,
            padding,
            points: Vec::new(),
        }
    }
}

/// Does throughput grow with thread count, and how far short of linear?
///
/// **The engine's own parallelism is used wherever the engine has any.** An
/// engine that answers [`Engine::set_threads`] gets one instance, is told how
/// many threads to use, and is handed the whole measured batch through
/// [`Engine::encode_batch`]. The harness then holds nothing but the clock.
///
/// Threading such an engine from outside is not a conservative approximation,
/// it is a different and wrong number: it leaves the engine's pool idle, or --
/// worse, because it looks plausible -- starts `n` pools of `n` threads each
/// and reports the resulting oversubscription as the engine's scaling. An
/// engine that parallelises internally would also post >100% efficiency under
/// external threading, having already used more than one core at "1 thread".
///
/// Only when `set_threads` returns false does the harness thread the engine
/// itself, and the curve is then tagged [`ScalingKind::External`] so a reader
/// can see it is the other question. Two choices in that fallback:
///
/// * **Per-thread engines, not one shared engine.** Most such libraries are
///   not `Sync`, and those that are often hide a mutex around a shared cache.
///   Giving every thread its own instance measures the best case the library
///   can offer, so a poor scaling number is a real property of the engine
///   rather than an artefact of how the harness shared it.
/// * **Work stealing, not a static split.** Documents differ in cost by more
///   than 10x across scripts. A contiguous split would leave threads idle at
///   the end and report that as poor scaling; a shared cursor keeps every
///   thread busy until the corpus is done.
///
/// Every repetition builds fresh engine instances. Each instance warms on a
/// representative slice of the corpus, while the timed region contains only
/// the disjoint remainder and encodes every measured document once. This is
/// the same warm-engine, unseen-text workload as [`measure`], and prevents an
/// engine's exact-input cache from turning corpus replay into apparent
/// throughput or scaling.
pub fn measure_scaling(
    make: &(dyn Fn() -> Option<Box<dyn Engine>> + Sync),
    chunks: &[String],
    counts: &[usize],
    reps: usize,
    padding: Padding,
) -> Option<Scaling> {
    if chunks.len() < 2 || reps == 0 {
        return Some(Scaling::empty(ScalingKind::External, padding));
    }

    // Spread warm-up throughout the input rather than taking a contiguous
    // prefix, which could be a single language or source in a mixed corpus.
    // The two sets remain strictly disjoint.
    let mut warm_chunks: Vec<&str> = Vec::new();
    let mut measured_chunks: Vec<&str> = Vec::new();
    for (index, chunk) in chunks.iter().enumerate() {
        if index % 6 == 0 {
            warm_chunks.push(chunk.as_str());
        } else {
            measured_chunks.push(chunk.as_str());
        }
    }
    let measured_bytes: usize = measured_chunks.iter().map(|c| c.len()).sum();

    // Ask the engine what it can do, once, on a probe instance that encodes
    // nothing. `set_threads` is the only honest capability signal --
    // `Info::internally_parallel` is a self-declaration, not a capability --
    // and `set_padding` decides whether this cell exists for this engine at
    // all.
    let mut probe = make()?;
    if !probe.set_padding(padding) {
        // No native padding. Reported as unsupported rather than measured
        // unpadded and labelled padded, and never padded by the harness.
        return None;
    }
    let native_batch = probe.has_native_batch();
    drop(probe);

    // Which thread counts will this engine actually honour? Asked up front, on
    // a fresh instance each time, so the timing loop never has to interpret a
    // refusal -- and so an engine whose parallelism is on/off rather than an
    // integer contributes the points it can instead of nothing.
    //
    // Probing every requested count, not just 1: an engine that can only run
    // at full width would look thread-less if 1 were the only question asked,
    // and would then be threaded from outside -- the bug this all exists to
    // prevent.
    let settable: Vec<usize> = counts
        .iter()
        .copied()
        .filter(|&n| match make() {
            Some(mut engine) => engine.set_threads(n),
            None => false,
        })
        .collect();

    let (kind, usable) = if !settable.is_empty() {
        (ScalingKind::Internal, settable)
    } else if native_batch {
        // Has its own batch fan-out but no width control. One point, at
        // whatever width it picked; `threads: 0` records "the engine's own
        // choice" rather than asserting a number the harness did not set.
        (ScalingKind::Internal, vec![0])
    } else {
        (ScalingKind::External, counts.to_vec())
    };

    // Padding is a property of a batch, so it only means anything where the
    // engine drives its own batch. An externally threaded engine is called one
    // document at a time, and there is no batch to pad to.
    if kind == ScalingKind::External && padding != Padding::Off {
        return None;
    }
    if usable.is_empty() {
        return Some(Scaling::empty(kind, padding));
    }

    let mut measured: Vec<(usize, f64)> = Vec::new();
    for &n in &usable {
        let mut samples = Vec::with_capacity(reps);
        for _ in 0..reps {
            let timed = match kind {
                ScalingKind::Internal => {
                    time_internal(make, &warm_chunks, &measured_chunks, n, padding)
                }
                ScalingKind::External => time_external(make, &warm_chunks, &measured_chunks, n),
            };
            match timed {
                Some(secs) => samples.push(secs),
                // `usable` already established the engine takes `n`, so a
                // refusal now is a bug in the adapter, not a capability.
                None => return Some(Scaling::empty(kind, padding)),
            }
        }

        let secs = median(samples);
        let mbps = (measured_bytes as f64 / (1024.0 * 1024.0)) / secs;
        measured.push((n, mbps));
    }

    // Measurement order may be reversed between complete runs to expose
    // thermal or temporal drift. Always pair against the measured 1-thread
    // point, then return a canonical thread-count order for JSON consumers.
    //
    // An engine that cannot be pinned to one thread has no baseline, so
    // efficiency is left at 0.0 rather than invented from its widest point --
    // which would report 100% for an engine whose scaling is unknown.
    let base = measured
        .iter()
        .find_map(|(threads, mbps)| (*threads == 1).then_some(*mbps));
    measured.sort_unstable_by_key(|(threads, _)| *threads);
    let points = measured
        .into_iter()
        .map(|(threads, mbps)| {
            let ideal = base.map(|b| b * threads as f64).unwrap_or(0.0);
            ThreadPoint {
                threads,
                mbps,
                efficiency_pct: if ideal > 0.0 {
                    mbps / ideal * 100.0
                } else {
                    0.0
                },
            }
        })
        .collect();
    Some(Scaling {
        kind,
        padding,
        points,
    })
}

/// One timed batch through the engine's own pool.
///
/// `None` when the engine will not take `threads`. That is a hard stop rather
/// than a silent fallback: a refusal here would otherwise run on whatever
/// thread count the engine felt like and report it as `threads`.
fn time_internal(
    make: &(dyn Fn() -> Option<Box<dyn Engine>> + Sync),
    warm: &[&str],
    measured: &[&str],
    threads: usize,
    padding: Padding,
) -> Option<f64> {
    let mut engine = make()?;
    // `0` means "the width the engine chose"; there is nothing to set.
    if threads != 0 && !engine.set_threads(threads) {
        return None;
    }
    if !engine.set_padding(padding) {
        return None;
    }

    // Warm through the same batch entry point, so a pool that grows to the
    // concurrency it has seen is already at full width. Unwarmed, the first
    // timed batch would pay to populate it.
    let mut buf: Ids = Vec::new();
    engine.encode_batch(warm, &mut buf);
    buf.clear();

    let t0 = Instant::now();
    engine.encode_batch(measured, &mut buf);
    let elapsed = t0.elapsed().as_secs_f64();
    std::hint::black_box(&buf);
    Some(elapsed)
}

/// One timed run of `threads` independent engines over a shared cursor, for
/// libraries that have no threading of their own.
fn time_external(
    make: &(dyn Fn() -> Option<Box<dyn Engine>> + Sync),
    warm: &[&str],
    measured: &[&str],
    threads: usize,
) -> Option<f64> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Barrier;

    // A fresh instance per worker and repetition makes every timed document
    // unseen by that instance. Construction stays outside the timer, as the
    // driver reports it separately.
    let mut engines: Vec<Box<dyn Engine>> = Vec::with_capacity(threads);
    for _ in 0..threads {
        engines.push(make()?);
    }
    for engine in engines.iter_mut() {
        let mut buf: Ids = Vec::new();
        for &chunk in warm {
            buf.clear();
            engine.encode(chunk, &mut buf);
        }
    }

    let cursor = AtomicUsize::new(0);
    let start = Barrier::new(threads + 1);
    let done = Barrier::new(threads + 1);
    Some(std::thread::scope(|scope| {
        for engine in engines.iter_mut() {
            let cursor = &cursor;
            let start = &start;
            let done = &done;
            scope.spawn(move || {
                let mut buf: Ids = Vec::with_capacity(4096);
                start.wait();
                loop {
                    let index = cursor.fetch_add(1, Ordering::Relaxed);
                    if index >= measured.len() {
                        break;
                    }
                    buf.clear();
                    engine.encode(measured[index], &mut buf);
                    std::hint::black_box(&buf);
                }
                done.wait();
            });
        }
        let t0 = Instant::now();
        start.wait();
        done.wait();
        t0.elapsed().as_secs_f64()
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stub engine: one id per byte. Enough to prove the harness itself
    /// counts bytes, hashes ids and produces a positive rate.
    struct Bytes;
    impl Engine for Bytes {
        fn info(&self) -> Info {
            Info {
                name: "bytes",
                version: "0",
                lang: "rust",
                class: Class::Native,
                url: "",
                also_computes: "",
                internally_parallel: false,
            }
        }
        fn encode(&mut self, text: &str, out: &mut Ids) {
            out.extend(text.as_bytes().iter().map(|&b| b as u32));
        }
        fn decode(&mut self, ids: &[u32], out: &mut String) -> Result<(), Unsupported> {
            let bytes: Vec<u8> = ids.iter().map(|&i| i as u8).collect();
            match String::from_utf8(bytes) {
                Ok(s) => {
                    out.push_str(&s);
                    Ok(())
                }
                Err(e) => unsupported(format!("not utf-8: {e}")),
            }
        }
    }

    /// An engine that only encodes. Proves the default `decode` keeps such an
    /// engine out of the decode ranking instead of scoring it zero.
    struct EncodeOnly;
    impl Engine for EncodeOnly {
        fn info(&self) -> Info {
            Bytes.info()
        }
        fn encode(&mut self, text: &str, out: &mut Ids) {
            Bytes.encode(text, out)
        }
    }

    #[test]
    fn harness_counts_and_verifies() {
        let text = "hello world, ".repeat(500);
        let chunks = chunk(&text, 1024, 100);
        assert!(chunks.len() > 1, "corpus should split into several chunks");
        assert_eq!(
            chunks.iter().map(|c| c.len()).sum::<usize>(),
            text.len(),
            "chunking must not drop or duplicate bytes"
        );

        let m = measure(&mut Bytes, &chunks, 3, true);
        // `bytes` is ONE timed slice, not the whole corpus: reps run on
        // disjoint slices, so there is no single duration covering everything.
        assert!(m.bytes > 0 && m.bytes <= text.len());
        // Tokens come from the verification pass, which does cover everything.
        assert_eq!(m.tokens, text.len(), "one id per byte, whole corpus");
        assert!(m.mbps > 0.0 && m.mbps.is_finite());

        let again = measure(&mut Bytes, &chunks, 1, true);
        assert_eq!(m.ids_hash, again.ids_hash);
        assert_ne!(ids_hash(&[1, 2, 3]), ids_hash(&[3, 2, 1]));
    }

    #[test]
    fn decode_harness_measures_and_verifies() {
        let text = "hello world, ".repeat(500);
        let chunks = chunk(&text, 1024, 100);

        // Built the way the driver builds them: one id slice per text chunk,
        // from the reference engine.
        let id_chunks: Vec<Ids> = chunks
            .iter()
            .map(|c| {
                let mut ids = Ids::new();
                Bytes.encode(c, &mut ids);
                ids
            })
            .collect();

        let d = measure_decode(&mut Bytes, &id_chunks, 3).expect("Bytes decodes");
        assert_eq!(d.bytes, text.len(), "round trip must reproduce every byte");
        assert_eq!(d.tokens, text.len(), "one id per byte");
        assert!(d.mbps > 0.0 && d.mbps.is_finite());
        assert!(d.ns_per_token > 0.0 && d.ns_per_token.is_finite());
        assert_eq!(d.text_hash, text_hash(&text), "hash of the decoded text");

        // An engine with no decode entry point is excluded from the decode
        // ranking, not scored zero in it.
        assert!(measure_decode(&mut EncodeOnly, &id_chunks, 1).is_err());

        // The hash has to discriminate, or verification is a no-op that
        // passes everything.
        assert_ne!(text_hash("abc"), text_hash("acb"));
    }

    #[test]
    fn latency_harness_measures_each_distinct_document_once() {
        let documents: Vec<String> = (0..101)
            .map(|index| format!("document {index:03} with distinct text"))
            .collect();
        let measured = measure_latency(&mut Bytes, &documents, 512)
            .expect("one warm-up document and 100 samples");
        assert_eq!(measured.samples, 100);
        assert_eq!(measured.document_bytes, 512);
        assert!(measured.p50_us > 0.0);
        assert!(measured.p99_us >= measured.p50_us);
    }

    /// No document may be encoded twice inside the timed region.
    ///
    /// This is the guard for the replay bug: warming on the same chunks the
    /// timed passes then re-encode turns the measurement into a test of the
    /// engine's memoization (gigatoken/llama-2 measured 256x too fast on
    /// `dense` that way). Warm-up, each rep, and the verification pass must
    /// touch disjoint slices, so no chunk is seen more than twice in total —
    /// once by warm-up-or-a-rep, once by verification.
    #[test]
    fn timed_passes_never_replay_a_document() {
        use std::collections::HashMap;
        use std::sync::Mutex;

        static SEEN: Mutex<Option<HashMap<String, usize>>> = Mutex::new(None);

        struct Recording;
        impl Engine for Recording {
            fn info(&self) -> Info {
                Info {
                    name: "recording",
                    version: "0",
                    lang: "rust",
                    class: Class::Native,
                    url: "",
                    also_computes: "",
                    internally_parallel: false,
                }
            }
            fn encode(&mut self, text: &str, out: &mut Ids) {
                let mut g = SEEN.lock().unwrap();
                *g.get_or_insert_with(HashMap::new)
                    .entry(text.to_string())
                    .or_insert(0) += 1;
                out.push(text.len() as u32);
            }
        }

        // 24 distinct chunks, 5 reps -> 6 slices of 4 chunks each.
        let text: String = (0..24)
            .map(|i| format!("{:-<1024}", format!("doc{i} ")))
            .collect();
        let chunks = chunk(&text, 1024, 100);
        assert!(chunks.len() >= 24);

        *SEEN.lock().unwrap() = Some(HashMap::new());
        let m = measure(&mut Recording, &chunks, 5, true);
        assert!(!m.reused, "24 chunks is enough for 5 reps + warm-up");

        let seen = SEEN.lock().unwrap().take().unwrap();
        let worst = seen.values().copied().max().unwrap_or(0);
        assert!(
            worst <= 2,
            "a chunk was encoded {worst} times; timed slices must be disjoint \
             (at most one timed/warm touch plus one verification touch)"
        );
        assert_eq!(
            seen.len(),
            chunks.len(),
            "verification must still cover the whole corpus"
        );
    }

    /// The work-stealing loop must hand every document to exactly one thread.
    /// If it dropped documents the corpus would shrink and throughput would
    /// look better with more threads; if it double-counted them, worse. Either
    /// way the scaling curve would be fiction, so this is checked directly.
    #[test]
    fn scaling_covers_every_document_exactly_once() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        static SEEN: AtomicUsize = AtomicUsize::new(0);

        struct Counting;
        impl Engine for Counting {
            fn info(&self) -> Info {
                Info {
                    name: "counting",
                    version: "0",
                    lang: "rust",
                    class: Class::Native,
                    url: "",
                    also_computes: "",
                    internally_parallel: false,
                }
            }
            fn encode(&mut self, text: &str, out: &mut Ids) {
                SEEN.fetch_add(text.len(), Ordering::Relaxed);
                out.push(text.len() as u32);
            }
        }

        let text = "lorem ipsum dolor sit amet ".repeat(400);
        let chunks = chunk(&text, 512, 100);
        let total: usize = chunks.iter().map(|c| c.len()).sum();
        let counts = vec![1usize, 2, 4];
        let reps = 1;

        let made = Arc::new(AtomicUsize::new(0));
        let m = made.clone();
        let make = move || -> Option<Box<dyn Engine>> {
            m.fetch_add(1, Ordering::Relaxed);
            Some(Box::new(Counting))
        };

        SEEN.store(0, Ordering::Relaxed);
        let scaling = measure_scaling(&make, &chunks, &counts, reps, Padding::Off).unwrap();
        assert_eq!(
            scaling.kind,
            ScalingKind::External,
            "an engine that refuses set_threads has to be threaded by the harness"
        );
        let pts = scaling.points;
        assert_eq!(pts.len(), counts.len());

        let warm_bytes: usize = chunks
            .iter()
            .enumerate()
            .filter(|(index, _)| index % 6 == 0)
            .map(|(_, chunk)| chunk.len())
            .sum();
        let measured_bytes = total - warm_bytes;
        let engines_per_rep: usize = counts.iter().sum();
        let expected = (warm_bytes * engines_per_rep + measured_bytes * counts.len()) * reps;
        assert_eq!(
            SEEN.load(Ordering::Relaxed),
            expected,
            "every document must be encoded exactly once per timed pass"
        );
        // Plus the probes: one instance to ask about padding and native
        // batch, then one per requested thread count to ask whether the engine
        // will take it. They encode nothing, so they do not appear in `SEEN`,
        // and they are built outside every timed region.
        assert_eq!(
            made.load(Ordering::Relaxed),
            engines_per_rep * reps + 1 + counts.len(),
            "one engine per thread and repetition, plus the capability probes"
        );

        // The single-thread point is the baseline, so it is 100% by definition.
        assert!((pts[0].efficiency_pct - 100.0).abs() < 1e-6);
        for p in &pts {
            assert!(p.mbps > 0.0 && p.mbps.is_finite());
        }
    }

    #[test]
    fn scaling_can_be_measured_in_reverse_but_is_reported_in_order() {
        let chunks = vec![
            "synthetic unit-test warm-up chunk".repeat(100),
            "synthetic unit-test measured chunk".repeat(100),
        ];
        let make = || Some(Box::new(Bytes) as Box<dyn Engine>);
        let pts = measure_scaling(&make, &chunks, &[4, 2, 1], 1, Padding::Off)
            .unwrap()
            .points;

        assert_eq!(
            pts.iter().map(|point| point.threads).collect::<Vec<_>>(),
            vec![1, 2, 4]
        );
        assert!((pts[0].efficiency_pct - 100.0).abs() < 1e-6);
    }

    /// An engine that owns its threading must be driven through its own batch
    /// entry point -- one instance, `set_threads`, one `encode_batch` -- and
    /// never cloned per thread. This is the regression the external-only
    /// harness shipped: it reported the harness's scaling as the engine's.
    #[test]
    fn scaling_uses_the_engines_own_parallelism_when_it_has_any() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        static MADE: AtomicUsize = AtomicUsize::new(0);
        static BATCHES: AtomicUsize = AtomicUsize::new(0);

        /// Reports a thread count it was actually given, and counts how often
        /// it was constructed and called as a batch.
        struct Pooled {
            threads: Arc<AtomicUsize>,
        }
        impl Engine for Pooled {
            fn info(&self) -> Info {
                Info {
                    name: "pooled",
                    version: "0",
                    lang: "rust",
                    class: Class::Native,
                    url: "",
                    also_computes: "",
                    internally_parallel: true,
                }
            }
            fn encode(&mut self, text: &str, out: &mut Ids) {
                out.extend(text.as_bytes().iter().map(|&b| b as u32));
            }
            fn set_threads(&mut self, threads: usize) -> bool {
                self.threads.store(threads, Ordering::Relaxed);
                true
            }
            fn encode_batch(&mut self, texts: &[&str], out: &mut Ids) {
                BATCHES.fetch_add(1, Ordering::Relaxed);
                for text in texts {
                    self.encode(text, out);
                }
            }
        }

        let chunks = vec![
            "warm chunk for the pooled engine".repeat(100),
            "measured chunk for the pooled engine".repeat(100),
            "another measured chunk for the pooled engine".repeat(100),
        ];
        let seen_threads = Arc::new(AtomicUsize::new(0));
        let t = seen_threads.clone();
        MADE.store(0, Ordering::Relaxed);
        BATCHES.store(0, Ordering::Relaxed);
        let make = move || -> Option<Box<dyn Engine>> {
            MADE.fetch_add(1, Ordering::Relaxed);
            Some(Box::new(Pooled { threads: t.clone() }))
        };

        // 1, 2, 8 rather than 1, 2, 4: internal mode builds
        // `1 + 2 * counts.len()` engines, and with 1, 2, 4 that happens to
        // equal `1 + 2 + 4`, so the assertion below could not tell correct
        // behaviour from one-engine-per-thread.
        let counts = [1usize, 2, 8];
        let scaling = measure_scaling(&make, &chunks, &counts, 1, Padding::Off).unwrap();

        assert_eq!(
            scaling.kind,
            ScalingKind::Internal,
            "an engine answering set_threads must be measured through its own pool"
        );
        assert_eq!(scaling.points.len(), counts.len());
        assert_eq!(
            seen_threads.load(Ordering::Relaxed),
            8,
            "the last thread count must have reached the engine"
        );
        // One capability probe, one prefilter probe per thread count, and one
        // engine per timed point -- NOT one engine per thread. The number to
        // fear is 1 + 2 + 8 = 11 timed engines, which would mean the harness
        // had gone back to re-implementing the parallelism it is supposed to
        // be delegating.
        assert_eq!(
            MADE.load(Ordering::Relaxed),
            1 + 2 * counts.len(),
            "internal mode builds one engine per point, plus probes"
        );
        // Two batch calls per point: one warm, one timed.
        assert_eq!(BATCHES.load(Ordering::Relaxed), counts.len() * 2);
        assert!((scaling.points[0].efficiency_pct - 100.0).abs() < 1e-6);
    }

    /// A library with its own batch fan-out but no width knob (tokie reads
    /// `available_parallelism()` and exposes nothing) must still be driven
    /// through that batch path. The bug this pins: classifying it as
    /// thread-less and then threading it from outside, which both idles its
    /// fan-out and reports the harness's curve as the engine's.
    #[test]
    fn an_uncontrollable_native_batch_is_never_threaded_by_the_harness() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static SINGLES: AtomicUsize = AtomicUsize::new(0);
        static BATCHES: AtomicUsize = AtomicUsize::new(0);

        struct FixedWidth;
        impl Engine for FixedWidth {
            fn info(&self) -> Info {
                Info {
                    name: "fixed-width",
                    version: "0",
                    lang: "rust",
                    class: Class::Native,
                    url: "",
                    also_computes: "",
                    internally_parallel: true,
                }
            }
            fn encode(&mut self, text: &str, out: &mut Ids) {
                SINGLES.fetch_add(1, Ordering::Relaxed);
                out.extend(text.as_bytes().iter().map(|&b| b as u32));
            }
            fn has_native_batch(&self) -> bool {
                true
            }
            fn set_threads(&mut self, _threads: usize) -> bool {
                false
            }
            fn encode_batch(&mut self, texts: &[&str], out: &mut Ids) {
                BATCHES.fetch_add(1, Ordering::Relaxed);
                for text in texts {
                    out.extend(text.as_bytes().iter().map(|&b| b as u32));
                }
            }
        }

        let chunks = vec![
            "warm".repeat(50),
            "measured one".repeat(50),
            "measured two".repeat(50),
        ];
        SINGLES.store(0, Ordering::Relaxed);
        BATCHES.store(0, Ordering::Relaxed);
        let make = || Some(Box::new(FixedWidth) as Box<dyn Engine>);

        let scaling = measure_scaling(&make, &chunks, &[1, 2, 4], 1, Padding::Off).unwrap();

        assert_eq!(
            scaling.kind,
            ScalingKind::Internal,
            "a native batch must be measured as internal, never threaded outside"
        );
        // One point, and `threads: 0` for "the width the engine chose" rather
        // than a number the harness did not set.
        assert_eq!(scaling.points.len(), 1);
        assert_eq!(scaling.points[0].threads, 0);
        assert_eq!(
            scaling.points[0].efficiency_pct, 0.0,
            "no 1-thread baseline exists, so efficiency must not be invented"
        );
        assert!(scaling.points[0].mbps > 0.0);
        // Warm + timed, through the batch path only. Any single-document
        // `encode` call would mean the harness had threaded it itself.
        assert_eq!(BATCHES.load(Ordering::Relaxed), 2);
        assert_eq!(
            SINGLES.load(Ordering::Relaxed),
            0,
            "the harness must not fall back to per-document encode"
        );
    }

    /// An engine with no padding of its own must yield no padded cell at all.
    /// The failure this prevents is a padded column quietly filled with
    /// unpadded numbers, which makes every engine in it look equally good at
    /// something only some of them did.
    #[test]
    fn padded_cells_do_not_exist_for_engines_that_cannot_pad() {
        let chunks = vec![
            "warm chunk".repeat(50),
            "measured chunk".repeat(50),
            "another measured chunk".repeat(50),
        ];
        let make = || Some(Box::new(Bytes) as Box<dyn Engine>);

        // `Bytes` takes the default `set_padding`, which accepts Off only.
        assert!(
            measure_scaling(&make, &chunks, &[1, 2], 1, Padding::Off).is_some(),
            "ragged output is always available"
        );
        assert!(
            measure_scaling(&make, &chunks, &[1, 2], 1, Padding::Longest).is_none(),
            "an engine that cannot pad must report no padded result"
        );
    }

    /// Chunk boundaries must never split a multi-byte character, or engines
    /// would be fed invalid text and disagree for the wrong reason.
    #[test]
    fn chunks_respect_char_boundaries() {
        let text = "日本語のテキストです。".repeat(200);
        for c in chunk(&text, 100, 50) {
            assert!(std::str::from_utf8(c.as_bytes()).is_ok());
        }
    }
}
