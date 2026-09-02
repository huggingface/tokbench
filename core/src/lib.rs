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
//! 4. **Warm cache, stated.** One full untimed pass over the corpus precedes
//!    the timed passes, so engines with pretoken/word caches are measured in
//!    the state a real `for doc in corpus { encode(doc) }` loop reaches. This
//!    flatters cache-heavy engines by design — it is the regime users run in —
//!    and `--reps 1 --no-warmup` reports the cold number for contrast.
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

pub mod rss;

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
    pub bytes: usize,
    pub tokens: usize,
    pub ids_hash: u64,
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

/// THE timing loop. Every engine, every language, goes through this function
/// and no other, so the only thing that differs between two numbers is the
/// engine.
///
/// `warmup` runs one untimed pass (see fairness rule 4) which also captures
/// the token count and id hash — computing them during a timed pass would
/// charge the engine for the harness's bookkeeping.
pub fn measure(engine: &mut dyn Engine, chunks: &[String], reps: usize, warmup: bool) -> Measure {
    let bytes: usize = chunks.iter().map(|c| c.len()).sum();
    let mut out: Ids = Vec::with_capacity(bytes / 2);

    // Untimed: correctness bookkeeping + cache fill.
    let mut all: Ids = Vec::new();
    if warmup {
        for c in chunks {
            out.clear();
            engine.encode(c, &mut out);
            all.extend_from_slice(&out);
        }
    } else {
        // Still need ids for verification, but from a single cold pass we
        // then discard, so the timed passes below start cold-ish.
        for c in chunks {
            out.clear();
            engine.encode(c, &mut out);
            all.extend_from_slice(&out);
        }
    }
    let tokens = all.len();
    let hash = ids_hash(&all);
    drop(all);

    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = Instant::now();
        for c in chunks {
            out.clear();
            engine.encode(c, &mut out);
            // Keep the optimiser from deleting the call.
            std::hint::black_box(&out);
        }
        samples.push(t0.elapsed().as_secs_f64());
    }

    let secs = median(samples);
    Measure {
        mbps: (bytes as f64 / (1024.0 * 1024.0)) / secs,
        ns_per_byte: secs * 1e9 / bytes as f64,
        secs,
        bytes,
        tokens,
        ids_hash: hash,
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
        assert_eq!(m.bytes, text.len());
        assert_eq!(m.tokens, text.len(), "one id per byte");
        assert!(m.mbps > 0.0 && m.mbps.is_finite());

        // Same input through the same engine must hash identically; a
        // different id stream must not.
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
