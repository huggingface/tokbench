//! The fairness contract every engine in `engines/` is measured through.
//!
//! There is exactly ONE timing loop (`measure`) and exactly ONE trait
//! (`Engine`). An engine folder's only job is to turn a model directory into
//! something that answers "give me the token ids for this `&str`". Everything
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

    let mut all: Ids = Vec::new();
    let secs;

    if warmup {
        // Untimed pass: fills caches and captures the ids for verification.
        for c in chunks {
            out.clear();
            engine.encode(c, &mut out);
            all.extend_from_slice(&out);
        }
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
        secs = median(samples);
    } else {
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
    }
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

/// Measure throughput at each thread count and derive scaling efficiency.
///
/// # How the work is parallelised, and why this way
///
/// Each thread gets **its own engine instance** (built by `make`, outside the
/// timer) and pulls documents off a shared atomic cursor. Two deliberate
/// choices:
///
/// * **Per-thread engines, not one shared engine.** Most of these libraries
///   are not `Sync`, and those that are often hide a mutex around a shared
///   cache. Giving every thread its own instance measures the best case the
///   library can offer, so a poor scaling number is a real property of the
///   engine rather than an artefact of how the harness shared it.
/// * **Work stealing, not a static split.** Documents differ in cost by more
///   than 10x across scripts. A contiguous split would leave threads idle at
///   the end and report that as poor scaling; a shared cursor keeps every
///   thread busy until the corpus is done, so what is measured is the engine,
///   not the partitioning.
///
/// Warm-up runs per thread over the whole corpus, matching what
/// [`measure`] does for the single-thread case, so the 1-thread point of this
/// curve is directly comparable with the headline number.
///
/// Engines that parallelise *internally* will show >100% efficiency here,
/// because they were already using more than one core at "1 thread". That is
/// why [`Info::internally_parallel`] exists and why the report flags it.
///
/// `target_secs` is how long one timed pass should last; the corpus is walked
/// as many times as needed to reach it (see below). Pass `0.0` to walk it
/// exactly once, which is only appropriate for tests.
pub fn measure_scaling(
    make: &(dyn Fn() -> Option<Box<dyn Engine>> + Sync),
    chunks: &[String],
    counts: &[usize],
    reps: usize,
    target_secs: f64,
) -> Vec<ThreadPoint> {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let bytes: usize = chunks.iter().map(|c| c.len()).sum();
    let mut out: Vec<ThreadPoint> = Vec::new();
    let mut base = f64::NAN;

    // How many times to walk the corpus inside ONE timed pass.
    //
    // Without this the sweep measures thread spawning, not tokenizing. A 200 kB
    // corpus is ~20 documents; at 14 threads that is 1.4 documents each, and a
    // fast engine finishes the whole corpus in a few hundred microseconds —
    // less than it costs to start the threads. The result is an efficiency
    // figure that collapses toward zero for precisely the fastest engines,
    // which looks like a damning scaling result and is pure artefact.
    //
    // So: time one warm single-thread pass, then repeat the corpus enough times
    // that the timed region is ~100 ms. Thread start-up becomes noise, and every
    // thread has real work queued. The corpus is walked in order and each
    // document is encoded the same number of times by construction, so the
    // measured throughput still refers to distinct documents rather than one
    // document replayed out of cache.
    let repeat = if target_secs <= 0.0 {
        1
    } else {
        let mut probe = match make() {
            Some(e) => e,
            None => return out,
        };
        let mut buf: Ids = Vec::new();
        for c in chunks {
            buf.clear();
            probe.encode(c, &mut buf); // warm
        }
        let t = Instant::now();
        for c in chunks {
            buf.clear();
            probe.encode(c, &mut buf);
        }
        let one = t.elapsed().as_secs_f64();
        if one > 0.0 {
            ((target_secs / one).ceil() as usize).clamp(1, 10_000)
        } else {
            1
        }
    };
    let total_units = chunks.len() * repeat;
    let total_bytes = bytes * repeat;

    for &n in counts {
        // Build and warm every engine before the clock starts.
        let mut engines: Vec<Box<dyn Engine>> = Vec::with_capacity(n);
        for _ in 0..n {
            match make() {
                Some(e) => engines.push(e),
                None => return out,
            }
        }
        for e in engines.iter_mut() {
            let mut buf: Ids = Vec::new();
            for c in chunks {
                buf.clear();
                e.encode(c, &mut buf);
            }
        }

        let mut samples = Vec::with_capacity(reps);
        for _ in 0..reps {
            let cursor = AtomicUsize::new(0);
            let t0 = Instant::now();
            std::thread::scope(|s| {
                for e in engines.iter_mut() {
                    let cursor = &cursor;
                    s.spawn(move || {
                        let mut buf: Ids = Vec::with_capacity(4096);
                        loop {
                            let i = cursor.fetch_add(1, Ordering::Relaxed);
                            if i >= total_units {
                                break;
                            }
                            buf.clear();
                            e.encode(&chunks[i % chunks.len()], &mut buf);
                            std::hint::black_box(&buf);
                        }
                    });
                }
            });
            samples.push(t0.elapsed().as_secs_f64());
        }

        let secs = median(samples);
        let mbps = (total_bytes as f64 / (1024.0 * 1024.0)) / secs;
        if n == counts[0] {
            base = mbps;
        }
        let ideal = base * n as f64 / counts[0] as f64;
        out.push(ThreadPoint {
            threads: n,
            mbps,
            efficiency_pct: if ideal > 0.0 {
                mbps / ideal * 100.0
            } else {
                0.0
            },
        });
    }
    out
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
        // target_secs = 0 → walk the corpus exactly once per pass, so the
        // expected byte count below is exact.
        let pts = measure_scaling(&make, &chunks, &counts, reps, 0.0);
        assert_eq!(pts.len(), counts.len());

        // Warm-up encodes the whole corpus once per engine, and each timed rep
        // encodes it once in total across all threads. For counts [1,2,4]:
        // warm-up = (1+2+4) corpora, timed = 3 corpora (one per count).
        let engines: usize = counts.iter().sum();
        let expected = total * engines + total * counts.len() * reps;
        assert_eq!(
            SEEN.load(Ordering::Relaxed),
            expected,
            "every document must be encoded exactly once per timed pass"
        );
        assert_eq!(
            made.load(Ordering::Relaxed),
            engines,
            "one engine per thread"
        );

        // The single-thread point is the baseline, so it is 100% by definition.
        assert!((pts[0].efficiency_pct - 100.0).abs() < 1e-6);
        for p in &pts {
            assert!(p.mbps > 0.0 && p.mbps.is_finite());
        }
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
