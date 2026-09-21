//! The measurement contract. One timing loop per direction, one `Engine`
//! trait, and every engine goes through them unchanged:
//!
//! 1. Ids are hashed and compared against the reference; a mismatch is never
//!    ranked.
//! 2. Load, vocabulary parsing and automaton construction happen before the
//!    clock starts.
//! 3. Warm-up and every timed rep get disjoint slices, so no engine is ever
//!    handed text it has already encoded.
//! 4. Single thread by default; the scaling sweep is a separate, labelled axis.
//! 5. Adapters call the library's ordinary public API. A conversion the public
//!    path forces stays in the number, because the user pays it too.

pub mod mem;

use std::path::{Path, PathBuf};
use std::time::Instant;

pub type Ids = Vec<u32>;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Class {
    Native,
    Cffi,
}

impl Class {
    pub fn as_str(self) -> &'static str {
        match self {
            Class::Native => "native",
            Class::Cffi => "cffi",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Info {
    pub name: &'static str,
    pub version: &'static str,
    pub lang: &'static str,
    pub class: Class,
    pub url: &'static str,
    pub also_computes: &'static str,
    pub internally_parallel: bool,
}

#[derive(Clone, Debug)]
pub struct Model {
    pub name: String,
    pub dir: PathBuf,
}

impl Model {
    pub fn tokenizer_json(&self) -> PathBuf {
        self.dir.join("tokenizer.json")
    }
    pub fn artifact(&self, file: &str) -> Option<PathBuf> {
        let p = self.dir.join(file);
        p.exists().then_some(p)
    }
}

#[derive(Debug)]
pub struct Unsupported(pub String);

impl std::fmt::Display for Unsupported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

pub fn unsupported<T>(msg: impl Into<String>) -> Result<T, Unsupported> {
    Err(Unsupported(msg.into()))
}

pub trait Engine: Send {
    fn info(&self) -> Info;

    fn encode(&mut self, text: &str, out: &mut Ids);

    fn set_threads(&mut self, _threads: usize) -> bool {
        false
    }

    fn encode_batch(&mut self, texts: &[&str], out: &mut Ids) {
        for text in texts {
            self.encode(text, out);
        }
    }

    fn has_native_batch(&self) -> bool {
        false
    }

    fn set_padding(&mut self, padding: Padding) -> bool {
        matches!(padding, Padding::Off)
    }

    fn decode(&mut self, _ids: &[u32], _out: &mut String) -> Result<(), Unsupported> {
        unsupported("library exposes no decode entry point")
    }
}

pub trait Build {
    fn build(model: &Model) -> Result<Box<dyn Engine>, Unsupported>
    where
        Self: Sized;

    fn build_with_cache_capacity(
        model: &Model,
        cache_capacity: Option<usize>,
    ) -> Result<Box<dyn Engine>, Unsupported>
    where
        Self: Sized,
    {
        match cache_capacity {
            None => Self::build(model),
            Some(_) => Err(Unsupported(
                "this library exposes no configurable cache capacity".into(),
            )),
        }
    }

    fn build_without_cache(_model: &Model) -> Result<Box<dyn Engine>, Unsupported>
    where
        Self: Sized,
    {
        Err(Unsupported(
            "this library exposes no way to disable its caches".into(),
        ))
    }

    fn build_without_cache_with_capacity(
        model: &Model,
        cache_capacity: Option<usize>,
    ) -> Result<Box<dyn Engine>, Unsupported>
    where
        Self: Sized,
    {
        match cache_capacity {
            None => Self::build_without_cache(model),
            Some(_) => Err(Unsupported(
                "an explicit cache capacity cannot be applied to a no-cache engine".into(),
            )),
        }
    }
}

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

pub fn text_hash(text: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in text.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[derive(Clone, Debug)]
pub struct Measure {
    pub mbps: f64,
    pub ns_per_byte: f64,
    pub secs: f64,
    pub bytes: usize,
    pub tokens: usize,
    pub ids_hash: u64,
    pub reused: bool,
}

#[derive(Clone, Debug)]
pub struct DecodeMeasure {
    pub mbps: f64,
    pub ns_per_token: f64,
    pub secs: f64,
    pub bytes: usize,
    pub tokens: usize,
    pub text_hash: u64,
}

#[derive(Clone, Debug)]
pub struct LatencyMeasure {
    pub p50_us: f64,
    pub p99_us: f64,
    pub samples: usize,
    pub document_bytes: usize,
}

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
            // Not enough distinct text; flag rather than replay silently.
            reused = true;
        }
        let per = per.max(1);
        let slice = |i: usize| -> &[String] {
            let start = (i * per) % chunks.len().max(1);
            let end = (start + per).min(chunks.len());
            &chunks[start..end]
        };

        // Warm on slice 0; verification below hashes the whole corpus, so
        // every engine is still hashed over identical text.
        for c in slice(0) {
            out.clear();
            engine.encode(c, &mut out);
        }

        // Slices differ in byte count, so the comparable quantity is s/byte.
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
                std::hint::black_box(&out);
            }
            spb.push(t0.elapsed().as_secs_f64() / sb as f64);
            timed_bytes = sb;
        }
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
        // Exactly one cold pass exists per instance, so `reps` cannot apply:
        // a second pass is warm by definition.
        let t0 = Instant::now();
        for c in chunks {
            out.clear();
            engine.encode(c, &mut out);
            std::hint::black_box(&out);
        }
        secs = t0.elapsed().as_secs_f64();

        // Ids captured afterwards so the harness's `extend` stays out of the
        // cold number.
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

/// Distinct whitespace-separated words of `text`, most frequent first.
pub fn vocabulary(text: &str) -> Vec<String> {
    let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for w in text.split_whitespace() {
        *counts.entry(w).or_insert(0) += 1;
    }
    let mut v: Vec<(&str, usize)> = counts.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    v.into_iter().map(|(w, _)| w.to_string()).collect()
}

/// Synthetic text whose whitespace pretokens recur `rate` times on average,
/// drawn from `vocab`.
///
/// Repetition is the variable that dominates BPE throughput: an engine whose
/// speed comes from a pretoken cache is fast in proportion to it. Sweeping it
/// is what separates the cache from the merge loop.
///
/// # The vocabulary is sampled uniformly, and that is the whole difficulty
///
/// Only the repetition rate may vary across a sweep. A first version took the
/// `rate`-appropriate number of words off the front of a frequency-ordered
/// vocabulary, and topped it up with generated tokens when the corpus ran
/// short. Both are fatal. Frequency order means the high-recurrence points are
/// built from the shortest, most common words and the low ones drag in every
/// rare long word, and generated filler tokenizes far worse than real text --
/// so tokens per byte moved by 2.6x across the sweep and every engine appeared
/// to speed up with repetition, including engines with no cache at all.
///
/// Sampling uniformly at random keeps the word-length and token-density
/// distribution the same at every point, so the curve isolates repetition.
/// `tokens` in the report is the audit: it should stay roughly flat across a
/// sweep, and a sweep where it does not has measured something else.
///
/// The corpus's own diversity is a floor on the achievable rate, and no filler
/// is invented to get under it. Returns the recurrence ACTUALLY achieved --
/// report that, never the request.
pub fn synth_recurrence(vocab: &[String], rate: f64, bytes: usize, seed: u64) -> (String, f64) {
    let mut state = seed | 1;
    let mut rand = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };

    let avg = if vocab.is_empty() {
        7.0
    } else {
        vocab.iter().map(|w| w.len() + 1).sum::<usize>() as f64 / vocab.len() as f64
    };
    let want_words = ((bytes as f64 / avg).ceil() as usize).max(1);
    let want_vocab =
        (((want_words as f64 / rate.max(1.0)).ceil() as usize).max(1)).min(vocab.len());

    // Uniform sample without replacement: a partial Fisher-Yates over indices.
    let mut idx: Vec<usize> = (0..vocab.len()).collect();
    for i in 0..want_vocab {
        let j = i + (rand() % (idx.len() - i) as u64) as usize;
        idx.swap(i, j);
    }
    let pool: Vec<&str> = idx[..want_vocab]
        .iter()
        .map(|&i| vocab[i].as_str())
        .collect();

    let mut out = String::with_capacity(bytes + 32);
    let mut used = vec![false; pool.len()];
    let mut distinct = 0usize;
    let mut words = 0usize;
    loop {
        let i = (rand() % pool.len() as u64) as usize;
        // Stop on a word boundary. Truncating to exactly `bytes` would leave a
        // fragment that is not a word from the corpus, and pretokenizes as its
        // own thing.
        if !out.is_empty() && out.len() + 1 + pool[i].len() > bytes {
            break;
        }
        if !used[i] {
            used[i] = true;
            distinct += 1;
        }
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(pool[i]);
        words += 1;
    }
    (out, words as f64 / distinct.max(1) as f64)
}

pub fn read_corpus(path: &Path) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

// ---------------------------------------------------------------------------
// Multi-thread scaling
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct ThreadPoint {
    pub threads: usize,
    pub mbps: f64,
    pub efficiency_pct: f64,
}

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Padding {
    Off,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalingKind {
    NativeThreads,
    IndependentInstances,
}

impl ScalingKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ScalingKind::NativeThreads => "native-threads",
            ScalingKind::IndependentInstances => "independent-instances",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ScalingMode {
    #[default]
    Auto,
    NativeThreads,
    IndependentInstances,
}

#[derive(Clone, Debug)]
pub struct Scaling {
    pub kind: ScalingKind,
    pub padding: Padding,
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

pub fn measure_scaling(
    make: &(dyn Fn() -> Option<Box<dyn Engine>> + Sync),
    chunks: &[String],
    counts: &[usize],
    reps: usize,
    padding: Padding,
) -> Option<Scaling> {
    measure_scaling_with_mode(make, chunks, counts, reps, padding, ScalingMode::Auto)
}

pub fn measure_scaling_with_mode(
    make: &(dyn Fn() -> Option<Box<dyn Engine>> + Sync),
    chunks: &[String],
    counts: &[usize],
    reps: usize,
    padding: Padding,
    mode: ScalingMode,
) -> Option<Scaling> {
    if chunks.len() < 2 || reps == 0 {
        return Some(Scaling::empty(ScalingKind::IndependentInstances, padding));
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

    let can_pin_one = settable.contains(&1);
    let (kind, usable, pin_instances) = match mode {
        ScalingMode::Auto if !settable.is_empty() => (ScalingKind::NativeThreads, settable, false),
        ScalingMode::Auto if native_batch => {
            // Has its own batch fan-out but no width control. One point, at
            // whatever width it picked; `threads: 0` records "the engine's own
            // choice" rather than asserting a number the harness did not set.
            (ScalingKind::NativeThreads, vec![0], false)
        }
        ScalingMode::Auto => (ScalingKind::IndependentInstances, counts.to_vec(), false),
        ScalingMode::NativeThreads if !settable.is_empty() => {
            (ScalingKind::NativeThreads, settable, false)
        }
        ScalingMode::NativeThreads if native_batch => (ScalingKind::NativeThreads, vec![0], false),
        ScalingMode::NativeThreads => {
            return Some(Scaling::empty(ScalingKind::NativeThreads, padding));
        }
        ScalingMode::IndependentInstances if native_batch && !can_pin_one => {
            // A fixed-width native pool cannot be made single-threaded. Running
            // several instances would oversubscribe the machine while claiming
            // one thread per instance.
            return Some(Scaling::empty(ScalingKind::IndependentInstances, padding));
        }
        ScalingMode::IndependentInstances => (
            ScalingKind::IndependentInstances,
            counts.to_vec(),
            can_pin_one,
        ),
    };

    // Padding is a property of a batch, so it only means anything where the
    // engine drives its own batch. An externally threaded engine is called one
    // document at a time, and there is no batch to pad to.
    if kind == ScalingKind::IndependentInstances && padding != Padding::Off {
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
                ScalingKind::NativeThreads => {
                    time_native_threads(make, &warm_chunks, &measured_chunks, n, padding)
                }
                ScalingKind::IndependentInstances => time_independent_instances(
                    make,
                    &warm_chunks,
                    &measured_chunks,
                    n,
                    pin_instances,
                ),
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

fn time_native_threads(
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

fn time_independent_instances(
    make: &(dyn Fn() -> Option<Box<dyn Engine>> + Sync),
    warm: &[&str],
    measured: &[&str],
    threads: usize,
    pin_single_thread: bool,
) -> Option<f64> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Barrier;

    // A fresh instance per worker and repetition makes every timed document
    // unseen by that instance. Construction stays outside the timer, as the
    // driver reports it separately.
    let mut engines: Vec<Box<dyn Engine>> = Vec::with_capacity(threads);
    for _ in 0..threads {
        let mut engine = make()?;
        if pin_single_thread && !engine.set_threads(1) {
            return None;
        }
        engines.push(engine);
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
            ScalingKind::IndependentInstances,
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

    #[test]
    fn scaling_uses_the_engines_own_parallelism_when_it_has_any() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        static MADE: AtomicUsize = AtomicUsize::new(0);
        static BATCHES: AtomicUsize = AtomicUsize::new(0);

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

        // 1, 2, 8 rather than 1, 2, 4: native-thread mode builds
        // `1 + 2 * counts.len()` engines, and with 1, 2, 4 that happens to
        // equal `1 + 2 + 4`, so the assertion below could not tell correct
        // behaviour from one-engine-per-thread.
        let counts = [1usize, 2, 8];
        let scaling = measure_scaling(&make, &chunks, &counts, 1, Padding::Off).unwrap();

        assert_eq!(
            scaling.kind,
            ScalingKind::NativeThreads,
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
            "native-thread mode builds one engine per point, plus probes"
        );
        // Two batch calls per point: one warm, one timed.
        assert_eq!(BATCHES.load(Ordering::Relaxed), counts.len() * 2);
        assert!((scaling.points[0].efficiency_pct - 100.0).abs() < 1e-6);

        BATCHES.store(0, Ordering::Relaxed);
        let independent = measure_scaling_with_mode(
            &make,
            &chunks,
            &counts,
            1,
            Padding::Off,
            ScalingMode::IndependentInstances,
        )
        .unwrap();
        assert_eq!(independent.kind, ScalingKind::IndependentInstances,);
        assert_eq!(independent.points.len(), counts.len());
        assert_eq!(
            BATCHES.load(Ordering::Relaxed),
            0,
            "independent instances must use the single-document entry point"
        );
    }

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
            ScalingKind::NativeThreads,
            "a native batch must use native threads, never unpinned independent instances"
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

        let independent = measure_scaling_with_mode(
            &make,
            &chunks,
            &[1, 2, 4],
            1,
            Padding::Off,
            ScalingMode::IndependentInstances,
        )
        .unwrap();
        assert!(
            independent.points.is_empty(),
            "a fixed-width native pool cannot honestly run as single-threaded instances"
        );
    }

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

    /// Two properties, and the second is the one that broke.
    ///
    /// The achieved rate must track the request, and the word-length
    /// distribution must NOT move across the sweep -- a frequency-ordered
    /// vocabulary made the high-recurrence points out of short common words,
    /// so throughput rose with repetition for every engine, cache or no cache.
    #[test]
    fn synthetic_recurrence_varies_only_repetition() {
        let corpus: String = (0..4000)
            .map(|i| format!("w{i:04}{} ", "x".repeat(i % 9)))
            .collect();
        let vocab = vocabulary(&corpus);
        assert!(vocab.len() >= 4000);

        let mut lengths = Vec::new();
        for rate in [2.0, 8.0, 64.0] {
            let (text, got) = synth_recurrence(&vocab, rate, 120_000, 7);
            assert!(text.len() <= 120_000);
            assert!(std::str::from_utf8(text.as_bytes()).is_ok());
            assert!(
                got >= rate * 0.5 && got <= rate * 2.0,
                "asked {rate}x, achieved {got}x"
            );
            let words: Vec<&str> = text.split_whitespace().collect();
            lengths.push(words.iter().map(|w| w.len()).sum::<usize>() as f64 / words.len() as f64);
        }
        let (lo, hi) = (
            lengths.iter().cloned().fold(f64::MAX, f64::min),
            lengths.iter().cloned().fold(0.0, f64::max),
        );
        assert!(
            hi / lo < 1.15,
            "mean word length moved {lo:.2} -> {hi:.2} across the sweep; \
             the sweep is varying more than repetition"
        );
    }

    /// The corpus's diversity is a floor, and it is reported rather than faked
    /// with generated filler, which tokenizes nothing like real text.
    #[test]
    fn recurrence_floor_is_reported_not_invented() {
        let vocab = vocabulary("alpha beta gamma delta");
        let (text, got) = synth_recurrence(&vocab, 1.0, 10_000, 3);
        assert!(got > 100.0, "4 distinct words cannot give 1x, got {got}x");
        for w in text.split_whitespace() {
            assert!(vocab.iter().any(|v| v == w), "invented token {w:?}");
        }
    }

    #[test]
    fn chunks_respect_char_boundaries() {
        let text = "日本語のテキストです。".repeat(200);
        for c in chunk(&text, 100, 50) {
            assert!(std::str::from_utf8(c.as_bytes()).is_ok());
        }
    }
}
