//! `tokbench` — run every compiled-in engine over the model × corpus matrix
//! through the one timing loop in `tokbench_core`, verify the ids, and write
//! the JSON the dashboard reads.
//!
//! There is no benchmarking framework underneath this. That is deliberate and
//! it is the smaller amount of code, not the larger: `divan` has no
//! machine-readable output (JSON/CSV is still a planned feature), and
//! `criterion`'s `estimates.json` is documented as a private implementation
//! detail that may change without warning, needs `cargo-criterion` plus
//! `harness = false`, and owns `fn main()`. Neither can express an
//! engine × model matrix, and neither verifies that two engines produced the
//! same ids — the property that makes a throughput comparison mean anything.
//! Wrapping either would mean parsing its output back into this schema. The
//! measurement itself is ~25 lines in `tokbench_core::measure`, shared by every
//! engine, and that is the whole of it.

mod registry;

use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use tokbench_core::{chunk, measure, measure_decode, measure_latency, Ids, Model, Phases};

/// ~10 kB documents: large enough that per-call overhead is amortised, small
/// enough to stay in cache. Matches the upstream pipeline benchmark so numbers
/// remain comparable with it.
const CHUNK_BYTES: usize = 10 * 1024;
const MAX_CHUNKS: usize = 100;

#[derive(Parser, Debug)]
#[command(
    name = "tokbench",
    about = "Benchmark tokenizer engines through one shared, verified timing loop."
)]
struct Args {
    /// Directory of model directories; each subdirectory holds one model's
    /// artifacts (tokenizer.json plus any engine-specific files).
    #[arg(long, default_value = "data/models")]
    models: PathBuf,

    /// Directory of `.txt` corpora.
    #[arg(long, default_value = "data/fixtures")]
    corpora: PathBuf,

    /// Timed passes per cell; the median is reported.
    #[arg(long, default_value_t = 5)]
    reps: usize,

    /// Skip the warm-up pass to report cold-cache numbers instead.
    #[arg(long)]
    no_warmup: bool,

    /// Only run these engines (repeatable). Default: everything compiled in.
    #[arg(long)]
    engine: Vec<String>,

    /// Only run these models (repeatable).
    #[arg(long)]
    model: Vec<String>,

    #[arg(long, default_value = "tokenizer_bench_results.json")]
    out: PathBuf,

    /// Open the dashboard in a browser once the JSON is written.
    #[arg(long)]
    open: bool,

    /// Interpreter used for scripted (Python) engines.
    #[arg(long, default_value = "python3")]
    python: String,

    /// Skip the per-engine footprint child processes (they roughly double wall
    /// time,
    /// since each one reloads the model).
    #[arg(long)]
    no_memory: bool,

    /// Skip the decode pass. Decode is measured over the reference engine's
    /// ids, so this also skips the extra reference encode that produces them.
    #[arg(long)]
    no_decode: bool,

    /// Measure call-level encode latency on these corpora (repeatable).
    #[arg(long = "latency")]
    latency: Vec<String>,

    /// Maximum bytes in each distinct latency document.
    #[arg(long, default_value_t = 512)]
    latency_bytes: usize,

    /// Number of call-level latency samples per engine and cell.
    #[arg(long, default_value_t = 1_000)]
    latency_samples: usize,

    /// Run the multi-thread scaling sweep on these corpora (repeatable).
    ///
    /// Scoped to named corpora rather than run everywhere because the sweep
    /// builds and warms one engine per thread at every thread count — on an
    /// 8-core box that is ~25 extra full-corpus encodes per engine per cell,
    /// which would dominate the wall time of a full matrix. Scaling behaviour
    /// barely varies by language, so one or two representative corpora give
    /// the same answer for a fraction of the cost.
    #[arg(long = "scaling")]
    scaling: Vec<String>,

    /// Do not include thread counts above this value in scaling sweeps.
    /// Useful on large cloud instances when the published claim is scoped to
    /// a fixed core count such as 8.
    #[arg(long)]
    max_threads: Option<NonZeroUsize>,

    /// Measure scaling points from the highest thread count down to one.
    /// Jobs alternate this with the default order to expose temporal drift.
    #[arg(long)]
    reverse_scaling: bool,

    /// Per-engine stripped binary deltas, as written by `scripts/binsize.sh`.
    /// Merged into the report when present.
    #[arg(long, default_value = "binary_sizes.json")]
    binary_sizes: PathBuf,

    /// Published package sizes, as written by `scripts/package_size.py`.
    #[arg(long, default_value = "package_sizes.json")]
    package_sizes: PathBuf,

    // --- internal: the memory child. One engine per process; see core::mem. ---
    #[arg(long, hide = true)]
    memory: Option<String>,
    #[arg(long, hide = true)]
    memory_model: Option<PathBuf>,
    #[arg(long, hide = true)]
    memory_corpus: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// Output schema. `dataset_metadata` + `results` are the shape the dashboard
// consumes; `runs` carries the full matrix, with the first run mirrored at the
// top level so a naive reader still sees a valid document.
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone, Debug)]
struct DatasetMetadata {
    file_size_bytes: usize,
    total_characters: usize,
    corpus: String,
    model: String,
    reps: usize,
    warmup: bool,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
struct Breakdown {
    normalization: u64,
    pre_tokenization: u64,
    core_encoding: u64,
    post_processing: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
struct EngineResult {
    tokenizer_name: String,
    total_tokens_produced: usize,
    mean_execution_time_seconds: f64,

    /// `None` when the engine's API cannot separate its stages. The dashboard
    /// renders that as "not instrumented"; it never invents a split.
    #[serde(skip_serializing_if = "Option::is_none")]
    breakdown_nanoseconds: Option<Breakdown>,

    // --- fairness disclosure; see tokbench_core's module docs ---
    engine_version: String,
    engine_lang: String,
    /// native | cffi | python | subprocess. Only same-class numbers are ranked
    /// against each other without a caveat.
    engine_class: String,
    /// Work done on the same pass beyond producing ids (e.g. "byte offsets").
    also_computes: String,
    internally_parallel: bool,
    /// Excluded from the timed region and reported separately.
    load_ms: f64,
    mbps: f64,
    ns_per_byte: f64,
    ids_hash: String,
    /// `true`/`false` against the reference engine; `None` when no reference
    /// ran, so nothing could be checked.
    verified: Option<bool>,
    /// Set when the engine could not run this cell, with the reason.
    #[serde(skip_serializing_if = "Option::is_none")]
    unsupported: Option<String>,

    // --- decode direction; all `None` under `--no-decode` ---
    /// MB/s of text produced, decoding the REFERENCE engine's ids. Not the
    /// engine's own ids: see fairness rule 7.
    #[serde(skip_serializing_if = "Option::is_none")]
    decode_mbps: Option<f64>,
    /// The input-side rate, and the one to compare when two engines produce
    /// text of different lengths from the same ids.
    #[serde(skip_serializing_if = "Option::is_none")]
    decode_ns_per_token: Option<f64>,
    /// Hash of the decoded text, the decode-side counterpart of `ids_hash`.
    #[serde(skip_serializing_if = "Option::is_none")]
    decode_text_hash: Option<String>,
    /// `true`/`false` against the reference engine's decoded text; `None`
    /// when no reference decode ran, so nothing could be checked.
    #[serde(skip_serializing_if = "Option::is_none")]
    decode_verified: Option<bool>,
    /// Why this engine produced no decode number — usually that its library
    /// has no decode entry point. Distinct from `unsupported`, which means it
    /// could not encode the cell either.
    #[serde(skip_serializing_if = "Option::is_none")]
    decode_unsupported: Option<String>,

    // --- footprint ---
    /// Memory the *loaded* tokenizer holds, measured in a dedicated child
    /// process so one engine's arenas cannot be credited to another.
    ///
    /// Live heap, not RSS: see `tokbench_core::mem` for why RSS ranked the
    /// engine holding the least as the one holding the most.
    #[serde(skip_serializing_if = "Option::is_none")]
    heap_load_mb: Option<f64>,
    /// Live heap after a full encode pass — the loaded tokenizer plus whatever
    /// caches it fills. The gap to `heap_load_mb` is the cache.
    #[serde(skip_serializing_if = "Option::is_none")]
    heap_encode_mb: Option<f64>,
    /// Published size of the engine's own package — the `.crate` tarball, the
    /// PyPI wheel, or the npm unpacked size. This is the dependency you take
    /// on. From `scripts/package_size.py`.
    #[serde(skip_serializing_if = "Option::is_none")]
    crate_size_kb: Option<f64>,
    /// Which registry and exact version the size above refers to.
    #[serde(skip_serializing_if = "Option::is_none")]
    package_ref: Option<String>,
    /// Stripped bytes this engine adds to a minimal binary, over a no-engine
    /// baseline. A different question from `crate_size_kb`: a small download
    /// can compile to a lot, and vice versa. From `scripts/binsize.sh`.
    #[serde(skip_serializing_if = "Option::is_none")]
    binary_delta_kb: Option<f64>,

    /// Multi-thread scaling curve, when the sweep ran for this cell.
    #[serde(skip_serializing_if = "Option::is_none")]
    scaling: Option<Vec<ScalePoint>>,

    /// True when the corpus could not supply `reps + 1` disjoint slices, so
    /// some text was encoded more than once inside the measurement. Such a
    /// cell partly measures the engine's memoization; the dashboard flags it.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    reused_text: bool,

    /// Call-level encode latency over distinct documents, when requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    latency_p50_us: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    latency_p99_us: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    latency_samples: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    latency_document_bytes: Option<usize>,
}

fn latency_documents(text: &str, max_bytes: usize, samples: usize) -> Vec<String> {
    if max_bytes == 0 || samples == 0 {
        return Vec::new();
    }
    let mut documents = Vec::with_capacity(samples + 1);
    let mut start = 0;
    while start < text.len() && documents.len() < samples + 1 {
        let mut end = (start + max_bytes).min(text.len());
        while end > start && !text.is_char_boundary(end) {
            end -= 1;
        }
        if end == start {
            break;
        }
        documents.push(text[start..end].to_string());
        start = end;
    }
    documents
}

/// One point on an engine's scaling curve.
#[derive(Serialize, Deserialize, Clone, Debug)]
struct ScalePoint {
    threads: usize,
    mbps: f64,
    /// Percentage of perfect linear scaling from the 1-thread number. 100% =
    /// each added core added a full core's worth of throughput.
    efficiency_pct: f64,
}

/// One entry of `package_sizes.json`.
#[derive(Deserialize, Debug, Clone)]
struct PackageSize {
    kb: Option<f64>,
    version: Option<String>,
    registry: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct Run {
    dataset_metadata: DatasetMetadata,
    results: Vec<EngineResult>,
}

#[derive(Serialize, Deserialize, Debug)]
struct Report {
    dataset_metadata: DatasetMetadata,
    results: Vec<EngineResult>,
    runs: Vec<Run>,
}

/// What a scripted engine prints on stdout. `python/harness.py` reproduces the
/// Rust protocol exactly and emits this.
#[derive(Deserialize, Debug)]
struct ScriptedReport {
    version: String,
    lang: String,
    secs: f64,
    tokens: usize,
    /// Bytes the runner actually encoded. Reported by the runner rather than
    /// taken from the file size: the harness caps at `max_chunks`, so a large
    /// corpus is only partly consumed and dividing by the full file size would
    /// silently inflate every scripted engine's MB/s.
    bytes: usize,
    ids_hash: u64,
    load_ms: f64,
    #[serde(default)]
    also_computes: String,
    #[serde(default)]
    internally_parallel: bool,
    #[serde(default)]
    phases: Option<BTreeMap<String, u64>>,
    #[serde(default)]
    unsupported: Option<String>,
}

fn list_dir(dir: &Path, want_dir: bool, ext: Option<&str>) -> Result<Vec<PathBuf>> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            if want_dir {
                p.is_dir()
            } else {
                p.is_file() && ext.is_none_or(|x| p.extension().is_some_and(|e| e == x))
            }
        })
        .collect();
    v.sort();
    Ok(v)
}

fn stem(p: &Path) -> String {
    p.file_stem().unwrap_or_default().to_string_lossy().into()
}

fn directory_name(p: &Path) -> String {
    p.file_name().unwrap_or_default().to_string_lossy().into()
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Child mode short-circuits everything: this process exists to load one
    // engine and report its memory, so it must not touch any other.
    if let Some(name) = args.memory.clone() {
        return memory_child(&args, &name);
    }

    // Stripped per-engine binary sizes, if `scripts/binsize.sh` has run.
    // Absent is normal (it needs a release build per engine); the column is
    // simply omitted rather than reported as zero.
    let bin_sizes: BTreeMap<String, f64> = std::fs::read_to_string(&args.binary_sizes)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    if bin_sizes.is_empty() {
        eprintln!(
            "note: no {} — binary-delta column omitted (run scripts/binsize.sh)",
            args.binary_sizes.display()
        );
    }
    let pkg_sizes: BTreeMap<String, PackageSize> = std::fs::read_to_string(&args.package_sizes)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    if pkg_sizes.is_empty() {
        eprintln!(
            "note: no {} — package-size column omitted (run scripts/package_size.py)",
            args.package_sizes.display()
        );
    }

    let models = list_dir(&args.models, true, None)?;
    if models.is_empty() {
        bail!(
            "no model directories under {} — run `make models` first",
            args.models.display()
        );
    }
    let corpora = list_dir(&args.corpora, false, Some("txt"))?;
    if corpora.is_empty() {
        bail!(
            "no .txt corpora under {} — run `make fixtures` first",
            args.corpora.display()
        );
    }

    let natives = registry::native();
    let scripted = registry::scripted();
    let mut thread_sweep = tokbench_core::thread_counts();
    if let Some(max_threads) = args.max_threads {
        thread_sweep.retain(|threads| *threads <= max_threads.get());
    }
    if args.reverse_scaling {
        thread_sweep.reverse();
    }
    if !args.scaling.is_empty() {
        eprintln!(
            "scaling sweep on {:?} at thread counts {:?}",
            args.scaling, thread_sweep
        );
    }
    let want = |n: &str| args.engine.is_empty() || args.engine.iter().any(|e| e == n);

    let cells = models.len() * corpora.len();
    eprintln!(
        "tokbench: {} engine(s) [{} native, {} scripted] × {} model(s) × {} corpus/corpora = {} cells, {} reps each",
        natives.len() + scripted.len(),
        natives.len(),
        scripted.len(),
        models.len(),
        corpora.len(),
        cells,
        args.reps
    );

    let mut runs: Vec<Run> = Vec::new();
    // Model + corpus behind each run, so the footprint pass below can rebuild
    // the same cells without re-deriving them.
    let mut cell_inputs: Vec<(Model, PathBuf)> = Vec::new();
    let started = Instant::now();
    let mut done = 0usize;

    for model_dir in &models {
        // A model directory may contain a dot (for example `glm-5.2`). Using
        // `file_stem` on a directory silently truncated it to `glm-5`, which
        // also made `--model glm-5.2` unable to select it.
        let model_name = directory_name(model_dir);
        if !args.model.is_empty() && !args.model.contains(&model_name) {
            continue;
        }
        let model = Model {
            name: model_name.clone(),
            dir: model_dir.clone(),
        };

        for corpus_path in &corpora {
            let corpus_name = stem(corpus_path);
            let Some(text) = tokbench_core::read_corpus(corpus_path) else {
                eprintln!("  skip {corpus_name}: unreadable");
                continue;
            };
            let chunks = chunk(&text, CHUNK_BYTES, MAX_CHUNKS);
            let bytes: usize = chunks.iter().map(|c| c.len()).sum();
            let chars: usize = chunks.iter().map(|c| c.chars().count()).sum();
            let latency_docs = if args.latency.contains(&corpus_name) {
                let documents = latency_documents(
                    &text,
                    args.latency_bytes,
                    args.latency_samples,
                );
                if documents.len() != args.latency_samples + 1 {
                    bail!(
                        "{model_name}/{corpus_name}: latency needs {} distinct documents, found {}",
                        args.latency_samples + 1,
                        documents.len()
                    );
                }
                documents
            } else {
                Vec::new()
            };

            let mut results: Vec<EngineResult> = Vec::new();

            // The one id stream every engine's decode is measured over
            // (fairness rule 7). Built here, once per cell, from a throwaway
            // reference engine: if each engine decoded its own encode output,
            // an engine that merges harder would feed itself fewer and longer
            // tokens and post a better token rate for strictly less work.
            //
            // Empty means no decode pass — either `--no-decode`, or this
            // binary was compiled without the reference engine, in which case
            // there is no oracle to verify a decode against anyway.
            let ref_ids: Vec<Ids> = if args.no_decode {
                Vec::new()
            } else {
                natives
                    .iter()
                    .find(|(n, _)| *n == registry::REFERENCE)
                    .and_then(|(_, ctor)| ctor(&model).ok())
                    .map(|mut r| {
                        chunks
                            .iter()
                            .map(|c| {
                                let mut ids = Ids::new();
                                r.encode(c, &mut ids);
                                ids
                            })
                            .collect()
                    })
                    .unwrap_or_default()
            };

            // Native engines: built, then timed in-process by core::measure.
            for (name, ctor) in &natives {
                if !want(name) {
                    continue;
                }
                let t0 = Instant::now();
                // Third-party code, adversarial inputs. A panic here is a
                // finding about that engine, not a reason to lose the run.
                let built =
                    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ctor(&model))) {
                        Ok(r) => r,
                        Err(_) => Err(tokbench_core::Unsupported("panicked while loading".into())),
                    };
                let load_ms = t0.elapsed().as_secs_f64() * 1e3;

                match built {
                    Err(why) => {
                        eprintln!("  {model_name}/{corpus_name} {name}: unsupported ({why})");
                        results.push(EngineResult {
                            tokenizer_name: name.to_string(),
                            total_tokens_produced: 0,
                            mean_execution_time_seconds: 0.0,
                            breakdown_nanoseconds: None,
                            engine_version: String::new(),
                            engine_lang: String::new(),
                            engine_class: String::new(),
                            also_computes: String::new(),
                            internally_parallel: false,
                            load_ms,
                            mbps: 0.0,
                            ns_per_byte: 0.0,
                            ids_hash: String::new(),
                            verified: None,
                            unsupported: Some(why.to_string()),
                            ..Default::default()
                        });
                    }
                    Ok(mut engine) => {
                        let info = engine.info();
                        let measured =
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                measure(engine.as_mut(), &chunks, args.reps, !args.no_warmup)
                            }));
                        let Ok(m) = measured else {
                            eprintln!(
                                "  {model_name}/{corpus_name} {name}: PANICKED while encoding"
                            );
                            results.push(EngineResult {
                                tokenizer_name: name.to_string(),
                                engine_version: info.version.into(),
                                engine_lang: info.lang.into(),
                                engine_class: info.class.as_str().into(),
                                load_ms,
                                unsupported: Some(
                                    "panicked while encoding this corpus".to_string(),
                                ),
                                ..Default::default()
                            });
                            continue;
                        };

                        let latency = if latency_docs.is_empty() {
                            None
                        } else {
                            measure_latency(
                                engine.as_mut(),
                                &latency_docs,
                                args.latency_bytes,
                            )
                        };

                        // Stage breakdown on a separate, untimed pass so the
                        // extra clock reads never inflate the headline.
                        let mut phases = Phases::default();
                        let mut any = false;
                        for c in &chunks {
                            if let Some(p) = engine.phases(c) {
                                phases.accumulate(p);
                                any = true;
                            }
                        }

                        // Decode, on the same clock, over the reference's ids.
                        // Runs while this engine is still alive and warm, so
                        // it costs no extra build.
                        let decoded = (!ref_ids.is_empty())
                            .then(|| measure_decode(engine.as_mut(), &ref_ids, args.reps));
                        let (
                            decode_mbps,
                            decode_ns_per_token,
                            decode_text_hash,
                            decode_unsupported,
                        ) = match &decoded {
                            None => (None, None, None, None),
                            Some(Ok(d)) => (
                                Some(d.mbps),
                                Some(d.ns_per_token),
                                Some(format!("{:016x}", d.text_hash)),
                                None,
                            ),
                            Some(Err(why)) => (None, None, None, Some(why.to_string())),
                        };

                        let decode_note = match &decoded {
                            Some(Ok(d)) => format!("  dec {:>7.1} MB/s", d.mbps),
                            // Distinguish "cannot decode" from "was not asked
                            // to", so a silent regression cannot hide as a
                            // blank column.
                            Some(Err(_)) => "  dec    n/a".to_string(),
                            None => String::new(),
                        };

                        eprintln!(
                            "  [{}/{}] {model_name}/{corpus_name} {name:<16} {:>8.1} MB/s  {:>6.2} ns/B  {} tok{}",
                            done + 1,
                            cells,
                            m.mbps,
                            m.ns_per_byte,
                            m.tokens,
                            decode_note
                        );

                        // Multi-thread sweep, only on the corpora asked for.
                        // Runs after the single-thread timing so it can never
                        // perturb the headline number.
                        let scaling = if args.scaling.contains(&corpus_name) {
                            let make = || ctor(&model).ok();
                            let pts =
                                // 100 ms per timed pass: long enough that thread
                                // start-up is noise even for the fastest
                                // engines, which would otherwise finish a small
                                // corpus before the threads were even up.
                                tokbench_core::measure_scaling(
                                    &make,
                                    &chunks,
                                    &thread_sweep,
                                    3,
                                    0.100,
                                );
                            if !pts.is_empty() {
                                let best = pts
                                    .iter()
                                    .max_by_key(|point| point.threads)
                                    .unwrap();
                                eprintln!(
                                    "        threads {} -> {:.1} MB/s ({:.0}% of linear)",
                                    best.threads, best.mbps, best.efficiency_pct
                                );
                            }
                            Some(
                                pts.into_iter()
                                    .map(|p| ScalePoint {
                                        threads: p.threads,
                                        mbps: p.mbps,
                                        efficiency_pct: p.efficiency_pct,
                                    })
                                    .collect(),
                            )
                        } else {
                            None
                        };

                        results.push(EngineResult {
                            tokenizer_name: name.to_string(),
                            total_tokens_produced: m.tokens,
                            mean_execution_time_seconds: m.secs,
                            breakdown_nanoseconds: any.then_some(Breakdown {
                                normalization: phases.normalization_ns,
                                pre_tokenization: phases.pre_tokenization_ns,
                                core_encoding: phases.core_encoding_ns,
                                post_processing: phases.post_processing_ns,
                            }),
                            engine_version: info.version.into(),
                            engine_lang: info.lang.into(),
                            engine_class: info.class.as_str().into(),
                            also_computes: info.also_computes.into(),
                            internally_parallel: info.internally_parallel,
                            load_ms,
                            mbps: m.mbps,
                            ns_per_byte: m.ns_per_byte,
                            ids_hash: format!("{:016x}", m.ids_hash),
                            verified: None,
                            unsupported: None,
                            decode_mbps,
                            decode_ns_per_token,
                            decode_text_hash,
                            decode_verified: None,
                            decode_unsupported,
                            scaling,
                            reused_text: m.reused,
                            latency_p50_us: latency.as_ref().map(|value| value.p50_us),
                            latency_p99_us: latency.as_ref().map(|value| value.p99_us),
                            latency_samples: latency.as_ref().map(|value| value.samples),
                            latency_document_bytes: latency
                                .as_ref()
                                .map(|value| value.document_bytes),
                            ..Default::default()
                        });
                    }
                }
            }

            // Scripted engines: the interpreter re-runs the same protocol and
            // reports back. Timed inside the interpreter, so process start-up
            // and imports are excluded — but still a different class, and
            // labelled as such.
            for (name, script) in &scripted {
                if !want(name) {
                    continue;
                }
                match run_scripted(&args, name, script, &model, corpus_path) {
                    Ok(r) => results.push(r),
                    Err(e) => eprintln!("  {model_name}/{corpus_name} {name}: {e}"),
                }
            }

            // Verification: the reference's id hash is the oracle. An engine
            // that produced different ids did different work, and its speed
            // is not a comparable number.
            if let Some(reference) = results
                .iter()
                .find(|r| r.tokenizer_name == registry::REFERENCE && r.unsupported.is_none())
                .map(|r| r.ids_hash.clone())
            {
                for r in results.iter_mut().filter(|r| r.unsupported.is_none()) {
                    r.verified = Some(r.ids_hash == reference);
                }
                let bad: Vec<&str> = results
                    .iter()
                    .filter(|r| r.verified == Some(false))
                    .map(|r| r.tokenizer_name.as_str())
                    .collect();
                if !bad.is_empty() {
                    eprintln!(
                        "  ! {model_name}/{corpus_name}: ids differ from {}: {}",
                        registry::REFERENCE,
                        bad.join(", ")
                    );
                }
            }

            // Same oracle, one level along: the reference's decoded text. Not
            // the original corpus — a lowercasing or accent-stripping
            // normalizer makes `decode(encode(t)) != t` for a tokenizer that
            // is behaving perfectly, so the corpus cannot be the expectation.
            if let Some(ref_text) = results
                .iter()
                .find(|r| r.tokenizer_name == registry::REFERENCE)
                .and_then(|r| r.decode_text_hash.clone())
            {
                for r in results.iter_mut().filter(|r| r.decode_text_hash.is_some()) {
                    r.decode_verified = Some(r.decode_text_hash.as_deref() == Some(&ref_text));
                }
                let bad: Vec<&str> = results
                    .iter()
                    .filter(|r| r.decode_verified == Some(false))
                    .map(|r| r.tokenizer_name.as_str())
                    .collect();
                if !bad.is_empty() {
                    eprintln!(
                        "  ! {model_name}/{corpus_name}: decoded text differs from {}: {}",
                        registry::REFERENCE,
                        bad.join(", ")
                    );
                }
            }

            for r in results.iter_mut() {
                r.binary_delta_kb = bin_sizes.get(&r.tokenizer_name).copied();
                if let Some(p) = pkg_sizes.get(&r.tokenizer_name) {
                    r.crate_size_kb = p.kb;
                    r.package_ref = match (&p.version, &p.registry) {
                        (Some(v), Some(reg)) => Some(format!("{v} ({reg})")),
                        (Some(v), None) => Some(v.clone()),
                        _ => None,
                    };
                }
            }

            cell_inputs.push((model.clone(), corpus_path.clone()));
            runs.push(Run {
                dataset_metadata: DatasetMetadata {
                    file_size_bytes: bytes,
                    total_characters: chars,
                    corpus: corpus_name,
                    model: model_name.clone(),
                    reps: args.reps,
                    warmup: !args.no_warmup,
                },
                results,
            });
            done += 1;
            let per = started.elapsed().as_secs_f64() / done as f64;
            eprintln!(
                "  -- {done}/{cells} cells | elapsed {:.0}s | eta ~{:.0}s",
                started.elapsed().as_secs_f64(),
                per * (cells - done) as f64
            );
        }
    }

    // ---- Footprint: a SECOND pass, after ALL timing is finished. ----
    //
    // This must not be interleaved with the timed passes, and the reason is
    // measured rather than theoretical. Each cell's footprint costs ~12 child
    // processes, every one loading a full model (llama-3's config alone is
    // 16 MB) and holding 20-150 MB resident. Run between cells, that churn
    // evicts the next cell's warm pages and its cost lands in the NEXT
    // measurement: fastokens on deepseek-v4/english measured 3.5 MB/s
    // interleaved against 62.3 MB/s with `--no-memory` on the identical
    // binary — an 18x error, in the throughput column, caused entirely by the
    // memory column. Separating the passes costs one extra walk of the matrix
    // and removes the interference completely.
    if !args.no_memory {
        let total = runs.len();
        eprintln!(
            "footprint pass: {total} cells (kept separate from timing — the child \
             processes would otherwise evict the next cell's caches)"
        );
        for (i, (run, (model, corpus))) in runs.iter_mut().zip(&cell_inputs).enumerate() {
            for r in run.results.iter_mut().filter(|r| r.unsupported.is_none()) {
                if let Some(m) = measure_memory(&r.tokenizer_name, model, corpus) {
                    r.heap_load_mb = m.heap_load_mb;
                    r.heap_encode_mb = m.heap_encode_mb;
                }
            }
            if (i + 1) % 10 == 0 {
                eprintln!("  footprint {}/{total}", i + 1);
            }
        }
    }

    let first = runs
        .first()
        .cloned()
        .context("no cells ran — every model/corpus pair was skipped")?;
    let report = Report {
        dataset_metadata: first.dataset_metadata,
        results: first.results,
        runs,
    };
    std::fs::write(&args.out, serde_json::to_string_pretty(&report)?)?;
    eprintln!("wrote {}", args.out.display());

    if args.open {
        let dash = Path::new("dashboard.html");
        if dash.exists() {
            open::that(dash).context("opening dashboard.html")?;
        } else {
            eprintln!(
                "dashboard.html not found; drop {} into it manually",
                args.out.display()
            );
        }
    }
    Ok(())
}

/// Run a scripted engine and adapt its report into an [`EngineResult`].
fn run_scripted(
    args: &Args,
    name: &str,
    script: &str,
    model: &Model,
    corpus: &Path,
) -> Result<EngineResult> {
    if !Path::new(script).exists() {
        bail!("{script} not found");
    }
    let mut cmd = if script.ends_with(".mjs") {
        let mut c = Command::new("node");
        c.arg(script);
        c
    } else {
        let mut c = Command::new(&args.python);
        c.arg(script);
        c
    };
    let out = cmd
        .arg("--model")
        .arg(&model.dir)
        .arg("--corpus")
        .arg(corpus)
        .arg("--reps")
        .arg(args.reps.to_string())
        .arg("--chunk-bytes")
        .arg(CHUNK_BYTES.to_string())
        .arg("--max-chunks")
        .arg(MAX_CHUNKS.to_string())
        .output()
        .with_context(|| format!("spawning {script}"))?;

    if !out.status.success() {
        bail!(
            "{script} exited {}: {}",
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    let text = String::from_utf8_lossy(&out.stdout);
    // The harness prints one JSON object as its last line, so an engine that
    // chatters on stdout does not break parsing.
    let line = text
        .lines()
        .rev()
        .find(|l| l.trim_start().starts_with('{'))
        .with_context(|| format!("{script} printed no JSON object"))?;
    let r: ScriptedReport =
        serde_json::from_str(line).with_context(|| format!("parsing {script} output: {line}"))?;

    if let Some(why) = r.unsupported {
        return Ok(EngineResult {
            tokenizer_name: name.into(),
            total_tokens_produced: 0,
            mean_execution_time_seconds: 0.0,
            breakdown_nanoseconds: None,
            engine_version: r.version,
            engine_lang: r.lang,
            engine_class: "subprocess".into(),
            also_computes: String::new(),
            internally_parallel: false,
            load_ms: r.load_ms,
            mbps: 0.0,
            ns_per_byte: 0.0,
            ids_hash: String::new(),
            verified: None,
            unsupported: Some(why),
            ..Default::default()
        });
    }

    let bytes = r.bytes as f64;
    let breakdown = r.phases.map(|p| Breakdown {
        normalization: *p.get("normalization").unwrap_or(&0),
        pre_tokenization: *p.get("pre_tokenization").unwrap_or(&0),
        core_encoding: *p.get("core_encoding").unwrap_or(&0),
        post_processing: *p.get("post_processing").unwrap_or(&0),
    });

    eprintln!(
        "  {}/{} {name:<16} {:>8.1} MB/s  (scripted)",
        model.name,
        stem(corpus),
        (bytes / (1024.0 * 1024.0)) / r.secs
    );

    Ok(EngineResult {
        tokenizer_name: name.into(),
        total_tokens_produced: r.tokens,
        mean_execution_time_seconds: r.secs,
        breakdown_nanoseconds: breakdown,
        engine_version: r.version,
        engine_lang: r.lang,
        engine_class: "subprocess".into(),
        also_computes: r.also_computes,
        internally_parallel: r.internally_parallel,
        load_ms: r.load_ms,
        mbps: (bytes / (1024.0 * 1024.0)) / r.secs,
        ns_per_byte: r.secs * 1e9 / bytes,
        ids_hash: format!("{:016x}", r.ids_hash),
        verified: None,
        unsupported: None,
        ..Default::default()
    })
}

/// Child process: build ONE engine, load and warm it, report its resident
/// memory, exit. Spawned once per engine by [`measure_memory`].
///
/// This must stay a separate process. Measuring several engines in one process
/// lets the allocator hand engine B the pages engine A just freed, which
/// reports B's footprint as near zero — see `tokbench_core::mem`.
fn memory_child(args: &Args, name: &str) -> Result<()> {
    let dir = args
        .memory_model
        .clone()
        .context("--memory-model required")?;
    let corpus = args
        .memory_corpus
        .clone()
        .context("--memory-corpus required")?;
    let model = Model {
        name: stem(&dir),
        dir,
    };
    let (ctor, _) = registry::native()
        .into_iter()
        .find(|(n, _)| *n == name)
        .map(|(n, c)| (c, n))
        .with_context(|| format!("engine {name} not compiled into this binary"))?;

    let text = tokbench_core::read_corpus(&corpus).context("reading corpus")?;
    let chunks = chunk(&text, CHUNK_BYTES, MAX_CHUNKS);

    let base = tokbench_core::mem::live_heap();
    let Some(mut engine) = ctor(&model).ok() else {
        println!("{{}}");
        return Ok(());
    };
    // Two samples, because "how much RAM does this engine use" is two questions:
    // what the loaded tokenizer holds, and what it holds once its caches are
    // warm. An engine can win one and lose the other.
    let after_load = tokbench_core::mem::live_heap();
    let mut out = Vec::new();
    for c in &chunks {
        out.clear();
        engine.encode(c, &mut out);
    }
    // The corpus is resident before the baseline is taken and `out` holds only
    // one chunk's ids, so neither is charged to the engine.
    let after_encode = tokbench_core::mem::live_heap();
    drop(out);

    let grew = |a: Option<u64>| match (base, a) {
        (Some(b), Some(a)) => Some(a.saturating_sub(b) as f64 / (1024.0 * 1024.0)),
        _ => None,
    };
    println!(
        "{}",
        serde_json::json!({
            "heap_load_mb": grew(after_load),
            "heap_encode_mb": grew(after_encode),
        })
    );
    // Held until after the readings: dropping earlier would free the very
    // allocations being measured.
    drop(engine);
    Ok(())
}

/// The four footprint numbers a child reports. See `memory_child`.
#[derive(Default)]
struct Footprint {
    heap_load_mb: Option<f64>,
    heap_encode_mb: Option<f64>,
}

/// Spawn `memory_child` for one engine and read back its footprint.
fn measure_memory(name: &str, model: &Model, corpus: &Path) -> Option<Footprint> {
    let exe = std::env::current_exe().ok()?;
    let out = Command::new(exe)
        .arg("--memory")
        .arg(name)
        .arg("--memory-model")
        .arg(&model.dir)
        .arg("--memory-corpus")
        .arg(corpus)
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let line = text
        .lines()
        .rev()
        .find(|l| l.trim_start().starts_with('{'))?;
    let v: serde_json::Value = serde_json::from_str(line).ok()?;
    let f = |k: &str| v.get(k).and_then(|x| x.as_f64());
    Some(Footprint {
        heap_load_mb: f("heap_load_mb"),
        heap_encode_mb: f("heap_encode_mb"),
    })
}
