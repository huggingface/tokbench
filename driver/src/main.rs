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
use std::io::{self, IsTerminal, Write};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use tokbench_core::{
    chunk, measure, measure_decode, measure_latency, measure_scaling_with_mode, Ids, Model, Phases,
    ScalingMode,
};

/// ~10 kB documents: large enough that per-call overhead is amortised, small
/// enough to stay in cache. Matches the upstream pipeline benchmark so numbers
/// remain comparable with it.
const CHUNK_BYTES: usize = 10 * 1024;
const MAX_CHUNKS: usize = 100;

#[derive(Subcommand, Clone, Copy, Debug, PartialEq, Eq)]
enum MeasureCommand {
    /// Single-thread encode throughput.
    Encode,
    /// Decode throughput over the reference engine's token IDs.
    Decode,
    /// Call-level encode latency.
    Latency,
    /// Multi-thread encode throughput and efficiency.
    Scaling,
    /// Live heap held by a loaded and warmed tokenizer.
    Memory,
}

#[derive(Subcommand, Debug)]
enum CliCommand {
    /// Run exactly one measurement family and skip every unrelated pass.
    Measure {
        #[command(subcommand)]
        command: MeasureCommand,
    },
}

#[derive(Parser, Debug)]
#[command(
    name = "tokbench",
    about = "Benchmark tokenizer engines through one shared, verified timing loop."
)]
struct Args {
    #[command(subcommand)]
    command: Option<CliCommand>,

    /// Directory of model directories; each subdirectory holds one model's
    /// artifacts (tokenizer.json plus any engine-specific files).
    #[arg(long, default_value = "data/models", global = true)]
    models: PathBuf,

    /// Directory of `.txt` corpora.
    #[arg(long, default_value = "data/fixtures", global = true)]
    corpora: PathBuf,

    /// Timed passes per cell; the median is reported.
    #[arg(long, default_value_t = 5, global = true)]
    reps: usize,

    /// Skip the warm-up pass to report cold-cache numbers instead.
    #[arg(long, global = true)]
    no_warmup: bool,

    /// Override the tokenizers v1 pipeline BPE word-cache capacity.
    /// `0` disables the cache; omission preserves the upstream default (65,536).
    #[arg(long, global = true)]
    cache_capacity: Option<usize>,

    /// Only run these engines (repeatable). Use `all`, or omit, for everything
    /// compiled in that supports the selected measurement.
    #[arg(long, global = true)]
    engine: Vec<String>,

    /// Also measure this engine and report the target's speed relative to it.
    #[arg(long, global = true)]
    compare_to: Option<String>,

    /// Only run these models (repeatable).
    #[arg(long, global = true)]
    model: Vec<String>,

    /// Only run these corpora (repeatable).
    #[arg(long, global = true)]
    corpus: Vec<String>,

    #[arg(long, default_value = "tokenizer_bench_results.json", global = true)]
    out: PathBuf,

    /// Open the dashboard in a browser once the JSON is written.
    #[arg(long, global = true)]
    open: bool,

    /// Interpreter used for scripted (Python) engines.
    #[arg(long, default_value = "python3", global = true)]
    python: String,

    /// Skip the per-engine footprint child processes (they roughly double wall
    /// time,
    /// since each one reloads the model).
    #[arg(long, global = true)]
    no_memory: bool,

    /// Skip the decode pass. Decode is measured over the reference engine's
    /// ids, so this also skips the extra reference encode that produces them.
    #[arg(long, global = true)]
    no_decode: bool,

    /// Measure call-level encode latency on these corpora (repeatable).
    #[arg(long = "latency")]
    latency: Vec<String>,

    /// Maximum bytes in each distinct latency document.
    #[arg(long, default_value_t = 512, global = true)]
    latency_bytes: usize,

    /// Maximum number of distinct call-level latency samples per engine and cell.
    #[arg(long, default_value_t = 1_000, global = true)]
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
    #[arg(long, global = true)]
    max_threads: Option<NonZeroUsize>,

    /// Worker count for `measure memory`. At one worker this measures one
    /// tokenizer. At larger counts the selected scaling mode decides whether
    /// that is one native pool or several independent tokenizer instances.
    #[arg(long, default_value_t = NonZeroUsize::new(1).unwrap(), global = true)]
    threads: NonZeroUsize,

    /// Measure scaling points from the highest thread count down to one.
    /// Jobs alternate this with the default order to expose temporal drift.
    #[arg(long, global = true)]
    reverse_scaling: bool,

    /// How scaling workers are supplied: the engine's native thread pool,
    /// independent single-threaded instances, or automatic capability-based selection.
    #[arg(
        long,
        value_parser = ["auto", "native-threads", "independent-instances"],
        default_value = "auto",
        global = true
    )]
    scaling_mode: String,

    /// Which padding modes the scaling sweep measures: `off`, `longest`, or
    /// `both`.
    ///
    /// Both by default. Padding to the longest document in a batch is a large
    /// and uneven cost -- a fill, often a second pass, sometimes a different
    /// output layout -- and it is a hard requirement for anyone feeding
    /// rectangular tensors, so the unpadded number alone is not usable for
    /// serving. Engines with no native padding report the padded cell as
    /// unsupported; the harness never pads on an engine's behalf.
    #[arg(
        long = "padding",
        value_parser = ["off", "longest", "both"],
        default_value = "both",
        global = true
    )]
    padding: String,

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
    #[arg(long, hide = true)]
    memory_threads: Option<NonZeroUsize>,
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
    /// Explicit tokenizers v1 pipeline cache capacity. Absent means the
    /// upstream default recorded by the pinned engine revision.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pipeline_cache_capacity: Option<usize>,
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
    /// Worker configuration used for the footprint measurement.
    #[serde(skip_serializing_if = "Option::is_none")]
    memory_threads: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    memory_parallelism: Option<String>,
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
    /// Padding modes this engine has no native support for, so no padded
    /// measurement exists rather than an unpadded one wearing a padded label.
    #[serde(skip_serializing_if = "Option::is_none")]
    padding_unsupported: Option<Vec<String>>,

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
    ///
    /// 0.0 means there was no 1-thread point to divide by: an engine whose
    /// parallelism cannot be pinned to one thread has no baseline, and
    /// inventing one from its widest point would report 100% for an engine
    /// whose scaling is in fact unknown.
    efficiency_pct: f64,
    /// `"off"` or `"longest"`. Points from different padding modes are NOT
    /// comparable and must never be mixed in one curve.
    padding: String,
    /// `"native-threads"` (the engine's own pool, via its batch API) or
    /// `"independent-instances"` (one single-threaded engine instance per
    /// harness thread).
    parallelism: String,
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

fn multiplier(numerator: f64, denominator: f64) -> String {
    if numerator.is_finite() && denominator.is_finite() && denominator > 0.0 {
        format!("×{:.2}", numerator / denominator)
    } else {
        "-".into()
    }
}

fn compact_duration(seconds: f64) -> String {
    let seconds = seconds.max(0.0).round() as u64;
    if seconds >= 3600 {
        format!("{}h{:02}m", seconds / 3600, seconds % 3600 / 60)
    } else if seconds >= 60 {
        format!("{}m{:02}s", seconds / 60, seconds % 60)
    } else {
        format!("{seconds}s")
    }
}

fn progress_line(done: usize, total: usize, elapsed_seconds: f64) -> String {
    const WIDTH: usize = 28;
    let filled = (WIDTH * done.min(total)).checked_div(total).unwrap_or(0);
    let bar = format!("{}{}", "#".repeat(filled), "-".repeat(WIDTH - filled));
    let eta = if done == 0 {
        "calculating".into()
    } else {
        compact_duration(elapsed_seconds / done as f64 * total.saturating_sub(done) as f64)
    };
    format!(
        "[{bar}] {done}/{total} cells  elapsed {}  eta {eta}",
        compact_duration(elapsed_seconds)
    )
}

fn print_progress(done: usize, total: usize, elapsed_seconds: f64) {
    if !io::stderr().is_terminal() {
        return;
    }
    eprint!("\r{}", progress_line(done, total, elapsed_seconds));
    let _ = io::stderr().flush();
    if done == total {
        eprintln!();
    }
}

fn median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let middle = values.len() / 2;
    Some(if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    })
}

fn median_multiplier(values: Vec<f64>, expected: usize) -> String {
    let comparable = values.len();
    if comparable == 0 {
        return format!("0/{expected} comparable");
    }
    median(values).map_or_else(
        || format!("0/{expected} comparable"),
        |value| format!("×{value:.2} on {comparable}/{expected} comparable"),
    )
}

fn format_scaling_efficiency(percent: f64) -> String {
    format!("{:.0}% observed", percent.round())
}

/// One engine's result for a cell, paired with the comparator's result for the
/// same cell when `--compare-to` named one.
type Compared<'a> = (&'a EngineResult, Option<&'a EngineResult>);

/// Padding modes that get their own columns, in order.
///
/// Fixed rather than derived from the run so the table has the same shape
/// whatever `--padding` was passed; a mode that was not measured renders `-`.
const PADDING_COLUMNS: [&str; 2] = ["off", "longest"];

/// The 1-thread point of ONE padding mode's curve.
///
/// `padding` is not optional and has no default: a curve is only a curve
/// within a single padding mode, and a helper that scanned every point would
/// silently pair a padded number with an unpadded one.
fn scaling_first<'a>(result: &'a EngineResult, padding: &str) -> Option<&'a ScalePoint> {
    result
        .scaling
        .as_deref()
        .unwrap_or_default()
        .iter()
        .filter(|point| point.padding == padding)
        .find(|point| point.threads == 1)
}

/// The widest point of ONE padding mode's curve.
fn scaling_last<'a>(result: &'a EngineResult, padding: &str) -> Option<&'a ScalePoint> {
    result
        .scaling
        .as_deref()
        .unwrap_or_default()
        .iter()
        .filter(|point| point.padding == padding)
        .max_by_key(|point| point.threads)
}

fn print_collapsed_measurement_table(
    runs: &[Run],
    measurement: MeasureCommand,
    compare_to: Option<&str>,
) {
    let mut groups: BTreeMap<(String, String), Vec<Compared<'_>>> = BTreeMap::new();
    for run in runs {
        let comparator = compare_to.and_then(|name| {
            run.results
                .iter()
                .find(|result| result.tokenizer_name == name)
        });
        for result in &run.results {
            if Some(result.tokenizer_name.as_str()) != compare_to {
                groups
                    .entry((
                        run.dataset_metadata.model.clone(),
                        result.tokenizer_name.clone(),
                    ))
                    .or_default()
                    .push((result, comparator));
            }
        }
    }
    let show_engine = groups
        .keys()
        .map(|(_, engine)| engine)
        .collect::<std::collections::BTreeSet<_>>()
        .len()
        > 1;
    let mut headers = vec!["model".to_string()];
    if show_engine {
        headers.push("engine".into());
    }
    headers.push("corpora".into());
    match measurement {
        MeasureCommand::Encode => {
            headers.extend(["median MB/s", "median ns/B"].map(str::to_string))
        }
        MeasureCommand::Decode => {
            headers.extend(["median MB/s", "median ns/token"].map(str::to_string))
        }
        MeasureCommand::Latency => {
            headers.extend(["median p50 us", "median p99 us", "samples"].map(str::to_string))
        }
        // One group of three per padding mode, in `PADDING_COLUMNS` order.
        MeasureCommand::Scaling => headers.extend(PADDING_COLUMNS.iter().flat_map(|padding| {
            [
                format!("median 1T MB/s [pad {padding}]"),
                format!("median max MB/s [pad {padding}]"),
                format!("median scaling [pad {padding}]"),
            ]
        })),
        MeasureCommand::Memory => {
            headers.extend(["median loaded MB", "median working MB", "workers"].map(str::to_string))
        }
    }
    if let Some(comparator) = compare_to {
        match measurement {
            MeasureCommand::Latency => {
                headers.push(format!("p50 vs {comparator}"));
                headers.push(format!("p99 vs {comparator}"));
            }
            MeasureCommand::Scaling => {
                for padding in PADDING_COLUMNS {
                    headers.push(format!("1T vs {comparator} [pad {padding}]"));
                    headers.push(format!("max vs {comparator} [pad {padding}]"));
                }
            }
            MeasureCommand::Memory => {
                headers.push(format!("loaded heap vs {comparator}"));
                headers.push(format!("working heap vs {comparator}"));
            }
            _ => headers.push(format!("vs {comparator}")),
        }
    }

    let fmt = |value: Option<f64>, decimals: usize| {
        value.map_or_else(|| "-".into(), |value| format!("{value:.decimals$}"))
    };
    let mut rows = Vec::new();
    for ((model, engine), entries) in groups {
        let expected = runs
            .iter()
            .filter(|run| run.dataset_metadata.model == model)
            .count();
        let mut row = vec![model];
        if show_engine {
            row.push(engine);
        }
        match measurement {
            MeasureCommand::Encode => {
                let complete: Vec<_> = entries
                    .iter()
                    .filter(|(result, _)| result.unsupported.is_none())
                    .collect();
                row.push(if compare_to.is_some() {
                    format!("{}/{} measured", complete.len(), expected)
                } else {
                    format!("{}/{}", complete.len(), expected)
                });
                row.push(fmt(
                    median(complete.iter().map(|(r, _)| r.mbps).collect()),
                    1,
                ));
                row.push(fmt(
                    median(complete.iter().map(|(r, _)| r.ns_per_byte).collect()),
                    2,
                ));
                if compare_to.is_some() {
                    let ratios = complete
                        .iter()
                        .filter_map(|(result, comparator)| {
                            let comparator = comparator.as_ref()?;
                            (comparator.unsupported.is_none()
                                && result.ids_hash == comparator.ids_hash
                                && comparator.mbps > 0.0)
                                .then_some(result.mbps / comparator.mbps)
                        })
                        .collect();
                    row.push(median_multiplier(ratios, expected));
                }
            }
            MeasureCommand::Decode => {
                let complete: Vec<_> = entries
                    .iter()
                    .filter(|(result, _)| result.decode_mbps.is_some())
                    .collect();
                row.push(if compare_to.is_some() {
                    format!("{}/{} measured", complete.len(), expected)
                } else {
                    format!("{}/{}", complete.len(), expected)
                });
                row.push(fmt(
                    median(complete.iter().filter_map(|(r, _)| r.decode_mbps).collect()),
                    1,
                ));
                row.push(fmt(
                    median(
                        complete
                            .iter()
                            .filter_map(|(r, _)| r.decode_ns_per_token)
                            .collect(),
                    ),
                    1,
                ));
                if compare_to.is_some() {
                    let ratios = complete
                        .iter()
                        .filter_map(|(result, comparator)| {
                            let comparator = comparator.as_ref()?;
                            let target = result.decode_mbps?;
                            let baseline = comparator.decode_mbps?;
                            (result.decode_text_hash == comparator.decode_text_hash
                                && baseline > 0.0)
                                .then_some(target / baseline)
                        })
                        .collect();
                    row.push(median_multiplier(ratios, expected));
                }
            }
            MeasureCommand::Latency => {
                let complete: Vec<_> = entries
                    .iter()
                    .filter(|(result, _)| result.latency_p50_us.is_some())
                    .collect();
                row.push(if compare_to.is_some() {
                    format!("{}/{} measured", complete.len(), expected)
                } else {
                    format!("{}/{}", complete.len(), expected)
                });
                row.push(fmt(
                    median(
                        complete
                            .iter()
                            .filter_map(|(r, _)| r.latency_p50_us)
                            .collect(),
                    ),
                    2,
                ));
                row.push(fmt(
                    median(
                        complete
                            .iter()
                            .filter_map(|(r, _)| r.latency_p99_us)
                            .collect(),
                    ),
                    2,
                ));
                let sample_counts: Vec<_> = complete
                    .iter()
                    .filter_map(|(result, _)| result.latency_samples)
                    .collect();
                row.push(
                    match (sample_counts.iter().min(), sample_counts.iter().max()) {
                        (Some(minimum), Some(maximum)) if minimum != maximum => {
                            format!("{minimum}–{maximum}")
                        }
                        (Some(samples), _) => samples.to_string(),
                        _ => "-".into(),
                    },
                );
                if compare_to.is_some() {
                    for percentile in [50, 99] {
                        let ratios = complete
                            .iter()
                            .filter_map(|(result, comparator)| {
                                let comparator = comparator.as_ref()?;
                                let (target, baseline) = if percentile == 50 {
                                    (result.latency_p50_us?, comparator.latency_p50_us?)
                                } else {
                                    (result.latency_p99_us?, comparator.latency_p99_us?)
                                };
                                (target > 0.0).then_some(baseline / target)
                            })
                            .collect();
                        row.push(median_multiplier(ratios, expected));
                    }
                }
            }
            MeasureCommand::Scaling => {
                // "measured" counts cells that produced any curve at all, in
                // any padding mode; the per-mode columns below say which.
                let any: Vec<_> = entries
                    .iter()
                    .filter(|(result, _)| {
                        result
                            .scaling
                            .as_deref()
                            .is_some_and(|points| !points.is_empty())
                    })
                    .collect();
                row.push(if compare_to.is_some() {
                    format!("{}/{} measured", any.len(), expected)
                } else {
                    format!("{}/{}", any.len(), expected)
                });

                for padding in PADDING_COLUMNS {
                    // Each column filters on its OWN point, not on the pair.
                    // An engine whose parallelism cannot be pinned to one
                    // thread has no 1T point, and requiring one would drop its
                    // wide-thread number from the table entirely.
                    row.push(fmt(
                        median(
                            entries
                                .iter()
                                .filter_map(|(r, _)| scaling_first(r, padding).map(|p| p.mbps))
                                .collect(),
                        ),
                        1,
                    ));
                    row.push(fmt(
                        median(
                            entries
                                .iter()
                                .filter_map(|(r, _)| scaling_last(r, padding).map(|p| p.mbps))
                                .collect(),
                        ),
                        1,
                    ));
                    // 0.0 is the "no 1-thread baseline" marker, not a real 0%.
                    let efficiencies: Vec<_> = entries
                        .iter()
                        .filter_map(|(result, _)| {
                            scaling_last(result, padding).map(|point| point.efficiency_pct)
                        })
                        .filter(|efficiency| *efficiency > 0.0)
                        .collect();
                    row.push(
                        match (
                            median(efficiencies.clone()),
                            efficiencies.iter().copied().min_by(f64::total_cmp),
                            efficiencies.iter().copied().max_by(f64::total_cmp),
                        ) {
                            (Some(median), Some(minimum), Some(maximum)) => format!(
                                "{} ({}\u{2013}{}% corpus range)",
                                format_scaling_efficiency(median),
                                minimum.round(),
                                maximum.round()
                            ),
                            _ => "-".into(),
                        },
                    );
                }

                if compare_to.is_some() {
                    for padding in PADDING_COLUMNS {
                        for at_maximum in [false, true] {
                            let ratios = entries
                                .iter()
                                .filter_map(|(result, comparator)| {
                                    let comparator = comparator.as_ref()?;
                                    let target = if at_maximum {
                                        scaling_last(result, padding)?
                                    } else {
                                        scaling_first(result, padding)?
                                    };
                                    let baseline = if at_maximum {
                                        scaling_last(comparator, padding)?
                                    } else {
                                        scaling_first(comparator, padding)?
                                    };
                                    (baseline.mbps > 0.0).then_some(target.mbps / baseline.mbps)
                                })
                                .collect();
                            row.push(median_multiplier(ratios, expected));
                        }
                    }
                }
            }
            MeasureCommand::Memory => {
                let complete: Vec<_> = entries
                    .iter()
                    .filter(|(result, _)| result.heap_encode_mb.is_some())
                    .collect();
                row.push(format!("{}/{} measured", complete.len(), expected));
                row.push(fmt(
                    median(
                        complete
                            .iter()
                            .filter_map(|(r, _)| r.heap_load_mb)
                            .collect(),
                    ),
                    1,
                ));
                row.push(fmt(
                    median(
                        complete
                            .iter()
                            .filter_map(|(r, _)| r.heap_encode_mb)
                            .collect(),
                    ),
                    1,
                ));
                let workers = complete
                    .first()
                    .and_then(|(r, _)| r.memory_threads)
                    .map_or_else(|| "-".into(), |n| n.to_string());
                row.push(workers);
                if compare_to.is_some() {
                    for loaded in [true, false] {
                        let ratios = complete
                            .iter()
                            .filter_map(|(result, comparator)| {
                                let comparator = comparator.as_ref()?;
                                let (target, baseline) = if loaded {
                                    (result.heap_load_mb?, comparator.heap_load_mb?)
                                } else {
                                    (result.heap_encode_mb?, comparator.heap_encode_mb?)
                                };
                                (baseline > 0.0).then_some(target / baseline)
                            })
                            .collect();
                        row.push(median_multiplier(ratios, expected));
                    }
                }
            }
        }
        rows.push(row);
    }

    if show_engine {
        let (headers, rows) = pivot_engine_columns(headers, rows);
        let columns = headers.len();
        print_table(headers, rows, columns, false);
    } else {
        print_table(headers, rows, 1, true);
    }
}

fn print_table(
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    label_columns: usize,
    align_last: bool,
) {
    let mut widths: Vec<usize> = headers.iter().map(|header| header.len()).collect();
    for row in &rows {
        for (width, value) in widths.iter_mut().zip(row) {
            *width = (*width).max(value.len());
        }
    }
    let render = |row: &[String]| {
        row.iter()
            .zip(&widths)
            .enumerate()
            .map(|(index, (value, width))| {
                if index >= label_columns && (index + 1 < row.len() || align_last) {
                    format!("{value:>width$}")
                } else if index + 1 == row.len() {
                    value.clone()
                } else {
                    format!("{value:<width$}")
                }
            })
            .collect::<Vec<_>>()
            .join("  ")
    };
    eprintln!();
    eprintln!("{}", render(&headers));
    eprintln!(
        "{}",
        widths
            .iter()
            .map(|width| "-".repeat(*width))
            .collect::<Vec<_>>()
            .join("  ")
    );
    for row in rows {
        eprintln!("{}", render(&row));
    }
}

fn pivot_engine_columns(
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
) -> (Vec<String>, Vec<Vec<String>>) {
    let engines: Vec<String> = rows
        .iter()
        .map(|row| row[1].clone())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let value_headers = &headers[2..];
    let mut pivoted_headers = vec![headers[0].clone()];
    pivoted_headers.extend(engines.iter().cloned());

    let mut by_model: BTreeMap<String, BTreeMap<String, Vec<String>>> = BTreeMap::new();
    for row in rows {
        by_model
            .entry(row[0].clone())
            .or_default()
            .insert(row[1].clone(), row[2..].to_vec());
    }
    let pivoted_rows = by_model
        .into_iter()
        .map(|(model, values)| {
            let mut row = vec![model];
            for engine in &engines {
                let Some(values) = values.get(engine) else {
                    row.push("-".into());
                    continue;
                };
                let parts = value_headers
                    .iter()
                    .zip(values)
                    .map(|(header, value)| {
                        if header == "corpora" || header == "status" || header.starts_with("vs ") {
                            value.clone()
                        } else if header.contains("1T MB/s") {
                            format!("1T {value} MB/s")
                        } else if header.contains("max MB/s") {
                            format!("max {value} MB/s")
                        } else if header.contains("MB/s") {
                            format!("{value} MB/s")
                        } else if header.contains("ns/token") {
                            format!("{value} ns/token")
                        } else if header.contains("ns/B") {
                            format!("{value} ns/B")
                        } else if header.contains("p50") && !header.contains(" vs ") {
                            format!("p50 {value} us")
                        } else if header.contains("p99") && !header.contains(" vs ") {
                            format!("p99 {value} us")
                        } else if header.contains("samples") {
                            format!("{value} samples")
                        } else if header.contains("efficiency") {
                            if value.contains('%') {
                                format!("{value} efficiency")
                            } else {
                                format!("{value}% efficiency")
                            }
                        } else if header == "tokens" {
                            format!("{value} tokens")
                        } else {
                            value.clone()
                        }
                    })
                    .collect::<Vec<_>>();
                row.push(parts.join(" · "));
            }
            row
        })
        .collect();
    (pivoted_headers, pivoted_rows)
}

fn print_measurement_table(runs: &[Run], measurement: MeasureCommand, compare_to: Option<&str>) {
    let show_corpus = runs
        .iter()
        .map(|run| run.dataset_metadata.corpus.as_str())
        .collect::<std::collections::BTreeSet<_>>()
        .len()
        > 1;
    if show_corpus {
        print_collapsed_measurement_table(runs, measurement, compare_to);
        return;
    }
    let show_engine = runs
        .iter()
        .flat_map(|run| {
            run.results
                .iter()
                .map(|result| result.tokenizer_name.as_str())
        })
        .filter(|engine| Some(*engine) != compare_to)
        .collect::<std::collections::BTreeSet<_>>()
        .len()
        > 1;

    let mut headers = vec!["model".to_string()];
    if show_corpus {
        headers.push("corpus".to_string());
    }
    if show_engine {
        headers.push("engine".to_string());
    }
    match measurement {
        MeasureCommand::Encode => headers.extend(["MB/s", "ns/B", "tokens"].map(str::to_string)),
        MeasureCommand::Decode => headers.extend(["MB/s", "ns/token"].map(str::to_string)),
        MeasureCommand::Latency => {
            headers.extend(["p50 us", "p99 us", "samples"].map(str::to_string))
        }
        MeasureCommand::Scaling => headers.extend(PADDING_COLUMNS.iter().flat_map(|padding| {
            [
                format!("1T MB/s [pad {padding}]"),
                format!("max MB/s [pad {padding}]"),
                format!("scaling [pad {padding}]"),
            ]
        })),
        MeasureCommand::Memory => headers
            .extend(["loaded MB", "working MB", "workers", "parallelism"].map(str::to_string)),
    }
    if let Some(comparator) = compare_to {
        match measurement {
            MeasureCommand::Latency => {
                headers.push(format!("p50 vs {comparator}"));
                headers.push(format!("p99 vs {comparator}"));
            }
            MeasureCommand::Scaling => {
                for padding in PADDING_COLUMNS {
                    headers.push(format!("1T vs {comparator} [pad {padding}]"));
                    headers.push(format!("max vs {comparator} [pad {padding}]"));
                }
            }
            MeasureCommand::Memory => {
                headers.push(format!("loaded heap vs {comparator}"));
                headers.push(format!("working heap vs {comparator}"));
            }
            _ => headers.push(format!("vs {comparator}")),
        }
    } else {
        headers.push("status".into());
    }

    let mut rows = Vec::new();
    for run in runs {
        for result in &run.results {
            if Some(result.tokenizer_name.as_str()) == compare_to {
                continue;
            }
            let comparator = compare_to.and_then(|name| {
                run.results
                    .iter()
                    .find(|candidate| candidate.tokenizer_name == name)
            });
            let mut row = vec![run.dataset_metadata.model.clone()];
            if show_corpus {
                row.push(run.dataset_metadata.corpus.clone());
            }
            if show_engine {
                row.push(result.tokenizer_name.clone());
            }
            let unsupported = result.unsupported.as_deref();
            match measurement {
                MeasureCommand::Encode => {
                    row.push(
                        unsupported.map_or_else(|| format!("{:.1}", result.mbps), |_| "-".into()),
                    );
                    row.push(
                        unsupported
                            .map_or_else(|| format!("{:.2}", result.ns_per_byte), |_| "-".into()),
                    );
                    row.push(
                        unsupported.map_or_else(
                            || result.total_tokens_produced.to_string(),
                            |_| "-".into(),
                        ),
                    );
                    row.push(if compare_to.is_some() {
                        match (unsupported, comparator) {
                            (Some(why), _) => format!("unsupported: {why}"),
                            (None, Some(other)) if other.unsupported.is_some() => {
                                "comparator unsupported".into()
                            }
                            (None, Some(other)) if result.ids_hash != other.ids_hash => {
                                "id mismatch".into()
                            }
                            (None, Some(other)) => multiplier(result.mbps, other.mbps),
                            (None, None) => "comparator missing".into(),
                        }
                    } else {
                        match (unsupported, result.verified) {
                            (Some(why), _) => format!("unsupported: {why}"),
                            (None, Some(true)) => "verified".into(),
                            (None, Some(false)) => "id mismatch".into(),
                            (None, None) => "unverified".into(),
                        }
                    });
                }
                MeasureCommand::Decode => {
                    let why = unsupported.or(result.decode_unsupported.as_deref());
                    row.push(
                        result
                            .decode_mbps
                            .map_or_else(|| "-".into(), |v| format!("{v:.1}")),
                    );
                    row.push(
                        result
                            .decode_ns_per_token
                            .map_or_else(|| "-".into(), |v| format!("{v:.1}")),
                    );
                    row.push(if compare_to.is_some() {
                        match (why, comparator) {
                            (Some(why), _) => format!("unsupported: {why}"),
                            (None, Some(other)) if other.decode_mbps.is_none() => {
                                "comparator unsupported".into()
                            }
                            (None, Some(other))
                                if result.decode_text_hash != other.decode_text_hash =>
                            {
                                "text mismatch".into()
                            }
                            (None, Some(other)) => {
                                multiplier(result.decode_mbps.unwrap(), other.decode_mbps.unwrap())
                            }
                            (None, None) => "comparator missing".into(),
                        }
                    } else {
                        match (why, result.decode_verified) {
                            (Some(why), _) => format!("unsupported: {why}"),
                            (None, Some(true)) => "verified".into(),
                            (None, Some(false)) => "text mismatch".into(),
                            (None, None) => "unverified".into(),
                        }
                    });
                }
                MeasureCommand::Latency => {
                    row.push(
                        result
                            .latency_p50_us
                            .map_or_else(|| "-".into(), |v| format!("{v:.2}")),
                    );
                    row.push(
                        result
                            .latency_p99_us
                            .map_or_else(|| "-".into(), |v| format!("{v:.2}")),
                    );
                    row.push(
                        result
                            .latency_samples
                            .map_or_else(|| "-".into(), |v| v.to_string()),
                    );
                    if compare_to.is_some() {
                        match (unsupported, comparator) {
                            (Some(why), _) => {
                                row.push(format!("unsupported: {why}"));
                                row.push("-".into());
                            }
                            (None, Some(other)) => {
                                row.push(match (result.latency_p50_us, other.latency_p50_us) {
                                    (Some(target), Some(baseline)) => multiplier(baseline, target),
                                    _ => "comparator unsupported".into(),
                                });
                                row.push(match (result.latency_p99_us, other.latency_p99_us) {
                                    (Some(target), Some(baseline)) => multiplier(baseline, target),
                                    _ => "comparator unsupported".into(),
                                });
                            }
                            (None, None) => {
                                row.push("comparator missing".into());
                                row.push("-".into());
                            }
                        }
                    } else {
                        row.push(
                            unsupported
                                .map_or_else(|| "ok".into(), |why| format!("unsupported: {why}")),
                        );
                    }
                }
                MeasureCommand::Scaling => {
                    // No native support for a mode is `no native padding`, not
                    // `-`: the distinction between "could not do it" and "was
                    // not measured" is the whole reason this axis exists.
                    let cannot_pad = |padding: &str| {
                        result
                            .padding_unsupported
                            .as_deref()
                            .is_some_and(|modes| modes.iter().any(|mode| mode == padding))
                    };
                    for padding in PADDING_COLUMNS {
                        let first = scaling_first(result, padding);
                        let last = scaling_last(result, padding);
                        // The throughput columns are rendered into "1T {} MB/s",
                        // so a prose marker there reads as "1T no native padding
                        // MB/s". They get "-"; the reason goes in the scaling
                        // column, which is free text.
                        let why_missing = || {
                            if cannot_pad(padding) {
                                "no native padding".to_string()
                            } else {
                                "-".to_string()
                            }
                        };
                        row.push(
                            first.map_or_else(
                                || "-".to_string(),
                                |point| format!("{:.1}", point.mbps),
                            ),
                        );
                        row.push(
                            last.map_or_else(
                                || "-".to_string(),
                                |point| format!("{:.1}", point.mbps),
                            ),
                        );
                        row.push(last.map_or_else(why_missing, |point| {
                            // efficiency 0.0 = no 1-thread baseline to divide
                            // by, which is unknown scaling, not 0% scaling.
                            if point.efficiency_pct > 0.0 {
                                format!(
                                    "{} @ {}T ({})",
                                    format_scaling_efficiency(point.efficiency_pct),
                                    point.threads,
                                    point.parallelism
                                )
                            } else {
                                format!(
                                    "n/a @ {}T ({}, no 1T baseline)",
                                    point.threads, point.parallelism
                                )
                            }
                        }));
                    }
                    if compare_to.is_some() {
                        for padding in PADDING_COLUMNS {
                            let first = scaling_first(result, padding);
                            let last = scaling_last(result, padding);
                            let other_first =
                                comparator.and_then(|other| scaling_first(other, padding));
                            let other_last =
                                comparator.and_then(|other| scaling_last(other, padding));
                            if let Some(why) = unsupported {
                                row.push(format!("unsupported: {why}"));
                                row.push("-".into());
                            } else if comparator
                                .is_some_and(|other| result.ids_hash != other.ids_hash)
                            {
                                row.push("id mismatch".into());
                                row.push("id mismatch".into());
                            } else {
                                row.push(match (first, other_first) {
                                    (Some(target), Some(baseline)) => {
                                        multiplier(target.mbps, baseline.mbps)
                                    }
                                    _ => "comparator unsupported".into(),
                                });
                                row.push(match (last, other_last) {
                                    (Some(target), Some(baseline)) => {
                                        multiplier(target.mbps, baseline.mbps)
                                    }
                                    _ => "comparator unsupported".into(),
                                });
                            }
                        }
                    } else {
                        row.push(match (unsupported, result.verified) {
                            (Some(why), _) => format!("unsupported: {why}"),
                            (None, Some(true)) => "verified".into(),
                            (None, Some(false)) => "id mismatch".into(),
                            (None, None) => "unverified".into(),
                        });
                    }
                }
                MeasureCommand::Memory => {
                    row.push(
                        result
                            .heap_load_mb
                            .map_or_else(|| "-".into(), |v| format!("{v:.1}")),
                    );
                    row.push(
                        result
                            .heap_encode_mb
                            .map_or_else(|| "-".into(), |v| format!("{v:.1}")),
                    );
                    row.push(
                        result
                            .memory_threads
                            .map_or_else(|| "-".into(), |v| v.to_string()),
                    );
                    row.push(
                        result
                            .memory_parallelism
                            .clone()
                            .unwrap_or_else(|| "-".into()),
                    );
                    if compare_to.is_some() {
                        for loaded in [true, false] {
                            let pair = comparator.map(|other| {
                                if loaded {
                                    (result.heap_load_mb, other.heap_load_mb)
                                } else {
                                    (result.heap_encode_mb, other.heap_encode_mb)
                                }
                            });
                            row.push(match (unsupported, pair) {
                                (Some(why), _) => format!("unsupported: {why}"),
                                (None, Some((Some(target), Some(baseline)))) if baseline > 0.0 => {
                                    multiplier(target, baseline)
                                }
                                (None, _) => "comparator unsupported".into(),
                            });
                        }
                    } else {
                        row.push(
                            unsupported
                                .map_or_else(|| "ok".into(), |why| format!("unsupported: {why}")),
                        );
                    }
                }
            }
            rows.push(row);
        }
    }

    let label_columns = 1 + usize::from(show_corpus) + usize::from(show_engine);
    if show_engine {
        let (headers, rows) = pivot_engine_columns(headers, rows);
        let columns = headers.len();
        print_table(headers, rows, columns, false);
    } else {
        print_table(headers, rows, label_columns, compare_to.is_some());
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let measurement = args
        .command
        .as_ref()
        .map(|CliCommand::Measure { command }| *command);
    let run_encode = measurement.is_none() || measurement == Some(MeasureCommand::Encode);
    let run_decode = match measurement {
        Some(command) => command == MeasureCommand::Decode,
        None => !args.no_decode,
    };
    let run_latency = measurement == Some(MeasureCommand::Latency);
    let run_scaling = measurement == Some(MeasureCommand::Scaling);
    let run_memory_atomic = measurement == Some(MeasureCommand::Memory);
    let run_memory = run_memory_atomic || (measurement.is_none() && !args.no_memory);
    if !run_memory_atomic && args.threads.get() != 1 {
        bail!("--threads is available with `tokbench measure memory` only");
    }

    // Child mode short-circuits everything: this process exists to load one
    // engine and report its memory, so it must not touch any other.
    if let Some(name) = args.memory.clone() {
        return memory_child(&args, &name);
    }

    // Stripped per-engine binary sizes, if `scripts/binsize.sh` has run.
    // Absent is normal (it needs a release build per engine); the column is
    // simply omitted rather than reported as zero.
    let (bin_sizes, pkg_sizes) = if measurement.is_none() {
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
        (bin_sizes, pkg_sizes)
    } else {
        (BTreeMap::new(), BTreeMap::new())
    };

    let models: Vec<_> = list_dir(&args.models, true, None)?
        .into_iter()
        .filter(|path| args.model.is_empty() || args.model.contains(&directory_name(path)))
        .collect();
    if models.is_empty() {
        bail!(
            "no selected model directories under {} — run `make models` or check --model",
            args.models.display()
        );
    }
    let corpora: Vec<_> = list_dir(&args.corpora, false, Some("txt"))?
        .into_iter()
        .filter(|path| args.corpus.is_empty() || args.corpus.contains(&stem(path)))
        .collect();
    if corpora.is_empty() {
        bail!(
            "no selected .txt corpora under {} — run `make fixtures` or check --corpus",
            args.corpora.display()
        );
    }
    if run_memory_atomic && (args.corpus.len() != 1 || corpora.len() != 1) {
        bail!("`tokbench measure memory` requires exactly one explicit --corpus");
    }

    let natives = registry::native();
    let scripted = registry::scripted();
    let known_engines: Vec<_> = natives
        .iter()
        .map(|(name, _)| *name)
        .chain(scripted.iter().map(|(name, _)| *name))
        .collect();
    let all_engines = args.engine.iter().any(|requested| requested == "all");
    if all_engines && args.engine.len() != 1 {
        bail!("--engine all cannot be combined with another --engine value");
    }
    if let Some(unknown) = args.engine.iter().find(|requested| {
        requested.as_str() != "all" && !known_engines.contains(&requested.as_str())
    }) {
        bail!(
            "engine {unknown} is not compiled in; available engines: {}",
            known_engines.join(", ")
        );
    }
    if let Some(comparator) = args.compare_to.as_deref() {
        if measurement.is_none() {
            bail!("--compare-to is available with `tokbench measure` only");
        }
        if !known_engines.contains(&comparator) {
            bail!(
                "comparison engine {comparator} is not compiled in; available engines: {}",
                known_engines.join(", ")
            );
        }
        if args.engine.iter().any(|engine| engine == comparator) {
            bail!("--engine and --compare-to must name different engines");
        }
    }
    if measurement.is_some()
        && !run_encode
        && args
            .engine
            .iter()
            .chain(args.compare_to.iter())
            .any(|requested| scripted.iter().any(|(name, _)| requested == name))
    {
        bail!("decode, latency, scaling, and memory measurements currently support native engines only");
    }
    if run_decode && !natives.iter().any(|(name, _)| *name == registry::REFERENCE) {
        bail!(
            "decode measurement requires the {} engine to provide shared token IDs",
            registry::REFERENCE
        );
    }
    let mut thread_sweep = tokbench_core::thread_counts();
    if let Some(max_threads) = args.max_threads {
        thread_sweep.retain(|threads| *threads <= max_threads.get());
    }
    if args.reverse_scaling {
        thread_sweep.reverse();
    }
    let padding_modes: Vec<tokbench_core::Padding> = match args.padding.as_str() {
        "off" => vec![tokbench_core::Padding::Off],
        "longest" => vec![tokbench_core::Padding::Longest],
        _ => vec![tokbench_core::Padding::Off, tokbench_core::Padding::Longest],
    };
    let scaling_mode = match args.scaling_mode.as_str() {
        "native-threads" => ScalingMode::NativeThreads,
        "independent-instances" => ScalingMode::IndependentInstances,
        _ => ScalingMode::Auto,
    };
    if !args.scaling.is_empty() {
        eprintln!(
            "scaling sweep on {:?} at thread counts {:?}, padding {:?}",
            args.scaling,
            thread_sweep,
            padding_modes
                .iter()
                .map(|padding| padding.as_str())
                .collect::<Vec<_>>()
        );
    }
    let want = |n: &str| {
        args.engine.is_empty()
            || all_engines
            || args.engine.iter().any(|engine| engine == n)
            || args.compare_to.as_deref() == Some(n)
    };
    if args.cache_capacity.is_some() {
        if !known_engines.contains(&"pipeline") {
            bail!("--cache-capacity requires the pipeline engine feature");
        }
        if !want("pipeline") {
            bail!("--cache-capacity requires selecting the pipeline engine");
        }
    }
    let native_count = natives.iter().filter(|(name, _)| want(name)).count();
    let scripted_count = if run_encode {
        scripted.iter().filter(|(name, _)| want(name)).count()
    } else {
        0
    };
    if native_count + scripted_count == 0 {
        bail!("no selected engines support this measurement");
    }

    let cells = models.len() * corpora.len();
    let measurement_detail = match measurement {
        Some(MeasureCommand::Latency) => {
            format!("up to {} distinct samples each", args.latency_samples)
        }
        Some(MeasureCommand::Scaling) => {
            format!("thread counts {thread_sweep:?}, {} reps", args.reps)
        }
        Some(MeasureCommand::Memory) => format!(
            "{} worker(s), {} parallelism, median of {} isolated runs",
            args.threads, args.scaling_mode, args.reps
        ),
        _ => format!("{} reps each", args.reps),
    };
    if let Some(measurement) = measurement {
        let name = match measurement {
            MeasureCommand::Encode => "encode",
            MeasureCommand::Decode => "decode",
            MeasureCommand::Latency => "latency",
            MeasureCommand::Scaling => "scaling",
            MeasureCommand::Memory => "memory",
        };
        eprintln!(
            "tokbench measure {name}: {} engine(s) × {} model(s) × {} corpus/corpora, {}",
            native_count + scripted_count,
            models.len(),
            corpora.len(),
            measurement_detail
        );
        print_progress(0, cells, 0.0);
    } else {
        eprintln!(
            "tokbench: {} engine(s) [{} native, {} scripted] × {} model(s) × {} corpus/corpora = {} cells, {}",
            native_count + scripted_count,
            native_count,
            scripted_count,
            models.len(),
            corpora.len(),
            cells,
            measurement_detail
        );
    }

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
            let measure_scaling_for_corpus =
                run_scaling || (measurement.is_none() && args.scaling.contains(&corpus_name));
            // Scaling needs enough unique input to keep all workers busy
            // without replay. Keep this separate from `chunks` so adding a
            // scaling sweep to the legacy full run cannot silently change its
            // ordinary encode workload.
            let scaling_chunks =
                measure_scaling_for_corpus.then(|| chunk(&text, CHUNK_BYTES, usize::MAX));
            let measure_latency_here =
                run_latency || (measurement.is_none() && args.latency.contains(&corpus_name));
            let latency_docs = if measure_latency_here {
                latency_documents(&text, args.latency_bytes, args.latency_samples)
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
            let ref_ids: Vec<Ids> = if !run_decode {
                Vec::new()
            } else {
                natives
                    .iter()
                    .find(|(n, _)| *n == registry::REFERENCE)
                    .and_then(|(_, ctor)| ctor(&model, None).ok())
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
                // The dedicated footprint child supplies both metadata and
                // measurements. Avoid loading every model once here only to
                // load it again in the isolated process below.
                if run_memory_atomic {
                    results.push(EngineResult {
                        tokenizer_name: name.to_string(),
                        ..Default::default()
                    });
                    continue;
                }
                let t0 = Instant::now();
                let engine_cache_capacity = (*name == "pipeline")
                    .then_some(args.cache_capacity)
                    .flatten();
                // Third-party code, adversarial inputs. A panic here is a
                // finding about that engine, not a reason to lose the run.
                let built = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    ctor(&model, engine_cache_capacity)
                })) {
                    Ok(r) => r,
                    Err(_) => Err(tokbench_core::Unsupported("panicked while loading".into())),
                };
                let load_ms = t0.elapsed().as_secs_f64() * 1e3;

                match built {
                    Err(why) => {
                        if measurement.is_none() {
                            eprintln!("  {model_name}/{corpus_name} {name}: unsupported ({why})");
                        }
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
                        let measured = if run_encode {
                            let measured =
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    measure(engine.as_mut(), &chunks, args.reps, !args.no_warmup)
                                }));
                            let Ok(value) = measured else {
                                if measurement.is_none() {
                                    eprintln!(
                                        "  {model_name}/{corpus_name} {name}: PANICKED while encoding"
                                    );
                                }
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
                            Some(value)
                        } else {
                            None
                        };

                        let latency = if latency_docs.is_empty() {
                            None
                        } else {
                            measure_latency(engine.as_mut(), &latency_docs, args.latency_bytes)
                        };

                        // Stage breakdown on a separate, untimed pass so the
                        // extra clock reads never inflate the headline.
                        let mut phases = Phases::default();
                        let mut any = false;
                        if measurement.is_none() {
                            for c in &chunks {
                                if let Some(p) = engine.phases(c) {
                                    phases.accumulate(p);
                                    any = true;
                                }
                            }
                        }

                        // Decode, on the same clock, over the reference's ids.
                        // Runs while this engine is still alive and warm, so
                        // it costs no extra build.
                        let decoded = (run_decode && !ref_ids.is_empty())
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

                        // Multi-thread sweep, only on the corpora asked for.
                        // Runs after the single-thread timing so it can never
                        // perturb the headline number.
                        let (scaling, padding_unsupported) = if let Some(scaling_chunks) =
                            &scaling_chunks
                        {
                            // One sweep per padding mode. They are separate
                            // curves, never merged: an engine can scale well
                            // ragged and badly padded, and that difference is
                            // the point of measuring both.
                            let mut points: Vec<ScalePoint> = Vec::new();
                            let mut missing: Vec<String> = Vec::new();
                            for &padding in &padding_modes {
                                let make = || ctor(&model, engine_cache_capacity).ok();
                                let curve = measure_scaling_with_mode(
                                    &make,
                                    scaling_chunks,
                                    &thread_sweep,
                                    args.reps,
                                    padding,
                                    scaling_mode,
                                );
                                let Some(curve) = curve else {
                                    // No native support for this padding mode.
                                    missing.push(padding.as_str().to_string());
                                    continue;
                                };
                                if measurement.is_none() {
                                    if let Some(best) =
                                        curve.points.iter().max_by_key(|point| point.threads)
                                    {
                                        eprintln!(
                                            "        pad {} · threads {} -> {:.1} MB/s ({:.0}% of linear, {} parallelism)",
                                            curve.padding.as_str(),
                                            best.threads,
                                            best.mbps,
                                            best.efficiency_pct,
                                            curve.kind.as_str()
                                        );
                                    }
                                }
                                points.extend(curve.points.iter().map(|point| ScalePoint {
                                    threads: point.threads,
                                    mbps: point.mbps,
                                    efficiency_pct: point.efficiency_pct,
                                    padding: curve.padding.as_str().to_string(),
                                    parallelism: curve.kind.as_str().to_string(),
                                }));
                            }
                            (
                                (!points.is_empty()).then_some(points),
                                (!missing.is_empty()).then_some(missing),
                            )
                        } else {
                            (None, None)
                        };

                        // A scaling-only run still needs the same correctness
                        // oracle as encode throughput. This pass is untimed as
                        // far as the reported scaling points are concerned and
                        // runs after them, so verification cannot warm or
                        // otherwise perturb the numbers above.
                        let scaling_identity = if run_scaling {
                            let identity =
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    measure(engine.as_mut(), &chunks, 1, false)
                                }));
                            let Ok(value) = identity else {
                                results.push(EngineResult {
                                    tokenizer_name: name.to_string(),
                                    engine_version: info.version.into(),
                                    engine_lang: info.lang.into(),
                                    engine_class: info.class.as_str().into(),
                                    load_ms,
                                    unsupported: Some(
                                        "panicked while verifying scaling output".to_string(),
                                    ),
                                    ..Default::default()
                                });
                                continue;
                            };
                            Some(value)
                        } else {
                            None
                        };
                        let identity = measured.as_ref().or(scaling_identity.as_ref());

                        if measurement.is_none() {
                            if let Some(m) = &measured {
                                let decode_note = match &decoded {
                                    Some(Ok(d)) => format!("  dec {:>7.1} MB/s", d.mbps),
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
                            } else if let Some(decoded) = &decoded {
                                match decoded {
                                    Ok(value) => eprintln!(
                                        "  [{}/{}] {model_name}/{corpus_name} {name:<16} decode {:>8.1} MB/s",
                                        done + 1,
                                        cells,
                                        value.mbps
                                    ),
                                    Err(why) => eprintln!(
                                        "  [{}/{}] {model_name}/{corpus_name} {name:<16} decode unsupported ({why})",
                                        done + 1,
                                        cells
                                    ),
                                }
                            } else if let Some(value) = &latency {
                                eprintln!(
                                    "  [{}/{}] {model_name}/{corpus_name} {name:<16} latency p50 {:.2} us  p99 {:.2} us",
                                    done + 1,
                                    cells,
                                    value.p50_us,
                                    value.p99_us
                                );
                            }
                        }

                        results.push(EngineResult {
                            tokenizer_name: name.to_string(),
                            total_tokens_produced: identity.map_or(0, |value| value.tokens),
                            mean_execution_time_seconds: measured
                                .as_ref()
                                .map_or(0.0, |value| value.secs),
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
                            mbps: measured.as_ref().map_or(0.0, |value| value.mbps),
                            ns_per_byte: measured.as_ref().map_or(0.0, |value| value.ns_per_byte),
                            ids_hash: identity
                                .map(|value| format!("{:016x}", value.ids_hash))
                                .unwrap_or_default(),
                            verified: None,
                            unsupported: None,
                            decode_mbps,
                            decode_ns_per_token,
                            decode_text_hash,
                            decode_verified: None,
                            decode_unsupported,
                            scaling,
                            padding_unsupported,
                            reused_text: measured.as_ref().is_some_and(|value| value.reused),
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
            if run_encode {
                for (name, script) in &scripted {
                    if !want(name) {
                        continue;
                    }
                    match run_scripted(&args, name, script, &model, corpus_path) {
                        Ok(r) => results.push(r),
                        Err(e) => eprintln!("  {model_name}/{corpus_name} {name}: {e}"),
                    }
                }
            }

            // Verification: the reference's id hash is the oracle. An engine
            // that produced different ids did different work, and its speed
            // is not a comparable number.
            if run_encode || run_scaling {
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
                    if measurement.is_none() && !bad.is_empty() {
                        eprintln!(
                            "  ! {model_name}/{corpus_name}: ids differ from {}: {}",
                            registry::REFERENCE,
                            bad.join(", ")
                        );
                    }
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
                if measurement.is_none() && !bad.is_empty() {
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
                    pipeline_cache_capacity: args.cache_capacity,
                },
                results,
            });
            done += 1;
            let per = started.elapsed().as_secs_f64() / done as f64;
            if measurement.is_none() {
                eprintln!(
                    "  -- {done}/{cells} cells | elapsed {:.0}s | eta ~{:.0}s",
                    started.elapsed().as_secs_f64(),
                    per * (cells - done) as f64
                );
            } else {
                print_progress(done, cells, started.elapsed().as_secs_f64());
            }
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
    if run_memory {
        let total = runs.len();
        eprintln!(
            "footprint pass: {total} cells (kept separate from timing — the child \
             processes would otherwise evict the next cell's caches)"
        );
        for (i, (run, (model, corpus))) in runs.iter_mut().zip(&cell_inputs).enumerate() {
            for r in run.results.iter_mut().filter(|r| r.unsupported.is_none()) {
                let cache_capacity = (r.tokenizer_name == "pipeline")
                    .then_some(args.cache_capacity)
                    .flatten();
                let threads = if run_memory_atomic {
                    args.threads.get()
                } else {
                    1
                };
                let mode = if run_memory_atomic {
                    args.scaling_mode.as_str()
                } else {
                    "independent-instances"
                };
                if let Some(m) = measure_memory_repeated(
                    &r.tokenizer_name,
                    model,
                    corpus,
                    cache_capacity,
                    threads,
                    mode,
                    if run_memory_atomic { args.reps } else { 1 },
                ) {
                    r.heap_load_mb = m.heap_load_mb;
                    r.heap_encode_mb = m.heap_encode_mb;
                    r.memory_threads = m.memory_threads;
                    r.memory_parallelism = m.memory_parallelism;
                    if let Some(value) = m.engine_version {
                        r.engine_version = value;
                    }
                    if let Some(value) = m.engine_lang {
                        r.engine_lang = value;
                    }
                    if let Some(value) = m.engine_class {
                        r.engine_class = value;
                    }
                    if let Some(value) = m.also_computes {
                        r.also_computes = value;
                    }
                    if let Some(value) = m.internally_parallel {
                        r.internally_parallel = value;
                    }
                    if m.unsupported.is_some() {
                        r.unsupported = m.unsupported;
                    }
                }
            }
            if (i + 1) % 10 == 0 {
                eprintln!("  footprint {}/{total}", i + 1);
            }
        }
    }

    if let Some(measurement) = measurement {
        print_measurement_table(&runs, measurement, args.compare_to.as_deref());
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
    let threads = args.memory_threads.map_or(1, NonZeroUsize::get);
    let base = tokbench_core::mem::live_heap();
    let cache_capacity = (name == "pipeline")
        .then_some(args.cache_capacity)
        .flatten();
    let build = || ctor(&model, cache_capacity).ok();
    let Some(mut first) = build() else {
        println!(
            "{}",
            serde_json::json!({"unsupported": "could not load engine"})
        );
        return Ok(());
    };
    let info = first.info();
    let native = match args.scaling_mode.as_str() {
        "native-threads" => first.set_threads(threads),
        "independent-instances" => false,
        _ => first.set_threads(threads),
    };
    if args.scaling_mode == "native-threads" && !native {
        println!(
            "{}",
            serde_json::json!({"unsupported": format!("engine cannot run with {threads} native threads")})
        );
        return Ok(());
    }
    let parallelism = if native {
        "native-threads"
    } else {
        "independent-instances"
    };

    // Two samples answer two different questions: the retained tokenizer
    // structures, and the state retained while the configured workers are
    // warm. Independent mode keeps every tokenizer alive at both snapshots.
    if !native && first.has_native_batch() && !first.set_threads(1) {
        println!(
            "{}",
            serde_json::json!({
                "unsupported": "engine has a native batch pool that cannot be pinned to one thread"
            })
        );
        return Ok(());
    }
    let mut engines = vec![first];
    if !native {
        while engines.len() < threads {
            let Some(mut engine) = build() else {
                println!(
                    "{}",
                    serde_json::json!({"unsupported": "could not load every worker instance"})
                );
                return Ok(());
            };
            // Prevent an engine's own pool from multiplying the requested
            // independent worker count whenever it exposes a width control.
            engine.set_threads(1);
            engines.push(engine);
        }
    }
    let after_load = tokbench_core::mem::live_heap();

    let after_encode = if native {
        let refs: Vec<&str> = chunks.iter().map(String::as_str).collect();
        let mut out = Vec::new();
        engines[0].encode_batch(&refs, &mut out);
        drop(out);
        tokbench_core::mem::live_heap()
    } else {
        use std::sync::{Arc, Barrier};
        let done = Arc::new(Barrier::new(threads + 1));
        let release = Arc::new(Barrier::new(threads + 1));
        let sample = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(threads);
            for (worker, mut engine) in engines.drain(..).enumerate() {
                let done = Arc::clone(&done);
                let release = Arc::clone(&release);
                let chunks = &chunks;
                handles.push(scope.spawn(move || {
                    let mut out = Vec::new();
                    for text in chunks.iter().skip(worker).step_by(threads) {
                        out.clear();
                        engine.encode(text, &mut out);
                    }
                    drop(out);
                    done.wait();
                    release.wait();
                    engine
                }));
            }
            done.wait();
            let sample = tokbench_core::mem::live_heap();
            release.wait();
            for handle in handles {
                if let Ok(engine) = handle.join() {
                    engines.push(engine);
                }
            }
            sample
        });
        sample
    };

    let grew = |a: Option<u64>| match (base, a) {
        (Some(b), Some(a)) => Some(a.saturating_sub(b) as f64 / (1024.0 * 1024.0)),
        _ => None,
    };
    println!(
        "{}",
        serde_json::json!({
            "heap_load_mb": grew(after_load),
            "heap_encode_mb": grew(after_encode),
            "memory_threads": threads,
            "memory_parallelism": parallelism,
            "engine_version": info.version,
            "engine_lang": info.lang,
            "engine_class": info.class.as_str(),
            "also_computes": info.also_computes,
            "internally_parallel": info.internally_parallel,
        })
    );
    // Held until after the readings: dropping earlier would free the very
    // allocations being measured.
    drop(engines);
    Ok(())
}

/// The four footprint numbers a child reports. See `memory_child`.
#[derive(Default)]
struct Footprint {
    heap_load_mb: Option<f64>,
    heap_encode_mb: Option<f64>,
    memory_threads: Option<usize>,
    memory_parallelism: Option<String>,
    engine_version: Option<String>,
    engine_lang: Option<String>,
    engine_class: Option<String>,
    also_computes: Option<String>,
    internally_parallel: Option<bool>,
    unsupported: Option<String>,
}

/// Spawn one isolated child per repetition and take the median footprint.
fn measure_memory_repeated(
    name: &str,
    model: &Model,
    corpus: &Path,
    cache_capacity: Option<usize>,
    threads: usize,
    scaling_mode: &str,
    reps: usize,
) -> Option<Footprint> {
    let samples: Vec<_> = (0..reps.max(1))
        .filter_map(|_| {
            measure_memory_once(name, model, corpus, cache_capacity, threads, scaling_mode)
        })
        .collect();
    let first = samples.first()?;
    if first.unsupported.is_some() {
        return Some(Footprint {
            unsupported: first.unsupported.clone(),
            ..Default::default()
        });
    }
    Some(Footprint {
        heap_load_mb: median(
            samples
                .iter()
                .filter_map(|sample| sample.heap_load_mb)
                .collect(),
        ),
        heap_encode_mb: median(
            samples
                .iter()
                .filter_map(|sample| sample.heap_encode_mb)
                .collect(),
        ),
        memory_threads: first.memory_threads,
        memory_parallelism: first.memory_parallelism.clone(),
        engine_version: first.engine_version.clone(),
        engine_lang: first.engine_lang.clone(),
        engine_class: first.engine_class.clone(),
        also_computes: first.also_computes.clone(),
        internally_parallel: first.internally_parallel,
        unsupported: None,
    })
}

fn measure_memory_once(
    name: &str,
    model: &Model,
    corpus: &Path,
    cache_capacity: Option<usize>,
    threads: usize,
    scaling_mode: &str,
) -> Option<Footprint> {
    let exe = std::env::current_exe().ok()?;
    let mut command = Command::new(exe);
    command
        .arg("--memory")
        .arg(name)
        .arg("--memory-model")
        .arg(&model.dir)
        .arg("--memory-corpus")
        .arg(corpus)
        .arg("--memory-threads")
        .arg(threads.to_string())
        .arg("--scaling-mode")
        .arg(scaling_mode);
    if let Some(capacity) = cache_capacity {
        command.arg("--cache-capacity").arg(capacity.to_string());
    }
    let out = command.output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let line = text
        .lines()
        .rev()
        .find(|l| l.trim_start().starts_with('{'))?;
    let v: serde_json::Value = serde_json::from_str(line).ok()?;
    let f = |k: &str| v.get(k).and_then(|x| x.as_f64());
    let s = |k: &str| v.get(k).and_then(|x| x.as_str()).map(str::to_string);
    Some(Footprint {
        heap_load_mb: f("heap_load_mb"),
        heap_encode_mb: f("heap_encode_mb"),
        memory_threads: v
            .get("memory_threads")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize),
        memory_parallelism: s("memory_parallelism"),
        engine_version: s("engine_version"),
        engine_lang: s("engine_lang"),
        engine_class: s("engine_class"),
        also_computes: s("also_computes"),
        internally_parallel: v.get("internally_parallel").and_then(|x| x.as_bool()),
        unsupported: s("unsupported"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_each_atomic_measurement() {
        for (name, expected) in [
            ("encode", MeasureCommand::Encode),
            ("decode", MeasureCommand::Decode),
            ("latency", MeasureCommand::Latency),
            ("scaling", MeasureCommand::Scaling),
            ("memory", MeasureCommand::Memory),
        ] {
            let args = Args::try_parse_from([
                "tokbench", "measure", name, "--engine", "pipeline", "--model", "gpt2", "--corpus",
                "eng_Latn",
            ])
            .unwrap();

            assert!(matches!(
                args.command,
                Some(CliCommand::Measure { command }) if command == expected
            ));
            assert_eq!(args.engine, ["pipeline"]);
            assert!(args.compare_to.is_none());
            assert_eq!(args.model, ["gpt2"]);
            assert_eq!(args.corpus, ["eng_Latn"]);
        }
    }

    #[test]
    fn parses_comparison_engine() {
        let args = Args::try_parse_from([
            "tokbench",
            "measure",
            "encode",
            "--engine",
            "pipeline",
            "--engine",
            "tiktoken",
            "--compare-to",
            "hf-tokenizers",
        ])
        .unwrap();

        assert_eq!(args.engine, ["pipeline", "tiktoken"]);
        assert_eq!(args.compare_to.as_deref(), Some("hf-tokenizers"));
    }

    #[test]
    fn parses_all_engines_selector() {
        let args =
            Args::try_parse_from(["tokbench", "measure", "encode", "--engine", "all"]).unwrap();

        assert_eq!(args.engine, ["all"]);
        assert!(args.model.is_empty());
        assert!(args.corpus.is_empty());
    }

    #[test]
    fn parses_pipeline_cache_capacity_including_zero() {
        let args = Args::try_parse_from([
            "tokbench",
            "measure",
            "encode",
            "--engine",
            "pipeline",
            "--cache-capacity",
            "0",
        ])
        .unwrap();

        assert_eq!(args.cache_capacity, Some(0));
    }

    #[test]
    fn parses_scaling_measurement_controls() {
        let args = Args::try_parse_from([
            "tokbench",
            "measure",
            "scaling",
            "--engine",
            "pipeline",
            "--reps",
            "7",
            "--scaling-mode",
            "independent-instances",
        ])
        .unwrap();

        assert_eq!(args.reps, 7);
        assert_eq!(args.scaling_mode, "independent-instances");
    }

    #[test]
    fn parses_memory_worker_controls() {
        let args = Args::try_parse_from([
            "tokbench",
            "measure",
            "memory",
            "--engine",
            "pipeline",
            "--corpus",
            "eng_Latn",
            "--threads",
            "8",
            "--scaling-mode",
            "independent-instances",
        ])
        .unwrap();

        assert_eq!(args.threads.get(), 8);
        assert_eq!(args.scaling_mode, "independent-instances");
    }

    #[test]
    fn formats_atomic_progress() {
        assert_eq!(compact_duration(65.0), "1m05s");
        assert_eq!(compact_duration(3_661.0), "1h01m");
        assert_eq!(
            progress_line(5, 10, 20.0),
            "[##############--------------] 5/10 cells  elapsed 20s  eta 20s"
        );
    }

    #[test]
    fn formats_partial_comparison_coverage() {
        assert_eq!(
            median_multiplier(vec![2.0, 4.0], 3),
            "×3.00 on 2/3 comparable"
        );
        assert_eq!(median_multiplier(Vec::new(), 3), "0/3 comparable");
    }

    #[test]
    fn labels_scaling_as_observed_without_reinterpreting_it() {
        assert_eq!(format_scaling_efficiency(98.2), "98% observed");
        assert_eq!(format_scaling_efficiency(107.6), "108% observed");
    }

    #[test]
    fn latency_samples_are_capped_by_available_documents() {
        let documents = latency_documents("abcdefghijkl", 4, 1_000);
        assert_eq!(documents, ["abcd", "efgh", "ijkl"]);
        assert_eq!(documents.len() - 1, 2);

        assert_eq!(latency_documents("abcdefghijkl", 4, 1).len(), 2);
    }

    #[test]
    fn keeps_the_existing_full_run_cli() {
        let args = Args::try_parse_from(["tokbench", "--no-memory", "--no-decode"]).unwrap();

        assert!(args.command.is_none());
        assert!(args.no_memory);
        assert!(args.no_decode);
    }
}
