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
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use tokbench_core::{chunk, measure, Model, Phases};

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

    /// Skip the per-engine RSS child processes (they roughly double wall time,
    /// since each one reloads the model).
    #[arg(long)]
    no_memory: bool,

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

    /// Per-engine stripped binary deltas, as written by `scripts/binsize.sh`.
    /// Merged into the report when present.
    #[arg(long, default_value = "binary_sizes.json")]
    binary_sizes: PathBuf,

    /// Published package sizes, as written by `scripts/package_size.py`.
    #[arg(long, default_value = "package_sizes.json")]
    package_sizes: PathBuf,

    // --- internal: the memory child. One engine per process; see core::rss. ---
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

    // --- footprint ---
    /// Resident memory this engine holds once loaded and warmed, measured in a
    /// dedicated child process so one engine's pages cannot be credited to
    /// another. See `tokbench_core::rss`.
    #[serde(skip_serializing_if = "Option::is_none")]
    rss_delta_mb: Option<f64>,
    /// Process high-water mark: catches engines that transiently allocate far
    /// more than they retain. Linux only.
    #[serde(skip_serializing_if = "Option::is_none")]
    rss_peak_mb: Option<f64>,
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
    let thread_sweep = tokbench_core::thread_counts();
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
    let started = Instant::now();
    let mut done = 0usize;

    for model_dir in &models {
        let model_name = stem(model_dir);
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

            let mut results: Vec<EngineResult> = Vec::new();

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

                        eprintln!(
                            "  [{}/{}] {model_name}/{corpus_name} {name:<16} {:>8.1} MB/s  {:>6.2} ns/B  {} tok",
                            done + 1,
                            cells,
                            m.mbps,
                            m.ns_per_byte,
                            m.tokens
                        );

                        // Multi-thread sweep, only on the corpora asked for.
                        // Runs after the single-thread timing so it can never
                        // perturb the headline number.
                        let scaling = if args.scaling.iter().any(|c| *c == corpus_name) {
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
                                let best = pts.last().unwrap();
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
                            scaling,
                            ..Default::default()
                        });
                    }
                }
            }

            // Footprint: one child process per engine, after the timing is
            // done so the spawn cost can never land inside a measurement.
            if !args.no_memory {
                for r in results.iter_mut().filter(|r| r.unsupported.is_none()) {
                    if let Some((delta, peak)) =
                        measure_memory(&r.tokenizer_name, &model, corpus_path)
                    {
                        r.rss_delta_mb = delta;
                        r.rss_peak_mb = peak;
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
/// reports B's footprint as near zero — see `tokbench_core::rss`.
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

    // Everything the engine retains — vocabulary, automata, and the caches the
    // warm pass fills — is allocated inside this closure.
    let (built, delta, peak) = tokbench_core::rss::around(|| {
        let mut engine = ctor(&model).ok()?;
        let mut out = Vec::new();
        for c in &chunks {
            out.clear();
            engine.encode(c, &mut out);
        }
        // Keep the engine alive across the second RSS reading, or its pages
        // would be freed before they are counted.
        Some(engine)
    });
    if built.is_none() {
        println!("{{}}");
        return Ok(());
    }
    let mb = |b: Option<u64>| b.map(|b| b as f64 / (1024.0 * 1024.0));
    println!(
        "{}",
        serde_json::json!({ "rss_delta_mb": mb(delta), "rss_peak_mb": mb(peak) })
    );
    Ok(())
}

/// Spawn `memory_child` for one engine and read back its footprint.
fn measure_memory(name: &str, model: &Model, corpus: &Path) -> Option<(Option<f64>, Option<f64>)> {
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
    Some((
        v.get("rss_delta_mb").and_then(|x| x.as_f64()),
        v.get("rss_peak_mb").and_then(|x| x.as_f64()),
    ))
}
