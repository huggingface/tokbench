//! gigatoken (marcelroed) — Rust BPE whose speed comes from a hand-written
//! SWAR pre-tokenizer and a pretoken cache rather than a faster merge loop.
//!
//! STATUS: scaffolded, not yet wired. Not published on crates.io (the PyPI
//! `gigatoken` wheel is the distributed artifact), so it enters as a pinned
//! git dependency.
//!
//! Two things to get right when wiring it, both of which decide whether the
//! published number means anything:
//!
//! * **Threads.** gigatoken's headline figures are whole-machine numbers on a
//!   many-core host. If its encode path spreads across cores while every other
//!   row here is single-threaded, the comparison is a core-count comparison.
//!   Pin it to one thread for the headline; if that is not possible, set
//!   `Info::internally_parallel = true` so the report marks the row.
//! * **The pretoken cache.** It is the main source of the speedup, and
//!   `tokbench_core::measure` runs a warm-up pass by design, so the cache will
//!   be warm — that is the regime real workloads see. `--no-warmup` gives the
//!   cold contrast, and both belong in any writeup that quotes a multiplier.
//!
//! Wiring: pin a rev in Cargo.toml, load `model.tokenizer_json()`, and report
//! the exact rev in `Info::version`.

use tokbench_core::{Build, Engine, Model, Unsupported};

pub struct Adapter;

impl Build for Adapter {
    fn build(_model: &Model) -> Result<Box<dyn Engine>, Unsupported> {
        Err(Unsupported(
            "not wired: pin a gigatoken git rev in Cargo.toml; \
             see engines/gigatoken/src/lib.rs"
                .into(),
        ))
    }
}
