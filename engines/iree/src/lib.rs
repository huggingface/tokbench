//! IREE's tokenizer (`iree-org/iree`, `runtime/src/iree/tokenizer`) — a
//! streaming tokenizer in C that loads HuggingFace `tokenizer.json` and
//! OpenAI `.tiktoken` directly, covering BPE, WordPiece and Unigram.
//!
//! STATUS: scaffolded, not yet wired.
//!
//! This is the best-matched foreign engine in the set: it reads the same
//! `tokenizer.json` as the reference, it is a C library so it needs no
//! interpreter, and it is documented as allocation-free in the hot path with
//! caller-provided buffers — which lines up exactly with the reused-`out`
//! buffer contract in `tokbench_core::Engine::encode`.
//!
//! Wiring notes:
//!
//! * IREE is a large CMake project; build only the tokenizer target rather
//!   than the whole runtime (`scripts/vendor_iree.sh`), then link statically.
//! * The API is pull-based/streaming: the encode loop feeds input and drains
//!   token ids until the input is consumed. Drive it to completion inside one
//!   `encode` call so the measured unit stays "one document → its ids", the
//!   same unit every other engine is charged for. Do NOT report a partial
//!   stream as a completed encode.
//! * It also tracks byte-exact offsets through normalization for training use
//!   cases. If offsets cannot be switched off, set
//!   `Info::also_computes = "byte offsets"` so the report discloses that it is
//!   doing strictly more work than the ids-only engines it sits next to.
//! * `iree-org/iree-tokenizer-py` exists; it is the same C core behind CPython.
//!   Prefer the C library here — an in-process C call is the fairer measurement
//!   and avoids attributing interpreter overhead to IREE.

use tokbench_core::{Build, Engine, Model, Unsupported};

pub struct Adapter;

impl Build for Adapter {
    fn build(_model: &Model) -> Result<Box<dyn Engine>, Unsupported> {
        Err(Unsupported(
            "not wired: build the iree tokenizer C target and bind it; \
             see engines/iree/src/lib.rs"
                .into(),
        ))
    }
}
