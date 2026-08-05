//! Microsoft BlingFire.
//!
//! STATUS: scaffolded, not yet wired — and the reason matters more than usual.
//!
//! The obvious move is the `blingfire` crate (1.0.0), and it is the wrong one:
//! it exposes exactly `text_to_words` and `text_to_sentences`, both of which
//! return a *string* of space-separated surface forms. Neither produces token
//! ids, so neither can be compared against any other engine here. Benchmarking
//! `text_to_words` and labelling the row "BlingFire" would compare a whitespace
//! segmenter against full subword tokenizers — the single most misleading thing
//! this repository could publish.
//!
//! The comparable entry point is `TextToIds` in `libblingfiretokdll`, which
//! BlingFire ships but the Rust crate does not bind:
//!
//! ```c
//! void* LoadModel(const char* pszModelPath);
//! int   TextToIds(void* handle, const char* pInUtf8Str, int InUtf8StrByteCount,
//!                 int32_t* pIdsArr, int MaxIdsArrLength, int UnkId);
//! int   FreeModel(void* handle);
//! ```
//!
//! To wire it:
//!   1. `scripts/vendor_blingfire.sh` builds `libblingfiretokdll` (CMake).
//!   2. A `build.rs` emits `cargo:rustc-link-lib=blingfiretokdll` plus the
//!      search path, and declares the three functions above in an `extern "C"`
//!      block.
//!   3. `build()` calls `LoadModel` on the model's `*.bin` artifact — BlingFire
//!      uses its own compiled model format, so `make models` must produce one
//!      for the vocabulary under test, or the engine stays `Unsupported` for
//!      models it has no `.bin` for.
//!   4. `TextToIds` writes into a caller-provided `i32` buffer, which suits the
//!      reused-buffer contract: size it once at `text.len()` ids and reuse.
//!
//! Class is [`tokbench_core::Class::Cffi`] once wired: in-process, same clock,
//! with the FFI call included.

use tokbench_core::{Build, Engine, Model, Unsupported};

pub struct Adapter;

impl Build for Adapter {
    fn build(_model: &Model) -> Result<Box<dyn Engine>, Unsupported> {
        Err(Unsupported(
            "not wired: needs libblingfiretokdll TextToIds (the blingfire crate only \
             segments words, which is not comparable); see engines/blingfire/src/lib.rs"
                .into(),
        ))
    }
}
