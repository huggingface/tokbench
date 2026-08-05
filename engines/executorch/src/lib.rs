//! ExecuTorch's tokenizers (`pytorch-labs/tokenizers`) — the C++ tokenizer
//! library used by the ExecuTorch and torchchat runners.
//!
//! Wired through a three-function C shim (`shim/shim.cpp`) over
//! `tokenizers::HFTokenizer`, compiled by `build.rs` and called in-process, so
//! this is [`Class::Cffi`]: same clock as the Rust engines, with the FFI call
//! and the C++ side's allocations included.
//!
//! # Which variant, and why it is the only defensible one
//!
//! Upstream is a family of classes over one base, not a single tokenizer:
//! `HFTokenizer`, `Tiktoken`, `SPTokenizer`, `Llama2cTokenizer` and Tekken all
//! implement
//!
//! ```cpp
//! Error load(const std::string& tokenizer_path);
//! Result<std::vector<uint64_t>> encode(const std::string& input,
//!                                      int8_t bos, int8_t eos) const;
//! ```
//!
//! This engine benchmarks **`HFTokenizer`**, and the report says so in
//! [`Info::version`] rather than just "executorch". Two reasons:
//!
//! * It reads the same `tokenizer.json` as the reference engine, so its ids
//!   can be checked against the reference one for one. An engine whose output
//!   cannot be verified cannot honestly be given a throughput number.
//! * `SPTokenizer` merely delegates to libsentencepiece, which already has its
//!   own row here. Benchmarking it under the "executorch" name would print the
//!   same library twice under two names and invite a reader to compare them.
//!
//! [`Info::version`] carries the pinned upstream commit as well. The library
//! has no releases, so a bare name would attach a number to "whatever main was
//! that day"; `scripts/vendor_executorch.sh` pins the same commit.
//!
//! # Matching the reference's call
//!
//! The shim passes `bos = 0, eos = 0`. That is not merely "no BOS/EOS
//! requested" — in `hf_tokenizer.cpp` it is what switches the post-processor
//! off entirely:
//!
//! ```cpp
//! bool add_special = (bos > 0 || eos > 0);
//! if (_postprocessor) tokens = _postprocessor->process(tokens, add_special);
//! ```
//!
//! which is exactly the `add_special_tokens = false` the reference is called
//! with.
//!
//! # What stays inside the timed region
//!
//! Two costs that a naive adapter would be tempted to hide, kept in on purpose
//! (see the fairness rules in `tokbench_core`):
//!
//! * **The id copy and narrowing.** `encode` returns an owned
//!   `std::vector<uint64_t>`; the shim walks it into the caller's `u32`
//!   buffer. Every Rust engine here pays the same via `out.extend(...)`.
//! * **The `&str` -> `std::string` copy.** `encode` takes `const std::string&`
//!   with no `string_view` overload, so any caller holding a pointer and a
//!   length must materialise a `std::string`. The shim reuses one scratch
//!   string across calls, so it is a memcpy into owned capacity rather than a
//!   malloc — which mirrors the reused `out` buffer the harness hands the Rust
//!   engines, and neither flatters nor penalises this engine.
//!
//! # Degrading gracefully
//!
//! Building the C++ side means five git submodules (abseil, re2,
//! sentencepiece, nlohmann/json, pcre2) and a CMake build. Requiring that of
//! anyone running `cargo check` on the workspace would be hostile, so
//! `build.rs` looks for the vendored archives and, finding none, compiles this
//! crate down to the `Unsupported` arm below.

// The engine's identity (name, url, pinned revision) is stated unconditionally
// so there is one definition of it however the crate was compiled; only the
// wired build actually reads it.
#![cfg_attr(not(have_executorch), allow(dead_code))]

use tokbench_core::{Build, Engine, Model, Unsupported};

const NAME: &str = "executorch";
const URL: &str = "https://github.com/pytorch-labs/tokenizers";

/// Set by `build.rs` only when the library was actually linked, so a build
/// without the vendored tree cannot accidentally claim a revision it did not
/// run.
const VERSION: &str = match option_env!("TOKBENCH_EXECUTORCH_VERSION") {
    Some(v) => v,
    None => "HFTokenizer (not vendored)",
};

// ---------------------------------------------------------------------------
// Wired path
// ---------------------------------------------------------------------------

#[cfg(have_executorch)]
mod wired {
    use super::*;
    use std::ffi::{c_char, c_void, CString};
    use tokbench_core::{Class, Ids, Info};

    // shim/shim.cpp. Three functions, declared by hand — see Cargo.toml for
    // why there is no bindgen here.
    extern "C" {
        /// Returns an opaque handle, or null if the config could not be
        /// loaded. Not called inside the timed region.
        fn tokbench_et_create(path: *const c_char) -> *mut c_void;

        /// Encodes into `out` (room for `cap` ids). Returns the id count, or
        /// -1 on failure. A return greater than `cap` means nothing was
        /// written and the buffer needs to grow.
        fn tokbench_et_encode(
            handle: *mut c_void,
            text: *const c_char,
            len: usize,
            out: *mut u32,
            cap: usize,
        ) -> i64;

        fn tokbench_et_destroy(handle: *mut c_void);
    }

    pub struct Adapter {
        handle: *mut c_void,
    }

    /// The handle is uniquely owned by this `Adapter` — never cloned, never
    /// aliased, and freed in `Drop`. The C++ object behind it holds a scratch
    /// buffer mutated by `encode`, so it is emphatically not `Sync`, but
    /// moving the sole owner between threads is sound. `Send` is required
    /// because `tokbench_core::Engine: Send`, and the scaling sweep builds one
    /// independent instance per thread.
    unsafe impl Send for Adapter {}

    impl Adapter {
        pub fn open(model: &Model) -> Result<Box<dyn Engine>, Unsupported> {
            let path = model.tokenizer_json();
            if !path.exists() {
                return Err(Unsupported("no tokenizer.json".into()));
            }
            let c = CString::new(path.as_os_str().as_encoded_bytes())
                .map_err(|_| Unsupported("tokenizer.json path contains a NUL".into()))?;

            // SAFETY: `c` is a valid NUL-terminated string that outlives the
            // call; the shim copies the path and does not retain the pointer.
            let handle = unsafe { tokbench_et_create(c.as_ptr()) };
            if handle.is_null() {
                // HFTokenizer does not implement every HuggingFace
                // normalizer/pre-tokenizer combination, so a rejection here is
                // a real capability gap, not a harness failure.
                return Err(Unsupported(
                    "HFTokenizer rejected this tokenizer.json (unsupported normalizer, \
                     pre-tokenizer or model type)"
                        .into(),
                ));
            }
            Ok(Box::new(Adapter { handle }))
        }
    }

    impl Drop for Adapter {
        fn drop(&mut self) {
            // SAFETY: `handle` came from `tokbench_et_create`, is non-null
            // (checked in `open`), and this is the sole owner, running once.
            unsafe { tokbench_et_destroy(self.handle) }
        }
    }

    impl Engine for Adapter {
        fn info(&self) -> Info {
            Info {
                name: NAME,
                version: VERSION,
                lang: "c++",
                class: Class::Cffi,
                url: URL,
                also_computes: "",
                internally_parallel: false,
            }
        }

        fn encode(&mut self, text: &str, out: &mut Ids) {
            // No subword tokenizer emits more tokens than the input has bytes
            // — the degenerate case is one token per byte — so `text.len()` is
            // a hard upper bound and one `reserve` removes the need for a
            // size-probing round trip. The harness reuses `out` across the
            // corpus, so after the first document this is a no-op, exactly as
            // it is for the Rust engines' `extend`.
            out.reserve(text.len());
            let cap = out.capacity();

            // SAFETY: `handle` is live; `text` is valid for `text.len()`
            // bytes; `out` has `cap` writable slots for `u32`. The shim writes
            // at most `min(n, cap)` of them and returns how many.
            let n = unsafe {
                tokbench_et_encode(
                    self.handle,
                    text.as_ptr().cast::<c_char>(),
                    text.len(),
                    out.as_mut_ptr(),
                    cap,
                )
            };

            if n < 0 {
                return; // encode failed; leaving `out` empty surfaces as a mismatch
            }
            let n = n as usize;
            if n <= cap {
                // SAFETY: the shim initialised exactly `n` elements.
                unsafe { out.set_len(n) };
                return;
            }

            // Unreachable given the bound above, but a wrong guess must not
            // silently truncate: grow and encode once more.
            out.reserve(n);
            let cap = out.capacity();
            // SAFETY: as above, with a buffer now known to be large enough.
            let n = unsafe {
                tokbench_et_encode(
                    self.handle,
                    text.as_ptr().cast::<c_char>(),
                    text.len(),
                    out.as_mut_ptr(),
                    cap,
                )
            };
            if n >= 0 && (n as usize) <= cap {
                // SAFETY: the shim initialised exactly `n` elements.
                unsafe { out.set_len(n as usize) };
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub struct Adapter;

impl Build for Adapter {
    #[cfg(have_executorch)]
    fn build(model: &Model) -> Result<Box<dyn Engine>, Unsupported> {
        wired::Adapter::open(model)
    }

    #[cfg(not(have_executorch))]
    fn build(_model: &Model) -> Result<Box<dyn Engine>, Unsupported> {
        Err(Unsupported(
            "pytorch-labs/tokenizers is not vendored: run \
             engines/executorch/scripts/vendor_executorch.sh (clones the pinned commit \
             and builds it with CMake), then rebuild"
                .into(),
        ))
    }
}
