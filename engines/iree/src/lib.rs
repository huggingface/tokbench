//! IREE's tokenizer (`iree-org/iree`, `runtime/src/iree/tokenizer`) — a
//! streaming tokenizer in C that loads HuggingFace `tokenizer.json` and
//! OpenAI `.tiktoken` directly, covering BPE, WordPiece and Unigram.
//!
//! # Why this engine is worth the FFI
//!
//! It is the best-matched foreign engine in the set. It parses the *same*
//! `tokenizer.json` the reference does, through its own C parser
//! (`iree_tokenizer_from_huggingface_json`), so there is no conversion step in
//! between that could quietly change the vocabulary — its ids are directly
//! comparable to `hf-tokenizers` and any divergence is a real disagreement
//! about tokenization, not an artefact of the harness. And being C, it needs no
//! interpreter: [`Class::Cffi`] means the only thing between the benchmark's
//! `Instant` and the tokenizer is a call instruction.
//!
//! # How it is driven, and why that is the fair way
//!
//! IREE's encode API is pull-based. You hand the encoder an input chunk and an
//! output window; it pulls bytes through normalizer → segmenter → model until
//! the output window is full or the input is exhausted, and reports how much of
//! each it used. Tokens still buffered in the pipeline come out on `finalize`.
//!
//! There is also a one-shot `iree_tokenizer_encode` convenience wrapper, and
//! this adapter deliberately does NOT use it. That wrapper does
//! `initialize → feed → finalize → deinitialize` internally, which means it
//! allocates and tears down ~120 kB of encoder state on *every* call. Measuring
//! that would charge IREE for a setup cost its own documentation tells you to
//! hoist out of the loop, and would contradict the library's headline claim of
//! being allocation-free in the hot path. So instead the state, its backing
//! storage and the transform scratch buffer are allocated once in
//! [`Build::build`] — before the clock starts, per fairness rule 3 — and each
//! `encode` does `reset → feed… → finalize` over the reused buffers. That is
//! the loop IREE is designed for and the one a real caller writes.
//!
//! What is emphatically NOT done is reporting a partial stream as a finished
//! encode. `encode` loops until every input byte has been consumed and then
//! finalizes; a document is only "done" when its last id is out. If the stream
//! ever stalls or the C side returns an error, the adapter stops and leaves
//! `out` short — which changes the id hash and shows up as `mismatch` in the
//! report. Failing loudly is the point: a truncated encode that looked fast
//! would be the single most misleading number this benchmark could produce.
//!
//! # Offsets are off
//!
//! IREE can map every token back to a byte range in the original input, but it
//! is opt-in twice over: you must pass a `token_offsets` array *and* set
//! `IREE_TOKENIZER_ENCODE_FLAG_TRACK_OFFSETS`. This adapter does neither, so
//! [`Info::also_computes`] is `""` — it is genuinely doing ids-only work, like
//! `tiktoken` and unlike `hf-tokenizers`, whose `encode` computes offsets and
//! word ids whether you want them or not. (This is why the scaffold's guess
//! that offsets might be unavoidable is not carried through: the header says
//! otherwise, and the header wins.)
//!
//! # Flags
//!
//! `AT_INPUT_START` only. In particular:
//!
//! * `ADD_SPECIAL_TOKENS` is **off**, matching the reference's
//!   `encode(text, false)` — no BOS/EOS from the post-processor template.
//! * `NO_SPECIAL_TOKEN_MATCHING` is **off**, which is the subtler of the two.
//!   Setting it would be tiktoken's `encode_ordinary`, where `<|endoftext|>` in
//!   the input tokenizes as literal text. HuggingFace's `add_special_tokens =
//!   false` does *not* do that — it still matches added/special tokens found in
//!   the input. Leaving this flag clear is what makes the id streams comparable.
//!
//! # Building the C side
//!
//! `scripts/vendor_iree.sh` fetches a sparse, blob-filtered, pinned checkout of
//! three IREE directories and compiles them into a static library with plain
//! `cc`. No CMake, because none is needed: the tokenizer's dependency cone is
//! self-contained C17 with no generated headers. `build.rs` links that archive
//! if it exists and otherwise leaves `cfg(iree_available)` unset, so an
//! un-vendored checkout still compiles and this engine simply reports
//! [`Unsupported`]. Nobody has to vendor IREE to build the workspace.

use tokbench_core::{Build, Engine, Model, Unsupported};

#[cfg(iree_available)]
use tokbench_core::{Class, Ids, Info};

#[cfg(iree_available)]
const URL: &str = "https://github.com/iree-org/iree";

/// The IREE revision that was actually linked, as recorded by `build.rs` from
/// `vendor/COMMIT`. Not a guess: when the library is absent there is no commit
/// to name and the engine is unsupported anyway.
#[cfg(iree_available)]
fn version() -> &'static str {
    option_env!("TOKBENCH_IREE_COMMIT").unwrap_or("not-vendored")
}

// ---------------------------------------------------------------------------
// Not vendored: compile, but decline to run.
// ---------------------------------------------------------------------------

#[cfg(not(iree_available))]
pub struct Adapter;

#[cfg(not(iree_available))]
impl Build for Adapter {
    fn build(_model: &Model) -> Result<Box<dyn Engine>, Unsupported> {
        Err(Unsupported(
            "iree tokenizer not vendored: run `bash engines/iree/scripts/vendor_iree.sh` \
             to fetch the pinned commit and build engines/iree/vendor/lib/libiree_tokenizer.a"
                .into(),
        ))
    }
}

// ---------------------------------------------------------------------------
// Vendored: the real engine.
// ---------------------------------------------------------------------------

#[cfg(iree_available)]
mod ffi {
    //! Hand-written declarations transcribed from the vendored headers.
    //!
    //! Every signature here was read out of the C source at the pinned commit
    //! — `runtime/src/iree/tokenizer/{tokenizer.h,types.h}`,
    //! `runtime/src/iree/tokenizer/format/huggingface/tokenizer_json.h` and
    //! `runtime/src/iree/base/{allocator.h,status.h,string_view.h}` — not
    //! inferred from names. bindgen is avoided so the crate has no build-time
    //! dependency on libclang; the surface is a dozen functions and it is
    //! cheaper to keep them honest by hand than to add that requirement.

    use std::os::raw::{c_char, c_int, c_void};

    /// `iree/base/config.h`: `#define IREE_HOST_SIZE_T size_t`.
    pub type HostSize = usize;

    /// `iree/base/status.h`: `typedef struct iree_status_handle_t* iree_status_t;`
    /// with `iree_status_is_ok(v)` defined as `(uintptr_t)(v) == IREE_STATUS_OK`
    /// and `IREE_STATUS_OK = 0` — so OK is exactly the null pointer.
    pub type Status = *mut c_void;

    #[inline]
    pub fn is_ok(s: Status) -> bool {
        s.is_null()
    }

    /// `iree/base/string_view.h`.
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct StringView {
        pub data: *const c_char,
        pub size: HostSize,
    }

    /// `iree/base/allocator.h`.
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct ByteSpan {
        pub data: *mut u8,
        pub data_length: HostSize,
    }

    /// `iree_allocator_ctl_fn_t`. We never call it — we only store its address
    /// in an [`Allocator`] and hand that to IREE — but the true signature is
    /// spelled out rather than erased to `*const c_void` so the declaration
    /// documents itself. `iree_allocator_command_t` is a plain C enum, hence
    /// `c_int`.
    pub type AllocatorCtlFn = unsafe extern "C" fn(
        self_: *mut c_void,
        command: c_int,
        params: *const c_void,
        inout_ptr: *mut *mut c_void,
    ) -> Status;

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct Allocator {
        pub self_: *mut c_void,
        pub ctl: AllocatorCtlFn,
    }

    /// `iree_tokenizer_offset_run_list_t`, passed empty because offset tracking
    /// is off.
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct OffsetRunList {
        pub capacity: HostSize,
        pub values: *mut c_void,
    }

    /// `iree_tokenizer_token_output_t` from `types.h`. `token_offsets` and
    /// `type_ids` are documented as "NULL to skip", which is what this adapter
    /// does for both.
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct TokenOutput {
        pub capacity: HostSize,
        pub token_ids: *mut i32,
        pub token_offsets: *mut c_void,
        pub type_ids: *mut u8,
    }

    pub enum Tokenizer {}
    pub enum EncodeState {}

    /// `iree_tokenizer_encode_flag_bits_e`.
    pub const FLAG_AT_INPUT_START: u32 = 1 << 0;

    unsafe extern "C" {
        /// The system allocator IREE itself defaults to
        /// (`IREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_libc_ctl` in
        /// `base/BUILD.bazel`); malloc/free underneath.
        pub safe fn iree_allocator_libc_ctl(
            self_: *mut c_void,
            command: c_int,
            params: *const c_void,
            inout_ptr: *mut *mut c_void,
        ) -> Status;

        pub fn iree_status_free(status: Status);

        pub fn iree_status_to_string(
            status: Status,
            allocator: *const Allocator,
            out_buffer: *mut *mut c_char,
            out_buffer_length: *mut HostSize,
        ) -> bool;

        pub fn iree_tokenizer_from_huggingface_json(
            json: StringView,
            allocator: Allocator,
            out_tokenizer: *mut *mut Tokenizer,
        ) -> Status;

        pub fn iree_tokenizer_free(tokenizer: *mut Tokenizer);

        pub fn iree_tokenizer_model_type_name(tokenizer: *const Tokenizer) -> StringView;

        pub fn iree_tokenizer_encode_state_calculate_size(
            tokenizer: *const Tokenizer,
            out_size: *mut HostSize,
        ) -> Status;

        pub fn iree_tokenizer_encode_state_initialize(
            tokenizer: *const Tokenizer,
            state_storage: ByteSpan,
            transform_buffer: ByteSpan,
            offset_runs: OffsetRunList,
            flags: u32,
            out_state: *mut *mut EncodeState,
        ) -> Status;

        pub fn iree_tokenizer_encode_state_deinitialize(state: *mut EncodeState);

        pub fn iree_tokenizer_encode_state_reset(state: *mut EncodeState, flags: u32);

        pub fn iree_tokenizer_encode_state_feed(
            state: *mut EncodeState,
            chunk: StringView,
            output: TokenOutput,
            out_bytes_consumed: *mut HostSize,
            out_token_count: *mut HostSize,
        ) -> Status;

        pub fn iree_tokenizer_encode_state_pending_token_bound(
            state: *const EncodeState,
        ) -> HostSize;

        pub fn iree_tokenizer_encode_state_finalize(
            state: *mut EncodeState,
            output: TokenOutput,
            out_token_count: *mut HostSize,
        ) -> Status;

        #[link_name = "free"]
        fn libc_free(ptr: *mut c_void);
    }

    /// The libc system allocator, reconstructed the way `iree_allocator_system()`
    /// does when `IREE_ALLOCATOR_SYSTEM_CTL` is defined: `{NULL, ctl}`.
    pub fn system_allocator() -> Allocator {
        Allocator {
            self_: std::ptr::null_mut(),
            ctl: iree_allocator_libc_ctl,
        }
    }

    /// Consume a status, returning its message if it is an error.
    ///
    /// IREE statuses carry a heap payload, so an error that is merely tested
    /// and dropped leaks. Every call site funnels through here so that cannot
    /// happen.
    pub fn take_error(status: Status) -> Option<String> {
        if is_ok(status) {
            return None;
        }
        let alloc = system_allocator();
        let mut buf: *mut c_char = std::ptr::null_mut();
        let mut len: HostSize = 0;
        let msg = unsafe {
            if iree_status_to_string(status, &alloc, &mut buf, &mut len) && !buf.is_null() {
                let s = std::slice::from_raw_parts(buf as *const u8, len);
                let owned = String::from_utf8_lossy(s).into_owned();
                libc_free(buf as *mut c_void);
                owned
            } else {
                "unknown IREE error".to_string()
            }
        };
        unsafe { iree_status_free(status) };
        Some(msg)
    }
}

#[cfg(iree_available)]
pub struct Adapter {
    tokenizer: *mut ffi::Tokenizer,
    state: *mut ffi::EncodeState,
    /// Backing storage for the encode state. IREE writes its state *into* this
    /// buffer rather than allocating; we own it and must keep it alive and at a
    /// stable address for as long as `state` is initialized. Boxed slices, not
    /// `Vec`, precisely so there is no `push` that could reallocate underneath
    /// a live pointer.
    state_storage: Box<[u8]>,
    /// Scratch the normalizer writes through. Sized per document; see
    /// [`Adapter::ensure_transform`].
    transform: Box<[u8]>,
    model_type: String,
}

// SAFETY: `Engine: Send`, and this type holds raw pointers, so the impl has to
// be written out. It is sound because every allocation an `Adapter` points at
// is exclusively owned by that `Adapter`: the tokenizer comes from a
// per-instance `iree_tokenizer_from_huggingface_json`, and the encode state
// lives in this struct's own `state_storage`. Nothing is shared between
// instances, which is exactly the arrangement `measure_scaling` relies on when
// it builds one engine per thread. IREE additionally documents the tokenizer
// itself as thread-safe, but this impl does not depend on that.
#[cfg(iree_available)]
unsafe impl Send for Adapter {}

#[cfg(iree_available)]
impl Build for Adapter {
    fn build(model: &Model) -> Result<Box<dyn Engine>, Unsupported> {
        let path = model.tokenizer_json();
        if !path.exists() {
            return Err(Unsupported(format!(
                "no tokenizer.json in {}",
                model.dir.display()
            )));
        }
        let json = std::fs::read_to_string(&path)
            .map_err(|e| Unsupported(format!("cannot read {}: {e}", path.display())))?;

        let alloc = ffi::system_allocator();
        let mut tokenizer: *mut ffi::Tokenizer = std::ptr::null_mut();
        let status = unsafe {
            ffi::iree_tokenizer_from_huggingface_json(
                ffi::StringView {
                    data: json.as_ptr() as *const std::os::raw::c_char,
                    size: json.len(),
                },
                alloc,
                &mut tokenizer,
            )
        };
        if let Some(msg) = ffi::take_error(status) {
            // UNIMPLEMENTED here means a component this tokenizer.json needs is
            // not supported yet — a legitimate sparse-matrix cell, not a bug.
            return Err(Unsupported(format!("iree cannot load this config: {msg}")));
        }
        if tokenizer.is_null() {
            return Err(Unsupported("iree returned a null tokenizer".into()));
        }

        let model_type = unsafe {
            let sv = ffi::iree_tokenizer_model_type_name(tokenizer);
            if sv.data.is_null() || sv.size == 0 {
                String::new()
            } else {
                String::from_utf8_lossy(std::slice::from_raw_parts(sv.data as *const u8, sv.size))
                    .into_owned()
            }
        };

        // State storage is sized by the library, once, outside the timer.
        let mut state_size: usize = 0;
        let status =
            unsafe { ffi::iree_tokenizer_encode_state_calculate_size(tokenizer, &mut state_size) };
        if let Some(msg) = ffi::take_error(status) {
            unsafe { ffi::iree_tokenizer_free(tokenizer) };
            return Err(Unsupported(format!("iree state sizing failed: {msg}")));
        }

        let mut adapter = Adapter {
            tokenizer,
            state: std::ptr::null_mut(),
            state_storage: vec![0u8; state_size].into_boxed_slice(),
            // Start at IREE's own minimum; grown on first use to fit the
            // corpus's document size. Growth happens inside `encode`, but only
            // ever on the warm-up pass in practice, because `measure` runs a
            // full untimed pass before the clock starts.
            transform: vec![0u8; 4096].into_boxed_slice(),
            model_type,
        };
        adapter
            .init_state()
            .map_err(|msg| Unsupported(format!("iree state init failed: {msg}")))?;
        Ok(Box::new(adapter))
    }
}

#[cfg(iree_available)]
impl Adapter {
    /// (Re)initialize the encode state over the current buffers.
    fn init_state(&mut self) -> Result<(), String> {
        if !self.state.is_null() {
            unsafe { ffi::iree_tokenizer_encode_state_deinitialize(self.state) };
            self.state = std::ptr::null_mut();
        }
        let mut state: *mut ffi::EncodeState = std::ptr::null_mut();
        let status = unsafe {
            ffi::iree_tokenizer_encode_state_initialize(
                self.tokenizer,
                ffi::ByteSpan {
                    data: self.state_storage.as_mut_ptr(),
                    data_length: self.state_storage.len(),
                },
                ffi::ByteSpan {
                    data: self.transform.as_mut_ptr(),
                    data_length: self.transform.len(),
                },
                // Offset tracking off: empty run list, no TRACK_OFFSETS flag.
                ffi::OffsetRunList {
                    capacity: 0,
                    values: std::ptr::null_mut(),
                },
                ffi::FLAG_AT_INPUT_START,
                &mut state,
            )
        };
        if let Some(msg) = ffi::take_error(status) {
            return Err(msg);
        }
        self.state = state;
        Ok(())
    }

    /// Guarantee the transform buffer can hold a whole document.
    ///
    /// IREE offers two sizing helpers. The streaming one caps at 16 kB, which
    /// is right for an endless stream but wrong here: it can deadlock when a
    /// single pre-token segment is longer than the buffer, and IREE's own
    /// header says so. The one-shot helper is uncapped —
    /// `next_pow2(3 * text_len)`, giving 1.5x the input in ring capacity — so a
    /// segment can never fail to fit. Since the harness hands us whole
    /// documents, one-shot sizing is the correct choice and removes the stall
    /// case entirely.
    ///
    /// The buffer only ever grows, and re-initializing the state is the price
    /// of growing it. In a `measure` run that cost is paid on the untimed
    /// warm-up pass and never again.
    fn ensure_transform(&mut self, text_len: usize) -> bool {
        let want = text_len.saturating_mul(3).max(4096).next_power_of_two();
        if want <= self.transform.len() {
            return true;
        }
        self.transform = vec![0u8; want].into_boxed_slice();
        self.init_state().is_ok()
    }

    /// The model family IREE decided this `tokenizer.json` is ("BPE",
    /// "WordPiece", "Unigram"). Exposed for debugging a mismatch: if IREE and
    /// the reference disagree on ids, the first thing worth checking is whether
    /// they even agree on the model type.
    pub fn model_type(&self) -> &str {
        &self.model_type
    }
}

#[cfg(iree_available)]
impl Engine for Adapter {
    fn info(&self) -> Info {
        Info {
            name: "iree",
            version: version(),
            lang: "c",
            class: Class::Cffi,
            url: URL,
            // Offsets and type ids are both opt-in and both left off; this is
            // an ids-only measurement. See the module docs.
            also_computes: "",
            internally_parallel: false,
        }
    }

    fn encode(&mut self, text: &str, out: &mut Ids) {
        if self.state.is_null() || !self.ensure_transform(text.len()) {
            return;
        }
        unsafe { ffi::iree_tokenizer_encode_state_reset(self.state, ffi::FLAG_AT_INPUT_START) };

        // Tokens are written straight into `out`'s spare capacity. IREE's
        // output type is `int32_t*` and `Ids` is `Vec<u32>`; the two have
        // identical layout and every bit pattern is valid for both, so this is
        // a reinterpretation, not a conversion, and there is no shadow buffer
        // to copy out of. This is the caller-provided-buffer API used the way
        // it is meant to be used — not a shortcut around work the public path
        // would otherwise do.
        let base = out.len();

        let mut consumed_total = 0usize;
        let mut stalls = 0u32;

        while consumed_total < text.len() {
            // Keep a workable window in front of the encoder. The output buffer
            // is what drives a pull-based encoder, so a tiny window means many
            // round trips.
            if out.capacity() - out.len() < 256 {
                out.reserve(out.len().max(text.len() / 2).max(1024));
            }
            let spare = out.capacity() - out.len();
            let output = ffi::TokenOutput {
                capacity: spare,
                token_ids: unsafe { out.as_mut_ptr().add(out.len()) } as *mut i32,
                token_offsets: std::ptr::null_mut(),
                type_ids: std::ptr::null_mut(),
            };

            let mut consumed: usize = 0;
            let mut produced: usize = 0;
            let status = unsafe {
                ffi::iree_tokenizer_encode_state_feed(
                    self.state,
                    ffi::StringView {
                        data: text.as_ptr().add(consumed_total) as *const std::os::raw::c_char,
                        size: text.len() - consumed_total,
                    },
                    output,
                    &mut consumed,
                    &mut produced,
                )
            };
            if ffi::take_error(status).is_some() {
                // Truncated on purpose: a short `out` changes the id hash and
                // the harness reports `mismatch` rather than a fast lie.
                out.truncate(base);
                return;
            }
            debug_assert!(produced <= spare);
            unsafe { out.set_len(out.len() + produced) };
            consumed_total += consumed;

            if consumed == 0 && produced == 0 {
                // The header promises progress whenever progress is possible,
                // so this means the output window was the binding constraint.
                // Grow it and try again; give up after a couple of rounds
                // rather than spinning forever.
                stalls += 1;
                if stalls > 2 {
                    out.truncate(base);
                    return;
                }
                out.reserve(out.capacity().max(1024));
            } else {
                stalls = 0;
            }
        }

        // All input consumed; flush whatever is still in the pipeline. finalize
        // is documented as non-retryable — it destroys pipeline state, so a
        // RESOURCE_EXHAUSTED from an undersized buffer cannot be recovered from
        // — which is why the bound is queried and reserved for first.
        let bound = unsafe { ffi::iree_tokenizer_encode_state_pending_token_bound(self.state) };
        if out.capacity() - out.len() < bound {
            out.reserve(bound);
        }
        let output = ffi::TokenOutput {
            capacity: out.capacity() - out.len(),
            token_ids: unsafe { out.as_mut_ptr().add(out.len()) } as *mut i32,
            token_offsets: std::ptr::null_mut(),
            type_ids: std::ptr::null_mut(),
        };
        let mut produced: usize = 0;
        let status =
            unsafe { ffi::iree_tokenizer_encode_state_finalize(self.state, output, &mut produced) };
        if ffi::take_error(status).is_some() {
            out.truncate(base);
            return;
        }
        unsafe { out.set_len(out.len() + produced) };
    }
}

#[cfg(iree_available)]
impl Drop for Adapter {
    fn drop(&mut self) {
        unsafe {
            if !self.state.is_null() {
                ffi::iree_tokenizer_encode_state_deinitialize(self.state);
            }
            if !self.tokenizer.is_null() {
                ffi::iree_tokenizer_free(self.tokenizer);
            }
        }
    }
}
