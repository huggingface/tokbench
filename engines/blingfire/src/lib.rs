//! Microsoft BlingFire, through `TextToIds` in `libblingfiretokdll`.
//!
//! # Why this engine does not use the `blingfire` crate
//!
//! The obvious move is `blingfire` 1.0.0 on crates.io, and it is the wrong
//! one. It exposes exactly `text_to_words` and `text_to_sentences`, both of
//! which return a *string* of space-separated surface forms. Neither produces
//! token ids, so neither can be compared against any other engine here.
//! Benchmarking `text_to_words` and labelling the row "BlingFire" would put a
//! whitespace segmenter in a table of subword tokenizers, where it would post
//! an enormous throughput number for a fundamentally cheaper job — the single
//! most misleading thing this repository could publish. The harness would not
//! even catch it: `ids_hash` would differ, the cell would read `mismatch`, and
//! a reader skimming the throughput column would still walk away with the
//! wrong impression.
//!
//! The comparable entry point is `TextToIds`, which BlingFire ships in its C
//! library but no Rust crate binds. So this adapter links that library
//! directly and declares the three functions it needs.
//!
//! # Getting the library
//!
//! `scripts/vendor_blingfire.sh`. Microsoft commit prebuilt binaries into the
//! BlingFire repo (`dist-pypi/blingfire/libblingfiretokdll.{so,dylib}`) and
//! ship the same files in the PyPI wheel, which is the fast path — but those
//! binaries are **x86_64 only** (`lipo -archs` on the dylib prints `x86_64`
//! and nothing else), so on Apple silicon the script falls back to a CMake
//! build of the `blingfiretokdll` target. It builds only that target, which
//! links `fsaClient` alone, so BlingFire's FSA compiler and its ~20 tools are
//! never compiled.
//!
//! If the library is absent the crate still compiles and `build()` returns
//! [`Unsupported`]; see `build.rs`.
//!
//! # Getting a model, and why the matrix is nearly empty
//!
//! BlingFire does not read `tokenizer.json`. It reads its own compiled FSA
//! image, a `.bin` produced by its offline toolchain, and that image carries
//! the vocabulary *and* the id assignment. That turns model selection into a
//! correctness question. Pointing this engine at a `.bin` whose vocabulary
//! differs from the `tokenizer.json` every other engine loaded would produce a
//! fast row answering a different question — the fairness contract's rule 1,
//! violated at the point of construction.
//!
//! Microsoft ship a good number of prebuilt `.bin` models, and exactly one of
//! them corresponds to a model in `data/models/`:
//!
//! | `data/models/` | BlingFire `.bin`  | verdict |
//! |----------------|-------------------|---------|
//! | `gpt2`         | `gpt2.bin`        | same vocabulary — the one real pairing |
//! | `bert-wiki`    | `bert_base_tok.bin` | **no.** `bert_base_tok.bin` is bert-base-uncased (`[PAD]`=0, `[CLS]`=101, `the`=1996). `data/models/bert-wiki` is a wiki-trained WordPiece with its own id space (`[UNK]`=0, `[CLS]`=1, `[SEP]`=2, `[PAD]`=3, `the`=7108). Same algorithm and same 30522 entries, different vocabulary. |
//! | `albert`       | `xlnet.bin`       | **no.** Unigram/30000 against SentencePiece/32000 — different models. |
//! | `llama-2`, `llama-3`, `deepseek-v4`, `mistral-nemo` | — | BlingFire ships nothing for these. |
//!
//! Everything but `gpt2` is therefore [`Unsupported`], with the reason stated
//! rather than left blank. That is the honest shape of this row, and a sparse
//! matrix with reasons beats a full one built on mismatched vocabularies.
//!
//! The vendor script installs `gpt2.bin` as `data/models/gpt2/blingfire.bin`.
//! To try another pairing, drop its `.bin` in as `<model>/blingfire.bin` and
//! extend the table in `Adapter::build`; the harness will report `mismatch`
//! against the reference if the vocabularies do not in fact agree.
//!
//! # The gpt2 cell does not match the reference, and the reason is structural
//!
//! It is the right vocabulary — `gpt2.bin` and `tokenizer.json` agree on
//! ordinary words token for token — but the id streams still differ, and the
//! difference is not a rounding error to be tuned away. **BlingFire's GPT-2
//! model does not emit whitespace tokens at all.**
//!
//! GPT-2's byte-level BPE represents every byte, so a run of spaces, a tab or
//! a newline each become tokens of their own (`220` = `Ġ`, `197` = `ĉ`,
//! `198` = `Ċ`). BlingFire's compiled image runs its own word-breaker first,
//! which treats a whitespace run as a boundary and then discards it. It has no
//! way to encode the run itself. Measured against the reference over the first
//! 20 chunks of each corpus, with the dummy prefix already switched off:
//!
//! | corpus  | reference ids | BlingFire ids | reference whitespace-only ids |
//! |---------|---------------|---------------|-------------------------------|
//! | english | 38 468        | 37 767        | 701                           |
//! | code    | 103 680       | 76 965        | 26 715                        |
//!
//! In both cases BlingFire's count equals the reference's count with every
//! whitespace-only token removed — exactly, not approximately. On `code` that
//! is 25.8% of the reference's output that BlingFire never produces.
//!
//! Two consequences, and the second is the one that matters for reading the
//! report:
//!
//! 1. The output is lossy. `IdsToText` on BlingFire's ids cannot reconstruct
//!    the input, because the whitespace is simply gone. GPT-2's tokenizer is
//!    a bijection on bytes; this is not.
//! 2. **It is doing less work, so its throughput is not comparable.** Fewer
//!    tokens out of the same bytes means fewer merges, and on whitespace-heavy
//!    input like source code that is a quarter of the job removed. A reader
//!    who took the MB/s at face value would be comparing a tokenizer against
//!    something doing three-quarters of the same task.
//!
//! This is why the cell is left to report `mismatch` rather than being quietly
//! made to agree. Fairness rule 1 exists for precisely this: the hash differs,
//! the cell is never presented as a speedup, and the speedup column skips it.
//! The number is still worth having — it is a real measurement of a real
//! library on a real vocabulary — but it belongs behind that label.
//!
//! Non-Latin scripts diverge further still: on `russian` and `chinese` only
//! 4% and 2% of positions agree even after whitespace is stripped from the
//! reference, so the disagreement there is not a whitespace story at all.
//!
//! # Measurement notes
//!
//! * [`Class::Cffi`]: in-process, same clock as the native engines, with the
//!   FFI call included.
//! * `also_computes` is `""`. BlingFire *can* return offsets, via a separate
//!   `TextToIdsWithOffsets` entry point; this adapter calls plain `TextToIds`,
//!   so it does ids and only ids.
//! * `UnkId`: taken from the `unk` token of the model's `tokenizer.json` when
//!   there is one, else 0. It is a `TextToIds` argument rather than a property
//!   of the `.bin`, so getting it wrong would silently change the id stream on
//!   every out-of-vocabulary input.
//! * Single-threaded. `TextToIds` does its own work on the calling thread, and
//!   the scaling sweep gives each thread its own `LoadModel` handle, so there
//!   is no shared state to contend on.

use tokbench_core::{Build, Engine, Model, Unsupported};

// ---------------------------------------------------------------------------
// Wired: the library was found at build time.
// ---------------------------------------------------------------------------

#[cfg(blingfire_linked)]
mod wired {
    use super::*;
    use std::os::raw::{c_char, c_int, c_void};
    use std::path::Path;
    use tokbench_core::{Class, Ids, Info};

    // The three entry points, transcribed from
    // `blingfiretools/blingfiretokdll/blingfiretokdll.h` (vendored to
    // `vendor/include/` by the script, so the transcription can be rechecked
    // without re-cloning).
    //
    // The C++ header declares these inside `extern "C"`, and the return types
    // are written `const int` there — a top-level `const` on a by-value return
    // is meaningless to the ABI, so `c_int` is the faithful mapping.
    // `MaxIdsArrLength` and `UnkId` have C++ default arguments upstream;
    // defaults are a compile-time source convenience with no ABI presence, so
    // both must be passed explicitly here.
    extern "C" {
        // Loads a compiled `.bin` and returns an opaque handle, or null.
        fn LoadModel(pszLdbFileName: *const c_char) -> *mut c_void;

        // Tokenizes into `pIdsArr` and returns the id count.
        //
        // The return value is the number of ids the input *produced*, which
        // may exceed `MaxIdsArrLength`; in that case the buffer holds only the
        // first `MaxIdsArrLength` of them. Comparing the two is the only way
        // to detect truncation, and this adapter does compare them.
        fn TextToIds(
            ModelPtr: *mut c_void,
            pInUtf8Str: *const c_char,
            InUtf8StrByteCount: c_int,
            pIdsArr: *mut i32,
            MaxIdsArrLength: c_int,
            UnkId: c_int,
        ) -> c_int;

        // Turns off the leading-space ("dummy prefix") that BlingFire's
        // SentencePiece-style path prepends to every input. Returns 1, or 0
        // for a null handle. See `Adapter::load` for why this is switched on.
        fn SetNoDummyPrefix(ModelPtr: *mut c_void, fNoDummyPrefix: bool) -> c_int;

        fn FreeModel(ModelPtr: *mut c_void) -> c_int;
    }

    pub struct Adapter {
        handle: *mut c_void,
        unk_id: c_int,
        /// Reused across calls, exactly as the harness's `out` buffer is.
        ///
        /// `TextToIds` writes into caller memory, so an adapter that allocated
        /// per call would be charging BlingFire for an allocation its API does
        /// not require — the mirror image of the `Vec`-returning engines,
        /// which are charged for theirs because theirs is unavoidable. The
        /// contract in `core` is explicit that `out` arrives cleared with its
        /// capacity intact; this buffer is the same idea one level down.
        scratch: Vec<i32>,
    }

    /// The handle is a private, per-instance model image. `LoadModel` maps a
    /// fresh one per call and `TextToIds` only reads it, so moving an
    /// `Adapter` to another thread is sound. It is deliberately NOT `Sync`:
    /// the scaling sweep builds one engine per thread rather than sharing one,
    /// so nothing here needs to be shared by reference.
    unsafe impl Send for Adapter {}

    impl Drop for Adapter {
        fn drop(&mut self) {
            if !self.handle.is_null() {
                unsafe { FreeModel(self.handle) };
            }
        }
    }

    impl Adapter {
        pub fn load(bin: &Path, unk_id: i32) -> Result<Box<dyn Engine>, Unsupported> {
            let c = std::ffi::CString::new(bin.as_os_str().as_encoded_bytes())
                .map_err(|_| Unsupported("model path contains an interior NUL".into()))?;

            // Loading is model parsing and FSA mapping: it belongs here, before
            // the timer, and is reported as load_ms (fairness rule 3).
            let handle = unsafe { LoadModel(c.as_ptr()) };
            if handle.is_null() {
                return Err(Unsupported(format!(
                    "LoadModel returned null for {} — the file is not a BlingFire \
                     model image, or was built by an incompatible version",
                    bin.display()
                )));
            }

            // Match the reference's convention rather than BlingFire's
            // default. Left on, BlingFire prepends a space to every input the
            // way SentencePiece's add_dummy_prefix does, so "The quick brown
            // fox" comes back as [383, ...] (" The") where GPT-2 gives
            // [464, ...] ("The"). That is an API default, not a property of
            // the vocabulary, and the reference is called with no such prefix
            // — so leaving it on would manufacture a disagreement on the first
            // token of every chunk. Measured: switching it off turns 4 of 8
            // short probe strings from differing to byte-identical.
            //
            // This is the same kind of adjustment as kitoken's
            // `encode(text, false)` for special tokens: putting the library in
            // the configuration the comparison is actually about.
            if unsafe { SetNoDummyPrefix(handle, true) } == 0 {
                unsafe { FreeModel(handle) };
                return Err(Unsupported(
                    "SetNoDummyPrefix rejected the model handle".into(),
                ));
            }

            Ok(Box::new(Adapter {
                handle,
                unk_id,
                scratch: Vec::new(),
            }))
        }
    }

    impl Engine for Adapter {
        fn info(&self) -> Info {
            Info {
                name: "blingfire",
                // Recorded by build.rs from the vendor script's PROVENANCE, so
                // this names the build that was actually linked.
                version: env!("TOKBENCH_BLINGFIRE_VERSION"),
                lang: "c++",
                class: Class::Cffi,
                url: "https://github.com/microsoft/BlingFire",
                // Ids only. Offsets are a different entry point
                // (TextToIdsWithOffsets) and are not called.
                also_computes: "",
                internally_parallel: false,
            }
        }

        fn encode(&mut self, text: &str, out: &mut Ids) {
            if text.is_empty() {
                return;
            }

            // One id per input byte is a hard upper bound for any of these
            // models: the worst case is every byte becoming its own token, and
            // no merge or piece can produce more ids than that. Sizing to it
            // once means the truncation branch below is unreachable in
            // practice while remaining a real check rather than an assumption.
            if self.scratch.len() < text.len() {
                self.scratch.resize(text.len(), 0);
            }

            let n = unsafe {
                TextToIds(
                    self.handle,
                    text.as_ptr() as *const c_char,
                    text.len() as c_int,
                    self.scratch.as_mut_ptr(),
                    self.scratch.len() as c_int,
                    self.unk_id,
                )
            };

            if n <= 0 {
                return;
            }

            // n is the count the input *produced*, not the count written. If it
            // exceeds the buffer, only the buffer's worth is valid — grow and
            // redo rather than read uninitialised tail, and never silently
            // emit a truncated id stream, which would corrupt the hash the
            // whole verification rests on.
            let n = n as usize;
            if n > self.scratch.len() {
                self.scratch.resize(n, 0);
                let again = unsafe {
                    TextToIds(
                        self.handle,
                        text.as_ptr() as *const c_char,
                        text.len() as c_int,
                        self.scratch.as_mut_ptr(),
                        self.scratch.len() as c_int,
                        self.unk_id,
                    )
                };
                if again <= 0 || again as usize > self.scratch.len() {
                    return;
                }
                out.extend(self.scratch[..again as usize].iter().map(|&v| v as u32));
                return;
            }

            // i32 -> u32 to match the harness's id type. The copy is real work
            // the caller pays, so it stays inside the timed region.
            out.extend(self.scratch[..n].iter().map(|&v| v as u32));
        }
    }
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

pub struct Adapter;

impl Build for Adapter {
    #[cfg(blingfire_linked)]
    fn build(model: &Model) -> Result<Box<dyn Engine>, Unsupported> {
        let Some(bin) = model.artifact("blingfire.bin") else {
            return Err(Unsupported(format!(
                "no blingfire.bin in {}. BlingFire reads its own compiled FSA image, \
                 not tokenizer.json, and ships a .bin matching only `gpt2` of the \
                 models here (its bert/xlnet images are different vocabularies — see \
                 engines/blingfire/src/lib.rs). Run \
                 engines/blingfire/scripts/vendor_blingfire.sh to install the ones \
                 that do match.",
                model.dir.display()
            )));
        };

        wired::Adapter::load(&bin, unk_id(model))
    }

    #[cfg(not(blingfire_linked))]
    fn build(_model: &Model) -> Result<Box<dyn Engine>, Unsupported> {
        Err(Unsupported(
            "libblingfiretokdll not linked. BlingFire's id-producing API (TextToIds) \
             lives in a C library that no Rust crate binds and cargo cannot fetch; \
             the `blingfire` crate only segments words, which is not comparable. \
             Run engines/blingfire/scripts/vendor_blingfire.sh, or set \
             BLINGFIRE_LIB_DIR to a directory holding libblingfiretokdll, then \
             rebuild."
                .into(),
        ))
    }
}

/// The id `TextToIds` should emit for out-of-vocabulary input.
///
/// This is an *argument* to `TextToIds`, not something the `.bin` carries, so
/// the default of 0 is only right when 0 really is the unk id. Read it out of
/// the same `tokenizer.json` the reference loads, so the two agree by
/// construction instead of by luck.
///
/// Parsed by hand rather than with serde: this crate has one dependency
/// (`tokbench-core`) and a whole JSON stack to read a single integer would
/// show up in the binary-size side-channel that `binsize/` measures, which
/// would misattribute the adapter's cost to BlingFire.
#[cfg(blingfire_linked)]
fn unk_id(model: &Model) -> i32 {
    let Ok(s) = std::fs::read_to_string(model.tokenizer_json()) else {
        return 0;
    };
    // "unk_token": "<name>"  →  find that token's id in "vocab".
    let Some(unk) = json_string_field(&s, "unk_token") else {
        return 0;
    };
    let Some(v) = s.find("\"vocab\"") else {
        return 0;
    };
    let needle = format!("{:?}:", unk);
    s[v..]
        .find(&needle)
        .and_then(|i| {
            let rest = &s[v + i + needle.len()..];
            let digits: String = rest
                .trim_start()
                .chars()
                .take_while(char::is_ascii_digit)
                .collect();
            digits.parse::<i32>().ok()
        })
        .unwrap_or(0)
}

/// `"key": "value"` → `value`, for the one string field above. Returns `None`
/// for a null or absent key, which is the common case (byte-level BPE has no
/// unk token at all).
#[cfg(blingfire_linked)]
fn json_string_field(s: &str, key: &str) -> Option<String> {
    let at = s.find(&format!("\"{key}\""))?;
    let rest = s[at..].split_once(':')?.1.trim_start();
    let inner = rest.strip_prefix('"')?;
    let end = inner.find('"')?;
    Some(inner[..end].to_string())
}
