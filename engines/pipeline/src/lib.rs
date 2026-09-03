//! The HuggingFace **rc0 pipeline** — `tk-encode` + `tk-serialize` from
//! [`feat/train_encode_split`](https://github.com/huggingface/tokenizers/tree/feat/train_encode_split),
//! the 1.0.0-rc.0 line.
//!
//! This is the interesting row in the table: it is the same project as the
//! reference engine, reading the same `tokenizer.json`, so the difference
//! between `hf-tokenizers` and `pipeline` is purely the new engine rather than
//! a different vocabulary, a different pre-tokenizer, or a different notion of
//! what a token is. Every other engine in this repository is a comparison
//! across projects; this one is a controlled before/after.
//!
//! Which makes verification matter more here, not less. A rewrite of the merge
//! loop, the pre-tokenizer and now the decoder is exactly the kind of change
//! that can be very fast and subtly wrong on some script, so the `verified`
//! and `decode_verified` flags against `tokenizers 0.23.1` are the whole point
//! — a mismatch on any corpus is a bug report, not a benchmark result.
//!
//! ## Why this engine was repinned, and what it invalidates
//!
//! This adapter used to hold a `tk_encode::Tokenizer` from
//! [#2279](https://github.com/huggingface/tokenizers/pull/2279)'s
//! `poc/target-encode` branch. On rc0 that type does not exist: the legacy
//! encode engine and its serde layer were stripped, and the runtime is
//! `PipelineTokenizer`, built by `tk-serialize` from a canonical config.
//!
//! So this is a different engine, not a version bump, and **every number it
//! reports is re-baselined** — encode as much as decode.
//!
//! The repin also fixes a measurement bug worth naming, because a branch pin
//! caused it. On `poc/target-engine` the pipeline's own decode was a
//! deliberate stub (`Err("PipelineTokenizer::decode is not implemented yet")`),
//! and this adapter reached the *legacy* `Tokenizer::decode` instead. That is
//! the "before" column of
//! [#2305](https://github.com/huggingface/tokenizers/pull/2305), so the decode
//! row published for this engine was the path #2305 replaced, measured twice —
//! once through each adapter — and read as "the new engine decodes no faster".
//! Hence `rev = "..."` in `Cargo.toml` rather than a branch.
//!
//! One thing the repin takes away. This adapter used to report a phase
//! breakdown by timing successive rungs of `encode_generic`'s monomorphised
//! `STAGE` ladder; rc0 has no such ladder, so `phases()` is gone and this
//! engine's `breakdown_nanoseconds` is omitted rather than guessed. The
//! dashboard renders that as "not instrumented", which is the honest reading:
//! the total is still measured, the split is not.
//!
//! ## Why `encode_into`
//!
//! `encode_into` is the allocation-free, offset-free path: no `Encoding`, no
//! `Vec<Encoding>`, no copy out of either. The reference engine is called
//! through `encode`, which also computes byte offsets, word ids and an
//! attention mask. Timing this side's *offset-computing* path against
//! competitors that only produce ids would be the wrong comparison, and timing
//! the reference's ids-only path while calling this one's full path would be
//! the wrong comparison in the other direction.
//!
//! Both are declared honestly instead of quietly equalised: the reference
//! discloses `also_computes: "byte offsets, word ids, attention mask"` and this
//! engine discloses nothing extra, so a reader can see that part of any gap
//! between the two is offset bookkeeping rather than raw encode speed. The ids
//! are identical either way, which is what `verified` checks.
//!
//! One cost is charged to this engine and is *not* an artefact to be excused:
//! `encode_into` fills `Vec<PipelineToken>`, and tokbench compares `Vec<u32>`,
//! so the adapter restates one as the other on every call. `PipelineToken` is
//! `repr(transparent)` over `u32`, so this is a copy and nothing more — but it
//! is a copy the public API forces, the user pays it too, and the fairness
//! contract says it stays in the number.

use tk_encode::pipeline::{PipelineToken, PipelineTokenizer};
use tokbench_core::{unsupported, Build, Class, Engine, Ids, Info, Model, Unsupported};

pub struct Adapter {
    tok: PipelineTokenizer,
    /// Reused across calls, so the timed loop never grows it — the same
    /// buffer-reuse a real encode loop does, and what `encode_into` is for.
    scratch: Vec<PipelineToken>,
}

impl Build for Adapter {
    fn build(model: &Model) -> Result<Box<dyn Engine>, Unsupported> {
        let path = model.tokenizer_json();
        if !path.exists() {
            return Err(Unsupported("no tokenizer.json".into()));
        }
        // Every config in `data/models` is still version 1.0, and rc0's reader
        // is canonical-only (`version: "2.0"`) by design. The upgrade pass is a
        // pure JSON->JSON rewrite and runs here, at load time, outside the
        // timed region — the same thing upstream's own benches do.
        let canonical = tk_convert::canonicalize_file(&path)
            .map_err(|e| Unsupported(format!("tk-convert cannot upgrade this config: {e}")))?;
        let tok = tk_serialize::from_json(&canonical)
            .map_err(|e| Unsupported(format!("tk-serialize cannot read this config: {e}")))?;
        Ok(Box::new(Adapter {
            tok,
            scratch: Vec::new(),
        }))
    }
}

impl Engine for Adapter {
    fn info(&self) -> Info {
        Info {
            name: "pipeline",
            // Not a release: a pinned rev on the rc0 branch. See Cargo.toml.
            version: "tk-encode 1.0.0-rc.0 (feat/train_encode_split @ 0743ac07)",
            lang: "rust",
            class: Class::Native,
            url: "https://github.com/huggingface/tokenizers/tree/feat/train_encode_split",
            also_computes: "",
            internally_parallel: false,
        }
    }

    fn encode(&mut self, text: &str, out: &mut Ids) {
        self.scratch.clear();
        // A cell that fails mid-run must not silently look fast. Leaving `out`
        // short changes the id hash, so verification flags it.
        if self.tok.encode_into(text, false, &mut self.scratch).is_ok() {
            out.extend(self.scratch.iter().map(|t| t.id()));
        }
    }

    /// `PipelineTokenizer::decode` — the path
    /// [#2305](https://github.com/huggingface/tokenizers/pull/2305) added.
    /// For a byte-level BPE model the vocab store already holds each entry's
    /// decoded raw bytes, so this is a concatenation of borrowed slices; other
    /// models still run the configured decoder chain.
    fn decode(&mut self, ids: &[u32], out: &mut String) -> Result<(), Unsupported> {
        match self.tok.decode(ids, false) {
            Ok(s) => {
                out.push_str(&s);
                Ok(())
            }
            Err(e) => unsupported(format!("decode failed: {e}")),
        }
    }
}
