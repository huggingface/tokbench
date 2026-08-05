//! llama.cpp's built-in tokenizer.
//!
//! STATUS: scaffolded, not yet wired.
//!
//! The relevant C API is small and stable:
//!
//! ```c
//! struct llama_model* llama_model_load_from_file(const char*, struct llama_model_params);
//! const struct llama_vocab* llama_model_get_vocab(const struct llama_model*);
//! int32_t llama_tokenize(const struct llama_vocab* vocab,
//!                        const char* text, int32_t text_len,
//!                        llama_token* tokens, int32_t n_tokens_max,
//!                        bool add_special, bool parse_special);
//! ```
//!
//! Call it with `add_special = false, parse_special = false` to match the
//! reference's `add_special_tokens = false`.
//!
//! Two constraints specific to this engine:
//!
//! * **It needs a GGUF, not `tokenizer.json`.** llama.cpp reads its vocabulary
//!   out of the model file. `make models` must place a `model.gguf` (a
//!   vocab-only GGUF is enough — `llama.cpp/convert_hf_to_gguf.py --vocab-only`
//!   produces one from the same upstream model) or this engine stays
//!   `Unsupported`. Loading a *different* model's vocabulary would produce a
//!   fast row that is silently answering a different question.
//! * **Link only the tokenizer.** Prefer a bindgen shim over `llama.h` with
//!   `GGML_*` backends off; `llama-cpp-2` with default features drags in the
//!   full inference stack, which lengthens builds without changing the number.

use tokbench_core::{Build, Engine, Model, Unsupported};

pub struct Adapter;

impl Build for Adapter {
    fn build(_model: &Model) -> Result<Box<dyn Engine>, Unsupported> {
        Err(Unsupported(
            "not wired: needs a vocab-only model.gguf + llama_tokenize shim; \
             see engines/llamacpp/src/lib.rs"
                .into(),
        ))
    }
}
