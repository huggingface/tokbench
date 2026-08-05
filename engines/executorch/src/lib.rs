//! ExecuTorch's tokenizers (`pytorch-labs/tokenizers`) — the C++ tokenizer
//! library used by ExecuTorch and torchchat runners.
//!
//! STATUS: scaffolded, not yet wired.
//!
//! The library is a family of classes rather than one tokenizer:
//! `HFTokenizer` (reads `tokenizer.json`), `Tiktoken`, `SPTokenizer`
//! (delegates to libsentencepiece), `Llama2cTokenizer`, and Tekken. They share
//! a `Tokenizer` base with roughly:
//!
//! ```cpp
//! Error  load(const std::string& tokenizer_path);
//! Result<std::vector<uint64_t>> encode(const std::string& input,
//!                                      int8_t bos, int8_t eos) const;
//! ```
//!
//! Wiring notes:
//!
//! * **Pick `HFTokenizer` and say so.** It is the variant that consumes the
//!   same `tokenizer.json` as the reference, so it is the one that can be
//!   verified id-for-id. Put the variant in `Info::version` (e.g.
//!   `"HFTokenizer @ <rev>"`) — "executorch" alone does not identify what ran,
//!   and `SPTokenizer` would really be measuring libsentencepiece, which
//!   already has its own row here.
//! * Call with `bos = 0, eos = 0` to match `add_special_tokens = false`.
//! * `encode` returns an owned `std::vector<uint64_t>`; the shim should copy
//!   into the caller's `out` and narrow to `u32`. That copy is part of the
//!   API's real cost and stays inside the timed region, exactly as the
//!   `Vec`-returning Rust engines are charged for theirs.
//! * A three-function C shim (`create/encode/destroy`) over the C++ object is
//!   enough; compile it with `cc` in a `build.rs` and link `libtokenizers`.

use tokbench_core::{Build, Engine, Model, Unsupported};

pub struct Adapter;

impl Build for Adapter {
    fn build(_model: &Model) -> Result<Box<dyn Engine>, Unsupported> {
        Err(Unsupported(
            "not wired: needs a C shim over pytorch-labs HFTokenizer; \
             see engines/executorch/src/lib.rs"
                .into(),
        ))
    }
}
