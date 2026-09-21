use tokbench_core::{Build, Engine, Model, Unsupported};

pub type Ctor = fn(&Model, Option<usize>) -> Result<Box<dyn Engine>, Unsupported>;

/// The correctness oracle. Every other engine's ids are compared against it.
pub const REFERENCE: &str = "hf-tokenizers";

#[allow(unused_mut)]
pub fn native() -> Vec<(&'static str, Ctor)> {
    // Each engine registers twice: as itself, and as `<name>-no-cache`, so a
    // cache's contribution is a measured row rather than a guess.
    // `build_without_cache` defaults to refusing, which reports `unsupported`
    // instead of a number that invites a wrong subtraction.
    macro_rules! engine {
        ($v:ident, $feature:literal, $name:literal, $adapter:path) => {
            #[cfg(feature = $feature)]
            {
                $v.push(($name, <$adapter>::build_with_cache_capacity as Ctor));
                $v.push((
                    concat!($name, "-no-cache"),
                    <$adapter>::build_without_cache_with_capacity as Ctor,
                ));
            }
        };
    }

    let mut v: Vec<(&'static str, Ctor)> = Vec::new();

    engine!(
        v,
        "hf-tokenizers",
        "hf-tokenizers",
        tokbench_hf_tokenizers::Adapter
    );
    engine!(v, "pipeline", "pipeline", tokbench_pipeline::Adapter);
    engine!(v, "kitoken", "kitoken", tokbench_kitoken::Adapter);
    engine!(v, "fastokens", "fastokens", tokbench_fastokens::Adapter);
    engine!(v, "tokie", "tokie", tokbench_tokie::Adapter);
    engine!(v, "tiktoken", "tiktoken", tokbench_tiktoken::Adapter);
    engine!(
        v,
        "rust-gems-bpe",
        "rust-gems-bpe",
        tokbench_rust_gems_bpe::Adapter
    );
    engine!(
        v,
        "wordchipper",
        "wordchipper",
        tokbench_wordchipper::Adapter
    );
    engine!(
        v,
        "sentencepiece",
        "sentencepiece",
        tokbench_sentencepiece::Adapter
    );
    engine!(v, "gigatoken", "gigatoken", tokbench_gigatoken::Adapter);
    engine!(v, "llamacpp", "llamacpp", tokbench_llamacpp::Adapter);
    engine!(v, "iree", "iree", tokbench_iree::Adapter);
    engine!(v, "executorch", "executorch", tokbench_executorch::Adapter);

    v
}
