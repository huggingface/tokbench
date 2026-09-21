#[cfg(gigatoken_wired)]
mod wired {
    use gigatoken_rs::load_tokenizer::hf::{load_hf_slice, HfTokenizer};
    use gigatoken_rs::EncodeState;
    use tokbench_core::{Build, Class, Engine, Ids, Info, Model, Padding, Unsupported};

    pub struct Adapter {
        workers: gigatoken_rs::WorkerPool,
        pool: Option<rayon::ThreadPool>,
        threads: usize,
        tok: HfTokenizer,
        sp_state: EncodeState,
    }

    impl Build for Adapter {
        fn build_without_cache(_model: &Model) -> Result<Box<dyn Engine>, Unsupported> {
            Err(Unsupported(
                "gigatoken's pretoken cache is pub(crate) with no bypass; disabling it needs a \
                 patched build of the crate"
                    .into(),
            ))
        }

        fn build(model: &Model) -> Result<Box<dyn Engine>, Unsupported> {
            let path = model.tokenizer_json();
            if !path.exists() {
                return Err(Unsupported(format!(
                    "no tokenizer.json in {}",
                    model.dir.display()
                )));
            }
            let data = std::fs::read(&path)
                .map_err(|e| Unsupported(format!("cannot read {}: {e}", path.display())))?;
            let tok = load_hf_slice(&data)
                .map_err(|e| Unsupported(format!("gigatoken cannot load this config: {e}")))?;
            Ok(Box::new(Adapter {
                workers: gigatoken_rs::WorkerPool::new(),
                pool: None,
                threads: 1,
                tok,
                sp_state: EncodeState::new(),
            }))
        }
    }

    impl Engine for Adapter {
        fn info(&self) -> Info {
            Info {
                name: "gigatoken",
                version: "0.10.0 (git 34a1599f0c0ae7d7cd0d1c530e6522320158b360)",
                lang: "rust",
                class: Class::Native,
                url: "https://github.com/marcelroed/gigatoken",
                also_computes: "",
                internally_parallel: self.threads > 1,
            }
        }

        fn encode(&mut self, text: &str, out: &mut Ids) {
            let Self { tok, sp_state, .. } = self;
            match tok {
                HfTokenizer::Bpe(bpe) => bpe.encode_with_added_tokens_flat(text.as_bytes(), out),
                HfTokenizer::SentencePiece(sp) => {
                    sp.encode_raw_cb(sp_state, text, &mut |ids| {
                        out.extend(ids.iter().copied().map(u32::from));
                    });
                }
            }
        }

        fn has_native_batch(&self) -> bool {
            true
        }

        fn set_threads(&mut self, threads: usize) -> bool {
            if threads == 0 {
                return false;
            }
            self.pool = match rayon::ThreadPoolBuilder::new().num_threads(threads).build() {
                Ok(pool) => Some(pool),
                Err(_) => return false,
            };
            self.threads = threads;
            true
        }

        fn encode_batch(&mut self, texts: &[&str], out: &mut Ids) {
            let workers = &self.workers;
            let run = || match &self.tok {
                HfTokenizer::Bpe(bpe) => {
                    let docs: Vec<&[u8]> = texts.iter().map(|t| t.as_bytes()).collect();
                    gigatoken_rs::encode_docs_ragged(workers, bpe, &docs)
                }
                HfTokenizer::SentencePiece(sp) => gigatoken_rs::sp_encode_docs_ragged(sp, texts),
            };
            let (flat, _row_lengths) = match &self.pool {
                Some(pool) => pool.install(run),
                None => run(),
            };
            out.extend_from_slice(&flat);
        }

        fn set_padding(&mut self, padding: Padding) -> bool {
            matches!(padding, Padding::Off)
        }
    }
}

#[cfg(gigatoken_wired)]
pub use wired::Adapter;

#[cfg(not(gigatoken_wired))]
pub struct Adapter;

#[cfg(not(gigatoken_wired))]
impl tokbench_core::Build for Adapter {
    fn build(
        _model: &tokbench_core::Model,
    ) -> Result<Box<dyn tokbench_core::Engine>, tokbench_core::Unsupported> {
        Err(tokbench_core::Unsupported(
            "gigatoken not wired: its manifest declares `cargo-features`, which Cargo \
             rejects in a dependency and which breaks resolution for the whole \
             workspace. Run engines/gigatoken/scripts/enable_gigatoken.sh and build \
             with RUSTFLAGS='--cfg gigatoken_wired' on nightly."
                .into(),
        ))
    }
}
