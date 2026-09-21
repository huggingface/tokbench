#[cfg(not(llamacpp))]
use tokbench_core::{Build, Engine, Model, Unsupported};

const VERSION: &str = env!("TOKBENCH_LLAMACPP_VERSION");

#[cfg(llamacpp)]
mod wired {
    use super::VERSION;
    use std::ffi::{c_char, c_int, CString};
    use std::path::Path;
    use tokbench_core::{Build, Class, Engine, Ids, Info, Model, Unsupported};

    #[repr(C)]
    pub struct LlamaModel {
        _private: [u8; 0],
    }
    #[repr(C)]
    pub struct LlamaVocab {
        _private: [u8; 0],
    }

    unsafe extern "C" {
        fn tokbench_llama_load_vocab(path: *const c_char) -> *mut LlamaModel;
        fn tokbench_llama_silence();

        fn llama_model_get_vocab(model: *const LlamaModel) -> *const LlamaVocab;
        fn llama_model_free(model: *mut LlamaModel);
        fn llama_tokenize(
            vocab: *const LlamaVocab,
            text: *const c_char,
            text_len: c_int,
            tokens: *mut i32,
            n_tokens_max: c_int,
            add_special: bool,
            parse_special: bool,
        ) -> c_int;
    }

    pub struct Adapter {
        model: *mut LlamaModel,
        vocab: *const LlamaVocab,
        scratch: Vec<i32>,
    }

    unsafe impl Send for Adapter {}

    impl Drop for Adapter {
        fn drop(&mut self) {
            unsafe { llama_model_free(self.model) };
        }
    }

    impl Build for Adapter {
        fn build(model: &Model) -> Result<Box<dyn Engine>, Unsupported> {
            let Some(path) = model.artifact("model.gguf") else {
                return Err(Unsupported(format!(
                    "no model.gguf (llama.cpp reads its vocabulary from a GGUF, \
                     not tokenizer.json); generate one from this model's own \
                     tokenizer.json with `python engines/llamacpp/scripts/make_gguf.py {}`",
                    model.name
                )));
            };
            let c_path = cstring(&path)?;

            unsafe { tokbench_llama_silence() };

            let handle = unsafe { tokbench_llama_load_vocab(c_path.as_ptr()) };
            if handle.is_null() {
                return Err(Unsupported(format!(
                    "llama.cpp could not load {} (unsupported tokenizer type, or \
                     a GGUF it does not recognise)",
                    path.display()
                )));
            }

            let vocab = unsafe { llama_model_get_vocab(handle) };
            if vocab.is_null() {
                unsafe { llama_model_free(handle) };
                return Err(Unsupported(format!(
                    "{} carries no vocabulary",
                    path.display()
                )));
            }

            Ok(Box::new(Adapter {
                model: handle,
                vocab,
                scratch: Vec::new(),
            }))
        }
    }

    impl Engine for Adapter {
        fn info(&self) -> Info {
            Info {
                name: "llamacpp",
                version: VERSION,
                lang: "c++",
                class: Class::Cffi,
                url: "https://github.com/ggml-org/llama.cpp",
                also_computes: "",
                internally_parallel: false,
            }
        }

        fn encode(&mut self, text: &str, out: &mut Ids) {
            let needed = text.len() + 1;
            if self.scratch.len() < needed {
                self.scratch.resize(needed, 0);
            }

            let mut n = unsafe {
                llama_tokenize(
                    self.vocab,
                    text.as_ptr() as *const c_char,
                    text.len() as c_int,
                    self.scratch.as_mut_ptr(),
                    self.scratch.len() as c_int,
                    false,
                    false,
                )
            };

            if n < 0 {
                let Ok(required) = usize::try_from(-(n as i64)) else {
                    return;
                };
                self.scratch.resize(required, 0);
                n = unsafe {
                    llama_tokenize(
                        self.vocab,
                        text.as_ptr() as *const c_char,
                        text.len() as c_int,
                        self.scratch.as_mut_ptr(),
                        self.scratch.len() as c_int,
                        false,
                        false,
                    )
                };
                if n < 0 {
                    return;
                }
            }

            out.extend(self.scratch[..n as usize].iter().map(|&id| id as u32));
        }
    }

    fn cstring(path: &Path) -> Result<CString, Unsupported> {
        CString::new(path.to_string_lossy().into_owned())
            .map_err(|_| Unsupported("model path contains a NUL byte".into()))
    }
}

#[cfg(llamacpp)]
pub use wired::Adapter;

#[cfg(not(llamacpp))]
pub struct Adapter;

#[cfg(not(llamacpp))]
impl Build for Adapter {
    fn build(_model: &Model) -> Result<Box<dyn Engine>, Unsupported> {
        let _ = VERSION;
        Err(Unsupported(
            "llama.cpp not found at build time: install it (`brew install \
             llama.cpp`, or a distro package providing llama.h + libllama) or \
             point LLAMA_CPP_DIR at a built checkout, then rebuild"
                .into(),
        ))
    }
}
