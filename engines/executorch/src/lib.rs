#![cfg_attr(not(have_executorch), allow(dead_code))]

use tokbench_core::{Build, Engine, Model, Unsupported};

const NAME: &str = "executorch";
const URL: &str = "https://github.com/pytorch-labs/tokenizers";

const VERSION: &str = match option_env!("TOKBENCH_EXECUTORCH_VERSION") {
    Some(v) => v,
    None => "HFTokenizer (not vendored)",
};

#[cfg(have_executorch)]
mod wired {
    use super::*;
    use std::ffi::{c_char, c_void, CString};
    use tokbench_core::{Class, Ids, Info};

    extern "C" {
        fn tokbench_et_create(path: *const c_char) -> *mut c_void;

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

    unsafe impl Send for Adapter {}

    impl Adapter {
        pub fn open(model: &Model) -> Result<Box<dyn Engine>, Unsupported> {
            let path = model.tokenizer_json();
            if !path.exists() {
                return Err(Unsupported("no tokenizer.json".into()));
            }
            let c = CString::new(path.as_os_str().as_encoded_bytes())
                .map_err(|_| Unsupported("tokenizer.json path contains a NUL".into()))?;

            let handle = unsafe { tokbench_et_create(c.as_ptr()) };
            if handle.is_null() {
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
            out.reserve(text.len());
            let cap = out.capacity();

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
                return;
            }
            let n = n as usize;
            if n <= cap {
                unsafe { out.set_len(n) };
                return;
            }

            out.reserve(n);
            let cap = out.capacity();
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
                unsafe { out.set_len(n as usize) };
            }
        }
    }
}

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
