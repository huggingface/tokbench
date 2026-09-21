use tokbench_core::{Build, Engine, Model, Unsupported};

#[cfg(iree_available)]
use tokbench_core::{Class, Ids, Info};

#[cfg(iree_available)]
const URL: &str = "https://github.com/iree-org/iree";

#[cfg(iree_available)]
fn version() -> &'static str {
    option_env!("TOKBENCH_IREE_COMMIT").unwrap_or("not-vendored")
}

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

#[cfg(iree_available)]
mod ffi {
    use std::os::raw::{c_char, c_int, c_void};

    pub type HostSize = usize;

    pub type Status = *mut c_void;

    #[inline]
    pub fn is_ok(s: Status) -> bool {
        s.is_null()
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct StringView {
        pub data: *const c_char,
        pub size: HostSize,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct ByteSpan {
        pub data: *mut u8,
        pub data_length: HostSize,
    }

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

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct OffsetRunList {
        pub capacity: HostSize,
        pub values: *mut c_void,
    }

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

    pub const FLAG_AT_INPUT_START: u32 = 1 << 0;

    unsafe extern "C" {
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

    pub fn system_allocator() -> Allocator {
        Allocator {
            self_: std::ptr::null_mut(),
            ctl: iree_allocator_libc_ctl,
        }
    }

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
    state_storage: Box<[u8]>,
    transform: Box<[u8]>,
    model_type: String,
}

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

    fn ensure_transform(&mut self, text_len: usize) -> bool {
        let want = text_len.saturating_mul(3).max(4096).next_power_of_two();
        if want <= self.transform.len() {
            return true;
        }
        self.transform = vec![0u8; want].into_boxed_slice();
        self.init_state().is_ok()
    }

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
            also_computes: "",
            internally_parallel: false,
        }
    }

    fn encode(&mut self, text: &str, out: &mut Ids) {
        if self.state.is_null() || !self.ensure_transform(text.len()) {
            return;
        }
        unsafe { ffi::iree_tokenizer_encode_state_reset(self.state, ffi::FLAG_AT_INPUT_START) };

        let base = out.len();

        let mut consumed_total = 0usize;
        let mut stalls = 0u32;

        while consumed_total < text.len() {
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
                out.truncate(base);
                return;
            }
            debug_assert!(produced <= spare);
            unsafe { out.set_len(out.len() + produced) };
            consumed_total += consumed;

            if consumed == 0 && produced == 0 {
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
