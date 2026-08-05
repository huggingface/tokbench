/*
 * A three-function C ABI over `tokenizers::HFTokenizer`, so the Rust adapter
 * can call the C++ object in-process (tokbench_core::Class::Cffi).
 *
 * The design rule for everything below: do not do the library's work for it,
 * and do not do the caller's work either. This file exists to cross the
 * language boundary and nothing else. In particular it must not pre-size,
 * cache or reuse anything that a C++ user of `HFTokenizer` would not, because
 * that would flatter the engine relative to the Rust engines measured through
 * the same harness.
 *
 * Upstream API being wrapped (include/pytorch/tokenizers/tokenizer.h):
 *
 *     virtual Error load(const std::string& tokenizer_path);
 *     virtual Result<std::vector<uint64_t>>
 *         encode(const std::string& input, int8_t bos = 0, int8_t eos = 0) const;
 *
 * `Result<T>` is upstream's own outcome type: `.ok()` then `.get()`. It has no
 * copy constructor and asserts on `.get()` when empty, so it is checked before
 * it is touched.
 */

#include <pytorch/tokenizers/error.h>
#include <pytorch/tokenizers/hf_tokenizer.h>
#include <pytorch/tokenizers/result.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace {

/*
 * The handle. It owns the tokenizer plus one scratch `std::string`.
 *
 * The scratch string is the honest way to bridge `&str` -> `const
 * std::string&`. `HFTokenizer::encode` takes its input by const reference to
 * `std::string`, so a caller holding a pointer+length — which is what Rust
 * has, and what any C++ caller reading from a buffer has — must materialise a
 * `std::string`. There is no `string_view` overload to escape through; that
 * copy is a real, unavoidable cost of this library's public API and it stays
 * inside the timed region.
 *
 * `assign()` rather than a fresh `std::string` per call because the buffer is
 * reused across the encode loop, so after the first document the copy is a
 * memcpy into already-owned capacity with no allocation. That mirrors exactly
 * what the harness does for the Rust engines: `core`'s `measure` hands every
 * engine an `out: &mut Ids` that was cleared but kept its capacity. Allocating
 * a new string per call would charge this engine for a malloc the Rust engines
 * are explicitly not charged for.
 */
struct Handle {
  tokenizers::HFTokenizer tok;
  std::string scratch;
};

} // namespace

extern "C" {

/*
 * Load `tokenizer.json`. Returns nullptr if the file is missing or if
 * HFTokenizer rejects the config (it does not implement every HuggingFace
 * normalizer/pre-tokenizer, and saying so is a legitimate Unsupported).
 *
 * Outside the timed region — the harness calls this from `Build::build`.
 */
void* tokbench_et_create(const char* path) {
  if (path == nullptr) {
    return nullptr;
  }
  auto* h = new (std::nothrow) Handle();
  if (h == nullptr) {
    return nullptr;
  }
  // `load` is declared to return Error, but the JSON parsing underneath can
  // throw on a malformed file; a C ABI must not let an exception cross it.
  tokenizers::Error err;
  try {
    err = h->tok.load(std::string(path));
  } catch (...) {
    delete h;
    return nullptr;
  }
  if (err != tokenizers::Error::Ok || !h->tok.is_loaded()) {
    delete h;
    return nullptr;
  }
  return h;
}

/*
 * Encode `text[0..len]` into `out`, which has room for `cap` ids.
 *
 * Returns the number of ids the encode produced, or -1 on failure. When the
 * return value exceeds `cap` nothing was written and the caller should grow
 * and retry — the Rust side sizes `out` to `text.len()`, which is a hard upper
 * bound (no subword tokenizer emits more tokens than input bytes), so this
 * path is a safety net rather than a normal occurrence.
 *
 * bos = 0, eos = 0 is deliberate and load-bearing. In hf_tokenizer.cpp:
 *
 *     bool add_special = (bos > 0 || eos > 0);
 *     if (_postprocessor) tokens = _postprocessor->process(tokens, add_special);
 *
 * so passing 0/0 is what makes the post-processor skip special tokens, which
 * is the `add_special_tokens = false` the reference engine is called with. Any
 * other value would add BOS/EOS ids and every cell would report a mismatch.
 *
 * The uint64 -> uint32 narrowing loop is inside the timed region on purpose.
 * `encode` hands back an owned `std::vector<uint64_t>` and a caller who wants
 * ids in their own buffer has to walk it; the Rust engines pay the identical
 * cost via `out.extend(...)` in their adapters (see engines/kitoken). Hoisting
 * it out would be measuring a different program than the one a user runs.
 */
int64_t tokbench_et_encode(
    void* handle,
    const char* text,
    size_t len,
    uint32_t* out,
    size_t cap) {
  auto* h = static_cast<Handle*>(handle);
  if (h == nullptr) {
    return -1;
  }
  try {
    h->scratch.assign(text, len);
    auto res = h->tok.encode(h->scratch, /*bos=*/0, /*eos=*/0);
    if (!res.ok()) {
      return -1;
    }
    const std::vector<uint64_t>& ids = res.get();
    const size_t n = ids.size();
    if (n > cap) {
      return static_cast<int64_t>(n); // caller grows and retries
    }
    for (size_t i = 0; i < n; ++i) {
      out[i] = static_cast<uint32_t>(ids[i]);
    }
    return static_cast<int64_t>(n);
  } catch (...) {
    return -1;
  }
}

void tokbench_et_destroy(void* handle) {
  delete static_cast<Handle*>(handle);
}

} // extern "C"
