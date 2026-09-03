// The only C we need: the two calls that pass a struct by value.
//
// `llama_model_params` is a 72-byte struct with function pointers, an enum, a
// float array and several bools, and llama.cpp reserves the right to grow it.
// Re-declaring it in Rust to call `llama_model_load_from_file` would be an ABI
// bet that silently corrupts the load parameters the day upstream inserts a
// field — the sort of bug that shows up as a mysteriously different token
// stream, which is exactly what this benchmark must never produce. Building
// this file against the real <llama.h> lets the compiler lay the struct out,
// so the Rust side only ever passes pointers and ints.
//
// Everything on the hot path (llama_tokenize) is called from Rust directly,
// not wrapped here: a shim would add a non-inlinable call per encode, and it
// buys nothing because that signature is pointers and ints already.

#include <stdbool.h>
#include <stdint.h>

#include "llama.h"

// llama_token is the id type llama_tokenize writes. The Rust side hands it an
// `*mut i32`, so if upstream ever changes the width this fails to compile here
// rather than writing past the end of the buffer.
_Static_assert(sizeof(llama_token) == sizeof(int32_t),
               "llama_token must be int32_t for the Rust buffer to match");

static void tokbench_llama_log_noop(enum ggml_log_level level, const char *text,
                                    void *user_data) {
    (void)level;
    (void)text;
    (void)user_data;
}

// llama.cpp narrates every load to stderr — ~80 lines of hparams, vocab stats
// and EOG tokens per model. In a matrix run that is thousands of lines
// interleaved with the driver's own progress output. `llama_log_set(NULL, ...)`
// restores the *default* callback rather than muting it, so muting needs a real
// no-op function, which needs the ggml_log_callback type, which lives here.
void tokbench_llama_silence(void) {
    llama_log_set(tokbench_llama_log_noop, NULL);
}

// vocab_only = true is the whole point of this engine: it makes llama.cpp read
// the GGUF's tokenizer.* keys and stop, skipping every tensor. No GGML backend
// is registered, no Metal device is opened, nothing is allocated for weights.
// (Note that llama_backend_init() is deliberately NOT called: it exists to
// bring up the compute backends, which is precisely the inference stack this
// engine is trying not to drag in. Vocabulary loading does not need it.)
struct llama_model *tokbench_llama_load_vocab(const char *path) {
    struct llama_model_params params = llama_model_default_params();
    params.vocab_only = true;
    params.use_mmap = true;
    params.check_tensors = false;
    return llama_model_load_from_file(path, params);
}
