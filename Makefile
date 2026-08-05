# tokbench — fetch inputs, build, measure.
#
# Everything a run needs is reproducible from this file: the corpora, the model
# artifacts each engine's format requires, and the two footprint side-channels.

HF ?= uvx --from huggingface_hub hf
PY ?= python3
DATA := data
MODELS := $(DATA)/models
FIXTURES := $(DATA)/fixtures

# Same fixture set as the upstream tokenizers pipeline benchmark, so numbers
# stay comparable with it.
FIXTURE_LANGS := amh_Ethi arb_Arab ben_Beng cmn_Hani ell_Grek eng_Latn heb_Hebr \
                 hin_Deva jpn_Jpan kat_Geor kor_Hang rus_Cyrl tam_Taml tha_Thai
FIXTURE_MODALITIES := agentic_swe code_mixed math_latex
HF_TEST_REPO := hf-internal-testing/tokenizers-bench

# One model per archetype: the shapes that stress different parts of a
# tokenizer (regex-heavy byte-level BPE, normalizer-heavy WordPiece, Unigram).
BENCH_MODELS := gpt2 llama-3 deepseek-v4 bert-base-uncased t5-base

.PHONY: all
all: fixtures models sizes bench

.PHONY: fixtures
fixtures:
	@mkdir -p $(FIXTURES)
	@for f in $(FIXTURE_LANGS); do \
	  [ -f $(FIXTURES)/$$f.txt ] || { echo "fetch lang/$$f"; \
	    $(HF) download $(HF_TEST_REPO) fixtures/lang/$$f.txt --repo-type dataset \
	      --local-dir $(DATA)/_dl >/dev/null && \
	    cp $(DATA)/_dl/fixtures/lang/$$f.txt $(FIXTURES)/ ; } ; \
	done
	@for f in $(FIXTURE_MODALITIES); do \
	  [ -f $(FIXTURES)/$$f.txt ] || { echo "fetch modalities/$$f"; \
	    $(HF) download $(HF_TEST_REPO) fixtures/modalities/$$f.txt --repo-type dataset \
	      --local-dir $(DATA)/_dl >/dev/null && \
	    cp $(DATA)/_dl/fixtures/modalities/$$f.txt $(FIXTURES)/ ; } ; \
	done
	@echo "fixtures ready: $$(ls $(FIXTURES) | wc -l | tr -d ' ') corpora"

# Fetch each model's tokenizer.json, then derive the per-engine artifacts from
# it. Deriving rather than downloading separately is deliberate: every engine
# must be measured on the SAME vocabulary, or the comparison is meaningless.
.PHONY: models
models:
	@mkdir -p $(MODELS)
	@for m in $(BENCH_MODELS); do \
	  [ -f $(MODELS)/$$m/tokenizer.json ] || { echo "fetch model $$m"; \
	    mkdir -p $(MODELS)/$$m && \
	    $(HF) download $(HF_TEST_REPO) models/$$m/tokenizer.json --repo-type dataset \
	      --local-dir $(DATA)/_dl >/dev/null && \
	    cp $(DATA)/_dl/models/$$m/tokenizer.json $(MODELS)/$$m/ ; } ; \
	done
	@$(PY) scripts/make_artifacts.py $(MODELS)

# Footprint side-channels. Both are optional; the driver omits the columns when
# the files are absent rather than reporting zeros.
.PHONY: sizes
sizes:
	$(PY) scripts/package_size.py
	-bash scripts/binsize.sh

.PHONY: bench
bench:
	cargo run --release -p tokbench --features rust-engines -- --reps 5

.PHONY: bench-open
bench-open:
	cargo run --release -p tokbench --features rust-engines -- --reps 5 --open

.PHONY: test
test:
	cargo test -p tokbench-core
	$(PY) python/harness.py

.PHONY: clean
clean:
	rm -f tokenizer_bench_results.json binary_sizes.json package_sizes.json
	cargo clean
