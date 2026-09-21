# tokbench — fetch inputs, measure, inspect.

HF ?= uvx --from huggingface_hub hf
PY ?= python3
DATA := data
MODELS := $(DATA)/models
FIXTURES := $(DATA)/fixtures
TOKBENCH := cargo run --locked --release -p tokbench --features rust-engines --

# Pin the input data to make a run reproducible.
HF_TEST_REVISION ?=
HF_REVISION_ARG := $(if $(strip $(HF_TEST_REVISION)),--revision $(HF_TEST_REVISION),)

BUCKET ?=
JOB_ID ?=
RUN ?=
RESULTS ?=
DASH_PORT ?= 8712

FIXTURE_LANGS := amh_Ethi arb_Arab ben_Beng cmn_Hani ell_Grek eng_Latn heb_Hebr \
                 hin_Deva jpn_Jpan kat_Geor kor_Hang rus_Cyrl tam_Taml tha_Thai
# Chat and agent traces are their own workload: short turns, many added tokens,
# and a normalizer path prose never reaches.
FIXTURE_MODALITIES := agentic-traces agentic_swe code_mixed math_latex \
                      added_special_dense added_special_sparse \
                      added_normalized_dense added_normalized_sparse
HF_TEST_REPO := hf-internal-testing/tokenizers-test-data

# One model per archetype: byte-level BPE, WordPiece, Unigram.
BENCH_MODELS := gpt2 llama-3 deepseek-v4 bert-base-uncased t5-base

.PHONY: all
all: fixtures models sizes bench

.PHONY: fixtures
fixtures:
	@mkdir -p $(FIXTURES)
	@for f in $(FIXTURE_LANGS); do \
	  [ -f $(FIXTURES)/$$f.txt ] || { echo "fetch lang/$$f"; \
	    $(HF) download $(HF_TEST_REPO) fixtures/lang/$$f.txt --repo-type dataset \
	      $(HF_REVISION_ARG) --local-dir $(DATA)/_dl >/dev/null && \
	    cp $(DATA)/_dl/fixtures/lang/$$f.txt $(FIXTURES)/ ; } ; \
	done
	@for f in $(FIXTURE_MODALITIES); do \
	  [ -f $(FIXTURES)/$$f.txt ] || { echo "fetch modalities/$$f"; \
	    $(HF) download $(HF_TEST_REPO) fixtures/modalities/$$f.txt --repo-type dataset \
	      $(HF_REVISION_ARG) --local-dir $(DATA)/_dl >/dev/null && \
	    cp $(DATA)/_dl/fixtures/modalities/$$f.txt $(FIXTURES)/ ; } ; \
	done
	@echo "fixtures ready: $$(ls $(FIXTURES) | wc -l | tr -d ' ') corpora"

# Agent traces, code and mixed scripts, rendered through each model's real
# Jinja chat_template so the bytes are what a served model tokenizes.
.PHONY: bigfixtures
bigfixtures:
	$(PY) scripts/make_fixtures.py 4

# Fetch each model's tokenizer.json, then derive every other engine's artifact
# from it. Deriving rather than downloading separately is the point: all
# engines must be measured on the same vocabulary.
.PHONY: models
models:
	@mkdir -p $(MODELS)
	@for m in $(BENCH_MODELS); do \
	  [ -f $(MODELS)/$$m/tokenizer.json ] || { echo "fetch model $$m"; \
	    case "$$m" in \
	      bert-base-uncased|t5-base) source="$$m.json" ;; \
	      *) source="models/$$m/tokenizer.json" ;; \
	    esac; \
	    mkdir -p $(MODELS)/$$m && \
	    $(HF) download $(HF_TEST_REPO) "$$source" --repo-type dataset \
	      $(HF_REVISION_ARG) --local-dir $(DATA)/_dl >/dev/null && \
	    cp "$(DATA)/_dl/$$source" $(MODELS)/$$m/tokenizer.json ; } ; \
	done
	@$(PY) scripts/make_artifacts.py $(MODELS)

# Footprint side-channels. Optional: the driver omits the columns rather than
# reporting zeros when the files are absent.
.PHONY: sizes
sizes:
	$(PY) scripts/package_size.py
	-bash scripts/binsize.sh

.PHONY: bench
bench:
	$(TOKBENCH) --reps 5

.PHONY: bench-open
bench-open:
	$(TOKBENCH) --reps 5 --open

# One measurement family instead of the whole matrix.
.PHONY: encode decode latency scaling memory
encode:
	$(TOKBENCH) measure encode --engine all
decode:
	$(TOKBENCH) measure decode --engine all
latency:
	$(TOKBENCH) measure latency --engine all
scaling:
	$(TOKBENCH) measure scaling --engine all --corpus eng_Latn
memory:
	$(TOKBENCH) measure memory --engine all --corpus eng_Latn

.PHONY: test
test:
	cargo test -p tokbench-core
	cargo clippy --features rust-engines --all-targets -- -D warnings
	cargo fmt --all -- --check
	$(PY) -m unittest discover -s hf-jobs -p 'test_*.py'

.PHONY: clean
clean:
	rm -f tokenizer_bench_results.json binary_sizes.json package_sizes.json
	cargo clean

# With BUCKET + JOB_ID, fetch and aggregate an HF Job; with RESULTS, stage a
# local report; with neither, serve the local tokenizer_bench_results.json.
# RUN selects one report instead of the median across the Job's runs.
.PHONY: dash
dash:
	$(PY) hf-jobs/dash.py \
	  $(if $(BUCKET),--bucket "$(BUCKET)") \
	  $(if $(JOB_ID),--job-id "$(JOB_ID)") \
	  $(if $(RUN),--run "$(RUN)") \
	  $(if $(RESULTS),--results "$(RESULTS)") \
	  --port "$(DASH_PORT)"
