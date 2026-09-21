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
CORPORA_REVISION ?=
CORPORA_REVISION_ARG := $(if $(strip $(CORPORA_REVISION)),--revision $(CORPORA_REVISION),)

BUCKET ?=
JOB_ID ?=
RUN ?=
RESULTS ?=
DASH_PORT ?= 8712

# The public corpora. Chat and agent traces are their own workload: short
# turns, many added tokens, and a normalizer path prose never reaches.
# Provenance and licences: scripts/corpora_card.md.
CORPORA_REPO ?= huggingface/tokbench-corpora
FIXTURE_CORPORA := amharic arabic bengali chinese english georgian greek hebrew \
                   hindi japanese korean russian tamil thai \
                   agentic-swe math-latex \
                   added-special-dense added-special-sparse \
                   added-normalized-dense added-normalized-sparse \
                   chat-llama3 chat-chatml chat-mistral chat-deepseek

# Not publicly redistributable, so these stay internal: code-mixed embeds redis
# (RSALv2/SSPLv1/AGPLv3) and junit5 (EPL-2.0) source verbatim, agentic-traces
# has no recorded provenance. `local:upstream` because the internal repo still
# uses the old names.
FIXTURE_INTERNAL := agentic-traces:agentic-traces code-mixed:code_mixed
HF_TEST_REPO := hf-internal-testing/tokenizers-test-data

# One model per archetype: byte-level BPE, WordPiece, Unigram.
BENCH_MODELS := gpt2 llama-3 deepseek-v4 bert-base-uncased t5-base

.PHONY: all
all: fixtures models sizes bench

.PHONY: fixtures
fixtures:
	@mkdir -p $(FIXTURES)
	@for f in $(FIXTURE_CORPORA); do \
	  [ -f $(FIXTURES)/$$f.txt ] || { echo "fetch $$f"; \
	    $(HF) download $(CORPORA_REPO) fixtures/$$f.txt --repo-type dataset \
	      $(CORPORA_REVISION_ARG) --local-dir $(DATA)/_dl >/dev/null && \
	    cp $(DATA)/_dl/fixtures/$$f.txt $(FIXTURES)/ ; } ; \
	done
	@for f in $(FIXTURE_INTERNAL); do \
	  n=$${f%%:*}; u=$${f##*:}; \
	  [ -f $(FIXTURES)/$$n.txt ] || { echo "fetch $$n (internal)"; \
	    $(HF) download $(HF_TEST_REPO) fixtures/modalities/$$u.txt --repo-type dataset \
	      $(HF_REVISION_ARG) --local-dir $(DATA)/_dl >/dev/null && \
	    cp $(DATA)/_dl/fixtures/modalities/$$u.txt $(FIXTURES)/$$n.txt ; } ; \
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
	$(TOKBENCH) measure scaling --engine all --corpus english
memory:
	$(TOKBENCH) measure memory --engine all --corpus english

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
