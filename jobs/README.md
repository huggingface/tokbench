# Running tokbench on Hugging Face Jobs

The Jobs path separates two kinds of reproducibility:

- the image digest, Cargo lockfile, source revision and input revision reproduce
  the software and data;
- repeated complete runs inside a Job, followed by runs on several Jobs,
  quantify timing noise and host-to-host variance.

A Jobs hardware flavor specifies allocated resources. It does not promise that
every Job lands on the same physical CPU model, so do not compare absolute MB/s
across Jobs without reading each run's `environment-before.json`.

## Build and publish an immutable image

From a clean tokbench checkout:

```bash
revision=$(git rev-parse HEAD)
docker build -f jobs/Dockerfile \
  --build-arg TOKBENCH_REVISION="$revision" \
  -t ghcr.io/huggingface/tokbench:"$revision" .
docker push ghcr.io/huggingface/tokbench:"$revision"
```

Resolve the pushed tag to its registry digest. The submitter rejects mutable
tags unless `--allow-mutable-image` is passed explicitly.

## Submit

Install a current `huggingface_hub`, log in, and use a writable Storage Bucket.
The test-data revision must be a commit SHA, not `main`:

```bash
python jobs/submit.py \
  --image ghcr.io/huggingface/tokbench@sha256:<digest> \
  --input-revision <tokenizers-test-data-commit> \
  --bucket huggingface/tokbench-results
```

The default is `cpu-performance`, five independent process-level runs, five
timed repetitions per cell, and 1/2/4/8-thread scaling sweeps on English and
Chinese. Use `--models` or `--engines` with comma-separated names for a smaller
validation run. Jobs default to a 30-minute timeout, so the submitter requests
six hours.

To reproduce the encode and scaling inputs used by Section 01 of the
tokenizers v1 blog, select its fixed eight-model matrix. The profile also skips
decode, which is not an input to those figures:

```bash
python jobs/submit.py \
  --profile blog-v1 \
  --image ghcr.io/huggingface/tokbench@sha256:<digest> \
  --input-revision <tokenizers-test-data-commit> \
  --bucket huggingface/tokenizers-v1-benchmarks
```

`blog-v1` fixes the models, engines, English/Chinese scaling corpora, and
eight-thread ceiling. Use the default profile instead when overriding that
matrix. A model selection controls both the downloaded artifacts and the
driver filter, so a selected model cannot be silently absent from the Job.
Add `--dry-run` to print the resolved, non-secret Job configuration without
submitting or consuming compute.

The private `hf-internal-testing/tokenizers-test-data` input requires an HF
token. By default the submitter forwards the locally configured token as an
encrypted Job secret. It is not written to the environment manifest.

## Artifacts

Each Job writes to `/outputs/$JOB_ID/` in the bucket:

- `run-01.json` through `run-05.json`: complete, independent reports;
- matching logs for diagnosis;
- the exact command for each run, with scaling order alternating between
  ascending and descending thread counts to expose temporal drift;
- `scaling-summary.json`, computed from paired points within each report;
- environment manifests captured before and after benchmarking;
- SHA-256 manifests for every input and output;
- the exact benchmark command.

Compute scaling efficiency within each report from its paired 1-thread and
8-thread points. Summarize that distribution across reports. Do not divide an
8-thread median pooled across reports by a separately pooled 1-thread median.

## View a completed Job

The native dashboard fetches through `huggingface_hub`, verifies the artifact
manifest, and displays the median of every complete `run-*.json` report. The
aggregate preserves median encode and decode metrics; the current native
dashboard renders its existing encode and scaling views:

```bash
make dash BUCKET=huggingface/tokbench-results JOB_ID=<job-id>
```

The aggregate keeps each thread-count result paired with its own run's
1-thread baseline before taking the median. The dashboard labels aggregated
data explicitly. To inspect one underlying run instead:

```bash
make dash BUCKET=huggingface/tokbench-results JOB_ID=<job-id> RUN=3
```

Local files use the same path. A directory is aggregated by default:

```bash
make dash RESULTS=results/<job-id>
make dash RESULTS=results/<job-id> RUN=3
make dash RESULTS=results/<job-id>/run-03.json
```
