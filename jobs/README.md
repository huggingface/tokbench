# Running tokbench on Hugging Face Jobs

The Jobs path separates two kinds of reproducibility:

- the image digest, Cargo lockfile, source revision and input revision reproduce
  the software and data;
- repeated complete runs inside a Job, followed by runs on several Jobs,
  quantify timing noise and host-to-host variance.

A Jobs hardware flavor specifies allocated resources. It does not promise that
every Job lands on the same physical CPU model, so do not compare absolute MB/s
across Jobs without reading each run's `environment-before.json`.

The `blog-v1` profile also pins the benchmark process to one logical CPU from
each of eight distinct physical cores. This keeps the 1/2/4/8-thread scaling
sweep off sibling SMT threads. The selected CPU set and effective task affinity
are recorded in the environment manifest.

## Publish a commit-specific Docker Space

Push the tokbench commit first, then create a private Docker Space from the
clean checkout. The generated Space recipe pins both base-image digests and
clones the exact tokbench commit:

```bash
revision=$(git rev-parse HEAD)
python jobs/publish_space.py \
  --repo-id "$USER/tokbench-jobs-${revision:0:7}"
```

The Space is an image builder and host. Benchmarks run on Jobs hardware, not on
the Space. A commit-specific Space is immutable by convention, so submit it
with `--allow-mutable-image`; do not update that Space to another tokbench
commit.

## Submit

Install a current `huggingface_hub`, log in, and use a writable Storage Bucket.
The test-data revision must be a commit SHA, not `main`:

```bash
python jobs/submit.py \
  --image hf.co/spaces/<user>/tokbench-jobs-<commit> \
  --allow-mutable-image \
  --input-revision <tokenizers-test-data-commit> \
  --bucket huggingface/tokbench-results
```

The default is `cpu-performance`, five independent process-level runs, five
timed repetitions per cell, and 1/2/4/8-thread scaling sweeps on English and
Chinese. Use `--models` or `--engines` with comma-separated names for a smaller
validation run. Jobs default to a 30-minute timeout, so the submitter requests
six hours.

To reproduce Section 01 of the tokenizers v1 blog, select its fixed eight-model
matrix. The profile measures encode, decode, 1/2/4/8-thread scaling on English
and Chinese, and call-level latency over 1,000 distinct 512-byte English
documents:

```bash
python jobs/submit.py \
  --profile blog-v1 \
  --image hf.co/spaces/<user>/tokbench-jobs-<commit> \
  --allow-mutable-image \
  --input-revision <tokenizers-test-data-commit> \
  --bucket huggingface/tokenizers-v1-benchmarks
```

`blog-v1` fixes the models, the `hf-tokenizers`, `pipeline`, and
`pipeline-no-cache` engines, English/Chinese scaling corpora, and eight-thread
ceiling. The third engine is the cache-disabled diagnostic used by Section 02.
Use the default profile instead when overriding that matrix. A model selection
controls both the downloaded artifacts and the driver filter, so a selected
model cannot be silently absent from the Job.
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
