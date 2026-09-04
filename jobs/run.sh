#!/usr/bin/env bash
set -Eeuo pipefail

: "${TOKBENCH_INPUT_REVISION:?set TOKBENCH_INPUT_REVISION to an immutable commit SHA}"
[[ "${TOKBENCH_INPUT_REVISION}" =~ ^[0-9a-fA-F]{40}$ ]] \
  || { echo "TOKBENCH_INPUT_REVISION must be a full 40-character commit SHA" >&2; exit 2; }

output_mount="${TOKBENCH_OUTPUT_DIR:-/outputs}"
job_name="${JOB_ID:-local-$(date -u +%Y%m%dT%H%M%SZ)}"
output_dir="${output_mount%/}/${job_name}"
mkdir -p "${output_dir}"

runs="${TOKBENCH_RUNS:-5}"
reps="${TOKBENCH_REPS:-5}"
features="${TOKBENCH_FEATURES:-rust-engines}"
scaling_csv="${TOKBENCH_SCALING:-eng_Latn,cmn_Hani}"
models_csv="${TOKBENCH_MODELS:-}"
engines_csv="${TOKBENCH_ENGINES:-}"
max_threads="${TOKBENCH_MAX_THREADS:-8}"
no_decode="${TOKBENCH_NO_DECODE:-0}"
pin_physical_cores="${TOKBENCH_PIN_PHYSICAL_CORES:-0}"
latency_csv="${TOKBENCH_LATENCY:-}"
latency_bytes="${TOKBENCH_LATENCY_BYTES:-512}"
latency_samples="${TOKBENCH_LATENCY_SAMPLES:-1000}"

for numeric in "${runs}" "${reps}" "${max_threads}" "${latency_bytes}" "${latency_samples}"; do
  [[ "${numeric}" =~ ^[1-9][0-9]*$ ]] \
    || { echo "runs, reps and max threads must be positive integers" >&2; exit 2; }
done
[[ "${no_decode}" == 0 || "${no_decode}" == 1 ]] \
  || { echo "TOKBENCH_NO_DECODE must be 0 or 1" >&2; exit 2; }
[[ "${pin_physical_cores}" == 0 || "${pin_physical_cores}" == 1 ]] \
  || { echo "TOKBENCH_PIN_PHYSICAL_CORES must be 0 or 1" >&2; exit 2; }

# Scaling must not accidentally place two workers on sibling SMT threads.
# Select one logical CPU for each distinct socket/core pair, restrict the whole
# process to the requested number of physical cores, then restart so every
# subsequently-created worker inherits that affinity.
if [[ "${pin_physical_cores}" == 1 && "${TOKBENCH_AFFINITY_PINNED:-0}" != 1 ]]; then
  cpuset="$({
    lscpu -p=CPU,CORE,SOCKET \
      | awk -F, '!/^#/ { key=$2 "," $3; if (!seen[key]++) print $1 }' \
      | head -n "${max_threads}"
  } | paste -sd, -)"
  cpu_count="$(awk -F, '{ print NF }' <<< "${cpuset}")"
  [[ -n "${cpuset}" && "${cpu_count}" == "${max_threads}" ]] || {
    echo "need ${max_threads} distinct physical cores, found ${cpu_count}: ${cpuset}" >&2
    exit 2
  }
  export TOKBENCH_AFFINITY_PINNED=1
  export TOKBENCH_PINNED_CPUSET="${cpuset}"
  echo "pinning benchmark to physical-core CPUs: ${cpuset}"
  exec taskset -c "${cpuset}" bash "$0" "$@"
fi

split_csv() {
  local value="$1"
  local -n destination="$2"
  IFS=',' read -r -a destination <<< "${value}"
  for item in "${destination[@]}"; do
    [[ -z "${item}" || "${item}" =~ ^[A-Za-z0-9._-]+$ ]] \
      || { echo "invalid benchmark selector: ${item}" >&2; exit 2; }
  done
}

split_csv "${models_csv}" model_items
split_csv "${engines_csv}" engine_items
split_csv "${scaling_csv}" scaling_items
split_csv "${latency_csv}" latency_items

python3 jobs/collect_environment.py "${output_dir}/environment-before.json"

# The source dataset is private today, so HF_TOKEN is normally supplied as a
# Job secret. The revision is mandatory because `main` is not reproducible.
make_args=(HF_TEST_REVISION="${TOKBENCH_INPUT_REVISION}")
if [[ -n "${models_csv}" ]]; then
  make_args+=(BENCH_MODELS="${model_items[*]}")
fi
make "${make_args[@]}" fixtures models

find data/models data/fixtures -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum > "${output_dir}/input-sha256.txt"

cargo build --locked --release -p tokbench --features "${features}"

args=(--reps "${reps}" --max-threads "${max_threads}" --no-memory)
args+=(--latency-bytes "${latency_bytes}" --latency-samples "${latency_samples}")
[[ "${no_decode}" == 1 ]] && args+=(--no-decode)
for item in "${scaling_items[@]}"; do
  [[ -n "${item}" ]] && args+=(--scaling "${item}")
done
for item in "${latency_items[@]}"; do
  [[ -n "${item}" ]] && args+=(--latency "${item}")
done
for item in "${model_items[@]}"; do
  [[ -n "${item}" ]] && args+=(--model "${item}")
done
for item in "${engine_items[@]}"; do
  [[ -n "${item}" ]] && args+=(--engine "${item}")
done

# Each iteration is a new process and a complete paired scaling sweep. Keeping
# every report lets downstream analysis compute efficiency within a sweep and
# measure run-to-run variance instead of dividing independently pooled medians.
for ((run = 1; run <= runs; run++)); do
  run_id="$(printf '%02d' "${run}")"
  run_args=("${args[@]}")
  if ((run % 2 == 0)); then
    run_args+=(--reverse-scaling)
  fi
  printf '%q ' target/release/tokbench "${run_args[@]}" \
    > "${output_dir}/run-${run_id}-command.txt"
  printf '\n' >> "${output_dir}/run-${run_id}-command.txt"
  target/release/tokbench "${run_args[@]}" \
    --out "${output_dir}/run-${run_id}.json" \
    2>&1 | tee "${output_dir}/run-${run_id}.log"
done

python3 jobs/summarize_scaling.py \
  "${output_dir}/scaling-summary.json" \
  "${output_dir}"/run-*.json
python3 jobs/collect_environment.py "${output_dir}/environment-after.json"
(
  cd "${output_dir}"
  find . -maxdepth 1 -type f ! -name artifact-sha256.txt -print0 \
    | sort -z \
    | xargs -0 sha256sum > artifact-sha256.txt
)

echo "results written to ${output_dir}"
