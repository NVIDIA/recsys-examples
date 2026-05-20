#!/usr/bin/env bash
# Sweep request batch_size and len_per_seq; write summarization totals, then plot.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

OUT_ROOT="${OUT_ROOT:-${ROOT}/profiler_python_result}"

BATCH_SIZES="${BATCH_SIZES:-8 32 64 128}"
LENS="${LENS:-1024 2048 4096}"
OFFLOAD_COUNTS="${OFFLOAD_COUNTS:-50,100,150,200,250,300}"
HOST_CAPACITY_SCALE="${HOST_CAPACITY_SCALE:-8}"
WARMUP="${WARMUP:-1}"
# Set NO_ORIGIN=1 to skip large origin_data CSV during long sweeps.
NO_ORIGIN="${NO_ORIGIN:-0}"

EXTRA_ARGS=()
if [[ "${NO_ORIGIN}" == "1" ]]; then
  EXTRA_ARGS+=(--no-origin-data)
fi

for bs in ${BATCH_SIZES}; do
  for len in ${LENS}; do
    echo "======== bs=${bs} len=${len} ========"
    python3 profiler_offload.py \
      --batch-size "${bs}" \
      --len-per-seq "${len}" \
      --max-seq-len "${len}" \
      --host-capacity-scale "${HOST_CAPACITY_SCALE}" \
      --offload-batch-counts "${OFFLOAD_COUNTS}" \
      --warmup-iterations "${WARMUP}" \
      --output-root "${OUT_ROOT}" \
      "${EXTRA_ARGS[@]}"
  done
done

python3 plot_offload_scenario_total.py \
  --summary-root "${OUT_ROOT}/summarization" \
  --origin-root "${OUT_ROOT}/origin_data" \
  --output-root "${OUT_ROOT}/plot" \
  --offload-batch-counts "${OFFLOAD_COUNTS}"

python3 plot_offload_scenario_panel.py \
  --summary-root "${OUT_ROOT}/summarization" \
  --origin-root "${OUT_ROOT}/origin_data" \
  --output "${OUT_ROOT}/plot/microbench2_bs8_len_panel.png" \
  --offload-batch-counts "${OFFLOAD_COUNTS}"

echo "[DONE] CSV under ${OUT_ROOT}/summarization/"
echo "[DONE] plots under ${OUT_ROOT}/plot/ (incl. microbench2_bs8_len_panel.png)"
