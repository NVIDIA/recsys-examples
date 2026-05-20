#!/usr/bin/env bash
# Micro-bench 2: nsys NVTX sweep (profiler_offload.py hooks -> .nsys-rep -> nvtx CSV).
#
# Default output tree (under breakdown_2/):
#   profiler_result_nsys/rep/<tag>.nsys-rep
#   profiler_result_nsys/csv/<tag>_nvtx_pushpop_sum.csv   (or nvtxsum / nvtxppsum)
#
# Env overrides (same semantics as run_sweep_offload.sh unless noted):
#   BATCH_SIZES, LENS, OFFLOAD_COUNTS, HOST_CAPACITY_SCALE, WARMUP, NO_ORIGIN
#   OUT_ROOT          default: ${SCRIPT_DIR}/profiler_result_nsys
#   NSYS_EXTRA        extra args passed to nsys profile (quoted)
#
# Example (full sweep, long):
#   cd breakdown_2 && bash run_sweep_offload_nsys.sh
#
# Example (quick smoke):
#   BATCH_SIZES=8 LENS=1024 OFFLOAD_COUNTS=50,100 NO_ORIGIN=1 bash run_sweep_offload_nsys.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/profiler_result_nsys}"
BATCH_SIZES="${BATCH_SIZES:-8 32 64 128}"
LENS="${LENS:-1024 2048 4096}"
OFFLOAD_COUNTS="${OFFLOAD_COUNTS:-50,100,150,200,250,300}"
HOST_CAPACITY_SCALE="${HOST_CAPACITY_SCALE:-8}"
WARMUP="${WARMUP:-1}"
NO_ORIGIN="${NO_ORIGIN:-1}"

mkdir -p "${OUT_ROOT}/rep" "${OUT_ROOT}/csv"

validate_nvtx_csv() {
  local csv_file="$1"
  python3 - "$csv_file" <<'PY'
import csv, sys
with open(sys.argv[1], newline="") as f:
    r = csv.reader(f)
    header = next(r, [])
h = set(header)
label_ok = bool({"Range", "Name", "NVTX Range"} & h)
time_ok = bool({"Total Time (ns)", "Total Time (us)", "Total Time (ms)", "Total Time"} & h)
raise SystemExit(0 if (label_ok and time_ok) else 1)
PY
}

export_nvtx_csv() {
  local tag="$1"
  local rep_file="${OUT_ROOT}/rep/${tag}.nsys-rep"
  local out_prefix="${OUT_ROOT}/csv/${tag}"
  local ok=0

  if [[ ! -f "${rep_file}" ]]; then
    echo "[ERR] missing rep: ${rep_file}"
    return 1
  fi

  for rpt in nvtx_pushpop_sum nvtxsum nvtxppsum; do
    rm -f "${out_prefix}"*"${rpt}"*.csv 2>/dev/null || true
    if nsys stats --report "${rpt}" --format csv --output "${out_prefix}" "${rep_file}" \
      >"/tmp/nsys_stats_${tag}.log" 2>&1; then
      shopt -s nullglob
      local cands=( "${out_prefix}"*"${rpt}"*.csv )
      shopt -u nullglob
      if (( ${#cands[@]} > 0 )) && validate_nvtx_csv "${cands[0]}"; then
        echo "[OK] ${tag} -> ${cands[0]} (report=${rpt})"
        ok=1
        break
      fi
    fi
  done

  if (( ok == 0 )); then
    echo "[ERR] ${tag} NVTX CSV export failed"
    cat "/tmp/nsys_stats_${tag}.log" || true
    return 1
  fi
}

PROF_ARGS=(
  --host-capacity-scale "${HOST_CAPACITY_SCALE}"
  --offload-batch-counts "${OFFLOAD_COUNTS}"
  --warmup-iterations "${WARMUP}"
  --output-root "${SCRIPT_DIR}"
)
if [[ "${NO_ORIGIN}" == "1" ]]; then
  PROF_ARGS+=(--no-origin-data)
fi

echo "[INFO] OUT_ROOT=${OUT_ROOT}"
echo "[INFO] BATCH_SIZES=${BATCH_SIZES}"
echo "[INFO] LENS=${LENS}"
echo "[INFO] OFFLOAD_COUNTS=${OFFLOAD_COUNTS}"

for bs in ${BATCH_SIZES}; do
  for len in ${LENS}; do
    tag="offload_bs${bs}_len${len}"
    echo ""
    echo "======== nsys profile ${tag} ========"

    # shellcheck disable=SC2086
    nsys profile \
      -t cuda,nvtx,osrt \
      --sample=none \
      --cuda-memory-usage=true \
      --force-overwrite true \
      ${NSYS_EXTRA:-} \
      -o "${OUT_ROOT}/rep/${tag}" \
      python3 profiler_offload.py \
        --batch-size "${bs}" \
        --len-per-seq "${len}" \
        --max-seq-len "${len}" \
        "${PROF_ARGS[@]}"

    export_nvtx_csv "${tag}"
  done
done

echo ""
echo "[DONE] NVTX sweep finished."
echo "  rep: ${OUT_ROOT}/rep/"
echo "  csv: ${OUT_ROOT}/csv/"
echo ""
echo "Inspect labels, e.g.:"
echo "  rg 'flexkv\\.' ${OUT_ROOT}/csv/ -n | head"
