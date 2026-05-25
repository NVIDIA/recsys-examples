#!/usr/bin/env bash
# Full pipeline: nsys rep -> csv -> plot -> profiling.xlsx
# All outputs under breakdown_1/profiler_result/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OUT_ROOT="${SCRIPT_DIR}/profiler_result"
MODE_DIR="flexkv_profile_fine"
SEQ_LENS=(1024 2048 4096)
BATCH_SIZE=8
REPEAT=1
mkdir -p \
  "${OUT_ROOT}/rep/${MODE_DIR}" \
  "${OUT_ROOT}/csv/${MODE_DIR}" \
  "${OUT_ROOT}/plot/plot" \
  "${OUT_ROOT}/plot/csv_summarization"

echo "[INFO] OUT_ROOT=${OUT_ROOT}"

# 1) nsys rep (new NVTX hooks)
for len in "${SEQ_LENS[@]}"; do
  tag="len${len}_bs${BATCH_SIZE}"
  echo "[RUN] nsys profile ${tag}"
  nsys profile \
    -t cuda,nvtx,osrt \
    --sample=none \
    --cuda-memory-usage=true \
    --force-overwrite true \
    -o "${OUT_ROOT}/rep/${MODE_DIR}/${tag}" \
    python3 test_flexkv_profile_fine.py \
      --len-per-seq "${len}" \
      --batch-size "${BATCH_SIZE}" \
      --repeat "${REPEAT}" \
      --max-seq-len "${len}"
done

# 2) rep -> NVTX csv
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

for len in "${SEQ_LENS[@]}"; do
  tag="len${len}_bs${BATCH_SIZE}"
  rep_file="${OUT_ROOT}/rep/${MODE_DIR}/${tag}.nsys-rep"
  out_prefix="${OUT_ROOT}/csv/${MODE_DIR}/${tag}"
  ok=0
  for rpt in nvtx_pushpop_sum nvtxppsum nvtxsum; do
    rm -f "${out_prefix}"*"${rpt}"*.csv || true
    if nsys stats --force-export=true --report "${rpt}" --format csv --output "${out_prefix}" "${rep_file}" >/tmp/nsys_stats_${tag}.log 2>&1; then
      shopt -s nullglob
      cands=( "${out_prefix}"*"${rpt}"*.csv )
      shopt -u nullglob
      if (( ${#cands[@]} > 0 )) && validate_nvtx_csv "${cands[0]}"; then
        echo "[OK] ${tag} -> ${cands[0]} (report=${rpt})"
        ok=1
        break
      fi
    fi
  done
  if (( ok == 0 )); then
    echo "[ERR] ${tag} export NVTX CSV failed"
    cat "/tmp/nsys_stats_${tag}.log" || true
    exit 1
  fi
done

# 3) plot (6 figures + csv_summarization)
SEQ_LENS_CSV="$(IFS=,; echo "${SEQ_LENS[*]}")"
python3 plot_flexkv_profile_fine.py \
  --csv-root "${OUT_ROOT}/csv" \
  --mode-dir "${MODE_DIR}" \
  --seq-lens "${SEQ_LENS_CSV}" \
  --batch-size "${BATCH_SIZE}" \
  --output-root "${OUT_ROOT}/plot"

# 4) quick sanity check
rg "flexkv\\.build_index_meta|recsys\\.merge_lookup_results|step1\\.gpu\\.lookup_py" \
  "${OUT_ROOT}/csv/${MODE_DIR}" -n || true

echo "[DONE] results tree:"
echo "  ${OUT_ROOT}/rep/${MODE_DIR}/"
echo "  ${OUT_ROOT}/csv/${MODE_DIR}/"
echo "  ${OUT_ROOT}/plot/plot/"
echo "  ${OUT_ROOT}/plot/csv_summarization/"
