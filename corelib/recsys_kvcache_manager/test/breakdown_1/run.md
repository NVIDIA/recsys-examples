# FlexKV Profiling 运行脚本

## 0) 路径与参数

```bash
set -euo pipefail
cd recsys-example-kvcache-origin/corelib/recsys_kvcache_manager/test
TEST_DIR="breakdown"
OUT_ROOT="profiler_result_finecpp"
MODE_DIR="flexkv_profile_fine"
SEQ_LENS=(1024 2048 4096)
BATCH_SIZE=8
REPEAT=1
BASE_PROFILE_MODE="flexkv_fine"

cd "${TEST_DIR}"
mkdir -p \
  "${OUT_ROOT}/rep/${MODE_DIR}" \
  "${OUT_ROOT}/csv/${MODE_DIR}" \
  "${OUT_ROOT}/plot/plot" \
  "${OUT_ROOT}/plot/csv_summarization"
```

## 1) 生成 `.nsys-rep`

```bash
for len in "${SEQ_LENS[@]}"; do
  tag="len${len}_bs${BATCH_SIZE}"
  nsys profile \
    -t cuda,nvtx,osrt \
    --sample=none \
    --cuda-memory-usage=true \
    --force-overwrite true \
    -o "${OUT_ROOT}/rep/${MODE_DIR}/${tag}" \
    python3 test_flexkv_profile_fine.py \
      --base-profile-mode "${BASE_PROFILE_MODE}" \
      --len-per-seq "${len}" \
      --batch-size "${BATCH_SIZE}" \
      --repeat "${REPEAT}" \
      --max-seq-len "${len}"
done
```

## 2) `.nsys-rep` 转 NVTX CSV（自动兼容 report 名）

```bash
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
  for rpt in nvtxsum nvtxppsum nvtx_pushpop_sum; do
    rm -f "${out_prefix}"*"${rpt}"*.csv || true
    if nsys stats --report "${rpt}" --format csv --output "${out_prefix}" "${rep_file}" >/tmp/nsys_stats_${tag}.log 2>&1; then
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
    echo "[ERR] ${tag} 导出 NVTX CSV 失败"
    cat "/tmp/nsys_stats_${tag}.log" || true
    exit 1
  fi
done
```

## 3) 画图

```bash
SEQ_LENS_CSV="$(IFS=,; echo "${SEQ_LENS[*]}")"

python3 plot_flexkv_profile_fine.py \
  --csv-root profiler_result/csv \
  --mode-dir flexkv_profile_fine \
  --seq-lens 1024,2048,4096 \
  --batch-size 8 \
  --output-root profiler_result/plot
```

> 分层输出（**L1–L3 + init**，不再生成 L4/L5）：
> - `step_flow_bs8.(png|csv)`           # L1 step1/2/3
> - `step_op_flow_bs8.(png|csv)`        # L2 九个 stepX.op
> - `L3_breakdown_bs8.(png|csv)`        # L3 函数级 timeline
> - `init_breakdown_bs8.(png|csv)`      # init
>
> 旧图可删：`cpu_gpu_py_breakdown_bs8*`、`gpu_cpu_cpp_breakdown_bs8*`、`step_op_wall_client_gpu_residual_bs8*`。

## 4) 快速检查是否出现 `gpu/cpu *_cpp` / `kernel` 标注

```bash
rg "gpu\\..*_cpp|cpu\\..*_cpp|gpu\\.kernel\\." "${OUT_ROOT}/csv/${MODE_DIR}" -n || true
echo "[DONE] 输出目录: ${OUT_ROOT}"
```

## 5) 单次 `offload_try_wait` 专项 profiling

```bash
OUT_ROOT_SINGLE="/home/scratch.noliu_gpu/recsys-example-kvcache-origin/corelib/recsys_kvcache_manager/test/profiler_result_finecpp/single_try_wait"
mkdir -p "${OUT_ROOT_SINGLE}/rep" "${OUT_ROOT_SINGLE}/csv" "${OUT_ROOT_SINGLE}/plot"

nsys profile \
  -t cuda,nvtx,osrt \
  --sample=none \
  --cuda-memory-usage=true \
  --force-overwrite true \
  -o "${OUT_ROOT_SINGLE}/rep/len1024_bs8_single_try_wait" \
  python3 single_try_wait_profiling.py \
    --base-profile-mode flexkv_fine \
    --len-per-seq 1024 \
    --batch-size 8 \
    --max-seq-len 1024

# Script exits via os._exit(0) after step1.offload_try_wait_once (no drain/shutdown).

nsys stats --report nvtx_pushpop_sum --format csv \
  --output "${OUT_ROOT_SINGLE}/csv/len1024_bs8_single_try_wait" \
  "${OUT_ROOT_SINGLE}/rep/len1024_bs8_single_try_wait.nsys-rep"

rg "step1\\.offload_try_wait_once|cpu\\..*_cpp|gpu\\..*_cpp" \
  "${OUT_ROOT_SINGLE}/csv" -n || true
```

### Why no drain / shutdown?

An extra `offload_try_wait` loop after the first call (`step1.offload_wait_drain` in older runs) or `client.shutdown()` often **hangs**. The script always exits after `step1.offload_try_wait_once`; read only that NVTX range for the single-shot metric.

## 6) 单次 try_wait 结果画图（rep -> csv 后）

```bash
python3 plot_nsys_breakdown.py \
  --csv-root "${OUT_ROOT_SINGLE}" \
  --mode-dir "csv" \
  --mode flexkv_fine \
  --seq-lens "1024" \
  --case-pattern "len{value}_bs8_single_try_wait" \
  --view mode_breakdown \
  --include-prefixes "step1.offload_try_wait_once,step1.offload_wait::cpu.,step1.offload_wait::gpu." \
  --output-png "${OUT_ROOT_SINGLE}/plot/single_try_wait_breakdown.png" \
  --output-csv "${OUT_ROOT_SINGLE}/plot/single_try_wait_breakdown.csv"
```

## 7) 连续 offload 计时（观察是否随次数变慢）

```bash
cd /home/scratch.noliu_gpu/recsys-example-kvcache-origin/corelib/recsys_kvcache_manager/test/breakdown_2

# 先跑你提到的 10/50/100（不求平均，直接看每次耗时）
python3 profiler_offload.py \
  --batch-size 8 \
  --len-per-seq 1024 \
  --max-seq-len 1024 \
  --host-capacity-scale 8 \
  --offload-batch-counts "10,50,100" \
  --output-csv "./offload_stress_10_50_100.csv"

# 如果要继续扩到 150/200，只改这个参数
python3 profiler_offload.py \
  --batch-size 8 \
  --len-per-seq 1024 \
  --max-seq-len 1024 \
  --host-capacity-scale 16 \
  --offload-batch-counts "50,100,150,200" \
  --output-csv "./offload_stress_50_100_150_200.csv"
```

脚本会：
- 每次 offload 都打印两列：`offload_only=xxx ms` 和 `end_to_end=xxx ms`
- 同时写入 CSV：`offload_batch_count, iteration, request_batch_size, offload_only_elapsed_ms, end_to_end_elapsed_ms`

快速看尾部几条（判断是否明显变慢）：

```bash
python3 - <<'PY'
import csv
from collections import defaultdict

path = "./offload_stress_10_50_100.csv"
d = defaultdict(list)
with open(path) as f:
    r = csv.DictReader(f)
    for row in r:
        d[int(row["offload_batch_count"])].append(float(row["offload_only_elapsed_ms"]))

for k in sorted(d):
    vals = d[k]
    tail = vals[-5:] if len(vals) >= 5 else vals
    print(f"scenario={k}, first={vals[0]:.3f}ms, last={vals[-1]:.3f}ms, tail={tail}")
PY
```

## 8) offload 理论带宽 vs 实际带宽（含 try_wait 轮询）

```bash
cd /home/scratch.noliu_gpu/recsys-example-kvcache-origin/corelib/recsys_kvcache_manager/test/breakdown_1

# 基于 profiler_result/plot/csv_summarization 的 step_op + L3 结果计算
python3 analyze_offload_bandwidth.py \
  --batch-size 8 \
  --seq-lens 1024,2048,4096 \
  --output-csv profiler_result/plot/csv_summarization/offload_bandwidth_bs8.csv
```

说明：
- `theoretical` = payload / `step1.flexkv.finish_task`
- `practical` = payload / `step1.offload_wait`（包含 `offload_try_wait` 轮询 + wait + release）
- `gap(%)` 表示理论带宽相对实际带宽的高估比例