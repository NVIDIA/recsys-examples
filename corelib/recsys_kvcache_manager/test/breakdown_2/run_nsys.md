# breakdown_2 — NVTX / nsys 扫参

与 `run_sweep_offload.sh`（`perf_counter` → `origin_data/` / `summarization/`）并行使用；本流程用 **Nsight Systems** 导出 NVTX 累计时间 CSV。

## 目录

| 路径 | 内容 |
|------|------|
| `profiler_result_nsys/rep/` | `offload_bs{bs}_len{len}.nsys-rep` |
| `profiler_result_nsys/csv/` | `offload_bs{bs}_len{len}_nvtx_pushpop_sum.csv`（或 `*nvtxsum*`） |

NVTX 标签来自 `profiler_offload.py` 的 hook，例如：

- `flexkv.offload_kvcache_launch`
- `flexkv.offload_kvcache_wait`
- `flexkv.finish_task`
- `flexkv.client.put_async` / `try_wait` / `wait`

## 全量扫参（耗时长）

```bash
cd /home/scratch.noliu_gpu/recsys-example-kvcache-origin/corelib/recsys_kvcache_manager/test/breakdown_2

bash run_sweep_offload_nsys.sh
```

## 快速冒烟（单 bs / 单 len / 少量 offload 轮数）

```bash
cd .../breakdown_2

BATCH_SIZES=8 \
LENS=1024 \
OFFLOAD_COUNTS=50,100 \
NO_ORIGIN=1 \
bash run_sweep_offload_nsys.sh
```

## 单次 profile + 手动导 CSV

```bash
cd .../breakdown_2
OUT=profiler_result_nsys
mkdir -p "${OUT}/rep" "${OUT}/csv"

nsys profile \
  -t cuda,nvtx,osrt \
  --sample=none \
  --cuda-memory-usage=true \
  --force-overwrite true \
  -o "${OUT}/rep/offload_bs8_len1024" \
  python3 profiler_offload.py \
    --batch-size 8 \
    --len-per-seq 1024 \
    --max-seq-len 1024 \
    --host-capacity-scale 8 \
    --offload-batch-counts 50,100,150 \
    --warmup-iterations 1 \
    --no-origin-data \
    --output-root .

nsys stats --report nvtx_pushpop_sum --format csv \
  --output "${OUT}/csv/offload_bs8_len1024" \
  "${OUT}/rep/offload_bs8_len1024.nsys-rep"
# 生成: ${OUT}/csv/offload_bs8_len1024_nvtx_pushpop_sum.csv
```

## 环境变量（`run_sweep_offload_nsys.sh`）

| 变量 | 默认 | 说明 |
|------|------|------|
| `OUT_ROOT` | `./profiler_result_nsys` | rep / csv 根目录 |
| `BATCH_SIZES` | `8 32 64 128` | 与 perf_counter 扫参一致 |
| `LENS` | `1024 2048 4096` | `len_per_seq` / `max_seq_len` |
| `OFFLOAD_COUNTS` | `50,100,...,300` | 单次 nsys 内连续跑多档 N |
| `HOST_CAPACITY_SCALE` | `8` | 同 `profiler_offload.py` |
| `WARMUP` | `1` | warmup 轮数 |
| `NO_ORIGIN` | `1` | 默认不写 `origin_data/`（减小 IO） |
| `NSYS_EXTRA` | 空 | 传给 `nsys profile` 的额外参数 |

## 注意

- NVTX CSV 的 **ms** 与 `summarization/*.csv` 的 `perf_counter` **不必数值相等**；用途是核对阶段、看 GPU timeline，定量对比仍以 hook CSV 为主。
- 嵌套 NVTX（如 `finish_task` 包住 `client.wait`）在 `nvtx_pushpop_sum` 里可能重叠，不要简单相加。
