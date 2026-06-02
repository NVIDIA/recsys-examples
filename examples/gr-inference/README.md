# GR-Inference

## 一、背景 & 动机

### 算法背景

基于 semantic id style 的搜广推模型建模方式，是当前推荐、搜索、广告生成式建模的主流趋势之一。它的算法核心可以概括为：

```text
离线建簇
-> 为每个真实 item id 映射出多级簇类 id
-> 在线推理时自回归生成有限步 semantic id
-> 将 semantic id 多元组映射回真实 item id
```

这个范式对推理系统有几个直接影响：

- 簇类级别通常较低，例如 3/5，因此 decode 自回归步数很短。
- 用户历史序列可能很长，因此 prefill 阶段的 context 计算可能很重。
- 搜广推是业务驱动场景，推荐结果通常需要强调多样性；在自回归 decode 中增大 beam width，是提升结果多样性的一种有效手段。因此 beam width 可能很大，且每一步的 beam width 也可能是可变的。

### 问题抽象

从算法背景出发，对应的推理架构问题可以抽象为：

```text
长 context + 短 decode + 大 beam width
```

这不是通用 chat LLM serving 的典型形态。vLLM、SGLang、TRT-LLM 这类公开框架优先解决的是：

```text
多用户请求
动态 batch
paged KV
长时间 decode
OpenAI/chat API
```

这些能力很重要，但 GR workload 的瓶颈不完全一样。推荐/搜索式生成通常是：

```text
一个请求有很长的用户/候选上下文
beam width 很大，例如 128/256
真正 decode 只有有限的几步（比如 2-3 步）
大量 beams 共享同一份 context
```

目前针对这种 pattern，现有通用 LLM 推理框架都不能很好满足：

- **vLLM：** 没有稳定、好用、生产级的原生 beam search serving 路径。用户通常需要在业务侧循环调用 vLLM 来实现 beam search。
- **TRT-LLM：** 功能侧原生不支持 logprobs 输出，并且 beam width > 1 时容易遇到显存分配过载问题；性能侧 decode attention 和后处理 kernel 在这个场景下也不够高效。
- **SGLang：** 目前看是主流开源框架中最能直接满足该架构问题的可用框架，但大 beam width 支持仍停留在 MR/PR 阶段，尚未合入 main 分支。

除此之外，主流 LLM 推理框架的功能极其复杂，代码量动辄十几万甚至几十万行。把如此繁重的框架适配到搜广推推理场景，是典型的“杀鸡用宰牛刀”。在传统三大框架上增删改，短期内或许可以满足单个业务场景的性能和功能需求，但从长期维护和追求 SOL 的角度，并不是最优解。

## 二、目标 & 怎么做

### 目标

- 在 GR “长 context + 短 decode + 大 beam width” 推理场景下，把经典 Qwen 模型的性能优化到 SOL。
- 从框架角度支持不同 GR 场景的实际需求，包括功能需求和性能需求。

### 怎么做

整体思路不是从零发明一个完全孤立的 serving system，也不是把 GR 场景硬塞进通用 LLM serving 框架，而是：

- **保留 GR 自己的核心抽象。** `ContextKV`、`BeamKV`、`BeamPath`、dynamic beam、item-constrained decode 和 request x active-beam batching 应该由 GR runtime 原生表达，而不是在业务侧用循环调用拼出来。
- **选择性参考开源框架的成熟能力。** 借鉴 vLLM / SGLang 在 continuous batching、paged KV、HTTP serving、benchmark tooling 和工程化接口上的经验；借鉴 TRT-LLM 在 kernel、CUDA graph、算子融合和模型层优化上的经验。但这些能力只作为可复用技术来源，不反向主导 GR 的核心 runtime contract。
- **用 benchmark 驱动实现。** 先围绕 Qwen-family、长 context、短 decode、大 beam width 建立稳定的 correctness / performance / Nsight breakdown 口径，再用这些口径决定哪些优化值得进入主路径。
- **用 AI coding 加速工程闭环。** 通过明确的接口契约、测试样例、benchmark 输出和文档目标，让 AI coding 帮助快速生成 adapter、scheduler 逻辑、HTTP glue、测试、benchmark script 和文档草稿；关键路径仍通过真实模型、真实 kernel、性能数据和 correctness regression 做验证。
- **保持框架小而专。** 对通用 LLM serving 已经解决得很好的部分只吸收必要能力；对 GR workload 真正敏感的 KV ownership、beam state、decode attention、batching unit 和业务约束保持自有设计。

### 当前进展

当前仓库已经完成 single-node alpha 主路径，而不是只有概念验证：

```text
Qwen3-1.7B real weights
+ GR-native ContextKV / BeamKV / BeamPath
+ real gr-decode_atten backend
+ continuous batching
+ BeamKV / ContextKV dense pool
+ direct pool-view decode CUDA graph
+ HTTP /generate
+ SGLang-equivalent beam_results 输出
+ offline / online GR vs SGLang benchmark
```

这条主路径已经验证了核心价值：在“长 context + 短 decode + 大 beam width”的已测矩阵下，GR 的 offline 性能稳定领先 SGLang beam-search PR 分支；online serving 已跑通同一 HTTP client benchmark，并已把 CUDA graph capture 收敛成启动期固定预热、服务期只 replay 的可复现口径。online tail latency 仍会受 HTTP client、Python 调度、请求到达时序和 batch fill 影响，后续继续产品化。

## 三、设计和实现亮点

| 维度 | 通用公开框架路径 | GR-Inference 路径 |
| --- | --- | --- |
| KV 抽象 | 以 sequence / token block / paged KV 为中心 | 显式拆成请求级 `ContextKV` 和短 `BeamKV` |
| 大 beam decode | `batch * beam` 展平成大量通用 decode rows | 按 request + beam tile 处理共享 context |
| attention kernel | 通用 paged decode attention | `gr-decode_atten` 直接接收 `ContextKV + BeamKV + BeamPath` |
| batch CUDA graph | 通用 batch bucket / paged KV 约束 | 固定 GR shape + pool slice，地址稳定后直接 replay |
| 输出口径 | 通用 API 返回和 beam 管理 | 性能路径不构造 debug `beam_details`，正确性路径再打开 |

核心实现点：

- **GR-native KV layout 已落地。** 一个请求的长 context 只保存一份 `ContextKV`；beam 的短 decode history 放在 `BeamKV`；`BeamPath` 只记录逻辑父子关系。这样避免把长 context 当成每条 beam 的独立 sequence 管理。
- **ContextKV 保持 dense hot path。** 当前 `ContextKV` 是连续 dense layout，优先服务 kernel 友好的 decode attention 和稳定 pool slice。固定长 context 下，dense ContextKV pool 可以减少临时分配并支持 CUDA graph replay；如果线上 context length 方差很大，后续需要多 context bucket 或 page-backed ContextKV，避免短请求占用大 slot。
- **BeamKV / ContextKV pool 已进入 hot path。** continuous serving 使用 dense BeamKV / ContextKV pool，带 lease、capacity、high watermark、utilization、leak check，便于 admission control、CUDA graph replay 和回归验证。
- **专用 decode attention。** `gr-decode_atten` 知道 `B`、`W`、共享 context 和短 BeamKV history，不需要把 `4 * 256` 个 beam 完全当成 1024 条普通 decode row。
- **面向 GR 的 continuous batching。** scheduler 按 decode step、beam width、context shape 组 batch，batch 单位是 `request x active beams`，不是普通独立 sequences。
- **decode CUDA graph 已走 direct pool-view replay。** 固定 shape graph 绑定稳定的 ContextKV / BeamKV pool slice；capture 后先 replay 再使用输出，避免 capture output 污染。online 启动 warmup 会覆盖 batch / pool-window / `/generate ignore_eos` 形状，服务开始后默认冻结新 graph capture；slot 不连续导致的动态 KV 拼接直接走 eager，避免随机地址污染 graph cache。
- **last-token logits only。** serving prefill 只计算下一 token 需要的最后位置 logits，避免对整个 context 做无用 `lm_head`。
- **Dynamic beam width 已有基础实现。** runtime 支持 fixed、scheduled、score-margin policies，并能从 HTTP request 构造。后续重点不是“实现动态 beam”，而是用真实质量指标确定默认策略、收缩规则和回归门槛。
- **Item-constrained generation 已有 runtime 基础。** item trie、mask、constrained topK、catalog reload/rollback 和 item metadata 已进入 runtime/HTTP。后续要把真实 catalog 压测和 item-level correctness 做成主线 benchmark。
- **性能和正确性一起对齐。** 性能口径使用默认 A `beam_results` 输出；debug-rich `beam_details` 只在正确性/debug 时打开。对齐指标包括 top1、topK overlap、score correlation 和 eager-vs-graph 一致性。

## 四、当前生产默认路径

- continuous serving 选择 `--decode-backend real` 时使用真实 `gr-decode_atten` backend。
- eligible continuous decode batch 默认启用 decode CUDA graph；可用 `GR_INFERENCE_DISABLE_DECODE_CUDA_GRAPH=1` 回退 eager decode。
- `ContextKV` / `BeamKV` 直接使用 pool slice；graph replay 前只更新小输入，如 beam token ids 和 topK indices，不拷贝整块 KV。
- decode graph cache 支持上限、LRU eviction 和 pointer guard；如果 pool slice 地址不匹配，不会错误复用 graph。online serving 默认在启动 warmup 后冻结新 capture，未覆盖的动态形状直接 eager，保证 benchmark 期间 graph capture 数不漂移。
- `scripts/serve_qwen3_gr_http.sh` 默认开启 online startup warmup：`GR_WARMUP_ONLINE_SHAPES=1`、`GR_WARMUP_ONLINE_POOL_WINDOWS=1`、`GR_WARMUP_ONLINE_MAX_CASES=64`、`GR_FREEZE_CUDA_GRAPHS_AFTER_WARMUP=1`。调试 graph coverage 时可设 `GR_FREEZE_CUDA_GRAPHS_AFTER_WARMUP=0`。
- QK norm + RoPE 默认先尝试当前性能最好的 SGLang-style inplace kernels；不可用时回退 FlashInfer/Torch。`sgl_kernel.fused_qk_norm_rope` 和 split-QKV/split-gate-up 这类实验分支不在 serving 热路径里。`trtllm_aligned` QK norm/RoPE JIT backend 只保留为显式实验路径。MLP activation 默认使用 bf16 等价的 exact packed SiLU/mul CUDA kernel，JIT 不可用时回退 torch；packed GEMM 默认仍走 torch。
- prefill 默认使用 SGLang-style piecewise CUDA graph；当前 Qwen3-1.7B bs4/ctx1000 形状会抓 6 段 graph：embed、4 个 layer chunk、output。可用 `GR_INFERENCE_DISABLE_PREFILL_CUDA_GRAPH=1` 回退 eager prefill。
- `/generate` 默认返回 SGLang-equivalent `beam_results`；当请求设置 `ignore_eos=true` 时，默认 suppress tokenizer special tokens 以对齐 SGLang 固定长度生成语义。
- 性能测试默认不构造完整 `beam_details`；正确性/debug 时再打开。

如需回退 eager decode：

```bash
GR_INFERENCE_DISABLE_DECODE_CUDA_GRAPH=1
```

如果需要关闭 GR 自带 experimental JIT kernels：

```bash
GR_INFERENCE_GR_TRTLLM_KERNELS_JIT=0
```

默认性能路径现在不需要这些旧开关：`GR_INFERENCE_SPLIT_QKV_PROJ`、`GR_INFERENCE_PACKED_QKV_PREFILL`、`GR_INFERENCE_PACKED_QKV_PREFILL_KV_WRITE`、`GR_INFERENCE_GR_TRTLLM_PACKED_QKV_PREFILL_JIT`、`GR_INFERENCE_GR_TRTLLM_PACKED_QKV_KV_WRITE_JIT`、`GR_INFERENCE_ENABLE_SGLANG_QKNORM`、`GR_INFERENCE_ENABLE_SGLANG_ROPE`、`GR_INFERENCE_FLASHINFER_STRIDED_QK_NORM_ROPE`、`GR_INFERENCE_SGL_KERNEL_QK_NORM_ROPE_PHASE`、`GR_INFERENCE_SPLIT_GATE_UP_PROJ`、`GR_INFERENCE_ENABLE_TRTLLM_GATED_MLP_OP`。这些分支已经从 serving 热路径删除。

decode graph cache 默认有上限，可通过下面环境变量调整：

```bash
GR_INFERENCE_DECODE_CUDA_GRAPH_MAX_ENTRIES=32
```

## 五、对比对象

当前 headline 数字对比的是 SGLang beam-search PR 分支：

```text
repo:   https://github.com/cswuyg/sglang.git
branch: feature/beam_search
PR:     https://github.com/sgl-project/sglang/pull/15645
```

这不是 SGLang upstream main 已合入功能，而是 beam search feature branch。

测试环境和 workload：

```text
GPU: NVIDIA H100 80GB HBM3
model: Qwen3-1.7B
context_len: 1000 / 5000
beam_width: 256
effective output length: 3 tokens
```

## 六、Offline 性能结果

性能口径：GR 使用默认 A 输出，也就是 SGLang-equivalent `beam_results`；不构造 debug-rich `beam_details`。GR 开启 prefill CUDA graph 和 direct pool-view decode CUDA graph，prefix / prefill cache 关闭，并在 `ignore_eos` 固定输出口径下 suppress Qwen special tokens。SGLang 关闭 radix cache，计时包住 beam-search PR 分支的 `Engine.generate`。

Radix cache off，也就是生产 no-prefix-reuse 口径：

| ctx | batch | GR ms | SGLang ms | SGLang/GR | winner | GR prefill | GR decode |
| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 1000 | 1 | 17.611 | 33.515 | 1.903x | GR | 7.027 | 9.952 |
| 1000 | 2 | 27.768 | 57.926 | 2.086x | GR | 12.633 | 14.170 |
| 1000 | 4 | 47.736 | 102.318 | 2.143x | GR | 23.442 | 22.566 |
| 1000 | 8 | 93.230 | 199.280 | 2.138x | GR | 46.521 | 43.707 |
| 5000 | 1 | 42.255 | 94.554 | 2.238x | GR | 31.029 | 10.701 |
| 5000 | 2 | 80.904 | 179.369 | 2.217x | GR | 63.763 | 16.216 |
| 5000 | 4 | 154.224 | 349.857 | 2.269x | GR | 126.087 | 26.772 |
| 5000 | 8 | 307.917 | 685.354 | 2.226x | GR | 253.345 | 51.448 |

Offline 结论：GR 在生产 no-prefix-reuse 口径的全部已测 case 中领先。`ctx=5000, batch=8` 下，GR 相比 SGLang 快 `2.23x`。实际生产场景里请求之间通常没有可复用 prefix，因此这里不再列 radix / prefix cache on 对比。

## 七、Online In-flight Serving 结果

online 口径使用 SGLang `bench_serving` 作为同一个 HTTP client，`request_rate=inf`、`max_concurrency=4`、`requests=64`。GR server 使用 `/generate` 兼容入口、A 默认 `beam_results` 输出、prefill CUDA graph、direct pool-view decode CUDA graph 和 SGLang `ignore_eos` 对齐的 special-token suppress；SGLang server 使用 beam-search PR 分支。正式计时命令使用 `warmup_requests=0`，外部 server startup warmup / priming 不计入结果。

GR 的稳定复现口径从 2026-05-28 起改成：

- server 启动时预热 online batch size 和 KV pool slot window。
- warmup 请求走和真实 `/generate` 一样的 `ignore_eos=true` / special-token suppress 路径。
- warmup 后冻结 prefill / decode CUDA graph capture；服务期只 replay 已知 graph。
- slot 不连续导致的动态 KV 拼接不进 CUDA graph，直接 eager 跑，避免随机 tensor 地址让 graph cache 随轮次增长。

固定 case：

```text
ctx=5000, beam=256, output=3, requests=64, max_concurrency=4
```

最新三轮复跑结果：

| server / output mode | round | req/s | median ms | p90 ms | p99 ms | input tok/s | output tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GR `/generate`, A `beam_results`, frozen graph capture | 1 | 19.41 | 199.32 | 240.16 | 323.75 | 97045 | 14906 |
| GR `/generate`, A `beam_results`, frozen graph capture | 2 | 20.10 | 195.62 | 223.38 | 252.40 | 100505 | 15438 |
| GR `/generate`, A `beam_results`, frozen graph capture | 3 | 19.66 | 199.92 | 235.61 | 304.19 | 98294 | 15098 |
| SGLang `/generate`, beam results, primed steady | 1 | 10.69 | 369.69 | 374.14 | 379.08 | 53450 | 8208 |
| SGLang `/generate`, beam results, primed steady | 2 | 10.65 | 370.59 | 373.94 | 378.19 | 53250 | 8177 |
| SGLang `/generate`, beam results, primed steady | 3 | 10.68 | 370.36 | 373.68 | 375.22 | 53400 | 8201 |

GR graph stability gate：

| checkpoint | prefill captures | decode captures | captures enabled | dynamic graph skips |
| --- | ---: | ---: | ---: | ---: |
| startup | 10 | 30 | 0 | 0 |
| round1 | 10 | 30 | 0 | 8 |
| round2 | 10 | 30 | 0 | 11 |
| round3 | 10 | 30 | 0 | 15 |

Online 结论：最新稳定复现口径下，GR 三轮 req/s 为 `19.41 / 20.10 / 19.66`，decode CUDA graph capture 固定为 `30`，不再随 online 调度继续增长。与 SGLang primed steady 三轮相比，GR 平均吞吐约 `1.85x`，median latency 低约 `46%`。p99 仍会有 online tail 抖动，这是 HTTP client、Python 调度、请求到达时序、batch fill 共同导致的，不再是服务期现场 capture 新 CUDA graph 导致的漂移。

复现 GR online 稳定口径：

```bash
BASE_OUT=benchmark_artifacts/sglang_compare/gr_online_repro
mkdir -p "${BASE_OUT}/gr"

env GR_MODEL_DIR=/workspace/models/Qwen3-1.7B \
  GR_CONTEXT_LEN=5000 \
  GR_DECODE_STEPS=3 \
  GR_BEAM_WIDTH=256 \
  GR_MAX_BATCH_SIZE=4 \
  GR_BEAM_KV_POOL_CAPACITY=4 \
  GR_CONTEXT_KV_POOL_CAPACITY=4 \
  GR_HTTP_HOST=0.0.0.0 \
  GR_HTTP_PORT=8000 \
  GR_DECODE_BACKEND=real \
  GR_DEVICE=cuda \
  GR_DECODE_CUDA_GRAPH_BATCH_BUCKETS=1,2,4,8 \
  GR_WARMUP_ONLINE_SHAPES=1 \
  GR_WARMUP_ONLINE_POOL_WINDOWS=1 \
  GR_WARMUP_ONLINE_MAX_CASES=64 \
  GR_FREEZE_CUDA_GRAPHS_AFTER_WARMUP=1 \
  GR_ENABLE_PREFILL_CACHE=0 \
  scripts/serve_qwen3_gr_http.sh \
  > "${BASE_OUT}/gr_server.log" 2>&1 &
SERVER_PID=$!

until curl -fsS http://127.0.0.1:8000/ready >/dev/null; do sleep 2; done
curl -fsS http://127.0.0.1:8000/metrics \
  -o "${BASE_OUT}/metrics_after_startup.json"

for round in 1 2 3; do
  OUT_DIR="${BASE_OUT}/gr/round${round}" \
  REQUESTS=64 CONTEXT_LEN=5000 DECODE_STEPS=3 BEAM_WIDTH=256 \
  REQUEST_RATE=inf MAX_CONCURRENCY=4 WARMUP_REQUESTS=0 \
  scripts/run_gr_sglang_bench_serving_beam_benchmark.sh \
  2>&1 | tee "${BASE_OUT}/gr_round${round}.log"
  curl -fsS http://127.0.0.1:8000/metrics \
    -o "${BASE_OUT}/metrics_after_round${round}.json"
done

kill "${SERVER_PID}"
```

直接使用当前 `scripts/serve_qwen3_gr_http.sh` + `scripts/run_gr_sglang_bench_serving_beam_benchmark.sh` 就是上述稳定复现方式：server 脚本默认打开 startup online warmup 和 warmup 后冻结 graph capture，GR client benchmark 脚本默认 `WARMUP_REQUESTS=0`。只有在调试 graph coverage 时，才建议显式设置 `GR_FREEZE_CUDA_GRAPHS_AFTER_WARMUP=0`。

online `/generate` top1 correctness smoke：

| ctx | beam | requests | max concurrency | top1 exact | topK overlap |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5000 | 256 | 64 | 4 | 58/64 | min 0.918 / mean 0.956 |

这组打的是真实 HTTP `/generate` 路径，使用默认 A `beam_results` 输出。top1 不完全一致的请求仍保持较高 TopK overlap。

## 八、正确性结果

正确性评测使用 GR 默认 A `beam_results` 输出与 SGLang `beam_results` 对比；GR 开启 CUDA graph，且 suppress Qwen special tokens 以对齐 SGLang `ignore_eos=true` 固定长度生成语义。

| ctx | batch | top1 exact | topK overlap | note |
| ---: | ---: | ---: | ---: | --- |
| 1000 | 1 | 1.00 | 0.949 | top1 exact |
| 1000 | 2 | 1.00 | 0.953 | top1 exact |
| 1000 | 4 | 1.00 | 0.956 | top1 exact |
| 1000 | 8 | 1.00 | 0.958 | top1 exact |
| 5000 | 1 | 1.00 | 0.969 | top1 exact |
| 5000 | 2 | 1.00 | 0.961 | top1 exact |
| 5000 | 4 | 1.00 | 0.955 | top1 exact |
| 5000 | 8 | 1.00 | 0.959 | top1 exact |

offline correctness 结论：生产 no-prefix-cache 口径下，beam=256 的 8 个固定 case Top1 全部一致，TopK 候选集合高度重合。全矩阵 24 个 case 的 top1 min / mean 为 `1.000 / 1.000`，TopK overlap min / mean 为 `0.945 / 0.960`。

## 九、性能 Breakdown

固定 case：

```text
ctx=1000, beam=256, batch=4, output=3
```

生产 profile Nsight 整体结果：

| metric | GR | SGLang |
| --- | ---: | ---: |
| active CUDA window | 46.856 ms | 99.944 ms |
| kernel total | 43.168 ms | 78.859 ms |
| CUDA runtime API total | 42.886 ms | 42.975 ms |
| CPU runtime gaps >50us | 2.186 ms | 28.305 ms |
| CUDA graph launches, active window total | 8 | 29 |
| kernel launch count | 1261 | 1620 |

Prefill stage：

| metric | GR | SGLang |
| --- | ---: | ---: |
| stage total | 24.404 ms | 27.549 ms |
| attention kernels | 1.367 ms | 1.389 ms |
| non-attention kernels | 20.825 ms | 19.102 ms |
| CPU overhead | 2.213 ms | 7.057 ms |
| CUDA graph launches | 6 | 29 |

Decode stage：

| metric | GR | SGLang |
| --- | ---: | ---: |
| stage total | 23.318 ms | 55.663 ms |
| attention kernels | 1.593 ms | 35.235 ms |
| non-attention kernels | 19.384 ms | 13.228 ms |
| CPU overhead | 2.341 ms | 7.200 ms |
| CUDA graph launches | 2 | 0 |

补充 kernel buckets：

| metric | GR | SGLang |
| --- | ---: | ---: |
| topK / beam selection | 4.027 ms | 4.716 ms |
| attention bucket total | 2.960 ms | 40.951 ms |

注：普通 latency 看 offline / online benchmark 表；Nsight 这里主要用于解释 active CUDA window、stage 和 kernel breakdown。原始 Nsight 输出在 `benchmark_artifacts/sglang_compare/prod_breakdown_ctx1000_beam256_b4_20260525_121548/`。

batch 4 的主要差距不是 topK 排序，而是大 beam decode attention 路径和通用 serving 调度开销。说人话：

```text
decode CUDA graph 让路径更短；
GR 专用 decode attention 让核心计算更省。
```

在这个 case 里，decode attention kernels 的差距是 `35.235 - 1.593 ~= 33.6 ms`，active CUDA window 差距是 `99.944 - 46.856 ~= 53.1 ms`，所以大头仍来自 decode attention / GR 专用 KV 结构。GR 没有把 `batch * beam = 4 * 256` 完全展平成 `1024` 条普通 decode rows，而是保留 request-level shared `ContextKV` 和很短的 `BeamKV` history。

CUDA graph 也有帮助：当前生产测试 profile 里 GR 开启了 prefill CUDA graph 和 decode CUDA graph，因此 Nsight 在 active window 里看到 8 次 CUDA graph launch。这里不是 prefill 单独 8 次，而是 `6 次 prefill piecewise graph + 2 次 decode graph`；SGLang 这次 29 次 CUDA graph launch 都落在 prefill，decode stage 没有命中 CUDA graph。小 batch，例如 batch 1，attention 差距不明显时，GR 的优势更多来自这个更短的 graph/runtime 路径。

SGLang decode attention 看到的是：

```text
batch * beam = 4 * 256 = 1024 decode rows
```

每个 row 走通用 paged decode attention。

GR decode attention 保留了 GR 结构：

```text
4 个请求
每个请求 256 beams
每个请求一份共享 ContextKV
每个 beam 只有很短的 BeamKV history
```

context 部分按 request / beam tile 处理，短 BeamKV 部分只 attend 少量 decode token。decode CUDA graph 再减少固定形状 decode step 的 launch 和 CPU 调度开销。

一句话：SGLang 是通用 beam serving 路径；GR 是专门为“长 context + 大 beam + 短 decode”做的路径。

## 十、TODO / Roadmap

当前已经完成 single-node alpha 的核心闭环。后续 TODO 不是“从零实现这些模块”，而是把已有基础能力产品化、扩大验证矩阵，并收敛成可维护的 serving framework。

| 方向 | 已有基础 | 后续产品化工作 |
| --- | --- | --- |
| Online serving hardening | HTTP `/generate`、background worker、continuous batching、pool metrics、online correctness/perf benchmark 已跑通；稳定口径下 server warmup 后冻结 CUDA graph capture，三轮复跑 graph capture 数不增长 | 优化 scheduler admission、batch fill 和 tail latency；补 request_rate、max_concurrency、arrival pattern、长时间 soak 回归；当前 online tail 仍会有 HTTP/Python/scheduler 抖动 |
| ContextKV 内存策略 | dense ContextKV pool 已接入 offline/online hot path，提供稳定 pool slice 给 CUDA graph；metadata-only paged KV allocator 和 scheduler page accounting 已有基础 | 根据真实 context length 分布增加多 context bucket；接入 page-backed ContextKV storage；长期让 decode attention 原生支持 page table，减少变长 context 显存浪费 |
| CUDA graph productionization | direct pool-view decode graph 默认启用；startup online warmup 覆盖 batch / pool-window / `/generate ignore_eos` 形状；服务期冻结新 capture；动态 KV 拼接跳过 graph | 扩大 shape 预热矩阵；补更多 request pattern 的 graph coverage、fallback、eviction 指标回归；评估是否把更多稳定形状纳入 warmup |
| Beam selection graph | decode forward 已进 graph；beam selection 仍在 graph 外 | 在 pool ownership、item mask、special-token suppress、输出裁剪都安全后，把 `log_softmax + topK + beam selection` 纳入 graph |
| GR vs SGLang benchmark | final offline A 口径已完成：`ctx=1000/5000, beam=256, batch=1/2/4/8`，GR 全部 case 领先；online 同一 HTTP client benchmark 已沉淀为稳定复现 recipe | 扩展更多 context length、beam width、dtype、模型尺寸和 GPU；做成一键 final offline / final online 汇总脚本 |
| Beam result 输出口径 | `/generate` 默认返回 SGLang-equivalent `beam_results`；`ignore_eos=true` 默认 suppress tokenizer special tokens；debug-rich `beam_details` 仅在 debug 时打开 | 继续优化 `beam_results` / `beam_details` 的 Python 构造和 JSON 序列化开销；明确 score normalization、length penalty 和 tie-break 策略 |
| Dynamic beam 策略 | fixed / scheduled / score-margin policy 已有基础，HTTP request 可配置，continuous scheduler 会按 active/next beam width 分组 | 用真实 GR 质量指标确定默认策略、score-margin 参数、自动收缩规则和质量回归门槛；量化速度收益与质量损失 |
| Item-constrained generation | item trie、legal-token mask、constrained topK、catalog reload/rollback、item metadata 已进入 runtime/HTTP | 接入真实大 catalog 压测；补 item-level correctness、illegal-token 检查、constrained topK 性能优化和线上语义回归 |
| 显存 admission 与回收 | KV budget、dense pool 指标、high watermark、leak check、cancel/timeout 生命周期已有基础 | 结合 page/offload 做更细粒度回收；完善高并发 admission policy；把显存估算和实际 pool 使用接入服务决策 |
| 模型和 backend matrix | 主线已验证 Qwen3-0.6B / H100 / BF16 | 扩展更多 Qwen-family 尺寸、dtype、quant、head 配置、checkpoint compatibility 和 backend fallback 表 |
| 多 GPU / scale-out | 当前主线是 single-node / single-GPU | 设计 TP/PP、多副本调度、跨卡 KV/beam ownership、负载均衡和部署编排 |
| 测试和文档收敛 | 已有 smoke、offline/online benchmark、Nsight breakdown、memory estimator | 收敛成少数稳定入口；一键跑 final offline / final online；一键刷新 README；补 CI/nightly correctness/perf regression |

## 十一、仓库结构

```text
src/gr_inference/gr_models/      Qwen-family model integration
src/gr_inference/gr_kv/          ContextKV, BeamKV, BeamPath
src/gr_inference/gr_kernels/     kernel wrappers and backend selection
src/gr_inference/gr_runtime/     beam search runtime and logits processing
src/gr_inference/gr_serving/     continuous batching, memory pools, HTTP serving
tools/                           benchmarks, comparison, profiling utilities
scripts/                         reproducible benchmark and serving entrypoints
tests/                           runtime, serving, model, kernel selection tests
```

## 十二、快速开始

### 默认环境和模型

默认 Docker 镜像：`lmsysorg/sglang:dev-cu13`

默认模型：`Qwen/Qwen3-1.7B`

```bash
git clone --recurse-submodules -b dev git@github.com:cb521/gr-inference.git
cd gr-inference
```

进入容器：

```bash
scripts/run_container.sh
```

### 指定模型

HuggingFace 模型：

```bash
MODEL=Qwen/Qwen3-0.6B scripts/run_container.sh scripts/quickstart_offline.sh
```

使用已有本地模型：

```bash
MODEL_ROOT=/path/to/models MODEL_DIR=/workspace/models/Qwen3-1.7B scripts/run_container.sh scripts/quickstart_offline.sh
```

固定 HuggingFace 版本：

```bash
MODEL=Qwen/Qwen3-1.7B MODEL_REVISION=main scripts/run_container.sh scripts/quickstart_offline.sh
```

### 常用命令

快速性能和精度检查：

```bash
RUN_ACCURACY=1 scripts/run_container.sh scripts/quickstart_offline.sh
```

完整 offline 性能和精度矩阵：

```bash
CONTEXT_LENS="1000 5000" \
BATCH_SIZES="1 2 4 8" \
REPEAT=3 \
RUN_ACCURACY=1 \
scripts/run_container.sh scripts/quickstart_offline.sh
```

结果默认写到：

```text
benchmark_artifacts/sglang_compare/offline_perf_YYYYmmdd_HHMMSS/summary.md
benchmark_artifacts/sglang_compare/offline_accuracy_YYYYmmdd_HHMMSS/summary.md
```

### 其它复现入口

在已经完成依赖安装的容器里，可以直接运行：

```bash
# 完整 offline 性能对比。
scripts/run_offline_perf_benchmark.sh

# 完整 offline 精度对齐。
scripts/run_offline_accuracy_benchmark.sh
```

需要同时跑 radix on/off、性能和正确性的全量公平评测时，使用：

```bash
OUT_DIR=benchmark_artifacts/sglang_compare/fair_eval_correctness_quick \
CONTEXT_LENS="1000" \
BEAM_WIDTHS="256" \
BATCH_SIZES="1 4" \
PERF_REPEAT=1 \
CORRECTNESS_REPEAT=1 \
scripts/run_gr_sglang_fair_eval.sh
```

跑固定 case 的 Nsight breakdown：

```bash
CONTEXT_LEN=5000 \
BEAM_WIDTH=256 \
REQUESTS=4 \
MAX_BATCH_SIZE=4 \
scripts/run_short_context_nsys_compare.sh
```
