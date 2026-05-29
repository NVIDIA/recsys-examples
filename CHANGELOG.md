# Changelog

## v26.04 - 2026-05-20

### Added

- Added `corelib/recsys_kvcache_manager` as a standalone RecSys KV cache package, including native and FlexKV-backed host storage paths.
- Added LLM-style KV-cache APIs for HSTU inference and updated the HSTU inference examples to use the package.
- Added `corelib/gr_decode_atten`, a beam-search decode attention kernel package for SID-GR KV-cache generation.
- Added `generate_beam_decode()` for SID-GR inference with context/beam KV-cache reuse.

### Changed

- Refactored the previous async KV-cache manager out of the HSTU inference example into the reusable KV-cache package.
- Updated SID-GR generation utilities with vectorized masking and CuTe beam-decode kernels.

## v26.03 - 2026-04-14

### Added

- Added Torch Export and AOTInductor packaging for end-to-end HSTU C++ inference.
- Added an HSTU end-to-end training benchmark suite with progressive optimization experiments.
- Added HSTU inference benchmark results on B200.

### Changed

- Improved DynamicEmb table fusion and expansion support.
- Relaxed DynamicEmb table sizing so embedding-table capacity is aligned to `bucket_capacity` instead of requiring power-of-two sizing.
- Migrated HSTU attention installation guidance to the `fbgemm_gpu_hstu` package.

## v26.01 - 2026-02-13

### Added

- Added workload-balanced batch shuffling for HSTU data-parallel training.
- Added caching and prefetching support for `EmbeddingBagCollection`.

### Changed

- Optimized HSTU KV-cache management with C++ onload/offload paths and asynchronous transfer overlap.

## v25.12 - 2026-01-13

### Added

- Added Triton Inference Server support for HSTU inference.
- Added the initial Semantic ID generative retrieval example.

## v25.11 - 2025-12-10

### Added

- Added DynamicEmb embedding admission support.

## v25.10 - 2025-11-11

### Added

- Added HSTU sequence parallelism support.
- Added DynamicEmb LRU score checkpointing and gradient clipping.
- Added HSTU support for SM89 GPUs in training.

### Changed

- Decoupled scaling sequence length from the maximum sequence length limit in HSTU attention.

## v25.09 - 2025-10-20

### Added

- Added prefetching and caching to the HSTU training example.
- Added distributed embedding dumping and memory scaling for DynamicEmb.
- Added HSTU inference kernel fusion and FP8 quantization support.

## v25.08 - 2025-09-08

### Added

- Added cache support for DynamicEmb.
- Added end-to-end HSTU inference examples.
- Added DynamicEmb evaluation-mode support.

## v25.07 - 2025-08-01

### Added

- Added the HSTU inference benchmark with paged KV cache, CUDA graph support, and related inference optimizations.
- Added tensor parallelism support in the HSTU layer.

## v25.06 - 2025-07-04

### Added

- Added LFU eviction support and DynamicEmb lookup performance improvements.
- Added pipeline support for the HSTU example.
- Added HSTU layer recompute support and custom CUDA ops for jagged tensor concatenation.

## v25.05 - 2025-05-29

### Added

- Added DynamicEmb support for `EmbeddingBagCollection`, truncated normal initialization, and Adagrad `initial_accumulator_value`.
- Added fused LayerNorm/dropout operations in the HSTU layer.

### Fixed

- Fixed convergence issues on the KuaiRand dataset.

## 0.1.0 - Initial Release

### Added

- Initial distributed recommender examples.
- HSTU attention operator support.
- DynamicEmb with GPU acceleration.
- Distributed training examples with TorchRec and Megatron-Core integration.
