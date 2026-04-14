# NVIDIA RecSys Examples

# Overview
NVIDIA RecSys Examples is a collection of optimized recommender models and components. 

The project includes:
- Examples for large-scale HSTU ranking and retrieval training through [TorchRec](https://github.com/pytorch/torchrec) and [Megatron-Core](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core) integration
- HSTU inference with paged KV cache, [Triton Inference Server](https://github.com/triton-inference-server/server) integration, CUDA graph usage, and C++ deployment with AOTInductor ([guide](./examples/hstu/inference/README.md))
- Examples for semantic-id based retrieval model through [TorchRec](https://github.com/pytorch/torchrec) and [Megatron-Core](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core) integration
- DynamicEmb for model-parallel dynamic embedding tables with zero-collision hashing, eviction, admission control, table fusion, and TorchRec integration ([documentation](./corelib/dynamicemb/README.md))

# What's New
- **[2026/4/14]** 🎉v26.03 released!
  - We added Torch export and AOTInductor packaging for end-to-end HSTU C++ inference. See the [HSTU inference overview](./examples/hstu/inference/README.md) and the [C++ inference guide](./examples/hstu/inference/GUIDE_TO_RUN_CPP_INFERENCE_DEMO.md).
  - We improved DynamicEmb with table fusion and expansion, relaxed embedding-table alignment (no longer power-of-two), and capacity sizing aligned to `bucket_capacity`. See [DynamicEmb](./corelib/dynamicemb/README.md).
  - We added an HSTU end-to-end training benchmark suite with progressive optimizations. See the [HSTU training benchmark](./examples/hstu/training/benchmark/README.md) and [E2E benchmark notes](./examples/hstu/training/benchmark/E2E_BENCHMARK.md).
  - We published HSTU inference benchmark results on B200 in the [HSTU inference benchmark](./examples/hstu/inference/benchmark/README.md).
  - We migrated HSTU attention to `fbgemm_gpu_hstu`, removed the legacy compatibility layer, and improved the training stack (fewer device-to-host syncs in jagged tensor handling, balancer tuning, and debug logging). See [HSTU training setup](./examples/hstu/training/README.md).
- **[2026/2/13]** 🎉v26.01 released!
  - We optimized HSTU KVCacheManager, moving Python-based KV cache management to optimized C++ implementation with asynchronous onload/offload operation and compression support. [Benchmark](https://github.com/NVIDIA/recsys-examples/tree/main/examples/hstu/inference/benchmark#1-end-to-end-inference-performance) shows onload and offload latency can be fully hidden under HSTU inference.
  - We introduced a HSTU training optimization with workload-balanced batch shuffling for data parallel training.
  - We added caching and prefetching support for `EmbeddingBagCollection`.
- **[2026/1/13]** 🎉v25.12 released!
  - Support TritonServer for HSTU inference. Follow [the HSTU inference TritonServer example](https://github.com/NVIDIA/recsys-examples/tree/main/examples/hstu/inference#example-hstu-model-inference-with-triton-server) to try it out.
  - We introduced our first semantic-id retrieval model example. Follow the semantic‑id retrieval (sid_gr) [documentation](https://github.com/NVIDIA/recsys-examples/tree/main/examples/sid_gr) to run it. 
- **[2025/12/10]** 🎉v25.11 released!
  - DynamicEmb supports embedding admission, that decides whether a new feature ID is allowed to create or update an embedding entry in the dynamic embedding table. By controlling admission, the system can prevent very rare or noisy IDs from consuming parameters and optimizer state that bring little training benefit.

<details>
<summary>More</summary>

- **[2025/11/11]** 🎉v25.10 released!
  - HSTU training example supports sequence parallelism.
  - DynamicEmb supports LRU score checkpointing, gradient clipping.
  - Decouple scaling sequence length from the maximum sequence length limit in HSTU attention and extend HSTU support to the SM89 GPU architecture for training.

- **[2025/10/20]** 🎉v25.09 released!
  - Integrated prefetching and caching into the HSTU training example.
  - DynamicEmb now supports distributed embedding dumping and memory scaling.
  - Added kernel fusion in the HSTU block for inference, including KVCache fixes.
  - HSTU attention now supports FP8 quantization.

- **[2025/9/8]** 🎉v25.08 released!
  - Added cache support for dynamicemb, enabling seamless hot embedding migration between cache and storage.
  - Released an end-to-end HSTU inference example, demonstrating precision aligned with training.
  - Enabled evaluation mode support for dynamicemb.

- **[2025/8/1]** 🎉v25.07 released!
  - Released HSTU inference benchmark, including paged kvcache HSTU kernel, kvcache manager based on trt-llm, CUDA graph, and other optimizations.
  - Added support for Tensor Parallelism in the HSTU layer.

- **[2025/7/4]** 🎉v25.06 released!
  - Dynamicemb lookup module performance improvement and LFU eviction support. 
  - Pipeline support for HSTU example, recompute support for HSTU layer and customized cuda ops for jagged tensor concat.

- **[2025/5/29]** 🎉v25.05 released! 
  - Enhancements to the dynamicemb functionality, including support for EmbeddingBagCollection, truncated normal initialization, and initial_accumulator_value for Adagrad.
  - Fusion of operations like layernorm and dropout in the HSTU layer, resulting in about 1.2x end-to-end speedup.
  - Fix convergence issues on the Kuairand dataset.
</details>

For more detailed release notes, please refer to our [releases][releases].

# Get Started
The examples we supported:
- [HSTU recommender examples](./examples/hstu/README.md)
- [HSTU inference](./examples/hstu/inference/README.md) — KV cache, Triton Inference Server, [C++ AOTInductor](./examples/hstu/inference/GUIDE_TO_RUN_CPP_INFERENCE_DEMO.md)
- [SID based generative recommender examples](./examples/sid_gr/README.md)

# Benchmarks
- [HSTU inference](./examples/hstu/inference/benchmark/README.md)
- [HSTU training](./examples/hstu/training/benchmark/README.md)
- [Dynamic embedding](./corelib/dynamicemb/benchmark/README.md)

# Contribution Guidelines
Please see our [contributing guidelines](./CONTRIBUTING.md) for details on how to contribute to this project.

# Resources
## Video
- [RecSys Examples 中的训练与推理优化实践](https://www.bilibili.com/video/BV1msMwzpE5B?buvid=638d217658211387f0a20e730604a780&from_spmid=united.player-video-detail.drama-float.0&is_story_h5=false&mid=V%2FD40L0stVy%2BZTgWdpjtGA%3D%3D&plat_id=116&share_from=ugc&share_medium=iphone&share_plat=ios&share_session_id=2DD6CE30-B189-4EEC-9FD4-8BAD6AEFE720&share_source=WEIXIN&share_tag=s_i&spmid=united.player-video-detail.0.0&timestamp=1749773222&unique_k=Sjcfmgy&up_id=1320140761&vd_source=7372540fd02b24a46851135aa003577c)
- [基于CUTLASS 3 的HSTU attention 算子开发与优化](https://www.bilibili.com/video/BV1TsMwzWEzS?buvid=638d217658211387f0a20e730604a780&from_spmid=united.player-video-detail.drama-float.0&is_story_h5=false&mid=V%2FD40L0stVy%2BZTgWdpjtGA%3D%3D&plat_id=116&share_from=ugc&share_medium=iphone&share_plat=ios&share_session_id=2DD6CE30-B189-4EEC-9FD4-8BAD6AEFE720&share_source=WEIXIN&share_tag=s_i&spmid=united.player-video-detail.0.0&timestamp=1749773222&unique_k=Sjcfmgy&up_id=1320140761&vd_source=7372540fd02b24a46851135aa003577c&spm_id_from=333.788.videopod.sections)

## Blog
- [NVIDIA Platform Delivers Lowest Token Cost Enabled by Extreme Co-Design](https://developer.nvidia.com/blog/nvidia-extreme-co-design-delivers-new-mlperf-inference-records/)
- [NVIDIA recsys-examples: 生成式推荐系统大规模训练推理的高效实践（上篇）](https://mp.weixin.qq.com/s/K9xtYC3azAccShpJ3ZxKbg)

# Community
Join our community channels to ask questions, provide feedback, and interact with other users and developers:
- GitHub Issues: For bug reports and feature requests
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)

# References
If you use RecSys Examples in your research, please cite:

```
@Manual{,
  title = {RecSys Examples: A collection of recommender system implementations},
  author = {NVIDIA Corporation},
  year = {2024},
  url = {https://github.com/NVIDIA/recsys-examples},
}
```

For more citation information and referenced papers, see [CITATION.md](./CITATION.md).

# License
This project is licensed under the Apache License - see the [LICENSE](./LICENSE) file for details.

[releases]: https://github.com/NVIDIA/recsys-examples/releases