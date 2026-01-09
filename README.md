# NVIDIA RecSys Examples

# Overview
NVIDIA RecSys Examples is a collection of optimized recommender models and components. 

The project includes:
- Examples for large-scale HSTU ranking and retrieval models through [TorchRec](https://github.com/pytorch/torchrec) and [Megatron-Core](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core) integration
- Examples for semantic-id based retrieval model through [TorchRec](https://github.com/pytorch/torchrec) and [Megatron-Core](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core) integration
- HSTU (Hierarchical Sequential Transduction Unit) attention operator support
- Dynamic Embeddings with GPU acceleration

# What's New
- **[2025/12/10]** 🎉v25.11 released!
  - DynamicEmb supports embedding admission, that decides whether a new feature ID is allowed to create or update an embedding entry in the dynamic embedding table. By controlling admission, the system can prevent very rare or noisy IDs from consuming parameters and optimizer state that bring little training benefit.
- **[2025/11/11]** 🎉v25.10 released!
  - HSTU training example supports sequence parallelism.
  - DynamicEmb supports LRU score checkpointing, gradient clipping.
  - Decouple scaling sequence length from the maximum sequence length limit in HSTU attention and extend HSTU support to the SM89 GPU architecture for training.
- **[2025/10/20]** 🎉v25.09 released!
  - Integrated prefetching and caching into the HSTU training example.
  - DynamicEmb now supports distributed embedding dumping and memory scaling.
  - Added kernel fusion in the HSTU block for inference, including KVCache fixes.
  - HSTU attention now supports FP8 quantization.

<details>
<summary>More</summary>

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
For more detailed release notes, please refer our [releases](https://github.com/NVIDIA/recsys-examples/releases).

# Get Started
The examples we supported:
- [HSTU recommender examples](./examples/hstu/README.md)
- [SID based generative recommender examples](./examples/sid_gr/README.md)

# Contribution Guidelines
Please see our [contributing guidelines](./CONTRIBUTING.md) for details on how to contribute to this project.

# Resources
## Video
- [RecSys Examples 中的训练与推理优化实践](https://www.bilibili.com/video/BV1msMwzpE5B?buvid=638d217658211387f0a20e730604a780&from_spmid=united.player-video-detail.drama-float.0&is_story_h5=false&mid=V%2FD40L0stVy%2BZTgWdpjtGA%3D%3D&plat_id=116&share_from=ugc&share_medium=iphone&share_plat=ios&share_session_id=2DD6CE30-B189-4EEC-9FD4-8BAD6AEFE720&share_source=WEIXIN&share_tag=s_i&spmid=united.player-video-detail.0.0&timestamp=1749773222&unique_k=Sjcfmgy&up_id=1320140761&vd_source=7372540fd02b24a46851135aa003577c)
- [基于CUTLASS 3 的HSTU attention 算子开发与优化](https://www.bilibili.com/video/BV1TsMwzWEzS?buvid=638d217658211387f0a20e730604a780&from_spmid=united.player-video-detail.drama-float.0&is_story_h5=false&mid=V%2FD40L0stVy%2BZTgWdpjtGA%3D%3D&plat_id=116&share_from=ugc&share_medium=iphone&share_plat=ios&share_session_id=2DD6CE30-B189-4EEC-9FD4-8BAD6AEFE720&share_source=WEIXIN&share_tag=s_i&spmid=united.player-video-detail.0.0&timestamp=1749773222&unique_k=Sjcfmgy&up_id=1320140761&vd_source=7372540fd02b24a46851135aa003577c&spm_id_from=333.788.videopod.sections)

## Blog
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