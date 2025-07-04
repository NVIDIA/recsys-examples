# NVIDIA RecSys Examples

# Overview
NVIDIA RecSys Examples is a collection of optimized recommender models and components. 

The project includes:
- Examples for large-scale HSTU ranking and retrieval models through [TorchRec](https://github.com/pytorch/torchrec) and [Megatron-Core](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core) integration
- HSTU (Hierarchical Sequential Transduction Unit) attention operator support
- Dynamic Embeddings with GPU acceleration

# Environment Setup
## Start from dockerfile

We provide [dockerfile](./docker/Dockerfile) for users to build environment. 
```
docker build -f docker/Dockerfile --platform linux/amd64 -t recsys-examples:latest .
```
If you want to build image for Grace, you can use 
```
docker build -f docker/Dockerfile --platform linux/arm64 -t recsys-examples:latest .
```
You can also set your own base image with args `--build-arg <BASE_IMAGE>`.

## Start from source file
Before running examples, build and install libs under corelib following instruction in documentation:
- [HSTU attention documentation](./corelib/hstu/README.md)
- [Dynamic Embeddings documentation](./corelib/dynamicemb/README.md)

On top of those two core libs, Megatron-Core along with other libs are required. You can install them via pypi package:

```bash
pip install torchx gin-config torchmetrics==1.0.3 typing-extensions iopath megatron-core==0.9.0
```

If you fail to install the megatron-core package, usually due to the python version incompatibility, please try to clone and then install the source code. 

```bash
git clone -b core_r0.9.0 https://github.com/NVIDIA/Megatron-LM.git megatron-lm && \
pip install -e ./megatron-lm
```

We provide our custom HSTU CUDA operators for enhanced performance. You need to install these operators using the following command:

```bash
cd /workspace/recsys-examples/examples/hstu && \
python setup.py install
```

# Get Started
The examples we supported:
- [HSTU recommender examples](./examples/hstu/README.md)

# Contribution Guidelines
Please see our [contributing guidelines](./CONTRIBUTING.md) for details on how to contribute to this project.

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