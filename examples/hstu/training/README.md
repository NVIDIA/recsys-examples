# HSTU Training Example

We support retrieval and ranking models whose backbones are HSTU layers. In this example collection, users specify model structures through gin config files. Supported datasets are listed below. For the gin-config interface, see the [inline comments](../utils/gin_config_args.py).

## Parallelism Introduction 
To facilitate large embedding tables and HSTU dense-layer scaling, this example integrates **[TorchRec](https://github.com/pytorch/torchrec)** for embedding table sharding and **[Megatron-Core](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core)** for dense parallelism, including data, tensor, sequence, pipeline, and context parallelism.
This integration ensures efficient training by coordinating sparse (embedding) and dense (context/data) parallelisms within a single model.
![parallelism](../figs/parallelism.png)

## Environment Setup
### Start from dockerfile

We provide [dockerfile](../../../docker/Dockerfile) for users to build environment. 
```bash
git clone --recursive https://github.com/NVIDIA/recsys-examples.git && cd recsys-examples
docker build -f docker/Dockerfile --platform linux/amd64 -t recsys-examples:latest .
```
If you want to build image for Grace, you can use 
```bash
git clone --recursive https://github.com/NVIDIA/recsys-examples.git && cd recsys-examples
docker build -f docker/Dockerfile --platform linux/arm64 -t recsys-examples:latest .
```
> **Note:** The `--recursive` flag is required to fetch submodules (e.g. `third_party/FBGEMM` for HSTU attention kernels).
> If you already cloned without it, run `git submodule update --init --recursive`.
You can also set your own base image with `--build-arg BASE_IMAGE=<image>`.

### Start from source file
Before running examples, build and install libs following the instructions below:
- [DynamicEmb documentation](../../../corelib/dynamicemb/README.md)

**HSTU attention kernels** are provided by the `fbgemm_gpu_hstu` package (import name: `hstu`),
included as a git submodule at `third_party/FBGEMM`. Install it from source:

```bash
git submodule update --init --recursive
cd third_party/FBGEMM/fbgemm_gpu/experimental/hstu && pip install .
```

On top of those core libraries, Megatron-Core and other Python dependencies are required. The Docker image installs Megatron-Core from `core_v0.13.1`. For a source environment, install the Python dependencies and then install the matching Megatron-Core source:

```bash
pip install torchx gin-config torchmetrics==1.0.3 typing-extensions iopath
git clone -b core_v0.13.1 https://github.com/NVIDIA/Megatron-LM.git megatron-lm
pip install --no-deps -e ./megatron-lm
```

We provide custom CUDA operators used by the HSTU examples under `examples/commons`. Install them with:

```bash
cd /workspace/recsys-examples/examples/commons
TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0" python3 setup.py install
```
### Dataset Introduction

We have supported several datasets as listed in the following sections:

### Dataset Information
#### **MovieLens**
refer to [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) and [MovieLens 20M](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset) for details.
#### **KuaiRand**

| dataset       | # users | seqlen max | seqlen min | seqlen mean | seqlen median | # items    |
|---------------|---------|------------|------------|-------------|---------------|------------|
| kuairand_pure | 27285   | 910        | 1          | 1           | 39            | 7551       |
| kuairand_1k   | 1000    | 49332      | 10         | 5038        | 3379          | 4369953    |
| kuairand_27k  | 27285   | 228000     | 100        | 11796       | 8591          | 32038725   |
 
refer to [KuaiRand](https://kuairand.com/) for details.

## Running the examples

Before getting started, make sure that all prerequisites are fulfilled. You can refer to the [Get Started](../../../README.md#get-started) section in the root README to set up the environment.


### Dataset preprocessing

In order to prepare the dataset for training, you can use our `hstu_data_preprocessor.py` under the commons folder of the project.

```bash
cd <root-to-repo>/examples/commons && 
mkdir -p ./tmp_data && python3 ./hstu_data_preprocessor.py --dataset_name <"ml-1m"|"ml-20m"|"kuairand-pure"|"kuairand-1k"|"kuairand-27k">

```

### Start training
The entrypoint for training are `pretrain_gr_retrieval.py` or `pretrain_gr_ranking.py`. We use gin-config to specify the model structure, training arguments, hyper-params etc.

Command to run retrieval task with `MovieLens 20m` dataset:

```bash
# Before running the `pretrain_gr_retrieval.py`, make sure that current working directory is `hstu`
cd <root-to-project>/examples/hstu
PYTHONPATH=${PYTHONPATH}:$(realpath ../) torchrun --nproc_per_node 1 --master_addr localhost --master_port 6000  ./training/pretrain_gr_retrieval.py --gin-config-file ./training/configs/movielen_retrieval.gin
```

To run ranking task with `MovieLens 20m` dataset:
```bash
# Before running the `pretrain_gr_ranking.py`, make sure that current working directory is `hstu`
cd <root-to-project>/examples/hstu
PYTHONPATH=${PYTHONPATH}:$(realpath ../) torchrun --nproc_per_node 1 --master_addr localhost --master_port 6000  ./training/pretrain_gr_ranking.py --gin-config-file ./training/configs/movielen_ranking.gin
```

