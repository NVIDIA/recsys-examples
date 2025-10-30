# HSTU Training example

We have supported both retrieval and ranking model whose backbones are HSTU layers. In this example collection, we allow user to specify the model structures via gin-config file. Supported datasets are listed below. Regarding the gin-config interface, please refer to [inline comments](../utils/gin_config_args.py) .

## Parallelism Introduction 
To facilitate large embedding tables and scaling-laws of HSTU dense, we have integrate **[TorchRec](https://github.com/pytorch/torchrec)** that does shard embedding tables and **[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)** that enable dense parallelism(e.g Data, Tensor, Sequence, Pipeline, and Context parallelism) in this example.
This integration ensures efficient training by coordinating sparse (embedding) and dense (context/data) parallelisms within a single model.
![parallelism](../figs/parallelism.png)

## Environment Setup
### Start from dockerfile

We provide [dockerfile](../../../docker/Dockerfile) for users to build environment. 
```
git clone https://github.com/NVIDIA/recsys-examples.git && cd recsys-examples
docker build -f docker/Dockerfile --platform linux/amd64 -t recsys-examples:latest .
```
If you want to build image for Grace, you can use 
```
git clone https://github.com/NVIDIA/recsys-examples.git && cd recsys-examples
docker build -f docker/Dockerfile --platform linux/arm64 -t recsys-examples:latest .
```
You can also set your own base image with args `--build-arg <BASE_IMAGE>`.

### Start from source file
Before running examples, build and install libs under corelib following instruction in documentation:
- [HSTU attention documentation](.../../../corelib/hstu/README.md)
- [Dynamic Embeddings documentation](.../../../corelib/dynamicemb/README.md)

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

Before getting started, please make sure that all pre-requisites are fulfilled. You can refer to [Get Started](../../../README) section in the root directory of the repo to set up the environment.


### Dataset preprocessing

In order to prepare the dataset for training, you can use our `preprocessor.py` under the hstu example folder of the project.

```bash
cd <root-to-repo>/examples/hstu && 
mkdir -p ./tmp_data && python3 ./preprocessor.py --dataset_name <"ml-1m"|"ml-20m"|"kuairand-pure"|"kuairand-1k"|"kuairand-27k">

```

### Start training
The entrypoint for training are `pretrain_gr_retrieval.py` or `pretrain_gr_ranking.py`. We use gin-config to specify the model structure, training arguments, hyper-params etc.

Command to run retrieval task with `MovieLens 20m` dataset:

```bash
# Before running the `pretrain_gr_retrieval.py`, make sure that current working directory is `hstu`
cd <root-to-project>examples/hstu 
PYTHONPATH=${PYTHONPATH}:$(realpath ../) torchrun --nproc_per_node 1 --master_addr localhost --master_port 6000  ./training/pretrain_gr_retrieval.py --gin-config-file ./training/configs/movielen_retrieval.gin
```

To run ranking task with `MovieLens 20m` dataset:
```bash
# Before running the `pretrain_gr_ranking.py`, make sure that current working directory is `hstu`
cd <root-to-project>examples/hstu 
PYTHONPATH=${PYTHONPATH}:$(realpath ../) torchrun --nproc_per_node 1 --master_addr localhost --master_port 6000  ./training/pretrain_gr_ranking.py --gin-config-file ./training/configs/movielen_ranking.gin
```


