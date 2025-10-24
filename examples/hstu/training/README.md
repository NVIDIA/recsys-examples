# HSTU Training example

We have supported both retrieval and ranking model whose backbones are HSTU layers. In this example collection, we allow user to specify the model structures via gin-config file. Supported datasets are listed below. Regarding the gin-config interface, please refer to [inline comments](../utils/gin_config_args.py) .

## Parallelism Introduction 
To facilitate large embedding tables and scaling-laws of HSTU dense, we have integrate **[TorchRec](https://github.com/pytorch/torchrec)** that does shard embedding tables and **[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)** that enable dense parallelism(e.g Data, Tensor, Sequence, Pipeline, and Context parallelism) in this example.
This integration ensures efficient training by coordinating sparse (embedding) and dense (context/data) parallelisms within a single model.
![parallelism](../figs/parallelism.png)


## Dataset Introduction

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

Before getting started, please make sure that all pre-requisites are fulfilled. You can refer to [Get Started][../../../README] section in the root directory of the repo to set up the environment.****


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


