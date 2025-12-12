#!/bin/bash

rm -rf ./inference/triton/hstu_model/1/
mkdir ./inference/triton/hstu_model/1/
cp ./inference/triton/hstu_model/model.py ./inference/triton/hstu_model/1/

rm -rf ./inference/triton/hstu_sparse/1/
mkdir ./inference/triton/hstu_sparse/1/
cp ./inference/triton/hstu_sparse/model.py ./inference/triton/hstu_sparse/1/

PYTHONPATH=${PYTHONPATH}:$(realpath ../) tritonserver --model-repository ./inference/triton/