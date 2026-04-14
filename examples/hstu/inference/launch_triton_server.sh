#!/bin/bash

CKPT_DIR=$1

# setup checkpoint for ps from dynamic embedding tables
DYNAMIC_MODULE_DIR="${CKPT_DIR}/dynamicemb_module/"
PS_MODULE_DIR="${CKPT_DIR}/ps_module/"
if [ ! -d "${PS_MODULE_DIR}" ]; then
    mkdir -p "${PS_MODULE_DIR}"
    for f in $(find ${DYNAMIC_MODULE_DIR} -regex .*\/.*_emb_.*); do 
        ln -s "$(realpath $f)" "${PS_MODULE_DIR}/$(basename $f).dyn";
    done
fi

# setup models and configs for triton
rm -rf ./inference/triton/hstu_model/1/
mkdir ./inference/triton/hstu_model/1/
cp ./inference/triton/hstu_model/model.py ./inference/triton/hstu_model/1/
echo $'parameters [\n  {\n    key: "HSTU_CHECKPOINT_DIR"\n    value: {\n      string_value: "'${CKPT_DIR}$'"\n    }\n  }\n]' >> ./inference/triton/hstu_model/config.pbtxt 

rm -rf ./inference/triton/hstu_sparse/1/
mkdir ./inference/triton/hstu_sparse/1/
cp ./inference/triton/hstu_sparse/model.py ./inference/triton/hstu_sparse/1/
echo $'parameters [\n  {\n    key: "HSTU_CHECKPOINT_DIR"\n    value: {\n      string_value: "'${CKPT_DIR}$'"\n    }\n  }\n]' >> ./inference/triton/hstu_sparse/config.pbtxt 

# launch
HSTU_INFERENCE_ONLY=1 PYTHONPATH=${PYTHONPATH}:$(realpath ../) tritonserver --model-repository ./inference/triton/