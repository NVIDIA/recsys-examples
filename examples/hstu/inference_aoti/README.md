# HSTU Model Inference with Pytorch AOTI and KVCache

## Purpose

This document describes the production workflow for building and running HSTU inference with KV-cache enabled AOTI export and C++ replay verification.

The workflow covers the following 5 steps:

1. Building required custom operators and runtime libraries
2. Exporting the HSTU ranking model through `torch.export` and AOTInductor
3. Starting the FlexKV-based KV-cache runtime service (inference with kvcache)
4. Running the C++ replay executable against exported artifacts and dumped tensors
5. Running the Triton Server demo path for the exported AOTI model

This guide is based on the checked-in workflow in [examples/hstu/inference_aoti/exported_with_kvcache_running_guide.sh](./exported_with_kvcache_running_guide.sh) and the container-oriented operational style in the local Triton/PyTorch guidebook.

---

## Scope

This guide covers the following tests/demos with related files:

Pytorch export and aoti testing demos:
- `export_inference_gr_ranking.py`
- `export_inference_gr_ranking_kvcache.py`

C++ aoti testing demos:
- `cpp_inference/inference_hstu_gr_ranking_exported_model.cpp`
- `cpp_inference/inference_hstu_gr_ranking_kvcache_exported_model.cpp`

Triton server aoti model testing demos:
- `inference_aoti/nve_init_hook/`
- `inference_aoti/triton_aoti/`
- triton client script: `send_one_kvcache_triton_request.py`

Flexkv Server launcher:
- `start_flexkv_server_for_kvcache_cpp.py`

It covers export, runtime setup, native C++ validation, and Triton Server deployment and request-replay path for the same exported AOTI package.

At the end of the workflow, the following key **artifacts** are expected:

1. Exported AOTI package in `examples/hstu/inference_aoti/hstu_gr_ranking_model/`
2. Replay tensors in `examples/hstu/inference_aoti/export_test_dump/`
3. C++ executable at `examples/hstu/inference_aoti/cpp_inference/build/inference_hstu_gr_ranking_kvcache_exported_model`
4. AOTI/Triton runtime libraries under `examples/hstu/triton_libs/`

---

## Environment

1. Building, testing (pytorch, C++ torch) and development relies the environment setup for `example/hstu`. It is based on NVIDIA PyTorch Release 26.05 (`nvcr.io/nvidia/pytorch:26.05-py3`) together with dependencies installed through `docker/Dockerfile`. You may refer to [`README`](./README.md#how-to-setup) of HSTU inference example as well

2. Testing with triton server demos requires the environment from Triton Inference Server Release 26.05 (`nvcr.io/nvidia/tritonserver:26.05-py3`). The triton server environment does not ship with PyTorch installed. The pytorch is manually installed in order to run the FlexKV server (and/or test client). Refer to `docker/Dockerfile` as an example.

3. **Important**: The inference with kvcache support with aoti requires a [customized FlexKV version](https://github.com/geoffreyQiu/FlexKV/tree/cpp_client).

---

## Repository Path Variables

Use the following variables throughout the workflow:

```bash
export REPO=/workspace/recsys-examples
export HSTU_DIR=${REPO}/examples/hstu
export GIN=${HSTU_DIR}/inference/configs/kuairand_1k_inference_ranking.gin
export CKPT=.../path/to/some/ckpt/...   # incomplete path
export KVCACHE_CONFIG=${HSTU_DIR}/inference_aoti/kvcache_cpp_runtime.yaml
export PYTORCH_BACKEND=.../path/to/tritonserver/pytorch_backend/...
```

Adjust `REPO`, `GIN`, `CKPT`, `KVCACHE_CONFIG` and `PYTORCH_BACKEND` for your environment.

---

## Example: HSTU Model Inference with AOTI based on Kuairand-1K

The following shows a quick demo for HSTU model inference with AOTI. See [guide_to_hstu_aoti_inference_setup.md](./guide_to_hstu_aoti_inference_setup.md) for the detailed setup guide.

1. Build the PyTorch-based image for training and exporting.

  There are two image builds below. The first builds the `fbgemm` base image and takes about 90 minutes, so it is worth keeping and reusing. The second build reuses that base image and covers the rest of dependencies and the build of recsys-examples, about 30 minutes.

```bash
# build FBGEMM base
DOCKER_BUILDKIT=1 docker build --progress=plain \
  --platform linux/amd64 --target base_fbgemm -t "recsys-fbgemm-base" -f "docker/Dockerfile" .

# build the rest for recsys-example
DOCKER_BUILDKIT=1 docker build --progress=plain \
  --platform linux/amd64 --build-arg BASE_FBGEMM_IMAGE=recsys-fbgemm-base -t "recsys-examples-dev" -f "docker/Dockerfile" .
```

2. Prepare the dataset and train the model.

   This step preprocesses the `kuairand-1k` dataset, runs single-GPU training with `./training/configs/kuairand_1k_ranking.gin`, and saves the final checkpoint into the `model_ckpt` volume for step 3.

   Note: Training only supported on Ampere, Hopper and Blackwell (sm100).

```bash
docker volume create recsys-data
docker volume create model_ckpt

docker run \
  --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus 1 \
  --volume recsys-data:/workspace/recsys-examples/examples/hstu/tmp_data \
  --volume model_ckpt:/workspace/recsys-examples/examples/hstu/ckpt \
  --hostname $(hostname) --name recsys-dev-training \
  --tmpfs /tmp:exec \
  recsys-examples-dev \
  bash -lecx "
    export PYTHONPATH=\${PYTHONPATH}:/workspace/recsys-examples/examples/
    export CUDA_VISIBLE_DEVICES=0

    cd /workspace/recsys-examples/examples/commons
    python3 ./hstu_data_preprocessor.py \
      --dataset_name kuairand-1k \
      --dataset_path /workspace/recsys-examples/examples/hstu/tmp_data
  "

docker run \
  --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus 1 \
  --volume recsys-data:/workspace/recsys-examples/examples/hstu/tmp_data \
  --volume model_ckpt:/workspace/recsys-examples/examples/hstu/ckpt \
  --hostname $(hostname) --name recsys-dev-training \
  --tmpfs /tmp:exec \
  recsys-examples-dev \
  bash -lecx "
    cd /workspace/recsys-examples/examples/hstu
    cp ./training/configs/kuairand_1k_ranking.gin /tmp/kuairand_1k_ranking_train_200.gin
    printf '\nTrainerArgs.log_interval = 50\nTrainerArgs.max_train_iters = 200\nTrainerArgs.ckpt_save_interval = 200\nTrainerArgs.ckpt_save_dir = \"./ckpt\"\n' >> /tmp/kuairand_1k_ranking_train_200.gin

    export PYTHONPATH=${PYTHONPATH}:/workspace/recsys-examples/examples/
    torchrun --nproc_per_node 1 --master_addr localhost --master_port 6000 \
      ./training/pretrain_gr_ranking.py \
      --gin-config-file /tmp/kuairand_1k_ranking_train_200.gin
    ls -la

    rm -rf ./ckpt/kuairand_1k_ckpt
    cp -apr ./ckpt/iter200 ./ckpt/kuairand_1k_ckpt
  "
```

3. Export inference model with aoti and kvcache, and test on Torch C++ runtime.

     Note: Inference/Exporting only supported on Ampere, Ada and Blackwell (sm120).

```bash
docker volume create exported_hstu_model
docker run \
  --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus 1 \
  --volume recsys-data:/workspace/recsys-examples/examples/hstu/tmp_data \
  --volume model_ckpt:/workspace/recsys-examples/examples/hstu/ckpt \
  --volume exported_hstu_model:/exported_hstu_model \
  --hostname $(hostname) --name recsys-dev-inference \
  --tmpfs /tmp:exec  \
  recsys-examples-dev \
  bash -lecx "
    export FLEXKV_LOG_LEVEL=WARNING
    export DYNAMICEMB_OPS_LIB_DIR=/workspace/recsys-examples/corelib/dynamicemb/torch_binding_build/
    export PYTHONPATH=${PYTHONPATH}:/workspace/recsys-examples/examples/

    cd /workspace/recsys-examples/examples/hstu
    export KVCACHE_MANAGER_CONFIG_FILE=./inference_aoti/kvcache_cpp_runtime.yaml
    python3 ./inference_aoti/export_inference_gr_ranking_kvcache.py \
      --gin_config_file ./inference/configs/kuairand_1k_inference_ranking.gin \
      --checkpoint_dir ./ckpt/kuairand_1k_ckpt \
      --max_bs 2 --kvcache_config_file \${KVCACHE_MANAGER_CONFIG_FILE}
    
    python3 ./inference_aoti/start_flexkv_server_for_kvcache_cpp.py \
      --config_file \${KVCACHE_MANAGER_CONFIG_FILE} > flexkv_cache_server.log 2>&1 &
    kvserver_pid=\$!
    sleep 10
    kill -0 \${kvserver_pid}
    ./inference_aoti/cpp_inference/build/inference_hstu_gr_ranking_kvcache_exported_model \
      ./inference_aoti/hstu_gr_ranking_kvcache_model \
      ./inference_aoti/export_test_dump
    kill \${kvserver_pid} || true

    mkdir -p /exported_hstu_model
    cp -apr /workspace/recsys-examples/examples/hstu/inference_aoti/hstu_gr_ranking_kvcache_model /exported_hstu_model/
    cp -apr /workspace/recsys-examples/examples/hstu/inference_aoti/export_test_dump /exported_hstu_model/ "
```

4. Pack the tritonserver-based image.
```bash
DOCKER_BUILDKIT=1 docker build --progress=plain \
  --build-arg PYTORCH_AOTI_IMAGE=recsys-examples-dev -f docker/Dockerfile.tritonserver  -t recsys-examples-tritonserver  .
```

5. Test with tritonserver.
```bash
docker run \
  --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus 1 \
  --volume exported_hstu_model:/exported_hstu_model \
  --hostname $(hostname) --name recsys-dev-tritonserver \
  --tmpfs /tmp:exec  \
  recsys-examples-tritonserver \
  bash -lecx "
    mkdir -p /triton_model_repo/
    cp -apr \
      /workspace/recsys-examples/examples/hstu/inference_aoti/triton_aoti/hstu_gr_ranking_kvcache \
      /triton_model_repo/
    cp -apr /exported_hstu_model/hstu_gr_ranking_kvcache_model \
      /triton_model_repo/hstu_gr_ranking_kvcache/1
    cp -apr /exported_hstu_model/export_test_dump \
      /workspace/recsys-examples/examples/hstu/inference_aoti

    cd /workspace/recsys-examples/examples/hstu/inference_aoti
    export FLEXKV_LOG_LEVEL=WARNING
    export KVCACHE_MANAGER_CONFIG_FILE=\${PWD}/kvcache_cpp_runtime.yaml
    python3 start_flexkv_server_for_kvcache_cpp.py --config_file \${KVCACHE_MANAGER_CONFIG_FILE} 2>&1 &
    kvserver_pid=\$!
    sleep 10
    kill -0 \${kvserver_pid}
    
    tritonserver --model-repository=/triton_model_repo/ &
    triton_pid=\$!
    sleep 30
    
    python3 test_tritonserver_aoti_hstu_model.py > test_benchmark.log
    cat test_benchmark.log
    kill \$triton_pid || true
    kill -9 \$triton_pid || true
    kill \$kvserver_pid || true "

docker volume rm exported_hstu_model
```