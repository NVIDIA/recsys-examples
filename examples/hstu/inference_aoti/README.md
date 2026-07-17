# HSTU AOTI Inference with KV Cache

## Purpose

This document describes the end-to-end workflow for exporting and running the HSTU ranking inference model with PyTorch AOTInductor (AOTI), native C++ replay, FlexKV-backed KV cache, and Triton Server.

The workflow covers five stages:

1. Building required custom operators and runtime libraries
2. Exporting the HSTU ranking model through `torch.export` and AOTInductor
3. Starting the FlexKV-based KV-cache runtime service
4. Running the C++ replay executable against exported artifacts and dumped tensors
5. Running the Triton Server demo path for the exported KV-cache AOTI model

For a more detailed build and deployment guide, see [guide_to_hstu_aoti_inference_setup.md](./guide_to_hstu_aoti_inference_setup.md).

---

## Scope

This guide covers these checked-in entrypoints:

PyTorch export and AOTI validation:
- `export_inference_gr_ranking.py`
- `export_inference_gr_ranking_kvcache.py`

C++ AOTI replay:
- `cpp_inference/inference_hstu_gr_ranking_exported_model.cpp`
- `cpp_inference/inference_hstu_gr_ranking_kvcache_exported_model.cpp`

Triton Server AOTI deployment:
- `nve_init_hook/`
- `triton_aoti/hstu_gr_ranking_kvcache/`
- `test_tritonserver_aoti_hstu_model.py`

FlexKV server launcher:
- `start_flexkv_server_for_kvcache_cpp.py`

The same exported model package and replay tensors are used by the Python AOTI check, native C++ replay executable, and Triton request replay path.

At the end of the KV-cache workflow, the following key artifacts are expected:

1. Exported KV-cache AOTI package in `examples/hstu/inference_aoti/hstu_gr_ranking_kvcache_model/`
2. Replay tensors in `examples/hstu/inference_aoti/export_test_dump/`
3. C++ executable at `examples/hstu/inference_aoti/cpp_inference/build/inference_hstu_gr_ranking_kvcache_exported_model`
4. AOTI/Triton runtime libraries under `examples/hstu/triton_libs/`

---

## Environment

1. PyTorch export, C++ replay, and development use the image built by `docker/Dockerfile`. Its base image is NVIDIA PyTorch 26.05 (`nvcr.io/nvidia/pytorch:26.05-py3`) plus the repo's FBGEMM, FlexKV, NVE, HSTU, DynamicEmb, commons, and KV-cache manager builds. See the HSTU example [README](../README.md) for the broader training and inference context.

2. Triton Server testing uses `docker/Dockerfile.tritonserver`, whose runtime base image is NVIDIA Triton Server 26.06 (`nvcr.io/nvidia/tritonserver:26.06-py3`). The Dockerfile copies the built PyTorch backend, PyTorch package, FlexKV, NVE, HSTU, FBGEMM, custom op libraries, and Triton client dependencies from the PyTorch AOTI image.

3. KV-cache AOTI support depends on the FlexKV tree vendored under `third_party/FlexKV` and copied into the image by `docker/Dockerfile`.

---

## Repository Path Variables

Use the following variables throughout the workflow:

```bash
export REPO=/workspace/recsys-examples
export HSTU_DIR=${REPO}/examples/hstu
export GIN=${HSTU_DIR}/inference/configs/kuairand_1k_inference_ranking.gin
export CKPT=${HSTU_DIR}/ckpt/kuairand_1k_ckpt
export KVCACHE_CONFIG=${HSTU_DIR}/inference_aoti/kvcache_cpp_runtime.yaml
```

Adjust `REPO`, `GIN`, `CKPT`, and `KVCACHE_CONFIG` for your environment.

---

## Example: HSTU AOTI Inference on KuaiRand-1K

The following commands run a compact KuaiRand-1K workflow: build images, prepare data, train a small checkpoint, export a KV-cache AOTI package, verify it through the C++ runtime, and replay it through Triton Server.

1. Build the PyTorch-based image for training, export, and C++ replay.

  There are two image builds below. The first builds the reusable `fbgemm` base image. The second reuses that base image and builds the remaining dependencies, in-tree custom ops, C++ replay executable, NVE init hook, PyTorch backend, and Triton runtime library staging.

```bash
# build FBGEMM base
DOCKER_BUILDKIT=1 docker build --progress=plain \
  --platform linux/amd64 --target base_fbgemm -t "recsys-fbgemm-base" -f "docker/Dockerfile" .

# build the rest for recsys-example
DOCKER_BUILDKIT=1 docker build --progress=plain \
  --platform linux/amd64 --build-arg BASE_FBGEMM_IMAGE=recsys-fbgemm-base -t "recsys-examples-dev" -f "docker/Dockerfile" .
```

2. Prepare the dataset and train a small model checkpoint.

   This step preprocesses the `kuairand-1k` dataset, runs single-GPU training with `./training/configs/kuairand_1k_ranking.gin`, and saves the final checkpoint into the `model_ckpt` volume for step 3.

  Note: training is supported on Ampere, Hopper, and Blackwell (sm100).

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

    export PYTHONPATH=\${PYTHONPATH}:/workspace/recsys-examples/examples/
    torchrun --nproc_per_node 1 --master_addr localhost --master_port 6000 \
      ./training/pretrain_gr_ranking.py \
      --gin-config-file /tmp/kuairand_1k_ranking_train_200.gin
    ls -la

    rm -rf ./ckpt/kuairand_1k_ckpt
    cp -apr ./ckpt/iter200 ./ckpt/kuairand_1k_ckpt
  "
```

3. Export the KV-cache AOTI inference model and verify it with the Torch C++ runtime.

  Note: inference/export is supported on Ampere, Ada, and Blackwell (sm120).

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
    export PYTHONPATH=\${PYTHONPATH}:/workspace/recsys-examples/examples/

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

4. Package the Triton Server runtime image.
```bash
DOCKER_BUILDKIT=1 docker build --progress=plain \
  --build-arg PYTORCH_AOTI_IMAGE=recsys-examples-dev -f docker/Dockerfile.tritonserver -t recsys-examples-tritonserver .
```

5. Replay the exported model through Triton Server.
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