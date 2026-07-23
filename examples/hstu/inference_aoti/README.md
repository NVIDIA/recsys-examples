# HSTU AOTI Inference with KV Cache

## Purpose

This document describes how to export and run the HSTU ranking inference model
with PyTorch AOTInductor (AOTI), native C++ replay, a FlexKV-backed KV cache,
and Triton Server.

The workflow covers five stages:

1. Building the required custom operators and runtime libraries
2. Exporting the HSTU ranking model with `torch.export` and AOTInductor
3. Starting the FlexKV-backed KV-cache runtime service
4. Validating the exported artifacts and replay tensors with native C++
5. Serving and testing the exported KV-cache AOTI model with Triton Server

For lower-level build and deployment details, see the
[HSTU KV-cache AOTI setup guide](./guide_to_hstu_aoti_inference_setup.md).

---

## Scope

This guide covers these checked-in entry points:

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

For each export variant, Python validation and native C++ replay consume the
same model package and replay tensors. The KV-cache artifacts are also used by
the Triton request-replay path.

The complete example below focuses on the KV-cache variant. The non-KV-cache
exporter and C++ replay executable remain available for direct AOTI validation.

At the end of the KV-cache workflow, the following key artifacts are expected:

1. Exported KV-cache AOTI package in `examples/hstu/inference_aoti/hstu_gr_ranking_kvcache_model/`
2. Replay tensors in `examples/hstu/inference_aoti/export_test_dump/`
3. C++ executable at `examples/hstu/inference_aoti/cpp_inference/build/inference_hstu_gr_ranking_kvcache_exported_model`
4. AOTI/Triton runtime libraries under `examples/hstu/triton_libs/`

---

## Exported Model Package

The AOTI export path converts the HSTU PyTorch model with `torch.export` and
`torch._inductor.aoti_compile_and_package` for native C++ loading.

The embedding implementation combines DynamicEmb `InferenceEmbeddingTable`
and `ScoredHashTable` with NVEmbedding layers. NVEmbedding's customized
`export_and_aot` stores layer metadata and embedding-table files alongside the
model `.pt2` archive, avoiding duplicate embedding-table copies at load time.
The complete exported model archive has this structure:

```text
path/to/model_archive
        ├── model.pt2                              # AOTI model package
        ├── metadata.json                          # NVEmbedding layer metadata
        └── weights/{emb_layer_module_name}.nve    # NVEmbedding weight data
```

---

## Environment

1. PyTorch export, C++ replay, and development use the image built from
   `docker/Dockerfile`. It extends NVIDIA PyTorch 26.05
   (`nvcr.io/nvidia/pytorch:26.05-py3`) with the repository's FBGEMM, FlexKV,
   NVE, HSTU, DynamicEmb, commons, and KV-cache manager builds. See the HSTU
   example [README](../README.md) for broader training and inference context.

2. Triton Server testing uses `docker/Dockerfile.tritonserver`, based on NVIDIA
   Triton Server 26.06 (`nvcr.io/nvidia/tritonserver:26.06-py3`). It copies the
   custom PyTorch backend, PyTorch installation, FlexKV, NVE, HSTU, FBGEMM,
   custom-operator libraries, and Triton client dependencies from the AOTI
   development image.

3. KV-cache AOTI support depends on the FlexKV source under
   `third_party/FlexKV`, which `docker/Dockerfile` copies into the image.

---

## Container Paths

The commands below assume these paths inside the containers:

| Purpose | Container path |
| --- | --- |
| Repository | `/workspace/recsys-examples` |
| HSTU example | `/workspace/recsys-examples/examples/hstu` |
| Gin configuration | `/workspace/recsys-examples/examples/hstu/inference/configs/kuairand_1k_inference_ranking.gin` |
| Checkpoint | `/workspace/recsys-examples/examples/hstu/ckpt/kuairand_1k_ckpt` |
| KV-cache configuration | `/workspace/recsys-examples/examples/hstu/inference_aoti/kvcache_cpp_runtime.yaml` |

If your layout differs, update the corresponding volume mounts and commands.

---

## Example: HSTU AOTI Inference on KuaiRand-1K

The following commands build the images, prepare KuaiRand-1K data, train a
small checkpoint, export a KV-cache AOTI package, validate it with native C++,
and replay requests through Triton Server.

### 1. Build the development image

The first build creates the reusable FBGEMM base image. The second reuses that
image and builds the remaining dependencies, in-tree custom operators, C++
replay executable, NVE initialization hook, PyTorch backend, and staged Triton
runtime libraries.

```bash
# Build the reusable FBGEMM base image.
DOCKER_BUILDKIT=1 docker build --progress=plain \
  --platform linux/amd64 \
  --target base_fbgemm \
  -t "recsys-fbgemm-base" \
  -f "docker/Dockerfile" .

# Build the recsys-examples development image from that base.
DOCKER_BUILDKIT=1 docker build --progress=plain \
  --platform linux/amd64 \
  --build-arg BASE_FBGEMM_IMAGE=recsys-fbgemm-base \
  -t "recsys-examples-dev" \
  -f "docker/Dockerfile" .
```

### 2. Prepare the dataset and train a checkpoint

This step preprocesses the `kuairand-1k` dataset, runs single-GPU training with
`./training/configs/kuairand_1k_ranking.gin`, and saves the final checkpoint in
the `model_ckpt` volume for step 3.

Training is supported on Ampere, Hopper, and Blackwell SM100 GPUs.

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
    rm -rf ./ckpt/kuairand_1k_ckpt
    cp -apr ./ckpt/iter200 ./ckpt/kuairand_1k_ckpt
  "
```

### 3. Export and validate the KV-cache AOTI model

AOTI export and inference are supported on Ampere, Ada, and Blackwell SM120
GPUs.

```bash
docker volume create exported_hstu_model
docker run \
  --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus 1 \
  --volume recsys-data:/workspace/recsys-examples/examples/hstu/tmp_data \
  --volume model_ckpt:/workspace/recsys-examples/examples/hstu/ckpt \
  --volume exported_hstu_model:/exported_hstu_model \
  --hostname $(hostname) --name recsys-dev-inference \
  --tmpfs /tmp:exec \
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

### 4. Package the Triton Server runtime image

```bash
DOCKER_BUILDKIT=1 docker build --progress=plain \
  --build-arg PYTORCH_AOTI_IMAGE=recsys-examples-dev \
  -f docker/Dockerfile.tritonserver \
  -t recsys-examples-tritonserver .
```

### 5. Replay the exported model through Triton Server

```bash
docker run \
  --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus 1 \
  --volume exported_hstu_model:/exported_hstu_model \
  --hostname $(hostname) --name recsys-dev-tritonserver \
  --tmpfs /tmp:exec \
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

# Remove the exported artifacts only when they are no longer needed.
docker volume rm exported_hstu_model
```
