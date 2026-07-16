# HSTU KV-Cache AOTI End-to-End Build and Run Guide

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
- `setup_kvcache_config_for_tritonserver.sh`
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
export ENV_FILE=/tmp/kvcache_cpp_runtime.env
export PYTORCH_BACKEND=.../path/to/tritonserver/pytorch_backend/...
```

Adjust `REPO`, `GIN`, `CKPT`, `ENV_FILE` and `PYTORCH_BACKEND` for your environment.

---

## Build

### Step 1: Build & Install Custom Ops used in HSTU inference

The HSTU model inference relies on these custom torch ops:

1. DynamicEmb inference ops
2. HSTU runtime ops from the C++ demo build
3. Paged KV-cache ops from `examples/commons`
4. FBGEMM shared libraries
5. KV-cache manager ops from `corelib/recsys_kvcache_manager`

The python package version is built and installed according `docker/Dockerfile`.
The torch bindings is built for aoti use.

```bash
# DynamicEmb ops
cd ${REPO}/corelib/dynamicemb
mkdir -p torch_binding_build && cd torch_binding_build
cmake .. && make -j

# KV-cache manager ops
cd ${REPO}/corelib/recsys_kvcache_manager/
mkdir -p build && cd build
cmake .. && make -j 
```

For HSTU runtime op and Paged KV-cache ops from `examples/commons`, their torch bindings are built in `inference_aoti/cpp_inference` together with C++ reply Executable (see step 3).

Expected output:

```text
${REPO}/corelib/dynamicemb/torch_binding_build/inference_emb_ops.so
```

### Step 2: Run Python Export for KV-Cache AOTI

Run the export workflow from the HSTU directory:

```bash
cd ${HSTU_DIR}
python3 inference_aoti/export_inference_gr_ranking_kvcache.py \
  --gin_config_file ${GIN} \
  --checkpoint_dir ${CKPT} \
  --max_bs 2
```

This step performs all of the following:

1. Builds the exportable model wrapper for KV-cache inference
2. Exports the model through `torch.export`
3. Produces the packaged AOTI archive under `inference_aoti/hstu_gr_ranking_model/`
4. Produces replay tensors under `inference_aoti/export_test_dump/`


### Step 3: Build the C++ Replay Executable

```bash
export PATH=/usr/local/cuda/bin:${PATH}
export CMAKE_PREFIX_PATH="$(python3 -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "share", "cmake"))')"

cmake -S "${HSTU_DIR}/inference_aoti/cpp_inference" -B "${HSTU_DIR}/inference_aoti/cpp_inference/build"
cmake --build "${HSTU_DIR}/inference_aoti/cpp_inference/build" -j 8
```

Expected outputs:

1. `examples/hstu/inference_aoti/cpp_inference/build/inference_hstu_gr_ranking_kvcache_exported_model`
2. `examples/hstu/inference_aoti/cpp_inference/build/libhstu_cuda_ops_runtime.so`
2. `examples/hstu/inference_aoti/cpp_inference/build/libpaged_kvcache_ops_runtime.so`


### Step 4: Verify with the C++ AOTI Replay

#### 4.1: Start the FlexKV Runtime Service

The C++ replay executable demonstrates the deployment scenario that kvcache server is separated from the inference framework.
Setup the KVCache config in `setup_kvcache_cpp_runtime_env.sh` in environment variables for FlexKV, and start the FlexKV server as follows:

```bash
cd ${HSTU_DIR}
rm -f ${ENV_FILE}
source setup_kvcache_cpp_runtime_env.sh
python3 inference_aoti/start_flexkv_server_for_kvcache_cpp.py --env_file ${ENV_FILE} > kv.log 2>&1 &
```

Verify the environment file created, and load the runtime server variables into the current shell:

```bash
cat ${ENV_FILE}
source ${ENV_FILE}
```

#### Step 4.2: Run the C++ Replay Verification

Run the replay executable against the exported package and dumped tensors:

```bash
${HSTU_DIR}/inference_aoti/cpp_inference/build/inference_hstu_gr_ranking_kvcache_exported_model \
  ${HSTU_DIR}/inference_aoti/hstu_gr_ranking_model \
  ${HSTU_DIR}/inference_aoti/export_test_dump
```

The executable will:

1. Load NVE metadata and weights
2. Load the packaged AOTI model
3. Replay each dumped batch
4. Compare C++ outputs against exported reference tensors
5. Report `max_abs_diff` for each batch

### Step 5: Triton Server Demo for the KV-Cache AOTI Model

This section show the deployment of export aoti model with Triton serving.
It has four major parts:

1. (temporary) Build the PyTorch backend with the model init hook (required by NVE layer loading; not in formal released triton environment).
2. Setup the dependency for triton server
3. Setup the exported aoti model
4. Launch Triton with the exported model, custom operator libraries, and FlexKV runtime


#### Step 5.1: Build the Modified Triton PyTorch Backend

If your environment requires the modified backend described in the local guidebook, build it first.

In the nvidia pytorch container:

```bash
## pytorch backend with model init hook support ##
python3 -m pip install --upgrade "cmake>=3.31.8"
cp /usr/local/lib/libjpeg.so.62 /usr/local/lib/python3.12/dist-packages/torch/lib/libjpeg.so.62

git clone git@github.com:triton-inference-server/pytorch_backend.git
cd pytorch_backend && git checkout ceeecb7
mkdir -p build && cd build
cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX:PATH="$PWD/install" \
  -DTRITON_PYTORCH_INCLUDE_PATHS="/usr/local/lib/python3.12/dist-packages/torch/include;/usr/local/lib/python3.12/dist-packages/torch/include/torch/csrc/api/include;/opt/pytorch/vision/torchvision/csrc" \
  -DTRITON_PYTORCH_LIB_PATHS="/usr/local/lib/python3.12/dist-packages/torch/lib" \
  -DTRITON_PYTORCH_ENABLE_TORCHVISION=OFF \
  -DTRITON_BACKEND_REPO_TAG=r26.05 \
  -DTRITON_CORE_REPO_TAG=r26.05 \
  -DTRITON_COMMON_REPO_TAG=r26.05 \
  ..
cmake --build . -j"$(nproc)" --target install


## nve init hook (nve layer loader) used by HSTU aoti model ##
cd ${HSTU_DIR}/nve_init_hook
mkdir -p build && cd build
cmake .. && make -j
```

Expected backend output:

```text
${PYTORCH_BACKEND}/build/install/backends/pytorch
${HSTU_DIR}/nve_init_hook/build/libnve_init_hook.so
```

#### Step 5.2: Setup the Triton Server Container

Copy the previously built `libtriton_pytorch.so` into the container. Or,
mount the install dir of pytorch backend to the container. (omitted here)

On the host:

```bash
docker run --rm -it \
  --gpus all \
  --ipc=host \
  --network=host \
  --shm-size=8G \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  --tmpfs /tmp:exec \
  --name triton_hstu \
  nvcr.io/nvidia/tritonserver:26.06-py3
```

Inside the triton server runtime container:

```bash
cp ${PYTORCH_BACKEND}/build/install/backends/pytorch/libtriton_pytorch.so \
  /opt/tritonserver/backends/pytorch/libtriton_pytorch.so
```

Install runtime dependencies expected by the HSTU KV-cache demo path.
**Pytorch is required from FlexKV**.

```bash
apt-get update -y --fix-missing
apt-get install -y libzmq3-dev liburing-dev libxxhash-dev libssl-dev
apt-get install -y cmake patchelf
pip3 install pandas rich cloudpickle psutil cython
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip3 install pip3 install tritonclient[all]
```

### Step 5.3: Stage the Exported Model and Libraries for Triton Server

After completing the export flow from Step 2, copy the generated AOTI package into the versioned Triton model directory:

```bash
cd ${HSTU_DIR}
rm -rf inference_aoti/triton_aoti/hstu_gr_ranking_kvcache/1/
cp -apr inference_aoti/hstu_gr_ranking_model inference_aoti/triton_aoti/hstu_gr_ranking_kvcache/1
```

This layout is required because Triton expects a versioned model directory under the repository.

The Triton model config must remain aligned with the export contract.
If the export wrapper changes its output contract, update triton model configs ([inference_aoti/triton_aoti/hstu_gr_ranking_kvcache/config.pbtxt](./inference_aoti/triton_aoti/hstu_gr_ranking_kvcache/config.pbtxt)) accordingly.

The HSTU aoti model also expects following runtime libraries set in `LD_LIBRARY_PATH` and `LD_PRELOAD`:


```text
${HSTU_DIR}/triton_libs
├─── libhstu_cuda_ops_runtime.so
├─── libpaged_kvcache_ops_runtime.so
├─── emb
│    └─── inference_emb_ops.so
├─── recsys_kvcache_manager
│    └─── kcache_manager_ops.so
├─── hstu_attn
│    └─── fbgemm_gpu_experimental_hstu.so
├─── fbgemm_gpu
│    ├─── fbgemm_gpu_py.so
│    ├─── fbgemm_gpu_sparse_async_cumsum.so
│    └─── ...
├─── pynve
│    ├─── libnve-common.so
│    └─── libnve-torch-ops.so
├─── lib/...
└─── ...
```

Example LD_PRELOAD value:

```bash
TRITON_LIBS=${HSTU_DIR}/triton_libs
LD_PRELOAD="$TRITON_LIBS/pynve/libnve-common.so:$TRITON_LIBS/pynve/libnve-torch-ops.so:$TRITON_LIBS/emb/inference_emb_ops.so:$TRITON_LIBS/recsys_kvcache_manager/kcache_manager_ops.so:$TRITON_LIBS/libhstu_cuda_ops_runtime.so:$TRITON_LIBS/libpaged_kvcache_ops_runtime.so:$TRITON_LIBS/hstu_attn/fbgemm_gpu_experimental_hstu.so:$TRITON_LIBS/fbgemm_gpu/fbgemm_gpu_py.so:$TRITON_LIBS/fbgemm_gpu/fbgemm_gpu_sparse_async_cumsum.so"
```

The libraries are gather from the **nvidia pytorch container used in step 1~4**:

1. `emb/inference_emb_ops.so`: from `${REPO}/corelib/dynamicemb/torch_binding_build`
2. `recsys_kvcache_manager/kcache_manager_ops.so`: from `${REPO}/corelib/recsys_kvcache_manager/build`
3. `fbgemm_gpu/`: from `/usr/local/lib/python3.12/dist-packages/fbgemm_gpu/`
4. `hstu_attn/`: from `/usr/local/lib/python3.12/dist-packages/hstu/`
5. `pynve/`: from `/usr/local/lib/python3.12/dist-packages/pynve/`
6. `libhstu_cuda_ops_runtime.so`: from `${HSTU_DIR}/inference_aoti/cpp_inference/build`
7. `libpaged_kvcache_ops_runtime.so`: from `${HSTU_DIR}/inference_aoti/cpp_inference/build`
8. `libpng16.so.16`: from `/usr/lib/x86_64-linux-gnu/`
9. `lib/`: from `/usr/local/lib/`

Before launching Triton, verify the expected critical files exist:

```bash
export TRITON_LIBS=${HSTU_DIR}/triton_libs

test -f ${TRITON_LIBS}/emb/inference_emb_ops.so
test -f ${TRITON_LIBS}/recsys_kvcache_manager/kcache_manager_ops.so
test -f ${TRITON_LIBS}/libhstu_cuda_ops_runtime.so
test -f ${TRITON_LIBS}/libpaged_kvcache_ops_runtime.so
test -f ${TRITON_LIBS}/hstu_attn/fbgemm_gpu_experimental_hstu.so
test -f ${TRITON_LIBS}/fbgemm_gpu/fbgemm_gpu_py.so
test -f ${TRITON_LIBS}/fbgemm_gpu/fbgemm_gpu_sparse_async_cumsum.so
test -f ${TRITON_LIBS}/pynve/libnve-common.so
test -f ${TRITON_LIBS}/pynve/libnve-torch-ops.so
```

### Step 5.4: Triton Server AOTI Model Test

The Triton model uses the same KV-cache runtime contract as the native replay executable.

Start the FlexKV server runtime in background:

```bash
cd ${HSTU_DIR}
rm -f ${ENV_FILE}

python3 inference_aoti/start_flexkv_server_for_kvcache_cpp.py --env_file ${ENV_FILE} > kvcache_server.log 2>&1 &
cat ${ENV_FILE} && source ${ENV_FILE}
```

Check `kvcache_server.log` for the status.

Setup `LD_PRELOAD` and `LD_LIBRARY_PATH` to ensure the required custom ops and runtime libraries are visible to Triton, and start the Triton Server in background:

```bash
cd ${HSTU_DIR}
rm -f triton_server.log
export TRITON_LIBS=${HSTU_DIR}/triton_libs

LD_PRELOAD="$TRITON_LIBS/pynve/libnve-common.so:$TRITON_LIBS/pynve/libnve-torch-ops.so:$TRITON_LIBS/emb/inference_emb_ops.so:$TRITON_LIBS/recsys_kvcache_manager/kcache_manager_ops.so:$TRITON_LIBS/libhstu_cuda_ops_runtime.so:$TRITON_LIBS/libpaged_kvcache_ops_runtime.so:$TRITON_LIBS/hstu_attn/fbgemm_gpu_experimental_hstu.so:$TRITON_LIBS/fbgemm_gpu/fbgemm_gpu_py.so:$TRITON_LIBS/fbgemm_gpu/fbgemm_gpu_sparse_async_cumsum.so" \
LD_LIBRARY_PATH="/usr/local/cuda/compat/lib.real:/opt/hpcx/ucc/lib/:/opt/hpcx/ucx/lib/:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/lib/python3.12/dist-packages/torch/lib/:$TRITON_LIBS/fbgemm_gpu/:$TRITON_LIBS/lib:$TRITON_LIBS" \
tritonserver --model-repository=${HSTU_DIR}/inference_aoti/triton_aoti/ > triton_server.log 2>&1 &
```

Check `triton_server.log` for the status.

Run the request sender against from the dumped replay tensors:

```bash
cd ${HSTU_DIR}
python3 inference_aoti/send_one_kvcache_triton_request.py \
  --dump_dir inference_aoti/export_test_dump \
  --batch_index 0 \
  --url localhost:8000 \
  --model_name hstu_gr_ranking_kvcache
```

This request path validates that:

1. Triton can load the exported AOTI package
2. the NVE model-init hook is working
3. the custom operator libraries are visible
4. the KV-cache runtime service is reachable
5. the model input/output contract matches the dumped replay data
