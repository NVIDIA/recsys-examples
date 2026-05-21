# Run Python Export + C++ AOTI Inference Demo

This document explains the full workflow:

1. Build and install required dependency:
  - `inference_emb_ops.so` for dynamicemb custom ops
  -  NVEmbedding package and shared libraries
2. Run the Python export pipeline to generate:
  - exported model → `model.pt2`, `metadata.json`, `weights/{nve_layer_module_name}.nve`
  - test cases → tensors `values.pt`, `lengths.pt`, `num_candidates.pt`, `ref_logits.pt` for multiple batches
3. Build the C++ E2E inference demos.
4. Run the C++ inference and compare output numerically.

---

## Path placeholders used in this doc

- `{RECSYS_DIR}`: root of `recsys-examples`
- `{NVE_DIR}`: root of `nv-embedding-cache` if building NVEmbedding outside the Docker image
- `{RECSYS_INFERENCE_DIR}`: `{RECSYS_DIR}/examples/hstu/inference`
- `{CPP_INFERENCE_BUILD_DIR}`: `{RECSYS_INFERENCE_DIR}/cpp_inference/build`
- `{CPP_INFERENCE_LIB_DIR}`: `{RECSYS_INFERENCE_DIR}/cpp_inference/lib`

---

## 1) Prerequisites

- Linux with CUDA GPU available
- CUDA-enabled PyTorch environment compatible with the repository Dockerfile
- DynamicEmb installed from this repository
- NVEmbedding (`pynve`) installed; the repository Docker image installs it from [nv-embedding-cache](https://github.com/NVIDIA/nv-embedding-cache)

---

## 2) Build custom ops library (`inference_emb_ops.so`)

From repository root:

```bash
cd {RECSYS_DIR}/corelib/dynamicemb
mkdir -p torch_binding_build && cd torch_binding_build
cmake .. && make -j
```

Expected output:

- `{RECSYS_DIR}/corelib/dynamicemb/torch_binding_build/inference_emb_ops.so`

---

## 3) Build and install NVEmbedding

The repository Docker image already installs NVEmbedding. If you are building a source environment outside Docker, install it from `nv-embedding-cache`:

```bash
cd {NVE_DIR}  # at the repository root dir
git submodule update --init --recursive
git clone https://github.com/NVIDIA/NVTX.git third_party/NVTX
CPLUS_INCLUDE_PATH=$(realpath ./third_party/NVTX/c/include/):${CPLUS_INCLUDE_PATH} pip install --no-deps .
```

Expected output:

- Python package: `pynve`
- Shared libraries discoverable under the installed `pynve` package, including `libnve-common.so` and `libnve-torch-ops.so`

---

## 4) Run Python export pipeline

From repository root:

```bash
cd {RECSYS_DIR}/examples/hstu/
export DYNAMICEMB_OPS_LIB_DIR=$(realpath ../../corelib/dynamicemb/torch_binding_build/)
python3 ./inference/export_inference_gr_ranking.py \
  --gin_config_file ./inference/configs/kuairand_1k_inference_ranking.gin \
  --checkpoint_dir ${PATH_TO_CHECKPOINT}
```

---

## 5) Build C++ inference demos

```bash
cd {RECSYS_INFERENCE_DIR}/cpp_inference
CMAKE_PREFIX_PATH="$(python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "share", "cmake"))')" cmake -S . -B build
cmake --build build --config Release -j
```

Expected output:

- Output library: `libhstu_cuda_ops_runtime.so`
- C++ Inference Executable: `inference_hstu_gr_ranking_exported_model`

---

## 6) Run C++ E2E demos

```bash
cd {RECSYS_INFERENCE_DIR}/cpp_inference

./build/inference_hstu_gr_ranking_exported_model \
  {RECSYS_INFERENCE_DIR}/hstu_gr_ranking_model \
  {RECSYS_INFERENCE_DIR}/export_test_dump
```

> ### Note (1)
> FBGEMM, HSTU, DynamicEmb inference ops, and NVEmbedding shared libraries are loaded by the C++ inference executable.
> The default lookup paths match the repository Docker image layout; pass explicit paths to the executable when using a custom environment.
>
> ### Note (2)
> The current export in `export_inference_gr_ranking.py` uses pytree inputs:
> 
> - `HSTUBatch`
> 
> So the C++ runtime sees a **flattened tensor list**, not those Python container objects. Current flattened input order
> 
> 1. `batch.features.values()`
> 2. `batch.features.lengths()`
> 3. `batch.num_candidates`
