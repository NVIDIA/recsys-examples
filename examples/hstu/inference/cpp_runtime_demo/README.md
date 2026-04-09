# Dense/full-sparse PT2 C++ demos

This folder now contains:

- `dense_module_aoti_demo`: loads `dense_module.pt2` and runs one inference with random CUDA inputs.
- `inferece_hstu_gr_ranking_exported_model`: loads `hstu_gr_ranking_model.pt2`, reads dumped tensors from `export_inference_gr_ranking.py`, runs runtime inference, and compares with dumped `ref_logits` using `torch::allclose`.

## Files

- [dense_module_aoti_demo.cpp](dense_module_aoti_demo.cpp)
- [inferece_hstu_gr_ranking_exported_model.cpp](inferece_hstu_gr_ranking_exported_model.cpp)
- [CMakeLists.txt](CMakeLists.txt)

## Important limitation

The current export in [../inference_gr_ranking_export.py](../inference_gr_ranking_export.py) uses pytree inputs:

- `HSTUBatch`
- `Dict[str, JaggedTensor]`

So the C++ runtime sees a **flattened tensor list**, not those Python container objects.

This demo therefore hardcodes the current flattened input convention.

### Current flattened input order

1. `batch.features.values()`
2. `batch.features.lengths()`
3. `batch.num_candidates`
4. `embeddings["user_id"].values()`
5. `embeddings["user_id"].lengths()`
6. `embeddings["video_id"].values()`
7. `embeddings["video_id"].lengths()`
8. `embeddings["user_active_degree"].values()`
9. `embeddings["user_active_degree"].lengths()`
10. `embeddings["follow_user_num_range"].values()`
11. `embeddings["follow_user_num_range"].lengths()`
12. `embeddings["fans_user_num_range"].values()`
13. `embeddings["fans_user_num_range"].lengths()`
14. `embeddings["friend_user_num_range"].values()`
15. `embeddings["friend_user_num_range"].lengths()`
16. `embeddings["register_days_range"].values()`
17. `embeddings["register_days_range"].lengths()`
18. `embeddings["action_weights"].values()`
19. `embeddings["action_weights"].lengths()`
### HSTU CUDA Operators Support

The model uses custom HSTU CUDA operators (`torch.ops.hstu_cuda_ops.compute_block_workloads`, `concat_2D_jagged_tensors_forward`, `concat_2D_jagged_tensors_backward`) which are called from the JaggedTensorOpFunction during batch processing. These operators get traced into the `.pt2` package during export.

At C++ runtime, PyTorch needs to resolve the operator schema for these HSTU operators. The CMakeLists.txt is configured to:

1. **Find and link hstu_cuda_ops library if available** - This statically registers operators at link time
2. **Dynamically load hstu_cuda_ops.so at runtime** - If linking wasn't possible, the demo attempts to load the shared library dynamically

**If you get "Could not find schema for hstu_cuda_ops::compute_block_workloads":**

1. **Option 1: Build and link hstu_cuda_ops (Recommended)**
    - Install from source: `cd examples/commons && pip install -e .`
    - Rebuild CMake - it should auto-detect and link the library
    - Check CMake output for: `Found hstu_cuda_ops library: ...`

2. **Option 2: Set CMake library path explicitly**
    ```bash
    # Find where hstu_cuda_ops is installed (after pip install -e .)
    python3 -c "import hstu_cuda_ops; import os; print(os.path.dirname(hstu_cuda_ops.__file__))"
    # Use output directory path in CMake
    cmake -S . -B build-release \
             -DTorch_DIR=... \
             -DCMAKE_PREFIX_PATH=/path/from/above
    ```

3. **Option 3: Runtime dynamic loading**
    - Install hstu_cuda_ops: `cd examples/commons && pip install -e .`
    - Rebuild - the demo will attempt to dynamically load it at runtime
    - Supported paths: system library paths and Python site-packages


That embedding order matches the current merge behavior in [../modules/inference_embedding.py](../modules/inference_embedding.py): dynamic tables first, then static tables.

If the Python export changes, this demo may also need to change.

## Build

### FBGEMM Operator Support

The model uses FBGEMM operators (specifically `torch.ops.fbgemm.asynchronous_complete_cumsum`) which are called from the HSTUProcessor during batch processing. These operators get traced into the `.pt2` package during export.

At C++ runtime, PyTorch needs to resolve the operator schema for these FBGEMM operators. The CMakeLists.txt is configured to:

1. **Find and link fbgemm_gpu library if available** - This statically registers operators at link time
2. **Dynamically load fbgemm_gpu_py.so at runtime** - If linking wasn't possible, the demo attempts to load the shared library dynamically

**If you get "Could not find schema for fbgemm::asynchronous_complete_cumsum":**

1. **Option 1: Link against fbgemm_gpu (Recommended)**
   - Ensure fbgemm_gpu is installed: `pip install fbgemm-gpu`
   - Rebuild CMake - it should auto-detect and link the library
   - Check CMake output for: `Found fbgemm_gpu library: ...`

2. **Option 2: Install system-wide**
   - Install fbgemm_gpu package which includes `libfbgemm_gpu_py.so`
   - Rebuild - the demo will attempt to dynamically load it at runtime

3. **Option 3: Set CMake library path explicitly**
   ```bash
   python3 -c "import fbgemm_gpu; import os; print(os.path.dirname(fbgemm_gpu.__file__))"
   # Use the output path as CMAKE_PREFIX_PATH
   cmake -S . -B build-release \
         -DTorch_DIR=... \
         -DCMAKE_PREFIX_PATH=/path/to/fbgemm_gpu/lib
   ```

### Build Instructions

```bash
cd examples/hstu/inference/cpp_runtime_demo
mkdir -p build-release
cmake -S . -B build-release -DTorch_DIR=/usr/local/lib/python3.12/dist-packages/torch/share/cmake/Torch
cmake --build build-release -j
```

For another separate build:

```bash
cmake --build build-debug -j
```

How to find : 
```bash
python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
```

## Run

```bash
./build-release/dense_module_aoti_demo ../../dense_module.pt2 model 0 1 64 10 512
```

Arguments:

1. package path
2. model name, default `model`
3. CUDA device index, default `0`
4. batch size, default `1`
5. history length, default `64`
6. number of candidates, default `10`
7. embedding dimension, default `512`

Example:

```bash
./build-release/dense_module_aoti_demo /abs/path/to/dense_module.pt2 model 0 1 64 10 512
```

### Full sparse runtime demo

Run all dumped batches:

```bash
./build-release/inferece_hstu_gr_ranking_exported_model /abs/path/to/hstu_gr_ranking_model.pt2 /abs/path/to/export_test_dump
```

Run a specific dumped batch index (e.g. `0`):

```bash
./build-release/inferece_hstu_gr_ranking_exported_model /abs/path/to/hstu_gr_ranking_model.pt2 /abs/path/to/export_test_dump model 0 0
```

Arguments:

1. `hstu_gr_ranking_model.pt2` path
2. dump folder path (contains `batch_XXXXXX_values.pt`, `lengths.pt`, `num_candidates.pt`, `ref_logits.pt`)
3. model name, default `model`
4. CUDA device index, default `0`
5. batch index, default `-1` (run all batches)
6. `inference_emb_ops.so` path (optional override)
7. `libnve_torch.so` path (optional override)
8. `libsplitops_cpu.so` path (optional override)
9. `libhstu_cuda_ops_runtime.so` path (optional override)

## Notes

- This demo assumes the package was compiled for CUDA.
- It uses random float32 embedding tensors, which matches the current sparse embedding output dtype expected by the dense module path.
- It prints the package call spec and metadata before running inference.
