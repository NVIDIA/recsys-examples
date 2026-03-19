# Run Python Export + C++ AOTI Inference Demo

This document explains the full workflow:

1. Run the Python export pipeline (`test_inference_emb`) to generate:
   - dumped embedding checkpoints
  - AOTInductor packaged models (`num_tables_*/model.pt2`)
   - C++-readable tensor inputs/expected output (`keys.pt`, `offsets.pt`, `embeddings.pt`)
2. Build required shared libraries:
  - `inference_emb_ops.so` (dynamicemb custom ops)
  - `libnve_torch.so` (NVE torch custom classes)
3. Build the C++ inference demos.
4. Manually copy the two `.so` files into `examples/hstu/inference/aoti_demo/lib/`.
5. Run the C++ E2E demo and compare output numerically.

---

## Path placeholders used in this doc

- `{RECSYS_DIR}`: root of `recsys-dynmicemb-alex`
- `{NVE_DIR}`: root of `trt-recsys`
- `{RECSYS_EXAMPLES_DIR}`: `{RECSYS_DIR}/examples/hstu/inference`
- `{AOTI_DEMO_BUILD_DIR}`: `{RECSYS_EXAMPLES_DIR}/aoti_demo/build`
- `{AOTI_DEMO_LIB_DIR}`: `{RECSYS_EXAMPLES_DIR}/aoti_demo/lib`

---

## 1) Prerequisites

- Linux with CUDA GPU available
- Python environment with PyTorch + your dynamic embedding dependencies installed
- DynamicEmb custom op library required at:
  - `{RECSYS_DIR}/corelib/dynamicemb/torch_binding_build/inference_emb_ops.so`
- NVE torch class library required at:
  - `{NVE_DIR}/build_dir/lib/libnve_torch.so`

---

## 2) Build custom ops library (`inference_emb_ops.so`)

From repository root:

```bash
cd {RECSYS_DIR}/corelib/dynamicemb
mkdir -p torch_binding_build
cd torch_binding_build
cmake ..
make -j
```

Expected output:

- `corelib/dynamicemb/torch_binding_build/inference_emb_ops.so`

Optional quick check:

```bash
ls -l {RECSYS_DIR}/corelib/dynamicemb/torch_binding_build/inference_emb_ops.so
```

---

## 3) Build NVE torch class library (`libnve_torch.so`)

Placeholder command block:

```bash
cd {NVE_DIR}  # at the repository root dir
git submodule update --init --recursive
git clone https://github.com/NVIDIA/NVTX.git third_party/NVTX
cp -apr third_party/json/single_include/nlohmann include/
mkdir build_dir
CMAKE_PREFIX_PATH="$(python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "share", "cmake"))')" cmake ..
make all -j
make install
```

Expected output:

- Output library: `build_dir/lib/libnve_torch.so`

Optional quick check:

```bash
ls -l {NVE_DIR}/build_dir/lib/libnve_torch.so
```

---

## 4) Run Python export pipeline

From repository root:

```bash
cd {RECSYS_DIR}
python examples/hstu/inference/test_export_demo.py
```

This runs `test_inference_emb()` and creates output under:

- `examples/hstu/inference/inference_emb_dump/num_tables_2/`
- `examples/hstu/inference/inference_emb_dump/num_tables_3/`

Each folder contains at least:

- `model.pt2`
- `keys.pt`
- `offsets.pt`
- `embeddings.pt`

---

## 5) Build C++ inference demo

```bash
cd {RECSYS_EXAMPLES_DIR}/aoti_demo
mkdir -p build
cd build

CMAKE_PREFIX_PATH="$(python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "share", "cmake"))')" cmake ..
cmake --build . --config Release -j
```

This builds:

- `inference_embedding_aoti_e2e_demo`

> Note: CMake no longer auto-copies shared libraries.

---

## 6) Manually copy shared libraries into lib folder

```bash
mkdir -p {AOTI_DEMO_LIB_DIR}
cd {AOTI_DEMO_LIB_DIR}

cp {RECSYS_DIR}/corelib/dynamicemb/torch_binding_build/inference_emb_ops.so ./inference_emb_ops.so
cp {NVE_DIR}/build_dir/lib/libnve_torch.so ./libnve_torch.so
```

Optional quick check:

```bash
ls -l ./inference_emb_ops.so ./libnve_torch.so
```

> Both Python and C++ demo loaders now use relative paths under `{RECSYS_EXAMPLES_DIR}/aoti_demo/lib/` by default.

---

## 7) Run C++ E2E demo (recommended)

### Run for `num_tables_2`

```bash
cd {AOTI_DEMO_BUILD_DIR}

./inference_embedding_aoti_e2e_demo \
  {RECSYS_EXAMPLES_DIR}/inference_emb_dump/num_tables_2/model.pt2 \
  {RECSYS_EXAMPLES_DIR}/inference_emb_dump/num_tables_2/keys.pt \
  {RECSYS_EXAMPLES_DIR}/inference_emb_dump/num_tables_2/offsets.pt \
  {RECSYS_EXAMPLES_DIR}/inference_emb_dump/num_tables_2/embeddings.pt
```

### Run for `num_tables_3`

```bash
cd {AOTI_DEMO_BUILD_DIR}

./inference_embedding_aoti_e2e_demo \
  {RECSYS_EXAMPLES_DIR}/inference_emb_dump/num_tables_3/model.pt2 \
  {RECSYS_EXAMPLES_DIR}/inference_emb_dump/num_tables_3/keys.pt \
  {RECSYS_EXAMPLES_DIR}/inference_emb_dump/num_tables_3/offsets.pt \
  {RECSYS_EXAMPLES_DIR}/inference_emb_dump/num_tables_3/embeddings.pt
```

> The C++ binaries load `inference_emb_ops.so` and `libnve_torch.so` from `../lib` relative to the executable by default.
> Keep both files in `{AOTI_DEMO_LIB_DIR}`.

Expected successful ending message:

```text
E2E comparison passed.
```

---

## 8) Troubleshooting

- If you changed Python export logic, rerun:
  - `python examples/hstu/inference/test_export_demo.py`
  to regenerate all `.pt2` / `.pt` artifacts.

- If C++ code changed, rebuild:
  - `cmake --build . --config Release -j`

- If custom op schema errors appear (`INFERENCE_EMB::*`), ensure:
  - `inference_emb_ops.so` was copied into `aoti_demo/lib`
  - the `.so` is loadable on your machine (dependencies available)

- If custom class / class registration errors appear (`torch.classes.nve.*`), ensure:
  - `libnve_torch.so` was copied into `aoti_demo/lib`
  - the `.so` was built against a compatible PyTorch/ABI/CUDA setup
  - the copied file is readable by your current user

- If CUDA errors occur, verify GPU visibility and CUDA compatibility in your environment.
