# Run Python Export + C++ AOTI Inference Demo

This document explains the full workflow:

1. Run the Python export pipeline (`test_inference_emb`) to generate:
   - dumped embedding checkpoints
   - AOTInductor packaged model (`model.pt2`)
   - C++-readable tensor inputs/expected output (`keys.pt`, `offsets.pt`, `embeddings.pt`)
2. Build the C++ inference demos.
3. Run the C++ E2E demo and compare output numerically.

---

## 1) Prerequisites

- Linux with CUDA GPU available
- Python environment with PyTorch + your dynamic embedding dependencies installed
- Custom op library required at:
  - `corelib/dynamicemb/torch_binding_build/inference_emb_ops.so`

---

## 2) Build custom ops library (`inference_emb_ops.so`)

From repository root:

```bash
cd /home/junyiq/newscratch/february/recsys-dynmicemb-alex/corelib/dynamicemb
mkdir -p torch_binding_build
cd torch_binding_build
cmake ..
make -j
```

Expected output:

- `corelib/dynamicemb/torch_binding_build/inference_emb_ops.so`

Optional quick check:

```bash
ls -l /home/junyiq/newscratch/february/recsys-dynmicemb-alex/corelib/dynamicemb/torch_binding_build/inference_emb_ops.so
```

---

## 3) Run Python export pipeline

From repository root:

```bash
cd /home/junyiq/newscratch/february/recsys-dynmicemb-alex
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

## 4) Build C++ inference demo

```bash
cd /home/junyiq/newscratch/february/recsys-dynmicemb-alex/examples/hstu/inference/aoti_demo
mkdir -p build
cd build

CMAKE_PREFIX_PATH="$(python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "share", "cmake"))')" cmake ..
cmake --build . --config Release -j
```

This builds:

- `inference_embedding_aoti_demo`
- `inference_embedding_aoti_e2e_demo`

---

## 5) Run C++ E2E demo (recommended)

### Run for `num_tables_2`

```bash
cd /home/junyiq/newscratch/february/recsys-dynmicemb-alex/examples/hstu/inference/aoti_demo/build

./inference_embedding_aoti_e2e_demo \
  /home/junyiq/newscratch/february/recsys-dynmicemb-alex/examples/hstu/inference/inference_emb_dump/num_tables_2/model.pt2 \
  /home/junyiq/newscratch/february/recsys-dynmicemb-alex/examples/hstu/inference/inference_emb_dump/num_tables_2/keys.pt \
  /home/junyiq/newscratch/february/recsys-dynmicemb-alex/examples/hstu/inference/inference_emb_dump/num_tables_2/offsets.pt \
  /home/junyiq/newscratch/february/recsys-dynmicemb-alex/examples/hstu/inference/inference_emb_dump/num_tables_2/embeddings.pt \
  /home/junyiq/newscratch/february/recsys-dynmicemb-alex/corelib/dynamicemb/torch_binding_build/inference_emb_ops.so
```

### Run for `num_tables_3`

```bash
cd /home/junyiq/newscratch/february/recsys-dynmicemb-alex/examples/hstu/inference/aoti_demo/build

./inference_embedding_aoti_e2e_demo \
  /home/junyiq/newscratch/february/recsys-dynmicemb-alex/examples/hstu/inference/inference_emb_dump/num_tables_3/model.pt2 \
  /home/junyiq/newscratch/february/recsys-dynmicemb-alex/examples/hstu/inference/inference_emb_dump/num_tables_3/keys.pt \
  /home/junyiq/newscratch/february/recsys-dynmicemb-alex/examples/hstu/inference/inference_emb_dump/num_tables_3/offsets.pt \
  /home/junyiq/newscratch/february/recsys-dynmicemb-alex/examples/hstu/inference/inference_emb_dump/num_tables_3/embeddings.pt \
  /home/junyiq/newscratch/february/recsys-dynmicemb-alex/corelib/dynamicemb/torch_binding_build/inference_emb_ops.so
```

Expected successful ending message:

```text
E2E comparison passed.
```

---

## 6) Troubleshooting

- If you changed Python export logic, rerun:
  - `python examples/hstu/inference/test_export_demo.py`
  to regenerate all `.pt2` / `.pt` artifacts.

- If C++ code changed, rebuild:
  - `cmake --build . --config Release -j`

- If custom op schema errors appear (`INFERENCE_EMB::*`), ensure:
  - `inference_emb_ops.so` path is correct
  - the `.so` is loadable on your machine (dependencies available)

- If CUDA errors occur, verify GPU visibility and CUDA compatibility in your environment.
