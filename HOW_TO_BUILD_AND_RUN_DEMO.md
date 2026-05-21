# Build `inference_emb_ops.so`

`inference_emb_ops.so` registers the `torch.ops.INFERENCE_EMB.*` operators used
by DynamicEmb exportable inference tables and the HSTU C++ AOTInductor demo.

## Build

From the repository root:

```bash
cd corelib/dynamicemb
mkdir -p torch_binding_build
cd torch_binding_build
cmake ..
make -j
```

Expected output:

```text
corelib/dynamicemb/torch_binding_build/inference_emb_ops.so
```

## Quick Check

From the repository root:

```bash
python3 - <<'PY'
import os
import torch

path = os.path.abspath("corelib/dynamicemb/torch_binding_build/inference_emb_ops.so")
torch.ops.load_library(path)
print("loaded", path)
PY
```

## Use with HSTU Export

The HSTU Python export path loads this library from `DYNAMICEMB_OPS_LIB_DIR`:

```bash
cd examples/hstu
export DYNAMICEMB_OPS_LIB_DIR=$(realpath ../../corelib/dynamicemb/torch_binding_build)
python3 ./inference/export_inference_gr_ranking.py \
  --gin_config_file ./inference/configs/kuairand_1k_inference_ranking.gin \
  --checkpoint_dir ${PATH_TO_CHECKPOINT}
```

For the full Python export plus C++ runtime workflow, see
[examples/hstu/inference/GUIDE_TO_RUN_CPP_INFERENCE_DEMO.md](./examples/hstu/inference/GUIDE_TO_RUN_CPP_INFERENCE_DEMO.md).
