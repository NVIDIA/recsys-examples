# Build and Run Guide (inference_emb_ops)

## 1) Build `inference_emb_ops.so`

From repository root:

```bash
cd corelib/dynamicemb
mkdir -p torch_binding_build
cd torch_binding_build
cmake ..
make -j
```

Expected output library:

- `corelib/dynamicemb/torch_binding_build/inference_emb_ops.so`

Optional quick check:

```bash
ls -l corelib/dynamicemb/torch_binding_build/inference_emb_ops.so
```

---

## 2) Run demo: `test_export_demo.py`

From repository root:

```bash
# Optional: ensure local dynamicemb package is importable
export PYTHONPATH=$PWD/corelib/dynamicemb:$PYTHONPATH

python examples/hstu/inference/test_export_demo.py
```

What the demo does:

- Loads `inference_emb_ops.so`
- Builds a small inference module using `INFERENCE_EMB` custom ops
- Runs eager forward test
- Runs `torch.export` smoke test

If you see a warning that the `.so` cannot be loaded, re-check build output and path:

- `corelib/dynamicemb/torch_binding_build/inference_emb_ops.so`
