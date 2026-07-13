# NVE model-init hook for the Triton PyTorch (AOTI) backend

`libnve_init_hook.so` loads an HSTU model's NVE embedding weights into the
process-global `NVELayerRegistry` at model load, so `nve_ops::embedding_lookup`
resolves at inference time. Those weights live **outside** `model.pt2`
(`<model_dir>/metadata.json` + `<model_dir>/weights/*.nve`), so loading the
package alone does not load them.

This is the **plug-in** for the generic `MODEL_INIT_LIBRARY` hook in the Triton
PyTorch backend: the backend `dlopen()`s this `.so` and calls its
`triton_pytorch_model_init(model_dir, device_index)` once at model load. The
backend itself links nothing NVE-specific — only this library does.

```
Triton PyTorch backend (generic)          this repo (NVE-specific)
  reads MODEL_INIT_LIBRARY param   ──►   libnve_init_hook.so
  dlopen + call the entry point          └─ nve::LayerDirectory(model_dir, dev)
```

## Build

Build in the same `nvcr.io/nvidia/pytorch:XX.YY-py3` image whose libtorch
matches the `tritonserver:XX.YY-py3` you deploy on, with NVE available
(`pip install` nv-embedding-cache → libs at `.../dist-packages/pynve`, headers
under `/workspace/deps/nve`):

```bash
cd nve_init_hook && mkdir build && cd build
cmake .. -DNVE_ROOT=/workspace/deps/nve \
         -DNVE_LIB_DIR=/usr/local/lib/python3.12/dist-packages/pynve
make -j
# -> libnve_init_hook.so
```

## Deploy

In the model's `config.pbtxt`:

```
parameters {
  key: "MODEL_INIT_LIBRARY"
  value: { string_value: "/abs/path/to/libnve_init_hook.so" }
}
```

At startup the NVE runtime libraries must still be discoverable
(`LD_LIBRARY_PATH` → `.../pynve`) and the HSTU custom ops registered, exactly as
for native inference. On model load you should see:

```
[nve-init-hook] loaded N NVE layer(s) into the registry
Ran model-init hook ".../libnve_init_hook.so" for model "hstu_gr_ranking" ...
```

> **Requires** the `MODEL_INIT_LIBRARY` hook in the Triton PyTorch backend.
> Until that lands upstream, mount a backend `.so` built with the patch over the
> stock backend in the official image.
