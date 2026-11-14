# CPU-only HostLayer: build and test

**Target:** CPU-only PyTorch + stock driverless NVE + `cudart`.
Build only DynamicEmb's CPU inference targets; test real HostLayer lookup,
export/replay and updates.

| Environment | Requirements |
|---|---|
| Builder | CUDA toolkit/nvcc, C++20 compiler, Python development headers, Make and a recursive checkout |
| CPU runtime | Matching `cudart`; C++ compiler/Python headers for AOT export. No GPU, driver or CUDA toolkit needed. |

Commands below use Ubuntu 24.04, Python 3.12 and CUDA 13.3 at `/usr/local/cuda`.
In an activated Python environment **on both builder and runtime**, install:

```bash
unset PYTHONPATH LD_PRELOAD
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install setuptools wheel psutil numpy pytest 'cmake<4' ninja
```

## 1. Build

Start in `corelib/dynamicemb`. NVE uses its minimum features with optional plugins,
MPI, NUMA/hwloc, NVTX and tests disabled. Stock NVE still compiles CUDA sources;
DynamicEmb's NVE-facing bindings also need the CUDA compiler.

This package runs on CPU regardless of the installed GPU, or with none. Leave
the compilation defaults unchanged. For GPU execution, use the [regular CUDA build](../../README.md#installation)
and matching `CMAKE_CUDA_ARCHITECTURES` (H100: `90`; SM120: `120`).

[NVEHostBuild.cmake](../../cmake/NVEHostBuild.cmake) supplies the optional-feature
settings through `CMAKE_TOOLCHAIN_FILE`: pinned NVE 26.10's `pip wheel` wrapper has no
CMake `-D` or `CMAKE_ARGS` passthrough.

```bash
demb="$PWD"
nve="$(realpath ../../third_party/nv-embedding-cache)"
out="$demb/dist/host_cpu"
mkdir -p "$out"
export PATH="/usr/local/cuda/bin:$PATH" LD_LIBRARY_PATH=/usr/local/cuda/lib64

CMAKE_TOOLCHAIN_FILE="$demb/cmake/NVEHostBuild.cmake" PYNVE_BUILD_DIR="$demb/build_nve_host_cpu" \
  PYNVE_DRIVERLESS_BUILD=1 PYNVE_DISABLE_MPI=1 PYNVE_DISABLE_AVX512=1 \
  PYNVE_BUILD_SAMPLES=0 PYNVE_DISABLE_TORCH_BINDINGS=0 \
  python -m pip wheel --no-build-isolation --no-deps "$nve" -w "$out"
python -m pip install --no-deps "$out"/pynve-*.whl

DYNAMICEMB_CPU_ONLY=1 NVE_ROOT="$nve" MAX_JOBS=4 \
  python -m pip wheel --no-build-isolation --no-deps . -w "$out"
python -m pip install --no-deps "$out"/dynamicemb-*.whl

site="$(python -c 'import sysconfig; print(sysconfig.get_path("platlib"))')"
cmake -S example/host_embedding -B example/host_embedding/build_cpu \
  -DDYNAMICEMB_CPU_ONLY=ON \
  -DDYNAMICEMB_LIB_DIR="$demb/torch_binding_build_cpu" \
  -DNVE_ROOT="$nve" -DNVE_LIB_DIR="$site/pynve"
cmake --build example/host_embedding/build_cpu --parallel 4
cp example/host_embedding/build_cpu/replay_host "$out/"
cp -L /usr/local/cuda/lib64/libcudart.so.13 "$out/"
```

## 2. Install on the CPU runtime

Copy `dist/host_cpu` to `/tmp/host_cpu_artifacts`. Keep the checkout available for
the demo/test files. Use the same Python/PyTorch versions as the builder.

```bash
demb=/path/to/recsys-examples/corelib/dynamicemb
out=/tmp/host_cpu_artifacts
python -m pip install --no-deps "$out"/*.whl
site="$(python -c 'import sysconfig; print(sysconfig.get_path("platlib"))')"
export LD_LIBRARY_PATH="$site/dynamicemb/exportable_embedding:$site/pynve:$site/torch/lib:$out"
export OMP_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
mkdir -p /tmp/host_cpu_results && cd /tmp/host_cpu_results
```

## 3. Test

One loop covers initial loading/lookup, export, 32 updates, fresh Python replay
and C++ replay for all three indexers. Run outside the source tree so imports use
the installed wheel.

```bash
demo="$demb/example/host_embedding"
for indexer in identity fused_identity linear_hash_map; do
  python "$demo/incremental_cpu.py" --indexer "$indexer" --package-dir "$PWD/$indexer"
  python "$demo/example.py" --device cpu --package-dir "$PWD/$indexer" --replay
  "$out/replay_host" "$PWD/$indexer" -1
done

mkdir -p tests
cp "$demb"/tests/test_{host_embedding,host_incremental_update,indexer_registration_lifetime}.py tests/
python -m pytest -q tests
```

All commands should succeed; CUDA-only unit cases skip. Identity/FusedIdentity
use concurrent lookup; LinearHashMap uses paused loading. See the
[Host concurrency contract](README.md#incremental-updates-on-cpu).
For pooled FP16, add `--dtype float16 --pooling 2` to `incremental_cpu.py`.
See the [example file map and API](README.md) for details.
