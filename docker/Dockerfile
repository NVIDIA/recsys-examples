ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.05-py3
ARG BASE_FBGEMM_IMAGE=base_fbgemm
ARG DEVEL_IMAGE=devel

# ============================================================================
# Stage 1: base_fbgemm — FBGEMM Path A + TorchRec (version-coupled with FBGEMM).
# Override BASE_FBGEMM_IMAGE to consume a pre-built image and skip the build.
# ============================================================================
FROM ${BASE_IMAGE} AS base_fbgemm
ARG TRITONSERVER_BUILD

WORKDIR /workspace/deps

# Layer 0-pre: bootstrap when base is nvcr/tritonserver. pytorch base ships
# all these (python alias, cmake/patchelf, ninja, torch, tabulate) — branch skips.
# tabulate is a FBGEMM v1.5.0 setup.py import; only missing on tritonserver.
# ninja: without it torch BuildExtension serial-fallbacks → HSTU build ~3h vs ~10min.
RUN if [ "${TRITONSERVER_BUILD}" = "1" ]; then \
      ln /bin/python3 /bin/python && \
      apt-get update -y --fix-missing && \
      apt-get install -y cmake patchelf && \
      pip3 install --no-cache-dir pandas rich cloudpickle psutil cython tabulate ninja && \
      pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130 && \
      apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*; \
    fi

# Layer 0a: FBGEMM Path A (slow ~55min nvcc compile).
RUN pip install --no-cache-dir setuptools-git-versioning scikit-build && \
    git clone --recursive -b v1.5.0 https://github.com/pytorch/FBGEMM.git fbgemm && \
    cd fbgemm/fbgemm_gpu && \
    python3 setup.py install --build-target=default --build-variant=cuda -DTORCH_CUDA_ARCH_LIST="7.5 8.0 9.0 10.0 12.0"

# Layer 0b: TorchRec (tightly coupled with FBGEMM C++ ops version).
RUN pip install --no-deps tensordict orjson && \
    git clone --recursive -b release/V1.5.0 https://github.com/pytorch/torchrec.git torchrec && \
    cd torchrec && \
    pip install --no-deps .

# ============================================================================
# Stage 2: devel — base_fbgemm + system setup + Megatron + pip
# + flash-attention + FlexKV/NVE + HSTU Path B.
# ============================================================================
FROM ${BASE_FBGEMM_IMAGE} AS devel
ARG TARGETPLATFORM
ARG TRITONSERVER_BUILD

WORKDIR /workspace/deps

# -- Layer 1: system setup + arch symlinks.

RUN apt-get update -y --fix-missing && apt-get install -y liburing-dev libxxhash-dev libssl-dev libzmq3-dev && \
    ARCH=$([ "${TARGETPLATFORM}" = "linux/arm64" ] && echo "aarch64" || echo "x86_64") && \
    rm -rf /usr/lib/${ARCH}-linux-gnu/libnvidia-ml.so.1 && \
    if [ ${ARCH} = "aarch64" ]; then \
      ln -s /usr/local/cuda-13/targets/sbsa-linux/lib/stubs/libnvidia-ml.so  /usr/lib/${ARCH}-linux-gnu/libnvidia-ml.so.1; \
    else \
      ln -s /usr/local/cuda-13/targets/${ARCH}-linux/lib/stubs/libnvidia-ml.so  /usr/lib/${ARCH}-linux-gnu/libnvidia-ml.so.1; \
    fi && \
    if [ "${TARGETPLATFORM}" = "linux/arm64" ]; then \
      CUDA_TARGET_ARCH=sbsa; \
    elif [ "${TARGETPLATFORM}" = "linux/amd64" ]; then \
      CUDA_TARGET_ARCH=x86_64; \
    else \
      CUDA_TARGET_ARCH=$(uname -m); \
    fi && \
    ln -sf /usr/local/cuda-13/targets/${CUDA_TARGET_ARCH}-linux/include/cccl/cuda \
    /usr/local/cuda/include/cuda && \
    apt update -y --fix-missing && \
    apt install -y gdb && \
    apt autoremove -y && apt clean && rm -rf /var/lib/apt/lists/*

# -- Layer 2: pip dependencies + Megatron-LM ---
# core_v0.13.1 picks up an optional skip for count_zeros_in_grad in
# ChainedOptimizer.step (~4 ms / step under HSTU bf16 when the stat is unused).
RUN git clone -b core_v0.13.1 https://github.com/NVIDIA/Megatron-LM.git megatron-lm && \
    pip install --no-deps -e ./megatron-lm && \
    pip install --no-cache-dir torchx gin-config torchmetrics==1.0.3 typing-extensions iopath pyvers expiring_dict \
        cloudpickle \
        'quack-kernels>=0.3.3,<0.5' 'apache-tvm-ffi>=0.1.6' torch-c-dlpack-ext pre-commit \
        pytest pytest-shard pytest-rerunfailures pytest-xdist pytest-flakefinder

# -- Layer 3: FlashAttention --
ARG FLASH_ATTN_ARBITRARY_MASK_COMMIT=b56db721d65afadd8cfe154cd12e83693bf0aba3
RUN git clone https://github.com/jiayus-nvidia/flash-attention.git flash-attention && \
    cd flash-attention && \
    git checkout ${FLASH_ATTN_ARBITRARY_MASK_COMMIT} && \
    pip install --no-deps --no-build-isolation -e ./flash_attn/cute

# -- Layer 4, 5: FlexKV and NVE. PYNVE_DISABLE_AVX512 only meaningful on x86;
# leave it unset on arm64 (AVX512 doesn't exist there).
COPY third_party/FlexKV /workspace/deps/FlexKV
RUN cd FlexKV && mkdir -p build && cd build && \
    TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0 10.0 12.0" cmake .. && \
    TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0 10.0 12.0" cmake --build . -j8 && \
    cd .. && mkdir -p flexkv/lib && cp -P build/lib/*.so* flexkv/lib/ && \
    MAX_JOBS=8 FLEXKV_DEBUG=0 TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0 10.0 12.0" python3 setup.py install && \
    cd /workspace/deps && \
    git clone -b v26.05 --recursive https://github.com/NVIDIA/nv-embedding-cache.git nve && \
    cd nve && git submodule update --init --recursive && \
    cd third_party/ && git clone https://github.com/NVIDIA/NVTX.git NVTX && cd .. && \
    if [ "${TARGETPLATFORM}" = "linux/arm64" ]; then AVX512_ENV=""; else AVX512_ENV="PYNVE_DISABLE_AVX512=1"; fi && \
    env TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0 10.0 12.0" PYNVE_WITH_TORCH_BINDINGS=1 ${AVX512_ENV} \
    CPLUS_INCLUDE_PATH=$(realpath ./third_party/NVTX/c/include/):${CPLUS_INCLUDE_PATH} pip install --no-deps .

# Install fbgemm_gpu_hstu (package: fbgemm_gpu_hstu, import: hstu) from submodule.
COPY third_party/FBGEMM /workspace/deps/fbgemm_hstu
RUN cd /workspace/deps/fbgemm_hstu/fbgemm_gpu/experimental/hstu && \
    export HSTU_ARCH_LIST="8.0 9.0 10.0" && \
    if [ "${TARGETPLATFORM}" = "linux/amd64" ]; then \
      export HSTU_DISABLE_120=TRUE; \
      export HSTU_ARCH_LIST="8.0 9.0 10.0 12.0"; \
    fi && \
    HSTU_DISABLE_86OR89=FALSE \
    HSTU_DISABLE_ARBITRARY=TRUE \
    HSTU_DISABLE_LOCAL=TRUE \
    HSTU_DISABLE_RAB=TRUE \
    HSTU_DISABLE_DRAB=TRUE \
    HSTU_DISABLE_FP16=TRUE \
    pip install --no-build-isolation . && \
    rm -rf /workspace/deps/fbgemm_hstu

# ============================================================================
# Stage 3: build — devel + recsys-examples in-tree builds (dynamicemb, commons,
# kvcache_manager) and runtime libcuda link.
# ============================================================================
FROM ${DEVEL_IMAGE} AS build
ARG TRITONSERVER_BUILD

RUN rm -rf /workspace/recsys-examples
WORKDIR /workspace/recsys-examples
COPY . .

RUN cd /workspace/recsys-examples/corelib/dynamicemb && \
    TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0 10.0 12.0" python setup.py install && \
    cd /workspace/deps && rm -rf nvcomp && \
    NVCOMP_ARCH=$([ "$(uname -m)" = "aarch64" ] && echo "linux-sbsa" || echo "linux-x86_64") && \
    wget https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/${NVCOMP_ARCH}/nvcomp-${NVCOMP_ARCH}-5.1.0.21_cuda12-archive.tar.xz && \
    tar -xf nvcomp-${NVCOMP_ARCH}-5.1.0.21_cuda12-archive.tar.xz && \
    mv nvcomp-${NVCOMP_ARCH}-5.1.0.21_cuda12-archive nvcomp && \
    rm nvcomp-${NVCOMP_ARCH}-5.1.0.21_cuda12-archive.tar.xz && \
    cd /workspace/recsys-examples/examples/commons && \
    TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0 10.0 12.0" python3 setup.py install && \
    cd /workspace/recsys-examples/corelib/recsys_kvcache_manager && \
    TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0 10.0 12.0" pip3 install --no-build-isolation . && \
    if [ "${TRITONSERVER_BUILD}" != "1" ]; then \
      rm -f /usr/lib/$(uname -m)-linux-gnu/libcuda.so.1 && \
      test -e /usr/local/cuda-13/compat/lib.real/libcuda.so.1 && \
      ln -s /usr/local/cuda-13/compat/lib.real/libcuda.so.1  /usr/lib/$(uname -m)-linux-gnu/libcuda.so.1 && \
      cd /workspace/recsys-examples/corelib/dynamicemb && \
      mkdir -p torch_binding_build && cd torch_binding_build && \
      cmake .. && make -j; \
    fi

# ============================================================================
# AOTI additions: C++ replay executable, NVE init hook, Triton backend prep,
# optional checkpoint export, and Triton runtime library staging.
# ============================================================================
RUN cd /workspace/recsys-examples/corelib/recsys_kvcache_manager && \
    mkdir -p build && cd build && \
    cmake .. && make -j

RUN export CMAKE_PREFIX_PATH="$(python3 -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "share", "cmake"))')" && \
    cmake -S "./examples/hstu/inference_aoti/cpp_inference" -B "./examples/hstu/inference_aoti/cpp_inference/build" && \
    cmake --build "./examples/hstu/inference_aoti/cpp_inference/build" -j 8 && \
    cmake -S "./examples/hstu/inference_aoti/nve_init_hook" -B "./examples/hstu/inference_aoti/nve_init_hook/build" && \
    cmake --build "./examples/hstu/inference_aoti/nve_init_hook/build" -j 8

RUN python3 -m pip install --no-cache-dir --upgrade "cmake>=3.31.8" && \
    cp /usr/local/lib/libjpeg.so.62 /usr/local/lib/python3.12/dist-packages/torch/lib/libjpeg.so.62 && \
    cd /workspace/deps && rm -rf pytorch_backend && \
    git clone https://github.com/triton-inference-server/pytorch_backend.git pytorch_backend && \
    cd pytorch_backend && git checkout ceeecb7 && \
    mkdir -p build && cd build && \
    cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX:PATH="$PWD/install" \
      -DTRITON_PYTORCH_INCLUDE_PATHS="/usr/local/lib/python3.12/dist-packages/torch/include;/usr/local/lib/python3.12/dist-packages/torch/include/torch/csrc/api/include;/opt/pytorch/vision/torchvision/csrc" \
      -DTRITON_PYTORCH_LIB_PATHS="/usr/local/lib/python3.12/dist-packages/torch/lib" \
      -DTRITON_PYTORCH_ENABLE_TORCHVISION=OFF \
      -DTRITON_BACKEND_REPO_TAG=r26.05 \
      -DTRITON_CORE_REPO_TAG=r26.05 \
      -DTRITON_COMMON_REPO_TAG=r26.05 \
      .. && \
    cmake --build . -j"$(nproc)" --target install

RUN cd ./examples/hstu && \
    mkdir -p \
      triton_libs/emb \
      triton_libs/recsys_kvcache_manager \
      triton_libs/hstu_attn \
      triton_libs/fbgemm_gpu \
      triton_libs/pynve \
      triton_libs/lib \
      triton_libs/system && \
    cp -P /workspace/recsys-examples/corelib/dynamicemb/torch_binding_build/inference_emb_ops.so triton_libs/emb/ && \
    cp -P /usr/local/lib/python3.12/dist-packages/recsys_kvcache_manager/kvcache_manager_ops.so triton_libs/recsys_kvcache_manager/ && \
    cp -P /workspace/recsys-examples/examples/hstu/inference_aoti/cpp_inference/build/libhstu_cuda_ops_runtime.so triton_libs/ && \
    cp -P /workspace/recsys-examples/examples/hstu/inference_aoti/cpp_inference/build/libpaged_kvcache_ops_runtime.so triton_libs/ && \
    cp -a /usr/local/lib/python3.12/dist-packages/hstu/. triton_libs/hstu_attn/ && \
    cp -a /usr/local/lib/python3.12/dist-packages/fbgemm_gpu/. triton_libs/fbgemm_gpu/ && \
    cp -a /usr/local/lib/python3.12/dist-packages/pynve/. triton_libs/pynve/ && \
    find /usr/local/lib -maxdepth 1 \( -type f -o -type l \) \( -name '*.so' -o -name '*.so.*' \) -exec cp -P {} triton_libs/lib/ \; && \
    cp -P /usr/lib/$(uname -m)-linux-gnu/libpng16.so.16* triton_libs/system/ && \
    sed -i 's#string_value: ".*libnve_init_hook.so"#string_value: "/workspace/recsys-examples/examples/hstu/inference_aoti/nve_init_hook/build/libnve_init_hook.so"#' \
      inference_aoti/triton_aoti/hstu_gr_ranking_kvcache/config.pbtxt

WORKDIR /workspace/recsys-examples/