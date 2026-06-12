ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.02-py3
ARG DEVEL_IMAGE=devel

FROM ${BASE_IMAGE} AS devel
ARG TARGETPLATFORM
ARG TRITONSERVER_BUILD

WORKDIR /workspace/deps

# -- Layer 1: system setup, arch symlinks, tritonserver deps ---
RUN apt-get update -y --fix-missing && apt-get install -y liburing-dev libxxhash-dev libssl-dev && \
    if [ "${TRITONSERVER_BUILD}" = "1" ]; then \
      ln /bin/python3 /bin/python && \
      apt-get install -y cmake patchelf && \
      pip3 install pandas rich cloudpickle psutil cython && \
      pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130; \
    fi && \
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
# Bump from core_v0.12.1 -> core_v0.13.1 to pick up
# d9608004f "Add an option to skip counting zeros in grad of ChainedOptimizer"
# (gates ChainedOptimizer.count_zeros on log_num_zeros_in_grad). 0.12.x release
# line still ships the unconditional count_zeros_fp32 call inside
# ChainedOptimizer.step, which costs ~4 ms / step on HSTU bf16 even when the
# stat is unused. Earliest tag containing the fix is core_v0.13.0; using 0.13.1
# (latest 0.13.x patch) for stability.
# nvidia-cutlass-dsl: --no-deps + --no-cache-dir + 3 explicit subpackages
# avoids a poisoned base-image pip cache wheel; keep the re-pins on the
# bundled install too.
RUN pip uninstall -y nvidia-cutlass-dsl nvidia-cutlass-dsl-libs-base nvidia-cutlass-dsl-libs-cu13 || true && \
    rm -rf /usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl* && \
    pip install --no-cache-dir --no-deps \
        'nvidia-cutlass-dsl==4.4.1' \
        'nvidia-cutlass-dsl-libs-base==4.4.1' \
        'nvidia-cutlass-dsl-libs-cu13==4.4.1' && \
    python -c "from cutlass.cute.arch import ProxyKind, SharedSpace" && \
    git clone -b core_v0.13.1 https://github.com/NVIDIA/Megatron-LM.git megatron-lm && \
    pip install --no-deps -e ./megatron-lm && \
    pip install --no-cache-dir torchx gin-config torchmetrics==1.0.3 typing-extensions iopath pyvers expiring_dict \
        cloudpickle triton==3.6.0 \
        'nvidia-cutlass-dsl==4.4.1' 'nvidia-cutlass-dsl-libs-base==4.4.1' 'nvidia-cutlass-dsl-libs-cu13==4.4.1' \
        'quack-kernels>=0.3.3' 'apache-tvm-ffi>=0.1.6' torch-c-dlpack-ext pre-commit

# -- Layer 3: FBGEMM (long build, own layer for caching) ---
RUN pip install --no-cache-dir setuptools-git-versioning scikit-build && \
    git clone --recursive -b v1.5.0 https://github.com/pytorch/FBGEMM.git fbgemm && \
    cd fbgemm/fbgemm_gpu && \
    python setup.py install --build-target=default --build-variant=cuda -DTORCH_CUDA_ARCH_LIST="7.5 8.0 9.0"

# -- Layer 4: TorchRec ---
RUN pip install --no-deps tensordict orjson && \
    git clone --recursive -b release/V1.5.0 https://github.com/pytorch/torchrec.git torchrec && \
    cd torchrec && \
    pip install --no-deps .

# -- Layer 5: FlashAttention --
ARG FLASH_ATTN_ARBITRARY_MASK_COMMIT=b56db721d65afadd8cfe154cd12e83693bf0aba3
RUN git clone https://github.com/jiayus-nvidia/flash-attention.git flash-attention && \
    cd flash-attention && \
    git checkout ${FLASH_ATTN_ARBITRARY_MASK_COMMIT} && \
    pip install --no-deps --no-build-isolation -e ./flash_attn/cute && \
    python -c "from flash_attn.cute.interface import flash_attn_func"

# -- Layer 6, 7: FlexKV and NVE : both inference dependency --
COPY third_party/FlexKV /workspace/deps/FlexKV
RUN cd FlexKV && mkdir -p build && cd build && \
    TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0" cmake .. && \
    TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0" cmake --build . -j8 && \
    cd .. && mkdir -p flexkv/lib && cp -P build/lib/*.so* flexkv/lib/ && \
    MAX_JOBS=8 FLEXKV_DEBUG=0 TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0" python3 setup.py install && \
    cd /tmp && python3 -c "from flexkv.kvmanager import KVManager; print('flexkv import ok')" && \
    cd /workspace/deps && \
    git clone --recursive https://github.com/NVIDIA/nv-embedding-cache.git nve && \
    cd nve && git submodule update --init --recursive && \
    cd third_party/ && git clone https://github.com/NVIDIA/NVTX.git NVTX && cd .. && \
    TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0" PYNVE_WITH_TORCH_BINDINGS=1 PYNVE_DISABLE_AVX512=1 \
    CPLUS_INCLUDE_PATH=$(realpath ./third_party/NVTX/c/include/):${CPLUS_INCLUDE_PATH} pip install --no-deps .

# Install fbgemm_gpu_hstu (package: fbgemm_gpu_hstu, import: hstu) from submodule
COPY third_party/FBGEMM /workspace/deps/fbgemm_hstu
RUN cd /workspace/deps/fbgemm_hstu/fbgemm_gpu/experimental/hstu && \
    HSTU_DISABLE_86OR89=FALSE \
    HSTU_DISABLE_ARBITRARY=TRUE \
    HSTU_DISABLE_LOCAL=TRUE \
    HSTU_DISABLE_RAB=TRUE \
    HSTU_DISABLE_DRAB=TRUE \
    HSTU_DISABLE_FP16=TRUE \
    HSTU_ARCH_LIST="8.0 9.0" \
    pip install --no-build-isolation . && \
    rm -rf /workspace/deps/fbgemm_hstu

FROM ${DEVEL_IMAGE} AS build
ARG TRITONSERVER_BUILD

RUN rm -rf /workspace/recsys-examples
WORKDIR /workspace/recsys-examples
COPY . .

RUN cd /workspace/recsys-examples/corelib/dynamicemb && \
    python setup.py install && \
    cd /workspace/deps && rm -rf nvcomp && \
    wget https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/linux-x86_64/nvcomp-linux-x86_64-5.1.0.21_cuda12-archive.tar.xz && \
    tar -xf nvcomp-linux-x86_64-5.1.0.21_cuda12-archive.tar.xz && \
    mv nvcomp-linux-x86_64-5.1.0.21_cuda12-archive nvcomp && \
    rm nvcomp-linux-x86_64-5.1.0.21_cuda12-archive.tar.xz && \
    cd /workspace/recsys-examples/examples/commons && \
    TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0" python3 setup.py install && \
    cd /workspace/recsys-examples/corelib/recsys_kvcache_manager && \
    TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0" pip3 install --no-build-isolation . && \
    if [ "${TRITONSERVER_BUILD}" != "1" ]; then \
      rm -f /usr/lib/$(uname -m)-linux-gnu/libcuda.so.1 && \
      ln -s /usr/local/cuda-13.1/compat/lib.real/libcuda.so.1  /usr/lib/$(uname -m)-linux-gnu/libcuda.so.1 && \
      cd /workspace/recsys-examples/corelib/dynamicemb && \
      mkdir -p torch_binding_build && cd torch_binding_build && \
      cmake .. && make -j; \
    fi
