ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.02-py3
ARG DEVEL_IMAGE=devel

FROM ${BASE_IMAGE} AS devel
ARG TARGETPLATFORM
ARG INFERENCEBUILD
ARG TRITONSERVER_BUILD

WORKDIR /workspace/deps

# -- Layer 1: system setup, arch symlinks, tritonserver deps ---
RUN if [ "${TRITONSERVER_BUILD}" = "1" ]; then \
      ln /bin/python3 /bin/python && \
      apt-get update -y --fix-missing && apt-get install -y cmake patchelf && \
      pip3 install pandas rich cloudpickle psutil && \
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
RUN git clone -b core_v0.12.1 https://github.com/NVIDIA/Megatron-LM.git megatron-lm && \
    pip install --no-deps -e ./megatron-lm && \
    pip install torchx gin-config torchmetrics==1.0.3 typing-extensions iopath pyvers \
    cloudpickle triton==3.6.0 nvidia-cutlass-dsl==4.3.0 --no-cache pre-commit

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

RUN git clone -b arbitrary_mask https://github.com/jiayus-nvidia/flash-attention.git flash-attention && \
    cd flash-attention && \
    pip install --no-deps -e .

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
    if [ "${TRITONSERVER_BUILD}" != "1" ]; then \
      rm -f /usr/lib/$(uname -m)-linux-gnu/libcuda.so.1 && \
      ln -s /usr/local/cuda-13.1/compat/lib.real/libcuda.so.1  /usr/lib/$(uname -m)-linux-gnu/libcuda.so.1 && \
      cd /workspace/recsys-examples/corelib/dynamicemb && \
      mkdir -p torch_binding_build && cd torch_binding_build && \
      cmake .. && make -j; \
    fi
