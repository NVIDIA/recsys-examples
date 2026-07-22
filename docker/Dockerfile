ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.05-py3
ARG BASE_FBGEMM_IMAGE=base_fbgemm
ARG BASE_TRITON_IMAGE=base_triton
ARG DEVEL_IMAGE=devel

# Triton r${TRITON_SERVER_VERSION} is built against local PyTorch libraries.
# Effective upstream version priority:
#   1. UPSTREAM_PYTORCH_CONTAINER_VERSION when explicitly supplied
#   2. Version parsed from nvcr.io/nvidia/pytorch:xx.xx-py3 BASE_IMAGE
#   3. DEFAULT_UPSTREAM_PYTORCH_CONTAINER_VERSION
ARG TRITON_SERVER_VERSION=26.06
ARG TRITON_CLIENT_VERSION=2.69.0
ARG UPSTREAM_PYTORCH_CONTAINER_VERSION=
ARG DEFAULT_UPSTREAM_PYTORCH_CONTAINER_VERSION=26.05
ARG TRITON_BUILD_JOBS=8

# ============================================================================
# Stage 1: base_fbgemm — FBGEMM Path A + TorchRec (version-coupled with FBGEMM).
# Override BASE_FBGEMM_IMAGE to consume a pre-built image and skip the build.
# ============================================================================
FROM ${BASE_IMAGE} AS base_fbgemm

WORKDIR /workspace/deps

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
# Stage 2: base_triton — base_fbgemm + Triton core, PyTorch backend, and
# Python backend. Override BASE_TRITON_IMAGE to consume a pre-built image.
# ============================================================================
FROM ${BASE_FBGEMM_IMAGE} AS base_triton

ARG BASE_IMAGE
ARG TRITON_SERVER_VERSION
ARG TRITON_CLIENT_VERSION
ARG UPSTREAM_PYTORCH_CONTAINER_VERSION
ARG DEFAULT_UPSTREAM_PYTORCH_CONTAINER_VERSION
ARG TRITON_BUILD_JOBS

USER root
WORKDIR /workspace/deps

# Dependencies required by the no-container Triton build. CMake is installed
# separately and pinned below because 3.31.6 is too old and 4.x breaks bundled
# dependencies encountered in this build.
RUN apt-get update -y --fix-missing && \
    apt-get install -y --no-install-recommends \
      autoconf automake build-essential ca-certificates git gperf \
      libarchive-dev libb64-dev libcurl4-openssl-dev libgoogle-perftools-dev \
      libnuma-dev libre2-dev libssl-dev libtool pkg-config \
      python3-dev python3-pip python3-setuptools python3-wheel rapidjson-dev \
      unzip wget zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

# Triton invokes /usr/bin/python3 to build its wheels. Install into that exact
# interpreter and bypass the NGC pip constraint only for these build tools.
RUN env -u PIP_CONSTRAINT /usr/bin/python3 -m pip install \
      --break-system-packages --no-cache-dir --upgrade \
      auditwheel build "cmake==3.31.10" "patchelf==0.17.2" pybind11 \
      setuptools virtualenv wheel && \
    cmake --version && \
    /usr/bin/python3 -c "import build, wheel, setuptools"

# Triton r26.06 requires Boost >= 1.78. Use the same Boost 1.80 headers and
# checksum encoded by Triton's build.py instead of relying on distro version.
RUN wget -O /tmp/boost.tar.gz \
      https://archives.boost.io/release/1.80.0/source/boost_1_80_0.tar.gz && \
    echo "4b2136f98bdd1f5857f1c3dea9ac2018effe65286cf251534b6ae20cc45e1847  /tmp/boost.tar.gz" \
      | sha256sum --check && \
    tar xzf /tmp/boost.tar.gz -C /tmp && \
    rm -rf /usr/include/boost && \
    mv /tmp/boost_1_80_0/boost /usr/include/boost && \
    rm -rf /tmp/boost.tar.gz /tmp/boost_1_80_0 && \
    grep BOOST_LIB_VERSION /usr/include/boost/version.hpp

# Build Triton directly against the PyTorch installation in this image.
# Torch-TensorRT, TorchVision custom operators, and NVSHMEM are deliberately
# disabled so their additional libraries and headers do not have to be staged.
# Temporary customized PyTorch backend build:
# ceeecb7 adds model-initialization-hook support for the PyTorch PT2 (AOTI)
# backend. This feature is not included in Triton releases through 26.07.
# Once it is included in release 26.08 or later, remove this override and use
# only the standard, release-aligned PyTorch backend produced by build.py.
# Until then, build ceeecb7 against the same backend/core/common release as the
# Triton server and install it over the standard /opt/tritonserver backend.
RUN set -eux; \
    parsed_upstream=; \
    case "${BASE_IMAGE}" in \
      nvcr.io/nvidia/pytorch:[0-9][0-9].[0-9][0-9]-py3) \
        parsed_upstream="${BASE_IMAGE#nvcr.io/nvidia/pytorch:}"; \
        parsed_upstream="${parsed_upstream%-py3}";; \
    esac; \
    effective_upstream="${UPSTREAM_PYTORCH_CONTAINER_VERSION:-${parsed_upstream}}"; \
    effective_upstream="${effective_upstream:-${DEFAULT_UPSTREAM_PYTORCH_CONTAINER_VERSION}}"; \
    pytorch_site=/usr/local/lib/python3.12/dist-packages/torch; \
    pytorch_includes="${pytorch_site}/include;${pytorch_site}/include/torch/csrc/api/include"; \
    echo "Triton container version: ${TRITON_SERVER_VERSION}"; \
    echo "Effective upstream PyTorch container version: ${effective_upstream}"; \
    git clone --depth 1 --branch "r${TRITON_SERVER_VERSION}" \
      https://github.com/triton-inference-server/server.git /workspace/deps/triton-server-src; \
    mkdir -p /opt/triton-pytorch-libs; \
    cp -a "${pytorch_site}/lib/." /opt/triton-pytorch-libs/; \
    cp -aL /usr/local/lib/libjpeg.so.62 /opt/triton-pytorch-libs/; \
    test -e /opt/triton-pytorch-libs/libtorch.so \
      && test -e /opt/triton-pytorch-libs/libtorch_cuda.so \
      && test -e /opt/triton-pytorch-libs/libjpeg.so.62; \
    cd /workspace/deps/triton-server-src; \
    python3 build.py -v \
      --no-container-build --build-dir=/workspace/triton-build --install-dir=/opt/tritonserver \
      --container-version="${TRITON_SERVER_VERSION}" \
      --upstream-container-version="${effective_upstream}" \
      --build-parallel="${TRITON_BUILD_JOBS}" \
      --enable-gpu --enable-logging --enable-stats --enable-metrics \
      --endpoint=http --endpoint=grpc \
      --backend="pytorch:r${TRITON_SERVER_VERSION}" --backend="python:r${TRITON_SERVER_VERSION}" \
      --extra-backend-cmake-arg="pytorch:TRITON_PYTORCH_INCLUDE_PATHS=${pytorch_includes}" \
      --extra-backend-cmake-arg=pytorch:TRITON_PYTORCH_LIB_PATHS=/opt/triton-pytorch-libs \
      --override-backend-cmake-arg=pytorch:TRITON_PYTORCH_ENABLE_TORCHTRT=OFF \
      --extra-backend-cmake-arg=pytorch:TRITON_PYTORCH_ENABLE_TORCHVISION=OFF \
      --extra-backend-cmake-arg=pytorch:TRITON_PYTORCH_NVSHMEM=OFF; \
    standard_backend=/opt/tritonserver/backends/pytorch/libtriton_pytorch.so; \
    test -f "${standard_backend}"; \
    standard_backend_sha256="$(sha256sum "${standard_backend}" | awk '{print $1}')"; \
    cd /workspace/deps; rm -rf pytorch_backend; \
    git clone --depth 1 \
      https://github.com/triton-inference-server/pytorch_backend.git pytorch_backend; \
    cd pytorch_backend; \
    git fetch --depth 1 origin r26.07; \
    git checkout --detach ceeecb7; \
    pinned_backend_commit="$(git rev-parse HEAD)"; \
    test "${pinned_backend_commit}" = ceeecb748caa785b359095a74c05d55dace2591a; \
    mkdir -p build; cd build; \
    cmake \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX:PATH="$PWD/install" \
      -DTRITON_PYTORCH_INCLUDE_PATHS="${pytorch_includes};/opt/pytorch/vision/torchvision/csrc" \
      -DTRITON_PYTORCH_LIB_PATHS=/opt/triton-pytorch-libs \
      -DTRITON_PYTORCH_ENABLE_TORCHTRT=OFF -DTRITON_PYTORCH_ENABLE_TORCHVISION=OFF \
      -DTRITON_PYTORCH_NVSHMEM=OFF \
      -DTRITON_BACKEND_REPO_TAG="r${TRITON_SERVER_VERSION}" \
      -DTRITON_CORE_REPO_TAG="r${TRITON_SERVER_VERSION}" \
      -DTRITON_COMMON_REPO_TAG="r${TRITON_SERVER_VERSION}" \
      ..; \
    cmake --build . --parallel "${TRITON_BUILD_JOBS}" --target install; \
    pinned_backend="$PWD/install/backends/pytorch/libtriton_pytorch.so"; \
    test -f "${pinned_backend}"; install -m 0755 "${pinned_backend}" "${standard_backend}"; \
    pinned_backend_sha256="$(sha256sum "${standard_backend}" | awk '{print $1}')"; \
    test "${pinned_backend_sha256}" = "$(sha256sum "${pinned_backend}" | awk '{print $1}')"; \
    provenance=/opt/tritonserver/backends/pytorch/pinned-backend-build.txt; \
    printf '%s\n' \
      "commit=${pinned_backend_commit}" \
      "repository_tags=r${TRITON_SERVER_VERSION}" \
      "replaced_sha256=${standard_backend_sha256}" \
      "installed_sha256=${pinned_backend_sha256}" \
      > "${provenance}"; \
    printf '%s\n' "${effective_upstream}" > /opt/tritonserver/upstream-pytorch-container-version; \
    rm -rf /workspace/triton-build /workspace/deps/triton-server-src /opt/triton-pytorch-libs

ENV TRITON_SERVER_VERSION=${TRITON_SERVER_VERSION}
ENV NVIDIA_TRITON_SERVER_VERSION=${TRITON_SERVER_VERSION}
ENV PATH=/opt/tritonserver/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/tritonserver/lib:/opt/tritonserver/backends/pytorch:${LD_LIBRARY_PATH}

# Triton release 26.06 corresponds to server/client version 2.69.0. Install
# only the HTTP client required by the same-container HSTU test workflow.
RUN env -u PIP_CONSTRAINT /usr/bin/python3 -m pip install \
      --break-system-packages --no-cache-dir "tritonclient[http]==${TRITON_CLIENT_VERSION}" && \
    /usr/bin/python3 -c "import tritonclient.http; print(tritonclient.http.__file__)"

# Keep runtime verification separate so an assertion failure does not invalidate
# the expensive Triton compilation layer above. Triton --help returns 1 in r26.06.
RUN set -eux; \
    provenance=/opt/tritonserver/backends/pytorch/pinned-backend-build.txt; \
    test -x /opt/tritonserver/bin/tritonserver \
      && test -f /opt/tritonserver/backends/pytorch/libtriton_pytorch.so; \
    grep -q '^commit=ceeecb7' "${provenance}"; \
    grep -q "^repository_tags=r${TRITON_SERVER_VERSION}$" "${provenance}"; \
    test -f /opt/tritonserver/backends/python/libtriton_python.so \
      && test -x /opt/tritonserver/backends/python/triton_python_backend_stub; \
    export LD_LIBRARY_PATH="/opt/tritonserver/lib:/opt/tritonserver/backends/pytorch:${LD_LIBRARY_PATH:-}"; \
    help_output=/tmp/tritonserver-help.txt; help_status=0; \
    /opt/tritonserver/bin/tritonserver --help > "${help_output}" 2>&1 || help_status=$?; \
    case "${help_status}" in 0|1) ;; *) exit "${help_status}";; esac; \
    grep -q '^Usage: tritonserver' "${help_output}"; rm -f "${help_output}"; \
    for target in \
      /opt/tritonserver/bin/tritonserver \
      /opt/tritonserver/backends/pytorch/libtriton_pytorch.so \
      /opt/tritonserver/backends/python/libtriton_python.so \
      /opt/tritonserver/backends/python/triton_python_backend_stub; do \
        if ldd "${target}" | grep -q 'not found'; then \
          ldd "${target}" | grep 'not found'; \
          exit 1; \
        fi; \
    done

# ============================================================================
# Stage 3: devel — base_triton + system setup + Megatron + pip
# + flash-attention + FlexKV/NVE + HSTU Path B.
# ============================================================================
FROM ${BASE_TRITON_IMAGE} AS devel
ARG TARGETPLATFORM

WORKDIR /workspace/deps

# -- Layer 1: system setup + arch symlinks.
RUN apt-get update -y --fix-missing && \
    apt-get install -y \
      gdb \
      liburing-dev \
      libxxhash-dev \
      libzmq3-dev && \
    ARCH=$([ "${TARGETPLATFORM}" = "linux/arm64" ] && echo "aarch64" || echo "x86_64") && \
    rm -rf /usr/lib/${ARCH}-linux-gnu/libnvidia-ml.so.1 && \
    if [ "${ARCH}" = "aarch64" ]; then \
      ln -s /usr/local/cuda-13/targets/sbsa-linux/lib/stubs/libnvidia-ml.so /usr/lib/${ARCH}-linux-gnu/libnvidia-ml.so.1; \
    else \
      ln -s /usr/local/cuda-13/targets/${ARCH}-linux/lib/stubs/libnvidia-ml.so /usr/lib/${ARCH}-linux-gnu/libnvidia-ml.so.1; \
    fi && \
    if [ "${TARGETPLATFORM}" = "linux/arm64" ]; then \
      CUDA_TARGET_ARCH=sbsa; \
    elif [ "${TARGETPLATFORM}" = "linux/amd64" ]; then \
      CUDA_TARGET_ARCH=x86_64; \
    else \
      CUDA_TARGET_ARCH=$(uname -m); \
    fi && \
    ln -sf /usr/local/cuda-13/targets/${CUDA_TARGET_ARCH}-linux/include/cccl/cuda /usr/local/cuda/include/cuda && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

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
# Stage 4: build — devel + recsys-examples in-tree builds (dynamicemb, commons,
# kvcache_manager) and runtime libcuda link.
# ============================================================================
FROM ${DEVEL_IMAGE} AS build

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
    rm -f /usr/lib/$(uname -m)-linux-gnu/libcuda.so.1 && \
    test -e /usr/local/cuda-13/compat/lib.real/libcuda.so.1 && \
    ln -s /usr/local/cuda-13/compat/lib.real/libcuda.so.1 /usr/lib/$(uname -m)-linux-gnu/libcuda.so.1 && \
    cd /workspace/recsys-examples/corelib/dynamicemb && \
    mkdir -p torch_binding_build && cd torch_binding_build && \
    cmake .. && make -j

# ============================================================================
# AOTI additions: C++ replay executable, NVE init hook, optional checkpoint
# export, and Triton runtime library staging.
# ============================================================================
RUN cd /workspace/recsys-examples/corelib/recsys_kvcache_manager && \
    mkdir -p build && cd build && \
    cmake .. && make -j

RUN export CMAKE_PREFIX_PATH="$(python3 -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "share", "cmake"))')" && \
    cmake -S "./examples/hstu/inference_aoti/cpp_inference" -B "./examples/hstu/inference_aoti/cpp_inference/build" && \
    cmake --build "./examples/hstu/inference_aoti/cpp_inference/build" -j 8 && \
    cmake -S "./examples/hstu/inference_aoti/nve_init_hook" -B "./examples/hstu/inference_aoti/nve_init_hook/build" && \
    cmake --build "./examples/hstu/inference_aoti/nve_init_hook/build" -j 8

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

