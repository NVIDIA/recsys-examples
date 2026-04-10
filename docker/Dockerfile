ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.02-py3
ARG DEVEL_IMAGE=devel

FROM ${BASE_IMAGE} AS devel
ARG TARGETPLATFORM
ARG INFERENCEBUILD
ARG TRITONSERVER_BUILD

WORKDIR /workspace/deps

RUN if [ "${TRITONSERVER_BUILD}" = "1" ]; then \
      ln /bin/python3 /bin/python && \
      apt-get update -y --fix-missing && apt-get install -y cmake && apt-get install -y patchelf; \
    fi

RUN if [ "${TRITONSERVER_BUILD}" = "1" ]; then \
      pip3 install pandas rich cloudpickle psutil && \
      pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130; \
    fi

RUN ARCH=$([ "${TARGETPLATFORM}" = "linux/arm64" ] && echo "aarch64" || echo "x86_64") && \
    rm -rf /usr/lib/${ARCH}-linux-gnu/libnvidia-ml.so.1 && \
    if [ ${ARCH} = "aarch64" ]; then \
      ln -s /usr/local/cuda-13/targets/sbsa-linux/lib/stubs/libnvidia-ml.so  /usr/lib/${ARCH}-linux-gnu/libnvidia-ml.so.1; \
    else \
      ln -s /usr/local/cuda-13/targets/${ARCH}-linux/lib/stubs/libnvidia-ml.so  /usr/lib/${ARCH}-linux-gnu/libnvidia-ml.so.1; \
    fi

RUN if [ "${INFERENCEBUILD}" != "1" ]; then \
      git clone -b core_v0.12.1 https://github.com/NVIDIA/Megatron-LM.git megatron-lm && \
      pip install --no-deps -e ./megatron-lm; \
    fi

RUN pip install torchx gin-config torchmetrics==1.0.3 typing-extensions iopath pyvers
RUN pip install cloudpickle
RUN pip install triton==3.6.0
RUN pip install nvidia-cutlass-dsl==4.3.0

RUN pip install --no-cache-dir setuptools-git-versioning scikit-build && \
  git clone --recursive -b v1.5.0 https://github.com/pytorch/FBGEMM.git fbgemm && \
  cd fbgemm/fbgemm_gpu && \
  python setup.py install --build-target=default --build-variant=cuda -DTORCH_CUDA_ARCH_LIST="7.5 8.0 9.0"

RUN pip install --no-deps tensordict orjson && \
  git clone --recursive -b release/V1.5.0 https://github.com/pytorch/torchrec.git torchrec && \
  cd torchrec && \
  pip install --no-deps .


# for dev
RUN apt update -y --fix-missing && \
    apt install -y gdb && \
    apt autoremove -y && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache pre-commit

RUN if [ "${TARGETPLATFORM}" = "linux/arm64" ]; then \
      CUDA_TARGET_ARCH=sbsa; \
    elif [ "${TARGETPLATFORM}" = "linux/amd64" ]; then \
      CUDA_TARGET_ARCH=x86_64; \
    else \
      CUDA_TARGET_ARCH=$(uname -m); \
    fi && \
    ln -sf /usr/local/cuda-13/targets/${CUDA_TARGET_ARCH}-linux/include/cccl/cuda \
    /usr/local/cuda/include/cuda

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
    python setup.py install
    
RUN cd /workspace/deps && rm -rf nvcomp && \
    wget https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/linux-x86_64/nvcomp-linux-x86_64-5.1.0.21_cuda12-archive.tar.xz && \
    tar -xf nvcomp-linux-x86_64-5.1.0.21_cuda12-archive.tar.xz && \
    mv nvcomp-linux-x86_64-5.1.0.21_cuda12-archive nvcomp && \
    rm  nvcomp-linux-x86_64-5.1.0.21_cuda12-archive.tar.xz 
  
RUN cd /workspace/recsys-examples/examples/commons && \
    TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0" python3 setup.py install

RUN if [ "${INFERENCEBUILD}" == "1" ]; then \
      rm -f /usr/lib/x86_64-linux-gnu/libcuda.so.1 && ln -s /usr/local/cuda-13.1/compat/lib.real/libcuda.so.1  /usr/lib/x86_64-linux-gnu/libcuda.so.1 && \
      cd /workspace/recsys-examples/corelib/dynamicemb && \
      mkdir -p torch_binding_build && cd torch_binding_build && \
      cmake .. && make -j && \
      cd /workspace/recsys-examples/examples/hstu/inference/cpp_inference && \
      CMAKE_PREFIX_PATH=/usr/local/lib/python3.12/dist-packages/torch/share/cmake cmake -S . -B build && \
      cmake --build build --config Release -j ; \
    fi
