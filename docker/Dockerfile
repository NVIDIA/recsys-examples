ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.06-py3
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
      pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129; \
    fi

RUN ARCH=$([ "${TARGETPLATFORM}" = "linux/arm64" ] && echo "aarch64" || echo "x86_64") && \
    rm -rf /usr/lib/${ARCH}-linux-gnu/libnvidia-ml.so.1 && \
    if [ ${ARCH} = "aarch64" ]; then \
      ln -s /usr/local/cuda-12.9/targets/sbsa-linux/lib/stubs/libnvidia-ml.so  /usr/lib/${ARCH}-linux-gnu/libnvidia-ml.so.1; \
    else \
      if [[ "${INFERENCEBUILD}" == "1" && "${TRITONSERVER_BUILD}" != "1" ]]; then \
        ln -s /usr/local/cuda-12.8/targets/${ARCH}-linux/lib/stubs/libnvidia-ml.so  /usr/lib/${ARCH}-linux-gnu/libnvidia-ml.so.1; \
      else \
        ln -s /usr/local/cuda-12.9/targets/${ARCH}-linux/lib/stubs/libnvidia-ml.so  /usr/lib/${ARCH}-linux-gnu/libnvidia-ml.so.1; \
      fi \
    fi

RUN if [ "${INFERENCEBUILD}" != "1" ]; then \
      git clone -b core_v0.12.1 https://github.com/NVIDIA/Megatron-LM.git megatron-lm && \
      pip install --no-deps -e ./megatron-lm; \
    fi

RUN pip install torchx gin-config torchmetrics==1.0.3 typing-extensions iopath pyvers

RUN pip install --no-cache-dir setuptools-git-versioning scikit-build && \
  git clone --recursive -b v1.2.0 https://github.com/pytorch/FBGEMM.git fbgemm && \
  cd fbgemm/fbgemm_gpu && \
  python setup.py install --package_variant=cuda -DTORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 9.0"

RUN pip install --no-deps tensordict orjson && \
  git clone --recursive -b v1.2.0 https://github.com/pytorch/torchrec.git torchrec && \
  cd torchrec && \
  pip install --no-deps .

RUN pip install nvidia-cutlass-dsl==4.3.0

# for dev
RUN apt update -y --fix-missing && \
    apt install -y gdb && \
    apt autoremove -y && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache pre-commit

RUN cd /workspace/recsys-examples/corelib/hstu && \
    HSTU_DISABLE_86OR89=FALSE HSTU_DISABLE_ARBITRARY=TRUE HSTU_DISABLE_LOCAL=TRUE HSTU_DISABLE_RAB=TRUE HSTU_DISABLE_DRAB=TRUE pip install . && \
    cd hopper && \
    HSTU_DISABLE_ARBITRARY=TRUE HSTU_DISABLE_SM8x=TRUE HSTU_DISABLE_LOCAL=TRUE HSTU_DISABLE_RAB=TRUE HSTU_DISABLE_DELTA_Q=FALSE HSTU_DISABLE_DRAB=TRUE pip install .
    
FROM ${DEVEL_IMAGE} AS build
ARG TRITONSERVER_BUILD

WORKDIR /workspace/recsys-examples
# to remove the pre-installed hstu packages of the previous build
RUN rm -rf /workspace/recsys-examples/*
# COPY the new cloned repo into workdir
COPY . .

RUN cd /workspace/recsys-examples/corelib/dynamicemb && \
    python setup.py install
    
RUN if [ "${TRITONSERVER_BUILD}" = "1" ]; then \
      pip3 uninstall -y hstu_attn hstu_hopper; \
    fi

RUN cd /workspace/recsys-examples/examples/commons && \
    TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 9.0" python3 setup.py install
