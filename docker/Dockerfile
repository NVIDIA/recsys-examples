ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:24.11-py3
ARG DEVEL_IMAGE=devel

FROM ${BASE_IMAGE} AS devel
ARG TARGETPLATFORM
ARG INFERENCEBUILD

WORKDIR /workspace/deps

RUN ARCH=$([ "${TARGETPLATFORM}" = "linux/arm64" ] && echo "aarch64" || echo "x86_64") && \
    rm -rf /usr/lib/${ARCH}-linux-gnu/libnvidia-ml.so.1 && \
    if [ "${INFERENCEBUILD}" == "1" ]; then \
      ln -s /usr/local/cuda-12.8/targets/x86_64-linux/lib/stubs/libnvidia-ml.so /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1;  \
    else \
      if [ ${ARCH} = "aarch64" ]; then \
          ln -s /usr/local/cuda-12.6/targets/sbsa-linux/lib/stubs/libnvidia-ml.so  /usr/lib/${ARCH}-linux-gnu/libnvidia-ml.so.1; \
      else \
          ln -s /usr/local/cuda-12.6/targets/${ARCH}-linux/lib/stubs/libnvidia-ml.so  /usr/lib/${ARCH}-linux-gnu/libnvidia-ml.so.1; \
      fi \
    fi

RUN if [ "${INFERENCEBUILD}" != "1" ]; then \
      git clone -b core_v0.12.1 https://github.com/NVIDIA/Megatron-LM.git megatron-lm && \
      pip install -e ./megatron-lm; \
    fi

RUN pip install torchx gin-config torchmetrics==1.0.3 typing-extensions iopath pyvers

RUN pip install --no-cache-dir setuptools==69.5.1 setuptools-git-versioning scikit-build && \
  git clone --recursive -b v1.2.0 https://github.com/pytorch/FBGEMM.git fbgemm && \
  cd fbgemm/fbgemm_gpu && \
  python setup.py install --package_variant=cuda -DTORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 9.0"

RUN pip install --no-deps tensordict orjson && \
  git clone --recursive -b v1.2.0 https://github.com/pytorch/torchrec.git torchrec && \
  cd torchrec && \
  pip install --no-deps .

# for dev
RUN apt update -y --fix-missing && \
    apt install -y gdb && \
    apt autoremove -y && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache pre-commit

FROM ${DEVEL_IMAGE} AS build

WORKDIR /workspace/recsys-examples
# to remove the pre-installed hstu packages of the previous build
RUN rm -rf /workspace/recsys-examples/*
# COPY the new cloned repo into workdir
COPY . .

RUN cd /workspace/recsys-examples/corelib/dynamicemb && \
    python setup.py install
    
RUN cd /workspace/recsys-examples/corelib/hstu && \
    HSTU_DISABLE_LOCAL=TRUE HSTU_DISABLE_RAB=TRUE HSTU_DISABLE_DRAB=TRUE pip install . && \
    cd hopper && \
    HSTU_DISABLE_SM8x=TRUE HSTU_DISABLE_LOCAL=TRUE HSTU_DISABLE_RAB=TRUE HSTU_DISABLE_DELTA_Q=FALSE HSTU_DISABLE_DRAB=TRUE pip install .

RUN cd /workspace/recsys-examples/examples/hstu && \
    python setup.py install