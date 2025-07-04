ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:24.11-py3
ARG DEVEL_IMAGE=devel

FROM ${BASE_IMAGE} AS devel
WORKDIR /workspace/deps

RUN git clone -b core_v0.12.1 https://github.com/NVIDIA/Megatron-LM.git megatron-lm && \
    pip install -e ./megatron-lm

RUN pip install torchx gin-config torchmetrics==1.0.3 typing-extensions iopath

RUN pip install --no-cache setuptools==69.5.1 setuptools-git-versioning scikit-build && \
  git clone --recursive -b main https://github.com/pytorch/FBGEMM.git fbgemm && \
  cd fbgemm/fbgemm_gpu && \
  git checkout 642ccb980d05aa1be00ccd131c5991b0914e2e64 && \
  python setup.py bdist_wheel --package_variant=cuda -DTORCH_CUDA_ARCH_LIST="8.0 9.0" && \
  python setup.py install --package_variant=cuda -DTORCH_CUDA_ARCH_LIST="8.0 9.0"

RUN pip install --no-deps tensordict orjson && \
  git clone --recursive -b main https://github.com/pytorch/torchrec.git torchrec && \
  cd torchrec && git checkout 6aaf1fa72e884642f39c49ef232162fa3772055e && \
  pip install --no-deps .

RUN test -f /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 || ln -s /usr/local/cuda-12.6/targets/x86_64-linux/lib/stubs/libnvidia-ml.so  /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1

# for dev
RUN apt update -y --fix-missing && \
    apt install -y gdb && \
    apt autoremove -y && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache pre-commit

FROM ${DEVEL_IMAGE} AS build

WORKDIR /workspace/recsys-examples
COPY . .

RUN cd /workspace/recsys-examples/corelib/dynamicemb && \
    python setup.py install
    
RUN cd /workspace/recsys-examples/corelib/hstu && \
    HSTU_DISABLE_LOCAL=TRUE HSTU_DISABLE_RAB=TRUE HSTU_DISABLE_DELTA_Q=FALSE HSTU_DISABLE_DRAB=TRUE pip install . && \
    cd hopper && \
    HSTU_DISABLE_SM8x=TRUE HSTU_DISABLE_LOCAL=TRUE HSTU_DISABLE_RAB=TRUE HSTU_DISABLE_DELTA_Q=TRUE HSTU_DISABLE_DRAB=TRUE pip install .

RUN cd /workspace/recsys-examples/examples/hstu && \
    python setup.py install
