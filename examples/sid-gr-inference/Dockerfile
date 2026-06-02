# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}

ARG INSTALL_KERNEL_DEPS=0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GR_HTTP_HOST=0.0.0.0 \
    GR_HTTP_PORT=8000

WORKDIR /app/sid-gr-inference

COPY pyproject.toml README.md ./
COPY src ./src
COPY tools ./tools
COPY scripts ./scripts
COPY docs ./docs
COPY third_party ./third_party

RUN python -m pip install --no-cache-dir --upgrade pip \
    && if [ "${INSTALL_KERNEL_DEPS}" = "1" ]; then \
        python -m pip install --no-cache-dir -e ".[kernels]"; \
    else \
        python -m pip install --no-cache-dir -e .; \
    fi \
    && chmod +x scripts/serve_qwen3_gr_http.sh

EXPOSE 8000

ENTRYPOINT ["scripts/serve_qwen3_gr_http.sh"]
