#!/usr/bin/env bash
# HybridStorage: build storage directly; stepwise find+insert; validate scores via export.
# Run from corelib/dynamicemb:
#   ./test/unit_tests/test_hybrid_storage_export.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DYNAMICEMB_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${DYNAMICEMB_ROOT}"

export PYTHONPATH="${DYNAMICEMB_ROOT}:${PYTHONPATH:-}"

NPROC="${NPROC_PER_NODE:-1}"

torchrun \
  --nnodes 1 \
  --nproc_per_node="${NPROC}" \
  -m pytest "${SCRIPT_DIR}/test_hybrid_storage_export.py" -v --tb=short "$@"
