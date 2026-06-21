#!/usr/bin/env bash
# Fake-mode unit tests: DYNAMICEMB_FAKE_MODE dispatch, host-side attribute
# parity vs real, CPU-only forward / backward / dump / load / scores, plus a
# single-rank gloo DMP sequence-mode wiring smoke. Runs without a GPU device;
# the attribute-parity test auto-skips when CUDA is unavailable.
#
# Run from corelib/dynamicemb:
#   ./test/unit_tests/test_fake_batched_dynamicemb_tables.sh
#
# Override the process count if you want torchrun to spawn more ranks
# (the DMP test itself is single-rank gloo regardless):
#   NPROC_PER_NODE=1 ./test/unit_tests/test_fake_batched_dynamicemb_tables.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DYNAMICEMB_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${DYNAMICEMB_ROOT}"

export PYTHONPATH="${DYNAMICEMB_ROOT}:${PYTHONPATH:-}"
export DYNAMICEMB_FAKE_MODE="${DYNAMICEMB_FAKE_MODE:-1}"

NPROC="${NPROC_PER_NODE:-1}"

torchrun \
  --nnodes 1 \
  --nproc_per_node="${NPROC}" \
  -m pytest "${SCRIPT_DIR}/test_fake_batched_dynamicemb_tables.py" -v --tb=short "$@"
