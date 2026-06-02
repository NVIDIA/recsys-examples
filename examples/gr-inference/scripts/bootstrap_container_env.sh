#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common_paths.sh
source "${SCRIPT_DIR}/common_paths.sh"
REPO_ROOT="${GR_INFERENCE_REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
KERNEL_ROOT="${GR_DECODE_ATTEN_ROOT:-$(gr_default_decode_atten_root)}"
INSTALL_PROJECT="${GR_BOOTSTRAP_INSTALL_PROJECT:-1}"
INSTALL_PROJECT_KERNELS="${GR_BOOTSTRAP_INSTALL_PROJECT_KERNELS:-0}"
INSTALL_TRTLLM_KERNELS="${GR_BOOTSTRAP_INSTALL_TRTLLM_KERNELS:-0}"
RUN_SMOKE="${GR_BOOTSTRAP_RUN_SMOKE:-0}"
PIP_USER="${GR_BOOTSTRAP_PIP_USER:-1}"
PIP_BREAK_SYSTEM_PACKAGES="${GR_BOOTSTRAP_PIP_BREAK_SYSTEM_PACKAGES:-auto}"

pip_install() {
  local args=(install)
  if [[ "${PIP_USER}" == "1" ]]; then
    args+=(--user)
  fi
  if [[ "${PIP_BREAK_SYSTEM_PACKAGES}" == "1" ]] || {
    [[ "${PIP_BREAK_SYSTEM_PACKAGES}" == "auto" ]] &&
      "${PYTHON_BIN}" -m pip install --help 2>/dev/null | grep -q -- "--break-system-packages"
  }; then
    args+=(--break-system-packages)
  fi
  "${PYTHON_BIN}" -m pip "${args[@]}" "$@"
}

usage() {
  cat <<'EOF'
Usage: scripts/bootstrap_container_env.sh [options]

Set up a fresh container for gr-inference development/serving.

Options:
  --kernel-root PATH        gr-decode-attention root. Default: $GR_DECODE_ATTEN_ROOT or third_party/gr-decode-attention
  --project-kernels         Also install gr-inference[kernels] optional deps
  --trtllm-kernels          Also install gr-inference[trtllm-kernels] optional deps
  --skip-project            Do not pip install this repository
  --skip-kernel-deps        Deprecated no-op. Submodule requirements are never installed.
  --smoke                   Run real decode attention smoke test after import checks
  --help                    Show this message

Typical:
  scripts/bootstrap_container_env.sh --project-kernels --smoke
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kernel-root)
      KERNEL_ROOT="$2"
      shift 2
      ;;
    --project-kernels)
      INSTALL_PROJECT_KERNELS=1
      shift
      ;;
    --trtllm-kernels)
      INSTALL_TRTLLM_KERNELS=1
      shift
      ;;
    --skip-project)
      INSTALL_PROJECT=0
      shift
      ;;
    --skip-kernel-deps)
      echo "WARNING: --skip-kernel-deps is deprecated; submodule requirements are never installed." >&2
      shift
      ;;
    --smoke)
      RUN_SMOKE=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

echo "== gr-inference container bootstrap =="
echo "repo root:    ${REPO_ROOT}"
echo "python:       $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
echo "kernel root:  ${KERNEL_ROOT}"

cd "${REPO_ROOT}"

if [[ "${INSTALL_PROJECT}" == "1" ]]; then
  if [[ "${INSTALL_PROJECT_KERNELS}" == "1" && "${INSTALL_TRTLLM_KERNELS}" == "1" ]]; then
    echo "== installing gr-inference with [kernels,trtllm-kernels] extras =="
    pip_install -e ".[kernels,trtllm-kernels]"
  elif [[ "${INSTALL_PROJECT_KERNELS}" == "1" ]]; then
    echo "== installing gr-inference with [kernels] extras =="
    pip_install -e ".[kernels]"
  elif [[ "${INSTALL_TRTLLM_KERNELS}" == "1" ]]; then
    echo "== installing gr-inference with [trtllm-kernels] extras =="
    pip_install -e ".[trtllm-kernels]"
  else
    echo "== installing gr-inference =="
    pip_install -e .
  fi
fi

if [[ ! -d "${KERNEL_ROOT}" ]]; then
  echo "ERROR: gr-decode-attention root does not exist: ${KERNEL_ROOT}" >&2
  echo "Initialize submodules with: git submodule update --init --recursive" >&2
  echo "Or set GR_DECODE_ATTEN_ROOT / pass --kernel-root PATH." >&2
  exit 1
fi

echo "== validating Python/kernel imports =="
GR_DECODE_ATTEN_ROOT="${KERNEL_ROOT}" "${PYTHON_BIN}" - <<'PY'
import os
import sys

import torch

print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

try:
    import flash_attn
    print("flash_attn OK", getattr(flash_attn, "__version__", "unknown"))
except Exception as exc:
    print("flash_attn unavailable", repr(exc))

try:
    import flashinfer
    import flashinfer.norm
    import flashinfer.rope

    print("flashinfer OK", getattr(flashinfer, "__version__", "unknown"))
    print("flashinfer.norm.rmsnorm", hasattr(flashinfer.norm, "rmsnorm"))
    print(
        "flashinfer.norm.fused_add_rmsnorm",
        hasattr(flashinfer.norm, "fused_add_rmsnorm"),
    )
    print(
        "flashinfer.rope.apply_rope_pos_ids",
        hasattr(flashinfer.rope, "apply_rope_pos_ids"),
    )
except Exception as exc:
    print("flashinfer unavailable", repr(exc))

try:
    import importlib
    import importlib.metadata

    dist = importlib.metadata.distribution("tokenspeed-trtllm-kernel")
    print("tokenspeed-trtllm-kernel OK", dist.version)
    loaded = None
    top_level = dist.read_text("top_level.txt") or ""
    candidates = [line.strip() for line in top_level.splitlines() if line.strip()]
    candidates.append("tokenspeed_trtllm_kernel")
    for candidate in candidates:
        try:
            importlib.import_module(candidate)
            loaded = candidate
            break
        except Exception:
            pass
    print("tokenspeed-trtllm-kernel loaded module", loaded)
except Exception as exc:
    print("tokenspeed-trtllm-kernel unavailable", repr(exc))

try:
    import tensorrt_llm

    print("tensorrt_llm OK", getattr(tensorrt_llm, "__version__", "unknown"))
except Exception as exc:
    print("tensorrt_llm unavailable", repr(exc))

print("torch.ops.trtllm", hasattr(torch.ops, "trtllm"))
print(
    "torch.ops.trtllm.fused_qk_norm_rope",
    hasattr(torch.ops.trtllm, "fused_qk_norm_rope")
    if hasattr(torch.ops, "trtllm")
    else False,
)

kernel_root = os.environ["GR_DECODE_ATTEN_ROOT"]
if not os.path.isdir(kernel_root):
    raise SystemExit(f"gr-decode_atten root does not exist: {kernel_root}")

sys.path.insert(0, kernel_root)
import cutlass  # noqa: F401
import quack  # noqa: F401
import cuda.bindings.driver  # noqa: F401
import interface

if not hasattr(interface, "beam_decode_attn"):
    raise SystemExit("gr-decode_atten interface imported, but beam_decode_attn is missing")

print("gr-decode_atten interface OK", interface.__file__)
print("has beam_decode_attn", hasattr(interface, "beam_decode_attn"))
PY

if [[ "${RUN_SMOKE}" == "1" ]]; then
  echo "== running real decode attention smoke =="
  GR_DECODE_ATTEN_ROOT="${KERNEL_ROOT}" "${PYTHON_BIN}" -m pytest tests/test_real_decode_attention_smoke.py -q -s
fi

echo "== bootstrap complete =="
