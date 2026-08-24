#!/usr/bin/env bash
set -Eeuo pipefail

readonly fbgemm_tag=v1.8.0
readonly fbgemm_source=/workspace/deps/fbgemm
readonly torch_tensor_body=/usr/local/lib/python3.12/dist-packages/torch/include/ATen/core/TensorBody.h

torch_tensor_body_backup=
backup_ready=false

restore_tensor_body() {
  local exit_status="$1"
  trap - EXIT
  set +e

  if [[ "${backup_ready}" == true ]]; then
    cp -p -- "${torch_tensor_body_backup}" "${torch_tensor_body}" || exit_status=1
    cmp -s -- "${torch_tensor_body_backup}" "${torch_tensor_body}" || exit_status=1
  fi
  if [[ -n "${torch_tensor_body_backup}" ]]; then
    rm -f -- "${torch_tensor_body_backup}" || exit_status=1
  fi

  exit "${exit_status}"
}
trap 'restore_tensor_body "$?"' EXIT

python3 -m pip install --no-cache-dir setuptools-git-versioning scikit-build
git clone --recursive --branch "${fbgemm_tag}" \
  https://github.com/pytorch/FBGEMM.git "${fbgemm_source}"
test "$(git -C "${fbgemm_source}" rev-parse HEAD)" = \
  "$(git -C "${fbgemm_source}" rev-list -n 1 "${fbgemm_tag}")"

cd "${fbgemm_source}/fbgemm_gpu"
torch_tensor_body_backup="$(mktemp /tmp/TensorBody.h.fbgemm-build.XXXXXX)"
cp -p -- "${torch_tensor_body}" "${torch_tensor_body_backup}"
backup_ready=true

test "$(grep -Fc 'return at::Tensor(std::forward<Args>(args)...);' "${torch_tensor_body}")" -eq 1
test "$(grep -Fc 'return repr_type(std::forward<Args>(args)...);' "${torch_tensor_body}")" -eq 0
sed -i \
  's@return at::Tensor(std::forward<Args>(args)\.\.\.);@return repr_type(std::forward<Args>(args)...);@' \
  "${torch_tensor_body}"
test "$(grep -Fc 'return at::Tensor(std::forward<Args>(args)...);' "${torch_tensor_body}")" -eq 0
test "$(grep -Fc 'return repr_type(std::forward<Args>(args)...);' "${torch_tensor_body}")" -eq 1

python3 setup.py install --build-target=default --build-variant=cuda \
  --package_channel=release \
  -DTORCH_CUDA_ARCH_LIST='8.0 9.0a 10.0'
