#!/usr/bin/env bash
set -Eeuo pipefail

readonly fbgemm_tag=v1.9.0
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

python3 -m pip install --no-cache-dir setuptools-git-versioning scikit-build wheel
git clone --recursive --branch "${fbgemm_tag}" \
  https://github.com/pytorch/FBGEMM.git "${fbgemm_source}"
test "$(git -C "${fbgemm_source}" rev-parse HEAD)" = \
  "$(git -C "${fbgemm_source}" rev-list -n 1 "${fbgemm_tag}")"

# FBGEMM v1.9.0 builds its host C++ with -Werror and a wider warning set, and
# upstream does not build it with this image's CUDA 13.4 and GCC: a new warning
# must not stop the build. Drop the one -Werror of _cc_common.
readonly fbgemm_cc_flags="${fbgemm_source}/cmake/modules/CppLibrary.cmake"
test "$(grep -c '^    -Werror$' "${fbgemm_cc_flags}")" -eq 1
sed -i '/^    -Werror$/d' "${fbgemm_cc_flags}"
test "$(grep -c '^    -Werror$' "${fbgemm_cc_flags}")" -eq 0

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

# Build a wheel and install it. With FBGEMM_WHEEL_DIR set, the wheel is also kept
# there (the Dockerfile's fbgemm_arm64_wheel target exports it for publishing).
python3 setup.py bdist_wheel --build-target=default --build-variant=cuda \
  --package_channel=release \
  -DTORCH_CUDA_ARCH_LIST='8.0 9.0a 10.0'
wheels=(dist/fbgemm_gpu-*.whl)
[[ ${#wheels[@]} -eq 1 && -f ${wheels[0]} ]]
python3 -m pip install --no-cache-dir --no-deps "${wheels[0]}"
if [[ -n "${FBGEMM_WHEEL_DIR:-}" ]]; then
  mkdir -p "${FBGEMM_WHEEL_DIR}"
  cp -- "${wheels[0]}" "${FBGEMM_WHEEL_DIR}/"
  # The wheel's .so files link this torch's libraries, whose C++ ABI is not
  # stable across builds: base_fbgemm checks the image's torch against it.
  python3 -c 'import torch; print(torch.__version__)' > "${FBGEMM_WHEEL_DIR}/torch_version.txt"
fi
