#!/bin/bash
# Adapts the HSTU package of the FBGEMM submodule
# (third_party/FBGEMM/fbgemm_gpu/experimental/hstu) to the 26.09 base. Run it
# from that directory, or give the directory, before building or installing.
# The Dockerfile's devel stage runs it, and so does the CI's wheel build.
#
# - PyTorch 2.14's headers need C++20; the submodule's setup.py asks for C++17.
# - CUTLASS DSL 4.8 drops CooperativeGroup's alignment argument, which 4.5.2
#   deprecated and ignored. The Blackwell backward still passes it, equal to
#   the size, in 7 calls.
#
# A second run changes nothing. Drop each edit once the submodule has it.
set -euo pipefail
cd "${1:-.}"

sed -i 's/-std=c++17/-std=c++20/g' setup.py
grep -q -- '-std=c++20' setup.py
! grep -q -- '-std=c++17' setup.py

python3 - <<'PY'
import re

path = "src/hstu_blackwell/hstu_bwd.py"
src = open(path).read()
src, n = re.subn(
    r"(CooperativeGroup\(\n\s+pipeline\.Agent\.Thread,\n(\s+)([^\n]+),\n)\2\3,\n",
    r"\1",
    src,
)
assert n in (0, 7), f"{path}: {n} CooperativeGroup calls with an alignment argument, expected 7 (or 0 once fixed)"
open(path, "w").write(src)
print(f"{path}: dropped the alignment argument of {n} CooperativeGroup calls")
PY
