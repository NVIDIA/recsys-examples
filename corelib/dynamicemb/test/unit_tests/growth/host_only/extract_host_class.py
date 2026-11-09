# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cut getTotalPhysicalMemory() and the HostVMMTensor class out of the real vmm_tensor.cu.

usage: extract_host_class.py <path of vmm_tensor.cu> <output directory>

Writes host_class.inc, which test_host_vmm.cpp includes after host_stubs.h, so that the
code under test is the code of the library and not a copy of it.
"""
import os
import sys


def extract(source: str) -> str:
    a = source.index("std::size_t getTotalPhysicalMemory() {")
    b = source.index("#include <pybind11/pybind11.h>")
    memory = source[a:b]
    c = source.index("// A buffer in pinned host memory that the GPU reads and writes")
    d = source.index("} // namespace dyn_emb\n\nvoid bind_vmm_op")
    return (
        memory + "\nnamespace dyn_emb {\n" + source[c:d] + "\n} // namespace dyn_emb\n"
    )


def main(argv) -> int:
    with open(argv[1]) as f:
        text = extract(f.read())
    with open(os.path.join(argv[2], "host_class.inc"), "w") as f:
        f.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
