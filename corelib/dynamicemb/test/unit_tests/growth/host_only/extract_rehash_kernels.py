# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cut the rehash kernels and their wrappers out of the real table_rows.cu.

usage: extract_rehash_kernels.py <path of table_rows.cu> <output directory>

Writes new_kernels.inc, which test_rehash_kernels.cpp includes after kernel_stubs.h:
``kernel<<<grid, block, 0, stream>>>(args)`` becomes ``emulate_launch(kernel, grid, block, 0,
stream, args)``, which runs every thread of every block one after the other.
"""
import os
import re
import sys

FIRST = "constexpr int64_t kEmptyKey"
LAST = "// Leaves the slots named by `slot_index`"


def launches(text: str) -> str:
    return re.sub(
        r"(\w+)<<<([^>]+)>>>\(",
        lambda m: "emulate_launch(%s, %s, " % (m.group(1), m.group(2)),
        text,
    )


def extract(source: str) -> str:
    a = source.index(FIRST)
    b = source.index(LAST, a)
    return launches(source[a:b])


def main(argv) -> int:
    with open(argv[1]) as f:
        text = extract(f.read())
    with open(os.path.join(argv[2], "new_kernels.inc"), "w") as f:
        f.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
