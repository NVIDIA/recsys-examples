// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import java.io.Serializable;

/** A copied row containing only the fields packed for a GPU kernel. */
interface PackedRow extends Serializable {
    Object field(int packedOrdinal);
}
