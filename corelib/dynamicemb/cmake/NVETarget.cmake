# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

function(dynamicemb_configure_nve_target target)
    if(NOT NVE_CACHE_LINE_SIZE)
        execute_process(
            COMMAND getconf LEVEL1_DCACHE_LINESIZE
            OUTPUT_VARIABLE NVE_CACHE_LINE_SIZE
            OUTPUT_STRIP_TRAILING_WHITESPACE
            COMMAND_ERROR_IS_FATAL ANY
        )
    endif()

    target_compile_definitions(${target} PRIVATE
        NVE_FEATURE_HT_PART_FNV1A=1
        NVE_FEATURE_HT_PART_MURMUR=1
        NVE_FEATURE_HT_PART_RRXMRRXMSX0=1
        NVE_FEATURE_HT_PART_STD_HASH=1
        NVE_FEATURE_HT_MASK_64=1
        NVE_FEATURE_HT_MASK_32=1
        NVE_FEATURE_HT_MASK_16=1
        NVE_FEATURE_HT_MASK_8=1
        NVE_FEATURE_HT_KEY_64=1
        NVE_FEATURE_HT_KEY_32=1
        NVE_FEATURE_HT_KEY_16=1
        NVE_FEATURE_HT_KEY_8=1
        NVE_CACHE_LINE_SIZE=${NVE_CACHE_LINE_SIZE}
    )
endfunction()
