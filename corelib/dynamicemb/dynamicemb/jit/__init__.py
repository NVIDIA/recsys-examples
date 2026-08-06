# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""JIT glue for the LruLfu (LRU_LFU strategy) eviction cubins.

The packaged eviction fatbins live alongside this subpackage (``dynamicemb/jit/
*.fatbin``); the C++ source that produces them is under ``src/jit/``."""

from dynamicemb.jit.score_jit import (
    ensure_lex_fatbin_loaded,
    register_score_function,
    score_function_key,
)

__all__ = [
    "ensure_lex_fatbin_loaded",
    "register_score_function",
    "score_function_key",
]
