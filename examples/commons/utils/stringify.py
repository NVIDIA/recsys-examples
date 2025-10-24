# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import torch


def recursive_traverse_dict(input_dict, key_prefix):
    ret = {}
    for key, value in input_dict.items():
        if isinstance(value, dict):
            sub_dict = recursive_traverse_dict(value, key_prefix=f"{key_prefix}.{key}")
            ret.update(sub_dict)
        else:
            ret.update({f"{key_prefix}.{key}": value})
    return ret


def stringify_dict(input_dict, prefix="", sep=","):
    ret = recursive_traverse_dict(input_dict, prefix)
    output = ""
    for key, value in ret.items():
        if isinstance(value, torch.Tensor):
            value.float()
            assert value.dim() == 0
            value = value.cpu().item()
            output += key + ": " + f"{value:6f}{sep}"
        elif isinstance(value, float):
            output += key + ": " + f"{value:6f}{sep}"
        elif isinstance(value, int):
            output += key + ": " + f"{value}{sep}"
        else:
            assert RuntimeError(f"stringify dict not supports type {type(value)}")
    # remove the ending sep
    pos = output.rfind(sep)
    return output[0:pos]
