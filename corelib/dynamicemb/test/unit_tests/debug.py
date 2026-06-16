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

import copy
import os
import shutil
import time
from itertools import accumulate
from typing import List

import torch  # usort:skip
import torch.distributed as dist
from torch import Tensor, nn  # usort:skip
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor


class Printer:
    def __init__(self, rank: int):
        self._rank = rank

    def print(self, info: str):
        print(f"[rank{self._rank}]", info)


def create_debug_folder(folder_path, printer=None):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        if printer:
            printer.print(f"folder already exists: {folder_path}")
        else:
            print(f"folder already exists: {folder_path}")


def monitor_debug_folder(folder_path, printer=None):
    while True:
        if os.path.exists(folder_path):
            if printer:
                printer.print(f"detected folder: {folder_path}")
            else:
                print(f"detected folder: {folder_path}")
            break
        time.sleep(1)


class Debugger:
    def __init__(self, is_sequence: bool = False) -> None:
        self._debug_iter = int(os.getenv("PYHKV_DEBUG_ITER", 0))

        self._rank = dist.get_rank()
        self.p = Printer(self._rank)
        self._world_size = dist.get_world_size()

        self._root_path = "./debug_folder/"
        if self._rank == 0:
            if os.path.exists(self._root_path):
                shutil.rmtree(self._root_path)

        dist.barrier()

        if self._rank == 0:
            create_debug_folder(self._root_path, self.p)
        else:
            monitor_debug_folder(self._root_path, self.p)

        self.rank_path = None
        self._cur_iter = 0

        self._feature_before_all2all_unsorted = []
        self._feature_before_all2all = []
        self._feature_before_all2all_kt = []
        self.merged_features = []
        self.embedding_values = []

    def feature_before_all2all(self, local_feature: KeyedJaggedTensor) -> None:
        self._cur_iter += 1
        if self._cur_iter > self._debug_iter:
            return

        iter_path = self._root_path + "iter" + str(self._cur_iter) + "/"
        self.rank_path = (
            iter_path + "rank" + str(self._rank) + "_feature_before_all2all.log"
        )
        if self._rank == 0:
            create_debug_folder(iter_path, self.p)
        else:
            monitor_debug_folder(iter_path, self.p)

        file_ = open(self.rank_path, "w")
        indices_np = local_feature.values().detach().cpu().numpy()
        offsets_np = local_feature.offsets().detach().cpu().numpy()
        indices_arr_str = ",".join(map(str, indices_np.flatten()))
        offsets_arr_str = ",".join(map(str, offsets_np.flatten()))
        file_.write(indices_arr_str + "\n")
        file_.write(offsets_arr_str + "\n")
        file_.close()

        feature_batch = offsets_np.shape[0] - 1
        self.p.print(
            f"[DynamicEmb] feature batch size before alltoall: {feature_batch}"
        )

        expected_feature = []
        for k in range(feature_batch):
            expected_feature.append([])

        for i in range(offsets_np.shape[0] - 1):
            for j in range(offsets_np[i], offsets_np[i + 1]):
                expected_feature[i].append(indices_np[j])
        self._feature_before_all2all_unsorted.append(copy.deepcopy(expected_feature))
        expected_feature = [sorted(i) for i in expected_feature]

        self._feature_before_all2all.append(expected_feature)
        self._feature_before_all2all_kt.append(local_feature)

    def pooled_embds_after_all2all(
        self,
        embds: torch.Tensor,
        dims: List[int],
        feature_num: int,
        target_features: List[int],
    ) -> None:
        if self._cur_iter > self._debug_iter:
            return

        print("DynamicEmb>>> embds size=", embds.size())

        # self_feature = self.merged_features[self._cur_iter - 1]
        self_feature = self._feature_before_all2all[self._cur_iter - 1]

        feature_batch = len(self_feature)
        batch_size = feature_batch // feature_num
        assert feature_batch % feature_num == 0

        prefix_dims = [0] + list(accumulate(dims))

        prefix_dims[-1]

        h_outputs = embds.to("cpu")

        print("DynamicEmb  feature batch=", feature_batch)
        print("DynamicEmb batch size=", batch_size, dims)

        for i, bag in enumerate(self_feature):
            feature_id = i // batch_size
            if feature_id not in target_features:
                continue
            batch = i % batch_size

            res = 0
            for index in bag:
                # print("- index:", feature_batch, index)
                res += float(index % 100000)

            first_idx2emb = prefix_dims[feature_id]
            last_idx2emb = first_idx2emb + dims[feature_id] - 1
            if (
                h_outputs[batch, first_idx2emb] != res
                or h_outputs[batch, last_idx2emb] != res
            ):
                print(
                    "DynamicEmb: lookup result after dist mismatched=",
                    h_outputs[batch, first_idx2emb].item(),
                    h_outputs[batch, last_idx2emb].item(),
                    res,
                )

    def sequence_embds_after_all2all(
        self,
        jagged_tensors: List[JaggedTensor],
        feature_names: List[str],
        dyn_emb_features: List[str],
        dim: int,
    ):
        if self._cur_iter > self._debug_iter:
            return
        feature_num = len(jagged_tensors)
        self.p.print(f"Total feature number = {feature_num}")

        self_feature_kt = self._feature_before_all2all_kt[self._cur_iter - 1]
        self_feature = self._feature_before_all2all_unsorted[self._cur_iter - 1]

        feature_batch = len(self_feature)
        feature_batch // feature_num
        assert feature_batch % feature_num == 0

        counter = 0
        for i, (feature_name, values) in enumerate(
            zip(self_feature_kt.keys(), jagged_tensors)
        ):
            embds = values.values().to("cpu")
            if feature_name not in dyn_emb_features:
                continue
            indices = self_feature_kt[feature_name].values().to("cpu")
            if indices.size()[0] != embds.size()[0]:
                print("DynamicEmb:mismatched")
            else:
                print("DynamicEmb:matched", indices.size()[0])
                print(type(indices))
            counter = 0
            for i in range(indices.size()[0]):
                index = indices[i].item()
                expected = float(index % 100000)
                if expected != embds[counter, 0] or expected != embds[counter, dim - 1]:
                    print(
                        "DynamicEmb: lookup result after dist mismatched=",
                        embds[counter, 0].item(),
                        embds[counter, dim - 1].item(),
                        expected,
                    )
                counter += 1
