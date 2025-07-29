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
import gc
import os

import torch
from megatron.core import parallel_state, tensor_parallel


def initialize_single_rank():
    if torch.distributed.is_initialized():
        return
    torch.set_printoptions(precision=6, sci_mode=False)
    rank = 0
    device: torch.device = torch.device(f"cuda:{rank}")
    backend = "nccl"
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(
        backend=backend, init_method="tcp://127.0.0.1:12345", rank=rank, world_size=1
    )


def initialize_distributed():
    if torch.distributed.is_initialized():
        return
    torch.set_printoptions(precision=6, sci_mode=False)
    rank = int(os.environ["LOCAL_RANK"])
    device: torch.device = torch.device(f"cuda:{rank}")
    backend = "nccl"
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend=backend)


def initialize_model_parallel(tensor_model_parallel_size=1):
    if parallel_state.model_parallel_is_initialized():
        return
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size,
    )
    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])


def destroy_global_state():
    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])

    # TODO, find the reason why destroying pg hit nccl error when tpsize > 1
    if parallel_state.model_parallel_is_initialized():
        if parallel_state.get_tensor_model_parallel_world_size() == 1:
            torch.distributed.destroy_process_group(
                group=parallel_state.get_tensor_model_parallel_group()
            )
            torch.distributed.destroy_process_group(
                group=parallel_state.get_data_parallel_group(with_context_parallel=True)
            )
        parallel_state.destroy_model_parallel()
    torch.cuda.empty_cache()
    gc.collect()


def set_random_seed(seed_):
    if not parallel_state.model_parallel_is_initialized():
        initialize_model_parallel()
    import random

    import numpy as np

    """Set random seed for reproducibility."""
    if seed_ is not None and seed_ > 0:
        # Only CP/TP ranks share the same seed
        # Ensure that different pipeline MP stages get different seeds.
        seed = seed_ + (100 * parallel_state.get_pipeline_model_parallel_rank())
        # Ensure different data parallel ranks get different seeds
        seed = seed + (10 * parallel_state.get_data_parallel_rank())

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.device_count() > 0:
            tensor_parallel.model_parallel_cuda_manual_seed(seed)

            # We must maintain an rng state for torchrec, because with different world size, the state evolution differ
            # guarantee randomness across DPxTPxCPxPP for embedding-group
            seed = seed + 1234
            seed = seed + (1000 * parallel_state.get_context_parallel_rank())
            seed = seed + (10000 * parallel_state.get_tensor_model_parallel_rank())
            rng_tracker = tensor_parallel.get_cuda_rng_tracker()
            rng_tracker.add("sharded-embedding-group-seed", seed)

    else:
        raise ValueError("Seed ({}) should be a positive integer.".format(seed))
