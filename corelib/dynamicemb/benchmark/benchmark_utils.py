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

import math
import os
import subprocess
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

# H100 SXM: 18 NVLink4 links x 25 GB/s per direction, one 400 Gb/s NDR NIC per GPU.
NVLINK_PEAK_GBPS = 450.0
IB_PEAK_GBPS = 50.0


class GPUTimer:
    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def start(self):
        self.start_event.record()

    def stop(self):
        self.end_event.record()

    def elapsed_time(self):
        """
        return in ms
        """
        self.dist_sync()
        torch.cuda.synchronize()
        return self.start_event.elapsed_time(self.end_event)

    def dist_sync(self):
        torch.distributed.barrier(device_ids=[torch.cuda.current_device()])


def init_distributed() -> Tuple[int, int, int, int]:
    """Init NCCL from torchrun's variables, or derive them from srun's.

    Returns (rank, world_size, local_rank, local_world_size): the GPUs of a
    host, which TorchRec's planner reads (the hierarchical dist's node is
    NCCL's LSA team instead: :func:`hier_node_size`). Exports
    LOCAL_WORLD_SIZE, and under a bare srun MASTER_ADDR (the step's first
    node) and MASTER_PORT (29500) unless they are set.
    """
    if "RANK" not in os.environ and "SLURM_PROCID" in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
        # env:// needs these too, and Slurm sets neither; rank 0 runs on the
        # step's first node.
        if "MASTER_ADDR" not in os.environ:
            nodes = (
                os.environ.get("SLURM_STEP_NODELIST")
                or os.environ["SLURM_JOB_NODELIST"]
            )
            os.environ["MASTER_ADDR"] = subprocess.check_output(
                ["scontrol", "show", "hostnames", nodes], text=True
            ).split()[0]
        os.environ.setdefault("MASTER_PORT", "29500")
    if "LOCAL_WORLD_SIZE" not in os.environ:
        # Some launchers (e.g. pyxis) set RANK and LOCAL_RANK but not this.
        if "SLURM_NTASKS" in os.environ and "SLURM_NNODES" in os.environ:
            tasks = int(os.environ["SLURM_NTASKS"])
            local_size = tasks // int(os.environ["SLURM_NNODES"])
        else:
            local_size = torch.cuda.device_count()
        os.environ["LOCAL_WORLD_SIZE"] = str(local_size)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl", device_id=torch.device(f"cuda:{local_rank}")
    )
    return rank, world_size, local_rank, local_world_size


def hier_node_size(
    local_world_size: int, group: Optional[dist.ProcessGroup] = None
) -> int:
    """The GPUs of a node as the hierarchical dist takes them: NCCL's LSA
    team on the ranks of ``group`` (the default one if None;
    ``dynamicemb.shard.node_size``, collective). A host's GPUs on Eos; with
    multi-node NVLink (GB200 and GB300 NVL72), the job's GPUs of the rack
    when the job runs inside one rack, and across racks what NCCL gives
    (docs/hier_output_dist.md, section 4.2). Where node_size raises, on
    every rank alike (a build without the hierarchical dist, an NCCL error:
    see its docstring), the GPUs of a host, ``local_world_size``, with a
    note on rank 0."""
    from dynamicemb.shard import node_size

    try:
        return node_size(group)
    except RuntimeError as err:
        if dist.get_rank(group) == 0:
            print(f"the node is a host of {local_world_size} GPUs ({err})", flush=True)
        return min(local_world_size, dist.get_world_size(group))


# The capacities of output_dist_capacity, by name.
HIER_CAPACITY_NAMES = ("lookup_keys", "remote_keys")


def hier_needs(
    batches: Sequence,
    tables: Mapping[str, int],
    owner_of: Callable[[torch.Tensor, str], torch.Tensor],
    local_size: Optional[int] = None,
    node_dedup: bool = True,
    group: Optional[dist.ProcessGroup] = None,
) -> Dict[str, int]:
    """What the hierarchical dist (``output_dist="hierarchical"``) of one
    row-wise sharding needs of each capacity for some calls: the most any
    rank needs in any of them, counted as the op counts its needs.
    ``remote_keys`` comes from the keys each GPU gets from its node for the
    owners of other nodes; ``lookup_keys`` from the keys of its own node
    each owner receives and the node-unique (table, key) it receives from
    each other node. A rank sends each (table, key) of a call once: a call
    on the dist deduplicates the sharding's keys (index dedup), whatever the
    sharder's ``use_index_dedup``.

    Args:
        batches: this rank's KeyedJaggedTensor of each call. Only the keys of
            the sharding's features count.
        tables: the table of each of the sharding's features.
        owner_of: ``owner_of(keys, feature)``, the rank that owns each key,
            given the keys in the KJT's own dtype. An ``int32`` key is owned
            as the ``int64`` key of its value (``key_ownership.owning_shard``).
        local_size: GPUs per node, as the dist takes them: NCCL's LSA team
            on the ranks of ``group`` (``dynamicemb.shard.node_size``), not
            LOCAL_WORLD_SIZE. None reads it so.
        node_dedup: across nodes an owner receives each (table, key) once
            per node; False: every occurrence.
        group: the sharding's process group (the default one if None).

    Collective: every rank of ``group`` calls it with its own batches, the
    same number of them on every rank.
    """
    if local_size is None:
        from dynamicemb.shard import node_size

        local_size = node_size(group)
    world_size, rank = dist.get_world_size(group), dist.get_rank(group)
    num_nodes, node = world_size // local_size, rank // local_size
    names = HIER_CAPACITY_NAMES
    device = torch.device("cuda", torch.cuda.current_device())
    needs = torch.zeros(len(names), dtype=torch.int64, device=device)
    ranks = torch.arange(world_size, device=device)
    for kjt in batches:
        # each feature's keys, without the totals a KJT keeps once asked for
        # them (a call on it must find its own)
        totals = [part.sum() for part in kjt.lengths().split(kjt.stride_per_key())]
        values = dict(zip(kjt.keys(), kjt.values().split(torch.stack(totals).tolist())))
        parts = []
        for feature, table in tables.items():
            keys = values[feature].to(device=device)
            owners = owner_of(keys, feature).to(torch.int64)
            keys = keys.to(torch.int64)
            parts.append(torch.stack([torch.full_like(keys, table), keys, owners], 1))
        # (table, key, owner) of each key this rank sends, each (table, key)
        # once (index dedup)
        sent = torch.unique(torch.cat(parts), dim=0)
        counts = torch.bincount(sent[:, 2], minlength=world_size)
        every = [torch.empty_like(counts) for _ in range(world_size)]
        dist.all_gather(every, counts, group=group)
        matrix = torch.stack(every)  # [requester, owner]
        # [requester node, owner node, rail]: what a node's GPUs send each owner
        by_node = matrix.view(num_nodes, local_size, num_nodes, local_size).sum(dim=1)
        node_keys = by_node.sum(dim=1)  # [node, rail]: what a GPU gets from its node
        own_node = by_node[ranks[:num_nodes], ranks[:num_nodes]]  # [node, rail]
        # the owner's keys, by source rank
        order = torch.argsort(sent[:, 2], stable=True)
        received = matrix[:, rank]
        keys = torch.empty(int(received.sum()), 2, dtype=torch.int64, device=device)
        dist.all_to_all_single(
            keys,
            sent[order][:, :2].contiguous(),
            received.tolist(),
            counts.tolist(),
            group=group,
        )
        source_node = torch.repeat_interleave(ranks // local_size, received)
        from_own_node = source_node == node
        if node_dedup:
            others = torch.cat(
                [source_node[~from_own_node, None], keys[~from_own_node]], 1
            )
            from_other_nodes = torch.unique(others, dim=0).size(0)
        else:
            from_other_nodes = int((~from_own_node).sum())
        call = torch.tensor(
            [
                int(from_own_node.sum()) + from_other_nodes,
                int((node_keys - own_node).max()),
            ],
            device=device,
        )
        needs = torch.maximum(needs, call)
    dist.all_reduce(needs, op=dist.ReduceOp.MAX, group=group)
    return dict(zip(names, needs.tolist()))


def hier_capacities(
    needs: Mapping[str, int], factor: float = 1.25, min_extra: int = 4096
) -> Dict[str, int]:
    """``output_dist_capacity`` for ``needs`` (:func:`hier_needs`): each
    capacity ``factor`` times its need, and at least ``min_extra`` entries
    more (small calls)."""
    return {
        name: max(math.ceil(factor * needs[name]), needs[name] + min_extra)
        for name in HIER_CAPACITY_NAMES
    }
