# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two local replicas can exchange aligned rows directly through peer memory."""

import socket

import torch
import torch.distributed as dist

try:
    from torch.distributed import _symmetric_memory as symm_mem
except ImportError:
    symm_mem = None


def peer_state(group, device):
    """Agree on the peer transport supported by both replicas once at startup."""
    if dist.get_world_size(group) != 2:
        return []
    props = [
        torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())
    ]
    uuids = [str(getattr(p, "uuid", "")) for p in props]
    available = all(
        callable(getattr(symm_mem, f, None)) for f in ("empty", "rendezvous")
    )
    peers = [None, None]
    me = (socket.gethostname(), uuids[device], uuids, available)
    dist.all_gather_object(peers, me, group=group)
    (host0, gpu0, devs0, avail0), (host1, gpu1, devs1, avail1) = peers
    if not (avail0 and avail1 and gpu0 and gpu1) or (
        host0 != host1 or gpu0 == gpu1 or gpu1 not in devs0 or gpu0 not in devs1
    ):
        return []
    peer = uuids.index(peers[1 - dist.get_rank(group)][1])
    caps = torch.zeros(2, dtype=torch.int32, device=device)
    if torch.cuda.can_device_access_peer(device, peer):
        access, direct = torch.ops.dyn_emb.replica_peer_capabilities(caps[:1], peer)
        caps[0], caps[1] = int(access), int(direct)
    dist.all_reduce(caps, op=dist.ReduceOp.MIN, group=group)
    access, direct = caps.tolist()
    return [torch.tensor(1 + direct, dtype=torch.int64, device="cpu")] if access else []
