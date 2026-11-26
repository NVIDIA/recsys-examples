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


class ReplicaPeer:
    def __init__(self, group, max_staging_bytes):
        self.group = group
        self.budget = max_staging_bytes
        self.supported = None
        self.direct_supported = False
        self.workspace = None
        self.merge_stream = None
        self.routes = {}

    def _can_map(self, device):
        # All peers take the same branch even when their visible devices,
        # driver capabilities, and memory limits differ.
        props = [
            torch.cuda.get_device_properties(i)
            for i in range(torch.cuda.device_count())
        ]
        uuids = [str(getattr(p, "uuid", "")) for p in props]
        available = all(
            callable(getattr(symm_mem, f, None)) for f in ("empty", "rendezvous")
        )
        peers = [None, None]
        me = (socket.gethostname(), uuids[device.index], uuids, available)
        dist.all_gather_object(peers, me, group=self.group)
        (host0, gpu0, devs0, avail0), (host1, gpu1, devs1, avail1) = peers
        if not (avail0 and avail1 and gpu0 and gpu1) or (
            host0 != host1 or gpu0 == gpu1 or gpu1 not in devs0 or gpu0 not in devs1
        ):
            return False
        peer = peers[1 - dist.get_rank(self.group)][1]
        can_access = torch.cuda.can_device_access_peer(device.index, uuids.index(peer))
        self.vote = torch.zeros(1, dtype=torch.int32, device=device)
        caps = (
            torch.ops.dyn_emb.replica_peer_capabilities(self.vote, uuids.index(peer))
            if can_access
            else (False, False)
        )
        flags = torch.tensor(caps, device=device, dtype=torch.int32)
        dist.all_reduce(flags, op=dist.ReduceOp.MIN, group=self.group)
        access, self.direct_supported = flags.tolist()
        return bool(access)

    def _all_ready(self, ready):
        self.vote.fill_(int(ready))
        dist.all_reduce(self.vote, op=dist.ReduceOp.MIN, group=self.group)
        return bool(self.vote.item())

    def release_workspace(self):
        if self.workspace is not None:
            # All accesses are joined back to the caller's stream. Drop both
            # peer mappings before allocating a different payload workspace.
            torch.cuda.current_stream(self.workspace[0].device).synchronize()
            self.workspace = None

    def _workspace(self, size, device):
        if self.workspace is not None and self.workspace[0].numel() >= size:
            return self.workspace
        self.release_workspace()
        try:
            arena = symm_mem.empty(size, device=device, dtype=torch.uint8)
        except RuntimeError:
            arena = None
        # Allocating is local and rendezvous is collective. One failed
        # allocation must send *both* ranks to the smaller transport.
        if not self._all_ready(arena is not None):
            return None
        handle = symm_mem.rendezvous(arena, self.group.group_name)
        remote = handle.get_buffer(1 - dist.get_rank(self.group), (size,), torch.uint8)
        self.workspace = arena, handle, remote
        return self.workspace

    def _direct(self, token, table, buckets, rows, slots, capacity):
        # A compact set of physical rows can go straight from the VMM table
        # into the peer inbox using the copy engines. The two row permutations
        # may be completely different. Release each row to the merge stream as
        # soon as its local AND remote source ranges have both been copied.
        row_bytes = table.size(1) * table.element_size()
        nbytes = rows.numel() * row_bytes
        if not self.direct_supported or nbytes + 8 > self.budget:
            return False
        rank = dist.get_rank(self.group)
        cached = self.routes.get(token)
        if cached is None or cached[0] is not rows:
            try:
                compact = bool(
                    ((rows.min() == 0) & (rows.max() == rows.numel() - 1)).item()
                )
            except torch.OutOfMemoryError:
                compact = False
            plan = None
            if self._all_ready(compact):
                try:
                    maps = torch.empty(
                        2 * rows.numel(), device=rows.device, dtype=rows.dtype
                    )
                except torch.OutOfMemoryError:
                    maps = None
                if not self._all_ready(maps is not None):
                    self.routes[token] = rows, None
                    return False
                dist.all_gather_into_tensor(maps, rows, group=self.group)
                peer_rows = maps.view(2, -1)[1 - rank]
                # Amortize the copy-engine notifications at large widths while
                # retaining enough waves to overlap the two streams.
                wave_bytes = min(192 << 20, max(8 << 20, nbytes // 32))
                rows_per_copy = max(1, wave_bytes // row_bytes)
                try:
                    local_wave = rows // rows_per_copy
                    peer_wave = peer_rows // rows_per_copy
                    last = (rows.numel() - 1) // rows_per_copy
                    # Replicas may have allocated rows in opposite orders. Choose
                    # the direction that makes both inputs ready at similar times.
                    reverse = (local_wave + peer_wave - last).abs().sum() < (
                        local_wave - peer_wave
                    ).abs().sum()
                    reverse = bool(reverse.item())
                    if reverse and rank == 1:
                        local_wave = last - local_wave
                    if reverse and rank == 0:
                        peer_wave = last - peer_wave
                    wave = torch.maximum(local_wave, peer_wave)
                    order = torch.argsort(wave * rows.numel() + rows)
                    ends = torch.bincount(wave).cumsum(0).tolist()
                    plan = (
                        rows[order],
                        peer_rows[order],
                        slots[order],
                        ends,
                        rows_per_copy,
                        reverse and rank == 1,
                    )
                except torch.OutOfMemoryError:
                    plan = None
                if not self._all_ready(plan is not None):
                    plan = None
            self.routes[token] = rows, plan
        plan = self.routes[token][1]
        if plan is None:
            return False
        workspace = self._workspace(nbytes + 8, table.device)
        if workspace is None:
            self.routes[token] = rows, None
            return False
        inbox, h, peer = workspace
        ordered, peer_ordered, positions, ends, rows_per_copy, reverse = plan
        inbox[nbytes : nbytes + 8].zero_()
        h.barrier(channel=0)
        torch.ops.dyn_emb.replica_copy_merge_(
            table,
            inbox,
            peer,
            ordered,
            peer_ordered,
            positions,
            buckets,
            ends,
            (capacity, rows_per_copy, reverse),
        )
        h.barrier(channel=1)
        return True

    def sync(self, token, table, buckets, rows, slots, capacity):
        device = table.device
        if self.supported is None:
            self.supported = self._can_map(device)
        if not self.supported:
            return False

        if self._direct(token, table, buckets, rows, slots, capacity):
            return True

        # Producer and consumer reuse two inboxes. The system-scope barrier
        # after the peer stores publishes a complete slab. The second barrier
        # prevents reuse until both consumers have written their table rows.
        slab = min(16 << 20, self.budget // 2) // 16 * 16
        row_bytes = table.size(1) * table.element_size()
        per_slab = slab // row_bytes
        if not per_slab:
            return False
        workspace = self._workspace(2 * slab, device)
        if workspace is None:
            return False
        arena, handle, remote = workspace
        if self.merge_stream is None:
            self.merge_stream = torch.cuda.Stream(device=device)
        merge_stream = self.merge_stream
        send_stream = torch.cuda.current_stream(device)
        ready = [None, None]
        for i, begin in enumerate(range(0, rows.numel(), per_slab)):
            which = i % 2
            if ready[which] is not None:
                send_stream.wait_event(ready[which])
            batch = slice(begin, begin + per_slab)
            region = slice(which * slab, (which + 1) * slab)
            args = rows[batch], slots[batch], buckets
            torch.ops.dyn_emb.replica_peer_transfer_(
                table, remote[region], *args, None, (capacity, False)
            )
            handle.barrier(channel=which)
            merge_stream.wait_stream(send_stream)
            with torch.cuda.stream(merge_stream):
                torch.ops.dyn_emb.replica_peer_transfer_(
                    table, arena[region], *args, None, (capacity, True)
                )
                handle.barrier(channel=2 + which)
                ready[which] = merge_stream.record_event()
        send_stream.wait_stream(merge_stream)
        return True
