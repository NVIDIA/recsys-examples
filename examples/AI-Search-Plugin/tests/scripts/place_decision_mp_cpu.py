# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""4-process gloo test of expert placement's auto-off decision: ranks that would decide differently whether to
re-place (here: a different ASRT_PLACE_MIN_IMB per rank) must all follow rank 0's decision, so no rank enters the
perm broadcast / expert move alone (that would hang the step).
Run: python3 tests/scripts/place_decision_mp_cpu.py"""
import os
import socket
import sys
import types

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

E = 16


def _fake_megatron():
    ps = types.ModuleType("megatron.core.parallel_state")
    ps.get_expert_model_parallel_world_size = lambda: 1
    ps.get_expert_model_parallel_group = lambda: None
    sys.modules.setdefault("megatron", types.ModuleType("megatron"))
    sys.modules.setdefault("megatron.core", types.ModuleType("megatron.core"))
    sys.modules["megatron.core"].parallel_state = ps
    sys.modules["megatron.core.parallel_state"] = ps
    torch.cuda.current_device = lambda: "cpu"
    torch.cuda.synchronize = lambda *a, **k: None


class _Experts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_local_experts = E


class _MoE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.router = torch.nn.Linear(4, E, bias=False)
        self.experts = _Experts()


class _Layer(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.layer_number = n
        self.mlp = _MoE()


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer(1), _Layer(2)])


def worker(rank, world, q, rank0_skips, port):
    _fake_megatron()
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    from ai_search_plugin.adapters.megatron_placement import ExpertPlacementManager
    # rank 0 skips (huge threshold) and the others want to move, or the other way round
    me_skips = (rank == 0) == rank0_skips
    os.environ["ASRT_PLACE_MIN_IMB"] = "100" if me_skips else "1.0"
    torch.manual_seed(0)
    mgr = ExpertPlacementManager([_Model()], None, num_layers=2, num_experts=E, ep_group=None)
    mgr.ep, mgr.n_local = 4, 4
    mgr.counts[:, :4] = 100.0  # hot cluster on "EP rank 0"
    # the toy model is not really expert-parallel: test only the decision + perm broadcast, not the data move
    mgr.apply = lambda perms: {"layers": len(perms), "bytes_sent": 0, "apply_s": 0.0}
    info = mgr.maybe_rebalance()
    q.put((rank, bool(info.get("skipped")), int(info.get("experts_moved", 0))))
    dist.destroy_process_group()


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def run(rank0_skips):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _free_port()
    ps = [ctx.Process(target=worker, args=(r, 4, q, rank0_skips, port)) for r in range(4)]
    for p in ps:
        p.start()
    for p in ps:
        p.join(120)
        assert p.exitcode == 0, ("worker failed or hung", p.exitcode)
    res = sorted(q.get() for _ in ps)
    assert len({(s, n) for _, s, n in res}) == 1, res
    assert res[0][1] == rank0_skips, res
    return res


if __name__ == "__main__":
    print("ok", run(True), run(False))
