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
"""The hierarchical dist of a row-wise sequence sharding
(``output_dist="fused"``) on the C++ operator ``HierOutputDist``
(``src/hier_output_dist``). Keys and rows move over NVLink inside a node and
over the rails (NCCL GIN) between nodes, with one host sync per call. Design:
docs/hier_output_dist.md (section 7.4 for this file).

:class:`HierSharding` builds the operator at the first ``input_dist()`` and
holds what the calls share. The input dist is in input_dist.py
(``RwSparseFeaturesDist.hier_forward``). The lookup writes its rows into the
operator's window (:meth:`HierSharding.lookup`), and
:class:`RwSequenceHierEmbeddingDist` moves them. A call that does not fit the
capacities runs on TorchRec's dists, and so does every eval forward.
:func:`quiesce` lets every rank free the operators at once.
"""

import os
import warnings
import weakref
from itertools import accumulate
from typing import Callable, Dict, List, Optional, Sequence, Union

import torch
import torch.distributed as dist
from torch.distributed.constants import default_pg_nccl_timeout
from torchrec.distributed import comm_ops
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.embedding_sharding import BaseEmbeddingDist
from torchrec.distributed.sharding.rw_sequence_sharding import RwSequenceEmbeddingDist
from torchrec.distributed.sharding.sequence_sharding import SequenceShardingContext
from torchrec.distributed.types import Awaitable
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

from ..dynamicemb_config import OUTPUT_DIST_CAPACITY_NAMES


def _timeout_s(pg: dist.ProcessGroup) -> float:
    """How long a kernel may wait for a peer: ``DYNAMICEMB_GIN_TIMEOUT_S``,
    else the process group's own timeout (how long its NCCL collectives
    wait)."""
    env = os.environ.get("DYNAMICEMB_GIN_TIMEOUT_S")
    if env is not None:
        return float(env)
    try:
        timeout = pg._get_backend(torch.device("cuda")).options._timeout
    except (AttributeError, RuntimeError):  # no NCCL backend options to read
        timeout = default_pg_nccl_timeout
    return timeout.total_seconds()


def _comparable(arguments: Dict) -> Dict:
    """The operator's arguments as values two ranks can compare (tensors as
    lists)."""
    return {
        name: value.tolist() if isinstance(value, torch.Tensor) else value
        for name, value in arguments.items()
    }


def _first_difference(every: List[Dict]) -> Optional[str]:
    """Where the first rank whose arguments differ from rank 0's differs, in
    words, or None when every rank has rank 0's."""
    for rank, arguments in enumerate(every[1:], start=1):
        for name, value in arguments.items():
            if value != every[0][name]:
                return (
                    f"rank {rank} has {name}={value!r} where rank 0 has "
                    f"{name}={every[0][name]!r}"
                )
    return None


class DynamicEmbSequenceShardingContext(SequenceShardingContext):
    """TorchRec's sequence sharding context, plus the plan of a call that
    took the hierarchical dist (None on TorchRec's dists).

    Not a dataclass: up to TorchRec 1.6 the parent's ``__init__`` is written by
    hand, and a generated one would accept only this class's own fields.
    """

    def __init__(self, *args, plan=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.plan = plan

    def record_stream(self, stream: torch.Stream) -> None:
        super().record_stream(stream)
        if self.plan is not None:
            self.plan.record_stream(stream)


class HierSharding:
    """One row-wise sequence sharding on the hierarchical dist: its operator,
    built at the first ``input_dist()`` (:meth:`setup`), and what its calls
    share. The knobs are read here, from the environment
    (docs/hier_output_dist.md, section 4.4).

    Args:
        pg: the sharding's process group (ranks node-contiguous).
        num_features: features of the sharding.
        embedding_dim: columns of a row of the lookup output.
        output_dtype: the lookup output's dtype.
        capacity: ``output_dist_capacity``.
        asked: the tables' ``output_dist``, ``"fused"`` or ``"auto"``:
            ``"fused"`` raises where ``"auto"`` warns and takes TorchRec's
            dists.
        tables: the table of each feature.
        node_dedup: across nodes a node sends each key once.
        lfu: some table counts frequencies (:attr:`counts_as_weights`).
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        num_features: int,
        embedding_dim: int,
        output_dtype: torch.dtype,
        capacity: Union[int, Dict[str, int]],
        asked: str = "fused",
        tables: Optional[List[int]] = None,
        node_dedup: bool = True,
        lfu: bool = False,
    ) -> None:
        self._pg = pg
        self._rank = dist.get_rank(pg)
        world_size = dist.get_world_size(pg)
        # GPUs per node, or the whole group when it is smaller than a node.
        local_size = min(get_local_size(), world_size)
        self._num_nodes = world_size // local_size
        self._asked = asked
        self._lfu = lfu
        self.dtype = output_dtype  # of the lookup output's rows
        if isinstance(capacity, int):
            capacity = dict.fromkeys(OUTPUT_DIST_CAPACITY_NAMES, capacity)
        # One node needs lookup_keys alone.
        capacities = {"remote_keys": 0, **capacity}
        proxy_mb = float(os.environ.get("DYNAMICEMB_GIN_PROXY_MB", 512))
        capacities["proxy_bytes"] = int(proxy_mb * (1 << 20))
        # What the operator is built from, but the bucketize's rules and the
        # weights (setup()).
        self._arguments = dict(
            world_size=world_size,
            local_size=local_size,
            feature_tables=list(range(num_features) if tables is None else tables),
            dim=embedding_dim,
            dtype=output_dtype,
            node_dedup=node_dedup,
            capacities=capacities,
            timeout_s=_timeout_s(pg),
            ctas=int(os.environ.get("DYNAMICEMB_GIN_CTAS", 0)),
            reduce_ctas=int(os.environ.get("DYNAMICEMB_GIN_REDUCE_CTAS", 8)),
        )
        self.op = None  # the HierOutputDist, once setup() made it
        # Whether the sharding takes the hierarchical dist; None until setup().
        self.takes_hier_dist: Optional[bool] = None
        self._weights = False  # the requesters' keys carry weights
        self._out_weights = False  # the owner's KJT carries weights
        # The owner's KJT of each call on this path, as its second wait
        # returned it, and the call's plan (owns()).
        self._plans: "weakref.WeakKeyDictionary[KeyedJaggedTensor, object]" = (
            weakref.WeakKeyDictionary()
        )
        # While a lookup runs (lookup()): the call's rows, and how many of
        # them the lookup groups took so far (lookup_buffer()).
        self._lookup_rows: Optional[torch.Tensor] = None
        self._rows_taken = 0
        # Calls that did not fit the capacities and took TorchRec's dists.
        self.fallbacks = 0

    @property
    def multi_node(self) -> bool:
        return self._num_nodes > 1

    @property
    def counts_as_weights(self) -> bool:
        """The owner's KJT carries occurrence counts as its weights (LFU
        across nodes without input weights), for LFU tables alone."""
        return self._out_weights and not self._weights

    def setup(
        self, block_sizes: torch.Tensor, dist_types: torch.Tensor, weights: bool
    ) -> bool:
        """Whether this sharding takes the hierarchical dist, the same on
        every rank. The first call builds the operator, collectively, with
        the bucketize's rule of each feature and whether the keys carry
        weights. Ranks whose arguments differ raise the same ValueError. A
        sharding the operator does not apply to takes TorchRec's dists with a
        warning. A configuration it refuses raises ValueError under
        ``"fused"`` and takes TorchRec's dists with a warning under
        ``"auto"``. If a rank cannot build it, every rank warns and takes
        TorchRec's dists. See docs/hier_output_dist.md, sections 6 and 7.2."""
        if self.takes_hier_dist is not None:
            return self.takes_hier_dist
        from dynamicemb_extensions import (
            HierOutputDist,
            HierOutputDistConfig,
            hier_output_dist_unique_id,
        )

        self.takes_hier_dist = False
        self._weights = weights
        self._out_weights = weights or (
            self._lfu and self._arguments["node_dedup"] and self.multi_node
        )
        arguments = dict(
            self._arguments,
            block_sizes=block_sizes,
            dist_types=dist_types,
            weights=self._weights,
            out_weights=self._out_weights,
        )
        config = HierOutputDistConfig(rank=self._rank, **arguments)
        refusal = HierOutputDist.invalid(config)
        # Making an id starts NCCL's listener for the other ranks: none for a
        # configuration the operator refuses.
        unique_id = (
            hier_output_dist_unique_id() if self._rank == 0 and not refusal else None
        )
        every = [None] * arguments["world_size"]
        dist.all_gather_object(
            every, (unique_id, _comparable(arguments)), group=self._pg
        )
        difference = _first_difference([theirs for _, theirs in every])
        if difference is not None:
            raise ValueError(
                f'output_dist="{self._asked}": the ranks of a sharding must build its '
                f"operator from the same arguments, and {difference} (set the "
                "environment variables and table options alike on every rank)"
            )
        why = HierOutputDist.unsupported(
            arguments["world_size"],
            arguments["local_size"],
            len(arguments["feature_tables"]),
            arguments["dim"] * torch.empty((), dtype=arguments["dtype"]).element_size(),
        )
        if why:
            warnings.warn(
                f'output_dist="{self._asked}": {why}; using TorchRec\'s dists'
            )
            return False
        # Every rank has rank 0's arguments, so every rank refuses alike.
        if refusal:
            if self._asked == "fused":
                raise ValueError(f'output_dist="fused": {refusal}')
            warnings.warn(
                f'output_dist="{self._asked}": {refusal}; using TorchRec\'s dists'
            )
            return False
        try:
            self.op = HierOutputDist(every[0][0], config)
        except RuntimeError as err:  # some rank cannot run it, every rank says so
            warnings.warn(
                f'output_dist="{self._asked}": {err}; using TorchRec\'s dists'
            )
            return False
        self.takes_hier_dist = True
        if self.multi_node and not self.op.gin_railed():
            warnings.warn(
                f'output_dist="{self._asked}": NCCL connected GIN to every rank, '
                "not only to the rails it uses, so its queues take memory that grows "
                "with the number of ranks; set NCCL_CROSS_NIC=0 for the job to connect "
                "the rails alone"
            )
        return True

    def owns(self, kjt: KeyedJaggedTensor) -> bool:
        """Whether ``kjt`` is the owner's KJT of a call on this path, as its
        second wait returned it. Its lookup then writes ``lookup_rows``,
        whatever the module's mode (a pipeline may switch it between a call's
        input dist and its lookup)."""
        return kjt in self._plans

    def owner_kjt(self, plan, keys: List[str]) -> KeyedJaggedTensor:
        """The owner's KJT of a call that fits."""
        length_per_key = plan.length_per_key
        kjt = KeyedJaggedTensor(
            keys=keys,
            values=plan.values,
            weights=plan.weights,
            lengths=plan.lengths,
            offsets=plan.offsets,
            stride=plan.num_sources,
            length_per_key=length_per_key,
            offset_per_key=[0] + list(accumulate(length_per_key)),
        )
        self._plans[kjt] = plan
        return kjt

    def fell_back(self, plan) -> None:
        """Counts a call that did not fit, and warns at the first such call
        and then at every power of two of them."""
        self.fallbacks += 1
        if self.fallbacks & (self.fallbacks - 1):
            return
        needs, capacities = plan.needs(), self.op.capacities()
        warnings.warn(
            f'output_dist="{self._asked}": a call does not fit the capacities, so it '
            f"runs on TorchRec's dists ({self.fallbacks} calls so far); raise "
            "output_dist_capacity's "
            + ", ".join(
                f"{name} from {capacities[name]} to at least {needs[name]}"
                for name in plan.overflows()
                if needs[name] > capacities[name]
            )
        )

    def lookup_buffer(
        self, rows: int, cols: int, dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        """``output_alloc`` of the sharding's DynamicEmb tables. Inside
        :meth:`lookup` it gives the next ``rows`` rows of the call's rows, so
        the lookup groups take consecutive slices in feature order. Elsewhere
        it gives None, and a lookup on TorchRec's dists allocates as usual."""
        if self._lookup_rows is None:
            return None
        taken = self._rows_taken
        self._rows_taken += rows
        return self._lookup_rows.narrow(0, taken, rows)

    def lookup(
        self, kjt: KeyedJaggedTensor, look_ups: Sequence[Callable[[], torch.Tensor]]
    ) -> torch.Tensor:
        """The lookup of a call on this path. Each of ``look_ups``, the lookup
        groups in feature order, writes its rows straight into the call's
        rows of ``op.lookup_rows`` (the tables' ``output_alloc``), and their
        outputs come back as those rows, flat, without a copy. The rows are
        valid until the next out kernel. ``op.lookup_rows`` raises if the
        gradient view of the last backward is still alive. See
        docs/hier_output_dist.md, section 11.2."""
        rows = self.op.lookup_rows(self._plans[kjt])
        # The lookup may run on a stream of its own (TorchRec's fused
        # pipeline), on which nobody else records the KJT's use.
        kjt.record_stream(torch.cuda.current_stream())
        self._lookup_rows, self._rows_taken = rows, 0
        try:
            outs = [look_up() for look_up in look_ups]
        finally:
            self._lookup_rows = None
        return _LookupOutput.apply(rows, *outs)


class _LookupOutput(torch.autograd.Function):
    """The lookup groups' outputs, consecutive slices of the call's rows of
    ``lookup_rows`` in feature order, as those rows, flat: a concatenation
    without a copy. The backward hands each group its slice of the
    gradient."""

    @staticmethod
    def forward(ctx, rows: torch.Tensor, *outs: torch.Tensor) -> torch.Tensor:
        ctx.shapes = [out.shape for out in outs]
        return rows.view(-1)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        parts = grad.split([shape.numel() for shape in ctx.shapes])
        return (None, *[part.view(shape) for part, shape in zip(parts, ctx.shapes)])


class _OutputAwaitable(Awaitable[torch.Tensor]):
    """The rows ``fwd_kernel`` wrote, on the stream it ran on. Waiting on them
    from another stream makes that stream wait for the kernel first (and
    records their use there): TorchRec's fused pipeline runs the lookup and
    the output dist on a stream of its own (``emb_lookup_stream="new"``) and
    reads the rows in the dense forward, on the default stream."""

    def __init__(self, out: torch.Tensor) -> None:
        super().__init__()
        self._out = out
        self._stream = torch.cuda.current_stream()
        self._written = torch.cuda.Event()
        self._written.record(self._stream)

    def _wait_impl(self) -> torch.Tensor:
        stream = torch.cuda.current_stream()
        if stream != self._stream:
            stream.wait_event(self._written)
            self._out.record_stream(stream)
        return self._out


class _OutputFunction(torch.autograd.Function):
    """The output dist of a call on this path: ``op.forward`` (``fwd_kernel``)
    in the forward, ``op.backward`` (``bwd_kernel``) in the backward, whose
    gradient for the lookup output is ``lookup_grad``, a view of
    ``lookup_rows``. The backward's maps, which the forward returns, are saved
    tensors: autograd keeps them for a second backward (``retain_graph``) and
    frees them after the last one."""

    @staticmethod
    def forward(ctx, lookup_out: torch.Tensor, plan, op) -> torch.Tensor:
        out, maps = op.forward(plan, lookup_out)
        ctx.save_for_backward(*maps)
        ctx.plan, ctx.op = plan, op
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        # Read at backward time, as TorchRec does: users flip it at runtime.
        lookup_grad = ctx.op.backward(
            ctx.plan, grad_out, ctx.saved_tensors, comm_ops.GRADIENT_DIVISION
        )
        return lookup_grad, None, None


class RwSequenceHierEmbeddingDist(
    BaseEmbeddingDist[SequenceShardingContext, torch.Tensor, torch.Tensor]
):
    """The output dist of a sharding on the hierarchical dist. A call on that
    path moves its rows by its plan (``ctx.plan``). Other calls (eval, or a
    call that did not fit) take TorchRec's ``RwSequenceEmbeddingDist``."""

    def __init__(
        self,
        hier: HierSharding,
        pg: dist.ProcessGroup,
        num_features: int,
        device: Optional[torch.device] = None,
        qcomm_codecs_registry=None,
    ) -> None:
        super().__init__()
        self._hier = hier
        self._torchrec = RwSequenceEmbeddingDist(
            pg, num_features, device, qcomm_codecs_registry=qcomm_codecs_registry
        )

    def forward(
        self,
        local_embs: torch.Tensor,
        sharding_ctx: Optional[DynamicEmbSequenceShardingContext] = None,
    ) -> Awaitable[torch.Tensor]:
        if sharding_ctx.plan is None:
            return self._torchrec(local_embs, sharding_ctx)
        return _OutputAwaitable(
            _OutputFunction.apply(local_embs, sharding_ctx.plan, self._hier.op)
        )


def quiesce(model: torch.nn.Module) -> None:
    """Lets every rank free the windows of ``model``'s hierarchical dists at
    once when the model is dropped. Each operator syncs its device and meets
    the other ranks (``op.quiesce()``), so no kernel of it runs and no credit
    is in flight. Collective: call it on every rank, after the last call and
    before the model is dropped. A call after it undoes it. Without it,
    dropping a sharding waits (up to ``min(timeout, 30 s)``) until every node
    and rail peer drops its own, and keeps the windows until the process
    exits if one does not (docs/hier_output_dist.md, section 4.6)."""
    for module in model.modules():
        for output_dist in getattr(module, "_output_dists", None) or []:
            if (
                isinstance(output_dist, RwSequenceHierEmbeddingDist)
                and output_dist._hier.op is not None
            ):
                output_dist._hier.op.quiesce()
