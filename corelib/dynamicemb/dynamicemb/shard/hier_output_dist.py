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
(``output_dist="hierarchical"``) on the C++ operator ``HierOutputDist``
(``src/hier_output_dist``). Keys and rows move over NVLink inside a node and
over the rails (NCCL GIN) between nodes, with one host sync per call. A node
is NCCL's LSA team, the ranks that reach each other's memory over NVLink
(:func:`node_size`, which refuses a team whose GPUs NVLink does not
connect). Design: docs/hier_output_dist.md (section 7.4 for this file).

:class:`HierDist` builds the operator at the first ``input_dist()`` and
holds what the calls share. The operator takes index-deduplicated keys: a
call on it deduplicates its sharding's keys, whatever the sharder's
``use_index_dedup`` (``ShardedDynamicEmbeddingCollection.input_dist``). The
input dist is in input_dist.py (``RwSparseFeaturesDist.hier_forward``). The
lookup writes its rows into the operator's window (:meth:`HierDist.lookup`),
and :class:`RwSequenceHierEmbeddingDist` moves them. A call that does not fit
the capacities raises at its second wait; every eval forward runs on
TorchRec's dists. :func:`quiesce` lets every rank free the operators at once.
"""

import contextlib
import ctypes
import os
import socket
import warnings
import weakref
from itertools import accumulate
from typing import (
    Callable,
    Dict,
    FrozenSet,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import torch
import torch.distributed as dist
from torch.distributed.constants import default_pg_nccl_timeout
from torchrec.distributed import comm_ops
from torchrec.distributed.embedding_sharding import BaseEmbeddingDist
from torchrec.distributed.sharding.rw_sequence_sharding import RwSequenceEmbeddingDist
from torchrec.distributed.sharding.sequence_sharding import SequenceShardingContext
from torchrec.distributed.types import Awaitable
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

from ..dynamicemb_config import OUTPUT_DIST_CAPACITY_NAMES


def _timeout_s(pg: dist.ProcessGroup) -> float:
    """How long a kernel may wait for a peer: ``DYNAMICEMB_HIER_TIMEOUT_S``,
    else the process group's own timeout (how long its NCCL collectives
    wait)."""
    env = os.environ.get("DYNAMICEMB_HIER_TIMEOUT_S")
    if env is not None:
        return float(env)
    try:
        timeout = pg._get_backend(torch.device("cuda")).options._timeout
    except (AttributeError, RuntimeError):  # no NCCL backend options to read
        timeout = default_pg_nccl_timeout
    return timeout.total_seconds()


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


# What node_size read for the ranks of each process group, by their global
# ranks, the same on every rank of the group: (size, None), or (None, why it
# could not).
_NODE_SIZES: Dict[Tuple[int, ...], Tuple[Optional[int], Optional[str]]] = {}


def node_size(pg: Optional[dist.ProcessGroup] = None) -> int:
    """The GPUs of a node of the hierarchical dist on the ranks of ``pg``
    (the default group if None): the ranks of NCCL's LSA team, which reach
    each other's memory over NVLink. With machines of 8 GPUs (Eos's H100
    nodes) it is a machine's GPUs, when the ranks are numbered host by host.
    With multi-node NVLink (GB200 and GB300 NVL72, MNNVL on) it is the job's
    GPUs of the rack (its NVLink clique) when the job runs inside one rack;
    across racks it is what NCCL gives (docs/hier_output_dist.md, section
    4.2). ``LOCAL_WORLD_SIZE`` stays the GPUs of a host, which TorchRec's
    planner and the devices take. See docs/hier_output_dist.md, section 2.

    Collective: every rank of ``pg`` calls it, at the same point. Rank 0 makes
    the id of a short-lived NCCL communicator of the group's ranks, made as
    the dist makes its own, and every rank reads its LSA team there
    (``hier_output_dist_lsa_size``). The ranks then agree that they read the
    same size and that it divides the group (NCCL's team is consecutive
    ranks). The answer is kept for the group's ranks: a later call, on every
    rank, returns it at once. Raises RuntimeError, the same on every rank,
    when a rank cannot read it (a build without the hierarchical dist, a GPU
    before compute capability 8.0, an NCCL error), when the ranks disagree,
    and when NVLink does not connect two GPUs of a node: NCCL's team takes
    the GPUs of a host whatever joins them, PCIe alone too, and the dist
    moves a node's keys and rows over NVLink alone. Each rank reads its
    GPU's NVLinks from NVML (nvidia-ml-py's ``pynvml``) with its team; in a
    node of more than one GPU, a rank that cannot read them refuses the node
    too (:func:`_first_unlinked`). A sharding then takes TorchRec's dists
    (:meth:`HierDist.setup`). But an NCCL error inside the communicator's
    initialization on some ranks only leaves the other ranks waiting there,
    as with any NCCL communicator (docs/hier_output_dist.md, section 19).
    Warns, on every rank, when the team splits the GPUs of a host between
    nodes (ranks not numbered host by host): the dist moves keys and rows
    between those over IB, not NVLink."""
    group = dist.group.WORLD if pg is None else pg
    ranks = tuple(dist.get_process_group_ranks(group))
    if ranks not in _NODE_SIZES:
        _NODE_SIZES[ranks] = _read_node_size(group)
    size, why = _NODE_SIZES[ranks]
    if why is not None:
        raise RuntimeError(why)
    return size


def _read_node_size(
    group: dist.ProcessGroup,
) -> Tuple[Optional[int], Optional[str]]:
    """:func:`node_size`'s read on ``group``, collective: (size, None), or
    (None, why not), the same on every rank: why not also when NVLink does
    not connect the GPUs of a node (:func:`_first_unlinked`). Warns, on
    every rank, when the nodes split a host (:func:`_split_host`)."""
    import dynamicemb_extensions

    rank, world_size = dist.get_rank(group), dist.get_world_size(group)
    probe = getattr(dynamicemb_extensions, "hier_output_dist_lsa_size", None)
    # Each rank tells the others whether it can take part before any of
    # them makes the communicator, which would wait for every rank.
    unique_id, why = None, None
    if probe is None:
        why = (
            "DynamicEmb was built without its hierarchical dist, which needs "
            "NCCL >= 2.30.7"
        )
    elif torch.cuda.get_device_capability() < (8, 0):
        # Its kernels take sm_80 instructions (input_dist/proxy.cu).
        major, minor = torch.cuda.get_device_capability()
        why = (
            f"its GPU is of compute capability {major}.{minor}, and the "
            "hierarchical dist needs 8.0 or newer"
        )
    elif rank == 0:
        try:
            unique_id = dynamicemb_extensions.hier_output_dist_unique_id()
        except RuntimeError as err:
            why = str(err)
    every = [None] * world_size
    dist.all_gather_object(every, (unique_id, why), group=group)
    why = _first_failure([theirs for _, theirs in every])
    if why is not None:
        return None, why
    try:
        size, why = probe(every[0][0], rank, world_size), None
    except RuntimeError as err:
        size, why = None, str(err)
    # With its team, each rank's host and its GPU's NVLinks, which decide
    # whether NVLink connects the GPUs of each node.
    dist.all_gather_object(
        every, (size, why, socket.gethostname(), _gpu_links()), group=group
    )
    why = _first_failure([theirs for _, theirs, _, _ in every])
    if why is not None:
        return None, why
    size, why = _agreed_node_size([size for size, _, _, _ in every], world_size)
    if why is not None:
        return None, why
    hosts = [host for _, _, host, _ in every]
    why = _first_unlinked([links for _, _, _, links in every], hosts, size)
    if why is not None:
        return None, why
    split = _split_host(hosts, size)
    if split is not None:
        warnings.warn(split)
    return size, None


def _first_failure(whys: Sequence[Optional[str]]) -> Optional[str]:
    """Why the first rank that could not read NCCL's LSA team could not
    (``whys``, every rank's reason or None, in rank order), in words, or None
    when every rank could."""
    for rank, why in enumerate(whys):
        if why is not None:
            return f"rank {rank} could not read NCCL's LSA team: {why}"
    return None


def _agreed_node_size(
    sizes: Sequence[int], world_size: int
) -> Tuple[Optional[int], Optional[str]]:
    """The node size of ``world_size`` ranks whose LSA teams have ``sizes``
    ranks, in rank order: (size, None), or (None, why not) when two ranks
    read different sizes or the size does not divide the ranks."""
    for rank, size in enumerate(sizes):
        if size != sizes[0]:
            return None, (
                f"the ranks read different LSA teams from NCCL: rank {rank} one "
                f"of {size} ranks, rank 0 one of {sizes[0]}"
            )
    if sizes[0] < 1 or world_size % sizes[0]:
        return None, (
            f"NCCL's LSA team of {sizes[0]} ranks does not divide the "
            f"{world_size} ranks"
        )
    return sizes[0], None


def _split_host(hosts: Sequence[str], size: int) -> Optional[str]:
    """Whether nodes of ``size`` consecutive ranks split the GPUs of a host
    (``hosts``: each rank's host name, in rank order): the first such host,
    in words, or None."""
    nodes: Dict[str, Set[int]] = {}
    for rank, host in enumerate(hosts):
        nodes.setdefault(host, set()).add(rank // size)
    for host, of_host in nodes.items():
        if len(of_host) > 1:
            return (
                f"the hierarchical dist's node, NCCL's LSA team of {size} ranks, "
                f"splits the GPUs of host {host} between {len(of_host)} nodes, "
                "which exchange keys and rows over IB, not NVLink; ranks "
                "numbered host by host, as many on every host, keep a host in "
                "one node (docs/hier_output_dist.md, section 4.2)"
            )
    return None


class _GpuLinks(NamedTuple):
    """What NVML says of a rank's GPU (:func:`_gpu_links`): its PCI address
    (domain, bus, device); those of the GPUs its active NVLinks reach
    directly (an NVLink bridge, or a mesh); whether an active NVLink reaches
    an NVSwitch; and, once the GPU joined an NVLink fabric (MNNVL), the
    fabric's cluster UUID and the GPU's clique in it, which NCCL reads to
    make one node of several hosts."""

    pci: Tuple[int, int, int]
    gpus: FrozenSet[Tuple[int, int, int]]
    switch: bool
    clique: Optional[Tuple[bytes, int]]


def _gpu_links() -> Tuple[Optional[_GpuLinks], Optional[str]]:
    """This rank's GPU as NVML (nvidia-ml-py's ``pynvml``) gives it:
    (links, None), or (None, why not). It never raises: :func:`_read_node_size`
    calls it between two all-gathers, and a rank that raised would leave the
    others waiting in the second."""
    try:
        import pynvml
    except ImportError:
        return None, "nvidia-ml-py (pynvml) is not installed"
    try:
        uuid = torch.cuda.get_device_properties(torch.cuda.current_device()).uuid
        pynvml.nvmlInit()
    except Exception as err:  # never raises: see the docstring
        return None, str(err)
    try:
        handle = pynvml.nvmlDeviceGetHandleByUUID(f"GPU-{uuid}")
        pci = pynvml.nvmlDeviceGetPciInfo(handle)
        gpus, switch = set(), False
        for link in range(pynvml.NVML_NVLINK_MAX_LINKS):
            try:
                if (
                    pynvml.nvmlDeviceGetNvLinkState(handle, link)
                    != pynvml.NVML_FEATURE_ENABLED
                ):
                    continue
                kind = pynvml.nvmlDeviceGetNvLinkRemoteDeviceType(handle, link)
                if kind == pynvml.NVML_NVLINK_DEVICE_TYPE_GPU:
                    peer = pynvml.nvmlDeviceGetNvLinkRemotePciInfo(handle, link)
                    gpus.add((peer.domain, peer.bus, peer.device))
                elif kind != pynvml.NVML_NVLINK_DEVICE_TYPE_IBMNPU:
                    switch = True  # an NVSwitch, or an end NVML does not name
            except pynvml.NVMLError:  # past the GPU's links, or no NVLink at all
                continue
        links = _GpuLinks(
            (pci.domain, pci.bus, pci.device),
            frozenset(gpus),
            switch,
            _fabric_clique(pynvml, handle),
        )
        return links, None
    except Exception as err:  # never raises: see the docstring
        return None, str(err)
    finally:
        with contextlib.suppress(Exception):
            pynvml.nvmlShutdown()


def _fabric_clique(pynvml, handle) -> Optional[Tuple[bytes, int]]:
    """The NVLink fabric's cluster UUID and the GPU's clique in it, once the
    GPU's registration with the fabric completed, as NCCL reads them for
    MNNVL; None without a fabric, or where nvidia-ml-py or the driver
    cannot tell."""
    info_type = getattr(pynvml, "c_nvmlGpuFabricInfoV_t", None)
    read = getattr(pynvml, "nvmlDeviceGetGpuFabricInfoV", None)
    if info_type is None or read is None:  # an nvidia-ml-py without it
        return None
    info = info_type()
    try:
        read(handle, ctypes.byref(info))
    except pynvml.NVMLError:
        return None
    cluster = bytes(info.clusterUuid)
    if (
        info.state != pynvml.NVML_GPU_FABRIC_STATE_COMPLETED
        or info.status != pynvml.NVML_SUCCESS
        or not any(cluster)
    ):
        return None
    return cluster, int(info.cliqueId)


def _nvlinked(first: _GpuLinks, second: _GpuLinks, one_host: bool) -> bool:
    """Whether NVLink connects two GPUs: one clique of an NVLink fabric
    (MNNVL, across hosts too), or on one host an NVLink between them, or
    from each to an NVSwitch."""
    if first.clique is not None and first.clique == second.clique:
        return True
    return one_host and (
        second.pci in first.gpus
        or first.pci in second.gpus
        or (first.switch and second.switch)
    )


def _pci_name(pci: Tuple[int, int, int]) -> str:
    domain, bus, device = pci
    return f"{domain:04x}:{bus:02x}:{device:02x}.0"


def _first_unlinked(
    every: Sequence[Tuple[Optional[_GpuLinks], Optional[str]]],
    hosts: Sequence[str],
    size: int,
) -> Optional[str]:
    """The first two ranks of a node (``size`` consecutive ranks) whose GPUs
    NVLink does not connect, or the first rank that could not read its GPU's
    NVLinks when nodes have more than one GPU, in words; or None.
    ``every``: each rank's :func:`_gpu_links`, in rank order; ``hosts``:
    each rank's host name."""
    if size == 1:
        return None  # nothing moves inside a node of one GPU
    for rank, (links, why) in enumerate(every):
        if links is None:
            return f"rank {rank} could not read its GPU's NVLinks from NVML: {why}"
    for start in range(0, len(every), size):
        for first in range(start, start + size):
            for second in range(first + 1, start + size):
                one_host = hosts[first] == hosts[second]
                if _nvlinked(every[first][0], every[second][0], one_host):
                    continue
                where = (
                    f"PCI {_pci_name(every[first][0].pci)} and "
                    f"{_pci_name(every[second][0].pci)} of host {hosts[first]}"
                    if one_host
                    else f"of hosts {hosts[first]} and {hosts[second]}, in no "
                    "NVLink clique together"
                )
                return (
                    f"NVLink does not connect the GPUs of ranks {first} and "
                    f"{second} ({where}), which NCCL's LSA team puts in one "
                    "node: the dist moves a node's keys and rows over NVLink "
                    "alone (docs/hier_output_dist.md, section 4.2)"
                )
    return None


# A call's status (int64 words; StatusSlots in
# src/hier_output_dist/common/common.h): the most any rank needs of
# lookup_keys and remote_keys, and the most occurrences of one key at a proxy
# (key_rows: at most local_size, the keys being index-deduplicated); the keys
# this owner looks up; the backward's sub-tasks of heavy rows; the remote
# occurrences this rank deduplicated as proxy; then per feature the keys this
# owner looks up, then the input KJT's keys.
(
    _NEED_LOOKUP_KEYS,
    _NEED_REMOTE_KEYS,
    _NEED_KEY_ROWS,
    _NUM_RECV,
    _HEAVY_SUBS,
    _OCCURRENCES,
    _LENGTH_PER_KEY,
) = range(7)
# Where the tensors the benchmark reads are in the plans' lists (the order of
# FwdPlan::tensors() and BwdPlan::tensors(), in
# src/hier_output_dist/input_dist/plan.cu).
_FWD_OFFSET_PER_BUCKET, _FWD_NUM_CHUNKS, _FWD_NODE_LENGTH_PER_BUCKET = 0, 14, 23
_BWD_NODE_LENGTH_PER_SOURCE = 20


class HierCall:
    """What the input dist of a call on the hierarchical dist leaves its
    output dist: the plans (``fwd_plan``, ``bwd_plan``: the op's lists of
    tensors, ``None`` for those the call has not) and, from the call's
    status, the backward's counts (``counts``: num_recv, heavy_subs). Also
    from the status, for tests and the benchmark: what the call needs of each
    capacity, the most of every rank (``needs``), and the remote occurrences
    this rank deduplicated as proxy (``occurrences``, 0 on one node)."""

    def __init__(
        self,
        fwd_plan: List[Optional[torch.Tensor]],
        bwd_plan: List[Optional[torch.Tensor]],
        counts: List[int],
        needs: Dict[str, int],
        occurrences: int,
    ) -> None:
        self.fwd_plan = fwd_plan
        self.bwd_plan = bwd_plan
        self.counts = counts
        self.needs = needs
        self.occurrences = occurrences

    def record_stream(self, stream: torch.Stream) -> None:
        """The plans' use on ``stream`` (the caching allocator's)."""
        for tensor in self.fwd_plan + self.bwd_plan:
            if tensor is not None:
                tensor.record_stream(stream)

    # Tables of the plans the benchmark reads.
    @property
    def offset_per_bucket(self) -> torch.Tensor:
        """int64 [world_size + 1]: where the requester's rows for each owner
        start."""
        return self.fwd_plan[_FWD_OFFSET_PER_BUCKET]

    @property
    def num_chunks(self) -> Optional[torch.Tensor]:
        """Across nodes, int64 0-dim: the chunks of this rank's stream as
        proxy."""
        return self.fwd_plan[_FWD_NUM_CHUNKS]

    @property
    def node_length_per_bucket(self) -> Optional[torch.Tensor]:
        """Across nodes, int64 [num_nodes]: this proxy's node-unique keys of
        each owner node."""
        return self.fwd_plan[_FWD_NODE_LENGTH_PER_BUCKET]

    @property
    def node_length_per_source(self) -> Optional[torch.Tensor]:
        """Across nodes, int64 [num_nodes]: the node-unique keys this owner
        gets from each source node."""
        return self.bwd_plan[_BWD_NODE_LENGTH_PER_SOURCE]


class DynamicEmbSequenceShardingContext(SequenceShardingContext):
    """TorchRec's sequence sharding context, plus what the input dist of a
    call that took the hierarchical dist leaves its output dist
    (``hier_result``, a :class:`HierCall`). It is None on TorchRec's dists,
    and once the output dist took it.

    Not a dataclass: up to TorchRec 1.6 the parent's ``__init__`` is written by
    hand, and a generated one would accept only this class's own fields.
    """

    def __init__(self, *args, hier_result=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hier_result = hier_result

    def record_stream(self, stream: torch.Stream) -> None:
        super().record_stream(stream)
        if self.hier_result is not None:
            self.hier_result.record_stream(stream)


class HierDist:
    """The hierarchical dist of one row-wise sequence sharding: its operator,
    built at the first ``input_dist()`` (:meth:`setup`), and what its calls
    share. The knobs are read here, from the environment
    (docs/hier_output_dist.md, section 4.4), and the node in :meth:`setup`,
    from NCCL (:func:`node_size`).

    Args:
        pg: the sharding's process group (ranks node-contiguous).
        num_features: features of the sharding.
        embedding_dim: columns of a row of the lookup output.
        output_dtype: the lookup output's dtype.
        capacity: ``output_dist_capacity``.
        asked: the tables' ``output_dist``, ``"hierarchical"`` or
            ``"auto"``: ``"hierarchical"`` raises where ``"auto"`` warns and
            takes TorchRec's dists.
        tables: the table of each feature.
        node_dedup: across nodes a node sends each key once.
        groups: the number of features of each lookup group, in feature order (the
            sharding's grouped configs); each group's rows take a slice of
            the call's rows of their own (None: one group).
        clipping: the tables clip gradients, which needs the sharder's index
            dedup (:meth:`setup`).
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        num_features: int,
        embedding_dim: int,
        output_dtype: torch.dtype,
        capacity: Union[int, Dict[str, int]],
        asked: str = "hierarchical",
        tables: Optional[List[int]] = None,
        node_dedup: bool = True,
        groups: Optional[List[int]] = None,
        clipping: bool = False,
    ) -> None:
        self._pg = pg
        self._rank = dist.get_rank(pg)
        world_size = dist.get_world_size(pg)
        # The nodes, once setup() read the node from NCCL.
        self._num_nodes: Optional[int] = None
        self._asked = asked
        self._clipping = clipping
        self.dtype = output_dtype  # of the lookup output's rows
        if isinstance(capacity, int):
            capacity = dict.fromkeys(OUTPUT_DIST_CAPACITY_NAMES, capacity)
        proxy_mb = float(os.environ.get("DYNAMICEMB_HIER_PROXY_MB", 512))
        # What the operator is built from, but the bucketize's rules and the
        # weighted flag (setup()), and the GPUs per node until setup() reads
        # them from NCCL (node_size). One node needs lookup_keys alone.
        self._arguments = dict(
            world_size=world_size,
            local_size=None,
            feature_tables=list(range(num_features) if tables is None else tables),
            lookup_groups=list(groups or []),
            dim=embedding_dim,
            dtype=output_dtype,
            node_dedup=node_dedup,
            lookup_keys=capacity["lookup_keys"],
            remote_keys=capacity.get("remote_keys", 0),
            proxy_bytes=int(proxy_mb * (1 << 20)),
            timeout_s=_timeout_s(pg),
            out_ctas=int(os.environ.get("DYNAMICEMB_HIER_OUT_CTAS", 0)),
            reduce_ctas=int(os.environ.get("DYNAMICEMB_HIER_REDUCE_CTAS", 8)),
            payload_kb=int(os.environ.get("DYNAMICEMB_HIER_PAYLOAD_KB", 512)),
        )
        self.op = None  # the HierOutputDist, once setup() made it
        # Whether the sharding takes the hierarchical dist; None until setup().
        self.takes_hier_dist: Optional[bool] = None
        # The owner's KJT of each call on this path, as its second wait
        # returned it: the rows its lookup writes, and the row of each key
        # there (owns(), lookup()).
        self._rows: "weakref.WeakKeyDictionary[KeyedJaggedTensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]" = (
            weakref.WeakKeyDictionary()
        )
        # While a lookup runs (lookup()): the call's rows and row map, and how
        # many of them the lookup groups took so far (lookup_buffer()).
        self._lookup_rows: Optional[torch.Tensor] = None
        self._lookup_map: Optional[torch.Tensor] = None
        self._rows_taken = 0

    @property
    def multi_node(self) -> bool:
        """Whether the sharding's ranks span more than one node: known once
        :meth:`setup` read the node from NCCL."""
        if self._num_nodes is None:
            raise RuntimeError("HierDist.multi_node: the node is read by setup()")
        return self._num_nodes > 1

    def setup(
        self,
        dist_types: Sequence[int],
        weighted: bool,
        index_dedup: bool,
    ) -> bool:
        """Whether this sharding takes the hierarchical dist, the same on
        every rank. The first call reads the node from NCCL (its LSA team on
        the sharding's ranks: :func:`node_size`), then builds the operator,
        collectively, with the bucketize's rule of each feature (DistType
        codes) and whether the keys carry weights. ``index_dedup`` is the
        sharder's ``use_index_dedup``: a call on the operator deduplicates the
        sharding's keys either way (without it, a sharding that takes the
        operator warns so here, once), but tables that clip gradients need it.
        Ranks whose arguments differ raise the same ValueError. A sharding
        the operator does not apply to takes TorchRec's dists with a warning.
        A configuration it refuses (a ``continuous`` table among them, or
        clipping without ``index_dedup``) raises ValueError under
        ``"hierarchical"`` and takes TorchRec's dists with a warning under
        ``"auto"``. If a rank cannot read the node or build the operator, or
        NVLink does not connect the GPUs of a node (:func:`node_size`), every
        rank warns and takes TorchRec's dists, but for an NCCL error
        inside a communicator's initialization on some ranks only, which
        leaves the others waiting there. See docs/hier_output_dist.md,
        sections 6, 7.2 and 19."""
        if self.takes_hier_dist is not None:
            return self.takes_hier_dist
        from dynamicemb_extensions import (
            HierOutputDist,
            hier_output_dist_invalid,
            hier_output_dist_unique_id,
            hier_output_dist_unsupported,
        )

        self.takes_hier_dist = False
        # The node: NCCL's LSA team on the sharding's ranks, which the
        # operator's own communicator gets too (its LSA check), and whose GPUs
        # NVLink connects. Every rank reads the same, or every rank raises
        # alike, but for an NCCL error inside the initialization of
        # node_size's communicator on some ranks only, which leaves the
        # others waiting there.
        try:
            local_size = node_size(self._pg)
        except RuntimeError as err:
            warnings.warn(
                f'output_dist="{self._asked}": {err}; using TorchRec\'s dists'
            )
            return False
        self._arguments["local_size"] = local_size
        self._num_nodes = self._arguments["world_size"] // local_size
        arguments = dict(
            self._arguments,
            dist_types=list(dist_types),
            weighted=weighted,
        )
        # The tables clip each row their lookup gets. TorchRec's dists without
        # index dedup send a row per occurrence; a call on the operator would
        # send a GPU's sum of a key's rows.
        refusal = (
            "the sharding's tables clip gradients, and it would add up a GPU's "
            "gradients of a key before the clip, where TorchRec's dists clip "
            "each occurrence's: give the sharder use_index_dedup=True"
            if self._clipping and not index_dedup
            else hier_output_dist_invalid(rank=self._rank, **arguments)
        )
        # Making an id starts NCCL's listener for the other ranks: none for a
        # configuration the operator refuses.
        unique_id = (
            hier_output_dist_unique_id() if self._rank == 0 and not refusal else None
        )
        every = [None] * arguments["world_size"]
        dist.all_gather_object(every, (unique_id, arguments), group=self._pg)
        difference = _first_difference([theirs for _, theirs in every])
        if difference is not None:
            raise ValueError(
                f'output_dist="{self._asked}": the ranks of a sharding must build its '
                f"operator from the same arguments, and {difference} (set the "
                "environment variables and table options alike on every rank)"
            )
        why = hier_output_dist_unsupported(
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
            if self._asked == "hierarchical":
                raise ValueError(f'output_dist="hierarchical": {refusal}')
            warnings.warn(
                f'output_dist="{self._asked}": {refusal}; using TorchRec\'s dists'
            )
            return False
        try:
            self.op = HierOutputDist(every[0][0], rank=self._rank, **arguments)
        except RuntimeError as err:  # some rank cannot run it, every rank says so
            warnings.warn(
                f'output_dist="{self._asked}": {err}; using TorchRec\'s dists'
            )
            return False
        self.takes_hier_dist = True
        if not index_dedup:
            # The training calls deduplicate the keys the sharder did not
            # (ShardedDynamicEmbeddingCollection.input_dist): say so once.
            warnings.warn(
                f'output_dist="{self._asked}": the sharder has '
                "use_index_dedup=False, so the training calls of this sharding "
                "deduplicate its keys themselves, as the hierarchical dist needs; "
                "index dedup changes no output row, and each key's gradient adds "
                "up the same terms, but with 16-bit rows a GPU's sum of them is "
                "rounded once more (docs/hier_output_dist.md, sections 7.4 and 18)"
            )
        return True

    def owns(self, kjt: KeyedJaggedTensor) -> bool:
        """Whether ``kjt`` is the owner's KJT of a call on this path, as its
        second wait returned it. Its lookup then writes ``lookup_rows``,
        whatever the module's mode (a pipeline may switch it between a call's
        input dist and its lookup)."""
        return kjt in self._rows

    def finish(
        self,
        in_dist: Tuple,
        keys: List[str],
        input_features: KeyedJaggedTensor,
        sharding_ctx: DynamicEmbSequenceShardingContext,
    ) -> KeyedJaggedTensor:
        """The second wait of a call: its status on the host, the call's one
        host sync, then the owner's KJT. ``in_dist`` is what ``op.input_dist``
        gave. A call that needs more of a capacity than it holds raises
        RuntimeError, on every rank alike: the needs are the most of every
        rank's. Else ``input_features`` gets the totals
        ``construct_jagged_tensors`` reads, ``sharding_ctx`` what the output
        dist takes, and the call's lookup will write the first num_recv rows
        of ``lookup_rows``, each key's where the row map says."""
        (
            values,
            weights,
            lengths,
            offsets,
            lookup_rows,
            row_map,
            status,
            fwd_plan,
            bwd_plan,
        ) = in_dist
        words = status.cpu().tolist()
        over = [
            (name, self._arguments[name], words[slot])
            for name, slot in (
                ("lookup_keys", _NEED_LOOKUP_KEYS),
                ("remote_keys", _NEED_REMOTE_KEYS),
            )
            if words[slot] > self._arguments[name]
        ]
        if over:
            raise RuntimeError(
                f'output_dist="{self._asked}": a call needs more than the capacities '
                "hold; raise output_dist_capacity's "
                + ", ".join(
                    f"{name} from {capacity} to at least {need}"
                    for name, capacity, need in over
                )
            )
        local_size = self._arguments["local_size"]
        if words[_NEED_KEY_ROWS] > local_size:
            raise RuntimeError(
                f'output_dist="{self._asked}": a proxy got one key '
                f"{words[_NEED_KEY_ROWS]} times from the {local_size} GPUs of its "
                "node; the keys must be index-deduplicated, each table's at most "
                "once per GPU"
            )
        num_features = len(keys)
        length_per_key = words[_LENGTH_PER_KEY : _LENGTH_PER_KEY + num_features]
        input_length_per_key = words[
            _LENGTH_PER_KEY + num_features : _LENGTH_PER_KEY + 2 * num_features
        ]
        input_features._length_per_key = input_length_per_key
        input_features._offset_per_key = [0] + list(accumulate(input_length_per_key))
        num_recv = words[_NUM_RECV]
        sharding_ctx.hier_result = HierCall(
            fwd_plan,
            bwd_plan,
            [num_recv, words[_HEAVY_SUBS]],
            {
                "lookup_keys": words[_NEED_LOOKUP_KEYS],
                "remote_keys": words[_NEED_REMOTE_KEYS],
                "key_rows": words[_NEED_KEY_ROWS],
            },
            words[_OCCURRENCES],
        )
        kjt = KeyedJaggedTensor(
            keys=keys,
            values=values[:num_recv],
            weights=None if weights is None else weights[:num_recv],
            lengths=lengths,
            offsets=offsets,
            stride=local_size + self._num_nodes - 1,
            length_per_key=length_per_key,
            offset_per_key=[0] + list(accumulate(length_per_key)),
        )
        self._rows[kjt] = (
            lookup_rows[:num_recv],
            None if row_map is None else row_map[:num_recv],
        )
        return kjt

    def key_rows(self, kjt: KeyedJaggedTensor) -> Optional[torch.Tensor]:
        """The row of each key of ``kjt`` (an owner's KJT of a call on this
        path, :meth:`owns`) among the call's rows that its lookup writes, int64
        on the GPU; None when key i takes row i. The lookup output's rows in
        the keys' order are those rows of it. No host sync: the group sizes
        come from the KJT's lengths per key, on the host."""
        row_map = self._rows[kjt][1]
        if row_map is None:
            return None
        groups = self._arguments["lookup_groups"] or [len(kjt.keys())]
        per_key = iter(kjt.length_per_key())
        group_keys = [sum(next(per_key) for _ in range(group)) for group in groups]
        # each key's group's first row (fills, no copy from the host)
        starts = torch.cat(
            [
                torch.full((keys,), start, dtype=torch.long, device=row_map.device)
                for start, keys in zip(accumulate([0] + group_keys), group_keys)
            ]
        )
        return row_map.long() + starts

    def lookup_buffer(
        self, rows: int, cols: int, dtype: torch.dtype
    ) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """``output_alloc`` of the sharding's DynamicEmb tables
        (``OutputAlloc``). Inside :meth:`lookup` it gives the next ``rows``
        rows of the call's rows, so the lookup groups take consecutive slices
        in feature order, and the row of each of the group's keys in its
        slice (int32), or None when key i takes row i. Within its slice a
        group's rows lie in the order the rail puts them: its keys of this
        node, then those of each other node (docs/hier_output_dist.md,
        section 11.2). Elsewhere it gives None, and a lookup on TorchRec's
        dists allocates as usual."""
        if self._lookup_rows is None:
            return None
        taken = self._rows_taken
        self._rows_taken += rows
        row_map = self._lookup_map
        return (
            self._lookup_rows.narrow(0, taken, rows),
            None if row_map is None else row_map.narrow(0, taken, rows),
        )

    def in_key_order(self, out: torch.Tensor) -> torch.Tensor:
        """The rows of the lookup group that took the last slice
        (:meth:`lookup_buffer`) in its keys' order: ``out`` itself when key i
        took row i, else a copy."""
        row_map = self._lookup_map
        if row_map is None:
            return out
        return out.view(-1, out.shape[-1]).index_select(
            0, row_map.narrow(0, self._rows_taken - out.shape[0], out.shape[0])
        )

    def lookup(
        self, kjt: KeyedJaggedTensor, look_ups: Sequence[Callable[[], torch.Tensor]]
    ) -> torch.Tensor:
        """The lookup of a call on this path. Each of ``look_ups``, the lookup
        groups in feature order, writes its rows straight into the call's
        rows of the op's ``lookup_rows`` (the tables' ``output_alloc``), and their
        outputs come back as those rows, flat, without a copy. The rows are
        valid until the next out kernel. See docs/hier_output_dist.md,
        section 11.2."""
        rows, row_map = self._rows[kjt]
        # The lookup may run on a stream of its own (TorchRec's fused
        # pipeline), on which nobody else records the use of the KJT and of
        # the row map (which the lookup's backward reads too).
        stream = torch.cuda.current_stream()
        kjt.record_stream(stream)
        if row_map is not None:
            row_map.record_stream(stream)
        self._lookup_rows, self._lookup_map, self._rows_taken = rows, row_map, 0
        try:
            outs = [look_up() for look_up in look_ups]
        finally:
            self._lookup_rows = self._lookup_map = None
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
    records their use there), as the awaitables of TorchRec's own dists do
    (their NCCL work's wait): TorchRec's fused pipeline runs the lookup and
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
    """The output dist of a call on this path: ``op.output_dist_fwd``
    (``fwd_kernel``) with the forward's plan in the forward,
    ``op.output_dist_bwd`` (``bwd_kernel``) with the backward's plan and
    counts in the backward, whose gradient for the lookup output is
    ``lookup_grad``, a view of ``lookup_rows`` valid until the next lookup.
    The backward's plan is saved tensors: autograd keeps them for a second
    backward (``retain_graph``) and frees them after the last one."""

    @staticmethod
    def forward(ctx, lookup_out: torch.Tensor, call: HierCall, op) -> torch.Tensor:
        # The out kernels may run on a stream of their own (TorchRec's fused
        # pipeline), on which nobody else records the plans' use; the
        # backward runs on this stream too.
        call.record_stream(torch.cuda.current_stream())
        out = op.output_dist_fwd(call.fwd_plan, lookup_out)
        ctx.save_for_backward(*call.bwd_plan)
        ctx.counts, ctx.op = call.counts, op
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        # Read at backward time, as TorchRec does: users flip it at runtime.
        lookup_grad = ctx.op.output_dist_bwd(
            list(ctx.saved_tensors), ctx.counts, grad_out, comm_ops.GRADIENT_DIVISION
        )
        return lookup_grad, None, None


class RwSequenceHierEmbeddingDist(
    BaseEmbeddingDist[SequenceShardingContext, torch.Tensor, torch.Tensor]
):
    """The output dist of a sharding on the hierarchical dist. A call on that
    path moves its rows by the plans its input dist made
    (``sharding_ctx.hier_result``), on the current stream. Eval calls take
    TorchRec's ``RwSequenceEmbeddingDist``."""

    def __init__(
        self,
        hier: HierDist,
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
        call = sharding_ctx.hier_result
        if call is None:
            return self._torchrec(local_embs, sharding_ctx)
        # The output dist takes the plans: the forward's go once its kernel is
        # queued, the backward's once the last backward is done.
        sharding_ctx.hier_result = None
        return _OutputAwaitable(_OutputFunction.apply(local_embs, call, self._hier.op))


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
