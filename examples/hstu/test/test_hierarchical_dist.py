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
"""DynamicEmb's hierarchical dist (``output_dist="hierarchical"``) training the
HSTU ranking model, end to end.

The model's model-parallel tables are DynamicEmb tables, on the hierarchical
dist or on TorchRec's dists. Both models load one checkpoint, from a model that
trained over the batches once, so it holds every key of them. Both train on the
same batches under one of the example's JaggedMegatron pipelines (``none``,
``native`` and ``prefetch``; the last two also with the balanced batch
shuffler). Each table has 2 GiB of HBM, which holds all its keys, also as the
cache of the prefetch pipeline. So no key is inserted (a random initializer
gives a new key a value that depends on the order of the keys), and none is
evicted (which keys a full cache evicts depends on their order too). Then:

* the first step's loss and logits are the same bits: the hierarchical dist's
  forward is a byte copy of the owner's rows (DynamicEmb's
  docs/hier_output_dist.md, section 18);
* each rank's tables hold the same keys;
* with fp32 (the PyTorch backend) on one node (the hierarchical dist's:
  NCCL's LSA team, the GPUs that reach each other over NVLink, which with
  multi-node NVLink spans hosts), every step's loss and logits,
  and the tables' rows, are within torch.allclose's defaults: the two dists
  add the same gradient terms per key, in another order. These runs have one
  table, ``item``. Where training is not the same bits from run to run on
  either dist, no tolerance tells the dists apart, so the test does not
  compare there: with a table of two features (``context``, for the two
  contextual features) under the plain loop, and on more than one node, two
  runs on TorchRec's dists already differ (by a few hundredths of the bound
  per step, which the steps after it can grow past the bound). There
  DynamicEmb's test/unit_tests/test_hier_dist.py checks the dist's sums bit
  for bit.
* bf16 (CUTLASS, the example's default) runs with the ``context`` table, and
  checks the parts that are the same bits: a last-bit difference of a row can
  flip a bf16 rounding a step later.

A call that needs more than ``output_dist_capacity`` raises RuntimeError on
every rank, under each pipeline.

It needs DynamicEmb built against NCCL >= 2.30.7 (else it skips) and 2 or more
GPUs (on one GPU the sharding takes TorchRec's dists: it skips). Run it from
``examples/hstu``, one process per GPU::

    PYTHONPATH=${PYTHONPATH}:$(realpath ../) torchrun --nproc_per_node 8 \\
        -m pytest -svv test/test_hierarchical_dist.py

Across nodes, run one torchrun per node (``--nnodes``, ``--node_rank`` and a
rendezvous endpoint) from the same folder on a shared filesystem: the
checkpoint goes to the working directory, and every rank reads it.
"""

import gc
import os
import shutil
from typing import Dict, List, Tuple

import commons.checkpoint as checkpoint
import commons.utils as init
import dynamicemb_extensions
import pytest
import torch
import torch.distributed as dist
from commons.checkpoint import get_unwrapped_module
from commons.distributed.batch_shuffler import IdentityBalancedBatchShuffler
from commons.pipeline.train_pipeline import (
    JaggedMegatronPrefetchTrainPipelineSparseDist,
    JaggedMegatronTrainNonePipeline,
    JaggedMegatronTrainPipelineSparseDist,
)
from commons.utils.distributed_utils import collective_assert
from configs import KernelBackend
from dynamicemb.shard import ShardedDynamicEmbeddingCollection, node_size, quiesce
from dynamicemb.shard.rw_sharding import RwSequenceDynamicEmbeddingSharding
from test_utils import create_model
from utils.hstu_batch_balancer import HASTUBalancedBatchShuffler

pytestmark = pytest.mark.skipif(
    not hasattr(dynamicemb_extensions, "HierOutputDist"),
    reason="DynamicEmb was built without NCCL >= 2.30.7: no hierarchical dist",
)

CONTEXTUAL = ["user0", "user1"]
# Each table's HBM: it holds all the keys of the batches, so nothing is evicted.
HBM = 2 << 30
BACKENDS = {
    torch.float32: KernelBackend.PYTORCH,  # only the PyTorch backend runs fp32
    torch.bfloat16: KernelBackend.CUTLASS,
}
PIPELINES = {
    "none": JaggedMegatronTrainNonePipeline,
    "native": JaggedMegatronTrainPipelineSparseDist,
    "prefetch": JaggedMegatronPrefetchTrainPipelineSparseDist,
}
CHECKPOINT = "./hier_dist_checkpoint"


def one_node() -> bool:
    """Whether every rank of the run is on one node of the hierarchical
    dist: NCCL's LSA team (node_size, collective), not LOCAL_WORLD_SIZE, the
    GPUs of a host."""
    return node_size(dist.group.WORLD) == dist.get_world_size()


def start() -> None:
    # A kernel that waits for a peer traps after two minutes rather than at
    # the process group's timeout (read when a model is sharded).
    os.environ.setdefault("DYNAMICEMB_HIER_TIMEOUT_S", "120")
    init.initialize_distributed()
    init.initialize_model_parallel(1)
    if dist.get_world_size() < 2:
        init.destroy_global_state()
        pytest.skip("on one GPU the sharding takes TorchRec's dists")


def contextual(dtype: torch.dtype) -> List[str]:
    """The contextual features of a run: with fp32 none (one table)."""
    return CONTEXTUAL if dtype == torch.bfloat16 else []


def table_names(dtype: torch.dtype) -> Tuple[str, ...]:
    """The model-parallel tables of a run, one row-wise sharding."""
    return ("item", "context") if contextual(dtype) else ("item",)


def make(output_dist: str, pipeline_type: str, dtype: torch.dtype, **kwargs):
    """The ranking model on ``output_dist``, its dense optimizer and its ten
    batches, the same for every model (one seed). The last batch is short,
    and may be empty on a rank."""
    return create_model(
        task_type="ranking",
        contextual_feature_names=contextual(dtype),
        max_num_candidates=10,
        optimizer_type_str="sgd",
        dtype=dtype,
        kernel_backend=BACKENDS[dtype],
        pipeline_type=pipeline_type,
        output_dist=output_dist,
        replicate_batches=False,
        seed=1234,
        global_hbm_for_values=HBM,
        **kwargs,
    )


def train(
    model, optimizer, pipeline_type: str, batches, balanced: bool = False
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Each step's loss and logits, the batches through the pipeline."""
    shuffler = (
        # create_model's attention heads
        HASTUBalancedBatchShuffler(num_heads=4, head_dim=64)
        if balanced
        else IdentityBalancedBatchShuffler()
    )
    pipeline = PIPELINES[pipeline_type](
        model,
        optimizer,
        device=torch.device("cuda", torch.cuda.current_device()),
        batch_shuffler=shuffler,
    )
    batch_iter = iter(batches)
    outs = []
    for _ in batches:
        loss, _, (_, logits, _, _) = pipeline.progress(batch_iter)
        outs.append((loss, logits))
    return outs


def takes_hier_dist(model) -> List[bool]:
    """Whether each row-wise sharding of ``model`` took the hierarchical
    dist."""
    return [
        sharding._hier is not None and bool(sharding._hier.takes_hier_dist)
        for ec in model.modules()
        if isinstance(ec, ShardedDynamicEmbeddingCollection)
        for sharding in ec._sharding_type_to_sharding.values()
        if isinstance(sharding, RwSequenceDynamicEmbeddingSharding)
    ]


def tables(model, names) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """This rank's keys of each table, in order, and their rows."""
    collection = get_unwrapped_module(model)._embedding_collection
    out = {}
    for name in names:
        keys, values = collection.export_local_embedding(name)
        keys, values = torch.from_numpy(keys), torch.from_numpy(values)
        order = torch.argsort(keys)
        out[name] = (keys[order], values[order])
    return out


def allclose_share(got: torch.Tensor, ref: torch.Tensor) -> float:
    """The largest share of torch.allclose's bound (its defaults,
    1e-8 + 1e-5 * |ref|) a difference takes: at most 1 where
    torch.allclose(got, ref) holds."""
    if got.numel() == 0:
        return 0.0
    ratio = (got - ref).abs() / (1e-8 + 1e-5 * ref.abs())
    # A NaN fails torch.allclose, but Python's max() may drop it: count it as inf.
    return torch.nan_to_num(ratio, nan=float("inf")).max().item()


def warm_up(dtype: torch.dtype):
    """A model on TorchRec's dists trains once over the batches, which
    inserts every key of them, and saves itself to CHECKPOINT. Its
    batches."""
    model, optimizer, batches = make("torchrec", "none", dtype)
    train(model, optimizer, "none", batches)
    if dist.get_rank() == 0:
        shutil.rmtree(CHECKPOINT, ignore_errors=True)
        os.makedirs(CHECKPOINT)
    dist.barrier(device_ids=[torch.cuda.current_device()])
    checkpoint.save(CHECKPOINT, model, dense_optimizer=optimizer)
    dist.barrier(device_ids=[torch.cuda.current_device()])
    del model, optimizer
    gc.collect()
    return batches


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "pipeline_type,balanced",
    [
        ("none", False),
        ("native", False),
        ("prefetch", False),
        ("native", True),
        ("prefetch", True),
    ],
)
def test_hierarchical_dist_vs_torchrec(
    pipeline_type: str, balanced: bool, dtype: torch.dtype
):
    start()
    batches = warm_up(dtype)
    runs = {}
    for output_dist in ("torchrec", "hierarchical"):
        model, optimizer, _ = make(output_dist, pipeline_type, dtype)
        checkpoint.load(CHECKPOINT, model, dense_optimizer=optimizer)
        outs = train(model, optimizer, pipeline_type, batches, balanced)
        runs[output_dist] = (
            outs,
            tables(model, table_names(dtype)),
            takes_hier_dist(model),
        )
        quiesce(model)
        del model, optimizer
        gc.collect()
    ref, ref_tables, ref_took = runs["torchrec"]
    got, got_tables, got_took = runs["hierarchical"]
    collective_assert(
        got_took == [True] and ref_took == [False],
        f"took the hierarchical dist: {got_took}, on TorchRec's dists {ref_took}",
    )
    (loss, logits), (ref_loss, ref_logits) = got[0], ref[0]
    collective_assert(
        torch.equal(loss, ref_loss) and torch.equal(logits, ref_logits),
        "step 0: the loss or logits differ from TorchRec's dists",
    )
    for name in table_names(dtype):
        collective_assert(
            torch.equal(got_tables[name][0], ref_tables[name][0]),
            f"{name}: the keys differ from TorchRec's dists",
        )
    if dtype == torch.float32 and one_node():
        shares = {
            "loss and logits": max(
                max(allclose_share(loss, ref_loss), allclose_share(logits, ref_logits))
                for (loss, logits), (ref_loss, ref_logits) in zip(got, ref)
            ),
            "rows": max(
                allclose_share(got_tables[name][1], ref_tables[name][1])
                for name in table_names(dtype)
            ),
        }
        largest = torch.tensor(list(shares.values()), device="cuda")
        dist.all_reduce(largest, op=dist.ReduceOp.MAX)
        if dist.get_rank() == 0:
            print(
                "fp32, the largest share of torch.allclose's bound:",
                dict(zip(shares, largest.tolist())),
                flush=True,
            )
        for what, share in shares.items():
            collective_assert(
                share <= 1,
                f"fp32: the {what} differ from TorchRec's dists beyond "
                f"torch.allclose's defaults (a share of {share:.3g} of its bound)",
            )
    dist.barrier(device_ids=[torch.cuda.current_device()])
    if dist.get_rank() == 0:
        shutil.rmtree(CHECKPOINT, ignore_errors=True)
    init.destroy_global_state()


@pytest.mark.parametrize("pipeline_type", ["none", "native", "prefetch"])
def test_hierarchical_dist_over_capacity(pipeline_type: str):
    """A capacity the first call does not fit: the call raises RuntimeError
    at its second wait, on every rank, naming the capacity to raise. Nothing
    hangs, and the model then goes."""
    start()
    model, optimizer, batches = make(
        "hierarchical", pipeline_type, torch.bfloat16, output_dist_capacity=1024
    )
    pipeline = PIPELINES[pipeline_type](
        model, optimizer, device=torch.device("cuda", torch.cuda.current_device())
    )
    try:
        pipeline.progress(iter(batches))
        raised = ""
    except RuntimeError as error:
        raised = str(error)
    collective_assert(
        "output_dist_capacity" in raised and "lookup_keys from 1024" in raised,
        f"[rank {dist.get_rank()}] raised {raised!r}",
    )
    torch.cuda.synchronize()
    dist.barrier(device_ids=[torch.cuda.current_device()])
    del pipeline, model, optimizer
    gc.collect()
    init.destroy_global_state()
