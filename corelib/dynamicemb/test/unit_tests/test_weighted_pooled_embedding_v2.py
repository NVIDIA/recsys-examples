# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Weighted pooled (SUM) tests at the BatchedDynamicEmbeddingTablesV2 level.

Every numeric test compares against an unsharded, weighted torchrec
EmbeddingBagCollection driven by the same KeyedJaggedTensor, so the slot layout
and the per-sample-weight semantics are checked against the framework that feeds
dynamicemb in production rather than against a formula derived by reading the
implementation.

Setup follows the same shape as ``init_embedding_tables`` in
test_batched_dynamic_embedding_tables_v2.py: the reference table is initialized
first and every one of its rows is inserted into dynamicemb, after which the two
sides run their own optimizers and stay in step on their own -- nothing here
re-syncs them or assumes a row still holds its initializer value.  The dynamicemb
tables are given twice the reference's capacity so eviction can never perturb the
comparison, and input keys are drawn from the reference's row range.

Each test runs the same batch twice.  The second pass exercises keys that are
already resident and already updated, which is where a mismatch in how repeated
keys are reduced would show up.  A final eval-mode forward then checks the
separate inference path against the same reference.
"""

import pytest
import torch
import torchrec
from dynamicemb import (
    DynamicEmbCheckMode,
    DynamicEmbPoolingMode,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
)
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from dynamicemb.key_value_table import DynamicEmbStorage
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from torchrec.distributed.types import BoundsCheckMode
from torchrec.modules.embedding_configs import PoolingType

# Rows in the reference table, and therefore the exclusive upper bound on input
# keys.  The reference is dense, so tests needing large or hashed keys cannot use
# it.
REF_NUM_EMBEDDINGS = 512
# Headroom in the dynamicemb tables so no key is ever evicted mid-test.
DEMB_CAPACITY = 2 * REF_NUM_EMBEDDINGS

# Both sides start from identical rows, so they should agree to roughly the same
# tolerance the module-level tests in test_batched_dynamic_embedding_tables_v2.py
# use against fbgemm.
RTOL, ATOL = 1e-6, 1e-6

LR = 0.5


def _table_name(i: int) -> str:
    return f"t{i}"


def _feature_name(i: int) -> str:
    return f"cate_{i}"


@pytest.fixture
def current_device():
    assert torch.cuda.is_available()
    return torch.cuda.current_device()


def _make_v2(dims, pooling_mode=DynamicEmbPoolingMode.SUM, lr=LR, device=None):
    device = device if device is not None else torch.cuda.current_device()
    opts = [
        DynamicEmbTableOptions(
            dim=d,
            max_capacity=DEMB_CAPACITY,
            init_capacity=DEMB_CAPACITY,
            index_type=torch.int64,
            embedding_dtype=torch.float32,
            device_id=device,
            bucket_capacity=128,
            safe_check_mode=DynamicEmbCheckMode.IGNORE,
            local_hbm_for_values=1024**3,
            score_strategy=DynamicEmbScoreStrategy.STEP,
        )
        for d in dims
    ]
    return BatchedDynamicEmbeddingTablesV2(
        table_options=opts,
        table_names=[_table_name(i) for i in range(len(opts))],
        feature_table_map=list(range(len(opts))),
        pooling_mode=pooling_mode,
        optimizer=EmbOptimType.SGD,
        learning_rate=lr,
        stochastic_rounding=False,
        bounds_check_mode=BoundsCheckMode.NONE,
    )


def _make_reference(dims, dev):
    """Unsharded weighted EmbeddingBagCollection mirroring the V2 tables."""
    ebc = torchrec.EmbeddingBagCollection(
        tables=[
            torchrec.EmbeddingBagConfig(
                name=_table_name(i),
                embedding_dim=d,
                num_embeddings=REF_NUM_EMBEDDINGS,
                feature_names=[_feature_name(i)],
                pooling=PoolingType.SUM,
            )
            for i, d in enumerate(dims)
        ],
        is_weighted=True,
        device=dev,
    )
    with torch.no_grad():
        for bag in ebc.embedding_bags.values():
            bag.weight.uniform_(0, 1)
    return ebc


def _seed_dynamicemb_from_reference(module, ref):
    """Insert every reference row into the dynamicemb tables.

    Mirrors ``init_embedding_tables`` in test_batched_dynamic_embedding_tables_v2:
    the reference is the source of the initial values, the optimizer state region
    is filled with the storage's own initial state, and the whole table is
    written in one shot so the two sides start identical.
    """
    storage = module.tables
    assert isinstance(storage, DynamicEmbStorage), type(storage)
    optimizer = module.optimizer
    max_emb_dim = storage.max_embedding_dim()
    max_value_dim = storage.max_value_dim()

    for table_idx, name in enumerate(module.table_names):
        w = ref.embedding_bags[name].weight.detach()
        num_emb, emb_dim = w.size(0), w.size(1)
        opt_state_dim = optimizer.get_state_dim(emb_dim)

        values = torch.zeros(num_emb, max_value_dim, dtype=w.dtype, device=w.device)
        values[:, :emb_dim] = w
        if opt_state_dim > 0:
            optimizer.reset_optimizer_states(
                values[:, max_emb_dim : max_emb_dim + opt_state_dim],
                emb_dims=emb_dim,
            )

        indices = torch.arange(num_emb, device=w.device, dtype=torch.int64)
        table_ids = torch.full(
            (num_emb,), table_idx, dtype=torch.int64, device=w.device
        )
        storage.set_score(1)
        storage.insert(indices, table_ids, values)


def _random_kjt(feature_num, batch_size, max_bag, dev, seed):
    """Feature-major KJT with keys inside the reference table's row range.

    Bags are small relative to the key range on purpose: with F*B bags drawn from
    REF_NUM_EMBEDDINGS rows the batch contains repeated keys both inside a single
    bag and across bags, which is what exercises the dedup + weighted reduce.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    lengths = torch.randint(
        0, max_bag + 1, (feature_num * batch_size,), generator=gen, dtype=torch.int64
    )
    total = int(lengths.sum())
    indices = torch.randint(
        0, REF_NUM_EMBEDDINGS, (total,), generator=gen, dtype=torch.int64
    )
    weights = torch.rand(total, generator=gen, dtype=torch.float32) * 1.5 + 0.5
    return torchrec.KeyedJaggedTensor(
        keys=[_feature_name(f) for f in range(feature_num)],
        values=indices.to(dev),
        lengths=lengths.to(dev),
        weights=weights.to(dev),
    )


def _assert_pooled_close(got, exp, dims):
    """got is [B, total_D] with features laid out in D_offsets order.

    The reference side is addressed by feature name, so a column-order mistake in
    dynamicemb cannot be cancelled out by the same mistake in the reference.
    """
    col = 0
    for f, d in enumerate(dims):
        torch.testing.assert_close(
            got[:, col : col + d].float(),
            exp[_feature_name(f)].float(),
            rtol=RTOL,
            atol=ATOL,
            msg=lambda m, f=f, lo=col, hi=col + d: (
                f"feature {f} (columns {lo}:{hi}): {m}"
            ),
        )
        col += d


def _assert_rows_close(module, ref, dev):
    """Every row dynamicemb holds must equal the reference's row for that key."""
    for name in module.table_names:
        keys, vals = module.export_keys_values(name, dev)
        assert (
            keys.numel() == REF_NUM_EMBEDDINGS
        ), f"table {name}: expected {REF_NUM_EMBEDDINGS} rows, got {keys.numel()}"
        w = ref.embedding_bags[name].weight.detach()
        torch.testing.assert_close(
            vals[:, : w.size(1)].float(),
            w[keys.long()].float(),
            rtol=RTOL,
            atol=ATOL,
            msg=lambda m, name=name: f"table {name}: {m}",
        )


def _step_both(module, ref, opt, kjt):
    """One forward + backward on both sides, driven by the same upstream grad."""
    got = module(kjt.values(), kjt.offsets().long(), pooling_weights=kjt.weights())
    exp = ref(kjt)

    # A random upstream gradient rather than a constant, so a mis-permuted row
    # cannot coincidentally produce the right update.
    grad = torch.rand_like(got)

    opt.zero_grad()
    (got * grad).sum().backward()  # dynamicemb's fused optimizer steps here
    (exp.values() * grad).sum().backward()
    opt.step()
    torch.cuda.synchronize()
    return got, exp


def _run_against_reference(dims, feature_num, dev, seed, batch_size=64, max_bag=8):
    module = _make_v2(dims, device=dev.index)
    ref = _make_reference(dims, dev)
    _seed_dynamicemb_from_reference(module, ref)
    opt = torch.optim.SGD(ref.parameters(), lr=LR)

    kjt = _random_kjt(feature_num, batch_size, max_bag, dev, seed)

    # Two passes over the same batch: the first inserts nothing (every key is
    # already resident) and the second sees rows both sides have already updated.
    for pass_idx in range(2):
        got, exp = _step_both(module, ref, opt, kjt)
        assert got.shape == (batch_size, sum(dims)), f"pass {pass_idx}: {got.shape}"
        _assert_pooled_close(got, exp, dims)
        _assert_rows_close(module, ref, dev)

    # Inference goes down a separate forward (dynamicemb_eval_forward) that has
    # to weight the pooling the same way the training path does.
    module.eval()
    ref.eval()
    with torch.no_grad():
        got = module(kjt.values(), kjt.offsets().long(), pooling_weights=kjt.weights())
        exp = ref(kjt)
    assert got.shape == (batch_size, sum(dims))
    _assert_pooled_close(got, exp, dims)


def test_weighted_sum_matches_torchrec(current_device):
    dev = torch.device(f"cuda:{current_device}")
    _run_against_reference(dims=[8], feature_num=1, dev=dev, seed=0)


def test_weighted_sum_mixed_D_matches_torchrec(current_device):
    dev = torch.device(f"cuda:{current_device}")
    # 8 + 4 = 12 columns; exercises the multi-dim gather / reduce path.
    _run_against_reference(dims=[8, 4], feature_num=2, dev=dev, seed=1)


# copy_multi_to_one falls back from the vec4 warp-per-ev kernel to the
# cta-per-ev one as soon as ev_size or total_D is not a multiple of 4,
# and the reduce kernels split the same way. The two tests above only ever hit
# the vec4 side, so cover the other one in both the uniform and multi-dim shape.
def test_weighted_sum_unaligned_dim_matches_torchrec(current_device):
    dev = torch.device(f"cuda:{current_device}")
    _run_against_reference(dims=[9], feature_num=1, dev=dev, seed=2)


def test_weighted_sum_unaligned_mixed_D_matches_torchrec(current_device):
    dev = torch.device(f"cuda:{current_device}")
    _run_against_reference(dims=[9, 6], feature_num=2, dev=dev, seed=3)


def test_weighted_sum_misaligned_feature_start_matches_torchrec(current_device):
    """A feature that is 4 wide but does not start on a 4-column boundary.

    max_D (4) and total_D (8) are both multiples of 4, so the coarse checks let
    this reach the vec4 kernels, but D_offsets is [0, 2, 6, 8]: feature 1
    occupies columns 2..5. Its width being exactly 4 means Vec4T takes its
    n == 4 path -- a real 16-byte access -- at dst_ptr + b*8 + 2, which is 8
    bytes off a 16-byte boundary. Only every feature dim being a multiple of 4
    rules that out, which is what layout.feature_dims_vec4 reports.
    """
    dev = torch.device(f"cuda:{current_device}")
    _run_against_reference(dims=[2, 4, 2], feature_num=3, dev=dev, seed=4)


def test_weighted_errors(current_device):
    device = torch.cuda.current_device()
    indices = torch.tensor([5, 12], dtype=torch.int64, device=device)
    offsets = torch.tensor([0, 1, 2], dtype=torch.int64, device=device)
    weights = torch.tensor([0.5, 1.5], dtype=torch.float32, device=device)

    # weights + MEAN -> ValueError
    mean_mod = _make_v2([8], pooling_mode=DynamicEmbPoolingMode.MEAN, device=device)
    with pytest.raises(ValueError, match="pooling_mode=SUM"):
        mean_mod(indices, offsets, pooling_weights=weights)

    # non-fp32 weights -> ValueError
    mod = _make_v2([8], device=device)
    with pytest.raises(ValueError, match="must be float32"):
        mod(indices, offsets, pooling_weights=weights.half())

    # numel mismatch -> ValueError
    with pytest.raises(ValueError, match="must equal indices.numel"):
        mod(indices, offsets, pooling_weights=weights[:1])

    # pooling_weights + frequency_counters -> ValueError: both come from the
    # single KJT weights channel, so a caller must pick one.
    counters = torch.tensor([1.0, 2.0], dtype=torch.float32, device=device)
    with pytest.raises(ValueError, match="mutually exclusive"):
        mod(indices, offsets, frequency_counters=counters, pooling_weights=weights)

    # differentiable weights -> ValueError. The backward returns no gradient for
    # them, so accepting these would leave the producing module training against
    # a silent zero.
    with pytest.raises(ValueError, match="requires_grad"):
        mod(indices, offsets, pooling_weights=weights.clone().requires_grad_())

    # ... but with grad mode off there is no backward to lose a gradient to, so
    # the same tensor is accepted. This is what keeps eval usable.
    with torch.no_grad():
        mod(indices, offsets, pooling_weights=weights.clone().requires_grad_())
