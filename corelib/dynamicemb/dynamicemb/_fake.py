# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""All fake-mode support consolidated into a single module.

Activated by ``DYNAMICEMB_FAKE_MODE`` env var (``1/true/yes/on``).

Three responsibilities live here:

1. ``is_enabled()`` — env-var reader.
2. ``bootstrap()`` — called from ``dynamicemb/__init__.py`` *before* any
   submodule import. If the real ``dynamicemb_extensions`` C++ extension
   is missing, install a ``sys.modules`` stub so that the package's
   transitive ``from dynamicemb_extensions import X`` lines succeed.
3. ``maybe_get_fake_class()`` — called from ``BatchedDynamicEmbeddingTablesV2.__new__``.
   Returns the fake subclass when fake mode is on, else ``None``. The subclass
   is built lazily because this module is imported before
   ``batched_dynamicemb_tables`` is, and the subclass cannot be defined at
   module top level without triggering that import too early.

The fake subclass keeps host-side bookkeeping identical to the real class
(attribute names, scores, optimizer args), but replaces storage / cache /
admission counter / forward / dump / load with no-op equivalents. Forward
returns a CPU zero tensor of the correct shape wired into autograd so
``loss.backward()`` is a runnable no-op.
"""

from __future__ import annotations

import enum
import os
import sys
import types
from typing import Optional, Type


# ===========================================================================
# 1. Env reader
# ===========================================================================

_TRUE = {"1", "true", "yes", "on"}


def is_enabled() -> bool:
    return os.environ.get("DYNAMICEMB_FAKE_MODE", "").strip().lower() in _TRUE


# ===========================================================================
# 2. dynamicemb_extensions stub  (installed only if real .so is missing)
# ===========================================================================


class _StubDynamicEmbDataType(enum.IntEnum):
    Float32 = 0
    Float16 = 1
    BFloat16 = 2
    Int64 = 3
    UInt64 = 4
    Int32 = 5
    UInt32 = 6
    Int8 = 7
    UInt8 = 8
    Size_t = 9


class _StubEvictStrategy(enum.IntEnum):
    KLru = 0
    KLfu = 1
    KEpochLru = 2
    KEpochLfu = 3
    KCustomized = 4
    LRU = 0
    LFU = 1
    CUSTOMIZED = 4


class _StubScorePolicy(enum.IntEnum):
    ASSIGN = 0
    ACCUMULATE = 1
    GLOBAL_TIMER = 2
    CONST = 3


class _OpaqueStub:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            f"dynamicemb_extensions stub: cannot construct {type(self).__name__} "
            "in fake mode (no GPU backend)."
        )


class _HostVMMTensor(_OpaqueStub):
    pass


class _VMMTensor(_OpaqueStub):
    pass


class _CurandStateContext(_OpaqueStub):
    pass


def _fake_callable(name: str):
    def _raise(*args, **kwargs):
        raise RuntimeError(
            f"dynamicemb_extensions.{name}() called in fake mode "
            "(GPU backend not available)."
        )

    _raise.__name__ = name
    return _raise


def _install_extension_stub() -> types.ModuleType:
    """Idempotently install a ``sys.modules['dynamicemb_extensions']`` stub."""
    existing = sys.modules.get("dynamicemb_extensions")
    if existing is not None and getattr(existing, "__dynamicemb_fake_stub__", False):
        return existing

    module = types.ModuleType("dynamicemb_extensions")
    module.__dynamicemb_fake_stub__ = True
    module.__doc__ = "Fake stub installed by dynamicemb._fake."

    module.DynamicEmbDataType = _StubDynamicEmbDataType
    module.EvictStrategy = _StubEvictStrategy
    module.ScorePolicy = _StubScorePolicy
    module.HostVMMTensor = _HostVMMTensor
    module.VMMTensor = _VMMTensor
    module.CurandStateContext = _CurandStateContext

    def __getattr__(name):  # PEP 562
        return _fake_callable(name)

    module.__getattr__ = __getattr__
    sys.modules["dynamicemb_extensions"] = module
    return module


def bootstrap() -> None:
    """Called from ``dynamicemb/__init__.py`` before any other import.

    If fake mode is on and the real extension is unavailable, install
    the stub so subsequent ``from dynamicemb_extensions import X`` lines
    resolve successfully (the resolved symbol only raises if actually called).
    """
    if not is_enabled():
        return
    try:
        import dynamicemb_extensions  # noqa: F401
    except ImportError:
        _install_extension_stub()


# ===========================================================================
# 3. FakeBatchedDynamicEmbeddingTablesV2 — built lazily on first dispatch
# ===========================================================================

_FAKE_CLASS: Optional[Type] = None


def maybe_get_fake_class() -> Optional[Type]:
    """Return the fake subclass if fake mode is enabled, else ``None``.

    Called by ``BatchedDynamicEmbeddingTablesV2.__new__``; ``None`` tells
    the caller to fall through to ``super().__new__(cls)`` for the real path.
    """
    if not is_enabled():
        return None
    return _build_fake_class()


def _build_fake_class() -> Type:
    """Lazily construct and cache the Fake subclass.

    Deferred because this module is imported during ``dynamicemb/__init__.py``
    bootstrap, before ``batched_dynamicemb_tables`` itself. Defining the
    class at module top level would import the base class too early.
    """
    global _FAKE_CLASS
    if _FAKE_CLASS is not None:
        return _FAKE_CLASS

    # Deferred imports — safe by the time anyone calls a constructor.
    import warnings
    from collections import deque
    from itertools import accumulate
    from typing import Deque, Dict, List, Tuple

    import torch
    from torch import Tensor, nn

    from dynamicemb.batched_dynamicemb_function import PrefetchState
    from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
    from dynamicemb.dynamicemb_config import (
        DynamicEmbPoolingMode,
        DynamicEmbScoreStrategy,
        DynamicEmbTableOptions,
    )
    from dynamicemb.optimizer import (
        BaseDynamicEmbeddingOptimizer,
        EmbOptimType,
    )
    from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
        BoundsCheckMode,
        CounterBasedRegularizationDefinition,
        CowClipDefinition,
        WeightDecayMode,
    )

    _ts_counter = [0]

    def _fake_timestamp() -> int:
        _ts_counter[0] += 1
        return _ts_counter[0]

    class _FakeStorage:
        """Stand-in for DynamicEmbStorage / HybridStorage.

        Implements the public ``Storage`` interface using host-side metadata
        only — sized fields (``embedding_dim``, ``value_dim``, …) come from
        the table options and optimizer captured at construction time. GPU
        primitives (``find`` / ``insert``) raise on call so that any path
        that reaches them under fake mode fails fast rather than silently
        producing garbage.

        Most callers from base class never reach this — fake overrides
        ``forward`` / ``prefetch`` / ``dump`` / ``load`` / ``fill_tables``
        / ``export_keys_values`` / ``incremental_dump`` at the V2 level.
        These methods exist for the few external readers (compute kernel
        metadata queries, planner introspection) that go through the
        ``module.tables`` property.
        """

        training: bool = True

        def __init__(self, options, optimizer, embedding_dtype, max_D):
            self._options = list(options)
            self._optimizer = optimizer
            self._embedding_dtype = embedding_dtype
            self._max_D = max_D
            # Per-table embedding dim and value dim (emb_dim + optimizer state dim).
            self._dims = [o.dim for o in self._options]
            self._value_dims = [
                o.dim + self._optimizer.get_state_dim(o.dim) for o in self._options
            ]
            self._max_value_dim = max(self._value_dims) if self._value_dims else max_D

        # ---- public Storage interface (host-side metadata only) ---------

        def embedding_dtype(self) -> "torch.dtype":
            return self._embedding_dtype

        def embedding_dim(self, table_id: int) -> int:
            return self._dims[table_id]

        def value_dim(self, table_id: int) -> int:
            return self._value_dims[table_id]

        def max_embedding_dim(self) -> int:
            return self._max_D

        def max_value_dim(self) -> int:
            return self._max_value_dim

        def init_optimizer_state(self) -> float:
            return 0.0

        # ---- score / lifecycle no-ops -----------------------------------

        def set_score(self, score) -> None:
            pass

        def dump(self, *args, **kwargs) -> None:
            pass

        def load(self, *args, **kwargs):
            return None

        def fill_tables(self, *args, **kwargs) -> None:
            pass

        def export_keys_values(self, device, batch_size=65536, table_id=0):
            return iter(())

        def incremental_dump(self, table_id, threshold, pg):
            return (
                torch.empty(0, dtype=torch.int64),
                torch.empty(0, 0),
            )

        # ---- GPU primitives — fail fast on call -------------------------

        def find(self, *args, **kwargs):
            raise RuntimeError(
                "_FakeStorage.find() called: fake mode does not implement "
                "GPU embedding lookup. The fake V2 subclass should have "
                "short-circuited this path; please report."
            )

        def insert(self, *args, **kwargs):
            raise RuntimeError(
                "_FakeStorage.insert() called: fake mode does not implement "
                "GPU embedding write. The fake V2 subclass should have "
                "short-circuited this path; please report."
            )

    class _FakeForward(torch.autograd.Function):
        """Zeros forward; routes grad through a sink Parameter."""

        @staticmethod
        def forward(ctx, sink: Tensor, shape, dtype):
            ctx.save_for_backward(sink)
            return torch.zeros(shape, dtype=dtype, device=sink.device)

        @staticmethod
        def backward(ctx, grad_output):
            (sink,) = ctx.saved_tensors
            return torch.zeros_like(sink), None, None

    class FakeBatchedDynamicEmbeddingTablesV2(BatchedDynamicEmbeddingTablesV2):
        """GPU-free counterpart of :class:`BatchedDynamicEmbeddingTablesV2`."""

        def __init__(
            self,
            table_options: List[DynamicEmbTableOptions],
            table_names=None,
            feature_table_map=None,
            use_index_dedup: bool = False,
            prefetch_pipeline: bool = False,
            pooling_mode: DynamicEmbPoolingMode = DynamicEmbPoolingMode.SUM,
            output_dtype: torch.dtype = torch.float32,
            device=None,
            enforce_hbm: bool = False,
            bounds_check_mode: BoundsCheckMode = BoundsCheckMode.WARNING,
            optimizer: EmbOptimType = EmbOptimType.SGD,
            stochastic_rounding: bool = True,
            gradient_clipping: bool = False,
            max_gradient: float = 1.0,
            max_norm: float = 0.0,
            learning_rate: float = 0.01,
            eps: float = 1.0e-8,
            initial_accumulator_value: float = 0.0,
            momentum: float = 0.9,
            weight_decay: float = 0.0,
            weight_decay_mode: WeightDecayMode = WeightDecayMode.NONE,
            eta: float = 0.001,
            beta1: float = 0.9,
            beta2: float = 0.999,
            counter_based_regularization=None,
            cowclip_regularization=None,
            *args,
            **kwargs,
        ) -> None:
            # Skip BatchedDynamicEmbeddingTablesV2.__init__ entirely; replicate
            # only the host-side bookkeeping. nn.Module init still required.
            nn.Module.__init__(self)

            warnings.warn(
                "dynamicemb fake mode active — GPU work is stubbed; "
                "dump/load/forward results are not real.",
                UserWarning,
                stacklevel=2,
            )

            assert len(table_options) >= 1
            table_option = table_options[0]
            for other_option in table_options:
                assert (
                    table_option == other_option
                ), "All tables must match in grouped keys."
            self._dynamicemb_options = table_options
            self.initializer_args = table_option.initializer_args
            self.index_type = table_option.index_type
            self.embedding_dtype = table_option.embedding_dtype
            self.output_dtype = output_dtype
            self.pooling_mode = pooling_mode
            self.use_index_dedup = use_index_dedup
            self._enable_prefetch = prefetch_pipeline
            self.prefetch_stream = None
            self._prefetch_outstanding_keys = torch.tensor(0, dtype=torch.int64)
            self._table_names = table_names
            self.bounds_check_mode_int = bounds_check_mode.value
            self._create_score()
            self._admit_strategy = self._dynamicemb_options[0].admit_strategy
            self._evict_strategy = self._dynamicemb_options[0].evict_strategy.value

            if device is not None:
                try:
                    self.device_id = int(str(device)[-1])
                except ValueError:
                    self.device_id = 0
            else:
                self.device_id = 0
            self._fake_device = torch.device("cpu")

            if table_option.device_id is None:
                for option in self._dynamicemb_options:
                    option.device_id = self.device_id

            self.dims: List[int] = [option.dim for option in self._dynamicemb_options]
            if pooling_mode == DynamicEmbPoolingMode.NONE:
                assert all(
                    d == self.dims[0] for d in self.dims
                ), f"Sequence mode requires uniform embedding dim, got {set(self.dims)}."

            T_ = len(self._dynamicemb_options)
            self.feature_table_map: List[int] = (
                feature_table_map if feature_table_map is not None else list(range(T_))
            )
            T = len(self.feature_table_map)
            assert T_ <= T
            table_has_feature = [False] * T_
            for t in self.feature_table_map:
                table_has_feature[t] = True
            assert all(table_has_feature), "Each table must have at least one feature!"

            feature_dims = [self.dims[t] for t in self.feature_table_map]
            D_offsets = [0] + list(accumulate(feature_dims))
            self.total_D: int = D_offsets[-1]
            self.max_D: int = max(self.dims)

            if self.max_D > min(self.dims):
                self.register_buffer(
                    "D_offsets_t",
                    torch.tensor(
                        D_offsets, device=self._fake_device, dtype=torch.int32
                    ),
                )
            else:
                self.register_buffer("D_offsets_t", None)

            self.feature_num = len(self.feature_table_map)
            self.table_offsets_in_feature: List[int] = []
            old_table_id = -1
            for idx, table_id in enumerate(self.feature_table_map):
                if table_id != old_table_id:
                    self.table_offsets_in_feature.append(idx)
                    old_table_id = table_id
            self.table_offsets_in_feature.append(self.feature_num)
            self.feature_offsets = torch.tensor(
                self.table_offsets_in_feature,
                device=self._fake_device,
                dtype=torch.int64,
            )

            for option in self._dynamicemb_options:
                if option.init_capacity is None:
                    option.init_capacity = option.max_capacity

            # Optimizer construction is host-only — reuse the real path.
            self._optimizer: BaseDynamicEmbeddingOptimizer = self._create_optimizer(
                optimizer,
                stochastic_rounding,
                gradient_clipping,
                max_gradient,
                max_norm,
                learning_rate,
                eps,
                initial_accumulator_value,
                beta1,
                beta2,
                weight_decay,
                eta,
                momentum,
                weight_decay_mode,
                counter_based_regularization,
                cowclip_regularization,
            )
            self._storage_externel = table_option.external_storage is not None

            # Stub backend.
            self._caching = False
            self._cache = None
            self._storage = _FakeStorage(
                self._dynamicemb_options,
                self._optimizer,
                self.embedding_dtype,
                self.max_D,
            )
            self._initializers: List = []
            self._eval_initializers: List = []
            self._admission_counter = None
            self._prefetch_states: Deque[PrefetchState] = deque()

            # Sink parameter — wires fake forward into autograd.
            self._empty_tensor = nn.Parameter(
                torch.empty(
                    10,
                    requires_grad=True,
                    device=self._fake_device,
                    dtype=self.embedding_dtype,
                )
            )

        # ---- forward / prefetch ----------------------------------------

        def forward(
            self,
            indices: Tensor,
            offsets: Tensor,
            per_sample_weights=None,
            feature_requires_grad=None,
            batch_size_per_feature_per_rank=None,
            total_unique_indices=None,
        ) -> Tensor:
            feature_batch_size = offsets.numel() - 1
            assert feature_batch_size > 0, "feature_batch_size must be greater than 0"
            assert (
                feature_batch_size % self.feature_num == 0
            ), "feature_batch_size must be divisible by feature_num"
            batch_size = feature_batch_size // self.feature_num

            if self.pooling_mode == DynamicEmbPoolingMode.NONE:
                shape = (int(indices.numel()), self.max_D)
            else:
                shape = (batch_size, self.total_D)

            return _FakeForward.apply(self._empty_tensor, shape, self.output_dtype)

        def prefetch(self, *args, **kwargs) -> None:
            return None

        # ---- lifecycle / metrics ---------------------------------------

        def flush(self) -> None:
            return None

        def reset_cache_states(self) -> None:
            return None

        def set_record_cache_metrics(self, record: bool) -> None:
            return None

        def split_embedding_weights(self) -> List[Tensor]:
            return [
                torch.empty(
                    (1, 1), device=self._fake_device, dtype=self.embedding_dtype
                )
                for _ in self._dynamicemb_options
            ]

        # ---- scores -----------------------------------------------------

        def get_score(self) -> Dict[str, int]:
            result: Dict[str, int] = {}
            ts = None
            for table_name, option in zip(self._table_names, self._dynamicemb_options):
                if option.score_strategy == DynamicEmbScoreStrategy.TIMESTAMP:
                    if ts is None:
                        ts = _fake_timestamp()
                    result[table_name] = ts
                else:
                    result[table_name] = self._scores[table_name]
            return result

        # ---- dump / load -----------------------------------------------

        def dump(self, save_dir, optim=False, counter=False, table_names=None, pg=None):
            warnings.warn(
                "dynamicemb fake mode: dump() is a no-op (no backend storage).",
                UserWarning,
            )

        def load(self, save_dir, optim=False, counter=False, table_names=None, pg=None):
            warnings.warn(
                "dynamicemb fake mode: load() is a no-op (no backend storage).",
                UserWarning,
            )

        def fill_tables(self, load_factor: float = 0.95, tolerance: float = 1e-5):
            return None

        def export_keys_values(
            self, table_name: str, device, batch_size: int = 65536
        ) -> Tuple[Tensor, Tensor]:
            return (
                torch.empty(0, dtype=torch.int64, device=device),
                torch.empty(0, 0, device=device),
            )

        def incremental_dump(
            self,
            named_thresholds=None,
            pg=None,
        ) -> Tuple[Dict[str, Tuple[Tensor, Tensor]], Dict[str, int]]:
            ret_tensors: Dict[str, Tuple[Tensor, Tensor]] = {}
            ret_scores: Dict[str, int] = {}
            if named_thresholds is None:
                return ret_tensors, ret_scores
            for table_name in named_thresholds:
                if table_name not in self._table_names:
                    continue
                ret_tensors[table_name] = (
                    torch.empty(0, dtype=torch.int64),
                    torch.empty(0, 0),
                )
                idx = self._table_names.index(table_name)
                option = self._dynamicemb_options[idx]
                if option.score_strategy == DynamicEmbScoreStrategy.TIMESTAMP:
                    ret_scores[table_name] = _fake_timestamp()
                else:
                    ret_scores[table_name] = self._scores[table_name]
            return ret_tensors, ret_scores

    _FAKE_CLASS = FakeBatchedDynamicEmbeddingTablesV2
    return _FAKE_CLASS


def __getattr__(name):
    """Expose ``FakeBatchedDynamicEmbeddingTablesV2`` for tests / isinstance.

    Building it triggers the deferred imports, so we only do it on demand.
    """
    if name == "FakeBatchedDynamicEmbeddingTablesV2":
        return _build_fake_class()
    raise AttributeError(f"module 'dynamicemb._fake' has no attribute {name!r}")
