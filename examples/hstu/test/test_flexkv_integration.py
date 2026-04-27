import gc
import os
import time
from uuid import uuid4

import pytest
import torch
import numpy as np
from types import SimpleNamespace
from configs import get_inference_hstu_config, get_kvcache_config
from modules.async_kvcache_manager import (
    AsyncHSTUKVCacheManager,
    FlexKVClientAdapter,
    FlexKVStorageManager,
    KVIndexMeta,
    SecondaryErrorCode,
    SecondaryTaskHandle,
    SecondaryTaskStatus,
    SecondaryWaitResult,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _new_ipc_endpoint(prefix: str) -> str:
    return f"ipc:///tmp/{prefix}_{os.getpid()}_{uuid4().hex}"


def _cleanup_cuda_and_gc() -> None:
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    gc.collect()


def _cancel_secondary_tasks(secondary_mgr) -> None:
    tasks = getattr(secondary_mgr, "_tasks", None)
    client = getattr(secondary_mgr, "_client", None)
    if not isinstance(tasks, dict) or client is None:
        return
    for state in list(tasks.values()):
        task_ids = [int(x) for x in state.get("task_ids", []) if int(x) >= 0]
        if len(task_ids) == 0:
            continue
        try:
            client.cancel(task_ids=task_ids)
        except Exception:
            pass
    tasks.clear()


def _build_lookup_tokens(lengths):
    max_len = max(int(x) for x in lengths) if len(lengths) > 0 else 0
    token_ids = torch.zeros((len(lengths), max_len), dtype=torch.int64)
    token_mask = torch.zeros((len(lengths), max_len), dtype=torch.bool)
    for i, ln in enumerate(lengths):
        ln = int(ln)
        if ln <= 0:
            continue
        token_ids[i, :ln] = torch.arange(1, ln + 1, dtype=torch.int64) + i * 1000
        token_mask[i, :ln] = True
    return token_ids, token_mask


def _build_mgr(mode="direct", fail_policy="fail_open"):
    hstu = get_inference_hstu_config(
        hidden_size=128, num_layers=2, num_attention_heads=2, head_dim=64,
        max_batch_size=4, max_seq_len=256, dtype=torch.bfloat16
    )
    extra_kv_args = {}
    if mode == "server_client":
        # server_client mode requires an explicit endpoint in current manager contract.
        extra_kv_args.update(
            flexkv_server_addr=_new_ipc_endpoint("flexkv_server_pytest"),
            flexkv_server_port=0,
        )
    kv = get_kvcache_config(
        blocks_in_primary_pool=512, page_size=32, offload_chunksize=128,
        secondary_backend="flexkv", flexkv_mode=mode, secondary_fail_policy=fail_policy,
        **extra_kv_args,
    )
    mgr = AsyncHSTUKVCacheManager.from_config(hstu, kv)
    return mgr


def _shutdown(mgr):
    secondary = getattr(mgr, "secondary_kvcache_manager", None)
    try:
        mgr.finish_or_cancel_kvcache_ops()
    except Exception:
        pass
    if secondary is not None:
        _cancel_secondary_tasks(secondary)
    client = getattr(secondary, "_client", None) if secondary is not None else None
    if client is not None:
        try:
            client.shutdown()
        except Exception:
            pass
    if hasattr(mgr, "executor"):
        try:
            mgr.executor.shutdown(wait=True, cancel_futures=True)
        except TypeError:
            mgr.executor.shutdown(wait=True)
    if hasattr(mgr, "onload_worker"):
        try:
            mgr.onload_worker.shutdown(wait=True, cancel_futures=True)
        except TypeError:
            mgr.onload_worker.shutdown(wait=True)
    time.sleep(0.05)
    _cleanup_cuda_and_gc()


def _shutdown_storage_mgr(storage_mgr):
    _cancel_secondary_tasks(storage_mgr)
    client = getattr(storage_mgr, "_client", None)
    if client is not None:
        try:
            client.shutdown()
        except Exception:
            pass
    time.sleep(0.05)
    _cleanup_cuda_and_gc()

def test_flexkv_lookup_onboard_offload_smoke():
    mgr = _build_mgr(mode="direct")
    try:
        user_ids = [11, 22]
        lengths = [64, 96]
        token_ids, token_mask = _build_lookup_tokens(lengths)
        lookup = mgr.lookup_kvcache(
            user_ids,
            lengths,
            token_ids=token_ids,
            token_mask=token_mask,
        )
        index_meta, _ = mgr.allocate_kvcache(lookup)
        onboard = mgr.onboard_launch_kvcache(index_meta)
        wait = mgr.onboard_try_wait_kvcache_or_fail(index_meta, onboard)
        assert wait is None or wait.ready
        offload = mgr.lazy_offload_kvcache(index_meta)
        assert offload is not None
        mgr.finish_or_cancel_kvcache_ops(kv_index_meta=index_meta)
    finally:
        _shutdown(mgr)

def test_flexkv_lookup_with_token_ids_and_mask():
    mgr = _build_mgr(mode="direct")
    try:
        user_ids = [101, 202]
        lengths = [6, 4]
        token_ids = torch.tensor(
            [
                [11, 12, 13, 14, 15, 16],
                [21, 22, 23, 24, 0, 0],
            ],
            dtype=torch.int64,
        )
        token_mask = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 0],
            ],
            dtype=torch.bool,
        )
        lookup = mgr.lookup_kvcache(
            user_ids,
            lengths,
            token_ids=token_ids,
            token_mask=token_mask,
        )
        assert lookup.token_ids is not None
        assert lookup.token_mask is not None
        assert lookup.token_ids.shape == token_ids.shape
        assert lookup.token_mask.shape == token_mask.shape
        assert isinstance(lookup.secondary_lookup, dict)
        assert lookup.secondary_lookup.get("backend") == "flexkv"
        hit_mask = lookup.secondary_lookup.get("hit_mask")
        assert hit_mask is None or hit_mask.shape == token_mask.shape
    finally:
        _shutdown(mgr)

def test_flexkv_task_handle_contract():
    mgr = _build_mgr(mode="direct")
    try:
        token_ids, token_mask = _build_lookup_tokens([32])
        lookup = mgr.lookup_kvcache([1], [32], token_ids=token_ids, token_mask=token_mask)
        idx, _ = mgr.allocate_kvcache(lookup)
        real_task_ids = [int(x) for x in (idx.secondary_get_task_ids or []) if int(x) >= 0]
        assert len(real_task_ids) > 0
        idx.restore_slot_mapping = {
            "task_ids": [real_task_ids[0]],
            "slot_mappings": [np.array([0], dtype=np.int64)],
        }
        task = mgr.onboard_launch_kvcache(idx)
        assert task.backend == "flexkv"
        assert task.handle is not None
        assert "task_key" in task.handle
    finally:
        _shutdown(mgr)


def test_flexkv_server_client_mode_smoke():
    mgr = _build_mgr(mode="server_client")
    try:
        token_ids, token_mask = _build_lookup_tokens([32])
        lookup = mgr.lookup_kvcache([1], [32], token_ids=token_ids, token_mask=token_mask)
        idx, _ = mgr.allocate_kvcache(lookup)
        real_task_ids = [int(x) for x in (idx.secondary_get_task_ids or []) if int(x) >= 0]
        assert len(real_task_ids) > 0
        idx.restore_slot_mapping = {
            "task_ids": [real_task_ids[0]],
            "slot_mappings": [np.array([0], dtype=np.int64)],
        }
        task = mgr.onboard_launch_kvcache(idx)
        assert task.handle["mode"] == "server_client"
    finally:
        _shutdown(mgr)

@pytest.mark.parametrize("fail_policy,should_raise", [
    ("fail_open", False),
    ("fail_close", True),
])
def test_flexkv_fail_policy_behavior(fail_policy, should_raise):
    mgr = _build_mgr(mode="direct", fail_policy=fail_policy)
    try:
        token_ids, token_mask = _build_lookup_tokens([32])
        lookup = mgr.lookup_kvcache([1], [32], token_ids=token_ids, token_mask=token_mask)
        idx, _ = mgr.allocate_kvcache(lookup)
        real_task_ids = [int(x) for x in (idx.secondary_get_task_ids or []) if int(x) >= 0]
        assert len(real_task_ids) > 0
        idx.restore_slot_mapping = {
            "task_ids": [real_task_ids[0]],
            "slot_mappings": [np.array([0], dtype=np.int64)],
        }
        handle = mgr.onboard_launch_kvcache(idx)
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            mgr.secondary_kvcache_manager,
            "onboard_wait_kvcache",
            lambda _task_handle: SecondaryWaitResult(
                status=SecondaryTaskStatus.FAILED,
                ready=False,
                error_code=SecondaryErrorCode.ONBOARD_WAIT_FAILED.value,
                message="mock onboard failed",
            ),
        )

        if should_raise:
            with pytest.raises(RuntimeError):
                mgr.onboard_try_wait_kvcache_or_fail(idx, handle)
        else:
            result = mgr.onboard_try_wait_kvcache_or_fail(idx, handle)
            assert result is not None
            assert result.status == SecondaryTaskStatus.FAILED
    finally:
        if 'monkeypatch' in locals():
            monkeypatch.undo()
        _shutdown(mgr)


def test_append_slot_mapping_indptr_to_slot_mapping_contract():
    adapter = FlexKVClientAdapter(mode="direct")
    page_size = 4
    index_meta = KVIndexMeta(
        request_id="req-append-contract",
        batch_size=1,
        user_ids=[7],
        namespaces=["uid:7"],
        total_history_lengths=[8],
        old_cached_lengths=[0],
        seq_start_indices=[0],
        seq_lengths=[8],
        new_tokens=8,
        token_ids=torch.tensor([[11, 12, 13, 14, 15, 16, 17, 18]], dtype=torch.int64),
        token_mask=torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.bool),
        secondary_matched_lengths=[0],
    )
    append_slot_mapping = torch.tensor([3, 4], dtype=torch.int64)   # 2 pages
    append_slot_indptr = torch.tensor([0, 2], dtype=torch.int64)
    reqs = adapter.to_offload_requests(
        index_meta=index_meta,
        append_slot_mapping=append_slot_mapping,
        append_slot_indptr=append_slot_indptr,
        tokens_per_block=page_size,
    )
    assert len(reqs) == 1
    req = reqs[0]
    assert req["token_ids"].dtype == np.int64
    assert req["slot_mapping"].dtype == np.int64
    assert req["token_mask"].dtype == np.bool_
    assert req["slot_mapping"].size == int(req["token_mask"].sum())
    # 核心契约：slot_mapping[::page_size] // page_size == page_ids
    np.testing.assert_array_equal(
        req["slot_mapping"][::page_size] // page_size,
        np.array([3, 4], dtype=np.int64),
    )
def test_offload_fail_policy_close_raises(monkeypatch):
    mgr = _build_mgr(mode="direct", fail_policy="fail_close")
    try:
        task_key = "offload:req-fail-close"
        handle = SecondaryTaskHandle(
            backend="flexkv",
            handle={"task_key": task_key, "task_ids": [101]},
            status=SecondaryTaskStatus.LAUNCHED,
        )
        mgr.ongoing_offload_tasks["req-fail-close"] = handle
        mgr.request_to_task_handles["req-fail-close"] = {"offload": handle}
        def _mock_offload_wait(_task_handle):
            return SecondaryWaitResult(
                status=SecondaryTaskStatus.FAILED,
                ready=False,
                error_code=SecondaryErrorCode.OFFLOAD_WAIT_FAILED.value,
                message="mock offload failed",
            )
        monkeypatch.setattr(
            mgr.secondary_kvcache_manager,
            "offload_wait_kvcache",
            _mock_offload_wait,
        )
        monkeypatch.setattr(
            mgr.secondary_kvcache_manager,
            "cancel_task",
            lambda _task_handle: None,
        )
        with pytest.raises(RuntimeError):
            mgr.finish_or_cancel_kvcache_ops(
                kv_index_meta=SimpleNamespace(request_id="req-fail-close")
            )
    finally:
        _shutdown(mgr)
def test_server_client_timeout_surface(monkeypatch):
    storage_mgr = FlexKVStorageManager(
        mode="server_client",
        server_addr=_new_ipc_endpoint("flexkv_test"),
        server_port=0,
        secondary_wait_timeout_ms=10,
    )
    try:
        task_key = "offload:req-timeout"
        storage_mgr._tasks[task_key] = {
            "task_ids": [201],
            "user_ids": [42],
        }
        handle = SecondaryTaskHandle(
            backend="flexkv",
            handle={"task_key": task_key, "task_ids": [201]},
            status=SecondaryTaskStatus.LAUNCHED,
        )
        # 强制返回 TIMEOUT
        monkeypatch.setattr(
            storage_mgr,
            "_wait_task_ids",
            lambda task_ids: {
                201: SimpleNamespace(status=SimpleNamespace(name="TIMEOUT"))
            },
        )
        result = storage_mgr.offload_wait_kvcache(handle)
        assert result.status == SecondaryTaskStatus.TIMEOUT
        assert result.error_code == SecondaryErrorCode.OFFLOAD_TIMEOUT.value
        assert result.failed_user_ids == [42]
        assert result.failed_mask is not None
        assert bool(result.failed_mask.all())
    finally:
        _shutdown_storage_mgr(storage_mgr)
def test_lookup_returns_task_ids_and_matched_lengths():
    storage_mgr = FlexKVStorageManager(mode="direct")
    try:
        # FlexKV now requires explicit GPU KV cache registration before lookup.
        cache_tables = [
            torch.zeros(
                (8, 2, storage_mgr.page_size, storage_mgr.num_heads, storage_mgr.head_dim),
                dtype=torch.bfloat16,
                device="cuda",
            )
            for _ in range(storage_mgr.num_layers)
        ]
        storage_mgr.register_gpu_cache_tensors(cache_tables)

        index_meta = KVIndexMeta(
            request_id="req-lookup-fields",
            batch_size=2,
            user_ids=[1, 2],
            namespaces=["uid:1", "uid:2"],
            total_history_lengths=[4, 2],
            old_cached_lengths=[0, 0],
            seq_start_indices=[0, 0],
            seq_lengths=[4, 2],
            new_tokens=6,
            token_ids=torch.tensor(
                [
                    [11, 12, 13, 14],
                    [21, 22, 0, 0],
                ],
                dtype=torch.int64,
            ),
            token_mask=torch.tensor(
                [
                    [1, 1, 1, 1],
                    [1, 1, 0, 0],
                ],
                dtype=torch.bool,
            ),
        )
        out = storage_mgr.lookup_kvcache(index_meta)
        assert out["backend"] == "flexkv"
        assert "task_ids" in out and len(out["task_ids"]) == 2
        assert "matched_lengths" in out and len(out["matched_lengths"]) == 2
        assert all(int(x) >= 0 for x in out["matched_lengths"])
        assert isinstance(out["hit_mask"], torch.Tensor)
        assert out["hit_mask"].shape == (2, 4)
    finally:
        _shutdown_storage_mgr(storage_mgr)