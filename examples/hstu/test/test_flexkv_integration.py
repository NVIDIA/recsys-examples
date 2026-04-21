import pytest
import torch
from configs import get_inference_hstu_config, get_kvcache_config
from modules.async_kvcache_manager import AsyncHSTUKVCacheManager, SecondaryTaskStatus
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def _build_mgr(mode="direct", fail_policy="fail_open"):
    hstu = get_inference_hstu_config(
        hidden_size=128, num_layers=2, num_attention_heads=2, head_dim=64,
        max_batch_size=4, max_seq_len=256, dtype=torch.bfloat16
    )
    kv = get_kvcache_config(
        blocks_in_primary_pool=512, page_size=32, offload_chunksize=128,
        secondary_backend="flexkv", flexkv_mode=mode, secondary_fail_policy=fail_policy
    )
    mgr = AsyncHSTUKVCacheManager.from_config(hstu, kv)
    return mgr
def _shutdown(mgr):
    if hasattr(mgr, "executor"):
        mgr.executor.shutdown(wait=False)
    if hasattr(mgr, "onload_worker"):
        mgr.onload_worker.shutdown(wait=False)
def test_flexkv_lookup_onboard_offload_smoke():
    mgr = _build_mgr(mode="direct")
    try:
        user_ids = [11, 22]
        lengths = [64, 96]
        lookup = mgr.lookup_kvcache(user_ids, lengths)
        index_meta, _ = mgr.allocate_kvcache(user_ids, lookup)
        onboard = mgr.onboard_launch_kvcache(user_ids, index_meta, lookup)
        wait = mgr.onboard_try_wait_kvcache_or_fail(user_ids, index_meta, lookup, onboard)
        assert wait is None or wait.ready
        offload = mgr.lazy_offload_kvcache(user_ids, index_meta, lookup)
        assert offload is not None
        mgr.finish_or_cancel_kvcache_ops(uid_or_uids=user_ids, kv_index_meta=index_meta)
    finally:
        _shutdown(mgr)
def test_flexkv_task_handle_contract():
    mgr = _build_mgr(mode="direct")
    try:
        lookup = mgr.lookup_kvcache([1], [32])
        idx, _ = mgr.allocate_kvcache([1], lookup)
        task = mgr.onboard_launch_kvcache([1], idx, lookup)
        assert task.backend == "flexkv"
        assert task.handle is not None
        assert "task_key" in task.handle
    finally:
        _shutdown(mgr)
def test_flexkv_server_client_mode_smoke():
    mgr = _build_mgr(mode="server_client")
    try:
        lookup = mgr.lookup_kvcache([1], [32])
        idx, _ = mgr.allocate_kvcache([1], lookup)
        task = mgr.onboard_launch_kvcache([1], idx, lookup)
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
        lookup = mgr.lookup_kvcache([1], [32])
        idx, _ = mgr.allocate_kvcache([1], lookup)
        handle = mgr.onboard_launch_kvcache([1], idx, lookup)

        # 注入失败：模拟 wait 路径异常
        if handle and handle.handle and "task_key" in handle.handle:
            task_key = handle.handle["task_key"]
            mgr.secondary_kvcache_manager._tasks[task_key]["error"] = "mock_wait_failed"

        if should_raise:
            with pytest.raises(RuntimeError):
                mgr.onboard_try_wait_kvcache_or_fail([1], idx, lookup, handle)
        else:
            result = mgr.onboard_try_wait_kvcache_or_fail([1], idx, lookup, handle)
            assert result is not None
            assert result.status == SecondaryTaskStatus.FAILED
    finally:
        _shutdown(mgr)
