from __future__ import annotations

import argparse
import time
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Dict

import torch

from recsys_kvcache_manager.host_kvstorage_manager import HostKVTaskStatus
from recsys_kvcache_manager.kvcache_config import get_kvcache_config
from recsys_kvcache_manager.kvcache_manager import KVCacheManager
from recsys_kvcache_manager.kvcache_utils import KVLookupResult

CURRENT_NVTX_SCOPE: ContextVar[str] = ContextVar("current_nvtx_scope", default="")


@contextmanager
def nvtx_range(name: str):
    scope_token = None
    if name.startswith("step") or name.startswith("init."):
        scope_token = CURRENT_NVTX_SCOPE.set(name)
    torch.cuda.nvtx.range_push(name)
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        torch.cuda.nvtx.range_pop()
        print(f"[NVTX] {name:<44} {elapsed_ms:9.3f} ms")
        if scope_token is not None:
            CURRENT_NVTX_SCOPE.reset(scope_token)


def _scoped_name(name: str) -> str:
    scope = CURRENT_NVTX_SCOPE.get()
    return f"{scope}::{name}" if scope else name


def wrap_with_nvtx(target, attr_name: str, nvtx_name: str) -> None:
    """Wrap an instance method or module function with scoped NVTX ranges."""
    if target is None or not hasattr(target, attr_name):
        return
    original = getattr(target, attr_name)
    if getattr(original, "__nvtx_wrapped__", False):
        return

    @wraps(original)
    def wrapped(*args, **kwargs):
        with nvtx_range(_scoped_name(nvtx_name)):
            return original(*args, **kwargs)

    wrapped.__nvtx_wrapped__ = True
    try:
        setattr(target, attr_name, wrapped)
    except Exception as e:  # noqa: BLE001
        print(
            f"[WARN] Failed to wrap {target!r}.{attr_name} with NVTX "
            f"({nvtx_name}): {e}"
        )


def _install_precreate_nvtx_hooks() -> None:
    from recsys_kvcache_manager.flex_kvcache_manager import FlexKVStorageManager

    wrap_with_nvtx(
        FlexKVStorageManager,
        "register_gpu_cache_tables",
        "flexkv.register_gpu_cache_tables",
    )


def install_nvtx_hooks(kvcache_mgr: KVCacheManager) -> None:
    flexkv_mgr = kvcache_mgr.host_kvstorage_manager
    wrap_with_nvtx(flexkv_mgr, "build_index_meta", "flexkv.build_index_meta")
    wrap_with_nvtx(flexkv_mgr, "lookup_kvcache", "flexkv.lookup_kvcache")
    wrap_with_nvtx(
        flexkv_mgr, "onboard_kvcache_launch", "flexkv.onboard_kvcache_launch"
    )
    wrap_with_nvtx(
        flexkv_mgr, "onboard_kvcache_wait", "flexkv.onboard_kvcache_wait"
    )
    wrap_with_nvtx(
        flexkv_mgr, "offload_kvcache_launch", "flexkv.offload_kvcache_launch"
    )
    wrap_with_nvtx(
        flexkv_mgr, "offload_kvcache_wait", "flexkv.offload_kvcache_wait"
    )
    wrap_with_nvtx(flexkv_mgr, "finish_task", "flexkv.finish_task")
    wrap_with_nvtx(flexkv_mgr, "cancel_task", "flexkv.cancel_task")

    adapter = getattr(flexkv_mgr, "_adapter", None)
    wrap_with_nvtx(
        adapter, "to_get_match_requests", "flexkv.adapter.to_get_match_requests"
    )
    wrap_with_nvtx(
        flexkv_mgr, "_build_slot_mappings", "flexkv._build_slot_mappings"
    )

    client = getattr(flexkv_mgr, "_client", None)
    wrap_with_nvtx(client, "get_match", "flexkv.client.get_match")
    wrap_with_nvtx(client, "put_async", "flexkv.client.put_async")
    wrap_with_nvtx(client, "launch", "flexkv.client.launch")
    wrap_with_nvtx(client, "try_wait", "flexkv.client.try_wait")
    wrap_with_nvtx(client, "wait", "flexkv.client.wait")

    install_recsys_glue_hooks(kvcache_mgr)


def install_recsys_glue_hooks(kvcache_mgr: KVCacheManager) -> None:
    if getattr(KVLookupResult.merge, "__nvtx_wrapped__", False):
        return
    original_merge = KVLookupResult.merge

    @classmethod
    @wraps(original_merge)
    def merge_with_nvtx(cls, lookup_res1, lookup_res2):
        with nvtx_range(_scoped_name("recsys.merge_lookup_results")):
            return original_merge(lookup_res1, lookup_res2)

    merge_with_nvtx.__nvtx_wrapped__ = True
    KVLookupResult.merge = merge_with_nvtx

    wrap_with_nvtx(
        kvcache_mgr, "offload_try_wait", "recsys.offload_try_wait_loop"
    )


def create_testing_kvcache_manager(
    max_batch_size: int,
    max_seq_len: int,
) -> KVCacheManager:
    _install_precreate_nvtx_hooks()
    kvcache_config = get_kvcache_config(
        num_layers=3,
        num_heads=4,
        head_dim=128,
        page_size=32,
        offload_chunksize=128,
        num_primary_cache_pages=512,
        num_buffer_pages=0,
        host_capacity_per_layer=max_seq_len * max_batch_size * 32 * 4 * 128 * 2,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        dtype=torch.bfloat16,
        device=torch.cuda.current_device(),
        host_kvstorage_backend="flexkv",
        offload_timeout_ms=100.0,
        offload_mode="lazy",
        extra_configs={
            "flexkv_mode": "direct",
            "flexkv_host_kvstorage_fail_policy": "fail_open",
            "flexkv_enable_mps": 0,
        },
    )
    gpu_gib = (
        kvcache_config.num_layers
        * kvcache_config.num_primary_cache_pages
        * kvcache_config.page_size
        * 2
        * kvcache_config.num_heads
        * kvcache_config.head_dim
        * 2
    ) / (1024.0**3)
    host_gib = (
        kvcache_config.num_layers * kvcache_config.host_capacity_per_layer
    ) / (1024.0**3)
    print(f"[DEBUG] KVCache GPU Memory Usage: {gpu_gib:.3f} GiB")
    print(f"[DEBUG] KVCache Host Memory Usage: {host_gib:.3f} GiB")
    kvcache_mgr = KVCacheManager.from_config(kvcache_config)
    install_nvtx_hooks(kvcache_mgr)
    return kvcache_mgr


def normalize_index_meta(index_meta) -> None:
    if not hasattr(index_meta, "sequence_lengths"):
        index_meta.sequence_lengths = index_meta.seq_lengths
    if not hasattr(index_meta, "slot_mappings"):
        index_meta.slot_mappings = None
    if hasattr(index_meta, "namespaces") and index_meta.namespaces is not None:
        index_meta.namespaces = [
            ns if isinstance(ns, list) else [ns] for ns in index_meta.namespaces
        ]


def build_uniform_batch(all_keys, all_values, len_per_seq: int, batch_size: int):
    seqlen = [len_per_seq] * batch_size
    user_ids = torch.tensor(list(range(batch_size)), dtype=torch.int64)
    sequence_lengths = torch.tensor(seqlen, dtype=torch.int32)
    keys = [
        all_keys[uid][:, : seqlen[i], ...]
        for i, uid in enumerate(range(batch_size))
    ]
    values = [
        all_values[uid][:, : seqlen[i], ...]
        for i, uid in enumerate(range(batch_size))
    ]
    return user_ids, sequence_lengths, keys, values, seqlen


def build_uniform_request(len_per_seq: int, batch_size: int):
    seqlen = [len_per_seq] * batch_size
    user_ids = torch.tensor(list(range(batch_size)), dtype=torch.int64)
    sequence_lengths = torch.tensor(seqlen, dtype=torch.int32)
    return user_ids, sequence_lengths


def run_step_1_offload(
    kvcache_mgr: KVCacheManager,
    all_keys,
    all_values,
    len_per_seq: int = 1024,
    batch_size: int = 1,
    mark_input_nvtx: bool = False,
) -> None:
    step_name = "step1"
    if mark_input_nvtx:
        with nvtx_range(f"{step_name}.input"):
            user_ids, sequence_lengths, keys, values, _ = build_uniform_batch(
                all_keys=all_keys,
                all_values=all_values,
                len_per_seq=len_per_seq,
                batch_size=batch_size,
            )
    else:
        user_ids, sequence_lengths, keys, values, _ = build_uniform_batch(
            all_keys=all_keys,
            all_values=all_values,
            len_per_seq=len_per_seq,
            batch_size=batch_size,
        )

    with nvtx_range(f"{step_name}.lookup"):
        index_meta, lookup_res = kvcache_mgr.lookup_kvcache(user_ids, sequence_lengths)
    normalize_index_meta(index_meta)
    assert torch.allclose(
        lookup_res.cached_lengths, torch.zeros((batch_size,), dtype=torch.int32)
    )

    with nvtx_range(f"{step_name}.allocate"):
        kvcache_metadata = kvcache_mgr.allocate_kvcache(index_meta, lookup_res)
    assert torch.allclose(kvcache_metadata.total_history_lengths, sequence_lengths.cuda())

    for layer_idx in range(3):
        kvcache_mgr.gpu_kvcache_mgr.put(
            torch.cat([k[layer_idx] for k in keys], dim=0),
            torch.cat([v[layer_idx] for v in values], dim=0),
            layer_idx,
            kvcache_metadata,
        )

    with nvtx_range(f"{step_name}.offload_launch"):
        task_handle = kvcache_mgr.offload_launch(
            index_meta=index_meta,
            kvcache_metadata=kvcache_metadata,
        )
    if task_handle is None or task_handle.handle is None:
        raise RuntimeError("step1: offload_launch did not return a valid handle")

    with nvtx_range(f"{step_name}.offload_wait"):
        while True:
            kvcache_mgr.offload_try_wait()
            if len(kvcache_mgr.ongoing_offload_tasks) == 0:
                break


def run_step_2_evict_gpu(kvcache_mgr: KVCacheManager, batch_size: int = 1) -> None:
    step_name = "step2"
    user_ids = torch.tensor(list(range(batch_size)), dtype=torch.int64)
    with nvtx_range(f"{step_name}.evict_gpu"):
        kvcache_mgr.evict(user_ids, for_gpu=True)


def run_step_3_onboard(
    kvcache_mgr: KVCacheManager,
    len_per_seq: int = 1024,
    batch_size: int = 1,
    mark_input_nvtx: bool = False,
) -> None:
    step_name = "step3"
    if mark_input_nvtx:
        with nvtx_range(f"{step_name}.input"):
            user_ids, sequence_lengths = build_uniform_request(
                len_per_seq=len_per_seq,
                batch_size=batch_size,
            )
    else:
        user_ids, sequence_lengths = build_uniform_request(
            len_per_seq=len_per_seq,
            batch_size=batch_size,
        )

    with nvtx_range(f"{step_name}.lookup"):
        index_meta, lookup_res = kvcache_mgr.lookup_kvcache(user_ids, sequence_lengths)
    normalize_index_meta(index_meta)
    assert torch.allclose(
        lookup_res.gpu_cached_lengths, torch.zeros((batch_size,), dtype=torch.int32)
    ), "step3 requires GPU cache to be evicted before onboard."

    with nvtx_range(f"{step_name}.allocate"):
        kvcache_metadata = kvcache_mgr.allocate_kvcache(index_meta, lookup_res)

    if getattr(kvcache_metadata, "new_history_nnz_cuda", None) is not None:
        new_history_nnz = int(kvcache_metadata.new_history_nnz_cuda.item())
    else:
        new_history_nnz = int(getattr(kvcache_metadata, "new_history_nnz", -1))
    assert new_history_nnz == 0, (
        "step3 expects same input and full onboard-only recovery; "
        f"new_history_nnz={new_history_nnz}. "
        "Use len_per_seq divisible by 32 (e.g. 1024/2048/4096)."
    )

    with nvtx_range(f"{step_name}.onboard_launch"):
        onboard_handle = kvcache_mgr.onboard_launch(index_meta, lookup_res, kvcache_metadata)
    assert onboard_handle is not None and onboard_handle.handle is not None, (
        "step3: onboard_launch did not return a valid handle"
    )
    assert onboard_handle.status == HostKVTaskStatus.LAUNCHED, (
        f"step3: onboard status is {onboard_handle.status}, expected LAUNCHED"
    )

    with nvtx_range(f"{step_name}.onboard_wait"):
        deadline = time.time() + 60.0
        onboard_ready = False
        while time.time() < deadline:
            onboard_wait_result = kvcache_mgr.onboard_wait(index_meta, onboard_handle)
            if onboard_wait_result.ready:
                onboard_ready = True
                break
            time.sleep(0.005)
    assert onboard_ready, "step3: onboard_wait did not reach ready=True"


class _NVTXProxy:
    def __init__(self, target, method_to_nvtx: Dict[str, str]):
        self._target = target
        self._method_to_nvtx = method_to_nvtx
        self._cache = {}

    def __getattr__(self, name):
        attr = getattr(self._target, name)
        if not callable(attr):
            return attr
        if name not in self._method_to_nvtx:
            return attr
        if name in self._cache:
            return self._cache[name]
        nvtx_name = self._method_to_nvtx[name]

        @wraps(attr)
        def wrapped(*args, **kwargs):
            with nvtx_range(_scoped_name(nvtx_name)):
                return attr(*args, **kwargs)

        wrapped.__nvtx_wrapped__ = True
        self._cache[name] = wrapped
        return wrapped


def install_gpu_cpp_kernel_hooks(kvcache_mgr) -> None:
    gpu_mgr = getattr(kvcache_mgr, "gpu_kvcache_mgr", None)
    if gpu_mgr is None:
        return

    wrap_with_nvtx(gpu_mgr, "lookup", "gpu.lookup_py")
    wrap_with_nvtx(gpu_mgr, "allocate", "gpu.allocate_py")
    wrap_with_nvtx(gpu_mgr, "check_for_offload", "gpu.check_for_offload_py")
    wrap_with_nvtx(
        gpu_mgr, "acquire_offload_pages", "gpu.acquire_offload_pages_py"
    )
    wrap_with_nvtx(
        gpu_mgr, "release_offload_pages", "gpu.release_offload_pages_py"
    )
    wrap_with_nvtx(
        gpu_mgr, "revoke_onboard_pages", "gpu.revoke_onboard_pages_py"
    )
    wrap_with_nvtx(gpu_mgr, "evict", "gpu.evict_py")
    wrap_with_nvtx(gpu_mgr, "evict_all", "gpu.evict_all_py")
    wrap_with_nvtx(gpu_mgr, "put", "gpu.put_py")
    wrap_with_nvtx(gpu_mgr, "get", "gpu.get_py")

    gpu_impl = getattr(gpu_mgr, "impl_", None)
    if gpu_impl is not None:
        gpu_mgr.impl_ = _NVTXProxy(
            gpu_impl,
            {
                "lookup": "gpu.lookup_cpp",
                "allocate": "gpu.allocate_cpp",
                "check_for_offload": "gpu.check_for_offload_cpp",
                "acquire_offload_pages": "gpu.acquire_offload_pages_cpp",
                "release_offload_pages": "gpu.release_offload_pages_cpp",
                "revoke_onboard_pages": "gpu.revoke_onboard_pages_cpp",
                "evict": "gpu.evict_cpp",
                "evict_all": "gpu.evict_all_cpp",
            },
        )

    try:
        import paged_kvcache_ops
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] paged_kvcache_ops import failed, skip kernel hook: {e}")
        return

    wrap_with_nvtx(
        paged_kvcache_ops, "append_kvcache", "gpu.kernel.append_kvcache"
    )


def install_cpu_cpp_hooks(kvcache_mgr) -> None:
    flexkv_mgr = getattr(kvcache_mgr, "host_kvstorage_manager", None)
    client = getattr(flexkv_mgr, "_client", None)
    if client is None:
        return

    method_to_nvtx = {
        "get_match": "cpu.get_match_cpp",
        "put_async": "cpu.put_async_cpp",
        "launch": "cpu.launch_cpp",
        "try_wait": "cpu.try_wait_cpp",
        "wait": "cpu.wait_cpp",
        "cancel": "cpu.cancel_cpp",
    }

    candidate_impl_attrs = [
        "impl_",
        "_impl",
        "impl",
        "_manager",
        "manager",
        "_kv_manager",
        "kv_manager",
        "_core",
        "core",
    ]
    for attr_name in candidate_impl_attrs:
        inner = getattr(client, attr_name, None)
        if inner is None:
            continue
        if not any(hasattr(inner, m) for m in method_to_nvtx):
            continue
        try:
            setattr(client, attr_name, _NVTXProxy(inner, method_to_nvtx))
            print(f"[INFO] CPU C++ hooks installed on _client.{attr_name}")
            return
        except Exception as e:  # noqa: BLE001
            print(
                f"[WARN] Failed to install CPU C++ hooks on _client.{attr_name}: {e}"
            )

    try:
        flexkv_mgr._client = _NVTXProxy(client, method_to_nvtx)
        print("[INFO] CPU C++ hooks installed on _client boundary")
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] Failed to install CPU C++ hooks on _client boundary: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-grained FlexKV profiler: "
            "step-level + backend-level + GPU C++/kernel breakdown."
        )
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=8192,
        help="Max tensor sequence shape used for random input generation",
    )
    parser.add_argument(
        "--len-per-seq",
        type=int,
        default=1024,
        help="Uniform sequence length per request, e.g. 1024/2048/4096",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size, e.g. 1,2,4,8,16,32",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat count for the 3-step flow in one run",
    )
    parser.add_argument(
        "--mark-input-nvtx",
        action="store_true",
        help="Also mark step1.input and step3.input.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def run_one_case(
    len_per_seq: int,
    max_seq_len: int,
    batch_size: int,
    args: argparse.Namespace,
) -> None:
    with nvtx_range("init.create_kvcache_manager"):
        kvcache_mgr = create_testing_kvcache_manager(
            max_batch_size=batch_size,
            max_seq_len=max_seq_len,
        )
        install_gpu_cpp_kernel_hooks(kvcache_mgr)
        install_cpu_cpp_hooks(kvcache_mgr)

    try:
        with nvtx_range("init.prepare_inputs"):
            g_keys = [
                torch.randn((3, max_seq_len, 4, 128), dtype=torch.bfloat16).cuda()
                for _ in range(batch_size)
            ]
            g_values = [
                torch.randn((3, max_seq_len, 4, 128), dtype=torch.bfloat16).cuda()
                for _ in range(batch_size)
            ]

        for repeat_idx in range(args.repeat):
            print(
                f"[RUN] repeat {repeat_idx + 1}/{args.repeat} "
                f"step1: input({len_per_seq} x {batch_size}) + lookup/allocate/offload"
            )
            run_step_1_offload(
                kvcache_mgr,
                g_keys,
                g_values,
                len_per_seq=len_per_seq,
                batch_size=batch_size,
                mark_input_nvtx=args.mark_input_nvtx,
            )

            print(
                f"[RUN] repeat {repeat_idx + 1}/{args.repeat} "
                f"step2: evict gpu (batch={batch_size})"
            )
            run_step_2_evict_gpu(kvcache_mgr, batch_size=batch_size)

            print(
                f"[RUN] repeat {repeat_idx + 1}/{args.repeat} "
                f"step3: input({len_per_seq} x {batch_size}) + lookup/allocate/onboard"
            )
            run_step_3_onboard(
                kvcache_mgr,
                len_per_seq=len_per_seq,
                batch_size=batch_size,
                mark_input_nvtx=args.mark_input_nvtx,
            )
        print(f"[DONE] fine C++/kernel profile completed. repeat={args.repeat}")
    finally:
        with nvtx_range("init.shutdown"):
            flexkv_mgr = getattr(kvcache_mgr, "host_kvstorage_manager", None)
            client = getattr(flexkv_mgr, "_client", None)
            if client is not None and hasattr(client, "shutdown"):
                try:
                    client.shutdown()
                except Exception as e:  # noqa: BLE001
                    print(f"[WARN] FlexKV client shutdown failed: {e}")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer")
    if args.repeat <= 0:
        raise ValueError("--repeat must be a positive integer")
    if args.len_per_seq <= 0:
        raise ValueError("--len-per-seq must be a positive integer")
    if args.len_per_seq > args.max_seq_len:
        raise ValueError("--max-seq-len must be >= --len-per-seq")

    torch.manual_seed(args.seed)
    print(
        f"[INFO] len_per_seq={args.len_per_seq}, "
        f"batch_size={args.batch_size}, repeat={args.repeat}"
    )
    print(f"[INFO] seqlen={[args.len_per_seq] * args.batch_size}")

    run_one_case(
        len_per_seq=args.len_per_seq,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        args=args,
    )


if __name__ == "__main__":
    main()
