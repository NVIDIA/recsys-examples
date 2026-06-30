import torch


def _append_kvcache_fake(
    append_key,
    append_value,
    batch_indices,
    positions,
    seqlen_offsets,
    nnz_cuda,
    kv_cache_table,
    kv_indices,
    kv_indptr,
    kv_last_page_len,
    kv_layout: int,
):
    del kv_layout
    torch._check(append_key.is_cuda)
    torch._check(append_value.is_cuda)
    torch._check(batch_indices.is_cuda)
    torch._check(positions.is_cuda)
    torch._check(seqlen_offsets.is_cuda)
    torch._check(nnz_cuda.is_cuda)
    torch._check(kv_cache_table.is_cuda)
    torch._check(kv_indices.is_cuda)
    torch._check(kv_indptr.is_cuda)
    torch._check(kv_last_page_len.is_cuda)
    return kv_cache_table


if hasattr(torch.ops, "paged_kvcache_ops") and hasattr(
    torch.ops.paged_kvcache_ops, "append_kvcache"
):
    torch.library.register_fake("paged_kvcache_ops::append_kvcache")(
        _append_kvcache_fake
    )