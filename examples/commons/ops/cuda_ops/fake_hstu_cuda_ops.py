import torch


@torch.library.register_fake("hstu_cuda_ops::split_by_lengths")
def _split_by_lengths_fake(values, lengths_1d, num_splits: int):
    torch._check(values.dim() == 1 or values.dim() == 2)
    torch._check(lengths_1d.dim() == 1)

    # Tensor[] return length is static (num_splits), which is export-friendly.
    # Each output tensor length along dim 0 is data-dependent; represent with symbolic sizes.
    # Keep trailing dimensions the same as `values`.
    ctx = torch.library.get_ctx()
    out = []
    for _ in range(num_splits):
        dyn_len = ctx.new_dynamic_size()
        if values.dim() == 1:
            out.append(values.new_empty((dyn_len,)))
        else:
            out.append(values.new_empty((dyn_len, values.size(1))))
    return out


@torch.library.register_fake("hstu_cuda_ops::lengths_splits")
def _lengths_splits_fake(lengths_1d, num_splits: int):
    torch._check(lengths_1d.dim() == 1)
    torch._check(num_splits > 0)

    # num_splits is static at export time; the second dim is data-dependent.
    torch.library.get_ctx()
    dyn_batch = lengths_1d.size(0) // num_splits
    out = []
    for _ in range(num_splits):
        out.append(lengths_1d.new_empty((dyn_batch,)))
    return out


@torch.library.register_fake("hstu_cuda_ops::lengths_reduce_dim1")
def _lengths_reduce_dim1_fake(lengths_1d, num_splits: int):
    torch._check(lengths_1d.dim() == 1)
    torch._check(num_splits > 0)

    # num_splits is static at export time.
    return lengths_1d.new_empty((num_splits,))
