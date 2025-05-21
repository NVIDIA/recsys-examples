import torch
import pytest
from typing import List, Tuple
from torchrec.sparse.jagged_tensor import JaggedTensor

import hstu_cuda_ops 
from JaggedTensorOpFunction import _JaggedTensorOpFunction 
from ops.triton_ops.triton_jagged import _Concat2DJaggedFunction

from JaggedTensorOpFunction import jagged_2D_tensor_concat
# Reference pytorch implementation
def concat_2D_jagged_tensors_pytorch(
    jagged_tensors: List[JaggedTensor],
    max_seqlens: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(jagged_tensors) == 0:
        raise ValueError("empty tensor list to concat")
    if len(jagged_tensors) == 1:
        return jagged_tensors[0].values(), jagged_tensors[0].lengths()
    padded_dense_list = []
    padded_mask_list = []
    for jt, max_seqlen in zip(jagged_tensors, max_seqlens):
        padded_dense = torch.ops.fbgemm.jagged_to_padded_dense(
            values=jt.values(),
            offsets=[jt.offsets()],
            max_lengths=[max_seqlen],
            padding_value=0.0,
        )
        padded_mask = torch.ops.fbgemm.jagged_to_padded_dense(
            values=torch.ones(
                (jt.values().numel(),), dtype=torch.long, device=jt.values().device
            ).view(-1, 1),
            offsets=[jt.offsets()],
            max_lengths=[max_seqlen],
            padding_value=0,
        ).to(torch.bool)
        padded_dense_list.append(padded_dense)
        padded_mask_list.append(padded_mask)

    concatted_dense = torch.cat(padded_dense_list, dim=1)
    concatted_mask = torch.cat(padded_mask_list, dim=1)
    return concatted_dense.flatten(0, 1)[concatted_mask.view(-1), :], torch.sum(
        torch.concat([jt.lengths().view(-1, 1) for jt in jagged_tensors], dim=1), dim=1
    )

# Create test jagged tensor
def create_test_jagged_tensor(batch_size, max_len, hidden_dim, dtype=torch.float32):

    lengths = torch.randint(
        1, max_len + 1, size=(batch_size,), device=torch.device("cuda")
    )
    offsets = torch.cat([torch.tensor([0]).cuda(), torch.cumsum(lengths, dim=0)]).cuda().to(torch.int32)
    
    offsets[1:] = torch.cumsum(lengths, dim=0, dtype=torch.int32)
    total_len = int(offsets[-1].item())
    values = (
        torch.empty(
            (total_len, hidden_dim),
            dtype=dtype,
            device=torch.device("cuda"),
        )
        .uniform_(-1.0, 1.0)
        .requires_grad_(True)
    )

    return JaggedTensor(
        values=values,
        lengths=lengths,
        offsets=offsets
    )

# Test jagged tensor creation
@pytest.mark.parametrize("batch_size,max_len,hidden_dim", [
    (2, 3, 4),
    (4, 5, 8),
    (1, 2, 16),
    (4, 10, 5)
])
def test_jagged_tensor_creation(batch_size, max_len, hidden_dim):
    jt = create_test_jagged_tensor(batch_size, max_len, hidden_dim)
    assert jt.values().shape[1] == hidden_dim
    assert jt.lengths().shape[0] == batch_size
    assert jt.offsets().shape[0] == batch_size + 1 

# Print jagged tensor
def print_jagged_tensor(jt):
    print("Values:", jt.values())
    print("Lengths:", jt.lengths())
    print("Offsets:", jt.offsets())

#test N jagged tensor concat kernel
@pytest.mark.parametrize("num", [1, 2, 3, 4])
@pytest.mark.parametrize("batch_size,max_len,hidden_dim", [
    (2, 3, 4),
    (4, 5, 8),
    (1, 2, 16),
    (4, 10, 5)
])
def test_n_jagged_tensor_concat_kernel(num, batch_size, max_len, hidden_dim):
    with torch.cuda.nvtx.range("Test Setup", color="blue"):
        jt_list = [create_test_jagged_tensor(batch_size, max_len, hidden_dim) for _ in range(num)]
        max_seqlens = [max(jt.lengths()) for jt in jt_list]
        from JaggedTensorOpFunction import jagged_2D_tensor_concat

    with torch.cuda.nvtx.range("Custom Implementation", color="purple"):
        result = jagged_2D_tensor_concat([jt.values() for jt in jt_list], [jt.offsets() for jt in jt_list], max_seqlens)
    
    with torch.cuda.nvtx.range("Original Implementation", color="green"):
        result2 = concat_2D_jagged_tensors_pytorch(jt_list, max_seqlens)

    with torch.cuda.nvtx.range("Result Verification", color="yellow"):
        assert torch.equal(result[0], result2[0])
        assert torch.equal(result[1], result2[1])
@pytest.mark.parametrize("batch_size,max_len,hidden_dim", [
    (2, 3, 4)
])
def test_triton_jagged_tensor_concat(batch_size, max_len, hidden_dim):
    with torch.cuda.nvtx.range("Test Setup", color="blue"):
        jt1 = create_test_jagged_tensor(batch_size, max_len, hidden_dim)
        jt2 = create_test_jagged_tensor(batch_size, max_len+1, hidden_dim)
        max_len_jt1 = torch.max(jt1.lengths()).item()
        max_len_jt2 = torch.max(jt2.lengths()).item()
        calculated_max_seq_len = max_len_jt1 + max_len_jt2
    # from triton_jagged import _Concat2DJaggedFunction
    from ops.triton_ops.triton_jagged import triton_concat_2D_jagged
    # for _ in range(20):
    with torch.cuda.nvtx.range("triton concat", color="purple"):
        # result = _Concat2DJaggedFunction.apply(
        #     calculated_max_seq_len,
        #     jt1.values(),
        #     jt2.values(),
        #     jt1.offsets(),
        #     jt2.offsets(),
        #     False,
        #     0
        # )
        result = triton_concat_2D_jagged(
            calculated_max_seq_len,
            jt1.values(),
            jt2.values(),
            jt1.offsets(),
            jt2.offsets(),
        )
    with torch.cuda.nvtx.range("cudaop concat", color="purple"):
        result2 = jagged_2D_tensor_concat([jt1.values(), jt2.values()], [jt1.offsets(), jt2.offsets()], [max_len_jt1, max_len_jt2])
    assert torch.equal(result, result2[0])

    grad_for_merged_values = torch.randn_like(result)
    with torch.cuda.nvtx.range("triton Backward", color="purple"):
        result.backward(gradient=grad_for_merged_values)
    grad_for_jt1 = jt1.values().grad
    grad_for_jt2 = jt2.values().grad
    jt1.values().grad = None
    jt2.values().grad = None
    with torch.cuda.nvtx.range("cudaop Backward", color="purple"):
        result2[0].backward(gradient=grad_for_merged_values)
    assert torch.equal(jt1.values().grad, grad_for_jt1)
    assert torch.equal(jt2.values().grad, grad_for_jt2)


@pytest.mark.parametrize("num", [1, 2, 3, 4])
@pytest.mark.parametrize("batch_size,max_len,hidden_dim", [
    (2, 3, 4),
    (4, 5, 8),
    (1, 2, 16),
    (4, 10, 5)
])
def test_cudaop_vs_pytorch(num, batch_size, max_len, hidden_dim):

    with torch.cuda.nvtx.range("Test Setup", color="blue"):
        jt_list = [create_test_jagged_tensor(batch_size, max_len, hidden_dim) for _ in range(num)]
        max_seqlens = [max(jt.lengths()) for jt in jt_list]

        jt_list2 = [create_test_jagged_tensor(batch_size, max_len, hidden_dim) for _ in range(num)]
        max_seqlens2 = [max(jt.lengths()) for jt in jt_list2]

    from JaggedTensorOpFunction import jagged_2D_tensor_concat

    for _ in range(20):
        if _ % 2 == 0:
            result = jagged_2D_tensor_concat([jt.values() for jt in jt_list], [jt.offsets() for jt in jt_list], max_seqlens)
            result2 = concat_2D_jagged_tensors_pytorch(jt_list, max_seqlens)
        else:
            result = jagged_2D_tensor_concat([jt.values() for jt in jt_list2], [jt.offsets() for jt in jt_list2], max_seqlens2)
            result2 = concat_2D_jagged_tensors_pytorch(jt_list2, max_seqlens2)
        assert torch.equal(result[0], result2[0])
        assert torch.equal(result[1], result2[1])
    
    #cuda event start from
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()

    start.record()
    
    for _ in range(1000):
        if _ % 2 == 0:
            with torch.cuda.nvtx.range("Custom Implementation", color="purple"):
                result = jagged_2D_tensor_concat([jt.values() for jt in jt_list], [jt.offsets() for jt in jt_list], max_seqlens)
            with torch.cuda.nvtx.range("Original Implementation", color="green"):
                result2 = concat_2D_jagged_tensors_pytorch(jt_list, max_seqlens)
        else:
            with torch.cuda.nvtx.range("Custom Implementation", color="purple"):
                result = jagged_2D_tensor_concat([jt.values() for jt in jt_list2], [jt.offsets() for jt in jt_list2], max_seqlens2)
            with torch.cuda.nvtx.range("Original Implementation", color="green"):
                result2 = concat_2D_jagged_tensors_pytorch(jt_list2, max_seqlens2)
        assert torch.equal(result[0], result2[0])
        assert torch.equal(result[1], result2[1])

    end.record()

    torch.cuda.synchronize()

    elapsed_time_ms = start.elapsed_time(end)
    print(f"CUDA kernel time: {elapsed_time_ms:.3f} ms")
    


@pytest.mark.parametrize("batch_size,max_len,hidden_dim", [
    (2, 3, 4),
    (4, 5, 8),
])
def test_jagged_tensor_concat_autograd(batch_size, max_len, hidden_dim):

    jt1 = create_test_jagged_tensor(batch_size, max_len, hidden_dim)
    jt2 = create_test_jagged_tensor(batch_size, max_len + 1, hidden_dim)
    jt_list = [jt1, jt2]
    assert jt1.values().requires_grad, "jt1.values() should require grad"
    assert jt2.values().requires_grad, "jt2.values() should require grad"
    max_seqlens = [torch.max(jt.lengths()).item() if jt.lengths().numel() > 0 else 0 for jt in jt_list]
    merged_offsets = jt1.offsets().clone()
    merged_offsets.add_(jt2.offsets())
    total_length = merged_offsets[-1].item()
    merged_values = (
        torch.empty(
            (total_length, hidden_dim),
            dtype=jt1.values().dtype,
            device=jt1.values().device,
        )
        .requires_grad_(True)
    )
#cuda backward
    merged_values, merged_lengths = _JaggedTensorOpFunction.apply([jt1.offsets(), jt2.offsets()], max_seqlens, *[jt1.values(), jt2.values()])
    grad_for_merged_values = torch.randn_like(merged_values)
    assert merged_values.requires_grad, "merged_values should require grad"
    merged_values.backward(gradient=grad_for_merged_values)

    grad_for_jt1 = jt1.values().grad
    grad_for_jt2 = jt2.values().grad
    jt1.values().grad = None
    jt2.values().grad = None
#reference backward
    result = concat_2D_jagged_tensors_pytorch(jt_list, max_seqlens)
    result[0].backward(gradient=grad_for_merged_values)

    # Check if the gradients accumulated in jtN.values().grad match the expected ones.
    assert jt1.values().grad is not None, "Gradient for jt1.values() should exist"
    assert torch.allclose(jt1.values().grad, grad_for_jt1), \
        "Autograd gradient for cuda op does not match expected pytorch output"

    assert jt2.values().grad is not None, "Gradient for jt2.values() should exist"
    assert torch.allclose(jt2.values().grad, grad_for_jt2), \
        "Autograd gradient for cuda op does not match expected pytorch output"
    
    print(f"Autograd test passed for bs={batch_size}, ml={max_len}, hd={hidden_dim}")

#Test different data type
@pytest.mark.parametrize("batch_size,max_len,hidden_dim, dtype", [
    (2, 3, 4, torch.float64),
    (4, 5, 8, torch.float32),
    (2, 3, 4, torch.bfloat16),
    (2, 3, 4, torch.half)
])
def test_different_type(batch_size, max_len, hidden_dim, dtype):
    jt1 = create_test_jagged_tensor(batch_size, max_len, hidden_dim, dtype=dtype)
    jt2 = create_test_jagged_tensor(batch_size, max_len, hidden_dim, dtype=dtype)
    from JaggedTensorOpFunction import jagged_2D_tensor_concat
    result = jagged_2D_tensor_concat([jt1.values(), jt2.values()], [jt1.offsets(), jt2.offsets()], [3, 4])
    print(result)
    
#test all jagged tensor concat function
@pytest.mark.parametrize("num", [1, 2, 3, 4])
@pytest.mark.parametrize("batch_size,max_len,hidden_dim", [
    (2, 3, 4),
    (4, 5, 8),
    (1, 2, 16),
    (4, 10, 5)
])
def test_all_jagged_tensor_concat_function(num, batch_size, max_len, hidden_dim):
    if num == 2:
        test_triton_jagged_tensor_concat(batch_size, max_len, hidden_dim)
    else:
        test_cudaop_vs_pytorch(num, batch_size, max_len, hidden_dim)


if __name__ == "__main__":
    # test_jagged_tensor_concat_kernel()
    # test_jagged_tensor_concat_autograd(batch_size=2, max_len=3, hidden_dim=4)
    # test_different_type(batchsize=2, max_len=3, hidden_dim=4, dtype=torch.float64)
    test_triton_jagged_tensor_concat(batch_size=2, max_len=3, hidden_dim=4)