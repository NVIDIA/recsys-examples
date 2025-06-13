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
        # .round(decimals=2)  # 四舍五入到两位小数
        .requires_grad_(True)
    )

    return JaggedTensor(
        values=values,
        lengths=lengths,
        offsets=offsets
    )
def print_jagged_tensor(jt_list):
    for jt in jt_list:
        print("Values:", jt.values())
        print("Lengths:", jt.lengths())
        print("Offsets:", jt.offsets(), "\n")
    

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

    with torch.cuda.nvtx.range("triton concat", color="purple"):
        result = triton_concat_2D_jagged(
            calculated_max_seq_len,
            jt1.values(),
            jt2.values(),
            jt1.offsets(),
            jt2.offsets(),
        )
    with torch.cuda.nvtx.range("cudaop concat", color="purple"):
        result2 = jagged_2D_tensor_concat([jt1.values(), jt2.values()], [jt1.offsets(), jt2.offsets()], [max_len_jt1, max_len_jt2], max(max_len_jt1, max_len_jt2))
    
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

#验证结果正确性
#@pytest.mark.parametrize("num", [2, 3, 4])
@pytest.mark.parametrize("num", [256])
@pytest.mark.parametrize("batch_size,max_len,hidden_dim", [
    (2, 3, 4),
    (4, 5, 8),
    (1, 2, 16),
    (4, 10, 5),
    (32, 1, 1),
    (40, 256, 256)
])
def test_forward_backward_verification(num, batch_size, max_len, hidden_dim):
# add triton here
    jt_list = [create_test_jagged_tensor(batch_size, max_len, hidden_dim) for _ in range(num)]
    max_seqlens = [max(jt.lengths()) for jt in jt_list]
    max_seqlen1 = max(max_seqlens)
    from JaggedTensorOpFunction import jagged_2D_tensor_concat

    result = jagged_2D_tensor_concat([jt.values() for jt in jt_list], [jt.offsets() for jt in jt_list], max_seqlens, max_seqlen1)
    result2 = concat_2D_jagged_tensors_pytorch(jt_list, max_seqlens)
    assert torch.equal(result[0], result2[0])
    assert torch.equal(result[1], result2[1])
    # 验证一次反向传播的正确性
    grad_for_merged_values = torch.randn_like(result[0])
    assert result[0].requires_grad, "merged_values should require grad" 
    # 使用 torch.autograd.grad 获取梯度（避免修改原始张量的 .grad 属性）
    input_tensors = [jt.values() for jt in jt_list]
    grad_list_cuda = torch.autograd.grad(
        outputs=result[0],
        inputs=input_tensors,
        grad_outputs=grad_for_merged_values,
        retain_graph=False,
        create_graph=False
    )
    
    grad_list_pytorch = torch.autograd.grad(
        outputs=result2[0],
        inputs=input_tensors,
        grad_outputs=grad_for_merged_values,
        retain_graph=False,
        create_graph=False
    )

    # Check if the gradients match
    for i, (grad_cuda, grad_pytorch) in enumerate(zip(grad_list_cuda, grad_list_pytorch)):
        assert grad_cuda is not None, f"CUDA gradient for tensor {i} should exist"
        assert grad_pytorch is not None, f"PyTorch gradient for tensor {i} should exist"
        assert torch.allclose(grad_cuda, grad_pytorch, atol=1e-5), \
            f"Autograd gradient for cuda op does not match expected pytorch output for tensor {i}"    

@pytest.mark.parametrize("num", [2, 3, 4])
@pytest.mark.parametrize("batch_size,max_len,hidden_dim", [
    (2, 3, 4),
    (4, 5, 8),
    (1, 2, 16),
])
def test_cudaop_vs_pytorch_benchmark(num, batch_size, max_len, hidden_dim, dtype=torch.float32):
#todo: move triton here
    with torch.cuda.nvtx.range("Test Setup", color="blue"):
        jt_list = [create_test_jagged_tensor(batch_size, max_len, hidden_dim) for _ in range(num)]
        max_seqlens = [max(jt.lengths()) for jt in jt_list]

        jt_list2 = [create_test_jagged_tensor(batch_size, max_len, hidden_dim) for _ in range(num)]
        max_seqlens2 = [max(jt.lengths()) for jt in jt_list2]
    max_seqlen1 = max(max_seqlens)
    max_seqlen2 = max(max_seqlens2)
    from JaggedTensorOpFunction import jagged_2D_tensor_concat

    for _ in range(20):
        if _ % 2 == 0:
            result = jagged_2D_tensor_concat([jt.values() for jt in jt_list], [jt.offsets() for jt in jt_list], max_seqlens, max_seqlen1)
            result2 = concat_2D_jagged_tensors_pytorch(jt_list, max_seqlens)
        else:
            result = jagged_2D_tensor_concat([jt.values() for jt in jt_list2], [jt.offsets() for jt in jt_list2], max_seqlens2, max_seqlen2)
            result2 = concat_2D_jagged_tensors_pytorch(jt_list2, max_seqlens2)
        assert torch.equal(result[0], result2[0])
        assert torch.equal(result[1], result2[1])
    
    #cuda event start from
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
# benchmark:
# 1000 jagged_2D_tensor_concat 保持
    torch.cuda.synchronize()

    start.record()
    for _ in range(100):
        current_jt_list = jt_list if _ % 2 == 0 else jt_list2
        current_max_seqlens = max_seqlens if _ % 2 == 0 else max_seqlens2
        current_max_seqlen = max_seqlen1 if _ % 2 == 0 else max_seqlen2
        with torch.cuda.nvtx.range("Custom Implementation", color="purple"):
            result_cudaop = jagged_2D_tensor_concat([jt.values() for jt in current_jt_list], [jt.offsets() for jt in current_jt_list], current_max_seqlens, current_max_seqlen)
    end.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start.elapsed_time(end) / 100
    print(f"CUDA kernel time: {elapsed_time_ms:.3f} ms")
    bytes = 4 if dtype == torch.float32 else 2
    throughput = sum(jt.values().numel() for jt in jt_list) * bytes / elapsed_time_ms / 1000
    print(f"Throughput: {throughput:.3f} GB/s")
##throughput
#value size  sum (jt.values().numel() for jt in jt_list) * bytes
#forward benchmark
    torch.cuda.synchronize()

    start.record()
    for _ in range(100):
        current_jt_list = jt_list if _ % 2 == 0 else jt_list2
        current_max_seqlens = max_seqlens if _ % 2 == 0 else max_seqlens2
        current_max_seqlen = max_seqlen1 if _ % 2 == 0 else max_seqlen2
        with torch.cuda.nvtx.range("Custom Implementation", color="purple"):
            result_pytorch = concat_2D_jagged_tensors_pytorch(current_jt_list, current_max_seqlens)
    end.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start.elapsed_time(end) / 100
    print(f"Pytorch time: {elapsed_time_ms:.3f} ms")
    throughput = sum(jt.values().numel() for jt in jt_list) * bytes / elapsed_time_ms / 1000
    print(f"Throughput: {throughput:.3f} GB/s")


#backward benchmark
    result_cudaop = jagged_2D_tensor_concat([jt.values() for jt in jt_list], [jt.offsets() for jt in jt_list], max_seqlens, max_seqlen1)
    grad_for_merged_values = torch.randn_like(result_cudaop[0])
    assert result_cudaop[0].requires_grad, "merged_values should require grad"

    torch.cuda.synchronize()
    start.record()
    for i in range(100):
        # 使用 torch.autograd.grad 避免梯度累积和计算图保留问题
        input_tensors = [jt.values() for jt in jt_list]
        grads = torch.autograd.grad(
            outputs=result_cudaop[0],
            inputs=input_tensors,
            grad_outputs=grad_for_merged_values,
            retain_graph=True,  # 保留计算图用于下次计算
            create_graph=False,  # 不创建梯度的计算图
            only_inputs=True  # 只计算输入的梯度
        )
        # 注意：torch.autograd.grad 不会自动累积梯度到 .grad 属性中
    end.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start.elapsed_time(end) / 100
    print(f"CUDA kernel backward time: {elapsed_time_ms:.3f} ms")

#reference backward benchmark
    result_pytorch = concat_2D_jagged_tensors_pytorch(jt_list, max_seqlens)

    torch.cuda.synchronize()
    start.record()
    for i in range(100):
        # 使用 torch.autograd.grad 避免梯度累积和计算图保留问题
        input_tensors = [jt.values() for jt in jt_list]
        grads = torch.autograd.grad(
            outputs=result_pytorch[0],
            inputs=input_tensors,
            grad_outputs=grad_for_merged_values,
            retain_graph=True,  # 保留计算图用于下次计算
            create_graph=False,  # 不创建梯度的计算图
            only_inputs=True  # 只计算输入的梯度
        )
        # 注意：torch.autograd.grad 不会自动累积梯度到 .grad 属性中
    end.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start.elapsed_time(end) / 100
    print(f"PyTorch backward time: {elapsed_time_ms:.3f} ms")




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
    result = jagged_2D_tensor_concat([jt1.values(), jt2.values()], [jt1.offsets(), jt2.offsets()], [3, 4], 4)
    print(result)
    
