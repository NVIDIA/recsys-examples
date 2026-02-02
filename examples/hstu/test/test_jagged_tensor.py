from typing import List, Tuple

import pytest
import torch
import torch.distributed as dist
from commons.ops.cuda_ops.JaggedTensorOpFunction import jagged_2D_tensor_concat
from torchrec.sparse.jagged_tensor import JaggedTensor

backend = "nccl"
dist.init_process_group(backend=backend)
local_rank = dist.get_rank()  # for one node
world_size = dist.get_world_size()
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

import atexit


def cleanup_process_group():
    if dist.is_initialized():
        dist.destroy_process_group()


atexit.register(cleanup_process_group)


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
    # Clear GPU memory before creating large tensors
    if device.type == "cuda":
        torch.cuda.empty_cache()

    lengths = torch.randint(1, max_len + 1, size=(batch_size,), device=device)
    offsets = (
        torch.cat([torch.tensor([0], device=device), torch.cumsum(lengths, dim=0)])
        .to(device)
        .to(torch.int64)
    )

    offsets[1:] = torch.cumsum(lengths, dim=0, dtype=torch.int64)
    total_len = int(offsets[-1].item())
    values = (
        torch.empty(
            (total_len, hidden_dim),
            dtype=dtype,
            device=device,
        )
        .uniform_(-1.0, 1.0)
        .requires_grad_(True)
    )

    return JaggedTensor(values=values, lengths=lengths, offsets=offsets)


def print_jagged_tensor(jt_list):
    for jt in jt_list:
        print("Values:", jt.values())
        print("Lengths:", jt.lengths())
        print("Offsets:", jt.offsets(), "\n")


# Test jagged tensor creation
@pytest.mark.parametrize(
    "batch_size,max_len,hidden_dim", [(2, 3, 4), (4, 5, 8), (1, 2, 16), (4, 10, 5)]
)
def test_jagged_tensor_creation(batch_size, max_len, hidden_dim):
    jt = create_test_jagged_tensor(batch_size, max_len, hidden_dim)
    assert jt.values().shape[1] == hidden_dim
    assert jt.lengths().shape[0] == batch_size
    assert jt.offsets().shape[0] == batch_size + 1


@pytest.mark.parametrize("batch_size,max_len,hidden_dim", [(2, 3, 4)])
def test_triton_jagged_tensor_concat(batch_size, max_len, hidden_dim):
    with torch.cuda.nvtx.range("Test Setup", color="blue"):
        jt1 = create_test_jagged_tensor(batch_size, max_len, hidden_dim)
        jt2 = create_test_jagged_tensor(batch_size, max_len + 1, hidden_dim)
        max_len_jt1 = torch.max(jt1.lengths()).item()
        max_len_jt2 = torch.max(jt2.lengths()).item()
        calculated_max_seq_len = max_len_jt1 + max_len_jt2
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
        result2 = jagged_2D_tensor_concat(
            [jt1.values(), jt2.values()],
            [jt1.offsets(), jt2.offsets()],
            [max_len_jt1, max_len_jt2],
        )

    assert torch.equal(result, result2[0])

    grad_for_merged_values = torch.randn_like(result)

    with torch.cuda.nvtx.range("cudaop Backward", color="purple"):
        result2[0].backward(gradient=grad_for_merged_values)
    grad_for_jt1 = jt1.values().grad
    grad_for_jt2 = jt2.values().grad
    jt1.values().grad = None
    jt2.values().grad = None
    with torch.cuda.nvtx.range("triton Backward", color="purple"):
        result.backward(gradient=grad_for_merged_values)
    assert torch.equal(jt1.values().grad, grad_for_jt1)
    assert torch.equal(jt2.values().grad, grad_for_jt2)


# Verify result correctness
@pytest.mark.parametrize("num", [1, 2, 127, 129, 256])
@pytest.mark.parametrize(
    "batch_size,max_len,hidden_dim",
    [
        (2, 3, 4),
        (4, 5, 8),
        (1, 2, 16),
        (4, 10, 5),
        (32, 1, 1),
        (32, 2048, 128),
        (40, 256, 256),
        (1, 1, 1),  # Minimum possible size
        (1, 1, 2),  # Minimum batch, minimum length
        (2, 1, 1),  # Minimum hidden_dim
        (1, 1000, 16),  # Long sequence
        (100, 10, 8),  # Large batch size
        (4, 4, 512),  # Large hidden dimension
        # (32, 8192, 128)
    ],
)
def test_forward_backward_verification(
    num, batch_size, max_len, hidden_dim, dtype=torch.half
):
    jt_list = [
        create_test_jagged_tensor(batch_size, max_len, hidden_dim, dtype)
        for _ in range(num)
    ]
    max_seqlens = [max(jt.lengths()) for jt in jt_list]
    max(max_seqlens)
    from commons.ops.cuda_ops.JaggedTensorOpFunction import jagged_2D_tensor_concat

    if num == 2:
        from ops.triton_ops.triton_jagged import triton_concat_2D_jagged

        max_len_jt1 = torch.max(jt_list[0].lengths()).item()
        max_len_jt2 = torch.max(jt_list[1].lengths()).item()
        calculated_max_seq_len = max_len_jt1 + max_len_jt2
        result = jagged_2D_tensor_concat(
            [jt_list[0].values(), jt_list[1].values()],
            [jt_list[0].offsets(), jt_list[1].offsets()],
            [max_len_jt1, max_len_jt2],
        )
        result2 = triton_concat_2D_jagged(
            calculated_max_seq_len,
            jt_list[0].values(),
            jt_list[1].values(),
            jt_list[0].offsets(),
            jt_list[1].offsets(),
        )
        assert torch.equal(result2, result[0])
        grad_for_merged_values = torch.randn_like(result[0])
        with torch.cuda.nvtx.range("cudaop Backward", color="purple"):
            result[0].backward(gradient=grad_for_merged_values)
        grad_for_jt1 = jt_list[0].values().grad
        grad_for_jt2 = jt_list[1].values().grad
        jt_list[0].values().grad = None
        jt_list[1].values().grad = None
        with torch.cuda.nvtx.range("triton Backward", color="purple"):
            result2.backward(gradient=grad_for_merged_values)
        assert torch.equal(jt_list[0].values().grad, grad_for_jt1)
        assert torch.equal(jt_list[1].values().grad, grad_for_jt2)

    result = jagged_2D_tensor_concat(
        [jt.values() for jt in jt_list],
        [jt.offsets() for jt in jt_list],
        max_seqlens,
    )
    result2 = concat_2D_jagged_tensors_pytorch(jt_list, max_seqlens)
    assert torch.equal(result[0], result2[0])
    assert torch.equal(result[1], result2[1])
    # Verify backward propagation correctness
    grad_for_merged_values = torch.randn_like(result[0])
    assert result[0].requires_grad, "merged_values should require grad"
    # Use torch.autograd.grad to get gradients (avoid modifying original tensor's .grad attribute)
    input_tensors = [jt.values() for jt in jt_list]
    grad_list_cuda = torch.autograd.grad(
        outputs=result[0],
        inputs=input_tensors,
        grad_outputs=grad_for_merged_values,
        retain_graph=False,
        create_graph=False,
    )

    grad_list_pytorch = torch.autograd.grad(
        outputs=result2[0],
        inputs=input_tensors,
        grad_outputs=grad_for_merged_values,
        retain_graph=False,
        create_graph=False,
    )

    # Check if the gradients match
    for i, (grad_cuda, grad_pytorch) in enumerate(
        zip(grad_list_cuda, grad_list_pytorch)
    ):
        assert grad_cuda is not None, f"CUDA gradient for tensor {i} should exist"
        assert grad_pytorch is not None, f"PyTorch gradient for tensor {i} should exist"
        assert torch.allclose(
            grad_cuda, grad_pytorch, atol=1e-5
        ), f"Autograd gradient for cuda op does not match expected pytorch output for tensor {i}"


@pytest.mark.parametrize("num", [2, 3, 4])
@pytest.mark.parametrize(
    "batch_size,max_len,hidden_dim",
    [
        (2, 3, 4),
        (4, 5, 8),
        (8, 10, 16),
        (16, 32, 32),
        (32, 64, 64),
        (64, 128, 128),
    ],
)
def test_cudaop_vs_pytorch_benchmark(
    num, batch_size, max_len, hidden_dim, dtype=torch.float32
):
    # Set random seeds for reproducibility
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    import numpy as np

    np.random.seed(0)
    import random

    random.seed(0)

    with torch.cuda.nvtx.range("Test Setup", color="blue"):
        jt_list = [
            create_test_jagged_tensor(batch_size, max_len, hidden_dim)
            for _ in range(num)
        ]
        max_seqlens = [max(jt.lengths()) for jt in jt_list]

        jt_list2 = [
            create_test_jagged_tensor(batch_size, max_len, hidden_dim)
            for _ in range(num)
        ]
        max_seqlens2 = [max(jt.lengths()) for jt in jt_list2]
    max_seqlen1 = max(max_seqlens)
    max_seqlen2 = max(max_seqlens2)
    from commons.ops.cuda_ops.JaggedTensorOpFunction import jagged_2D_tensor_concat

    for _ in range(20):
        if _ % 2 == 0:
            result = jagged_2D_tensor_concat(
                [jt.values() for jt in jt_list],
                [jt.offsets() for jt in jt_list],
                max_seqlens,
            )
            result2 = concat_2D_jagged_tensors_pytorch(jt_list, max_seqlens)
        else:
            result = jagged_2D_tensor_concat(
                [jt.values() for jt in jt_list2],
                [jt.offsets() for jt in jt_list2],
                max_seqlens2,
            )
            result2 = concat_2D_jagged_tensors_pytorch(jt_list2, max_seqlens2)
        assert torch.equal(result[0], result2[0])
        assert torch.equal(result[1], result2[1])

    # cuda event start from
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()

    start.record()
    for _ in range(100):
        current_jt_list = jt_list if _ % 2 == 0 else jt_list2
        current_max_seqlens = max_seqlens if _ % 2 == 0 else max_seqlens2
        current_max_seqlen = max_seqlen1 if _ % 2 == 0 else max_seqlen2
        with torch.cuda.nvtx.range("Custom Implementation", color="purple"):
            result_cudaop = jagged_2D_tensor_concat(
                [jt.values() for jt in current_jt_list],
                [jt.offsets() for jt in current_jt_list],
                current_max_seqlens,
            )
    end.record()
    torch.cuda.synchronize()
    cuda_forward_time = start.elapsed_time(end) / 100
    print(f"CUDA kernel time: {cuda_forward_time:.3f} ms")
    # Calculate bytes per element based on dtype
    if dtype == torch.float32:
        bytes_per_element = 4
    elif dtype in [torch.float16, torch.bfloat16, torch.half]:
        bytes_per_element = 2
    elif dtype == torch.float64:
        bytes_per_element = 8
    else:
        bytes_per_element = 4
    total_elements = sum(jt.values().numel() for jt in jt_list)
    total_bytes = total_elements * bytes_per_element
    print(
        f"Test config: num={num}, batch_size={batch_size}, max_len={max_len}, hidden_dim={hidden_dim}"
    )
    print(f"Data size: {total_elements:,} elements, {total_bytes/1024/1024:.3f} MB")
    cuda_throughput = total_bytes / cuda_forward_time / 1e3  # MB/s
    print(f"cuda_throughput: {cuda_throughput:.3f} MB/s")

    # forward benchmark
    torch.cuda.synchronize()

    start.record()
    for _ in range(100):
        current_jt_list = jt_list if _ % 2 == 0 else jt_list2
        current_max_seqlens = max_seqlens if _ % 2 == 0 else max_seqlens2
        current_max_seqlen = max_seqlen1 if _ % 2 == 0 else max_seqlen2
        with torch.cuda.nvtx.range("Custom Implementation", color="purple"):
            result_pytorch = concat_2D_jagged_tensors_pytorch(
                current_jt_list, current_max_seqlens
            )
    end.record()
    torch.cuda.synchronize()
    pytorch_forward_time = start.elapsed_time(end) / 100
    print(f"Pytorch time: {pytorch_forward_time:.3f} ms")
    pytorch_total_bytes = sum(jt.values().numel() for jt in jt_list) * bytes_per_element
    pytorch_throughput = pytorch_total_bytes / pytorch_forward_time / 1e3
    print(f"pytorch_throughput: {pytorch_throughput:.3f} MB/s")

    # backward benchmark
    result_cudaop = jagged_2D_tensor_concat(
        [jt.values() for jt in jt_list],
        [jt.offsets() for jt in jt_list],
        max_seqlens,
    )
    grad_for_merged_values = torch.randn_like(result_cudaop[0])
    assert result_cudaop[0].requires_grad, "merged_values should require grad"

    torch.cuda.synchronize()
    start.record()
    for i in range(100):
        # Use torch.autograd.grad to avoid gradient accumulation and graph retention issues
        input_tensors = [jt.values() for jt in jt_list]
        grads = torch.autograd.grad(
            outputs=result_cudaop[0],
            inputs=input_tensors,
            grad_outputs=grad_for_merged_values,
            retain_graph=True,  # Retain computation graph for next calculation
            create_graph=False,  # Don't create computation graph for gradients
            only_inputs=True,  # Only compute gradients for inputs
        )
    end.record()
    torch.cuda.synchronize()
    cuda_backward_time = start.elapsed_time(end) / 100
    print(f"CUDA kernel backward time: {cuda_backward_time:.3f} ms")

    # reference backward benchmark
    result_pytorch = concat_2D_jagged_tensors_pytorch(jt_list, max_seqlens)

    torch.cuda.synchronize()
    start.record()
    for i in range(100):
        # Use torch.autograd.grad to avoid gradient accumulation and graph retention issues
        input_tensors = [jt.values() for jt in jt_list]
        grads = torch.autograd.grad(
            outputs=result_pytorch[0],
            inputs=input_tensors,
            grad_outputs=grad_for_merged_values,
            retain_graph=True,  # Retain computation graph for next calculation
            create_graph=False,  # Don't create computation graph for gradients
            only_inputs=True,  # Only compute gradients for inputs
        )
    end.record()
    torch.cuda.synchronize()
    pytorch_backward_time = start.elapsed_time(end) / 100
    print(f"PyTorch backward time: {pytorch_backward_time:.3f} ms")

    # Calculate throughput
    cuda_forward_throughput = total_bytes / cuda_forward_time / 1e3  # MB/s
    pytorch_forward_throughput = (
        pytorch_total_bytes / pytorch_forward_time / 1e3
    )  # MB/s
    cuda_backward_throughput = total_bytes / cuda_backward_time / 1e3  # MB/s
    pytorch_backward_throughput = (
        pytorch_total_bytes / pytorch_backward_time / 1e3
    )  # MB/s

    print(f"\nPerformance Results:")
    print(
        f"{'Operation':<15} {'CUDA (ms)':<12} {'PyTorch (ms)':<12} {'Speedup':<10} {'CUDA MB/s':<12} {'PyTorch MB/s':<12}"
    )
    print("-" * 80)

    forward_speedup = pytorch_forward_time / cuda_forward_time
    backward_speedup = pytorch_backward_time / cuda_backward_time

    print(
        f"{'Forward':<15} {cuda_forward_time:<12.3f} {pytorch_forward_time:<12.3f} {forward_speedup:<10.2f}x {cuda_forward_throughput:<12.1f} {pytorch_forward_throughput:<12.1f}"
    )
    print(
        f"{'Backward':<15} {cuda_backward_time:<12.3f} {pytorch_backward_time:<12.3f} {backward_speedup:<10.2f}x {cuda_backward_throughput:<12.1f} {pytorch_backward_throughput:<12.1f}"
    )

    total_cuda_time = cuda_forward_time + cuda_backward_time
    total_pytorch_time = pytorch_forward_time + pytorch_backward_time
    total_speedup = total_pytorch_time / total_cuda_time

    print(
        f"{'Total':<15} {total_cuda_time:<12.3f} {total_pytorch_time:<12.3f} {total_speedup:<10.2f}x"
    )

    if forward_speedup > 1:
        print(f"CUDA is {forward_speedup:.2f}x faster than PyTorch for forward pass")
    else:
        print(f"PyTorch is {1/forward_speedup:.2f}x faster than CUDA for forward pass")

    if backward_speedup > 1:
        print(f"CUDA is {backward_speedup:.2f}x faster than PyTorch for backward pass")
    else:
        print(
            f"PyTorch is {1/backward_speedup:.2f}x faster than CUDA for backward pass"
        )


# Test different data type
@pytest.mark.parametrize(
    "batch_size,max_len,hidden_dim, dtype",
    [
        (2, 3, 4, torch.float64),
        (4, 5, 8, torch.float32),
        (2, 3, 4, torch.bfloat16),
        (2, 3, 4, torch.half),
    ],
)
def test_different_type(batch_size, max_len, hidden_dim, dtype):
    jt1 = create_test_jagged_tensor(batch_size, max_len, hidden_dim, dtype=dtype)
    jt2 = create_test_jagged_tensor(batch_size, max_len, hidden_dim, dtype=dtype)
    from commons.ops.cuda_ops.JaggedTensorOpFunction import jagged_2D_tensor_concat

    result = jagged_2D_tensor_concat(
        [jt1.values(), jt2.values()], [jt1.offsets(), jt2.offsets()], [3, 4]
    )
    print(result)


@pytest.mark.parametrize(
    "batch_size,max_len,hidden_dim",
    [
        (1, 8, 1),
        (32, 1024, 128),
        (32, 2048, 128),
        (32, 4096, 128),
        (32, 8192, 128),
    ],
)
def test_cudaop_vs_tritonop_benchmark(
    batch_size, max_len, hidden_dim, dtype=torch.float32
):
    from commons.ops.cuda_ops.JaggedTensorOpFunction import jagged_2D_tensor_concat
    from ops.triton_ops.triton_jagged import triton_concat_2D_jagged

    # Set random seeds for reproducibility
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    import numpy as np

    np.random.seed(0)
    import random

    random.seed(0)

    with torch.cuda.nvtx.range("Test Setup", color="blue"):
        jt_list1 = [
            create_test_jagged_tensor(batch_size, max_len, hidden_dim, dtype)
            for _ in range(2)
        ]
        jt_list2 = [
            create_test_jagged_tensor(batch_size, max_len, hidden_dim, dtype)
            for _ in range(2)
        ]

        max_len_jt1_1 = torch.max(jt_list1[0].lengths()).item()
        max_len_jt2_1 = torch.max(jt_list1[1].lengths()).item()
        max_len_jt1_2 = torch.max(jt_list2[0].lengths()).item()
        max_len_jt2_2 = torch.max(jt_list2[1].lengths()).item()

        calculated_max_seq_len1 = max_len_jt1_1 + max_len_jt2_1
        calculated_max_seq_len2 = max_len_jt1_2 + max_len_jt2_2

        max_seqlen1 = max(max_len_jt1_1, max_len_jt2_1)
        max_seqlen2 = max(max_len_jt1_2, max_len_jt2_2)

        # GEMM setup
        size = 8192
        a = torch.randn(size, size, device="cuda", dtype=torch.float32)
        b = torch.randn(size, size, device="cuda", dtype=torch.float32)
    # Warmup
    # todo: backward and gemm need warmup
    print("Warming up...")
    for _ in range(10):
        current_jt_list = jt_list1 if _ % 2 == 0 else jt_list2
        current_max_len_1 = max_len_jt1_1 if _ % 2 == 0 else max_len_jt1_2
        current_max_len_2 = max_len_jt2_1 if _ % 2 == 0 else max_len_jt2_2
        current_max_seqlen = max_seqlen1 if _ % 2 == 0 else max_seqlen2
        current_calculated_max_seq_len = (
            calculated_max_seq_len1 if _ % 2 == 0 else calculated_max_seq_len2
        )

        result = jagged_2D_tensor_concat(
            [current_jt_list[0].values(), current_jt_list[1].values()],
            [current_jt_list[0].offsets(), current_jt_list[1].offsets()],
            [current_max_len_1, current_max_len_2],
        )
        result2 = triton_concat_2D_jagged(
            current_calculated_max_seq_len,
            current_jt_list[0].values(),
            current_jt_list[1].values(),
            current_jt_list[0].offsets(),
            current_jt_list[1].offsets(),
        )
        grad_for_merged_values = torch.randn_like(result[0])
        with torch.cuda.nvtx.range("cudawarmup", color="purple"):
            result[0].backward(gradient=grad_for_merged_values)
        current_jt_list[0].values().grad
        current_jt_list[1].values().grad
        current_jt_list[0].values().grad = None
        current_jt_list[1].values().grad = None
        with torch.cuda.nvtx.range("triton warmup", color="purple"):
            result2.backward(gradient=grad_for_merged_values)
        # GEMM warmup
        with torch.cuda.nvtx.range("GEMM warmup", color="purple"):
            for _ in range(10):
                torch.mm(a, b)

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.cuda.nvtx.range("GEMM", color="purple"):
        for _ in range(10):
            torch.mm(a, b)

    start.record()
    for _ in range(100):
        current_jt_list = jt_list1 if _ % 2 == 0 else jt_list2
        current_max_len_1 = max_len_jt1_1 if _ % 2 == 0 else max_len_jt1_2
        current_max_len_2 = max_len_jt2_1 if _ % 2 == 0 else max_len_jt2_2
        current_max_seqlen = max_seqlen1 if _ % 2 == 0 else max_seqlen2

        with torch.cuda.nvtx.range("CUDA Forward", color="red"):
            cuda_result = jagged_2D_tensor_concat(
                [current_jt_list[0].values(), current_jt_list[1].values()],
                [current_jt_list[0].offsets(), current_jt_list[1].offsets()],
                [current_max_len_1, current_max_len_2],
            )
    end.record()
    torch.cuda.synchronize()
    cuda_forward_time = start.elapsed_time(end) / 100

    # Before benchmark add workload GEMM
    with torch.cuda.nvtx.range("GEMM", color="purple"):
        for _ in range(10):
            torch.mm(a, b)

    # Triton Forward Benchmark
    start.record()
    for _ in range(100):
        current_jt_list = jt_list1 if _ % 2 == 0 else jt_list2
        current_calculated_max_seq_len = (
            calculated_max_seq_len1 if _ % 2 == 0 else calculated_max_seq_len2
        )

        with torch.cuda.nvtx.range("Triton Forward", color="green"):
            triton_result = triton_concat_2D_jagged(
                current_calculated_max_seq_len,
                current_jt_list[0].values(),
                current_jt_list[1].values(),
                current_jt_list[0].offsets(),
                current_jt_list[1].offsets(),
            )
    end.record()
    torch.cuda.synchronize()
    triton_forward_time = start.elapsed_time(end) / 100

    with torch.cuda.nvtx.range("GEMM", color="purple"):
        for _ in range(10):
            torch.mm(a, b)
    # CUDA Backward Benchmark
    cuda_result_for_backward = jagged_2D_tensor_concat(
        [jt_list1[0].values(), jt_list1[1].values()],
        [jt_list1[0].offsets(), jt_list1[1].offsets()],
        [max_len_jt1_1, max_len_jt2_1],
    )
    grad_for_backward = torch.randn_like(cuda_result_for_backward[0])

    start.record()
    for _ in range(100):
        input_tensors = [jt_list1[0].values(), jt_list1[1].values()]
        with torch.cuda.nvtx.range("CUDA Backward", color="red"):
            grads = torch.autograd.grad(
                outputs=cuda_result_for_backward[0],
                inputs=input_tensors,
                grad_outputs=grad_for_backward,
                retain_graph=True,
                create_graph=False,
                only_inputs=True,
            )
    end.record()
    torch.cuda.synchronize()
    cuda_backward_time = start.elapsed_time(end) / 100

    with torch.cuda.nvtx.range("GEMM", color="purple"):
        for _ in range(10):
            torch.mm(a, b)
    # Triton Backward Benchmark
    triton_result_for_backward = triton_concat_2D_jagged(
        calculated_max_seq_len1,
        jt_list1[0].values(),
        jt_list1[1].values(),
        jt_list1[0].offsets(),
        jt_list1[1].offsets(),
    )

    start.record()
    for _ in range(100):
        input_tensors = [jt_list1[0].values(), jt_list1[1].values()]
        with torch.cuda.nvtx.range("Triton Backward", color="green"):
            grads = torch.autograd.grad(
                outputs=triton_result_for_backward,
                inputs=input_tensors,
                grad_outputs=grad_for_backward,
                retain_graph=True,
                create_graph=False,
                only_inputs=True,
            )
    end.record()
    torch.cuda.synchronize()
    triton_backward_time = start.elapsed_time(end) / 100

    # Calculate throughput
    if dtype == torch.float32:
        bytes_per_element = 4
    elif dtype in [torch.float16, torch.bfloat16, torch.half]:
        bytes_per_element = 2
    elif dtype == torch.float64:
        bytes_per_element = 8
    else:
        bytes_per_element = 4
    total_elements = sum(jt.values().numel() for jt in jt_list1)
    total_bytes = total_elements * bytes_per_element
    print(
        f"Test config: num=2, batch_size={batch_size}, max_len={max_len}, hidden_dim={hidden_dim}"
    )
    print(f"Data size: {total_elements:,} elements, {total_bytes/1024/1024:.3f} MB")

    cuda_forward_throughput = total_bytes / cuda_forward_time / 1e3  # MB/s
    triton_forward_throughput = total_bytes / triton_forward_time / 1e3  # MB/s
    cuda_backward_throughput = total_bytes / cuda_backward_time / 1e3  # MB/s
    triton_backward_throughput = total_bytes / triton_backward_time / 1e3  # MB/s

    # Print results
    print(f"\nPerformance Results:")
    print(
        f"{'Operation':<15} {'CUDA (ms)':<12} {'Triton (ms)':<12} {'Speedup':<10} {'CUDA MB/s':<12} {'Triton MB/s':<12}"
    )
    print("-" * 80)

    forward_speedup = triton_forward_time / cuda_forward_time
    backward_speedup = triton_backward_time / cuda_backward_time

    print(
        f"{'Forward':<15} {cuda_forward_time:<12.3f} {triton_forward_time:<12.3f} {forward_speedup:<10.2f}x {cuda_forward_throughput:<12.1f} {triton_forward_throughput:<12.1f}"
    )
    print(
        f"{'Backward':<15} {cuda_backward_time:<12.3f} {triton_backward_time:<12.3f} {backward_speedup:<10.2f}x {cuda_backward_throughput:<12.1f} {triton_backward_throughput:<12.1f}"
    )

    total_cuda_time = cuda_forward_time + cuda_backward_time
    total_triton_time = triton_forward_time + triton_backward_time
    total_speedup = total_triton_time / total_cuda_time

    print(
        f"{'Total':<15} {total_cuda_time:<12.3f} {total_triton_time:<12.3f} {total_speedup:<10.2f}x"
    )

    if forward_speedup > 1:
        print(f"CUDA is {forward_speedup:.2f}x faster than Triton for forward pass")
    else:
        print(f"Triton is {1/forward_speedup:.2f}x faster than CUDA for forward pass")

    if backward_speedup > 1:
        print(f"CUDA is {backward_speedup:.2f}x faster than Triton for backward pass")
    else:
        print(f"Triton is {1/backward_speedup:.2f}x faster than CUDA for backward pass")
