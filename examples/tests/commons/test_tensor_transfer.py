import pytest
import torch
from commons.utils.tensor_transfer import (
    copy_tensor_to_pinned_cpu,
    tensor_from_cpu_array_like,
    tensor_to_cpu_list,
)


def test_tensor_from_cpu_array_like_to_cpu() -> None:
    tensor = tensor_from_cpu_array_like(
        [[1, 2, 3], [4, 5, 6]],
        dtype=torch.int64,
        device="cpu",
    )

    assert tensor.device.type == "cpu"
    assert tensor.dtype == torch.int64
    assert tensor.tolist() == [[1, 2, 3], [4, 5, 6]]


def test_tensor_from_cpu_array_like_from_tensor_converts_dtype() -> None:
    source = torch.tensor([1, 2, 3], dtype=torch.int32)

    tensor = tensor_from_cpu_array_like(source, dtype=torch.float32, device="cpu")

    assert tensor.device.type == "cpu"
    assert tensor.dtype == torch.float32
    assert tensor.tolist() == [1.0, 2.0, 3.0]


def test_tensor_to_cpu_list_cpu_tensor() -> None:
    tensor = torch.tensor([[1.5, 2.5], [3.5, 4.5]], dtype=torch.float32)

    assert tensor_to_cpu_list(tensor) == [[1.5, 2.5], [3.5, 4.5]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_copy_tensor_to_pinned_cpu_from_cpu() -> None:
    tensor = torch.arange(4, dtype=torch.int64)

    cpu = copy_tensor_to_pinned_cpu(tensor)

    assert cpu.device.type == "cpu"
    assert cpu.is_pinned()
    assert cpu.tolist() == [0, 1, 2, 3]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_tensor_from_cpu_array_like_to_cuda() -> None:
    tensor = tensor_from_cpu_array_like(
        [[1, 2], [3, 4]],
        dtype=torch.int64,
        device=torch.device("cuda"),
    )

    assert tensor.is_cuda
    assert tensor.dtype == torch.int64
    assert tensor.cpu().tolist() == [[1, 2], [3, 4]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_tensor_from_cpu_array_like_to_cuda_without_pinned_staging() -> None:
    source = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)

    tensor = tensor_from_cpu_array_like(
        source,
        dtype=torch.int64,
        device=torch.device("cuda"),
        pinned_staging=False,
    )

    assert tensor.is_cuda
    assert tensor.dtype == torch.int64
    assert tensor.cpu().tolist() == [[1, 2], [3, 4]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_tensor_from_cpu_array_like_rejects_cuda_tensor() -> None:
    source = torch.arange(4, device="cuda", dtype=torch.int64)

    with pytest.raises(ValueError, match="CPU-side data"):
        tensor_from_cpu_array_like(source, device="cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_copy_tensor_to_pinned_cpu_from_cuda() -> None:
    tensor = torch.arange(4, device="cuda", dtype=torch.int64)

    cpu = copy_tensor_to_pinned_cpu(tensor)

    assert cpu.device.type == "cpu"
    assert cpu.is_pinned()
    assert cpu.tolist() == [0, 1, 2, 3]
