import abc
from typing import Optional, Protocol, TypeVar, Union, cast

import numpy as np

T_co = TypeVar("T_co", covariant=True)


class ArrayLike(Protocol[T_co]):
    """Array-like protocol supporting indexing and length"""

    def __getitem__(self, idx: int) -> T_co:
        ...

    def __len__(self) -> int:
        ...


class BaseTask(abc.ABC):
    @abc.abstractmethod
    def get_workloads(self, *args, **kwargs):
        pass


class IdentityTask(BaseTask):
    def __init__(self):
        pass

    def get_workloads(self, *args, **kwargs):
        return 0


class BinaryTask(BaseTask):
    """
    Mul, Add, Max, Min, Reduction, etc.
    """

    def __init__(self):
        pass

    def get_workloads(self, length: Union[int, ArrayLike[int]]):
        length_val = cast(Union[int, ArrayLike[int]], length)
        return length_val  # type: ignore[return-value, operator]

    def __repr__(self):
        return f"BinaryTask"


class FMATask(BaseTask):
    def __init__(self):
        pass

    def get_workloads(self, length: Union[int, ArrayLike[int]]):
        length_val = cast(Union[int, ArrayLike[int]], length)
        return 2 * length_val  # type: ignore[return-value, operator]

    def __repr__(self):
        return f"FMATask"


class GEMMTask(BaseTask):
    def __init__(self):
        pass

    def compute_flops(
        self,
        m: Union[int, ArrayLike[int]],
        n: Union[int, ArrayLike[int]],
        k: Union[int, ArrayLike[int]],
    ) -> Union[int, ArrayLike[int]]:
        """Compute FLOPs (integer or array-like of integers)"""
        # Cast to avoid mypy arithmetic errors - works for np.ndarray, torch.Tensor, etc.
        m_val = cast(Union[int, ArrayLike[int]], m)
        n_val = cast(Union[int, ArrayLike[int]], n)
        k_val = cast(Union[int, ArrayLike[int]], k)
        return 2 * m_val * n_val * k_val  # type: ignore[return-value, operator]

    def compute_memory(
        self,
        m: Union[int, ArrayLike[int]],
        n: Union[int, ArrayLike[int]],
        k: Union[int, ArrayLike[int]],
        dtype_bytes: int = 2,
    ) -> Union[float, ArrayLike[float]]:
        """Compute memory usage (float or array-like of floats)"""
        # Cast to avoid mypy arithmetic errors - works for np.ndarray, torch.Tensor, etc.
        m_val = cast(Union[int, ArrayLike[int]], m)
        n_val = cast(Union[int, ArrayLike[int]], n)
        k_val = cast(Union[int, ArrayLike[int]], k)
        return dtype_bytes * (m_val * k_val + k_val * n_val + m_val * n_val)  # type: ignore[return-value, operator]

    def estimate_time(
        self,
        peak_flops: float,
        bandwidth: float,
        m: Union[int, ArrayLike[int]],
        n: Union[int, ArrayLike[int]],
        k: Union[int, ArrayLike[int]],
        dtype_bytes: int = 2,
    ) -> Union[float, ArrayLike[float]]:
        """
        Estimate execution time (in seconds) using the Roofline model.

        Args:
            peak_flops: GPU peak compute performance (FLOPS)
            bandwidth: GPU memory bandwidth (Bytes/s)
            dtype_bytes: Data type size in bytes (2 for FP16)

        Returns:
            Estimated execution time (seconds or array-like of floats)
        """
        flops = self.compute_flops(m, n, k)
        memory = self.compute_memory(m, n, k, dtype_bytes)

        # Roofline: Time = max(compute_time, memory_time)
        compute_time = flops / peak_flops  # type: ignore[operator]
        memory_time = memory / bandwidth  # type: ignore[operator]

        # Handle both scalar and array cases
        if isinstance(compute_time, (int, float)) and isinstance(
            memory_time, (int, float)
        ):
            return max(compute_time, memory_time)  # type: ignore[return-value]
        else:
            return np.maximum(compute_time, memory_time)  # type: ignore[arg-type,return-value]

    def __repr__(self):
        return f"GEMMTask"

    # we now only consider the compute workload
    def get_workloads(
        self,
        m: Union[int, ArrayLike[int]],
        n: Union[int, ArrayLike[int]],
        k: Union[int, ArrayLike[int]],
        dtype_bytes: int = 2,
    ) -> Union[int, ArrayLike[int]]:
        """Get workloads in FLOPs (integer or array-like of integers)"""
        compute_flops = self.compute_flops(m, n, k)
        return compute_flops  # type: ignore[return-value]


class BaseAttentionTask(BaseTask):
    """
    It's legal to leave num_heads and head_dim to None, in this case, the task will be estimated as a single head attention task.
    In most cases, we only need to know the relative ordering。
    """

    def __init__(self):
        pass


# TODO@junzhang, causal or not?
class SelfAttentionTask(BaseAttentionTask):
    """
    q,k,v = W_q* x, W_k* x, W_v* x
    attention = qk^T * v
    """

    def __init__(self):
        super().__init__()
        # assume k (embedding dim) = head_dim * num_heads
        self._qkv_proj_task = GEMMTask()
        # QK, PV
        self._attention_single_head_task = GEMMTask()
        #
        self._out_proj_task = GEMMTask()
        # the up dim is 4x
        self._fc_task = GEMMTask()

    def get_workloads(
        self,
        seqlen: Union[int, ArrayLike[int]],
        num_heads: Optional[int] = 1,
        head_dim: Optional[int] = 1,
    ) -> Union[int, ArrayLike[int]]:
        """Get workloads in FLOPs (integer or array-like of integers)"""
        # Ensure num_heads and head_dim are not None
        _num_heads: int = num_heads if num_heads is not None else 1
        _head_dim: int = head_dim if head_dim is not None else 1

        # q,k,v proj
        qkv_proj_flops = self._qkv_proj_task.get_workloads(
            seqlen, 3 * _num_heads * _head_dim, _head_dim * _num_heads
        )
        attention_flops = (
            self._attention_single_head_task.get_workloads(
                seqlen, seqlen, _head_dim * _num_heads
            )
            * _num_heads
            * 2
        )  # QK, PV
        out_proj = self._out_proj_task.get_workloads(
            seqlen, _head_dim * _num_heads, _head_dim * _num_heads
        )
        fc_flops = (
            self._fc_task.get_workloads(
                seqlen, 4 * _head_dim * _num_heads, _head_dim * _num_heads
            )
            * 2
        )
        # print(f"qkv_proj_flops: {qkv_proj_flops}, attention_flops: {attention_flops}, fc_flops: {fc_flops}, out_proj_flops: {out_proj}")
        return qkv_proj_flops + attention_flops + fc_flops + out_proj  # type: ignore[return-value, operator]


class HSTUAttentionTask(BaseAttentionTask):
    """
    q,k,v = W_q* x, W_k* x, W_v* x
    attention = qk^T * v
    """

    def __init__(self):
        super().__init__()
        # assume k (embedding dim) = head_dim * num_heads
        self._qkvu_proj_task = GEMMTask()
        # QK, PV
        self._attention_single_head_task = GEMMTask()
        self._out_proj_task = GEMMTask()
        # no mlp

    def get_workloads(
        self,
        seqlen: Union[int, ArrayLike[int]],
        num_heads: Optional[int] = 1,
        head_dim: Optional[int] = 1,
    ) -> Union[int, ArrayLike[int]]:
        """Get workloads in FLOPs (integer or array-like of integers)"""
        # Ensure num_heads and head_dim are not None
        _num_heads: int = num_heads if num_heads is not None else 1
        _head_dim: int = head_dim if head_dim is not None else 1

        # q,k,v proj
        qkvu_proj_flops = self._qkvu_proj_task.get_workloads(
            seqlen, 4 * _num_heads * _head_dim, _head_dim * _num_heads
        )
        attention_flops = (
            self._attention_single_head_task.get_workloads(
                seqlen, seqlen, _head_dim * _num_heads
            )
            * _num_heads
            * 2
        )  # QK, PV
        out_proj = self._out_proj_task.get_workloads(
            seqlen, _head_dim * _num_heads, _head_dim * _num_heads
        )
        return qkvu_proj_flops + attention_flops + out_proj  # type: ignore[return-value, operator]


if __name__ == "__main__":
    min_seq = 4
    max_seq = 1024
    seqlen = np.random.randint(min_seq, max_seq, size=100)
    task = SelfAttentionTask()
    W = task.get_workloads(seqlen, num_heads=16, head_dim=64)
    print(W)
