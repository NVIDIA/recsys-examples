import abc


class BaseTask(abc.ABC):
    pass
    # @abc.abstractmethod
    # def compute_flops(self) -> float:
    #     pass

    # @abc.abstractmethod
    # def compute_memory(self, dtype_bytes: int = 2) -> float:
    #     pass

    # @abc.abstractmethod
    # def compute_ai(self, dtype_bytes: int = 2) -> float:
    #     pass

    # @abc.abstractmethod
    # def estimate_time(self) -> float:
    #     pass


# class GEMMTask(BaseTask):
#     def __init__(self, m: int, n: int, k: int):
#         self.m = m
#         self.n = n
#         self.k = k

#     def compute_flops(self) -> float:
#         """计算量（FLOPs）"""
#         return 2 * self.m * self.n * self.k

#     def compute_memory(self, dtype_bytes: int = 2) -> float:
#         """访存量（Bytes）"""
#         return dtype_bytes * (self.m * self.k + self.k * self.n + self.m * self.n)

#     def compute_ai(self, dtype_bytes: int = 2) -> float:
#         """计算访存比（FLOPs/Byte）"""
#         return self.compute_flops() / self.compute_memory(dtype_bytes)

#     def estimate_time(
#         self, peak_flops: float, bandwidth: float, dtype_bytes: int = 2
#     ) -> float:
#         """
#         使用Roofline模型估计执行时间（秒）

#         Args:
#             peak_flops: GPU峰值算力 (FLOPS)
#             bandwidth: GPU内存带宽 (Bytes/s)
#             dtype_bytes: 数据类型字节数 (2 for FP16)

#         Returns:
#             估计执行时间（秒）
#         """
#         flops = self.compute_flops()
#         memory = self.compute_memory(dtype_bytes)

#         # Roofline: Time = max(compute_time, memory_time)
#         compute_time = flops / peak_flops
#         memory_time = memory / bandwidth

#         return max(compute_time, memory_time)

#     def __repr__(self):
#         return f"GEMM_[{self.m}x{self.k}x{self.n}]"


# class BaseAttentionTask(BaseTask):
#     """
#     It's legal to leave num_heads and head_dim to None, in this case, the task will be estimated as a single head attention task.
#     In most cases, we only need to know the relative ordering。
#     """

#     def __init__(
#         self,
#         batch_size: int,
#         seqlen: Union[int, List[int]],
#         num_heads: Optional[int] = 1,
#         head_dim: Optional[int] = 1,
#     ):
#         self.batch_size = batch_size
#         self.num_heads = num_heads
#         self.head_dim = head_dim

#         if isinstance(seqlen, list):
#             assert (
#                 len(seqlen) == batch_size
#             ), f"Length of seqlen must be equal to batch_size"
#             self.seqlen = seqlen
#         else:
#             self.seqlen = [seqlen] * batch_size

#     def _attention_ops(self) -> int:
#         pass


# class MHATask(BaseAttentionTask):
#     def compute_flops(self) -> float:
#         return self.batch_size * self.seqlen * self.num_heads * self.head_dim

#     def compute_memory(self, dtype_bytes: int = 2) -> float:
#         return (
#             self.batch_size * self.seqlen * self.num_heads * self.head_dim * dtype_bytes
#         )


# class MLATask(BaseAttentionTask):
#     def __init__(
#         self,
#         batch_size: int,
#         seqlen: Union[int, List[int]],
#         num_heads: Optional[int] = 1,
#         head_dim: Optional[int] = 1,
#     ):
#         super().__init__(batch_size, seqlen, num_heads, head_dim)

#     def compute_flops(self) -> float:
#         return self.batch_size * self.seqlen * self.num_heads * self.head_dim

#     def compute_memory(self, dtype_bytes: int = 2) -> float:
#         return (
#             self.batch_size * self.seqlen * self.num_heads * self.head_dim * dtype_bytes
#         )


# class HSTUTask(BaseAttentionTask):
#     def compute_flops(self) -> float:
#         return self.batch_size * self.seqlen * self.num_heads * self.head_dim

#     def compute_memory(self, dtype_bytes: int = 2) -> float:
#         return (
#             self.batch_size * self.seqlen * self.num_heads * self.head_dim * dtype_bytes
#         )
