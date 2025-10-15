import abc
from typing import Generic, Optional, Tuple, TypeVar

import torch

TableOptionType = TypeVar("TableOptionType")
OptimizerInterface = TypeVar("OptimizerInterface")


# make it standalone to avoid recursive references.
class Storage(abc.ABC, Generic[TableOptionType, OptimizerInterface]):
    @abc.abstractmethod
    def __init__(
        self,
        options: TableOptionType,
        optimizer: OptimizerInterface,
    ):
        pass

    @abc.abstractmethod
    def find(
        self,
        unique_keys: torch.Tensor,
        unique_vals: torch.Tensor,
        founds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_missing: torch.Tensor
        missing_keys: torch.Tensor
        missing_indices: torch.Tensor
        return num_missing, missing_keys, missing_indices

    @abc.abstractmethod
    def find_embeddings(
        self,
        unique_keys: torch.Tensor,
        unique_embs: torch.Tensor,
        founds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_missing: torch.Tensor
        missing_keys: torch.Tensor
        missing_indices: torch.Tensor
        return num_missing, missing_keys, missing_indices

    @abc.abstractmethod
    def insert(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
    ) -> None:
        pass

    @abc.abstractmethod
    def update(
        self, keys: torch.Tensor, grads: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_missing: torch.Tensor
        missing_keys: torch.Tensor
        missing_indices: torch.Tensor
        return num_missing, missing_keys, missing_indices

    @abc.abstractmethod
    def enable_update(self) -> bool:
        ...

    @abc.abstractmethod
    def dump(
        self,
        meta_file_path: str,
        emb_key_path: str,
        embedding_file_path: str,
        score_file_path: Optional[str],
        opt_file_path: Optional[str],
    ) -> None:
        pass

    @abc.abstractmethod
    def load(
        self,
        meta_file_path: str,
        emb_file_path: str,
        embedding_file_path: str,
        score_file_path: Optional[str],
        opt_file_path: Optional[str],
        include_optim: bool,
    ) -> None:
        pass

    @abc.abstractmethod
    def embedding_dtype(
        self,
    ) -> torch.dtype:
        pass

    @abc.abstractmethod
    def embedding_dim(
        self,
    ) -> int:
        pass

    @abc.abstractmethod
    def value_dim(
        self,
    ) -> int:
        pass

    @abc.abstractmethod
    def init_optimizer_state(
        self,
    ) -> float:
        pass


class Cache(abc.ABC):
    @abc.abstractmethod
    def find(
        self,
        unique_keys: torch.Tensor,
        unique_vals: torch.Tensor,
        founds: Optional[torch.Tensor] = None,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        num_missing: int
        missing_keys: torch.Tensor
        missing_indices: torch.Tensor
        return num_missing, missing_keys, missing_indices

    @abc.abstractmethod
    def find_embeddings(
        self,
        unique_keys: torch.Tensor,
        unique_embs: torch.Tensor,
        founds: Optional[torch.Tensor] = None,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        num_missing: int
        missing_keys: torch.Tensor
        missing_indices: torch.Tensor
        return num_missing, missing_keys, missing_indices

    @abc.abstractmethod
    def find_missed_keys(
        self,
        unique_keys: torch.Tensor,
        founds: Optional[torch.Tensor] = None,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        num_missing: int
        missing_keys: torch.Tensor
        missing_indices: torch.Tensor
        return num_missing, missing_keys, missing_indices

    @abc.abstractmethod
    def insert_and_evict(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_evicted: int
        evicted_keys: torch.Tensor
        evicted_values: torch.Tensor
        evicted_scores: torch.Tensor
        return num_evicted, evicted_keys, evicted_values, evicted_scores

    @abc.abstractmethod
    def update(
        self, keys: torch.Tensor, grads: torch.Tensor
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        num_missing: int
        missing_keys: torch.Tensor
        missing_indices: torch.Tensor
        return num_missing, missing_keys, missing_indices

    @abc.abstractmethod
    def flush(self, storage: Storage) -> None:
        pass

    @abc.abstractmethod
    def reset(
        self,
    ) -> None:
        pass

    @abc.abstractmethod
    def cache_metrics(
        self,
    ) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def set_record_cache_metrics(self, record: bool) -> None:
        pass
