import abc

from dynamicemb.dynamicemb_config import *
from dynamicemb_extensions import (
    CurandStateContext,
    const_load_or_initialize_mixed,
    const_init,
    const_init_flat,
    debug_load_or_initialize_mixed,
    debug_init,
    debug_init_flat,
    normal_load_or_initialize_mixed,
    normal_init,
    normal_init_flat,
    truncated_normal_load_or_initialize_mixed,
    truncated_normal_init,
    truncated_normal_init_flat,
    uniform_load_or_initialize_mixed,
    uniform_init,
    uniform_init_flat,
)


class BaseDynamicEmbInitializer(abc.ABC):
    def __init__(self, args: DynamicEmbInitializerArgs):
        self._args = args
        if self._args.lower is None:
            self._args.lower = 0.0
        if self._args.upper is None:
            self._args.upper = 1.0

    @abc.abstractmethod
    def __call__(
        self,
        buffer: torch.Tensor,
        indices: torch.Tensor,
        keys: Optional[torch.Tensor],  # remove it when debug mode is removed
    ) -> None:
        """Initialize rows selected by integer indices or a boolean mask."""
        ...

    @abc.abstractmethod
    def initialize_flat(
        self,
        table_ptrs: torch.Tensor,
        slot_indices: torch.Tensor,
        table_ids: torch.Tensor,
        table_value_dims: torch.Tensor,
        table_emb_dims: torch.Tensor,
        mask: torch.Tensor,
        keys: Optional[torch.Tensor],
        value_dtype: torch.dtype,
        initial_optim_state: float,
    ) -> None:
        """Initialize selected rows directly in multi-table flat storage."""
        ...

    @abc.abstractmethod
    def load_or_initialize_mixed(
        self,
        output: torch.Tensor,
        cache_table_ptrs: torch.Tensor,
        cache_rows: torch.Tensor,
        storage_table_ptrs: torch.Tensor,
        storage_rows: torch.Tensor,
        table_ids: torch.Tensor,
        table_value_dims: torch.Tensor,
        table_emb_dims: torch.Tensor,
        all_dims_vec4: bool,
        keys: Optional[torch.Tensor],
        preinitialized_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Load cache/storage rows or initialize transient rows in one kernel.

        Source-less rows selected by ``preinitialized_mask`` are retained.
        """
        ...


class NormalInitializer(BaseDynamicEmbInitializer):
    def __init__(self, args: DynamicEmbInitializerArgs):
        super().__init__(args)
        self._curand_state = CurandStateContext()

    def __call__(
        self,
        buffer: torch.Tensor,
        indices: torch.Tensor,
        keys: Optional[torch.Tensor],  # remove it when debug mode is removed
    ) -> None:
        normal_init(
            buffer, indices, self._curand_state, self._args.mean, self._args.std_dev
        )

    def initialize_flat(
        self,
        table_ptrs: torch.Tensor,
        slot_indices: torch.Tensor,
        table_ids: torch.Tensor,
        table_value_dims: torch.Tensor,
        table_emb_dims: torch.Tensor,
        mask: torch.Tensor,
        keys: Optional[torch.Tensor],
        value_dtype: torch.dtype,
        initial_optim_state: float,
    ) -> None:
        normal_init_flat(
            table_ptrs,
            slot_indices,
            table_ids,
            table_value_dims,
            table_emb_dims,
            mask,
            value_dtype,
            initial_optim_state,
            self._curand_state,
            self._args.mean,
            self._args.std_dev,
        )

    def load_or_initialize_mixed(
        self,
        output: torch.Tensor,
        cache_table_ptrs: torch.Tensor,
        cache_rows: torch.Tensor,
        storage_table_ptrs: torch.Tensor,
        storage_rows: torch.Tensor,
        table_ids: torch.Tensor,
        table_value_dims: torch.Tensor,
        table_emb_dims: torch.Tensor,
        all_dims_vec4: bool,
        keys: Optional[torch.Tensor],
        preinitialized_mask: Optional[torch.Tensor] = None,
    ) -> None:
        normal_load_or_initialize_mixed(
            output,
            cache_table_ptrs,
            cache_rows,
            storage_table_ptrs,
            storage_rows,
            table_ids,
            table_value_dims,
            table_emb_dims,
            all_dims_vec4,
            self._curand_state,
            self._args.mean,
            self._args.std_dev,
            preinitialized_mask,
        )


class TruncatedNormalInitializer(BaseDynamicEmbInitializer):
    def __init__(self, args: DynamicEmbInitializerArgs):
        super().__init__(args)
        self._curand_state = CurandStateContext()

    def __call__(
        self,
        buffer: torch.Tensor,
        indices: torch.Tensor,
        keys: Optional[torch.Tensor],  # remove it when debug mode is removed
    ) -> None:
        truncated_normal_init(
            buffer,
            indices,
            self._curand_state,
            self._args.mean,
            self._args.std_dev,
            self._args.lower,
            self._args.upper,
        )

    def initialize_flat(
        self,
        table_ptrs: torch.Tensor,
        slot_indices: torch.Tensor,
        table_ids: torch.Tensor,
        table_value_dims: torch.Tensor,
        table_emb_dims: torch.Tensor,
        mask: torch.Tensor,
        keys: Optional[torch.Tensor],
        value_dtype: torch.dtype,
        initial_optim_state: float,
    ) -> None:
        truncated_normal_init_flat(
            table_ptrs,
            slot_indices,
            table_ids,
            table_value_dims,
            table_emb_dims,
            mask,
            value_dtype,
            initial_optim_state,
            self._curand_state,
            self._args.mean,
            self._args.std_dev,
            self._args.lower,
            self._args.upper,
        )

    def load_or_initialize_mixed(
        self,
        output: torch.Tensor,
        cache_table_ptrs: torch.Tensor,
        cache_rows: torch.Tensor,
        storage_table_ptrs: torch.Tensor,
        storage_rows: torch.Tensor,
        table_ids: torch.Tensor,
        table_value_dims: torch.Tensor,
        table_emb_dims: torch.Tensor,
        all_dims_vec4: bool,
        keys: Optional[torch.Tensor],
        preinitialized_mask: Optional[torch.Tensor] = None,
    ) -> None:
        truncated_normal_load_or_initialize_mixed(
            output,
            cache_table_ptrs,
            cache_rows,
            storage_table_ptrs,
            storage_rows,
            table_ids,
            table_value_dims,
            table_emb_dims,
            all_dims_vec4,
            self._curand_state,
            self._args.mean,
            self._args.std_dev,
            self._args.lower,
            self._args.upper,
            preinitialized_mask,
        )


class UniformInitializer(BaseDynamicEmbInitializer):
    def __init__(self, args: DynamicEmbInitializerArgs):
        super().__init__(args)
        self._curand_state = CurandStateContext()

    def __call__(
        self,
        buffer: torch.Tensor,
        indices: torch.Tensor,
        keys: Optional[torch.Tensor],  # remove it when debug mode is removed
    ) -> None:
        uniform_init(
            buffer, indices, self._curand_state, self._args.lower, self._args.upper
        )

    def initialize_flat(
        self,
        table_ptrs: torch.Tensor,
        slot_indices: torch.Tensor,
        table_ids: torch.Tensor,
        table_value_dims: torch.Tensor,
        table_emb_dims: torch.Tensor,
        mask: torch.Tensor,
        keys: Optional[torch.Tensor],
        value_dtype: torch.dtype,
        initial_optim_state: float,
    ) -> None:
        uniform_init_flat(
            table_ptrs,
            slot_indices,
            table_ids,
            table_value_dims,
            table_emb_dims,
            mask,
            value_dtype,
            initial_optim_state,
            self._curand_state,
            self._args.lower,
            self._args.upper,
        )

    def load_or_initialize_mixed(
        self,
        output: torch.Tensor,
        cache_table_ptrs: torch.Tensor,
        cache_rows: torch.Tensor,
        storage_table_ptrs: torch.Tensor,
        storage_rows: torch.Tensor,
        table_ids: torch.Tensor,
        table_value_dims: torch.Tensor,
        table_emb_dims: torch.Tensor,
        all_dims_vec4: bool,
        keys: Optional[torch.Tensor],
        preinitialized_mask: Optional[torch.Tensor] = None,
    ) -> None:
        uniform_load_or_initialize_mixed(
            output,
            cache_table_ptrs,
            cache_rows,
            storage_table_ptrs,
            storage_rows,
            table_ids,
            table_value_dims,
            table_emb_dims,
            all_dims_vec4,
            self._curand_state,
            self._args.lower,
            self._args.upper,
            preinitialized_mask,
        )


class ConstantInitializer(BaseDynamicEmbInitializer):
    def __init__(self, args: DynamicEmbInitializerArgs):
        super().__init__(args)

    def __call__(
        self,
        buffer: torch.Tensor,
        indices: torch.Tensor,
        keys: Optional[torch.Tensor],  # remove it when debug mode is removed
    ) -> None:
        const_init(buffer, indices, self._args.value)

    def initialize_flat(
        self,
        table_ptrs: torch.Tensor,
        slot_indices: torch.Tensor,
        table_ids: torch.Tensor,
        table_value_dims: torch.Tensor,
        table_emb_dims: torch.Tensor,
        mask: torch.Tensor,
        keys: Optional[torch.Tensor],
        value_dtype: torch.dtype,
        initial_optim_state: float,
    ) -> None:
        const_init_flat(
            table_ptrs,
            slot_indices,
            table_ids,
            table_value_dims,
            table_emb_dims,
            mask,
            value_dtype,
            initial_optim_state,
            self._args.value,
        )

    def load_or_initialize_mixed(
        self,
        output: torch.Tensor,
        cache_table_ptrs: torch.Tensor,
        cache_rows: torch.Tensor,
        storage_table_ptrs: torch.Tensor,
        storage_rows: torch.Tensor,
        table_ids: torch.Tensor,
        table_value_dims: torch.Tensor,
        table_emb_dims: torch.Tensor,
        all_dims_vec4: bool,
        keys: Optional[torch.Tensor],
        preinitialized_mask: Optional[torch.Tensor] = None,
    ) -> None:
        const_load_or_initialize_mixed(
            output,
            cache_table_ptrs,
            cache_rows,
            storage_table_ptrs,
            storage_rows,
            table_ids,
            table_value_dims,
            table_emb_dims,
            all_dims_vec4,
            self._args.value,
            preinitialized_mask,
        )


class DebugInitializer(BaseDynamicEmbInitializer):
    def __init__(self, args: DynamicEmbInitializerArgs):
        super().__init__(args)

    def __call__(
        self,
        buffer: torch.Tensor,
        indices: torch.Tensor,
        keys: Optional[torch.Tensor],  # remove it when debug mode is removed
    ) -> None:
        debug_init(buffer, indices, keys)

    def initialize_flat(
        self,
        table_ptrs: torch.Tensor,
        slot_indices: torch.Tensor,
        table_ids: torch.Tensor,
        table_value_dims: torch.Tensor,
        table_emb_dims: torch.Tensor,
        mask: torch.Tensor,
        keys: Optional[torch.Tensor],
        value_dtype: torch.dtype,
        initial_optim_state: float,
    ) -> None:
        if keys is None:
            raise ValueError("keys are required by the debug initializer")
        debug_init_flat(
            table_ptrs,
            slot_indices,
            table_ids,
            table_value_dims,
            table_emb_dims,
            mask,
            value_dtype,
            initial_optim_state,
            keys,
        )

    def load_or_initialize_mixed(
        self,
        output: torch.Tensor,
        cache_table_ptrs: torch.Tensor,
        cache_rows: torch.Tensor,
        storage_table_ptrs: torch.Tensor,
        storage_rows: torch.Tensor,
        table_ids: torch.Tensor,
        table_value_dims: torch.Tensor,
        table_emb_dims: torch.Tensor,
        all_dims_vec4: bool,
        keys: Optional[torch.Tensor],
        preinitialized_mask: Optional[torch.Tensor] = None,
    ) -> None:
        if keys is None:
            raise ValueError("keys are required by the debug initializer")
        debug_load_or_initialize_mixed(
            output,
            cache_table_ptrs,
            cache_rows,
            storage_table_ptrs,
            storage_rows,
            table_ids,
            table_value_dims,
            table_emb_dims,
            all_dims_vec4,
            keys,
            preinitialized_mask,
        )


def create_initializer_from_args(
    initializer_args: DynamicEmbInitializerArgs,
) -> BaseDynamicEmbInitializer:
    """
    Factory function to create an initializer instance from initializer arguments.
    """
    mode = initializer_args.mode
    if mode == DynamicEmbInitializerMode.NORMAL:
        return NormalInitializer(initializer_args)
    elif mode == DynamicEmbInitializerMode.TRUNCATED_NORMAL:
        return TruncatedNormalInitializer(initializer_args)
    elif mode == DynamicEmbInitializerMode.UNIFORM:
        return UniformInitializer(initializer_args)
    elif mode == DynamicEmbInitializerMode.CONSTANT:
        return ConstantInitializer(initializer_args)
    elif mode == DynamicEmbInitializerMode.DEBUG:
        return DebugInitializer(initializer_args)
    else:
        raise ValueError(f"Not supported initializer type: {mode}")
