import abc

from dynamicemb.dynamicemb_config import *
from dynamicemb_extensions import (
    CurandStateContext,
    const_init,
    debug_init,
    normal_init,
    truncated_normal_init,
    uniform_init,
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
