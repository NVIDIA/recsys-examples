import abc
from dynamicemb.dynamicemb_config import *
from dynamicemb_extensions import (
    CurandStateContext,
    normal_init,
    truncated_normal_init,
    uniform_init,
    const_init,
    debug_init,
)


class BaseDynamicEmbInitializer(abc.ABC):
    def __init__(
        self,
        args: DynamicEmbInitializerArgs
    ):
        self._args = args

    @abc.abstractmethod
    def __call__(
        self,
        buffer: torch.Tensor,
        indices: torch.Tensor,
        keys: Optional[torch.Tensor], # remove it when debug mode is removed
    ) -> None:
        ...

class NormalInitializer(BaseDynamicEmbInitializer):
    def __init__(
        self,
        args: DynamicEmbInitializerArgs
    ):
        super().__init__()
        self._curand_state = CurandStateContext()
    
    def __call__(
        self,
        buffer: torch.Tensor,
        indices: torch.Tensor,
        keys: Optional[torch.Tensor], # remove it when debug mode is removed
    ) -> None:
        normal_init(buffer, indices, self._curand_state, self._args.mean, self._args.std_dev)

class TruncatedNormalInitializer(BaseDynamicEmbInitializer):
    def __init__(
        self,
        args: DynamicEmbInitializerArgs
    ):
        super().__init__()
        self._curand_state = CurandStateContext()
    
    def __call__(
        self,
        buffer: torch.Tensor,
        indices: torch.Tensor,
        keys: Optional[torch.Tensor], # remove it when debug mode is removed
    ) -> None:
        truncated_normal_init(buffer, indices, self._curand_state, self._args.mean, self._args.std_dev, self._args.lower, self._args.upper)

class UniformInitializer(BaseDynamicEmbInitializer):
    def __init__(
        self,
        args: DynamicEmbInitializerArgs
    ):
        super().__init__()
        self._curand_state = CurandStateContext()
    
    def __call__(
        self,
        buffer: torch.Tensor,
        indices: torch.Tensor,
        keys: Optional[torch.Tensor], # remove it when debug mode is removed
    ) -> None:
        uniform_init(buffer, indices, self._curand_state, self._args.lower, self._args.upper)

class ConstantInitializer(BaseDynamicEmbInitializer):
    def __init__(
        self,
        args: DynamicEmbInitializerArgs
    ):
        super().__init__()
    
    def __call__(
        self,
        buffer: torch.Tensor,
        indices: torch.Tensor,
        keys: Optional[torch.Tensor], # remove it when debug mode is removed
    ) -> None:
        const_init(buffer, indices, self._args.value)

class DebugInitializer(BaseDynamicEmbInitializer):
    def __init__(
        self,
        args: DynamicEmbInitializerArgs
    ):
        super().__init__()
    
    def __call__(
        self,
        buffer: torch.Tensor,
        indices: torch.Tensor,
        keys: Optional[torch.Tensor], # remove it when debug mode is removed
    ) -> None:
        debug_init(buffer, indices, keys)