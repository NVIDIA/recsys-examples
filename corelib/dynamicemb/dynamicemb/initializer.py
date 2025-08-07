import abc
from dynamicemb.dynamicemb_config import *
from dynamicemb_extensions import (
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
        pass

    @abc.abstractmethod
    def __call__(
        self,
        buffer: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        ...

class NormalInitializer(BaseDynamicEmbInitializer):
    def __init__(
        self,
        args: DynamicEmbInitializerArgs     
    ):
        super().__init__()
    
    def __call__(
        self,
        buffer: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        normal_init(buffer, indices)

class TruncatedNormalInitializer(BaseDynamicEmbInitializer):
    def __init__(
        self,
        args: DynamicEmbInitializerArgs     
    ):
        super().__init__()
    
    def __call__(
        self,
        buffer: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        truncated_normal_init(buffer, indices)

class UniformInitializer(BaseDynamicEmbInitializer):
    def __init__(
        self,
        args: DynamicEmbInitializerArgs     
    ):
        super().__init__()
    
    def __call__(
        self,
        buffer: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        uniform_init(buffer, indices)

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
    ) -> None:
        const_init(buffer, indices)

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
    ) -> None:
        debug_init(buffer, indices)