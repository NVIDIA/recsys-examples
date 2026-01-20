"""
Abstract Factory Pattern for creating Training Pipeline instances.

This module provides a factory registry system that allows different modules
to register their training pipeline implementations and create them by name.

Example usage:
    # Register a pipeline class
    TrainPipelineFactory.register("none", JaggedMegatronTrainNonePipeline)
    
    # Create an instance
    pipeline = TrainPipelineFactory.create(
        "none",
        model=model,
        optimizer=optimizer,
        device=device,
        batch_shuffler=shuffler
    )
    
    # Or use config dict
    config = {
        "type": "none",
        "execute_all_batches": True,
        "apply_jit": False
    }
    pipeline = TrainPipelineFactory.create_from_config(
        config,
        model=model,
        optimizer=optimizer,
        device=device
    )
"""

from typing import Any, Dict, Type

import torch


class TrainPipelineFactory:
    """
    Factory class for creating Training Pipeline instances.

    This factory implements the registry pattern, allowing different modules
    to register their pipeline implementations without creating circular dependencies.

    Note: This class uses only class methods and should NOT be instantiated.
          The registry is shared at the class level, ensuring singleton behavior.
    """

    _registry: Dict[str, Type] = {}

    def __init__(self):
        """
        Prevent instantiation of the factory class.

        Raises:
            TypeError: Always raised to prevent instantiation
        """
        raise TypeError(
            f"{self.__class__.__name__} is a factory class with class methods only "
            "and should not be instantiated. Use class methods directly: "
            f"{self.__class__.__name__}.create(...)"
        )

    @classmethod
    def register(
        cls,
        name: str,
        pipeline_class: Type,
    ) -> None:
        """
        Register a training pipeline class with a given name.

        Args:
            name: Unique identifier for the pipeline (e.g., "none", "sparse_dist", "prefetch")
            pipeline_class: The class to register

        Raises:
            ValueError: If name is already registered
        """
        if name in cls._registry:
            raise ValueError(
                f"Training pipeline '{name}' is already registered. "
                f"Existing: {cls._registry[name]}, New: {pipeline_class}"
            )

        cls._registry[name] = pipeline_class

    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Unregister a training pipeline class.

        Args:
            name: The name of the pipeline to unregister

        Raises:
            KeyError: If name is not registered
        """
        if name not in cls._registry:
            raise KeyError(f"Training pipeline '{name}' is not registered")
        del cls._registry[name]

    @classmethod
    def create(
        cls,
        name: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        **kwargs: Any,
    ):
        """
        Create a training pipeline instance by name.

        Args:
            name: The registered name of the pipeline
            model: The model to train
            optimizer: The optimizer to use
            device: The device to use
            **kwargs: Additional arguments to pass to the pipeline's __init__ method

        Returns:
            An instance of the requested training pipeline

        Raises:
            KeyError: If name is not registered
        """
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise KeyError(
                f"Training pipeline '{name}' is not registered. "
                f"Available pipelines: {available or 'none'}"
            )

        pipeline_class = cls._registry[name]
        return pipeline_class(model=model, optimizer=optimizer, device=device, **kwargs)

    @classmethod
    def create_from_config(
        cls,
        config: Dict[str, Any],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ):
        """
        Create a training pipeline from a configuration dictionary.

        Args:
            config: Configuration dict with 'type' key and other parameters
                   Example: {"type": "prefetch", "execute_all_batches": True}
            model: The model to train
            optimizer: The optimizer to use
            device: The device to use

        Returns:
            An instance of the requested training pipeline

        Raises:
            KeyError: If 'type' key is missing or the pipeline is not registered
        """
        if "type" not in config:
            raise KeyError("Configuration must contain 'type' key")

        pipeline_type = config["type"]
        # Create a copy without the 'type' key for kwargs
        kwargs = {k: v for k, v in config.items() if k != "type"}

        return cls.create(
            pipeline_type, model=model, optimizer=optimizer, device=device, **kwargs
        )

    @classmethod
    def list_available(cls) -> Dict[str, Type]:
        """
        List all registered training pipeline classes.

        Returns:
            Dictionary mapping pipeline names to their classes
        """
        return cls._registry.copy()

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if a pipeline name is registered.

        Args:
            name: The name to check

        Returns:
            True if registered, False otherwise
        """
        return name in cls._registry

    @classmethod
    def clear_registry(cls) -> None:
        """
        Clear all registered pipelines. Useful for testing.
        """
        cls._registry.clear()


def register_train_pipeline(name: str):
    """
    Decorator for registering training pipeline classes.

    Example:
        @register_train_pipeline("my_pipeline")
        class MyTrainPipeline:
            def __init__(self, model, optimizer, device, **kwargs):
                ...
    """

    def decorator(cls: Type):
        TrainPipelineFactory.register(name, cls)
        return cls

    return decorator
