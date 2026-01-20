from typing import Any, Dict, Type

from .batch_shuffler import BaseTaskBalancedBatchShuffler


class BatchShufflerFactory:
    """
    Abstract Factory Pattern for creating BaseTaskBalancedBatchShuffler instances.

    This module provides a factory registry system that allows different modules
    to register their batch shuffler implementations and create them by name.

    Example usage:
        # Register a shuffler class
        BatchShufflerFactory.register("hstu", HASTUBalancedBatchShuffler)

        # Create an instance
        shuffler = BatchShufflerFactory.create("hstu", num_heads=16, head_dim=64)

        # Or use config dict
        config = {"type": "hstu", "num_heads": 16, "head_dim": 64}
        shuffler = BatchShufflerFactory.create_from_config(config)

    Note: This class uses only class methods and should NOT be instantiated.
          The registry is shared at the class level, ensuring singleton behavior.
    """

    _registry: Dict[str, Type[BaseTaskBalancedBatchShuffler]] = {}

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
        shuffler_class: Type[BaseTaskBalancedBatchShuffler],
    ) -> None:
        """
        Register a batch shuffler class with a given name.

        Args:
            name: Unique identifier for the shuffler (e.g., "hstu", "sid_gr")
            shuffler_class: The class to register (must inherit from BaseTaskBalancedBatchShuffler)

        Raises:
            ValueError: If name is already registered
            TypeError: If shuffler_class doesn't inherit from BaseTaskBalancedBatchShuffler
        """
        if name in cls._registry:
            raise ValueError(
                f"Batch shuffler '{name}' is already registered. "
                f"Existing: {cls._registry[name]}, New: {shuffler_class}"
            )

        if not issubclass(shuffler_class, BaseTaskBalancedBatchShuffler):
            raise TypeError(
                f"shuffler_class must inherit from BaseTaskBalancedBatchShuffler, "
                f"got {shuffler_class}"
            )

        cls._registry[name] = shuffler_class

    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Unregister a batch shuffler class.

        Args:
            name: The name of the shuffler to unregister

        Raises:
            KeyError: If name is not registered
        """
        if name not in cls._registry:
            raise KeyError(f"Batch shuffler '{name}' is not registered")
        del cls._registry[name]

    @classmethod
    def create(
        cls,
        name: str,
        **kwargs: Any,
    ) -> BaseTaskBalancedBatchShuffler:
        """
        Create a batch shuffler instance by name.

        Args:
            name: The registered name of the shuffler
            **kwargs: Arguments to pass to the shuffler's __init__ method

        Returns:
            An instance of the requested batch shuffler

        Raises:
            KeyError: If name is not registered
        """
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise KeyError(
                f"Batch shuffler '{name}' is not registered. "
                f"Available shufflers: {available or 'none'}"
            )

        shuffler_class = cls._registry[name]
        return shuffler_class(**kwargs)

    @classmethod
    def create_from_config(
        cls, config: Dict[str, Any]
    ) -> BaseTaskBalancedBatchShuffler:
        """
        Create a batch shuffler from a configuration dictionary.

        Args:
            config: Configuration dict with 'type' key and other parameters
                   Example: {"type": "hstu", "num_heads": 16, "head_dim": 64}

        Returns:
            An instance of the requested batch shuffler

        Raises:
            KeyError: If 'type' key is missing or the shuffler is not registered
        """
        if "type" not in config:
            raise KeyError("Configuration must contain 'type' key")

        shuffler_type = config["type"]
        # Create a copy without the 'type' key for kwargs
        kwargs = {k: v for k, v in config.items() if k != "type"}

        return cls.create(shuffler_type, **kwargs)

    @classmethod
    def list_available(cls) -> Dict[str, Type[BaseTaskBalancedBatchShuffler]]:
        """
        List all registered batch shuffler classes.

        Returns:
            Dictionary mapping shuffler names to their classes
        """
        return cls._registry.copy()

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if a shuffler name is registered.

        Args:
            name: The name to check

        Returns:
            True if registered, False otherwise
        """
        return name in cls._registry

    @classmethod
    def clear_registry(cls) -> None:
        """
        Clear all registered shufflers. Useful for testing.
        """
        cls._registry.clear()


# Register the built-in identity shuffler
from .batch_shuffler import IdentityBalancedBatchShuffler

BatchShufflerFactory.register("identity", IdentityBalancedBatchShuffler)


def register_batch_shuffler(name: str):
    """
    Decorator for registering batch shuffler classes.

    Example:
        @register_batch_shuffler("my_shuffler")
        class MyBalancedBatchShuffler(BaseTaskBalancedBatchShuffler):
            ...
    """

    def decorator(cls: Type[BaseTaskBalancedBatchShuffler]):
        BatchShufflerFactory.register(name, cls)
        return cls

    return decorator
