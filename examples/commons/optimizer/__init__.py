from dataclasses import dataclass


@dataclass
class OptimizerParam:
    """
    Configuration for the embedding optimizer.

    Args:
        optimizer_str (str): The optimizer type as a string: ``'adam'`` | ``'sgd'``.
        learning_rate (float): The learning rate for the optimizer.
        adam_beta1 (float, optional): The beta1 parameter for the Adam optimizer. Defaults to 0.9.
        adam_beta2 (float, optional): The beta2 parameter for the Adam optimizer. Defaults to 0.95.
        adam_eps (float, optional): The epsilon parameter for the Adam optimizer. Defaults to 1e-08.
    """

    optimizer_str: str
    learning_rate: float
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-08
    weight_decay: float = 0.01
