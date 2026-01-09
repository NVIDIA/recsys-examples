import torch


def module_hook_check_act_has_nan(
    module, input, output, msg: str = "", print_nan_indices: bool = False
) -> torch.Tensor:
    if isinstance(output, torch.Tensor) and torch.isnan(output).all():
        if print_nan_indices:
            nan_indices = output.isnan().nonzero()
            print(f"{msg} module {module} has nan output at indices {nan_indices}")
        else:
            print(f"{msg} module {module} has nan output")
        assert False
    return output


def tensor_hook_check_grad_has_nan(
    grad: torch.Tensor, msg: str = "", print_nan_indices: bool = False
) -> torch.Tensor:
    if grad.isnan().any():
        if print_nan_indices:
            nan_indices = grad.isnan().nonzero()
            print(f"{msg} grad has nan at indices {nan_indices}")
        else:
            print(f"{msg} grad has nan")
        assert False
    return grad


def module_hook_check_act_has_inf(
    module, input, output, msg: str = "", print_inf_indices: bool = False
) -> torch.Tensor:
    if isinstance(output, torch.Tensor) and torch.isinf(output).any():
        if print_inf_indices:
            inf_indices = output.isinf().nonzero()
            print(f"{msg} module {module} has inf output at indices {inf_indices}")
        else:
            print(f"{msg} module {module} has inf output")
        assert False
    return output


def tensor_hook_assert_grad_has_nan(grad: torch.Tensor, msg: str = "") -> torch.Tensor:
    assert grad.isnan().any(), f"{msg} grad has nan"
    return grad


def tensor_hook_check_grad_has_inf(
    grad: torch.Tensor, msg: str = "", print_inf_indices: bool = False
) -> torch.Tensor:
    if grad.isinf().any():
        if print_inf_indices:
            inf_indices = grad.isinf().nonzero()
            print(f"{msg} grad has inf at indices {inf_indices}")
        else:
            print(f"{msg} grad has inf")
    return grad


def tensor_hook_print_grad(grad: torch.Tensor, msg: str = "") -> torch.Tensor:
    print(f"{msg} grad[-1,...]: {grad[-1,...]}")
    return grad
