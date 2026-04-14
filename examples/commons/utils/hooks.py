import torch


def register_perf_hooks(
    attn_module: torch.nn.Module,
    num_heads: int,
    attention_dim: int,
    linear_dim: int,
    is_causal: bool,
) -> None:
    """Attach forward / backward timing hooks to an ``HSTUAttention`` module.

    This measures per-layer attention FLOPs and kernel time using CUDA events,
    then reports TFLOPS / MFU via :class:`_AttnPerfAccumulator` (see
    ``attn_perf_tracker.py``).
    """
    from commons.utils.attn_perf_tracker import _get_attn_perf_accum
    from commons.utils.perf import _compute_attn_fwd_flops

    accum = _get_attn_perf_accum()

    # Per-module scratch space (stored as module attributes to avoid closures
    # capturing mutable lists).
    attn_module._perf_fwd_start = None  # type: ignore[attr-defined]
    attn_module._perf_fwd_flops = 0.0  # type: ignore[attr-defined]

    # -- forward pre hook: record start event & compute FLOPs ----------------
    def _fwd_pre_hook(module, args, kwargs):
        offsets = args[3]
        num_candidates = kwargs.get("num_candidates", None)
        num_contextuals = kwargs.get("num_contextuals", None)

        fwd_flops = _compute_attn_fwd_flops(
            offsets,
            num_heads,
            attention_dim,
            linear_dim,
            is_causal,
            num_candidates,
            num_contextuals,
        )
        module._perf_fwd_flops = fwd_flops

        start = torch.cuda.Event(enable_timing=True)
        start.record()
        module._perf_fwd_start = start

    # -- forward post hook: record end event ---------------------------------
    def _fwd_post_hook(module, _input, output):
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        accum.add_fwd(module._perf_fwd_start, end, module._perf_fwd_flops)

    # -- backward pre hook: record start event -------------------------------
    def _bwd_pre_hook(module, grad_output):
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        module._perf_bwd_start = start

    # -- backward post hook: record end event --------------------------------
    def _bwd_post_hook(module, grad_input, grad_output):
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        # backward FLOPs ≈ 2.5 × forward FLOPs
        bwd_flops = module._perf_fwd_flops * 2.5
        accum.add_bwd(module._perf_bwd_start, end, bwd_flops)

    attn_module.register_forward_pre_hook(_fwd_pre_hook, with_kwargs=True)
    attn_module.register_forward_hook(_fwd_post_hook)
    attn_module.register_full_backward_pre_hook(_bwd_pre_hook)
    attn_module.register_full_backward_hook(_bwd_post_hook)


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
