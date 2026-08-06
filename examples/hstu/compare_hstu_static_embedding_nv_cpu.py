"""Compare NV zero-static-embedding ranking loss with a CPU FP32 reference."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from configs.hstu_config import HSTULayerType, KernelBackend, get_hstu_config
from modules.fused_hstu_layer import FusedHSTULayer
from modules.jagged_data import JaggedData

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[2] / "fixed_hstu_weights.pt"


class FP32MasterAdam(torch.optim.Optimizer):
    """Adam with FP32 master copies for low-precision model parameters."""

    def __init__(self, params, **kwargs):
        params = list(params)
        super().__init__(params, kwargs)
        self._masters = {
            parameter: parameter.detach().float().clone()
            for parameter in params
        }
        self._adam = torch.optim.Adam(self._masters.values(), **kwargs)

    @torch.no_grad()
    def step(self, closure=None):
        for parameter, master in self._masters.items():
            if parameter.grad is not None:
                master.grad = parameter.grad.detach().float()
        loss = self._adam.step(closure)
        for parameter, master in self._masters.items():
            parameter.copy_(master.to(dtype=parameter.dtype))
        return loss

    def zero_grad(self, set_to_none=True):
        for parameter in self._masters:
            parameter.grad = None
        return super().zero_grad(set_to_none=set_to_none)


class CPUHSTULayer(torch.nn.Module):
    def __init__(self, hidden, heads, scaling, num_targets):
        super().__init__()
        self.hidden, self.heads = hidden, heads
        self.head_dim = hidden // heads
        self.scaling = scaling
        self.num_targets = num_targets
        self.uvqk = torch.nn.Parameter(torch.empty(hidden, 4 * hidden))
        self.proj = torch.nn.Parameter(torch.empty(hidden, hidden))

    def forward(self, x):
        batch, seq, _ = x.shape
        norm = F.layer_norm(x, (self.hidden,), eps=1e-6)
        uvqk = F.silu(norm @ self.uvqk).reshape(
            batch, seq, 4, self.heads, self.head_dim
        )
        u, v, q, k = uvqk.unbind(2)
        scores = torch.einsum("bihd,bjhd->bhij", q, k) * (self.head_dim ** -0.5)
        scores = F.silu(scores) / self.scaling
        positions = torch.arange(seq, device=x.device)
        row = positions.view(seq, 1)
        col = positions.view(1, seq)
        causal = row >= col
        target_ids = torch.clamp(row - seq + self.num_targets, min=-1)
        target_dist = target_ids - target_ids.t()
        target_mask = (target_dist == 0) | (target_ids < 0) | (target_ids.t() < 0)
        scores = scores * (causal & target_mask).view(1, 1, seq, seq)
        attn = torch.einsum("bhij,bjhd->bihd", scores, v).reshape(
            batch, seq, self.hidden
        )
        post = F.layer_norm(attn, (self.hidden,), eps=1e-6) * u.reshape(
            batch, seq, self.hidden
        )
        return post @ self.proj + x


class CPUHSTU(torch.nn.Module):
    def __init__(self, args, weights):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            CPUHSTULayer(
                args.hidden_size, args.num_heads, args.seq_len * 2,
                args.num_target_tokens * 2
            )
            for _ in range(args.num_layers)
        ])
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                layer.uvqk.copy_(weights["blocks"][i]["uvqk_in_out"].float())
                layer.proj.copy_(weights["blocks"][i]["projection_in_out"].float())

    def forward(self, x):
        batch = x.shape[0] // self.layers[0].scaling
        x = x.reshape(batch, self.layers[0].scaling, -1)
        for layer in self.layers:
            x = layer(x)
        return x.flatten(0, 1)


def print_gradient_comparison(name, nv_grad, cpu_grad):
    if nv_grad is None or cpu_grad is None:
        print(f"  {name:<24s} missing_grad nv={nv_grad is None} cpu={cpu_grad is None}")
        return
    nv = nv_grad.detach().float().cpu().reshape(-1)
    cpu = cpu_grad.detach().float().cpu().reshape(-1)
    diff = nv - cpu
    relative_l2 = diff.norm() / cpu.norm().clamp_min(1e-30)
    cosine = F.cosine_similarity(nv, cpu, dim=0, eps=1e-30)
    print(
        f"  {name:<24s}"
        f" nv_max={nv.abs().max().item():.6e}"
        f" cpu_max={cpu.abs().max().item():.6e}"
        f" diff_max={diff.abs().max().item():.6e}"
        f" rel_l2={relative_l2.item():.6e}"
        f" cosine={cosine.item():.6f}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--num-target-tokens", type=int, default=2)
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument(
        "--same-state-cpu",
        action="store_true",
        help="Reset the CPU reference from the current NV model before each step",
    )
    p.add_argument("--debug-gradients", action="store_true")
    p.add_argument("--debug-gradient-steps", type=int, default=3)
    args = p.parse_args()
    weights = torch.load(args.weights, map_location="cpu", weights_only=False)
    device, dtype = torch.device("cuda"), torch.bfloat16
    dim = args.hidden_size // args.num_heads

    # NV's ranking path uses a TorchRec EmbeddingCollection. Keep both tables,
    # but explicitly zero their storage before lookup.
    ec = torch.nn.ModuleDict({
        "item_id": torch.nn.Embedding(1024, args.hidden_size, device=device),
        "action": torch.nn.Embedding(args.num_classes, args.hidden_size, device=device),
    })
    with torch.no_grad():
        for parameter in ec.parameters():
            parameter.zero_()
    n = args.batch_size * args.seq_len
    lengths = torch.full((args.batch_size,), args.seq_len, device=device, dtype=torch.long)
    ids = (torch.arange(n, device=device) * 17 + 3) % 1024
    actions = (torch.arange(n, device=device) * 3 + 1) % args.num_classes
    timestamps = torch.arange(args.seq_len, device=device).repeat(args.batch_size)
    offsets = torch.arange(0, n * 2 + 1, args.seq_len * 2, device=device, dtype=torch.long)
    seq_lengths = torch.full((args.batch_size,), args.seq_len * 2, device=device, dtype=torch.long)
    target_lengths = torch.full(
        (args.batch_size,), args.num_target_tokens * 2, device=device, dtype=torch.long
    )
    target_offsets = torch.cat(
        (target_lengths.new_zeros(1), torch.cumsum(target_lengths, dim=0))
    )
    config = get_hstu_config(hidden_size=args.hidden_size, kv_channels=dim,
        num_attention_heads=args.num_heads, num_layers=args.num_layers, dtype=dtype,
        hidden_dropout=0.0, norm_epsilon=1e-6, is_causal=True,
        kernel_backend=KernelBackend.CUTLASS, target_group_size=1,
        hstu_layer_type=HSTULayerType.FUSED, learnable_input_layernorm=False,
        learnable_output_layernorm=False, residual=True, add_uvqk_bias=False,
        scaling_seqlen=args.seq_len * 2, disable_contextual_mask=False,
        # Bypass get_hstu_config's TP query for this standalone single-card run.
        is_inference=True)
    # get_hstu_config uses is_inference only to bypass Megatron TP lookup.
    # FusedHSTULayer itself must run in training mode for input gradients.
    config.is_inference = False
    layers = torch.nn.ModuleList([FusedHSTULayer(config) for _ in range(args.num_layers)]).to(device=device, dtype=dtype)
    head = torch.nn.Linear(args.hidden_size, args.num_classes, device=device, dtype=dtype)
    with torch.no_grad():
        for i, layer in enumerate(layers):
            layer._linear_uvqk_weight.copy_(weights["blocks"][i]["uvqk_in_out"].to(device, dtype))
            layer._linear_uvqk_bias.zero_()
            layer._linear_proj_weight.copy_(weights["blocks"][i]["projection_in_out"].to(device, dtype))
            layer._output_layernorm_weight.fill_(1.0); layer._output_layernorm_bias.zero_()
        head.weight.copy_(weights["prediction_head_weight_out_in"].to(device, dtype))
        head.bias.copy_(weights["prediction_head_bias"].to(device, dtype))
    # The production BF16 path keeps dense parameters in BF16 but updates an
    # FP32 master copy. Direct Adam(BF16 parameters) can quantize the update
    # away, making the NV loss appear stuck (especially from zero embeddings).
    optimizer = FP32MasterAdam(
        list(ec.parameters()) + list(layers.parameters()) + list(head.parameters()),
        lr=1e-3,
    )
    cpu_ec = torch.nn.ModuleDict({
        "item_id": torch.nn.Embedding(1024, args.hidden_size),
        "action": torch.nn.Embedding(args.num_classes, args.hidden_size),
    })
    cpu_hstu = CPUHSTU(args, weights)
    cpu_head = torch.nn.Linear(args.hidden_size, args.num_classes, dtype=torch.float32)
    with torch.no_grad():
        for parameter in cpu_ec.parameters():
            parameter.zero_()
        cpu_head.weight.copy_(weights["prediction_head_weight_out_in"].float())
        cpu_head.bias.copy_(weights["prediction_head_bias"].float())
    cpu_optimizer = torch.optim.Adam(
        list(cpu_ec.parameters()) + list(cpu_hstu.parameters()) + list(cpu_head.parameters()), lr=1e-3
    )
    for step in range(args.num_steps):
        optimizer.zero_grad(set_to_none=True)
        cpu_optimizer.zero_grad(set_to_none=True)
        if args.same_state_cpu or args.debug_gradients:
            # Compare both implementations at exactly the same model state.
            # Casting the NV model tensors to FP32 isolates fused BF16 numeric
            # error from divergence caused by two independent optimizers.
            with torch.no_grad():
                for name in ("item_id", "action"):
                    cpu_ec[name].weight.copy_(ec[name].weight.detach().cpu().float())
                for nv_layer, cpu_layer in zip(layers, cpu_hstu.layers):
                    cpu_layer.uvqk.copy_(
                        nv_layer._linear_uvqk_weight.detach().cpu().float()
                    )
                    cpu_layer.proj.copy_(
                        nv_layer._linear_proj_weight.detach().cpu().float()
                    )
                cpu_head.weight.copy_(head.weight.detach().cpu().float())
                cpu_head.bias.copy_(head.bias.detach().cpu().float())
        with torch.enable_grad():
            embedded = {"item_id": ec["item_id"](ids), "action": ec["action"](actions)}
            values = torch.stack((embedded["item_id"], embedded["action"]), 1).flatten(0, 1)
            values = values.to(dtype).reshape(n * 2, args.hidden_size)
            values.retain_grad()
            jagged = JaggedData(
                values, seq_lengths, offsets, args.seq_len * 2,
                max_num_candidates=args.num_target_tokens * 2,
                num_candidates=target_lengths,
                num_candidates_offsets=target_offsets,
                has_interleaved_action=True, scaling_seqlen=args.seq_len * 2
            )
            out = jagged
            nv_layer_outputs = []
            for layer in layers:
                out = layer(out)
                if args.debug_gradients and step < args.debug_gradient_steps:
                    out.values.retain_grad()
                nv_layer_outputs.append(out.values)
            states = out.values.reshape(
                args.batch_size, args.seq_len * 2, args.hidden_size
            )[:, -args.num_target_tokens * 2::2]
            nv_logits = head(states).reshape(-1, args.num_classes)
            labels = torch.tensor([1, 4, 7, 0], device=device)[:nv_logits.shape[0]]
            nv_loss = F.cross_entropy(nv_logits.float(), labels)
            cpu_embedded = {
                "item_id": cpu_ec["item_id"](ids.cpu()),
                "action": cpu_ec["action"](actions.cpu()),
            }
            cpu_values = torch.stack(
                (cpu_embedded["item_id"], cpu_embedded["action"]), 1
            ).flatten(0, 1)
            if args.debug_gradients and step < args.debug_gradient_steps:
                cpu_values.retain_grad()
            cpu_x = cpu_values.reshape(
                args.batch_size, args.seq_len * 2, args.hidden_size
            )
            cpu_layer_outputs = []
            for layer in cpu_hstu.layers:
                cpu_x = layer(cpu_x)
                if args.debug_gradients and step < args.debug_gradient_steps:
                    cpu_x.retain_grad()
                cpu_layer_outputs.append(cpu_x)
            cpu_out = cpu_x.flatten(0, 1)
            cpu_states = cpu_out.reshape(
                args.batch_size, args.seq_len * 2, args.hidden_size
            )[:, -args.num_target_tokens * 2::2].reshape(-1, args.hidden_size)
            cpu_logits = cpu_head(cpu_states)
            cpu_labels = torch.tensor([1, 4, 7, 0], dtype=torch.long)[:cpu_logits.shape[0]]
            cpu_loss = F.cross_entropy(cpu_logits, cpu_labels)
            nv_loss.backward()
            cpu_loss.backward()
            if args.debug_gradients and step < args.debug_gradient_steps:
                print(f"gradient_comparison step={step}")
                print_gradient_comparison("hstu_input", values.grad, cpu_values.grad)
                for index, (nv_output, cpu_output) in enumerate(
                    zip(nv_layer_outputs, cpu_layer_outputs)
                ):
                    print_gradient_comparison(
                        f"layer{index}.output", nv_output.grad, cpu_output.grad
                    )
                    print_gradient_comparison(
                        f"layer{index}.uvqk",
                        layers[index]._linear_uvqk_weight.grad,
                        cpu_hstu.layers[index].uvqk.grad,
                    )
                    print_gradient_comparison(
                        f"layer{index}.projection",
                        layers[index]._linear_proj_weight.grad,
                        cpu_hstu.layers[index].proj.grad,
                    )
                print_gradient_comparison("head.weight", head.weight.grad, cpu_head.weight.grad)
                print_gradient_comparison("head.bias", head.bias.grad, cpu_head.bias.grad)
                for name in ("item_id", "action"):
                    print_gradient_comparison(
                        f"embedding.{name}",
                        ec[name].weight.grad,
                        cpu_ec[name].weight.grad,
                    )
            embedding_grads = [
                parameter.grad.detach().float().abs().max().item()
                for parameter in ec.parameters()
                if parameter.grad is not None
            ]
            embedding_grad = max(embedding_grads, default=0.0)
            input_grad = (
                values.grad.detach().float().abs().max().item()
                if values.grad is not None else 0.0
            )
            trainable_nv_parameters = [
                parameter
                for parameter in list(layers.parameters()) + list(head.parameters())
                if parameter.requires_grad
            ]
            dense_grad = max(
                (
                    parameter.grad.detach().float().abs().max().item()
                    for parameter in trainable_nv_parameters
                    if parameter.grad is not None
                ),
                default=0.0,
            )
            dense_before = [
                parameter.detach().float().clone()
                for parameter in trainable_nv_parameters
            ]
            optimizer.step()
            if not (args.same_state_cpu or args.debug_gradients):
                cpu_optimizer.step()
            dense_update = max(
                (
                    (parameter.detach().float() - before).abs().max().item()
                    for parameter, before in zip(
                        trainable_nv_parameters, dense_before
                    )
                ),
                default=0.0,
            )
            embedding_weight = max(
                parameter.detach().float().abs().max().item()
                for parameter in ec.parameters()
            )
            print(
                f"step={step} "
                f"NV_BF16_loss={nv_loss.item():.10f} "
                f"CPU_FP32_loss={cpu_loss.item():.10f}"
            )


if __name__ == "__main__":
    main()
