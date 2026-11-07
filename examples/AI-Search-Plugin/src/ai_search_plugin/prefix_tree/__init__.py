# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prefix-tree training of GRPO groups (``prefix_tree``): the shared prompt of a group is trained once.

The n samples of a GRPO group share their prompt.  The training micro-batch is packed as a tree (the shared prefix
once = trunk, then one branch per sample); token-wise layers run on the tree rows only, sequence mixers run on a
virtual pack whose documents are exactly the original samples, and outputs are mapped back to the flat layout.  Loss
and gradients equal the flat path up to floating-point rounding (sibling gradients of a trunk row are summed).

Modules (torch / Megatron / slime / verl are imported lazily, only by the modules that need them):
  plan               host-side layout, index maps, sibling grouping (numpy)
  model              Megatron layer wrappers + RankPlan (torch, Megatron)
  adapters.slime     slime training actor: ``megatron_init`` (wraps get_batch + TransformerBlock.forward)
  adapters.slime_pack  slime rollout manager: sibling-aware micro-batch schedule (``--rollout-function-path``)
  adapters.verl      verl Megatron engine: tree packing, positions, output mapping, micro-batching, R3 order
  adapters.verl_driver verl driver: whole GRPO groups per DP rank
  verl_plugin        ``VERL_USE_EXTERNAL_MODULES`` entry (driver + workers)

Switch: ASRT_PREFIX_TREE=1 in every process.  Options: PREFIX_TREE_MIN_TRUNK (slime, default 16 tokens),
ASRT_PREFIX_TREE_MIN_FRAC (verl, default 0.25), PREFIX_TREE_PACK (slime packer, default 1), PREFIX_TREE_ATTN
(full | core), PREFIX_TREE_GDN (full | state), PREFIX_TREE_FAST_ROWS, PREFIX_TREE_SLIME_CHECK=K (numerics log).
"""
import os

NAME = "prefix_tree"


def enabled(environ=None):
    env = os.environ if environ is None else environ
    return env.get("ASRT_PREFIX_TREE", "0") not in ("0", "", "false", "False")


def install_verl():
    """Register the verl hooks in this process (driver: DP balancing; workers: Megatron engine).  Returns the list
    of registered hooks ([] when ASRT_PREFIX_TREE is off)."""
    if not enabled():
        return []
    from ..adapters._hooks import when_imported
    from .adapters import verl_driver
    out = []
    if verl_driver.install():
        out.append("driver:" + verl_driver.TRAINER_MODULE)

    def on_engine(_mod):
        from .adapters import verl as V
        rep = V.install()
        print(f"[asrt] prefix_tree verl engine pid={os.getpid()} {rep}", flush=True)

    when_imported("verl.workers.engine.megatron.transformer_impl", on_engine)
    out.append("engine:verl.workers.engine.megatron.transformer_impl")
    return out


def install(framework="slime"):
    """Install in THIS process: slime -> the training-actor hooks (same as adapters.slime.megatron_init);
    verl -> the verl hooks (same as importing verl_plugin)."""
    if framework == "slime":
        from .adapters import slime
        return slime.megatron_init()
    if framework == "verl":
        return install_verl()
    raise ValueError(f"framework must be 'slime' or 'verl', got {framework!r}")
