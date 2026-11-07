# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""R3 EP plan: pure-Python / numpy core (no torch, no Megatron).

With rollout routing replay (R3) every training token's experts are known before the step.  Megatron's alltoall MoE
token dispatcher (``MoEAlltoAllTokenDispatcher.preprocess``) nevertheless, on every MoE layer call,

  * all-gathers the per-expert token counts over the tp x ep group (one collective per layer call), and
  * copies the all-to-all splits / tokens-per-expert to the host and waits for them before the EP all-to-all
    (``cuda_sync_point = "before_ep_alltoall"``): the CPU stalls on the GPU once per MoE layer call.

The plan counts tokens per (micro-batch, layer, expert) from the replayed ids once per step, does one all-gather and
one host copy, and then every layer call sets the same dispatcher attributes from the plan.  This module holds the
parts of that which need no framework: the counting, the derivation of the dispatcher attributes from the gathered
counts (same arithmetic as Megatron), the micro-batch cursor of slime's replay stages, the eligibility rules and the
per-step guard state machine.  The Megatron patch is :mod:`ai_search_plugin.adapters.megatron_r3plan`, the slime glue
:mod:`ai_search_plugin.adapters.slime_r3plan`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

log = logging.getLogger("ai_search_plugin.r3plan")

# Layouts on which the plan was checked in the source implementation (0 plan/actual mismatches): (PP, VPP).
# Anything else runs too, but relies on the runtime guard (verify steps + per-step device check).
VALIDATED_LAYOUTS = ((1, 1), (2, 2), (4, 3))


def layout_status(pp: int, vpp: Optional[int]) -> str:
    """'validated' for a (PP, VPP) checked in the source, else 'guarded' (correctness relies on the runtime guard)."""
    return "validated" if (int(pp or 1), int(vpp or 1)) in VALIDATED_LAYOUTS else "guarded"


# --------------------------------------------------------------------------------------------------- counting
def count_routes(ids, num_experts: int) -> np.ndarray:
    """ids [T, k] expert ids of one micro-batch / layer (negative = no route) -> [E] int64 token counts per expert.

    Each (token, slot) adds one, which is what ``routing_map.sum(dim=0)`` gives for a top-k routing map (top-k ids of a
    token are distinct)."""
    a = np.asarray(ids).reshape(-1).astype(np.int64)
    a = a[a >= 0]
    if a.size and int(a.max()) >= num_experts:
        raise ValueError(f"expert id {int(a.max())} >= num_experts {num_experts}")
    return np.bincount(a, minlength=num_experts).astype(np.int64)


def count_plan(lists: Sequence[Sequence], num_experts: int) -> np.ndarray:
    """lists[layer][microbatch] = ids [T_m, k] -> counts [M, L, E] int64 (same layout as the device plan)."""
    L = len(lists)
    if L == 0:
        return np.zeros((0, 0, num_experts), dtype=np.int64)
    M = len(lists[0])
    if any(len(x) != M for x in lists):
        raise ValueError("every layer must have the same number of micro-batches")
    out = np.zeros((M, L, num_experts), dtype=np.int64)
    for li, per_mb in enumerate(lists):
        for m, ids in enumerate(per_mb):
            out[m, li] = count_routes(ids, num_experts)
    return out


def local_expert_columns(glob: np.ndarray, local_expert_start: int, num_local_experts: int) -> np.ndarray:
    """[..., E] -> [..., nle]: the columns of this rank's local experts (all a rank needs from the gathered counts)."""
    return np.ascontiguousarray(glob[..., local_expert_start:local_expert_start + num_local_experts])


# --------------------------------------------------------------------------------------------------- attributes
@dataclass
class Splits:
    """The dispatcher attributes ``preprocess`` sets (dropless alltoall path), all int64 on the host."""
    input_splits: np.ndarray                       # [ep]   tokens this rank sends to each EP rank
    output_splits: np.ndarray                      # [ep]   tokens this rank receives from each EP rank (own TP rank)
    output_splits_tp: np.ndarray                   # [tp]   tokens received per TP rank (after the TP all-gather)
    tokens_per_local_expert: np.ndarray            # [nle]  the return value of preprocess
    num_global_tokens_per_local_expert: np.ndarray  # [tp*ep, nle] (only used by Megatron when nle > 1)


def splits_from_counts(local_counts, glob_local, *, ep: int, tp: int, tp_rank: int,
                       num_local_experts: int) -> Splits:
    """Dispatcher attributes of one layer call from the plan.

    local_counts [E]        this rank's tokens per expert (its row of the gathered counts)
    glob_local   [tp*ep, nle] every tp x ep rank's tokens to THIS rank's local experts, in tp_ep-group rank order
                 (Megatron's tp_ep group is ep-major / tp-minor: group rank = ep_idx * tp + tp_idx)
    Same arithmetic as ``MoEAlltoAllTokenDispatcher.preprocess`` after its all-gather."""
    nle = int(num_local_experts)
    local_counts = np.asarray(local_counts, dtype=np.int64).reshape(-1)
    glob_local = np.asarray(glob_local, dtype=np.int64)
    if local_counts.size != ep * nle:
        raise ValueError(f"local counts have {local_counts.size} experts, expected ep*nle = {ep * nle}")
    if glob_local.shape != (ep * tp, nle):
        raise ValueError(f"gathered counts {glob_local.shape}, expected {(ep * tp, nle)}")
    input_splits = local_counts.reshape(ep, nle).sum(axis=1)
    ngtle = glob_local.reshape(ep, tp, nle).transpose(1, 0, 2)          # [tp, ep, nle]
    per_rank = ngtle.sum(axis=2)                                          # [tp, ep]
    return Splits(input_splits=input_splits,
                  output_splits=np.ascontiguousarray(per_rank[tp_rank]),
                  output_splits_tp=per_rank.sum(axis=1),
                  tokens_per_local_expert=ngtle.sum(axis=(0, 1)),
                  num_global_tokens_per_local_expert=np.ascontiguousarray(ngtle).reshape(-1, nle))


# --------------------------------------------------------------------------------------------------- cursor
REPLAY_STAGES = ("replay_forward", "replay_backward")


def replay_cursor(stage: Optional[str], forward_index: int, backward_index: int) -> Optional[int]:
    """Micro-batch of the current layer call from slime's RoutingReplay counters.

    The router pops its replay entry right before the dispatcher runs, so the entry just used is index - 1 of the
    counter that the stage selects ('replay_forward' for log-prob / training forwards, 'replay_backward' for the
    activation-recompute forward inside backward).  Any other stage ('fallthrough', 'record', unset) means the
    routing was computed, not replayed: no plan (None)."""
    if stage == "replay_forward":
        m = int(forward_index) - 1
    elif stage == "replay_backward":
        m = int(backward_index) - 1
    else:
        return None
    return m if m >= 0 else None


# --------------------------------------------------------------------------------------------------- eligibility
def ineligible_reason(dispatcher) -> Optional[str]:
    """Why a dispatcher call cannot use the plan (None = it can).  Config-only, so the answer is the same on every rank
    of the tp x ep group (a per-rank fallback would desynchronize the group's collectives)."""
    cfg = getattr(dispatcher, "config", None)
    if getattr(dispatcher, "drop_and_pad", False):
        return "drop_and_pad"
    if cfg is None:
        return "no config"
    if getattr(cfg, "moe_expert_capacity_factor", None) is not None:
        return "capacity factor (token dropping)"
    if getattr(cfg, "moe_router_padding_for_quantization", False):
        return "router padding for quantization"
    cg = getattr(cfg, "cuda_graph_impl", "none")
    if cg not in (None, "none", False):
        return f"cuda graphs ({cg})"
    if getattr(dispatcher, "cuda_dtoh_point", "before_permutation_1") != "before_permutation_1":
        return "non-default dtoh point"
    return None


# --------------------------------------------------------------------------------------------------- guard
@dataclass
class StepGuard:
    """Per-step guard (same decisions on every rank: the caller feeds it the group-wide MAX of the bad counts).

    verify_steps  first N steps that built a plan run the ORIGINAL preprocess on every call and compare it exactly
                  with the plan (the original values are used, so these steps are exact even if the plan is wrong)
    strict        raise on a mismatch instead of falling back
    On any mismatch (verify difference or device check of the actual routing vs the plan) the plan is switched off
    for the rest of the run and every later call takes Megatron's original path; a warning names the step."""
    verify_steps: int = 1
    strict: bool = False
    disabled: bool = False
    disabled_why: str = ""
    steps: int = 0
    planned_steps: int = 0
    history: List[Dict] = field(default_factory=list)

    def verifying(self) -> bool:
        return not self.disabled and self.planned_steps < self.verify_steps

    def end_step(self, *, mismatch: int, verify_fail: int, planned_calls: int, fallback_calls: int,
                 built: bool) -> Dict:
        """Close one step.  mismatch / verify_fail: group-wide counts (already reduced).  Returns the step record."""
        was_verifying = self.verifying()
        rec = {"step": self.steps, "built": bool(built), "verify": bool(built and was_verifying),
               "planned_calls": int(planned_calls), "fallback_calls": int(fallback_calls),
               "mismatch": int(mismatch), "verify_fail": int(verify_fail), "disabled": self.disabled}
        self.steps += 1
        if built:
            self.planned_steps += 1
        bad = int(mismatch) + int(verify_fail)
        if bad and not self.disabled:
            what = (f"{verify_fail} verified calls differed from Megatron's preprocess (original values were used)"
                    if verify_fail else
                    f"{mismatch} layer calls routed differently from the plan (this step's MoE dispatch is suspect)")
            self.disabled = True
            self.disabled_why = f"step {rec['step']}: {what}"
            rec["disabled"] = True
            rec["action"] = "raise" if self.strict else "fallback"
            msg = f"r3 EP plan: {what}; plan switched off, Megatron's original dispatcher path from now on"
            if self.strict:
                self.history.append(rec)
                raise RuntimeError(msg + " (ASRT_R3PLAN_STRICT=1)")
            log.warning(msg)
        self.history.append(rec)
        del self.history[:-32]
        return rec
