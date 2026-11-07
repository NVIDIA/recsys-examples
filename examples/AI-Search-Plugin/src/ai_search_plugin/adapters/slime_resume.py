""" Keep slime's weight version continuous across a checkpoint resume (non-straw transport).

slime numbers weight syncs from ``args.update_weight_start_version``: the updater starts there
(backends/megatron_utils/update_weight/__init__.py:55) and the first sync publishes start + 1 (train.py pushes
the weights once before the loop).  A run from rollout 0 starts at 0, so rollout r is generated at version r + 1.

On a resume with ``--load <megatron ckpt>`` and object-store transport, nothing sets it:
  * resolve_checkpoint (data/checkpoint.py:247) returns at once unless the transport is straw, so the
    ``update_weight_start_version = weight_version / step + 1`` seeding (checkpoint.py:391) never runs;
  * RayTrainGroup copies getattr(args, "update_weight_start_version", 0) = 0 (ray/actor_group.py:59) and writes it
    back onto args before every actor init (actor_group.py:214);
  * so the actor's own fallback "Model-only resumes keep the serving version aligned with the next rollout"
    (backends/megatron_utils/actor.py:147-152, only ``if not hasattr(args, ...)``) never fires.
The engines then restart at version 1 after every resume (929s segment b: rollout 2 generated at 1, not 3), so
the versions on restored samples are newer than the engines', slime's own staleness metric marks them unknown,
and anything that counts versions (asrt ticket ceiling, token-level staleness log) loses its reference.

This sets ``args.update_weight_start_version = start_rollout_id`` (= loaded rollout id + 1, slime's own intent at
actor.py:150-151) in each training actor right before the weight updater is created, only when a slime save was
loaded (not --finetune, $load/rollout/global_dataset_state_dict_<loaded id>.pt exists) and the value is still
slime's default 0.  A run from an HF model (finetune) or a converted Megatron base checkpoint (no data-source state)
is unchanged; straw / recovery resumes that already seeded a nonzero version and the full disk transport (the driver
keeps its own weight_v<N> counter) are left alone.  slime's flush schedule handles a nonzero start (utils/weight_sync.py:4-12: first
publication flushes, phase kept relative to version 1).

Installed by ``ai_search_plugin.adapters.slime_weightsync.megatron_init``, or on its own for a stock-slime
run:  ``--custom-megatron-init-path ai_search_plugin.adapters.slime_resume.megatron_init`` (no other asrt code is
loaded).  ASRT_RESUME_VERSION=0 turns it off.
"""
from __future__ import annotations

import os
import sys

_ACTOR = "slime.backends.megatron_utils.actor"


def _on() -> bool:
    return os.environ.get("ASRT_RESUME_VERSION", "1").strip().lower() not in ("0", "off", "no", "false")


def _rank0() -> bool:
    try:
        import torch.distributed as dist

        return not dist.is_initialized() or dist.get_rank() == 0
    except Exception:
        return True


def seed_version(args, loaded_rollout_id):
    """Set args.update_weight_start_version for a resumed run; returns the value set, or None (unchanged)."""
    if not _on() or loaded_rollout_id is None:
        return None
    try:
        lr = int(loaded_rollout_id)
    except (TypeError, ValueError):
        return None
    if lr < 0 or getattr(args, "finetune", False):
        return None  # fresh run (HF / base model, nothing loaded): slime's 0 is right
    if getattr(args, "update_weight_mode", None) == "full" and getattr(args, "update_weight_transport", None) == "disk":
        # full disk transport: the driver numbers weight_v<N> dirs itself from its own counter (actor_group.py:59,
        # 184-187), which stays 0 - seeding only the actor would make the two disagree
        return None
    load = getattr(args, "load", None)
    if not load or not os.path.exists(os.path.join(str(load), "rollout", f"global_dataset_state_dict_{lr}.pt")):
        # only a slime resume: a slime save writes the rollout data-source state next to the model.  A converted
        # Megatron base checkpoint loaded with --load (real-weight runs, iteration 0) has none and stays unchanged
        return None
    cur = getattr(args, "update_weight_start_version", 0) or 0
    try:
        cur = int(cur)
    except (TypeError, ValueError):
        return None
    if cur != 0:
        return None  # straw / training recovery already seeded the version
    sid = getattr(args, "start_rollout_id", None)
    start = int(sid) if sid is not None else lr + 1   # = actor.py:150-151
    if start <= 0:
        return None
    args.update_weight_start_version = start
    if _rank0():
        print(f"[asrt] resume: weight version continues from {start} (loaded rollout {lr}; first sync publishes "
              f"{start + 1}; slime left it at 0 for a non-straw resume)", flush=True)
    return start


def install(args=None) -> bool:
    """Patch the actor module's initialize_model_and_optimizer (records the loaded rollout id) and
    create_weight_updater (seeds the version first).  Both are looked up as module globals at call time inside
    MegatronTrainRayActor.init, which calls the custom Megatron init hook before either."""
    if not _on():
        return False
    mod = sys.modules.get(_ACTOR)
    if mod is None:
        try:
            import importlib

            mod = importlib.import_module(_ACTOR)
        except Exception as e:  # noqa: BLE001
            print(f"[asrt] slime_resume: actor module unavailable: {e!r}", flush=True)
            return False
    if getattr(mod, "_asrt_resume_patched", False):
        return True
    orig_init = getattr(mod, "initialize_model_and_optimizer", None)
    orig_cwu = getattr(mod, "create_weight_updater", None)
    if orig_init is None or orig_cwu is None:
        print("[asrt] slime_resume: actor module has no initialize_model_and_optimizer/create_weight_updater; "
              "version seeding not installed", flush=True)
        return False
    seen = {}

    def initialize_model_and_optimizer(*a, **kw):
        out = orig_init(*a, **kw)
        try:
            seen["loaded"] = out[3]
        except Exception:  # noqa: BLE001
            pass
        return out

    def create_weight_updater(args_, *a, **kw):
        try:
            seed_version(args_, seen.get("loaded"))
        except Exception as e:  # noqa: BLE001  never break actor init
            print(f"[asrt] slime_resume: seeding failed: {e!r}", flush=True)
        return orig_cwu(args_, *a, **kw)

    initialize_model_and_optimizer.__wrapped__ = orig_init
    create_weight_updater.__wrapped__ = orig_cwu
    mod.initialize_model_and_optimizer = initialize_model_and_optimizer
    mod.create_weight_updater = create_weight_updater
    mod._asrt_resume_patched = True
    return True


def megatron_init(args):
    """Entry for ``--custom-megatron-init-path`` (stock-slime baseline arms)."""
    ok = install(args)
    if _rank0():
        print(f"[asrt] slime_resume installed: {ok}", flush=True)
    return ok
