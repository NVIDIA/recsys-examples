"""Driver wrapper for slime's train.py with asrt control-plane patches.

    python3 -m ai_search_plugin.adapters.slime_train <path/to/slime>/train.py <normal slime args>

Behaviour with ``ASRT_PREFETCH`` unset: identical to ``python3 train.py``.
With ``ASRT_PREFETCH`` containing ``overlap`` the train loop is replaced by
one that requests batch i+1 from the rollout manager while the trainers are
running update_weights(i) and posts it to the prefetch board, so every
trainer's background thread pulls it during the weight sync (see
``slime_prefetch``).  The RolloutManager is created as a threaded actor
(``max_concurrency``) so control calls made during the update are not
queued behind ``generate``.

Unsupported settings (colocate, offload, critic, straw transport, external
rollout, fault tolerance, eval steps) silently use slime's loop/step.
"""
from __future__ import annotations

import logging
import os
import runpy
import sys
import time

log = logging.getLogger("ai_search_plugin.slime_train")


def _supported(args) -> bool:
    return not (
        args.colocate or args.offload_train or args.offload_rollout or args.use_critic
        # object-store and straw both hand the trainer a reference that slime's own _get_rollout_data resolves;
        # with straw (slime's streaming handoff) this adds the TRAINER-side half: next batch pulled during the sync
        or getattr(args, "rollout_data_transport", "object-store") not in ("object-store", "straw")
        or getattr(args, "rollout_external", False) or getattr(args, "use_fault_tolerance", False)
        or getattr(args, "release_train", False) or getattr(args, "debug_train_only", False)
        or getattr(args, "debug_rollout_only", False)
    )


def make_train(orig_train, T):
    """T = the imported slime train module (for its globals)."""
    import ray

    from . import slime_prefetch

    def train(args, pgs, rollout_manager, num_rollout_per_epoch, restore_plan):
        if "overlap" not in slime_prefetch.opts() or not _supported(args):
            log.warning("asrt overlap disabled for this configuration; using slime's loop")
            return orig_train(args, pgs, rollout_manager, num_rollout_per_epoch, restore_plan)
        should = T.should_run_periodic_action
        board = slime_prefetch.get_or_create_board()
        actor_model, _ = T.create_training_models(args, pgs, rollout_manager)
        actor_model.update_weights()
        ray.get(rollout_manager.training_ready.remote())
        if args.check_weight_update_equal:
            ray.get(rollout_manager.check_weights.remote(action="compare"))
        if args.num_rollout == 0 and args.eval_interval is not None:
            ray.get(rollout_manager.eval.remote(rollout_id=0))

        pending = None  # (rollout_id, ObjectRef of generate)
        for rollout_id in range(args.start_rollout_id, args.num_rollout):
            if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
                ray.get(rollout_manager.eval.remote(rollout_id))
            t0 = time.time()
            if pending is None or pending[0] != rollout_id:
                pending = (rollout_id, rollout_manager.generate.remote(rollout_id))
            rollout_data_ref = ray.get(pending[1])
            pending = None
            t_gen = time.time() - t0
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
            ray.get(rollout_manager.training_completed.remote(rollout_id))
            if should(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
                T.save_checkpoint(args, rollout_id, actor_model, None, rollout_manager, actor_trains=True)
            actor_model.clear_memory()

            nxt = rollout_id + 1
            eval_now = should(rollout_id, args.eval_interval, num_rollout_per_epoch)
            overlap = nxt < args.num_rollout and not eval_now
            was_paused = ray.get(rollout_manager.pause_rollout_admission.remote())
            t1 = time.time()
            upd = [a.update_weights.remote() for a in actor_model._actor_handlers]
            posted_at = None
            if overlap:
                gen = rollout_manager.generate.remote(nxt)
                pending = (nxt, gen)
                left = list(upd)
                while left:
                    if posted_at is None:
                        ready, _ = ray.wait([gen], timeout=0.0)
                        if ready:
                            board.post.remote(nxt, ray.get(gen))
                            posted_at = time.time() - t1
                    _, left = ray.wait(left, num_returns=len(left), timeout=0.05)
            ray.get(upd)
            t_upd = time.time() - t1
            if nxt < args.num_rollout:
                ray.get(rollout_manager.resume_rollout_admission.remote(was_paused))
            if overlap and posted_at is None:
                # still post early: the trainers' threads start pulling as soon as it is ready,
                # train(nxt) joins that pull instead of starting its own
                refs = ray.get(gen)
                board.post.remote(nxt, refs)
            print(f"[asrt] step {rollout_id}: wait_generate {t_gen:.1f}s update_weights {t_upd:.1f}s "
                  f"next_batch_posted_during_update={posted_at is not None}"
                  + (f" at +{posted_at:.1f}s" if posted_at is not None else ""), flush=True)
            if eval_now:
                ray.get(rollout_manager.eval.remote(rollout_id))

    return train


def _threaded_manager():
    """Make ``RolloutManager.options(...)`` add max_concurrency (threaded actor)."""
    from slime.ray import rollout as pgm  # create_rollout_manager does `from .rollout import RolloutManager`

    RM = pgm.RolloutManager
    if getattr(RM, "_asrt_wrapped", False):
        return
    conc = int(os.environ.get("ASRT_MANAGER_CONCURRENCY", 4))

    class _Wrap:
        _asrt_wrapped = True

        def __getattr__(self, k):
            return getattr(RM, k)

        def options(self, **kw):
            kw.setdefault("max_concurrency", conc)
            return RM.options(**kw)

        def remote(self, *a, **kw):
            return RM.options(max_concurrency=conc).remote(*a, **kw)

    pgm.RolloutManager = _Wrap()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or not argv[0].endswith(".py"):
        raise SystemExit("usage: python -m ai_search_plugin.adapters.slime_train <path/to/slime>/train.py <slime args>")
    script = os.path.abspath(argv[0])
    sys.argv = [script] + argv[1:]
    sys.path.insert(0, os.path.dirname(script))
    from . import slime_prefetch

    if "overlap" not in slime_prefetch.opts():
        runpy.run_path(script, run_name="__main__")
        return
    import importlib

    T = importlib.import_module(os.path.splitext(os.path.basename(script))[0])
    _threaded_manager()
    T.train = make_train(T.train, T)
    args, restore_plan = T.parse_args(return_restore_plan=True)
    T.configure_logger()
    T.main(args, restore_plan)


if __name__ == "__main__":
    main()
