# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The order in which the growth lane is validated on a GPU, and what each step decides.

Nothing in this branch was compiled by nvcc or run on a GPU when it was written. The gates of
this directory decide whether the LAST commit (the one that deletes the
``torch.cuda.synchronize()`` of ``ExtendableBuffer.extend``) may merge, but they are not the
only things that have to pass, and the first thing that fails on a GPU will not be a gate.
Run the steps in this order; stop at the first failure, and read ``python run_order.py`` for
this text. ``R`` is ``corelib/dynamicemb``, one H100 (``DEMB_ARCHS=90``), the PyTorch of the CI
image. Torchrun suites bind port 29500 and run one at a time; the single-GPU scripts can share
a node with ``CUDA_VISIBLE_DEVICES``.

0. Build. ``cd R && DEMB_ARCHS=90 MAX_JOBS=96 python setup.py build_ext --inplace``. The
   kernels of ``src/table_operation/table_rows.cu`` (``table_carry_slot_state``,
   ``table_carry_reclaimed_rows``), the bindings of ``table.cu`` / ``table.cuh`` and the
   buffers of ``src/vmm_tensor.cu`` have only ever been compiled by a host compiler against
   stand-ins. Build the tree before the lane (``355a1f61``) the same way, in another
   directory, for step 8.

1. The checks that need no GPU, on this box too (the harness of ``host_only`` needs a C++
   compiler)::

       cd R/test/unit_tests && python -m pytest -q growth/test_growth_order_logic.py \\
           growth/test_growth_hooks.py growth/test_value_move_model.py \\
           growth/host_only/test_host_only_vmm.py no_host_sync/test_sync_free_lint.py \\
           no_host_sync/test_host_sync_probe_unit.py no_host_sync/test_sync_ratchet.py

2. The suites that run the code this lane changed (all of ``test/unit_test.sh``, but these
   first): ``test_table_rows.py``, ``test_expansion_value_capacity.py``,
   ``test_no_eviction_row_indexing.py``, ``test_vmm_tensor.py`` (also with
   ``DYNAMICEMB_VMM_STRESS_FULL=1``), ``table_operation/test_lru_lfu.py`` (a growth with both
   score words), ``erase_async/test_storage_erase_by_score.py`` (a growth under an erase),
   ``incremental_dump/*``, ``test_hybrid_storage_export.py``,
   ``test_batched_dynamic_embedding_tables_v2.py``, ``test_alignment.sh`` at ``WORLD_SIZE`` 1, 4
   and 8 (the address space a host table reserves is exercised there, and a failure reads as
   "construction failed").

3. The new tests of the lane: ``growth/test_growth_order.py``, ``growth/test_rehash_padded.py``,
   ``growth/test_host_value_buffer.py``, ``no_host_sync/test_expand_async.py``,
   ``no_host_sync/test_no_host_syncs.py``.

4. E1 (``test_gate_e1_device_extend.py``): the device buffer under a running kernel, the control
   ``GATE_E1_SYNC=before``, the sanitizer on both, and the recorded loop of ``test_vmm_buffer``
   without its synchronize, twenty times and under the sanitizer (``recorded_doubling_loop``).
   A failure of the last one means copy-on-grow, not "no wait".

5. E2 (``test_gate_e2_host_in_place.py``), with ``ulimit -l unlimited``: growth under load, 1000
   growths with the address checked, 20 pieces against one registration, 128 reservations, the
   attribute; the ``nsys`` and the sanitizer runs of its docstring.

6. E3 (``test_gate_e3_stream_order.py``): 5000 steps, 96 under ``compute-sanitizer``, and the
   negative control ``GATE_E3_DISABLE_ORDER=1`` once per configuration, one process each (a
   failed control can leave the context unusable). Then one run of the REAL pipeline with a
   table that grows: the HSTU pipeline test, from ``examples/hstu``
   (``python -m pytest test/test_pipeline.py -k prefetch``; its pipeline is a subclass of
   TorchRec's ``PrefetchTrainPipelineSparseDist``), with ``init_capacity`` set small in the
   ``DynamicEmbTableOptions`` that ``create_model`` of ``examples/hstu/test_utils.py`` builds for
   the ``item`` table, so that the batches of the test grow the storage under the cache (assert
   ``key_map_generation > 0`` on it). The loop of E3 is the order of
   ``PrefetchTrainPipelineSparseDist.progress``, not the pipeline.

7. The census, on the tree with the last commit: ``python no_host_sync/host_sync_probe.py --json
   census.json``, then ``python no_host_sync/sync_budgets.py --from-report census.json`` (what
   would change; a case above its claim names the call that still waits), then ``--write`` (which
   also retires ``_meta.unmeasured`` for the configs it measured). Set
   ``ALLOC_EVENTS_PER_GROWTH`` of ``sync_budgets.py`` from the allocator counts of the growth
   cases of every layout, and run ``test_sync_ratchet.py`` with ``HSP_RATCHET_STRICT=1``.

8. E4 (``test_gate_e4_rehash_speed.py``) on BOTH trees, joined by ``GATE_E4_OUT`` /
   ``GATE_E4_BASELINE``. It gates the padded rehash (commit ``b1c5b448``), not only the last
   commit: that commit must not merge before E4 passes (at most 1.2 times the old wall time up
   to 2**22 slots, at most 1.0 above).

9. The costs the lane adds, on one node, tree before and tree after: the throughput of the HSTU
   benchmark (the PR validated "neutral" with it: every forward, backward and evaluation now takes
   a lock, a dictionary lookup and a stream registration), and the peak memory of a growth of
   the host tier of a hybrid storage (E3's ``hybrid`` line, and the peak tests of
   ``test_rehash_padded.py``).

10. Only then merge the last commit. When it has, delete the ``torch.cuda.synchronize()`` that
    ``test_vmm_buffer`` calls after every ``extend`` (its comment says why it was there), so
    that CI runs the recorded loop without it.

At integration with the other lanes (``git merge`` does not show these):

* the contract lane's ``no_host_sync/test_audited_reads.py`` lists every ``read_now`` of the
  package: ``read_table_sizes_now`` is one more (the commit that made it one call has the entry);
* every other lane still has ``DYNAMICEMB_EXACT_GROWTH`` in code, docs and tests: this lane's
  deletion has to win;
* the probe's ``Cfg.dim`` (the 256-wide growth configs that cross the 2 MiB granule) and the step
  lane's ``Cfg.dims``: ``dims = cfg.dims or (cfg.dim or DIM,) * 2``, and the wide configs must
  still cross the granule;
* ``budgets.json`` is rewritten by every lane: regenerate it once, from a census, at the end.
"""

if __name__ == "__main__":
    print(__doc__)
