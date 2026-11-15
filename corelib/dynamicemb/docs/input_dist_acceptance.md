# Issue #459 acceptance and code review evidence

Review updated: 2026-11-15. Decision: **Needs Fixes / validation before merge; #459 is not accepted as complete.** See the [full RFC](input_dist_indices_design.md) for contracts and alternatives, and [dense-lookup evidence](../benchmark/input_dist_dense_lookup_results.json), [preceding optimization evidence](../benchmark/input_dist_optimization_results.json) and [historical evidence](../benchmark/input_dist_review_results.json) for samples, source digests, log hashes and profiler summaries. This report distinguishes fixed reproducible defects, unverified deployment risks and design tradeoffs.

## Baselines and environment

The original PR was `7ef51667b75d9e498776ee9c4a29426828282d7c` on main `476cdc01`. Review incorporated #528 (`d53b9c8`), then #550/#553 through `8accb875333f82a9446877ec929f53ed42a283b7`, and #555/#478/#554 through `85e628d`. The current integrated baseline is `4466f9d2ea34fd74105f57e1d81291798cd23690` (merge `772999e`). All three #556 patches were inspected and retained: two empty-batch norm flattening fixes and FlexKV's MPS configuration update. The entire DynamicEmb core tree is identical on85e628d and4466f9d (`05747fb85efe182ea78c58a6b5c96e714836f187`), so existing core baseline artifacts remain applicable, with their actual85e628d provenance. This is not a full HSTU/FlexKV deployment test; the unfused CPU norm smoke test is separately scoped below. The README retains native TorchRec training and export documentation; checkpoint reexports, topology tests and #553 NVLink checks remain intact. The #528 transport is unchanged. Native counter/projection sources were rebuilt for both ABIs and linked with matching baseline objects and the prior storage-publication object; source/header/object/extension hashes and compile/link commands are recorded. The optional NVE inference extension was disabled and is not validated. Immutable runtime snapshots pair production Python, the correct extension ABI and matching JIT fatbins. Current code is `4f998af`, production Python SHA-256 `b1f21e4e0a67183c63514eecec2b00c11d125d2426e77e45106f776b49ce8016`.

Hardware: four RTX PRO 6000 Blackwell Server Edition GPUs (SM120), driver 580.126.09, one shared Linux host. `nvidia-smi topo -m` reports PCIe NODE/SYS connections, no NVLink. Runs use GPU subsets and may contend with other jobs; they are not physical multi-node tests.

| Environment | Exact stack / configuration | Limits |
| --- | --- | --- |
| P: compatible older stack | Python 3.12, PyTorch 2.7.0+cu128, TorchRec 1.2.0, FBGEMM 1.2.0+cu128, CUDA 12.8, NCCL 2.28.9 preload; extension without hierarchical ops | Two-rank tests disable P2P/IB and cuMem host allocation and use loopback; tracker API is absent |
| T: target dependency validation | Python 3.12, PyTorch 2.13.0+cu130, TorchRec `v1.8.0-rc1` commit `0ea0ec0` (package 1.8.0+0ea0ec0), official FBGEMM 1.8.0+cu130 wheel pinned by docker/Dockerfile; CUDA 13.0.88 compiler, newer CCCL headers, NCCL 2.30.7 headers/extension/preload | Private clean venv, not the complete NGC 26.07 container; no Docker executable. PyTorch's dependency/compiled NCCL version remains 2.29.7, and both library paths can be mapped. `ncclGetVersion` on the preloaded symbols reports 23007. This must not be described as a certified single-NCCL container |

The target venv includes nvidia-ml-py 13.615.71, so final topology refusal is based on actual NVML links, not a missing dependency. An initial score-script attempt lacked `click`; installing it allowed the actual distributed score checks to run. The first hierarchical-script attempt lacked torch/lib in `LD_LIBRARY_PATH` and failed importing libc10; later scripts include that path. These setup failures are not functional test passes or code defects.

## Confirmed findings, ordered by impact

All source paths below are relative to `corelib/dynamicemb`. The original-code negative controls and final regression results are retained separately. A fixed finding is not an outstanding blocker.

| Severity / disposition | Trigger and code evidence | Actual effect / minimal repair | Required regression |
| --- | --- | --- | --- |
| Blocking correctness, fixed; inherited from main | [`insert_and_evict.cu:70`](../src/table_operation/insert_and_evict.cu#L70) previously used non-atomic counter RMW. Real cache backward releases on the consumer stream while public prefetch prepares another batch on a producer stream | The pre-fix PR left2835 live keys under-pinned and920 over-pinned in trial5; unmodified main85 left2067/2029 in trial0. An under-pinned live row can lose eviction protection. Use `atomicAdd` in the shared plain/fused kernel; retain gates, last-use ordering and rehash handling | C `test_counter_updates_on_different_streams_preserve_each_reference`; L `test_cache_backward_and_prefetch_keep_live_pins_across_streams`. Two negative counter cases fail; both positive cases pass on P/T. Twenty real-cache trials per stack all preserve4096 live pins |
| Blocking correctness, fixed | Original [`PrefetchState`](https://github.com/WEIXILIN0307/recsys-examples/blob/7ef51667b75d9e498776ee9c4a29426828282d7c/corelib/dynamicemb/dynamicemb/batched_dynamicemb_function.py#L193) and [`DynamicEmbeddingFunction`](../dynamicemb/batched_dynamicemb_function.py#L205) had no abandoned pin owner. Resolve an HBM/cache input and drop it, or drop its output graph without backward | Both original layouts retained 32 pins. Cache capacity/erase/eviction cannot reclaim those identities. Add an independently owned idempotent lease with stream-ordered finalization; defer GC releases outside a held table gate | `test_abandoned_pinned_batch_releases_rows`, `test_discard_retained_graph_releases_rows`, `test_cancellation_inside_table_gate_defers_cleanup` |
| Blocking correctness, fixed; inherited from main too | [Original backward](https://github.com/WEIXILIN0307/recsys-examples/blob/7ef51667b75d9e498776ee9c4a29426828282d7c/corelib/dynamicemb/dynamicemb/batched_dynamicemb_function.py#L1394) unconditionally called `release_slot_pins`. Backward the same custom Function twice with `retain_graph=True` on the first call | Original #551 and unmodified d53 main reached 0 pins after the first backward and -32 after the second. Hold the reservation for retained graphs and release once after final use; record retained backward's last GPU use for cancellation | `test_retained_backward_observes_intervening_update`, `test_retained_backward_cancelled_on_another_stream_waits_for_update` |
| Blocking integration, fixed | #528 [`HierDist.finish` / ownership transfer](../dynamicemb/shard/hier_output_dist.py#L814) registers output ownership by KJT identity; resolving creates another object. Its `output_alloc` may supply a mapped shared window | A naive merge loses window ownership or sends wrong row order/gradients. Preserve completion status ordering, rebind ownership, honor allocator and pass row map to gradient reduction. An ordinary in-place window copy also produced a real autograd shared-view RuntimeError across two groups; use the existing externally owned raw-window write contract | `test_hier_receiver_completion_contract`, `test_output_allocator_preserves_values_and_gradient_rows`, `test_shared_output_window_accepts_multiple_lookup_groups`; supported-hardware communication CI still pending |
| Should-fix correctness, fixed | [Original native wrapper](https://github.com/WEIXILIN0307/recsys-examples/blob/7ef51667b75d9e498776ee9c4a29426828282d7c/corelib/dynamicemb/dynamicemb/shard/input_dist.py#L225) appended receiver callbacks without caching; TorchRec `Awaitable.wait()` reruns callbacks | Waiting the same request again re-prepared/pinned rows. Cache the successful resolved result and attach the callback once while preserving native tensors-awaitable metadata | `test_native_receiver_resolution_runs_once`, which fails on original #551 |
| Should-fix compatibility, fixed | [Original table forward](https://github.com/WEIXILIN0307/recsys-examples/blob/7ef51667b75d9e498776ee9c4a29426828282d7c/corelib/dynamicemb/dynamicemb/batched_dynamicemb_tables.py#L1397) rejected a train/eval mode difference between input_dist and compute | Valid pipeline mode transitions raised `Module mode changed after input_dist resolved the batch`. Consume/release training-prepared rows in eval, and prepare fresh state for eval-to-training or recomputation | `test_resolved_batch_train_eval_transition`, `test_recomputed_resolved_batch_owns_separate_state` |
| Should-fix observability/optimizer integration, fixed | [Native grouped lookup observer](../dynamicemb/shard/rw_sharding.py#L154) passed the resolved KJT to registered optimizer/model tracker | Tracker received physical row IDs as keys; original PR fails the new test on TorchRec 1.5. Reconstruct original local key identities only for observers | `test_optimizer_tracker_observes_original_keys`; final positive on target TorchRec 1.8, explicit skip on 1.2 |
| Should-fix failure cleanup, fixed | [Later kernel-group resolution](../dynamicemb/shard/rw_sharding.py#L241) raises after an earlier group took pins | Release states already prepared by that receiver operation. Completed backend mutation is not rolled back | `test_later_group_resolution_failure_releases_earlier_pins` |
| Should-fix recovery, fixed; inherited cache logic | [Cache capacity rejection](../dynamicemb/batched_dynamicemb_function.py#L1051) incremented `_prefetch_outstanding_keys` before checking the limit | A real rejected third reservation left 12288 outstanding instead of 8192; releasing the two successful batches could not return the rejected reservation. Check before incrementing; no rollback API is introduced | L `test_rejected_cache_reservation_can_recover`, failed before the fix |
| Should-fix follow-up compatibility, fixed | New dense adapter initially selected pooled D>2048 in `torchrec_lookup.py` | FBGEMM 1.8 raises; FBGEMM 1.2 aborts the standalone width probe. Check the bound before dispatch and retain existing EBC behavior | `test_dense_lookup_and_fallback_match_reference`, D=2048/2052/4096, both dependency stacks |
| Should-fix follow-up correctness, fixed | Eval's early return in [`BatchedDynamicEmbeddingTablesV2.forward`](../dynamicemb/batched_dynamicemb_tables.py#L1221) omitted the training path's consumer-stream registration. Prepare an explicit eval state on stream A, correctly finish its production, queue lookup on B, then drop state and reuse allocations on A | Real warmed allocator reuse overwrote all 256 output elements on HBM/cache/host/hybrid in both stacks. Register eval metadata before lookup. Native pipeline KJT registration already protected that call path; the defect concerns explicit raw-table state handoffs | L `test_eval_state_survives_drop_on_producer_stream`, including same-stream metadata replacement; no-fix and incomplete-cache negative controls fail |
| Should-fix follow-up CPU lifetime, fixed | `resolve_at_receive` creates a KJT subclass per native request; inherited FX metaclass initialization strongly registers every class | Discarding 1000 requests and forcing GC still retained 1000 classes, about 3.25 MB traced Python memory. A local transport metaclass skips constructor registration while preserving inherited ABC creation; no FX global registry mutation | `test_request_local_receiver_types_are_collectible`; standalone 1000-request negative and positive controls on both dependency stacks |
| Outstanding performance / acceptance issue | Current `4f998af` versus DynamicEmb-identical main85/main4466 imported checkouts below | Final HBM SparseDist medians17.195/17.133 ms versus main16.423/16.361; raw HBM2.441/2.333 versus2.049/1.901 ms. The PR does not meet the user's performance-preservation requirement | Counter/projection fusion and snapshot aliasing remove one kernel per receiver call; stable end-to-end parity remains unproven. All earlier/withdrawn series retain their sources |

The lease/gate cancellation and output-copy defects were found while implementing the integration, and repaired before publication. They are not misrepresented as behaviors of original #551, which did not contain #528. The generic Storage snapshot path already had weak ownership and full-key matching; this review did not invent a new backend API or replace it with an unproven concurrency mechanism.

## Executable acceptance matrix

Run commands from `corelib/dynamicemb` in an installed compatible build. `U` is `test/unit_tests/test_input_dist_indices.py`, `L` is `test/unit_tests/test_resolved_lifecycle.py`, and `C` is `test/unit_tests/table_operation/test_counter_row_projection.py`; a named test is executable as `python -m pytest U::test_name -q` with the actual file path substituted. A two-GPU invocation is `torchrun --standalone --nproc_per_node=2 -m pytest U -q -rs`. Results are suite evidence, not an assertion that every Cartesian product of options was run.

| #459 condition and observable acceptance criterion | Main code location | Executable test | Environment / result / remaining scope |
| --- | --- | --- | --- |
| 1. Resolution has finished before lookup; first compute performs no hash/backend key lookup | `shard/input_dist.py`, `shard/rw_sharding.py`, `batched_dynamicemb_compute_kernel.py` | U `test_input_dist_resolves_before_lookup` | P/T one/two GPUs; all six layouts, dedup flags, int32/int64, stream transition and fused-awaitable wrapper; passed |
| 2. Lookup reuses native TorchRec operations with correct indices | `dense_lookup.py`, `torchrec_lookup.py`, `shard/resolved_input.py` | U `test_pooled_lookup_and_update_match_unsharded_torchrec`; L `test_single_persistent_matrix_uses_physical_rows`; `test_dense_lookup.py` | P/T compact and persistent matrices, SGD/Adam; aligned FP32/FP16 native and BF16/odd/mixed/fractional-stride EC/EBC fallback passed; direct TorchRec dense-kernel comparison passed |
| 3. Evicted physical rows initialize embedding and moments for the new key | `_prefetch_hbm_direct_path`, `_prefetch_cache_path` in `batched_dynamicemb_function.py` | U `test_eviction_reinitializes_reused_value_rows_and_adam_state` | Real GPU eviction with an assertion that row reuse actually occurred; passed; not an artificial slot-number-only test |
| 4. Rehash does not invalidate value-row identity or ref-counter release | `refresh_slots_after_rehash`, `_PinnedRowLease` | U `test_expansion_and_out_of_order_batches`; existing `test_no_grad_release_after_rehash_by_a_later_prefetch` | Passed; full table suite also exercises expansion and non-evicting row capacity |
| 5. All backends preserve current values and optimizer state across overlapping input resolution and arbitrary backward order | `resolved_storage.py`, DEFAULT update/write-through in `batched_dynamicemb_function.py` | U `test_prefetched_batch_reads_updates_from_previous_batch`, `test_in_flight_batches_share_current_optimizer_state` | HBM/cache/host/hybrid/PyDict PS/cache+PS, SGD/Adam, both backward orders and dropped input owners; passed |
| 6. Finite backing-store eviction cannot discard an in-flight update | `_StorageWithActiveRows.find`, `ActiveStorageRows.publish` | U `test_pending_rows_survive_backing_store_eviction` | Host/hybrid/PyDict PS interfaces; passed; remote PS durability/consistency is not certified |
| 7. Full key identity and table identity survive matching | `matching_rows` | U `test_live_row_matching_keeps_table_and_full_key_identity` | Signed/unsigned 64-bit identities and same key in different tables; passed |
| 8. Non-admitted temporary rows produce output but no persistent optimizer write | `_apply_admission`, persisted-row selection | U `test_non_admitted_rows_survive_input_dist`; existing weighted/no-grad suites | Passed; direct matrix adapter intentionally falls back to scratch workspace for rejected rows |
| 9. Dedup frequency and score semantics stay intact | segmented uniqueness, `shard/input_dist.py`, unchanged hier transport | U `test_optimizer_and_score_parity_with_legacy`; `per_key_score/test_per_key_score_dist.py` | Target two-rank score checks passed, including eval, dump/load and pipeline reconstruction; precise scores remain unsupported on the existing Host/Hybrid/PS-only combinations |
| 10. Successful receiver completion is idempotent | `_ResolveOnce`, `_PostprocessedSplitsAwaitable`, `HierKJTAwaitable` | U `test_native_receiver_resolution_runs_once`, `test_hier_receiver_completion_contract` | Native real communication and hierarchical supplied-result receiver test passed; the latter is not a transport test |
| 11. Hierarchical overflow is checked before backend mutation and preserves later valid calls | `HierDist.finish`, `HierKJTAwaitable` | U `test_hier_receiver_completion_contract[True]`; `test_hier_overflow.py` | Receiver status-before-resolution passed with supplied GPU tensors; full supported-hardware overflow/next-call matrix not run on current main |
| 12. Output windows and row maps route forward and gradients, including multiple groups and eval | `place_lookup_output`, `reduce_grads(grad_rows=...)`, `HierDist.rebind_input` | L allocator/shared-window tests; U hierarchical completion test | Real GPU window views/permutation, varied row gradients, all four local layouts and two groups passed; NVLink/GIN transport placement still needs CI |
| 13. Native fallback, score fallback and topology refusal keep row preparation | `shard/rw_sharding.py`, #553 `HierDist.setup` | U `test_output_dist_selection_keeps_receiver_resolution`, `test_hierarchical_refuses_unlinked_topology_before_resolution`; `test_hier_pipeline.py --core` | Actual two-rank PCIe auto fallback and explicit hierarchical refusal passed; one-rank fallback passed; no forced topology override |
| 14. Pipelined split fusion preserves per-request completion and collective ordering | `resolve_at_receive`, RW/TW-RW `_dist`; score/single-rank `_key_dist` | U `test_split_fusion_keeps_each_collections_receiver_state`, `test_native_sparse_dist_pipeline`, `test_native_pooled_pipeline` | T/P real two-collection split collective and native pipelines passed. Unscored multi-rank fusion enabled; separate score-channel completion retained |
| 15. Original output indices and observer keys remain keys | sequence output context, `original_features`, tracker wrapper | U `test_input_dist_resolves_before_lookup`, `test_optimizer_tracker_observes_original_keys` | `need_indices=True` and target tracker passed; old TorchRec tracker is unavailable, not silently tested |
| 16. Multiple kernel groups, mixed static/dynamic groups and feature processing retain their contracts | `_DynamicLookupMixin`, `ResolvedEmbeddingInput.split` | U mixed-group and `test_feature_processing_precedes_resolution_and_runs_once`, `test_feature_processor_preserves_no_grad_contract` | Passed; invalid learned pooling-weight gradient remains an explicit error |
| 17. Cancellation/exception releases resources once without recycling metadata used on another stream | `_PinnedRowLease`, `TableAccessGate`, `PrefetchState.record_stream` | L cancellation/retained/gate tests; U storage cancellation and later-group failure | Actual CUDA delayed-copy/cancellation stress and GC passed; arbitrary concurrent host-thread forwards are not a supported claim |
| 18. Retained backward, recomputation and mode changes do not share a consumed lease | table forward + final-backward release | L retained/mode/recompute tests | HBM/cache/host/hybrid, both backward orders and SUM/NONE modes passed |
| 19. Eval resolves through all tiers without mutation; cold restore preserves RNG | `dynamicemb_resolve_eval`, `eval_lookup`, meta construction | U `test_host_and_evaluation_resolve_rows`, `test_cold_lookup_after_restore_preserves_rng`; full table train/eval | Passed on interface backends; live network PS not exercised |
| 20. Low precision, weighted pooling and mixed dimensions preserve accumulation/update | `torchrec_lookup.py`, unchanged DynamicEmb optimizer kernels | U low-precision pooling tests, `test_half_pooling_keeps_finite_fp32_output`; `test_weighted_pooled_embedding_v2.py` | FP32 accumulation and requested output casts passed; SGD/Adam direct matrix with all three float dtypes passed |
| 21. Raw table and FIFO prefetch APIs remain usable | table `prefetch`, `forward`, grouped `_prefetch_raw` | U `test_raw_grouped_prefetch_keeps_legacy_contract`, `test_legacy_prefetch_owns_rows_until_forward`; L `test_all_hit_sequence_projects_rows_with_unique_pins`; full table suite | Passed on P/T; FIFO owns unconsumed batches until consumed/destroyed. Raw/FIFO defaults keep `occurrence_rows=None`, preserving prior allocation timing |
| 22. TW-RW pooled sharding keeps gradient scaling and empty-group behavior | `shard/twrw_sharding.py` | `test_table_row_wise.py --node-size 1 --mode grad --dist-type hash_roundrobin` | Target two GPUs/two emulated nodes and four GPUs/two emulated nodes: passed (-0.05 and -0.025 updates respectively); physical multi-node execution remains untested |
| 23. Async erase still orders table access and spares pinned rows | original gate/erase plus deferred cleanup scope | `erase_async/test_async_erase.py`, `erase_async/test_erase_end_to_end.py` | Final4f998af T/P:31 passed,1 skipped each; one visible GPU skips device handoff. Earlier actual two-GPU handoffs retain their source boundaries |
| 24. Complete current-main and implementation regression results are separate | Full table suite and source manifests | `python -m pytest test/test_batched_dynamic_embedding_tables_v2.py -q` | See final run ledger below; do not substitute historical 476 results |
| 25. No unacceptable regression in complete training or backend overlap cost | `benchmark_resolved_lookup.py` | Commands below and profiler traces | Not satisfied: measured regression and missing production/NVLink workloads; optimization acceptance remains open |
| 26. Receiver fusion updates U pins while producing N rows; DEFAULT snapshots reuse reverse mapping | Counter/projection kernel, `_prefetch_hbm_direct_path`, `ResolvedEmbeddingInput.from_rows` | C `test_counter_row_projection_matches_separate_operations`, `test_counter_projection_validates_metadata_before_pinning`; L all-hit test; U Host/eval resolution | P/T: single/multiple tables, overflow, missing slots, N=0/N<U/N>U and metadata rejection passed. SGD/Adam x three dtypes verify live updates/out-of-order/retained backward and full moments; identity assertions detect re-projection or hash-slot substitution |
| 27. Concurrent batches cannot lose counter references | Shared atomic counter update and actual cache prefetch/backward | C `test_counter_updates_on_different_streams_preserve_each_reference`; L `test_cache_backward_and_prefetch_keep_live_pins_across_streams` | Final P/T both pass; non-atomic counter negative fails both. Initial queue-saturated stress skipped and is retained as skipped, not passed |

## Receiver projection, atomic pins and main4466: current implementation and evidence

Final code `4f998af` includes fusion `a71a57c`, C++ allocation/API refinement `9292242`, main merge `772999e`, and atomic pin arithmetic. Final runtime `fuse6`/`fuse6-primary` has production Python digest `b1f21e4e0a67183c63514eecec2b00c11d125d2426e77e45106f776b49ce8016`; target/older native extension hashes are `cb2717ba4e262c5901baef13914518d637a76041810eeba22c6972c6ebf998aa` / `3a8cd8ae6f4c4a21abc1b3cb73972c7dd276c5228fe85d20e3bebbefd3ec51c3`. The `fusion_row_projection_followup` evidence records all three native builds per ABI, source/header/object/extension hashes, runtime deltas, test sources, scripts, log hashes and every process sample. Pilot runtimes retain their own digests; final-source counts are not inferred from them.

All-hit HBM input_dist projects physical occurrence rows inside its unique-key pin kernel. N output rows and U unique counter changes are separate launch ranges. Misses preserve pin-before-insert/eviction ordering. DEFAULT/eval aliases the existing reverse map, including scratch rows. Raw/FIFO preparation defaults to no retained occurrence map; cache follows its original preparation path. The old ten-argument counter/None-return API is unchanged; a separate returning C++ entry point allocates the fused output. Metadata errors are checked before pin mutation. C's56 cases cover1/3 tables, overflow, +/- updates, missing slots, N=0/N<U/N>U, malformed metadata and actual stream contention.

The confirmed inherited non-atomic counter defect is reproduced with real cache forward/backward plus public prefetch and valid table options; instrumentation only aligns the actual kernels with CUDA events. Pre-fix PR trial5 has2835 under-pinned and920 over-pinned live keys; main85 trial0 has2067/2029. The first diagnostic omitted option dtypes and failed setup with `KeyError: None`; it is not race evidence. A first2000-launch unit stress saturated host queues and skipped its overlap condition. The final65536-slot/128-launch stress queues both streams before the start event: two non-atomic negatives fail; both plain/fused atomic positives pass in P and T with no skips. Twenty standalone real-cache trials per final stack each retain exactly4096 live pins and reach zero after ordinary FIFO consumption/backward. L also covers that actual cache overlap without substituting results. Atomic arithmetic does not replace eviction, rehash, table-gate or cancellation ordering.

| Actual final validation / command family | Environment / source | Result |
| --- | --- | --- |
| `torchrun --standalone --nproc_per_node=2 -m pytest U L test/unit_tests/test_dense_lookup.py C test/unit_tests/table_operation/test_multitable_ref_counter.py -q` | T / fuse6, GPU1,2 | 836 passed,4 skipped per rank,46.72s |
| Same command on older dependency stack | P / fuse6-primary, GPU0,3 | 833 passed,7 skipped per rank,41.45s |
| `python -m pytest test/test_batched_dynamic_embedding_tables_v2.py -q` | T / fuse6, final source | 391 passed,4 skipped,323 warnings,569.65s; existing NCCL process-group shutdown warning |
| C stream-contention cases, final test form | T / P | 2 passed each,3.83 /3.35s; pre-fix negative2 expected failures |
| Actual cache backward/prefetch standalone,20 trials | T / P | Both passed; every live row has exactly one pin, no under/over-pinned key |
| Async erase and end-to-end erase suites | T / P final runtime | 31 passed,1 skipped each,6.38 /5.99s; one visible GPU skips device handoff |
| #556 unfused norm CPU smoke, current source vs85 control | CPU, outside GPU venv | Current16 passed; old control8 normal passes +8 expected empty-batch reshape failures. TP/Megatron and FlexKV deployment not run |

Exact harness invocations, environment settings and source manifests are recorded in machine-readable evidence. The two-rank command covers the committed final C/L assertions. The first counter-only run's61 passes +2 skips per stack precedes the corrected contention fixture; the final full matrices include those two positives. Earlier fusion matrices and391-case table run belong to their stated pilot source. Re-projection negatives fail6 HBM alias checks; a hash-slot-as-row negative fails6 cases; the DEFAULT re-projection negative fails9 Host/eval checks. None is an outstanding final-source defect. The SGD/Adam x three-dtype all-hit test verifies row-map reuse, one pin per unique identity, raw no-retention behavior, out-of-order/retained backward and complete optimizer-moment parity.

Formal timing uses complete main85, preceding `9f33797` and final `4f998af` installations, resident128-feature/B32/hotness4/D64/FP32/SGD input, capacity4096,20 warmups, six100-step trials and five separate profile steps. Each workload runs main/preceding/current/current/preceding/main. GPU1,2 pipeline ranks use CPUs4-7/36-39 before Torch import; raw uses GPU1. CPU isolation/memory binding are absent. No local GPU regression overlaps timing; external jobs share the host. Main85 is DynamicEmb-identical to current main4466, rather than an invented rebuild of the new base.

| Complete workload / two process medians in ms | main85 / main4466 core | preceding9f33797 | final4f998af |
| --- | --- | --- | --- |
| Raw HBM SGD | 2.049 /1.901 | 2.218 /2.350 | 2.441 /2.333 |
| HBM SparseDist + MLP/SGD | 16.423 /16.361 | 17.196 /17.039 | 17.195 /17.133 |
| HBM Prefetch SparseDist + MLP/SGD | 16.857 /16.980 | 17.547 /18.080 | 17.916 /17.288 |
| Host SparseDist + MLP/SGD | 16.946 /16.944 | 17.431 /17.519 | 17.520 /17.268 |
| Raw Cache SGD, depth0, capacity16384 | 2.468 /2.286 | 2.794 /2.653 | 2.695 /2.872 |
| Raw Cache SGD, depth2, capacity16384 | 2.401 /2.776 | 2.760 /2.917 | 2.887 /2.628 |

All24 final processes,12 cache processes and24 first-fusion pilot processes are retained separately. HBM receiver fusion reduces five-step PR pipeline kernels595 ->590; Host aliasing620 ->615. Main HBM SparseDist is590, main Prefetch595, Host main630. Pipeline syncs remain71; raw HBM150/11 versus main145/11, raw Cache170/16 versus main165/16. Inclusive final CPU attribution reduces HBM `from_rows`114.61/100.88 ->23.77/25.09 us per rank, while the fused counter call rises40.78/41.09 ->75.44/64.28 us. Host `from_rows` falls99.09/100.93 ->23.94/24.12 us. Nested stages must not be summed; those component changes do not explain the whole device-wait/shared-host latency. **End-to-end performance preservation remains unmet.** Earlier series retain their original sources and are not relabeled. NVLink/physical multi-node GIN, real network PS, complete NGC deployment and repository CI remain validation gaps.

## Eval lifetime and borrowed matrix views: preceding implementation and evidence

The eval fix is `b779019`; persistent matrix-wrapper reuse is `b7a950d`. Final code `9f33797` withdraws the unhelpful registration-cache experiment and has production Python SHA-256 `d6ed2ad30f274b5c1569429331928c1f641903308ce57a86b9b7d7e2d028cea1`, byte-identical to the `matrix1` snapshot. All retain integrated main `85e628d` and the matching previously built extensions. The `metadata_reuse_followup` evidence separates production snapshots, test-source boundaries and all timing series. No C++ kernel, backend mutation protocol or optimizer changes in this follow-up.

Evaluation now protects hidden state allocations when an explicit raw-table state crosses streams. A first short-delay diagnostic passed without exercising the race; after warming the exact lookup/allocator and checking that the consumer was still pending, value-buffer reuse corrupted all 256 output elements. Removing only the registration gives four failures on each stack. This is a confirmed defect in the preceding PR implementation; native pipeline KJT registration already protected its call path. The final test also replaces the value Tensor after an earlier stream registration: all eight layout/replacement cases pass with the simple registration loop.

Persistent matrix views are borrowed under weak buffer identity and current logical shape. Real-buffer tests cover in-place updates, actual extension, a same-shape replacement and owner GC, for FP32/FP16/BF16 and row strides 8/24/9. Existing direct HBM/cache SGD/Adam tests also pass. Public buffer wrappers/version semantics remain unchanged. Removing the shape check causes nine failures; owning the buffer strongly causes nine lifecycle failures. Instrumented raw forward calls reduce per-call buffer-wrapper work from 38.98 us to 3.79 us for the cached-view access, and observe zero warmed `buffer.tensor()` calls in that path. This is component evidence, not an end-to-end speedup claim.

A second candidate (`af28238`, `streams1`) used weak Tensor identities and the most recent stream to avoid repeated allocator registrations. It passed functional matrices and reduced five-step SparseDist `record_stream` calls from 120 to 90, Prefetch from 250 to 220. However, inclusive method CPU timing increased from 36.05/36.19 us to 44.24/44.37 us per call; complete-workload timing did not improve reliably. **That cache was removed in `9f33797`.** Its source, tests and all 18 timing processes remain labeled as a withdrawn experiment. Removing its identity checks caused eight control failures, including four actual corrupted eval outputs after same-shape Tensor replacement. These controls do not imply that this cache is in the final code.

| Actual command / scope | T: target stack | P: older stack |
| --- | --- | --- |
| Final `matrix1` production, two-rank `pytest U L test_dense_lookup.py -q`, final test source | 709 passed / 4 skipped per rank, 46.73 s | Preceding same-production full matrix: 702 passed / 7 skipped per rank, 41.87 s; final eight-case eval extension separately passes below |
| Final production full table suite | 391 passed / 4 skipped, 353.30 s; existing process-group shutdown warning | Not rerun; earlier full-suite evidence remains source-bounded |
| Final eight-case eval lifetime/replacement test | Included in final full matrix | 8 passed / 56 deselected, 8.02 s |
| Earlier same-production full matrix before added eval replacement parameter | 705 passed / 4 skipped per rank, 51.67 s | 702 passed / 7 skipped per rank, 41.87 s |
| Eval-only four-layout positive / no-fix negative, preceding test revision | 4 passed / 4 expected failures | 4 passed / 4 expected failures |
| Matrix growth / owner negative controls | 9 / 9 expected failures | Not repeated; positive matrix above |
| Withdrawn registration-cache candidate full matrix | 713 passed / 4 skipped per rank; single-rank 707 / 10 | 710 passed / 7 skipped per rank |
| Withdrawn identity-check negative, candidate test source | 8 expected failures | Not repeated |

Two initial eval commands used relative test paths from the staged working directory and ran no tests (exit 4). Initial matrix-view test drafts had test-scaffolding failures: an instance patch introduced a bound-method cycle requiring GC, then a replacement reference had int64 instead of float32 dtype. Those failed runs are retained; production did not change during the test corrections. The withdrawn candidate's full table run was deliberately interrupted at 336 passed after rejecting its CPU cost; this partial run is not full-table acceptance. Skips, deselections and interruption are not passes.

Complete unchanged installations use GPU1 for raw training and GPUs1,2 for pipelines; all workers bind CPUs before Torch import (rank0 CPUs4-7, rank1 CPUs36-39). CPUs/memory are not isolated. Resident inputs: 128 features, batch32, hotness4, D64/FP32/SGD, capacity4096, 20 warmups, six 100-step trials and five separate profiled calls. Each series uses main/before/current/current/before/main order. No other local GPU validation overlaps timing; external jobs share the host.

| Final production series / two process medians (ms) | main85 | preceding `8efd951` | retained matrix-view implementation |
| --- | --- | --- | --- |
| Raw HBM SGD | 2.422 / 2.197 | 2.784 / 2.758 | 2.801 / 2.764 |
| SparseDist + MLP/SGD | 20.360 / 19.920 | 21.538 / 21.421 | 20.413 / 21.143 |
| Prefetch SparseDist + MLP/SGD | 18.567 / 18.571 | 18.358 / 17.878 | 16.955 / 19.067 |

These samples do not establish stable parity or improvement over main. Five-call raw profiles remove five `aten::view` operations (10 -> 5); GPU kernel/sync counts remain 150/11. SparseDist stays 595/71 (main 590/71); Prefetch stays 595/71. Kernel count, host synchronization and full value-copy traffic are not reduced by a CPU wrapper cache. Profiler allocator/API readings, complete samples and trace hashes remain in the JSON.

| Withdrawn registration-cache series / two process medians (ms) | main85 | retained matrix-view implementation | withdrawn cache |
| --- | --- | --- | --- |
| Raw HBM SGD | 1.880 / 1.760 | 2.320 / 2.335 | 2.328 / 2.326 |
| SparseDist + MLP/SGD | 17.000 / 17.167 | 18.100 / 18.168 | 18.590 / 18.057 |
| Prefetch SparseDist + MLP/SGD | 17.739 / 17.790 | 19.116 / 18.743 | 19.000 / 19.202 |

Retain both series and every unfavorable sample. Their different host conditions preclude attributing cross-series differences to code. No stable end-to-end gain is claimed. **Performance preservation remains unmet; complete #459 acceptance, supported NVLink/physical multi-node GIN and deployment-specific remote PS validation remain open.** The PR remains Draft and no comments or issue closure are performed.

## Preceding bounded receiver-type reuse and evidence

Code `8efd951` (production Python SHA-256 `d6ef7ec8f598fcf5f53bcab948715cc8aa970c6d00367b06763ea1d2a09c9a6c`), based on `e778622`. Remote main remains `85e628d`. Only `shard/input_dist.py` changes in production; raw table, dense lookup, CUDA and optimizer code are unchanged. The `receiver_type_pool_followup` evidence retains exact Python manifest deltas, matching extension hashes, all 16 first-candidate processes, 12 final processes and 12 NUMA-control processes. The first classmethod/global-pool experiments are explicitly not the final implementation and are not silently dropped.

The final distributor-local cache retains at most eight idle `(type, weak-reference cell)` entries. Every live request has an exclusive entry; native/fused awaitables keep their source KJT and resolver alive. A weakref callback clears the cell before returning it, so cached entries own no source, resolver, result or backend rows. The static `dist_init` protocol stays unchanged. There is no in-flight limit: a cache miss creates a fresh type. Deepcopy/pickling resets the cache instead of copying mutable bindings. The negative runtime without that reset returns the wrong owner's value on deepcopy and cannot pickle its local class; the final implementation passes both controls. Those are intentional negative experiments, not outstanding production defects.

| Actual validation | Source / environment | Result and scope |
| --- | --- | --- |
| Full receiver + lifecycle matrix, two ranks | pool2 / T | 395 passed, 4 skipped per rank, 34.54 s |
| Same matrix on older stack | pool2-primary / P | 392 passed, 7 skipped per rank, 29.39 s |
| Same matrix, one rank | pool2 / T | 389 passed, 10 skipped, 24.44 s |
| Final receiver ownership/copy/cache controls | T / P, two ranks | 7 passed per rank / 6 passed + 1 skipped per rank; includes added weak-result assertions |
| TW-RW versus RW gradient, node-size 2 | T / P, two ranks | Both passed; -0.05 step |
| Same gradient test, two emulated nodes | T / P, four ranks | Both passed; -0.025 step; not physical multi-node infrastructure |
| Cache-copy negative without reduction | Independent pool2-copy-negative runtime | 2 expected failures: wrong value for deepcopy and pickle failure |

The full matrices precede additional assertions for completed-result reclamation and distinct active types. The final seven-case contract run repeats the weak-result checks; a final single-case burst run repeats all strengthened assertions on both stacks (1 passed each). Production Python is identical throughout, and the exact test-source hashes before each addition are retained. It checks 128 simultaneously live requests, no retained resolvers/results after drop, at most eight idle types, independent out-of-order completion, repeated waits and copied-cache ownership. The real native-awaitable test drops the caller's source reference before completion and confirms that the awaitable still owns the binding. It also exercises `clear_inputs()` on TorchRec 1.8; TorchRec 1.2 lacks that API and explicitly skips that case. Earlier table-suite/dense-kernel evidence remains bounded to its source; those implementations did not change and were not rerun solely for this controller cache. Black/isort/autoflake/codespell/AST/whitespace checks pass; full repository CI is not inferred.

Lightweight CPU attribution measures 100 warmed pipeline calls, with inclusive method timers. Receiver preparation falls from about 151/158 us per rank to 23.8/24.3 us. The production counter observes two total type constructions per rank, zero during the 100 warmed calls. This is a component result, not end-to-end throughput. Nested stages must not be summed; the raw attribution only covers forward. Remaining lookup, state registration and lease costs are still visible. In both tested TorchRec implementations, the ordinary SparseDist runtime finishes the current tensors awaitable inside `PipelinedForward`, immediately before compute: the input_dist boundary does not automatically move key resolution off the current forward's critical path.

| Complete workload / two process medians (ms), original affinity | main85 | preceding e778622 | final pool2 |
| --- | --- | --- | --- |
| SparseDist + MLP/SGD | 15.773 / 15.605 | 18.494 / 16.610 | 18.272 / 16.608 |
| Prefetch SparseDist + MLP/SGD | 16.014 / 16.652 | 16.971 / 17.020 | 16.733 / 16.901 |

These six-process series use main/preceding/current/current/preceding/main order, resident inputs and the same 128-feature, B32, hotness4, D64/FP32/SGD parameters; profiling is separate. Both ranks originally shared CPUs4-11 on GPUs1,2. GPU1 is NUMA0 and GPU2 NUMA2, making rank1 remote in that configuration. An additional CPU-affinity control places rank0 on CPUs4-7 and rank1 on CPUs36-39 before importing Torch. It neither isolates those CPUs nor sets memory binding, and uses a different process order (current/preceding/main/main/preceding/current):

| Complete workload / two process medians (ms), NUMA CPU-affinity control | main85 | preceding e778622 | final pool2 |
| --- | --- | --- | --- |
| SparseDist + MLP/SGD | 15.857 / 16.140 | 18.815 / 17.022 | 17.667 / 18.203 |
| Prefetch SparseDist + MLP/SGD | 18.020 / 17.849 | 18.684 / 19.198 | 18.700 / 17.825 |

Every result and placement log is retained, including slower samples and the first candidate's adverse timing series. No other local GPU validation overlaps timing; external jobs still share the host. Affinity and process order differ between series, so their cross-series differences cannot be attributed to code. Kernel/sync counts remain 595/71 in these PR pipeline profiles. The cache removes repeated Python type construction, but these samples do not establish stable end-to-end speedup or parity with main. **Performance preservation remains unmet; #459 is not accepted as complete.** Supported NVLink/physical multi-node GIN and deployment-specific remote PS validation still require a target environment; no Draft transition or comments are performed.

## Receiver lifetime and gate follow-up: preceding implementation and evidence

Code `e778622a0cc5dc1f445cbdf210d98c7c8b2385b3`, based on `2bf1a40`; production Python SHA-256 `02d58e7c1643ea3bb1e160afe4e93a36428b566412776287c6f767c41e9fe5d9`. Main remains `85e628d`, rechecked against the remote. The `critical_path_followup` section of the [machine-readable evidence](../benchmark/input_dist_dense_lookup_results.json) retains source/extension manifests, all 24 pilot and 12 final timing processes, per-rank CPU diagnostics, failed control and test log hashes. No C++ or optimizer-format changes are introduced in this follow-up.

The new request-local KJT type had a real CPU retention defect: TorchRec's metaclass initializes each subclass in FX's strong global constructor registry. The weak resolver prevented row ownership from being retained, but did not free the classes. Dropping 1000 requests and forcing GC retained 1000 classes and 3,246,884 traced Python bytes. `_ReceiverInputMeta` now preserves inherited ABC creation and skips FX constructor registration for these transport-only types. The same diagnostic retains zero classes and 91,280 bytes on both stacks; that remaining diagnostic allocation includes the list of 1000 weakref handles. These are Python traced bytes, not process RSS or GPU memory. `test_request_local_receiver_types_are_collectible` covers the lifetime; real two-collection fusion and native pipeline tests continue to cover reconstruction and independent resolver ownership. Dependency upgrades must recheck the metaclass/tracing contract.

The table gate retains its fair host lock, exception cleanup and cross-stream ordering. It reuses an event on the same CUDA device and omits a wait when the preceding section used the same stream. A device change creates a fresh event. Delayed queued handoffs across three streams, failure after a queued write, cross-device handoffs, fairness and all existing cancellation cases were exercised. This is a local reduction of event/stream work, not removal of the gate around HBM rehash or cancellation. In five raw HBM calls the explicit gate `cudaStreamWaitEvent` count falls from 10 to 0; overall host synchronization and kernel counts remain 11 and 150. Pipeline stream-wait counts remain 125 because those handoffs cross streams. A separate 100-call CPU diagnostic reduces Python CUDA Event constructions from 600 to 400; its instrumented elapsed times are not throughput evidence.

| Actual validation | Environment / latest production source | Result |
| --- | --- | --- |
| Full `test/test_batched_dynamic_embedding_tables_v2.py` | T / gate2 | 391 passed, 4 skipped, 359.16 s; existing NCCL process-group shutdown warning |
| Two-rank U + L + `test_dense_lookup.py` | T / gate2 | 686 passed, 4 skipped per rank, 36.62 s |
| Same two-rank matrix | P / gate2-primary | 684 passed, 6 skipped per rank, 31.23 s |
| Async erase + end-to-end erase | T / P | 31 passed, 1 skipped in each environment, 5.66 / 5.39 s; single visible device skips the new device-handoff case |
| Device-handoff case with two visible GPUs | T / P | 1 passed per rank in each environment |
| Discard 1000 transport requests + GC | T / P | Zero retained request types on both; original-source negative fails with 1000 retained |
| Black/isort/autoflake/codespell/AST/whitespace | Changed source/tests and documentation | Passed; repository-wide CI is not inferred |

Latest timing uses complete unchanged main85, preceding `2bf1a40` (dense4) and current code (gate2), with the same resident-input parameters below. Process order is main/preceding/current/current/preceding/main. GPU regression jobs finish before timing starts; external jobs still share the machine. Every sample is retained.

| Two-rank training workload / process medians (ms) | main85 | preceding dense4 | current gate2 |
| --- | --- | --- | --- |
| SparseDist + dense MLP/SGD | 19.030 / 18.734 | 20.303 / 20.265 | 20.390 / 20.293 |
| Prefetch SparseDist + dense MLP/SGD | 19.981 / 19.526 | 20.687 / 20.279 | 20.359 / 20.364 |

Both current pipelines still measure slower than main in this series. Kernel/host-sync counts are 595/71 for current code; reduced event construction and repaired type retention do not establish end-to-end speedup. Pilot gate-only results are separate: raw HBM preceding 2.406/2.419 versus gate-only 2.356/2.446 ms, Host depth-2 preceding 2.933/2.971 versus gate-only 2.877/2.849 ms. Those raw samples precede only the transport-type fix, and are not relabeled as final-source measurements. The pilot SparseDist preceding processes vary from 20.333 to 18.269 ms, while main varies 19.240 to 18.684 ms. Keep that variability rather than discarding it. CPU profiles identify remaining adapter/state work and per-request class creation; no claim is made that these component costs fully explain the observed wall-time difference.

Supported NVLink/physical multi-node GIN validation remains missing. The fork runner API currently reports no self-hosted runners; upstream Blossom authorization is restricted. No build comment, workflow dispatch, review request, Draft transition or issue closure was performed. A usable target validation environment has been requested separately. The performance-preservation gate and complete #459 acceptance remain open.

## Dense forward follow-up: preceding implementation and evidence

Measured adapter code: `361fe13a2432d458135a9ff5aaa9a83b62eb98ca`; production Python SHA-256 `6cf177ccde3e6416c7eb7ab284a2bc606c9db6931bd86f20d5b65822f3f7d44b`. No C++/optimizer-format change. The [dedicated evidence](../benchmark/input_dist_dense_lookup_results.json) contains the matching runtime/extension manifests, current test hashes, failed/negative controls and all 40 timed processes. Existing evidence below remains bounded by its earlier source.

`dynamicemb/dense_lookup.py` now calls the same FBGEMM CUDA forward primitives as TorchRec dense lookup for uniform aligned FP32/FP16 matrices. Physical rows and compact rows share one flat-view contract; integer row-stride scaling excludes optimizer columns and final padding. The selector in `torchrec_lookup.py` retains public EC/EBC for BF16 storage, odd/mixed widths, pooled D above 2048 and fractional strides. Both dependency stacks produced numerically matching NONE/SUM/MEAN outputs and identical kernel names to TorchRec's own dense module; both reject BF16 dense storage. Output dtype, FP32 pooling accumulation and DynamicEmb backward/state ownership are tested separately. Generated operator schemas remain a dependency-version contract to validate on upgrades.

| Actual run / command family | Source / environment | Result and boundary |
| --- | --- | --- |
| Full `test/test_batched_dynamic_embedding_tables_v2.py` | dense3 / T | 391 passed, 4 skipped, 351.23 s; existing process-group shutdown warning |
| Receiver/lifecycle/dense/weighted/no-grad/native-copy focused matrix | dense3 / T | 646 passed, 8 skipped, 30.00 s; before adding six cancellation-sum test cases |
| Corresponding older-stack focused matrix | dense3-primary / P | 645 passed, 9 skipped, 28.41 s; before those six test-only additions |
| Final `torchrun --standalone --nproc_per_node=2 -m pytest` on U, L and `test_dense_lookup.py` | dense3 / T | 577 passed, 4 skipped per rank, 39.27 s |
| Same final two-rank command | dense3-primary / P | 575 passed, 6 skipped per rank, 28.42 s |
| Final `python -m pytest test/unit_tests/test_dense_lookup.py -q` | dense3 / T | 189 passed, 4.74 s |
| Async erase and end-to-end erase | dense3 / T | 29 passed, 5.69 s |
| Direct comparison with TorchRec dense module, standalone diagnostic | dense3 / T and P | NONE/SUM/MEAN numerics and kernel names match; BF16 storage refusal observed |
| Final receiver/lifecycle/expanded dense matrix, same two-rank command | dense4 / T | 685 passed, 4 skipped per rank, 46.73 s |
| Same final expanded matrix on older stack | dense4-primary / P | 683 passed, 6 skipped per rank, 43.43 s |

The initial 189 adapter cases cover storage/output dtypes, aligned/odd/padded row layouts, nonzero storage offset, empty bags, weighted pooling, exact end-of-storage bounds, unchanged optimizer/padding columns and cross-stream metadata lifetime. Six half-input cancellation tests ensure a large intermediate sum accumulates before output conversion. The receiver matrix now checks both aligned native and mixed-D EC/EBC pooling against an independent unsharded TorchRec reference, including SGD/Adam updates. The two-device lifecycle regression invokes final backward with another CUDA device current and checks updates, pin release and restoration of the caller's current device.

Removing metadata `record_stream` in an isolated negative runtime fails the queued-use/deletion regression. Removing the ready-event wait initially passed because first CUDA kernel loading concealed the race; warming the exact operator before delayed initialization makes that negative fail, while the unchanged normal implementation passes. The initial test-only `torch.full` size argument failure was corrected; neither that failure nor the passing unwarmed negative is advertised as production correctness evidence. The source manifest distinguishes those runtime copies from production. Black 23.9.1, isort 5.12.0, autoflake, Python AST, shell syntax and whitespace checks passed for this follow-up; repository-wide CI is not inferred.

A subsequent width probe exposed a missing pooled-kernel eligibility check in the new adapter: D=3072 raised on FBGEMM 1.8 and aborted the isolated FBGEMM 1.2 process. The selector now routes pooled D>2048 to the existing EBC path before dispatch. This is a repaired follow-up defect, not an inherited main defect or a new public API restriction. The final production Python digest is `2c32b97868ca68ba631c5f5cf93d46360f2f75565513355a9b93b55a7feeda80`. The adapter matrix expands to 297 cases with D=2048/2052/4096: it exercises the exact pooled boundary, above-limit fallback and wider native sequence kernels on both stacks. Final two-rank runs above include every expanded case. The full table suite and 40-process timing series precede only this selector guard and extra tests; they were not rerun after it and retain their original source hashes.

### Current performance: components improve, acceptance remains open

Parameters and affinity match the earlier main85 measurements below. Each row uses sequential main/old/dense/dense/old/main process order, six 100-step trials per process, with five separately profiled calls. `old` is complete production `6297e55`; `dense` is complete production `361fe13`. No other local GPU validation overlapped timings. The machine still has external jobs; preserve every sample and process, including the reverse-order SparseDist control.

| Workload / two process medians (ms) | main85 | old public adapter | dense adapter |
| --- | --- | --- | --- |
| HBM raw training | 1.740 / 1.761 | 2.803 / 2.584 | 2.394 / 2.207 |
| Cache raw, depth 2 | 2.193 / 2.231 | 3.001 / 2.667 | 2.601 / 2.613 |
| Host raw, depth 2 | 2.343 / 2.326 | 3.375 / 3.361 | 2.908 / 2.899 |
| Hybrid raw, depth 2 | 2.851 / 2.593 | 3.976 / 3.378 | 3.511 / 3.217 |
| Two-rank SparseDist + MLP/SGD | 16.090 / 15.706 | 17.262 / 17.169 | 18.606 / 18.718 |
| Two-rank Prefetch SparseDist + MLP/SGD | 16.009 / 16.391 | 18.116 / 17.477 | 17.516 / 16.970 |

The additional SparseDist dense/old/old/dense sequence measured 16.895 / 17.205 / 19.482 / 18.017 ms. It neither erases the slower first pair nor establishes a stable end-to-end gain. The raw HBM and Host pairs show lower adapter overhead than the preceding implementation, while all current comparisons still leave overhead relative to main. In a separate 100-call raw cProfile diagnostic, the adapter cumulative CPU time fell from about 58 ms (old) to 18 ms (dense2, before the final release/device-query reductions). That diagnostic attributes wrapper costs; it does not measure uninstrumented end-to-end speed.

Five-call dense profiles retain kernel/sync counts HBM 150/11, Cache 170/16, Host 190/11 and Hybrid 250/36; pipelines are 595/71. Thus the adapter change does not itself reduce those counts. Host/Hybrid still enqueue ten full-row publication kernels across five calls at depth 2. Raw HBM peak allocator increment is 9,936,384 bytes; Host/Hybrid are 10,190,848 bytes, excluding VMM and some NCCL allocations. Copy/API counts, allocation readings and trace hashes are in the JSON. Earlier publication changes removed substantial compact-matching overhead; the new adapter mainly removes CPU KJT/module/parameter work and eligible low-precision conversions. None of these component facts establishes production throughput, NVLink performance or complete #459 acceptance.

## 2026-11-14 preceding optimization validation

Code commit: `6297e5578ba8e6337ce203f306effd2e4ccb4d28`. Final production Python manifest: `d184aeb4e0d7c556136c0c1509a9e5577a46129f93ad03f88ab10979806301b7`; before the final single-group concatenation elision it was `3451b6edb3776226f7851e4b7907cab23f66a6b4a482ab96127151f88959b493`. The initial receiver-fusion integration and storage fast paths were tested before the final native skip-missing copy change; the final runs below identify the extension revision separately. Formatting preserves the Python AST. `opt6`, `opt7` and `opt8` share this Python digest; `opt8` uses the final CUDA operation with a compile-time skip-missing option. Existing row-copy callers keep its default false setting.

| Run / command | Environment / revision | Actual result |
| --- | --- | --- |
| Full unmodified main table suite | T / main85 | 391 passed, 4 skipped, 373.87 s |
| Full PR table suite with final CUDA refinement, before single-group KJT alias change | T / opt8 | 391 passed, 4 skipped, 348.78 s |
| Focused receiver/lifetime/weighted/no-grad plus native storage-row copy | T / opt8 | 427 passed, 6 skipped, 23.68 s |
| Two-rank focused receiver plus native storage-row copy | T / opt8 | 323 passed, 4 skipped per rank, 32.33 s |
| Same focused matrix with final CUDA refinement, before KJT alias change | P / opt8-primary | 426 passed, 7 skipped, 24.28 s |
| Same two-rank matrix before final CUDA refinement | P / opt7-primary | 321 passed, 6 skipped per rank, 26.68 s |
| Async erase and end-to-end erase | T / opt6 | 29 passed, 5.81 s |
| TW-RW gradient / two emulated nodes | T / opt7 two ranks; opt8 four ranks | Passed: both tables moved by -0.05 and -0.025 respectively |
| Checkpoint-file discovery / upstream helper extraction | T / opt8 | 8 passed, 3.97 s |
| Final receiver/lifetime matrix after single-group KJT alias change | T / opt9 | 348 passed, 6 skipped, 24.35 s |
| Final two-rank receiver matrix after KJT alias change | T / opt9; P / opt9-primary | 296/4 and 294/6 per rank, 29.10 s / 26.13 s |

`test_storage_hits_skip_snapshot_matching_and_keep_optimizer_state` reads a real backing store after a retained Adam update, asserts that an all-hit read performs no live matching, and compares complete embedding/moment rows. `test_storage_row_copy_preserves_table_key_admission_and_moments` covers FP32/FP16/BF16, full uint64 identities, table isolation, source/target admission masks, and widths 7/64/1024/3072; the companion test covers empty tables and batches. The cross-stream cancellation test uses a delayed real GPU copy and a callable replacement that does not retain the source through mock call history.

One standalone kernel-test attempt failed allocating its first six-key CUDA tensor (`out of memory`) before calling the new operation on shared GPU 0. A subsequent GPU 1 run passed all 27 cases. An intermediate cancellation-test failure was caused by `MagicMock` retaining the new helper's state arguments; replacing it with a plain delayed callable restored the test's intended ownership condition. These failed runs are retained as distinct evidence; neither was counted as a pass. Both main and PR full suites emit the existing process-group shutdown warning, recorded in their logs.

## Commands and historical run ledger

Use the installed ABI that matches the imported Python package. The local review used isolated runtime copies because the ignored extension in the source checkout belonged to the old pre-#528 ABI. Reproduction in a correctly rebuilt installation does not require that staging helper.

```bash
# Focused one-GPU correctness/lifecycle matrix
python -m pytest test/unit_tests/test_input_dist_indices.py test/unit_tests/test_resolved_lifecycle.py test/unit_tests/test_weighted_pooled_embedding_v2.py test/unit_tests/test_no_grad_slot_lifetime.py -q -rs
# Real two-rank native receiver, storage, score, pipeline and fallback matrix
torchrun --standalone --nproc_per_node=2 -m pytest test/unit_tests/test_input_dist_indices.py -q -rs
torchrun --standalone --nproc_per_node=2 test/unit_tests/per_key_score/test_per_key_score_dist.py
# Full existing tables and async erase, run as separate jobs to avoid fixed test ports
python -m pytest test/test_batched_dynamic_embedding_tables_v2.py -q
python -m pytest test/unit_tests/erase_async/test_async_erase.py test/unit_tests/erase_async/test_erase_end_to_end.py -q
# TW-RW: the test sets node size internally; torchrun otherwise overwrites LOCAL_WORLD_SIZE
torchrun --standalone --nproc_per_node=2 test/unit_tests/test_table_row_wise.py --node-size 1 --mode grad --dist-type hash_roundrobin
# Supported NVLink hardware, and separately physical multi-node infrastructure, still required
bash test/unit_tests/test_hier_dist.sh
bash test/unit_tests/test_hier_overflow.sh
bash test/unit_tests/test_hier_pipeline.sh --core
bash test/unit_test.sh hier_fwd_bwd
bash test/unit_test.sh hier_load_dump
```

| Final / bounded run | Source snapshot | Actual result |
| --- | --- | --- |
| Unmodified main 8acc, target full table suite | `main-target-8acc` | 391 passed, 4 skipped |
| PR, isolated target full table suite | `feature-target-r5` | 391 passed, 4 skipped; before final capacity-rejection-only change |
| Final target focused + lifecycle + weighted/no-grad | `feature-target-r6` | 397 passed, 5 skipped |
| Final older-stack focused + lifecycle + weighted/no-grad | `feature-primary-r6` | 396 passed, 6 skipped |
| Target two-rank focused | `feature-target-r4` | 292 passed, 4 skipped per rank |
| Older-stack two-rank focused | `feature-primary-r4` | 290 passed, 6 skipped per rank |
| Earlier target tables + lifecycle + async erase | `feature-target-final` | 465 passed, 4 skipped; bounded by its source manifest |
| Target per-key score script | `feature-target-r4` | Passed, exit 0 |
| Target TW-RW gradient, two then four GPUs | `r4`, `r5` | Both passed, exit 0; emulated nodes |
| Target one-rank fallback; older-stack two-rank no-hierarchical fallback/pipeline | `r4`, `r5` | Both ALL PASSED, exit 0 |

The initial parallel PR full-table attempt had 43 checkpoint failures (`EADDRINUSE` on fixed port 29500), 348 passes and four skips. A separate diagnostic reproduced that port error. After the main job exited, an isolated full rerun passed 391/4 without changing the tests' port. These are separate recorded runs, not a synthesized green result. Final r6 differs from r5 by the three-line cache reservation precheck/recovery fix, exercised by the new negative/positive regression and both final focused matrices; the complete table suite was not rerun after that final conditional fix. R4 performance samples precede only unused-import cleanup and this rejection fix and are labeled by their actual source.

Final run ledger is generated from the retained logs in the machine-readable evidence. Current results include target focused 397 passed/5 skipped; target two-rank 292 passed/4 skipped per rank; older-stack focused 396 passed/6 skipped; older-stack two-rank 290 passed/6 skipped per rank. The four common skips are existing precise-score/storage restrictions. The extra older-stack skips are tracker unavailability and no hierarchical-enabled multi-rank topology test. Single-rank target skips the multi-rank refusal test. A target one-rank hierarchical pipeline fallback, older-stack two-rank no-hierarchical-build fallback/pipeline, and the target two-rank score script passed. Final one-GPU suite counts include the cache-reservation recovery regression.

The pre-#553 d53 hierarchical experiment ran the real extension on two PCIe GPUs and, after correcting the observer to reconstruct keys, matched forward, per-key gradients, table values, LFU/admission and eval in its cases. The old transport-only synchronization assertions were split from newly moved row-resolution work, with resolution syncs reported separately. A run excluding the isolated standalone timeout-teardown printed ALL PASSED, explicitly with that one stage skipped. The timeout-teardown hung on unmodified d53 main too. Latest main now refuses this PCIe topology. These observations are diagnostic evidence, not a supported-hardware hierarchical pass; current-main NVLink/GIN overflow, pipeline overlap, window reuse and teardown remain validation gates.

## Preceding main85 profiling and performance

All figures below come from complete imported main85, merged-but-unoptimized `d1151e6` (before85), and optimized `6297e55` (opt9). The repeated-key inputs fit the warmed working set: 128 features, per-rank batch 32, hotness 4, D=64, FP32/SGD, 20 warmups and six trials of 100 steps per process. Five extra steps are profiled separately. Raw cases use GPU 1 / CPU affinity 4-7; two-rank pipelines use GPUs 1,2 / affinity 4-11. Cache depth 2 uses capacity 16384 (4096 cache rows), allowing three outstanding 1024-key batches under the existing reservation contract; other cases use capacity 4096. This is a small resident-input benchmark, not a production training claim.

| Workload / repeated process medians (ms) | main85 | before85 | optimized | Five-step kernels / host syncs, before -> optimized |
| --- | --- | --- | --- | --- |
| HBM raw, ABBA repeat | 1.775 / 1.774 | 2.729 initial series | 2.552 / 2.551 | 155/11 -> 150/11 |
| Cache raw, depth 2, ABBA repeat | 2.195 / 2.204 | 7.086 initial, unstable series | 2.943 / 3.011 | 175/16 -> 170/16 |
| Host raw, depth 2, main/before/after/after/before/main | 4.737 / 2.310 | 8.770 / 9.507 | 3.352 / 3.487 | 925/121 -> 190/11 |
| Hybrid raw, depth 2, same order | 2.855 / 3.582 | 10.056 / 9.281 | 3.982 / 4.146 | 985/146 -> 250/36 |
| Two-rank SparseDist + dense MLP/SGD | 15.640 | 17.459 | 17.626 | 600/56 -> 595/71 |
| Two-rank Prefetch SparseDist + dense MLP/SGD | 15.993 | 17.370 | 17.582 | 600/56 -> 595/71 |

Storage publication substantially reduces the previous overlap cost and removes its additional count synchronizations. It does not establish parity with main. The stable HBM repeats remain about 44% slower in this small raw case; SparseDist and Prefetch medians remain about 12.7% and 9.9% slower respectively. The public EC/EBC adapter and state/lifetime work remain costs to optimize. Restoring fusion recovers the native protocol, but this single-collection pipeline benchmark does not demonstrate a throughput gain from it. Multi-collection correctness was tested; multi-collection throughput was not measured.

Keep the outliers: the initial HBM optimized process measured 4.766 ms, with five-step CUDA API CPU time 7723.8 us in memcpy and 5398.5 us in stream synchronization, versus 245.6/66.7 us in the initial main process. Later ABBA medians were 2.552/2.551 ms for the same optimized source. Host main process medians also vary materially. These observations justify separating scheduling/wait variability from fixed adapter overhead; they do not identify the external cause or warrant dropping samples. The first r1/r2 repeat series overlapped a local GPU regression and is excluded from performance conclusions. R3/r4 and storage R5/R6 ran without another local validation job; external jobs still share the machine. All samples, commands, trace hashes, API counts, selected operators, allocator readings and native publication-kernel counts are in `input_dist_optimization_results.json`.

```bash
python benchmark/benchmark_resolved_lookup.py --lookup default --layout host --train --prefetch-depth 2 --features 128 --capacity 4096 --warmup 20 --iterations 100 --trials 6 --profile-steps 5 --profile-dir /tmp/resolved-host
# Execute this separately in each complete main/before/optimized installation.
torchrun --standalone --nproc_per_node=2 benchmark/benchmark_resolved_lookup.py --lookup default --pipeline sparse --layout hbm --features 128 --capacity 4096 --warmup 20 --iterations 100 --trials 6 --profile-steps 5 --profile-dir /tmp/resolved-sparse
```

## Historical profiling and performance

The table below is historical main-8acc evidence. It predates the latest-main merge, restored split fusion and fused storage publication, and must not be interpreted as a measurement of the optimized revision. Current-baseline measurements are recorded separately in the JSON; no throughput improvement over main has been established.

Run the same command in complete main and PR installations, with identical GPU/dependency settings. `--lookup default` measures each installation's actual data flow; `--lookup both` is only an adapter A/B and must not be labeled a complete-main baseline.

```bash
# Raw training, one table / 128 features, resident working set
python benchmark/benchmark_resolved_lookup.py --lookup default --train --layout hbm --features 128 --warmup 20 --iterations 100 --trials 8 --profile-steps 5 --profile-dir /tmp/dynamicemb-profile
# ActiveStorageRows: compare zero and two outstanding batches, host then hybrid
python benchmark/benchmark_resolved_lookup.py --lookup default --train --layout host --prefetch-depth 2 --features 128 --warmup 20 --iterations 100 --trials 8 --profile-steps 5 --profile-dir /tmp/dynamicemb-host-profile
# Complete SparseDist training, dense MLP + sparse optimizer included
torchrun --standalone --nproc_per_node=2 benchmark/benchmark_resolved_lookup.py --lookup default --pipeline sparse --features 128 --warmup 20 --iterations 100 --trials 8 --profile-steps 5 --profile-dir /tmp/dynamicemb-pipeline-profile
```

All rows below use T, D=64, per-rank batch=32, hotness=4, capacity=4096, a 1024-key resident working set, twenty warmup steps and eight trials of one hundred timed steps. Profile counts cover five extra calls outside timing, with one terminal profiler synchronization. These shared-host synthetic numbers are evidence of overhead, not production estimates.

| Complete workload | Main 8acc median ms | PR median ms | Main -> PR kernels in five calls | Main -> PR host-sync calls |
| --- | ---: | ---: | ---: | ---: |
| HBM raw training | 1.696 | 2.878 | 145 -> 155 | 11 -> 11 |
| Cache raw training | 2.156 | 3.143 | 165 -> 175 | 16 -> 16 |
| Host raw training, depth 0 | 2.002 | 2.745 | 185 -> 175 | 11 -> 11 |
| Hybrid raw training, depth 0 | 2.697 | 3.447 | 245 -> 235 | 36 -> 36 |
| Host raw training, depth 2 | 2.058 | 8.782 | 185 -> 925 | 11 -> 121 |
| Hybrid raw training, depth 2 | 2.573 | 10.217 | 245 -> 985 | 36 -> 146 |
| Two-rank SparseDist + dense MLP/SGD | 15.537 | 17.871 | 590 -> 600 | 71 -> 56 |

The before/after local optimization profile is separately retained: pre-optimization Host depth-0 had 245 kernels/31 sync calls in five steps; the empty-live-registry fast path reduced this to 195/11, and removing identity persisted-row gathers/duplicate ranges reduced it further to 175/11. Because adjacent trials/hardware load differ, the corresponding wall-time change is not presented as an isolated causal speedup. The persistent single-matrix adapter removes the compact HBM embedding load; raw callers still compute occurrence rows, so its raw kernel count need not decrease. A multi-table/rejected-row workspace and hierarchical output copy still exist.

For overlap, the added dynamic nonzero/matching/sorting/searchsorted and publishing copies are visible in the CPU/CUDA trace. Their increased kernels/synchronizations explain why fewer synchronization calls in the complete SparseDist row do not imply higher throughput. A separate warmed Host memory profile records resident allocated bytes 4,498,944 at depth 0 versus 5,580,288 at depth 2; peak allocated bytes are 14,689,280 versus 15,770,624. The depth-2 resident increase is 1,081,344 bytes. These are real PyTorch allocator readings, not total device memory. Peak allocator increments in these warmed profiles exclude pre-existing live snapshots, VMM reservations and some NCCL memory; resident/peak readings are recorded separately where collected. No network PS performance, physical multi-node GIN performance or production HSTU throughput was measured.

## Remaining gates and non-blocking design choices

1. **Merge/complete-issue validation gate:** run current-main native/hierarchical receiver, overflow recovery, output-window and pipeline suites on supported NVLink GPUs and physical multi-node GIN. Do not bypass #553 to obtain a green local test.
2. **Performance gate:** reduce native adapter/split-fusion overhead and bound active-snapshot matching cost under representative overlap, or have the maintainer explicitly accept the cost. The measured regressions violate the requested performance-preservation criterion.
3. **Deployment risk, not a proved defect:** PyDict tests validate the Storage/PS interface and finite-store eviction behavior, not remote service errors, durability, latency or third-party asynchronous buffer ownership. Run the deployment's PS integration suite before making that deployment claim. The registry is local and supplies no invalidation/coherence mechanism for independent external writers; no such multi-writer PS deployment was tested.
4. **Design choices requiring review, not speculative bugs:** compact adaptation for general backends; an extra mapped output-window copy; the remaining score-channel fusion opt-out; serialized backend event chain; private PyTorch retained-graph query; no new host-thread-reentrant model API. Their invariants, costs and alternatives are explicit in the RFC.

No outstanding functional defect was inferred merely from a missing experiment, and no full-repository CI pass is claimed. Historical or failed setup runs are kept distinguishable from successful final checks. This work updates existing RFC/PR bodies and commits only; it does not close any issue, change Draft state or publish GitHub comments.

## Static checks

Changed Python files pass Black and isort with the repository profile. Root-config autoflake retains two unused assignments in the existing full-table test, also present on main; the changed production files have no new autoflake findings. Historical root-command mypy against 8acc reported 199 errors in 53 files on both main 8acc and this branch, with no added normalized diagnostic messages. The root mypy configuration excludes corelib, so this is not a claim of type-checking DynamicEmb. Whitespace and new-document spelling checks pass. Full repository CI is not claimed.

Current-change checks: Black 23.9.1, isort 5.12.0 with the repository profile, autoflake, Python AST parsing, shell syntax and whitespace checks pass. The historical mypy comparison was not rerun against main85; corelib remains excluded by root configuration.
