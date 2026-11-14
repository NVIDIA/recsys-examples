# Issue #459 acceptance and code review evidence

Review updated: 2026-11-14. Decision: **Needs Fixes / validation before merge; #459 is not accepted as complete.** See the [full RFC](input_dist_indices_design.md) for contracts and alternatives, and [dense-lookup evidence](../benchmark/input_dist_dense_lookup_results.json), [preceding optimization evidence](../benchmark/input_dist_optimization_results.json) and [historical evidence](../benchmark/input_dist_review_results.json) for samples, source digests, log hashes and profiler summaries. This report distinguishes fixed reproducible defects, unverified deployment risks and design tradeoffs.

## Baselines and environment

The original PR was `7ef51667b75d9e498776ee9c4a29426828282d7c` on main `476cdc01`. Review incorporated #528 (`d53b9c8`), then #550/#553 through `8accb875333f82a9446877ec929f53ed42a283b7`. The current integrated baseline is `85e628db7a6ac4d5f5f070d1e2d15d51eead0534` (merge `d1151e6`), including #555, #478 and #554. The README conflict retains both the TorchRec training description and upstream export documentation. Checkpoint-helper extraction/reexports and the corrected topology test are retained. In particular, the #553 NVLink checks are preserved. The transport C++ sources did not change between d53 and 8acc. New 85e628d training extensions were rebuilt for both ABIs; modified storage-publication code was compiled with those flags and linked with the matching baseline objects/fatbins. The optional NVE inference extension was disabled and is not validated. Incremental-build commands and modified-source hashes are recorded separately from Python manifests. Runtime snapshots pair production Python, the correct extension ABI and matching JIT fatbins; their SHA-256 manifests identify what tests imported.

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
| Blocking correctness, fixed | Original [`PrefetchState`](https://github.com/WEIXILIN0307/recsys-examples/blob/7ef51667b75d9e498776ee9c4a29426828282d7c/corelib/dynamicemb/dynamicemb/batched_dynamicemb_function.py#L193) and [`DynamicEmbeddingFunction`](../dynamicemb/batched_dynamicemb_function.py#L205) had no abandoned pin owner. Resolve an HBM/cache input and drop it, or drop its output graph without backward | Both original layouts retained 32 pins. Cache capacity/erase/eviction cannot reclaim those identities. Add an independently owned idempotent lease with stream-ordered finalization; defer GC releases outside a held table gate | `test_abandoned_pinned_batch_releases_rows`, `test_discard_retained_graph_releases_rows`, `test_cancellation_inside_table_gate_defers_cleanup` |
| Blocking correctness, fixed; inherited from main too | [Original backward](https://github.com/WEIXILIN0307/recsys-examples/blob/7ef51667b75d9e498776ee9c4a29426828282d7c/corelib/dynamicemb/dynamicemb/batched_dynamicemb_function.py#L1394) unconditionally called `release_slot_pins`. Backward the same custom Function twice with `retain_graph=True` on the first call | Original #551 and unmodified d53 main reached 0 pins after the first backward and -32 after the second. Hold the reservation for retained graphs and release once after final use; record retained backward's last GPU use for cancellation | `test_retained_backward_observes_intervening_update`, `test_retained_backward_cancelled_on_another_stream_waits_for_update` |
| Blocking integration, fixed | #528 [`HierDist.finish` / ownership transfer](../dynamicemb/shard/hier_output_dist.py#L814) registers output ownership by KJT identity; resolving creates another object. Its `output_alloc` may supply a mapped shared window | A naive merge loses window ownership or sends wrong row order/gradients. Preserve completion status ordering, rebind ownership, honor allocator and pass row map to gradient reduction. An ordinary in-place window copy also produced a real autograd shared-view RuntimeError across two groups; use the existing externally owned raw-window write contract | `test_hier_receiver_completion_contract`, `test_output_allocator_preserves_values_and_gradient_rows`, `test_shared_output_window_accepts_multiple_lookup_groups`; supported-hardware communication CI still pending |
| Should-fix correctness, fixed | [Original native wrapper](https://github.com/WEIXILIN0307/recsys-examples/blob/7ef51667b75d9e498776ee9c4a29426828282d7c/corelib/dynamicemb/dynamicemb/shard/input_dist.py#L225) appended receiver callbacks without caching; TorchRec `Awaitable.wait()` reruns callbacks | Waiting the same request again re-prepared/pinned rows. Cache the successful resolved result and attach the callback once while preserving native tensors-awaitable metadata | `test_native_receiver_resolution_runs_once`, which fails on original #551 |
| Should-fix compatibility, fixed | [Original table forward](https://github.com/WEIXILIN0307/recsys-examples/blob/7ef51667b75d9e498776ee9c4a29426828282d7c/corelib/dynamicemb/dynamicemb/batched_dynamicemb_tables.py#L1397) rejected a train/eval mode difference between input_dist and compute | Valid pipeline mode transitions raised `Module mode changed after input_dist resolved the batch`. Consume/release training-prepared rows in eval, and prepare fresh state for eval-to-training or recomputation | `test_resolved_batch_train_eval_transition`, `test_recomputed_resolved_batch_owns_separate_state` |
| Should-fix observability/optimizer integration, fixed | [Native grouped lookup observer](../dynamicemb/shard/rw_sharding.py#L154) passed the resolved KJT to registered optimizer/model tracker | Tracker received physical row IDs as keys; original PR fails the new test on TorchRec 1.5. Reconstruct original local key identities only for observers | `test_optimizer_tracker_observes_original_keys`; final positive on target TorchRec 1.8, explicit skip on 1.2 |
| Should-fix failure cleanup, fixed | [Later kernel-group resolution](../dynamicemb/shard/rw_sharding.py#L241) raises after an earlier group took pins | Release states already prepared by that receiver operation. Completed backend mutation is not rolled back | `test_later_group_resolution_failure_releases_earlier_pins` |
| Should-fix recovery, fixed; inherited cache logic | [Cache capacity rejection](../dynamicemb/batched_dynamicemb_function.py#L1051) incremented `_prefetch_outstanding_keys` before checking the limit | A real rejected third reservation left 12288 outstanding instead of 8192; releasing the two successful batches could not return the rejected reservation. Check before incrementing; no rollback API is introduced | L `test_rejected_cache_reservation_can_recover`, failed before the fix |
| Outstanding performance / acceptance issue | Historical main8acc imported-checkout A/B, followed by the main85 measurements below | SparseDist training is 17.871 ms versus main 15.537 ms (+15.0%); opaque-storage overlap is substantially more expensive. The PR does not yet meet the user's requirement to preserve performance | Publication/fusion and the dense adapter reduce specific costs; current main85 comparisons below still do not establish performance preservation |

The lease/gate cancellation and output-copy defects were found while implementing the integration, and repaired before publication. They are not misrepresented as behaviors of original #551, which did not contain #528. The generic Storage snapshot path already had weak ownership and full-key matching; this review did not invent a new backend API or replace it with an unproven concurrency mechanism.

## Executable acceptance matrix

Run commands from `corelib/dynamicemb` in an installed compatible build. `U` below is `test/unit_tests/test_input_dist_indices.py`, `L` is `test/unit_tests/test_resolved_lifecycle.py`; a named test is executable as `python -m pytest U::test_name -q` with the actual file path substituted. A two-GPU invocation is `torchrun --standalone --nproc_per_node=2 -m pytest U -q -rs`. Results are suite evidence, not an assertion that every Cartesian product of options was run.

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
| 21. Raw table and FIFO prefetch APIs remain usable | table `prefetch`, `forward`, grouped `_prefetch_raw` | U `test_raw_grouped_prefetch_keeps_legacy_contract`, `test_legacy_prefetch_owns_rows_until_forward`; full table suite | Passed on compatible/target suites; FIFO intentionally owns unconsumed batches until consumed or destroyed |
| 22. TW-RW pooled sharding keeps gradient scaling and empty-group behavior | `shard/twrw_sharding.py` | `test_table_row_wise.py --node-size 1 --mode grad --dist-type hash_roundrobin` | Target two GPUs/two emulated nodes and four GPUs/two emulated nodes: passed (-0.05 and -0.025 updates respectively); physical multi-node execution remains untested |
| 23. Async erase still orders table access and spares pinned rows | original gate/erase plus deferred cleanup scope | `erase_async/test_async_erase.py`, `erase_async/test_erase_end_to_end.py` | Combined target table/lifecycle/erase run: 465 passed, 4 skipped before final DEFAULT fast-path/import-only edits; targeted final lifetime/native suites passed afterward |
| 24. Complete current-main and implementation regression results are separate | Full table suite and source manifests | `python -m pytest test/test_batched_dynamic_embedding_tables_v2.py -q` | See final run ledger below; do not substitute historical 476 results |
| 25. No unacceptable regression in complete training or backend overlap cost | `benchmark_resolved_lookup.py` | Commands below and profiler traces | Not satisfied: measured regression and missing production/NVLink workloads; optimization acceptance remains open |

## Dense forward follow-up: current implementation and evidence

Code: `361fe13a2432d458135a9ff5aaa9a83b62eb98ca`; production Python SHA-256 `6cf177ccde3e6416c7eb7ab284a2bc606c9db6931bd86f20d5b65822f3f7d44b`. No C++/optimizer-format change. The [dedicated evidence](../benchmark/input_dist_dense_lookup_results.json) contains the matching runtime/extension manifests, current test hashes, failed/negative controls and all 40 timed processes. Existing evidence below remains bounded by its earlier source.

`dynamicemb/dense_lookup.py` now calls the same FBGEMM CUDA forward primitives as TorchRec dense lookup for uniform aligned FP32/FP16 matrices. Physical rows and compact rows share one flat-view contract; integer row-stride scaling excludes optimizer columns and final padding. The selector in `torchrec_lookup.py` retains public EC/EBC for BF16 storage, odd/mixed widths and fractional strides. Both dependency stacks produced numerically matching NONE/SUM/MEAN outputs and identical kernel names to TorchRec's own dense module; both reject BF16 dense storage. Output dtype, FP32 pooling accumulation and DynamicEmb backward/state ownership are tested separately. Generated operator schemas remain a dependency-version contract to validate on upgrades.

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

The 189 adapter cases cover storage/output dtypes, aligned/odd/padded row layouts, nonzero storage offset, empty bags, weighted pooling, exact end-of-storage bounds, unchanged optimizer/padding columns and cross-stream metadata lifetime. Six half-input cancellation tests ensure a large intermediate sum accumulates before output conversion. The receiver matrix now checks both aligned native and mixed-D EC/EBC pooling against an independent unsharded TorchRec reference, including SGD/Adam updates. The two-device lifecycle regression invokes final backward with another CUDA device current and checks updates, pin release and restoration of the caller's current device.

Removing metadata `record_stream` in an isolated negative runtime fails the queued-use/deletion regression. Removing the ready-event wait initially passed because first CUDA kernel loading concealed the race; warming the exact operator before delayed initialization makes that negative fail, while the unchanged normal implementation passes. The initial test-only `torch.full` size argument failure was corrected; neither that failure nor the passing unwarmed negative is advertised as production correctness evidence. The source manifest distinguishes those runtime copies from production. Black 23.9.1, isort 5.12.0, autoflake, Python AST, shell syntax and whitespace checks passed for this follow-up; repository-wide CI is not inferred.

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
4. **Design choices requiring review, not speculative bugs:** compact adaptation for general backends; an extra mapped output-window copy; the remaining score-channel score-channel fusion opt-out; serialized backend event chain; private PyTorch retained-graph query; no new host-thread-reentrant model API. Their invariants, costs and alternatives are explicit in the RFC.

No outstanding functional defect was inferred merely from a missing experiment, and no full-repository CI pass is claimed. Historical or failed setup runs are kept distinguishable from successful final checks. This work updates existing RFC/PR bodies and commits only; it does not close any issue, change Draft state or publish GitHub comments.

## Static checks

Changed Python files pass Black and isort with the repository profile. Root-config autoflake retains two unused assignments in the existing full-table test, also present on main; the changed production files have no new autoflake findings. Historical root-command mypy against 8acc reported 199 errors in 53 files on both main 8acc and this branch, with no added normalized diagnostic messages. The root mypy configuration excludes corelib, so this is not a claim of type-checking DynamicEmb. Whitespace and new-document spelling checks pass. Full repository CI is not claimed.

Current-change checks: Black 23.9.1, isort 5.12.0 with the repository profile, autoflake, Python AST parsing, shell syntax and whitespace checks pass. The historical mypy comparison was not rerun against main85; corelib remains excluded by root configuration.
