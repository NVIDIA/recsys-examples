set -e
FWD_BWD_TEST_FILES=(
    "test/unit_tests/table_operation/test_table_operation.sh"
    "test/unit_tests/erase_async/test_erase_async.sh"
    "test/unit_tests/retain_evicted_keys/test_retain_evicted_keys.sh"
    "test/unit_tests/test_hybrid_storage_export.sh"
    "test/unit_tests/test_hybrid_masked.py"
    "test/unit_tests/test_eval_masked.py"
    "test/unit_tests/test_lfu_scores.sh"
    "test/test_batched_dynamic_embedding_tables_v2.py"
    "test/test_unique_op.py"
    "test/unit_tests/test_sequence_embedding.sh"
    "test/unit_tests/test_pooled_embedding.sh"
    "test/unit_tests/test_weighted_pooled_embedding_v2.py"
    "test/unit_tests/test_twin_module.sh"
    "test/unit_tests/test_alignment.sh"
    "test/unit_tests/test_hash_roundrobin_kuairand.py"
    "test/unit_tests/test_vmm_tensor.py"
    "test/unit_tests/optimizer/test_padded_buffer_optimizer.py"
    "test/unit_tests/optimizer/test_ftrl_optimizer.py"
    # Added by #495 and never registered, so its CUDA checks on the packed
    # fp32 Adagrad accumulator -- odd dims, checkpoint conversion, state
    # precision -- have not run in CI since.
    "test/unit_tests/optimizer/test_rowwise_adagrad_state_precision.py"
    "test/unit_tests/test_no_eviction_row_indexing.py"
    "test/unit_tests/test_per_table_hbm_budget.py"
    "test/unit_tests/test_expansion_value_capacity.py"
    # A growth queues its kernels on one stream and returns; the order it keeps against
    # the streams that use the table (the logic needs no GPU).
    "test/unit_tests/growth/test_growth_order_logic.py"
    "test/unit_tests/growth/test_growth_order.py"
    "test/unit_tests/no_host_sync/test_expand_async.py"
    "test/unit_tests/no_host_sync/test_outstanding_keys.py"
    "test/unit_tests/test_check_scores_late.py"
    "test/unit_tests/test_table_rows.py"
    "test/unit_tests/test_write_back_masked.py"
    "test/unit_tests/test_cache_metrics_device.py"
    "test/unit_tests/test_find_load_primitives.py"
    "test/unit_tests/test_int64_weights_channel.py"
    "test/unit_tests/test_frequency_requirements.py"
    "test/unit_tests/test_frequency_counter_guard.py"
    "test/unit_tests/per_key_score/test_per_key_score.sh"
    "test/unit_tests/test_planner_plan_cost.py"
    "test/unit_tests/test_planner_placement.py"
    "test/unit_tests/test_table_row_wise.sh"
    # The padded tail of the unique op's outputs and of the evicted arrays is
    # marked (EmptyKey, table id -1, index -1) and the table, initializer and
    # optimizer kernels skip it.
    "test/unit_tests/sync_free/test_padded_tail.py"
    "test/unit_tests/sync_free/test_initializer_negative_index.py"
    "test/unit_tests/sync_free/test_evicted_prefill.py"
    "test/unit_tests/sync_free/test_padded_buffer_optimizer_tail.py"
    # Host syncs on the training thread: the static lint and the probe's own logic
    # need no GPU; the helpers and the padded-array invariants need one; the ratchet
    # and the acceptance test take a census with the CUDA profiler and compare it
    # with budgets.json, which a change that removes syncs lowers.
    "test/unit_tests/no_host_sync/test_host_sync_probe_unit.py"
    "test/unit_tests/no_host_sync/test_sync_free_lint.py"
    "test/unit_tests/no_host_sync/test_sync_free_helpers.py"
    "test/unit_tests/no_host_sync/test_tail_invariants.py"
    "test/unit_tests/no_host_sync/test_sync_ratchet.py"
    "test/unit_tests/no_host_sync/test_no_host_syncs.py"
)

LOAD_DUMP_TEST_FILES=(
    "test/unit_tests/test_checkpoint_file_discovery.py"
    "test/unit_tests/admission/test_embedding_admission.sh"
    "test/unit_tests/test_embedding_dump_load.sh"
    "test/unit_tests/incremental_dump/test_incremental_dump.sh"
    # Registered in both groups: it covers forward and checkpoints, and runs
    # only the half named by the group it is passed.
    "test/unit_tests/test_table_row_wise.sh"
)

case "$1" in
    fwd_bwd)
        TEST_FILES=("${FWD_BWD_TEST_FILES[@]}")
        ;;
    load_dump)
        TEST_FILES=("${LOAD_DUMP_TEST_FILES[@]}")
        ;;
    *)
        TEST_FILES=("${FWD_BWD_TEST_FILES[@]}" "${LOAD_DUMP_TEST_FILES[@]}")
        ;;
esac

# Run each test file using the appropriate command
for TEST_FILE in "${TEST_FILES[@]}"; do
    echo "Running tests in $TEST_FILE"
    if [[ "$TEST_FILE" == *.py ]]; then
        # Run Python test files with pytest
        pytest -svv "$TEST_FILE"
        # Check if the test failed
        if [ $? -ne 0 ]; then
            echo "ERROR: Test failed in $TEST_FILE"
            exit 1
        fi
    elif [[ "$TEST_FILE" == *.sh ]]; then
        # Run shell scripts with bash. The group travels in the environment
        # rather than as an argument, so that a script spanning both groups
        # can run only the half being asked for without the others seeing an
        # argument they did not expect -- test_hybrid_storage_export.sh
        # forwards "$@" to pytest, which would read "fwd_bwd" as a path to
        # collect and fail.
        DYNAMICEMB_TEST_GROUP="$1" bash "$TEST_FILE"
        # Check if the test failed
        if [ $? -ne 0 ]; then
            echo "ERROR: Test failed in $TEST_FILE"
            exit 1
        fi
    else
        echo "Unknown test file type: $TEST_FILE"
    fi
done
