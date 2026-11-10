set -e
FWD_BWD_TEST_FILES=(
    "test/unit_tests/table_operation/test_table_operation.sh"
    "test/unit_tests/erase_async/test_erase_async.sh"
    "test/unit_tests/retain_evicted_keys/test_retain_evicted_keys.sh"
    "test/unit_tests/test_hybrid_storage_export.sh"
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
    "test/unit_tests/test_table_rows.py"
    "test/unit_tests/test_int64_weights_channel.py"
    "test/unit_tests/test_frequency_requirements.py"
    "test/unit_tests/test_frequency_counter_guard.py"
    "test/unit_tests/per_key_score/test_per_key_score.sh"
    "test/unit_tests/test_planner_plan_cost.py"
    "test/unit_tests/test_planner_placement.py"
    "test/unit_tests/test_table_row_wise.sh"
    "test/unit_tests/test_hier_dist.sh"
    "test/unit_tests/test_hier_overflow.sh"
    "test/unit_tests/test_hier_pipeline.sh"
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

# Files of the groups above that build sharded EmbeddingCollections, run
# again on the hierarchical output dist with their configurations unchanged
# (test/unit_tests/hier_switch.py; docs/hier_output_dist.md, section 18).
FWD_BWD_HIER_FILES=(
    "test/unit_tests/test_sequence_embedding.sh"
    "test/unit_tests/test_twin_module.sh"
    "test/unit_tests/erase_async/test_erase_async.sh"
    "test/unit_tests/per_key_score/test_per_key_score.sh"
    "test/unit_tests/test_lfu_scores.sh"
)

LOAD_DUMP_HIER_FILES=(
    "test/unit_tests/admission/test_embedding_admission.sh"
    "test/unit_tests/incremental_dump/test_incremental_dump.sh"
    # Not test_embedding_dump_load.sh: each of its launches shards without
    # index dedup, which the dist does not take, so none would run on it.
)

case "$1" in
    fwd_bwd)
        TEST_FILES=("${FWD_BWD_TEST_FILES[@]}")
        HIER_FILES=("${FWD_BWD_HIER_FILES[@]}")
        ;;
    load_dump)
        TEST_FILES=("${LOAD_DUMP_TEST_FILES[@]}")
        HIER_FILES=("${LOAD_DUMP_HIER_FILES[@]}")
        ;;
    *)
        TEST_FILES=("${FWD_BWD_TEST_FILES[@]}" "${LOAD_DUMP_TEST_FILES[@]}")
        HIER_FILES=("${FWD_BWD_HIER_FILES[@]}" "${LOAD_DUMP_HIER_FILES[@]}")
        ;;
esac

# Run one test file of a group using the appropriate command
run_file() {
    local group=$1
    local TEST_FILE=$2
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
        DYNAMICEMB_TEST_GROUP="$group" bash "$TEST_FILE"
        # Check if the test failed
        if [ $? -ne 0 ]; then
            echo "ERROR: Test failed in $TEST_FILE"
            exit 1
        fi
    else
        echo "Unknown test file type: $TEST_FILE"
    fi
}

# Run each test file
for TEST_FILE in "${TEST_FILES[@]}"; do
    echo "Running tests in $TEST_FILE"
    run_file "$1" "$TEST_FILE"
done

# Then the hierarchical runs, unless the build has no dist: there, each model
# given output_dist="hierarchical" raises ImportError. The probe loads torch
# first, for libc10 (dynamicemb/__init__.py). A probe that fails does not skip
# the runs.
if [ "$(python -c 'import torch, dynamicemb_extensions as ext; print(hasattr(ext, "HierOutputDist"))')" = False ]; then
    echo "this build has no hierarchical dist: the runs on it are skipped"
else
    for TEST_FILE in "${HIER_FILES[@]}"; do
        echo "Running tests in $TEST_FILE on the hierarchical output dist"
        (
            export DYNAMICEMB_TEST_OUTPUT_DIST=hierarchical PYHKV_DEBUG_STRICT=1
            run_file "$1" "$TEST_FILE"
        ) || exit 1
    done
fi
