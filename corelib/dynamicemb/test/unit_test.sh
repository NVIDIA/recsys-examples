set -e
FWD_BWD_TEST_FILES=(
    "test/unit_tests/table_operation/test_table_operation.sh"
    "test/unit_tests/test_lfu_scores.sh"
    "test/test_batched_dynamic_embedding_tables_v2.py"
    "test/test_unique_op.py"
    "test/unit_tests/test_sequence_embedding.sh"
    "test/unit_tests/test_pooled_embedding.sh"
    "test/unit_tests/test_twin_module.sh"
)

LOAD_DUMP_TEST_FILES=(
    "test/unit_tests/test_embedding_admission.sh"
    "test/unit_tests/test_embedding_dump_load.sh"
    "test/unit_tests/incremental_dump/test_incremental_dump.sh"
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
        # Run shell scripts with bash
        bash "$TEST_FILE"
        # Check if the test failed
        if [ $? -ne 0 ]; then
            echo "ERROR: Test failed in $TEST_FILE"
            exit 1
        fi
    else
        echo "Unknown test file type: $TEST_FILE"
    fi
done
