set -e
TEST_FILES=(
    # "test/unit_tests/test_embedding_admission.sh" # Passed
    # "test/unit_tests/table_operation/test_table_operation.sh" # Passed
    # "test/unit_tests/test_lfu_scores.sh" # Failed: TODO
    # "test/test_batched_dynamic_embedding_tables_v2.py" # Passed
    # "test/test_unique_op.py" # Passed
    # "test/unit_tests/test_sequence_embedding.sh" # Passed
    # "test/unit_tests/test_pooled_embedding.sh" # Failed: pooled implementation
    # "test/unit_tests/test_dynamicemb_table_dump_load.sh" # Failed: reserve
    # "test/unit_tests/test_embedding_dump_load.sh" # Passed
    # "test/unit_tests/test_twin_module.sh" # Passed
    "test/unit_tests/incremental_dump/test_incremental_dump.sh" # Passed
)
export DYNAMICEMB_DUMP_LOAD_DEBUG=1
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
