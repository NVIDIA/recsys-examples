#!/bin/bash
# Erasing the keys whose score is below a threshold, in the background of
# training. Today: the kernel and its Python wrapper on a hash table, and what an
# erase leaves in a slot. The storage layer, the worker thread and the API come
# with their own files here.
set -e

DIR=./test/unit_tests/erase_async

pytest -svv $DIR/test_erase_by_score.py
