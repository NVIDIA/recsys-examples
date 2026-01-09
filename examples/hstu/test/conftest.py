# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Pytest configuration for limiting parametrized test cases during debugging.

Usage:
    # Run only the first parameter combination for all parametrized tests
    PYTEST_FIRST_PARAM_ONLY=1 pytest
    
    # Run all parameter combinations (default behavior)
    pytest
"""

import os


def pytest_configure(config):
    """Register custom marker for first parameter only mode."""
    config.addinivalue_line(
        "markers",
        "first_param_only: automatically added when PYTEST_FIRST_PARAM_ONLY is set",
    )


def pytest_generate_tests(metafunc):
    """
    Hook to modify parametrized tests to only run the first parameter combination.

    This is useful during debugging when you want to quickly test if the test
    infrastructure works without running all parameter combinations.

    Set environment variable PYTEST_FIRST_PARAM_ONLY=1 to enable this behavior.
    """
    # Check if we should limit to first parameter only
    if os.getenv("PYTEST_FIRST_PARAM_ONLY", "0") == "1":
        # Check if this test function has parametrize
        if hasattr(metafunc, "definition") and hasattr(metafunc.definition, "callspec"):
            # This will be handled by pytest_collection_modifyitems
            pass


def pytest_collection_modifyitems(config, items):
    """
    Modify collected test items to only keep the first parameter combination
    for each parametrized test function when PYTEST_FIRST_PARAM_ONLY is set.
    """
    if os.getenv("PYTEST_FIRST_PARAM_ONLY", "0") != "1":
        return

    # Group items by their base test name (without parameter suffix)
    test_groups = {}
    for item in items:
        # Get the base test name (remove parameter suffix like [param0])
        base_name = item.nodeid.split("[")[0]
        if base_name not in test_groups:
            test_groups[base_name] = []
        test_groups[base_name].append(item)

    # Filter to keep only the first item from each group that has multiple items
    items_to_keep = []
    for base_name, group_items in test_groups.items():
        if len(group_items) > 1:
            # This is a parametrized test, keep only the first one
            items_to_keep.append(group_items[0])
            print(
                f"  [PYTEST_FIRST_PARAM_ONLY] Keeping only first parameter for: {base_name}"
            )
        else:
            # Not parametrized or only one parameter, keep it
            items_to_keep.extend(group_items)

    # Replace the items list
    items[:] = items_to_keep
