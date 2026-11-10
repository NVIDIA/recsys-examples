# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every public method of a storage orders the stream after a growth, or says why it need not.

A growth only queues its kernels on the stream that decided to grow, so whoever reads the key
map or a value buffer from another stream has to be ordered after it first
(``DynamicEmbTableState.wait_for_growth``, see ``growth_order.py``). The forward, the
backward and the evaluation of the module do it themselves; every other caller -- a dump, an
export, an erase, a flush, a replay -- gets it from the storage's own method, which is
decorated with ``_after_growth`` (or, for a generator, calls ``self.wait_for_growth()`` in
its body, because a generator runs at its first ``next()``).

A method that is added without it reopens the race the decorator closes (a dump on a stream
that never ran a forward reads a key map the rehash has not built), and nothing at runtime
would say so. This reads ``key_value_table.py`` and fails for the next one, with no GPU and
no torch.
"""

import ast
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # corelib/dynamicemb
KVT = os.path.join(ROOT, "dynamicemb", "key_value_table.py")
STORAGES = ("DynamicEmbStorage", "HybridStorage")
DECORATOR = "_after_growth"

# The public methods that are not decorated, and why none needs to be.
NOT_ORDERED = {
    "wait_for_growth": "it is the order itself",
    "expand_if_need": "a growth entry: the section it runs in orders itself against every "
    "stream that has used the table (GrowthOrder.section)",
    "note_forwarded": "called by the forward, which has registered its stream and ordered it "
    "after the last growth (_note_table_users) before it gets here",
    "pop_evicted_keys": "hands back the keys earlier inserts retained, tensors those calls "
    "made: neither the key map nor a value buffer, which a growth replaces",
    "pop_erased_keys": "the same, for the keys an erase retained",
    "pop_erased_slot_index": "the same, for the slots an erase retained",
    "erase_untracked": "reads a set of table ids kept on the host",
    "embedding_dtype": "reads the options of the tables, not their state",
    "embedding_dim": "a host list of the dims",
    "value_dim": "a host list of the dims",
    "max_embedding_dim": "a host number",
    "max_value_dim": "a host number",
    "embedding_dims": "the dims of the tables, which a growth does not change",
    "all_dims_vec4": "a host bool",
    "init_optimizer_state": "a host number",
    "set_score": "sets a host number",
    "evict_strategy": "a host enum",
}


def storages():
    """``{class name: {method name: (FunctionDef, [decorator source])}}`` of the public
    methods that are not properties."""
    with open(KVT) as f:
        tree = ast.parse(f.read())
    found = {}
    for node in tree.body:
        if not (isinstance(node, ast.ClassDef) and node.name in STORAGES):
            continue
        methods = {}
        for item in node.body:
            if not isinstance(item, ast.FunctionDef) or item.name.startswith("_"):
                continue
            decorators = [ast.unparse(d) for d in item.decorator_list]
            if any(
                d == "property" or d.endswith((".setter", ".getter"))
                for d in decorators
            ):
                continue
            methods[item.name] = (item, decorators)
        found[node.name] = methods
    return found


def is_generator(function: ast.FunctionDef) -> bool:
    return any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(function))


def test_the_storages_are_found():
    found = storages()
    assert set(found) == set(STORAGES)
    assert all(len(methods) > 10 for methods in found.values())


def test_every_public_method_of_a_storage_is_ordered_after_a_growth_or_listed():
    unordered = [
        "%s.%s" % (cls, name)
        for cls, methods in storages().items()
        for name, (function, decorators) in methods.items()
        # a generator orders itself in its body, which the last test of this file reads
        if DECORATOR not in decorators
        and name not in NOT_ORDERED
        and not is_generator(function)
    ]
    assert not unordered, (
        "these public methods of a storage neither carry @%s nor are on NOT_ORDERED of "
        "this file: %s. A method that reads the key map or a value buffer has to order the "
        "current stream after the last growth (see the docstring of this file); one that "
        "does not has to say why." % (DECORATOR, ", ".join(unordered))
    )


def test_the_list_of_the_methods_that_need_no_order_is_not_stale():
    methods = {}
    for found in storages().values():
        for name, (_, decorators) in found.items():
            methods.setdefault(name, []).append(DECORATOR in decorators)
    for name, reason in NOT_ORDERED.items():
        assert reason.strip(), name
        assert name in methods, "%s is on NOT_ORDERED and is no public method" % name
        assert not any(methods[name]), "%s is on NOT_ORDERED and is decorated" % name


def test_a_generator_orders_itself_when_it_runs_not_when_it_is_made():
    generators = [
        (cls, name, function, decorators)
        for cls, methods in storages().items()
        for name, (function, decorators) in methods.items()
        if is_generator(function)
    ]
    assert generators, "the storages have export_keys_values"
    for cls, name, function, decorators in generators:
        where = "%s.%s" % (cls, name)
        assert DECORATOR not in decorators, (
            "%s is a generator: @%s would order the stream that made the iterator, not the "
            "one that runs it" % (where, DECORATOR)
        )
        first_yield = min(
            n.lineno
            for n in ast.walk(function)
            if isinstance(n, (ast.Yield, ast.YieldFrom))
        )
        waits = [
            n.lineno
            for n in ast.walk(function)
            if isinstance(n, ast.Call) and ast.unparse(n.func) == "self.wait_for_growth"
        ]
        assert waits and min(waits) < first_yield, (
            "%s has to call self.wait_for_growth() before its first yield" % where
        )


def test_the_decorator_refuses_a_generator():
    with open(KVT) as f:
        tree = ast.parse(f.read())
    (decorator,) = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == DECORATOR
    ]
    refusals = [
        n
        for n in ast.walk(decorator)
        if isinstance(n, ast.Assert) and "isgeneratorfunction" in ast.unparse(n.test)
    ]
    assert refusals, "%s must assert that it is not given a generator" % DECORATOR
