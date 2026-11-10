# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Every place where the package waits for the device on purpose is listed, with its reason.

Searching the package for ``read_now``, ``pull_to_host``, ``.wait(`` and ``flagged_compact`` (the
helper that reads a count back) finds each of them: the audited reads of ``sync_free``, the wait of
a ring of late reads, and the compaction whose size is a host int. ``AUDITED_READS`` says, for each
``(file, function, kind)``, how many calls there are and why they are by design. The test fails

* for a call that is not listed, with the line to add: a new read is a decision that a reviewer
  makes, with its reason, and not a side effect of a change;
* for a listed entry that is gone or whose number of calls changed, so that the table cannot go
  stale and stays the answer to "what can still wait for the GPU".

Only calls are matched, by the syntax tree (a comment, a string or a definition is not one), in
every file of ``dynamicemb/``. A call in a method is named ``Class.method``, in a nested function
``outer.inner``. The reasons begin with what kind of read it is:

* ``contract``: a participant that is host-sized by its own contract (a storage, a counter or an
  initializer that did not declare the padded contract) is given exact-length tensors, so the
  number of rows is read once per call into it. A class that declared it costs nothing;
* ``public API``: the method returns host data (a count, a tensor of a size only the device knows);
* ``late check``: an answer that was copied back without waiting and is reported later; this is
  the one place that waits for it, when the host is a whole ring ahead of the device;
* ``user wait``: the caller asked to wait, on a handle it holds;
* ``TorchRec``: the input distribution of TorchRec, which is not DynamicEmb's code.

Branches. A change that adds, removes or moves a blocking read edits its row in the same commit,
as one that lowers ``budgets.json`` edits that file; this file is the one place that says what
can wait for the device, and it is true of the tree it is in. It cannot know what another
branch adds, so merging branches is a step of its own: run this file on the merged tree. The
first test then names each call that has no row, with the line to add (the decision whether it
is by design, and why, is the reviewer's), and the second each row whose count changed or whose
call is gone. Rows are keyed by ``(file, function, kind)``, so branches that change different
functions merge without a textual conflict, and two that change the same function show up as a
count that is wrong, not as a pass.

CPU only, standard library only.
"""
import ast
import collections
import os
from typing import Dict, Iterator, List, Tuple

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
# corelib/dynamicemb: the files are named relative to it, "dynamicemb/types.py" as in the lint
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PACKAGE = os.path.join(ROOT, "dynamicemb")

# A call to a function or method of this name is a read; ``wait`` only as a method (``ring.wait(t)``,
# ``handle.wait()``): a bare ``wait(...)`` is some other function.
READ_FUNCTIONS = ("read_now", "pull_to_host", "flagged_compact")
READ_METHODS = ("wait",)
CATEGORIES = ("contract", "public API", "late check", "user wait", "TorchRec")

# (file, function, kind) -> (number of calls, reason)
AUDITED_READS: Dict[Tuple[str, str, str], Tuple[int, str]] = {
    # ---- sync_free.read_now ----------------------------------------------------------------
    ("dynamicemb/types.py", "_live_rows", "read_now"): (
        1,
        "contract: how many rows are live, before the find / insert of a storage that did not "
        "declare the padded contract (OldContractStorage), the one place that makes this read",
    ),
    ("dynamicemb/sync_free.py", "initialize_masked", "read_now"): (
        1,
        "contract: how many rows an initializer written for exact index lists is to fill "
        "(dense=True; an initializer that declared skips_negative_index costs nothing)",
    ),
    ("dynamicemb/batched_dynamicemb_function.py", "_read_miss_counts", "read_now"): (
        1,
        "contract: the number of misses of a step whose tier below storage did not declare the "
        "padded contract, so that a step that missed nothing does not call it (and the "
        "number of keys it will be asked for, which saves its adapter a read)",
    ),
    ("dynamicemb/key_value_table.py", "_resolve_pending_evicted", "read_now"): (
        1,
        "public API: pop_evicted_keys with no ticket for its count: the result has a "
        "data-dependent size",
    ),
    ("dynamicemb/key_value_table.py", "DynamicEmbCache.cache_metrics", "read_now"): (
        1,
        "public API: the getter returns the hit counts, which are kept on the device",
    ),
    (
        "dynamicemb/shard/embedding.py",
        "ShardedDynamicEmbeddingCollection._dedup_indices",
        "read_now",
    ): (
        1,
        "TorchRec: only for a sharding whose input distributor is TorchRec's (table-wise, "
        "column-wise, data-parallel): its all-to-all needs the arrays at their exact length, which "
        "the row-wise distributor of DynamicEmb cuts itself once the splits are known",
    ),
    ("dynamicemb/key_value_table.py", "read_table_sizes_now", "read_now"): (
        1,
        "public API: a checkpoint load reads the exact table sizes once, to size the tables for "
        "the keys it keeps; a step decides to grow from the late snapshots and never comes here",
    ),
    (
        "dynamicemb/scored_hashtable.py",
        "LinearBucketTable._sorted_insert_and_evict",
        "read_now",
    ): (
        1,
        "public API: insert_and_evict without async_count returns the number of evictions as a "
        "host int; every path of a step passes async_count=True and leaves the count on the device",
    ),
    # ---- the wait of a ring of late reads ---------------------------------------------------
    ("dynamicemb/key_value_table.py", "_resolve_pending_evicted", "wait"): (
        1,
        "public API: pop_evicted_keys waits for the count of each pending entry (a step "
        "only folds what has landed)",
    ),
    (
        "dynamicemb/batched_dynamicemb_tables.py",
        "BatchedDynamicEmbeddingTablesV2._settle_oldest",
        "wait",
    ): (
        1,
        "late check: flush() waits for every answer of the negative-score check in flight "
        "(wait=True); a step only polls and never waits",
    ),
    # ---- the count of a compaction, read back ----------------------------------------------
    ("dynamicemb/key_value_table.py", "_compact_missing", "flagged_compact"): (
        1,
        "public API: Storage.find of the in-tree storages returns the number of missing keys as "
        "a host int; the padded bodies use lookup_masked / find_masked and never come here",
    ),
    ("dynamicemb/key_value_table.py", "_find_keys", "flagged_compact"): (
        1,
        "public API: the compact=True form (the default) is the host-count find; every call in "
        "the package passes compact=False",
    ),
    (
        "dynamicemb/embedding_admission.py",
        "MultiTableFrequencyAdmitter.admit",
        "flagged_compact",
    ): (
        1,
        "contract: a counter that did not declare the padded contract takes the admitted keys "
        "compacted to erase them; one that declared it is given the whole batch",
    ),
    # ---- what the caller waits for ---------------------------------------------------------
    ("dynamicemb/async_erase.py", "EraseHandle.result", "wait"): (
        1,
        "user wait: result() blocks until the erase has finished",
    ),
    ("dynamicemb/async_erase.py", "EraseHandleGroup.wait", "wait"): (
        1,
        "user wait: the group of handles of one erase call is waited for as one",
    ),
    ("dynamicemb/shard/input_dist.py", "_ScoredTensorsAwaitable._wait_impl", "wait"): (
        1,
        "TorchRec: the awaitable of the scores' all-to-all, waited for where TorchRec asks for "
        "the result",
    ),
    ("dynamicemb/shard/input_dist.py", "_ScoredSplitsAwaitable._wait_impl", "wait"): (
        2,
        "TorchRec: the awaitables of the keys' and the scores' all-to-all, waited for where "
        "TorchRec asks for the result",
    ),
}


# --------------------------------------------------------------------------
# the search
# --------------------------------------------------------------------------
class _Reads(ast.NodeVisitor):
    def __init__(self) -> None:
        self.scope: List[str] = []
        self.sites: List[Tuple[str, str, int]] = []  # (function, kind, line)

    def _enter(self, node) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _enter

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        kind = None
        if isinstance(func, ast.Name) and func.id in READ_FUNCTIONS:
            kind = func.id
        elif isinstance(func, ast.Attribute):
            if func.attr in READ_FUNCTIONS or func.attr in READ_METHODS:
                kind = func.attr
        if kind is not None:
            self.sites.append((".".join(self.scope) or "<module>", kind, node.lineno))
        self.generic_visit(node)


def find_reads(source: str) -> List[Tuple[str, str, int]]:
    """``(function, kind, line)`` of every read in ``source``."""
    visitor = _Reads()
    visitor.visit(ast.parse(source))
    return visitor.sites


def _package_files() -> Iterator[Tuple[str, str]]:
    for dirpath, dirnames, filenames in os.walk(PACKAGE):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for name in sorted(filenames):
            if name.endswith(".py"):
                path = os.path.join(dirpath, name)
                yield os.path.relpath(path, ROOT).replace(os.sep, "/"), path


def reads_of_the_package() -> Dict[Tuple[str, str, str], List[int]]:
    """``(file, function, kind)`` -> the lines of its calls, over the whole package."""
    found: Dict[Tuple[str, str, str], List[int]] = collections.defaultdict(list)
    for relative, path in _package_files():
        with open(path) as f:
            for function, kind, line in find_reads(f.read()):
                found[(relative, function, kind)].append(line)
    return found


def _entry(key: Tuple[str, str, str], count: int) -> str:
    file, function, kind = key
    return '    (%r, %r, %r): (%d, "<why it is by design>"),' % (
        file,
        function,
        kind,
        count,
    )


# --------------------------------------------------------------------------
# the table is the tree
# --------------------------------------------------------------------------
def test_every_read_of_the_package_is_audited():
    found = reads_of_the_package()
    unlisted = [
        "%s line %s: %s() in %s\n  to accept it, add to AUDITED_READS:\n%s"
        % (key[0], ", ".join(map(str, lines)), key[2], key[1], _entry(key, len(lines)))
        for key, lines in sorted(found.items())
        if key not in AUDITED_READS
    ]
    assert not unlisted, (
        "a blocking read that is not in AUDITED_READS (if it is by design, list it with its "
        "reason, else use the padded forms):\n" + "\n".join(unlisted)
    )


def test_no_audited_read_is_stale():
    found = reads_of_the_package()
    stale = []
    for key, (count, _) in sorted(AUDITED_READS.items()):
        now = len(found.get(key, ()))
        if now != count:
            stale.append(
                "%s: %s() in %s is listed with %d call(s), the tree has %d"
                % (key[0], key[2], key[1], count, now)
            )
    assert not stale, (
        "AUDITED_READS names reads that are gone or changed (remove or correct the entry):\n"
        + "\n".join(stale)
    )


def test_every_reason_names_its_kind_and_says_why():
    for key, (count, reason) in AUDITED_READS.items():
        assert count >= 1, key
        assert reason.split(":")[0] in CATEGORIES, (key, reason)
        assert len(reason.split(":", 1)[1].split()) >= 4, (key, reason)
        assert key[2] in READ_FUNCTIONS + READ_METHODS, key


# --------------------------------------------------------------------------
# the search itself
# --------------------------------------------------------------------------
SAMPLE = """
import sync_free as sf
from sync_free import read_now, pull_to_host


def read_now(*tensors):             # a definition is not a call
    return tensors


def flat(t):
    return sf.read_now(t)           # a method call, and the call below a name


class K:
    def m(self, ring, t, h):
        read_now(t)
        ring.wait(t)
        h.wait()
        pull_to_host(t)
        flagged_compact(t, [t])
        wait(t)                     # a function named wait is not a read
        x = "read_now(t)"           # a string
        # read_now(t)               a comment

        def inner():
            return sf.pull_to_host(t)

        return inner


class Outer:
    class Inner:
        def go(self, r):
            return r.wait()


async def late(r):
    return r.wait()


LAMBDA = lambda r: r.wait()         # at module level
"""


def test_the_search_finds_calls_only_and_names_their_scope():
    sites = collections.Counter(
        (function, kind) for function, kind, _ in find_reads(SAMPLE)
    )
    assert sites == collections.Counter(
        {
            ("flat", "read_now"): 1,
            ("K.m", "read_now"): 1,
            ("K.m", "wait"): 2,
            ("K.m", "pull_to_host"): 1,
            ("K.m", "flagged_compact"): 1,
            ("K.m.inner", "pull_to_host"): 1,
            ("Outer.Inner.go", "wait"): 1,
            ("late", "wait"): 1,
            ("<module>", "wait"): 1,
        }
    )


def test_the_search_gives_the_lines():
    lines = {(function, kind): line for function, kind, line in find_reads(SAMPLE)}
    text = SAMPLE.split("\n")
    assert "read_now(t)" in text[lines[("flat", "read_now")] - 1]
    assert "flagged_compact" in text[lines[("K.m", "flagged_compact")] - 1]


def test_the_package_is_found():
    """A path that is wrong would make both tests above pass on an empty search."""
    assert os.path.isfile(os.path.join(PACKAGE, "sync_free.py")), PACKAGE
    assert reads_of_the_package(), "the search found no read at all"
    for key in (
        ("dynamicemb/types.py", "_live_rows", "read_now"),
        ("dynamicemb/shard/input_dist.py", "_ScoredSplitsAwaitable._wait_impl", "wait"),
    ):
        assert key in reads_of_the_package(), key


@pytest.mark.parametrize("name", READ_FUNCTIONS)
def test_the_helpers_the_search_looks_for_exist(name):
    """The helper a search looks for exists, so that a rename cannot make the search find
    nothing: ``read_now``, ``pull_to_host`` and ``flagged_compact`` are defined in the package
    or imported from the extension."""
    defined = False
    for _, path in _package_files():
        with open(path) as f:
            source = f.read()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                defined = True
            if isinstance(node, ast.ImportFrom):
                defined = defined or any(alias.name == name for alias in node.names)
    assert defined, "%s is not defined or imported anywhere in the package" % name
