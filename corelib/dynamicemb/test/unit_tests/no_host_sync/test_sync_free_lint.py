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

"""Static lint of the sync-free code: the host-sync idioms, found in the syntax tree.

CPU only, standard library only (``ast``, ``tokenize``): it runs without torch or a GPU.

Code that runs on the training thread of a sync-free path must not read a CUDA tensor,
and the idioms that do are easy to write by accident. This lint walks the functions named
in TARGETS (the sync-free paths of the design) and flags, per function:

  sync-call           .item() .cpu() .tolist() .numpy() .nonzero() .unique() .masked_select()
                      .bincount() .synchronize() torch.equal/allclose, .to("cpu")
  host-read-helper    flagged_compact(...) and the pinned-copy helpers that read a count
  mask-index          x[bool_mask] and x[bool_mask] = v (the index is a comparison, a negation,
                      an & or | of masks, or is named like a mask)
  scalar-put          x[tensor_index] = <python number> (a blocking H2D copy: use index_fill_)
  host-tensor-ctor    torch.tensor(..., device=...), and host data copied to the device: .cuda()
                      without non_blocking=True, .cuda() / .to(device) of a torch.tensor(...)
                      (pageable memory: a blocking H2D copy)
  repeat-interleave   without output_size
  where-1arg          torch.where(cond) (it is nonzero)
  pin-memory          pin_memory=True / .pin_memory() (an allocation: only at construction)
  blocking-copy       non_blocking=False
  scalar-of-tensor    int() / bool() / float() of a reduction (x.sum(), x.max(), ...) or of a
                      name that holds a tensor; range() or a slice bound that is such a name
  reduction-in-branch a reduction (.any() .all() .sum() .max() ..., torch.sum(x)) in an if /
                      while / conditional expression / assert, whatever it is compared with:
                      `if miss.sum() > 0:` reads the sum
  branch-on-tensor    an if / while / conditional expression / assert whose test is a bare name,
                      attribute or call that is not a known host fact (a tensor in an `if`
                      reads it); host facts are bool/int/str parameters, names assigned from
                      `is None`, isinstance, len, gate.needed(), read_now(), arithmetic of host
                      facts ... and a few well-known flags. A parameter annotated as a tensor,
                      and a name assigned from a tensor expression (`is_miss = valid & ~founds`,
                      torch.where, a reduction), is a tensor whatever it is called, so
                      `if is_miss:` is flagged
  async-count-read    the count of evicted keys that a call returns on the device (a call that
                      passes async_count=True or collect_evicted=True, or table_insert_and_evict,
                      as the second element of its result; or a parameter or free name like
                      num_evicted / unique_total that the function did not make a host value) is
                      used in a branch condition, a slice bound, range(), or int()/bool()/float()

It is a lint, not a proof: a CPU tensor may call .item() and a Python list may be indexed by
a name called `hits`. A line that is right anyway carries a `# sync-ok: <why>` comment (on
the line, on the first line of its statement, or on a comment line right above it).

TARGETS lists the functions. A target that is new in the sync-free design (it does not exist
before it) is enforced as soon as it exists: there is no legacy body to excuse. A target that
exists today keeps its legacy branches until the package that rewrites it lands. It may keep
the violations it has on the tree before the removal (LEGACY_BASELINE, per rule: reported as an
expected failure with the count per rule) and may not gain one: a clean legacy function stays
clean, so the rewrite of `forward` or `backward` cannot add a sync unnoticed. Its flag `enforce`
is switched on when its package lands, with a `# sync-ok` on each branch that stays by design
(the external PS, the determinism mode, the host-count mode of async_count).
SYNC_LINT_ENFORCE_ALL=1 enforces every target now, SYNC_LINT_REQUIRE_NEW=1 fails a new target
that is missing and SYNC_LINT_STRICT=1 fails a legacy target that is below its baseline, so that
the baseline is lowered with the change that removed the violation. `python
test_sync_free_lint.py` prints the report.

The same file also compares the whole tree with the regex lint of the sync census
(sync_lint.py, lint_baseline.json): a line that matches a host-sync pattern and was not in the
tree before the removal must be on ALLOWED_NEW_HITS, with the reason, and a line that a change
claims to remove (EXPECTED_GONE, appended to by that change) must be gone.
"""
import ast
import functools
import io
import json
import os
import re
import sys
import textwrap
import tokenize
from typing import Dict, Iterable, List, NamedTuple, Optional, Set, Tuple

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # corelib/dynamicemb

import sync_lint  # noqa: E402

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 9),
    reason="the lint reads ast.Subscript.slice of Python 3.9+",
)


# --------------------------------------------------------------------------
# the engine
# --------------------------------------------------------------------------
class Violation(NamedTuple):
    lineno: int
    rule: str
    message: str
    source: str  # the source line, stripped


SYNC_METHODS = frozenset(
    {
        "item",
        "cpu",
        "tolist",
        "numpy",
        "nonzero",
        "argwhere",
        "unique",
        "unique_consecutive",
        "masked_select",
        "bincount",
        "synchronize",
        "equal",
        "allclose",
    }
)
HOST_READ_HELPERS = frozenset(
    {"flagged_compact", "_copy_cuda_tensor_to_pinned_cpu", "_scalar_item", "_bool_item"}
)
REDUCTIONS = frozenset(
    {
        "sum",
        "max",
        "min",
        "any",
        "all",
        "prod",
        "mean",
        "item",
        "argmax",
        "argmin",
        "count_nonzero",
        "amax",
        "amin",
    }
)
# calls whose result is a Python value (int(), bool() and float() of a device reduction are
# flagged as scalar-of-tensor where they are called)
HOST_FUNCS = frozenset(
    {
        "isinstance",
        "issubclass",
        "hasattr",
        "callable",
        "len",
        "all",
        "any",
        "range",
        "id",
        "getattr",
        "int",
        "bool",
        "float",
        "list",
        "dict",
        "set",
        "tuple",
        "sorted",
    }
)
HOST_PREFIXES = (
    "_in_tree",
    "_uses_",
    "_is_",
    "_masked_",
    "is_",
    "has_",
    "can_",
    "need_",
    "needs_",
    "use_",
    "enable_",
    "should_",
)
HOST_METHODS = frozenset(
    {
        "get",
        "startswith",
        "endswith",
        "needed",
        "ready",
        "query",
        "free_slots",
        "is_pinned",
        "is_contiguous",
        "numel",
        "nelement",
        "dim",
        "size",
        "stride",
        "element_size",
    }
)
# names that are Python flags wherever they appear
KNOWN_HOST_NAMES = frozenset(
    {
        "self",
        "cls",
        "training",
        "block",
        "exact",
        "masked",
        "async_count",
        "compact",
        "need",
        "miss_path",
        "rng",
        "ps",
        "verbose",
    }
)
HOST_NAME_RE = re.compile(
    r"^(is|has|use|enable|enabled|need|needs|can|should|allow)_|_(flag|enabled|mode|strategy|dtype)$"
)
# a bare attribute in a branch test is a host flag (`self.training`) unless it is named like a tensor
TENSOR_ATTR_RE = re.compile(r"(mask|masks|num_evicted|founds)$")
# an index with such a name is probably a boolean mask
MASK_NAME_RE = re.compile(
    r"(^|_)(mask|masks|flag|flags|found|founds|valid|keep|live|hit|hits|miss|misses|admitted|refused|persisted|gone|present|selected)(_|$)"
    r"|^(is|in|has|non)_"
)
# ... unless it ends like the name of positions: `refused_pos`, `hit_rows`
INDEX_NAME_RE = re.compile(r"(^|_)(pos|positions|idx|indices|index|rows|slots|ids)$")
_VALUE_COMPARISONS = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
PRAGMA = "sync-ok"
KNOWN_RULES = frozenset(
    {
        "sync-call",
        "host-read-helper",
        "mask-index",
        "scalar-put",
        "host-tensor-ctor",
        "repeat-interleave",
        "where-1arg",
        "pin-memory",
        "blocking-copy",
        "scalar-of-tensor",
        "reduction-in-branch",
        "branch-on-tensor",
        "async-count-read",
    }
)
# `torch.<name>(...)` calls that return a tensor (what they are assigned to is a tensor whatever
# it is called) and the tensor methods whose result is a tensor when the receiver is one
TENSOR_FUNCS = frozenset(
    {
        "where",
        "logical_and",
        "logical_or",
        "logical_not",
        "logical_xor",
        "isin",
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "bitwise_not",
        "isnan",
        "isinf",
        "isfinite",
        "zeros",
        "ones",
        "full",
        "empty",
        "arange",
        "zeros_like",
        "ones_like",
        "full_like",
        "empty_like",
        "cat",
        "stack",
        "clamp",
        "maximum",
        "minimum",
    }
) | (REDUCTIONS - {"item"})
TENSOR_METHODS = frozenset(
    {
        "to",
        "reshape",
        "view",
        "clone",
        "contiguous",
        "detach",
        "float",
        "long",
        "int",
        "bool",
        "flatten",
        "squeeze",
        "unsqueeze",
        "logical_not",
        "logical_and",
        "logical_or",
    }
)
# calls whose result is host data that was read from the device (a list of CPU tensors)
HOST_READ_CALLS = frozenset({"read_now", "pull_to_host"})
HOST_READ_METHODS = frozenset({"wait", "value"})
# methods whose result is a Python value or a CPU tensor (the call itself is a sync-call)
HOST_RESULT_METHODS = frozenset({"item", "tolist", "cpu", "numpy"})
# A call that passes one of these keywords as True returns the device count of evicted keys as the
# second element of its result (collect_evicted: `(rows, num_evicted, keys, table_ids)`), and the raw
# eviction kernels always do.
DEVICE_COUNT_KEYWORDS = ("async_count", "collect_evicted")
DEVICE_COUNT_CALLS = frozenset(
    {"table_insert_and_evict", "table_insert_collect_evicted"}
)
# A name that is such a count, and a device tensor in the sync-free code unless the function made a
# host value of it (`n = int(sf.read_now(count)[0])`).
DEVICE_COUNT_NAME_RE = re.compile(
    r"^(num|n|total)_(evicted|ev|unique|uniques|missing|miss|new|admitted|refused|gone|spilled)$"
    r"|^unique_total$"
)
# tensors built on the host (pageable memory) and the arguments that name a device
HOST_TENSOR_CTORS = frozenset(
    {
        "torch.tensor",
        "torch.as_tensor",
        "torch.from_numpy",
        "torch.Tensor",
        "torch.LongTensor",
        "torch.IntTensor",
        "torch.FloatTensor",
        "torch.BoolTensor",
    }
)
DEVICE_NAME_RE = re.compile(r"(^|_)(device|dev)$")
# questions about a tensor object that do not read it: isinstance(x, T), x.is_cuda, x.dtype
OBJECT_QUERIES = frozenset({"isinstance", "hasattr", "id", "type"})
OBJECT_ATTRIBUTES = frozenset(
    {"is_cuda", "device", "dtype", "shape", "ndim", "is_pinned", "is_contiguous"}
)


def _dotted(node: ast.AST) -> str:
    """``torch.cuda.synchronize`` for that attribute chain, '' when it is not one."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def _is_number(node: ast.AST) -> bool:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        node = node.operand
    return isinstance(node, ast.Constant) and isinstance(node.value, (bool, int, float))


def _is_plain_index(node: ast.AST) -> bool:
    """An index that cannot be a tensor: a constant, a slice, an Ellipsis, or a tuple of those."""
    if isinstance(node, (ast.Constant, ast.Slice)):
        return True
    if isinstance(node, ast.Tuple):
        return all(_is_plain_index(e) for e in node.elts)
    return False


def _is_host_call(call: ast.Call) -> bool:
    f = call.func
    if isinstance(f, ast.Name):
        return f.id in HOST_FUNCS or f.id.startswith(HOST_PREFIXES)
    if isinstance(f, ast.Attribute):
        return f.attr in HOST_METHODS or f.attr.startswith(HOST_PREFIXES)
    return False


def _callee(call: ast.Call) -> str:
    """The name a call is made through: `f` for f(...), `m` for x.m(...)."""
    f = call.func
    if isinstance(f, ast.Attribute):
        return f.attr
    return f.id if isinstance(f, ast.Name) else ""


def _host_value(expr: ast.AST, names: Iterable[str] = ()) -> bool:
    """Is the Python value of `expr` certainly a host fact (used to learn which names are)?
    `names` are the names already known to hold host facts."""
    if isinstance(
        expr,
        (
            ast.Constant,
            ast.List,
            ast.Tuple,
            ast.Dict,
            ast.Set,
            ast.ListComp,
            ast.SetComp,
            ast.DictComp,
        ),
    ):
        return True  # (a list of tensors is still a Python list: its truth value is its length)
    if isinstance(expr, ast.Name):
        return expr.id in names
    if isinstance(expr, ast.Compare):
        # `x is None` and `k in d` are Python facts; `a == b` or `a < b` is one when both sides are
        if not any(isinstance(op, _VALUE_COMPARISONS) for op in expr.ops):
            return True
        return all(_host_value(o, names) for o in [expr.left] + list(expr.comparators))
    if isinstance(expr, ast.BoolOp):
        return all(_host_value(v, names) for v in expr.values)
    if isinstance(expr, ast.UnaryOp):
        return _host_value(expr.operand, names)
    if isinstance(expr, ast.BinOp):  # `n - 1`, `n // 2`: arithmetic of host facts
        return _host_value(expr.left, names) and _host_value(expr.right, names)
    if isinstance(expr, ast.IfExp):
        return _host_value(expr.body, names) and _host_value(expr.orelse, names)
    if isinstance(expr, ast.Call):
        return (
            _is_host_call(expr)
            or _callee(expr) in HOST_READ_CALLS
            or (
                isinstance(expr.func, ast.Attribute)
                and expr.func.attr in HOST_RESULT_METHODS
            )
        )
    if isinstance(expr, ast.Subscript):
        # `read_now(x)[0]` and `ring.value(t)[0]` are CPU tensors, `x.shape[0]` a Python int
        base = expr.value
        if isinstance(base, ast.Call):
            return _callee(base) in HOST_READ_CALLS | HOST_READ_METHODS
        return isinstance(base, ast.Attribute) and base.attr == "shape"
    return False


def _host_names(func: ast.AST) -> Set[str]:
    """The names of `func` that hold host facts: parameters that are flags or numbers, and
    names assigned from host expressions."""
    names: Set[str] = set()
    args = func.args
    positional = args.posonlyargs + args.args
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    for a, d in list(zip(positional, defaults)) + list(
        zip(args.kwonlyargs, args.kw_defaults)
    ):
        if a.annotation is not None:
            host = "Tensor" not in ast.unparse(
                a.annotation
            )  # `Optional[EvictStrategy]` is a Python value
        else:
            host = (isinstance(d, ast.Constant) and d.value is not None) or bool(
                HOST_NAME_RE.search(a.arg)
            )
        if host:
            names.add(a.arg)
    for _ in range(2):  # a name assigned from another host name
        for node in ast.walk(func):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                if _host_value(node.value, names):
                    names.add(node.targets[0].id)
            elif (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], (ast.Tuple, ast.List))
                and isinstance(node.value, ast.Call)
                and _callee(node.value) in HOST_READ_CALLS
            ):  # `n_spilled, n_gone = sf.read_now(a, b)`: CPU tensors
                names.update(
                    e.id for e in node.targets[0].elts if isinstance(e, ast.Name)
                )
            elif isinstance(node, ast.For) and isinstance(node.target, ast.Name):
                if isinstance(node.iter, ast.Call) and _dotted(node.iter.func) in (
                    "range",
                    "enumerate",
                    "zip",
                ):
                    names.add(node.target.id)
    return names


def _may_be_tensor(expr: ast.AST, host: Set[str], tensor: Set[str]) -> bool:
    """Can `expr`, a side of a comparison, be a tensor? A name that is not a host fact (an
    unannotated parameter is not one: annotate a number as `n: int`), an element of one, a
    tensor expression, an attribute that is named like a mask."""
    if isinstance(expr, ast.Name):
        return expr.id in tensor or not (
            expr.id in host or HOST_NAME_RE.search(expr.id) or expr.id.isupper()
        )
    if isinstance(expr, ast.Subscript):
        return _may_be_tensor(expr.value, host, tensor)
    if isinstance(expr, ast.Attribute):
        return bool(TENSOR_ATTR_RE.search(expr.attr))
    if isinstance(expr, ast.BinOp):
        return _may_be_tensor(expr.left, host, tensor) or _may_be_tensor(
            expr.right, host, tensor
        )
    if isinstance(expr, ast.UnaryOp):
        return _may_be_tensor(expr.operand, host, tensor)
    return _tensor_value(expr, host, tensor)


def _tensor_value(expr: ast.AST, host: Set[str], tensor: Set[str]) -> bool:
    """Is `expr` a tensor? A mask built from other masks (`a & ~b`), a comparison with a side
    that can be a tensor, a torch function that returns a tensor, a reduction, a name that holds
    one. `host` and `tensor` are the names known to hold host facts and tensors."""
    if isinstance(expr, ast.Name):
        return expr.id in tensor
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Invert):
        return not _host_value(expr.operand, host)
    if isinstance(expr, ast.BinOp) and isinstance(
        expr.op, (ast.BitAnd, ast.BitOr, ast.BitXor)
    ):
        return not (_host_value(expr.left, host) and _host_value(expr.right, host))
    if isinstance(expr, ast.IfExp):
        return _tensor_value(expr.body, host, tensor) or _tensor_value(
            expr.orelse, host, tensor
        )
    if isinstance(expr, ast.Compare):
        return any(isinstance(op, _VALUE_COMPARISONS) for op in expr.ops) and any(
            _may_be_tensor(o, host, tensor)
            for o in [expr.left] + list(expr.comparators)
        )
    if isinstance(expr, ast.Call):
        dotted = _dotted(expr.func)
        if dotted.startswith("torch.") and dotted.rsplit(".", 1)[-1] in TENSOR_FUNCS:
            return True
        f = expr.func
        if isinstance(f, ast.Attribute):
            if f.attr in REDUCTIONS and f.attr != "item":
                return True
            return f.attr in TENSOR_METHODS and _tensor_value(f.value, host, tensor)
    return False


def _assigned_values(func: ast.AST) -> Iterable[Tuple[str, ast.AST]]:
    """(name, value) of every assignment of `func` to a single name."""
    for node in ast.walk(func):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            yield node.targets[0].id, node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is not None
        ):
            yield node.target.id, node.value


def _tensor_params(func: ast.AST) -> Set[str]:
    """The parameters that are annotated as tensors (`is_miss: Tensor`): a prefix like `is_` does
    not make them host facts."""
    args = func.args
    return {
        a.arg
        for a in args.posonlyargs + args.args + args.kwonlyargs
        if a.annotation is not None and "Tensor" in ast.unparse(a.annotation)
    }


def _tensor_names(func: ast.AST, host: Set[str]) -> Set[str]:
    """The names of `func` that hold tensors: the parameters annotated as tensors, and the names
    assigned from tensor expressions, however they are called: `is_miss = valid & ~founds` is a
    mask, and a bare `if is_miss:` reads it. Every place that binds a name has to agree (a
    tensor, or None): the analysis does not follow the flow, so a name that is a tensor here and a
    Python int there (`e` of a function that has an empty early return and a loop) is left out.
    """
    params = _tensor_params(func)
    tensor = set(params)
    for _ in range(3):  # a name assigned from another tensor name
        tensor = params | {
            name
            for name, kinds in _bindings(func, host, tensor).items()
            if "tensor" in kinds and kinds <= {"tensor", "none"}
        }
    return tensor


def _bindings(func: ast.AST, host: Set[str], tensor: Set[str]) -> Dict[str, Set[str]]:
    """Every place of `func` that binds a name, by what it binds: 'tensor' (a value that
    _tensor_value knows is one), 'none', or 'other' (anything else, unpacking, loop targets).
    """
    kinds: Dict[str, Set[str]] = {}

    def bind(target: ast.AST, kind: str) -> None:
        # (the `state` of `state.x = ...` is loaded, not bound: only Store names count)
        for sub in ast.walk(target):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                kinds.setdefault(sub.id, set()).add(kind)

    def kind_of(value: ast.AST) -> str:
        if _tensor_value(value, host, tensor):
            return "tensor"
        return (
            "none"
            if isinstance(value, ast.Constant) and value.value is None
            else "other"
        )

    for node in ast.walk(func):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                bind(
                    target,
                    kind_of(node.value) if isinstance(target, ast.Name) else "other",
                )
        elif isinstance(node, ast.AnnAssign):
            bind(node.target, "other" if node.value is None else kind_of(node.value))
        elif isinstance(node, ast.AugAssign):
            masks = (ast.BitAnd, ast.BitOr, ast.BitXor)
            bind(node.target, "tensor" if isinstance(node.op, masks) else "other")
        elif isinstance(node, (ast.For, ast.comprehension)):
            bind(node.target, "other")
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            bind(node.optional_vars, "other")
    return kinds


def _unbound_names(func: ast.AST) -> Set[str]:
    """The parameters of `func`, and every name it loads without binding it: what the caller
    handed in, so that a name like `num_evicted` there is the caller's count (a device tensor
    in sync-free code), where a name that `func` assigned itself is whatever it assigned.
    """
    args = func.args
    params = {a.arg for a in args.posonlyargs + args.args + args.kwonlyargs}
    params.update(a.arg for a in (args.vararg, args.kwarg) if a is not None)
    stored = {
        n.id
        for n in ast.walk(func)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    }
    loaded = {
        n.id
        for n in ast.walk(func)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
    return params | (loaded - stored)


def _is_host_tensor_ctor(call: ast.Call) -> bool:
    """torch.tensor(list), torch.as_tensor(array), ...: a tensor in pageable host memory (a
    `device=` that is not the CPU makes it a device tensor)."""
    if _dotted(call.func) not in HOST_TENSOR_CTORS:
        return False
    for k in call.keywords:
        if k.arg == "device":
            return isinstance(k.value, ast.Constant) and k.value.value == "cpu"
    return True


def _host_tensor_names(func: ast.AST) -> Set[str]:
    """The names of `func` assigned from a tensor built on the host."""
    return {
        name
        for name, value in _assigned_values(func)
        if isinstance(value, ast.Call) and _is_host_tensor_ctor(value)
    }


def _returns_device_count(call: ast.Call) -> bool:
    """Does this call return the device count of evicted keys as the second element of its
    result: async_count=True, collect_evicted=True, or a raw eviction kernel?"""
    return _callee(call) in DEVICE_COUNT_CALLS or any(
        k.arg in DEVICE_COUNT_KEYWORDS
        and isinstance(k.value, ast.Constant)
        and k.value.value is True
        for k in call.keywords
    )


def _async_count_names(func: ast.AST) -> Tuple[Set[str], Set[str]]:
    """(counts, results): the names bound to the device count that a call returning one
    (async_count=True, collect_evicted=True, a raw eviction kernel) returns as the second
    element of its result, and the names bound to the whole result (whose [1] is that count).
    """
    counts: Set[str] = set()
    results: Set[str] = set()
    for node in ast.walk(func):
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
            continue
        if not _returns_device_count(node.value):
            continue
        for target in node.targets:
            if isinstance(target, (ast.Tuple, ast.List)) and len(target.elts) > 1:
                if isinstance(target.elts[1], ast.Name):
                    counts.add(target.elts[1].id)
            elif isinstance(target, ast.Name):
                results.add(target.id)
    return counts, results


class _FunctionLinter(ast.NodeVisitor):
    def __init__(
        self, func: ast.AST, lines: List[str], pragmas: Set[int], standalone: Set[int]
    ):
        self.lines = lines
        self.pragmas = pragmas
        self.standalone = standalone
        self.found: List[Violation] = []
        self.host_names = _host_names(func) | KNOWN_HOST_NAMES
        self.tensor_names = _tensor_names(func, self.host_names) - KNOWN_HOST_NAMES
        self.host_tensors = _host_tensor_names(func)
        self.unbound = _unbound_names(func)
        self.counts, self.results = _async_count_names(func)
        self._stmts: List[ast.AST] = []
        self._flagged: Set[Tuple[int, int, str, str]] = set()
        self.visit(func)

    # -- bookkeeping
    def visit(self, node):  # keep the stack of enclosing statements
        is_stmt = isinstance(node, ast.stmt)
        if is_stmt:
            self._stmts.append(node)
        try:
            super().visit(node)
        finally:
            if is_stmt:
                self._stmts.pop()

    def _suppressed(self, lineno: int) -> bool:
        lines = {lineno}
        if self._stmts:
            lines.add(self._stmts[-1].lineno)
        return any(n in self.pragmas or (n - 1) in self.standalone for n in lines)

    def flag(self, node: ast.AST, rule: str, message: str) -> None:
        key = (node.lineno, node.col_offset, rule, message)
        if key in self._flagged or self._suppressed(node.lineno):
            return
        self._flagged.add(key)
        text = (
            self.lines[node.lineno - 1].strip()
            if 0 < node.lineno <= len(self.lines)
            else ""
        )
        self.found.append(Violation(node.lineno, rule, message, text))

    # -- calls
    def visit_Call(self, node: ast.Call) -> None:
        f = node.func
        name = (
            f.attr
            if isinstance(f, ast.Attribute)
            else (f.id if isinstance(f, ast.Name) else "")
        )
        dotted = _dotted(f)
        kwargs = {k.arg: k.value for k in node.keywords if k.arg}
        if isinstance(f, ast.Attribute) and name in SYNC_METHODS:
            self.flag(node, "sync-call", ".%s() reads or waits for the device" % name)
        if name == "to" and isinstance(f, ast.Attribute):
            targets = list(node.args[:1]) + [
                kwargs[k] for k in ("device",) if k in kwargs
            ]
            if any(isinstance(t, ast.Constant) and t.value == "cpu" for t in targets):
                self.flag(node, "sync-call", '.to("cpu") reads the device')
        if name in HOST_READ_HELPERS:
            self.flag(node, "host-read-helper", "%s() blocks to read a count" % name)
        if name == "pin_memory" and isinstance(f, ast.Attribute):
            self.flag(node, "pin-memory", ".pin_memory() allocates pinned memory")
        pin = kwargs.get("pin_memory")
        if isinstance(pin, ast.Constant) and pin.value is True:
            self.flag(node, "pin-memory", "pin_memory=True allocates pinned memory")
        nb = kwargs.get("non_blocking")
        if isinstance(nb, ast.Constant) and nb.value is False:
            self.flag(node, "blocking-copy", "non_blocking=False blocks the host")
        if dotted in ("torch.tensor", "torch.as_tensor") and "device" in kwargs:
            device = kwargs["device"]
            if not (isinstance(device, ast.Constant) and device.value == "cpu"):
                self.flag(
                    node,
                    "host-tensor-ctor",
                    "%s(..., device=...) is a blocking H2D copy" % dotted,
                )
        if isinstance(f, ast.Attribute) and name == "cuda":
            if self._host_built(f.value) or not (
                isinstance(nb, ast.Constant) and nb.value is True
            ):
                self.flag(
                    node,
                    "host-tensor-ctor",
                    ".cuda() of host data is a blocking pageable H2D copy",
                )
        elif (
            isinstance(f, ast.Attribute)
            and name == "to"
            and self._host_built(f.value)
            and self._names_a_device(node, kwargs)
        ):
            self.flag(
                node,
                "host-tensor-ctor",
                ".to(device) of a tensor built on the host is a blocking pageable H2D copy",
            )
        if isinstance(f, ast.Name) and f.id == "range":
            for arg in node.args:
                self._check_bound(arg, node, "range()")
        if name == "repeat_interleave" and "output_size" not in kwargs:
            self.flag(
                node,
                "repeat-interleave",
                "repeat_interleave without output_size reads the repeats",
            )
        if dotted == "torch.where" and len(node.args) == 1 and not node.keywords:
            self.flag(node, "where-1arg", "torch.where(cond) is nonzero")
        if (
            isinstance(f, ast.Name)
            and f.id in ("int", "bool", "float")
            and len(node.args) == 1
        ):
            arg = node.args[0]
            if isinstance(arg, ast.Call):
                af = arg.func
                reads = (
                    isinstance(af, ast.Attribute) and af.attr in REDUCTIONS
                ) or _dotted(af).startswith("torch.")
                if reads:
                    self.flag(
                        node,
                        "scalar-of-tensor",
                        "%s() of a device reduction reads it" % f.id,
                    )
            if self._is_device_count(arg):
                self.flag(
                    node,
                    "async-count-read",
                    "%s() of %s" % (f.id, self._count_text(arg)),
                )
            elif isinstance(arg, ast.Name) and arg.id in self.tensor_names:
                self.flag(
                    node,
                    "scalar-of-tensor",
                    "%s() of a tensor (%s) reads it" % (f.id, arg.id),
                )
        self.generic_visit(node)

    def _host_built(self, expr: ast.AST) -> bool:
        """A tensor made on the host (pageable memory): torch.tensor(...) or a name assigned from it."""
        if isinstance(expr, ast.Name):
            return expr.id in self.host_tensors
        return isinstance(expr, ast.Call) and _is_host_tensor_ctor(expr)

    @staticmethod
    def _names_a_device(call: ast.Call, kwargs: Dict[str, ast.AST]) -> bool:
        """Is the target of this .to(...) a device (and not a dtype)?"""
        targets = list(call.args[:1]) + [kwargs[k] for k in ("device",) if k in kwargs]
        for t in targets:
            if isinstance(t, ast.Constant) and isinstance(t.value, str):
                if t.value.startswith("cuda"):
                    return True
            elif isinstance(t, ast.Name) and DEVICE_NAME_RE.search(t.id):
                return True
            elif isinstance(t, ast.Attribute) and DEVICE_NAME_RE.search(t.attr):
                return True
            elif isinstance(t, ast.Call) and _dotted(t.func) == "torch.device":
                return True
        return False

    def _check_bound(self, bound: ast.AST, node: ast.AST, what: str) -> None:
        """`what` (range(), a slice) is sized by `bound`: a device count or a tensor reads it."""
        if self._is_device_count(bound):
            self.flag(
                node,
                "async-count-read",
                "%s by %s" % (what, self._count_text(bound)),
            )
        elif isinstance(bound, ast.Name) and bound.id in self.tensor_names:
            self.flag(
                node,
                "scalar-of-tensor",
                "%s by a tensor (%s) reads it" % (what, bound.id),
            )

    # -- indexing
    def _mask_like(self, node: ast.AST) -> bool:
        if isinstance(node, (ast.Compare, ast.BoolOp)):
            return True
        if isinstance(node, ast.UnaryOp):
            return isinstance(node.op, (ast.Not, ast.Invert))
        if isinstance(node, ast.BinOp):
            return isinstance(node.op, (ast.BitAnd, ast.BitOr, ast.BitXor))
        if isinstance(node, (ast.Name, ast.Attribute)):
            name = node.id if isinstance(node, ast.Name) else node.attr
            return bool(MASK_NAME_RE.search(name)) and not INDEX_NAME_RE.search(name)
        if isinstance(node, ast.Call):
            return _dotted(node.func) in (
                "torch.logical_not",
                "torch.logical_and",
                "torch.logical_or",
                "torch.isin",
            )
        if isinstance(node, ast.Tuple):
            return any(self._mask_like(e) for e in node.elts)
        return False

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if self._mask_like(node.slice):
            self.flag(node, "mask-index", "indexing by a boolean mask reads its count")
        if isinstance(node.slice, ast.Slice):
            for bound in (node.slice.lower, node.slice.upper, node.slice.step):
                if bound is not None:
                    self._check_bound(bound, node, "slicing")
        self.generic_visit(node)

    def _check_put(
        self, targets: Iterable[ast.AST], value: Optional[ast.AST], node: ast.AST
    ) -> None:
        if value is None or not _is_number(value):
            return
        for t in targets:
            if isinstance(t, ast.Subscript) and not _is_plain_index(t.slice):
                self.flag(
                    node,
                    "scalar-put",
                    "a Python number through a tensor index is a blocking H2D: index_fill_",
                )

    def visit_Assign(self, node: ast.Assign) -> None:
        self._check_put(node.targets, node.value, node)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._check_put([node.target], node.value, node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._check_put([node.target], node.value, node)
        self.generic_visit(node)

    # -- branch conditions
    def _leaves(self, expr: ast.AST) -> Iterable[ast.AST]:
        if isinstance(expr, ast.BoolOp):
            for v in expr.values:
                yield from self._leaves(v)
        elif isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
            yield from self._leaves(expr.operand)
        else:
            yield expr

    def _host_leaf(self, expr: ast.AST) -> bool:
        if isinstance(expr, ast.Constant):
            return True
        if isinstance(expr, ast.Compare):
            # `x is None` and `k in d` are Python facts; `a > 0` is a tensor when a side is the
            # name of one (a reduction in it is reported by the walk of the test)
            return not any(
                isinstance(op, _VALUE_COMPARISONS) for op in expr.ops
            ) or not any(
                isinstance(o, ast.Name) and o.id in self.tensor_names
                for o in [expr.left] + list(expr.comparators)
            )
        if isinstance(expr, ast.Name):
            if expr.id in self.tensor_names:
                # `is_miss = valid & ~founds`: a mask, whatever it is called
                return False
            return (
                expr.id in self.host_names
                or bool(HOST_NAME_RE.search(expr.id))
                or expr.id.isupper()
            )
        if isinstance(expr, ast.Subscript):
            return isinstance(
                expr.value, ast.Attribute
            )  # `self._busy[i]`: a list kept on an object
        if isinstance(expr, ast.Attribute):
            return not TENSOR_ATTR_RE.search(expr.attr)
        if isinstance(expr, ast.Call):
            return _is_host_call(expr)
        if isinstance(expr, ast.BinOp):
            return not isinstance(expr.op, (ast.BitAnd, ast.BitOr, ast.BitXor))
        return False

    def _is_device_count(self, node: ast.AST) -> bool:
        """The device count of an eviction call (its name, or `result[1]`), or a name that is
        such a count (num_evicted, unique_total, ...) unless the function made a host value of it.
        """
        if isinstance(node, ast.Name):
            if node.id in self.host_names:
                return False
            return node.id in self.counts or (
                node.id in self.unbound and bool(DEVICE_COUNT_NAME_RE.match(node.id))
            )
        return (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id in self.results
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == 1
        )

    def _count_text(self, node: ast.AST) -> str:
        """How a message names the count that `_is_device_count` found."""
        if isinstance(node, ast.Name) and node.id not in self.counts:
            return (
                "`%s`, a count that may be on the device (annotate a host count as `: int`, "
                "or read it with sf.read_now)" % node.id
            )
        return "the device count of an eviction call"

    @staticmethod
    def _reduction(node: ast.AST) -> str:
        """The name of the device reduction that `node` calls (x.sum(), x.any(), torch.max(x), ...),
        '' when it is not one. .item() is a sync-call of its own."""
        if not isinstance(node, ast.Call):
            return ""
        name = _callee(node)
        if name not in REDUCTIONS or name == "item":
            return ""
        f = node.func
        if isinstance(f, ast.Attribute) or _dotted(f).startswith("torch."):
            return name
        return ""  # the builtins any(), all(), max(), min(), sum() of a Python iterable

    @staticmethod
    def _asks_about_the_object(node: ast.AST, parents: Dict[ast.AST, ast.AST]) -> bool:
        """`isinstance(count, T)`, `count.is_cuda`, `count.dtype`: a question about the Python
        object, answered without reading the tensor."""
        parent = parents.get(node)
        if isinstance(parent, ast.Call) and _callee(parent) in OBJECT_QUERIES:
            return node in parent.args
        return (
            isinstance(parent, ast.Attribute)
            and parent.value is node
            and parent.attr in OBJECT_ATTRIBUTES
        )

    def _check_test(self, test: ast.AST) -> None:
        for leaf in self._leaves(test):
            if self._is_device_count(leaf) or self._reduction(leaf):
                continue  # reported below with the rest of the test
            if not self._host_leaf(leaf):
                self.flag(
                    leaf,
                    "branch-on-tensor",
                    "a branch on %s reads it if it is a CUDA tensor"
                    % ast.unparse(leaf),
                )
        parents = {
            child: parent
            for parent in ast.walk(test)
            for child in ast.iter_child_nodes(parent)
        }
        for sub in ast.walk(test):
            if self._is_device_count(sub):
                if not self._asks_about_the_object(sub, parents):
                    self.flag(
                        sub,
                        "async-count-read",
                        "%s in a branch condition" % self._count_text(sub),
                    )
            elif self._reduction(sub):
                self.flag(
                    sub,
                    "reduction-in-branch",
                    ".%s() in a branch condition reads it" % self._reduction(sub),
                )

    def visit_If(self, node: ast.If) -> None:
        self._check_test(node.test)
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self._check_test(node.test)
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self._check_test(node.test)
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        self._check_test(node.test)
        self.generic_visit(node)


def _pragma_lines(source: str) -> Tuple[Set[int], Set[int]]:
    """(lines with a `# sync-ok` comment, the subset of them that is only a comment)."""
    pragmas: Set[int] = set()
    standalone: Set[int] = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT and PRAGMA in tok.string:
                pragmas.add(tok.start[0])
                if tok.line.strip().startswith("#"):
                    standalone.add(tok.start[0])
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return pragmas, standalone


def qualified_functions(tree: ast.AST) -> Dict[str, ast.AST]:
    """Every function of a module by qualified name (``Class.method``, ``function``); the
    functions nested in a function are part of it."""
    out: Dict[str, ast.AST] = {}

    def walk(body: List[ast.stmt], prefix: str) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                walk(node.body, prefix + node.name + ".")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out[prefix + node.name] = node

    walk(tree.body, "")
    return out


def lint_function(source: str, qualname: str) -> Optional[List[Violation]]:
    """The violations of one function of `source`, or None when it has no such function."""
    tree = ast.parse(source)
    func = qualified_functions(tree).get(qualname)
    if func is None:
        return None
    pragmas, standalone = _pragma_lines(source)
    linter = _FunctionLinter(func, source.splitlines(), pragmas, standalone)
    return sorted(linter.found, key=lambda v: (v.lineno, v.rule))


def functions_passing_async_count(source: str) -> List[str]:
    """The functions of `source` with a call that passes async_count=True."""
    out = []
    for qualname, func in qualified_functions(ast.parse(source)).items():
        for node in ast.walk(func):
            if isinstance(node, ast.Call) and any(
                k.arg == "async_count"
                and isinstance(k.value, ast.Constant)
                and k.value.value is True
                for k in node.keywords
            ):
                out.append(qualname)
                break
    return out


# --------------------------------------------------------------------------
# the targets
# --------------------------------------------------------------------------
class Target(NamedTuple):
    file: str  # relative to corelib/dynamicemb
    qualname: str  # 'function', 'Class.method' or 'Class.*' (every method)
    on_main: bool  # does it exist before the sync removal (then it keeps its legacy branches)
    enforce: bool  # a legacy body is reported until its package flips this


BDF = "dynamicemb/batched_dynamicemb_function.py"
KVT = "dynamicemb/key_value_table.py"
SHT = "dynamicemb/scored_hashtable.py"
SF = "dynamicemb/sync_free.py"
ADM = "dynamicemb/embedding_admission.py"
BDT = "dynamicemb/batched_dynamicemb_tables.py"


def _new(file: str, *qualnames: str) -> List[Target]:
    return [Target(file, q, False, True) for q in qualnames]


def _legacy(file: str, *qualnames: str) -> List[Target]:
    return [Target(file, q, True, False) for q in qualnames]


TARGETS: List[Target] = (
    # the lookup paths (masked branches of the existing names, and the functions that are new)
    _new(
        BDF,
        "_prefetch_cache_path_masked",
        "_prefetch_hbm_direct_path_masked",
        "_generic_forward_path_masked",
    )
    + _legacy(
        BDF,
        "dynamicemb_prefetch",
        "dynamicemb_eval_forward",
        "DynamicEmbeddingFunction.forward",
        "DynamicEmbeddingFunction.backward",
    )
    + _new(
        BDF, "OutstandingKeys.add", "OutstandingKeys.release", "OutstandingKeys.check"
    )
    # key_value_table: the masked storage API, the lazy RETAIN_KEY append, the growth estimate
    + _new(
        KVT,
        "_write_back_evicted_masked",
        "_append_evicted_masked",
        "_resolve_pending_evicted",
        "DynamicEmbStorage.lookup_masked",
        "DynamicEmbStorage.load_masked",
        "HybridStorage.lookup_masked",
        "HybridStorage.load_masked",
        "request_size_snapshot",
        "DynamicEmbStorage.request_size_snapshot",
        "HybridStorage.request_size_snapshot",
        "SizeSnapshots.*",
    )
    + _legacy(
        KVT,
        "HybridStorage.insert",
        "_eval_lookup_storage",
        "_eval_lookup_cached",
        "DynamicEmbCache.lookup",
        "DynamicEmbCache.insert_and_evict",
        "expand_if_need_impl",
    )
    # the hash table: the async_count branch of the two evicting inserts
    + _legacy(
        SHT,
        "LinearBucketTable.insert_and_evict",
        "LinearBucketTable.insert_and_evict_with_counter_and_overflow",
    )
    # the helpers every path calls
    + _new(
        SF,
        "mask_keys",
        "canonical_unique",
        "inert_evicted",
        "inert_collected",
        "initialize_masked",
        "compact_by_count",
        "padded_optimizer_table_ids",
        "DeferredRead.enqueue",
        "DeferredRead.ready",
        "DeferredRead.value",
        "DeferredRead.release",
        "MissGate.begin",
        "MissGate.needed",
    )
    + _legacy(
        ADM,
        "MultiTableFrequencyAdmitter.admit",
        "MultiTableProbabilisticAdmitter.admit",
    )
    + _legacy(BDT, "BatchedDynamicEmbeddingTablesV2._check_scores")
    + _new(BDT, "BatchedDynamicEmbeddingTablesV2._table_key_bounds")
)

ENFORCE_ALL = os.environ.get("SYNC_LINT_ENFORCE_ALL") == "1"
REQUIRE_NEW = os.environ.get("SYNC_LINT_REQUIRE_NEW") == "1"
STRICT = os.environ.get("SYNC_LINT_STRICT") == "1"
FINAL = os.environ.get("SYNC_LINT_FINAL") == "1"

# What the targets that exist before the removal hold on the tree before it, per rule. A legacy
# target may keep these (its package removes them, or marks the branches that stay by design with
# `# sync-ok`) and may not gain one: a clean function stays clean, so the rewrite of `forward` or
# `backward` cannot add a read unnoticed, and a function with a legacy branch cannot grow a new
# one. A target that is not listed has none. SYNC_LINT_STRICT=1 also fails a target that is below
# its baseline, so that the baseline goes down with the change that removed the violation; the
# entry of a target that is flipped to `enforce` goes away.
LEGACY_BASELINE: Dict[str, Dict[str, int]] = {
    "dynamicemb/batched_dynamicemb_function.py::dynamicemb_prefetch": {
        "host-read-helper": 1,
    },
    "dynamicemb/key_value_table.py::HybridStorage.insert": {
        "mask-index": 6,
        "reduction-in-branch": 2,
    },
    "dynamicemb/key_value_table.py::_eval_lookup_cached": {
        "host-read-helper": 1,
    },
    "dynamicemb/key_value_table.py::DynamicEmbCache.lookup": {
        "sync-call": 1,
    },
    "dynamicemb/key_value_table.py::expand_if_need_impl": {
        "blocking-copy": 1,
        "pin-memory": 1,
    },
    "dynamicemb/scored_hashtable.py::LinearBucketTable.insert_and_evict": {
        "sync-call": 2,
    },
    "dynamicemb/scored_hashtable.py::LinearBucketTable.insert_and_evict_with_counter_and_overflow": {
        "sync-call": 2,
    },
    "dynamicemb/embedding_admission.py::MultiTableFrequencyAdmitter.admit": {
        "host-read-helper": 1,
    },
    "dynamicemb/batched_dynamicemb_tables.py::BatchedDynamicEmbeddingTablesV2._check_scores": {
        "reduction-in-branch": 1,
        "scalar-of-tensor": 1,
    },
}


def _source(file: str) -> Optional[str]:
    path = os.path.join(ROOT, file)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return f.read()


def expand(targets: Iterable[Target]) -> List[Target]:
    """`Class.*` becomes one target per method of the class that is in the file now."""
    out: List[Target] = []
    for t in targets:
        if not t.qualname.endswith(".*"):
            out.append(t)
            continue
        source = _source(t.file)
        prefix = t.qualname[:-1]
        names = (
            sorted(q for q in qualified_functions(ast.parse(source))) if source else []
        )
        methods = [
            q for q in names if q.startswith(prefix) and "." not in q[len(prefix) :]
        ]
        out.extend(t._replace(qualname=q) for q in methods)
    return out


def report_of(target: Target) -> Optional[List[Violation]]:
    """The violations of the target in this tree, None when the target is not there."""
    source = _source(target.file)
    return None if source is None else lint_function(source, target.qualname)


def rule_counts(violations: Iterable[Violation]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for v in violations:
        counts[v.rule] = counts.get(v.rule, 0) + 1
    return counts


def _summary(violations: List[Violation]) -> str:
    return ", ".join("%s x%d" % kv for kv in sorted(rule_counts(violations).items()))


def _key(target: Target) -> str:
    return "%s::%s" % (target.file, target.qualname)


def legacy_verdict(
    violations: List[Violation], baseline: Dict[str, int], strict: bool = False
) -> Tuple[str, str, List[Violation]]:
    """What to say of a target that exists before the removal: (verdict, why, violations to
    show). 'pass': it is clean. 'xfail': it has violations that its baseline allows (report mode).
    'fail': it has more of a rule than its baseline, or (strict) fewer, which means that the
    baseline was not lowered with the change that removed them."""
    counts = rule_counts(violations)
    grown = sorted(r for r, n in counts.items() if n > baseline.get(r, 0))
    if grown:
        why = (
            "gained a host-sync idiom that the tree before the removal did not have (now > "
            "baseline): %s. Remove it, or mark the line `# sync-ok: <why>` if it stays by design"
            % ", ".join(
                "%s %d > %d" % (r, counts[r], baseline.get(r, 0)) for r in grown
            )
        )
        return "fail", why, [v for v in violations if v.rule in grown]
    lowered = sorted(r for r, n in baseline.items() if counts.get(r, 0) < n)
    if strict and lowered:
        why = (
            "is below its baseline (lower LEGACY_BASELINE with the change that removed the "
            "violation): %s"
            % ", ".join(
                "%s %d < %d" % (r, counts.get(r, 0), baseline[r]) for r in lowered
            )
        )
        return "fail", why, []
    if violations:
        return (
            "xfail",
            "report mode (a legacy body until its package lands): %s"
            % _summary(violations),
            violations,
        )
    return "pass", "", []


def _format(target: Target, violations: List[Violation]) -> str:
    rows = [
        "  %s:%d [%s] %s\n      %s"
        % (target.file, v.lineno, v.rule, v.message, v.source)
        for v in violations
    ]
    return "%s::%s: %d violation(s)\n%s" % (
        target.file,
        target.qualname,
        len(violations),
        "\n".join(rows),
    )


# --------------------------------------------------------------------------
# tests of the engine: what it flags and what it lets through
# --------------------------------------------------------------------------
def _rules(body: str, params: str = "a, mask, *, flag=False, n: int = 0") -> List[str]:
    source = "def f(%s):\n%s" % (params, textwrap.indent(textwrap.dedent(body), "    "))
    found = lint_function(source, "f")
    assert found is not None
    return sorted(v.rule for v in found)


FORBIDDEN = [
    ("x = a.item()", ["sync-call"]),
    ("x = a.cpu()", ["sync-call"]),
    ("x = a.tolist()", ["sync-call"]),
    ("x = a.numpy()", ["sync-call"]),
    ("x = a.nonzero()", ["sync-call"]),
    ("x = torch.nonzero(a)", ["sync-call"]),
    ("x = torch.unique(a)", ["sync-call"]),
    ("x = a.masked_select(mask)", ["sync-call"]),
    ("torch.cuda.synchronize()", ["sync-call"]),
    ("ev.synchronize()", ["sync-call"]),
    ("x = torch.equal(a, a)", ["sync-call"]),
    ('x = a.to("cpu")', ["sync-call"]),
    ('x = a.to(device="cpu")', ["sync-call"]),
    ("x = flagged_compact(mask, [a])", ["host-read-helper"]),
    ("x = _copy_cuda_tensor_to_pinned_cpu(a)", ["host-read-helper"]),
    ("x = a[mask]", ["mask-index"]),
    ("a[mask] = b", ["mask-index"]),
    ("x = a[a > 0]", ["mask-index"]),
    ("x = a[~mask]", ["mask-index"]),
    ("x = a[mask & other]", ["mask-index"]),
    ("x = a[is_miss]", ["mask-index"]),
    ("x = a[:, mask]", ["mask-index"]),
    ("a[idx] = 0", ["scalar-put"]),
    ("a[idx] = -1", ["scalar-put"]),
    ("a[idx] = 0.5", ["scalar-put"]),
    ("a[idx] = True", ["scalar-put"]),
    ("a[idx] += 1", ["scalar-put"]),
    ("x = torch.tensor([1, 2], device=a.device)", ["host-tensor-ctor"]),
    ('x = torch.as_tensor(rows, device="cuda")', ["host-tensor-ctor"]),
    ("x = torch.repeat_interleave(a, b)", ["repeat-interleave"]),
    ("x = a.repeat_interleave(b)", ["repeat-interleave"]),
    ("x = torch.where(mask)", ["where-1arg"]),
    ("x = torch.empty(3, pin_memory=True)", ["pin-memory"]),
    ("x = a.pin_memory()", ["pin-memory"]),
    ("a.copy_(b, non_blocking=False)", ["blocking-copy"]),
    ("x = int(a.sum())", ["scalar-of-tensor"]),
    ("x = bool(a.any())", ["scalar-of-tensor"]),
    ("x = float(a.max())", ["scalar-of-tensor"]),
    ("x = int(torch.sum(a))", ["scalar-of-tensor"]),
    ("x = torch.tensor(rows).cuda()", ["host-tensor-ctor"]),
    ("x = torch.tensor(rows).to(a.device)", ["host-tensor-ctor"]),
    ("x = torch.as_tensor(rows).to(device, non_blocking=True)", ["host-tensor-ctor"]),
    ("t = torch.tensor(rows)\nx = t.to('cuda')", ["host-tensor-ctor"]),
    ("x = a.cuda()", ["host-tensor-ctor"]),
    ("if a.any():\n    pass", ["reduction-in-branch"]),
    ("if not a.all():\n    pass", ["reduction-in-branch"]),
    ("while a.any():\n    pass", ["reduction-in-branch"]),
    ("assert a.all()", ["reduction-in-branch"]),
    # a reduction compared with a number is still a read of the device
    ("if a.sum() > 0:\n    pass", ["reduction-in-branch"]),
    ("if founds.sum() == 0:\n    pass", ["reduction-in-branch"]),
    ("while a.max() > 3:\n    pass", ["reduction-in-branch"]),
    ("x = 1 if a.min() < 0 else 2", ["reduction-in-branch"]),
    ("assert a.sum() > 0", ["reduction-in-branch"]),
    ("if torch.sum(a) > 0:\n    pass", ["reduction-in-branch"]),
    ("if int(a.sum()) > 0:\n    pass", ["reduction-in-branch", "scalar-of-tensor"]),
    # a name assigned from a tensor expression is a tensor, whatever it is called
    ("is_miss = a & ~mask\nif is_miss:\n    pass", ["branch-on-tensor"]),
    ("is_miss = ~a\nif not is_miss:\n    pass", ["branch-on-tensor"]),
    (
        "need_evict = torch.where(mask, a, 0) > 0\nif need_evict:\n    pass",
        ["branch-on-tensor"],
    ),
    ("is_new = a & ~mask\nx = 1 if is_new else 2", ["branch-on-tensor"]),
    ("need_evict = a > 0\nif need_evict:\n    pass", ["branch-on-tensor"]),
    ("is_new = a & ~mask\nif is_new > 0:\n    pass", ["branch-on-tensor"]),
    ("total = a.sum()\nx = a[:total]", ["scalar-of-tensor"]),
    ("total = a.sum()\nfor i in range(total):\n    pass", ["scalar-of-tensor"]),
    ("total = a.sum()\nx = int(total)", ["scalar-of-tensor"]),
    # a count of evicted keys that the caller handed in is on the device
    ("for i in range(num_evicted):\n    pass", ["async-count-read"]),
    ("x = a[:num_evicted]", ["async-count-read"]),
    ("if num_evicted > 0:\n    pass", ["async-count-read"]),
    ("if num_evicted:\n    pass", ["async-count-read"]),
    ("x = int(num_evicted)", ["async-count-read"]),
    ("if mask:\n    pass", ["branch-on-tensor"]),
    ("if not mask:\n    pass", ["branch-on-tensor"]),
    ("if a and flag:\n    pass", ["branch-on-tensor"]),
    ("x = 1 if mask else 2", ["branch-on-tensor"]),
    ("while a:\n    pass", ["branch-on-tensor"]),
    ("if foo(a):\n    pass", ["branch-on-tensor"]),
    ("if state.persisted_mask:\n    pass", ["branch-on-tensor"]),
    ("if a[0]:\n    pass", ["branch-on-tensor"]),
]

ALLOWED = [
    "x = torch.where(mask, a, -1)",
    "x = torch.where(mask, a, b)",
    "a.index_fill_(0, idx, -1)",
    "a.masked_fill_(mask, 0)",
    "x = a[idx]",
    "x = a[idx_rows]",
    "a[idx] = b",
    "a[idx] = a[idx] + 1",
    "x = torch.sort(a, stable=True).indices",
    "pinned.copy_(a, non_blocking=True)",
    "ev.record()\nx = ev.query()",
    "s = a.sum()",
    "x = torch.arange(n, device=a.device)",
    "x = torch.cat([a, a])",
    "x = torch.repeat_interleave(a, b, output_size=n)",
    "x = a.reshape(1).to(torch.int64)",
    "x = a.to(device=a.device, non_blocking=True)",
    "x = a[:n]",
    "x = a[1:]",
    "x = a[:, :3]",
    "x = a[0]",
    "x = a[..., 0]",
    "x = a[None, :]",
    "x = int(len(a))",
    "x = int(n)",
    "x = int(a[0])",
    "x = float(flag)",
    "if flag:\n    x = 1",
    "if n:\n    x = 1",
    "if not flag:\n    x = 1",
    "if mask is None:\n    x = 1",
    "if mask is not None and flag:\n    x = 1",
    "if a.numel() == 0:\n    x = 1",
    "if isinstance(a, int):\n    x = 1",
    "if hasattr(a, 'q') and len(a) > 1:\n    x = 1",
    "need = gate.needed(p)\nif not need:\n    x = 1",
    "present = mask is None\nif present:\n    x = 1",
    "x = 1 if flag else 2",
    "for t in range(n):\n    if t:\n        x = 1",
    "if os.environ.get('A') == '1':\n    x = 1",
    "assert flag",
    "if self.training:\n    x = 1",
    "if state.no_eviction:\n    x = 1",
    "if a > 0:\n    x = 1",
    "x = a[idx_rows[:, 0]]",
    "x = bool(flag)",
    "if self.per_key_score and self.training:\n    x = 1",
    "if optimizer.need_gradient_clipping():\n    x = 1",
    "if is_pooling:\n    x = 1",
    "x = a[refused_pos]",
    "x = a[hit_rows]",
    "if a.numel() > 0:\n    x = 1",
    "if len(a) > 1:\n    x = 1",
    "if a.shape[0] == 0:\n    x = 1",
    "x = a.cuda(non_blocking=True)",
    "x = torch.tensor(rows).to(torch.int64)",
    "x = torch.tensor(rows, dtype=torch.int64)",
    "k = n // 2\nx = a[:k]",
    "k = int(a.shape[0])\nx = a[:k]\nfor i in range(k):\n    y = 1",
    "k = int(sf.read_now(a.sum())[0])\nx = a[:k]\nfor i in range(k):\n    y = 1\nif k > 0:\n    y = 2",
    "n_spilled, n_gone = sf.read_now(a, mask)\nx = a[:n_spilled]\nif n_gone > 0:\n    y = 1",
    "if isinstance(num_evicted, torch.Tensor) and num_evicted.is_cuda:\n    x = 1",
    "is_miss = a & ~mask\nif is_miss is not None:\n    x = 1",
    "is_empty = n == 0\nif is_empty:\n    x = 1",
    "need_evict = n > 0\nif need_evict:\n    x = 1",
    "has_tail = flag != 1\nif has_tail:\n    x = 1",
    "e = torch.empty(0)\ne = n + 1\nx = a[:e]",  # a tensor here and a number there: not decided
    "total = a.sum()\nx = a[:n]",
]


@pytest.mark.parametrize(
    "body, expected", FORBIDDEN, ids=[b.splitlines()[0] for b, _ in FORBIDDEN]
)
def test_the_lint_flags_the_forbidden_idioms(body, expected):
    assert _rules(body) == expected


@pytest.mark.parametrize("body", ALLOWED, ids=[b.splitlines()[0] for b in ALLOWED])
def test_the_lint_lets_the_allowed_idioms_through(body):
    assert _rules(body) == []


def test_a_sync_ok_comment_suppresses_its_statement():
    assert _rules("x = a.item()  # sync-ok: a CPU tensor") == []
    assert (
        _rules("# sync-ok: the external PS reads its keys on the host\nx = a.item()")
        == []
    )
    # on the first line of a statement that spans lines
    assert _rules("x = foo(  # sync-ok: why\n    a.item(),\n)") == []
    # not on an unrelated line, and not past a blank line
    assert _rules("y = 1  # sync-ok\nx = a.item()") == ["sync-call"]
    assert _rules("# sync-ok\n\nx = a.item()") == ["sync-call"]


def test_the_lint_scans_nested_functions_and_comprehensions():
    body = """
    def inner(t):
        return t.item()
    ys = [t.cpu() for t in a]
    """
    assert _rules(body) == ["sync-call", "sync-call"]


def test_an_optional_tensor_parameter_is_not_a_host_flag():
    # `valid=None` may be a tensor: a bare `if valid:` reads it; `is None` is fine
    assert _rules("if valid:\n    x = 1", params="a, valid=None") == [
        "branch-on-tensor"
    ]
    assert _rules(
        "if valid:\n    x = 1", params="a, valid: Optional[torch.Tensor] = None"
    ) == ["branch-on-tensor"]
    assert _rules("if valid is None:\n    x = 1", params="a, valid=None") == []
    assert _rules("if block:\n    x = 1", params="a, block: bool") == []
    # an annotated Python value is a host fact
    assert (
        _rules(
            "x = 1 if strategy else 2",
            params="a, strategy: Optional[EvictStrategy] = None",
        )
        == []
    )


def test_a_name_that_holds_a_tensor_is_not_a_host_flag_whatever_it_is_called():
    # a parameter annotated as a tensor (the `is_miss` of MissGate.begin) ...
    for annotation in ("Tensor", "torch.Tensor", "Optional[torch.Tensor] = None"):
        assert _rules("if is_miss:\n    x = 1", params="is_miss: " + annotation) == [
            "branch-on-tensor"
        ]
    assert (
        _rules(
            "if is_miss is not None:\n    x = 1",
            params="is_miss: Optional[torch.Tensor] = None",
        )
        == []
    )
    # ... and a name assigned from a mask expression, the other places that name is a tensor
    assert _rules("is_miss = valid & ~founds\nif is_miss:\n    x = 1") == [
        "branch-on-tensor"
    ]
    # while the same names are host facts when they are
    for params in ("is_miss: bool", "is_miss=False", "is_miss: int = 0"):
        assert _rules("if is_miss:\n    x = 1", params=params) == []
    assert _rules("is_miss = n == 0\nif is_miss:\n    x = 1") == []


ASYNC_COUNT = """
def f(cache, keys, ids, rows):
    (slots, num_ev, ev_keys, ev_slots, ev_scores, ev_tids) = cache.insert_and_evict(
        keys, ids, None, async_count=True
    )
    {use}
"""


@pytest.mark.parametrize(
    "use",
    [
        "if num_ev:\n        pass",
        "if num_ev != 0:\n        pass",
        "if not num_ev:\n        pass",
        "while num_ev > 0:\n        pass",
        "x = ev_keys[:num_ev]",
        "x = int(num_ev)",
        "x = 1 if num_ev else 2",
    ],
)
def test_the_device_count_of_an_async_count_call_is_not_read(use):
    found = lint_function(ASYNC_COUNT.format(use=use), "f")
    assert found and {v.rule for v in found} == {"async-count-read"}, found


def test_the_whole_result_of_an_async_count_call_is_tracked():
    source = textwrap.dedent(
        """
        def f(cache, keys, ids):
            r = cache.insert_and_evict(keys, ids, async_count=True)
            if r[1]:
                pass
        """
    )
    assert [v.rule for v in lint_function(source, "f")] == ["async-count-read"]


def test_what_an_async_count_call_returns_may_be_used_on_the_device():
    use = "loaded = load(ev_slots, ev_tids)\n    store(slots, loaded)\n    total = num_ev.sum()\n    y = ev_keys[ev_slots]"
    assert lint_function(ASYNC_COUNT.format(use=use), "f") == []


@pytest.mark.parametrize(
    "call",
    [
        "table.insert(keys, ids, score, collect_evicted=True)",
        "table.insert(keys, ids, score, collect_evicted=True, return_slots=True)",
        "table_insert_and_evict(*args)",
        "table_insert_collect_evicted(*args)",
    ],
)
def test_the_device_count_of_every_eviction_call_is_tracked(call):
    """async_count=True is not the only way to a count on the device: insert(collect_evicted=True)
    and the raw kernels return one as the second element of their result too."""
    source = (
        "def f(table, keys, ids, score, args):\n"
        "    (rows, num_ev, ev_keys, ev_tids) = %s\n"
        "    x = ev_keys[:num_ev]\n"
        "    if num_ev > 0:\n"
        "        pass\n" % call
    )
    assert [v.rule for v in lint_function(source, "f")] == ["async-count-read"] * 2


COUNT_MADE_HOST = """
def f(cache, keys, ids, a):
    (slots, num_ev, ev_keys, ev_slots, ev_scores, ev_tids) = cache.insert_and_evict(
        keys, ids, None, async_count=True
    )
    n = int(sf.read_now(num_ev)[0])
    x = ev_keys[:n]
    for i in range(n):
        pass
    if n > 0:
        pass
"""
HOST_COUNT_CALL = """
def f(cache, keys, ids, a):
    (slots, num_evicted, ev_keys) = cache.insert_and_evict(keys, ids)
    h_num_evicted = num_evicted.cpu().item()
    if num_evicted > 0:
        x = ev_keys[:num_evicted]
    y = ev_keys[:h_num_evicted]
"""


def test_a_count_that_the_function_made_a_host_value_may_be_used():
    assert lint_function(COUNT_MADE_HOST, "f") == []


def test_a_count_that_a_host_count_call_returned_is_not_a_device_count():
    """The legacy form of the call returns a host int: it is assigned by the function, and the
    name alone does not make it a tensor. Only the .cpu().item() read itself is reported.
    """
    assert [v.rule for v in lint_function(HOST_COUNT_CALL, "f")] == [
        "sync-call",
        "sync-call",
    ]


def test_the_functions_that_pass_async_count_are_found():
    assert functions_passing_async_count(ASYNC_COUNT.format(use="pass")) == ["f"]
    assert (
        functions_passing_async_count("def g():\n    h(async_count=False)\n    k()")
        == []
    )


# Lines of the tree before the sync removal (bdf, kvt, sht) that the lint has to see.
LEGACY_SNIPPET = """
def legacy(storage, slot_indices, found_mask, unique_table_ids, inserted_mask, new_slots, admitted_tids,
           num_evicted, table_offsets_cpu, num_tables, is_hit, slots, tids, refused_pos):
    found_slots = slot_indices[found_mask]
    storage.increment_counter(found_slots, unique_table_ids[found_mask])
    if inserted_mask.any():
        storage.increment_counter(new_slots[inserted_mask], admitted_tids[inserted_mask])
    slot_indices[refused_pos] = -1
    h_num_evicted = num_evicted.cpu().item()
    total_unique = table_offsets_cpu[num_tables].item()
    num_hit, _, (hit_slots, hit_tids) = flagged_compact(is_hit, [slots, tids])
    reps = torch.repeat_interleave(torch.arange(3), counts)
    key_persisted[positions_in_unique] = True
    assert bool((slots >= 0).all()), "rehash insert must not reject keys"
"""


def test_the_lint_sees_the_syncs_of_the_tree_before_the_removal():
    rules = [v.rule for v in lint_function(LEGACY_SNIPPET, "legacy")]
    assert rules.count("mask-index") == 4  # found_mask x2, inserted_mask x2
    assert (
        rules.count("reduction-in-branch") == 2
    )  # `if inserted_mask.any()` and the assert
    assert rules.count("scalar-put") == 2  # refused_pos, positions_in_unique
    assert rules.count("sync-call") == 3  # .cpu(), .item(), .item()
    assert rules.count("host-read-helper") == 1
    assert rules.count("repeat-interleave") == 1
    assert rules.count("scalar-of-tensor") == 1  # bool((...).all()) is a reduction call
    assert lint_function("def g():\n    pass", "legacy") is None


# --------------------------------------------------------------------------
# the targets in this tree
# --------------------------------------------------------------------------
def test_the_targets_that_exist_before_the_removal_are_spelled_right():
    """A mistyped qualified name would skip a function for good: every target that is
    marked as existing today has to be in the file."""
    missing = [
        "%s::%s" % (t.file, t.qualname)
        for t in TARGETS
        if t.on_main and report_of(t) is None
    ]
    assert not missing, "targets marked on_main that are not in the tree: %s" % missing


@pytest.mark.parametrize(
    "target",
    expand(TARGETS),
    ids=lambda t: "%s::%s" % (os.path.basename(t.file), t.qualname),
)
def test_sync_free_functions_are_free_of_host_syncs(target):
    violations = report_of(target)
    if violations is None:
        if REQUIRE_NEW and not target.on_main:
            pytest.fail(
                "%s::%s is part of the design and is not in the tree"
                % (target.file, target.qualname)
            )
        pytest.skip("%s::%s is not in this tree (yet)" % (target.file, target.qualname))
    if target.enforce or ENFORCE_ALL:
        if violations:
            pytest.fail(_format(target, violations))
        return
    verdict, why, shown = legacy_verdict(
        violations, LEGACY_BASELINE.get(_key(target), {}), STRICT
    )
    if verdict == "fail":
        pytest.fail(
            "%s %s\n%s"
            % (_key(target), why, _format(target, shown) if shown else "(no violation)")
        )
    if verdict == "xfail":
        pytest.xfail(why)


def _violation(rule: str) -> Violation:
    return Violation(1, rule, "message", "source")


def test_a_legacy_target_may_keep_its_violations_but_not_gain_one():
    baseline = {"sync-call": 2}
    assert legacy_verdict([], {})[0] == "pass"
    assert legacy_verdict([_violation("sync-call")] * 2, baseline)[0] == "xfail"
    assert legacy_verdict([_violation("sync-call")], baseline)[0] == "xfail"
    assert legacy_verdict([], baseline)[0] == "pass"
    verdict, why, shown = legacy_verdict([_violation("sync-call")] * 3, baseline)
    assert verdict == "fail" and "sync-call 3 > 2" in why and len(shown) == 3
    # a rule that the baseline does not have, in a function that has others
    verdict, why, shown = legacy_verdict(
        [_violation("sync-call")] * 2 + [_violation("mask-index")], baseline
    )
    assert verdict == "fail" and "mask-index 1 > 0" in why
    assert [v.rule for v in shown] == ["mask-index"]
    # a clean function that gains one: the case of the rewrite of forward / backward
    assert legacy_verdict([_violation("branch-on-tensor")], {})[0] == "fail"


def test_strict_mode_wants_the_baseline_lowered_with_the_change_that_lowers_it():
    baseline = {"sync-call": 2}
    assert legacy_verdict([_violation("sync-call")] * 2, baseline, True)[0] == "xfail"
    for violations in ([_violation("sync-call")], []):
        verdict, why, _ = legacy_verdict(violations, baseline, True)
        assert verdict == "fail" and "lower LEGACY_BASELINE" in why
    assert legacy_verdict([], {}, True)[0] == "pass"


def test_the_legacy_baseline_is_well_formed():
    legacy = {_key(t) for t in TARGETS if t.on_main and not t.enforce}
    for key, rules in LEGACY_BASELINE.items():
        assert key in legacy, (
            "%s is not a legacy target that is in report mode: a target that is flipped to "
            "enforce, or that is new, has no baseline" % key
        )
        assert rules, "%s: an empty baseline is the default, leave the target out" % key
        for rule, n in rules.items():
            assert rule in KNOWN_RULES, (key, rule)
            assert isinstance(n, int) and n > 0, (key, rule, n)


def test_every_rule_of_the_lint_has_a_case():
    reported = {rule for _, expected in FORBIDDEN for rule in expected}
    assert reported == KNOWN_RULES


def test_every_function_that_passes_async_count_leaves_the_count_on_the_device():
    """The extra rule of the lint: a count that comes back from an async_count call is a
    device tensor, and a branch on it (`if num_evicted:`) reads it."""
    failures = []
    package = os.path.join(ROOT, "dynamicemb")
    for dirpath, dirnames, filenames in os.walk(package):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for name in sorted(filenames):
            if not name.endswith(".py"):
                continue
            path = os.path.join(dirpath, name)
            with open(path) as f:
                source = f.read()
            for qualname in functions_passing_async_count(source):
                found = lint_function(source, qualname) or []
                failures += [
                    "%s::%s:%d: %s"
                    % (os.path.relpath(path, ROOT), qualname, v.lineno, v.message)
                    for v in found
                    if v.rule == "async-count-read"
                ]
    assert not failures, "\n".join(failures)


# --------------------------------------------------------------------------
# the regex lint of the census against the tree before the removal
# --------------------------------------------------------------------------
# A line that matches a host-sync pattern and is not in lint_baseline.json is new, and has to be
# listed here with its reason: (file, label or None for every label, why).
ALLOWED_NEW_HITS: List[Tuple[str, Optional[str], str]] = [
    (
        "dynamicemb/sync_free.py",
        None,
        "the audited home of every read that remains: read_now, DeferredRead.wait, pull_to_host",
    ),
    (
        "dynamicemb/batched_dynamicemb_tables.py",
        "any_all_in_cond",
        "_check_scores_now: the check for scores the host holds, or a module without a ring; "
        "the late check of a CUDA score goes through the ring",
    ),
    (
        "dynamicemb/key_value_table.py",
        "blocking_copy",
        "a docstring that names non_blocking=False (the exact growth confirmation), not a call",
    ),
    (
        "dynamicemb/key_value_table.py",
        "flagged_compact",
        "_compact_missing: the one read of the public Storage.find, whose contract returns a "
        "host count; a caller that can mask uses lookup_masked",
    ),
    (
        "dynamicemb/key_value_table.py",
        "tolist",
        "rows_est and unique_size_per_table are CPU tensors (the host's growth estimates): "
        "no device read",
    ),
    (
        "dynamicemb/scored_hashtable.py",
        "item",
        "DEMB_DETERMINISM_MODE reads the count of the evictions it accumulated over its waves: "
        "inherent to that opt-in mode",
    ),
    (
        "dynamicemb/scored_hashtable.py",
        "nonzero",
        "DEMB_DETERMINISM_MODE drops the reserved keys of a batch before it buckets them: "
        "one read in that opt-in mode",
    ),
    (
        "dynamicemb/scored_hashtable.py",
        "repeat_interleave",
        "the call passes output_size on its next line (the lint sees the first line of a "
        "multi-line call)",
    ),
    (
        "dynamicemb/key_value_table.py",
        "pin_memory",
        "construction-time pinned buffers (create_table_state, the size snapshot rings): "
        "allocated once, never in a step; the estimated_row_counts line existed before and "
        "black joined its lines",
    ),
    (
        "dynamicemb/async_erase.py",
        "pin_memory",
        "AsyncEraser.submit: the pinned landing buffer of an erase's counts, allocated on the "
        "worker thread (the host allocator caches it), which makes EraseHandle.result() one "
        "wait for the device",
    ),
]


def _allowed(hit_key: Tuple[str, str, str]) -> bool:
    file, label, _ = hit_key
    return any(
        file == f and (lab is None or lab == label) for f, lab, _ in ALLOWED_NEW_HITS
    )


# A line of lint_baseline.json that a change claims to remove: (file, label, the start of the
# line with its whitespace collapsed, the commit). The change appends its claims in the same diff
# (the way it lowers budgets.json), and the test below checks them: none of these lines may be in
# the tree. The text has to be unique enough that it does not name a line the change keeps.
# SYNC_LINT_FINAL=1 (the last commit of the series) also wants every line that left the baseline
# to be claimed here, so that no removal goes unreviewed.
EXPECTED_GONE: List[Tuple[str, str, str, str]] = [
    # E10: the counts of an erase come back in pinned memory behind the event result() waits
    # for, instead of a second, pageable copy after it
    ("dynamicemb/async_erase.py", "cpu", "erased, spared = counts.cpu()", "E10"),
    ("dynamicemb/async_erase.py", "tolist", "erased, spared = counts.cpu()", "E10"),
]


def _claim_matches(claim: Tuple[str, str, str, str], key: Tuple[str, str, str]) -> bool:
    return (
        key[0] == claim[0]
        and key[1] == claim[1]
        and key[2].startswith(re.sub(r"\s+", " ", claim[2].strip()))
    )


def stale_claims(baseline_hits, claims) -> list:
    """The claims that name no line of the baseline: a typo would make the check vacuous."""
    keys = {sync_lint.norm(h) for h in baseline_hits}
    return [c for c in claims if not any(_claim_matches(c, k) for k in keys)]


def claims_still_present(current_hits, claims) -> list:
    """The claims whose line is still in the tree."""
    keys = {sync_lint.norm(h) for h in current_hits}
    return [c for c in claims if any(_claim_matches(c, k) for k in keys)]


def unclaimed_removals(gone, claims) -> list:
    """The ((file, label, text), count) of `gone` (sync_lint.diff_hits) that no claim names."""
    return [(k, n) for k, n in gone if not any(_claim_matches(c, k) for c in claims)]


@functools.lru_cache(maxsize=None)
def _baseline_hits() -> list:
    with open(os.path.join(HERE, "lint_baseline.json")) as f:
        return json.load(f)


@functools.lru_cache(maxsize=None)
def _current_hits() -> list:
    return sync_lint.scan(ROOT, False)


def _hit(file: str, label: str, text: str) -> dict:
    return {"file": file, "line": 1, "label": label, "text": text}


def test_the_lines_that_a_change_claims_to_remove_are_gone():
    stale = stale_claims(_baseline_hits(), EXPECTED_GONE)
    assert not stale, (
        "EXPECTED_GONE names no line of lint_baseline.json (a typo, or a line that is not in "
        "the baseline): %s" % stale
    )
    present = claims_still_present(_current_hits(), EXPECTED_GONE)
    assert not present, (
        "these lines are claimed to be removed (EXPECTED_GONE) and are still in the tree:\n%s"
        % "\n".join(
            "  %s [%s] %s (%s)" % (f, lab, t[:100], c) for f, lab, t, c in present
        )
    )


def test_every_line_that_left_the_baseline_is_claimed_by_a_change():
    if not FINAL:
        pytest.skip(
            "the check of the last commit of the series (SYNC_LINT_FINAL=1): every line of "
            "lint_baseline.json that is no longer in the tree has a claim in EXPECTED_GONE"
        )
    gone, _new = sync_lint.diff_hits(_baseline_hits(), _current_hits())
    unclaimed = sorted(unclaimed_removals(gone, EXPECTED_GONE))
    assert not unclaimed, (
        "%d line(s) of the baseline are gone and no change claims them. Add them to "
        "EXPECTED_GONE with the commit that removed them:\n%s"
        % (
            len(unclaimed),
            "\n".join(
                '    ("%s", "%s", "%s", "<commit>"),' % (k[0], k[1], k[2][:100])
                for k, _ in unclaimed
            ),
        )
    )


def test_the_removal_claims_are_well_formed():
    labels = {label for label, _ in sync_lint.PY_PATTERNS} | {
        label for label, _ in sync_lint.CPP_PATTERNS
    }
    assert len(set(EXPECTED_GONE)) == len(EXPECTED_GONE), "a claim is listed twice"
    for file, label, text, commit in EXPECTED_GONE:
        assert file.startswith(("dynamicemb/", "src/")), file
        assert label in labels, (file, label)
        assert text.strip(), "a claim names the line it removes: %s" % file
        assert commit.strip(), "a claim names the commit that removes it: %s" % file


def test_the_removal_checks_tell_a_removed_line_from_one_that_is_still_there():
    baseline = [
        _hit("dynamicemb/a.py", "cpu", "n = counts.cpu().item()"),
        _hit("dynamicemb/a.py", "item", "n = counts.cpu().item()"),
        _hit("dynamicemb/b.py", "sync_call", "ev.synchronize()"),
        _hit("dynamicemb/c.py", "cpu", "y  =   x.cpu()"),
    ]
    now = [
        _hit("dynamicemb/b.py", "sync_call", "ev.synchronize()"),
        _hit("dynamicemb/c.py", "cpu", "y  =   x.cpu()"),
    ]
    gone, _ = sync_lint.diff_hits(baseline, now)
    removed = ("dynamicemb/a.py", "cpu", "n = counts.cpu()", "C07")
    # the claim names a baseline line that the tree no longer has; the line next to it, which
    # matches another pattern, is not claimed
    assert stale_claims(baseline, [removed]) == []
    assert claims_still_present(now, [removed]) == []
    assert [k for k, _ in unclaimed_removals(gone, [removed])] == [
        ("dynamicemb/a.py", "item", "n = counts.cpu().item()")
    ]
    # a line that is still there is caught, however its whitespace is spelled
    kept = ("dynamicemb/c.py", "cpu", "y =  x.cpu", "C08")
    assert claims_still_present(now, [kept]) == [kept]
    # a typo, a wrong label and a wrong file name nothing
    for wrong in (
        ("dynamicemb/a.py", "cpu", "m = counts.cpu()", "C07"),
        ("dynamicemb/a.py", "unique", "n = counts.cpu()", "C07"),
        ("dynamicemb/z.py", "cpu", "n = counts.cpu()", "C07"),
    ):
        assert stale_claims(baseline, [wrong]) == [wrong]
        assert claims_still_present(now, [wrong]) == []


def test_no_line_with_a_sync_pattern_is_new_unless_it_is_allowed():
    _gone, new = sync_lint.diff_hits(_baseline_hits(), _current_hits())
    unlisted = sorted((k, n) for k, n in new if not _allowed(k))
    entries = sorted({(k[0], k[1]) for k, _ in unlisted})
    assert not unlisted, (
        "%d line(s) match a host-sync pattern (sync_lint.py) and are not in the baseline of the tree "
        "before the removal. Remove the sync, or add to ALLOWED_NEW_HITS, with the reason it is "
        "inherent (see the list of inherent syncs of the design):\n%s\n\n%s"
        % (
            len(unlisted),
            "\n".join(
                "  %s [%s] x%d: %s" % (k[0], k[1], n, k[2][:120]) for k, n in unlisted
            ),
            "\n".join('    ("%s", "%s", "<why>"),' % e for e in entries),
        )
    )


def test_the_allow_list_is_well_formed():
    labels = {label for label, _ in sync_lint.PY_PATTERNS} | {
        label for label, _ in sync_lint.CPP_PATTERNS
    }
    for file, label, why in ALLOWED_NEW_HITS:
        assert file.startswith(("dynamicemb/", "src/")), file
        assert label is None or label in labels, (file, label)
        assert why.strip(), "an allowed hit needs its reason: %s" % file


def main() -> int:
    """Print the report of every target found in this tree."""
    status = 0
    for target in expand(TARGETS):
        violations = report_of(target)
        if violations is None:
            print("%s::%s: not in this tree" % (target.file, target.qualname))
        elif not violations:
            print("%s::%s: clean" % (target.file, target.qualname))
        elif target.enforce or ENFORCE_ALL:
            print(_format(target, violations))
            status = 1
        else:
            verdict, why, _ = legacy_verdict(
                violations, LEGACY_BASELINE.get(_key(target), {}), STRICT
            )
            print("%s\n  %s: %s" % (_format(target, violations), verdict, why))
            status |= 1 if verdict == "fail" else 0
    return status


if __name__ == "__main__":
    sys.exit(main())
