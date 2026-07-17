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

"""JIT glue for the LruLfu (LRU_LFU strategy) eviction cubins.

- ``ensure_lex_fatbin_loaded()`` hands the default (Lex) evictor fatbin to the
  C++ loader once. The default eviction path needs only this -- no numba, and the
  custom fatbin is not read/required.
- ``register_score_function(fn, score_strategy, cc)`` numba-compiles a user decay function to
  LTO-IR, links it into the custom cubin (nvJitLink, C++ side), and returns an
  integer key used to route inserts to that custom evictor. Cached per function.
"""
import ast
import hashlib
import inspect
import os
import textwrap
import threading

import dynamicemb_extensions as _ext

_lock = threading.Lock()
_lex_loaded = False
# score_function group key (int) -> True once registered with the C++ cache.
_registered_keys = set()

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_LEX_FATBIN = os.path.join(_PKG_DIR, "evict_lrulfu_lex.fatbin")
_CUSTOM_FATBIN = os.path.join(_PKG_DIR, "evict_lrulfu_custom.fatbin")


def _read_fatbin(path: str) -> bytes:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"LruLfu evict fatbin missing: {path}. Rebuild the extension "
            "(python setup.py build_ext --inplace) so the fatbins are produced "
            "as package_data."
        )
    with open(path, "rb") as fh:
        return fh.read()


def ensure_lex_fatbin_loaded() -> None:
    """Load the default (Lex) LruLfu evict fatbin into the C++ side (idempotent).

    This is all the default eviction path needs -- no numba, and the custom
    fatbin is not read/required unless a score_function is registered."""
    global _lex_loaded
    if _lex_loaded:
        return
    with _lock:
        if _lex_loaded:
            return
        _ext.demb_set_lex_fatbin(_read_fatbin(_LEX_FATBIN))
        _lex_loaded = True


def _remap_score_function(fn, perm):
    """Return a function equivalent to *fn* but with every ``scores[c]`` (c a
    logical index) rewritten to ``scores[perm[c]]`` (physical word), so a
    user-written function (indexing in the tuple/logical order) reads the correct
    device words.

    ``scores`` is the function's first parameter. Only integer-constant subscripts
    are allowed (the static remap requires it) and must be in ``[0, len(perm))``;
    anything else raises. The check runs even when *perm* is identity."""
    src = textwrap.dedent(inspect.getsource(fn))
    tree = ast.parse(src)
    func = tree.body[0]
    if not isinstance(func, ast.FunctionDef) or not func.args.args:
        raise TypeError("score_function must be a plain function taking "
                        "(scores, cur_timestamp).")
    func.decorator_list = []  # don't re-apply decorators when we recompile
    scores_name = func.args.args[0].arg
    n = len(perm)

    class _Remap(ast.NodeTransformer):
        def visit_Subscript(self, node):
            self.generic_visit(node)
            if isinstance(node.value, ast.Name) and node.value.id == scores_name:
                idx = node.slice
                if isinstance(idx, ast.Index):  # Python < 3.9
                    idx = idx.value
                if not (isinstance(idx, ast.Constant) and isinstance(idx.value, int)):
                    raise ValueError(
                        f"score_function must index `{scores_name}` with integer "
                        f"constants (required for the logical->physical remap)."
                    )
                c = idx.value
                if not (0 <= c < n):
                    raise IndexError(
                        f"score_function subscript {scores_name}[{c}] is out of "
                        f"range [0, {n})."
                    )
                node.slice = ast.copy_location(ast.Constant(perm[c]), node.slice)
            return node

    _Remap().visit(tree)
    ast.fix_missing_locations(tree)
    code = compile(tree, filename=f"<score_function_remap:{func.name}>", mode="exec")
    ns = dict(getattr(fn, "__globals__", {}))
    exec(code, ns)
    return ns[func.name]


def score_function_key(fn, perm) -> int:
    """Stable non-zero int key for a (score_function, physical permutation).

    Derived from (module, qualname, source hash, perm) so the same function under
    different logical score orders gets DIFFERENT keys/cubins (the remapped source
    differs). Truncated to a positive int64."""
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        src = repr(fn)
    ident = (
        f"{getattr(fn, '__module__', '')}.{getattr(fn, '__qualname__', '')}"
        f":{tuple(perm)}:{src}"
    )
    digest = hashlib.sha1(ident.encode("utf-8")).hexdigest()
    key = int(digest[:15], 16)  # 60 bits, always positive, nonzero in practice
    return key or 1


def register_score_function(
    fn, score_strategy, cc_major: int, cc_minor: int
) -> int:
    """numba-compile fn -> LTO-IR, link into the custom cubin, cache under its
    key. Returns the key to pass as score_fn_key on inserts. Idempotent.

    *score_strategy* is the table's (logical) score strategy tuple; the user's
    score_function is written against that order, so its ``scores[c]`` subscripts
    are remapped to physical device order before compiling. The custom fatbin is
    read only here (the default path never touches it)."""
    ensure_lex_fatbin_loaded()
    from dynamicemb.dynamicemb_config import score_dump_permutation

    # perm[j] = physical word holding logical column j.
    perm = score_dump_permutation(score_strategy)
    key = score_function_key(fn, perm)
    if key in _registered_keys:
        return key
    with _lock:
        if key in _registered_keys:
            return key
        # Validate (constant, in-bounds subscripts) and remap logical->physical.
        remapped = _remap_score_function(fn, perm)
        try:
            from numba import cuda, types
        except ImportError as e:
            raise ImportError(
                "score_function requires numba-cuda to JIT-compile the eviction "
                "decay. Install numba-cuda, or omit score_function to use the "
                "default (frequency, older-timestamp tiebreak) evictor."
            ) from e
        # After remap, scores is indexed in PHYSICAL order (word 0 = timestamp,
        # word 1 = frequency). Signature is (scores, cur_timestamp) -> float64; any
        # decay constant is baked into the function body. cc MUST be the device's
        # (numba's default sm_50 is rejected by recent CUDA NVVM).
        ltoir, _ = cuda.compile(
            remapped,
            sig=(types.CPointer(types.uint64), types.uint64),
            device=True,
            output="ltoir",
            abi="c",
            abi_info={"abi_name": "user_score_fn"},
            cc=(cc_major, cc_minor),
        )
        _ext.demb_register_score_function(
            key, bytes(ltoir), _read_fatbin(_CUSTOM_FATBIN), cc_major, cc_minor
        )
        _registered_keys.add(key)
    return key
