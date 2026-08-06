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
- ``register_score_function(fn, score_strategy, cc_major, cc_minor)`` numba-compiles
  a user decay function to LTO-IR, links it into the custom cubin (nvJitLink, C++
  side), and returns an integer key used to route inserts to that custom evictor.
  Cached per function.
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


def score_function_key(fn, perm, cc_major: int, cc_minor: int) -> int:
    """Stable non-zero int key for a (score_function, physical permutation, device
    compute capability).

    Derived from (module, qualname, source hash, perm, cc). Including the perm
    means the same function under different logical score orders gets a DIFFERENT
    key/cubin (the remapped source differs). Including cc is what makes
    heterogeneous multi-GPU correct: a different-arch device gets a different key,
    so it numba-compiles and registers its OWN LTO-IR (for its cc) instead of
    reusing the first-registered arch's -- otherwise nvJitLink would link the wrong
    arch and cuModuleLoadData would fail on the mismatched device. Same-arch devices
    share the key (one compile, module cached per device). Positive int64."""
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        src = repr(fn)
    ident = (
        f"{getattr(fn, '__module__', '')}.{getattr(fn, '__qualname__', '')}"
        f":{tuple(perm)}:sm{cc_major}{cc_minor}:{src}"
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

    # perm[j] = physical word holding logical column j. The key includes the
    # device cc so each arch registers its own (arch-specific) numba LTO-IR.
    perm = score_dump_permutation(score_strategy)
    key = score_function_key(fn, perm, cc_major, cc_minor)
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
        # word 1 = frequency). The return type is PINNED to float64 to match the
        # C++ `extern "C" __device__ double user_score_fn(...)` ABI: without it
        # numba infers the return type from the body, so a function that returns
        # an integer expression (e.g. `return -scores[1]`, no float() cast) would
        # compile to an integer-register return that the caller reads as junk
        # double bits -- a silent miscompile. Pinning float64 makes numba cast the
        # integer result instead. Any decay constant is baked into the body; cc
        # MUST be the device's (numba's default sm_50 is rejected by recent NVVM).
        ltoir, _ = cuda.compile(
            remapped,
            sig=types.float64(types.CPointer(types.uint64), types.uint64),
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
