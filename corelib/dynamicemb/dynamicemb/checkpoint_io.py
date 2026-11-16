# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint discovery and decoding shared by training and CPU/CUDA inference."""

import bisect
import fnmatch
import json
import os
import re
from functools import partial
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor

from .key_ownership import owned_key_mask
from .types import (
    EMBEDDING_TYPE,
    KEY_TYPE,
    OPT_STATE_TYPE,
    SCORE_TYPE,
    torch_dtype_to_np_dtype,
)

META_EMBEDDING_DTYPE = "embedding_dtype"
META_EMBEDDING_DIM = "embedding_dim"
META_OPT_STATE_DTYPE = "optim_state_dtype"


def encode_meta_json_file_path(root_path: str, table_name: str) -> str:
    return os.path.join(root_path, f"{table_name}_opt_args.json")


def encode_checkpoint_file_path(
    root_path: str, table_name: str, rank: int, world_size: int, item: str
) -> str:
    assert item in ["keys", "values", "scores", "opt_values"]
    return os.path.join(
        root_path, f"{table_name}_emb_{item}.rank_{rank}.world_size_{world_size}"
    )


def encode_counter_checkpoint_file_path(
    root_path: str, table_name: str, rank: int, world_size: int, item: str
) -> str:
    assert item in ["keys", "frequencies"]
    return os.path.join(
        root_path, f"{table_name}_counter_{item}.rank_{rank}.world_size_{world_size}"
    )


def list_checkpoint_files(root_path: str, table_names: Iterable[str]) -> List[str]:
    """Sorted paths, relative to *root_path*, that find_files looks tables up in.

    A table named "<scope>/<name>" keeps its files in the subdirectory *scope*
    of *root_path*, so each distinct table directory is listed once, together
    with *root_path* itself.
    """
    file_names = os.listdir(root_path)
    for table_dir in sorted({os.path.dirname(name) for name in table_names} - {""}):
        dir_path = os.path.join(root_path, table_dir)
        if os.path.isdir(dir_path):
            file_names.extend(
                os.path.join(table_dir, name) for name in os.listdir(dir_path)
            )
    return sorted(file_names)


def find_files(
    root_path: str, table_name: str, suffix: str, sorted_file_names: Sequence[str]
) -> Tuple[List[str], int]:
    """Find one table's checkpoint files of *suffix* among *sorted_file_names*.

    *sorted_file_names* is the listing from list_checkpoint_files, taken once by
    the caller: a checkpoint directory holds every table's files, so globbing it
    per table and per suffix lists it 6N times for N tables, which is slow where
    a listing is expensive, such as object-storage mounts. Being sorted, the
    names that can match sit together, so a lookup does not scan the whole
    listing. Names are relative to *root_path*, so a table named
    "<scope>/<name>" is matched against its subdirectory's entries.
    """
    suffix_to_encode_file_path_func = {
        "emb_keys": partial(encode_checkpoint_file_path, item="keys"),
        "emb_values": partial(encode_checkpoint_file_path, item="values"),
        "emb_scores": partial(encode_checkpoint_file_path, item="scores"),
        "opt_values": partial(encode_checkpoint_file_path, item="opt_values"),
        "counter_keys": partial(encode_counter_checkpoint_file_path, item="keys"),
        "counter_frequencies": partial(
            encode_counter_checkpoint_file_path, item="frequencies"
        ),
    }
    if suffix not in suffix_to_encode_file_path_func:
        raise RuntimeError(f"Invalid suffix: {suffix}")
    encode_file_path_func = suffix_to_encode_file_path_func[suffix]

    # v2 version
    pattern = encode_file_path_func("", table_name, "*", "*")
    prefix = re.split(r"[*?[]", pattern, maxsplit=1)[0]
    start = end = bisect.bisect_left(sorted_file_names, prefix)
    while end < len(sorted_file_names) and sorted_file_names[end].startswith(prefix):
        end += 1
    files = [
        os.path.join(root_path, name)
        for name in fnmatch.filter(sorted_file_names[start:end], pattern)
    ]
    if len(files) == 0:
        return [], 0
    files = sorted(files)
    world_size = int(files[0].split(".")[-1].split("_")[-1])
    if len(files) != world_size:
        raise RuntimeError(
            f"Checkpoints is corrupted. Found {len(files)} under path {root_path} for table {table_name}, but the number of checkpointed world size is {world_size}."
        )

    for i in range(world_size):
        expected_file_path = encode_file_path_func(root_path, table_name, i, world_size)
        if expected_file_path not in set(files):
            raise RuntimeError(
                f"Checkpoints is corrupted. Expected file path {expected_file_path} for table {table_name}, but it is not found."
            )

    return files, len(files)


def get_loading_files(
    root_path: str,
    name: str,
    rank: int,
    world_size: int,
    sorted_file_names: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[str], List[str], List[str], int, int]:
    """Resolve one table's checkpoint files under *root_path*.

    Pass *sorted_file_names*, from list_checkpoint_files, when resolving many
    tables, so the directory is listed once rather than once per table.
    """
    if not os.path.exists(root_path):
        raise RuntimeError(f"can't find path to load, path:", root_path)
    if sorted_file_names is None:
        sorted_file_names = list_checkpoint_files(root_path, [name])

    key_files, num_key_files = find_files(
        root_path, name, "emb_keys", sorted_file_names
    )
    value_files, num_value_files = find_files(
        root_path, name, "emb_values", sorted_file_names
    )
    score_files, num_score_files = find_files(
        root_path, name, "emb_scores", sorted_file_names
    )
    opt_files, num_opt_files = find_files(
        root_path, name, "opt_values", sorted_file_names
    )

    if num_key_files != num_value_files:
        assert (
            num_key_files > 0
        ), "No key files found under path {root_path} for table {name}"
        raise RuntimeError(
            f"The number of key files under path {root_path} for table {name} does not match the number of value files."
        )

    counter_key_files, num_counter_key_files = find_files(
        root_path, name, "counter_keys", sorted_file_names
    )
    counter_freq_files, num_counter_freq_files = find_files(
        root_path, name, "counter_frequencies", sorted_file_names
    )

    if num_counter_key_files != num_counter_freq_files:
        raise RuntimeError(
            f"The number of key files of admission counter under path {root_path} for table {name} does not match the number of frequency files({num_counter_key_files}/{num_counter_freq_files})."
        )

    if num_counter_key_files > 0 and num_counter_key_files != num_key_files:
        raise RuntimeError(
            f"The number of key files under path {root_path} for table {name} does not match the number of keys files of admission counter({num_key_files}/{num_counter_key_files})."
        )

    if world_size == num_key_files:
        return (
            [encode_checkpoint_file_path(root_path, name, rank, world_size, "keys")],
            [encode_checkpoint_file_path(root_path, name, rank, world_size, "values")],
            [encode_checkpoint_file_path(root_path, name, rank, world_size, "scores")]
            if num_score_files == num_key_files
            else [],
            [
                encode_checkpoint_file_path(
                    root_path, name, rank, world_size, "opt_values"
                )
            ]
            if num_opt_files == num_key_files
            else [],
            [
                encode_counter_checkpoint_file_path(
                    root_path, name, rank, world_size, "keys"
                )
            ]
            if num_counter_key_files == num_key_files
            else [],
            [
                encode_counter_checkpoint_file_path(
                    root_path, name, rank, world_size, "frequencies"
                )
            ]
            if num_counter_freq_files == num_key_files
            else [],
        )
    # TODO: support skipping files.
    return (
        key_files,
        value_files,
        score_files,
        opt_files,
        counter_key_files,
        counter_freq_files,
    )


def load_from_json(file_path: str) -> Dict[str, Any]:
    try:
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
        return data
    except Exception as e:
        raise RuntimeError(f"Error loading data from JSON file: {e}")


def dtype_to_meta_name(dtype: torch.dtype) -> str:
    """``torch.bfloat16`` -> ``"bfloat16"``, the spelling checkpoint meta uses."""
    return str(dtype).split(".", 1)[-1]


def dtype_from_meta_name(name: Any, field_name: str) -> torch.dtype:
    """Inverse of :func:`dtype_to_meta_name`, rejecting anything not a dtype.

    Resolves against ``torch`` itself rather than a hand-kept table so any dtype
    a table can be built with round-trips, and a corrupt or hand-edited meta
    fails here instead of silently mis-striding the value file.
    """
    dtype = getattr(torch, name, None) if isinstance(name, str) else None
    if not isinstance(dtype, torch.dtype):
        raise ValueError(
            f"Checkpoint meta field {field_name!r} is {name!r}, which is not a "
            "torch dtype."
        )
    return dtype


def resolve_checkpoint_value_layout(
    meta_data: Dict[str, Any],
    embedding_file_path: str,
    num_keys: int,
    runtime_dim: int,
) -> Tuple[torch.dtype, torch.dtype, int]:
    """Decide how to read a value file: (embedding dtype, opt-state dtype, dim).

    Prefers what the dump recorded. For a checkpoint written before the meta
    carried a dtype, the historical answer is fp32 for both value files, and the
    row width is recovered from the value file's length -- exact, given the dtype
    and the key count. An empty shard is the one case the length cannot answer,
    so it defers to the runtime dim.

    Either way the result is cross-checked against the file size, which turns a
    wrong dtype assumption (or a truncated file) into an error here rather than
    a silently misparsed table.

    Raises:
        ValueError: the value file's size disagrees with the resolved layout, or
            a meta dtype field is not a torch dtype.
    """
    emb_dtype = (
        dtype_from_meta_name(meta_data[META_EMBEDDING_DTYPE], META_EMBEDDING_DTYPE)
        if META_EMBEDDING_DTYPE in meta_data
        else EMBEDDING_TYPE
    )
    opt_state_dtype = (
        dtype_from_meta_name(meta_data[META_OPT_STATE_DTYPE], META_OPT_STATE_DTYPE)
        if META_OPT_STATE_DTYPE in meta_data
        else OPT_STATE_TYPE
    )

    emb_bytes = os.path.getsize(embedding_file_path)
    ckpt_dim = meta_data.get(META_EMBEDDING_DIM, None)
    if ckpt_dim is None:
        if num_keys == 0:
            ckpt_dim = runtime_dim
        else:
            key_block = num_keys * emb_dtype.itemsize
            if emb_bytes % key_block != 0:
                raise ValueError(
                    f"Embedding file {embedding_file_path} size {emb_bytes} is not "
                    f"divisible by {key_block} (num_keys={num_keys}, dtype "
                    f"{emb_dtype}), so its row width cannot be recovered. This "
                    "checkpoint predates "
                    f"{META_EMBEDDING_DIM!r}/{META_EMBEDDING_DTYPE!r} in the meta "
                    "JSON, so it is assumed to be fp32."
                )
            ckpt_dim = emb_bytes // key_block
    ckpt_dim = int(ckpt_dim)

    expected_bytes = num_keys * ckpt_dim * emb_dtype.itemsize
    if emb_bytes != expected_bytes:
        raise ValueError(
            f"Embedding file {embedding_file_path} holds {emb_bytes} bytes but "
            f"{expected_bytes} were expected for {num_keys} keys of {ckpt_dim} "
            f"{emb_dtype} elements."
        )
    return emb_dtype, opt_state_dtype, ckpt_dim


def _tensor_from_bytes(
    buf: bytes, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Decode *buf* as a 1-D tensor of *dtype* on *device*.

    ``torch.frombuffer`` rather than ``np.frombuffer`` for the same bfloat16
    reason as :func:`_raw_bytes`. The ``bytearray`` copy makes the buffer writable
    so torch does not warn about aliasing read-only memory; the ``to(device)``
    would copy regardless.
    """
    if len(buf) == 0:
        return torch.empty(0, dtype=dtype, device=device)
    return torch.frombuffer(bytearray(buf), dtype=dtype).to(device)


def _iter_batches_from_files(
    emb_key_path: str,
    embedding_file_path: str,
    score_file_path: Optional[str],
    opt_file_path: Optional[str],
    dim: int,
    optstate_dim: int,
    device: torch.device,
    dist_type: str,
    batch_size: int = 65536,
    num_scores: int = 1,
    emb_dtype: torch.dtype = EMBEDDING_TYPE,
    opt_state_dtype: torch.dtype = OPT_STATE_TYPE,
    shard_index: Optional[int] = None,
    num_shards: Optional[int] = None,
) -> Iterator[Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]]:
    """Yield (keys, embeddings, scores, opt_states) batches from checkpoint files.

    *shard_index* / *num_shards* are the table's own shard geometry (its position
    among the ranks that hold it, and how many there are): the world for a
    row-wise table, one node for a table-row-wise one. Left ``None`` they fall
    back to the global rank and world size, which is the row-wise answer.

    ``num_scores`` is the number of score words per key in the score file. When
    > 1 (e.g. LruLfu's timestamp+frequency) the yielded ``scores`` is
    [n, num_scores]; otherwise it is [n].

    *emb_dtype* / *opt_state_dtype* are the precisions the files were written at,
    which is how far apart the batches step through them -- get them wrong and
    the files misparse rather than fail. Callers holding the table's meta should
    pass what :func:`resolve_checkpoint_value_layout` resolved; the fp32 defaults
    are what every checkpoint written before the meta recorded a dtype used.
    The yielded tensors keep the file's dtype; the insert path casts to the
    table's.

    Handles file I/O, deserialization, and keeping only the keys this rank owns.

    *dist_type* selects the ownership rule, which is the table's own -- the load
    path checks it against the checkpoint's before getting here
    (``_validate_load_meta``). It matters in two ways. When the checkpoint's
    shard count matches the world size each rank reads only its own file, every
    key in it already belongs to that rank, and the filter is a no-op -- but
    only if the rule is the right one. When the counts differ the files are
    re-split across ranks and the filter is what does the resharding.
    Pass *score_file_path* / *opt_file_path* as ``None`` to skip those files.
    """
    fkey = open(emb_key_path, "rb")
    fembedding = open(embedding_file_path, "rb")
    fscore = (
        open(score_file_path, "rb")
        if score_file_path and os.path.exists(score_file_path)
        else None
    )
    fopt = open(opt_file_path, "rb") if opt_file_path else None
    num_keys = os.path.getsize(emb_key_path) // KEY_TYPE.itemsize

    if num_shards is None or shard_index is None:
        num_shards = dist.get_world_size() if dist.is_initialized() else 1
        shard_index = dist.get_rank() if dist.is_initialized() else 0

    try:
        for start in range(0, num_keys, batch_size):
            n = min(num_keys - start, batch_size)

            keys_bytes = fkey.read(KEY_TYPE.itemsize * n)
            keys = torch.tensor(
                np.frombuffer(keys_bytes, dtype=torch_dtype_to_np_dtype[KEY_TYPE]),
                dtype=KEY_TYPE,
                device=device,
            )

            emb_bytes = fembedding.read(emb_dtype.itemsize * dim * n)
            embeddings = _tensor_from_bytes(emb_bytes, emb_dtype, device).view(-1, dim)

            scores = None
            if fscore:
                score_bytes = fscore.read(SCORE_TYPE.itemsize * n * num_scores)
                scores = torch.tensor(
                    np.frombuffer(
                        score_bytes, dtype=torch_dtype_to_np_dtype[SCORE_TYPE]
                    ),
                    dtype=SCORE_TYPE,
                    device=device,
                )
                if num_scores > 1:
                    scores = scores.view(-1, num_scores)

            opt_states = None
            if fopt:
                opt_bytes = fopt.read(opt_state_dtype.itemsize * optstate_dim * n)
                opt_states = _tensor_from_bytes(
                    opt_bytes, opt_state_dtype, device
                ).view(-1, optstate_dim)

            masks = owned_key_mask(keys, shard_index, num_shards, dist_type)
            if masks is not None:
                keys = keys[masks]
                embeddings = embeddings[masks]
                if scores is not None:
                    scores = scores[masks]
                if opt_states is not None:
                    opt_states = opt_states[masks]

            yield keys, embeddings, scores, opt_states
    finally:
        fkey.close()
        fembedding.close()
        if fscore:
            fscore.close()
        if fopt:
            fopt.close()
