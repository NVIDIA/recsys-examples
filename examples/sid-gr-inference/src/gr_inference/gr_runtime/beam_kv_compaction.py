# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BeamKV history compaction for dynamic beam widths."""

from __future__ import annotations

from typing import Any

from gr_inference.gr_kv import BatchedBeamPath, BeamKV


def needs_batched_beam_kv_history_compaction(
    path: BatchedBeamPath,
    *,
    decode_nums: int,
    active_beam_width: int,
) -> bool:
    """Return true when current beams reference history outside active width."""

    _validate_compaction_args(path, decode_nums, active_beam_width)
    for beam_path in path.paths:
        for query_beam in range(active_beam_width):
            ancestry = _beam_ancestry(
                beam_path,
                query_beam=query_beam,
                decode_nums=decode_nums,
            )
            if any(beam >= active_beam_width for beam in ancestry[:-1]):
                return True
    return False


def compact_batched_beam_kv_history(
    beam_kv: BeamKV,
    path: BatchedBeamPath,
    *,
    decode_nums: int,
    active_beam_width: int,
) -> BeamKV:
    """Gather each current beam's ancestors into BeamKV prefix slots.

    The external decode attention kernel flattens the sliced BeamKV history with
    ``decode_step * active_beam_width + beam``. After dynamic beam shrink, an
    ancestor can live outside ``active_beam_width`` in an earlier step. This
    helper creates a layout-compatible BeamKV where prior-step ancestors for
    current query beam ``j`` are copied into slot ``j``.
    """

    _validate_compaction_args(path, decode_nums, active_beam_width)
    if beam_kv.batch_size != path.batch_size:
        raise ValueError("BeamKV batch size must match BatchedBeamPath")
    if decode_nums > beam_kv.max_decode_steps:
        raise ValueError("decode_nums exceeds BeamKV decode capacity")
    if active_beam_width > beam_kv.max_beam_width:
        raise ValueError("active_beam_width exceeds BeamKV beam capacity")
    if not hasattr(beam_kv.key, "new_empty") or not hasattr(beam_kv.value, "new_empty"):
        raise TypeError("BeamKV compaction requires torch-like tensors")

    compact_key = beam_kv.key.new_empty(beam_kv.key_shape)
    compact_value = beam_kv.value.new_empty(beam_kv.value_shape)
    history_steps = decode_nums - 1
    for batch_idx, beam_path in enumerate(path.paths):
        for query_beam in range(active_beam_width):
            ancestry = _beam_ancestry(
                beam_path,
                query_beam=query_beam,
                decode_nums=decode_nums,
            )
            for step, ancestor_beam in enumerate(ancestry[:history_steps]):
                compact_key[:, batch_idx, step, query_beam] = beam_kv.key[
                    :,
                    batch_idx,
                    step,
                    ancestor_beam,
                ]
                compact_value[:, batch_idx, step, query_beam] = beam_kv.value[
                    :,
                    batch_idx,
                    step,
                    ancestor_beam,
                ]
    return BeamKV(compact_key, compact_value)


def _validate_compaction_args(
    path: BatchedBeamPath,
    decode_nums: int,
    active_beam_width: int,
) -> None:
    if decode_nums <= 0:
        raise ValueError("decode_nums must be positive")
    if active_beam_width <= 0:
        raise ValueError("active_beam_width must be positive")
    if path.steps_done < decode_nums:
        raise ValueError("BatchedBeamPath does not contain enough decode steps")


def _beam_ancestry(
    beam_path: Any,
    *,
    query_beam: int,
    decode_nums: int,
) -> tuple[int, ...]:
    if query_beam < 0 or query_beam >= beam_path.entries[decode_nums - 1].width:
        raise ValueError("query_beam outside active beam width")

    ancestry = [0 for _ in range(decode_nums)]
    current = query_beam
    for step in range(decode_nums - 1, -1, -1):
        entry = beam_path.entries[step]
        if current < 0 or current >= entry.width:
            raise ValueError("beam ancestry points outside active width")
        ancestry[step] = current
        current = entry.parent_beams[current]
    return tuple(ancestry)
