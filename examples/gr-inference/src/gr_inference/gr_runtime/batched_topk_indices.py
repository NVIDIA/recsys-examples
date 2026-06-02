"""Build batched topK indices for GR decode attention."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from gr_inference.gr_kv import BatchedBeamPath


def make_batched_topk_indices(
    path: BatchedBeamPath,
    *,
    num_q_heads: int,
    decode_nums: int,
    beam_width: int,
    device: Any | None = None,
):
    """Build kernel topK indices shaped [B, 1, Hq, decode_nums, W].

    The external GR decode kernel indexes flattened BeamKV history as
    ``decode_step * beam_width + beam``. For each current query beam, this helper
    follows the parent chain in ``BatchedBeamPath`` to select the matching beam
    slot at each decode history step.
    """

    if num_q_heads <= 0:
        raise ValueError("num_q_heads must be positive")
    if decode_nums <= 0:
        raise ValueError("decode_nums must be positive")
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    if path.steps_done < decode_nums:
        raise ValueError("BatchedBeamPath does not contain enough decode steps")

    import torch

    if decode_nums == 1:
        return _make_initial_topk_indices(
            torch,
            batch_size=path.batch_size,
            num_q_heads=num_q_heads,
            beam_width=beam_width,
            device=device,
        )

    base_indices: list[int] = []
    for beam_path in path.paths:
        batch_values = [0 for _ in range(decode_nums * beam_width)]
        for query_beam in range(beam_width):
            beam_at_step = _beam_ancestry(
                beam_path,
                query_beam=query_beam,
                decode_nums=decode_nums,
            )
            for decode_idx, beam in enumerate(beam_at_step):
                batch_values[decode_idx * beam_width + query_beam] = (
                    decode_idx * beam_width + beam
                )
        base_indices.extend(batch_values)

    pattern = torch.tensor(
        base_indices,
        dtype=torch.int32,
        device=device,
    ).view(path.batch_size, decode_nums, beam_width)
    return pattern.view(path.batch_size, 1, 1, decode_nums, beam_width).expand(
        path.batch_size,
        1,
        num_q_heads,
        decode_nums,
        beam_width,
    ).contiguous()


def make_compacted_batched_topk_indices(
    *,
    batch_size: int,
    num_q_heads: int,
    decode_nums: int,
    beam_width: int,
    device: Any | None = None,
):
    """Build topK indices for compacted BeamKV history.

    In compacted history, every query beam owns the same slot id at every decode
    history step, so the kernel lookup is simply ``step * W + query_beam``.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_q_heads <= 0:
        raise ValueError("num_q_heads must be positive")
    if decode_nums <= 0:
        raise ValueError("decode_nums must be positive")
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")

    import torch

    step_offsets = torch.arange(decode_nums, dtype=torch.int32, device=device)
    step_offsets = step_offsets.view(decode_nums, 1) * beam_width
    beam_offsets = torch.arange(beam_width, dtype=torch.int32, device=device).view(
        1,
        beam_width,
    )
    pattern = step_offsets + beam_offsets
    return pattern.view(1, 1, 1, decode_nums, beam_width).expand(
        batch_size,
        1,
        num_q_heads,
        decode_nums,
        beam_width,
    ).contiguous()


def _make_initial_topk_indices(
    torch,
    *,
    batch_size: int,
    num_q_heads: int,
    beam_width: int,
    device: Any | None = None,
):
    pattern = _cached_initial_pattern(
        beam_width=beam_width,
        device_type=_device_type(device),
        device_index=_device_index(device),
    )
    return pattern.view(1, 1, 1, 1, beam_width).expand(
        batch_size,
        1,
        num_q_heads,
        1,
        beam_width,
    ).contiguous()


@lru_cache(maxsize=128)
def _cached_initial_pattern(
    *,
    beam_width: int,
    device_type: str,
    device_index: int | None,
):
    import torch

    device = torch.device(device_type, device_index)
    return torch.arange(beam_width, dtype=torch.int32, device=device)


def _device_type(device: Any | None) -> str:
    if device is None:
        return "cpu"
    return getattr(device, "type", str(device).split(":", 1)[0])


def _device_index(device: Any | None) -> int | None:
    index = getattr(device, "index", None)
    if index is not None:
        return index
    if isinstance(device, str) and ":" in device:
        return int(device.split(":", 1)[1])
    return None


def _beam_ancestry(
    beam_path,
    *,
    query_beam: int,
    decode_nums: int,
) -> tuple[int, ...]:
    """Return the beam id used by one query beam at each decode history step."""

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
