"""Batch-aware beam path tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from gr_inference.gr_kv.beam_path import BeamPath

if TYPE_CHECKING:
    from gr_inference.gr_runtime.batched_beam_search import BatchedBeamSelection


@dataclass
class BatchedBeamPath:
    """Collection of request-local BeamPath objects for a decode batch."""

    paths: tuple[BeamPath, ...]

    @classmethod
    def create(
        cls,
        *,
        batch_size: int,
        max_decode_steps: int,
        max_beam_width: int,
    ) -> "BatchedBeamPath":
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return cls(
            tuple(
                BeamPath(
                    max_decode_steps=max_decode_steps,
                    max_beam_width=max_beam_width,
                )
                for _ in range(batch_size)
            )
        )

    @property
    def batch_size(self) -> int:
        return len(self.paths)

    @property
    def steps_done(self) -> int:
        if not self.paths:
            return 0
        return self.paths[0].steps_done

    def active_widths(self) -> tuple[int, ...]:
        return tuple(path.active_beam_width for path in self.paths)

    def append(self, selection: "BatchedBeamSelection") -> None:
        if selection.batch_size != self.batch_size:
            raise ValueError("selection batch_size must match BatchedBeamPath")
        for path, parents, tokens, scores in zip(
            self.paths,
            selection.parent_beams,
            selection.token_ids,
            selection.scores,
        ):
            path.append(
                parent_beams=parents,
                token_ids=tokens,
                scores=scores,
            )

    def token_trace(
        self,
        *,
        batch_idx: int,
        beam: int,
        step: int | None = None,
    ) -> tuple[int, ...]:
        return self.paths[batch_idx].token_trace(beam=beam, step=step)


@dataclass
class BatchedBeamPathBuilder:
    """Incrementally build a BatchedBeamPath."""

    batch_size: int
    max_decode_steps: int
    max_beam_width: int
    path: BatchedBeamPath = field(init=False)

    def __post_init__(self) -> None:
        self.path = BatchedBeamPath.create(
            batch_size=self.batch_size,
            max_decode_steps=self.max_decode_steps,
            max_beam_width=self.max_beam_width,
        )

    def append(self, selection: "BatchedBeamSelection") -> BatchedBeamPath:
        self.path.append(selection)
        return self.path

