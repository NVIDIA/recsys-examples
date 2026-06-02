"""Fixed-beam decode runtime wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gr_inference.gr_kernels.attention import GRDecodeAttention, GRDecodeAttentionInputs
from gr_inference.gr_runtime.outputs import DecodeStepOutput
from gr_inference.gr_runtime.request import GRRequestState


@dataclass
class GRDecodeEngine:
    """MVP runtime around the existing GR decode attention kernel."""

    attention: GRDecodeAttention
    fixed_beam_width: int

    def __post_init__(self) -> None:
        if self.fixed_beam_width <= 0:
            raise ValueError("fixed_beam_width must be positive")

    def decode_attention_step(
        self,
        request: GRRequestState,
        q: Any,
        *,
        layer_idx: int,
        step: int,
        active_beam_width: int | None = None,
        topk_indices: Any | None = None,
        decode_nums: int | None = None,
        return_lse: bool = False,
        backend_name: str = "dsl",
    ) -> DecodeStepOutput:
        """Run one attention step for a fixed beam width.

        The model projection, item mask, and topK stages will be layered around
        this method. For the MVP skeleton, this is the integration point for the
        existing decode attention kernel.
        """

        request.validate()
        if active_beam_width is None:
            active_beam_width = self.fixed_beam_width
        if active_beam_width <= 0 or active_beam_width > self.fixed_beam_width:
            raise ValueError("active_beam_width must be in (0, fixed_beam_width]")
        request.beam_kv.validate_step(step, active_beam_width)

        inputs = GRDecodeAttentionInputs(
            q=q,
            context_kv=request.context_kv,
            beam_kv=request.beam_kv,
            beam_path=request.beam_path,
            layer_idx=layer_idx,
            step=step,
            active_beam_width=active_beam_width,
            topk_indices=topk_indices,
            decode_nums=decode_nums,
            return_lse=return_lse,
            backend_name=backend_name,
        )
        attention_output = self.attention(inputs)
        return DecodeStepOutput(
            request_id=request.request_id,
            layer_idx=layer_idx,
            step=step,
            active_beam_width=active_beam_width,
            attention_output=attention_output,
        )
