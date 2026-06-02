"""Request-level logits processor hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import inf
from typing import Any, Literal, Mapping, Sequence

LogitsProcessorPhase = Literal["prefill", "decode"]


@dataclass(frozen=True)
class LogitsProcessorContext:
    """Context passed to request-level logits processors."""

    request_id: str
    phase: LogitsProcessorPhase
    step: int
    beam_width: int
    beam_path: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TokenSuppressLogitsProcessor:
    """Set selected token logits to a low value before beam selection."""

    token_ids: tuple[int, ...]
    fill_value: float = -inf
    phases: tuple[LogitsProcessorPhase, ...] = ("prefill", "decode")

    def __init__(
        self,
        token_ids: Sequence[int],
        *,
        fill_value: float = -inf,
        phases: Sequence[LogitsProcessorPhase] = ("prefill", "decode"),
    ) -> None:
        object.__setattr__(self, "token_ids", _normalize_token_ids(token_ids))
        object.__setattr__(self, "fill_value", float(fill_value))
        object.__setattr__(self, "phases", _normalize_phases(phases))

    def process_logits(self, logits: Any, context: LogitsProcessorContext) -> Any:
        if context.phase not in self.phases or not self.token_ids:
            return logits
        processed = _clone_logits(logits)
        processed[..., list(self.token_ids)] = self.fill_value
        return processed

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "token_suppress",
            "token_ids": self.token_ids,
            "fill_value": self.fill_value,
            "phases": self.phases,
        }


@dataclass(frozen=True)
class TokenBiasLogitsProcessor:
    """Add fixed token biases before beam selection."""

    token_bias: tuple[tuple[int, float], ...]
    phases: tuple[LogitsProcessorPhase, ...] = ("prefill", "decode")

    def __init__(
        self,
        token_bias: Mapping[int | str, float] | Sequence[tuple[int, float]],
        *,
        phases: Sequence[LogitsProcessorPhase] = ("prefill", "decode"),
    ) -> None:
        object.__setattr__(self, "token_bias", _normalize_token_bias(token_bias))
        object.__setattr__(self, "phases", _normalize_phases(phases))

    def process_logits(self, logits: Any, context: LogitsProcessorContext) -> Any:
        if context.phase not in self.phases or not self.token_bias:
            return logits
        processed = _clone_logits(logits)
        for token_id, bias in self.token_bias:
            processed[..., token_id] = processed[..., token_id] + bias
        return processed

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "token_bias",
            "token_bias": dict(self.token_bias),
            "phases": self.phases,
        }


def apply_logits_processors(
    logits: Any,
    processors: tuple[Any, ...],
    context: LogitsProcessorContext,
) -> Any:
    """Apply a processor chain, returning the logits used for beam selection."""

    processed = logits
    for processor in processors:
        process = getattr(processor, "process_logits", None)
        if process is None:
            process = processor
        processed = process(processed, context)
        if processed is None:
            raise ValueError("logits processor must return logits")
    return processed


def validate_logits_processors(processors: tuple[Any, ...]) -> None:
    for processor in processors:
        process = getattr(processor, "process_logits", None)
        if process is None:
            process = processor
        if not callable(process):
            raise ValueError(
                "logits_processors must be callable or define process_logits"
            )


def logits_processors_metadata(
    processors: tuple[Any, ...]
) -> tuple[dict[str, Any], ...]:
    """Return debug metadata for response payloads."""

    rows: list[dict[str, Any]] = []
    for processor in processors:
        metadata = getattr(processor, "metadata", None)
        if callable(metadata):
            rows.append(dict(metadata()))
        else:
            rows.append({"type": type(processor).__name__})
    return tuple(rows)


def logits_processor_from_spec(spec: Mapping[str, Any]) -> Any:
    """Build a serializable built-in processor from an HTTP/request spec."""

    processor_type = spec.get("type")
    phases = spec.get("phases", ("prefill", "decode"))
    if processor_type in {"token_suppress", "suppress_tokens", "bad_tokens"}:
        token_ids = spec.get("token_ids", spec.get("suppressed_token_ids"))
        if token_ids is None:
            raise ValueError("token_suppress logits processor requires token_ids")
        return TokenSuppressLogitsProcessor(
            token_ids,
            fill_value=float(spec.get("fill_value", -inf)),
            phases=phases,
        )
    if processor_type in {"token_bias", "bias_tokens"}:
        token_bias = spec.get("token_bias", spec.get("biases"))
        if token_bias is None:
            raise ValueError("token_bias logits processor requires token_bias")
        return TokenBiasLogitsProcessor(token_bias, phases=phases)
    raise ValueError(f"unsupported logits processor type: {processor_type!r}")


def logits_processors_from_specs(specs: Any) -> tuple[Any, ...]:
    if specs is None:
        return ()
    if not isinstance(specs, Sequence) or isinstance(specs, (str, bytes, bytearray)):
        raise ValueError("logits_processors must be a list")
    processors = []
    for spec in specs:
        if not isinstance(spec, Mapping):
            raise ValueError("logits processor spec must be a JSON object")
        processors.append(logits_processor_from_spec(spec))
    return tuple(processors)


def _clone_logits(logits: Any) -> Any:
    if hasattr(logits, "clone"):
        return logits.clone()
    raise TypeError(f"logits processor requires cloneable logits, got {type(logits)!r}")


def _normalize_token_ids(token_ids: Sequence[int]) -> tuple[int, ...]:
    if isinstance(token_ids, (str, bytes, bytearray)):
        raise ValueError("token_ids must be a sequence of integers")
    normalized = tuple(int(token_id) for token_id in token_ids)
    if any(token_id < 0 for token_id in normalized):
        raise ValueError("token_ids must be non-negative")
    return tuple(dict.fromkeys(normalized))


def _normalize_token_bias(
    token_bias: Mapping[int | str, float] | Sequence[tuple[int, float]],
) -> tuple[tuple[int, float], ...]:
    if isinstance(token_bias, Mapping):
        items = token_bias.items()
    elif isinstance(token_bias, Sequence) and not isinstance(
        token_bias, (str, bytes, bytearray)
    ):
        items = token_bias
    else:
        raise ValueError("token_bias must be an object or list of pairs")

    normalized: list[tuple[int, float]] = []
    for raw_token, raw_bias in items:
        token_id = int(raw_token)
        if token_id < 0:
            raise ValueError("token_bias token ids must be non-negative")
        normalized.append((token_id, float(raw_bias)))
    return tuple(dict(normalized).items())


def _normalize_phases(
    phases: Sequence[LogitsProcessorPhase],
) -> tuple[LogitsProcessorPhase, ...]:
    if isinstance(phases, (str, bytes, bytearray)):
        phases = (phases,)  # type: ignore[assignment]
    normalized = tuple(phases)
    invalid = [phase for phase in normalized if phase not in {"prefill", "decode"}]
    if invalid:
        raise ValueError(f"invalid logits processor phases: {invalid!r}")
    if not normalized:
        raise ValueError("logits processor phases must be non-empty")
    return normalized
