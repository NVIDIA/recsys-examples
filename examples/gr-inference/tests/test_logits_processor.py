import importlib.util

import pytest
from gr_inference.gr_runtime import (
    LogitsProcessorContext,
    TokenBiasLogitsProcessor,
    TokenSuppressLogitsProcessor,
    apply_logits_processors,
    logits_processors_from_specs,
    logits_processors_metadata,
)


def test_logits_processor_specs_build_builtin_processors() -> None:
    processors = logits_processors_from_specs(
        [
            {
                "type": "token_suppress",
                "token_ids": [1, 2, 2],
                "phases": ["prefill"],
            },
            {
                "type": "token_bias",
                "token_bias": {"3": 1.5, "4": -2.0},
            },
        ]
    )

    assert isinstance(processors[0], TokenSuppressLogitsProcessor)
    assert isinstance(processors[1], TokenBiasLogitsProcessor)
    assert logits_processors_metadata(processors) == (
        {
            "type": "token_suppress",
            "token_ids": (1, 2),
            "fill_value": float("-inf"),
            "phases": ("prefill",),
        },
        {
            "type": "token_bias",
            "token_bias": {3: 1.5, 4: -2.0},
            "phases": ("prefill", "decode"),
        },
    )


def test_logits_processor_specs_reject_invalid_payloads() -> None:
    with pytest.raises(ValueError, match="logits_processors must be a list"):
        logits_processors_from_specs({"type": "token_bias"})

    with pytest.raises(ValueError, match="unsupported logits processor"):
        logits_processors_from_specs([{"type": "temperature"}])

    with pytest.raises(ValueError, match="invalid logits processor phases"):
        logits_processors_from_specs(
            [{"type": "token_suppress", "token_ids": [1], "phases": ["sample"]}]
        )


def test_builtin_logits_processors_apply_to_tensor_logits() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch

    logits = torch.zeros((1, 2, 5), dtype=torch.float32)
    processed = apply_logits_processors(
        logits,
        (
            TokenBiasLogitsProcessor({3: 4.0}),
            TokenSuppressLogitsProcessor([1], phases=("decode",)),
        ),
        LogitsProcessorContext(
            request_id="req",
            phase="decode",
            step=0,
            beam_width=2,
        ),
    )

    assert torch.all(logits == 0)
    assert torch.all(processed[..., 3] == 4.0)
    assert torch.all(torch.isneginf(processed[..., 1]))
