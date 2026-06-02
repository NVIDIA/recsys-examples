# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from gr_inference.gr_serving import (
    FIFOBatchAssembler,
    GRRequestQueue,
    GRServingRequest,
    SchedulerPolicy,
)


def make_request(idx: int) -> GRServingRequest:
    return GRServingRequest(
        request_id=f"req-{idx}",
        input_ids=object(),
        max_decode_steps=1,
        beam_width=1,
    )


def test_fifo_batch_assembler_respects_max_batch_size() -> None:
    queue = GRRequestQueue()
    for idx in range(3):
        queue.submit(make_request(idx))
    assembler = FIFOBatchAssembler(SchedulerPolicy(max_batch_size=2))

    batch = assembler.assemble(queue)

    assert batch.size == 2
    assert [request.request_id for request in batch.requests] == ["req-0", "req-1"]
    assert len(queue) == 1
    assert batch.metadata()["size"] == 2


def test_scheduler_policy_validates_max_batch_size() -> None:
    try:
        SchedulerPolicy(max_batch_size=0)
    except ValueError as exc:
        assert "max_batch_size" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_batch_compatibility_checks_shape_and_decode_config() -> None:
    queue = GRRequestQueue()
    queue.submit(make_request(0))
    queue.submit(make_request(1))
    batch = FIFOBatchAssembler(SchedulerPolicy(max_batch_size=2)).assemble(queue)

    assert batch.compatible_for_tensor_batch()
