import time

from gr_inference.gr_serving import (
    GRContinuousBatchingPolicy,
    GRContinuousScheduler,
    GRInProcessServingFacade,
    GRServingRequest,
    GRServingWorker,
)


class FakeInputIds:
    def __init__(self, shape):
        self.shape = shape


def make_request(idx: int, *, decode_steps: int = 1) -> GRServingRequest:
    return GRServingRequest(
        request_id=f"req-{idx}",
        input_ids=FakeInputIds((1, 8)),
        max_decode_steps=decode_steps,
        beam_width=2,
    )


def wait_for_result(worker: GRServingWorker, request_id: str, *, timeout_s: float = 1.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        response = worker.poll(request_id)
        if response is not None:
            return response
        time.sleep(0.005)
    raise AssertionError(f"request {request_id} did not finish")


def test_serving_worker_auto_ticks_until_request_finishes() -> None:
    worker = GRServingWorker(
        GRInProcessServingFacade(
            GRContinuousScheduler(
                policy=GRContinuousBatchingPolicy(
                    max_prefill_batch_size=1,
                    max_decode_batch_size=1,
                )
            )
        ),
        tick_interval_s=0.001,
        idle_sleep_s=0.001,
    )

    worker.start()
    try:
        worker.submit(make_request(0, decode_steps=2))
        response = wait_for_result(worker, "req-0")

        assert response.metadata["stop_reason"] == "max_decode_steps"
        assert worker.status()["worker"]["running"] is True
        assert worker.metrics()["worker_ticks"] >= 2
    finally:
        worker.stop()

    assert worker.running is False

def test_serving_worker_is_facade_compatible_for_manual_tick() -> None:
    worker = GRServingWorker(GRInProcessServingFacade(GRContinuousScheduler()))

    request_id = worker.submit(make_request(0))
    assert worker.status()["pending_submissions"] == 1
    tick = worker.tick()

    assert request_id == "req-0"
    assert tick.finished_request_ids == ("req-0",)
    assert worker.require_result("req-0").request_id == "req-0"
    assert worker.metrics()["worker_running"] == 0
    assert worker.metrics()["worker_pending_submissions"] == 0


def test_serving_worker_stop_is_idempotent() -> None:
    worker = GRServingWorker(GRInProcessServingFacade(GRContinuousScheduler()))

    worker.start()
    worker.stop()
    worker.stop()

    assert worker.running is False
