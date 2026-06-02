from gr_inference.gr_runtime import TimingRecorder


def test_timing_recorder_records_sections() -> None:
    recorder = TimingRecorder()

    with recorder.section("foo"):
        pass
    with recorder.section("foo"):
        pass

    summary = recorder.summary()

    assert "foo" in summary
    assert summary["foo"]["count"] == 2
    assert summary["foo"]["total_ms"] >= 0.0
    assert summary["foo"]["avg_ms"] >= 0.0


def test_timing_recorder_stores_detail_level() -> None:
    recorder = TimingRecorder(detail="fine")

    assert recorder.detail == "fine"


def test_timing_recorder_can_emit_nvtx_ranges() -> None:
    class FakeNvtx:
        def __init__(self) -> None:
            self.events = []

        def range_push(self, name: str) -> None:
            self.events.append(("push", name))

        def range_pop(self) -> None:
            self.events.append(("pop", None))

    class FakeCuda:
        def __init__(self) -> None:
            self.nvtx = FakeNvtx()

        def is_available(self) -> bool:
            return False

    class FakeModule:
        def __init__(self) -> None:
            self.cuda = FakeCuda()

    module = FakeModule()
    recorder = TimingRecorder(sync_module=module, emit_nvtx=True)

    with recorder.section("decode.step"):
        pass

    assert module.cuda.nvtx.events == [("push", "decode.step"), ("pop", None)]


def test_timing_recorder_can_skip_cuda_sync() -> None:
    class FakeCuda:
        def __init__(self) -> None:
            self.sync_count = 0

        def is_available(self) -> bool:
            return True

        def synchronize(self) -> None:
            self.sync_count += 1

    class FakeModule:
        def __init__(self) -> None:
            self.cuda = FakeCuda()

    module = FakeModule()
    recorder = TimingRecorder(sync_module=module, sync_timing=False)

    with recorder.section("decode.step"):
        pass

    assert module.cuda.sync_count == 0
