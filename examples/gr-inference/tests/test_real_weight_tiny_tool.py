import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest


def torch_or_skip():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch

    return torch


def real_serving_args(**overrides) -> Namespace:
    values = {
        "decode_backend": "real",
        "device": "cuda",
        "decode_steps": 1,
        "batched_decode": True,
        "requests": 1,
        "max_batch_size": 1,
    }
    values.update(overrides)
    return Namespace(**values)


def sweep_args(**overrides) -> Namespace:
    values = {
        "model_dir": "/models/qwen",
        "decode_backend": "real",
        "device": "cuda",
        "beam_score_mode": "logprob",
        "warmup_runs": 1,
        "repeat": 1,
        "continuous": True,
        "arrival_stagger_ticks": 0,
        "arrival_burst_size": 1,
        "profile_continuous_decode": False,
        "profile_detail": "coarse",
        "beam_kv_pool_capacity": 0,
        "scheduled_fractions": "1.0,0.5,0.25",
    }
    values.update(overrides)
    return Namespace(**values)


def http_request_factory(default_decode_steps=1, default_beam_width=2):
    torch = torch_or_skip()
    from serve_qwen3_gr_http import make_torch_request_factory

    return make_torch_request_factory(
        torch,
        device="cpu",
        default_decode_steps=default_decode_steps,
        default_beam_width=default_beam_width,
    )


def _http_result(status: int, body: dict):
    from soak_http_serving import HTTPResult

    return HTTPResult(status, body)


class FakeSoakClient:
    def __init__(
        self,
        *,
        overload_suffixes: tuple[str, ...] = (),
        failed_request_id: str | None = None,
    ) -> None:
        self.overload_suffixes = overload_suffixes
        self.failed_request_id = failed_request_id
        self.submitted = []
        self.polls = {}

    def request(self, method, path, payload=None):
        if method == "GET" and path in {"/health", "/config", "/status", "/metrics"}:
            return _http_result(200, {"ok": True})
        if method == "GET" and path == "/build":
            return _http_result(200, {"build": {"framework": "gr-inference"}})
        if method == "GET" and path == "/kv/events":
            return _http_result(200, {"events": [{"type": "allocate"}], "count": 1})
        if method == "GET" and path == "/ready":
            return _http_result(
                200, {"ready": True, "reasons": [], "admission": {"queue_full": False}}
            )
        if method == "POST" and path == "/submit_many":
            request_ids = [row["request_id"] for row in payload["requests"]]
            if any(
                request_id.endswith(self.overload_suffixes)
                for request_id in request_ids
            ):
                return _http_result(
                    429, {"error": {"code": "overloaded", "message": "full"}}
                )
            self.submitted.extend(request_ids)
            return _http_result(202, {"request_ids": request_ids})
        if method == "POST" and path == "/submit":
            self.submitted.append(payload["request_id"])
            return _http_result(202, {"request_id": payload["request_id"]})
        if method == "POST" and path == "/cancel":
            return _http_result(
                200,
                {
                    "request_id": payload["request_id"],
                    "token_ids": [],
                    "scores": [],
                    "metadata": {"cancelled": True},
                },
            )
        if method == "GET" and path.startswith("/poll/"):
            request_id = path.rsplit("/", 1)[-1]
            self.polls[request_id] = self.polls.get(request_id, 0) + 1
            metadata = {"stop_reason": "max_decode_steps"}
            if request_id == self.failed_request_id:
                metadata = {
                    "failed": True,
                    "stop_reason": "decode_failed",
                    "error_type": "RuntimeError",
                    "error_message": "shape mismatch",
                }
            return _http_result(
                200,
                {
                    "ready": True,
                    "response": {
                        "request_id": request_id,
                        "token_ids": [],
                        "scores": [],
                        "metadata": metadata,
                    },
                },
            )
        raise AssertionError((method, path, payload))


def test_build_identity_topk_indices() -> None:
    torch = torch_or_skip()
    from run_qwen3_real_weight_tiny_gr import build_identity_topk_indices

    indices = build_identity_topk_indices(
        torch,
        batch=1,
        head_q=2,
        decode_nums=3,
        beam_width=4,
        device="cpu",
    )

    assert tuple(indices.shape) == (1, 1, 2, 3, 4)
    assert indices[0, 0, 0, 0].tolist() == [0, 1, 2, 3]
    assert indices[0, 0, 0, 1].tolist() == [4, 5, 6, 7]
    assert indices[0, 0, 1, 2].tolist() == [8, 9, 10, 11]


def test_benchmark_time_call_runs() -> None:
    torch = torch_or_skip()
    from bench_qwen3_real_weight import time_call

    counter = {"value": 0}

    def fn():
        counter["value"] += 1

    elapsed = time_call(torch, fn, warmup=1, iters=2)

    assert elapsed >= 0.0
    assert counter["value"] == 3


def test_aggregate_profile_groups_sections() -> None:
    from bench_qwen3_real_weight import aggregate_profile

    aggregate = aggregate_profile(
        {
            "layer0.qkv": {"total_ms": 1.0},
            "layer1.qkv": {"total_ms": 2.0},
            "layer0.input_norm": {"total_ms": 0.1},
            "layer0.qkv_proj": {"total_ms": 0.2},
            "layer0.qk_norm_rope": {"total_ms": 0.3},
            "layer0.q_norm": {"total_ms": 0.11},
            "layer0.k_norm": {"total_ms": 0.12},
            "layer0.rope": {"total_ms": 0.13},
            "layer0.decode_attention": {"total_ms": 3.0},
            "layer0.post_attention": {"total_ms": 4.0},
            "layer0.o_proj": {"total_ms": 0.4},
            "layer0.post_norm": {"total_ms": 0.5},
            "layer0.mlp": {"total_ms": 0.6},
            "layer0.gate_up_proj": {"total_ms": 0.21},
            "layer0.silu_mul": {"total_ms": 0.22},
            "layer0.down_proj": {"total_ms": 0.23},
            "layer0.beam_kv_write": {"total_ms": 5.0},
            "layer0.decode_total": {"total_ms": 6.0},
            "model.forward_decode_step": {"total_ms": 7.0},
            "flashinfer.calls.rmsnorm": {"total_ms": 4.0},
            "flashinfer.calls.rope": {"total_ms": 5.0},
        }
    )

    assert aggregate["qkv_ms"] == 3.0
    assert aggregate["input_norm_ms"] == 0.1
    assert aggregate["qkv_proj_ms"] == 0.2
    assert aggregate["qk_norm_rope_ms"] == 0.3
    assert aggregate["q_norm_ms"] == 0.11
    assert aggregate["k_norm_ms"] == 0.12
    assert aggregate["rope_ms"] == 0.13
    assert aggregate["decode_attention_ms"] == 3.0
    assert aggregate["post_attention_mlp_ms"] == 4.0
    assert aggregate["o_proj_ms"] == 0.4
    assert aggregate["post_norm_ms"] == 0.5
    assert aggregate["mlp_ms"] == 0.6
    assert aggregate["gate_up_proj_ms"] == 0.21
    assert aggregate["silu_mul_ms"] == 0.22
    assert aggregate["down_proj_ms"] == 0.23
    assert aggregate["beam_kv_write_ms"] == 5.0
    assert aggregate["layer_total_ms"] == 6.0
    assert aggregate["model_forward_decode_step_ms"] == 7.0
    assert aggregate["flashinfer_rmsnorm_calls"] == 4.0
    assert aggregate["flashinfer_rope_calls"] == 5.0


def test_real_weight_serving_fake_backend() -> None:
    from run_qwen3_real_weight_serving import make_decode_backend

    backend = make_decode_backend(
        real_serving_args(
            decode_backend="fake",
            device="cpu",
            batched_decode=False,
        ),
        device="cpu",
    )

    class Inputs:
        q = "q"

    assert backend(Inputs()) == "q"


def test_real_weight_serving_real_backend_requires_batched_decode() -> None:
    from run_qwen3_real_weight_serving import make_decode_backend

    with pytest.raises(RuntimeError, match="requires --batched-decode"):
        make_decode_backend(
            real_serving_args(
                batched_decode=False,
            ),
            device="cuda",
        )


def test_real_weight_serving_real_backend_allows_single_request_batch(
    monkeypatch,
) -> None:
    class FakeExistingBackend:
        def __init__(self):
            pass

    import run_qwen3_real_weight_serving

    monkeypatch.setattr(
        run_qwen3_real_weight_serving,
        "ExistingGRDecodeAttentionBackend",
        FakeExistingBackend,
    )

    backend = run_qwen3_real_weight_serving.make_decode_backend(
        real_serving_args(
            decode_steps=2,
        ),
        device="cuda",
    )

    assert isinstance(backend, FakeExistingBackend)


def test_real_weight_serving_median_helper() -> None:
    from run_qwen3_real_weight_serving import _median

    assert _median([3.0, 1.0, 2.0]) == 2.0
    assert _median(value for value in [1.0, 3.0]) == 2.0


def test_real_weight_serving_compact_run() -> None:
    from run_qwen3_real_weight_serving import _compact_run

    compact = _compact_run(
        {
            "first_token_ids": tuple(range(16)),
            "first_metadata": {
                "beam_details": tuple({"rank": idx} for idx in range(8)),
                "decode_ms": 2.0,
            },
        }
    )

    assert "first_token_ids" not in compact
    assert compact["first_token_ids_count"] == 16
    assert compact["first_token_ids_sample"] == tuple(range(8))
    assert "beam_details" not in compact["first_metadata"]
    assert compact["first_metadata"]["beam_details_count"] == 8
    assert len(compact["first_metadata"]["beam_details_sample"]) == 4


def test_real_weight_serving_metadata_metric_median() -> None:
    from run_qwen3_real_weight_serving import _metadata_metric_median

    runs = [
        {"first_metadata": {"decode_ms": 3.0}},
        {"first_metadata": {"decode_ms": 1.0}},
        {"first_metadata": {"decode_ms": 2.0}},
    ]

    assert _metadata_metric_median(runs, "decode_ms") == 2.0
    assert _metadata_metric_median(runs, "missing") is None


def test_sweep_real_weight_serving_builds_fixed_command() -> None:
    from sweep_real_weight_serving import build_command

    command = build_command(
        sweep_args(
            warmup_runs=2,
            repeat=3,
            profile_continuous_decode=True,
            beam_kv_pool_capacity=2,
        ),
        context_len=4700,
        decode_steps=3,
        beam_width=128,
        requests=2,
        beam_policy="fixed",
    )

    assert "--context-len" in command
    assert command[command.index("--context-len") + 1] == "4700"
    assert "--decode-steps" in command
    assert command[command.index("--decode-steps") + 1] == "3"
    assert "--continuous" in command
    assert "--profile-continuous-decode" in command
    assert "--beam-kv-pool-capacity" in command
    assert "--beam-schedule" not in command


def test_sweep_real_weight_serving_builds_scheduled_command() -> None:
    from sweep_real_weight_serving import build_command

    command = build_command(
        sweep_args(),
        context_len=16,
        decode_steps=3,
        beam_width=128,
        requests=2,
        beam_policy="scheduled",
    )

    assert "--beam-schedule" in command
    assert command[command.index("--beam-schedule") + 1] == "0:128,1:64,2:32"


def test_sweep_real_weight_serving_builds_staggered_arrival_command() -> None:
    from sweep_real_weight_serving import build_command

    command = build_command(
        sweep_args(
            arrival_stagger_ticks=1,
            arrival_burst_size=2,
        ),
        context_len=16,
        decode_steps=3,
        beam_width=128,
        requests=2,
        beam_policy="fixed",
    )

    assert "--arrival-stagger-ticks" in command
    assert command[command.index("--arrival-stagger-ticks") + 1] == "1"
    assert "--arrival-burst-size" in command
    assert command[command.index("--arrival-burst-size") + 1] == "2"


def test_sweep_real_weight_serving_scheduled_widths_reuse_last_fraction() -> None:
    from sweep_real_weight_serving import scheduled_widths

    assert scheduled_widths(
        beam_width=100,
        decode_steps=4,
        fractions="1.0,0.5",
    ) == {
        0: 100,
        1: 50,
        2: 50,
        3: 50,
    }


def test_sweep_real_weight_serving_rejects_invalid_policy() -> None:
    from sweep_real_weight_serving import build_command

    with pytest.raises(ValueError, match="unsupported beam policy"):
        build_command(
            sweep_args(
                continuous=False,
                scheduled_fractions="1.0,0.5",
            ),
            context_len=16,
            decode_steps=1,
            beam_width=8,
            requests=1,
            beam_policy="score_margin",
        )


def test_real_weight_serving_continuous_profile_aggregate() -> None:
    from run_qwen3_real_weight_serving import _aggregate_decode_profile

    aggregate = _aggregate_decode_profile(
        {
            "continuous.prefill": {"total_ms": 10.0},
            "continuous.decode_microbatch_total": {"total_ms": 20.0},
            "continuous.decode_batch_build": {"total_ms": 1.0},
            "continuous.topk_indices": {"total_ms": 2.0},
            "continuous.beam_selection": {"total_ms": 3.0},
            "continuous.beam_kv_scatter": {"total_ms": 4.0},
            "model.forward_decode_step": {"total_ms": 15.0},
            "layer0.decode_attention": {"total_ms": 5.0},
            "layer1.decode_attention": {"total_ms": 6.0},
            "layer0.mlp": {"total_ms": 7.0},
        }
    )

    assert aggregate["continuous_prefill_ms"] == 10.0
    assert aggregate["continuous_decode_microbatch_total_ms"] == 20.0
    assert aggregate["continuous_decode_batch_build_ms"] == 1.0
    assert aggregate["continuous_topk_indices_ms"] == 2.0
    assert aggregate["continuous_beam_selection_ms"] == 3.0
    assert aggregate["continuous_beam_kv_scatter_ms"] == 4.0
    assert aggregate["model_forward_decode_step_ms"] == 15.0
    assert aggregate["decode_attention_ms"] == 11.0
    assert aggregate["mlp_ms"] == 7.0


def test_real_weight_serving_writes_summary_json(tmp_path) -> None:
    from run_qwen3_real_weight_serving import _write_summary_json

    output_json = tmp_path / "profiles" / "summary.json"

    _write_summary_json({"wall_ms_median": 1.5, "samples": (1, 2)}, str(output_json))

    assert output_json.read_text(encoding="utf-8")
    assert '"wall_ms_median": 1.5' in output_json.read_text(encoding="utf-8")


def test_real_weight_serving_writes_verbose_beam_path_json(tmp_path) -> None:
    from gr_inference.gr_kv import BeamPath
    from run_qwen3_real_weight_serving import _write_summary_json

    beam_path = BeamPath(max_decode_steps=2, max_beam_width=2)
    beam_path.append(parent_beams=(0, 0), token_ids=(311, 389), scores=(-1.0, -2.0))
    output_json = tmp_path / "summary.json"

    _write_summary_json({"first_metadata": {"_beam_path": beam_path}}, str(output_json))

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["first_metadata"]["_beam_path"]["entries"][0]["token_ids"] == [
        311,
        389,
    ]


def test_compare_gr_sglang_beam_reports_gr_output_token_budget() -> None:
    from compare_gr_sglang_beam import compare, render_markdown

    gr = {
        "engine_status": {"max_decode_steps": 2, "max_beam_width": 2},
        "outputs": [
            {
                "workload_id": "req-0",
                "beam_results": [
                    {"output_ids": [1, 2, 3], "meta_info": {"sequence_score": -3.0}},
                    {"output_ids": [1, 2, 4], "meta_info": {"sequence_score": -4.0}},
                ],
            }
        ],
        "responses": 1,
        "wall_ms_median": 10.0,
    }
    sglang = {
        "decode_steps": 3,
        "runs": [
            {
                "outputs": [
                    {
                        "workload_id": "req-0",
                        "beams": [
                            {"rank": 0, "token_ids": [1, 2, 3], "score": -1.0},
                            {"rank": 1, "token_ids": [1, 2, 4], "score": -1.3},
                        ],
                    }
                ]
            }
        ],
    }

    report = compare(gr, sglang)
    markdown = render_markdown(report)

    assert report["workload"]["gr_output_token_budget"] == 3
    assert report["workload"]["sglang_output_token_budget"] == 3
    assert report["workload"]["output_token_budget_match"] is True
    assert report["correctness"]["token_length_match_rate"] == 1.0
    assert report["correctness"]["top1_exact_match_rate"] == 1.0
    assert "GR output_token_budget: `3`" in markdown


def test_real_weight_serving_inflight_arrival_metrics() -> None:
    from run_qwen3_real_weight_serving import _inflight_arrival_metrics

    metrics = _inflight_arrival_metrics(
        [
            {
                "prefill_request_ids": ["req-0"],
                "decode_batches": [
                    {"step": 0, "size": 1, "request_ids": ["req-0"]},
                ],
            },
            {
                "prefill_request_ids": ["req-1"],
                "decode_batches": [
                    {"step": 0, "size": 1, "request_ids": ["req-1"]},
                    {"step": 1, "size": 1, "request_ids": ["req-0"]},
                ],
            },
        ],
        arrival_stagger_ticks=1,
        submitted_after_start=1,
    )

    assert metrics["arrival_stagger_ticks"] == 1
    assert metrics["inflight_submitted_after_start"] == 1
    assert metrics["inflight_admission_ticks"] == 1
    assert metrics["inflight_mixed_decode_step_ticks"] == 1
    assert metrics["inflight_max_decode_batch_size"] == 1


def test_real_weight_serving_staggered_arrivals_support_bursts() -> None:
    from run_qwen3_real_weight_serving import _submit_staggered_arrivals

    class FakeTorch:
        @staticmethod
        def randint(*args, **kwargs):
            return object()

    class FakeExecutor:
        def __init__(self):
            self.request_ids = []

        def submit(self, request):
            self.request_ids.append(request.request_id)

    args = Namespace(
        requests=5,
        arrival_stagger_ticks=2,
        arrival_burst_size=2,
        context_len=16,
        decode_steps=3,
        beam_width=128,
        beam_schedule=None,
    )
    config = Namespace(vocab_size=32)
    executor = FakeExecutor()

    next_request = _submit_staggered_arrivals(
        args,
        FakeTorch,
        config,
        "cpu",
        executor,
        request_prefix="req",
        next_request=0,
        current_tick=0,
        workload=(),
    )
    assert next_request == 2
    assert executor.request_ids == ["req-0", "req-1"]

    next_request = _submit_staggered_arrivals(
        args,
        FakeTorch,
        config,
        "cpu",
        executor,
        request_prefix="req",
        next_request=next_request,
        current_tick=2,
        workload=(),
    )
    assert next_request == 4
    assert executor.request_ids == ["req-0", "req-1", "req-2", "req-3"]


def test_real_weight_serving_make_beam_kv_pool() -> None:
    torch = torch_or_skip()
    from run_qwen3_real_weight_serving import _make_beam_kv_pool

    config = Namespace(
        num_layers=2,
        num_kv_heads=3,
        head_dim=4,
    )
    args = Namespace(
        continuous=True,
        beam_kv_pool_capacity=5,
        decode_steps=2,
        beam_width=7,
    )

    pool = _make_beam_kv_pool(args, torch, config, "cpu")

    assert tuple(pool.key.shape) == (2, 5, 2, 7, 3, 4)
    assert tuple(pool.value.shape) == (2, 5, 2, 7, 3, 4)
    assert pool.usage()["beam_kv_pool_free"] == 5


def test_real_weight_serving_beam_kv_pool_requires_continuous() -> None:
    from run_qwen3_real_weight_serving import _make_beam_kv_pool

    with pytest.raises(RuntimeError, match="requires --continuous"):
        _make_beam_kv_pool(
            Namespace(
                continuous=False,
                beam_kv_pool_capacity=1,
                decode_steps=1,
                beam_width=2,
            ),
            object(),
            Namespace(num_layers=1, num_kv_heads=1, head_dim=1),
            "cpu",
        )


def test_real_weight_serving_parses_beam_schedule() -> None:
    from run_qwen3_real_weight_serving import _parse_beam_schedule

    assert _parse_beam_schedule("0:128, 1:64,2:32") == {
        0: 128,
        1: 64,
        2: 32,
    }


def test_real_weight_serving_beam_schedule_requires_step_zero() -> None:
    from run_qwen3_real_weight_serving import _parse_beam_schedule

    with pytest.raises(ValueError, match="include step 0"):
        _parse_beam_schedule("1:64,2:32")


def test_real_weight_serving_builds_scheduled_beam_policy() -> None:
    from run_qwen3_real_weight_serving import _make_beam_width_policy

    policy = _make_beam_width_policy(
        Namespace(
            beam_width=128,
            beam_schedule="0:128,1:64,2:32",
        )
    )

    assert policy.width_for_step(0) == 128
    assert policy.width_for_step(1) == 64
    assert policy.width_for_step(2) == 32
    assert policy.width_for_step(3) == 32


def test_real_weight_serving_rejects_scheduled_width_over_max() -> None:
    from run_qwen3_real_weight_serving import _make_beam_width_policy

    with pytest.raises(ValueError, match="exceeds --beam-width"):
        _make_beam_width_policy(
            Namespace(
                beam_width=64,
                beam_schedule="0:128,1:64",
            )
        )


def test_http_serving_request_factory_tensorizes_input_ids() -> None:
    factory = http_request_factory(default_decode_steps=2, default_beam_width=4)

    request = factory({"request_id": "req-1", "input_ids": [1, 2, 3]})

    assert tuple(request.input_ids.shape) == (1, 3)
    assert request.max_decode_steps == 2
    assert request.beam_width == 4

    with_timeout = factory(
        {"request_id": "req-timeout", "input_ids": [1, 2, 3], "timeout_ticks": 5}
    )
    assert with_timeout.metadata["timeout_ticks"] == 5

    with_processor = factory(
        {
            "request_id": "req-processor",
            "input_ids": [1, 2, 3],
            "logits_processors": [
                {"type": "token_suppress", "token_ids": [9]},
            ],
        }
    )
    assert with_processor.logits_processors[0].metadata()["type"] == "token_suppress"


def test_http_serving_request_factory_builds_dynamic_beam_policy() -> None:
    factory = http_request_factory(default_beam_width=4)

    request = factory(
        {
            "request_id": "req-dynamic",
            "input_ids": [1, 2],
            "beam_width_policy": {
                "type": "score_margin",
                "score_margin": 0.5,
            },
        }
    )

    assert request.beam_width_policy.width_for_step(0) == 4


def test_http_serving_request_factory_builds_scheduled_beam_policy() -> None:
    factory = http_request_factory(default_decode_steps=3, default_beam_width=4)

    request = factory(
        {
            "request_id": "req-scheduled",
            "input_ids": [1, 2],
            "beam_width_policy": {
                "type": "scheduled",
                "schedule": {
                    "0": 4,
                    "1": 2,
                    "2": 1,
                },
            },
        }
    )

    assert request.beam_width_policy.width_for_step(0) == 4
    assert request.beam_width_policy.width_for_step(1) == 2
    assert request.beam_width_policy.width_for_step(2) == 1


def test_http_serving_request_factory_rejects_batched_inputs() -> None:
    factory = http_request_factory()

    with pytest.raises(ValueError, match="input_ids"):
        factory({"request_id": "bad", "input_ids": [[1], [2]]})


def test_http_serving_normalizes_real_backend_args() -> None:
    from serve_qwen3_gr_http import _normalize_args

    args = Namespace(max_batch_size=2, requests=1)

    _normalize_args(args)

    assert args.continuous is True
    assert args.batched_decode is True
    assert args.requests == 2
    assert args.warmup_online_pool_windows is True
    assert args.warmup_online_max_cases == 64
    assert args.freeze_cuda_graphs_after_warmup is True


def test_http_serving_online_warmup_cases_cover_pool_windows() -> None:
    from serve_qwen3_gr_http import _online_warmup_cases

    args = Namespace(
        max_batch_size=4,
        beam_width=256,
        warmup_online_pool_windows=True,
        warmup_online_max_cases=64,
    )

    assert _online_warmup_cases(args, pool_capacity=4) == (
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 1),
        (0, 2),
        (1, 2),
        (2, 2),
        (0, 3),
        (1, 3),
        (0, 4),
    )


def test_http_serving_online_warmup_cases_can_use_legacy_batch_shapes() -> None:
    from serve_qwen3_gr_http import _online_warmup_cases

    args = Namespace(
        max_batch_size=4,
        beam_width=256,
        warmup_online_pool_windows=False,
        warmup_online_max_cases=64,
    )

    assert _online_warmup_cases(args, pool_capacity=4) == (
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
    )


def test_http_serving_online_warmup_prefills_blockers_before_targets(
    monkeypatch,
) -> None:
    torch = torch_or_skip()
    import serve_qwen3_gr_http

    events = []

    class FakeScheduler:
        def tick(self, *, prefill_executor=None, decode_executor=None):
            assert callable(prefill_executor)
            assert callable(decode_executor)
            events.append(("tick", decode_executor(("unused",))))

    class FakeExecutor:
        def __init__(self, *args, **kwargs) -> None:
            self.scheduler = FakeScheduler()
            self._run_prefill = lambda request_ids: None

        def submit(self, request) -> None:
            events.append(
                (
                    "submit",
                    request.request_id,
                    request.beam_width,
                    request.metadata["warmup_target_request"],
                    request.metadata["ignore_eos"],
                    len(request.logits_processors),
                )
            )

        def run_until_empty(self, *, max_ticks=None, timeout_unfinished=False):
            events.append(("drain", max_ticks, timeout_unfinished))
            return ()

    monkeypatch.setattr(
        serve_qwen3_gr_http,
        "GRContinuousServingExecutor",
        FakeExecutor,
    )
    monkeypatch.setattr(
        serve_qwen3_gr_http,
        "_online_warmup_cases",
        lambda args, pool_capacity: ((2, 1),),
    )
    args = Namespace(
        max_batch_size=4,
        beam_kv_pool_capacity=4,
        decode_steps=3,
        context_len=8,
        beam_width=256,
        executor_sync_timing=False,
        enable_prefill_cache=False,
        prefill_cache_max_entries=None,
        prefill_cache_max_tokens=0,
        prefill_cache_page_size=None,
        prefill_cache_min_prefix_tokens=None,
        prefill_cache_max_decode_extend_tokens=None,
        decode_cuda_graph_batch_buckets=(1, 2, 4, 8),
        suppress_token_ids="9,10",
        suppress_special_tokens_on_ignore_eos=True,
    )
    engine = Namespace(model=Namespace(config=Namespace(vocab_size=32)))

    serve_qwen3_gr_http._warmup_online_shapes(
        args,
        torch=torch,
        engine=engine,
        beam_kv_pool=None,
        context_kv_pool=None,
        device="cpu",
    )

    assert [event[0] for event in events] == [
        "submit",
        "submit",
        "tick",
        "submit",
        "drain",
    ]
    assert events[0][2:] == (1, False, True, 1)
    assert events[1][2:] == (1, False, True, 1)
    assert events[2] == ("tick", ())
    assert events[3][2:] == (256, True, True, 1)
    assert events[4] == ("drain", 5, False)


def test_http_serving_freezes_cuda_graph_runners() -> None:
    from serve_qwen3_gr_http import _freeze_cuda_graph_captures

    class FakeRunner:
        def __init__(self) -> None:
            self.frozen = False

        def freeze_captures(self) -> None:
            self.frozen = True

    prefill = FakeRunner()
    decode = FakeRunner()
    engine = Namespace(
        _prefill_cuda_graph_runner=prefill,
        _decode_cuda_graph_runner=decode,
    )

    _freeze_cuda_graph_captures(engine)

    assert prefill.frozen is True
    assert decode.frozen is True


def test_http_serving_builds_validation_policy() -> None:
    from serve_qwen3_gr_http import _make_http_validation_policy

    policy = _make_http_validation_policy(
        Namespace(
            context_len=16,
            decode_steps=2,
            beam_width=8,
            max_http_request_bytes=4096,
            max_http_context_len=None,
            max_http_decode_steps=None,
            max_http_beam_width=None,
            max_http_submit_many=4,
            max_http_waiting_requests=8,
            max_http_timeout_ticks=32,
            allow_manual_tick=False,
            allow_catalog_reload=True,
        )
    )

    assert policy.max_request_bytes == 4096
    assert policy.max_context_len == 16
    assert policy.max_decode_steps == 2
    assert policy.max_beam_width == 8
    assert policy.max_submit_many == 4
    assert policy.max_waiting_requests == 8
    assert policy.max_timeout_ticks == 32
    assert policy.allow_manual_tick is False
    assert policy.allow_catalog_reload is True


def test_http_serving_builds_adapter_without_loading_real_model(monkeypatch) -> None:
    torch_or_skip()
    import serve_qwen3_gr_http

    config = Namespace(
        vocab_size=32,
        model_name="fake-qwen3",
    )

    class FakeModel:
        pass

    fake_model = FakeModel()
    fake_model.config = config

    monkeypatch.setattr(
        serve_qwen3_gr_http,
        "load_model",
        lambda args, torch: (fake_model, config, "cpu"),
    )
    monkeypatch.setattr(
        serve_qwen3_gr_http,
        "make_decode_backend",
        lambda args, device: (lambda inputs: inputs.q),
    )

    adapter = serve_qwen3_gr_http.build_http_serving_adapter(
        Namespace(
            model_dir="/models/qwen",
            context_len=16,
            decode_steps=1,
            beam_width=2,
            max_batch_size=2,
            decode_backend="fake",
            device="cpu",
            return_beam_details=False,
            beam_score_mode="logprob",
            profile_continuous_decode=False,
            profile_detail="coarse",
            beam_kv_pool_capacity=0,
            catalog_jsonl=None,
            catalog_vocab_size=None,
            catalog_eos_token_id=None,
            catalog_allow_eos_for_terminal=True,
            catalog_item_id_field="item_id",
            catalog_token_ids_field="token_ids",
            catalog_metadata_field="metadata",
            catalog_allow_duplicate_item_ids=False,
            catalog_allow_duplicate_token_paths=False,
            disable_background_worker=True,
            worker_tick_interval_s=0.001,
            worker_idle_sleep_s=0.005,
            decode_log_interval=2,
            max_http_request_bytes=1 << 20,
            max_http_context_len=None,
            max_http_decode_steps=None,
            max_http_beam_width=None,
            max_http_submit_many=32,
            max_http_waiting_requests=4,
            max_http_timeout_ticks=3,
            max_finished_requests=2,
            allow_manual_tick=False,
            allow_catalog_reload=False,
            api_key="secret",
            enable_log_requests=True,
            log_requests_level="summary",
        )
    )

    headers = {"Authorization": "Bearer secret"}
    response = adapter.handle(
        "POST",
        "/submit",
        {"request_id": "req-1", "input_ids": [1, 2]},
        headers=headers,
    )
    config_response = adapter.handle("GET", "/config", headers=headers)
    build_response = adapter.handle("GET", "/build", headers=headers)
    startup_lines = serve_qwen3_gr_http._startup_config_lines(adapter)

    assert response.status == 202
    assert adapter.facade.status()["waiting_prefill"] == 1
    assert config_response.body["validation_policy"]["max_waiting_requests"] == 4
    assert config_response.body["validation_policy"]["max_timeout_ticks"] == 3
    assert config_response.body["scheduler_policy"]["max_finished_requests"] == 2
    assert config_response.body["auth"]["enabled"] is True
    assert config_response.body["logging"]["request_logging_enabled"] is True
    assert config_response.body["build"]["decode_backend"] == "fake"
    assert config_response.body["build"]["model_name"] == "fake-qwen3"
    assert build_response.body["build"]["device"] == "cpu"
    assert build_response.body["build"]["beam_kv_pool_enabled"] is False
    assert isinstance(build_response.body["build"]["cuda_available"], bool)
    assert build_response.body["build"]["enable_log_requests"] is True
    assert build_response.body["build"]["decode_log_interval"] == 2
    assert any("max_waiting_requests=4" in line for line in startup_lines)
    assert any("max_finished_requests=2" in line for line in startup_lines)
    assert any("Auth: enabled=True" in line for line in startup_lines)
    assert any("decode_backend=fake" in line for line in startup_lines)


def test_http_soak_config_from_args() -> None:
    from soak_http_serving import _config_from_args

    config = _config_from_args(
        Namespace(
            requests=3,
            submit_batch_size=2,
            input_len=4,
            decode_steps=2,
            beam_width=8,
            timeout_ticks=5,
            cancel_every=0,
            poll_interval_s=0.0,
            max_polls=7,
            manual_tick=True,
            ready_sample_interval=1,
            progress_interval=2,
            request_prefix="test",
            drain_at_end=False,
            shutdown_at_end=True,
        )
    )

    assert config.requests == 3
    assert config.submit_batch_size == 2
    assert config.input_len == 4
    assert config.timeout_ticks == 5
    assert config.manual_tick is True
    assert config.progress_interval == 2
    assert config.shutdown_at_end is True


def test_http_soak_run_counts_completed_cancelled_and_overload() -> None:
    from soak_http_serving import SoakConfig, run_soak

    summary = run_soak(
        FakeSoakClient(overload_suffixes=("-2",)),
        SoakConfig(
            requests=4,
            submit_batch_size=2,
            input_len=3,
            decode_steps=1,
            beam_width=2,
            cancel_every=2,
            poll_interval_s=0.0,
            max_polls=2,
            request_prefix="req",
        ),
    )

    assert summary["requested"] == 4
    assert summary["accepted"] == 2
    assert summary["completed"] == 2
    assert summary["overload_errors"] == 1
    assert summary["outcomes"]["succeeded"] == 1
    assert summary["outcomes"]["cancelled"] == 1
    assert summary["pending"] == []
    assert summary["response_status_counts"]["429"] == 1
    assert summary["schema_version"] == "gr_http_soak_v1"
    assert summary["build"]["build"]["framework"] == "gr-inference"
    assert summary["kv_events"]["count"] == 1


def test_http_soak_run_records_failed_response_diagnostics() -> None:
    from soak_http_serving import SoakConfig, run_soak

    summary = run_soak(
        FakeSoakClient(failed_request_id="req-0"),
        SoakConfig(
            requests=1,
            submit_batch_size=1,
            input_len=3,
            decode_steps=1,
            beam_width=2,
            poll_interval_s=0.0,
            max_polls=1,
            request_prefix="req",
        ),
    )

    diagnostics = summary["response_diagnostics"]
    assert summary["outcomes"]["failed"] == 1
    assert diagnostics["stop_reason_counts"] == {"decode_failed": 1}
    assert diagnostics["error_type_counts"] == {"RuntimeError": 1}
    assert diagnostics["failed_samples"][0]["request_id"] == "req-0"
    assert diagnostics["failed_samples"][0]["error_message"] == "shape mismatch"


def test_http_soak_summary_tool_reports_pass_and_failures(tmp_path) -> None:
    from summarize_http_soak import summarize_soak

    summary = {
        "schema_version": "gr_http_soak_v1",
        "config": {"request_prefix": "soak"},
        "requested": 4,
        "accepted": 4,
        "completed": 4,
        "pending": [],
        "overload_errors": 0,
        "outcomes": {
            "succeeded": 3,
            "cancelled": 1,
            "failed": 0,
            "timed_out": 0,
        },
        "metrics": {
            "decode_ms": 12.0,
            "prefill_ms": 8.0,
            "beam_kv_pool_max_used": 2,
            "kv_health_kv_allocator_leak_detected": 0,
            "beam_kv_pool_health_beam_kv_pool_leak_detected": 0,
            "worker_errors": 0,
        },
        "response_diagnostics": {
            "stop_reason_counts": {"max_decode_steps": 3},
            "error_type_counts": {},
            "failed_samples": [],
        },
    }

    report = summarize_soak(summary, expected_requests=4, expected_cancelled=1)
    failed = summarize_soak(
        {
            **summary,
            "accepted": 3,
            "outcomes": {**summary["outcomes"], "failed": 1},
        },
        expected_requests=4,
        expected_cancelled=1,
    )

    assert report["passed"] is True
    assert report["request_prefix"] == "soak"
    assert report["beam_pool_max_used"] == 2
    assert report["response_diagnostics"]["stop_reason_counts"] == {
        "max_decode_steps": 3
    }
    assert failed["passed"] is False
    assert "accepted=3 requested=4" in failed["failures"]
    assert "failed=1" in failed["failures"]


def test_http_soak_summary_tool_cli_writes_json(tmp_path) -> None:
    from summarize_http_soak import main

    input_path = tmp_path / "soak.json"
    output_path = tmp_path / "summary.json"
    input_path.write_text(
        json.dumps(
            {
                "requested": 1,
                "accepted": 1,
                "completed": 1,
                "pending": [],
                "overload_errors": 0,
                "outcomes": {
                    "succeeded": 1,
                    "cancelled": 0,
                    "failed": 0,
                    "timed_out": 0,
                },
                "metrics": {
                    "kv_health_kv_allocator_leak_detected": 0,
                    "beam_kv_pool_health_beam_kv_pool_leak_detected": 0,
                    "worker_errors": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    old_argv = sys.argv
    try:
        sys.argv = [
            "summarize_http_soak.py",
            str(input_path),
            "--expected-requests",
            "1",
            "--output-json",
            str(output_path),
            "--fail-on-error",
        ]
        main()
    finally:
        sys.argv = old_argv

    assert json.loads(output_path.read_text(encoding="utf-8"))["passed"] is True


def test_serving_profile_summary_tool_reports_buckets_and_comparison(tmp_path) -> None:
    from summarize_serving_profile import summarize_profile

    summary = {
        "responses": 2,
        "wall_ms": 100.0,
        "scheduler_metrics": {"decode_ms": 80.0, "prefill_ms": 20.0},
        "kernel_backend_selection": {"rmsnorm": "flashinfer", "rope": "torch"},
        "flashinfer_call_counts": {"rmsnorm": 3},
        "gr_trtllm_call_counts": {"fused_qk_norm_rope_cuda": 2},
        "scheduler_status": {
            "decode_inputs_cache": {"entries": 1, "hits": 2, "misses": 1},
            "topk_indices_cache": {"entries": 1, "hits": 3, "misses": 1},
        },
        "decode_profile_aggregate": {
            "continuous_decode_batch_build_ms": 10.0,
            "continuous_topk_indices_ms": 5.0,
            "decode_attention_ms": 20.0,
        },
    }
    baseline = {
        "decode_profile_aggregate": {
            "continuous_decode_batch_build_ms": 20.0,
            "continuous_topk_indices_ms": 10.0,
            "decode_attention_ms": 25.0,
        },
    }

    report = summarize_profile(summary, baseline=baseline)
    rows = {row["name"]: row for row in report["top_buckets"]}

    assert report["wall_ms"] == 100.0
    assert report["decode_ms"] == 80.0
    assert report["kernel_backend_selection"]["rmsnorm"] == "flashinfer"
    assert report["flashinfer_call_counts"]["rmsnorm"] == 3
    assert report["gr_trtllm_call_counts"]["fused_qk_norm_rope_cuda"] == 2
    assert report["cache_metrics"]["decode_inputs_cache"]["hits"] == 2
    assert rows["decode_attention_ms"]["pct_of_decode"] == 25.0
    assert rows["continuous_decode_batch_build_ms"]["baseline_ms"] == 20.0
    assert rows["continuous_decode_batch_build_ms"]["improvement_pct"] == 50.0


def test_serving_profile_summary_tool_cli_writes_json(tmp_path) -> None:
    from summarize_serving_profile import main

    input_path = tmp_path / "profile.json"
    output_path = tmp_path / "profile_summary.json"
    input_path.write_text(
        json.dumps(
            {
                "wall_ms": 10.0,
                "scheduler_metrics": {"decode_ms": 5.0},
                "decode_profile_aggregate": {"decode_attention_ms": 2.0},
            }
        ),
        encoding="utf-8",
    )
    old_argv = sys.argv
    try:
        sys.argv = [
            "summarize_serving_profile.py",
            str(input_path),
            "--output-json",
            str(output_path),
        ]
        main()
    finally:
        sys.argv = old_argv

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["top_buckets"][0]["name"] == "decode_attention_ms"


def test_http_serving_launcher_exposes_production_env_knobs() -> None:
    launcher = (
        Path(__file__).resolve().parents[1] / "scripts" / "serve_qwen3_gr_http.sh"
    )

    text = launcher.read_text(encoding="utf-8")

    assert launcher.exists()
    assert "--max-http-waiting-requests" in text
    assert "--max-http-timeout-ticks" in text
    assert "--max-finished-requests" in text
    assert "--api-key" in text
    assert "--enable-log-requests" in text
    assert "--decode-log-interval" in text
    assert "GR_MODEL_DIR" in text
    assert "GR_HTTP_HOST" in text
    assert "GR_HTTP_API_KEY" in text


def test_container_artifacts_wire_serving_launcher() -> None:
    root = Path(__file__).resolve().parents[1]
    dockerfile = root / "Dockerfile"
    dockerignore = root / ".dockerignore"

    dockerfile_text = dockerfile.read_text(encoding="utf-8")
    dockerignore_text = dockerignore.read_text(encoding="utf-8")

    assert dockerfile.exists()
    assert "INSTALL_KERNEL_DEPS" in dockerfile_text
    assert 'ENTRYPOINT ["scripts/serve_qwen3_gr_http.sh"]' in dockerfile_text
    assert "EXPOSE 8000" in dockerfile_text
    assert "*.safetensors" in dockerignore_text
    assert "profiles" in dockerignore_text


def test_sweep_real_weight_serving_helpers() -> None:
    from sweep_real_weight_serving import build_command, parse_csv_ints, parse_summary

    assert parse_csv_ints("64, 128") == [64, 128]

    args = sweep_args(repeat=3)
    command = build_command(
        args,
        context_len=16,
        decode_steps=1,
        beam_width=128,
        requests=2,
        beam_policy="fixed",
    )

    assert "--model-dir" in command
    assert "/models/qwen" in command
    assert "--beam-width" in command
    assert "128" in command
    assert "--requests" in command
    assert "2" in command
    assert "--beam-score-mode" in command
    assert "logprob" in command
    assert "--continuous" in command

    summary = parse_summary(
        "\n".join(
            [
                "wall_ms_median: 123.4",
                "decode_ms_median: 68.6",
                "first_metadata: {'ignored': True}",
            ]
        )
    )
    assert summary == {
        "wall_ms_median": 123.4,
        "decode_ms_median": 68.6,
    }
