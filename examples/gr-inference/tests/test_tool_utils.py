import sys
from pathlib import Path

import pytest
from tool_utils import (  # noqa: E402
    bootstrap_repo_paths,
    iter_jsonl,
    numeric_median,
    read_jsonl,
    summary_metric,
)


def test_bootstrap_repo_paths_adds_requested_import_roots(tmp_path: Path) -> None:
    anchor = tmp_path / "repo" / "tools" / "tool.py"
    original_path = list(sys.path)
    try:
        repo_root = bootstrap_repo_paths(
            anchor, include_tools=True, include_benchmarks=True
        )

        assert repo_root == tmp_path / "repo"
        assert str(repo_root / "src") in sys.path
        assert str(repo_root / "tools") in sys.path
        assert str(repo_root / "benchmarks") in sys.path
    finally:
        sys.path[:] = original_path


def test_read_jsonl_skips_blank_lines_and_respects_limit(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")

    assert read_jsonl(path, limit=1) == [{"a": 1}]
    assert list(iter_jsonl(path)) == [(1, {"a": 1}), (3, {"b": 2})]


def test_read_jsonl_rejects_non_object_rows(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text("[1, 2, 3]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a JSON object"):
        read_jsonl(path)


def test_summary_metric_uses_preferred_sources() -> None:
    assert summary_metric({"wall_ms_median": 3.0}, "wall_ms") == 3.0
    assert summary_metric({"wall_ms_samples": [4.0, 2.0]}, "wall_ms") == 3.0
    assert summary_metric({"scheduler_metrics": {"decode_ms": 5.0}}, "decode_ms") == 5.0
    assert numeric_median([3, 1, 2]) == 2.0
