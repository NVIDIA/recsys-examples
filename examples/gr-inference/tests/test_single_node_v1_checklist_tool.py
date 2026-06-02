import os
import subprocess
import sys


def test_single_node_v1_checklist_dry_run_builds_plan(tmp_path, capsys) -> None:
    from run_single_node_v1_checklist import main

    assert (
        main(
            [
                "--dry-run",
                "--output-dir",
                str(tmp_path),
                "--skip-tests",
                "--skip-profile",
                "--context-len",
                "16",
                "--decode-steps",
                "1",
                "--beam-width",
                "128",
                "--num-layers",
                "28",
                "--num-kv-heads",
                "8",
                "--head-dim",
                "128",
                "--vocab-size",
                "151936",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert '"dry_run": true' in output
    assert "memory_estimate" in output
    assert (tmp_path / "single_node_v1_checklist_plan.json").exists()


def test_single_node_v1_checklist_can_include_profile_and_soak_steps(tmp_path) -> None:
    from run_single_node_v1_checklist import REPO_ROOT, _is_git_worktree, build_parser, build_plan

    args = build_parser().parse_args(
        [
            "--dry-run",
            "--output-dir",
            str(tmp_path),
            "--run-soak",
        ]
    )
    plan = build_plan(args)

    names = [step["name"] for step in plan]
    assert "focused_pytest" in names
    assert "full_pytest" in names
    if _is_git_worktree(REPO_ROOT):
        assert "diff_check" in names
    else:
        assert "diff_check" not in names
    assert "memory_estimate" in names
    assert "profile_serving" in names
    assert "profile_summary" in names
    assert "http_soak" in names
    profile = plan[names.index("profile_serving")]
    profile_command = profile["command"]
    assert profile_command[profile_command.index("--model-dir") + 1] == "models/Qwen3-1.7B"
    assert profile_command[profile_command.index("--beam-schedule") + 1] == "0:256,1:192,2:192"
    soak = plan[names.index("http_soak")]
    command = soak["command"]
    assert command[command.index("--input-len") + 1] == "16"
    assert command[command.index("--decode-steps") + 1] == "1"
    assert command[command.index("--beam-width") + 1] == "128"
    assert command[command.index("--progress-interval") + 1] == "512"
    server_command = soak["server_command"]
    assert server_command[server_command.index("--context-len") + 1] == "16"
    assert server_command[server_command.index("--decode-steps") + 1] == "1"
    assert server_command[server_command.index("--beam-width") + 1] == "128"
    report_command = soak["report_command"]
    assert "tools/summarize_http_soak.py" in report_command
    assert "--fail-on-error" in report_command


def test_single_node_v1_checklist_can_include_dynamic_beam_quality(tmp_path) -> None:
    from run_single_node_v1_checklist import build_parser, build_plan

    args = build_parser().parse_args(
        [
            "--dry-run",
            "--output-dir",
            str(tmp_path),
            "--skip-tests",
            "--skip-profile",
            "--run-dynamic-beam-quality",
        ]
    )
    plan = build_plan(args)

    names = [step["name"] for step in plan]
    assert "dynamic_beam_quality" in names
    command = plan[names.index("dynamic_beam_quality")]["command"]
    assert "tools/compare_beam_policies.py" in command
    schedule_args = [
        command[index + 1]
        for index, item in enumerate(command)
        if item == "--schedule"
    ]
    assert schedule_args == [
        "0:256,1:192,2:192",
        "0:256,1:192,2:128",
        "0:256,1:128,2:64",
    ]
    assert "--fail-on-quality-gate" in command
    assert "--output-markdown" in command


def test_single_node_v1_checklist_can_skip_diff_check(tmp_path) -> None:
    from run_single_node_v1_checklist import build_parser, build_plan

    args = build_parser().parse_args(
        [
            "--dry-run",
            "--output-dir",
            str(tmp_path),
            "--skip-diff-check",
        ]
    )
    plan = build_plan(args)

    names = [step["name"] for step in plan]
    assert "focused_pytest" in names
    assert "full_pytest" in names
    assert "diff_check" not in names


def test_single_node_v1_checklist_logs_failed_command_output(tmp_path) -> None:
    from run_single_node_v1_checklist import _run_command

    log_path = tmp_path / "failed.log"
    command = [
        sys.executable,
        "-c",
        "print('checklist failure detail'); raise SystemExit(7)",
    ]

    try:
        _run_command(command, env=os.environ.copy(), log_path=log_path)
    except subprocess.CalledProcessError as exc:
        assert exc.returncode == 7
    else:
        raise AssertionError("expected failed command to raise")

    assert "checklist failure detail" in log_path.read_text(encoding="utf-8")
