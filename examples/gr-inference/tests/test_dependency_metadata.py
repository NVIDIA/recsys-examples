from __future__ import annotations

import re
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _kernel_extra_block() -> str:
    text = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(
        r"^kernels = \[\n(?P<body>.*?)^\]", text, flags=re.MULTILINE | re.DOTALL
    )
    assert match is not None, "pyproject.toml must define the kernels extra"
    return match.group("body")


def test_kernel_extra_bounds_flashinfer_and_cuda_tile() -> None:
    block = _kernel_extra_block()

    assert '"flashinfer-python>=0.6.11,<0.7"' in block
    assert '"cuda-tile>=1.3.0,<1.4"' in block


def test_kernel_extra_keeps_cutlass_wheels_matched() -> None:
    block = _kernel_extra_block()
    versions = dict(
        re.findall(
            r'"(nvidia-cutlass-dsl(?:-libs-(?:base|cu13))?)==([^"]+)"',
            block,
        )
    )

    assert versions == {
        "nvidia-cutlass-dsl": "4.5.1",
        "nvidia-cutlass-dsl-libs-base": "4.5.1",
        "nvidia-cutlass-dsl-libs-cu13": "4.5.1",
    }


def test_kernel_extra_includes_gr_decode_attention_runtime_dependency() -> None:
    block = _kernel_extra_block()

    assert '"quack-kernels>=0.3.3,<0.5"' in block
    assert '"apache-tvm-ffi>=0.1.6"' in block
    assert '"torch-c-dlpack-ext"' in block


def test_kernel_extra_keeps_packaging_compatible_with_dali_cuda13() -> None:
    block = _kernel_extra_block()

    assert '"packaging>=24.2,<=25.0"' in block


def test_bootstrap_never_installs_submodule_requirements() -> None:
    script = (REPO_ROOT / "scripts" / "bootstrap_container_env.sh").read_text(
        encoding="utf-8"
    )

    assert "pip install -r" not in script
    assert "GR_BOOTSTRAP_INSTALL_EXTERNAL_KERNEL_DEPS" not in script


def test_gr_decode_attention_is_relative_submodule() -> None:
    gitmodules = (REPO_ROOT / ".gitmodules").read_text(encoding="utf-8")

    assert "path = third_party/gr-decode-attention" in gitmodules
    assert "url = ../gr-decode-attention.git" in gitmodules
