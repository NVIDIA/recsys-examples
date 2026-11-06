# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise CPU build checks without importing GPU-dependent runtime packages."""

import ast
import builtins
import importlib.metadata
import importlib.util
import io
import os
import re
import runpy
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch


def setup_function(name, namespace):
    setup_path = Path(__file__).resolve().parents[1] / "setup.py"
    tree = ast.parse(setup_path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(setup_path), "exec"),
        namespace,
    )
    return namespace[name]


def block_runtime_imports(original):
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {
            "torchrec",
            "fbgemm_gpu",
            "dynamicemb",
            "dynamicemb_extensions",
        }:
            raise AssertionError(f"CPU build tried to import {name}")
        return original(name, *args, **kwargs)

    return guarded


class BuildArchsTest(unittest.TestCase):
    def setUp(self):
        namespace = {
            "os": os,
            "re": re,
            "subprocess": subprocess,
            "CUDA_HOME": "/toolkit",
        }
        self.select = setup_function("get_cuda_archs", namespace)

    def test_cuda13_includes_gb300(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(
            subprocess,
            "check_output",
            return_value="sm_75\nsm_80\nsm_90\nsm_100\nsm_103\nsm_120\n",
        ) as query:
            self.assertEqual(self.select(), ["75", "80", "90", "100", "103", "120"])
            query.assert_called_once_with(
                ["/toolkit/bin/nvcc", "--list-gpu-code"], text=True
            )

    def test_older_compiler_does_not_receive_new_targets(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(
            subprocess, "check_output", return_value="sm_70\nsm_75\nsm_80\nsm_90\n"
        ):
            self.assertEqual(self.select(), ["75", "80", "90"])

    def test_explicit_gb300_on_older_compiler_fails(self):
        with patch.dict(os.environ, {"DEMB_ARCHS": "103"}, clear=True), patch.object(
            subprocess, "check_output", return_value="sm_100\n"
        ):
            with self.assertRaisesRegex(RuntimeError, "Invalid DEMB_ARCHS"):
                self.select()

    def test_explicit_targets_and_separators(self):
        with patch.dict(
            os.environ, {"DEMB_ARCHS": "90, 103;"}, clear=True
        ), patch.object(subprocess, "check_output", return_value="sm_90\nsm_103\n"):
            self.assertEqual(self.select(), ["90", "103"])

    def test_empty_and_invalid_targets_fail(self):
        for requested in ("", ";", "10.3", "103;garbage"):
            with self.subTest(requested=requested), patch.dict(
                os.environ, {"DEMB_ARCHS": requested}, clear=True
            ), patch.object(subprocess, "check_output", return_value="sm_103\n"):
                with self.assertRaises(RuntimeError):
                    self.select()

    def test_compiler_failure_is_not_hidden(self):
        with patch.object(subprocess, "check_output", side_effect=FileNotFoundError):
            with self.assertRaises(FileNotFoundError):
                self.select()


class TorchRecMetadataTest(unittest.TestCase):
    def setUp(self):
        self.metadata = SimpleNamespace(
            version=Mock(return_value="1.8.0+source"),
            PackageNotFoundError=importlib.metadata.PackageNotFoundError,
        )
        self.check = setup_function(
            "check_torchrec_version",
            {
                "importlib": SimpleNamespace(metadata=self.metadata),
                "re": re,
                "compatible_versions": "1.2.0",
            },
        )

    def test_version_check_does_not_load_cuda_dependencies(self):
        with patch.object(
            builtins,
            "__import__",
            side_effect=block_runtime_imports(builtins.__import__),
        ):
            self.check()
        self.metadata.version.assert_called_once_with("torchrec")

    def test_multi_digit_minor_version(self):
        self.metadata.version.return_value = "1.10.0"
        self.check()

    def test_old_version_fails(self):
        self.metadata.version.return_value = "1.1.9"
        with self.assertRaisesRegex(RuntimeError, "required"):
            self.check()

    def test_missing_distribution_fails(self):
        self.metadata.version.side_effect = self.metadata.PackageNotFoundError(
            "torchrec"
        )
        with self.assertRaisesRegex(RuntimeError, "not installed"):
            self.check()

    def test_invalid_version_fails(self):
        self.metadata.version.return_value = "unknown"
        with self.assertRaisesRegex(RuntimeError, "Cannot read"):
            self.check()


class BuildOnlyCheckTest(unittest.TestCase):
    def test_build_check_without_driver_or_gpu_runtime_imports(self):
        self.run_build_check(tensordict_available=True)

    def test_missing_tensordict_dependency_fails_on_cpu(self):
        self.run_build_check(tensordict_available=False)

    def run_build_check(self, tensordict_available):
        torch = ModuleType("torch")
        torch.__path__ = []
        torch.__version__ = "2.13.0"
        torch.version = SimpleNamespace(cuda="13.3")
        torch.backends = SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 92400))
        torch.utils = ModuleType("torch.utils")
        torch.utils.__path__ = []
        extension = ModuleType("torch.utils.cpp_extension")
        extension.CUDA_HOME = "/toolkit"
        torch.utils.cpp_extension = extension
        tensordict = ModuleType("tensordict")
        tensordict.__file__ = "/packages/tensordict/__init__.py"
        modules = {
            "tensordict": tensordict,
            "torch": torch,
            "torch.utils": torch.utils,
            "torch.utils.cpp_extension": extension,
        }
        versions = {
            "torchrec": "1.8.0",
            "fbgemm-gpu": "1.8.0",
            "dynamicemb": "0.1.0",
            "tensordict": "0.15.4",
            "pyvers": "0.2.3",
        }
        import_without_gpu = block_runtime_imports(builtins.__import__)

        def import_dependencies(name, *args, **kwargs):
            if name == "tensordict" and not tensordict_available:
                raise ModuleNotFoundError("No module named 'pyvers'")
            return import_without_gpu(name, *args, **kwargs)

        def compiler_output(command, **kwargs):
            self.assertEqual(command[0], "/toolkit/bin/nvcc")
            if command[1] == "--version":
                return "Cuda compilation tools, release 13.3, V13.3.1"
            self.assertEqual(command[1], "--list-gpu-code")
            return "sm_103\n"

        with tempfile.TemporaryDirectory() as directory:
            specs = {}
            for name in (
                "torchrec",
                "fbgemm_gpu",
                "dynamicemb",
                "dynamicemb_extensions",
            ):
                path = Path(directory) / (name + ".so")
                path.touch()
                specs[name] = SimpleNamespace(origin=str(path))
            output = io.StringIO()
            with patch.dict(sys.modules, modules), patch.object(
                sys, "argv", ["check_gb300.py", "--build-only"]
            ), patch.object(
                subprocess, "check_output", side_effect=compiler_output
            ), patch.object(
                importlib.metadata, "version", side_effect=versions.__getitem__
            ), patch.object(
                importlib.util, "find_spec", side_effect=specs.__getitem__
            ), patch.object(
                builtins,
                "__import__",
                side_effect=import_dependencies,
            ), redirect_stdout(
                output
            ):
                script = str(Path(__file__).with_name("check_gb300.py"))
                if tensordict_available:
                    runpy.run_path(script, run_name="__main__")
                else:
                    with self.assertRaisesRegex(ModuleNotFoundError, "pyvers"):
                        runpy.run_path(script, run_name="__main__")
            if tensordict_available:
                self.assertTrue(
                    '"scope": "build/metadata/artifacts"' in output.getvalue()
                )


if __name__ == "__main__":
    unittest.main()
