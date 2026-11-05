# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise setup's compiler-target selection without importing torch or running pip."""

import ast
import os
import re
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch


class BuildArchsTest(unittest.TestCase):
    def setUp(self):
        setup_path = Path(__file__).resolve().parents[1] / "setup.py"
        tree = ast.parse(setup_path.read_text())
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "get_cuda_archs"
        )
        namespace = {
            "os": os,
            "re": re,
            "subprocess": subprocess,
            "CUDA_HOME": "/toolkit",
        }
        exec(
            compile(
                ast.Module(body=[function], type_ignores=[]), str(setup_path), "exec"
            ),
            namespace,
        )
        self.select = namespace["get_cuda_archs"]

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


if __name__ == "__main__":
    unittest.main()
