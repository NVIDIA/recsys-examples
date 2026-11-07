# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Entry point for ``VERL_USE_EXTERNAL_MODULES=...,ai_search_plugin.prefix_tree.verl_plugin`` (with ASRT_PREFIX_TREE=1).

verl imports the listed modules in every process.  This one registers post-import hooks only, so it costs nothing in
processes that never import the patched modules:
  * driver: ``verl.trainer.ppo.v1.trainer_base`` -> group-level DP balancing (``adapters.verl_driver``);
  * training workers: ``verl.workers.engine.megatron.transformer_impl`` -> tree packing in the Megatron engine
    (``adapters.verl``; the model-side wrappers install themselves on the first micro-batch with siblings).
ASRT_PREFIX_TREE unset / 0 = nothing is installed.
"""
import os

from . import install_verl

installed = install_verl()
if installed and os.environ.get('ASRT_VERL_QUIET', '0') != '1':
    print(f'[asrt] ai_search_plugin prefix_tree verl pid={os.getpid()} hooks={installed}', flush=True)
