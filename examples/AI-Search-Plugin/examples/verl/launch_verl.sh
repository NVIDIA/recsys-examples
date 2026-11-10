#!/usr/bin/env bash
# Minimal verl launch with ai_search_plugin (illustrative; adapt the usual verl arguments).
# verl imports every module in VERL_USE_EXTERNAL_MODULES in every process; the plugin entry installs the features
# listed in ASRT_VERL as post-import patches.  The variables must reach every Ray worker: export them before
# `ray start` on every node, or pass them as runtime_env env_vars.
set -euo pipefail

# 1. print the recipe and export it (default components; see `python -m ai_search_plugin list`)
python3 -m ai_search_plugin plan verl default max_staleness=2 log_dir=/tmp/asp-logs
eval "$(python3 -m ai_search_plugin plan verl default max_staleness=2 log_dir=/tmp/asp-logs | grep '^export ')"
# this exports e.g.
#   VERL_USE_EXTERNAL_MODULES=ai_search_plugin.verl_plugin
#   ASRT_VERL=sglhooks,inplace,ceiling,toolcache,response_logits
#   ASRT_SGL_HOOKS=nosync,fastload  ASRT_MAX_STALENESS=2  ASRT_LOG_DIR=/tmp/asp-logs

# 2. launch verl unchanged (async trainer + SGLang rollout; response_logits needs the Megatron backend)
python3 -m verl.trainer.main_ppo ${VERL_ARGS:-}
