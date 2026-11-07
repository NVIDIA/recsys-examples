#!/usr/bin/env bash
# Minimal slime launch with ai_search_plugin (illustrative; adapt paths and the usual slime arguments).
# Requirements: slime's fully-async rollout mode (the plugin's rollout wrappers end in slime's stock
# generate_rollout_fully_async), ai-search-plugin installed (pip install -e .) or on PYTHONPATH in every Ray process.
set -euo pipefail
SLIME=${SLIME:-/path/to/slime}            # slime checkout (contains train.py)

# 1. the plugin recipe: Ray runtime env (env vars + PYTHONPATH) and the extra slime arguments
python3 examples/slime/runtime_env.py --out runtime_env.json --args-out plugin_args.txt \
    --generate my_agent.generate --tools my_agent.search --log-dir /tmp/asp-logs
PLUGIN_ARGS=$(cat plugin_args.txt)
# PLUGIN_ARGS is e.g.
#   --custom-megatron-init-path ai_search_plugin.frameworks.slime.megatron_init
#   --rollout-function-path ai_search_plugin.adapters.slime_rshard.generate_rollout_fully_async
#   --custom-generate-function-path ai_search_plugin.adapters.slime_toolcache.generate
# If you already use --custom-megatron-init-path, pass it as option chain_megatron_init instead.

# 2. the trainer runs through the plugin driver (needed by the `prefetch` component; identical to train.py otherwise)
ray job submit --address="http://127.0.0.1:8265" --runtime-env-json="$(cat runtime_env.json)" -- \
    python3 -m ai_search_plugin.adapters.slime_train "$SLIME/train.py" \
    ${SLIME_ARGS:-} \
    $PLUGIN_ARGS
