"""Build the ai_search_plugin recipe for a slime job.

    python examples/slime/runtime_env.py --out runtime_env.json --args-out plugin_args.txt \
        --generate my_agent.generate --tools my_agent.search --log-dir /tmp/asp-logs

Writes a Ray runtime-env JSON (env_vars incl. PYTHONPATH with the SGLang-engine sitecustomize first) and the slime
arguments to add, and prints the trainer launcher.  See launch_slime.sh for how they are used.
NOTE: illustrative; not exercised against a live slime cluster in this repository's tests.
"""
import argparse
import json
import os

import ai_search_plugin as asp

ap = argparse.ArgumentParser()
ap.add_argument("--components", default="default", help='"default", "all", or a comma list ("default,-tool_hedge")')
ap.add_argument("--generate", default="", help="module.attr of your slime custom generate function (for toolcache)")
ap.add_argument("--tools", default="", help="comma list of module.attr read-only tool functions (toolcache / hedge)")
ap.add_argument("--num-experts", type=int, default=0, help="MoE expert count (r3_transfer)")
ap.add_argument("--log-dir", default="")
ap.add_argument("--out", default="runtime_env.json")
ap.add_argument("--args-out", default="plugin_args.txt")
a = ap.parse_args()

opts = {"toolcache_generate": a.generate, "tool_targets": a.tools, "log_dir": a.log_dir}
if a.num_experts:
    opts["num_experts"] = a.num_experts
plan = asp.plan("slime", a.components, {k: v for k, v in opts.items() if v})
env = plan.full_env(os.environ.get("PYTHONPATH", ""))
json.dump({"env_vars": env}, open(a.out, "w"), indent=1)
open(a.args_out, "w").write(" ".join(plan.args) + "\n")
print(plan.shell())
print(f"\nwrote {a.out} and {a.args_out}; trainer launcher: {' '.join(plan.launcher) or 'python3'} <slime>/train.py")
