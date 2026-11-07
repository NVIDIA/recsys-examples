"""Enable the plugin from a Python launcher that starts slime itself (same effect as launch_slime.sh).

install() writes the env into os.environ; anything this process spawns (and a Ray job started from here with
runtime_env={"env_vars": plan.full_env(...)}) inherits it.  The patches themselves install inside slime's processes
through the hooks in plan.args.  NOTE: illustrative; not run against slime in this repository's tests.
"""
import ai_search_plugin as asp

plan = asp.install(
    framework="slime",
    components=["ticket_admission", "pool_fill", "prefetch", "weight_sync", "sampler_nosync", "response_logits"],
    options={"max_staleness": 2, "log_dir": "/tmp/asp-logs"},
)
print(plan.shell())
slime_argv = ["--your", "--usual", "--slime", "--args"] + plan.args
print("launch:", " ".join(plan.launcher or ["python3"]), "<slime>/train.py", " ".join(slime_argv))
