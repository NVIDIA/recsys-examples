"""Enable the plugin for verl from Python (same effect as launch_verl.sh).

Call this in the driver BEFORE ray.init / before verl starts its workers, so the env reaches them (or pass
plan.full_env() as runtime_env env_vars).  NOTE: illustrative; not run against verl in this repository's tests.
"""
import ai_search_plugin as asp

plan = asp.install(framework="verl", components="default,-verl_inplace,verl_kvfresh",
                   options={"max_staleness": 2})
print(plan.shell())
# then start verl as usual, e.g.:  from verl.trainer.main_ppo import main; main()
