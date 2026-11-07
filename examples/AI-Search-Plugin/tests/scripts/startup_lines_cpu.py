"""Startup lines of tool_hedge (via slime_toolcache.install) and the ticket DEFER switch.
    python3 tests/cpu/test_startup_lines_cpu.py"""
import contextlib
import io
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

# fake tool module, listed as a hedge target
m = types.ModuleType("my_agent")


async def search(*a, **k):
    return "ok"


m.search = search
sys.modules["my_agent"] = m
os.environ["ASRT_TOOL_HEDGE"] = "auto"
os.environ["ASRT_TOOL_HEDGE_TARGETS"] = "my_agent.search"
from ai_search_plugin.adapters import slime_toolcache as tc  # noqa: E402
from ai_search_plugin.adapters import tool_hedge as th  # noqa: E402
from ai_search_plugin.adapters import slime_admit as sa  # noqa: E402

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    tc.install(["my_agent.search"])
out = buf.getvalue()
assert "[asrt] tool_hedge: on targets=my_agent.search delay=auto(q0.9" in out and "max_extra=2" in out, out
os.environ["ASRT_TOOL_HEDGE"] = "7.5"
assert th.describe(["x.y"]).startswith("on targets=x.y delay=7.5 s"), th.describe(["x.y"])
os.environ["ASRT_TOOL_HEDGE"] = "0"
assert th.describe().startswith("off"), th.describe()

os.environ.pop("ASRT_TICKET_DEFER", None)
os.environ.pop("ASRT_TICKET_DROP_S", None)
l1 = sa.defer_line()
assert l1.startswith("[asrt] ticket defer: on (ASRT_TICKET_DEFER=<default 1>)") and "ASRT_TICKET_DROP_S=auto" in l1, l1
os.environ["ASRT_TICKET_DEFER"] = "0"
l2 = sa.defer_line()
assert l2.startswith("[asrt] ticket defer: off (ASRT_TICKET_DEFER=0)") and "ASRT_TICKET_DROP_S=<off>" in l2, l2
print("OK\n" + out.strip() + "\n" + l1 + "\n" + l2)
