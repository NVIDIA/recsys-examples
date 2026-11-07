"""tool_hedge: hung first attempt is overtaken by the hedge; error docs only win if all attempts fail; off = identity.
Run: python3 tests/scripts/tool_hedge_cpu.py"""
import asyncio, os, time
os.environ["ASRT_TOOL_HEDGE"] = "0.2"
from ai_search_plugin.adapters import tool_hedge as th

calls = []
async def tool(q):
    n = len(calls); calls.append(q)
    if q == "hang" and n == 0:
        await asyncio.sleep(5); return "late"
    if q == "err":
        await asyncio.sleep(0.01); return "Doc 1(Title: tool_error) Error: search service failed"
    if q == "err1" and n == 0:
        return "Doc 1(Title: tool_error) Error"
    await asyncio.sleep(0.05); return f"ok:{q}"

async def main():
    h = th.hedged(tool, max_extra=2)
    t0 = time.monotonic(); out = await h("hang"); dt = time.monotonic() - t0
    assert out == "ok:hang" and dt < 1.0 and len(calls) == 2, (out, dt, calls)
    calls.clear(); assert await h("fast") == "ok:fast" and len(calls) == 1
    calls.clear(); out = await h("err"); assert "tool_error" in out and len(calls) == 3, calls
    calls.clear(); out = await h("err1"); assert out == "ok:err1" and len(calls) == 2, (out, calls)
    assert th.hedged(tool, spec=None) is not tool  # env on
    os.environ["ASRT_TOOL_HEDGE"] = "0"; assert th.hedged(tool) is tool
    os.environ.pop("ASRT_TOOL_HEDGE_TARGETS", None)
    assert not th.allowed("my_agent.search") and not th.allowed("tools.run_code")   # no default targets: opt-in per tool
    os.environ["ASRT_TOOL_HEDGE_TARGETS"] = "my.fetch"; assert th.allowed("my.fetch") and not th.allowed("my_agent.search")
    print("ok", th.stats())
asyncio.run(main())
