"""slime_toolcache: a final tool error document (HTTP 200 + error body) is returned but not cached; the next
identical query calls the tool again; normal answers stay cached; errors_not_cached counts it.
Run: python3 tests/scripts/toolcache_nocache_err_cpu.py"""
import asyncio
from ai_search_plugin.adapters import slime_toolcache as tc

n = {"q": 0}
async def tool(q):
    n["q"] += 1
    if q == "bad" and n["q"] == 1:
        return "Doc 1(Title: tool_error) Error: search service failed after 4 attempts"
    return f"Doc 1(Title: ok) {q}"

async def main():
    c = tc.cached(tool)
    assert "tool_error" in await c("bad")
    assert await c("bad") == "Doc 1(Title: ok) bad" and n["q"] == 2      # re-requested, not served from cache
    assert await c("bad") == "Doc 1(Title: ok) bad" and n["q"] == 2      # success is cached
    assert await c("good") == "Doc 1(Title: ok) good" and await c("good") and n["q"] == 3
    st = c.stats(); assert st["errors_not_cached"] == 1 and st["hits"] == 2, st
    print("ok", st)
asyncio.run(main())
