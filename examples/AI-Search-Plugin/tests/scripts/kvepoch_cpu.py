"""kvepoch: new requests' radix extra_key carries a weight-update epoch that advances around every update;
requests before / during / after an update never share a key; the (mixin) update method still runs. No SGLang needed.
Run: python3 tests/scripts/kvepoch_cpu.py"""
import asyncio, dataclasses, os, types
os.environ["ASRT_SGL_HOOKS"] = "kvepoch"
from ai_search_plugin.adapters import slime_sglang_hooks as H

@dataclasses.dataclass(frozen=True)
class Tok:
    ids: tuple
    extra_key: str = None

calls = []
class Mixin:
    async def update_weights_from_distributed(self, obj, request=None):
        calls.append(obj); await asyncio.sleep(0); return True, "ok"
class TokenizerManager(Mixin):
    def _create_tokenized_object(self, obj, text, ids):
        return Tok(tuple(ids), obj.get("extra_key"))

mod = types.ModuleType("fake_tm"); mod.TokenizerManager = TokenizerManager
H.patch_kvepoch(mod)
tm = TokenizerManager()
async def main():
    a = tm._create_tokenized_object({}, "", [1, 2])
    b = tm._create_tokenized_object({"extra_key": "lora7"}, "", [1, 2])
    r = await tm.update_weights_from_distributed("bucket0")
    c = tm._create_tokenized_object({}, "", [1, 2])
    assert r == (True, "ok") and calls == ["bucket0"], (r, calls)
    assert a.extra_key == "@wv0" and b.extra_key == "lora7@wv0" and c.extra_key == "@wv2", (a, b, c)
    assert a.extra_key != c.extra_key
    print("ok", a.extra_key, b.extra_key, c.extra_key)
asyncio.run(main())
