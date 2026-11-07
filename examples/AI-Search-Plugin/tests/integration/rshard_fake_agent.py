"""Test helpers for test_slime_rshard_ray.py: deterministic fake SGLang engines + fake router (aiohttp, own process)
and a multi-turn agent (custom generate: model turn -> tool observation (loss-masked) -> ... -> final turn) and reward.
Engine outputs are a pure function of the request (input ids, max_new_tokens), so stock and sharded slime must
produce bit-identical samples."""
import asyncio
import hashlib
import random
import sys

TURNS = 3


def _toks(input_ids, n):
    h = (sum(input_ids[-16:]) * 2654435761 + len(input_ids)) & 0xFFFFFFFF
    out = []
    for i in range(n):
        h = (h * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(1000 + h % 5000)
    return out


def serve(n_engines: int, base_port: int):
    """Run E engines on base_port+1.. and a router on base_port (round robin /generate, /workers)."""
    from aiohttp import ClientSession, web

    stats = {"inflight": [0] * n_engines, "max_inflight": [0] * n_engines, "calls": [0] * n_engines,
             "aborts": [0] * n_engines}
    seen = set()

    def engine_app(e):
        async def generate(req):
            body = await req.json()
            ids = body["input_ids"]
            n = int(body["sampling_params"]["max_new_tokens"])
            stats["inflight"][e] += 1
            stats["max_inflight"][e] = max(stats["max_inflight"][e], stats["inflight"][e])
            stats["calls"][e] += 1
            try:
                await asyncio.sleep(random.uniform(0.005, 0.03))
            finally:
                stats["inflight"][e] -= 1
            key = int(hashlib.md5(f"{ids}|{n}".encode()).hexdigest()[:8], 16)  # unique per request content
            if key % 11 == 0 and key not in seen:  # first attempt of some requests: aborted (weight-sync style)
                seen.add(key)
                stats["aborts"][e] += 1
                return web.json_response({"text": "", "meta_info": {
                    "output_token_logprobs": [], "finish_reason": {"type": "abort"}, "completion_tokens": 0,
                    "prompt_tokens": len(ids), "cached_tokens": 0, "weight_version": "7", "e2e_latency": 0.0}})
            t = _toks(ids, n)
            lp = [[-((x % 97) / 97.0), x, None] for x in t]
            return web.json_response({"text": " ".join(map(str, t)), "meta_info": {
                "output_token_logprobs": lp, "finish_reason": {"type": "length", "length": n},
                "completion_tokens": n, "prompt_tokens": len(ids), "cached_tokens": 0, "weight_version": "7",
                "e2e_latency": 0.01}})

        async def metrics(_):
            return web.Response(text=f'sglang:num_running_reqs{{tp_rank="0"}} {stats["inflight"][e]}\n'
                                     f'sglang:num_queue_reqs{{tp_rank="0"}} 0\n')

        async def st(_):
            return web.json_response({k: v[e] for k, v in stats.items()})

        app = web.Application(client_max_size=1 << 26)
        app.add_routes([web.post("/generate", generate), web.get("/metrics", metrics), web.get("/stats", st)])
        return app

    rr = {"i": 0}

    async def r_generate(req):
        body = await req.read()
        e = rr["i"] % n_engines
        rr["i"] += 1
        async with ClientSession() as s:
            async with s.post(f"http://127.0.0.1:{base_port + 1 + e}/generate", data=body,
                              headers={"Content-Type": "application/json"}) as r:
                return web.Response(body=await r.read(), content_type="application/json")

    async def rstats(_):
        return web.json_response({"router_generate": rr["i"]})

    async def workers(_):
        return web.json_response({"workers": [{"id": str(e), "url": f"http://127.0.0.1:{base_port + 1 + e}"}
                                              for e in range(n_engines)]})

    async def main():
        runners = []
        router = web.Application(client_max_size=1 << 26)
        router.add_routes([web.post("/generate", r_generate), web.get("/workers", workers), web.get("/rstats", rstats)])
        for port, app in [(base_port, router)] + [(base_port + 1 + e, engine_app(e)) for e in range(n_engines)]:
            r = web.AppRunner(app)
            await r.setup()
            await web.TCPSite(r, "127.0.0.1", port).start()
            runners.append(r)
        print("READY", flush=True)
        while True:
            await asyncio.sleep(3600)

    asyncio.run(main())


async def generate(args, sample, sampling_params):
    """Multi-turn agent with loss-masked tool observations (search-agent-like), deterministic."""
    from slime.rollout.sglang_rollout import GenerateState, _prepare_prompt_ids
    from slime.utils.http_utils import post
    from slime.utils.types import Sample

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    if not sample.tokens:
        sample.tokens = _prepare_prompt_ids(sample, state.tokenizer, state.processor)
    for turn in range(TURNS):
        sp = {**sampling_params, "max_new_tokens": 8 + (sample.index % 5) + 4 * turn}
        out = await post(url, {"input_ids": sample.tokens, "sampling_params": sp, "return_logprob": True})
        mi = out["meta_info"]
        if mi["finish_reason"]["type"] == "abort":
            sample.status = Sample.Status.ABORTED
            return sample
        toks = [x[1] for x in mi["output_token_logprobs"]]
        lps = [x[0] for x in mi["output_token_logprobs"]]
        sample.append_response_tokens(args, tokens=toks, log_probs=lps, trainable=True, meta_info=mi, text=out["text"])
        if turn == TURNS - 1:
            break
        await asyncio.sleep(0.002)  # the "tool"
        obs = [7, 8, 9] + [t % 100 for t in toks[:5]]
        sample.append_response_tokens(args, tokens=obs, trainable=False)
    sample.status = Sample.Status.COMPLETED
    return sample


async def reward(args, sample, **kw):
    return float(sum(sample.tokens[-4:]) % 13) / 13.0


if __name__ == "__main__":
    serve(int(sys.argv[1]), int(sys.argv[2]))
