"""slime_rshard on slime's REAL fully-async worker + REAL generate_and_rm(_group) + Ray shard actors, against
deterministic fake SGLang engines (rshard_fake_agent.py: E engines + a round-robin router, some first attempts
aborted like a weight sync does).  Multi-turn custom generate with loss-masked tool tokens, custom reward.

Checks: (1) stock and sharded runs return bit-identical samples for every group (tokens, logprobs, loss masks,
response, rewards, weight versions, status); (2) sharded: no /generate goes through the router, every engine is used,
per-engine in-flight never exceeds ASRT_RS_INFLIGHT; (3) aborted groups are requeued and finished (same as stock);
(4) a killed shard falls back to the stock path without losing groups; (5) ASRT_RS=auto turns itself off for a small
trajectory pool.
Needs the slime image (slime, ray, aiohttp, httpx).  usage (inside the image):
    RS_TEST_TOKENIZER=<any HF tokenizer dir> PYTHONPATH=<repo>/src:<this dir>:<slime checkout> \
        python3 tests/integration/slime_rshard_ray.py
NOT run in this repository's CI (needs slime + Ray + a tokenizer); ported unchanged apart from paths.
"""
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
import urllib.request
from argparse import Namespace

HERE = os.path.dirname(os.path.abspath(__file__))
TOK = os.environ.get("RS_TEST_TOKENIZER", "")   # any HF tokenizer directory (required)
E = 3
GROUPS, N = 40, 4


class Args(Namespace):
    def __getattr__(self, k):  # any slime option the test does not set: None (= slime's "unset")
        if k.startswith("__"):
            raise AttributeError(k)
        return None


def make_args(port):
    return Args(hf_checkpoint=TOK, sglang_router_ip="127.0.0.1", sglang_router_port=port,
                rollout_num_gpus=E, rollout_num_gpus_per_engine=1, sglang_server_concurrency=64,
                sglang_max_running_requests=None, n_samples_per_prompt=N, rollout_batch_size=8,
                rollout_temperature=1.0, rollout_top_p=1.0, rollout_top_k=-1, rollout_max_response_len=512,
                rollout_stop=None, rollout_stop_token_ids=None, rollout_skip_special_tokens=False,
                custom_generate_function_path="rshard_fake_agent.generate", custom_rm_path="rshard_fake_agent.reward",
                group_rm=False, partial_rollout=False, use_rollout_routing_replay=False, sglang_dp_size=None,
                fully_async_group_prefetch_factor=1, use_distributed_post=False, ci_test=False,
                rollout_data_transport="object-store", use_score_centering=False)


class Buffer:
    def __init__(self):
        from slime.utils.types import Sample

        self.q = [[Sample(prompt=f"question number {g} about topic {g % 7}", index=g * N + j, group_index=g,
                          label="x") for j in range(N)] for g in range(GROUPS)]
        self.requeued = 0

    def get_samples(self, n):
        out, self.q = self.q[:n], self.q[n:]
        return out

    def add_samples(self, groups):
        self.requeued += len(groups)
        self.q = list(groups) + self.q


def run_arm(arm, port, out_path):
    """One arm in a fresh process (patches are process-global)."""
    os.environ["ASRT_LOG_DIR"] = os.path.dirname(out_path)
    os.environ["ASRT_RS"] = {"stock": "off", "shard": "on,profile", "kill": "on", "auto": "auto"}[arm]
    os.environ["ASRT_RS_INFLIGHT"] = "4"
    os.environ["ASRT_RS_LOG_S"] = "1"
    import ray

    ray.init(address="local", num_cpus=8, include_dashboard=False, log_to_driver=False,
             runtime_env={"env_vars": {"PYTHONPATH": os.environ.get("PYTHONPATH", "")}})
    from slime.rollout import fully_async_rollout as far
    from slime.utils.http_utils import init_http_client

    from ai_search_plugin.adapters import slime_rshard

    args = make_args(port)
    if os.environ["ASRT_RS"] != "off":
        slime_rshard.install(args)
    if arm == "auto":
        os.environ["ASRT_RS_MIN"] = "100000"
    init_http_client(args)
    buf = Buffer()
    w = far.AsyncRolloutWorker(args, buf, concurrency=args.sglang_server_concurrency * E)
    w.start()
    got, t0, killed = [], time.time(), False
    while len(got) < GROUPS and time.time() - t0 < 300:
        if w.error is not None:
            raise w.error
        got += [g for _, g in w.get_completed_groups()]
        if arm == "kill" and not killed and len(got) >= 5 and slime_rshard._S["pool"] is not None:
            ray.kill(slime_rshard._S["pool"].shards[0], no_restart=True)
            killed = True
        time.sleep(0.05)
    pool = slime_rshard._S["pool"]
    info = {"arm": arm, "n": len(got), "requeued": buf.requeued, "secs": time.time() - t0,
            "pool": None if pool is None else {**pool.cfg, "dead": sorted(pool.dead), "local": pool.local_groups},
            "why_off": slime_rshard._S["why"]}
    w.close()
    rows = {}
    for g in got:
        for s in g:
            rows[s.index] = dict(tokens=list(s.tokens), lp=list(s.rollout_log_probs or []), lm=list(s.loss_mask or []),
                                 resp=s.response, rl=s.response_length, reward=s.reward, wv=list(s.weight_versions),
                                 status=str(s.status), g=s.group_index)
    with open(out_path, "wb") as f:
        pickle.dump({"info": info, "rows": rows}, f)
    ray.shutdown()


def _get(port, path):
    return json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5).read())


def run_all():
    tmp = tempfile.mkdtemp(prefix="rshard-test-")
    res = {}
    for i, arm in enumerate(["stock", "shard", "kill", "auto"]):
        port = 18600 + 20 * i
        srv = subprocess.Popen([sys.executable, os.path.join(HERE, "rshard_fake_agent.py"), str(E), str(port)],
                               stdout=subprocess.PIPE, text=True)
        try:
            assert srv.stdout.readline().strip() == "READY"
            out = os.path.join(tmp, f"{arm}.pkl")
            r = subprocess.run([sys.executable, __file__, "--arm", arm, str(port), out], timeout=600)
            assert r.returncode == 0, f"arm {arm} failed"
            with open(out, "rb") as f:
                res[arm] = pickle.load(f)
            res[arm]["router"] = _get(port, "/rstats")["router_generate"]
            res[arm]["engines"] = [_get(port + 1 + e, "/stats") for e in range(E)]
        finally:
            srv.kill()
        print(arm, res[arm]["info"], "router calls", res[arm]["router"],
              [(x["calls"], x["max_inflight"], x["aborts"]) for x in res[arm]["engines"]], flush=True)
    base = res["stock"]["rows"]
    assert len(base) == GROUPS * N and res["stock"]["info"]["requeued"] > 0, "stock: aborts must requeue groups"
    for arm in ("shard", "kill", "auto"):
        rows = res[arm]["rows"]
        assert rows.keys() == base.keys(), f"{arm}: different sample set"
        bad = [i for i in base if rows[i] != base[i]]
        for i in bad[:3]:
            print(arm, "sample", i, {k: (base[i][k], rows[i][k]) for k in base[i] if base[i][k] != rows[i][k] and k != "tokens"},
                  "tokens len", len(base[i]["tokens"]), len(rows[i]["tokens"]), flush=True)
        if arm == "shard" or not bad:
            print(arm, "differs in", len(bad), "samples", flush=True)
    for arm in ("shard", "kill", "auto"):
        bad = [i for i in base if res[arm]["rows"][i] != base[i]]
        assert not bad, f"{arm}: {len(bad)} samples differ, e.g. {bad[:3]}"
    sh = res["shard"]
    assert sh["router"] == 0, "sharded: generate calls must bypass the router"
    assert all(x["calls"] > 0 for x in sh["engines"]), "sharded: every engine used"
    assert all(x["max_inflight"] <= 4 for x in sh["engines"]), "sharded: per-engine in-flight cap"
    assert sh["info"]["requeued"] > 0
    assert res["stock"]["router"] > 0
    kl = res["kill"]["info"]["pool"]
    assert kl["dead"] == [0] and kl["local"] > 0, "killed shard -> stock fallback"
    assert res["auto"]["info"]["pool"] is None and "auto" in res["auto"]["info"]["why_off"]
    print("ALL OK: stock == sharded == shard-killed == auto-off, bit for bit;", GROUPS * N, "samples", flush=True)


def test_rshard_ray():
    run_all()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--arm":
        run_arm(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    else:
        run_all()
