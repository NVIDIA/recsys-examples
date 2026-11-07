"""DirectRPC: weight-sync control calls posted straight to engine HTTP servers (CPU, no slime/ray).

Checks: the JSON body equals what slime's SGLangEngine._make_request would post; node_rank != 0 engines are
skipped (slime's actor no-ops there); a late engine is named (rpc_skew / worst); HTTP errors propagate.
Run: python3 tests/scripts/slime_ws_direct_cpu.py
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ai_search_plugin.adapters import slime_weightsync as ws

got = []
DELAY = {}


def make_server(tag):
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
            got.append((tag, self.path, body))
            time.sleep(DELAY.get(tag, 0.0))
            code = 400 if body.get("fail") else 200
            out = json.dumps({"success": code == 200, "message": f"{tag} ok"}).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main():
    s0, s1 = make_server("e0"), make_server("e1")
    d = ws.DirectRPC([f"http://127.0.0.1:{s0.server_port}", None, f"http://127.0.0.1:{s1.server_port}"])

    class Prof:
        t = {}

        def add(self, k, v, n=0):
            self.t[k] = self.t.get(k, 0) + v

    import torch

    payload = {"names": ["a", "b"], "dtypes": ws._jsonable_dtypes([torch.bfloat16, torch.float32]),
               "shapes": ws._jsonable_shapes([torch.Size([2, 3]), (4,)]), "group_name": "slime-pp_0",
               "flush_cache": False, "weight_version": "3", "load_format": "flattened_bucket"}
    assert payload["dtypes"] == ["bfloat16", "float32"] and payload["shapes"] == [[2, 3], [4]]
    DELAY["e1"] = 0.3
    p = Prof()
    out = d.gather(d.submit([0, 1, 2], "update_weights_from_distributed", payload), p, "bucket")
    assert out[1] is None and out[0]["success"] and out[2]["message"] == "e1 ok", out
    assert sorted(t for t, _, _ in got) == ["e0", "e1"]  # engine 1 (node_rank != 0) not called
    assert all(path == "/update_weights_from_distributed" and body == payload for _, path, body in got)
    assert p.t["rpc_skew"] > 0.2 and d.worst[1] == 2 and d.worst[2] == "bucket", (p.t, d.worst)
    got.clear()
    DELAY.clear()
    d.gather(d.submit([0, 2], "pause_generation", {"mode": "in_place"}))
    assert [b for _, _, b in got] == [{"mode": "in_place"}] * 2
    try:
        d.gather(d.submit([0], "update_weights_from_distributed", {"fail": 1}))
        raise AssertionError("HTTP 400 must raise like slime's _make_request")
    except Exception as e:
        assert "400" in str(e), e
    print("ok")


if __name__ == "__main__":
    main()
