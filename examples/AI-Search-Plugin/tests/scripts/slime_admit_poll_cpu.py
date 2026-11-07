# engine poll unit check: fake HTTP router/engine, versions picked up, minimum kept
import sys, threading, json, time, http.server, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from ai_search_plugin.adapters import slime_admit as sa
V = {"a": 3, "b": 2}
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path == "/workers":
            body = {"workers": [{"url": f"http://127.0.0.1:{e.server_address[1]}"} for e in ENG]}
        else:
            body = {"weight_version": str(V[self.server.name])}
        b = json.dumps(body).encode(); self.send_response(200); self.end_headers(); self.wfile.write(b)
def srv(name):
    s = http.server.HTTPServer(("127.0.0.1", 0), H); s.name = name
    threading.Thread(target=s.serve_forever, daemon=True).start(); return s
ENG = [srv("a"), srv("b")]; R = srv("r")
from types import SimpleNamespace
tk = sa.Tickets(8, 2); tk.v0 = 1; tk.vmax = 1
os.environ["ASRT_TICKET_POLL_S"] = "0.2"
tk.start_engine_poll(SimpleNamespace(sglang_router_ip="127.0.0.1", sglang_router_port=R.server_address[1]))
time.sleep(1.0)
assert tk.v_eng == 2 and tk.rel() == 1, (tk.v_eng, tk.rel())
V["b"] = 4; time.sleep(1.0)
assert tk.v_eng == 3 and tk.rel() == 2, (tk.v_eng, tk.rel())
print("poll ok: min engine version tracked", tk.v_eng, "rel", tk.rel())
