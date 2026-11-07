"""asrt prefix/group-affinity load balancer for verl's rollout router (feature
'route' of ai_search_plugin.adapters.verl; see verl_rollout.py). Importing this module
imports verl.workers.rollout.router, so it is only imported when the feature is
on. Ray pickles the class by reference (module path), so the balancer actor
process imports it from here."""
import os

from verl.workers.rollout.router import GlobalRequestLoadBalancer

from .verl_rollout import AffinityState, _log


class AsrtAffinityBalancer(GlobalRequestLoadBalancer):
    """Sticky multi-turn routing like verl; a request id seen for the first time
    goes to argmin(in-flight + uncached_prompt_tokens / tok_eq), where the server
    that last received the same prompt (same asrt_prefix_key) has 0 uncached."""

    def __init__(self, servers, max_cache_size=None, full_determinism=False, **kw):
        if max_cache_size is None:
            super().__init__(servers)
        else:
            super().__init__(servers, max_cache_size=max_cache_size)
        self._asrt = AffinityState()
        self._asrt_n = 0
        self._asrt_every = int(os.environ.get("ASRT_VERL_ROUTE_LOG_EVERY", 2000))

    def require_acquire_fields(self):
        return ["asrt_prefix_key", "asrt_ntok"]

    def acquire_server(self, request_id, asrt_prefix_key=None, asrt_ntok=0, **extra):
        self._asrt_n += 1
        if self._asrt_every and self._asrt_n % self._asrt_every == 0:
            rec = {"acquire": self._asrt_n, **self._asrt.st, "inflight": dict(self._inflight_requests)}
            print(f"[asrt] verl route {rec}", flush=True)
            _log("route", rec)
        sid = self._request_id_to_server.get(request_id) if request_id in self._request_id_to_server else None
        if sid is not None and sid in self._inflight_requests:
            self._asrt.st["sticky"] += 1
            self._inflight_requests[sid] += 1
            return sid, self._servers[sid]
        if not self._inflight_requests:
            raise RuntimeError("No available servers in load balancer")
        sid = self._asrt.pick(dict(self._inflight_requests), asrt_prefix_key, int(asrt_ntok or 0))
        if sid is None:
            return super().acquire_server(request_id)
        self._request_id_to_server[request_id] = sid
        self._inflight_requests[sid] += 1
        return sid, self._servers[sid]

    def get_status(self):
        st = super().get_status()
        st["asrt_route"] = dict(self._asrt.st)
        return st
