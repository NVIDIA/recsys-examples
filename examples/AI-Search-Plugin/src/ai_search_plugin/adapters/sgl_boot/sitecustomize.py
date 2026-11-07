"""Engine-side asrt hooks for SGLang processes spawned by a framework (verl: SGLangHttpServer actor ->
scheduler subprocesses). ai_search_plugin.adapters.verl_engine puts this directory first on PYTHONPATH (for slime, put this directory first on
PYTHONPATH yourself, see ai_search_plugin.frameworks.slime). Active only if ASRT_SGL_HOOKS is set; always chains to the system sitecustomize."""
import os, sys

_here = os.path.dirname(os.path.abspath(__file__))
# chain without recursion.  Another sitecustomize that also chains to "the first other
# sitecustomize on sys.path" (e.g. a site shim) ran us again, we ran it again, ... -> RecursionError at
# interpreter start, asrt engine hooks half-installed.  Every sitecustomize file runs at most
# once per process (set on sys); a re-entered copy of this file only continues the chain for its caller (next
# sitecustomize on sys.path that has not run yet), so a third one later on the path still runs.
# Registries (all on sys, per process):
#   _asrt_sitecustomize_ran   shared: files that started (our own path at start) or finished running (others, added
#                             only AFTER they ran - a registry-aware chainer that reads "my path is in the registry" as
#                             "re-entered" must not be told so before it ran: an earlier version did that, and also
#                             trusted that registry for its own re-entry -> asrt hooks silently skipped)
#   _asrt_sgl_boot_started    private: sgl_boot copies that started -> our own re-entry test
#   _asrt_sgl_boot_chained    private: files we chained, marked BEFORE running them (recursion guard for chainers that
#                             know no registry)
_RAN = sys.__dict__.setdefault("_asrt_sitecustomize_ran", set())
_STARTED = sys.__dict__.setdefault("_asrt_sgl_boot_started", set())
_CHAINED = sys.__dict__.setdefault("_asrt_sgl_boot_chained", set())
_me = os.path.abspath(__file__)
_reentered = _me in _STARTED
_STARTED.add(_me)
_RAN.add(_me)
_top = getattr(sys.modules.get("sitecustomize"), "__file__", None)   # the file Python imported first


def _chain_next():
    for _p in sys.path:
        if not _p:
            continue
        _cand = os.path.abspath(os.path.join(_p, "sitecustomize.py"))
        if (_cand in _RAN or _cand in _CHAINED or _cand in _STARTED or (_top and _cand == os.path.abspath(_top))
                or os.path.dirname(_cand) == _here or not os.path.isfile(_cand)):
            continue
        _CHAINED.add(_cand)
        try:
            import runpy
            runpy.run_path(_cand, run_name="sitecustomize_chained")
        except Exception as e:
            print(f"[asrt] chained sitecustomize {_cand} failed: {e!r}", file=sys.stderr)
        _RAN.add(_cand)
        return


_chain_next()
if os.environ.get("ASRT_SGL_HOOKS") and not _reentered:
    try:
        from ai_search_plugin.adapters import slime_sglang_hooks
        slime_sglang_hooks.register_import_hooks()
    except Exception as e:
        # a half-installed hook set must not run silently (seen: RecursionError, then a CUDA fault at the
        # first weight sync).  This sitecustomize also runs in helper interpreters (pip, ray tools, compile workers) that
        # inherit PYTHONPATH: those only log.  A process that turns out to be an SGLang engine (imports
        # sglang.srt.managers.*) is stopped right there with exit 78, unless ASRT_SGL_HOOKS_STRICT=0.
        _why = f"{os.environ.get('ASRT_SGL_HOOKS')}: {e!r}"
        print(f"[asrt] ERROR sgl hooks not installed ({_why})", file=sys.stderr, flush=True)
        if os.environ.get("ASRT_SGL_HOOKS_STRICT", "1") != "0":
            import importlib.abc

            class _AbortOnSGLangEngine(importlib.abc.MetaPathFinder):
                def find_spec(self, name, path, target=None):
                    # only the modules an engine process imports (scheduler subprocess, tokenizer-manager host);
                    # not sglang.srt.managers.io_struct etc., which plain verl processes import too
                    if name in ("sglang.srt.managers.scheduler", "sglang.srt.managers.tokenizer_manager"):
                        print(f"[asrt] ERROR SGLang engine process {os.getpid()} without its asrt hooks ({_why}) - "
                              f"exit 78 (ASRT_SGL_HOOKS_STRICT=0 to run anyway)", file=sys.stderr, flush=True)
                        os._exit(78)
                    return None

            sys.meta_path.insert(0, _AbortOnSGLangEngine())
