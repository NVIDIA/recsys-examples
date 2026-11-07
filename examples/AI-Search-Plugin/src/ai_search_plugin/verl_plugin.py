"""Entry point for ``VERL_USE_EXTERNAL_MODULES=ai_search_plugin.verl_plugin``.

verl imports this module from ``verl/__init__.py`` in every process.  It installs the
features listed in ``ASRT_VERL`` (written by ``ai_search_plugin.install(framework="verl")`` /
``ai_search_plugin.frameworks.verl.plan``).  The features are post-import hooks, so verl modules imported later are
patched when they load.  ``ASRT_VERL`` unset / ``off`` = plain verl.
"""
import os

from .adapters import verl as _verl

installed = _verl.install()
if installed and os.environ.get("ASRT_VERL_QUIET", "0") != "1":
    print(f"[asrt] ai_search_plugin verl pid={os.getpid()} features={sorted(installed)}", flush=True)
