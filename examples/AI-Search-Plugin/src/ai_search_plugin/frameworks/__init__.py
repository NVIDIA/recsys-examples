"""Framework entry points.

Each module turns a component selection into a :class:`Plan` (environment variables, framework CLI arguments,
launcher prefix) and knows the hook files that framework already accepts:

* slime: ``--custom-megatron-init-path`` (training actors), ``--rollout-function-path`` (rollout manager),
  ``--custom-generate-function-path`` (per-sample generate), and a ``sitecustomize`` directory first on
  ``PYTHONPATH`` for the SGLang engine processes.
* verl: ``VERL_USE_EXTERNAL_MODULES=ai_search_plugin.verl_plugin`` (verl imports it in every process).
"""
from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from typing import Dict, List

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # .../ai_search_plugin
SRC_ROOT = os.path.dirname(PKG_DIR)                                       # directory that contains ai_search_plugin/
SGL_BOOT_DIR = os.path.join(PKG_DIR, "adapters", "sgl_boot")


@dataclass
class Plan:
    framework: str
    components: List[str]
    env: Dict[str, str] = field(default_factory=dict)          # every Ray process of the job needs these
    pythonpath_prepend: List[str] = field(default_factory=list)  # put first on PYTHONPATH (engine sitecustomize)
    args: List[str] = field(default_factory=list)              # framework CLI arguments to add / replace
    launcher: List[str] = field(default_factory=list)          # prefix for the training entry script (slime)
    skipped: Dict[str, str] = field(default_factory=dict)      # component -> reason it was not enabled
    notes: List[str] = field(default_factory=list)

    def full_env(self, base_pythonpath: str = "") -> Dict[str, str]:
        """``env`` plus PYTHONPATH (prepend dirs + ``base_pythonpath``) - what to pass as Ray runtime_env env_vars."""
        env = dict(self.env)
        if self.pythonpath_prepend:
            parts = list(self.pythonpath_prepend) + [p for p in base_pythonpath.split(":") if p]
            env["PYTHONPATH"] = ":".join(dict.fromkeys(parts))
        return env

    def shell(self) -> str:
        """A human-readable recipe (export lines + CLI arguments)."""
        lines = [f"# ai_search_plugin plan for {self.framework}: {', '.join(self.components) or '(nothing)'}"]
        for k, v in self.skipped.items():
            lines.append(f"# skipped {k}: {v}")
        for n in self.notes:
            lines.append(f"# note: {n}")
        if self.pythonpath_prepend:
            lines.append(f"export PYTHONPATH={shlex.quote(':'.join(self.pythonpath_prepend))}:$PYTHONPATH")
        for k, v in sorted(self.env.items()):
            lines.append(f"export {k}={shlex.quote(v)}")
        if self.launcher:
            lines.append("# launch the trainer as:  " + " ".join(self.launcher) + " <path/to/slime>/train.py <args>")
        if self.args:
            lines.append("# add to the framework arguments:")
            lines.append("  " + " ".join(shlex.quote(a) for a in self.args))
        return "\n".join(lines)


def apply_env(plan: Plan, environ=None) -> None:
    """Write ``plan.env`` (and the PYTHONPATH prepend) into ``environ`` (default ``os.environ``)."""
    environ = os.environ if environ is None else environ
    for k, v in plan.env.items():
        environ[k] = v
    if plan.pythonpath_prepend:
        environ["PYTHONPATH"] = plan.full_env(environ.get("PYTHONPATH", ""))["PYTHONPATH"]
