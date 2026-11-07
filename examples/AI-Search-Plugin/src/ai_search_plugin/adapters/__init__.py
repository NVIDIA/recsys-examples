"""Ported component modules (monkey patches + env switches).

Each module imports its framework (slime / verl / SGLang / Megatron / torch) lazily, inside its install functions,
so every module here can be imported - and most of them unit-tested - on a CPU-only machine without any framework
installed.  Use :func:`ai_search_plugin.install` (or the framework entry points in :mod:`ai_search_plugin.frameworks`)
rather than switching these modules on by hand; the env variables they read (``ASRT_*``) are documented in each
module docstring and in docs/COMPONENTS.md.
"""
