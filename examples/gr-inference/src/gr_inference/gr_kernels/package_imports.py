"""Optional kernel package discovery helpers."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def distribution_available(name: str) -> bool:
    try:
        importlib.metadata.distribution(name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def module_attribute_available(module_name: str, attribute: str) -> bool:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return False
    return hasattr(module, attribute)


def load_trtllm_kernel_registration() -> str | None:
    loaded = import_first_available("gr_inference_trtllm_kernels")
    if loaded is not None:
        return loaded
    loaded = import_distribution_modules("tokenspeed-trtllm-kernel")
    if loaded is not None:
        return loaded
    return import_first_available(
        "tokenspeed_trtllm_kernel",
        "tensorrt_llm",
    )


def import_distribution_modules(distribution_name: str) -> str | None:
    for module_name in distribution_top_level_modules(distribution_name):
        loaded = import_first_available(module_name)
        if loaded is not None:
            return loaded
    return None


def distribution_top_level_modules(distribution_name: str) -> tuple[str, ...]:
    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return ()
    top_level = distribution.read_text("top_level.txt")
    if top_level:
        candidates = tuple(
            line.strip()
            for line in top_level.splitlines()
            if line.strip() and not line.strip().endswith(".dist-info")
        )
        if candidates:
            return candidates
    modules: set[str] = set()
    for file in distribution.files or ():
        parts = tuple(file.parts)
        if not parts:
            continue
        first = parts[0]
        if first.endswith(".dist-info") or first == "__pycache__":
            continue
        if len(parts) > 1:
            modules.add(first)
            continue
        if first.endswith((".py", ".so")):
            modules.add(first.split(".", 1)[0])
    return tuple(sorted(modules))


def import_first_available(*module_names: str) -> str | None:
    for module_name in module_names:
        try:
            importlib.import_module(module_name)
            return module_name
        except Exception:
            continue
    return None
