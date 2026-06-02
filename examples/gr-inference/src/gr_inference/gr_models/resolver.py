"""Resolve local or HuggingFace-hosted model checkpoints."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

SnapshotDownloader = Callable[..., str]

_INDEX_FILES = (
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)
_WEIGHT_GLOBS = (
    "*.safetensors",
    "*.bin",
    "*.pt",
)


def resolve_model_dir(
    *,
    model_dir: str | Path | None = None,
    model: str | Path | None = None,
    default_model: str | None = None,
    revision: str | None = None,
    download_dir: str | Path | None = None,
    downloader: SnapshotDownloader | None = None,
) -> str:
    """Resolve a model reference to a local HuggingFace checkpoint directory.

    ``model_dir`` is an explicit local path and never downloads. ``model`` may
    be either a local directory or a HuggingFace Hub repo id. When ``model`` is a
    repo id, the snapshot is downloaded through ``huggingface_hub``. If
    ``download_dir`` is set, repo snapshots are materialized there instead of
    only using the HuggingFace cache.
    """

    if model_dir:
        return str(validate_local_model_dir(model_dir, label="model_dir"))

    model_ref = str(model or default_model or "").strip()
    if not model_ref:
        raise ValueError("one of model_dir, model, or default_model must be set")

    model_path = Path(model_ref).expanduser()
    if model_path.is_dir():
        return str(validate_local_model_dir(model_path, label="model"))
    if _looks_like_local_path(model_ref):
        raise FileNotFoundError(f"model path does not exist: {model_ref}")

    downloaded = _download_model_snapshot(
        model_ref,
        revision=revision,
        download_dir=download_dir,
        downloader=downloader,
    )
    return str(validate_local_model_dir(downloaded, label=f"model {model_ref!r}"))


def validate_local_model_dir(
    model_dir: str | Path,
    *,
    label: str = "model_dir",
) -> Path:
    """Validate that a directory looks like a HuggingFace model checkpoint."""

    path = Path(model_dir).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {path}")
    if not (path / "config.json").is_file():
        raise ValueError(
            f"{label}={path} does not look like a HuggingFace checkpoint: "
            "missing config.json"
        )
    if not _has_weight_files(path):
        raise ValueError(
            f"{label}={path} does not look like a HuggingFace checkpoint: "
            "missing model weight files (*.safetensors, *.bin, *.pt) or "
            "a sharded weight index"
        )
    return path


def _download_model_snapshot(
    repo_id: str,
    *,
    revision: str | None = None,
    download_dir: str | Path | None = None,
    downloader: SnapshotDownloader | None = None,
) -> str:
    snapshot_download = downloader
    if snapshot_download is None:
        try:
            from huggingface_hub import snapshot_download
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "Downloading MODEL from HuggingFace requires huggingface-hub. "
                "Install it with `pip install huggingface-hub`, or pass "
                "`MODEL_DIR=/path/to/local/checkpoint`."
            ) from exc
    kwargs = {"repo_id": repo_id, "revision": revision}
    if download_dir:
        kwargs["local_dir"] = str(Path(download_dir).expanduser())
    return str(snapshot_download(**kwargs))


def _has_weight_files(path: Path) -> bool:
    if any((path / name).is_file() for name in _INDEX_FILES):
        return True
    return any(next(path.glob(pattern), None) is not None for pattern in _WEIGHT_GLOBS)


def _looks_like_local_path(value: str) -> bool:
    return value.startswith(("/", "./", "../", "~")) or value == "." or value == ".."
