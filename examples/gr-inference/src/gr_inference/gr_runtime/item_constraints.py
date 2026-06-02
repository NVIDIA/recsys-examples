"""Constrained item-token decoding helpers."""

from __future__ import annotations

from collections.abc import Iterable as IterableABC
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
import json
from pathlib import Path
from threading import RLock
from typing import Any, Iterable, Mapping

from gr_inference.gr_runtime.generation import GRGenerationState


@dataclass
class TokenTrieNode:
    children: dict[int, "TokenTrieNode"] = field(default_factory=dict)
    terminal: bool = False
    item_ids: list[Any] = field(default_factory=list)


@dataclass(frozen=True)
class SemanticItem:
    """One catalog item represented by a semantic token path."""

    item_id: Any
    token_ids: tuple[int, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SemanticItemCatalog:
    """Validated semantic-id catalog used to build item constraints."""

    items: tuple[SemanticItem, ...]
    source: str | None = None

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        *,
        item_id_field: str = "item_id",
        token_ids_field: str = "token_ids",
        metadata_field: str | None = "metadata",
        vocab_size: int | None = None,
        allow_duplicate_item_ids: bool = False,
        allow_duplicate_token_paths: bool = False,
        source: str | None = None,
    ) -> "SemanticItemCatalog":
        if vocab_size is not None and vocab_size <= 0:
            raise ValueError("vocab_size must be positive when provided")

        items: list[SemanticItem] = []
        seen_item_ids: set[Any] = set()
        seen_paths: dict[tuple[int, ...], Any] = {}
        for row_number, record in enumerate(records, start=1):
            item_id = _record_field(record, item_id_field, row_number=row_number)
            token_ids = _coerce_token_ids(
                _record_field(record, token_ids_field, row_number=row_number),
                row_number=row_number,
                vocab_size=vocab_size,
            )
            if not allow_duplicate_item_ids and item_id in seen_item_ids:
                raise ValueError(f"duplicate item_id at row {row_number}: {item_id!r}")
            previous_item_id = seen_paths.get(token_ids)
            if previous_item_id is not None and not allow_duplicate_token_paths:
                raise ValueError(
                    "duplicate semantic token path at row "
                    f"{row_number}: {token_ids!r} already used by {previous_item_id!r}"
                )
            seen_item_ids.add(item_id)
            seen_paths[token_ids] = item_id
            items.append(
                SemanticItem(
                    item_id=item_id,
                    token_ids=token_ids,
                    metadata=_record_metadata(
                        record,
                        item_id_field=item_id_field,
                        token_ids_field=token_ids_field,
                        metadata_field=metadata_field,
                        row_number=row_number,
                    ),
                )
            )

        if not items:
            raise ValueError("semantic item catalog must contain at least one item")
        return cls(items=tuple(items), source=source)

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        item_id_field: str = "item_id",
        token_ids_field: str = "token_ids",
        metadata_field: str | None = "metadata",
        vocab_size: int | None = None,
        allow_duplicate_item_ids: bool = False,
        allow_duplicate_token_paths: bool = False,
    ) -> "SemanticItemCatalog":
        catalog_path = Path(path)
        records: list[Mapping[str, Any]] = []
        with catalog_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON in semantic item catalog at line {line_number}: {exc}"
                    ) from exc
                if not isinstance(record, MappingABC):
                    raise ValueError(
                        f"semantic item catalog line {line_number} must be a JSON object"
                    )
                records.append(record)
        return cls.from_records(
            records,
            item_id_field=item_id_field,
            token_ids_field=token_ids_field,
            metadata_field=metadata_field,
            vocab_size=vocab_size,
            allow_duplicate_item_ids=allow_duplicate_item_ids,
            allow_duplicate_token_paths=allow_duplicate_token_paths,
            source=str(catalog_path),
        )

    @property
    def item_count(self) -> int:
        return len(self.items)

    def trie(self) -> "TokenTrie":
        return TokenTrie.from_items(self.items)

    def provider(
        self,
        *,
        vocab_size: int,
        eos_token_id: int | None = None,
        allow_eos_for_terminal: bool = True,
    ) -> "TrieItemMaskProvider":
        return TrieItemMaskProvider(
            self.trie(),
            vocab_size=vocab_size,
            eos_token_id=eos_token_id,
            allow_eos_for_terminal=allow_eos_for_terminal,
        )

    def status(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "item_count": self.item_count,
        }


class TrieItemMaskProviderStore:
    """Atomic-ish holder for hot-swapping catalog-backed constraint providers."""

    def __init__(
        self,
        provider: "TrieItemMaskProvider",
        *,
        version: int = 1,
        metadata: Mapping[str, Any] | None = None,
        max_reload_history: int = 16,
    ) -> None:
        self._lock = RLock()
        self._provider = provider
        self._version = int(version)
        self._metadata = dict(metadata or {})
        self._previous_provider: TrieItemMaskProvider | None = None
        self._previous_version: int | None = None
        self._previous_metadata: dict[str, Any] | None = None
        self._reload_history: list[dict[str, Any]] = []
        self._max_reload_history = int(max_reload_history)

    @classmethod
    def from_catalog(
        cls,
        catalog: SemanticItemCatalog,
        *,
        vocab_size: int,
        eos_token_id: int | None = None,
        allow_eos_for_terminal: bool = True,
    ) -> "TrieItemMaskProviderStore":
        return cls(
            catalog.provider(
                vocab_size=vocab_size,
                eos_token_id=eos_token_id,
                allow_eos_for_terminal=allow_eos_for_terminal,
            ),
            metadata=catalog.status(),
        )

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        vocab_size: int,
        eos_token_id: int | None = None,
        allow_eos_for_terminal: bool = True,
        item_id_field: str = "item_id",
        token_ids_field: str = "token_ids",
        metadata_field: str | None = "metadata",
        allow_duplicate_item_ids: bool = False,
        allow_duplicate_token_paths: bool = False,
    ) -> "TrieItemMaskProviderStore":
        catalog = SemanticItemCatalog.from_jsonl(
            path,
            item_id_field=item_id_field,
            token_ids_field=token_ids_field,
            metadata_field=metadata_field,
            vocab_size=vocab_size,
            allow_duplicate_item_ids=allow_duplicate_item_ids,
            allow_duplicate_token_paths=allow_duplicate_token_paths,
        )
        return cls.from_catalog(
            catalog,
            vocab_size=vocab_size,
            eos_token_id=eos_token_id,
            allow_eos_for_terminal=allow_eos_for_terminal,
        )

    def snapshot(self) -> "TrieItemMaskProvider":
        with self._lock:
            return self._provider

    def swap(
        self,
        provider: "TrieItemMaskProvider",
        *,
        metadata: Mapping[str, Any] | None = None,
        operation: str = "swap",
    ) -> int:
        with self._lock:
            previous_version = self._version
            self._previous_provider = self._provider
            self._previous_version = self._version
            self._previous_metadata = dict(self._metadata)
            self._provider = provider
            self._version += 1
            self._metadata = dict(metadata or {})
            self._record_reload_event(
                {
                    "operation": operation,
                    "status": "succeeded",
                    "previous_version": previous_version,
                    "version": self._version,
                    **self._metadata,
                }
            )
            return self._version

    def reload_jsonl(
        self,
        path: str | Path,
        *,
        vocab_size: int,
        eos_token_id: int | None = None,
        allow_eos_for_terminal: bool = True,
        item_id_field: str = "item_id",
        token_ids_field: str = "token_ids",
        metadata_field: str | None = "metadata",
        allow_duplicate_item_ids: bool = False,
        allow_duplicate_token_paths: bool = False,
    ) -> int:
        try:
            catalog = SemanticItemCatalog.from_jsonl(
                path,
                item_id_field=item_id_field,
                token_ids_field=token_ids_field,
                metadata_field=metadata_field,
                vocab_size=vocab_size,
                allow_duplicate_item_ids=allow_duplicate_item_ids,
                allow_duplicate_token_paths=allow_duplicate_token_paths,
            )
            return self.swap(
                catalog.provider(
                    vocab_size=vocab_size,
                    eos_token_id=eos_token_id,
                    allow_eos_for_terminal=allow_eos_for_terminal,
                ),
                metadata=catalog.status(),
                operation="reload_jsonl",
            )
        except Exception as exc:
            with self._lock:
                self._record_reload_event(
                    {
                        "operation": "reload_jsonl",
                        "status": "failed",
                        "version": self._version,
                        "source": str(path),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
            raise

    def rollback(self) -> int:
        with self._lock:
            if self._previous_provider is None or self._previous_version is None:
                raise RuntimeError("no previous catalog version available for rollback")
            target_version = self._previous_version
            current_provider = self._provider
            current_version = self._version
            current_metadata = dict(self._metadata)
            self._provider = self._previous_provider
            self._version += 1
            self._metadata = dict(self._previous_metadata or {})
            self._previous_provider = current_provider
            self._previous_version = current_version
            self._previous_metadata = current_metadata
            self._record_reload_event(
                {
                    "operation": "rollback",
                    "status": "succeeded",
                    "rolled_back_from_version": current_version,
                    "rolled_back_to_version": target_version,
                    "version": self._version,
                    **self._metadata,
                }
            )
            return self._version

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "version": self._version,
                "previous_version": self._previous_version,
                "reload_history": tuple(self._reload_history),
                "last_reload": self._reload_history[-1] if self._reload_history else None,
                **self._metadata,
            }

    def _record_reload_event(self, event: Mapping[str, Any]) -> None:
        self._reload_history.append(dict(event))
        if self._max_reload_history > 0 and len(self._reload_history) > self._max_reload_history:
            del self._reload_history[: len(self._reload_history) - self._max_reload_history]


def _record_field(
    record: Mapping[str, Any],
    field_name: str,
    *,
    row_number: int,
) -> Any:
    if field_name not in record:
        raise ValueError(f"missing required field {field_name!r} at row {row_number}")
    return record[field_name]


def _coerce_token_ids(
    raw_token_ids: Any,
    *,
    row_number: int,
    vocab_size: int | None,
) -> tuple[int, ...]:
    if isinstance(raw_token_ids, (str, bytes)) or not isinstance(raw_token_ids, IterableABC):
        raise ValueError(f"token_ids at row {row_number} must be a non-empty sequence")

    token_ids: list[int] = []
    for token in raw_token_ids:
        if isinstance(token, bool):
            raise ValueError(f"token id at row {row_number} must be an integer")
        try:
            token_id = int(token)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"token id at row {row_number} must be an integer") from exc
        if token_id < 0:
            raise ValueError(f"token id at row {row_number} must be non-negative")
        if vocab_size is not None and token_id >= vocab_size:
            raise ValueError(
                f"token id {token_id} at row {row_number} exceeds vocab_size={vocab_size}"
            )
        token_ids.append(token_id)

    if not token_ids:
        raise ValueError(f"token_ids at row {row_number} must be non-empty")
    return tuple(token_ids)


def _record_metadata(
    record: Mapping[str, Any],
    *,
    item_id_field: str,
    token_ids_field: str,
    metadata_field: str | None,
    row_number: int,
) -> Mapping[str, Any]:
    if metadata_field is None:
        return {
            key: value
            for key, value in record.items()
            if key not in {item_id_field, token_ids_field}
        }
    metadata = record.get(metadata_field, {})
    if not isinstance(metadata, MappingABC):
        raise ValueError(f"metadata field at row {row_number} must be a JSON object")
    return dict(metadata)


class TokenTrie:
    """Trie over tokenized item ids / semantic ids."""

    def __init__(self) -> None:
        self.root = TokenTrieNode()

    @classmethod
    def from_sequences(cls, sequences: Iterable[Iterable[int]]) -> "TokenTrie":
        trie = cls()
        for sequence in sequences:
            trie.insert(sequence)
        return trie

    @classmethod
    def from_items(
        cls,
        items: Iterable[SemanticItem | tuple[Any, Iterable[int]]],
    ) -> "TokenTrie":
        trie = cls()
        for item in items:
            if isinstance(item, SemanticItem):
                trie.insert(item.token_ids, item_id=item.item_id)
            else:
                item_id, sequence = item
                trie.insert(sequence, item_id=item_id)
        return trie

    def insert(self, sequence: Iterable[int], *, item_id: Any | None = None) -> None:
        node = self.root
        seen = False
        for token in sequence:
            seen = True
            token_id = int(token)
            node = node.children.setdefault(token_id, TokenTrieNode())
        if not seen:
            raise ValueError("cannot insert an empty item-token sequence")
        node.terminal = True
        if item_id is not None and item_id not in node.item_ids:
            node.item_ids.append(item_id)

    def allowed_next(self, prefix: Iterable[int] = ()) -> set[int]:
        node = self._find_node(prefix)
        if node is None:
            return set()
        return set(node.children)

    def is_terminal(self, prefix: Iterable[int]) -> bool:
        node = self._find_node(prefix)
        return bool(node is not None and node.terminal)

    def item_ids(self, prefix: Iterable[int]) -> tuple[Any, ...]:
        node = self._find_node(prefix)
        if node is None or not node.terminal:
            return ()
        return tuple(node.item_ids)

    def _find_node(self, prefix: Iterable[int]) -> TokenTrieNode | None:
        node = self.root
        for token in prefix:
            token_id = int(token)
            if token_id not in node.children:
                return None
            node = node.children[token_id]
        return node


class TrieItemMaskProvider:
    """Build torch boolean masks from a TokenTrie and current BeamPath."""

    def __init__(
        self,
        trie: TokenTrie,
        *,
        vocab_size: int,
        eos_token_id: int | None = None,
        allow_eos_for_terminal: bool = True,
    ) -> None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if eos_token_id is not None and not 0 <= eos_token_id < vocab_size:
            raise ValueError("eos_token_id must be within vocabulary")
        self.trie = trie
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.allow_eos_for_terminal = allow_eos_for_terminal

    @property
    def stop_token_ids(self) -> tuple[int, ...]:
        return (self.eos_token_id,) if self.eos_token_id is not None else ()

    def initial_mask(self, logits: Any):
        import torch

        device = getattr(logits, "device", None)
        mask = torch.zeros(self.vocab_size, dtype=torch.bool, device=device)
        for token in self.allowed_next(()):
            if 0 <= token < self.vocab_size:
                mask[token] = True
        return mask

    def step_mask(self, generation: GRGenerationState, logits: Any):
        import torch

        if generation.beam_path.steps_done == 0:
            raise ValueError("beam_path must be initialized before step masks")
        if not hasattr(logits, "shape") or len(logits.shape) != 3:
            raise ValueError("decode logits must be shaped [B, W, vocab]")
        if logits.shape[0] != 1:
            raise ValueError("TrieItemMaskProvider currently supports batch_size=1")

        width = logits.shape[1]
        device = getattr(logits, "device", None)
        mask = torch.zeros((width, self.vocab_size), dtype=torch.bool, device=device)
        for beam in range(width):
            prefix = generation.beam_path.token_trace(beam)
            for token in self.allowed_next(prefix):
                if 0 <= token < self.vocab_size:
                    mask[beam, token] = True
        return mask

    def allowed_next(self, prefix: Iterable[int] = ()) -> set[int]:
        normalized = self._semantic_prefix(prefix)
        if self._has_eos(prefix):
            return set(self.stop_token_ids)

        allowed = self.trie.allowed_next(normalized)
        if (
            self.eos_token_id is not None
            and self.allow_eos_for_terminal
            and self.trie.is_terminal(normalized)
        ):
            allowed.add(self.eos_token_id)
        return allowed

    def is_complete(self, token_ids: Iterable[int]) -> bool:
        return self.trie.is_terminal(self._semantic_prefix(token_ids))

    def resolve_item_ids(self, token_ids: Iterable[int]) -> tuple[Any, ...]:
        return self.trie.item_ids(self._semantic_prefix(token_ids))

    def beam_item_results(
        self,
        beam_path: Any,
        *,
        beam_width: int,
    ) -> tuple[dict[str, Any], ...]:
        results = []
        for beam in range(beam_width):
            token_ids = beam_path.token_trace(beam)
            semantic_token_ids = self._semantic_prefix(token_ids)
            item_ids = self.resolve_item_ids(token_ids)
            results.append(
                {
                    "rank": beam,
                    "token_ids": token_ids,
                    "semantic_token_ids": semantic_token_ids,
                    "is_complete": self.is_complete(token_ids),
                    "item_ids": item_ids,
                    "item_id": item_ids[0] if len(item_ids) == 1 else None,
                }
            )
        return tuple(results)

    def _semantic_prefix(self, token_ids: Iterable[int]) -> tuple[int, ...]:
        values = tuple(int(token) for token in token_ids)
        if self.eos_token_id is None:
            return values
        if self.eos_token_id not in values:
            return values
        return values[: values.index(self.eos_token_id)]

    def _has_eos(self, token_ids: Iterable[int]) -> bool:
        return self.eos_token_id is not None and self.eos_token_id in set(
            int(token) for token in token_ids
        )
