# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prefix-aware prefill cache for GR serving.

This is a compact radix-style index over token ids.  GR still stores complete
per-request ``PrefillResult`` payloads, but the index borrows the production
parts of SGLang's radix cache that fit GR's dense KV layout: namespace
isolation, optional page-aligned matching, LRU eviction, and token-budget
control.
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from gr_inference.gr_runtime.generation import PrefillResult


@dataclass(frozen=True)
class GRPrefixCacheMatch:
    """Longest prefix match returned by ``GRPromptPrefixCache``."""

    prefix_len: int
    prefill: PrefillResult
    exact: bool
    source_token_count: int


@dataclass(frozen=True)
class _GRPrefixEntry:
    cache_key: tuple[Any, tuple[int, ...]]
    extra_key: Any
    tokens: tuple[int, ...]
    tree_tokens: tuple[int, ...]
    prefill: PrefillResult

    @property
    def token_count(self) -> int:
        return len(self.tokens)

    @property
    def tree_token_count(self) -> int:
        return len(self.tree_tokens)


@dataclass
class _GRPrefixNode:
    key: tuple[int, ...] = ()
    children: dict[Any, "_GRPrefixNode"] = field(default_factory=dict)
    source_entry: _GRPrefixEntry | None = None
    last_access_time: float = field(default_factory=time.monotonic)
    lock_ref: int = 0


class GRPromptPrefixCache:
    """LRU-bounded compressed radix index for prompt prefill results.

    GR does not yet have SGLang's paged KV allocator, so cached payloads remain
    complete cloned prefill results.  The radix tree is still useful for exact
    hits and for guarded partial-prefix experiments, while capacity and
    namespace controls keep the cache bounded and safe for multi-tenant serving.
    """

    def __init__(
        self,
        *,
        max_entries: int | None = 16,
        max_tokens: int | None = None,
        page_size: int = 1,
    ) -> None:
        if max_entries is not None and max_entries < 0:
            raise ValueError("max_entries must be non-negative or None")
        if max_tokens is not None and max_tokens < 0:
            raise ValueError("max_tokens must be non-negative or None")
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        self.max_entries = max_entries
        self.max_tokens = max_tokens
        self.page_size = page_size
        self.entries: OrderedDict[
            tuple[Any, tuple[int, ...]], _GRPrefixEntry
        ] = OrderedDict()
        self.root = _GRPrefixNode()
        self.insertions = 0
        self.evictions = 0
        self.evicted_tokens = 0
        self.skipped_insertions = 0
        self.rebuilds = 0
        self.total_tokens = 0
        self.total_tree_tokens = 0

    def __len__(self) -> int:
        return len(self.entries)

    def clear(self) -> None:
        self.entries.clear()
        self.root = _GRPrefixNode()
        self.total_tokens = 0
        self.total_tree_tokens = 0
        self.rebuilds += 1

    def set_max_entries(self, max_entries: int | None) -> None:
        self.configure(
            max_entries=max_entries,
            max_tokens=self.max_tokens,
            page_size=self.page_size,
        )

    def configure(
        self,
        *,
        max_entries: int | None,
        max_tokens: int | None,
        page_size: int,
    ) -> None:
        if max_entries is not None and max_entries < 0:
            raise ValueError("max_entries must be non-negative or None")
        if max_tokens is not None and max_tokens < 0:
            raise ValueError("max_tokens must be non-negative or None")
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        page_size_changed = page_size != self.page_size
        self.max_entries = max_entries
        self.max_tokens = max_tokens
        self.page_size = page_size
        if page_size_changed:
            self._recompute_tree_tokens()
            self._rebuild_tree()
        elif self._enforce_capacity():
            self._rebuild_tree()

    def insert(
        self,
        input_ids: Any,
        prefill: PrefillResult,
        *,
        extra_key: Any = None,
    ) -> bool:
        if self.max_entries == 0 or self.max_tokens == 0:
            self.skipped_insertions += 1
            return False
        tokens = input_ids_to_token_tuple(input_ids)
        if not tokens:
            self.skipped_insertions += 1
            return False
        if self.max_tokens is not None and len(tokens) > self.max_tokens:
            self.skipped_insertions += 1
            return False

        normalized_extra_key = normalize_prefix_cache_extra_key(extra_key)
        cache_key = (normalized_extra_key, tokens)
        previous = self.entries.get(cache_key)
        if previous is not None:
            self.total_tokens -= previous.token_count
            self.total_tree_tokens -= previous.tree_token_count

        entry = _GRPrefixEntry(
            cache_key=cache_key,
            extra_key=normalized_extra_key,
            tokens=tokens,
            tree_tokens=_page_aligned_tokens(tokens, self.page_size),
            prefill=prefill,
        )
        self.entries[cache_key] = entry
        self.entries.move_to_end(cache_key)
        self.total_tokens += entry.token_count
        self.total_tree_tokens += entry.tree_token_count
        self.insertions += 1
        self._enforce_capacity()
        self._rebuild_tree()
        return True

    def match(
        self, input_ids: Any, *, extra_key: Any = None
    ) -> GRPrefixCacheMatch | None:
        tokens = input_ids_to_token_tuple(input_ids)
        if not tokens or not self.entries:
            return None

        normalized_extra_key = normalize_prefix_cache_extra_key(extra_key)
        exact = self.entries.get((normalized_extra_key, tokens))
        if exact is not None:
            self.entries.move_to_end(exact.cache_key)
            return GRPrefixCacheMatch(
                prefix_len=len(tokens),
                prefill=exact.prefill,
                exact=True,
                source_token_count=len(tokens),
            )

        tree_tokens = _page_aligned_tokens(tokens, self.page_size)
        if not tree_tokens:
            return None

        node = self.root
        position = 0
        at_root = True
        best_entry: _GRPrefixEntry | None = None
        best_len = 0

        while position < len(tree_tokens):
            child = node.children.get(
                _child_lookup_key(
                    tree_tokens,
                    position,
                    extra_key=normalized_extra_key if at_root else None,
                    page_size=self.page_size,
                )
            )
            if child is None:
                break
            common = _common_prefix_len(
                tree_tokens, child.key, position, self.page_size
            )
            if common == 0:
                break
            child.last_access_time = time.monotonic()
            if child.source_entry is not None:
                best_entry = child.source_entry
                best_len = position + common
            if common < len(child.key):
                break
            position += common
            node = child
            at_root = False

        if best_entry is None or best_len <= 0:
            return None
        self.entries.move_to_end(best_entry.cache_key)
        return GRPrefixCacheMatch(
            prefix_len=best_len,
            prefill=best_entry.prefill,
            exact=False,
            source_token_count=best_entry.token_count,
        )

    def status(self) -> dict[str, int | None]:
        return {
            "entries": len(self.entries),
            "max_entries": self.max_entries,
            "max_tokens": self.max_tokens,
            "page_size": self.page_size,
            "insertions": self.insertions,
            "evictions": self.evictions,
            "evicted_tokens": self.evicted_tokens,
            "skipped_insertions": self.skipped_insertions,
            "rebuilds": self.rebuilds,
            "nodes": _count_nodes(self.root),
            "tokens": self.total_tokens,
            "tree_tokens": self.total_tree_tokens,
        }

    def _enforce_capacity(self) -> bool:
        evicted = False
        while self.max_entries is not None and len(self.entries) > self.max_entries:
            _cache_key, entry = self.entries.popitem(last=False)
            self._subtract_entry(entry)
            evicted = True
        while self.max_tokens is not None and self.total_tokens > self.max_tokens:
            _cache_key, entry = self.entries.popitem(last=False)
            self._subtract_entry(entry)
            evicted = True
        return evicted

    def _subtract_entry(self, entry: _GRPrefixEntry) -> None:
        self.total_tokens -= entry.token_count
        self.total_tree_tokens -= entry.tree_token_count
        self.evicted_tokens += entry.token_count
        self.evictions += 1

    def _recompute_tree_tokens(self) -> None:
        rebuilt: OrderedDict[
            tuple[Any, tuple[int, ...]], _GRPrefixEntry
        ] = OrderedDict()
        self.total_tree_tokens = 0
        for cache_key, entry in self.entries.items():
            updated = _GRPrefixEntry(
                cache_key=cache_key,
                extra_key=entry.extra_key,
                tokens=entry.tokens,
                tree_tokens=_page_aligned_tokens(entry.tokens, self.page_size),
                prefill=entry.prefill,
            )
            rebuilt[cache_key] = updated
            self.total_tree_tokens += updated.tree_token_count
        self.entries = rebuilt

    def _rebuild_tree(self) -> None:
        self.root = _GRPrefixNode()
        for entry in self.entries.values():
            if entry.tree_tokens:
                self._insert_entry(entry)
        self.rebuilds += 1

    def _insert_entry(self, entry: _GRPrefixEntry) -> None:
        tokens = entry.tree_tokens
        node = self.root
        position = 0
        at_root = True
        while position < len(tokens):
            lookup_key = _child_lookup_key(
                tokens,
                position,
                extra_key=entry.extra_key if at_root else None,
                page_size=self.page_size,
            )
            child = node.children.get(lookup_key)
            if child is None:
                node.children[lookup_key] = _GRPrefixNode(
                    key=tokens[position:],
                    source_entry=entry,
                )
                return

            common = _common_prefix_len(tokens, child.key, position, self.page_size)
            if common == len(child.key):
                position += common
                node = child
                at_root = False
                continue
            if common == 0:
                raise RuntimeError("prefix cache radix child key collision")

            split = _GRPrefixNode(
                key=child.key[:common],
                source_entry=child.source_entry,
                last_access_time=child.last_access_time,
                lock_ref=child.lock_ref,
            )
            child.key = child.key[common:]
            split.children[
                _child_lookup_key(
                    child.key, 0, extra_key=None, page_size=self.page_size
                )
            ] = child
            node.children[lookup_key] = split

            position += common
            if position == len(tokens):
                split.source_entry = entry
                return

            new_child = _GRPrefixNode(
                key=tokens[position:],
                source_entry=entry,
            )
            split.children[
                _child_lookup_key(
                    new_child.key,
                    0,
                    extra_key=None,
                    page_size=self.page_size,
                )
            ] = new_child
            return

        node.source_entry = entry


def input_ids_to_token_tuple(input_ids: Any) -> tuple[int, ...]:
    """Return a single request's token ids as a flat tuple of Python ints."""

    value = input_ids
    if hasattr(value, "detach"):
        value = value.detach()
        if hasattr(value, "to"):
            value = value.to(device="cpu", non_blocking=False)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        raw = value
    elif isinstance(value, list):
        raw = value
    else:
        raw = [value]
    if raw and isinstance(raw[0], list | tuple):
        if len(raw) != 1:
            raise ValueError(
                "prefix cache expects one request per cached prefill entry"
            )
        raw = raw[0]
    return tuple(int(token) for token in raw)


def normalize_prefix_cache_extra_key(extra_key: Any) -> Any:
    """Normalize optional cache namespace data into a stable hashable key."""

    if extra_key is None or isinstance(extra_key, str | int | float | bool | bytes):
        return extra_key
    if isinstance(extra_key, tuple):
        return tuple(normalize_prefix_cache_extra_key(value) for value in extra_key)
    try:
        return json.dumps(extra_key, sort_keys=True, default=str)
    except TypeError:
        return repr(extra_key)


def _page_aligned_tokens(tokens: tuple[int, ...], page_size: int) -> tuple[int, ...]:
    if page_size <= 1:
        return tokens
    aligned_len = len(tokens) // page_size * page_size
    return tokens[:aligned_len]


def _child_lookup_key(
    tokens: tuple[int, ...],
    position: int,
    *,
    extra_key: Any,
    page_size: int,
) -> Any:
    if page_size <= 1:
        plain: Any = tokens[position]
    else:
        plain = tokens[position : position + page_size]
    return (extra_key, plain) if extra_key is not None else plain


def _common_prefix_len(
    tokens: tuple[int, ...],
    segment: tuple[int, ...],
    position: int,
    page_size: int = 1,
) -> int:
    limit = min(len(tokens) - position, len(segment))
    common = 0
    while common < limit and tokens[position + common] == segment[common]:
        common += 1
    if page_size > 1:
        common = common // page_size * page_size
    return common


def _count_nodes(node: _GRPrefixNode) -> int:
    return 1 + sum(_count_nodes(child) for child in node.children.values())
