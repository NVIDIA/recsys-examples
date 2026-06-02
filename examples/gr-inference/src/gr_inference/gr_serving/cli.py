"""CLI parsing helpers shared by serving tools."""

from __future__ import annotations


def parse_unique_int_list(value: str) -> tuple[int, ...]:
    if not value:
        return ()
    normalized = value.replace(",", " ").split()
    return tuple(dict.fromkeys(int(item) for item in normalized))
