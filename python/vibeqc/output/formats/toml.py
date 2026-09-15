"""Small TOML emitter for deterministic metadata exports.

The stdlib can read TOML but cannot write it. This helper covers the simple
subset vibe-qc uses for registry exports: scalars, inline arrays, tables, and
arrays of tables. It deliberately refuses unsupported values instead of
guessing at a lossy representation.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


_BARE_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def dumps_toml(data: Mapping[str, Any]) -> str:
    """Return *data* as a TOML document.

    The input mapping order is preserved, so callers can keep public registry
    exports stable across runs.
    """
    lines: list[str] = []
    _write_mapping(lines, data, ())
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"


def write_toml(data: Mapping[str, Any], path: str | Path) -> None:
    """Write *data* as UTF-8 TOML at *path*."""
    Path(path).write_text(dumps_toml(data), encoding="utf-8")


def _write_mapping(
    lines: list[str],
    data: Mapping[str, Any],
    path: tuple[str, ...],
) -> None:
    scalar_items: list[tuple[str, Any]] = []
    table_items: list[tuple[str, Mapping[str, Any]]] = []
    array_table_items: list[tuple[str, Sequence[Mapping[str, Any]]]] = []

    for key, value in data.items():
        if isinstance(value, Mapping):
            table_items.append((key, value))
        elif _is_array_of_tables(value):
            array_table_items.append((key, value))
        else:
            scalar_items.append((key, value))

    for key, value in scalar_items:
        lines.append(f"{_format_key(key)} = {_format_value(value)}")

    if scalar_items and (table_items or array_table_items):
        lines.append("")

    for index, (key, value) in enumerate(table_items):
        lines.append(f"[{_format_dotted_key(path + (key,))}]")
        _write_mapping(lines, value, path + (key,))
        if index != len(table_items) - 1 or array_table_items:
            lines.append("")

    for table_index, (key, records) in enumerate(array_table_items):
        for record_index, record in enumerate(records):
            lines.append(f"[[{_format_dotted_key(path + (key,))}]]")
            _write_mapping(lines, record, path + (key,))
            if record_index != len(records) - 1:
                lines.append("")
        if table_index != len(array_table_items) - 1:
            lines.append("")


def _is_array_of_tables(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and bool(value)
        and all(isinstance(item, Mapping) for item in value)
    )


def _format_dotted_key(parts: Sequence[str]) -> str:
    return ".".join(_format_key(part) for part in parts)


def _format_key(key: str) -> str:
    if _BARE_KEY_RE.match(key):
        return key
    return json.dumps(key, ensure_ascii=True)


def _format_value(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=True)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("TOML registry exports cannot contain NaN or infinity")
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_value(item) for item in value) + "]"
    raise TypeError(f"unsupported TOML value {value!r} ({type(value).__name__})")
