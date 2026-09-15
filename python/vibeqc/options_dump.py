"""Active-settings dumper for SCF (and other) options objects.

Every option in vibe-qc has a meaningful default, every default is
overridable through a kwarg or a struct field, and the active values
must be printable at SCF startup so the user can see what's in effect
without having to read the driver source. This module provides the
last piece -- a polymorphic ``format_options`` that walks any options
object (a pybind-bound C++ struct, a Python dataclass, or a plain
class with public attributes) and renders the active settings in a
stable, human-readable form.

Usage from an SCF driver (typical):

.. code-block:: python

    from .options_dump import dump_active_settings

    dump_active_settings(plog, [
        ("PeriodicRHFOptions",     opts),
        ("LatticeSumOptions",      opts.lattice_opts),
        ("Driver kwargs", {
            "omega":         omega,
            "grid_shape":    grid_shape_t,
            "spacing_bohr":  spacing_bohr,
            "linear_dep_threshold": linear_dep_threshold,
        }),
    ])

The output is a banner + one ``key = value`` line per setting,
nested by group. Numeric values use a fixed format
(``{:.6g}`` for floats, ``{!r}`` for everything else) so log-diffing
two SCF runs with different settings stays readable.

Design notes
------------

* **Polymorphic over object type**. ``dir(obj)`` + filtering catches
  pybind-bound structs (LatticeSumOptions etc.); ``fields(obj)``
  catches Python dataclasses; raw ``dict`` is treated as-is. The
  fallback is "if it's not callable and the name doesn't start with
  ``_``, dump it".
* **Defaults are visible**. The dumper doesn't know what's a default
  vs. an override -- it shows the *active* value either way. That's
  the right user model: the active value is what matters.
* **Stable ordering**. Pybind structs are walked in ``dir()`` order
  (alphabetical); dataclasses use ``fields()`` declaration order;
  dicts preserve insertion order. No hidden re-sorting.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Any, Iterable, List, Mapping, Sequence, Tuple


__all__ = ["format_options", "dump_active_settings"]


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def _looks_like_pybind_enum(value: Any) -> bool:
    """Pybind11 enums aren't ``enum.Enum`` subclasses but expose
    ``.name``, ``.value``, and a class whose name reads like an enum.
    Detect ducktype-style so the formatter handles them like Python
    enums (``Class.NAME`` instead of ``<Class.NAME: 0>``).
    """
    if isinstance(value, (int, float, str, bool)):
        return False
    if value is None:
        return False
    return (
        hasattr(value, "name")
        and hasattr(value, "value")
        and not callable(value.name)
        and not callable(value.value)
        and isinstance(value.value, int)
        and isinstance(value.name, str)
    )


def _looks_like_options_struct(value: Any) -> bool:
    """A pybind-bound options struct exposes public attributes but
    no callable methods we'd recognise as part of an option object's
    contract. Detection is heuristic: it's a non-builtin instance
    with a class name ending in ``Options`` and at least one
    non-callable, non-dunder attribute.
    """
    if isinstance(value, (int, float, str, bool, list, tuple, dict)):
        return False
    if value is None:
        return False
    cls_name = type(value).__name__
    if not cls_name.endswith("Options"):
        return False
    return any(not _is_dunder(n) and not n.startswith("_")
               for n in dir(value))


def _format_value(value: Any) -> str:
    """Stable, log-diffable repr for a single option value.

    * floats -> ``{:.6g}`` (drops trailing zeros, keeps precision)
    * bools / ints / Nones / enums (Python and pybind) -> name
    * sequences <= 8 items -> list-repr; longer -> ``[...]`` with len
    * nested options struct -> ``<NestedOptions; print separately>``
      (caller is expected to emit it as its own group)
    * everything else -> truncated ``repr()``
    """
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, enum.Enum):
        return f"{type(value).__name__}.{value.name}"
    if _looks_like_pybind_enum(value):
        return f"{type(value).__name__}.{value.name}"
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, (tuple, list)):
        if len(value) <= 8:
            return "[" + ", ".join(_format_value(v) for v in value) + "]"
        return f"<{type(value).__name__} len={len(value)}>"
    if isinstance(value, dict):
        if len(value) <= 4:
            inner = ", ".join(f"{k!r}: {_format_value(v)}" for k, v in value.items())
            return "{" + inner + "}"
        return f"<dict len={len(value)}>"
    if _looks_like_options_struct(value):
        return f"<{type(value).__name__}; print separately>"
    r = repr(value)
    if len(r) > 80:
        r = r[:77] + "..."
    return r


def _walk_attributes(obj: Any) -> Iterable[Tuple[str, Any]]:
    """Yield ``(name, value)`` pairs for every public, non-callable
    attribute of ``obj``.

    Handles three object kinds:

    1. ``Mapping`` (dict-like) -- yields its items directly.
    2. Python dataclass -- yields fields in declaration order.
    3. Anything else -- falls back to ``dir()`` + filter (excludes
       dunders, callables, properties whose getter raises).
    """
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            yield str(k), v
        return

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            try:
                yield f.name, getattr(obj, f.name)
            except Exception:   # pragma: no cover -- defensive
                yield f.name, "<unreadable>"
        return

    for name in dir(obj):
        if _is_dunder(name):
            continue
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except Exception:       # pragma: no cover -- defensive
            continue
        if callable(value):
            continue
        yield name, value


def format_options(
    obj: Any,
    *,
    title: str = "options",
    indent: str = "  ",
) -> str:
    """Render ``obj``'s active settings as a banner + list of
    ``key = value`` lines.

    ``obj`` can be a pybind-bound struct (LatticeSumOptions, ...), a
    Python dataclass, a plain class with public attributes, or a
    dict. ``title`` is shown as a header.
    """
    items = list(_walk_attributes(obj))
    lines: List[str] = []
    lines.append(f"[{title}]")
    if not items:
        lines.append(f"{indent}<no public settings>")
        return "\n".join(lines)

    # Right-align keys for visual alignment up to a sensible width.
    max_key = max(len(str(k)) for k, _ in items)
    pad = min(max_key, 32)
    for name, value in items:
        key_str = str(name).ljust(pad)
        lines.append(f"{indent}{key_str} = {_format_value(value)}")
    return "\n".join(lines)


def dump_active_settings(
    plog,
    groups: Sequence[Tuple[str, Any]],
    *,
    banner_title: str = "Active settings",
) -> None:
    """Write ``groups`` to a :class:`vibeqc.progress.ProgressLogger` as
    a single multi-line block.

    ``groups`` is a sequence of ``(title, obj)`` pairs. Each pair
    becomes one ``[title]`` block followed by its key-value lines.
    Empty groups are skipped silently.

    Designed to be the first thing every SCF driver emits after
    the basic startup banner, so an SCF log self-documents its
    inputs without the user needing to introspect the call site.
    """
    if not groups:
        return
    blocks: List[str] = [f"=== {banner_title} ==="]
    for title, obj in groups:
        if obj is None:
            continue
        blocks.append(format_options(obj, title=title))
    blocks.append("=" * (len(banner_title) + 8))
    plog.write_raw("\n".join(blocks))
