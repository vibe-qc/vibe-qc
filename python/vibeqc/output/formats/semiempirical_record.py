"""The physics qualifications a semiempirical run rests on, as a record.

A semiempirical result can be numerically clean and still not be chemistry:
the route may be experimental, and a DFTB run may have evaluated element
pairs whose repulsive potential is the in-house ``A/R^12`` placeholder
rather than a fitted curve. vibe-qc already detects both facts before it
computes a number. Until issues #318 and #345 it stated the first only in
the ``.system`` manifest and the second only on stderr, a channel most batch
harnesses discard, so an EOS curve over placeholder pairs read like a result
(0 of 84 archived scan ``.out`` files and 0 of 2664 manifests carried the
warning, measured 2026-09-17).

This module is the one place that spells those facts for every surface a
consumer reads: the ``.out`` options block, the ``.system`` ``[run]``
section and the archived QVF job spec. The runners compute the values (the
route plan's maturity, :func:`vibeqc.semiempirical.dftb0.placeholder_repulsive_pairs`)
and hand them here; nothing in this module evaluates physics.

Keys and spellings are shared with the stderr warning
(:func:`vibeqc.semiempirical.dftb0.warn_placeholder_repulsives`): a pair is
always ``"Z1-Z2"`` with ``Z1 <= Z2``, and the manifest key is
``placeholder_repulsive_pairs``. An empty list means "checked, none"; an
absent key means there was no repulsive table to check: a route without one
(GFN2-xTB, PM6, OMx, MSINDO), or parameters that could not be probed.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

__all__ = [
    "PLACEHOLDER_REPULSIVE_RUN_FIELD",
    "ROUTE_MATURITY_RUN_FIELD",
    "ROUTE_STATUS_RUN_FIELD",
    "placeholder_repulsive_labels",
    "semiempirical_record_fields",
    "format_semiempirical_record_lines",
    "format_semiempirical_record_section",
]

# ``.system`` ``[run]`` and QVF ``job.spec.options`` keys.
ROUTE_MATURITY_RUN_FIELD = "route_maturity"
ROUTE_STATUS_RUN_FIELD = "route_status"
PLACEHOLDER_REPULSIVE_RUN_FIELD = "placeholder_repulsive_pairs"

# One sentence, the same one the stderr warning ends with, so a reader of
# the ``.out`` learns what the pair list means without the warning.
_PLACEHOLDER_QUALIFICATION = (
    "A/R^12 placeholder repulsive: absolute energies, equilibrium "
    "geometries and EOS fits are placeholder physics; fixed-geometry "
    "differences remain exactly valid"
)


def placeholder_repulsive_labels(pairs: Iterable[Sequence[int]] | None) -> list[str]:
    """``"Z1-Z2"`` labels for placeholder pairs, sorted and unique.

    Accepts the ``(z1, z2)`` tuples of
    :func:`vibeqc.semiempirical.dftb0.placeholder_repulsive_pairs` or labels
    that are already strings. ``None`` yields an empty list.
    """
    if pairs is None:
        return []
    labels: set[str] = set()
    for pair in pairs:
        if isinstance(pair, str):
            labels.add(pair)
            continue
        z1, z2 = (int(pair[0]), int(pair[1]))
        if z2 < z1:
            z1, z2 = z2, z1
        labels.add(f"{z1}-{z2}")

    def _key(label: str) -> tuple[int, int]:
        a, b = label.split("-", 1)
        return (int(a), int(b))

    return sorted(labels, key=_key)


def semiempirical_record_fields(
    *,
    route_maturity: str | None = None,
    route_status: str | None = None,
    placeholder_pairs: Iterable[Sequence[int]] | None = None,
) -> dict[str, Any]:
    """The ``[run]`` / job-spec fields for the qualifications that apply.

    ``placeholder_pairs=None`` means the route has no repulsive table and the
    key is left out; an empty iterable records ``[]``, "checked, none".
    """
    fields: dict[str, Any] = {}
    if route_maturity is not None:
        fields[ROUTE_MATURITY_RUN_FIELD] = str(route_maturity)
    if route_status is not None:
        fields[ROUTE_STATUS_RUN_FIELD] = str(route_status)
    if placeholder_pairs is not None:
        fields[PLACEHOLDER_REPULSIVE_RUN_FIELD] = placeholder_repulsive_labels(
            placeholder_pairs
        )
    return fields


def format_semiempirical_record_lines(
    *,
    route_maturity: str | None = None,
    placeholder_pairs: Iterable[Sequence[int]] | None = None,
    indent: str = "    ",
    key_width: int = 20,
) -> str:
    """Options-block lines stating the route maturity and placeholder pairs.

    Rendered in the same ``key = value`` shape as the surrounding
    semiempirical options so the block keeps one alignment. The maturity line
    is written whenever a maturity is known, production included: a stable
    shape is what lets a consumer grep for the key rather than infer from its
    absence. The placeholder line is written whenever the route has a
    repulsive table (``placeholder_pairs`` is not ``None``); ``none`` states
    that every pair was checked and fitted.
    """
    def _key(name: str) -> str:
        # Pad to the block's column; a key wider than the column keeps one
        # space before "=" instead of running into it.
        padded = name.ljust(key_width)
        if not padded.endswith(" "):
            padded += " "
        return f"{indent}{padded}= "

    lines: list[str] = []
    if route_maturity is not None:
        lines.append(_key("maturity") + f"{route_maturity}\n")
    if placeholder_pairs is not None:
        labels = placeholder_repulsive_labels(placeholder_pairs)
        key = _key(PLACEHOLDER_REPULSIVE_RUN_FIELD)
        if labels:
            lines.append(key + ", ".join(labels) + "\n")
            lines.append(f"{indent}  ({_PLACEHOLDER_QUALIFICATION})\n")
        else:
            lines.append(key + "none\n")
    return "".join(lines)


def format_semiempirical_record_section(
    *,
    placeholder_pairs: Iterable[Sequence[int]],
) -> str:
    """The molecular ``.out`` section for a DFTB run's repulsive qualification.

    The molecular semiempirical ``.out`` has no options block of its own, so
    the placeholder record is a short section beside the parameter identity
    one. Written for every DFTB0 / SCC-DFTB run, ``none`` included, so the
    section's presence says the check ran.
    """
    return format_semiempirical_record_lines(
        placeholder_pairs=placeholder_pairs, indent="  ", key_width=28
    )
