"""Plain-text writer for the assembled citation list.

Emits ``{stem}.references`` -- a Chicago-ish numbered reference list,
one entry per assembled citation. The same content is also used as
the ``## References`` block appended to ``{stem}.out``, via
:func:`format_references_block`.

The format is human-readable, not machine-parseable. For
machine-readable citation data, downstream tools should consume
``{stem}.bibtex`` (or the manifest's ``[plan]`` section).
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

from ..channel import Level
from ..channel import write as _channel_write
from .registry import AssembledCitations, Citation
from .._stem_paths import stem_sibling


__all__ = [
    "write_references",
    "write_references_block",
    "format_references",
    "format_references_block",
    "citation_to_plain",
]


# Used by format_references_block to align with the rest of .out.
_INDENT = " " * 2
_WRAP_WIDTH = 78


def write_references(
    stem: os.PathLike | str,
    citations: AssembledCitations | Iterable[Citation],
) -> Path:
    """Write ``{stem}.references`` (plain-text numbered list)."""
    target = stem_sibling(stem, ".references")
    body = format_references(citations)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return target


def format_references(
    citations: AssembledCitations | Iterable[Citation],
) -> str:
    """Render the assembled list as a stand-alone reference document.

    Includes a short header explaining where the file came from.

    Citations carried by an :class:`AssembledCitations` are filtered to
    the user-visible subset (``print = True``). Entries flagged
    ``print = False`` are retained by the manifest writer for internal
    provenance only and never enter the references block. When called
    with a raw iterable of :class:`Citation`, every entry is rendered
    (the caller has already chosen the subset).
    """
    items: Sequence[Citation]
    warnings: tuple[str, ...] = ()
    if isinstance(citations, AssembledCitations):
        items = citations.printable
        warnings = citations.warnings
    else:
        items = tuple(citations)

    lines: list[str] = [
        "# vibe-qc auto-generated reference list",
        "#",
        "# This file is regenerated on every run. The matching BibTeX",
        "# entries are in the .bibtex sibling.",
        "",
    ]
    if not items:
        lines.append("(no citations assembled)")
        lines.append("")
    else:
        for i, c in enumerate(items, 1):
            lines.append(citation_to_plain(c, index=i))
            lines.append("")

    if warnings:
        lines.append("# --- citation routing warnings ---")
        for w in warnings:
            lines.append(f"# {w}")
        lines.append("")

    return "\n".join(lines) + "\n"


def format_references_block(
    citations: AssembledCitations | Iterable[Citation],
    *,
    indent: str = "",
    width: int = _WRAP_WIDTH,
) -> str:
    """Render the citation list as the ``## References`` block embedded
    in ``{stem}.out``.

    Returned text starts with a section header line so the caller can
    splice it verbatim into the SCF text log. The hard-wrap width
    matches the SCF-trace layout in :mod:`vibeqc.scf_log` so the .out
    stays visually consistent.
    """
    items: Sequence[Citation]
    if isinstance(citations, AssembledCitations):
        items = citations.printable
    else:
        items = tuple(citations)

    lines: list[str] = [
        f"{indent}## References",
        "",
        f"{indent}Please cite the references below when reporting results",
        f"{indent}from this run. The corresponding BibTeX entries are",
        f"{indent}written to the .bibtex sibling.",
        "",
    ]
    if not items:
        lines.append(f"{indent}  (no citations assembled)")
        lines.append("")
        return "\n".join(lines)

    wrapper = textwrap.TextWrapper(
        width=width,
        initial_indent=indent + "  ",
        subsequent_indent=indent + "     ",
        break_long_words=False,
        break_on_hyphens=False,
    )
    for i, c in enumerate(items, 1):
        body = citation_to_plain(c, index=i, multiline=False)
        wrapped = wrapper.fill(body)
        lines.append(wrapped)
        lines.append("")
    return "\n".join(lines)


def write_references_block(
    citations: Optional[AssembledCitations | Iterable[Citation]] = None,
    *,
    block: Optional[str] = None,
    write: Optional[Callable[..., object]] = None,
    level: Level = Level.STANDARD,
    leading: str = "\n",
    trailing: str = "\n",
) -> bool:
    """Emit the ``## References`` block through the output channel.

    The level-aware counterpart to :func:`format_references_block`: rather
    than returning a string for the caller to splice with ``write(...)`` by
    hand, this renders the block and emits it through ``write`` (the
    ambient :func:`vibeqc.output.write` by default) tagged with ``level``,
    the same way :func:`vibeqc.output.formats.scf_log.write_scf_trace`
    emits the SCF trace. ``leading`` / ``trailing`` are written around the
    block (the runners used ``"\\n"`` on each side).

    Pass either ``citations`` (assembled) or a pre-built ``block`` string.
    Returns whether anything was written -- ``False`` for no citations, so
    the caller's ``if`` guard is folded in.
    """
    if block is None:
        if citations is None:
            return False
        block = format_references_block(citations)
    if not block:
        return False
    emit = _channel_write if write is None else write
    emit(f"{leading}{block}{trailing}", level)
    return True


def citation_to_plain(
    c: Citation,
    *,
    index: int | None = None,
    multiline: bool = True,
) -> str:
    """Render one citation as a plain-text paragraph.

    ``multiline=True`` (the default) emits a header line with the
    index + bibtex_key plus the formatted citation underneath -- used
    for ``{stem}.references``. ``multiline=False`` emits the single
    formatted-citation line -- used for the embedded ``.out`` block,
    where ``textwrap.fill`` handles the line breaks.
    """
    prefix = f"[{index}] " if index is not None else ""
    authors = _format_authors(c.authors)
    parts: list[str] = [authors]
    year = "" if c.year in (None, "") else str(c.year)
    if year:
        # "(Year)." reads cleanly whether or not the author already ends
        # in "." -- the parenthesised year is the visual separator.
        parts[-1] = f"{parts[-1]} ({year})."
    else:
        # No year -- only add a closing period if the author block
        # doesn't already end with one (e.g. "Valeev, Edward F.").
        if not parts[-1].endswith("."):
            parts[-1] = f"{parts[-1]}."
    parts.append(f"{c.title}.")
    if c.journal:
        venue = c.journal
        if c.volume not in (None, ""):
            venue = f"{venue}, {c.volume}"
            if c.issue not in (None, ""):
                venue = f"{venue}({c.issue})"
        if c.pages:
            venue = f"{venue}, {c.pages}"
        parts.append(f"{venue}.")
    elif c.publisher:
        parts.append(f"{c.publisher}.")
    elif c.kind == "software":
        sw = "Software"
        if c.version:
            sw = f"{sw} v{c.version}"
        if c.license:
            sw = f"{sw}, {c.license}"
        parts.append(f"[{sw}].")
    if c.doi:
        parts.append(f"doi:{c.doi}")
    if c.url and not c.doi:
        parts.append(f"<{c.url}>")
    body = " ".join(p for p in parts if p)

    if not multiline:
        return f"{prefix}{body}"

    head = f"{prefix}({c.bibtex_key})"
    return f"{head}\n  {body}"


def _format_authors(authors: tuple[str, ...]) -> str:
    """Render an author tuple as a single string.

    Database entries store authors as ``"Family, Given"``. We keep
    that form for the citation text -- it's the canonical form for
    Chicago and most journal house styles, and matches the BibTeX
    surface. Three-or-fewer authors are joined with commas + "and";
    four-or-more get ", ... et al." after the third.
    """
    if not authors:
        return "(no author)"
    if len(authors) == 1:
        return authors[0]
    if len(authors) == 2:
        return f"{authors[0]} and {authors[1]}"
    if len(authors) == 3:
        return f"{authors[0]}, {authors[1]}, and {authors[2]}"
    return f"{authors[0]}, {authors[1]}, et al."
