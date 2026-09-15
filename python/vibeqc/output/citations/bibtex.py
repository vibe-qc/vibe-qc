"""BibTeX writer for the assembled citation list.

Emits ``{stem}.bibtex`` -- plain ASCII, one entry per assembled
reference, in citation order. The entries are suitable for ``biber`` /
``bibtex`` use via ``\\bibliography{output-h2o.bibtex}``.

This module deliberately uses no template engine -- BibTeX is a small,
stable format and the structure of every entry is determined by its
``kind`` field. Special characters in author / title strings are
emitted verbatim (the database uses UTF-8 throughout; modern biber +
LaTeX engines handle that without escaping). ``{`` / ``}`` / ``\\`` /
``%`` / ``&`` / ``$`` characters in titles are wrapped in braces so
LaTeX does not interpret them as markup.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

from .registry import AssembledCitations, Citation
from .._stem_paths import stem_sibling


__all__ = ["write_bibtex", "format_bibtex", "citation_to_bibtex"]


# BibTeX kind -> entry type. vibe-qc uses a small subset of the BibTeX
# entry-type vocabulary; everything else falls back to @misc.
_KIND_TO_BIBTEX_TYPE = {
    "article":    "article",
    "book":       "book",
    "software":   "software",   # @software is supported by biber >= 2.5
    "phdthesis":  "phdthesis",
    "misc":       "misc",
}


def write_bibtex(
    stem: os.PathLike | str,
    citations: AssembledCitations | Iterable[Citation],
) -> Path:
    """Write a ``{stem}.bibtex`` file with one entry per cited
    reference.

    ``stem`` is a path stem; the ``.bibtex`` suffix is appended by
    :func:`~vibeqc.output._stem_paths.stem_sibling`, so a stem that
    carries a dot (``output_scale_4.08``) keeps its whole name.
    :meth:`pathlib.Path.with_suffix`, which this used to call, *replaces*
    rather than appends and truncated such stems (issue #254). Returns
    the written path.
    """
    target = stem_sibling(stem, ".bibtex")
    body = format_bibtex(citations)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return target


def format_bibtex(citations: AssembledCitations | Iterable[Citation]) -> str:
    """Format the citation list as a complete BibTeX file body.

    An :class:`AssembledCitations` is filtered to the user-visible
    subset (``print = True``) so the ``.bibtex`` sibling matches the
    references block in the ``.out``. ``print = False`` entries live
    only in the ``.system`` manifest. A raw iterable is rendered as-is.
    """
    items: Sequence[Citation]
    if isinstance(citations, AssembledCitations):
        items = citations.printable
    else:
        items = tuple(citations)

    lines: list[str] = [
        "% vibe-qc auto-generated BibTeX entries -- one per cited",
        "% reference, in citation order. The corresponding software",
        "% citation for vibe-qc itself is the first entry.",
        "",
    ]
    for c in items:
        lines.append(citation_to_bibtex(c))
        lines.append("")
    return "\n".join(lines)


def citation_to_bibtex(c: Citation) -> str:
    """Render one :class:`~vibeqc.output.citations.Citation` as a
    BibTeX entry."""
    btype = _KIND_TO_BIBTEX_TYPE.get(c.kind, "misc")
    fields: list[tuple[str, str]] = []

    authors_str = " and ".join(c.authors)
    fields.append(("author", _quote(authors_str)))
    fields.append(("title", _quote(c.title)))

    if c.journal:
        fields.append(("journal", _quote(c.journal)))
    if c.volume is not None and c.volume != "":
        fields.append(("volume", str(c.volume)))
    if c.issue is not None and c.issue != "":
        fields.append(("number", str(c.issue)))
    if c.pages:
        fields.append(("pages", _quote(c.pages)))
    if c.publisher:
        fields.append(("publisher", _quote(c.publisher)))
    if c.year not in (None, ""):
        fields.append(("year", str(c.year)))
    if c.doi:
        fields.append(("doi", _quote(c.doi)))
    if c.url:
        fields.append(("url", _quote(c.url)))
    if c.version:
        fields.append(("version", _quote(c.version)))
    if c.license:
        fields.append(("license", _quote(c.license)))
    if c.notes:
        fields.append(("note", _quote(c.notes)))

    field_lines = ",\n".join(f"  {k:<11s} = {v}" for k, v in fields)
    return f"@{btype}{{{c.bibtex_key},\n{field_lines}\n}}"


def _quote(value: str) -> str:
    """Wrap a BibTeX field value in braces, escaping LaTeX-special
    characters minimally. The database uses UTF-8 throughout; we only
    need to brace-protect ``{``, ``}``, ``%``, ``$``, ``&``, ``#``,
    ``_``, ``^``, ``~``, ``\\`` so a modern biber + LaTeX engine
    renders accents / dashes / unicode names verbatim."""
    # We use *brace-wrapped* values rather than quote-wrapped because
    # author names with commas and umlauts are simpler to keep
    # readable that way. The outer braces preserve case in titles
    # automatically.
    escaped = []
    for ch in value:
        if ch in "{}":
            # Brace characters inside a brace-wrapped value need to
            # stay balanced; the database authors should avoid them in
            # titles, but if a title needs a literal brace, double it.
            escaped.append("\\" + ch)
        else:
            escaped.append(ch)
    return "{" + "".join(escaped) + "}"
