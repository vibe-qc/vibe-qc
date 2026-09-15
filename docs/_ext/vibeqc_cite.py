"""Sphinx extension — render citations from ``database.toml``.

Closes the no-drift loop on the v0.8.x citation surface. The single
source of truth is
``python/vibeqc/output/citations/database.toml`` (per AGENTS.md
rule 8); this extension lets MyST / RST pages render any subset of
that database inline, so prose pages stay consistent with what
``run_job`` actually emits at runtime.

Three directives:

* ``{vibeqc-cite}`<key>``` (role) — inline reference text for a
  single entry. E.g. ``{vibeqc-cite}`peintinger_pob_tzvp_2013```.
* ``vibeqc-cite-entry`` (directive) — block-form rendering of one
  entry. Takes the entry key as the directive argument.
* ``vibeqc-cite-route`` (directive) — render whatever entries fire
  for a given ``[routes.<category>][<key>]``. E.g.
  ``vibeqc-cite-route:: basis_sets pob-tzvp-rev2`` renders the
  Peintinger 2013 + Vilela Oliveira 2019 pair.

Register in ``docs/conf.py``::

    sys.path.insert(0, str(_REPO_ROOT / "docs" / "_ext"))
    extensions += ["vibeqc_cite"]

The database is parsed once per Sphinx build and cached on the
extension module; warnings on unknown keys / unknown routes fire via
the normal Sphinx ``logger.warning`` channel so a typo in a doc
page surfaces in the build log just like a broken cross-reference.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from docutils import nodes
from docutils.parsers.rst import Directive, directives
from docutils.statemachine import StringList
from sphinx.util import logging as sphinx_logging
from sphinx.util.docutils import SphinxDirective, SphinxRole


__all__ = ["setup"]


logger = sphinx_logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# Database load (cached per build)                                       #
# ---------------------------------------------------------------------- #

_DB_CACHE: dict[str, Any] | None = None


def _database_path(repo_root: Path) -> Path:
    """Path to the bundled citation database."""
    return repo_root / "python" / "vibeqc" / "output" / "citations" / \
        "database.toml"


def _load_database(app: Any) -> dict[str, Any]:
    """Parse ``database.toml`` once per build, cache for the
    remainder of the build."""
    global _DB_CACHE
    if _DB_CACHE is not None:
        return _DB_CACHE
    repo_root = Path(app.srcdir).resolve().parent
    db_path = _database_path(repo_root)
    if not db_path.is_file():
        logger.error(
            "vibeqc-cite: database.toml not found at %s — "
            "directives that depend on it will render an empty "
            "placeholder",
            db_path,
        )
        _DB_CACHE = {"entries": {}, "routes": {}}
        return _DB_CACHE
    with db_path.open("rb") as fh:
        _DB_CACHE = tomllib.load(fh)
    return _DB_CACHE


def _entry(app: Any, key: str) -> dict[str, Any] | None:
    db = _load_database(app)
    entries = db.get("entries", {})
    return entries.get(key)


def _route(
    app: Any,
    category: str,
    route_key: str,
) -> list[str]:
    """Look up an entry-key list from ``routes.<category>[<key>]``,
    or empty list when not found."""
    db = _load_database(app)
    routes = db.get("routes", {})
    cat = routes.get(category)
    if not isinstance(cat, dict):
        return []
    return list(cat.get(route_key.lower(), []))


# ---------------------------------------------------------------------- #
# Entry rendering                                                        #
# ---------------------------------------------------------------------- #

def _format_entry_text(entry: dict[str, Any]) -> str:
    """Single-line Chicago-ish citation string.

    Mirrors :func:`vibeqc.output.citations.plain.citation_to_plain`
    (single-line form), kept in sync by hand. The two are
    intentionally separate: the Sphinx directive runs in the docs-
    build environment (no vibeqc package importable in CI), so we
    re-implement the small formatter here rather than depending on
    the runtime module.
    """
    authors = entry.get("authors") or []
    if isinstance(authors, str):
        authors = [authors]
    if not authors:
        author_str = "(no author)"
    elif len(authors) == 1:
        author_str = authors[0]
    elif len(authors) == 2:
        author_str = f"{authors[0]} and {authors[1]}"
    elif len(authors) == 3:
        author_str = f"{authors[0]}, {authors[1]}, and {authors[2]}"
    else:
        author_str = f"{authors[0]}, {authors[1]}, et al."

    year = entry.get("year")
    parts: list[str] = []
    if year not in (None, ""):
        if not author_str.endswith("."):
            parts.append(f"{author_str} ({year}).")
        else:
            parts.append(f"{author_str} ({year}).")
    else:
        if author_str.endswith("."):
            parts.append(author_str)
        else:
            parts.append(f"{author_str}.")

    parts.append(f"*{entry['title']}.*")

    journal = entry.get("journal")
    volume = entry.get("volume")
    issue = entry.get("issue")
    pages = entry.get("pages")
    publisher = entry.get("publisher")
    if journal:
        venue = f"{journal}"
        if volume not in (None, ""):
            venue = f"{venue}, **{volume}**"
            if issue not in (None, ""):
                venue = f"{venue}({issue})"
        if pages:
            venue = f"{venue}, {pages}"
        parts.append(f"{venue}.")
    elif publisher:
        parts.append(f"{publisher}.")
    elif entry.get("kind") == "software":
        sw = "[Software"
        version = entry.get("version")
        license_ = entry.get("license")
        if version:
            sw += f" v{version}"
        if license_:
            sw += f", {license_}"
        sw += "]"
        parts.append(f"{sw}.")

    doi = entry.get("doi")
    url = entry.get("url")
    if doi:
        parts.append(f"doi:[{doi}](https://doi.org/{doi})")
    elif url:
        parts.append(f"<{url}>")
    return " ".join(parts)


# ---------------------------------------------------------------------- #
# Directives                                                             #
# ---------------------------------------------------------------------- #

class _CiteBaseDirective(SphinxDirective):
    """Common helpers for the entry / route directives."""

    has_content = False
    final_argument_whitespace = True

    def _render_entries(
        self,
        keys: list[str],
        *,
        block_class: str = "vibeqc-cite-block",
    ) -> list[nodes.Node]:
        """Build a docutils container with one paragraph per cited
        entry, MyST-parsed so the inline emphasis / DOI links render
        correctly."""
        if not keys:
            return []

        # Render each entry as MyST source, then have docutils parse
        # it back into nodes. This is the standard idiom in Sphinx
        # extensions that want to emit rich content from a directive.
        lines: list[str] = []
        for k in keys:
            entry = _entry(self.env.app, k)
            if entry is None:
                logger.warning(
                    "vibeqc-cite: unknown entry key %r "
                    "(referenced from %s)",
                    k, self.env.docname,
                )
                lines.append(f"> ⚠ unknown entry key `{k}`")
                lines.append("")
                continue
            line = _format_entry_text(entry)
            # Render as a blockquote per entry — pairs nicely with
            # the existing hand-written citation pages that use
            # `> Author …` blockquotes.
            lines.append(f"> {line}")
            lines.append(">")  # empty quote line as separator
            lines.append("")
        # Drop the trailing empty blockquote separator if present.
        while lines and lines[-1] in ("", ">"):
            lines.pop()

        container = nodes.container(classes=[block_class])
        self.state.nested_parse(
            StringList(lines, source=self.env.docname),
            self.content_offset,
            container,
        )
        return [container]


class VibeqcCiteEntryDirective(_CiteBaseDirective):
    """Render one (or several) database entry by key.

    Usage in MyST::

        ```{vibeqc-cite-entry} peintinger_pob_tzvp_2013
        ```

    or with multiple keys::

        ```{vibeqc-cite-entry} pulay_diis_1980 pulay_diis_1982
        ```
    """

    required_arguments = 1
    optional_arguments = 1000  # accept arbitrarily many keys
    final_argument_whitespace = True

    def run(self) -> list[nodes.Node]:
        # All positional arguments are entry keys. Sphinx splits on
        # whitespace by default; allow commas too for the natural
        # ``foo, bar, baz`` author-list shape.
        raw = " ".join(self.arguments)
        keys = [k.strip() for k in raw.replace(",", " ").split()
                if k.strip()]
        return self._render_entries(keys)


class VibeqcCiteRouteDirective(_CiteBaseDirective):
    """Render whatever entries fire for a given route.

    Usage::

        ```{vibeqc-cite-route} basis_sets pob-tzvp-rev2
        ```

    Renders the Peintinger 2013 + Vilela Oliveira 2019 pair (whatever
    the database currently says).
    """

    required_arguments = 2
    optional_arguments = 0
    final_argument_whitespace = False

    def run(self) -> list[nodes.Node]:
        category, route_key = self.arguments[0], self.arguments[1]
        keys = _route(self.env.app, category, route_key)
        if not keys:
            logger.warning(
                "vibeqc-cite: no entries for route %s[%s] "
                "(referenced from %s)",
                category, route_key, self.env.docname,
            )
            return [nodes.warning(
                "",
                nodes.paragraph(
                    text=f"⚠ no entries for route "
                         f"{category}[{route_key}]",
                ),
            )]
        return self._render_entries(keys)


# ---------------------------------------------------------------------- #
# Inline role                                                            #
# ---------------------------------------------------------------------- #

class VibeqcCiteRole(SphinxRole):
    """Inline citation by key.

    Usage in MyST: ``{vibeqc-cite}`peintinger_pob_tzvp_2013``` →
    renders the Chicago-ish reference text inline. Useful inside
    prose like *"following {vibeqc-cite}`grimme_d3bj_2011`, …"*.
    """

    def run(self) -> tuple[list[nodes.Node], list[Any]]:
        key = self.text.strip()
        entry = _entry(self.env.app, key)
        if entry is None:
            logger.warning(
                "vibeqc-cite: unknown entry key %r "
                "(referenced from %s)",
                key, self.env.docname,
            )
            return [nodes.literal(text=f"?{key}?")], []
        text = _format_entry_text(entry)
        # The inline role can't itself trigger nested MyST parsing
        # (roles are leaf-node producers), so we render the entry as
        # a plain inline node and accept that the *emphasis* /
        # **bold** markdown markup will appear literally. Doc authors
        # who need rich inline rendering should use the directive
        # form instead.
        return [nodes.inline(text=text)], []


# ---------------------------------------------------------------------- #
# Sphinx setup hook                                                      #
# ---------------------------------------------------------------------- #

def setup(app: Any) -> dict[str, Any]:
    app.add_directive("vibeqc-cite-entry", VibeqcCiteEntryDirective)
    app.add_directive("vibeqc-cite-route", VibeqcCiteRouteDirective)
    app.add_role("vibeqc-cite", VibeqcCiteRole())
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
