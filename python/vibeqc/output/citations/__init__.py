"""``vibeqc.output.citations`` -- runtime citation surface.

The single source of truth is :mod:`database.toml` in this package.
Public entry points:

* :func:`load_database` / :func:`load_default_database` -- read the
  TOML file and return a :class:`CitationDatabase`. The default-loader
  pulls the bundled ``database.toml`` (plus the basissetdev sibling
  if running on that branch).
* :class:`CitationDatabase` -- parsed in-memory form. Exposes
  :meth:`entries`, :meth:`routes`, and :meth:`assemble(...)`.
* :class:`Citation` -- one citable reference, with helpers to render
  it as BibTeX (:meth:`to_bibtex`) and plain text (:meth:`to_plain`).
* :func:`write_bibtex` / :func:`write_references` -- emit
  ``{stem}.bibtex`` and ``{stem}.references`` files.
* :func:`format_references_block` -- the in-``.out`` reference block.

The full schema and routing rules are described in
``docs/design_output_module.md``.
"""

from __future__ import annotations

from .bibtex import format_bibtex, write_bibtex
from .cli import main as cite_cli_main
from .emitter import emit_citations
from .plain import (
    format_references,
    format_references_block,
    write_references,
    write_references_block,
)
from .registry import (
    Citation,
    CitationDatabase,
    DatabaseError,
    assemble,
    citation_manifest_rows,
    load_database,
    load_default_database,
)


__all__ = [
    # registry.py
    "Citation",
    "CitationDatabase",
    "DatabaseError",
    "assemble",
    "citation_manifest_rows",
    "load_database",
    "load_default_database",
    # bibtex.py
    "format_bibtex",
    "write_bibtex",
    # plain.py
    "format_references",
    "format_references_block",
    "write_references",
    "write_references_block",
    # emitter.py -- assemble + write siblings in one call. Used by the
    # standalone post-SCF dispatchers (run_b2plyp, run_pwpb95, run_mp2,
    # run_wb97x_d, run_cpcm_scf, run_dftb0, run_scc_dftb) when the
    # caller passes a non-None ``output=`` kwarg.
    "emit_citations",
    # cli.py (re-exported for programmatic use; the entry point is the
    # `vibeqc-cite` console script declared in pyproject.toml).
    "cite_cli_main",
]
