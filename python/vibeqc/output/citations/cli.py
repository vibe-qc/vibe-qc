"""``vibeqc-cite`` -- reprint citations for an already-run job.

Re-reads ``{stem}.system`` (the manifest produced by ``run_job``),
extracts the ``[plan]`` section, walks the citation database, and
either:

* prints the plain-text reference list to stdout (default), or
* writes ``{stem}.bibtex`` / ``{stem}.references`` next to the
  manifest (``--write`` flag).

Useful when:

* A job ran before the citation surface existed (pre-v0.8.x) and
  you want to assemble its references now.
* A generated output is being copied between machines and the
  maintainer wants to regenerate the citation files without re-running
  the SCF.
* A tutorial walks through "here are the references this run
  cited" -- calling ``vibeqc-cite output-h2o`` is cheaper than
  embedding the output verbatim.

The ``[plan]`` section in the manifest carries the resolved method
+ basis + functional that the job used, so the assembled citation
list matches what would have been auto-emitted at run time.

Exit codes
----------

* ``0`` -- citations assembled and emitted successfully.
* ``1`` -- manifest missing or malformed.
* ``2`` -- citation database load error.
"""

from __future__ import annotations

import argparse
import os
import sys
import tomllib
from pathlib import Path
from typing import Any

from .bibtex import write_bibtex
from .plain import format_references, write_references
from .registry import AssembledCitations, load_default_database


__all__ = ["main"]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vibeqc-cite",
        description=(
            "Reprint or regenerate the citation list for an "
            "already-run vibe-qc job. Reads {stem}.system, walks "
            "the citation database, and emits plain-text refs to "
            "stdout (default) or writes .bibtex + .references "
            "siblings (--write)."
        ),
    )
    p.add_argument(
        "stem",
        type=Path,
        help=(
            "Job output stem (e.g. 'output-h2o'). The .system "
            "manifest is read from {stem}.system; with --write, "
            "the .bibtex and .references files are placed next to "
            "it."
        ),
    )
    p.add_argument(
        "--write",
        action="store_true",
        help=(
            "Write {stem}.bibtex + {stem}.references files instead "
            "of printing to stdout."
        ),
    )
    p.add_argument(
        "--bibtex-only",
        action="store_true",
        help=(
            "With --write, only emit the .bibtex file. Without "
            "--write, print the BibTeX entries to stdout instead "
            "of the plain-text list."
        ),
    )
    return p


def _load_manifest(stem: Path) -> dict[str, Any]:
    """Read ``{stem}.system`` and return its parsed contents.

    The stem may already carry a suffix (``output-h2o.system``,
    ``output-h2o.out``); we normalise to ``with_suffix('.system')``.
    """
    manifest_path = stem.with_suffix(".system")
    if not manifest_path.is_file():
        sys.stderr.write(
            f"vibeqc-cite: manifest not found at {manifest_path}\n"
            "  (try passing the path to your job stem, e.g. "
            "'output-h2o' or 'output-h2o.out')\n"
        )
        sys.exit(1)
    try:
        with manifest_path.open("rb") as fh:
            return tomllib.load(fh)
    except Exception as exc:  # tomllib raises TOMLDecodeError, OSError, ...
        sys.stderr.write(
            f"vibeqc-cite: failed to parse {manifest_path}: "
            f"{type(exc).__name__}: {exc}\n"
        )
        sys.exit(1)


def _job_descriptor(manifest: dict[str, Any]) -> dict[str, Any]:
    """Pull the bits we need to walk the citation routes out of
    a manifest dict.

    The post-Phase-O1 manifest carries ``[plan]`` with ``method`` /
    ``basis`` / ``functional``. For pre-Phase-O1 manifests (no
    ``[plan]`` section yet) we fall back to the legacy ``[run]``
    block, which only records the basename -- the user is told to
    re-run the job once the new manifest format is in place.
    """
    plan = manifest.get("plan")
    if plan is None:
        sys.stderr.write(
            "vibeqc-cite: manifest has no [plan] section (Phase O1 "
            "format). For pre-v0.8.x manifests, re-run the job -- "
            "the new manifest shape is required for citation "
            "regeneration.\n"
        )
        sys.exit(1)
    functional = plan.get("functional") or None
    if isinstance(functional, str) and not functional.strip():
        functional = None
    return {
        "method":     str(plan.get("method") or ""),
        "basis":      str(plan.get("basis")  or ""),
        "functional": functional,
    }


def _assemble(manifest: dict[str, Any]) -> AssembledCitations:
    try:
        db = load_default_database()
    except Exception as exc:
        sys.stderr.write(
            f"vibeqc-cite: failed to load citation database: "
            f"{type(exc).__name__}: {exc}\n"
        )
        sys.exit(2)
    job = _job_descriptor(manifest)
    return db.assemble(
        method=job["method"] or None,
        basis=job["basis"] or None,
        functional=job["functional"],
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    stem = args.stem
    manifest = _load_manifest(stem)
    result = _assemble(manifest)

    if args.write:
        wrote: list[Path] = []
        if not args.bibtex_only:
            wrote.append(write_references(stem, result))
        wrote.append(write_bibtex(stem, result))
        for p in wrote:
            sys.stdout.write(f"wrote {p}\n")
        return 0

    # Stdout path.
    if args.bibtex_only:
        from .bibtex import format_bibtex
        sys.stdout.write(format_bibtex(result))
    else:
        sys.stdout.write(format_references(result))
    if result.warnings:
        sys.stderr.write("\n# citation routing warnings:\n")
        for w in result.warnings:
            sys.stderr.write(f"#   {w}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
