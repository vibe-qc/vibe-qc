"""Top-level ``vibe-qc`` CLI -- thin dispatcher for vibe-qc sub-commands.

The ``vibe-qc`` entry point ships with two sub-commands:

* ``vibe-qc cite <stem> [options]`` -- reprints citations for an
  already-run job (shipped as ``vibeqc-cite`` as well).
* ``vibe-qc outputs <stem> [options]`` -- inspects a job's
  ``.system`` manifest (shipped as ``vibeqc-outputs`` as well).

When invoked with no sub-command, the CLI prints a brief banner and
usage hint.

Examples
--------
>>> # Show help
>>> main(["--help"])  # doctest: +SKIP
>>> # Reprint citations
>>> main(["cite", "output-h2o"])  # doctest: +SKIP
>>> # Inspect manifest
>>> main(["outputs", "output-h2o.system"])  # doctest: +SKIP

Entry point in ``pyproject.toml``:

.. code-block:: toml

   [project.scripts]
   vibe-qc = "vibeqc.cli:main"
"""

from __future__ import annotations

import argparse
import sys

from .banner import banner


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vibe-qc",
        description=(
            "vibe-qc -- quantum chemistry for molecules and solids. "
            "Run with a sub-command to see its help."
        ),
    )
    p.add_argument(
        "--version",
        action="store_true",
        help="Print the vibe-qc version and exit.",
    )
    sub = p.add_subparsers(
        dest="subcommand",
        help="Sub-command (run ``vibe-qc <cmd> --help`` for help).",
    )

    # --- cite ---
    cite_p = sub.add_parser(
        "cite",
        help="Reprint citations for an already-run job.",
    )
    cite_p.add_argument(
        "stem",
        help=(
            "Job output stem (e.g. 'output-h2o'). "
            "The .system manifest is read from {stem}.system."
        ),
    )
    cite_p.add_argument(
        "--write",
        action="store_true",
        help="Write .bibtex + .references files instead of stdout.",
    )
    cite_p.add_argument(
        "--bibtex-only",
        action="store_true",
        help="Emit BibTeX only (with --write or to stdout).",
    )

    # --- outputs ---
    out_p = sub.add_parser(
        "outputs",
        help="Inspect a job's .system manifest.",
    )
    out_p.add_argument(
        "stem",
        help=(
            "Job stem or manifest path. Accepts "
            "'output-h2o' / 'output-h2o.out' / 'output-h2o.system'."
        ),
    )
    out_p.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any 'always' file is missing.",
    )
    out_p.add_argument(
        "--paths-only",
        action="store_true",
        help="Print just the declared file paths.",
    )
    out_p.add_argument(
        "--missing-only",
        action="store_true",
        help="Restrict the summary to missing files.",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _make_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(banner())
        return 0

    if args.subcommand is None:
        print(banner())
        print(
            "\nUsage: vibe-qc <subcommand> [options]\n\n"
            "Sub-commands:\n"
            "  cite       Reprint citations for an already-run job\n"
            "  outputs    Inspect a job's .system manifest\n\n"
            "Run ``vibe-qc <subcommand> --help`` for usage details.\n"
            "Or import the Python API directly:  import vibeqc as vq\n",
            file=sys.stderr,
        )
        return 0

    # Dispatch to the appropriate handler.
    if args.subcommand == "cite":
        from .output.citations.cli import main as cite_main

        return cite_main(
            [
                args.stem,
                *(["--write"] if args.write else []),
                *(["--bibtex-only"] if args.bibtex_only else []),
            ],
        )

    if args.subcommand == "outputs":
        from .output.outputs_cli import main as outputs_main

        opts: list[str] = []
        if args.strict:
            opts.append("--strict")
        if args.paths_only:
            opts.append("--paths-only")
        if args.missing_only:
            opts.append("--missing-only")

        return outputs_main([args.stem, *opts])

    parser.print_help(file=sys.stderr)
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
