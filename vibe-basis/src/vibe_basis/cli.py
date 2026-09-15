"""`vb` — top-level CLI for vibe-basis.

Initial v0.1.0 surface:

* ``vb --version`` — package version + supported backends.
* ``vb parse <backend> <out>`` — parse an external-SCF output
  file and print the extracted energy + convergence flags. Useful
  for ad-hoc diagnosis ("did that CRYSTAL run actually converge?")
  without spinning up an optimization loop.

Future verbs (planned):

* ``vb emit-test-set --backend crystal --basis pob-tzvp --out d12/``
* ``vb fit --recipe mpei_tzvp --stage 2 --transport vq:compute-host``
* ``vb parity --reference pt2013-si-table-2 --backend crystal``
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from . import __version__
from .backends import crystal


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="vb")
def main() -> None:
    """vibe-basis — basis-set optimization driver for external SCF programs."""


#: Backend names ``vb parse`` accepts, mapped to their parser module.
#: ``crystal`` is canonical. The version-suffixed spellings are kept as
#: aliases because they are honest, not merely for compatibility: one
#: parser genuinely reads CRYSTAL14 / 17 / 23 output, and since 0.3.0 it
#: *reports* which one it saw (``result.crystal_version``) rather than
#: relying on the caller to have picked the right name.
_PARSE_BACKENDS = {
    "crystal": crystal,
    "crystal14": crystal,
    "crystal17": crystal,
    "crystal23": crystal,
}


@main.command()
@click.argument(
    "backend",
    type=click.Choice(sorted(_PARSE_BACKENDS), case_sensitive=False),
)
@click.argument(
    "output_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
def parse(backend: str, output_path: Path) -> None:
    """Parse an external-SCF OUTPUT_PATH file and print the result.

    Exit code 0 if the SCF converged AND the file is complete;
    non-zero otherwise (with the failure mode printed to stderr).
    Useful for ``vq``-style scripting:

    \b
        vb parse crystal ./mgo.out || echo "rerun this one"

    BACKEND is ``crystal`` (canonical); ``crystal14`` / ``crystal17`` /
    ``crystal23`` are accepted aliases for the same parser, which
    detects the actual version from the output banner.
    """
    module = _PARSE_BACKENDS.get(backend.lower())
    if module is None:  # pragma: no cover — Click already constrained the choice
        raise click.UsageError(f"unknown backend: {backend!r}")
    result = module.parse_output_file(output_path)

    if result.ok:
        version = (
            f"CRYSTAL{result.crystal_version}"
            if result.crystal_version is not None
            else "CRYSTAL?"
        )
        click.echo(f"{result.method:<6} {result.energy:.10f} Ha  "
                   f"(cycle {result.last_cycle}, {version})")
        sys.exit(0)
    else:
        click.echo(
            f"FAILED ({result.failure_mode}) — "
            f"method={result.method!r}, last_cycle={result.last_cycle}, "
            f"energy={result.energy!r}, truncated={result.truncated}, "
            f"converged={result.converged}",
            err=True,
        )
        sys.exit(2)


if __name__ == "__main__":  # pragma: no cover
    main()
