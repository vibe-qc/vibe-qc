#!/usr/bin/env python3
"""H2 / def2-SVP fixed-geometry RKS with Microsoft SKALA-1.1.

The two hydrogen coordinates are in bohr. This is a small molecular API
example, not a benchmark energy. Real evaluation requires Linux, Python
3.11 through 3.13, the ``[skala]`` extra, and the verified checkpoint.

Run:
    .venv-skala/bin/python examples/molecular/input-h2-rks-skala.py

Plan the files without importing PyTorch or downloading the model:
    .venv/bin/python examples/molecular/input-h2-rks-skala.py --dry-run

Outputs:
    output-h2-rks-skala.out          SCF trace and energy components
    output-h2-rks-skala.system       runtime, model, and grid provenance
    output-h2-rks-skala.molden       molecular orbitals
    output-h2-rks-skala.xyz          molecular structure
    output-h2-rks-skala.bibtex       machine-readable citations
    output-h2-rks-skala.references   human-readable citations
    output-h2-rks-skala.population.txt/.json
                                     population analysis
    output-h2-rks-skala.qvf          structure and wavefunction archive
    output-h2-rks-skala.dump         written only after a captured failure

Success means the SCF reports ``converged: True`` and a finite energy. It
does not establish accuracy outside the molecular domain studied for SKALA.
"""

from __future__ import annotations

import argparse
import math
import platform
import sys
from pathlib import Path

import vibeqc as vq

HERE = Path(__file__).resolve().parent
OUTPUT_STEM = HERE / "output-h2-rks-skala"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write the output plan without loading PyTorch or the checkpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_STEM,
        help="output stem (default: beside this script)",
    )
    return parser.parse_args()


def require_supported_runtime() -> None:
    """Stop before Torch import on runtimes not supported by SKALA."""
    if sys.version_info >= (3, 14):
        raise SystemExit(
            "SKALA requires Python 3.13 or earlier; this interpreter is "
            f"{platform.python_version()}."
        )
    if sys.platform == "darwin":
        raise SystemExit(
            "In-process SKALA evaluation is not supported on macOS yet. "
            "Use a compatible Linux CPU environment, or pass --dry-run."
        )
    if sys.platform != "linux":
        raise SystemExit(
            "In-process SKALA evaluation is currently validated only on "
            f"Linux CPU environments (got {sys.platform!r}); pass --dry-run "
            "for output planning on this platform."
        )


def main() -> None:
    args = parse_args()
    if not args.dry_run:
        require_supported_runtime()

    h2 = vq.Molecule(
        [
            vq.Atom(1, [0.0, 0.0, -0.7]),
            vq.Atom(1, [0.0, 0.0, 0.7]),
        ]
    )

    result = vq.run_job(
        h2,
        basis="def2-svp",
        method="rks",
        functional="skala-1.1",
        output=args.output,
        dry_run=args.dry_run,
        progress=False,
        record_hostname=False,
    )

    if result is None:
        print("Dry-run complete. Inspect the .system file beside the output stem.")
        return
    if not result.converged or not math.isfinite(result.energy):
        raise RuntimeError("SKALA RKS did not produce a converged finite energy")

    print(f"converged: {result.converged}")
    print(f"E(SKALA) = {result.energy:.10f} Ha")


if __name__ == "__main__":
    main()
