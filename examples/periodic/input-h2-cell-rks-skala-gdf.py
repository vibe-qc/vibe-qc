#!/usr/bin/env python3
"""Experimental 3D H2-cell RKS / SKALA-1.1 / Gamma GDF smoke run.

The cell and coordinates are in bohr. This deliberately simple molecular-limit
cell demonstrates that the general full-grid XC provider reaches the periodic
GDF driver. It is not a converged solid-state calculation or a reference
energy. Converge the cell, k mesh, and GDF controls for scientific use.

Run:
    .venv-skala/bin/python \
        examples/periodic/input-h2-cell-rks-skala-gdf.py

Plan the files without importing PyTorch or downloading the model:
    .venv/bin/python \
        examples/periodic/input-h2-cell-rks-skala-gdf.py --dry-run

Outputs:
    output-h2-cell-rks-skala-gdf.out          SCF trace and energy components
    output-h2-cell-rks-skala-gdf.system       model and periodic provenance
    output-h2-cell-rks-skala-gdf.molden/.xyz  orbitals and structure
    output-h2-cell-rks-skala-gdf.bibtex       machine-readable citations
    output-h2-cell-rks-skala-gdf.references   human-readable citations
    output-h2-cell-rks-skala-gdf.qvf          structure and wavefunction archive
    output-h2-cell-rks-skala-gdf.xsf/.cif     periodic structures

Success means the experimental route reports ``converged: True`` and a finite
per-cell energy. It does not validate periodic SKALA accuracy.
"""

from __future__ import annotations

import argparse
import math
import platform
import sys
from pathlib import Path

import numpy as np

import vibeqc as vq

HERE = Path(__file__).resolve().parent
OUTPUT_STEM = HERE / "output-h2-cell-rks-skala-gdf"


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

    cell_length = 8.0
    system = vq.PeriodicSystem(
        dim=3,
        lattice=np.eye(3) * cell_length,
        unit_cell=[
            vq.Atom(
                1,
                [cell_length / 2, cell_length / 2, cell_length / 2 - 0.7],
            ),
            vq.Atom(
                1,
                [cell_length / 2, cell_length / 2, cell_length / 2 + 0.7],
            ),
        ],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    result = vq.run_periodic_job(
        system,
        basis,
        method="RKS",
        functional="skala-1.1",
        jk_method="gdf",
        output=args.output,
        dry_run=args.dry_run,
        progress=False,
        record_hostname=False,
        # GDF currently exposes only a molecular Gamma-block population proxy,
        # not a validated periodic bond-analysis formula.
        write_population_file=False,
        write_density=False,
    )

    if result is None:
        print("Dry-run complete. Inspect the .system file beside the output stem.")
        return
    if not result.converged or not math.isfinite(result.energy):
        raise RuntimeError("periodic SKALA RKS did not converge to a finite energy")

    print(f"converged: {result.converged}")
    print(f"E(SKALA)/cell = {result.energy:.10f} Ha")


if __name__ == "__main__":
    main()
