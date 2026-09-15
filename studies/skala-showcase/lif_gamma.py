#!/usr/bin/env python3
"""Experimental LiF rocksalt SKALA-1.1 Gamma GDF gate.

The two-atom primitive cell is from ``qc-input-library/scripts/_geometries.py``
(``lif_rocksalt``), with a = 4.03 Angstrom. Periodic SKALA is experimental;
this calculation establishes execution and finite-energy behavior, not
solid-state accuracy.
"""

from __future__ import annotations

import argparse
import math
import os
import platform
import sys
from pathlib import Path

import numpy as np
import vibeqc as vq

ANGSTROM_TO_BOHR = 1.0 / 0.529177210903


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="plan dispatch and outputs without loading PyTorch or the checkpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="explicit output stem; parent directory must already exist",
    )
    return parser.parse_args()


def require_supported_runtime() -> None:
    if not ((3, 11) <= sys.version_info[:2] <= (3, 13)):
        raise SystemExit(
            "SKALA requires Python 3.11 through 3.13; this interpreter is "
            f"{platform.python_version()}."
        )
    if sys.platform != "linux":
        raise SystemExit(
            f"SKALA real evaluation requires Linux; platform is {sys.platform!r}."
        )


def reserve_output_stem(requested: Path | None) -> Path:
    if requested is not None:
        stem = requested.expanduser().resolve()
        if not stem.parent.is_dir():
            raise FileNotFoundError(f"output parent does not exist: {stem.parent}")
        return stem

    root = Path(os.environ.get("VQ_WORKDIR", ".")).resolve()
    output_dir = root / "lif-gamma-results"
    output_dir.mkdir(exist_ok=False)
    probe = output_dir / ".write-probe"
    probe.write_text("ok\n", encoding="utf-8")
    probe.unlink()
    return output_dir / "lif-gamma"


def build_lif() -> vq.PeriodicSystem:
    lattice_constant = 4.03 * ANGSTROM_TO_BOHR
    half = lattice_constant / 2.0
    lattice = np.array(
        [[0.0, half, half], [half, 0.0, half], [half, half, 0.0]],
        dtype=float,
    )
    atoms = [
        vq.Atom(3, [0.0, 0.0, 0.0]),
        vq.Atom(9, [half, half, half]),
    ]
    return vq.PeriodicSystem(dim=3, lattice=lattice, unit_cell=atoms)


def main() -> None:
    args = parse_args()
    if not args.dry_run:
        require_supported_runtime()
    output = reserve_output_stem(args.output)
    system = build_lif()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    result = vq.run_periodic_job(
        system,
        basis,
        method="RKS",
        functional="skala-1.1",
        jk_method="gdf",
        output=output,
        dry_run=args.dry_run,
        max_iter=128,
        conv_tol_energy=1.0e-8,
        write_molden_file=True,
        # The current GDF population path is a molecular Gamma-block proxy,
        # not a validated periodic bond analysis. Keep it out of the campaign.
        write_population_file=False,
        write_density=False,
        density_spacing_bohr=0.25,
        output_qvf=True,
        progress=False,
        record_hostname=False,
    )

    if result is None:
        print(f"dry-run complete: {output}.system")
        return
    if not result.converged or not math.isfinite(result.energy):
        raise RuntimeError("LiF Gamma SKALA did not converge to a finite energy")

    print(f"output stem: {output}")
    print(f"E(SKALA-1.1, Gamma)/cell = {result.energy:.12f} hartree")
    print("periodic SKALA status: experimental")


if __name__ == "__main__":
    main()
