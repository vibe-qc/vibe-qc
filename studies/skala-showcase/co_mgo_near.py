#!/usr/bin/env python3
"""Exploratory near-geometry CO/MgO(001) periodic SKALA calculation.

This fixed six-atom model uses a one-layer MgO(001) checkerboard with upright
C-down CO. The rounded Mg-C and C-O distances are illustrative coordinates
for contrasting a near and separated geometry; they are not literature
benchmark structures.

This is a vacuum-padded 3D, high-coverage, unrelaxed STO-3G surrogate. It is
not a vacuum-free 2D slab or a quantitative adsorption-energy benchmark, and
periodic SKALA dispersion is unavailable.
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
MG_C_DISTANCE_ANGSTROM = 2.5


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
    output_dir = root / "co-mgo-near-results"
    output_dir.mkdir(exist_ok=False)
    probe = output_dir / ".write-probe"
    probe.write_text("ok\n", encoding="utf-8")
    probe.unlink()
    return output_dir / "co-mgo-near"


def build_system() -> vq.PeriodicSystem:
    """Build a 3D vacuum-padded MgO(001) layer plus C-down CO."""
    lattice_a = 4.21
    bookkeeping_c = 30.0
    slab_z = 10.0
    half = lattice_a / 2.0
    adsorbate_x = half
    adsorbate_y = half
    carbon_z = slab_z + MG_C_DISTANCE_ANGSTROM
    oxygen_z = carbon_z + 1.15

    lattice_angstrom = np.diag([lattice_a, lattice_a, bookkeeping_c])
    atoms_angstrom = [
        (12, (0.0, 0.0, slab_z)),
        (8, (0.0, half, slab_z)),
        (8, (half, 0.0, slab_z)),
        (12, (half, half, slab_z)),
        (6, (adsorbate_x, adsorbate_y, carbon_z)),
        (8, (adsorbate_x, adsorbate_y, oxygen_z)),
    ]
    atoms = [
        vq.Atom(z, [coordinate * ANGSTROM_TO_BOHR for coordinate in xyz])
        for z, xyz in atoms_angstrom
    ]
    return vq.PeriodicSystem(
        dim=3,
        lattice=lattice_angstrom * ANGSTROM_TO_BOHR,
        unit_cell=atoms,
    )


def main() -> None:
    args = parse_args()
    if not args.dry_run:
        require_supported_runtime()
    output = reserve_output_stem(args.output)
    system = build_system()
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
        # The generic GDF population sidecar is not a validated periodic formula.
        write_population_file=False,
        write_density=False,
        density_spacing_bohr=0.3,
        output_qvf=True,
        progress=False,
        record_hostname=False,
    )

    if result is None:
        print(f"dry-run complete: {output}.system")
        return
    if not result.converged or not math.isfinite(result.energy):
        raise RuntimeError("near CO/MgO SKALA did not converge to a finite energy")

    print(f"output stem: {output}")
    print(f"Mg-C distance = {MG_C_DISTANCE_ANGSTROM:.3f} Angstrom")
    print(f"E_near/cell   = {result.energy:.12f} hartree")
    print("interpretation: exploratory fixed-composition point, not E_ads")
    print("periodic SKALA status: experimental")


if __name__ == "__main__":
    main()
