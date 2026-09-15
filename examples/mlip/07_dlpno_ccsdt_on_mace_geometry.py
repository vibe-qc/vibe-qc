#!/usr/bin/env python3
"""Stage 2: DLPNO-CCSD(T) on the MACE-optimized water geometry.

First run ``06_optimize_water_off23_for_dlpno.py --accept-asl`` in a
Python <= 3.13 MACE environment. Then run this input in a normal VibeQC
environment:

    .venv/bin/python examples/mlip/07_dlpno_ccsdt_on_mace_geometry.py

DLPNO-CCSD(T) supplies the correlated total energy. The emitted Molden
orbitals, populations, charges, bond orders, and dipole are properties of the
RHF reference because a correlated DLPNO property density is not implemented.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import vibeqc as vq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("mace-handoff-output"),
        help="directory produced by the MACE geometry stage",
    )
    args = parser.parse_args()
    work_dir = args.work_dir.resolve()
    geometry = work_dir / "mace-to-dlpno-water-opt.xyz"
    output = work_dir / "mace-to-dlpno-water-dlpno-ccsd-t"
    if not geometry.exists():
        raise SystemExit(
            f"Missing {geometry}. Run "
            "06_optimize_water_off23_for_dlpno.py --accept-asl first."
        )

    molecule = vq.Molecule.from_xyz(geometry, charge=0, multiplicity=1)
    result = vq.run_job(
        molecule,
        basis="cc-pvdz",
        method="dlpno-ccsd(t)",
        write_molden_file=True,
        write_population_file=True,
        output=output,
    )
    cc = result.dlpno_ccsd
    print(f"E(RHF reference)      = {cc.e_hf: .12f} Ha")
    print(f"E_corr(DLPNO-CCSD)    = {cc.e_corr: .12f} Ha")
    print(f"E((T1))               = {cc.e_t: .12f} Ha")
    print(f"E(DLPNO-CCSD(T))      = {result.energy_total: .12f} Ha")
    print(
        "Molden/population/dipole files describe the RHF reference; "
        "the correlated result supplied here is the energy."
    )


if __name__ == "__main__":
    main()
