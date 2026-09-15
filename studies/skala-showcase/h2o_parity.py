#!/usr/bin/env python3
"""Microsoft upstream H2O/def2-SVP SKALA-1.1 parity calculation.

The geometry and numerical targets are from ``skala/tests/test_ase.py`` in
Microsoft's SKALA repository. Coordinates are converted from Angstrom to the
bohr convention used by vibe-qc. Analytic SKALA forces are deliberately not
requested because that derivative path is gated in vibe-qc.
"""

from __future__ import annotations

import argparse
import math
import os
import platform
import sys
from pathlib import Path

import vibeqc as vq

ANGSTROM_TO_BOHR = 1.0 / 0.529177210903
EV_PER_HARTREE = 27.211386245988
E_ANGSTROM_TO_DEBYE = 4.803204712570263

# Microsoft SKALA upstream targets, skala/tests/test_ase.py, SKALA-1.1 row.
# The upstream ASE calculator enables the checkpoint's B3LYP5 D3(BJ)
# correction by default, so its energy target must be compared with
# ``result.energy_total`` rather than the bare SCF energy.
REFERENCE_ENERGY_EV = -2076.839069353949
REFERENCE_ENERGY_HARTREE = REFERENCE_ENERGY_EV / EV_PER_HARTREE
REFERENCE_DIPOLE_E_ANGSTROM = 0.41354587147386074
REFERENCE_DIPOLE_DEBYE = REFERENCE_DIPOLE_E_ANGSTROM * E_ANGSTROM_TO_DEBYE


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
    parser.add_argument(
        "--energy-tol",
        type=float,
        default=5.0e-4,
        help="maximum absolute energy difference in hartree (default: 5e-4)",
    )
    parser.add_argument(
        "--dipole-tol",
        type=float,
        default=5.0e-3,
        help="maximum absolute dipole difference in debye (default: 5e-3)",
    )
    return parser.parse_args()


def require_supported_runtime() -> None:
    """Fail before model import on unsupported real-run platforms."""
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
    """Reserve a vq-safe output child or validate an explicit stem."""
    if requested is not None:
        stem = requested.expanduser().resolve()
        if not stem.parent.is_dir():
            raise FileNotFoundError(f"output parent does not exist: {stem.parent}")
        return stem

    root = Path(os.environ.get("VQ_WORKDIR", ".")).resolve()
    output_dir = root / "h2o-parity-results"
    output_dir.mkdir(exist_ok=False)
    probe = output_dir / ".write-probe"
    probe.write_text("ok\n", encoding="utf-8")
    probe.unlink()
    return output_dir / "h2o-parity"


def build_water() -> vq.Molecule:
    """Return the exact ``ase.build.molecule('H2O')`` test geometry."""
    atoms_angstrom = [
        (8, (0.0, 0.0, 0.119262)),
        (1, (0.0, 0.763239, -0.477047)),
        (1, (0.0, -0.763239, -0.477047)),
    ]
    return vq.Molecule(
        [
            vq.Atom(z, [coordinate * ANGSTROM_TO_BOHR for coordinate in xyz])
            for z, xyz in atoms_angstrom
        ]
    )


def main() -> None:
    args = parse_args()
    if not args.dry_run:
        require_supported_runtime()
    output = reserve_output_stem(args.output)
    molecule = build_water()

    result = vq.run_job(
        molecule,
        basis="def2-svp",
        method="rks",
        functional="skala-1.1",
        density_fit=True,
        aux_basis="def2-svp-jk",
        grid_level="skala",
        dispersion="b3lyp5",
        output=output,
        dry_run=args.dry_run,
        write_molden_file=True,
        write_population_file=True,
        write_cube=["density", "homo", "lumo"],
        cube_spacing=0.2,
        cube_padding=4.0,
        output_qvf=True,
        qtaim=True,
        localize="ibo",
        structured_log=True,
        perf_log=True,
        progress=False,
        record_hostname=False,
    )

    if result is None:
        print(f"dry-run complete: {output}.system")
        return
    if not result.converged or not math.isfinite(result.energy):
        raise RuntimeError("H2O SKALA calculation did not converge to a finite energy")

    basis = vq.BasisSet(molecule, "def2-svp")
    dipole_debye = vq.dipole_moment(result, basis, molecule).total_debye
    energy_delta = float(result.energy_total) - REFERENCE_ENERGY_HARTREE
    dipole_delta = float(dipole_debye) - REFERENCE_DIPOLE_DEBYE

    print(f"output stem: {output}")
    print(f"E(SKALA-1.1)      = {result.energy:.12f} hartree")
    print(f"E[D3(BJ), B3LYP5] = {result.e_dispersion:+.12f} hartree")
    print(f"E(total)          = {result.energy_total:.12f} hartree")
    print(f"total-energy delta = {energy_delta:+.6e} hartree")
    print(f"|mu|               = {dipole_debye:.9f} debye")
    print(f"dipole delta       = {dipole_delta:+.6e} debye")

    if abs(energy_delta) > args.energy_tol:
        raise RuntimeError(
            "H2O energy parity failed: "
            f"|delta|={abs(energy_delta):.6e} > {args.energy_tol:.6e} hartree"
        )
    if abs(dipole_delta) > args.dipole_tol:
        raise RuntimeError(
            "H2O dipole parity failed: "
            f"|delta|={abs(dipole_delta):.6e} > {args.dipole_tol:.6e} debye"
        )


if __name__ == "__main__":
    main()
