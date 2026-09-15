#!/usr/bin/env python3
"""Benzene SKALA-1.1 property and localized-orbital showcase.

The D6h geometry is the existing vibe-qc regression geometry in
``examples/regression/systems/molecules/benzene.py``. This is a fixed-geometry
visualization study, not a geometry optimization or benchmark energy.
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
    output_dir = root / "benzene-properties-results"
    output_dir.mkdir(exist_ok=False)
    probe = output_dir / ".write-probe"
    probe.write_text("ok\n", encoding="utf-8")
    probe.unlink()
    return output_dir / "benzene-properties"


def build_benzene() -> vq.Molecule:
    """Build planar D6h benzene with C-C 1.397 A and C-H 1.084 A."""
    carbon_radius = 1.397 * ANGSTROM_TO_BOHR
    hydrogen_radius = (1.397 + 1.084) * ANGSTROM_TO_BOHR
    carbons: list[vq.Atom] = []
    hydrogens: list[vq.Atom] = []
    for index in range(6):
        angle = index * math.pi / 3.0
        direction = (math.cos(angle), math.sin(angle), 0.0)
        carbons.append(vq.Atom(6, [carbon_radius * value for value in direction]))
        hydrogens.append(
            vq.Atom(1, [hydrogen_radius * value for value in direction])
        )
    return vq.Molecule(carbons + hydrogens)


def main() -> None:
    args = parse_args()
    if not args.dry_run:
        require_supported_runtime()
    output = reserve_output_stem(args.output)
    molecule = build_benzene()

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
        cube_spacing=0.25,
        cube_padding=4.0,
        output_qvf=True,
        qtaim=True,
        localize=["ibo", "boys", "pipek-mezey"],
        structured_log=True,
        perf_log=True,
        progress=False,
        record_hostname=False,
    )

    if result is None:
        print(f"dry-run complete: {output}.system")
        return
    if not result.converged or not math.isfinite(result.energy):
        raise RuntimeError(
            "benzene SKALA calculation did not converge to a finite energy"
        )

    print(f"output stem: {output}")
    print(f"E(SKALA-1.1)       = {result.energy:.12f} hartree")
    print(f"E[D3(BJ), B3LYP5]  = {result.e_dispersion:+.12f} hartree")
    print(f"E(SKALA-1.1 + D3)  = {result.energy_total:.12f} hartree")


if __name__ == "__main__":
    main()
