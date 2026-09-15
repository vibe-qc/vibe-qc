#!/usr/bin/env python3
"""S22B water-dimer binding energy with experimental SKALA-1.1.

This reproduces one molecular geometry and reference reaction from the S22
subset used in the SKALA paper's GMTKN55 evaluation.  The official reaction is
``-E(02) + 2 E(02a)``; structure ``02b`` is not part of this benchmark entry.
This is not a reproduction of the paper's aggregate S22 mean absolute error.
The public GMTKN55 geometry data are CC-BY-4.0; see the references and
attribution in the showcase README.
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
KCAL_PER_HARTREE = 627.5094740631
S22B_BINDING_KCAL_MOL = 4.989

# GMTKN55/S22 structures 02 and 02a.  Coordinates are Angstrom.
# Source: grimme-lab/GMTKN55, S22 subset, CC-BY-4.0.
_DIMER = (
    (8, (-1.65542049075742, -0.12330037532823, 0.0)),
    (8, (1.24621235228714, 0.10268869233895, 0.0)),
    (1, (-0.70409020630450, 0.03193167180819, 0.0)),
    (1, (-2.03867258517291, 0.75372288209565, 0.0)),
    (1, (1.57598546497384, -0.38252143545728, -0.75856123927884)),
    (1, (1.57598546497384, -0.38252143545728, 0.75856123927884)),
)
_MONOMER_02A = (
    (8, (0.0, 0.0, -0.38904988213560)),
    (1, (0.75861214478398, 0.0, 0.19452494106780)),
    (1, (-0.75861214478398, 0.0, 0.19452494106780)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="plan both jobs without loading PyTorch or the checkpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="explicit output prefix; parent directory must already exist",
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


def reserve_output_prefix(requested: Path | None) -> Path:
    if requested is not None:
        prefix = requested.expanduser().resolve()
        if not prefix.parent.is_dir():
            raise FileNotFoundError(f"output parent does not exist: {prefix.parent}")
        return prefix

    root = Path(os.environ.get("VQ_WORKDIR", ".")).resolve()
    output_dir = root / "s22-water-dimer-results"
    output_dir.mkdir(exist_ok=False)
    probe = output_dir / ".write-probe"
    probe.write_text("ok\n", encoding="utf-8")
    probe.unlink()
    return output_dir / "s22-water"


def molecule(atoms_angstrom: tuple[tuple[int, tuple[float, float, float]], ...]):
    return vq.Molecule(
        [
            vq.Atom(z, [coordinate * ANGSTROM_TO_BOHR for coordinate in xyz])
            for z, xyz in atoms_angstrom
        ]
    )


def run_component(
    component: str,
    atoms_angstrom: tuple[tuple[int, tuple[float, float, float]], ...],
    *,
    output_prefix: Path,
    dry_run: bool,
    properties: bool,
):
    kwargs: dict[str, object] = {}
    if properties:
        kwargs.update(
            write_molden_file=True,
            write_population_file=True,
            write_cube=["density", "homo", "lumo"],
            cube_spacing=0.2,
            cube_padding=4.0,
            output_qvf=True,
            qtaim=True,
            localize="ibo",
        )
    else:
        kwargs.update(
            write_molden_file=False,
            write_population_file=False,
            output_qvf=False,
        )

    return vq.run_job(
        molecule(atoms_angstrom),
        basis="def2-qzvp",
        method="rks",
        functional="skala-1.1",
        density_fit=True,
        aux_basis="def2-universal-jkfit",
        grid_level="skala",
        dispersion="b3lyp5",
        output=Path(f"{output_prefix}-{component}"),
        dry_run=dry_run,
        structured_log=True,
        perf_log=True,
        progress=False,
        record_hostname=False,
        **kwargs,
    )


def main() -> None:
    args = parse_args()
    if not args.dry_run:
        require_supported_runtime()
    prefix = reserve_output_prefix(args.output)

    dimer = run_component(
        "dimer",
        _DIMER,
        output_prefix=prefix,
        dry_run=args.dry_run,
        properties=True,
    )
    monomer = run_component(
        "monomer-02a",
        _MONOMER_02A,
        output_prefix=prefix,
        dry_run=args.dry_run,
        properties=False,
    )

    if dimer is None or monomer is None:
        if not (dimer is None and monomer is None):
            raise RuntimeError("S22 dry-run returned an inconsistent result set")
        print(f"dry-run complete: {prefix}-{{dimer,monomer-02a}}.system")
        return

    results = (dimer, monomer)
    if any(not result.converged for result in results) or any(
        not math.isfinite(float(result.energy_total)) for result in results
    ):
        raise RuntimeError("one or more S22 SKALA calculations did not converge")

    bare_binding = (
        2.0 * float(monomer.energy) - float(dimer.energy)
    ) * KCAL_PER_HARTREE
    total_binding = (
        2.0 * float(monomer.energy_total) - float(dimer.energy_total)
    ) * KCAL_PER_HARTREE
    d3_contribution = total_binding - bare_binding
    error = total_binding - S22B_BINDING_KCAL_MOL

    print(f"output prefix: {prefix}")
    print(f"D_e(SKALA-1.1)          = {bare_binding:+.6f} kcal/mol")
    print(f"D_e[D3(BJ), B3LYP5]     = {d3_contribution:+.6f} kcal/mol")
    print(f"D_e(SKALA-1.1 + D3)     = {total_binding:+.6f} kcal/mol")
    print(f"S22B reference           = {S22B_BINDING_KCAL_MOL:+.6f} kcal/mol")
    print(f"signed error              = {error:+.6f} kcal/mol")


if __name__ == "__main__":
    main()
