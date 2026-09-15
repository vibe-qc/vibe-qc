"""Build a pending periodic QVF job container.

The example is deliberately tiny: one helium atom in an eight-bohr cubic
cell, Gamma-only SCC-DFTB. It exercises the basis-free periodic container
route without turning a format tutorial into a long solid-state calculation.

Run from the repository root:

    .venv/bin/python examples/qvf_containers/build_periodic_he_job.py
    .venv/bin/vibeqc run \
        examples/qvf_containers/runs/he-periodic-scc.qvf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import vibeqc as vq


HERE = Path(__file__).resolve().parent
DEFAULT_STEM = HERE / "runs" / "he-periodic-scc"


def build_container(stem: Path) -> Path:
    """Create a Gamma-only periodic He SCC-DFTB container."""
    box_bohr = 8.0
    system = vq.PeriodicSystem(
        dim=3,
        lattice=np.eye(3) * box_bohr,
        unit_cell=[vq.Atom(2, [0.0, 0.0, 0.0])],
        charge=0,
        multiplicity=1,
    )
    stem.parent.mkdir(parents=True, exist_ok=True)
    return vq.write_pending_qvf(
        system,
        stem,
        method="scc_dftb",
        kpoints=[1, 1, 1],
        tasks=["single_point"],
        options={
            "progress": False,
            "write_xyz_file": False,
            "write_xsf_structure_file": False,
            "write_cif_file": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a pending periodic He SCC-DFTB QVF container."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_STEM,
        help="Output stem; .qvf is added automatically.",
    )
    args = parser.parse_args()

    path = build_container(args.output)
    container = vq.load_job_container(path)
    print(f"Wrote {path}")
    print(f"run_status = {container.run_status}")
    print(f"job_type   = {container.job_type}")
    print(f"kpoints    = {container.spec['kpoints']}")
    print("Run it with:")
    print(f"  vibeqc run {path}")


if __name__ == "__main__":
    main()
