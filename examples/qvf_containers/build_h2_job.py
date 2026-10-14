"""Build a pending molecular QVF job container.

Success criterion: ``runs/h2-rhf.qvf`` exists with ``run_status = pending``
and ``job_type = molecular`` (``inspect_job.py`` prints both). After
``vibeqc run`` the same file reports ``run_status = converged``,
``run_count = 1`` and an RHF/STO-3G energy of E = -1.1167143 Ha for H2 at
1.4 bohr (converges in 2 iterations; vibe-qc 0.17.10 development build).

Run from the repository root:

    .venv/bin/python examples/qvf_containers/build_h2_job.py

Then execute the same artifact in place:

    .venv/bin/vibeqc run examples/qvf_containers/runs/h2-rhf.qvf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import vibeqc as vq


HERE = Path(__file__).resolve().parent
DEFAULT_STEM = HERE / "runs" / "h2-rhf"


def build_container(stem: Path) -> Path:
    """Create H2/STO-3G RHF as a pending QVF container."""
    molecule = vq.Molecule(
        [
            vq.Atom(1, [0.0, 0.0, 0.0]),
            vq.Atom(1, [0.0, 0.0, 1.4]),
        ],
        charge=0,
        multiplicity=1,
    )
    stem.parent.mkdir(parents=True, exist_ok=True)
    return vq.write_pending_qvf(
        molecule,
        stem,
        method="rhf",
        basis="sto-3g",
        tasks=["single_point"],
        options={
            "perf_log": True,
            "structured_log": True,
            "progress": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a pending H2/STO-3G RHF QVF job container."
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
    print("Run it with:")
    print(f"  vibeqc run {path}")


if __name__ == "__main__":
    main()
