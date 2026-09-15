#!/usr/bin/env python3
"""Stage 1: optimize molecular water with MACE-OFF23.

MACE-OFF23 is the registered organic-family model and its weights are under
the Academic Software License (academic, non-commercial use only). Run only
after confirming that the license covers your work:

    .venv-mace/bin/python examples/mlip/06_optimize_water_off23_for_dlpno.py \
        --accept-asl

The optimized geometry is written under ``--work-dir`` (default:
``mace-handoff-output``). Run the stage-2 input with a normal VibeQC
environment; it does not need MACE.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import vibeqc as vq
from vibeqc.mlip import MLIPOptions

HERE = Path(__file__).resolve().parent
INPUT = HERE.parent / "h2o.xyz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--accept-asl",
        action="store_true",
        help=(
            "acknowledge that this MACE-OFF23 run is academic and "
            "non-commercial under the ASL"
        ),
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("mace-handoff-output"),
        help="directory for the geometry and calculation siblings",
    )
    args = parser.parse_args()
    if not args.accept_asl:
        raise SystemExit(
            "MACE-OFF23 requires explicit ASL acknowledgment. Review the "
            "license, then rerun with --accept-asl for academic, "
            "non-commercial use."
        )
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    output = work_dir / "mace-to-dlpno-water-opt"

    molecule = vq.Molecule.from_xyz(INPUT)
    result = vq.run_job(
        molecule,
        method="mace",
        mlip_options=MLIPOptions(
            model="off23-medium",
            accept_academic_license=True,
            device="cpu",
            dtype="float64",
        ),
        optimize=True,
        fmax=0.02,
        output=output,
    )
    print(f"MACE-OFF23 optimized energy: {float(result.energy):.12f} Ha")
    print(f"Next input geometry: {output.with_suffix('.xyz')}")
    print(
        "Do not compare this absolute MACE energy with the subsequent "
        "DLPNO-CCSD(T) electronic total."
    )


if __name__ == "__main__":
    main()
