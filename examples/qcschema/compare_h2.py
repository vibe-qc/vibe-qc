"""Run one QCSchema H2 input with HF, MP2, and FCI in STO-3G.

Run from any directory with the vibe-qc environment's Python. Result JSON
files are written to the chosen output directory, never beside this script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import vibeqc as vq

# Pachucki, Phys. Rev. A 82, 032509 (2010), Table II, R = 1.4011 bohr.
BO_REFERENCE_HARTREE = -1.1744759314002167


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path.cwd(),
        help="directory for atomic result JSON files (default: current directory)",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name("h2_input.json")
    atomic_input = vq.read_qcschema(source)

    print("H2 at R = 1.4011 bohr, STO-3G")
    print("Method       Energy (Hartree)       E - BO reference (Hartree)")
    for method in ("HF", "MP2", "FCI"):
        job = {**atomic_input, "model": {**atomic_input["model"], "method": method}}
        output_path = args.output_dir / f"h2_{method.lower()}_result.json"
        result = vq.run_qcschema(job, output_path=output_path)
        energy = result["return_result"]
        print(f"{method:<6} {energy:>22.12f} {energy - BO_REFERENCE_HARTREE:>28.12f}")
    print(f"Reference: {BO_REFERENCE_HARTREE:.16f} Hartree")


if __name__ == "__main__":
    main()
