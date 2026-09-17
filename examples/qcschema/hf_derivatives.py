"""Run H2 HF gradient and Hessian QCSchema drivers.

Result JSON files are written to the chosen output directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import vibeqc as vq


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

    for driver in ("gradient", "hessian"):
        job = {**atomic_input, "driver": driver}
        output_path = args.output_dir / f"h2_hf_{driver}.json"
        result = vq.run_qcschema(job, output_path=output_path)
        values = result["return_result"]
        print(f"{driver}: {len(values)} values, {output_path}")


if __name__ == "__main__":
    main()
