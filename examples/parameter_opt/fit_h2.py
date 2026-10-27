"""First working native fits to normalized H2 reference curves.

For queued execution, include the normalized JSON alongside this script and
use vq's directory submission with an explicit Python command. The interpreter
must contain the intended parameter-opt source. See the parameter-opt guide.

All supplied observations are training data. This small workflow demonstrates
fitting, not transferability or scientific acceptance of a parameter set.
"""

import argparse
from collections import defaultdict
from pathlib import Path

from vibeqc.parameter_opt import (
    Dataset,
    DFTBRepulsion,
    FitProblem,
    Observation,
    ParameterSet,
    PM6Block,
    read_references,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", choices=("dftb", "pm6"), required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    records = read_references(args.references)
    if any(
        r.system.atomic_numbers != (1, 1)
        or r.system.charge
        or r.system.multiplicity != 1
        for r in records
    ):
        raise ValueError("this example requires neutral singlet H2 records")
    groups = defaultdict(list)
    for record in records:
        groups[(record.protocol.sha256, record.system.parent)].append(record)
    observations = []
    for key, group in sorted(groups.items()):
        if len(group) < 2:
            raise ValueError(
                "each protocol needs at least two geometries for contrasts"
            )
        anchor = min(group, key=lambda r: r.energy)
        for record in group:
            if record is not anchor:
                observations.append(
                    Observation(
                        record.sha256,
                        (record, anchor),
                        (1, -1),
                        category=record.protocol.sha256,
                        tolerance=0.001,
                    )
                )
    adapter = (
        DFTBRepulsion(((1, 1, 1.5),))
        if args.adapter == "dftb"
        else PM6Block(((1, "betas"),))
    )
    parameters = ParameterSet.initial(adapter.specs())
    problem = FitProblem(adapter, parameters, Dataset(tuple(observations)))
    result = problem.fit(checkpoint=args.output, resume=args.resume)
    print("status:", result["status"])
    print("parameters:", result["values"])
    print("baseline:", result["baseline_metrics"])
    print("fitted:", result["fitted_metrics"])
    print("native SHA256:", result["adapter"]["native_sha256"])
    if result["status"] != "converged":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
