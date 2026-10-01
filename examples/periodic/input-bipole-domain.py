"""LiH BIPOLE domain selection, paired with the physical-domain tutorial.

No calculation starts without --run. --historical selects reproduction support.
The example is an input recipe, not a validated numerical reference.
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="execute one RHF SCF")
    parser.add_argument("--historical", action="store_true",
                        help="use historical index-ball support")
    parser.add_argument("--output", default="lih-bipole-domain",
                        help="output stem; choose separate stems for comparisons")
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return

    import numpy as np
    import vibeqc as vq
    from vibeqc.periodic_runner import run_periodic_job

    a = 4.084 / 0.529177210903  # Angstrom to bohr.
    lattice = (a / 2) * np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
    system = vq.PeriodicSystem(3, lattice, [
        vq.Atom(3, [0., 0., 0.]), vq.Atom(1, [a / 2, 0., 0.]),
    ])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    result = run_periodic_job(
        system, basis, method="RHF", jk_method="bipole", kpoints=(1, 1, 1),
        bipole_historical_domain=args.historical,
        output=args.output, max_iter=100, progress=True,
    )
    # The runner uses vibeqc.output for .out/.system/QVF. No hand-written log.
    print(f"converged={result.converged}; energy={result.energy:.12f} Ha")
    if not result.converged:
        raise SystemExit("SCF did not converge; this energy is not a reference.")


if __name__ == "__main__":
    main()
