#!/usr/bin/env python3
"""Export and restart water RHF, OH UHF, or NaH ECP RHF through TREXIO.

All Cartesian coordinates below are in bohr. QVF remains enabled.
See docs/tutorial/trexio_exchange.md for the walkthrough.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vibeqc import (
    Atom, InitialGuess, Molecule, UHFOptions, compute_overlap,
    read_trexio, run_job, run_rhf, run_uhf,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("water", "oh", "nah"), default="water")
    parser.add_argument("--backend", choices=("hdf5", "text"), default="hdf5")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.case == "water":
        molecule = Molecule([Atom(8, [0., 0., 0.]),
                             Atom(1, [0., 1.43, -0.98]), Atom(1, [0., -1.43, -0.98])])
        basis_name, method = "def2-svp", "rhf"
    elif args.case == "oh":
        molecule = Molecule([Atom(8, [0., 0., 0.]), Atom(1, [0., 0., 1.83])],
                            multiplicity=2)
        basis_name, method = "def2-svp", "uhf"
    else:
        molecule = Molecule([Atom(11, [0., 0., 0.]), Atom(1, [0., 0., 3.5])])
        basis_name, method = "lanl2dz", "rhf"

    stem = args.output_dir / args.case
    # BEGIN export
    result = run_job(molecule, basis=basis_name, method=method, output=stem,
                     trexio=True, trexio_backend=args.backend,
                     localize=False, progress=False, verbose=False)
    suffix = ".trexio.h5" if args.backend == "hdf5" else ".trexio"
    path = Path(str(stem) + suffix)
    data = read_trexio(path)  # detects either backend
    # END export

    # BEGIN inspect
    basis = data.basis_set()
    overlap = compute_overlap(basis)
    blocks = data.mo_blocks()  # native AO ordering; columns are MOs
    orth_error = max(float(np.max(np.abs(
        block.coefficients.conj().T @ overlap @ block.coefficients
        - np.eye(block.coefficients.shape[1])))) for block in blocks)
    counts = [float(np.trace(density @ overlap).real)
              for density in data.density_matrices()]
    assert orth_error < 1e-9, orth_error
    np.testing.assert_allclose(sum(counts), data.n_electrons, rtol=0, atol=1e-9)
    np.testing.assert_allclose(data.energy, result.energy, rtol=0, atol=1e-11)
    if method == "uhf":
        assert [block.spin for block in blocks] == [0, 1]
        np.testing.assert_allclose(counts, [data.n_up, data.n_dn], rtol=0, atol=1e-9)
    if args.case == "nah":
        np.testing.assert_array_equal(data.atomic_numbers, [11, 1])
        np.testing.assert_array_equal(data.fields["ecp_z_core"], [10, 0])
        assert data.n_electrons == 2
    # END inspect

    # BEGIN restart
    molecule = data.molecule()
    options = data.ecp_options(UHFOptions()) if method == "uhf" else data.ecp_options()
    options.initial_guess = InitialGuess.READ
    driver = run_uhf if method == "uhf" else run_rhf
    restarted = driver(molecule, basis, options, read_from=path)
    assert restarted.converged
    restart_error = abs(restarted.energy - data.energy)
    assert restart_error < 1e-8, restart_error
    # END restart

    assert Path(str(stem) + ".qvf").is_file()  # TREXIO did not replace the default.
    print(f"PASS {args.case}: {path}")
    print(f"Energy / Ha: {data.energy:.12f}; electron counts: {counts}")
    print(f"MO orthogonality error: {orth_error:.3e}; READ energy error / Ha: {restart_error:.3e}")


if __name__ == "__main__":
    main()
