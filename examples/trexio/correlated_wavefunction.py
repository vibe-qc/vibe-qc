#!/usr/bin/env python3
"""Export LiH CASCI/CASSCF(2,2), with one frozen core orbital, or full CI.

The Li-H distance is 3 bohr and the basis is STO-3G. This small model
demonstrates wavefunction interchange, not converged molecular energetics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vibeqc import Atom, Molecule, compute_overlap, read_trexio, run_job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("casci", "casscf", "fci"), default="casscf")
    parser.add_argument("--backend", choices=("hdf5", "text"), default="hdf5")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / ("lih-" + args.method)

    # BEGIN calculation
    molecule = Molecule([Atom(3, [0., 0., 0.]), Atom(1, [0., 0., 3.])])
    options = {} if args.method == "fci" else {"active_space": (2, 2)}
    result = run_job(molecule, basis="sto-3g", method=args.method,
                     output=stem, trexio=True, trexio_backend=args.backend,
                     progress=False, verbose=False, **options)
    suffix = ".trexio.h5" if args.backend == "hdf5" else ".trexio"
    path = Path(str(stem) + suffix)
    data = read_trexio(path)
    # END calculation

    # BEGIN checks
    coefficients = np.asarray(data.fields["determinant_coefficient"])
    norm = float(np.vdot(coefficients, coefficients).real)
    np.testing.assert_allclose(norm, 1., rtol=0, atol=1e-10)
    assert data.mo_type == "CI"
    assert "mo_energy" not in data.fields  # no invented canonical SCF energies
    np.testing.assert_allclose(data.energy, result.energy, rtol=0, atol=1e-10)
    for spin, count in (("up", data.n_up), ("dn", data.n_dn)):
        np.testing.assert_allclose(np.trace(data.fields["rdm_1e_" + spin]), count,
                                   rtol=0, atol=1e-9)
    density = data.density_matrices()[0]  # uses the full correlated 1-RDM
    overlap = compute_overlap(data.basis_set())
    np.testing.assert_allclose(np.trace(density @ overlap), 4., rtol=0, atol=1e-9)
    if args.method != "fci":
        assert data.mo_class.count("Core") == 1
        assert data.mo_class.count("Active") == 2
        # Six spatial orbitals fit in one 64-bit word per spin. Core MO 0
        # must be occupied in BOTH words of every determinant.
        assert all(int(word) & 1 for det in data.fields["determinant_list"] for word in det)
    # END checks

    assert Path(str(stem) + ".qvf").is_file()
    print(f"PASS {args.method}: {path}")
    print(f"Energy / Ha: {data.energy:.12f}; determinants: {len(coefficients)}; CI norm: {norm:.12f}")


if __name__ == "__main__":
    main()
