#!/usr/bin/env python3
"""Export and restart a two-k-point He cell using TREXIO.

The cubic lattice constant is 7 bohr. STO-3G, a two-point mesh and a
small GDF cutoff make this an IO demonstration, not a converged solid.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vibeqc import Atom, BasisSet, PeriodicSystem, read_trexio, run_periodic_job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("hdf5", "text"), default="hdf5")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # BEGIN calculation
    system = PeriodicSystem(3, np.eye(3) * 7., [Atom(2, [0., 0., 0.])])
    basis = BasisSet(system.unit_cell_molecule(), "sto-3g")
    settings = dict(method="RHF", jk_method="gdf", kpoints=(2, 1, 1),
                    aux_basis="def2-svp-jk", gdf_method="rsgdf", rsgdf_ke_cutoff=12.,
                    convergence="off", max_iter=60, conv_tol_energy=1e-8,
                    progress=False, verbose=0,
                    write_molden_file=False, write_population_file=False)
    result = run_periodic_job(system, basis, output=args.output_dir / "helium",
                              trexio=True, trexio_backend=args.backend, **settings)
    suffix = ".trexio.h5" if args.backend == "hdf5" else ".trexio"
    path = args.output_dir / ("helium" + suffix)
    data = read_trexio(path)
    # END calculation

    # BEGIN checks
    assert result.converged
    assert data.periodic
    blocks = data.mo_blocks()
    assert [(block.k_point, block.spin) for block in blocks] == [(0, 0), (1, 0)]
    np.testing.assert_allclose(data.lattice, system.lattice, rtol=0, atol=1e-12)
    np.testing.assert_allclose(sum(data.kpoint_weights), 1., rtol=0, atol=1e-12)
    electrons = sum(data.kpoint_weights[block.k_point] * sum(block.occupations)
                    for block in blocks)
    np.testing.assert_allclose(electrons, 2., rtol=0, atol=1e-10)
    for block, original in zip(blocks, result.mo_coeffs):
        np.testing.assert_allclose(block.coefficients, original, rtol=0, atol=1e-12)
    # END checks

    # BEGIN restart
    restarted = run_periodic_job(data.periodic_system(), data.basis_set(),
                                 output=args.output_dir / "helium-read",
                                 initial_guess="read", read_from=path, **settings)
    assert restarted.converged
    error = abs(restarted.energy - result.energy)
    assert error < 1e-8, error
    # END restart
    assert (args.output_dir / "helium.qvf").is_file()
    print(f"PASS periodic: {path}")
    print(f"Energy / Ha: {data.energy:.12f}; weighted electron count: {electrons:.12f}")
    print(f"READ energy error / Ha: {error:.3e}")


if __name__ == "__main__":
    main()
