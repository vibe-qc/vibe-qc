"""Restart a native PBE water-cation calculation with OpenTrustRegion.

The public READ route takes the prior result in memory. Every calculation
writes through run_job to --output-dir. The compact grid is for teaching.
"""

import argparse
from pathlib import Path

import numpy as np
import vibeqc as vq


def options():
    opts = vq.UKSOptions()
    opts.max_iter = 100
    opts.conv_tol_energy = 1e-9
    opts.conv_tol_grad = 1e-7
    opts.stability_check = False
    opts.grid.n_radial = 25
    opts.grid.n_theta = 12
    opts.grid.n_phi = 24
    return opts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not vq.has_opentrustregion():
        parser.error("Rebuild vibe-qc with -DVIBEQC_ENABLE_OPENTRUSTREGION=ON.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    molecule = vq.Molecule(
        [vq.Atom(8, [0, 0, 0]), vq.Atom(1, [0, 1.43, 1.1]),
         vq.Atom(1, [0, -1.43, 1.1])],  # bohr
        charge=1, multiplicity=2,
    )
    basis = vq.BasisSet(molecule, "sto-3g")
    seed = vq.run_job(
        molecule, basis="sto-3g", method="uks", functional="PBE",
        uks_options=options(), initial_guess="hcore",
        output=args.output_dir / "uks-seed", name_molecule=False, num_threads=1,
    )
    if not seed.converged:
        raise RuntimeError("The native seed did not converge.")
    restarted = vq.run_job(
        molecule, basis="sto-3g", method="uks", functional="PBE",
        uks_options=options(), initial_guess="read", read_from=seed,
        orbital_optimizer="opentrustregion",
        output=args.output_dir / "uks-restarted", name_molecule=False, num_threads=1,
    )
    if not restarted.converged:
        raise RuntimeError("The restart did not converge; inspect uks-restarted.out.")
    np.testing.assert_allclose(restarted.energy, seed.energy, rtol=0, atol=2e-8)
    overlap = np.asarray(vq.compute_overlap(basis))
    for spin, electrons in (("alpha", 5), ("beta", 4)):
        density = np.asarray(getattr(restarted, "density_" + spin))
        np.testing.assert_allclose(np.trace(density @ overlap), electrons, rtol=0, atol=1e-9)
        np.testing.assert_allclose(density, getattr(seed, "density_" + spin), rtol=0, atol=2e-6)
    report = restarted.opentrustregion
    print(f"\nSeed / restart energy: {seed.energy:.12f} / {restarted.energy:.12f} Ha")
    print(f"Energy difference:     {restarted.energy - seed.energy:+.3e} Ha")
    print(f"Accepted evaluations:  {report.accepted_evaluations}")
    print(f"Final AO residual:     {report.final_residual:.3e}")
    print("Electron counts:       alpha=5, beta=4 (verified)")
    print(f"Termination:           {report.termination}")


if __name__ == "__main__":
    main()
