"""Follow a spin-breaking orbital instability of H2 at a fixed 4 bohr bond.

Starts UHF from the RHF density, retaining one electron in each spin sector.
This is electronic orbital optimization; the nuclei never move. The resulting
broken-symmetry determinant is not a spin-pure singlet wavefunction.
"""

import argparse
from pathlib import Path

import numpy as np
import vibeqc as vq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not vq.has_opentrustregion():
        parser.error("Rebuild vibe-qc with -DVIBEQC_ENABLE_OPENTRUSTREGION=ON.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    molecule = vq.Molecule([vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 4.0])])
    basis = vq.BasisSet(molecule, "sto-3g")
    rhf_options = vq.RHFOptions()
    rhf_options.initial_guess = vq.InitialGuess.HCORE
    rhf_options.stability_check = False
    rhf_options.conv_tol_energy = 1e-9
    rhf_options.conv_tol_grad = 1e-7
    rhf = vq.run_job(
        molecule, basis="sto-3g", method="rhf", rhf_options=rhf_options,
        output=args.output_dir / "h2-rhf", name_molecule=False, num_threads=1,
    )
    if not rhf.converged:
        raise RuntimeError("The restricted seed did not converge.")

    uhf_options = vq.UHFOptions()
    uhf_options.initial_guess = vq.InitialGuess.READ
    # A restricted density contains both spins, so split it in half.
    uhf_options.read_density_alpha = np.asarray(rhf.density) / 2
    uhf_options.read_density_beta = np.asarray(rhf.density) / 2
    uhf_options.max_iter = 100
    uhf_options.conv_tol_energy = 1e-9
    uhf_options.conv_tol_grad = 1e-7
    otr_options = vq.OpenTrustRegionOptions()
    otr_options.stability = "follow"
    uhf = vq.run_job(
        molecule, basis="sto-3g", method="uhf", uhf_options=uhf_options,
        orbital_optimizer="opentrustregion", opentrustregion_options=otr_options,
        output=args.output_dir / "h2-uhf-follow", name_molecule=False, num_threads=1,
    )
    report = uhf.opentrustregion
    if not (uhf.converged and report.stability_checked
            and report.stability_converged and report.stable):
        raise RuntimeError("The unrestricted solution is not converged and internally stable.")
    if uhf.energy >= rhf.energy - 0.05:
        raise RuntimeError("The calculation did not escape the restricted saddle.")
    overlap = np.asarray(vq.compute_overlap(basis))
    for density in (uhf.density_alpha, uhf.density_beta):
        np.testing.assert_allclose(np.trace(density @ overlap), 1.0, rtol=0, atol=1e-9)
    print(f"\nRestricted energy:    {rhf.energy:.12f} Ha")
    print(f"Unrestricted energy:  {uhf.energy:.12f} Ha")
    print(f"Energy lowering:      {rhf.energy - uhf.energy:.6f} Ha")
    print(f"Unrestricted <S^2>:   {uhf.s_squared:.6f}")
    print("Electron counts:      alpha=1, beta=1 (verified)")
    print(f"Stable manifold:      {report.manifold}")


if __name__ == "__main__":
    main()
