"""Compare native and OpenTrustRegion SCF for water or its doublet cation.

Run with --method rhf, uhf, rks, or uks and --output-dir outside the checkout.
KS calculations use PBE and a small teaching grid, not a converged grid study.
See docs/tutorial/opentrustregion.md for the interpretation.
"""

import argparse
from pathlib import Path

import numpy as np
import vibeqc as vq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("rhf", "uhf", "rks", "uks"), default="rhf")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not vq.has_opentrustregion():
        parser.error("Rebuild vibe-qc with -DVIBEQC_ENABLE_OPENTRUSTREGION=ON.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    unrestricted = args.method in ("uhf", "uks")
    # Atom coordinates are in bohr. Both backends use exactly this geometry.
    molecule = vq.Molecule(
        [vq.Atom(8, [0, 0, 0]), vq.Atom(1, [0, 1.43, 1.1]),
         vq.Atom(1, [0, -1.43, 1.1])],
        charge=1 if unrestricted else 0,
        multiplicity=2 if unrestricted else 1,
    )
    results = {}
    for backend in ("native", "opentrustregion"):
        options = getattr(vq, args.method.upper() + "Options")()
        options.initial_guess = vq.InitialGuess.HCORE
        options.max_iter = 100
        options.conv_tol_energy = 1e-9
        options.conv_tol_grad = 1e-7
        # Compare cold starts without a native stability restart schedule.
        options.stability_check = False
        if args.method.endswith("ks"):
            options.grid.n_radial = 25
            options.grid.n_theta = 12
            options.grid.n_phi = 24
        result = vq.run_job(
            molecule, basis="sto-3g", method=args.method,
            functional="PBE" if args.method.endswith("ks") else None,
            orbital_optimizer=backend,
            output=args.output_dir / f"{args.method}-{backend}",
            name_molecule=False, num_threads=1,
            **{args.method + "_options": options},
        )
        if not result.converged:
            raise RuntimeError(f"{backend} did not converge; inspect its .out file.")
        results[backend] = result

    native, otr = results["native"], results["opentrustregion"]
    delta = otr.energy - native.energy
    np.testing.assert_allclose(otr.energy, native.energy, rtol=0, atol=2e-8)
    densities = ("density_alpha", "density_beta") if unrestricted else ("density",)
    density_error = max(
        np.max(np.abs(np.asarray(getattr(otr, key)) - getattr(native, key)))
        for key in densities
    )
    if density_error > 2e-6:
        raise RuntimeError(f"The two solutions have different densities: {density_error}.")
    report = otr.opentrustregion
    if not (report.energy_converged and report.gradient_converged
            and report.final_residual < 1e-7 and report.gradient_rms < 1e-8):
        raise RuntimeError("The final physical convergence checks failed.")
    if not (report.stability_checked and report.stability_converged and report.stable):
        raise RuntimeError("The final internal-stability check did not certify a minimum.")
    print(f"\n{args.method.upper()} / STO-3G comparison")
    print(f"Native energy:           {native.energy:.12f} Ha")
    print(f"OpenTrustRegion energy:  {otr.energy:.12f} Ha")
    print(f"Energy difference:       {delta:+.3e} Ha")
    print(f"Maximum density error:   {density_error:.3e}")
    print(f"AO residual / grad RMS:  {report.final_residual:.3e} / {report.gradient_rms:.3e}")
    print(f"Termination:             {report.termination}")
    print(f"Stable manifold:         {report.manifold}")
    print(f"Accepted/trial/response: {report.accepted_evaluations}/"
          f"{report.trial_evaluations}/{report.response_evaluations}")


if __name__ == "__main__":
    main()
