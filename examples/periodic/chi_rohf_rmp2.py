"""Proposed LiH2 chi ROHF/RMP2 Validator case; scientific acceptance is pending.

Use a delivered runtime containing the ROHF increment and submit through vq.
Select Gamma (--mesh 1) or complex characters (--mesh 3), then compare
--frozen-core 0 and 1 with the conventions in docs/chi_rohf_mp2.md. These
finite support settings do not establish physical convergence. Chi remains
experimental and D93 fitted/post-HF remains not-qualified.

Add --checkpoint to retain the accepted reference and correlation progress;
run again with --resume and the same inputs/output directory in a new process.
See docs/chi_restart.md. Restart acceptance also remains pending.

--output is a required directory outside the source checkout. --help needs
only the Python standard library and does not import the native runtime.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=int, choices=(1, 3), default=1,
                        help="complete unshifted character mesh (M, 1, 1)")
    parser.add_argument("--frozen-core", type=int, choices=(0, 1), default=0,
                        help="shared closed orbitals frozen per character")
    parser.add_argument("--output", type=Path, required=True,
                        help="output directory outside the source checkout")
    parser.add_argument("--checkpoint", action="store_true",
                        help="save separate HF and RMP2 restart archives")
    parser.add_argument("--resume", action="store_true",
                        help="restore saved HF and resume RMP2 with identical inputs")
    args = parser.parse_args()
    output_dir = args.output.expanduser().resolve()
    source_root = Path(__file__).resolve().parents[2]
    if output_dir == source_root or source_root in output_dir.parents:
        parser.error("--output must be outside the source checkout")

    import numpy as np
    import vibeqc as vq
    from vibeqc.periodic.chi.fitted import ChiCoulombControls

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"lih2-rohf-rmp2-mesh{args.mesh}-core{args.frozen_core}"
    system = vq.PeriodicSystem(
        3, np.diag([24.0, 26.0, 28.0]),
        [vq.Atom(3, [0.0, 0.0, 0.0]),
         vq.Atom(1, [0.0, 0.0, 3.0]),
         vq.Atom(1, [2.5, 0.0, -1.0])],
        charge=0, multiplicity=2,
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    controls = ChiCoulombControls(
        one_electron_cutoff_bohr=12.0,
        pair_cutoff_bohr=12.0,
        reciprocal_cutoff_ha=20.0,
        nuclear_real_cutoff_bohr=12.0,
        nuclear_reciprocal_cutoff_bohr_inv=8.0,
        nuclear_alpha_bohr_inv=0.4,
        tail_reciprocal_cutoff_ha=None,
        integral_screen_error=0.0,
        fit_screen_threshold=0.0,
        support_policy="physical_pairs",
    )
    restart = {}
    if args.checkpoint or args.resume:
        restart = dict(chi_reference_checkpoint=stem.with_suffix(".hf.vqr"),
                       chi_correlation_checkpoint=stem.with_suffix(".mp2.vqr"))
    if args.resume:
        restart.update(chi_reference_from=stem.with_suffix(".hf.vqr"),
                       chi_correlation_resume=stem.with_suffix(".mp2.vqr"))
    result = vq.run_periodic_job(
        system, basis,
        method="aiccm", variant="chi", scf_reference="rohf",
        correlation="mp2", aiccm_backend="ri", fit_source="classic",
        aux_basis="def2-SVP-JKFIT", chi_coulomb_controls=controls,
        chi_frozen_core=args.frozen_core, kpoints=(args.mesh, 1, 1),
        initial_guess="HCORE", smearing_temperature=0.0, max_iter=200,
        conv_tol_energy=1e-11, conv_tol_grad=1e-9,
        output=str(stem) + ("-resumed" if args.resume else ""),
        write_density=False, output_qvf=False, record_hostname=False, **restart,
    )
    corr = result.correlation
    # The public wrapper's energy is HF; e_total_correlated includes RMP2.
    print(f"ROHF / cell            {result.energy: .12f} Ha")
    print(f"RMP2 singles / cell    {corr.e_singles_per_cell: .12f} Ha")
    print(f"RMP2 doubles / cell    {corr.e_doubles_per_cell: .12f} Ha")
    print(f"RMP2 correlation / cell {result.e_correlation: .12f} Ha")
    print(f"Correlated total / cell {result.e_total_correlated: .12f} Ha")
    print(f"Method: {corr.method}; canonicalization: {corr.canonicalization}")
    print(f"Frozen convention: {corr.frozen_core_convention}")
    print(f"Restart mode: {corr.restart_mode}; resumed work: {corr.resumed_work_units}")
    print(f"Source identity: {corr.source_identity}")
    print(f"Reference identity: {corr.reference_identity}")
    print("Native scientific acceptance is pending; experimental, not-qualified.")


if __name__ == "__main__":
    main()
