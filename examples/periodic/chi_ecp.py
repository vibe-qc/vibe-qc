"""Reproducible scalar ECP chi inputs for later vq Validator acceptance.

H2S/RHF and SH/UHF or ROHF use the bundled LANL2DZ basis and its ten-electron
sulfur core. The cell is neutral; explicit electron counts are eight and seven.
Coordinates and the 24 x 26 x 28 cell are in bohr. Mesh 3 includes complex
characters. The explicit 256 GiB numerical reservation cap and 512 MiB support
record cap clear the conservative metadata preflight (two workers). They do
not allocate that memory eagerly or impose a scheduler/process RAM limit;
Validator must request a suitable job resource envelope. These finite settings
are not a physical convergence claim.

Compare library/runner, resumed/uninterrupted, and frozen_core=0/1 only on
admitted RHF/ROHF MP2 and RHF/UHF CC routes. ORCA/CRYSTAL comparisons must use the exported exact
basis/ECP coefficients and the same geometry, charge, spin and explicit core
convention. A molecular ORCA result is context for a cell-size limit, not an
exact oracle for this finite periodic fitted Hamiltonian. Chi is experimental;
D93 fitted/post-HF remains not-qualified. This file contains no measured verdict.

--help uses only the standard library. --output must be outside the checkout.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', choices=('rhf', 'uhf', 'rohf'), default='rhf')
    parser.add_argument('--correlation', choices=('none', 'mp2', 'ccsd'), default='mp2')
    parser.add_argument('--mesh', type=int, choices=(1, 3), default=1)
    parser.add_argument('--frozen-core', type=int, choices=(0, 1), default=0,
                        help='explicit occupied orbitals per spin; never the ten ECP core electrons')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.reference == 'rohf' and args.correlation == 'ccsd':
        parser.error('ROHF coupled cluster remains unsupported')
    if args.reference == 'uhf' and args.correlation == 'mp2' and args.frozen_core:
        parser.error('public UMP2 frozen active spaces remain unsupported')
    if args.correlation == 'none' and args.frozen_core:
        parser.error('--frozen-core requires correlation')
    destination = args.output.expanduser().resolve()
    # A vq-staged standalone copy has no repository-relative location.
    source_root = next((p for p in Path(__file__).resolve().parents
                        if (p/'pyproject.toml').is_file()
                        and (p/'python/vibeqc').is_dir()), None)
    if source_root is not None and (destination == source_root or source_root in destination.parents):
        parser.error('--output must be outside the source checkout')

    import numpy as np
    import vibeqc as vq
    from vibeqc.periodic.chi.fitted import ChiCoulombControls

    atoms = [vq.Atom(16, [0., 0., 0.]), vq.Atom(1, [0., 0., 2.54])]
    if args.reference == 'rhf':
        atoms.append(vq.Atom(1, [2.538, 0., -.093]))
    system = vq.PeriodicSystem(3, np.diag([24., 26., 28.]), atoms,
                               charge=0, multiplicity=1 if args.reference == 'rhf' else 2)
    basis = vq.BasisSet(system.unit_cell_molecule(), 'lanl2dz')
    controls = ChiCoulombControls(
        one_electron_cutoff_bohr=12., pair_cutoff_bohr=12., reciprocal_cutoff_ha=20.,
        nuclear_real_cutoff_bohr=12., nuclear_reciprocal_cutoff_bohr_inv=8.,
        nuclear_alpha_bohr_inv=.4, support_policy='physical_pairs',
        memory_byte_cap=256*1024**3, support_record_byte_cap=512*1024**2)
    destination.mkdir(parents=True, exist_ok=True)
    name = f"{'h2s' if args.reference == 'rhf' else 'sh'}-{args.reference}-{args.correlation}-m{args.mesh}-f{args.frozen_core}"
    stem = destination/name
    correlation = None if args.correlation == 'none' else args.correlation
    restart = {}
    if args.checkpoint or args.resume:
        restart['chi_reference_checkpoint'] = stem.with_suffix('.hf.vqr')
        if correlation:
            restart['chi_correlation_checkpoint'] = stem.with_suffix('.correlation.vqr')
    if args.resume:
        restart['chi_reference_from'] = stem.with_suffix('.hf.vqr')
        if correlation:
            restart['chi_correlation_resume'] = stem.with_suffix('.correlation.vqr')
    vq.run_periodic_job(system, basis,
        method='aiccm', variant='chi', scf_reference=args.reference,
        correlation=correlation, aiccm_backend='ri', fit_source='classic',
        aux_basis='def2-universal-jkfit', chi_coulomb_controls=controls,
        chi_frozen_core=args.frozen_core if correlation else None,
        kpoints=(args.mesh, 1, 1), initial_guess='HCORE', smearing_temperature=0.,
        conv_tol_energy=1e-11, conv_tol_grad=1e-9, max_iter=250,
        output=str(stem)+('-resumed' if args.resume else ''),
        write_density=False, output_qvf=False, record_hostname=False, **restart)


if __name__ == '__main__':
    main()
