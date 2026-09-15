"""Focused diagnostic: LiH 8-atom conv-cell GDF SCF component breakdown.

The Steps 1+2 gauge fix passes H2/12-bohr (molecular limit) but
diverges on every tight ionic crystal — LiH GDF gives -729 Ha vs
EWALD_3D's -29.33 Ha.  This script isolates *which* Fock piece is
wrong by running both GDF and EWALD_3D on the identical LiH cell
and dumping per-iteration energy components + matrix norms.

Capped at 6 SCF iters so it finishes fast (~10 min on compute-reference).

Run via vq:
    vq submit examples/regression/crystal_parity/baseline_sto3g/diag_lih_gdf_components.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("PYSCF_TMPDIR", "/tmp/pyscf_vibeqc")
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import vibeqc as vq

ANG2BOHR = 1.0 / 0.529177210903


def build_lih():
    a = 4.084 * ANG2BOHR
    cation_frac = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    anion_frac = [(0.5, 0.5, 0.5), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)]
    atoms = [vq.Atom(3, [fx * a, fy * a, fz * a]) for fx, fy, fz in cation_frac]
    atoms += [vq.Atom(1, [fx * a, fy * a, fz * a]) for fx, fy, fz in anion_frac]
    system = vq.PeriodicSystem(3, np.diag([a, a, a]), atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis, a


def main() -> int:
    system, basis, a = build_lih()
    print(f"vibeqc {vq.__version__}")
    print(f"LiH 8-atom conv cell: {basis.nbasis} BFs, "
          f"{system.n_electrons()} electrons, a={a:.4f} bohr")
    print()

    def _opts(method=None):
        o = vq.PeriodicRHFOptions()
        o.use_diis = True
        o.damping = 0.0
        o.max_iter = 6
        o.conv_tol_energy = 1e-8
        if method is not None:
            o.lattice_opts.coulomb_method = method
        return o

    # --- EWALD_3D reference ------------------------------------------
    print("=== EWALD_3D RHF (reference path) ===", flush=True)
    try:
        r = vq.run_rhf_periodic_gamma_ewald3d(
            system, basis, _opts(vq.CoulombMethod.EWALD_3D),
            omega=0.5, spacing_bohr=0.3,
        )
        print(f"  E_total={r.energy:.6f}  e_nuclear={r.e_nuclear:.6f}  "
              f"e_elec={r.e_electronic:.6f}  conv={r.converged}  iter={r.n_iter}")
        for it in r.scf_trace:
            print(f"    iter {it.iteration:2d}  E={it.energy:18.8f}  "
                  f"dE={it.delta_energy:12.3e}")
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
    print()

    # --- GDF Steps 1+2 path ------------------------------------------
    print("=== GDF RHF (Steps 1+2 gauge fix, default aux) ===", flush=True)
    try:
        r = vq.run_rhf_periodic_gamma_gdf(
            system, basis, _opts(), progress=False,
        )
        print(f"  E_total={r.energy:.6f}  e_nuclear={r.e_nuclear:.6f}  "
              f"e_elec={r.e_electronic:.6f}  conv={r.converged}  iter={r.n_iter}")
        print(f"  e_coulomb={r.e_coulomb:.6f}  "
              f"e_hf_exchange={r.e_hf_exchange:.6f}  "
              f"aux={r.aux_basis_name}  n_aux={r.n_aux}")
        for it in r.scf_trace:
            print(f"    iter {it.iteration:2d}  E={it.energy:18.8f}  "
                  f"dE={it.delta_energy:12.3e}")
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
    print()

    # --- GDF with explicit def2-svp-jk aux (what the sweep used) -----
    print("=== GDF RHF (Steps 1+2, aux=def2-svp-jk) ===", flush=True)
    try:
        r = vq.run_rhf_periodic_gamma_gdf(
            system, basis, _opts(), aux_basis="def2-svp-jk", progress=False,
        )
        print(f"  E_total={r.energy:.6f}  e_nuclear={r.e_nuclear:.6f}  "
              f"e_elec={r.e_electronic:.6f}  conv={r.converged}  iter={r.n_iter}")
        print(f"  e_coulomb={r.e_coulomb:.6f}  "
              f"e_hf_exchange={r.e_hf_exchange:.6f}  "
              f"aux={r.aux_basis_name}  n_aux={r.n_aux}")
        for it in r.scf_trace:
            print(f"    iter {it.iteration:2d}  E={it.energy:18.8f}  "
                  f"dE={it.delta_energy:12.3e}")
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()

    return 0


if __name__ == "__main__":
    sys.exit(main())
