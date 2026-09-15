"""Demonstrate the bare-aux GDF divergence on diffuse aux (def2-svp-jkfit).

The current vibe-qc build_lpq_native uses a direct image sum on the
aux 2c metric M_PQ = <P_0 | 1/r12 | Q_T>_T. For any aux basis with
diffuse primitives (every standard JKfit aux) this sum is divergent
in real space — the L=0 component of an isolated Gaussian extends
indefinitely.

The fix (Sun 2017, PySCF _CCGDFBuilder) is the compensated-charge
construction: subtract a smooth model density with the same multipole
moment from each aux shell, so that the (aux−chg) lattice sum has
zero net L-th multipole and converges.

This script measures the divergence directly: build M and the 3c
tensor T at successively larger cutoffs, and report:

    cutoff_bohr,  ||M||_F,  max|M|,  cond(M),  ||T||_F

If the bare-aux path is working, all four should plateau. If broken
(the documented bug), ||M||_F and max|M| grow monotonically with
cutoff while cond(M) explodes.

Run locally:
    .venv/bin/python examples/debug/gdf_bare_aux_divergence.py
"""
from __future__ import annotations

import numpy as np
import vibeqc as vq
from vibeqc.aux_basis import build_lpq_native, make_aux_basis_set

ANG2BOHR = 1.0 / 0.529177210903


def build_h2_in_box(separation_bohr: float = 1.4, box_bohr: float = 12.0):
    half = 0.5 * separation_bohr
    atoms = [
        vq.Atom(1, [0.0, 0.0, -half]),
        vq.Atom(1, [0.0, 0.0, +half]),
    ]
    system = vq.PeriodicSystem(3, np.diag([box_bohr, box_bohr, box_bohr]), atoms)
    ao = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    aux = make_aux_basis_set(system.unit_cell_molecule(), aux_name="def2-svp-jk")
    return system, ao, aux


def main() -> int:
    system, ao, aux = build_h2_in_box()
    print(f"vibeqc {vq.__version__}")
    print(f"H2 / STO-3G / def2-svp-jk in {12.0:.1f}-bohr cubic box")
    print(f"  n_ao={ao.nbasis}, n_aux={aux.nbasis}")
    print()

    from vibeqc._vibeqc_core import (
        compute_2c_eri_lattice,
        compute_3c_eri_lattice,
    )
    from vibeqc.aux_basis import modrho_scales

    alpha = modrho_scales(aux)

    cutoffs = [12.0, 16.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0]
    print("BARE-AUX (the bug):")
    print(f"{'cutoff':>8s}  {'||M||_F':>12s}  {'max|M|':>12s}  "
          f"{'cond(M)':>12s}  {'||T||_F':>12s}  {'||Lpq||_F':>12s}")
    print("-" * 80)

    e_scf_history = []
    for cut in cutoffs:
        lo = vq.LatticeSumOptions()
        lo.cutoff_bohr = cut
        lo.nuclear_cutoff_bohr = cut

        M = np.asarray(compute_2c_eri_lattice(aux, system, lo))
        T = np.asarray(compute_3c_eri_lattice(ao, aux, system, lo))
        M = 0.5 * (M + M.T)

        # apply modrho (same as build_lpq_native does)
        M_mr = (alpha[:, None] * M) * alpha[None, :]
        T_mr = T * alpha[:, None, None]

        eig = np.linalg.eigvalsh(M_mr)
        cond = eig[-1] / max(abs(eig[0]), 1e-30)

        try:
            Lpq = build_lpq_native(
                system, ao, aux,
                lat_opts=lo, linear_dep_thr=1e-9, apply_modrho=True,
                algorithm="bare",
            )
            lpq_norm = float(np.linalg.norm(Lpq))
        except Exception as exc:
            lpq_norm = float("nan")
            print(f"  build_lpq_native failed at cutoff {cut}: {type(exc).__name__}: {exc}")

        print(f"{cut:>8.1f}  {np.linalg.norm(M_mr):>12.4e}  "
              f"{np.max(np.abs(M_mr)):>12.4e}  {cond:>12.4e}  "
              f"{np.linalg.norm(T_mr):>12.4e}  {lpq_norm:>12.4e}")

    print()
    print("Bug signature: ||M||_F grows monotonically with cutoff → the")
    print("bare-aux image sum on the 2c metric diverges for diffuse aux.")
    print()

    # ------------------------------------------------------------------
    # COMPCELL — the fix. Should PLATEAU with cutoff.
    # ------------------------------------------------------------------
    from vibeqc.aux_basis import (
        make_modrho_aux_basis, make_compensating_basis, make_fused_basis,
        fuse_transform_matrix, build_lpq_compcell,
    )
    mol = system.unit_cell_molecule()
    modrho = make_modrho_aux_basis(aux, mol)
    chg = make_compensating_basis(modrho, mol, eta=0.2)
    fused = make_fused_basis(modrho, chg, mol)
    A = fuse_transform_matrix(modrho, chg)
    print(f"COMPCELL (eta=0.2):  n_modrho={modrho.nbasis}, n_chg={chg.nbasis}, "
          f"n_fused={fused.nbasis}, A.shape={A.shape}")
    print(f"{'cutoff':>8s}  {'||M||_F':>12s}  {'max|M|':>12s}  "
          f"{'cond(M)':>12s}  {'||T||_F':>12s}  {'||Lpq||_F':>12s}")
    print("-" * 80)
    for cut in cutoffs:
        lo = vq.LatticeSumOptions()
        lo.cutoff_bohr = cut
        lo.nuclear_cutoff_bohr = cut
        M_fused = np.asarray(compute_2c_eri_lattice(fused, system, lo))
        T_fused = np.asarray(compute_3c_eri_lattice(ao, fused, system, lo))
        M_fused = 0.5 * (M_fused + M_fused.T)
        Mc = A @ M_fused @ A.T
        Mc = 0.5 * (Mc + Mc.T)
        Tc = np.einsum("iP,Pmn->imn", A, T_fused, optimize=True)
        eig = np.linalg.eigvalsh(Mc)
        cond = eig[-1] / max(abs(eig[0]), 1e-30)
        try:
            Lpq = build_lpq_compcell(system, ao, aux, lat_opts=lo,
                                     compcell_eta=0.2)
            lpq_norm = float(np.linalg.norm(Lpq))
        except Exception as exc:
            lpq_norm = float("nan")
            print(f"  build_lpq_compcell failed at cutoff {cut}: "
                  f"{type(exc).__name__}: {exc}")
        print(f"{cut:>8.1f}  {np.linalg.norm(Mc):>12.4e}  "
              f"{np.max(np.abs(Mc)):>12.4e}  {cond:>12.4e}  "
              f"{np.linalg.norm(Tc):>12.4e}  {lpq_norm:>12.4e}")
    print()
    print("Expected: ||M||_F PLATEAUS with cutoff → compcell construction")
    print("            converges in real space (Sun 2017 / PySCF _CCGDFBuilder).")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
