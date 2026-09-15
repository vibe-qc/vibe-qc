"""Do vibe-qc MDF and PySCF MDF converge to the SAME Ne energy?

The G=0 term is proven complete (no higher-multipole series). The
remaining vibe-qc-vs-PySCF-MDF residual (~0.1 mHa) is mesh-related. This
settles whether it is a real discrepancy or just different default mesh
sizes, by converging BOTH methods in their mesh and anchoring to the
molecular Ne/sto-3g RHF limit (10-bohr box ~ isolated atom).

PySCF out-of-process (CLAUDE.md sec 10); debug script, not runtime.
"""

from __future__ import annotations

import numpy as np

import vibeqc as vq
from vibeqc.pbc_gdf import run_pbc_gdf_rhf

BOX = 10.0


def vibeqc_mdf(ke, eta=0.5):
    system = vq.PeriodicSystem(3, np.diag([BOX] * 3), [vq.Atom(10, [0, 0, 0])])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    opt = vq.PeriodicRHFOptions()
    opt.max_iter = 150
    opt.conv_tol_energy = 1e-11
    r = run_pbc_gdf_rhf(
        system, basis, opt, aux_basis="def2-svp-jk", exxdiv="ewald",
        gdf_method="mdf", mdf_ke_cutoff=ke, compcell_eta=eta, progress=False,
    )
    return r.energy


def vibeqc_rsgdf(ke):
    system = vq.PeriodicSystem(3, np.diag([BOX] * 3), [vq.Atom(10, [0, 0, 0])])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    opt = vq.PeriodicRHFOptions()
    opt.max_iter = 150
    opt.conv_tol_energy = 1e-11
    r = run_pbc_gdf_rhf(
        system, basis, opt, aux_basis="def2-svp-jk", exxdiv="ewald",
        gdf_method="rsgdf", rsgdf_ke_cutoff=ke, progress=False,
    )
    return r.energy


def pyscf_mdf(mesh_n):
    from pyscf.pbc import gto, scf, df

    cell = gto.Cell()
    cell.atom = "Ne 0 0 0"
    cell.a = np.eye(3) * BOX
    cell.unit = "B"
    cell.basis = "sto-3g"
    cell.mesh = [mesh_n, mesh_n, mesh_n]
    cell.verbose = 0
    cell.build()
    mdf = df.MDF(cell)
    mdf.auxbasis = "def2-svp-jkfit"
    mdf.mesh = [mesh_n, mesh_n, mesh_n]
    mdf.build()
    mf = scf.RHF(cell).density_fit(with_df=mdf)
    mf.exxdiv = "ewald"
    mf.conv_tol = 1e-10
    return mf.kernel()


def pyscf_gdf_dense():
    """PySCF GDF (pure Gaussian, dense aux) as an independent anchor."""
    from pyscf.pbc import gto, scf, df

    cell = gto.Cell()
    cell.atom = "Ne 0 0 0"
    cell.a = np.eye(3) * BOX
    cell.unit = "B"
    cell.basis = "sto-3g"
    cell.verbose = 0
    cell.build()
    gdf = df.GDF(cell)
    gdf.auxbasis = "def2-svp-jkfit"
    gdf.build()
    mf = scf.RHF(cell).density_fit(with_df=gdf)
    mf.exxdiv = "ewald"
    mf.conv_tol = 1e-10
    return mf.kernel()


def main():
    def ke_of_mesh(n):
        b = 2 * np.pi / BOX
        gmax = ((n - 1) // 2) * b
        return 0.5 * gmax * gmax

    print("=" * 70)
    print("Ne / sto-3g / def2-svp-jk(fit), 10-bohr box — MDF convergence")
    print("=" * 70)

    e_gdf = pyscf_gdf_dense()
    print(f"\nPySCF GDF (dense aux, no PW)  E = {e_gdf:.8f}  [Gaussian-DF anchor]")

    print("\nPySCF MDF vs mesh:")
    e_pyscf = {}
    for n in (11, 19, 31, 41, 51):
        e = pyscf_mdf(n)
        e_pyscf[n] = e
        print(f"  mesh=({n:2d})^3  ke~{ke_of_mesh(n):6.1f}   E = {e:.8f}")

    print("\nvibe-qc MDF vs ke:")
    for ke in (40.0, 80.0, 160.0, 320.0):
        e = vibeqc_mdf(ke)
        print(f"  ke={ke:6.0f}              E = {e:.8f}")

    print("\nvibe-qc rsgdf (pure Gaussian) vs ke:")
    for ke in (200.0, 400.0):
        e = vibeqc_rsgdf(ke)
        print(f"  ke={ke:6.0f}              E = {e:.8f}")

    print("\nConverged comparison:")
    e_mdf_vq = vibeqc_mdf(320.0)
    e_mdf_py = e_pyscf[51]
    print(f"  vibe-qc MDF (ke=320)        = {e_mdf_vq:.8f}")
    print(f"  PySCF   MDF (mesh 51^3)     = {e_mdf_py:.8f}")
    print(f"  Delta (converged)           = {(e_mdf_vq - e_mdf_py)*1e3:.4f} mHa")
    print(f"  vibe-qc MDF vs PySCF GDF    = {(e_mdf_vq - e_gdf)*1e3:.4f} mHa")


if __name__ == "__main__":
    main()
