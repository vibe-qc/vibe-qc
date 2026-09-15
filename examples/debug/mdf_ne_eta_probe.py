"""Characterize the vibe-qc-MDF vs PySCF-MDF residual on Ne-in-box.

The G=0 self-term is proven complete by the per-aux gap (monopole
reproduces T_real - T_recip_dense to 7.5e-9; compensated-aux dipole is
~5e-7 -> zero). So the 0.14 mHa Ne vs PySCF MDF is NOT a higher-multipole
G=0 term. This script tests the leading alternative hypothesis: the
Gaussian/PW split point eta (vibe-qc fixed at 1.0; PySCF auto-guesses).

A true MDF result is eta-independent at convergence; eta-sensitivity at
the 0.1 mHa level quantifies the aux/mesh incompleteness floor and tells
us whether matching PySCF's eta closes the gap.

PySCF invoked out-of-process (CLAUDE.md sec 10) -- this is a debug script,
not vibe-qc runtime.
"""

from __future__ import annotations

import numpy as np

import vibeqc as vq
from vibeqc.pbc_gdf import run_pbc_gdf_rhf

BOX = 10.0


def vibeqc_mdf(eta, ke):
    system = vq.PeriodicSystem(3, np.diag([BOX] * 3), [vq.Atom(10, [0, 0, 0])])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    opt = vq.PeriodicRHFOptions()
    opt.max_iter = 120
    opt.conv_tol_energy = 1e-10
    r = run_pbc_gdf_rhf(
        system, basis, opt, aux_basis="def2-svp-jk", exxdiv="ewald",
        gdf_method="mdf", mdf_ke_cutoff=ke, compcell_eta=eta, progress=False,
    )
    return r.energy, r.converged


def pyscf_mdf():
    """PySCF MDF reference + the eta it auto-guesses + its mesh."""
    from pyscf.pbc import gto, scf, df

    cell = gto.Cell()
    cell.atom = "Ne 0 0 0"
    cell.a = np.eye(3) * BOX
    cell.unit = "B"
    cell.basis = "sto-3g"
    cell.verbose = 0
    cell.build()

    mdf = df.MDF(cell)
    mdf.auxbasis = "def2-svp-jkfit"
    mdf.build()
    eta_guess = getattr(mdf, "eta", None)
    mesh = list(getattr(mdf, "mesh", []))

    mf = scf.RHF(cell).density_fit(with_df=mdf)
    mf.exxdiv = "ewald"
    mf.conv_tol = 1e-10
    e = mf.kernel()
    return e, eta_guess, mesh


def main():
    print("=" * 64)
    print("Ne / sto-3g / def2-svp-jk(fit), 10-bohr box — MDF eta probe")
    print("=" * 64)

    e_pyscf, eta_pyscf, mesh_pyscf = pyscf_mdf()
    print(f"\nPySCF MDF:  E = {e_pyscf:.8f}")
    print(f"  PySCF eta guess = {eta_pyscf}")
    print(f"  PySCF mesh      = {mesh_pyscf}")

    print(f"\n{'eta':>6} {'ke':>5} {'E(vibeqc MDF)':>16} "
          f"{'dE vs PySCF (mHa)':>18} {'conv':>5}")
    etas = [0.1, 0.2, 0.5, 1.0, 2.0]
    if eta_pyscf is not None and float(eta_pyscf) not in etas:
        etas.append(float(eta_pyscf))
    for eta in sorted(etas):
        for ke in (60.0, 120.0):
            e, conv = vibeqc_mdf(eta, ke)
            print(f"{eta:6.3f} {ke:5.0f} {e:16.8f} "
                  f"{(e - e_pyscf) * 1e3:18.4f} {str(conv):>5}")


if __name__ == "__main__":
    main()
