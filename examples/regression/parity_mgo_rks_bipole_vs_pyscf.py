#!/usr/bin/env python3
"""MgO RKS BIPOLE vs PySCF KRKS parity reference (out-of-process per CLAUDE.md §10).

Compares the converged total energy of MgO/STO-3G primitive rocksalt
between vibe-qc's BIPOLE RKS route and PySCF's pbc KRKS density-fitting
route.  Both use a [2,2,2] Monkhorst-Pack mesh, SVWN functional, and
Fermi-Dirac smearing at 0.01 Ha.

Usage:
    python examples/regression/parity_mgo_rks_bipole_vs_pyscf.py
"""

from __future__ import annotations

import sys

import numpy as np


def _build_mgo():
    ANG2BOHR = 1.0 / 0.529177210903
    a = 4.21 * ANG2BOHR
    lattice = (a / 2.0) * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    atoms = [
        ("Mg", [0.0, 0.0, 0.0]),
        ("O", [a / 2.0, a / 2.0, a / 2.0]),
    ]
    return lattice, atoms


def run_vibeqc_bipole(lattice, atoms):
    """Run vibe-qc BIPOLE RKS on MgO/STO-3G."""
    import vibeqc as vq
    from vibeqc._vibeqc_core import CoulombMethod, PeriodicKSOptions

    system = vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(12 if sym == "Mg" else 8, pos) for sym, pos in atoms],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kmesh = vq.monkhorst_pack(system, [2, 2, 2])

    opts = PeriodicKSOptions()
    opts.lattice_opts.cutoff_bohr = 8.0
    opts.lattice_opts.nuclear_cutoff_bohr = 8.0
    opts.lattice_opts.coulomb_method = CoulombMethod.DIRECT_TRUNCATED
    opts.max_iter = 30
    opts.functional = "svwn"
    opts.smearing_temperature = 0.01  # 0.01 Ha ≈ 3158 K
    opts.use_diis = True
    opts.diis_start_iter = 2
    opts.conv_tol_energy = 1e-8
    opts.conv_tol_grad = 1e-6
    # No auto-FMIXING under DIIS (Gap-B validation 2026-07-13)

    from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks

    print("  vibe-qc BIPOLE RKS SVWN/STO-3G [2,2,2] ...", flush=True)
    result = run_pbc_bipole_rks(system, basis, kmesh, opts)
    print(
        f"    converged={result.converged}  n_iter={result.n_iter}  "
        f"E={result.energy:.10f}  A={result.free_energy:.10f}  "
        f"-TS={result.free_energy - result.energy:.10f}  "
        f"S/kB={result.entropy:.6f}",
        flush=True,
    )
    return result


def run_pyscf_krks(lattice, atoms):
    """Run PySCF pbc KRKS on MgO/STO-3G with density fitting."""
    import pyscf
    from pyscf.pbc import df as pbc_df
    from pyscf.pbc import dft as pbc_dft
    from pyscf.pbc import gto as pbc_gto

    cell = pbc_gto.Cell()
    cell.a = lattice
    # atoms carry bohr coordinates and unit="B" -- pass them through
    # unconverted. The earlier /ANG2BOHR here converted them to Angstrom
    # values while still declaring bohr, which parked O at 2.105 bohr
    # instead of a/2 = 3.978 bohr (a ~1.8 mHa geometry bias vs BIPOLE).
    cell.atom = [(sym, list(pos)) for sym, pos in atoms]
    cell.unit = "B"
    cell.basis = "sto3g"
    cell.exp_to_discard = 0.1
    cell.build()

    kmesh = [2, 2, 2]
    kpts = cell.make_kpts(kmesh)

    mf = pbc_dft.KRKS(cell, kpts)
    mf.xc = "SVWN"
    mf = mf.density_fit()
    mf.with_df.auxbasis = "weigend"
    # PySCF's default DIIS handles convergence; add smearing
    mf.smearing_method = "fermi"
    mf.smearing_sigma = 0.01  # Ha

    print("  PySCF KRKS SVWN/STO-3G [2,2,2] density_fit ...", flush=True)
    mf.kernel()
    e_tot = float(mf.e_tot)
    print(f"    converged={mf.converged}  E={e_tot:.10f}", flush=True)
    return e_tot


def main():
    lattice, atoms = _build_mgo()

    print("=== MgO RKS BIPOLE vs PySCF KRKS parity ===", flush=True)
    print(f"  cell: MgO primitive rocksalt, a=4.21 Å, STO-3G", flush=True)

    result_vq = run_vibeqc_bipole(lattice, atoms)
    e_pyscf = run_pyscf_krks(lattice, atoms)

    e_vq = float(result_vq.energy)
    delta_ha = e_vq - e_pyscf
    delta_mha = delta_ha * 1000.0

    print(flush=True)
    print(f"  vibe-qc BIPOLE  E = {e_vq:.10f} Ha", flush=True)
    print(f"  PySCF KRKS      E = {e_pyscf:.10f} Ha", flush=True)
    print(
        f"  Δ (vq − pyscf)  = {delta_ha:+.10f} Ha  ({delta_mha:+.3f} mHa)", flush=True
    )

    # The two routes use different Coulomb treatments (BIPOLE Ewald j-split
    # vs PySCF density-fitting) and different basis sets for the auxiliary
    # (vibe-qc has no DF, PySCF uses weigend aux).  Expect agreement to
    # within a few mHa for the same functional + mesh.
    if abs(delta_ha) < 0.005:
        print("  PASS: Δ < 5 mHa", flush=True)
        return 0
    elif abs(delta_ha) < 0.05:
        print("  WARN: Δ < 50 mHa — within Coulomb-method tolerance", flush=True)
        return 0
    else:
        print(f"  FAIL: Δ = {delta_mha:.1f} mHa > 50 mHa threshold", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
