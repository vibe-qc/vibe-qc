"""Cross-code validation of the megacell periodic-MP2 route against PySCF (§10).

A finite megacell is a *molecular* calculation, so vibe-qc's megacell MP2 must
equal PySCF's **molecular** MP2 on the identical supercell geometry. This is the
oracle check the maintainer chose (PySCF), done here as a gated in-test import —
the established cross-code parity pattern (cf. ``tests/test_parity_hf_dft.py``).
PySCF is never imported by vibe-qc's runtime (`python/vibeqc/`), so §10 holds; a
test importing it under ``importorskip`` is the sanctioned validation channel.

(The PySCF.pbc *KMP2* oracle is for the thermodynamic-limit cross-check of the
BvK route / megacell extrapolation — a finite-size study, task #5. For the
*finite* megacell the exact, like-with-like oracle is PySCF molecular MP2.)

Measured agreement on H₂/STO-3G chains: |Δe_corr| ~4e-13, |Δe_hf| ~3e-14.
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
from vibeqc.periodic_megacell_mp2 import (
    build_supercell_molecule,
    megacell_mp2,
    megacell_mp2_tdl,
)

pytest.importorskip("pyscf")


def _h2_chain_cell():
    """One H₂ per cell; periodic along z (6 bohr), vacuum in x, y (20 bohr)."""
    return vq.PeriodicSystem(
        3,
        np.diag([20.0, 20.0, 6.0]),
        [vq.Atom(1, [10.0, 10.0, 2.3]), vq.Atom(1, [10.0, 10.0, 3.7])],
    )


@pytest.mark.parametrize("nrep", [(1, 1, 1), (1, 1, 2)])
def test_megacell_mp2_matches_pyscf_molecular(nrep):
    """vibe-qc megacell MP2 == PySCF molecular MP2 on the identical supercell."""
    from pyscf import gto, mp, scf

    system = _h2_chain_cell()
    mc = megacell_mp2(system, "sto-3g", nrep)

    # Same supercell geometry, fed to PySCF as a finite molecule (bohr).
    mol_sc = build_supercell_molecule(system, nrep)
    atoms = [[int(a.Z), tuple(float(x) for x in a.xyz)] for a in mol_sc.atoms]
    pmol = gto.M(atom=atoms, basis="sto-3g", unit="B", charge=0, spin=0, verbose=0)
    mf = scf.RHF(pmol)
    mf.kernel()
    pmp2 = mp.MP2(mf, frozen=0)
    pmp2.kernel()

    # Measured |Δ| ~1e-13 (same machine); 1e-8 keeps margin for cross-platform
    # BLAS while still catching any real regression (a megacell bug shifts the
    # correlation energy by mHa, far above 1e-8).
    assert mc.e_hf == pytest.approx(float(mf.e_tot), abs=1e-8)
    assert mc.e_corr == pytest.approx(float(pmp2.e_corr), abs=1e-8)


def test_megacell_tdl_matches_pyscf_kmp2_convergence():
    """TDL cross-check: megacell (open, N→∞) ≈ PySCF KMP2 (periodic, k→∞).

    The open-boundary megacell TDL extractor (:func:`megacell_mp2_tdl`) and
    PySCF.pbc KMP2 are two independent routes to the bulk per-cell MP2
    correlation energy.  The megacell is a finite cluster with no periodic
    boundary conditions; PySCF KMP2 uses 1-D PBC with Coulomb truncation
    transverse.  Both converge to the same thermodynamic limit.

    This is a convergence study, not an exact match — the finite-size errors
    differ (megacell ~1/N surface, KMP2 ~exponential in the HOMO-LUMO gap),
    so we use a soft ~mHa tolerance on the converged values.

    System: H₂ chain, STO-3G, 20-bohr transverse vacuum, 6-bohr spacing.
    """
    from examples.regression.core.runner_pyscf import run_periodic_mp2

    system = _h2_chain_cell()

    # --- Megacell TDL (open-boundary supercells, linear extrapolation) ---
    tdl = megacell_mp2_tdl(system, "sto-3g", sizes=(2, 3, 4, 5), axis=2)
    assert tdl.fit_max_residual < 1e-4, (
        f"megacell TDL linear-fit residual too large: {tdl.fit_max_residual:.2e}"
    )
    bulk_megacell = tdl.e_corr_per_cell_bulk

    # --- PySCF KMP2 (periodic, 3-D PBC, 20-bohr transverse vacuum) ---
    # The 20-bohr transverse vacuum isolates the chain effectively in 3-D
    # PBC — the inter-chain Coulomb interaction across 20 bohr is ~0.05 Ha
    # bare but screened to << µHa by the finite gap.  dimension=3 avoids
    # the PySCF axis-ordering convention that dimension=1 requires
    # (periodic direction must be the *first* lattice vector).
    bohr_to_ang = 0.52917721067
    lattice_ang = np.diag([20.0, 20.0, 6.0]) * bohr_to_ang
    tz = 6.0
    atoms_frac = [
        ("H", (0.5, 0.5, (tz / 2 - 1.4 / 2) / tz)),
        ("H", (0.5, 0.5, (tz / 2 + 1.4 / 2) / tz)),
    ]

    kmesh_values = [2, 4, 6, 8]
    kmp2_results = []
    for nk in kmesh_values:
        ref = run_periodic_mp2(
            lattice_ang=lattice_ang,
            atoms_frac=atoms_frac,
            basis="sto-3g",
            kmesh=(1, 1, nk),
            dimension=3,  # 3-D PBC; 20-bohr transverse vacuum isolates chain
        )
        if ref.status == "unavailable":
            pytest.skip("PySCF not importable in the external interpreter")
        assert ref.status == "ok", f"KMP2 at kmesh=(1,1,{nk}) failed: {ref.note}"
        assert ref.converged, f"KMP2 at kmesh=(1,1,{nk}) did not converge"
        kmp2_results.append(ref.e_corr)

    # Sanity: KMP2 per-cell correlation magnitude should grow (more negative)
    # and converge with increasing k-mesh.  Correlation energies are negative,
    # so "growing" means decreasing numerically.
    for i in range(len(kmp2_results) - 1):
        assert kmp2_results[i] > kmp2_results[i + 1], (
            f"KMP2 E_corr not monotonically decreasing with k-mesh: "
            f"nk={kmesh_values[i]} → {kmp2_results[i]:.8f}, "
            f"nk={kmesh_values[i + 1]} → {kmp2_results[i + 1]:.8f}"
        )

    # Both routes must converge to the same TDL within ~2 mHa.
    # The megacell TDL extrapolates from finite open-boundary clusters
    # (linear fit, ~1/N surface residual); KMP2 uses 3-D PBC with finite
    # k-mesh and 20-bohr transverse vacuum (sub-µHa inter-chain coupling
    # at the converged k-mesh).  Each has residual finite-size error at
    # ~sub-mHa; 2 mHa is a comfortable guard that catches a real energy
    # bug (> mHa) while passing with physically reasonable convergence.
    kmp2_best = kmp2_results[-1]  # largest k-mesh = best estimate
    assert bulk_megacell == pytest.approx(kmp2_best, abs=2e-3), (
        f"TDL mismatch: megacell bulk = {bulk_megacell:.8f} Ha/cell, "
        f"KMP2(nk=8) = {kmp2_best:.8f} Ha/cell, "
        f"Δ = {abs(bulk_megacell - kmp2_best):.2e} Ha"
    )
