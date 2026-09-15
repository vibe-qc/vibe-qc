"""Thermodynamic-limit cross-check: megacell MP2 vs PySCF.pbc KMP2 (§10, §7).

The headline open validation of the megacell periodic-correlation route. Two
*independent* routes must reach the **same** per-unit-cell MP2 correlation
energy in the thermodynamic limit (TDL):

* **megacell -> infinity** — vibe-qc's own finite-supercell route. A size series
  of open-boundary supercells, with ``E_corr(N) = b·N + a`` extrapolated; the
  slope ``b`` is the bulk per-cell correlation (``megacell_mp2_tdl``). This side
  is a *molecular* calculation — no periodic exxdiv (see
  ``handovers/HANDOVER_PERIODIC_LOCAL_CORRELATION.md`` § 6.5).

* **k-mesh -> infinity** — PySCF.pbc periodic KMP2 (the maintainer-chosen
  oracle), a BvK-torus calculation sampled with a growing Monkhorst-Pack mesh.
  Run **out of process** through ``runner_pyscf.run_periodic_mp2`` so PySCF is
  never imported into ``python/vibeqc/`` (§10).

These are genuinely different boundary conditions (an open cluster vs a periodic
torus) computed by two independent codes, so their finite-size *extrapolations*
agree only to within truncation error — here ~7e-6 Ha/cell (~0.05% of e_corr),
which is the honest combined finite-size error, not a bug (§7).

Why the chain is isolated correctly on each side
------------------------------------------------
The open-boundary megacell has no periodic images at all. To make PySCF describe
the *same* isolated 1-D chain (and not a 3-D array of chains coupled by spurious
inter-image Coulomb), the KMP2 reference uses ``cell.dimension = 1`` with
``low_dim_ft_type = 'inf_vacuum'`` (Coulomb truncated transverse). Measured: the
KMP2 per-cell correlation is then bit-identical for 16/20/24/28-bohr transverse
boxes — the transverse vacuum has been removed from the physics, exactly as for
the open megacell. (A naive ``dimension=3`` supercell-with-vacuum does *not*
converge to the isolated-chain TDL — it drifts past it.)

PySCF reference — pinned with provenance
----------------------------------------
The KMP2 k-mesh series below is pinned as committed reference values (the
§0 "committed reference values with provenance" pattern of the handover, used
here because the periodic KMP2 run is the external oracle). ``test_pyscf_kmp2_
pins_reproduce`` regenerates a subset live through ``run_periodic_mp2`` so the
pin cannot silently rot.
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
from vibeqc.periodic_megacell_mp2 import megacell_mp2_tdl

# ---------------------------------------------------------------------------
# Pinned PySCF.pbc KMP2 reference — H2 chain / STO-3G, per unit cell (Ha).
#
#   Provenance
#   ----------
#   Code      : PySCF 2.13.0, pyscf.pbc.scf.KRHF(.density_fit()) + pyscf.pbc.mp.KMP2
#   System    : 1-D H2 chain, one H2 per cell; H-H bond 1.4 bohr, cell spacing
#               6.0 bohr along the periodic axis; STO-3G.
#   PBC setup : cell.dimension = 1, low_dim_ft_type = 'inf_vacuum' (Coulomb
#               truncated transverse -> isolated chain), default exxdiv='ewald';
#               20-bohr transverse box (per-cell e_corr is box-independent).
#   k-mesh    : Monkhorst-Pack (Nk, 1, 1).
#   Generated : 2026-06-17, out of process via
#               examples/regression/core/runner_pyscf.py::run_periodic_mp2
#               (PySCF never imported inside python/vibeqc/, §10).
#   Regenerate: test_pyscf_kmp2_pins_reproduce (below) re-runs a subset live.
#
# PySCF normalizes the k-sampled energy per unit cell, so e_corr is directly
# comparable to the megacell route's per-cell correlation energy.
PYSCF_KMP2_ECORR_PER_CELL = {
    4: -0.0132304239,
    6: -0.0132544700,
    8: -0.0132606381,
    10: -0.0132628880,
    12: -0.0132638982,
    16: -0.0132647079,
}

# Two-route agreement is limited by the combined finite-size truncation error of
# an open-boundary (megacell) extrapolation and a BvK k-mesh extrapolation.
# Measured residual after a 1/Nk fit of the KMP2 tail: ~6.8e-6 Ha/cell. The guard
# below (2e-5) keeps a comfortable margin over that while still catching any real
# regression: a genuine megacell bug shifts the correlation energy by mHa, ~50x
# above this tolerance.
TOL_TDL = 2.0e-5
# The raw largest-Nk point has not fully reached the limit; allow it a slightly
# looser band than the extrapolated value (measured gap ~9.9e-6).
TOL_RAW = 1.5e-5


def _h2_chain_cell():
    """One H2 per cell; periodic along z (6 bohr), 20-bohr transverse vacuum.

    Same physical chain as the pinned PySCF reference (which labels the periodic
    axis ``a[0]`` for dimension=1); the per-cell correlation energy is
    axis-orientation independent.
    """
    return vq.PeriodicSystem(
        3,
        np.diag([20.0, 20.0, 6.0]),
        [vq.Atom(1, [10.0, 10.0, 2.3]), vq.Atom(1, [10.0, 10.0, 3.7])],
    )


def _extrapolate_inverse_nk(nks, energies):
    """Fit E(Nk) = e_inf + c/Nk over the supplied tail; return e_inf."""
    nk = np.asarray(nks, dtype=float)
    y = np.asarray(energies, dtype=float)
    amat = np.vstack([np.ones_like(nk), 1.0 / nk]).T
    coef, *_ = np.linalg.lstsq(amat, y, rcond=None)
    return float(coef[0])


def test_megacell_tdl_matches_pyscf_kmp2_tdl():
    """megacell -> inf and PySCF KMP2 k-mesh -> inf reach the same per-cell TDL."""
    # --- vibe-qc megacell route (live): bulk per-cell = slope of E_corr(N) ---
    b_megacell = megacell_mp2_tdl(
        _h2_chain_cell(), "sto-3g", (2, 3, 4, 5, 6)
    ).e_corr_per_cell_bulk
    # Regression guard on the vibe-qc side (measured -0.0132746 on this system).
    assert b_megacell == pytest.approx(-0.0132746, abs=1e-5)

    # --- PySCF KMP2 series (pinned): monotone approach to the same limit ------
    nks = sorted(PYSCF_KMP2_ECORR_PER_CELL)
    series = [PYSCF_KMP2_ECORR_PER_CELL[n] for n in nks]

    # Approaches the megacell limit from above (less negative), monotonically:
    # each finer mesh is more negative and strictly closer to b_megacell.
    for lo, hi in zip(series, series[1:]):
        assert hi < lo  # more negative as Nk grows
    gaps = [abs(e - b_megacell) for e in series]
    for lo, hi in zip(gaps, gaps[1:]):
        assert hi < lo  # gap to the megacell TDL shrinks
    assert all(e > b_megacell for e in series)  # not yet overshot the limit

    # --- the cross-check: extrapolated limits agree within truncation error ---
    e_inf_kmp2 = _extrapolate_inverse_nk(nks[-3:], series[-3:])
    assert abs(e_inf_kmp2 - b_megacell) < TOL_TDL
    # Even the un-extrapolated finest mesh already lands within a tight band.
    assert abs(series[-1] - b_megacell) < TOL_RAW


def test_pyscf_kmp2_pins_reproduce():
    """The pinned PySCF KMP2 values still reproduce live (guards against rot).

    Out of process via the extended runner — PySCF is launched in a separate
    interpreter, never imported here (§10). Skips cleanly when PySCF is absent.
    """
    from examples.regression.core.runner_pyscf import run_periodic_mp2

    bohr_to_ang = 0.52917721067
    lattice_ang = np.diag([6.0, 20.0, 20.0]) * bohr_to_ang  # periodic axis = a[0]
    atoms_frac = [
        ("H", (2.3 / 6.0, 0.5, 0.5)),
        ("H", (3.7 / 6.0, 0.5, 0.5)),
    ]

    for nk in (4, 12):  # one coarse, one fine; full series is pinned above
        ref = run_periodic_mp2(
            lattice_ang=lattice_ang,
            atoms_frac=atoms_frac,
            basis="sto-3g",
            kmesh=(nk, 1, 1),
            dimension=1,
            low_dim_ft_type="inf_vacuum",
        )
        if ref.status == "unavailable":
            pytest.skip("PySCF not importable in the external interpreter")
        assert ref.status == "ok" and ref.converged
        assert ref.e_corr == pytest.approx(
            PYSCF_KMP2_ECORR_PER_CELL[nk], abs=1e-7
        )
