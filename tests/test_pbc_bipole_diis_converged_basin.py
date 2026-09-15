"""Regression: DIIS must not move the BIPOLE RHF SCF off the converged
fixed point.

Bug (found 2026-06-05 certifying the BIPOLE production force path):
H₂/STO-3G RHF on the metallic-flavoured [2,1,1] mesh converged to a
*spurious* E = −0.347 Ha at bond length bz≈1.449 bohr, while every
neighbouring geometry gave the smooth, correct E ≈ −1.728 Ha. The FD
production gradient at bond 1.45 straddles 1.449, so |F| blew up to
~690 Ha/bohr.

Root cause — *not* a physical near-degeneracy (PySCF KRHF on the same
system/mesh gives a single, smooth E(bz) curve with no discontinuity at
1.449; vibe-qc with DIIS off is equally smooth). It is a numerical
artifact of the SCF driver:

  1. At bz≈1.449 the SAD→Fock→density step lands essentially *exactly*
     on the fixed point, so the DIIS error history collapses to
     machine-zero (‖e‖² ~ 1e-32). The Pulay B-matrix goes singular
     (cond ~1e19) and the degenerate solve returns large ±coefficients
     summing to 1.
  2. The driver declared convergence on the energy of the (good) input
     density, then *unconditionally* DIIS-extrapolated + re-diagonalised
     + rebuilt the density and returned that. Diagonalising the garbage
     extrapolated Fock occupied the antibonding orbital → −0.347.

Fix (``pbc_bipole.py``): on the converged iteration, diagonalise the
*physical* Fock F(D_used), not the extrapolated/level-shifted one. At a
true fixed point F(D) commutes with D, so the bare-Fock diagonalisation
reproduces the converged density exactly and yields canonical orbitals.

This is the CLAUDE.md §7 discipline: a wrong-energy "convergence" is a
bug to be tracked to its cause, not papered over by tightening DIIS.

Run:
    .venv/bin/python -m pytest tests/test_pbc_bipole_diis_converged_basin.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
from vibeqc._vibeqc_core import (
    InitialGuess,
    PeriodicRHFOptions,
    monkhorst_pack,
)
from vibeqc.pbc_bipole import run_pbc_bipole_rhf


def _opts(use_diis: bool) -> PeriodicRHFOptions:
    opts = PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 8.0
    opts.lattice_opts.nuclear_cutoff_bohr = 8.0
    opts.max_iter = 300
    opts.use_diis = use_diis
    opts.conv_tol_energy = 1e-9
    opts.initial_guess = InitialGuess.SAD
    return opts


def _h2(bz: float) -> vq.PeriodicSystem:
    lattice = 5.0 * np.eye(3)
    return vq.PeriodicSystem(
        3, lattice, [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, bz])]
    )


def _run(bz: float, use_diis: bool):
    sysp = _h2(bz)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    # k-mesh built once on the reference geometry, as the gradient path does.
    kmesh = monkhorst_pack(_h2(1.45), [2, 1, 1])
    # Pinned to the LEGACY exchange gauge (use_exchange_ewald_split=False).
    # This whole file validates the converged-iteration DIIS fix against
    # the SPECIFIC spurious basins it was written for — the −0.347 at
    # bz≈1.449 / −0.528 at bz≈1.399 (§0a), which are legacy-gauge DIIS
    # artifacts. The corrected gauge (the Phase-5 multi-k default since
    # 2026-06-13) converges cleanly to ≈ −1.277 on this tight 5-bohr box
    # with no spurious basin at these geometries, so the legacy gauge is
    # the meaningful vehicle here. The fix itself is gauge-independent.
    return run_pbc_bipole_rhf(
        sysp, basis, kmesh, _opts(use_diis),
        use_ewald_j_split=True, use_exchange_ewald_split=False,
        ewald_precision=1e-8, progress=False,
    )


# The geometry that triggered the original 2026-06-05 spurious basin
# (bz≈1.449, −0.347 Ha), plus its neighbours — uniformly spaced for the
# second-difference smoothness stencil below.
_SCAN = [1.447, 1.448, 1.449, 1.450, 1.451, 1.452, 1.453]

# Additional trigger geometries for the basin assertion only: bz=1.399
# (the 2026-06-10 −0.528 Ha basin re-exposed when a merge dropped the
# converged-iteration DIIS/FMIXING/LEVSHIFT guards — an FD displaced
# geometry of the bond-1.4 multi-k gradient tests) and its FD partner.
_BASIN_ONLY = [1.399, 1.401]


@pytest.mark.parametrize("bz", _SCAN + _BASIN_ONLY)
def test_rhf_211_lands_in_correct_basin(bz):
    """RHF + [2,1,1] lands in the bonding basin at every sampled bz,
    never either known spurious antibonding basin. Legacy gauge — see
    ``_run`` for why."""
    r = _run(bz, use_diis=True)
    assert r.converged, f"bz={bz} did not converge"
    # The bonding solution stays below -1.5 at this cutoff; the known
    # spurious basins were -0.347 and -0.528 Ha.
    assert r.energy < -1.5, (
        f"bz={bz}: E={r.energy:.6f} — landed in the spurious "
        "antibonding basin"
    )
    # The occupied (lowest) MO at every k must be the bonding orbital
    # (negative eigenvalue); the spurious basin occupied a positive one.
    for ik, eps in enumerate(r.mo_energies):
        occ = float(np.real(eps[0]))
        assert occ < 0.0, (
            f"bz={bz} k={ik}: occupied MO ε={occ:.4f} > 0 — "
            f"wrong-orbital occupation"
        )


def test_rhf_211_diis_matches_diis_off():
    """At the trigger geometry, DIIS-on must reproduce the DIIS-off
    (steepest-descent) converged energy. DIIS is an accelerator; it must
    not change *which* fixed point the SCF returns."""
    bz = 1.449
    e_on = _run(bz, use_diis=True).energy
    e_off = _run(bz, use_diis=False).energy
    assert abs(e_on - e_off) < 1e-7, (
        f"bz={bz}: DIIS-on E={e_on:.8f} vs DIIS-off E={e_off:.8f} — "
        f"DIIS moved the SCF to a different solution"
    )


def test_rhf_211_energy_curve_is_smooth():
    """E(bz) must be smooth across the trigger geometry: no single-point
    discontinuity (the −0.347 outlier produced a ~690 Ha/bohr FD spike)."""
    energies = np.array([_run(bz, use_diis=True).energy for bz in _SCAN])
    # Central second difference is a stencil for d²E/dbz²; a single-point
    # collapse would make the middle point's curvature enormous.
    d2 = np.abs(energies[2:] - 2.0 * energies[1:-1] + energies[:-2])
    assert np.all(d2 < 1e-4), (
        f"E(bz) not smooth — |Δ²E| = {d2} (energies={energies})"
    )
