"""EXPERIMENTAL: double hybrids on the direct-torus (real-Γ) route.

Gates for :func:`vibeqc.periodic.ccm.direct.run_ccm_double_hybrid_direct`
(milestone 2b of the real-Γ development line, 2026-07-18): hybrid-KS SCF
half on the direct route + spin-component-scaled MP2 on the converged KS
orbitals, both on ONE neutral cderi ``L`` (the BvK-ewald direct-reference
composition of ``run_ccm_ri_mp2(reference="direct")``, with KS orbitals).

This does NOT touch the ``run_ccm_ri_mp2`` default (``reference="neutral"``)
— that flip is the pending maintainer decision of ``HANDOVER_D2_EXXDIV.md``.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import Atom, Molecule, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.direct import (
    ccm_exchange_q0_madelung,
    run_ccm_double_hybrid_direct,
    run_ccm_rks_direct,
    run_ccm_uks_direct,
)

pytestmark = pytest.mark.experimental  # Γ-CCM (aiccm2026dev-a) research lane


def _h2_cubic_ccm(nrep=(2, 1, 1)):
    cell = PeriodicSystem(
        3, np.diag([6.0, 6.0, 6.0]),
        [Atom(1, [3.0, 3.0, 2.3]), Atom(1, [3.0, 3.0, 3.7])], 0, 1)
    return CCMSystem(cell, nrep, "sto-3g")


def _h2_box_pair():
    atoms = [Atom(1, [10.0, 10.0, 9.3]), Atom(1, [10.0, 10.0, 10.7])]
    cell = PeriodicSystem(3, np.diag([20.0, 20.0, 20.0]), atoms, 0, 1)
    return CCMSystem(cell, (1, 1, 1), "sto-3g"), Molecule(atoms, 0, 1)


@pytest.mark.slow
def test_double_hybrid_direct_vacuum_limit_matches_molecular():
    """Vacuum limit vs the molecular ``run_double_hybrid`` (canonical MP2,
    no DF, so the comparison has no aux-basis mismatch on the molecular
    side). Measured 2026-07-18: total −6.7e-4 — the SCF half carries
    −7.0e-4 (higher-order periodic-image residual, scaling with the
    a_x = 0.53 exact-exchange fraction; PBE's pure-XC residual is 4.5e-5,
    HSE06's SR residual −2.1e-4), and the PT2 halves agree to 3e-5 (the
    RI-vs-canonical floor). Gate at 2e-3."""
    from vibeqc import make_basis, run_double_hybrid

    ccm, mol = _h2_box_pair()
    r = run_ccm_double_hybrid_direct(ccm, "b2plyp")
    m = run_double_hybrid(mol, make_basis(mol, "sto-3g"), "b2plyp",
                          density_fit=False, density_fit_mp2=False)
    assert r.converged
    assert r.energy / ccm.n_cells == pytest.approx(m.e_total, abs=2e-3)
    # PT2 halves separately, at the RI floor:
    e_pt2_mol = m.e_total - m.rks.energy
    assert r.e_pt2 == pytest.approx(e_pt2_mol, abs=2e-4)


def test_double_hybrid_direct_composition_identity():
    """The result is exactly its declared composition: E = E_SCF +
    c_os·E_os + c_ss·E_ss with the functional's own coefficients, and the
    SCF half equals ``run_ccm_rks_direct`` under the double-hybrid bypass
    on the same L (b2plyp: c_os = c_ss = 0.27)."""
    ccm = _h2_cubic_ccm((2, 1, 1))
    r = run_ccm_double_hybrid_direct(ccm, "b2plyp")
    assert r.guess_selection == r.ks.guess_selection
    assert r.guess_selection.requested.name == "AUTO"
    assert r.guess_selection.effective.name == "HCORE"
    assert np.trace(r.ks.density @ r.ks.overlap) == pytest.approx(
        ccm.supercell.n_electrons(), abs=2e-12)
    assert r.mp2_c_os == pytest.approx(0.27, abs=1e-12)
    assert r.mp2_c_ss == pytest.approx(0.27, abs=1e-12)
    assert r.energy == pytest.approx(
        r.e_scf + r.mp2_c_os * r.e_pt2_os + r.mp2_c_ss * r.e_pt2_ss,
        abs=1e-13)
    assert r.e_scf == pytest.approx(r.ks.energy, abs=0.0)
    assert r.exchange_q0 == "BvK-ewald"
    assert r.exchange_q0_applicability == "active"


def test_double_hybrid_direct_seam_moves_scf_not_density():
    """The exchange-q=0 convention acts on the SCF half exactly as for a
    global hybrid: the ``exxdiv=None`` control's SCF sits above ewald by
    a_x·ξ_N·N_e/2 per supercell with the same converged density, while the
    PT2 halves differ only through the KS eigenvalue denominators (the
    exxdiv-on-denominators mechanism of HANDOVER_D2_EXXDIV.md § Part 2)."""
    ccm = _h2_cubic_ccm((2, 1, 1))
    r_ew = run_ccm_double_hybrid_direct(ccm, "b2plyp", exxdiv="ewald")
    r_no = run_ccm_double_hybrid_direct(ccm, "b2plyp", exxdiv=None)
    a_x = 0.53
    xi = ccm_exchange_q0_madelung(ccm)
    n_e = ccm.supercell.n_electrons()
    assert r_no.e_scf - r_ew.e_scf == pytest.approx(
        a_x * xi * n_e / 2.0, rel=1e-6)
    assert np.max(np.abs(np.asarray(r_no.ks.density)
                         - np.asarray(r_ew.ks.density))) < 1e-6
    # Ewald shifts every occupied down by a_x·ξ_N => larger |denominators|
    # => strictly less negative PT2 (the same direction as PySCF's ewald
    # KMP2 denominators).
    assert r_ew.e_pt2 > r_no.e_pt2


def test_double_hybrid_direct_fail_closed_surface():
    """Boundary: non-DH functionals are rejected here; DH functionals are
    rejected by the plain KS drivers with a pointer; open shells are
    rejected (no ROKS-CCM half)."""
    ccm = _h2_cubic_ccm((2, 1, 1))
    with pytest.raises(ValueError, match="not a"):
        run_ccm_double_hybrid_direct(ccm, "pbe0")
    with pytest.raises(NotImplementedError,
                       match="run_ccm_double_hybrid_direct"):
        run_ccm_rks_direct(ccm, "b2plyp")
    with pytest.raises(NotImplementedError,
                       match="run_ccm_double_hybrid_direct"):
        run_ccm_uks_direct(ccm, "b2plyp")
    odd = CCMSystem(
        PeriodicSystem(3, np.diag([7.0, 7.0, 7.0]),
                       [Atom(3, [3.5, 3.5, 3.5])], 0, 2),
        (1, 1, 1), "sto-3g")
    with pytest.raises(NotImplementedError, match="closed-shell"):
        run_ccm_double_hybrid_direct(odd, "b2plyp")
