"""Post-SCF localization of CCM crystalline orbitals (Task 1, Wannier).

``localise_ccm`` rotates the selected finite Γ-CCM Hamiltonian's occupied MOs
into localized orbitals by a unitary transform within that occupied space.
These tests pin the hard correctness gate — the rotation leaves the occupied
density (hence the total energy) invariant — plus unitarity, genuine
localization, and reality for a centrosymmetric cell. They do not claim a
Γ-CCM/χ-CCM or Γ-CCM/neutral-GDF construction equivalence.

Reference: Marzari & Vanderbilt, Phys. Rev. B 56, 12847 (1997); reuses
``vibeqc.periodic_localise.localise_periodic_gamma``.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import (
    CCMSystem,
    WannierAliasingReport,
    localise_ccm,
    localization_aliasing,
    localization_density_residual,
)
from vibeqc.periodic.ccm.scf import run_ccm_rhf

pytestmark = pytest.mark.experimental  # Γ-CCM (aiccm2026dev-a) research lane


def _h2_chain(nrep=(3, 1, 1)):
    """Dense 1-D H₂ chain (6-bohr cell, real inter-unit bonding; gapped)."""
    unit = PeriodicSystem(3, np.diag([6.0, 30.0, 30.0]),
                          [Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])], 0, 1)
    return CCMSystem(unit, nrep, "sto-3g")


@pytest.fixture(scope="module")
def ccm_scf():
    ccm = _h2_chain((3, 1, 1))
    return ccm, run_ccm_rhf(ccm, method="aiccm2026dev-a")


@pytest.mark.parametrize("method", ["pipek-mezey", "boys"])
def test_localise_preserves_density_and_energy(ccm_scf, method):
    """Unitary rotation within occ ⇒ occupied density (and total energy) invariant."""
    ccm, r = ccm_scf
    w = localise_ccm(r, ccm, method=method)
    assert localization_density_residual(r, w) < 1e-10


@pytest.mark.parametrize("method", ["pipek-mezey", "boys"])
def test_localise_is_unitary(ccm_scf, method):
    """Localized orbitals stay S^CCM-orthonormal: C_locᵀ S C_loc = I."""
    ccm, r = ccm_scf
    w = localise_ccm(r, ccm, method=method)
    S = np.asarray(r.overlap, dtype=float)
    G = w.C_loc.T @ S @ w.C_loc
    assert np.max(np.abs(G - np.eye(w.n_occ))) < 1e-9


@pytest.mark.parametrize("method", ["pipek-mezey", "boys"])
def test_localise_increases_locality(ccm_scf, method):
    """The localization objective genuinely increases (final ≥ initial)."""
    ccm, r = ccm_scf
    w = localise_ccm(r, ccm, method=method)
    assert w.objective_final >= w.objective_initial - 1e-9
    assert w.objective_final > w.objective_initial  # the canonical set is delocalized


def test_localise_real_for_centrosymmetric(ccm_scf):
    """Centrosymmetric cell ⇒ real localized orbitals (up to global phase)."""
    ccm, r = ccm_scf
    w = localise_ccm(r, ccm, method="pipek-mezey")
    assert not np.iscomplexobj(w.C_loc)
    # bond-centered Wannier functions: centers at the H₂ midpoint (z = 0.7 bohr,
    # folded into the home cell by the plain-r operator).
    assert np.allclose(w.centroids[:, 2], 0.7, atol=0.15)


def test_n_occ_is_full_supercell_count(ccm_scf):
    """All N·(occ per cell) Wannier functions are produced (3 H₂ units → 3)."""
    ccm, r = ccm_scf
    w = localise_ccm(r, ccm, method="pipek-mezey")
    assert w.n_occ == ccm.supercell.n_electrons() // 2 == 3


# --------------------------------------------------------------------------- #
# Task 1 M2 — Wannier-aliasing diagnostic (the cluster must exceed the
# localization length, else a Wannier tail wraps the BvK torus).
# --------------------------------------------------------------------------- #
def test_aliasing_not_flagged_for_well_localized_chain(ccm_scf):
    """A well-localized H2-bond Wannier in a roomy cell is not aliased, and the
    per-orbital ratios are exactly sqrt(Ω_i)/R_wsc."""
    ccm, r = ccm_scf
    w = localise_ccm(r, ccm, method="pipek-mezey")
    rep = localization_aliasing(ccm, w, safety=0.5)
    assert isinstance(rep, WannierAliasingReport)
    assert not rep.any_aliased
    assert rep.max_ratio < 0.5
    # ratios are exactly sqrt(spread)/R_wsc
    expect = np.sqrt(np.clip(np.asarray(w.spreads, float), 0.0, None)) / ccm.wsc_inscribed_radius
    assert np.allclose(np.asarray(rep.ratios), expect, atol=1e-12)
    assert rep.aliased == [bool(x > 0.5) for x in expect]


def test_aliasing_threshold_flips_deterministically(ccm_scf):
    """Tightening safety below the worst ratio flips the cluster verdict."""
    ccm, r = ccm_scf
    w = localise_ccm(r, ccm, method="pipek-mezey")
    loose = localization_aliasing(ccm, w, safety=0.5)
    tight = localization_aliasing(ccm, w, safety=0.5 * loose.max_ratio)
    assert not loose.any_aliased
    assert tight.any_aliased            # safety now below max_ratio -> fires
    assert tight.max_ratio == loose.max_ratio  # ratios are safety-independent


def test_aliasing_rejects_nonpositive_safety(ccm_scf):
    ccm, r = ccm_scf
    w = localise_ccm(r, ccm, method="pipek-mezey")
    with pytest.raises(ValueError):
        localization_aliasing(ccm, w, safety=0.0)
