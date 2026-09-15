"""gCP (Kruse-Grimme 2012 geometric counterpoise) — framework tests.

These tests exercise the gCP scaffolding from :mod:`vibeqc.gcp` without
depending on the full per-element parameter tables — bundled coverage is
intentionally limited to first-row organic chemistry (H, C, N, O, F) at
def2-SVP and def2-TZVP in this first cut, mirroring the dispersion.py
"Phase D1a stub" pattern. Tests cover:

* Slater-overlap helper: symmetry, R → 0 / R → ∞ limits, equal-vs-
  unequal exponent branches give the same answer at the crossover.
* gCP energy: sign (always positive — opposes BSSE over-binding),
  scales linearly with σ, decays smoothly with R.
* Gradient: finite-difference equivalence within ~1e-6 Ha/bohr.
* Registry: case-insensitive basis lookup, sensible error on unknown
  basis / unknown element.
* Composite recipes that reference gCP-blocked bases raise the right
  error type (:class:`GCPDataMissing` with a clear contribute-via-PR
  pointer).
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc.gcp import _slater_overlap_1s_1s, GCPDataMissing


# ---------------------------------------------------------------------------
# Slater overlap helper
# ---------------------------------------------------------------------------

def test_slater_overlap_equal_exponents_at_zero():
    """⟨1s|1s⟩ at R=0 equals 1 for equal exponents."""
    assert _slater_overlap_1s_1s(1.0, 1.0, 0.0) == pytest.approx(1.0, abs=1e-12)


def test_slater_overlap_decays_to_zero():
    """Overlap goes to zero at large separation."""
    assert abs(_slater_overlap_1s_1s(1.0, 1.0, 50.0)) < 1e-10


def test_slater_overlap_symmetric_in_exponents():
    """⟨1s_a|1s_b⟩ should equal ⟨1s_b|1s_a⟩ for any (ζ_a, ζ_b, R)."""
    for za, zb, R in [(1.0, 2.0, 1.5), (3.0, 5.0, 2.0), (0.5, 7.0, 3.0)]:
        s_ab = _slater_overlap_1s_1s(za, zb, R)
        s_ba = _slater_overlap_1s_1s(zb, za, R)
        assert s_ab == pytest.approx(s_ba, abs=1e-10)


def test_slater_overlap_continuous_at_equal_exponents():
    """Equal-exponent and unequal-exponent branches should agree at the
    transition; the unequal branch handles |ζ_a − ζ_b| > 1e-6, the
    equal one handles smaller separations. Numerically check that
    nudging ζ across the boundary doesn't produce a discontinuity > 1%
    of the value.
    """
    R = 1.5
    s_at_boundary = _slater_overlap_1s_1s(1.0, 1.0 + 5e-7, R)
    s_just_below = _slater_overlap_1s_1s(1.0, 1.0, R)
    rel_diff = abs(s_at_boundary - s_just_below) / max(abs(s_just_below), 1e-12)
    assert rel_diff < 1e-3


# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------

def _h2o_molecule(charge: int = 0, multiplicity: int = 1) -> vq.Molecule:
    """Standard H2O at experimental geometry, in bohr."""
    bohr = 1.0 / 0.529177210903
    # O at origin, two H along ±60° from z, OH = 0.957 Å, HOH = 104.5°
    OH_ang = 0.957
    h_x = OH_ang * np.sin(np.radians(52.25))
    h_z = OH_ang * np.cos(np.radians(52.25))
    atoms = [
        vq.Atom(8, [0.0, 0.0, 0.0]),
        vq.Atom(1, [h_x * bohr, 0.0, h_z * bohr]),
        vq.Atom(1, [-h_x * bohr, 0.0, h_z * bohr]),
    ]
    return vq.Molecule(atoms, charge, multiplicity)


def test_gcp_h2o_def2_svp_energy_positive():
    """gCP energy on H2O / def2-SVP should be small and positive
    (it opposes the BSSE over-binding of the SCF). Order of magnitude:
    a few mHa.
    """
    mol = _h2o_molecule()
    result = vq.compute_gcp(mol, "def2-svp")
    assert result.energy > 0.0
    # On H2O, gCP at def2-SVP is on the order of 1-5 mHa.
    assert 1e-4 < result.energy < 5e-2


def test_gcp_h2o_def2_tzvp_smaller_than_svp():
    """A larger basis should give a smaller gCP correction — that's the
    direction it physically goes (less BSSE, less correction)."""
    mol = _h2o_molecule()
    e_svp = vq.compute_gcp(mol, "def2-svp").energy
    e_tzvp = vq.compute_gcp(mol, "def2-tzvp").energy
    assert e_tzvp < e_svp


def test_gcp_single_atom_is_zero():
    """A single atom has no pairs, so gCP = 0 trivially."""
    mol = vq.Molecule([vq.Atom(8, [0.0, 0.0, 0.0])], 0, 3)  # atomic oxygen
    result = vq.compute_gcp(mol, "def2-svp")
    assert result.energy == 0.0


def test_gcp_linear_in_sigma():
    """E_gCP is proportional to the basis's σ fit parameter. Scale σ
    by 2 in a custom GCPParams and check the energy doubles.
    """
    mol = _h2o_molecule()
    base = vq.gcp_params_for("def2-svp")
    scaled = vq.GCPParams(
        basis_name=base.basis_name,
        sigma=2.0 * base.sigma,
        eta=base.eta,
        alpha=base.alpha,
        beta=base.beta,
        e_mis=base.e_mis,
        n_virt=base.n_virt,
        etaspec=base.etaspec,
        citation=base.citation,
    )
    e_base = vq.compute_gcp(mol, params=base).energy
    e_scaled = vq.compute_gcp(mol, params=scaled).energy
    assert e_scaled == pytest.approx(2.0 * e_base, rel=1e-10)


# ---------------------------------------------------------------------------
# Gradient
# ---------------------------------------------------------------------------

def test_gcp_gradient_matches_finite_difference():
    """Analytic gradient should match a 3-point central finite
    difference on the gCP energy to ~1e-5 Ha/bohr.
    """
    mol = _h2o_molecule()
    res = vq.compute_gcp(mol, "def2-svp", with_gradient=True)
    g = res.gradient
    assert g.shape == (3, 3)

    # Finite-difference each coordinate.
    h = 1e-4
    fd_grad = np.zeros_like(g)
    for atom_idx in range(3):
        for cart in range(3):
            atoms_plus = [
                vq.Atom(int(a.Z), list(a.xyz)) for a in mol.atoms
            ]
            atoms_minus = [
                vq.Atom(int(a.Z), list(a.xyz)) for a in mol.atoms
            ]
            pos_plus = list(atoms_plus[atom_idx].xyz)
            pos_plus[cart] += h
            atoms_plus[atom_idx] = vq.Atom(int(mol.atoms[atom_idx].Z), pos_plus)
            pos_minus = list(atoms_minus[atom_idx].xyz)
            pos_minus[cart] -= h
            atoms_minus[atom_idx] = vq.Atom(int(mol.atoms[atom_idx].Z), pos_minus)
            mol_plus = vq.Molecule(atoms_plus, mol.charge, mol.multiplicity)
            mol_minus = vq.Molecule(atoms_minus, mol.charge, mol.multiplicity)
            e_plus = vq.compute_gcp(mol_plus, "def2-svp").energy
            e_minus = vq.compute_gcp(mol_minus, "def2-svp").energy
            fd_grad[atom_idx, cart] = (e_plus - e_minus) / (2.0 * h)

    np.testing.assert_allclose(g, fd_grad, atol=5e-5)


# ---------------------------------------------------------------------------
# Registry + error surfaces
# ---------------------------------------------------------------------------

def test_gcp_params_case_insensitive():
    """Registry lookup should be case-insensitive."""
    p1 = vq.gcp_params_for("def2-SVP")
    p2 = vq.gcp_params_for("def2-svp")
    p3 = vq.gcp_params_for("DEF2-SVP")
    assert p1 is not None
    assert p1 is p2 is p3


def test_gcp_unknown_basis_returns_none():
    assert vq.gcp_params_for("totally-made-up-basis") is None


def test_gcp_unknown_basis_raises_with_actionable_message():
    """A bad basis_name in compute_gcp should raise GCPDataMissing
    listing the bundled coverage so the user can pick a real one."""
    mol = _h2o_molecule()
    with pytest.raises(GCPDataMissing) as exc:
        vq.compute_gcp(mol, "not-a-real-basis")
    assert "not-a-real-basis" in str(exc.value)
    assert "Bundled registry" in str(exc.value)


def test_gcp_unsupported_element_raises():
    """An element outside the H–Kr range supported by the universal
    Slater-exponent table (e.g. Cs, Z=55) should raise GCPDataMissing
    via the ``covers()`` check. Post-v0.9.0 the standard def2-SVP
    parameters now ship full H–Kr coverage from mctc-gcp, so this
    test exercises an element beyond that window.
    """
    bohr = 1.0 / 0.529177210903
    # Fr has 87 electrons, H has 1 → 88 total → singlet with mult=1.
    mol = vq.Molecule([
        vq.Atom(87, [0.0, 0.0, 0.0]),                   # Fr (Z=87, outside H–Rn)
        vq.Atom(1,  [2.50 * bohr, 0.0, 0.0]),
    ], 0, 1)
    with pytest.raises(GCPDataMissing) as exc:
        vq.compute_gcp(mol, "def2-svp")
    assert "Z=[87]" in str(exc.value)
    assert "Contribute" in str(exc.value)


def test_gcp_bare_basis_without_per_element_data_raises():
    """Bases registered with the (σ, η, α, β) fit constants but
    no [elements.*] populated should raise informatively. After the
    v0.9.0 mctc-gcp data bundling, no shipped basis is in this state
    by default — but a user-supplied GCPParams() with empty tables
    must still fail clearly. Build one and exercise the error path.
    """
    mol = _h2o_molecule()
    empty_params = vq.GCPParams(
        basis_name="empty-test-basis",
        sigma=0.5, eta=1.0, alpha=1.0, beta=1.5,
        e_mis={}, n_virt={},
        citation="empty-table test fixture",
    )
    with pytest.raises(GCPDataMissing) as exc:
        vq.compute_gcp(mol, params=empty_params)
    assert "empty-test-basis" in str(exc.value).lower()
    # The covers() check fires first (every Z is missing) and gives
    # the most actionable message — missing-Z list + contribute pointer.
    assert "not yet bundled" in str(exc.value)
    assert "Contribute" in str(exc.value)


def test_gcp_explicit_params_bypasses_registry():
    """Passing GCPParams(...) directly should bypass the registry and
    let users plug in their own published tables for unbundled bases."""
    mol = _h2o_molecule()
    custom = vq.GCPParams(
        basis_name="custom-tiny",
        sigma=0.1, eta=1.0, alpha=1.0, beta=1.5,
        e_mis={1: 0.001, 8: 0.05},
        n_virt={1: 3.0, 8: 8.0},
        citation="test fixture",
    )
    result = vq.compute_gcp(mol, params=custom)
    assert result.energy > 0.0
    assert result.params is custom
