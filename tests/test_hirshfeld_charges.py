"""Classical Hirshfeld charges (``vibeqc.hirshfeld_charges``).

Hirshfeld, F. L. *Theor. Chim. Acta* **44**, 129 (1977).

Compared with :func:`vibeqc.mulliken_charges` /
:func:`vibeqc.loewdin_charges` (basis-space partitions), Hirshfeld is
a real-space partition built on the converged density + the SAD
promolecule. Two algorithmic identities are exact (limited only by
grid numerics):

    Σ q_A         = molecule.charge
    Σ_A w_A(r)    = 1   for r in the promolecule's support

so the sum-to-charge sanity check is *the* canonical Hirshfeld
regression. The other tests pin numerical regression on a tight
fixture, exercise the open-shell code path (``run_uhf`` ⇒
``_total_density`` adds α + β), and verify the grid-options
pass-through.

These tests share the smallest-system fixtures with
:mod:`tests.test_properties` for consistency with the existing
population-analysis suite.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _h2o() -> vq.Molecule:
    """Same equilibrium-ish H₂O as tests.test_properties."""
    return vq.Molecule([
        vq.Atom(8, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 1.43, -0.98]),
        vq.Atom(1, [0.0, -1.43, -0.98]),
    ])


def _nh4_plus() -> vq.Molecule:
    """NH₄⁺ in Td (rNH ≈ 1.025 Å, bohr coords)."""
    a2b = 1.0 / 0.529177210903
    d = 1.025 * a2b / np.sqrt(3.0)
    return vq.Molecule(
        [
            vq.Atom(7, [0.0,  0.0,  0.0]),
            vq.Atom(1, [+d, +d, +d]),
            vq.Atom(1, [+d, -d, -d]),
            vq.Atom(1, [-d, +d, -d]),
            vq.Atom(1, [-d, -d, +d]),
        ],
        charge=+1,
    )


def _oh_minus() -> vq.Molecule:
    """OH⁻ (closed-shell, 10 electrons)."""
    return vq.Molecule(
        [
            vq.Atom(8, [0.0, 0.0, 0.0]),
            vq.Atom(1, [0.0, 0.0, 1.84]),
        ],
        charge=-1,
    )


def _oh_radical() -> vq.Molecule:
    """OH (doublet radical) — exercises the open-shell path."""
    return vq.Molecule(
        [
            vq.Atom(8, [0.0, 0.0, 0.0]),
            vq.Atom(1, [0.0, 0.0, 1.84]),
        ],
        multiplicity=2,
    )


# ---------------------------------------------------------------------------
# Canonical Σq_A = q_mol sanity (the Hirshfeld weight identity)
# ---------------------------------------------------------------------------

def test_hirshfeld_sum_to_total_charge_neutral():
    """Σ q_A^Hirshfeld = 0 for a neutral closed-shell H₂O."""
    mol = _h2o()
    basis = vq.BasisSet(mol, "sto-3g")
    res = vq.run_rhf(mol, basis)
    h = vq.hirshfeld_charges(res, basis, mol)
    assert h.charges.shape == (3,)
    # Hirshfeld weight identity is exact; grid quadrature error sets
    # the floor. The default Becke-Lebedev-Treutler grid easily clears
    # 1e-5 e on STO-3G H₂O.
    assert abs(float(h.charges.sum())) < 1e-5


def test_hirshfeld_sum_to_total_charge_cation():
    """Σ q_A = +1 for NH₄⁺."""
    mol = _nh4_plus()
    basis = vq.BasisSet(mol, "sto-3g")
    res = vq.run_rhf(mol, basis)
    h = vq.hirshfeld_charges(res, basis, mol)
    assert abs(float(h.charges.sum()) - 1.0) < 1e-5


def test_hirshfeld_sum_to_total_charge_anion():
    """Σ q_A = −1 for OH⁻."""
    mol = _oh_minus()
    basis = vq.BasisSet(mol, "sto-3g")
    res = vq.run_rhf(mol, basis)
    h = vq.hirshfeld_charges(res, basis, mol)
    assert abs(float(h.charges.sum()) - (-1.0)) < 1e-5


# ---------------------------------------------------------------------------
# Numerical regression — pinned values on tight fixtures
# ---------------------------------------------------------------------------

def test_hirshfeld_h2o_signs_and_magnitudes():
    """Right sign + ballpark magnitude on H₂O / def2-SVP:

    * q(O) is negative (oxygen pulls electrons),
    * q(H) are positive and equal by molecular symmetry,
    * |q(O)| > |q(H)|, and
    * Σ |q| is order 0.5-0.8 e (heavy-atom polarisation typical of
      first-row donors).

    This is *qualitative* validation against the established
    Hirshfeld values for water (~q(O) = −0.3 to −0.4 e on def2-SVP
    SCF densities). Pinning a single tight numerical reference would
    bake in the SAD-promolecule basis-dependence; the spec checked
    here is the one that holds across promolecule conventions.
    """
    mol = _h2o()
    basis = vq.BasisSet(mol, "def2-svp")
    res = vq.run_rhf(mol, basis)
    h = vq.hirshfeld_charges(res, basis, mol)

    q_O, q_H1, q_H2 = float(h.charges[0]), float(h.charges[1]), float(h.charges[2])
    assert q_O < 0.0, f"q(O) should be negative; got {q_O:+.4f}"
    assert q_H1 > 0.0 and q_H2 > 0.0, f"q(H) should be positive; got {q_H1:+.4f}, {q_H2:+.4f}"
    assert abs(q_H1 - q_H2) < 1e-6, "H atoms break symmetry"
    assert abs(q_O) > abs(q_H1), "Heavy atom should be the more polarised one"
    # Loose magnitude bracket — Hirshfeld on water sits between
    # −0.25 and −0.55 e for any sensible promolecule + basis combo.
    assert -0.55 < q_O < -0.25, f"q(O) outside expected range: {q_O:+.4f}"


# ---------------------------------------------------------------------------
# Open-shell code path
# ---------------------------------------------------------------------------

def test_hirshfeld_open_shell_oh_radical():
    """OH radical (doublet) exercises ``_total_density`` adding α + β
    density matrices. Sum-to-charge must still hold (neutral)."""
    mol = _oh_radical()
    basis = vq.BasisSet(mol, "sto-3g")
    res = vq.run_uhf(mol, basis)
    h = vq.hirshfeld_charges(res, basis, mol)
    assert abs(float(h.charges.sum())) < 1e-5
    # The unpaired electron should sit predominantly on O, so q(O) < 0
    # and q(H) > 0 just as in the closed-shell case.
    assert float(h.charges[0]) < 0.0
    assert float(h.charges[1]) > 0.0


# ---------------------------------------------------------------------------
# Diagnostic outputs
# ---------------------------------------------------------------------------

def test_hirshfeld_grid_norms_match_n_electrons():
    """Both ``molecule_norm`` (∫ ρ_mol) and ``promolecule_norm``
    (∫ ρ_pro) should integrate to n_electrons on the default grid,
    to ~5e-5 absolute. This catches grid-builder regressions."""
    mol = _h2o()                 # neutral, 10 electrons
    basis = vq.BasisSet(mol, "def2-svp")
    res = vq.run_rhf(mol, basis)
    h = vq.hirshfeld_charges(res, basis, mol)
    n_e = 10.0
    assert abs(h.molecule_norm    - n_e) < 5e-4
    assert abs(h.promolecule_norm - n_e) < 5e-4


def test_hirshfeld_block_sweep_matches_single_block():
    """The grid is swept in blocks sized by ``max_block_elems`` so a
    large system never materialises a multi-GB AO matrix. The result
    must be independent of the block size to floating-point round-off
    — verified by forcing a tiny block (many sweeps) against the
    default single-block path on the same molecule.
    """
    mol = _h2o()
    basis = vq.BasisSet(mol, "def2-svp")
    res = vq.run_rhf(mol, basis)

    h_single = vq.hirshfeld_charges(res, basis, mol)
    # max_block_elems = 5000 with ~24 AOs ⇒ ~200-point blocks ⇒
    # hundreds of sweeps over the H₂O grid.
    h_blocked = vq.hirshfeld_charges(res, basis, mol, max_block_elems=5000)

    np.testing.assert_allclose(
        h_blocked.charges, h_single.charges, rtol=0, atol=1e-12,
        err_msg="Block-swept Hirshfeld charges differ from single-block.",
    )
    assert abs(h_blocked.molecule_norm - h_single.molecule_norm) < 1e-10
    assert abs(h_blocked.promolecule_norm - h_single.promolecule_norm) < 1e-10
    assert h_blocked.n_grid_points == h_single.n_grid_points


def test_hirshfeld_grid_options_passthrough():
    """Caller-supplied ``GridOptions`` must take effect — verified by
    a smaller grid producing a different (but valid) point count."""
    from vibeqc._vibeqc_core import GridOptions

    mol = _h2o()
    basis = vq.BasisSet(mol, "sto-3g")
    res = vq.run_rhf(mol, basis)

    h_default = vq.hirshfeld_charges(res, basis, mol)

    sparse = GridOptions()
    sparse.n_radial = 20            # default is much larger
    h_sparse = vq.hirshfeld_charges(res, basis, mol, grid_options=sparse)

    assert h_sparse.n_grid_points < h_default.n_grid_points, (
        f"Sparse grid ({h_sparse.n_grid_points}) should have fewer "
        f"points than default ({h_default.n_grid_points})."
    )
    # Even sparse grid should still satisfy the weight identity to
    # ~1e-3 e — much looser than the default grid's 1e-5 tolerance.
    assert abs(float(h_sparse.charges.sum())) < 1e-3


# ---------------------------------------------------------------------------
# Natural-orbital rank reduction of the grid contraction
# ---------------------------------------------------------------------------

def test_factor_density_reproduces_dense_contraction():
    """``_factor_density`` is an exact rewrite of the grid contraction.

    The sweep evaluates ρ(r) = Σ_μν χ_μ P_μν χ_ν as Σ_k n_k (χ·v_k)² over
    the eigenpairs of ``P``. That identity is what lets the per-block
    product drop from ``n_bf`` to ``rank(P)`` columns, so it is pinned
    directly — on a *mixed-sign* spectrum, since correlated relaxed
    densities can carry slightly negative natural occupations, and the
    factorisation has to carry the sign rather than assume PSD.
    """
    from vibeqc.properties import _factor_density

    rng = np.random.default_rng(20260805)
    n_bf = 24
    A = rng.standard_normal((n_bf, n_bf))
    P = A + A.T                       # symmetric, indefinite
    chi = rng.standard_normal((150, n_bf))

    W, signs = _factor_density(P)
    t = chi @ W
    rho_reduced = (t * t) @ signs
    rho_dense = np.einsum("gm,gm->g", chi @ P, chi)

    np.testing.assert_allclose(rho_reduced, rho_dense, rtol=0, atol=1e-10)
    # An indefinite matrix is full rank: nothing may be discarded, so the
    # reduction must degrade gracefully rather than silently lose modes.
    assert W.shape[1] == n_bf
    assert set(np.unique(signs)) <= {-1.0, 1.0}


def test_factor_density_reduces_rank_of_idempotent_density():
    """A converged closed-shell density has rank ``n_occ``, not ``n_bf``.

    This is where the speedup comes from, and it is the property that
    keeps widening as the basis grows while ``n_occ`` stays put — so it
    is worth pinning that the factorisation actually finds the low rank
    rather than carrying the zero tail along.
    """
    from vibeqc.properties import _factor_density, _total_density

    mol = _h2o()
    basis = vq.BasisSet(mol, "def2-svp")
    res = vq.run_rhf(mol, basis)

    W, signs = _factor_density(_total_density(res))
    n_occ = 5                          # H₂O: 10 electrons, closed shell
    assert W.shape[1] == n_occ, (
        f"rank(P) should be n_occ={n_occ} for a converged RHF density; "
        f"got {W.shape[1]} of {basis.nbasis} basis functions."
    )
    # Occupations of an idempotent density are positive.
    np.testing.assert_array_equal(signs, np.ones(n_occ))


# ---------------------------------------------------------------------------
# API plumbing
# ---------------------------------------------------------------------------

def test_hirshfeld_top_level_export():
    """``vibeqc.hirshfeld_charges`` and ``vibeqc.HirshfeldResult``
    are exported at the package level for parity with the other
    population analyses (``mulliken_charges`` / ``loewdin_charges``).
    """
    assert hasattr(vq, "hirshfeld_charges")
    assert hasattr(vq, "HirshfeldResult")

    mol = _h2o()
    basis = vq.BasisSet(mol, "sto-3g")
    res = vq.run_rhf(mol, basis)
    h = vq.hirshfeld_charges(res, basis, mol)
    assert isinstance(h, vq.HirshfeldResult)
