"""Phase D1b — ADIIS extrapolator tests.

ADIIS (augmented-Roothaan-Hall DIIS; Hu & Yang, J. Chem. Phys. 132,
054109 (2010)) is the EDIIS sibling: same positive-simplex constraints
and QP solver, a different (ARH) energy objective expanded about the
most recent iterate.

Pins the contract:

  1. The ADIIS coefficient solver returns a valid simplex vector
     (sum = 1, all non-negative) for every call past the first.
  2. The subspace is capped at ``max_subspace``.
  3. Open-shell extrapolation works (one coefficient set, two spin
     blocks).
  4. ADIIS-driven SCF reaches the same energy plateau as plain DIIS
     on RHF / UHF / RKS / UKS reference systems — the accelerator
     changes the path, not the fixed point.
  5. ADIIS and EDIIS converge near-identically at the HF level
     (Garza & Scuseria, J. Chem. Phys. 137, 054110 (2012)).
  6. ``"adiis"`` resolves through ``scf_accelerator_from_string``.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import (
    ADIIS,
    BasisSet,
    RHFOptions,
    RKSOptions,
    SCFAccelerator,
    UHFOptions,
    UKSOptions,
    run_rhf,
    run_rks,
    run_uhf,
    run_uks,
    scf_accelerator_from_string,
)
from .conftest import ANGSTROM_TO_BOHR, make_molecule


# ---------------------------------------------------------------------------
# Geometries
# ---------------------------------------------------------------------------

def _h2o_atoms():
    return [
        (8, [0.0, 0.0,  0.117 * ANGSTROM_TO_BOHR]),
        (1, [0.0,  0.755 * ANGSTROM_TO_BOHR, -0.471 * ANGSTROM_TO_BOHR]),
        (1, [0.0, -0.755 * ANGSTROM_TO_BOHR, -0.471 * ANGSTROM_TO_BOHR]),
    ]


def _oh_radical_atoms():
    """Hydroxyl radical (doublet) — 9 electrons total, m = 2."""
    return [
        (8, [0.0, 0.0, 0.0]),
        (1, [0.0, 0.0, 0.97 * ANGSTROM_TO_BOHR]),
    ]


def _random_symmetric(n, rng):
    A = rng.standard_normal((n, n))
    return 0.5 * (A + A.T)


@pytest.fixture
def h2o_basis():
    mol = make_molecule(_h2o_atoms())
    return mol, BasisSet(mol, "sto-3g")


# ---------------------------------------------------------------------------
# Enum surface.
# ---------------------------------------------------------------------------

def test_scf_accelerator_has_adiis_value():
    assert hasattr(SCFAccelerator, "ADIIS")


def test_adiis_resolves_from_string():
    assert scf_accelerator_from_string("adiis") == SCFAccelerator.ADIIS
    assert scf_accelerator_from_string("ADIIS") == SCFAccelerator.ADIIS


# ---------------------------------------------------------------------------
# ADIIS coefficient solver — directly via the bound class.
# ---------------------------------------------------------------------------

def test_adiis_first_call_returns_input_unchanged():
    rng = np.random.default_rng(0)
    a = ADIIS(max_subspace=8)
    F = _random_symmetric(5, rng)
    D = _random_symmetric(5, rng)
    F_out = a.extrapolate(F, D)
    np.testing.assert_array_equal(F_out, F)
    assert a.subspace_size() == 1


def test_adiis_coefficients_form_a_simplex():
    """For any sequence of (F, D), the ADIIS coefficients must lie on
    the simplex: c_i >= 0 and sum(c_i) == 1."""
    rng = np.random.default_rng(42)
    a = ADIIS(max_subspace=6)
    n_ao = 4
    for k in range(8):
        F = _random_symmetric(n_ao, rng)
        D = _random_symmetric(n_ao, rng)
        a.extrapolate(F, D)
        coeffs = np.array(a.last_coeffs())
        if k == 0:
            assert coeffs.shape == (1,)
            assert coeffs[0] == pytest.approx(1.0)
        else:
            assert (coeffs >= -1e-12).all(), f"negative coefficient: {coeffs}"
            assert coeffs.sum() == pytest.approx(1.0, abs=1e-9)


def test_adiis_concave_hull_returns_a_minimum_not_the_stationary_maximum():
    """ADIIS must minimise its concave ARH model on the boundary.

    With scalar iterates ``(D, F) = (0, 1), (1, -1)``, the ADIIS
    objective is ``2s - 2s^2`` where ``s`` is the first coefficient.
    Its stationary point at ``s = 1/2`` is the maximum (1/2), while
    both simplex vertices have the minimum value zero.  The historical
    convex-only KKT solver returned the maximum (issue #486).
    """
    a = ADIIS(max_subspace=8)
    f0, d0 = np.array([[1.0]]), np.array([[0.0]])
    f1, d1 = np.array([[-1.0]]), np.array([[1.0]])

    a.extrapolate(f0, d0)
    a.discard_last_extrapolation()
    a.extrapolate(f1, d1)
    coeffs = np.asarray(a.last_coeffs())
    s_first = float(coeffs[0])
    objective = 2.0 * s_first - 2.0 * s_first**2

    assert coeffs.shape == (2,)
    assert coeffs.sum() == pytest.approx(1.0, abs=1.0e-12)
    assert (coeffs >= 0.0).all()
    np.testing.assert_allclose(coeffs, [0.0, 1.0], rtol=0.0, atol=1.0e-12)
    assert objective <= 1.0e-12


def test_adiis_duplicate_history_is_finite_and_prefers_the_newest_vertex():
    """A rank-deficient flat face falls back to an equivalent boundary."""
    a = ADIIS(max_subspace=8)
    fock = np.array([[2.0]])
    density = np.array([[0.5]])
    a.extrapolate(fock, density)
    a.discard_last_extrapolation()
    a.extrapolate(fock, density)

    coeffs = np.asarray(a.last_coeffs())
    np.testing.assert_allclose(coeffs, [0.0, 1.0], rtol=0.0, atol=1.0e-12)


def test_adiis_differences_extensive_traces_before_multiplying():
    """The ARH model must form newest-relative products directly."""
    a = ADIIS(max_subspace=8)
    base = float(2**52)
    d0, d1 = np.array([[base]]), np.array([[base + 1.0]])
    f0, f1 = np.array([[-1.5]]), np.array([[0.5]])

    a.extrapolate(f0, d0)
    a.discard_last_extrapolation()
    a.extrapolate(f1, d1)

    coeffs = np.asarray(a.last_coeffs())
    np.testing.assert_allclose(coeffs, [0.25, 0.75], rtol=0.0, atol=1.0e-12)


def test_adiis_rejects_history_too_deep_for_exact_face_enumeration_on_use():
    """A shared large DIIS cap must fail only when ADIIS is invoked."""
    a = ADIIS(max_subspace=13)
    with pytest.raises(ValueError, match=r"max_subspace must be <= 12"):
        a.extrapolate(np.eye(1), np.eye(1))


def test_adiis_exact_face_enumeration_accepts_depth_twelve():
    a = ADIIS(max_subspace=12)
    for index in range(12):
        a.extrapolate(np.array([[float(index)]]), np.zeros((1, 1)))
        a.discard_last_extrapolation()

    coeffs = np.asarray(a.last_coeffs())
    assert a.subspace_size() == 12
    assert coeffs.shape == (12,)
    np.testing.assert_allclose(coeffs[:-1], 0.0, rtol=0.0, atol=0.0)
    assert coeffs[-1] == 1.0


def test_adiis_subspace_capped_at_max_subspace():
    rng = np.random.default_rng(7)
    a = ADIIS(max_subspace=4)
    n_ao = 3
    for _ in range(10):
        F = _random_symmetric(n_ao, rng)
        D = _random_symmetric(n_ao, rng)
        a.extrapolate(F, D)
    assert a.subspace_size() == 4


def test_adiis_requires_subspace_at_least_two():
    with pytest.raises(ValueError, match="subspace"):
        ADIIS(max_subspace=1)


def test_adiis_open_shell_extrapolate_works():
    rng = np.random.default_rng(99)
    a = ADIIS(max_subspace=4)
    n = 3
    Fa = _random_symmetric(n, rng)
    Fb = _random_symmetric(n, rng)
    Da = _random_symmetric(n, rng)
    Db = _random_symmetric(n, rng)
    out_a, out_b = a.extrapolate_uhf(Fa, Fb, Da, Db)
    np.testing.assert_array_equal(out_a, Fa)
    np.testing.assert_array_equal(out_b, Fb)

    Fa2 = _random_symmetric(n, rng)
    Fb2 = _random_symmetric(n, rng)
    Da2 = _random_symmetric(n, rng)
    Db2 = _random_symmetric(n, rng)
    out_a2, out_b2 = a.extrapolate_uhf(Fa2, Fb2, Da2, Db2)
    coeffs = np.array(a.last_coeffs())
    assert coeffs.sum() == pytest.approx(1.0, abs=1e-9)
    assert (coeffs >= -1e-12).all()
    assert out_a2.shape == Fa.shape
    assert out_b2.shape == Fb.shape


# ---------------------------------------------------------------------------
# Block-vector overload — multi-k / multi-block generalisation, mirroring
# the EDIIS block-vector contract. ARH energy model with the augmented
# trace cross-term ⟨D_i − D_n | F_j − F_n⟩ summed over all blocks.
# ---------------------------------------------------------------------------

def test_adiis_blocks_first_call_returns_input_unchanged():
    rng = np.random.default_rng(5)
    a = ADIIS(max_subspace=8)
    blocks_F = [_random_symmetric(4, rng), _random_symmetric(4, rng)]
    blocks_D = [_random_symmetric(4, rng), _random_symmetric(4, rng)]
    out = a.extrapolate_blocks(blocks_F, blocks_D)
    assert len(out) == 2
    for o, f in zip(out, blocks_F):
        np.testing.assert_array_equal(np.asarray(o), f)


def test_adiis_blocks_with_one_block_matches_closed_shell():
    """A single block must reproduce the closed-shell extrapolate."""
    rng = np.random.default_rng(11)
    n = 4
    a_blocks = ADIIS(max_subspace=6)
    a_ref = ADIIS(max_subspace=6)
    for _ in range(5):
        F = _random_symmetric(n, rng)
        D = _random_symmetric(n, rng)
        out_blocks = a_blocks.extrapolate_blocks([F], [D])
        out_ref = a_ref.extrapolate(F, D)
        np.testing.assert_allclose(
            np.asarray(out_blocks[0]), out_ref, atol=1e-12
        )


def test_adiis_blocks_count_mismatch_raises():
    """A second extrapolate_blocks call with a different block count
    must raise rather than segfault — mirrors the EDIIS contract."""
    rng = np.random.default_rng(23)
    F = _random_symmetric(3, rng)
    D = F.copy()
    a = ADIIS(max_subspace=4)
    a.extrapolate_blocks([F, F], [D, D])
    with pytest.raises(ValueError):
        a.extrapolate_blocks([F], [D])


def test_adiis_blocks_per_block_dim_mismatch_raises():
    rng = np.random.default_rng(24)
    a = ADIIS(max_subspace=4)
    a.extrapolate_blocks(
        [_random_symmetric(3, rng)], [_random_symmetric(3, rng)],
    )
    with pytest.raises(ValueError):
        a.extrapolate_blocks(
            [_random_symmetric(4, rng)], [_random_symmetric(4, rng)],
        )


def test_adiis_blocks_with_two_blocks_matches_uhf_overload():
    """Two blocks must match the open-shell UHF extrapolate."""
    rng = np.random.default_rng(17)
    n = 3
    a_blocks = ADIIS(max_subspace=4)
    a_uhf = ADIIS(max_subspace=4)
    for _ in range(4):
        Fa = _random_symmetric(n, rng)
        Fb = _random_symmetric(n, rng)
        Da = _random_symmetric(n, rng)
        Db = _random_symmetric(n, rng)
        out_blocks = a_blocks.extrapolate_blocks([Fa, Fb], [Da, Db])
        out_uhf_a, out_uhf_b = a_uhf.extrapolate_uhf(Fa, Fb, Da, Db)
        np.testing.assert_allclose(np.asarray(out_blocks[0]), out_uhf_a,
                                    atol=1e-12)
        np.testing.assert_allclose(np.asarray(out_blocks[1]), out_uhf_b,
                                    atol=1e-12)


# ---------------------------------------------------------------------------
# SCF parity — ADIIS reaches the same fixed point as DIIS.
# ---------------------------------------------------------------------------

def _rhf_with_accelerator(mol, basis, method, max_iter=160):
    opts = RHFOptions()
    opts.max_iter = max_iter
    opts.conv_tol_energy = 1e-10
    opts.conv_tol_grad = 1e-8
    opts.scf_accelerator = method
    return run_rhf(mol, basis, opts)


def test_rhf_h2o_adiis_reaches_correct_energy(h2o_basis):
    """Like pure EDIIS, pure ADIIS is not expected to drive the
    gradient as tight as DIIS — the positive-simplex constraint
    c_i >= 0 caps extrapolation near the minimum (the standard
    motivation for the EDIIS+DIIS / ADIIS+DIIS hybrids). It must
    still land on the correct energy plateau."""
    mol, basis = h2o_basis
    r_diis = _rhf_with_accelerator(mol, basis, SCFAccelerator.DIIS)
    r_adiis = _rhf_with_accelerator(mol, basis, SCFAccelerator.ADIIS)
    assert r_diis.converged
    assert r_adiis.energy == pytest.approx(r_diis.energy, abs=1e-6)


def test_rhf_h2o_adiis_matches_ediis(h2o_basis):
    """Garza & Scuseria 2012: ADIIS and EDIIS converge near-identically
    at the HF level. Same system, same settings — same energy."""
    mol, basis = h2o_basis
    r_ediis = _rhf_with_accelerator(mol, basis, SCFAccelerator.EDIIS)
    r_adiis = _rhf_with_accelerator(mol, basis, SCFAccelerator.ADIIS)
    assert r_adiis.energy == pytest.approx(r_ediis.energy, abs=1e-7)


def test_uhf_oh_radical_adiis_matches_diis():
    """Open-shell ADIIS on OH·/sto-3g converges to the same
    broken-symmetry doublet fixed point as DIIS (same energy +
    ⟨S²⟩). OH· from a near-symmetric SAD guess needs a
    symmetry-breaking oscillation; pure energy-functional
    accelerators damp it, so this case wants a generous iteration
    budget — ADIIS closes it around iter ~260, where DIIS takes
    ~150."""
    mol = make_molecule(_oh_radical_atoms(), multiplicity=2)
    basis = BasisSet(mol, "sto-3g")
    common = UHFOptions()
    common.max_iter = 400
    common.conv_tol_energy = 1e-10
    common.conv_tol_grad = 1e-8
    common.scf_accelerator = SCFAccelerator.DIIS
    r_diis = run_uhf(mol, basis, common)
    common.scf_accelerator = SCFAccelerator.ADIIS
    r_adiis = run_uhf(mol, basis, common)
    assert r_diis.converged
    assert r_adiis.converged
    assert r_adiis.energy == pytest.approx(r_diis.energy, abs=1e-8)
    assert r_adiis.s_squared == pytest.approx(r_diis.s_squared, abs=1e-6)


def test_rks_h2o_pbe_adiis_matches_diis(h2o_basis):
    mol, basis = h2o_basis
    common = RKSOptions()
    common.functional = "PBE"
    common.max_iter = 200
    common.conv_tol_energy = 1e-9
    common.conv_tol_grad = 1e-7
    common.scf_accelerator = SCFAccelerator.DIIS
    r_diis = run_rks(mol, basis, common)
    common.scf_accelerator = SCFAccelerator.ADIIS
    r_adiis = run_rks(mol, basis, common)
    assert r_diis.converged
    assert r_adiis.converged
    assert r_adiis.energy == pytest.approx(r_diis.energy, abs=1e-7)


def test_uks_oh_radical_pbe_adiis_matches_diis():
    mol = make_molecule(_oh_radical_atoms(), multiplicity=2)
    basis = BasisSet(mol, "sto-3g")
    common = UKSOptions()
    common.functional = "PBE"
    common.max_iter = 250
    common.conv_tol_energy = 1e-9
    common.conv_tol_grad = 1e-7
    common.scf_accelerator = SCFAccelerator.DIIS
    r_diis = run_uks(mol, basis, common)
    common.scf_accelerator = SCFAccelerator.ADIIS
    r_adiis = run_uks(mol, basis, common)
    assert r_diis.converged
    assert r_adiis.converged
    assert r_adiis.energy == pytest.approx(r_diis.energy, abs=1e-7)
    assert r_adiis.s_squared == pytest.approx(r_diis.s_squared, abs=1e-6)
