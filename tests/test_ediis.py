"""Phase D1a / D1c — EDIIS extrapolator + EDIIS+DIIS hybrid tests.

Pins the contract:

  1. EDIIS coefficient solver returns a valid simplex vector (sum=1,
     all non-negative) for every call past the first.

  2. EDIIS at the converged density returns weight 1 on the last
     iterate (Brillouin condition collapses the cross-term to zero).

  3. EDIIS-only and EDIIS+DIIS hybrid SCFs converge to the *same*
     energy as plain DIIS on standard reference systems (RHF / UHF /
     RKS / UKS). The accelerator changes the path through the SCF
     manifold, not the fixed point.

  4. The new RHFOptions / UHFOptions / RKSOptions / UKSOptions
     fields exist with the documented defaults.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import (
    BasisSet,
    EDIIS,
    RHFOptions,
    RKSOptions,
    SCFAccelerator,
    UHFOptions,
    UKSOptions,
    run_rhf,
    run_rks,
    run_uhf,
    run_uks,
)
from .conftest import ANGSTROM_TO_BOHR, make_molecule


# ---------------------------------------------------------------------------
# Geometries
# ---------------------------------------------------------------------------

def _h2o_atoms():
    """Water at experimental geometry, in bohr (Z, xyz)."""
    return [
        (8, [0.0, 0.0,  0.117 * ANGSTROM_TO_BOHR]),
        (1, [0.0,  0.755 * ANGSTROM_TO_BOHR, -0.471 * ANGSTROM_TO_BOHR]),
        (1, [0.0, -0.755 * ANGSTROM_TO_BOHR, -0.471 * ANGSTROM_TO_BOHR]),
    ]


def _oh_radical_atoms():
    """Hydroxyl radical (doublet) — 9 electrons total, m=2."""
    return [
        (8, [0.0, 0.0, 0.0]),
        (1, [0.0, 0.0, 0.97 * ANGSTROM_TO_BOHR]),
    ]


# ---------------------------------------------------------------------------
# Defaults — Track F (v0.8.0): SCFAccelerator.EDIIS_DIIS (PySCF/ORCA-style
# hybrid), switch threshold 1e-1.
# ---------------------------------------------------------------------------

def test_rhf_options_scf_accelerator_default_is_ediis_diis():
    opts = RHFOptions()
    assert opts.scf_accelerator == SCFAccelerator.EDIIS_DIIS
    assert opts.ediis_diis_switch_threshold == pytest.approx(1e-1)


def test_uhf_rks_uks_options_have_scf_accelerator_default():
    for cls in (UHFOptions, RKSOptions, UKSOptions):
        opts = cls()
        assert opts.scf_accelerator == SCFAccelerator.EDIIS_DIIS
        assert opts.ediis_diis_switch_threshold == pytest.approx(1e-1)


# ---------------------------------------------------------------------------
# EDIIS coefficient solver — directly via the bound class.
# ---------------------------------------------------------------------------

def _random_symmetric(n, rng):
    A = rng.standard_normal((n, n))
    return 0.5 * (A + A.T)


def test_ediis_first_call_returns_input_unchanged():
    rng = np.random.default_rng(0)
    e = EDIIS(max_subspace=8)
    F = _random_symmetric(5, rng)
    D = _random_symmetric(5, rng)
    F_out = e.extrapolate(F, D, energy=-10.0)
    np.testing.assert_array_equal(F_out, F)
    assert e.subspace_size() == 1


def test_ediis_coefficients_form_a_simplex():
    """For any sequence of (F, D, E), the EDIIS coefficients must lie
    on the simplex: c_i >= 0 and sum(c_i) == 1."""
    rng = np.random.default_rng(42)
    e = EDIIS(max_subspace=6)
    n_ao = 4
    for k in range(8):
        F = _random_symmetric(n_ao, rng)
        D = _random_symmetric(n_ao, rng)
        E = -10.0 + 0.1 * rng.standard_normal()
        e.extrapolate(F, D, E)
        coeffs = np.array(e.last_coeffs())
        if k == 0:
            # First call returns trivial coefficients.
            assert coeffs.shape == (1,)
            assert coeffs[0] == pytest.approx(1.0)
        else:
            assert (coeffs >= -1e-12).all(), f"negative coefficient: {coeffs}"
            assert coeffs.sum() == pytest.approx(1.0, abs=1e-9)


def test_ediis_concave_hull_returns_a_minimum_not_the_stationary_maximum():
    """EDIIS must minimise a concave energy on the simplex boundary.

    For the scalar density ``D``, let ``E(D) = -(D - 1/2)^2`` and
    ``F(D) = dE/dD = 1 - 2D``.  The two stored vertices ``D = 0, 1``
    both have energy -1/4, while the equality-constrained stationary
    point ``D = 1/2`` is the maximum at zero.  A convex-only KKT solver
    returned coefficients ``(1/2, 1/2)`` and therefore made the EDIIS
    result worse than every stored iterate (issue #486).
    """
    e = EDIIS(max_subspace=8)
    f0, d0 = np.array([[1.0]]), np.array([[0.0]])
    f1, d1 = np.array([[-1.0]]), np.array([[1.0]])

    e.extrapolate(f0, d0, energy=-0.25)
    e.discard_last_extrapolation()
    e.extrapolate(f1, d1, energy=-0.25)
    coeffs = np.asarray(e.last_coeffs())
    d_hull = float(coeffs[0] * d0[0, 0] + coeffs[1] * d1[0, 0])
    e_hull = -(d_hull - 0.5) ** 2

    assert coeffs.shape == (2,)
    assert coeffs.sum() == pytest.approx(1.0, abs=1.0e-12)
    assert (coeffs >= 0.0).all()
    np.testing.assert_allclose(coeffs, [0.0, 1.0], rtol=0.0, atol=1.0e-12)
    assert e_hull <= -0.25 + 1.0e-12


def test_ediis_indefinite_hull_finds_minimum_inside_a_boundary_face():
    """The global solver must search faces, not only pure vertices.

    On the triangle ``(-1, 0), (1, 0), (0, 2)``, the quadratic
    ``E(x, y) = x^2 - y^2 + 3y`` has a full-face stationary point at
    ``(0, 3/2)`` with E=9/4 and vertex energies 1, 1, 2.  Its global
    hull minimum is instead the midpoint of the bottom edge, E=0.
    Comparing a full-face KKT result with the vertices would still miss
    that minimum; enumerating the active-constraint faces finds it.
    """
    e = EDIIS(max_subspace=8)
    points = [(-1.0, 0.0), (1.0, 0.0), (0.0, 2.0)]

    for index, (x_coord, y_coord) in enumerate(points):
        density = np.diag([x_coord, y_coord])
        fock = np.diag([2.0 * x_coord, -2.0 * y_coord + 3.0])
        energy = x_coord**2 - y_coord**2 + 3.0 * y_coord
        e.extrapolate(fock, density, energy)
        if index == 1:
            # The second return is the bottom-edge minimum.  The third
            # call is testing the QP, not the separate anti-replay policy.
            e.discard_last_extrapolation()

    coeffs = np.asarray(e.last_coeffs())
    point = sum(c * np.asarray(p) for c, p in zip(coeffs, points))
    energy = point[0] ** 2 - point[1] ** 2 + 3.0 * point[1]

    np.testing.assert_allclose(
        coeffs, [0.5, 0.5, 0.0], rtol=0.0, atol=1.0e-10
    )
    assert energy == pytest.approx(0.0, abs=1.0e-12)


def test_ediis_old_face_replay_is_evicted_before_returning_the_same_fock():
    """Anti-replay must also handle a minimum inside an old face.

    The third iterate extends the indefinite triangle used above. Its
    global minimum remains the bottom-edge midpoint, exactly the Fock
    returned on call two, with zero weight on the newest iterate. Since
    call two is treated as consumed here, the guard must evict one old
    constituent and re-solve rather than replay that same step.
    """
    e = EDIIS(max_subspace=8)
    points = [(-1.0, 0.0), (1.0, 0.0), (0.0, 2.0)]
    outputs = []
    for x_coord, y_coord in points:
        density = np.diag([x_coord, y_coord])
        fock = np.diag([2.0 * x_coord, -2.0 * y_coord + 3.0])
        energy = x_coord**2 - y_coord**2 + 3.0 * y_coord
        outputs.append(np.asarray(e.extrapolate(fock, density, energy)))

    assert e.subspace_size() == 2
    assert np.asarray(e.last_coeffs()).shape == (2,)
    assert not np.array_equal(outputs[2], outputs[1])


def test_ediis_duplicate_history_is_finite_and_prefers_the_newest_vertex():
    """A rank-deficient flat face falls back to an equivalent boundary."""
    e = EDIIS(max_subspace=8)
    fock = np.array([[2.0]])
    density = np.array([[0.5]])
    e.extrapolate(fock, density, energy=-1.0)
    e.discard_last_extrapolation()
    e.extrapolate(fock, density, energy=-1.0)

    coeffs = np.asarray(e.last_coeffs())
    np.testing.assert_allclose(coeffs, [0.0, 1.0], rtol=0.0, atol=1.0e-12)


def test_ediis_extensive_objective_still_compares_the_old_vertex():
    """Large Fock scales must not make the old vertex look singular.

    Both scalar iterates have the same extensive quadratic contribution,
    so their energy difference uniquely selects the older one. A raw KKT
    rank test can discard that vertex when ``M_ii`` dwarfs the unit simplex
    border, violating the guarantee that EDIIS is no worse than its history.
    """
    e = EDIIS(max_subspace=8)
    density = np.array([[1.0]])
    fock = np.array([[1.0e20]])
    e.extrapolate(fock, density, energy=-2.0e20)
    e.discard_last_extrapolation()
    e.extrapolate(fock, density, energy=-1.0e20)

    coeffs = np.asarray(e.last_coeffs())
    np.testing.assert_allclose(coeffs, [1.0, 0.0], rtol=0.0, atol=1.0e-12)


def test_ediis_absolute_energy_offset_does_not_blur_distinct_vertices():
    """A common extensive energy offset is irrelevant on the simplex."""
    e = EDIIS(max_subspace=8)
    density = np.array([[0.0]])
    e.extrapolate(np.array([[1.0]]), density, energy=1.0e15)
    e.discard_last_extrapolation()
    e.extrapolate(np.array([[2.0]]), density, energy=1.0e15 + 1.0)

    coeffs = np.asarray(e.last_coeffs())
    np.testing.assert_allclose(coeffs, [1.0, 0.0], rtol=0.0, atol=1.0e-12)


def test_ediis_extensive_density_gauge_preserves_an_interior_minimum():
    """Affine ``Tr(FD)`` terms must not hide the physical curvature.

    ``E(D) = (D - (2**51 + 1/2))**2`` has its hull minimum halfway
    between the exactly representable densities ``2**51`` and
    ``2**51 + 1``. The raw EDIIS matrix is order 2**51 while its tangent
    curvature is order one; centering the simplex quadratic gauge must
    recover the same coefficients as an origin-centred convex fixture.
    """
    e = EDIIS(max_subspace=8)
    base = float(2**51)
    d0, d1 = np.array([[base]]), np.array([[base + 1.0]])
    f0, f1 = np.array([[-1.0]]), np.array([[1.0]])

    e.extrapolate(f0, d0, energy=0.25)
    e.discard_last_extrapolation()
    e.extrapolate(f1, d1, energy=0.25)

    coeffs = np.asarray(e.last_coeffs())
    np.testing.assert_allclose(coeffs, [0.5, 0.5], rtol=0.0, atol=1.0e-12)


def test_ediis_differences_extensive_traces_before_multiplying():
    """Relative assembly must preserve curvature lost by absolute traces."""
    e = EDIIS(max_subspace=8)
    base = float(2**52)
    d0, d1 = np.array([[base]]), np.array([[base + 1.0]])
    f0, f1 = np.array([[-0.5]]), np.array([[1.5]])

    e.extrapolate(f0, d0, energy=0.0625)
    e.discard_last_extrapolation()
    e.extrapolate(f1, d1, energy=0.5625)

    coeffs = np.asarray(e.last_coeffs())
    np.testing.assert_allclose(coeffs, [0.75, 0.25], rtol=0.0, atol=1.0e-12)


def test_ediis_rejects_history_too_deep_for_exact_face_enumeration_on_use():
    """A shared large DIIS cap must fail only when EDIIS is invoked."""
    e = EDIIS(max_subspace=13)
    with pytest.raises(ValueError, match=r"max_subspace must be <= 12"):
        e.extrapolate(np.eye(1), np.eye(1), energy=0.0)


def test_ediis_exact_face_enumeration_accepts_depth_twelve():
    e = EDIIS(max_subspace=12)
    for index in range(12):
        e.extrapolate(np.array([[float(index)]]), np.zeros((1, 1)), energy=0.0)
        e.discard_last_extrapolation()

    coeffs = np.asarray(e.last_coeffs())
    assert e.subspace_size() == 12
    assert coeffs.shape == (12,)
    np.testing.assert_allclose(coeffs[:-1], 0.0, rtol=0.0, atol=0.0)
    assert coeffs[-1] == 1.0


def test_ediis_subspace_capped_at_max_subspace():
    rng = np.random.default_rng(7)
    e = EDIIS(max_subspace=4)
    n_ao = 3
    for _ in range(10):
        F = _random_symmetric(n_ao, rng)
        D = _random_symmetric(n_ao, rng)
        e.extrapolate(F, D, energy=-1.0)
    assert e.subspace_size() == 4


def test_ediis_at_a_converged_iterate_picks_that_iterate():
    """At convergence, the cross-term Tr[(D_i - D_j)(F_i - F_j)]
    vanishes pairwise, so the energy functional reduces to
    Σ c_i E_i. With the latest iterate having the lowest energy,
    EDIIS should put all weight on it (vertex of the simplex)."""
    rng = np.random.default_rng(123)
    e = EDIIS(max_subspace=6)
    n_ao = 4
    F0 = _random_symmetric(n_ao, rng)
    D0 = _random_symmetric(n_ao, rng)
    # Push a few "noisy" iterates with higher energies.
    for k in range(3):
        e.extrapolate(F0 + 0.01 * _random_symmetric(n_ao, rng),
                      D0 + 0.01 * _random_symmetric(n_ao, rng),
                      energy=-10.0 + 0.5 * (3 - k))
    # Now push the "converged" iterate.
    e.extrapolate(F0, D0, energy=-12.0)
    coeffs = np.array(e.last_coeffs())
    # The converged iterate (lowest energy, last) should dominate.
    assert coeffs[-1] > 0.5
    # Coefficients still on the simplex.
    assert coeffs.sum() == pytest.approx(1.0, abs=1e-9)


def test_ediis_pinned_vertex_does_not_replay_previous_return():
    """Anti-replay guard: a QP solution pinned on an old vertex must not
    hand the SCF the identical Fock twice in a row.

    Deterministic pinning setup (scalars): iterate 1 = (F=0, D=1,
    E=-1), iterate 2 = (F=1, D=1, E=0). The EDIIS objective along the
    segment c = (1-t, t) is  -1 + t  (the D-difference cross-term
    vanishes because D_1 == D_2), so the minimiser sits at the t = 0
    vertex — iterate 1 — and the raw extrapolation would return F_1
    again, exactly as the previous call did. The deterministic SCF
    would then diagonalise the same Fock, rebuild the same iterate 2,
    and freeze until the FIFO evicts iterate 1 (the c-diamond
    KRHF-GDF kmesh (2,2,2) 8-cycle stall, 2026-07-09). The guard must
    drop the exploited vertex and return the newest Fock instead.
    """
    e = EDIIS(max_subspace=8)
    F1 = np.array([[0.0]])
    F2 = np.array([[1.0]])
    D = np.array([[1.0]])

    out1 = e.extrapolate(F1, D, energy=-1.0)
    np.testing.assert_array_equal(out1, F1)

    out2 = e.extrapolate(F2, D, energy=0.0)
    # Pre-guard behaviour returned F1 (the pinned vertex) — a bit-exact
    # replay of out1. The guard must yield the newest Fock.
    np.testing.assert_array_equal(out2, F2)
    coeffs = np.array(e.last_coeffs())
    assert coeffs.sum() == pytest.approx(1.0, abs=1e-9)
    assert (coeffs >= -1e-12).all()


def test_ediis_pinned_vertex_scf_loop_makes_progress():
    """SCF-shaped replay loop: feeding the (F, D, E) pair the frozen
    diagonalisation would regenerate must not freeze the extrapolation
    for ``max_subspace`` cycles. Every return past the first must carry
    weight on fresh information (here: differ from the pinned F_1)."""
    e = EDIIS(max_subspace=8)
    F1 = np.array([[0.0]])
    D = np.array([[1.0]])
    e.extrapolate(F1, D, energy=-1.0)
    # The stalled SCF replays the same iterate-2 data every cycle.
    F2 = np.array([[1.0]])
    for _ in range(4):
        out = e.extrapolate(F2, D, energy=0.0)
        assert not np.array_equal(out, F1), (
            "EDIIS returned the exploited vertex again — replay deadlock"
        )


def test_adiis_pinned_vertex_does_not_replay_previous_return():
    """ADIIS sibling of the EDIIS anti-replay guard test. The ARH
    objective for the same scalar setup is  2 c_1 (D_1 - D_2) F_2 + ...
    = 0 along the whole segment (D_1 == D_2), so any simplex point is a
    minimiser; if the solver lands on the old vertex twice in a row the
    guard must reroute to the newest Fock rather than replay."""
    from vibeqc import ADIIS

    a = ADIIS(max_subspace=8)
    F1 = np.array([[0.0]])
    F2 = np.array([[1.0]])
    D1 = np.array([[1.0]])
    D2 = np.array([[2.0]])

    out1 = a.extrapolate(F1, D1)
    np.testing.assert_array_equal(out1, F1)
    prev = out1
    for _ in range(4):
        out = a.extrapolate(F2, D2)
        coeffs = np.array(a.last_coeffs())
        if coeffs[-1] <= 1e-10:
            # Newest iterate unused: the guard must have prevented a
            # bit-exact replay of the previous return.
            assert not np.array_equal(out, prev)
        prev = out


def test_ediis_extrapolated_density_follows_the_guard_erasure():
    """``last_extrapolated_density`` is Σ_i c_i D_i over the history that
    SURVIVED the anti-replay guard, not over the sequence of pushes.

    GitLab #487: the ROHF/ROKS Roothaan loop needs D_tilde = Σ_i c_i D_i
    (the density whose Fock EDIIS returned) to build its projectors. It
    used to mirror the density history in Python and was never told when
    ``finish_extrapolation`` erased an interior entry, so a single guard
    fire left the mirror one entry too long and the loop silently fell
    back to the *current* density. The accessor is combined on the C++
    side over the live history, so it cannot desync.

    Scalar construction (F, D, E), all 1x1:
      1: (0.0, 1.0, -1.0)   2: (0.5, 1.2, -1.2)   3: (1.2, 1.4, -1.0)
    Call 2 puts all weight on the lower iterate 2 (guard idle, return
    F_2 = 0.5). Call 3: the three-vertex minimum is the pure vertex 2
    again (every mixture loses 0.2 s in the linear term and gains at most
    0.07 s from the cross terms), which would replay the previous return
    with zero weight on the newest iterate -- so the guard erases vertex
    2 and re-solves over {1, 3}. That two-vertex objective is
    -1 - 0.24 t (1 - t): convex, minimum at t = 1/2. Hence the surviving
    coefficients are (1/2, 1/2), the returned Fock is 0.6 and the
    density it belongs to is (1.0 + 1.4) / 2 = 1.2 -- which is neither
    the current density (1.4) nor anything a three-entry mirror could
    index with a two-entry coefficient vector.
    """
    e = EDIIS(max_subspace=8)
    F = [np.array([[0.0]]), np.array([[0.5]]), np.array([[1.2]])]
    D = [np.array([[1.0]]), np.array([[1.2]]), np.array([[1.4]])]
    E = [-1.0, -1.2, -1.0]

    assert e.last_extrapolated_density() == []
    e.extrapolate(F[0], D[0], energy=E[0])
    np.testing.assert_allclose(e.last_extrapolated_density()[0], D[0])

    out2 = e.extrapolate(F[1], D[1], energy=E[1])
    np.testing.assert_allclose(out2, F[1])
    np.testing.assert_allclose(e.last_coeffs(), [0.0, 1.0], atol=1e-10)
    assert e.replay_guard_erasures() == 0
    np.testing.assert_allclose(e.last_extrapolated_density()[0], D[1])

    out3 = e.extrapolate(F[2], D[2], energy=E[2])
    assert e.replay_guard_erasures() == 1
    assert e.subspace_size() == 2
    coeffs = np.array(e.last_coeffs())
    np.testing.assert_allclose(coeffs, [0.5, 0.5], atol=1e-8)
    np.testing.assert_allclose(out3, 0.5 * F[0] + 0.5 * F[2], atol=1e-8)
    (d_tilde,) = e.last_extrapolated_density()
    np.testing.assert_allclose(d_tilde, 0.5 * D[0] + 0.5 * D[2], atol=1e-8)
    # The two things a desynced driver could have used instead:
    assert not np.allclose(d_tilde, D[2])  # the current iterate
    assert len(coeffs) != len(D)  # a push-ordered mirror cannot be indexed


def test_adiis_extrapolated_density_follows_the_guard_erasure():
    """ADIIS sibling of the test above: after the guard evicts the pinned
    old vertex, ``last_extrapolated_density`` is the survivor's density,
    and the erasure counter says the guard fired."""
    from vibeqc import ADIIS

    a = ADIIS(max_subspace=8)
    F1, F2 = np.array([[0.0]]), np.array([[1.0]])
    D1, D2 = np.array([[1.0]]), np.array([[2.0]])
    assert a.last_extrapolated_density() == []
    a.extrapolate(F1, D1)
    np.testing.assert_allclose(a.last_extrapolated_density()[0], D1)
    fired = False
    for _ in range(4):
        a.extrapolate(F2, D2)
        coeffs = np.array(a.last_coeffs())
        (d_tilde,) = a.last_extrapolated_density()
        if a.replay_guard_erasures() > 0:
            fired = True
            # Only the newest iterate can survive a two-entry erasure.
            assert a.subspace_size() == 1
            np.testing.assert_allclose(coeffs, [1.0])
            np.testing.assert_allclose(d_tilde, D2)
            break
        # No erasure yet: the accessor is the plain hull combination.
        expected = coeffs[0] * D1 + coeffs[1:].sum() * D2
        np.testing.assert_allclose(d_tilde, expected, atol=1e-12)
    assert fired, "the ADIIS anti-replay guard never fired on the pinned setup"


def test_ediis_uhf_extrapolated_density_is_the_hull_combination_per_spin():
    """Open-shell: two blocks, the same coefficients on both spins, each
    block Σ_i c_i D_{σ,i} -- the pair ``run_roothaan_scf`` projects with."""
    rng = np.random.default_rng(487)
    n = 4

    def spd():
        m = rng.standard_normal((n, n))
        return m @ m.T + n * np.eye(n)

    e = EDIIS(max_subspace=8)
    fa_hist, fb_hist, da_hist, db_hist = [], [], [], []
    for it in range(3):
        fa, fb, da, db = spd(), spd(), spd(), spd()
        fa_hist.append(fa)
        fb_hist.append(fb)
        da_hist.append(da)
        db_hist.append(db)
        out_a, out_b = e.extrapolate_uhf(fa, fb, da, db, energy=-1.0 - 0.1 * it)
        e.discard_last_extrapolation()  # keep the guard out of this test
        c = np.array(e.last_coeffs())
        assert len(c) == it + 1
        blocks = e.last_extrapolated_density()
        assert len(blocks) == 2
        np.testing.assert_allclose(
            blocks[0], sum(ci * d for ci, d in zip(c, da_hist)), atol=1e-12
        )
        np.testing.assert_allclose(
            blocks[1], sum(ci * d for ci, d in zip(c, db_hist)), atol=1e-12
        )
        # ... and the returned Fock pair is the same combination of the
        # stored Focks, so at the HF level it is F(D_tilde).
        np.testing.assert_allclose(
            out_a, sum(ci * f for ci, f in zip(c, fa_hist)), atol=1e-12
        )
        np.testing.assert_allclose(
            out_b, sum(ci * f for ci, f in zip(c, fb_hist)), atol=1e-12
        )


def test_ediis_clear_resets_the_erasure_counter_and_density():
    e = EDIIS(max_subspace=8)
    e.extrapolate(np.array([[0.0]]), np.array([[1.0]]), energy=-1.0)
    e.extrapolate(np.array([[1.0]]), np.array([[1.0]]), energy=0.0)
    assert e.replay_guard_erasures() == 1
    e.clear()
    assert e.replay_guard_erasures() == 0
    assert e.last_extrapolated_density() == []


def _unit_diag(i, n=3):
    """i-th diagonal unit matrix — F_i = D_i = these makes the EDIIS QP
    Hessian exactly the identity (M_ij = delta_ij), so the coefficient
    solve is strictly convex and its minimiser is analytic."""
    m = np.zeros((n, n))
    m[i, i] = 1.0
    return m


def test_ediis_discarded_extrapolation_is_not_a_replay():
    """The anti-replay guard must key on the previously CONSUMED return,
    not merely the previously produced one.

    The LM02 pattern in miniature. In the EDIIS_DIIS hybrid both branches
    run every cycle and only one result is kept, so an EDIIS Fock
    produced on a DIIS-branch cycle never reaches the SCF; reproducing it
    later is a FIRST use, not a replay. Before this was distinguished the
    guard erased live history mid-run and forked n-octane RKS/PBE/def2-SVP
    onto a basin ~10 mHa above the v0.15.21 solution (shipped v0.15.31 to
    v0.15.45) — see
    ``.agents/agentic-loop-state/blockers/
    ediis-antireplay-guard-molecular-basin-regression.md``.

    With ``M = I`` and energies (3, 0, 3) the QP is analytic. Vertex 2 is
    the strict minimiser at cycle 2, where it is the NEWEST iterate
    (c_newest = 1, guard inert), and again at cycle 3, where it is an OLD
    vertex (c_newest = 0) whose combination reproduces the cycle-2 return
    bit-exactly. Cycle 2 takes the DIIS branch, so that Fock is discarded.

    Pre-fix the guard fired at cycle 3, erased iterate 2 and returned
    (F1 + F3)/2 at depth 2. It must now return F2 with the history whole.
    """
    e = EDIIS(max_subspace=8)
    energies = [3.0, 0.0, 3.0]

    e.extrapolate(_unit_diag(0), _unit_diag(0), energy=energies[0])

    out2 = e.extrapolate(_unit_diag(1), _unit_diag(1), energy=energies[1])
    np.testing.assert_allclose(out2, _unit_diag(1), atol=1e-12)
    # The hybrid's switch metric selected DIIS this cycle: out2 is
    # dropped and never diagonalised.
    e.discard_last_extrapolation()

    out3 = e.extrapolate(_unit_diag(2), _unit_diag(2), energy=energies[2])
    np.testing.assert_allclose(out3, _unit_diag(1), atol=1e-12)
    assert e.subspace_size() == 3, (
        "guard erased history on a first use of a discarded extrapolation"
    )


def test_ediis_guard_still_fires_when_extrapolation_was_consumed():
    """The original anti-replay protection (1a7e807b) must stay armed.

    Same construction as the discard test, but cycle 2's return IS handed
    to the SCF (no discard call). Cycle 3 then reproduces a Fock the SCF
    has already diagonalised, which is a genuine replay: the guard must
    still drop the exploited vertex and re-solve.
    """
    e = EDIIS(max_subspace=8)
    energies = [3.0, 0.0, 3.0]

    e.extrapolate(_unit_diag(0), _unit_diag(0), energy=energies[0])
    out2 = e.extrapolate(_unit_diag(1), _unit_diag(1), energy=energies[1])
    np.testing.assert_allclose(out2, _unit_diag(1), atol=1e-12)
    # No discard: the hybrid took the EDIIS branch, so out2 was consumed.

    out3 = e.extrapolate(_unit_diag(2), _unit_diag(2), energy=energies[2])
    assert not np.allclose(out3, out2, atol=1e-12), (
        "guard failed to break a genuine replay of a consumed return"
    )
    assert e.subspace_size() == 2, (
        "guard should have evicted the fully-exploited pinned vertex"
    )


def test_ediis_discard_covers_the_pre_diis_start_warmup():
    """``iter < diis_start_iter`` is the second discard path.

    The molecular and periodic drivers compute the extrapolation every
    cycle but apply it only when ``diis_active``; below
    ``diis_start_iter`` the return is dropped exactly as the hybrid's
    losing branch drops it. Retracting it keeps the guard from reading a
    later first use as a replay.
    """
    e = EDIIS(max_subspace=8)
    energies = [3.0, 0.0, 3.0]

    e.extrapolate(_unit_diag(0), _unit_diag(0), energy=energies[0])
    # Warm-up cycle: extrapolated, then dropped because DIIS is not yet
    # active. Identical bookkeeping to the hybrid's DIIS branch.
    e.extrapolate(_unit_diag(1), _unit_diag(1), energy=energies[1])
    e.discard_last_extrapolation()

    out3 = e.extrapolate(_unit_diag(2), _unit_diag(2), energy=energies[2])
    np.testing.assert_allclose(out3, _unit_diag(1), atol=1e-12)
    assert e.subspace_size() == 3


def test_adiis_discarded_extrapolation_is_not_a_replay():
    """ADIIS sibling: ADIIS_DIIS has the same produced-vs-consumed
    distinction, so ``discard_last_extrapolation`` must retract a dropped
    return there too."""
    from vibeqc import ADIIS

    a = ADIIS(max_subspace=8)
    F1 = np.array([[0.0]])
    F2 = np.array([[1.0]])
    D1 = np.array([[1.0]])
    D2 = np.array([[2.0]])

    a.extrapolate(F1, D1)
    prev = a.extrapolate(F2, D2)
    # DIIS branch won: retract, so a later identical hull point is a
    # first use rather than a replay.
    a.discard_last_extrapolation()
    out = a.extrapolate(F2, D2)
    coeffs = np.array(a.last_coeffs())
    if coeffs[-1] <= 1e-10 and np.array_equal(out, prev):
        # Reproducing a DISCARDED return is legitimate; the guard must
        # not have torn down the history to avoid it.
        assert a.subspace_size() >= 2


def test_ediis_open_shell_extrapolate_works():
    rng = np.random.default_rng(99)
    e = EDIIS(max_subspace=4)
    n = 3
    Fa = _random_symmetric(n, rng)
    Fb = _random_symmetric(n, rng)
    Da = _random_symmetric(n, rng)
    Db = _random_symmetric(n, rng)
    out_a, out_b = e.extrapolate_uhf(Fa, Fb, Da, Db, energy=-5.0)
    np.testing.assert_array_equal(out_a, Fa)
    np.testing.assert_array_equal(out_b, Fb)

    # Second call: must produce a valid simplex of coefficients and
    # return convex combinations of historical Fa / Fb.
    Fa2 = _random_symmetric(n, rng)
    Fb2 = _random_symmetric(n, rng)
    Da2 = _random_symmetric(n, rng)
    Db2 = _random_symmetric(n, rng)
    out_a2, out_b2 = e.extrapolate_uhf(Fa2, Fb2, Da2, Db2, energy=-6.0)
    coeffs = np.array(e.last_coeffs())
    assert coeffs.sum() == pytest.approx(1.0, abs=1e-9)
    assert (coeffs >= -1e-12).all()
    assert out_a2.shape == Fa.shape
    assert out_b2.shape == Fb.shape


# ---------------------------------------------------------------------------
# Block-vector overload — multi-k / multi-block generalisation. With one
# block, must reduce to the closed-shell extrapolate; with two blocks,
# must reduce to extrapolate_uhf. The QP cross-term ⟨F_i | D_j⟩ sums
# over all blocks — this is what makes the same kernel cover (a) closed-
# shell, (b) UHF α + β, and (c) multi-k periodic SCF (real-space cells).
# ---------------------------------------------------------------------------

def test_ediis_blocks_first_call_returns_input_unchanged():
    rng = np.random.default_rng(5)
    e = EDIIS(max_subspace=8)
    blocks_F = [_random_symmetric(4, rng), _random_symmetric(4, rng)]
    blocks_D = [_random_symmetric(4, rng), _random_symmetric(4, rng)]
    out = e.extrapolate_blocks(blocks_F, blocks_D, energy=-7.0)
    assert len(out) == 2
    for o, f in zip(out, blocks_F):
        np.testing.assert_array_equal(np.asarray(o), f)
    assert e.subspace_size() == 1


def test_ediis_blocks_with_one_block_matches_closed_shell():
    """A single block must reproduce the closed-shell extrapolate
    exactly — same QP, same coefficients, same output."""
    rng = np.random.default_rng(11)
    n = 4
    e_blocks = EDIIS(max_subspace=6)
    e_ref = EDIIS(max_subspace=6)
    for k in range(5):
        F = _random_symmetric(n, rng)
        D = _random_symmetric(n, rng)
        E = -10.0 + 0.1 * rng.standard_normal()
        out_blocks = e_blocks.extrapolate_blocks([F], [D], energy=E)
        out_ref = e_ref.extrapolate(F, D, energy=E)
        np.testing.assert_allclose(
            np.asarray(out_blocks[0]), out_ref, atol=1e-12
        )
        np.testing.assert_allclose(
            np.array(e_blocks.last_coeffs()),
            np.array(e_ref.last_coeffs()),
            atol=1e-12,
        )


def test_ediis_blocks_with_two_blocks_matches_uhf_overload():
    """Two blocks must match the open-shell UHF extrapolate — the
    block-vector ⟨F_i | D_j⟩ sum over two blocks is exactly the
    UHF α + β trace cross-term."""
    rng = np.random.default_rng(17)
    n = 3
    e_blocks = EDIIS(max_subspace=4)
    e_uhf = EDIIS(max_subspace=4)
    for k in range(4):
        Fa = _random_symmetric(n, rng)
        Fb = _random_symmetric(n, rng)
        Da = _random_symmetric(n, rng)
        Db = _random_symmetric(n, rng)
        E = -5.0 + 0.2 * rng.standard_normal()
        out_blocks = e_blocks.extrapolate_blocks(
            [Fa, Fb], [Da, Db], energy=E
        )
        out_uhf_a, out_uhf_b = e_uhf.extrapolate_uhf(
            Fa, Fb, Da, Db, energy=E
        )
        np.testing.assert_allclose(np.asarray(out_blocks[0]), out_uhf_a,
                                    atol=1e-12)
        np.testing.assert_allclose(np.asarray(out_blocks[1]), out_uhf_b,
                                    atol=1e-12)


def test_ediis_blocks_count_mismatch_raises():
    """A second extrapolate_blocks call with a different block count
    must raise rather than segfault — solve_qp would otherwise index
    a mis-sized history. See cpp/include/vibeqc/ediis.hpp:110 contract."""
    rng = np.random.default_rng(23)
    F = _random_symmetric(3, rng)
    D = F.copy()
    e = EDIIS(max_subspace=4)
    e.extrapolate_blocks([F, F], [D, D], energy=-1.0)
    with pytest.raises(ValueError):
        e.extrapolate_blocks([F], [D], energy=-2.0)


def test_ediis_blocks_per_block_dim_mismatch_raises():
    """A second call with the same block count but a different
    per-block matrix dimension must raise rather than corrupt state."""
    rng = np.random.default_rng(24)
    e = EDIIS(max_subspace=4)
    e.extrapolate_blocks(
        [_random_symmetric(3, rng)], [_random_symmetric(3, rng)],
        energy=-1.0,
    )
    with pytest.raises(ValueError):
        e.extrapolate_blocks(
            [_random_symmetric(4, rng)], [_random_symmetric(4, rng)],
            energy=-2.0,
        )


# ---------------------------------------------------------------------------
# SCF parity — EDIIS / EDIIS+DIIS converge to the same fixed point as DIIS.
# ---------------------------------------------------------------------------

@pytest.fixture
def h2o_basis():
    mol = make_molecule(_h2o_atoms())
    return mol, BasisSet(mol, "sto-3g")


def _rhf_with_accelerator(mol, basis, method):
    opts = RHFOptions()
    opts.max_iter = 80
    opts.conv_tol_energy = 1e-10
    opts.conv_tol_grad = 1e-8
    opts.scf_accelerator = method
    return run_rhf(mol, basis, opts)


def test_rhf_h2o_diis_and_ediis_diis_hybrid_match(h2o_basis):
    """The production hybrid converges to the same fixed point as
    plain DIIS — the accelerator changes the path through the SCF
    manifold, not the converged energy."""
    mol, basis = h2o_basis
    r_diis = _rhf_with_accelerator(mol, basis, SCFAccelerator.DIIS)
    r_hyb  = _rhf_with_accelerator(mol, basis, SCFAccelerator.EDIIS_DIIS)
    assert r_diis.converged
    assert r_hyb.converged
    assert r_hyb.energy == pytest.approx(r_diis.energy, abs=1e-9)


def test_rhf_plain_diis_can_use_a_history_deeper_than_twelve(h2o_basis):
    """Inactive exact-face accelerators must not reject a DIIS-only run."""
    mol, basis = h2o_basis
    opts = RHFOptions()
    opts.scf_accelerator = SCFAccelerator.DIIS
    opts.diis_subspace_size = 13
    opts.max_iter = 80
    opts.conv_tol_energy = 1e-10
    opts.conv_tol_grad = 1e-8

    result = run_rhf(mol, basis, opts)
    assert result.converged


def test_rhf_h2o_pure_ediis_reaches_correct_energy(h2o_basis):
    """Pure EDIIS is *not* expected to converge to the same tight
    gradient threshold as DIIS — the simplex constraint c_i ≥ 0
    prevents it from extrapolating below the convex hull near the
    minimum (this is the standard motivation for the EDIIS+DIIS
    hybrid). What we *can* assert: it lands on the correct energy
    plateau within the EDIIS-typical accuracy of ~1e-6 Ha."""
    mol, basis = h2o_basis
    r_diis  = _rhf_with_accelerator(mol, basis, SCFAccelerator.DIIS)
    r_ediis = _rhf_with_accelerator(mol, basis, SCFAccelerator.EDIIS)
    assert r_diis.converged
    assert r_ediis.energy == pytest.approx(r_diis.energy, abs=1e-6)


def test_uhf_oh_radical_ediis_diis_matches_diis():
    """OH/sto-3g UHF starts from a near-closed-shell SAD guess and
    needs an oscillation to break symmetry into the doublet — both
    DIIS and EDIIS+DIIS reach the same fixed point, just along
    different paths."""
    mol = make_molecule(_oh_radical_atoms(), multiplicity=2)
    basis = BasisSet(mol, "sto-3g")
    common = UHFOptions()
    common.max_iter = 200
    common.conv_tol_energy = 1e-10
    common.conv_tol_grad = 1e-8
    common.scf_accelerator = SCFAccelerator.DIIS
    r_diis = run_uhf(mol, basis, common)
    common.scf_accelerator = SCFAccelerator.EDIIS_DIIS
    r_hyb = run_uhf(mol, basis, common)
    assert r_diis.converged and r_hyb.converged
    assert r_hyb.energy == pytest.approx(r_diis.energy, abs=1e-9)
    assert r_hyb.s_squared == pytest.approx(r_diis.s_squared, abs=1e-6)


def test_rks_h2o_pbe_ediis_diis_matches_diis(h2o_basis):
    mol, basis = h2o_basis
    common = RKSOptions()
    common.functional = "PBE"
    common.max_iter = 80
    common.conv_tol_energy = 1e-9
    common.conv_tol_grad = 1e-7
    common.scf_accelerator = SCFAccelerator.DIIS
    r_diis = run_rks(mol, basis, common)
    common.scf_accelerator = SCFAccelerator.EDIIS_DIIS
    r_hyb = run_rks(mol, basis, common)
    assert r_diis.converged and r_hyb.converged
    assert r_hyb.energy == pytest.approx(r_diis.energy, abs=1e-8)


def test_uks_oh_radical_pbe_ediis_diis_matches_diis():
    mol = make_molecule(_oh_radical_atoms(), multiplicity=2)
    basis = BasisSet(mol, "sto-3g")
    common = UKSOptions()
    common.functional = "PBE"
    common.max_iter = 200
    common.conv_tol_energy = 1e-9
    common.conv_tol_grad = 1e-7
    common.scf_accelerator = SCFAccelerator.DIIS
    r_diis = run_uks(mol, basis, common)
    common.scf_accelerator = SCFAccelerator.EDIIS_DIIS
    r_hyb = run_uks(mol, basis, common)
    assert r_diis.converged and r_hyb.converged
    assert r_hyb.energy == pytest.approx(r_diis.energy, abs=1e-8)


# ---------------------------------------------------------------------------
# ADIIS_DIIS hybrid — the ADIIS analogue of EDIIS_DIIS. Same contract:
# converges to the same fixed point as plain DIIS across RHF/UHF/RKS/UKS.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["adiis_diis", "ADIIS_DIIS", "adiis+diis"])
def test_adiis_diis_keyword_resolves(name):
    assert vq.scf_accelerator_from_string(name) == SCFAccelerator.ADIIS_DIIS


def test_rhf_h2o_adiis_diis_matches_diis(h2o_basis):
    mol, basis = h2o_basis
    r_diis = _rhf_with_accelerator(mol, basis, SCFAccelerator.DIIS)
    r_hyb = _rhf_with_accelerator(mol, basis, SCFAccelerator.ADIIS_DIIS)
    assert r_diis.converged and r_hyb.converged
    assert r_hyb.energy == pytest.approx(r_diis.energy, abs=1e-9)


def test_uhf_oh_radical_adiis_diis_matches_diis():
    mol = make_molecule(_oh_radical_atoms(), multiplicity=2)
    basis = BasisSet(mol, "sto-3g")
    common = UHFOptions()
    common.max_iter = 200
    common.conv_tol_energy = 1e-10
    common.conv_tol_grad = 1e-8
    common.scf_accelerator = SCFAccelerator.DIIS
    r_diis = run_uhf(mol, basis, common)
    common.scf_accelerator = SCFAccelerator.ADIIS_DIIS
    r_hyb = run_uhf(mol, basis, common)
    assert r_diis.converged and r_hyb.converged
    assert r_hyb.energy == pytest.approx(r_diis.energy, abs=1e-9)
    assert r_hyb.s_squared == pytest.approx(r_diis.s_squared, abs=1e-6)


def test_rks_h2o_pbe_adiis_diis_matches_diis(h2o_basis):
    mol, basis = h2o_basis
    common = RKSOptions()
    common.functional = "PBE"
    common.max_iter = 80
    common.conv_tol_energy = 1e-9
    common.conv_tol_grad = 1e-7
    common.scf_accelerator = SCFAccelerator.DIIS
    r_diis = run_rks(mol, basis, common)
    common.scf_accelerator = SCFAccelerator.ADIIS_DIIS
    r_hyb = run_rks(mol, basis, common)
    assert r_diis.converged and r_hyb.converged
    assert r_hyb.energy == pytest.approx(r_diis.energy, abs=1e-8)


def test_uks_oh_radical_pbe_adiis_diis_matches_diis():
    mol = make_molecule(_oh_radical_atoms(), multiplicity=2)
    basis = BasisSet(mol, "sto-3g")
    common = UKSOptions()
    common.functional = "PBE"
    common.max_iter = 200
    common.conv_tol_energy = 1e-9
    common.conv_tol_grad = 1e-7
    common.scf_accelerator = SCFAccelerator.DIIS
    r_diis = run_uks(mol, basis, common)
    common.scf_accelerator = SCFAccelerator.ADIIS_DIIS
    r_hyb = run_uks(mol, basis, common)
    assert r_diis.converged and r_hyb.converged
    assert r_hyb.energy == pytest.approx(r_diis.energy, abs=1e-8)


# ---------------------------------------------------------------------------
# The EDIIS energy model must BE the energy (Kudin/Scuseria/Cances 2002)
# ---------------------------------------------------------------------------


def test_ediis_convex_hull_uses_vibeqc_total_density_energy_convention():
    """The exact QP objective must use vibe-qc's total-density convention.

    For ``E(D) = (D - 1/4)^2`` and ``F = dE/dD``, the convex hull of
    ``D = 0, 1`` has its unique minimum at coefficient ``t = 1/4`` on
    the second density. The literal pair-density factor from Kudin et
    al. Eq. (8), applied incorrectly to vibe-qc's total density, puts
    the stationary point at ``t = 3/8`` instead (the #119 objective-
    convention regression).
    """
    e = EDIIS(max_subspace=8)
    d0, d1 = np.array([[0.0]]), np.array([[1.0]])
    f0, f1 = np.array([[-0.5]]), np.array([[1.5]])
    e0, e1 = 0.0625, 0.5625

    e.extrapolate(f0, d0, energy=e0)
    e.discard_last_extrapolation()
    e.extrapolate(f1, d1, energy=e1)

    coeffs = np.asarray(e.last_coeffs())
    np.testing.assert_allclose(coeffs, [0.75, 0.25], rtol=0.0, atol=1.0e-12)
    d_hull = float(coeffs[1])
    assert (d_hull - 0.25) ** 2 == pytest.approx(0.0, abs=1.0e-14)


def test_ediis_open_shell_hull_uses_the_same_energy_convention():
    """The total-density factor applies to the summed spin-block traces."""
    e = EDIIS(max_subspace=8)
    d0a = d0b = np.array([[0.0]])
    d1a = d1b = np.array([[0.5]])
    f0a = f0b = np.array([[-0.5]])
    f1a = f1b = np.array([[1.5]])

    e.extrapolate_uhf(f0a, f0b, d0a, d0b, energy=0.0625)
    e.discard_last_extrapolation()
    e.extrapolate_uhf(f1a, f1b, d1a, d1b, energy=0.5625)

    coeffs = np.asarray(e.last_coeffs())
    np.testing.assert_allclose(coeffs, [0.75, 0.25], rtol=0.0, atol=1.0e-12)


def test_ediis_concave_nh2_hull_selects_the_lower_vertex():
    """The physical #486 hull is minimised, not merely extremised.

    Kudin, Scuseria & Cances, J. Chem. Phys. 116, 8255 (2002),
    doi:10.1063/1.1470195, Eqs. (7), (15)-(16): the coefficients minimise
    ``E(sum_i c_i D_i)`` on the simplex. The doublet/sextet fills below
    produce a concave HF hull whose interior stationary point is a maximum;
    the lower sextet vertex must win. The former convex-only KKT solver
    returned the maximum at ``t = 0.206889`` (issue #486).

    The two iterates are deliberately far apart -- the doublet and the
    sextet aufbau fills of the same Hcore orbitals -- so the objective
    error is large and the pair is fully determined by the geometry and
    basis (no guess or SCF trajectory in the fixture).
    """
    from vibeqc import _vibeqc_core as _core
    from vibeqc.rohf import (
        ROHFOptions,
        _aufbau_densities,
        _make_hf_fock_builder,
        _nuclear_repulsion,
        _orthonormaliser,
        _resolve_jk_builder,
    )

    mol = make_molecule(
        [
            (7, [0.0, 0.0, 0.0]),
            (1, [0.0, 1.5185884, 1.1988259]),
            (1, [0.0, -1.5185884, 1.1988259]),
        ],
        charge=0,
        multiplicity=2,
    )
    basis = BasisSet(mol, "sto-3g")
    s = np.asarray(_core.compute_overlap(basis), dtype=float)
    hcore = np.asarray(_core.compute_kinetic(basis), dtype=float) + np.asarray(
        _core.compute_nuclear(basis, mol), dtype=float
    )
    e_nuc = float(_nuclear_repulsion(mol))
    x = _orthonormaliser(s, 1.0e-7)
    build_j, build_k = _resolve_jk_builder(mol, basis, ROHFOptions())
    fock_builder = _make_hf_fock_builder(hcore, build_j, build_k)

    _eps, c_orth = np.linalg.eigh(x.T @ hcore @ x)
    c_mo = x @ c_orth
    d0 = _aufbau_densities(c_mo, 5, 4)  # doublet fill
    d1 = _aufbau_densities(c_mo, 7, 2)  # sextet fill

    def energy_at(t):
        da = (1.0 - t) * d0[0] + t * d1[0]
        db = (1.0 - t) * d0[1] + t * d1[1]
        _fa, _fb, e_elec = fock_builder(da, db)
        return e_elec + e_nuc

    e_at_0, e_at_1, e_at_half = energy_at(0.0), energy_at(1.0), energy_at(0.5)
    # E(t) = E(0) + b t + c t^2, exactly (HF energy is quadratic in D).
    quad = 2.0 * (e_at_0 + e_at_1 - 2.0 * e_at_half)
    lin = e_at_1 - e_at_0 - quad
    assert quad < -1.0e-3, "fixture must remain a concave hull"
    # The parabola is exact: verify off the three fitting points.
    assert e_at_0 + lin * 0.25 + quad * 0.0625 == pytest.approx(
        energy_at(0.25), abs=1.0e-10
    )
    t_star = -lin / (2.0 * quad)
    assert 0.0 < t_star < 1.0
    assert energy_at(t_star) > max(e_at_0, e_at_1)

    ediis = EDIIS(8)
    fa0, fb0, e0 = fock_builder(*d0)
    fa1, fb1, e1 = fock_builder(*d1)
    ediis.extrapolate_uhf(fa0, fb0, d0[0], d0[1], e0 + e_nuc)
    # The hybrid's contract: a produced-but-unused extrapolation is
    # discarded, which also disarms the anti-replay guard.
    ediis.discard_last_extrapolation()
    ediis.extrapolate_uhf(fa1, fb1, d1[0], d1[1], e1 + e_nuc)

    coeffs = np.asarray(ediis.last_coeffs(), dtype=float)
    assert coeffs.size == 2
    assert coeffs.sum() == pytest.approx(1.0, abs=1.0e-12)
    np.testing.assert_allclose(coeffs, [0.0, 1.0], rtol=0.0, atol=1.0e-10)
    assert energy_at(float(coeffs[1])) <= min(e_at_0, e_at_1) + 1.0e-10
