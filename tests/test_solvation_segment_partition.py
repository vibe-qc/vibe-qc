"""The Becke fuzzy-cell partition that makes the CFC coarsening smooth (#757).

Klamt & Diedenhofen 2018 assign each basis point to exactly one segment by
nearest centre, so a point on a boundary flips at an infinitesimal
displacement and its *whole* area moves -- a discontinuous energy. The paper
does not address this; its own treatment of related discrete problems is to
avoid differentiating them (COC area gradients "neglected", triple-segment
area gradients "neglected", exact symmetric coincidences broken with "a small
geometrical noise function", p. 1652).

So the fix is Becke's fuzzy cells (J. Chem. Phys. 88, 2547 (1988),
doi:10.1063/1.454033), the canonical way to make a point-to-centre assignment
continuous, already used in this codebase for DFT molecular grids.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from vibeqc.solvation.segment_partition import (
    BECKE_K,
    becke_switch,
    smooth_partition,
)


def _sphere(n, seed, radius=3.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    return radius * x / np.linalg.norm(x, axis=1, keepdims=True)


# ---------------------------------------------------------------------
# The switch: one definition, shared with the DFT grid
# ---------------------------------------------------------------------


def test_becke_switch_is_the_published_recurrence():
    """``s(mu) = (1 - p^k(mu))/2`` with ``p(mu) = (3 mu - mu^3)/2``.

    Becke 1988 eq. 19-21. Endpoints are exact by construction: ``s(-1) = 1``,
    ``s(0) = 1/2``, ``s(1) = 0``, and ``s`` is odd about ``mu = 0``.
    """
    assert becke_switch(np.array([-1.0, 0.0, 1.0])) == pytest.approx([1.0, 0.5, 0.0])
    mu = np.linspace(-0.99, 0.99, 41)
    assert becke_switch(mu) + becke_switch(-mu) == pytest.approx(np.ones_like(mu))
    # Explicitly, one iteration at a time.
    def p(x):
        return 1.5 * x - 0.5 * x ** 3
    for k in (1, 2, 3, 5):
        expect = 0.5 * (1.0 - np.array([p(p(p(p(p(m)[:k][0] if False else m)))) for m in [0.0]]))
        got = becke_switch(np.array([0.37]), k)
        m = 0.37
        for _ in range(k):
            m = p(m)
        assert got[0] == pytest.approx(0.5 * (1.0 - m), abs=0.0, rel=1e-15)


def test_smoothing_order_matches_the_dft_grid_default():
    """One definition, not two that agree today.

    ``GridOptions.becke_k`` defaults to 3 in ``cpp/include/vibeqc/grid.hpp``,
    which is also Becke's published choice. Pinning them equal is what stops
    the cavity and the DFT grid from drifting apart -- the same
    two-copies-of-one-quantity hazard as #546.
    """
    assert BECKE_K == 3
    # Located from this file, not from the working directory: the assertion is
    # about the C++ header, so it must find the header wherever pytest was
    # started. cpp/ and python/ stay in the same repository after the 2026-09
    # split (vibeqc#581), so this cross-directory read survives it -- but the
    # path is written to break loudly rather than silently if that changes.
    header_path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "cpp" / "include" / "vibeqc" / "grid.hpp"
    )
    assert header_path.is_file(), f"missing {header_path}"
    assert "int becke_k  = 3;" in header_path.read_text()


# ---------------------------------------------------------------------
# Partition of unity, and what it buys
# ---------------------------------------------------------------------


@pytest.mark.parametrize("n_cen", [2, 8, 24])
def test_weights_are_a_partition_of_unity(n_cen):
    """Exact, and load-bearing: it is what conserves the cavity's total area.

    The CFC's area feeds COSMO-RS sigma profiles, so a partition that only
    approximately summed to one would leak surface.
    """
    cen = _sphere(n_cen, 0)
    pts = _sphere(30, 1, radius=3.1)
    part = smooth_partition(pts, cen)
    assert part.row_sum_residual() < 1e-14
    assert np.all(part.weights >= 0.0)


def test_a_point_at_a_centre_takes_all_of_its_weight():
    """The hard-assignment limit is recovered exactly where it is unambiguous.

    At ``t = c_S`` the ratio ``mu_ST`` saturates the triangle inequality for
    every other centre, so ``s = 0`` exactly and the point belongs wholly to
    ``S``. A partition that blurred even here would not reduce to the paper's
    construction anywhere.
    """
    cen = _sphere(12, 2)
    part = smooth_partition(cen[3:4], cen)
    assert part.weights[0, 3] == pytest.approx(1.0, abs=1e-14)
    assert np.max(np.abs(np.delete(part.weights[0], 3))) < 1e-14


def test_weights_move_continuously_across_a_former_assignment_boundary():
    """The whole point of #757, at the level of the primitive.

    A point swept past the midpoint between two centres transfers its weight
    smoothly. Under the hard assignment this was a step from 1 to 0; here the
    largest single-step change across a fine sweep is bounded, and the weight
    passes through 1/2 at the midpoint by symmetry.
    """
    cen = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    ts = np.linspace(-0.4, 0.4, 401)
    pts = np.stack([ts, np.full_like(ts, 2.0), np.zeros_like(ts)], axis=1)
    w = smooth_partition(pts, cen).weights[:, 0]
    assert w[0] > 0.5 > w[-1]
    assert w[len(ts) // 2] == pytest.approx(0.5, abs=1e-12)
    assert np.max(np.abs(np.diff(w))) < 0.02
    assert np.all(np.diff(w) < 0.0)          # monotone, no wobble


def test_no_nearest_neighbour_truncation():
    """A cutoff would be faster and would reintroduce the defect.

    Truncating the product at the nearest few centres is tempting -- a distant
    factor is close to 1 -- but ``s(mu) = 0`` needs ``mu >= 1``, so distant
    factors are near 1 and not equal to it: at 16 nearest centres the weights
    moved by 1.2e-3. Worse, the cutoff itself is discontinuous, since the m-th
    and (m+1)-th nearest centres swap as a point moves. This pins that the
    implementation uses the full product, by checking a centre far outside any
    plausible cutoff still changes the answer.
    """
    cen = _sphere(20, 3)
    pts = _sphere(5, 4, radius=3.05)
    base = smooth_partition(pts, cen).weights
    far = np.vstack([cen, [[0.0, 0.0, 40.0]]])
    with_far = smooth_partition(pts, far).weights[:, :-1]
    assert 0.0 < np.max(np.abs(base - with_far))


# ---------------------------------------------------------------------
# The adjoints
# ---------------------------------------------------------------------


def test_vjp_matches_finite_differences():
    """Both halves: the points and the centres."""
    cen = _sphere(24, 5)
    pts = _sphere(15, 6, radius=3.0)
    rng = np.random.default_rng(7)
    adj = rng.normal(size=(15, 24))
    part = smooth_partition(pts, cen)
    d_pts, d_cen = part.vjp(adj)

    def g(p, c):
        return float(np.sum(adj * smooth_partition(p, c).weights))

    h = 1e-6
    for b in (0, 4, 11):
        for k in range(3):
            up, dn = pts.copy(), pts.copy()
            up[b, k] += h
            dn[b, k] -= h
            fd = (g(up, cen) - g(dn, cen)) / (2 * h)
            assert abs(fd - d_pts[b, k]) < 1e-6 * max(abs(fd), 1.0)
    for S in (0, 9, 20):
        for k in range(3):
            up, dn = cen.copy(), cen.copy()
            up[S, k] += h
            dn[S, k] -= h
            fd = (g(pts, up) - g(pts, dn)) / (2 * h)
            assert abs(fd - d_cen[S, k]) < 1e-6 * max(abs(fd), 1.0)


def test_constant_adjoint_gives_exactly_zero():
    """``sum_S W_S = 1`` implies ``sum_S dW_S = 0``, for every input.

    An exact identity, so it needs no finite differences and no tolerance
    argument -- the sharpest cheap check on the adjoints, and the reason the
    cavity's *total* area derivative is immune to partition errors.
    """
    cen = _sphere(18, 8)
    pts = _sphere(20, 9, radius=2.9)
    part = smooth_partition(pts, cen)
    d_pts, d_cen = part.vjp(np.ones((20, 18)))
    assert np.max(np.abs(d_pts)) < 1e-13
    assert np.max(np.abs(d_cen)) < 1e-13


def test_chunking_does_not_change_the_answer():
    """The point loop is chunked to bound memory; results must not depend on it."""
    import vibeqc.solvation.segment_partition as sp

    cen = _sphere(14, 10)
    pts = _sphere(37, 11, radius=3.02)
    adj = np.random.default_rng(12).normal(size=(37, 14))
    saved = sp.POINT_CHUNK
    try:
        sp.POINT_CHUNK = 64
        w_a = smooth_partition(pts, cen).weights
        v_a = smooth_partition(pts, cen).vjp(adj)
        sp.POINT_CHUNK = 7
        w_b = smooth_partition(pts, cen).weights
        v_b = smooth_partition(pts, cen).vjp(adj)
    finally:
        sp.POINT_CHUNK = saved
    assert np.array_equal(w_a, w_b)
    assert np.max(np.abs(v_a[0] - v_b[0])) < 1e-15
    assert np.max(np.abs(v_a[1] - v_b[1])) < 1e-13


def test_coincident_centres_are_refused():
    """Two centres at one point make Becke's ratio undefined, as for two atoms."""
    cen = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    with pytest.raises(ValueError, match="coincide"):
        smooth_partition(np.zeros((1, 3)), cen)
