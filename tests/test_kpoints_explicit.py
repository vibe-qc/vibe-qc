"""Phase K4 — explicit user-supplied k-list ``KPoints.from_list``.

Pinned contracts:

1. **API surface** — ``KPoints.from_list(system, k_frac, weights=None,
   normalize=True)`` is public; returns a ``KPoints`` of
   ``kind="explicit"``.

2. **Default uniform weights** — passing only ``k_frac`` gives
   ``weights == 1/N`` for every point and ``weights.sum() == 1``.

3. **Custom weights normalized** — passing ``weights=[1, 8, 6]``
   (raw multiplicities) auto-normalises to fractional weights summing
   to 1.0 (default ``normalize=True``).

4. **Verbatim weights** — passing ``normalize=False`` preserves the
   user's weights as-is. Useful when feeding an already-normalized
   list from another code's symmetry analysis.

5. **Cartesian round-trip** — Cartesian ``kpoints_cart`` is computed
   from fractional via ``B @ k_frac.T`` to <1e-12.

6. **Negative weights rejected** — any negative entry raises
   ``ValueError``.

7. **Single-point convenience** — passing a length-3 array as
   ``k_frac`` (instead of (1, 3)) produces a 1-point list.

8. **Round-trip to BlochKMesh** — ``to_bloch_kmesh()`` produces a
   native ``BlochKMesh`` of the same length, weights, and Cartesian
   positions; periodic SCF drivers consume it unchanged.

9. **Empty input rejected** — zero-length k_frac raises ``ValueError``
   with a clear message.

10. **Shape validation** — wrong-shape input (not (N, dim), (dim,),
    (N, 3), or (3,)) raises ``ValueError``.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


ANGSTROM_TO_BOHR = 1.8897261339213


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _cubic_mg(a_ang: float = 3.0) -> vq.PeriodicSystem:
    a = a_ang * ANGSTROM_TO_BOHR
    return vq.PeriodicSystem(3, np.eye(3) * a, [vq.Atom(12, [0, 0, 0])])


def _low_dim(dim: int) -> vq.PeriodicSystem:
    if dim == 1:
        lat = np.diag([6.0, 30.0, 30.0])
    elif dim == 2:
        lat = np.diag([6.0, 7.0, 30.0])
    else:
        raise ValueError(dim)
    return vq.PeriodicSystem(dim, lat, [vq.Atom(1, [0.0, 0.0, 0.0])])


# ---------------------------------------------------------------------------
# 1. API surface
# ---------------------------------------------------------------------------

def test_from_list_is_public():
    assert hasattr(vq.KPoints, "from_list")


def test_returns_kpoints_with_kind_explicit():
    sys = _cubic_mg()
    kp = vq.KPoints.from_list(sys, [[0.0, 0.0, 0.0]])
    assert isinstance(kp, vq.KPoints)
    assert kp.kind == "explicit"
    assert kp.mesh is None
    assert kp.shift is None
    assert kp.ir_mapping.size == 0


# ---------------------------------------------------------------------------
# 2. Default uniform weights
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_points", [1, 2, 3, 5, 8])
def test_default_uniform_weights(n_points):
    sys = _cubic_mg()
    k_frac = [[i / n_points, 0.0, 0.0] for i in range(n_points)]
    kp = vq.KPoints.from_list(sys, k_frac)
    np.testing.assert_allclose(kp.weights, np.full(n_points, 1.0 / n_points),
                                atol=1e-12)
    np.testing.assert_allclose(kp.weights.sum(), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 3. Custom weights normalized
# ---------------------------------------------------------------------------

def test_custom_weights_auto_normalised():
    sys = _cubic_mg()
    raw = [1, 8, 6]
    kp = vq.KPoints.from_list(
        sys, [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0, 0]], weights=raw)
    expected = np.array(raw, dtype=np.float64) / sum(raw)
    np.testing.assert_allclose(kp.weights, expected, atol=1e-12)
    np.testing.assert_allclose(kp.weights.sum(), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 4. Verbatim weights with normalize=False
# ---------------------------------------------------------------------------

def test_unnormalised_preserves_input():
    sys = _cubic_mg()
    raw = [1.0, 8.0, 6.0]
    kp = vq.KPoints.from_list(
        sys, [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0, 0]],
        weights=raw, normalise=False)
    np.testing.assert_array_equal(kp.weights, raw)


# ---------------------------------------------------------------------------
# 5. Cartesian round-trip from fractional
# ---------------------------------------------------------------------------

def test_cartesian_from_fractional():
    """``k_cart == B @ k_frac`` to FP precision, where ``B`` has columns
    = reciprocal lattice vectors (vibe-qc convention)."""
    sys = _cubic_mg()
    k_frac = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.5, 0.5],
    ])
    kp = vq.KPoints.from_list(sys, k_frac.tolist())
    B = np.asarray(sys.reciprocal_lattice())
    expected_cart = (B @ k_frac.T).T
    np.testing.assert_allclose(kp.kpoints_cart, expected_cart, atol=1e-12)


# ---------------------------------------------------------------------------
# 6. Negative weights rejected
# ---------------------------------------------------------------------------

def test_negative_weight_raises():
    sys = _cubic_mg()
    with pytest.raises(ValueError, match="negative weight"):
        vq.KPoints.from_list(
            sys, [[0, 0, 0], [0.5, 0.5, 0.5]], weights=[1.0, -0.1])


# ---------------------------------------------------------------------------
# 7. Single-point convenience (length-3 array instead of (1, 3))
# ---------------------------------------------------------------------------

def test_single_point_via_length3():
    sys = _cubic_mg()
    kp = vq.KPoints.from_list(sys, [0.0, 0.0, 0.0])
    assert len(kp) == 1
    assert kp.kpoints_frac.shape == (1, 3)
    np.testing.assert_allclose(kp.weights, [1.0], atol=1e-12)


def test_one_dimensional_explicit_list_accepts_scalars():
    sys = _low_dim(1)
    kp = vq.KPoints.from_list(sys, [0.0, 0.25, 0.5])

    assert len(kp) == 3
    np.testing.assert_allclose(
        kp.kpoints_frac,
        [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0]],
        atol=1e-12,
    )
    np.testing.assert_allclose(kp.weights, [1.0 / 3.0] * 3, atol=1e-12)


def test_two_dimensional_explicit_list_pads_inactive_axis():
    sys = _low_dim(2)
    kp = vq.KPoints.from_list(sys, [[0.0, 0.0], [0.5, 0.25]])

    assert len(kp) == 2
    np.testing.assert_allclose(
        kp.kpoints_frac,
        [[0.0, 0.0, 0.0], [0.5, 0.25, 0.0]],
        atol=1e-12,
    )


def test_two_dimensional_single_point_via_length2():
    sys = _low_dim(2)
    kp = vq.KPoints.from_list(sys, [0.25, 0.5])

    assert len(kp) == 1
    np.testing.assert_allclose(kp.kpoints_frac, [[0.25, 0.5, 0.0]])


def test_low_dimensional_explicit_inactive_component_raises():
    sys = _low_dim(2)
    with pytest.raises(ValueError, match="inactive"):
        vq.KPoints.from_list(sys, [[0.0, 0.0, 0.1]])


def test_single_point_at_gamma_matches_gamma_constructor():
    """``from_list(sys, [0,0,0])`` should be equivalent to
    ``KPoints.gamma(sys)`` modulo construction kind."""
    sys = _cubic_mg()
    kp_a = vq.KPoints.from_list(sys, [0.0, 0.0, 0.0])
    kp_b = vq.KPoints.gamma(sys)
    np.testing.assert_allclose(kp_a.kpoints_cart, kp_b.kpoints_cart, atol=1e-12)
    np.testing.assert_allclose(kp_a.weights, kp_b.weights, atol=1e-12)


# ---------------------------------------------------------------------------
# 8. Round-trip to BlochKMesh
# ---------------------------------------------------------------------------

def test_to_bloch_kmesh_round_trip():
    sys = _cubic_mg()
    k_frac = [[0, 0, 0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5]]
    kp = vq.KPoints.from_list(sys, k_frac, weights=[1, 8, 1])
    bm = kp.to_bloch_kmesh()
    assert isinstance(bm, vq.BlochKMesh)
    assert len(bm) == 3
    np.testing.assert_allclose(
        np.asarray(bm.weights),
        np.array([1, 8, 1]) / 10.0,
        atol=1e-12,
    )
    bm_cart = np.asarray([np.asarray(k) for k in bm.kpoints])
    np.testing.assert_allclose(bm_cart, kp.kpoints_cart, atol=1e-12)


# ---------------------------------------------------------------------------
# 9. Empty input rejected
# ---------------------------------------------------------------------------

def test_empty_kfrac_raises():
    sys = _cubic_mg()
    with pytest.raises(ValueError, match="at least 1"):
        vq.KPoints.from_list(sys, np.zeros((0, 3)))


# ---------------------------------------------------------------------------
# 10. Shape validation
# ---------------------------------------------------------------------------

def test_wrong_inner_dim_raises():
    sys = _cubic_mg()
    with pytest.raises(ValueError, match="shape"):
        vq.KPoints.from_list(sys, [[0.0, 0.0]])  # length-2 inner


def test_wrong_outer_dim_raises():
    sys = _cubic_mg()
    with pytest.raises(ValueError, match="shape"):
        vq.KPoints.from_list(sys, np.zeros((2, 2, 3)))  # 3D


def test_weights_length_mismatch_raises():
    sys = _cubic_mg()
    with pytest.raises(ValueError, match="length"):
        vq.KPoints.from_list(sys, [[0, 0, 0], [0.5, 0, 0]],
                              weights=[1, 2, 3])


def test_zero_total_weight_with_normalise_raises():
    sys = _cubic_mg()
    with pytest.raises(ValueError, match="normalize"):
        vq.KPoints.from_list(sys, [[0, 0, 0], [0.5, 0, 0]],
                              weights=[0.0, 0.0])
