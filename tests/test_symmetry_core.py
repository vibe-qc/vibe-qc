"""Phase SYM1 tests: real-basis Wigner D-matrices.

Four contract tests drive the validation — any bug in the Euler-angle
extraction, the complex Wigner formula, or the real↔complex basis
transform shows up in at least one:

1. **Identity**        ``D^l(I) = I``.
2. **Orthogonality**    ``D^l(R) · D^l(R)ᵀ = I`` for any proper
   rotation R (real sph harmonics transform under real orthogonal
   representations).
3. **Homomorphism**     ``D^l(R₁ R₂) = D^l(R₁) · D^l(R₂)`` — composing
   rotations composes D-matrices.
4. **Inverse**          ``D^l(Rᵀ) = D^l(R)ᵀ``.

Plus a spot check for ``l = 1``: p orbitals rotate like Cartesian
vectors, so ``D^1(R)`` equals R up to the basis-ordering permutation
between real-spherical (y, z, x) and Cartesian (x, y, z).

Run across ``l = 0, 1, 2, 3, 4`` and a spread of random rotations.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


# SciPy's Rotation gives us a clean uniform sampler over SO(3) and
# trusted Euler-angle extraction for the spot checks.
from scipy.spatial.transform import Rotation as SciRot


L_RANGE = [0, 1, 2, 3, 4]


def _random_rotations(n: int, seed: int = 12345):
    rng = np.random.default_rng(seed)
    return [SciRot.random(num=1, rng=rng).as_matrix()[0] for _ in range(n)]


# ---------------------------------------------------------------------------
# Group identities
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("l", L_RANGE)
def test_identity_rotation_gives_identity_matrix(l):
    D = vq.wigner_d_real(l, np.eye(3))
    assert D.shape == (2 * l + 1, 2 * l + 1)
    assert np.allclose(D, np.eye(2 * l + 1), atol=1e-12)


@pytest.mark.parametrize("l", L_RANGE)
@pytest.mark.parametrize("idx", range(5))
def test_orthogonality(l, idx):
    """Real orthogonality: D · Dᵀ = I. Proper rotations act by real
    orthogonal matrices on real spherical harmonics, so this is a
    sanity check on the whole pipeline."""
    R = _random_rotations(5, seed=7777)[idx]
    D = vq.wigner_d_real(l, R)
    n = 2 * l + 1
    assert np.allclose(D @ D.T, np.eye(n), atol=1e-10)


@pytest.mark.parametrize("l", L_RANGE)
@pytest.mark.parametrize("idx", range(5))
def test_composition_homomorphism(l, idx):
    """``D^l`` is a group homomorphism: composing rotations composes
    D-matrices. Strongest of the four identities — catches bugs in
    Euler-angle extraction, the complex Wigner formula, and the
    real-basis transform simultaneously."""
    pairs = list(zip(
        _random_rotations(5, seed=1001),
        _random_rotations(5, seed=2002),
    ))
    R1, R2 = pairs[idx]
    D1 = vq.wigner_d_real(l, R1)
    D2 = vq.wigner_d_real(l, R2)
    D12 = vq.wigner_d_real(l, R1 @ R2)
    assert np.allclose(D12, D1 @ D2, atol=1e-10)


@pytest.mark.parametrize("l", L_RANGE)
@pytest.mark.parametrize("idx", range(5))
def test_inverse_rotation_gives_transpose(l, idx):
    """For a proper rotation R, ``R^{-1} = Rᵀ`` and the representation
    D inherits this: ``D^l(Rᵀ) = D^l(R)ᵀ``."""
    R = _random_rotations(5, seed=3030)[idx]
    D = vq.wigner_d_real(l, R)
    D_inv = vq.wigner_d_real(l, R.T)
    assert np.allclose(D_inv, D.T, atol=1e-12)


# ---------------------------------------------------------------------------
# l=1 specifics: p orbitals transform as Cartesian vectors
# ---------------------------------------------------------------------------

def test_l1_reflects_rotation_matrix_up_to_permutation():
    """Real spherical harmonics for l=1 are ordered (Y_{1,-1}, Y_{1,0},
    Y_{1,+1}) = (y, z, x) in the Condon–Shortley convention. A 3D
    rotation R acts on the Cartesian basis (x, y, z). The permutation
    (x, y, z) → (y, z, x) is P = [[0,1,0],[0,0,1],[1,0,0]].

    With Y^R = (y, z, x) ordering and Cartesian = (x, y, z), we have
    Y^R = P · (x, y, z)ᵀ, so R acting on Cartesian pulls back to
    ``D^1(R) = P · R · Pᵀ`` on the real-spherical basis.
    """
    # Rotation by 60° around z = rotate x→x cos + y sin, y→-x sin + y cos, z→z
    theta = np.deg2rad(60.0)
    R = SciRot.from_euler("z", theta).as_matrix()
    P = np.array([[0, 1, 0],
                  [0, 0, 1],
                  [1, 0, 0]], dtype=float)
    D_expected = P @ R @ P.T

    D_got = vq.wigner_d_real(1, R)
    assert np.allclose(D_got, D_expected, atol=1e-12)


def test_l1_pure_z_rotation_leaves_y_and_x_mixed():
    """A rotation around the z-axis mixes only the (y, x) = (Y_{1,-1},
    Y_{1,+1}) components; Y_{1,0} = z is invariant."""
    theta = 0.7  # radians, arbitrary
    R = SciRot.from_euler("z", theta).as_matrix()
    D = vq.wigner_d_real(1, R)
    # Middle row / column (m=0, z-component) is the second index.
    assert D[1, 1] == pytest.approx(1.0, abs=1e-12)
    assert D[1, 0] == pytest.approx(0.0, abs=1e-12)
    assert D[1, 2] == pytest.approx(0.0, abs=1e-12)
    assert D[0, 1] == pytest.approx(0.0, abs=1e-12)
    assert D[2, 1] == pytest.approx(0.0, abs=1e-12)
    # And (y, x) block should be a 2x2 rotation.
    yx_block = D[np.ix_([0, 2], [0, 2])]
    assert np.allclose(yx_block @ yx_block.T, np.eye(2), atol=1e-12)


# ---------------------------------------------------------------------------
# Special rotations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("l", L_RANGE)
def test_2pi_rotation_is_identity(l):
    """Rotation by 2π around any axis returns the identity for any
    integer l (Wigner D-matrices are single-valued reps of SO(3))."""
    axis = np.array([1.0, 2.0, 3.0])
    axis = axis / np.linalg.norm(axis)
    R = SciRot.from_rotvec(axis * 2 * np.pi).as_matrix()
    D = vq.wigner_d_real(l, R)
    assert np.allclose(D, np.eye(2 * l + 1), atol=1e-10)


@pytest.mark.parametrize("l", L_RANGE)
def test_pi_rotation_squared_is_identity(l):
    """A π-rotation squared is a 2π-rotation — the identity for any
    integer l. Tests the composition homomorphism on a specific
    non-trivial rotation."""
    R = SciRot.from_euler("y", np.pi).as_matrix()
    D = vq.wigner_d_real(l, R)
    D_sq = D @ D
    assert np.allclose(D_sq, np.eye(2 * l + 1), atol=1e-10)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_rejects_non_rotation_matrix():
    """``wigner_d_real`` accepts proper (det=+1) and improper
    (det=−1) rotations. A genuine non-rotation (scaling, shear) must
    still be rejected."""
    with pytest.raises(ValueError, match="not a rotation"):
        vq.wigner_d_real(2, 2.0 * np.eye(3))


def test_rejects_wrong_shape():
    with pytest.raises(ValueError, match="3x3"):
        vq.wigner_d_real(1, np.eye(4))


# ---------------------------------------------------------------------------
# Improper rotations: reflections, inversion, rotoreflections
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("l", L_RANGE)
def test_inversion_gives_signed_identity(l):
    """Inversion ``i = -I`` sends Y_{l,m}(r̂) to Y_{l,m}(-r̂) =
    (-1)^l Y_{l,m}(r̂). So D^l(i) = (-1)^l · I at every l."""
    D = vq.wigner_d_real(l, -np.eye(3))
    expected = ((-1) ** l) * np.eye(2 * l + 1)
    assert np.allclose(D, expected, atol=1e-12)


@pytest.mark.parametrize("l", L_RANGE)
def test_mirror_sigma_xy_squared_is_identity(l):
    """σ_h (reflection in the xy plane, det = -1) composed with
    itself is the identity — both for the rotation in 3D and for
    every D^l. Any σ² = E identity works; this is the cleanest."""
    sigma_h = np.diag([1.0, 1.0, -1.0])   # det = -1, σ² = I
    D = vq.wigner_d_real(l, sigma_h)
    D_sq = D @ D
    assert np.allclose(D_sq, np.eye(2 * l + 1), atol=1e-10)


@pytest.mark.parametrize("l", L_RANGE)
def test_mirror_is_orthogonal(l):
    """Any reflection is an orthogonal transformation, so D^l
    inherits: D^l · (D^l)^T = I."""
    sigma_xz = np.diag([1.0, -1.0, 1.0])
    D = vq.wigner_d_real(l, sigma_xz)
    n = 2 * l + 1
    assert np.allclose(D @ D.T, np.eye(n), atol=1e-12)


@pytest.mark.parametrize("l", L_RANGE)
def test_improper_homomorphism_with_proper(l):
    """D^l(σ · R) = D^l(σ) · D^l(R) for σ improper and R proper.
    Cross-check of the improper handler's consistency with the
    group product."""
    sigma = np.diag([1.0, 1.0, -1.0])
    R = SciRot.from_euler("z", 0.7).as_matrix()
    D_sigma = vq.wigner_d_real(l, sigma)
    D_R = vq.wigner_d_real(l, R)
    D_composed = vq.wigner_d_real(l, sigma @ R)
    assert np.allclose(D_composed, D_sigma @ D_R, atol=1e-10)


@pytest.mark.parametrize("l", L_RANGE)
def test_improper_squared_is_proper(l):
    """Improper · improper = proper (the two det=-1 signs cancel).
    Verify the product is correctly classified and handled."""
    sigma1 = np.diag([1.0, 1.0, -1.0])
    sigma2 = np.diag([1.0, -1.0, 1.0])
    # σ_h · σ_v = C2 about the x axis (a proper rotation).
    product = sigma1 @ sigma2
    assert abs(np.linalg.det(product) - 1.0) < 1e-12   # det = +1
    D_sigma1 = vq.wigner_d_real(l, sigma1)
    D_sigma2 = vq.wigner_d_real(l, sigma2)
    D_product = vq.wigner_d_real(l, product)
    assert np.allclose(D_product, D_sigma1 @ D_sigma2, atol=1e-10)


def test_l1_mirror_xy_flips_z_component():
    """Reflection through the xy plane flips z but leaves x and y
    invariant. In the real-spherical (y, z, x) ordering at l=1, this
    should appear as the sign flip on the z-component (middle entry)
    and nothing else."""
    sigma_h = np.diag([1.0, 1.0, -1.0])
    D = vq.wigner_d_real(1, sigma_h)
    # Expected behavior: y → y, z → -z, x → x.
    # In (y, z, x) basis, this is diag(1, -1, 1).
    expected = np.diag([1.0, -1.0, 1.0])
    assert np.allclose(D, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Complex Wigner D spot check
# ---------------------------------------------------------------------------

def test_wigner_small_d_symmetry():
    """d^l_{m',m}(β) satisfies d^l_{m',m}(-β) = d^l_{m,m'}(β) — see
    e.g. Sakurai §3.8. Tests the small-d formula directly."""
    for l in L_RANGE:
        for beta in (0.3, 1.1, 2.7):
            d_pos = vq.wigner_small_d(l, beta)
            d_neg = vq.wigner_small_d(l, -beta)
            assert np.allclose(d_neg, d_pos.T, atol=1e-12)


def test_wigner_small_d_at_zero_is_identity():
    for l in L_RANGE:
        d = vq.wigner_small_d(l, 0.0)
        assert np.allclose(d, np.eye(2 * l + 1), atol=1e-14)


# ---------------------------------------------------------------------------
# Euler-angle extractor round-trip
# ---------------------------------------------------------------------------

def test_euler_extraction_roundtrip():
    """R → (α, β, γ) → R via Rz(α) Ry(β) Rz(γ). Must be bitwise
    (to tolerance) invertible for non-degenerate rotations."""
    for R in _random_rotations(8, seed=909090):
        alpha, beta, gamma = vq.euler_angles_from_rotation(R)
        R_back = (
            SciRot.from_euler("z", alpha).as_matrix()
            @ SciRot.from_euler("y", beta).as_matrix()
            @ SciRot.from_euler("z", gamma).as_matrix()
        )
        assert np.allclose(R, R_back, atol=1e-10)


def test_euler_extraction_at_identity_is_zero():
    alpha, beta, gamma = vq.euler_angles_from_rotation(np.eye(3))
    assert alpha == pytest.approx(0.0, abs=1e-14)
    assert beta == pytest.approx(0.0, abs=1e-14)
    assert gamma == pytest.approx(0.0, abs=1e-14)
