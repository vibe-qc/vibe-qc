"""One screening-factor derivation per language, carried not recomputed.

Guards the structural hazard behind #546 (issues #548, #549).

`f(e)` was written out independently in four places: the Python
`dielectric_factor`, the C++ `cpcm_dielectric_factor`, the solvation gradient
(hard-coded `variant="cpcm"`, the #546 wrong answer), and by hand inside the
C++ MSINDO reaction field. Nothing pinned any two of them equal, and a drift
is silent because every path still yields a smooth plausible number.

Klamt & Schuurmann 1993 p. 800 gives the family these are all points of:
`f(e) = (e - 1)/(e + x)` with `x` in 0-2. CPCM is `x = 0`, COSMO is `x = 1/2`
(1993 p. 801). Expressing it once with a per-variant `x` makes a new variant a
table entry instead of a branch, and lets one test pin Python against C++
across the whole family.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from vibeqc.solvation.screening import (
    SCREENING_X,
    ScreeningModel,
    dielectric_factor,
    screening_factor,
    screening_x,
)

# A grid that spans the range where the variants actually diverge. n-pentane
# and cyclohexane sit near the bottom of the shipped presets; water at the top.
EPSILON_GRID = [1.84, 2.02, 2.27, 4.71, 8.93, 20.49, 35.94, 78.39, 1e6]


# ---------------------------------------------------------------------
# The one-parameter family
# ---------------------------------------------------------------------


def test_variants_are_points_of_the_klamt_family():
    """CPCM and COSMO are `x = 0` and `x = 1/2` of one formula."""
    assert SCREENING_X["cpcm"] == 0.0
    assert SCREENING_X["cosmo"] == 0.5
    for eps in EPSILON_GRID:
        assert screening_factor(eps, 0.0) == pytest.approx(
            (eps - 1.0) / eps, rel=1e-15
        )
        assert screening_factor(eps, 0.5) == pytest.approx(
            (eps - 1.0) / (eps + 0.5), rel=1e-15
        )


def test_screening_x_rejects_unknown_variant():
    with pytest.raises(ValueError, match="unknown variant"):
        screening_x("bogus")


@pytest.mark.parametrize("variant", sorted(SCREENING_X))
def test_conductor_limit_is_exactly_one(variant):
    """`epsilon = inf` gives exactly 1, not the closed form's NaN.

    The conductor is the limit COSMO is derived from (Klamt 1993 eq. 2), so
    the generic layer validates against it; `(inf - 1)/(inf + x)` is NaN.
    """
    sm = ScreeningModel.from_variant(math.inf, variant)
    assert sm.is_conductor
    assert sm.f == 1.0
    assert dielectric_factor(math.inf, variant=variant) == 1.0


@pytest.mark.parametrize("variant", sorted(SCREENING_X))
def test_screening_model_is_self_consistent(variant):
    for eps in EPSILON_GRID:
        sm = ScreeningModel.from_variant(eps, variant)
        assert sm.variant == variant
        assert sm.epsilon == eps
        assert sm.x == SCREENING_X[variant]
        assert sm.f == screening_factor(eps, sm.x)
        assert sm.f == dielectric_factor(eps, variant=variant)
        assert not sm.is_conductor


@pytest.mark.parametrize("eps", [0.5, 1.0, -3.0])
def test_rejects_non_screening_epsilon(eps):
    with pytest.raises(ValueError, match="epsilon must be > 1"):
        screening_factor(eps, 0.0)


# ---------------------------------------------------------------------
# Python and C++ pinned equal -- neither can move alone
# ---------------------------------------------------------------------


def _native_or_skip():
    try:
        from vibeqc import _vibeqc_core as core
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"native core unavailable: {exc}")
    if not hasattr(core, "cpcm_dielectric_factor"):  # pragma: no cover
        pytest.skip("native cpcm_dielectric_factor unavailable")
    return core


@pytest.mark.parametrize("variant", sorted(SCREENING_X))
def test_python_and_cpp_screening_factors_agree_bitwise(variant):
    """The two independent implementations must not drift.

    Same intent as `tests/test_smearing_frontier_resolution.py` pinning the
    Python roundoff threshold equal to its C++ twin: `f` is a closed-form
    rational function, so bitwise equality is the right bar, and anything
    weaker would let the #546 class of divergence back in.
    """
    core = _native_or_skip()
    for eps in EPSILON_GRID:
        py = dielectric_factor(eps, variant=variant)
        cpp = core.cpcm_dielectric_factor(float(eps), variant)
        assert py == cpp, f"{variant} at eps={eps}: python {py!r} != c++ {cpp!r}"


def test_cpp_rejects_the_same_variants_python_does():
    core = _native_or_skip()
    with pytest.raises((ValueError, RuntimeError)):
        core.cpcm_dielectric_factor(10.0, "bogus")
    with pytest.raises((ValueError, RuntimeError)):
        core.cpcm_dielectric_factor(1.0, "cpcm")


# ---------------------------------------------------------------------
# The model is carried on the result, not re-derived downstream (#548)
# ---------------------------------------------------------------------


def _water(vq):
    return vq.Molecule(
        [
            vq.Atom(8, [0.0, 0.0, 0.0]),
            vq.Atom(1, [1.81, 0.0, 0.20]),
            vq.Atom(1, [-0.38, 1.75, 0.0]),
        ],
        0,
        1,
    )


@pytest.mark.parametrize("variant", sorted(SCREENING_X))
def test_solvent_result_carries_the_screening_model(variant):
    """A consumer that cannot recompute `f` cannot recompute it wrongly."""
    vq = pytest.importorskip("vibeqc")
    mol = _water(vq)
    basis = vq.BasisSet(mol, "sto-3g")
    sm = vq.SolventModel(
        epsilon=2.27,
        name="benzene",
        variant=variant,
        n_points_per_sphere=50,
        max_macro_iter=40,
        tol_e_solv=1e-9,
    )
    sol = vq.run_cpcm_scf(mol, basis, method="rhf", solvent=sm)
    assert sol.converged

    screening = sol.screening
    assert isinstance(screening, ScreeningModel)
    assert screening.variant == variant
    assert screening.epsilon == 2.27
    assert screening.f == dielectric_factor(2.27, variant=variant)

    # The charges really were built with that f: q = -f A^-1 V, so scaling it
    # back out must reproduce the unscaled conductor solution.
    from vibeqc.solvation.cpcm import build_A_matrix

    A = build_A_matrix(sol.cavity.points, sol.cavity.weights)
    q0 = -np.linalg.solve(A, np.asarray(sol.cpcm.V, dtype=np.float64))
    np.testing.assert_allclose(
        np.asarray(sol.cpcm.q, dtype=np.float64), screening.f * q0,
        rtol=1e-9, atol=1e-12,
    )


def test_gas_phase_result_has_no_screening_model():
    """Gas phase is *absent* screening, not an out-of-range epsilon (#549)."""
    vq = pytest.importorskip("vibeqc")
    mol = _water(vq)
    basis = vq.BasisSet(mol, "sto-3g")
    sol = vq.run_cpcm_scf(mol, basis, method="rhf", solvent="vacuum")
    assert sol.screening is None
    assert float(np.abs(np.asarray(sol.cpcm.q)).max()) == 0.0


def test_gas_phase_gradient_equals_the_plain_gas_phase_gradient():
    """#549: the uniform gas-phase return type must survive the gradient.

    `_gas_phase_solvent_result` exists to give downstream code one shape.
    Rejecting it with `epsilon must be > 1` defeats that, and with `q`
    identically zero the reaction field contributes exactly nothing, so the
    answer is the gas-phase gradient itself.
    """
    vq = pytest.importorskip("vibeqc")
    mol = _water(vq)
    basis = vq.BasisSet(mol, "sto-3g")
    sol = vq.run_cpcm_scf(mol, basis, method="rhf", solvent="vacuum")

    grad = np.asarray(
        vq.cpcm_gradient(sol.scf, mol, basis, sol, method="rhf"), dtype=np.float64
    )
    ref = np.asarray(vq.compute_gradient(mol, basis, sol.scf, None), dtype=np.float64)
    # Machine precision, not bitwise: the two calls reduce the same sums in a
    # thread-count-dependent order, so they differ by ~1 ulp (measured 8.9e-16
    # on components of order 0.05, i.e. ~3e-14 relative, and only once the
    # lane sandbox changed OMP_NUM_THREADS). A reaction field that actually
    # contributed would show up at 1e-3 or above, four orders of magnitude
    # clear of this bound.
    np.testing.assert_allclose(grad, ref, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------
# MSINDO's C++ reaction field takes the variant instead of assuming it (#548)
# ---------------------------------------------------------------------


def test_cpp_msindo_reaction_field_accepts_the_screening_variant():
    """The MSINDO reaction field must not bake the COSMO form in.

    It re-derived ``(e - 1)/(e + 0.5)`` by hand, in a header that did not even
    include ``solvation_cpcm.hpp``. That is both a copy that can drift and a
    special case: MSINDO COSMO structurally could not be given CPCM screening
    even though the generic layer supports both.
    """
    _native_or_skip()
    from vibeqc._vibeqc_core.semiempirical import indo

    rng = np.random.default_rng(20260902)
    nseg, nsto = 6, 4
    # Component layout: nsto diagonal components, then one per off-diagonal
    # pair, whose component index must sit at or past nsto.
    pairs = [(nsto + k, mu, nu) for k, (mu, nu) in enumerate([(0, 1), (0, 2), (1, 2)])]
    n_comp = nsto + len(pairs)

    B = np.asfortranarray(rng.normal(size=(n_comp, nseg)))
    pts = rng.normal(size=(nseg, 3)) * 3.0
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    A = np.zeros((nseg, nseg))
    off = d > 0
    A[off] = 1.0 / d[off]
    np.fill_diagonal(A, 6.0)
    A = np.asfortranarray(0.5 * (A + A.T))
    Vc = rng.normal(size=nseg)
    P = rng.normal(size=(nsto, nsto))
    P = np.asfortranarray(0.5 * (P + P.T))

    eps = 2.27
    q = {}
    for variant in ("cpcm", "cosmo"):
        res = indo.cosmo_reaction_field(B, A, Vc, P, nsto, pairs, eps, variant)
        q[variant] = np.asarray(res.q, dtype=np.float64)

    assert np.abs(q["cosmo"]).max() > 0.0
    # q = -f A^-1 V scales linearly in f, so the two variants must differ by
    # exactly the ratio of their screening factors. Anything else means the
    # new argument is being ignored.
    ratio = dielectric_factor(eps, variant="cosmo") / dielectric_factor(
        eps, variant="cpcm"
    )
    np.testing.assert_allclose(
        q["cosmo"], ratio * q["cpcm"], rtol=1e-12, atol=1e-14
    )


def test_cpp_msindo_reaction_field_defaults_to_cosmo():
    """Default is unchanged: omitting the variant is the COSMO form."""
    _native_or_skip()
    from vibeqc._vibeqc_core.semiempirical import indo

    rng = np.random.default_rng(4242)
    nseg, nsto = 5, 3
    pairs = [(nsto + k, mu, nu) for k, (mu, nu) in enumerate([(0, 1), (1, 2)])]
    B = np.asfortranarray(rng.normal(size=(nsto + len(pairs), nseg)))
    A = np.asfortranarray(np.eye(nseg) * 5.0 + 0.1)
    Vc = rng.normal(size=nseg)
    P = rng.normal(size=(nsto, nsto))
    P = np.asfortranarray(0.5 * (P + P.T))

    implicit = indo.cosmo_reaction_field(B, A, Vc, P, nsto, pairs, 8.93)
    explicit = indo.cosmo_reaction_field(B, A, Vc, P, nsto, pairs, 8.93, "cosmo")
    np.testing.assert_array_equal(
        np.asarray(implicit.q), np.asarray(explicit.q)
    )


def test_cpp_msindo_reaction_field_rejects_unscreening_epsilon():
    """Rejection is the shared helper's job, so the message is uniform."""
    _native_or_skip()
    from vibeqc._vibeqc_core.semiempirical import indo

    nseg, nsto = 4, 2
    pairs = [(nsto, 0, 1)]
    B = np.asfortranarray(np.ones((nsto + len(pairs), nseg)))
    A = np.asfortranarray(np.eye(nseg) * 4.0)
    Vc = np.ones(nseg)
    P = np.asfortranarray(np.eye(nsto))
    with pytest.raises((ValueError, RuntimeError), match="epsilon must be > 1"):
        indo.cosmo_reaction_field(B, A, Vc, P, nsto, pairs, 1.0)
    with pytest.raises((ValueError, RuntimeError)):
        indo.cosmo_reaction_field(B, A, Vc, P, nsto, pairs, 8.93, "bogus")


def test_msindo_cosmo_accepts_a_screening_variant_end_to_end():
    """MSINDO is no longer the one route that cannot take CPCM screening.

    The screening variant is a property of the generic reaction-field layer,
    not of the Hamiltonian. Before #548 it was a constant compiled into
    ``indo_engine.hpp``, so this call was impossible.
    """
    pytest.importorskip("vibeqc")
    from vibeqc.semiempirical.methods.msindo_cosmo import msindo_cosmo

    Z = [8, 1, 1]
    xyz = [[0.0, 0.0, 0.0], [0.9584, 0.0, 0.0], [-0.2396, 0.9268, 0.0]]

    # Low dielectric, where the two factors are 18% apart.
    cosmo = msindo_cosmo(Z, xyz, epsilon=2.27, variant="cosmo", cavity="gepol")
    cpcm = msindo_cosmo(Z, xyz, epsilon=2.27, variant="cpcm", cavity="gepol")

    assert cosmo.converged and cpcm.converged
    # CPCM screens harder (f 0.559 vs 0.458), so it must stabilise more.
    assert cpcm.e_pol < cosmo.e_pol < 0.0


def test_msindo_cosmo_default_variant_is_unchanged():
    """Defaulting must remain the Klamt COSMO form, bit for bit."""
    pytest.importorskip("vibeqc")
    from vibeqc.semiempirical.methods.msindo_cosmo import msindo_cosmo

    Z = [8, 1, 1]
    xyz = [[0.0, 0.0, 0.0], [0.9584, 0.0, 0.0], [-0.2396, 0.9268, 0.0]]
    implicit = msindo_cosmo(Z, xyz, epsilon=78.39, cavity="gepol")
    explicit = msindo_cosmo(Z, xyz, epsilon=78.39, variant="cosmo", cavity="gepol")
    assert implicit.total_energy == explicit.total_energy
    assert implicit.e_solv == explicit.e_solv
    assert implicit.e_pol == explicit.e_pol
