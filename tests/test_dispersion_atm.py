"""Axilrod-Teller-Muto (ATM) three-body term of D3(BJ) -- the ``s9`` scale.

The two-body D3(BJ) energy is what ``s9 = 0`` gives; ``s9 = 1`` adds the
three-body dipole-dipole-dipole term. See ``cpp/include/vibeqc/dispersion.hpp``
for the equations and ``python/vibeqc/dispersion.py`` for the backend split.

Two families of check here:

1. **Oracle parity** against Grimme's reference ``dftd3`` library. Rare-gas
   clusters isolate the ATM machinery because their C6 values have a single
   free-atom reference environment; the RP208 tests additionally cover the
   full coordination-dependent interpolation and its gradient.

2. **Self-consistency**: analytic ATM gradient vs finite differences, and
   the C++ r0ab radii vs the Python table they were copied from.

Regression guard, and the reason this file exists: ``dftd3``'s
``RationalDampingParam`` defaults ``s9 = 1.0``. Before v0.15.34 vibe-qc
omitted ``s9`` from that call, so the ``dftd3`` backend silently applied
ATM to *every* functional while the builtin backend never did -- the same
``run_job`` gave different physics depending on whether an optional pip
package was installed. ``test_dftd3_backend_does_not_silently_apply_atm``
is the test that would have caught it.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc._d3_r0ab import r0ab as _r0ab_python
from vibeqc._vibeqc_core import d3_r0ab as _r0ab_cpp


requires_dftd3 = pytest.mark.skipif(
    not vq.dftd3_available(),
    reason="optional dftd3 package not installed",
)

# Damping used throughout: HSE-3c's set. s8 = 0 keeps the two-body term
# pure-C6, which isolates the ATM contribution cleanly.
_HSE3C = dict(s6=1.0, s8=0.0, a1=0.4411, a2=4.5182)

# Rare gases provide a coordination-independent ATM isolation case.
_RARE_GASES = (2, 10, 18)


def _mol(numbers, positions) -> "vq.Molecule":
    return vq.Molecule(
        [vq.Atom(int(z), [float(c) for c in r])
         for z, r in zip(numbers, positions)]
    )


def _dftd3_energy(numbers, positions, *, s9: float, **damping) -> float:
    """Reference D3 energy straight from Grimme's library."""
    from dftd3.interface import DispersionModel, RationalDampingParam

    model = DispersionModel(
        np.asarray(numbers, dtype=np.int32), np.asarray(positions, dtype=float)
    )
    param = RationalDampingParam(s9=s9, **damping)
    return float(model.get_dispersion(param, grad=False)["energy"])


def _builtin_energy(numbers, positions, *, s9: float, **damping) -> float:
    params = vq.D3BJParams(s9=s9, **damping)
    return vq.compute_d3bj(_mol(numbers, positions), params,
                           backend="builtin").energy


def _cluster(numbers, seed: int) -> np.ndarray:
    """A well-separated, non-degenerate cluster geometry (bohr)."""
    rng = np.random.default_rng(seed)
    n = len(numbers)
    return (rng.uniform(-2.5, 2.5, (n, 3))
            + np.arange(n)[:, None] * np.array([4.4, 0.0, 0.0]))


# ---------------------------------------------------------------------------
# Parameter surface
# ---------------------------------------------------------------------------

def test_s9_defaults_to_zero():
    """Published per-functional (s8, a1, a2) sets were fit two-body-only,
    so a bare D3BJParams must not carry a three-body term."""
    assert vq.D3BJParams().s9 == 0.0
    assert vq.D3BJParams(s6=1.0, s8=0.5, a1=0.4, a2=4.5).s9 == 0.0


def test_s9_is_settable_and_shown_in_repr():
    p = vq.D3BJParams(s6=1.0, s8=0.0, a1=0.4, a2=4.5, s9=1.0)
    assert p.s9 == 1.0
    assert "s9" in repr(p)


def test_per_functional_lookup_has_no_three_body_term():
    """d3bj_params_for('pbe') is plain D3(BJ): two-body only."""
    for backend in ("builtin",) + (("dftd3",) if vq.dftd3_available() else ()):
        p = vq.d3bj_params_for("pbe", backend=backend)
        assert p is not None
        assert p.s9 == 0.0, backend


# ---------------------------------------------------------------------------
# r0ab table: the C++ copy must not drift from the Python one
# ---------------------------------------------------------------------------

def test_cpp_r0ab_matches_python_table():
    max_z = 18
    for za in range(1, max_z + 1):
        for zb in range(1, max_z + 1):
            assert _r0ab_cpp(za, zb) == pytest.approx(
                _r0ab_python(za, zb), abs=1e-9
            ), f"r0ab({za},{zb})"


def test_cpp_r0ab_is_symmetric_and_nan_out_of_range():
    assert _r0ab_cpp(6, 1) == _r0ab_cpp(1, 6)
    assert np.isnan(_r0ab_cpp(19, 1))
    assert np.isnan(_r0ab_cpp(1, 0))


# ---------------------------------------------------------------------------
# Physics: the ATM term is repulsive for a compact equilateral triangle
# ---------------------------------------------------------------------------

def test_atm_is_repulsive_for_equilateral_triangle():
    """For an equilateral triangle every angle is 60 deg, so
    3.cos^3(60) + 1 = 1.375 > 0 and the term raises the energy."""
    side = 7.0
    pos = [[0.0, 0.0, 0.0], [side, 0.0, 0.0],
           [side / 2, side * np.sqrt(3) / 2, 0.0]]
    nums = [18, 18, 18]
    e_2body = _builtin_energy(nums, pos, s9=0.0, **_HSE3C)
    e_3body = _builtin_energy(nums, pos, s9=1.0, **_HSE3C)
    assert e_2body < 0.0
    assert e_3body > e_2body, "ATM must be repulsive here"


def test_atm_scales_linearly_in_s9():
    pos = _cluster([18, 18, 18], seed=1)
    nums = [18, 18, 18]
    e0 = _builtin_energy(nums, pos, s9=0.0, **_HSE3C)
    e1 = _builtin_energy(nums, pos, s9=1.0, **_HSE3C)
    e_half = _builtin_energy(nums, pos, s9=0.5, **_HSE3C)
    assert e_half == pytest.approx(e0 + 0.5 * (e1 - e0), rel=1e-12)


def test_atm_vanishes_below_three_atoms():
    for nums, pos in (([18], [[0.0, 0.0, 0.0]]),
                      ([18, 18], [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]])):
        e0 = _builtin_energy(nums, pos, s9=0.0, **_HSE3C)
        e1 = _builtin_energy(nums, pos, s9=1.0, **_HSE3C)
        assert e1 == pytest.approx(e0, rel=0, abs=0)


# ---------------------------------------------------------------------------
# Oracle parity (homonuclear rare gases)
# ---------------------------------------------------------------------------

@requires_dftd3
@pytest.mark.parametrize("Z", _RARE_GASES)
@pytest.mark.parametrize("n_atoms", [3, 4, 6])
def test_builtin_atm_matches_dftd3_for_homonuclear_rare_gas(Z, n_atoms):
    """Pins the whole ATM machinery bit-for-bit: damping (r0ab-based zero
    damping with exponent alp+2 = 16), the angular factor, C9 as the
    geometric mean of the three pair C6, the sign, and the A<B<C sum."""
    nums = [Z] * n_atoms
    pos = _cluster(nums, seed=100 + Z + n_atoms)
    got = _builtin_energy(nums, pos, s9=1.0, **_HSE3C)
    ref = _dftd3_energy(nums, pos, s9=1.0, **_HSE3C)
    assert got == pytest.approx(ref, rel=0, abs=1e-12)


@requires_dftd3
@pytest.mark.parametrize("Z", _RARE_GASES)
def test_builtin_two_body_matches_dftd3_when_s9_zero(Z):
    """Turning ATM off must reproduce the pre-existing two-body number."""
    nums = [Z] * 4
    pos = _cluster(nums, seed=7)
    got = _builtin_energy(nums, pos, s9=0.0, **_HSE3C)
    ref = _dftd3_energy(nums, pos, s9=0.0, **_HSE3C)
    assert got == pytest.approx(ref, rel=0, abs=1e-12)


@requires_dftd3
def test_atm_increment_agrees_with_dftd3():
    """The isolated E(s9=1) - E(s9=0) increment, not just the total."""
    nums = [18] * 5
    pos = _cluster(nums, seed=42)
    d_builtin = (_builtin_energy(nums, pos, s9=1.0, **_HSE3C)
                 - _builtin_energy(nums, pos, s9=0.0, **_HSE3C))
    d_ref = (_dftd3_energy(nums, pos, s9=1.0, **_HSE3C)
             - _dftd3_energy(nums, pos, s9=0.0, **_HSE3C))
    assert d_builtin == pytest.approx(d_ref, rel=0, abs=1e-13)
    assert d_builtin != 0.0


# ---------------------------------------------------------------------------
# The regression this file was written for
# ---------------------------------------------------------------------------

@requires_dftd3
def test_dftd3_backend_does_not_silently_apply_atm():
    """``dftd3``'s RationalDampingParam defaults s9=1.0. vibe-qc must pass
    s9 explicitly, so that a default D3BJParams (s9=0) yields the *two-body*
    reference number and not the ATM-inclusive one."""
    nums = [18, 18, 18]
    pos = _cluster(nums, seed=11)
    mol = _mol(nums, pos)

    got = vq.compute_d3bj(
        mol, vq.D3BJParams(**_HSE3C), backend="dftd3"
    ).energy
    ref_two_body = _dftd3_energy(nums, pos, s9=0.0, **_HSE3C)
    ref_with_atm = _dftd3_energy(nums, pos, s9=1.0, **_HSE3C)

    assert ref_with_atm != ref_two_body, "test geometry has no ATM signal"
    assert got == pytest.approx(ref_two_body, rel=0, abs=0)


@requires_dftd3
def test_dftd3_backend_honours_s9_when_requested():
    nums = [18, 18, 18]
    pos = _cluster(nums, seed=12)
    mol = _mol(nums, pos)
    got = vq.compute_d3bj(
        mol, vq.D3BJParams(**_HSE3C, s9=1.0), backend="dftd3"
    ).energy
    assert got == pytest.approx(
        _dftd3_energy(nums, pos, s9=1.0, **_HSE3C), rel=0, abs=0
    )


@requires_dftd3
def test_backends_agree_on_atm_so_auto_routing_is_physics_neutral():
    """The whole point: which backend answers must not change the method."""
    nums = [10] * 4
    pos = _cluster(nums, seed=21)
    mol = _mol(nums, pos)
    p = vq.D3BJParams(**_HSE3C, s9=1.0)
    e_builtin = vq.compute_d3bj(mol, p, backend="builtin").energy
    e_dftd3 = vq.compute_d3bj(mol, p, backend="dftd3").energy
    assert e_builtin == pytest.approx(e_dftd3, rel=0, abs=1e-12)


# ---------------------------------------------------------------------------
# Analytic ATM gradient
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nums", [[18, 18, 18], [2, 2, 2, 2], [10] * 5])
def test_atm_gradient_matches_finite_differences(nums):
    """The complete analytic ATM gradient reproduces finite differences."""
    pos = _cluster(nums, seed=5 + len(nums))
    damping = dict(s6=1.0, s8=0.9, a1=0.4, a2=4.5)

    def grad(s9):
        res = vq.compute_d3bj(
            _mol(nums, pos), vq.D3BJParams(**damping, s9=s9),
            backend="builtin", with_gradient=True,
        )
        return np.asarray(res.gradient, dtype=float)

    def energy_atm(p):
        return (_builtin_energy(nums, p, s9=1.0, **damping)
                - _builtin_energy(nums, p, s9=0.0, **damping))

    analytic = grad(1.0) - grad(0.0)

    h = 1e-5
    fd = np.zeros_like(analytic)
    for i in range(len(nums)):
        for k in range(3):
            plus = pos.copy(); plus[i, k] += h
            minus = pos.copy(); minus[i, k] -= h
            fd[i, k] = (energy_atm(plus) - energy_atm(minus)) / (2 * h)

    assert np.max(np.abs(analytic - fd)) < 1e-9
    assert np.max(np.abs(analytic)) > 1e-9, "ATM gradient is trivially zero"


def test_atm_gradient_is_translationally_invariant():
    nums = [18, 18, 18, 18]
    pos = _cluster(nums, seed=9)
    res = vq.compute_d3bj(
        _mol(nums, pos), vq.D3BJParams(**_HSE3C, s9=1.0),
        backend="builtin", with_gradient=True,
    )
    g = np.asarray(res.gradient, dtype=float)
    assert np.allclose(g.sum(axis=0), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Composite recipes
# ---------------------------------------------------------------------------

def test_pbeh3c_and_hse3c_request_the_three_body_term():
    """Both papers define the method to include the three-body C9 term:
    PBEh-3c -- "the three-body dispersion is always included in the PBEh-3c
    method" (Grimme et al., JCP 143, 054107 (2015)); HSE-3c -- ESI Sec. A.
    """
    for name in ("pbeh-3c", "hse-3c"):
        recipe = vq.resolve_composite(name)
        assert recipe.d3bj_damping is not None, name
        assert recipe.d3bj_damping.s9 == 1.0, name


def test_composites_without_published_atm_stay_two_body():
    """HF-3c's paper never mentions ATM; B97-3c / B3LYP-3c carry no
    published three-body scale either. Don't invent one."""
    for name in ("hf-3c", "b97-3c", "b3lyp-3c"):
        recipe = vq.resolve_composite(name)
        assert recipe.d3bj_damping is not None, name
        assert recipe.d3bj_damping.s9 == 0.0, name


def test_hse3c_uses_its_own_damping_not_pbeh3c_s():
    """Regression: hse-3c used to inherit PBEh-3c's (a1, a2) = (0.4860,
    4.5000). The larger a1 widens the BJ damping radius, so those values
    under-bind benzene's D3 energy by ~0.97 mHa. HSE-3c's own re-fit set
    reproduces the ESI's Table S2 gas-phase benzene number."""
    hse = vq.resolve_composite("hse-3c").d3bj_damping
    pbeh = vq.resolve_composite("pbeh-3c").d3bj_damping
    assert (hse.s6, hse.s8, hse.a1, hse.a2, hse.s9) == (
        1.0, 0.0, 0.4411, 4.5182, 1.0
    )
    assert (hse.a1, hse.a2) != (pbeh.a1, pbeh.a2)


@pytest.mark.parametrize("name", ["pbeh-3c", "hse-3c"])
def test_composite_resolution_propagates_s9_into_d3bj_params(name):
    """run_job(method='pbeh-3c') must hand the ATM scale to compute_d3bj."""
    from vibeqc.runner import _apply_composite

    mol = _mol([1, 1], [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    _, _, _, dispersion, recipe = _apply_composite(
        name, basis=None, functional=None, dispersion=None, molecule=mol,
    )
    assert isinstance(dispersion, vq.D3BJParams)
    assert dispersion.s9 == 1.0
    assert recipe.name == name


def test_composite_resolution_keeps_two_body_recipes_at_s9_zero():
    from vibeqc.runner import _apply_composite

    mol = _mol([1, 1], [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    _, _, _, dispersion, _ = _apply_composite(
        "hf-3c", basis=None, functional=None, dispersion=None, molecule=mol,
    )
    assert isinstance(dispersion, vq.D3BJParams)
    assert dispersion.s9 == 0.0


# ---------------------------------------------------------------------------
# Benzene: the published magnitude of the ATM term
# ---------------------------------------------------------------------------

def _benzene_bohr() -> tuple[list[int], np.ndarray]:
    """D6h benzene, r(CC) = 1.3915 A, r(CH) = 1.0800 A."""
    ang_to_bohr = 1.8897261254578281
    r_c, r_h = 1.3915, 2.4715
    numbers, coords = [], []
    for i in range(6):
        theta = np.pi / 3.0 * i
        numbers.append(6)
        coords.append([r_c * np.cos(theta), r_c * np.sin(theta), 0.0])
        numbers.append(1)
        coords.append([r_h * np.cos(theta), r_h * np.sin(theta), 0.0])
    return numbers, np.asarray(coords) * ang_to_bohr


@requires_dftd3
def test_benzene_hse3c_atm_magnitude():
    """The HSE-3c ESI's own numerical example. With HSE-3c damping, the ATM
    term on gas-phase benzene is a small *repulsive* shift of ~+0.0094 mHa.

    That is only ~0.14% of the two-body energy here -- well under the
    "2%-3% of the two-body dispersion energy" that Grimme et al. (2015)
    quote as typical, because HSE-3c sets s8 = 0 (so the two-body
    reference is C6-only) and benzene is a small, flat, sparse molecule.
    The term is repulsive and grows with density, so it is the condensed
    phase (the ESI's benzene *crystal*) where it earns its place.
    """
    numbers, pos = _benzene_bohr()
    e_2body = _dftd3_energy(numbers, pos, s9=0.0, **_HSE3C)
    e_atm = _dftd3_energy(numbers, pos, s9=1.0, **_HSE3C) - e_2body

    assert e_atm > 0.0, "ATM is repulsive for benzene"
    assert e_atm == pytest.approx(9.44e-6, rel=0.02)
    assert abs(e_atm / e_2body) == pytest.approx(1.4e-3, rel=0.1)


@requires_dftd3
def test_benzene_old_hse3c_damping_under_binds():
    """Pins the direction and size of the damping bug that was fixed: the
    inherited PBEh-3c (a1, a2) has a larger a1, hence a wider BJ damping
    radius, hence a *weaker* (less negative) dispersion energy."""
    numbers, pos = _benzene_bohr()
    e_wrong = _dftd3_energy(numbers, pos, s9=1.0,
                            s6=1.0, s8=0.0, a1=0.4860, a2=4.5000)
    e_right = _dftd3_energy(numbers, pos, s9=1.0, **_HSE3C)

    assert e_wrong > e_right, "old params must under-bind, not over-bind"
    assert (e_wrong - e_right) == pytest.approx(9.7e-4, rel=0.02)


@requires_dftd3
def test_benzene_hse3c_recipe_energy_matches_reference():
    """End-to-end: the hse-3c recipe's damping, driven through vibe-qc's
    public compute_d3bj, equals Grimme's reference for the same (s6, s8,
    a1, a2, s9)."""
    numbers, pos = _benzene_bohr()
    damping = vq.resolve_composite("hse-3c").d3bj_damping
    params = vq.D3BJParams(
        s6=damping.s6, s8=damping.s8, a1=damping.a1, a2=damping.a2,
        s9=damping.s9,
    )
    got = vq.compute_d3bj(_mol(numbers, pos), params, backend="dftd3").energy
    ref = _dftd3_energy(
        numbers, pos, s9=1.0,
        s6=damping.s6, s8=damping.s8, a1=damping.a1, a2=damping.a2,
    )
    assert got == pytest.approx(ref, rel=0, abs=0)
