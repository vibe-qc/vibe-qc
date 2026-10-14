"""The shared translation-covariant lattice-summed shell gamma kernel.

Covers ``cpp/src/semiempirical/core/periodic_gamma.cpp``: the Elstner 1998
Eq. 17/18 short-range function against the Appendix closed forms, the
Klopman-Ohno remainder, the 1-D/2-D/3-D Ewald lattice potential against the
textbook Madelung constants, and the record-driven assembly's translation
covariance, gradient and strain derivative against finite differences.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from vibeqc._vibeqc_core import Atom, Molecule
from vibeqc._vibeqc_core import semiempirical as _se

KO = _se.ShellGammaForm.KlopmanOhno
ELSTNER = _se.ShellGammaForm.Elstner


def _spec(form, average=_se.KlopmanOhnoAverage.HardnessMean):
    spec = _se.ShellGammaSpec()
    spec.form = form
    spec.ko_average = average
    return spec


# ---------------------------------------------------------------------------
# Scalar kernels
# ---------------------------------------------------------------------------


def _elstner_equal(tau, R):
    return np.exp(-tau * R) * (
        1.0 / R + 11.0 * tau / 16.0 + 3.0 * tau * tau * R / 16.0
        + tau ** 3 * R * R / 48.0
    )


def _elstner_unequal(ta, tb, R):
    def half(x, y):
        D = x * x - y * y
        return np.exp(-x * R) * (
            y ** 4 * x / (2.0 * D * D) - (y ** 6 - 3.0 * y ** 4 * x * x) / (R * D ** 3)
        )

    return half(ta, tb) + half(tb, ta)


@pytest.mark.parametrize("tau,R", [(1.28, 1.4), (2.0, 3.0), (0.9, 6.0)])
def test_elstner_equal_tau_matches_appendix_closed_form(tau, R):
    """Elstner et al. PRB 58, 7260 (1998), Eq. 18 with tau_a = tau_b."""
    assert _se.elstner_short_range(tau, tau, R) == pytest.approx(
        _elstner_equal(tau, R), rel=0.0, abs=1e-15
    )


@pytest.mark.parametrize("ta,tb,R", [(1.5, 0.7, 1.4), (2.4, 1.1, 3.0), (0.9, 1.6, 0.8)])
def test_elstner_unequal_tau_matches_appendix_closed_form(ta, tb, R):
    assert _se.elstner_short_range(ta, tb, R) == pytest.approx(
        _elstner_unequal(ta, tb, R), rel=1e-13, abs=0.0
    )
    assert _se.elstner_short_range(tb, ta, R) == _se.elstner_short_range(ta, tb, R)


def test_elstner_short_range_is_continuous_across_the_degenerate_window():
    """The delta^4 series and the closed form agree to 1e-11 at the switch."""
    tau, R = 1.6, 1.1
    for rel in (0.0199, 0.02, 0.0201, 0.05):
        d = rel * tau
        series_or_exact = _se.elstner_short_range(tau + d, tau - d, R)
        assert series_or_exact == pytest.approx(
            _elstner_unequal(tau + d, tau - d, R), rel=0.0, abs=2e-11
        )
    # Deep inside the window the closed form is numerically unusable
    # (catastrophic cancellation) but the kernel remains smooth.
    values = [
        _se.elstner_short_range(tau * (1 + r), tau * (1 - r), R)
        for r in (1e-7, 1e-5, 1e-3)
    ]
    assert values[0] == pytest.approx(_elstner_equal(tau, R), rel=0.0, abs=1e-13)
    assert values[2] > values[0]
    assert values[2] - values[0] < 1e-5


def test_elstner_gamma_reaches_the_hubbard_limit_at_zero_distance():
    """Eq. 18: S -> 5 tau/16 + 1/R as R -> 0, i.e. gamma(0) = U = 5 tau/16."""
    tau = 1.28
    R = 1e-6
    assert 1.0 / R - _se.elstner_short_range(tau, tau, R) == pytest.approx(
        5.0 * tau / 16.0, rel=0.0, abs=1e-6
    )
    ta, tb = 1.5, 0.7
    limit = 0.5 * (ta * tb / (ta + tb) + (ta * tb) ** 2 / (ta + tb) ** 3)
    assert 1.0 / R - _se.elstner_short_range(ta, tb, R) == pytest.approx(
        limit, rel=0.0, abs=1e-6
    )


@pytest.mark.parametrize("ta,tb", [(1.28, 1.28), (1.5, 0.7), (1.6, 1.6 * 1.01)])
def test_elstner_short_range_derivative_matches_finite_difference(ta, tb):
    for R in (0.7, 1.4, 4.0):
        h = 1e-5
        fd = (
            _se.elstner_short_range(ta, tb, R + h)
            - _se.elstner_short_range(ta, tb, R - h)
        ) / (2.0 * h)
        assert _se.elstner_short_range_derivative(ta, tb, R) == pytest.approx(
            fd, rel=1e-8, abs=1e-12
        )


def test_klopman_ohno_pair_kernel_reproduces_both_average_conventions():
    Ua, Ub, R = 0.405771, 0.583349, 2.3
    gfn2 = _spec(KO, _se.KlopmanOhnoAverage.HardnessMean)
    dftb = _spec(KO, _se.KlopmanOhnoAverage.InverseHardnessMean)
    # GFN2 (build_gfn2_shell_gamma): 1/sqrt(R^2 + 1/gab^2), gab = (Ua+Ub)/2.
    gab = 0.5 * (Ua + Ub)
    assert _se.shell_gamma_pair(gfn2, Ua, Ub, R) == pytest.approx(
        1.0 / np.sqrt(R * R + 1.0 / gab ** 2), rel=1e-15
    )
    assert _se.shell_gamma_onsite(gfn2, Ua, Ub) == pytest.approx(gab, rel=1e-15)
    # SCC-DFTB (gamma_matrix): 1/sqrt(R^2 + eta^2), eta = (1/Ua + 1/Ub)/2.
    eta = 0.5 * (1.0 / Ua + 1.0 / Ub)
    assert _se.shell_gamma_pair(dftb, Ua, Ub, R) == pytest.approx(
        1.0 / np.sqrt(R * R + eta * eta), rel=1e-15
    )
    assert _se.shell_gamma_onsite(dftb, Ua, Ua) == pytest.approx(Ua, rel=1e-15)
    for spec in (gfn2, dftb):
        assert _se.shell_gamma_remainder(spec, Ua, Ub, R) == pytest.approx(
            _se.shell_gamma_pair(spec, Ua, Ub, R) - 1.0 / R, rel=1e-12
        )


@pytest.mark.parametrize("form", [KO, ELSTNER])
def test_pair_and_remainder_derivatives_match_finite_difference(form):
    spec = _spec(form)
    Ua, Ub = 0.405771, 0.583349
    for R in (0.9, 2.3, 7.0):
        h = 1e-5
        fd_pair = (
            _se.shell_gamma_pair(spec, Ua, Ub, R + h)
            - _se.shell_gamma_pair(spec, Ua, Ub, R - h)
        ) / (2 * h)
        fd_rem = (
            _se.shell_gamma_remainder(spec, Ua, Ub, R + h)
            - _se.shell_gamma_remainder(spec, Ua, Ub, R - h)
        ) / (2 * h)
        assert _se.shell_gamma_pair_derivative(spec, Ua, Ub, R) == pytest.approx(
            fd_pair, rel=1e-7, abs=1e-12
        )
        assert _se.shell_gamma_remainder_derivative(spec, Ua, Ub, R) == pytest.approx(
            fd_rem, rel=1e-7, abs=1e-12
        )


def test_elstner_remainder_decays_exponentially_while_klopman_ohno_does_not():
    """The property that gives the Elstner form a thermodynamic limit (#444)."""
    Ua, Ub = 0.405771, 0.583349
    ko = [abs(_se.shell_gamma_remainder(_spec(KO), Ua, Ub, R)) for R in (10.0, 20.0)]
    el = [abs(_se.shell_gamma_remainder(_spec(ELSTNER), Ua, Ub, R)) for R in (10.0, 20.0)]
    assert ko[1] / ko[0] > 0.1  # ~ R^-3: 1/8
    assert el[1] / el[0] < 1e-4  # ~ e^{-tau R}


# ---------------------------------------------------------------------------
# Ewald lattice potential
# ---------------------------------------------------------------------------

MADELUNG_NACL_3D = 1.7475645946331822
MADELUNG_SQUARE_2D = 1.6155426267128247
MADELUNG_CHAIN_1D = 2.0 * np.log(2.0)


def _pair_energy(kernel, positions, charges):
    energy = 0.0
    for a, b in itertools.product(range(len(positions)), repeat=2):
        d = positions[b] - positions[a]
        energy += 0.5 * charges[a] * charges[b] * kernel.potential(d, a == b)
    return energy


def test_ewald_3d_reproduces_the_rocksalt_madelung_constant():
    a = 5.0  # conventional cube edge; nearest-neighbour distance a/2
    translations = np.array([[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]])
    positions = [np.zeros(3), np.array([a / 2, 0.0, 0.0])]
    for alpha in (0.0, 0.7, 1.4):
        kernel = _se.EwaldCoulombKernel(translations, alpha)
        assert kernel.dimension == 3
        energy = _pair_energy(kernel, positions, [1.0, -1.0])
        assert energy == pytest.approx(-MADELUNG_NACL_3D / (a / 2), rel=0.0, abs=1e-10)


def test_ewald_2d_reproduces_the_square_lattice_madelung_constant():
    r0 = 2.0
    translations = np.array([[2 * r0, 0.0, 0.0], [0.0, 2 * r0, 0.0]])
    positions = [np.zeros(3), np.array([r0, 0.0, 0.0]),
                 np.array([0.0, r0, 0.0]), np.array([r0, r0, 0.0])]
    charges = [1.0, -1.0, -1.0, 1.0]
    for alpha in (0.0, 0.5):
        kernel = _se.EwaldCoulombKernel(translations, alpha)
        energy = _pair_energy(kernel, positions, charges) / 2.0  # per ion pair
        assert energy == pytest.approx(-MADELUNG_SQUARE_2D / r0, rel=0.0, abs=1e-9)


def test_ewald_1d_reproduces_the_alternating_chain_madelung_constant():
    r0 = 1.7
    translations = np.array([[2 * r0, 0.0, 0.0]])
    positions = [np.zeros(3), np.array([r0, 0.0, 0.0])]
    for alpha in (0.0, 1.0):
        kernel = _se.EwaldCoulombKernel(translations, alpha)
        energy = _pair_energy(kernel, positions, [1.0, -1.0])
        assert energy == pytest.approx(-MADELUNG_CHAIN_1D / r0, rel=0.0, abs=1e-9)


@pytest.mark.parametrize(
    "translations",
    [
        np.array([[3.1, 0.2, -0.4]]),
        np.array([[3.1, 0.2, -0.4], [0.5, 2.9, 0.3]]),
        np.array([[3.1, 0.2, -0.4], [0.5, 2.9, 0.3], [-0.2, 0.4, 3.4]]),
    ],
    ids=["1d", "2d", "3d"],
)
def test_ewald_potential_is_lattice_periodic_and_even(translations):
    kernel = _se.EwaldCoulombKernel(translations)
    rng = np.random.default_rng(7)
    for _ in range(4):
        d = rng.normal(size=3) * 1.5
        shift = sum(int(n) * translations[i] for i, n in enumerate(rng.integers(-3, 4, size=len(translations))))
        base = kernel.potential(d)
        assert kernel.potential(d + shift) == pytest.approx(base, rel=0.0, abs=1e-12)
        assert kernel.potential(-d) == pytest.approx(base, rel=0.0, abs=1e-12)
        grad = kernel.gradient(d)
        np.testing.assert_allclose(kernel.gradient(d + shift), grad, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(kernel.gradient(-d), -grad, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize(
    "translations",
    [
        np.array([[3.1, 0.2, -0.4]]),
        np.array([[3.1, 0.2, -0.4], [0.5, 2.9, 0.3]]),
        np.array([[3.1, 0.2, -0.4], [0.5, 2.9, 0.3], [-0.2, 0.4, 3.4]]),
    ],
    ids=["1d", "2d", "3d"],
)
def test_ewald_gradient_matches_finite_difference(translations):
    kernel = _se.EwaldCoulombKernel(translations)
    rng = np.random.default_rng(11)
    for _ in range(3):
        d = rng.normal(size=3) * 1.2
        grad = kernel.gradient(d)
        h = 1e-5
        for axis in range(3):
            e = np.zeros(3)
            e[axis] = h
            fd = (kernel.potential(d + e) - kernel.potential(d - e)) / (2 * h)
            assert grad[axis] == pytest.approx(fd, rel=1e-7, abs=1e-9)
    # The same-site lattice self potential is stationary.
    np.testing.assert_allclose(kernel.gradient(np.zeros(3), True), 0.0, atol=1e-13)


# ---------------------------------------------------------------------------
# Record-driven assembly
# ---------------------------------------------------------------------------


def _pair_cutoff_records(positions, translations, cutoff):
    """Every (a, b, g) with |R_b + g - R_a| <= cutoff, the g = 0 self pair excluded."""
    translations = np.asarray(translations, dtype=float)
    dim = len(translations)
    n = len(positions)
    span = max(np.linalg.norm(positions[a] - positions[b]) for a in range(n) for b in range(n))
    dual = np.linalg.solve(translations @ translations.T, translations)
    n_max = [int(np.ceil((cutoff + span) * np.linalg.norm(dual[i]))) + 1 for i in range(dim)]
    ranges = [range(-n_max[i], n_max[i] + 1) for i in range(dim)]
    rec_a, rec_b, shifts, weights = [], [], [], []
    for labels in itertools.product(*ranges):
        g = sum(l * translations[i] for i, l in enumerate(labels))
        zero = all(l == 0 for l in labels)
        for a in range(n):
            for b in range(n):
                if zero and a == b:
                    continue
                if np.linalg.norm(positions[b] + g - positions[a]) <= cutoff:
                    rec_a.append(a)
                    rec_b.append(b)
                    shifts.append(np.asarray(g, dtype=float))
                    weights.append(1.0)
    return rec_a, rec_b, np.asarray(shifts).reshape(-1, 3), weights


def _mgo_like(oxygen_shift=None):
    a = 7.96
    translations = np.array([[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]])
    oxygen = np.array([a / 2 + 0.3, 0.1, -0.2])
    if oxygen_shift is not None:
        oxygen = oxygen + oxygen_shift
    return translations, [np.zeros(3), oxygen]


# Mg: s,p shells; O: s,p shells with distinct hardnesses.
_SITE_ATOM = [0, 0, 1, 1]
_HARDNESS = [0.42, 0.48, 0.55, 0.6]
_DQ = np.array([0.35, -0.55, 0.1, 0.1])


def _gamma(form, translations, positions, cutoff=12.0, alpha=0.0):
    kernel = _se.EwaldCoulombKernel(np.asarray(translations), alpha)
    rec_a, rec_b, shifts, weights = _pair_cutoff_records(positions, translations, cutoff)
    return _se._periodic_shell_gamma_from_records(
        _SITE_ATOM, _HARDNESS, np.asarray(positions), _spec(form), kernel,
        rec_a, rec_b, shifts, weights,
    )


@pytest.mark.parametrize("form", [KO, ELSTNER])
def test_assembled_gamma_is_symmetric_and_translation_covariant(form):
    translations, positions = _mgo_like()
    gamma = _gamma(form, translations, positions)
    np.testing.assert_allclose(gamma, gamma.T, rtol=0.0, atol=1e-13)
    # Relabelling the oxygen by any lattice vector is exactly a no-op.
    for labels in ((1, 0, 0), (0, -1, 0), (2, 1, -1)):
        shift = sum(l * translations[i] for i, l in enumerate(labels))
        _, moved = _mgo_like(shift)
        np.testing.assert_allclose(_gamma(form, translations, moved), gamma, rtol=0.0, atol=1e-11)
    # And alpha-independent.
    np.testing.assert_allclose(
        _gamma(form, translations, positions, alpha=0.9), gamma, rtol=0.0, atol=1e-10
    )


def test_elstner_gamma_converges_in_the_pair_cutoff_while_klopman_ohno_drifts():
    """The #444 defect and its resolution, on one fixed charge pattern."""
    translations, positions = _mgo_like()
    energies = {}
    for form in (KO, ELSTNER):
        energies[form] = [
            0.5 * _DQ @ _gamma(form, translations, positions, cutoff=c) @ _DQ
            for c in (16.0, 24.0, 32.0)
        ]
    # tau = 16/5 U >= 1.34 bohr^-1 here, so e^{-tau R} is below 1e-9 at the
    # first cutoff and the remainder sum is converged from there on.
    el = energies[ELSTNER]
    assert abs(el[2] - el[1]) < 1e-9
    assert abs(el[1] - el[0]) < 1e-8
    ko = energies[KO]
    assert abs(ko[2] - ko[1]) > 1e-6


def test_molecular_limit_of_the_kernel_matches_the_molecular_assembly():
    """A huge cell with a tight cutoff reduces to the free-boundary pair kernel."""
    translations = np.eye(3) * 80.0
    positions = [np.zeros(3), np.array([2.1, 0.4, -0.3])]
    for form in (KO, ELSTNER):
        spec = _spec(form)
        molecular = _se._molecular_shell_gamma(_SITE_ATOM, _HARDNESS, np.asarray(positions), spec)
        kernel = _se.EwaldCoulombKernel(translations)
        rec_a, rec_b, shifts, weights = _pair_cutoff_records(positions, translations, 10.0)
        periodic = _se._periodic_shell_gamma_from_records(
            _SITE_ATOM, _HARDNESS, np.asarray(positions), spec, kernel,
            rec_a, rec_b, shifts, weights,
        )
        # The lattice potential of an isolated pair in an 80-bohr box differs
        # from the free kernel only by the (uniform) image tail, which cancels
        # in any neutral contraction.
        neutral = np.array([0.5, -0.5, 0.25, -0.25])
        assert neutral @ periodic @ neutral == pytest.approx(
            neutral @ molecular @ neutral, rel=0.0, abs=1e-6
        )


@pytest.mark.parametrize("form", [KO, ELSTNER])
@pytest.mark.parametrize(
    "translations",
    [
        np.array([[4.1, 0.3, -0.2]]),
        np.array([[4.1, 0.3, -0.2], [0.4, 4.4, 0.5]]),
        np.array([[0, 3.98, 3.98], [3.98, 0, 3.98], [3.98, 3.98, 0]]),
    ],
    ids=["1d", "2d", "3d"],
)
def test_assembled_gamma_gradient_matches_finite_difference(form, translations):
    positions = [np.array([0.1, -0.2, 0.05]), np.array([2.3, 0.9, -0.4])]
    spec = _spec(form)
    kernel = _se.EwaldCoulombKernel(translations)
    cutoff = 14.0

    def energy(pos):
        rec_a, rec_b, shifts, weights = _pair_cutoff_records(pos, translations, cutoff)
        gamma = _se._periodic_shell_gamma_from_records(
            _SITE_ATOM, _HARDNESS, np.asarray(pos), spec, kernel,
            rec_a, rec_b, shifts, weights,
        )
        return 0.5 * _DQ @ gamma @ _DQ

    rec_a, rec_b, shifts, weights = _pair_cutoff_records(positions, translations, cutoff)
    grad = np.asarray(_se._periodic_shell_gamma_gradient_from_records(
        _SITE_ATOM, _HARDNESS, np.asarray(positions), spec, kernel,
        rec_a, rec_b, shifts, weights, _DQ,
    ))
    h = 1e-5
    for atom in range(2):
        for axis in range(3):
            plus = [p.copy() for p in positions]
            minus = [p.copy() for p in positions]
            plus[atom][axis] += h
            minus[atom][axis] -= h
            fd = (energy(plus) - energy(minus)) / (2 * h)
            assert grad[atom, axis] == pytest.approx(fd, rel=1e-6, abs=1e-9)
    # Newton's third law on the pair.
    np.testing.assert_allclose(grad.sum(axis=0), 0.0, atol=1e-10)


@pytest.mark.parametrize("form", [KO, ELSTNER])
def test_assembled_gamma_strain_derivative_matches_finite_difference(form):
    translations = np.array([[0, 3.98, 3.98], [3.98, 0, 3.98], [3.98, 3.98, 0]]) * 1.05
    positions = [np.array([0.1, -0.2, 0.05]), np.array([2.3, 0.9, -0.4])]
    spec = _spec(form)
    cutoff = 14.0
    kernel = _se.EwaldCoulombKernel(translations)
    rec_a, rec_b, shifts, weights = _pair_cutoff_records(positions, translations, cutoff)

    def energy(transform):
        strained_t = (transform @ translations.T).T
        strained_p = [transform @ p for p in positions]
        strained_s = (transform @ shifts.T).T
        strained_kernel = _se.EwaldCoulombKernel(strained_t, kernel.alpha)
        gamma = _se._periodic_shell_gamma_from_records(
            _SITE_ATOM, _HARDNESS, np.asarray(strained_p), spec, strained_kernel,
            rec_a, rec_b, strained_s, weights,
        )
        return 0.5 * _DQ @ gamma @ _DQ

    strain = np.asarray(_se._periodic_shell_gamma_strain_from_records(
        _SITE_ATOM, _HARDNESS, np.asarray(positions), spec, kernel,
        rec_a, rec_b, shifts, weights, _DQ,
    ))
    h = 1e-5
    for i in range(3):
        for j in range(3):
            eps = np.zeros((3, 3))
            eps[i, j] = h
            fd = (energy(np.eye(3) + eps) - energy(np.eye(3) - eps)) / (2 * h)
            assert strain[i, j] == pytest.approx(fd, rel=1e-6, abs=1e-9)


def test_molecular_assembly_reproduces_the_shipped_gfn2_shell_gamma():
    """KO/HardnessMean on GFN2 shells is build_gfn2_shell_gamma bit-for-bit."""
    pytest.importorskip("vibeqc.semiempirical.methods.gfn2_params")
    from vibeqc.semiempirical.methods.gfn2_params import (
        gfn2_parameter_cache_available,
        load_gfn2_params,
    )

    if not gfn2_parameter_cache_available():
        pytest.skip("GFN2 parameter cache unavailable")
    params = load_gfn2_params()
    mol = Molecule(
        [Atom(8, [0.0, 0.0, 0.0]), Atom(1, [1.43, 1.11, 0.0]), Atom(1, [-1.43, 1.11, 0.0])],
        0, 1,
    )
    basis = _se.SemiempiricalBasis.build(mol, params, 0)
    shells = _se.gfn2_enumerate_shells(basis, mol, params)
    shipped = np.asarray(_se.build_gfn2_shell_gamma(shells, mol, params))
    positions = np.array([atom.xyz for atom in mol.atoms])
    ours = np.asarray(_se._molecular_shell_gamma(
        [s.atom_idx for s in shells], [s.hardness for s in shells], positions, _spec(KO),
    ))
    # The shipped on-site block is 1/sqrt(1/gab^2), one ulp from 1/(1/gab).
    np.testing.assert_allclose(ours, shipped, rtol=0.0, atol=2e-16)


# ---------------------------------------------------------------------------
# External anchor: xtb's periodic GFN1-xTB Coulomb matrix (#245)
# ---------------------------------------------------------------------------

# Converged GFN1-xTB shell charges of the 64-atom conventional MgO cell
# (a = 4.212 A, atoms in the order of _mgo_conventional_2x2x2 below; two
# shells per atom, s then p), read from xtb 6.7.0's xtbrestart after an SCC
# converged to 2e-8 e with the Ewald tolerance set to 1e-14.  Only the
# quadratic form 0.5 qsh^T J qsh is compared, so the charges are inputs, not
# something vibe-qc has to reproduce.
_XTB_MGO64_QSH = np.array([
    1.6408725534422306, -0.6949241476534399, 0.19366898254555817,
    -1.1395139584730398, 1.6412651687424011, -0.6954763247296132,
    0.19366849756427781, -1.1394485049560956, 1.641265168742403,
    -0.6954763247296146, 0.1936682618534974, -1.139480927877689,
    1.6412651687423996, -0.6954763247296121, 0.19366826185349562,
    -1.139480927877687, 1.641069115185367, -0.6952004331994002,
    0.19366826185349628, -1.1394809278776852, 1.6410691151853676,
    -0.6952004331994013, 0.19366826185349856, -1.1394809278776872,
    1.6410691151853676, -0.6952004331994021, 0.19366898254555895,
    -1.1395139584730427, 1.6414607129335177, -0.6957518219578049,
    0.19366849756427634, -1.1394485049560956, 1.6410691151853665,
    -0.6952004331994, 0.19366826185349567, -1.1394809278776867,
    1.6410691151853656, -0.6952004331994, 0.19366826185349684,
    -1.1394809278776854, 1.6414607129335197, -0.6957518219578065,
    0.1936684975642773, -1.1394485049560976, 1.6410691151853682,
    -0.6952004331994021, 0.19366898254555678, -1.1395139584730416,
    1.6412651687423998, -0.6954763247296122, 0.1936684975642761,
    -1.1394485049560945, 1.6408725534422302, -0.69492414765344,
    0.193668982545559, -1.1395139584730398, 1.6412651687424005,
    -0.6954763247296121, 0.19366826185349728, -1.1394809278776874,
    1.6412651687424011, -0.6954763247296131, 0.1936682618534983,
    -1.139480927877688, 1.64106911518537, -0.6952004331994012,
    0.19367066862629606, -1.1395476098080626, 1.641460712933519,
    -0.6957518219578048, 0.1936682618534972, -1.1394809278776874,
    1.6410691151853674, -0.6952004331994015, 0.19366898254555717,
    -1.1395139584730405, 1.6410691151853678, -0.6952004331994004,
    0.19366898254555737, -1.1395139584730427, 1.6412651687423985,
    -0.6954763247296104, 0.19366898254555867, -1.1395139584730436,
    1.6412651687424007, -0.6954763247296136, 0.19366898254555845,
    -1.1395139584730418, 1.6408725534422293, -0.6949241476534398,
    0.193670668626297, -1.139547609808063, 1.6412651687424,
    -0.6954763247296125, 0.19366826185349542, -1.1394809278776887,
    1.6412651687423994, -0.69547632472961, 0.1936689825455604,
    -1.1395139584730427, 1.6412651687424, -0.6954763247296111,
    0.19366898254556017, -1.1395139584730432, 1.6412651687424005,
    -0.6954763247296135, 0.1936682618534969, -1.1394809278776892,
    1.6408725534422308, -0.6949241476534419, 0.1936706686262961,
    -1.1395476098080664, 1.6414607129335215, -0.6957518219578058,
    0.19366826185349525, -1.1394809278776867, 1.6410691151853685,
    -0.6952004331994014, 0.19367066862629628, -1.1395476098080661,
    1.6410691151853638, -0.695200433199398, 0.1936689825455569,
    -1.1395139584730403, 1.6410691151853662, -0.6952004331993998,
    0.19366898254555895, -1.1395139584730434,
])
# 0.5 qsh^T J qsh with xtb's own Coulomb matrix at that tolerance, evaluated
# from the matrix xtb dumped (studies/probe_periodic_gfn1_coulomb_anchor.py
# documents the dump); xtb prints electrostatic 3.960220733566 Eh, of which
# 0.997398261248 Eh is the atomic third-order term.
_XTB_MGO64_SECOND_ORDER = 2.9628224723181216
_XTB_MGO64_ALPHA = 0.15859712
_XTB_MGO64_R_CUT = 32.54779904
_XTB_MGO64_TOLERANCE = 1.0e-14

# Isolated MgO diatomic at 3.5 bohr, same provenance: xtb prints
# electrostatic 0.080549134921 Eh; minus the third-order 0.006487670743 Eh.
_XTB_MGO_DIATOMIC_QSH = np.array([
    1.033001583894995, -0.4724668055543587,
    0.2122868267380975, -0.77282160507872,
])
_XTB_MGO_DIATOMIC_SECOND_ORDER = 0.080549134921 - 0.006487670743

# GFN1-xTB per-shell hardnesses U_Z (1 + kappa_l), xtb src/xtb/gfn1.f90
# (chemicalHardness, shellHardness); Gamma_A is not needed here.
_GFN1_SHELL_HARDNESS = {
    8: (0.583349, 0.583349 * (1.0 + 0.0374608)),
    12: (0.354151, 0.354151 * (1.0 + 0.0674819)),
}


def _mgo_conventional_2x2x2(a_angstrom: float = 4.212):
    """64-atom rocksalt cell as xtb saw it: fractional coordinates wrapped
    into [0, 1) with xtb's 1e-14 epsilon, so the tied Wigner-Seitz images
    come out exactly as in its run."""
    bohr = 1.8897261246257702
    edge = 2 * a_angstrom * bohr
    fcc = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    coords, Z = [], []
    for i, j, k in itertools.product(range(2), repeat=3):
        for site in fcc:
            base = (np.array([i, j, k]) + site) * a_angstrom * bohr
            coords.append(base)
            Z.append(12)
            coords.append(base + np.array([a_angstrom / 2 * bohr, 0.0, 0.0]))
            Z.append(8)
    coords = np.array(coords)
    frac = coords / edge
    frac = np.where(np.abs(frac - 1.0) < 1.0e-14, frac - 1.0, frac)
    frac = np.where(frac > 1.0 + 1.0e-14, frac - np.floor(frac), frac)
    return frac * edge, Z, np.eye(3) * edge


def _xtb_wigner_seitz_records(coords, lattice, alpha, r_cut):
    """xtb's real-space set (klopmanohno.F90 getCoulombMatrixPBC3D with a
    TWignerSeitzCell generated non-symmetrically): for every pair (a, b >= a)
    the closest images of b around a within r_cut (ties within 0.01 bohr^2
    share the weight), each combined with every lattice translation of
    length <= r_cut.  Returns the directed record arrays and the erfc
    deficit: xtb sums erfc(alpha r)/r over this weighted set instead of the
    whole lattice, so images reachable from fewer than all tied images are
    under-weighted; the shared kernel's Ewald potential carries the whole
    lattice and that difference has to be removed before comparing."""
    from scipy.special import erfc

    def points(cutoff):
        tmax = int(math.ceil(cutoff / float(lattice[0, 0])))
        out = []
        for n in itertools.product(range(-tmax, tmax + 1), repeat=3):
            T = lattice @ np.asarray(n, float)
            d2 = float(T @ T)
            if d2 <= cutoff ** 2 + 1.0e-10:
                out.append((d2, T))
        out.sort(key=lambda p: p[0])
        return np.array([p[1] for p in out])

    r_trans = points(r_cut)
    n = len(coords)
    lists = {a: [] for a in range(n)}
    for i in range(n):
        for j in range(i + 1):
            candidates = []
            for k, T in enumerate(r_trans):
                d2 = float(np.sum((coords[i] + T - coords[j]) ** 2))
                if d2 > r_cut ** 2 or (d2 < 1.0e-4 and k == 0 and i == j):
                    continue
                candidates.append((d2, k))
            candidates.sort(key=lambda c: c[0])
            keep = [c for c in candidates if c[0] <= candidates[0][0] + 0.01 ** 2]
            for _d2, k in keep:
                lists[j].append((i, 1.0 / len(keep), coords[i] + r_trans[k]))
    rec_a, rec_b, shifts, weights = [], [], [], []
    for a, entries in lists.items():
        for b, w, image in entries:
            vec = image - coords[a]
            for T in r_trans:
                d = vec + T
                if float(d @ d) < 1.0e-18:
                    continue
                shift = d - (coords[b] - coords[a])
                rec_a.append(a)
                rec_b.append(b)
                shifts.append(shift)
                weights.append(w)
                if a != b:
                    rec_a.append(b)
                    rec_b.append(a)
                    shifts.append(-shift)
                    weights.append(w)
    big = points(r_cut + 2.0 * float(lattice[0, 0]) + 5.0)
    deficit = np.zeros((n, n))
    for a in range(n):
        for b in range(a, n):
            r = np.linalg.norm((coords[b] - coords[a]) + big, axis=1)
            full = float(np.sum(erfc(alpha * r[r > 1.0e-9]) / r[r > 1.0e-9]))
            partial = 0.0
            for bb, w, image in lists[a]:
                if bb != b:
                    continue
                rr = np.linalg.norm((image - coords[a]) + r_trans, axis=1)
                partial += w * float(np.sum(erfc(alpha * rr[rr > 1.0e-9]) / rr[rr > 1.0e-9]))
            deficit[a, b] = deficit[b, a] = full - partial
    return rec_a, rec_b, np.asarray(shifts).reshape(-1, 3), weights, deficit, len(r_trans)


def test_klopman_ohno_kernel_reproduces_xtb_periodic_gfn1_coulomb_energy():
    """The lane's external anchor (#245): the shared kernel against xtb.

    xtb's periodic GFN1-xTB Coulomb matrix is the same Ewald-split
    Klopman-Ohno construction as build_periodic_shell_gamma with the
    harmonic hardness average and gExp = 2, so on the same cell, the same
    per-shell hardnesses and the same real-space truncation the two must
    agree to their Ewald convergence.  Measured 2026-09-24 against xtb 6.7.0
    with its Ewald tolerance at 1e-14: element-wise 2.5e-13 Ha over the
    128 x 128 matrix and 9e-14 Eh on the quadratic form, once xtb's
    under-weighted erfc sum over tied Wigner-Seitz images (7.9e-8 per
    element here; 5.9e-3 at xtb's default tolerance 1e-8) is removed.  The
    tolerance below is the one fixed on the issue before measuring.
    """
    coords, Z, lattice = _mgo_conventional_2x2x2()
    site_atom, hardness = [], []
    for a, z in enumerate(Z):
        for U in _GFN1_SHELL_HARDNESS[z]:
            site_atom.append(a)
            hardness.append(U)
    qsh = _XTB_MGO64_QSH
    assert len(qsh) == len(hardness) == 128
    assert abs(float(qsh.sum())) < 1.0e-8

    rec_a, rec_b, shifts, weights, deficit, n_trans = _xtb_wigner_seitz_records(
        coords, lattice, _XTB_MGO64_ALPHA, _XTB_MGO64_R_CUT
    )
    assert n_trans == 33
    spec = _spec(KO, _se.KlopmanOhnoAverage.InverseHardnessMean)
    deficit_sites = deficit[np.ix_(site_atom, site_atom)]
    energies = {}
    for alpha in (0.0, _XTB_MGO64_ALPHA, 2.0 * _XTB_MGO64_ALPHA):
        kernel = _se.EwaldCoulombKernel(lattice.copy(), alpha)
        gamma = _se._periodic_shell_gamma_from_records(
            site_atom, hardness, coords, spec, kernel, rec_a, rec_b, shifts, weights
        )
        np.testing.assert_allclose(gamma, gamma.T, rtol=0.0, atol=1.0e-13)
        energies[alpha] = 0.5 * qsh @ (gamma - deficit_sites) @ qsh
    # The vibe-qc side must not move with its own Ewald splitting parameter.
    values = list(energies.values())
    assert max(values) - min(values) < 1.0e-9
    # The anchor itself, at the tolerance fixed before measuring.
    for value in values:
        assert value == pytest.approx(_XTB_MGO64_SECOND_ORDER, abs=1.0e-8)


def test_klopman_ohno_kernel_reproduces_xtb_gfn1_coulomb_energy_of_a_molecule():
    """Molecular limit of the same anchor: tables, harmonic average and the
    on-site rule, through build_molecular_shell_gamma (xtb's cluster
    Coulomb matrix); agrees to 5e-13 Eh."""
    coords = np.array([[0.0, 0.0, 0.0], [3.5, 0.0, 0.0]])
    site_atom, hardness = [], []
    for a, z in enumerate((12, 8)):
        for U in _GFN1_SHELL_HARDNESS[z]:
            site_atom.append(a)
            hardness.append(U)
    spec = _spec(KO, _se.KlopmanOhnoAverage.InverseHardnessMean)
    gamma = _se._molecular_shell_gamma(site_atom, hardness, coords, spec)
    qsh = _XTB_MGO_DIATOMIC_QSH
    assert 0.5 * qsh @ gamma @ qsh == pytest.approx(
        _XTB_MGO_DIATOMIC_SECOND_ORDER, abs=1.0e-10
    )
