"""Conservative 3D shell-pair envelopes and independently integrated tails."""
import math

import numpy as np
import pytest
from scipy.special import gamma, gammaincc

import vibeqc as vq
from vibeqc import manta
from vibeqc import _vibeqc_core as core


def _basis(angular=(0, 2), pure=True, shift=(0., 0., 0.), rotation=None, exponents=(.12, 1.7), coefficients=(.8, -.3)):
    positions = np.array([[.1, -.2, .3], [1.4, .6, -.4]])
    if rotation is not None:
        positions = positions @ rotation.T
    positions += shift
    mol = vq.Molecule([vq.Atom(2, p.tolist()) for p in positions], 0, 1)
    shells = []
    for i, l in enumerate(angular):
        s = core.ShellInfo(); s.atom_index, s.l, s.pure = i % 2, l, pure
        s.origin = positions[i % 2].tolist(); s.exponents, s.coefficients = exponents, coefficients
        shells.append(s)
    return vq.BasisSet(mol, shells, "manta-extents", True)


def _cart(l):
    return np.array([(x, y, l-x-y) for x in range(l, -1, -1) for y in range(l-x, -1, -1)])


def _ao(shell, points):
    # Independent direct polynomial evaluation of the saved shell convention.
    d = points-np.array(shell.origin)
    radial = sum(c*np.exp(-a*np.sum(d*d, axis=1)) for a, c in zip(shell.exponents, shell.coefficients, strict=True))
    values = np.prod(d[:, None, :]**_cart(shell.l)[None, :, :], axis=2)*radial[:, None]
    if shell.pure:
        values = values @ manta.cartesian_to_spherical(shell.l).T
    return values


def _sphere(n=256):
    z = 1-2*(np.arange(n)+.5)/n
    phi = np.arange(n)*math.pi*(3-math.sqrt(5))
    return np.column_stack([np.sqrt(1-z*z)*np.cos(phi), np.sqrt(1-z*z)*np.sin(phi), z])


@pytest.mark.parametrize("l", [0, 1, 3, 6])
@pytest.mark.parametrize("pure", [False, True])
def test_off_axis_signed_angular_shells_and_integrated_tails(l, pure):
    basis = _basis((l, l), pure)
    opts = manta.ExtentOptions(); opts.minimum_radius = opts.maximum_radius = 1.2
    opts.moment_order = 6
    result = manta.compute_pair_extents(basis, opts)
    shells = basis.shells(); directions = _sphere()
    nodes, weights = np.polynomial.legendre.leggauss(80)
    t, w = (nodes+1)/2, weights/2
    for pair in result.pairs:
        r = pair.radius+t/(1-t)
        points = (np.array(pair.center)+r[:, None, None]*directions[None, :, :]).reshape(-1, 3)
        a, b = (_ao(shells[i], points) for i in pair.shells)
        assert np.max(np.max(abs(a), axis=1)*np.max(abs(b), axis=1)) <= pair.pointwise_bound*(1+1e-12)
        radial_weights = w*r*r/(1-t)**2
        for order in [0, 2, 6]:
            quadrature = np.repeat(radial_weights*r**order*(4*math.pi/len(directions)), len(directions))
            absolute_integrals = abs(a).T @ (quadrature[:, None]*abs(b))
            assert absolute_integrals.max() <= pair.tail_moment_bounds[order]*(1+1e-10)
        assert pair.tail_mass_bound == pair.tail_moment_bounds[0]
        assert pair.always_near and pair.reason == "radius-cap"
    assert result.receipt.always_near_pairs == len(result.pairs)


@pytest.mark.parametrize("exponent", [1e-4, .7, 100.])
@pytest.mark.parametrize("order", [0, 6, 12])
def test_s_shell_analytic_norms_tail_moments_and_potential(exponent, order):
    basis = _basis((0,), exponents=[exponent], coefficients=[.9])
    opts = manta.ExtentOptions(); opts.moment_order = order; opts.minimum_radius = 0.
    pair = manta.compute_pair_extents(basis, opts).pairs[0]
    p, coefficient = 2*exponent, .9**2
    mass = coefficient*(math.pi/p)**1.5
    assert pair.l1_norm_bound == pytest.approx(mass, rel=1e-12)
    assert pair.linfinity_bound == pytest.approx(coefficient, rel=1e-12)
    assert pair.potential_bound >= 2*math.pi*coefficient/p
    assert not pair.always_near
    assert pair.pointwise_bound <= opts.pointwise_tolerance and pair.tail_mass_bound <= opts.tail_mass_tolerance
    for k, bound in enumerate(pair.tail_moment_bounds):
        shape = (k+3)/2
        exact = 2*math.pi*coefficient*gamma(shape)*gammaincc(shape,p*pair.radius**2)/p**shape
        assert exact <= bound*(1+1e-12)
        assert bound <= exact*1.2


@pytest.mark.parametrize("pure", [False, True])
def test_potential_bound_covers_independent_libint_attraction(pure):
    basis = _basis((1, 2), pure)
    extents = manta.compute_pair_extents(basis)
    offsets = np.cumsum([0]+[len(_cart(s.l)) if not pure else 2*s.l+1 for s in basis.shells()])
    for pair in extents.pairs:
        i, j = pair.shells
        for direction in [[0., 0., 0.], [.4, -.2, .1], [4., 3., 2.]]:
            position = np.array(pair.center)+direction
            potential = vq.compute_nuclear_with_charges(basis, [position.tolist()], [1.])
            block = potential[offsets[i]:offsets[i+1], offsets[j]:offsets[j+1]]
            assert np.max(abs(block)) <= pair.potential_bound


def test_translation_rotation_tightening_and_owned_output():
    basis = _basis(); opts = manta.ExtentOptions()
    first = manta.compute_pair_extents(basis, opts)
    angle = .7; rotation = np.array([[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    transformed = manta.compute_pair_extents(_basis(rotation=rotation, shift=(.7, -.2, .3)), opts)
    for a, b in zip(first.pairs, transformed.pairs, strict=True):
        np.testing.assert_allclose(np.array(a.center) @ rotation.T+[.7, -.2, .3], b.center, atol=1e-14)
        assert a.radius == pytest.approx(b.radius, rel=1e-13)
        assert a.l1_norm_bound == pytest.approx(b.l1_norm_bound, rel=1e-13)
    opts.tail_mass_tolerance *= 1e-3; opts.pointwise_tolerance *= 1e-3
    tighter = manta.compute_pair_extents(basis, opts)
    assert all(b.radius >= a.radius for a, b in zip(first.pairs, tighter.pairs, strict=True))
    assert first.receipt.source_identity != tighter.receipt.source_identity != transformed.receipt.source_identity
    before = first.pairs[0].tail_moment_bounds
    before[0] = -1.
    assert first.pairs[0].tail_moment_bounds[0] >= 0.
    assert first.receipt.center_policy == "geometric-midpoint"


def test_diffuse_cap_zero_coefficients_and_positive_underflow():
    opts = manta.ExtentOptions(); opts.maximum_radius = 3.
    diffuse = manta.compute_pair_extents(_basis((0,), exponents=[1e-8], coefficients=[1.]), opts).pairs[0]
    assert diffuse.always_near and diffuse.tail_mass_bound > opts.tail_mass_tolerance
    zero = manta.compute_pair_extents(_basis((0,), exponents=[1.], coefficients=[0.]), opts).pairs[0]
    assert not zero.always_near and zero.l1_norm_bound == zero.potential_bound == zero.tail_mass_bound == 0.
    tiny = manta.compute_pair_extents(_basis((0,), exponents=[1.], coefficients=[1e-200]), opts).pairs[0]
    assert 0 < tiny.l1_norm_bound <= np.finfo(float).tiny
    assert 0 < tiny.tail_mass_bound <= np.finfo(float).tiny


@pytest.mark.parametrize("field,plan_field", [("memory_bytes", "modeled_peak_bytes"), ("shell_calls", "shell_pairs"),
    ("primitive_quartets", "primitive_pairs"), ("kernel_terms", "kernel_terms"), ("scalar_terms", "scalar_terms")])
def test_strict_resource_boundaries(field, plan_field):
    basis = _basis(); opts = manta.ExtentOptions(); plan = manta.plan_pair_extents(basis, opts)
    budget = manta.ResourceBudget(); setattr(budget, field, getattr(plan, plan_field))
    result = manta.compute_pair_extents(basis, opts, budget)
    assert result.receipt.modeled_peak_bytes == plan.modeled_peak_bytes
    assert result.receipt.envelope_evaluations <= 132*plan.shell_pairs
    setattr(budget, field, getattr(plan, plan_field)-1)
    with pytest.raises(ValueError, match="budget"):
        manta.compute_pair_extents(basis, opts, budget)


@pytest.mark.parametrize("field,value", [("pointwise_tolerance", 0.), ("tail_mass_tolerance", -1.),
    ("pointwise_tolerance", np.nan), ("maximum_radius", np.inf), ("minimum_radius", -1.),
    ("maximum_radius", 1.), ("moment_order", 13), ("moment_order", -1)])
def test_invalid_extent_policy_refused(field, value):
    opts = manta.ExtentOptions(); setattr(opts, field, value)
    with pytest.raises(ValueError):
        manta.plan_pair_extents(_basis(), opts)


def test_extreme_separation_never_turns_a_positive_envelope_into_exact_zero():
    basis = _basis((0, 0), exponents=[1e100], coefficients=[1.])
    shells = basis.shells(); shells[1].origin = [1e150, 0., 0.]
    mol = vq.Molecule([vq.Atom(2, shells[0].origin), vq.Atom(2, shells[1].origin)], 0, 1)
    extreme = vq.BasisSet(mol, shells, "manta-extents-extreme", True)
    try:
        result = manta.compute_pair_extents(extreme)
    except OverflowError as error:
        # long double has platform-dependent range; explicit refusal is valid.
        assert "overflow" in str(error)
    else:
        assert all(p.l1_norm_bound > 0 for p in result.pairs)
        assert all(p.tail_mass_bound > 0 for p in result.pairs)


def test_tight_angular_positive_bounds_do_not_vanish_by_peak_arithmetic_overflow():
    basis = _basis((3,), exponents=[8e307], coefficients=[1.])
    opts = manta.ExtentOptions(); opts.minimum_radius = 0.; opts.maximum_radius = 1e-200
    pair = manta.compute_pair_extents(basis, opts).pairs[0]
    assert pair.linfinity_bound > 0 and pair.potential_bound > 0
    assert pair.l1_norm_bound > 0 and pair.tail_mass_bound > 0


def test_search_cap_and_exact_contraction_cancellation_are_explicit():
    opts = manta.ExtentOptions(); opts.minimum_radius = 0.; opts.maximum_radius = 1e120
    opts.moment_order = 0  # Higher moments of this enormous diffuse cloud overflow binary64.
    pair = manta.compute_pair_extents(_basis((0,), exponents=[5e-201], coefficients=[1e-150]), opts).pairs[0]
    assert pair.always_near and pair.reason == "search-cap"
    assert pair.radius < opts.maximum_radius and pair.tail_mass_bound > opts.tail_mass_tolerance
    cancelled = _basis((0,), exponents=[.4, .4], coefficients=[1., -1.])
    result = manta.compute_pair_extents(cancelled).pairs[0]
    assert np.isfinite(result.radius) and result.l1_norm_bound > 0
    np.testing.assert_array_equal(_ao(cancelled.shells()[0], np.array([[0., 0., 0.], [1., 2., 3.]])), 0.)
