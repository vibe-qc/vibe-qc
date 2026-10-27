"""Single-auxiliary Gaussian moments and conservative extent envelopes."""
import gc

import numpy as np
import pytest
from scipy.special import gamma, gammaincc

import vibeqc as vq
from vibeqc import manta
from vibeqc import _vibeqc_core as core


def _basis(l=2, pure=True, exponents=(1.2, 2.6), coefficients=(.8, -.2), origin=(.1, -.2, .3)):
    molecule = vq.Molecule([vq.Atom(2, list(origin))], 0, 1)
    shell = core.ShellInfo(); shell.atom_index = 0; shell.l = l; shell.pure = pure
    shell.origin = list(origin); shell.exponents = exponents; shell.coefficients = coefficients
    return vq.BasisSet(molecule, [shell], "manta-auxiliary-distribution", True)


def _powers(l):
    return np.array([(x, y, l-x-y) for x in range(l, -1, -1) for y in range(l-x, -1, -1)])


def _quadrature(shell, origin, indices):
    # Direct polynomial quadrature has no MD recurrence or primitive-pair
    # product. The integrand has degree at most l+order <= 19.
    nodes, weights = np.polynomial.hermite.hermgauss(16)
    powers, indices = _powers(shell.l), np.asarray(indices)
    cartesian = np.zeros((len(powers), len(indices)))
    for alpha, coefficient in zip(shell.exponents, shell.coefficients, strict=True):
        value = np.ones_like(cartesian)
        for axis in range(3):
            relative = nodes/np.sqrt(alpha)
            xyz = relative+shell.origin[axis]-origin[axis]
            basis_values = relative[None, :]**powers[:, axis, None]
            moment_values = xyz[None, :]**indices[:, axis, None]
            value *= np.einsum("it,qt,t->iq", basis_values, moment_values, weights)
        cartesian += coefficient*value/alpha**1.5
    transform = manta.cartesian_to_spherical(shell.l) if shell.pure else np.eye(len(powers))
    return transform@cartesian


@pytest.mark.parametrize("l", [0, 2, 7])
@pytest.mark.parametrize("pure", [False, True])
@pytest.mark.parametrize("order", [0, 1, 6, 12])
def test_single_auxiliary_panels_match_independent_polynomial_quadrature(l, pure, order):
    basis = _basis(l, pure); origin = [.2, -.1, .05]
    result = manta.compute_auxiliary_moments(basis, 0, origin, order)
    expected = _quadrature(basis.shells()[0], origin, result.indices)
    np.testing.assert_allclose(result.values, expected, atol=3e-11, rtol=1e-11)
    assert result.values.shape == (basis.nbasis, (order+1)*(order+2)*(order+3)//6)
    r = result.receipt
    assert r.shell_calls == 1 and r.primitives == r.primitive_expansions == 2
    assert r.primitive_moment_products == 2*len(result.indices)
    assert r.normalization == "raw-cartesian-moments;libint-shell-axial-v1"
    assert len(r.source_identity) == 64


@pytest.mark.parametrize("exponent", [.001, .7, 100.])
def test_auxiliary_s_moments_are_single_gaussians_not_squared_products(exponent):
    basis = _basis(0, False, [exponent], [.8]); shell = basis.shells()[0]
    result = manta.compute_auxiliary_moments(basis, 0, shell.origin, 12)
    for column, powers in enumerate(result.indices):
        expected = .8
        for power in powers:
            expected *= 0. if power % 2 else gamma((power+1)/2)/exponent**((power+1)/2)
        assert result.values[0, column] == pytest.approx(expected, rel=2e-12, abs=1e-14)
    assert result.values[0, 0] == pytest.approx(.8*(np.pi/exponent)**1.5, rel=1e-13)
    pair = manta.compute_pair_moments(basis, [0, 0], shell.origin, 0)
    assert not np.isclose(result.values[0, 0], pair.values[0, 0], rtol=1e-3, atol=0.)


def test_translation_fourier_monopoles_and_owned_exports():
    basis = _basis(7); first_origin = np.array([.2, -.3, .1]); second_origin = np.array([-.2, .1, .4])
    first = manta.compute_auxiliary_moments(basis, 0, first_origin, 12)
    second = manta.compute_auxiliary_moments(basis, 0, second_origin, 12)
    translation = manta.multipole_translation(first_origin-second_origin, 12)
    np.testing.assert_allclose(first.values@translation.values.T, second.values, atol=2e-10, rtol=2e-11)
    zero = manta.compute_auxiliary_fourier_panel(basis, np.zeros((1, 3)))
    np.testing.assert_allclose(first.values[:, 0], zero.values[0].real, atol=2e-12, rtol=2e-12)
    assert first.receipt.source_identity != second.receipt.source_identity
    value = first.values; saved = value.copy(); value[:] = -50.
    del basis, second; gc.collect()
    np.testing.assert_array_equal(first.values, saved)
    with pytest.raises(AttributeError):
        first.receipt.source_identity = "changed"


def test_mixed_shell_inventory_is_linear_and_keeps_physical_centers():
    shells = [_basis(0).shells()[0], _basis(2, exponents=[.4], coefficients=[.3], origin=[2., .1, -.4]).shells()[0]]
    shells[1].atom_index = 1
    molecule = vq.Molecule([vq.Atom(2, s.origin) for s in shells], 0, 1)
    basis = vq.BasisSet(molecule, shells, "manta-mixed-auxiliary", True)
    extents = manta.compute_auxiliary_extents(basis)
    plan = manta.plan_auxiliary_extents(basis)
    assert plan.shells == extents.receipt.shells == 2
    assert plan.primitives == extents.receipt.primitives == 3
    assert plan.maximum_shell_primitives == 2
    monopoles = []
    for index, (shell, extent) in enumerate(zip(shells, extents.shells, strict=True)):
        assert extent.shell == index and extent.center == pytest.approx(shell.origin)
        moments = manta.compute_auxiliary_moments(basis, index, shell.origin, 0)
        assert moments.receipt.primitives == len(shell.exponents)
        monopoles.extend(moments.values[:, 0])
    fourier = manta.compute_auxiliary_fourier_panel(basis, np.zeros((1, 3)))
    np.testing.assert_allclose(monopoles, fourier.values[0].real, atol=2e-12, rtol=2e-12)


def test_exact_signed_cancellation_keeps_absolute_extent_bounds():
    basis = _basis(2, exponents=[1.2, 1.2], coefficients=[.8, -.8])
    moments = manta.compute_auxiliary_moments(basis, 0, [.2, -.1, .05], 12)
    np.testing.assert_array_equal(moments.values, np.zeros_like(moments.values))
    options = manta.ExtentOptions(); options.retain_full_moments = True
    extent = manta.compute_auxiliary_extents(basis, options).shells[0]
    assert extent.l1_norm_bound > 0 and extent.potential_bound > 0
    assert min(extent.full_moment_bounds) > 0


@pytest.mark.parametrize("l", [0, 2, 7])
@pytest.mark.parametrize("pure", [False, True])
def test_auxiliary_envelopes_bound_full_moments_radial_tails_and_exterior_values(l, pure):
    basis = _basis(l, pure); shell = basis.shells()[0]
    options = manta.ExtentOptions(); options.moment_order = 12; options.retain_full_moments = True
    result = manta.compute_auxiliary_extents(basis, options); extent = result.shells[0]
    assert extent.shell == 0 and extent.center == pytest.approx(shell.origin)
    assert not extent.always_near and not extent.reason
    assert extent.pointwise_bound <= options.pointwise_tolerance
    assert extent.tail_mass_bound <= options.tail_mass_tolerance
    assert result.receipt.center_policy == "shell-center"
    assert result.receipt.shells == 1 and result.receipt.primitives == 2
    powers = _powers(l)
    transform = manta.cartesian_to_spherical(l) if pure else np.eye(len(powers))
    representation_norm = np.max(np.sum(abs(transform), axis=1))
    # Radial integral of the independently written absolute r^l envelope.
    full, tail = np.zeros(13), np.zeros(13)
    for k in range(13):
        shape = (l+k+3)/2
        for alpha, coefficient in zip(shell.exponents, shell.coefficients, strict=True):
            term = 2*np.pi*abs(coefficient)*representation_norm*gamma(shape)/alpha**shape
            full[k] += term
            tail[k] += term*gammaincc(shape, alpha*extent.radius**2)
    np.testing.assert_allclose(extent.full_moment_bounds, full, atol=0., rtol=2e-12)
    assert np.all(np.asarray(extent.tail_moment_bounds) >= tail*(1-5e-13))
    assert np.all(np.asarray(extent.tail_moment_bounds) <= extent.full_moment_bounds)
    moments = manta.compute_auxiliary_moments(basis, 0, extent.center, 12)
    for column, index in enumerate(moments.indices):
        assert np.max(abs(moments.values[:, column])) <= extent.full_moment_bounds[sum(index)]
    directions = np.random.default_rng(473).normal(size=(100, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    for radius in [extent.radius, 1.3*extent.radius, 2*extent.radius]:
        relative = radius*directions
        polynomial = np.prod(relative[:, None, :]**powers[None, :, :], axis=2)
        radial = sum(c*np.exp(-a*radius**2) for a, c in zip(shell.exponents, shell.coefficients, strict=True))
        values = radial*polynomial@transform.T
        assert np.max(abs(values)) <= extent.pointwise_bound


@pytest.mark.parametrize("exponent", [.001, .7, 100.])
def test_auxiliary_s_potential_bound_and_radius_refusal(exponent):
    basis = _basis(0, False, [exponent], [.8]); options = manta.ExtentOptions()
    result = manta.compute_auxiliary_extents(basis, options)
    extent = result.shells[0]
    assert extent.potential_bound >= 2*np.pi*.8/exponent
    options.minimum_radius = options.maximum_radius = .001
    refused = manta.compute_auxiliary_extents(basis, options)
    assert refused.shells[0].always_near and refused.shells[0].reason == "radius-cap"
    assert refused.receipt.always_near_shells == 1
    assert refused.shells[0].tail_mass_bound > options.tail_mass_tolerance
    assert result.receipt.source_identity != refused.receipt.source_identity
    shells = result.shells; shells.clear()
    assert len(result.shells) == 1


@pytest.mark.parametrize("adapter", ["moments", "extents"])
@pytest.mark.parametrize("field,plan_field", [("memory_bytes", "modeled_peak_bytes"), ("primitive_quartets", "primitives"),
    ("scalar_terms", "scalar_terms"), ("kernel_terms", "kernel_terms"), ("shell_calls", None)])
def test_auxiliary_resource_boundaries(adapter, field, plan_field):
    basis = _basis(7); budget = manta.ResourceBudget(); budget.other_live_bytes = 12345
    options = manta.ExtentOptions(); options.moment_order = 12; options.retain_full_moments = True
    def plan():
        return (manta.plan_auxiliary_moments(basis, 0, 12, budget) if adapter == "moments" else
                manta.plan_auxiliary_extents(basis, options, budget))
    def compute():
        return (manta.compute_auxiliary_moments(basis, 0, [.2, -.1, .05], 12, budget) if adapter == "moments" else
                manta.compute_auxiliary_extents(basis, options, budget))
    baseline = plan(); cap = 1 if plan_field is None else getattr(baseline, plan_field)
    setattr(budget, field, cap)
    assert compute().receipt.modeled_peak_bytes == baseline.modeled_peak_bytes
    setattr(budget, field, cap-1)
    with pytest.raises(ValueError, match="budget"):
        compute()


def test_auxiliary_invalid_inputs_source_policy_and_zero_sign_identity():
    basis = _basis()
    for order in [-1, 13]:
        with pytest.raises(ValueError, match="order"):
            manta.compute_auxiliary_moments(basis, 0, [0., 0., 0.], order)
    with pytest.raises(IndexError, match="shell"):
        manta.plan_auxiliary_moments(basis, 1, 6)
    with pytest.raises(ValueError, match="finite"):
        manta.compute_auxiliary_moments(basis, 0, [np.nan, 0., 0.], 6)
    with pytest.raises(OverflowError):
        manta.compute_auxiliary_moments(basis, 0, [1e300, 0., 0.], 12)
    for operation in [lambda b: manta.plan_auxiliary_moments(b, 0, 0), manta.plan_auxiliary_extents]:
        with pytest.raises(ValueError, match="angular"):
            operation(_basis(8))
    a = manta.compute_auxiliary_moments(basis, 0, [0., 0., 0.], 6)
    b = manta.compute_auxiliary_moments(basis, 0, [-0., 0., 0.], 6)
    assert a.receipt.source_identity == b.receipt.source_identity
    c = manta.compute_auxiliary_moments(_basis(coefficients=[.8, -.3]), 0, [0., 0., 0.], 6)
    assert c.receipt.source_identity != a.receipt.source_identity
    options = manta.ExtentOptions(); simple = manta.compute_auxiliary_extents(basis, options)
    options.retain_full_moments = True; full = manta.compute_auxiliary_extents(basis, options)
    assert not simple.shells[0].full_moment_bounds
    assert len(full.shells[0].full_moment_bounds) == 7
    assert simple.receipt.source_identity != full.receipt.source_identity
    assert simple.receipt.modeled_peak_bytes < full.receipt.modeled_peak_bytes
