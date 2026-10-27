"""Shared primitive moment panels against direct polynomial quadrature."""
import numpy as np
import pytest
from scipy.special import gamma

import vibeqc as vq
from vibeqc import manta
from vibeqc import _vibeqc_core as core


def _basis(l=2, pure=True, exponents=(.7, 1.6), shift=(0., 0., 0.)):
    positions = np.array([[.1, -.2, .3], [.4, .1, -.1]])+shift
    mol = vq.Molecule([vq.Atom(2, p.tolist()) for p in positions], 0, 1)
    shells = []
    for i, p in enumerate(positions):
        s = core.ShellInfo(); s.atom_index = i; s.l = l if i == 0 else max(l-1, 0)
        s.pure = pure; s.origin = p.tolist(); s.exponents = exponents
        s.coefficients = [.8, -.2][:len(exponents)]; shells.append(s)
    return vq.BasisSet(mol, shells, "manta-moment-fixture", True)


def _cart(l):
    return np.array([(x, y, l-x-y) for x in range(l, -1, -1) for y in range(l-x, -1, -1)])


def _quadrature(a, b, origin, indices):
    # Tensor-product Gauss-Hermite integration evaluated as three 1D polynomial
    # quadratures, independent of the MD recurrence and its origin expansion.
    nodes, weights = np.polynomial.hermite.hermgauss(16)
    ca, cb = _cart(a.l), _cart(b.l); indices = np.array(indices)
    result = np.zeros((len(ca), len(cb), len(indices)))
    for alpha, ac in zip(a.exponents, a.coefficients, strict=True):
        for beta, bc in zip(b.exponents, b.coefficients, strict=True):
            p = alpha+beta; center = (alpha*np.array(a.origin)+beta*np.array(b.origin))/p
            value = np.ones_like(result)
            for axis in range(3):
                xyz = nodes/np.sqrt(p)+center[axis]
                va = (xyz-a.origin[axis])[None, :]**np.arange(a.l+1)[:, None]
                vb = (xyz-b.origin[axis])[None, :]**np.arange(b.l+1)[:, None]
                vm = (xyz-origin[axis])[None, :]**np.arange(indices[:, axis].max()+1)[:, None]
                integral = np.einsum("it,jt,kt,t->ijk", va, vb, vm, weights)
                value *= integral[ca[:, axis, None, None], cb[None, :, axis, None], indices[None, None, :, axis]]
            prefactor = ac*bc*np.exp(-alpha*beta/p*np.sum((np.array(a.origin)-b.origin)**2))/p**1.5
            result += prefactor*value
    ta = manta.cartesian_to_spherical(a.l) if a.pure else np.eye(len(ca))
    tb = manta.cartesian_to_spherical(b.l) if b.pure else np.eye(len(cb))
    return np.einsum("ai,ijq,bj->abq", ta, result, tb).reshape(ta.shape[0]*tb.shape[0], len(indices))


@pytest.mark.parametrize("l", [0, 2, 6])
@pytest.mark.parametrize("pure", [False, True])
@pytest.mark.parametrize("order", [0, 1, 4, 6, 9, 12])
def test_complete_signed_contracted_panels_by_independent_quadrature(l, pure, order):
    basis = _basis(l, pure); origin = [.15, -.1, .05]
    result = manta.compute_pair_moments(basis, [0, 1], origin, order)
    expected = _quadrature(*basis.shells(), origin, result.indices)
    np.testing.assert_allclose(result.values, expected, atol=2e-11, rtol=5e-12)
    receipt = result.receipt
    assert receipt.shell_calls == 1 and receipt.primitive_pairs == receipt.primitive_expansions == 4
    assert receipt.primitive_moment_products == 4*len(result.indices)
    assert result.values.shape == (np.prod(result.ao_shape), len(result.indices))
    assert len(result.indices) == (order+1)*(order+2)*(order+3)//6
    assert receipt.normalization == "raw-cartesian-moments;libint-shell-axial-v1"


@pytest.mark.parametrize("order", [0, 6, 12])
def test_origin_translation_pair_exchange_and_owned_exports(order):
    basis = _basis(); a = np.array([.2, -.3, .1]); b = np.array([.1, .2, .4])
    first = manta.compute_pair_moments(basis, [0, 1], a, order)
    second = manta.compute_pair_moments(basis, [0, 1], b, order)
    translation = manta.multipole_translation(a-b, order)
    np.testing.assert_allclose(first.values@translation.values.T, second.values, atol=2e-11, rtol=5e-12)
    reverse = manta.compute_pair_moments(basis, [1, 0], a, order)
    values = first.values.reshape(*first.ao_shape, -1)
    np.testing.assert_allclose(values.transpose(1, 0, 2).reshape(reverse.values.shape), reverse.values, atol=2e-11, rtol=5e-12)
    assert len({x.receipt.source_identity for x in [first, second, reverse]}) == 3
    copied = first.values; saved = copied.copy(); copied[:] = -555.
    np.testing.assert_array_equal(first.values, saved)
    assert len(first.receipt.source_identity) == 64


def test_monopole_and_dipoles_match_independent_libint():
    basis = _basis(2, False); origin = [.3, .1, -.2]
    result = manta.compute_pair_moments(basis, [0, 1], origin, 1)
    n, m = result.ao_shape; indices = [tuple(x) for x in result.indices]
    matrix = result.values.reshape(n, m, -1)
    np.testing.assert_allclose(matrix[:, :, 0], vq.compute_overlap(basis)[:n, n:], atol=2e-12, rtol=3e-12)
    dipole = vq.compute_dipole(basis, origin)
    for axis, expected in enumerate([dipole.x, dipole.y, dipole.z]):
        power = [0, 0, 0]; power[axis] = 1
        np.testing.assert_allclose(matrix[:, :, indices.index(tuple(power))], expected[:n, n:], atol=2e-12, rtol=3e-12)


@pytest.mark.parametrize("field,plan_field", [("memory_bytes", "modeled_peak_bytes"), ("primitive_quartets", "primitive_pairs"), ("scalar_terms", "scalar_terms"), ("kernel_terms", "kernel_terms")])
def test_panel_resource_boundaries(field, plan_field):
    basis = _basis(); plan = manta.plan_pair_moments(basis, [0, 1], 9)
    budget = manta.ResourceBudget(); setattr(budget, field, getattr(plan, plan_field))
    actual = manta.compute_pair_moments(basis, [0, 1], [0., 0., 0.], 9, budget)
    assert actual.receipt.modeled_peak_bytes == plan.modeled_peak_bytes
    setattr(budget, field, getattr(plan, plan_field)-1)
    with pytest.raises(ValueError, match="budget"):
        manta.compute_pair_moments(basis, [0, 1], [0., 0., 0.], 9, budget)


def test_invalid_requests_zero_sign_identity_and_per_panel_admission():
    basis = _basis()
    for order in [-1, 13]:
        with pytest.raises(ValueError, match="order"):
            manta.compute_pair_moments(basis, [0, 1], [0., 0., 0.], order)
    with pytest.raises(IndexError, match="shell"):
        manta.plan_pair_moments(basis, [0, 2], 1)
    with pytest.raises(ValueError, match="finite"):
        manta.compute_pair_moments(basis, [0, 1], [np.nan, 0., 0.], 1)
    with pytest.raises(OverflowError):
        manta.compute_pair_moments(basis, [0, 1], [1e300, 0., 0.], 2)
    budget = manta.ResourceBudget(); budget.shell_calls = 1
    a = manta.compute_pair_moments(basis, [0, 1], [0., 0., 0.], 2, budget)
    b = manta.compute_pair_moments(basis, [0, 1], [-0., 0., 0.], 2, budget)
    assert a.receipt.source_identity == b.receipt.source_identity
    budget.shell_calls = 0
    with pytest.raises(ValueError, match="budget"):
        manta.compute_pair_moments(basis, [0, 1], [0., 0., 0.], 2, budget)


@pytest.mark.parametrize("l,pure", [(0, False), (2, False), (2, True), (6, False), (6, True)])
def test_full_absolute_moment_bounds_cover_every_signed_cartesian_moment(l, pure):
    basis = _basis(l, pure)
    options = manta.ExtentOptions(); options.moment_order = 12; options.retain_full_moments = True
    extents = manta.compute_pair_extents(basis, options)
    for pair in extents.pairs:
        moments = manta.compute_pair_moments(basis, pair.shells, pair.center, 12)
        assert len(pair.full_moment_bounds) == 13
        assert pair.full_moment_bounds[0] == pair.l1_norm_bound
        for q, index in enumerate(moments.indices):
            assert np.max(abs(moments.values[:, q])) <= pair.full_moment_bounds[sum(index)]*(1+1e-12)
        assert np.all(np.array(pair.tail_moment_bounds) <= pair.full_moment_bounds)
    # The tree's retained-byte model includes these optional vectors.
    full = manta.plan_bubbles(extents)
    options.retain_full_moments = False
    simple = manta.compute_pair_extents(basis, options)
    assert simple.receipt.source_identity != extents.receipt.source_identity
    assert all(not p.full_moment_bounds for p in simple.pairs)
    assert full.modeled_peak_bytes > manta.plan_bubbles(simple).modeled_peak_bytes


@pytest.mark.parametrize("exponent", [.001, .7, 100.])
def test_full_radial_moments_match_analytic_s_gaussian(exponent):
    basis = _basis(0, exponents=[exponent]); options = manta.ExtentOptions()
    options.moment_order = 12; options.retain_full_moments = True
    pair = manta.compute_pair_extents(basis, options).pairs[0]
    p = 2*exponent; c = .8**2
    expected = [2*np.pi*c*gamma((k+3)/2)/p**((k+3)/2) for k in range(13)]
    np.testing.assert_allclose(pair.full_moment_bounds, expected, atol=0., rtol=2e-12)


@pytest.mark.parametrize("field,plan_field", [("memory_bytes", "modeled_peak_bytes"), ("scalar_terms", "scalar_terms")])
def test_optional_full_bound_storage_is_admitted(field, plan_field):
    basis = _basis(); options = manta.ExtentOptions(); options.retain_full_moments = True
    plan = manta.plan_pair_extents(basis, options)
    budget = manta.ResourceBudget(); setattr(budget, field, getattr(plan, plan_field))
    manta.compute_pair_extents(basis, options, budget)
    setattr(budget, field, getattr(plan, plan_field)-1)
    with pytest.raises(ValueError, match="budget"):
        manta.compute_pair_extents(basis, options, budget)
