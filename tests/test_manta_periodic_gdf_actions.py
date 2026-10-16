"""Producer-owned periodic GDF actions: complex adjoints before any RI solve."""
import itertools

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta, _vibeqc_core as core
from tests.test_aux_eri_sr_lattice_enumeration import _weighted_gradient_fixture
from tests.test_manta_periodic_fitting import CAPS, libint_reference
from tests.test_manta_periodic_gdf import case, independent, legacy, quadrature_long_range
from tests.test_manta_periodic_gdf_tiles import assemble
from tests.test_manta_periodic_range_fitting import physical, reference as range_reference
from tests.test_manta_periodic_ri_paired import paired_case
from tests.test_manta_periodic_ri_q import builder, densities


def inputs(source, seed=481):
    rng = np.random.default_rng(seed)
    p = source.plan
    shape = (p.momenta, p.n_orb, p.n_orb)
    x = rng.normal(size=shape)+1j*rng.normal(size=shape)
    c = rng.normal(size=p.n_aux)+1j*rng.normal(size=p.n_aux)
    return x, c


def check_actions(source, tensor, x, c, budget, atol=4e-12):
    f, a = source.forward(x, budget), source.adjoint(c, budget)
    np.testing.assert_allclose(f.values, np.einsum('kpij,kij->p', tensor.conj(), x), atol=atol, rtol=3e-13)
    np.testing.assert_allclose(a.values, np.einsum('kpij,p->kij', tensor, c), atol=atol, rtol=3e-13)
    assert np.vdot(c, f.values) == pytest.approx(np.vdot(a.values, x), abs=atol, rel=3e-13)
    for r, direction in ((f, 'forward'), (a, 'adjoint')):
        assert r.receipt.direction == direction
        assert r.receipt.coupling_identity == source.three_center_source_identity
        assert r.receipt.coupling_passes == 1
        assert r.receipt.shell_tiles == np.prod(source.shell_extents)
        assert r.receipt.integral_elements_consumed == tensor.size == r.receipt.contraction_products
        assert r.receipt.shell_calls <= r.plan.shell_calls
        assert r.receipt.primitive_products <= r.plan.primitive_products
    return f, a


@pytest.mark.parametrize('gamma', [False, True])
@pytest.mark.parametrize('shift', [False, True])
@pytest.mark.parametrize('kernel', ['coulomb', 'erfc', 'erf'])
def test_unweighted_complex_actions_match_independent_finite_sources(gamma, shift, kernel):
    args = case(gamma, shift)
    if kernel != 'coulomb':
        args['physical_kernel'] = physical(kernel, .43)
    source = manta.PeriodicGDFTiles(**args); x, c = inputs(source)
    # Arbitrary transition inputs, including non-Hermitian q=0 matrices.
    assert np.max(abs(x-x.swapaxes(-1, -2).conj())) > .1
    tensor = (independent(args) if kernel == 'coulomb' else range_reference(args))[1]
    check_actions(source, tensor, x, c, args['budget'])
    if kernel == 'coulomb':
        check_actions(source, legacy(args)[1], x, c, args['budget'])


@pytest.mark.parametrize('angular,pure', [(1, True), (2, False), (3, True)])
def test_signed_angular_actions_against_libint_and_polynomial_fourier(angular, pure):
    a = case(); system, orb, aux = _weighted_gradient_fixture(angular, orbital_angular=1)
    def basis(b):
        shells = b.shells()
        for s in shells:
            s.pure = pure
        return vq.BasisSet(system.unit_cell_molecule(), shells, 'coupling-angular', True)
    a.update(system=system, orbital=basis(orb), auxiliary=basis(aux),
        vectors=np.array(list(itertools.product(range(-1, 2), repeat=3)))@system.reciprocal_lattice().T+a['q'])
    a['options'].pair_cutoff_bohr = 4.3; a['options'].auxiliary_cutoff_bohr = 5.1
    source = manta.PeriodicGDFTiles(**a)
    tensor = libint_reference(a)[1]+quadrature_long_range(a)[1]
    check_actions(source, tensor, *inputs(source), a['budget'], atol=4e-11)


def test_screened_source_optional_metric_layouts_linearity_and_owned_exports():
    a = case(); a['options'].integral_screen_error = 1e-4
    source = manta.PeriodicGDFTiles(**a); x, c = inputs(source)
    tensor = assemble(source); f, rev = check_actions(source, tensor, x, c, a['budget'])
    a['options'].compute_metric = False
    no_metric = manta.PeriodicGDFTiles(**a)
    assert no_metric.metric_source_identity == ''
    for obj in (source, no_metric):
        np.testing.assert_array_equal(obj.forward(np.asfortranarray(x), a['budget']).values, f.values)
        np.testing.assert_array_equal(obj.adjoint(c, a['budget']).values, rev.values)
        assert obj.forward(x, a['budget']).receipt.snapshot_identity == f.receipt.snapshot_identity
        np.testing.assert_allclose(obj.forward((2-1j)*x, a['budget']).values, (2-1j)*f.values, atol=4e-12, rtol=0)
        np.testing.assert_allclose(obj.adjoint((1+2j)*c, a['budget']).values, (1+2j)*rev.values, atol=4e-12, rtol=0)
        np.testing.assert_array_equal(obj.forward(np.zeros_like(x), a['budget']).values, np.zeros_like(c))
        np.testing.assert_array_equal(obj.adjoint(np.zeros_like(c), a['budget']).values, np.zeros_like(x))
    saved = f.values.copy(); saved_rev = rev.values.copy(); identity = f.receipt.snapshot_identity
    f.values[:] = np.nan; rev.values[:] = np.nan; x[:] = np.nan; c[:] = np.nan
    for name in ('q', 'ket_kpoints', 'vectors'):
        a[name][:] = np.nan
    a['options'].omega = 0.; a['system'].lattice = 99*np.eye(3)
    np.testing.assert_array_equal(f.values, saved); np.testing.assert_array_equal(rev.values, saved_rev)
    assert f.receipt.snapshot_identity == identity
    with pytest.raises(AttributeError):
        f.receipt.coupling_identity = 'forged'
    with pytest.raises(TypeError):
        manta.PeriodicGDFCouplingResult()


@pytest.mark.parametrize('weights', [[.27, .73], [0., 1.]])
def test_ri_reuses_actions_with_method_owned_weights_factor_and_spin_sum(weights):
    a = paired_case(); source = manta.PeriodicGDFTiles(**a)
    b = builder(a); d = densities(); w = np.array(weights)
    f = source.forward(w[:, None, None]*d.sum(axis=0), a['budget'])
    rho = b.factor.T@f.values; c = b.factor.conj()@rho
    adjoint = source.adjoint(c, a['budget']); result = b.build(d, w, make_k=False)
    assert result.receipt.coulomb_contraction == 'paired-raw'
    np.testing.assert_allclose(result.J, adjoint.values, atol=2e-13, rtol=3e-13)
    assert result.receipt.coulomb_energy == pytest.approx(.5*np.vdot(rho, rho).real, abs=2e-13)
    assert result.receipt.shell_calls == f.receipt.shell_calls+adjoint.receipt.shell_calls
    assert result.receipt.primitive_products == f.receipt.primitive_products+adjoint.receipt.primitive_products
    assert result.receipt.coupling_passes == 2
    # The producer has no weights, spin density identity or second metric solve.
    assert not hasattr(f.receipt, 'density_identity')
    assert source.metric_source_identity == b.metric_info.metric_source_identity


@pytest.mark.parametrize('field,quantity', CAPS)
def test_single_action_joint_caps_precede_input_values(field, quantity):
    a = case(); source = manta.PeriodicGDFTiles(**a); x, c = inputs(source)
    budget = a['budget']; budget.other_live_bytes = 23456
    p = source.plan_coupling(budget)
    for name, count in CAPS:
        setattr(budget, name, getattr(p, count))
    source.forward(x, budget); source.adjoint(c, budget)
    setattr(budget, field, getattr(p, quantity)-1); x[:] = np.nan; c[:] = np.nan
    for action, args in ((source.plan_coupling, (budget,)), (source.forward, (x, budget)), (source.adjoint, (c, budget))):
        with pytest.raises((ValueError, RuntimeError), match='budget'):
            action(*args)


def test_action_memory_has_no_full_coupling_tensor_or_metric_workspace():
    a = case(True); system = vq.PeriodicSystem(3, 8*np.eye(3), [vq.Atom(2, [0., 0., 0.])])
    def basis(count):
        shells = [core.ShellInfo(0, 0, True, [.3+.02*i], [1.], [0., 0., 0.]) for i in range(count)]
        return vq.BasisSet(system.unit_cell_molecule(), shells, 'coupling-memory-census', True)
    a.update(system=system, orbital=basis(24), auxiliary=basis(80),
        vectors=np.zeros((1, 3)), ket_kpoints=np.zeros((4, 3)))
    a['options'].compute_metric = False
    for field, _ in CAPS:
        setattr(a['budget'], field, 2**64-1)
    source = manta.PeriodicGDFTiles(**a); p = source.plan_coupling(a['budget'])
    extra = p.modeled_peak_bytes-source.plan.coupling_peak_bytes
    assert extra == a['budget'].backend_margin_bytes+64*(4*24**2+80)
    assert 16*4*80*24**2 > 10*(extra-a['budget'].backend_margin_bytes)
    assert p.maximum_tile_elements == 4
    # Count-only witness. Do not launch this large traversal.


@pytest.mark.parametrize('direction', ['forward', 'adjoint'])
@pytest.mark.parametrize('fault', ['dtype', 'shape', 'strides', 'nonfinite', 'overflow'])
def test_action_rejects_bad_input_without_silent_projection(direction, fault):
    a = case(True); source = manta.PeriodicGDFTiles(**a); x, c = inputs(source)
    data = x if direction == 'forward' else c
    if fault == 'dtype':
        data = data.astype(np.complex64)
    elif fault == 'shape':
        data = data[:-1]
    elif fault == 'strides':
        data = data[::-1]
    elif fault == 'nonfinite':
        data.flat[0] = np.nan
    else:
        data[:] = complex(1e308, 1e308)
    with pytest.raises((ValueError, RuntimeError, TypeError, OverflowError)):
        getattr(source, direction)(data, a['budget'])
