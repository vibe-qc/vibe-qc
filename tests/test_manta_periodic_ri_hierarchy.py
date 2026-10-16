"""Ordered finite-support auxiliary image hierarchy and paired consumers."""
import dataclasses

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta
from tests import test_manta_periodic_ri_multipole as leaf
from tests.test_manta_periodic_fitting import CAPS
from tests.test_manta_periodic_gdf_tiles import assemble
from tests.test_manta_periodic_gdf_actions import inputs


def hierarchy_job(directory, method='uhf', species='Li', limit=100):
    job, args, k = leaf.multipole_job(directory, method, species, limit)
    args['options'].fitting.coupling_multipole_hierarchy = True
    job['periodic_fitting'] = dataclasses.replace(job['periodic_fitting'], coupling_multipole_hierarchy=True)
    return job, args, k


def hierarchy_ks_job(directory, spins=1, limit=100):
    job, args, a, k, old = leaf.multipole_ks_job(directory, spins, limit)
    args['options'].fitting.coupling_multipole_hierarchy = True
    job['periodic_fitting'] = dataclasses.replace(job['periodic_fitting'], coupling_multipole_hierarchy=True)
    return job, args, a, k, old


@pytest.fixture
def enable(monkeypatch):
    original = leaf.controls
    def controls(o, *args, **kw):
        original(o, *args, **kw); o.coupling_multipole_hierarchy = True
    monkeypatch.setattr(leaf, 'controls', controls)


@pytest.mark.parametrize('gamma', [False, True])
@pytest.mark.parametrize('order', [0, 2, 4])
def test_finite_source_independent_integrals_and_complex_paired_actions(enable, gamma, order):
    leaf.test_ordered_source_and_complex_adjoints_with_resolvable_far_terms(gamma, order)


def test_reciprocal_and_atom_image_gauge(enable):
    leaf.test_image_gauge_and_reciprocal_relabelling_preserve_the_approximate_map()


@pytest.mark.parametrize('rank', [0, 1, 6])
def test_unchanged_metric_and_quadratic_energy_derivative(enable, rank):
    leaf.test_fixed_metric_rank_energy_derivative_and_evaluated_error(rank)


@pytest.mark.parametrize('field,quantity', CAPS)
def test_joint_tree_moment_envelope_and_action_admission(enable, field, quantity):
    leaf.test_all_joint_caps_and_no_values_before_admission(field, quantity)


@pytest.mark.parametrize('pure', [False, True])
def test_signed_angular_against_independent_libint_and_fourier(enable, pure):
    leaf.test_signed_angular_leaf_contraction_against_libint(pure)


@pytest.mark.parametrize('order', [0, 2])
def test_higher_nodes_represent_multiple_images_and_partition_is_density_independent(enable, order):
    a = leaf.multipole_case(False, order=order); source = manta.PeriodicGDFTiles(**a)
    x, c = inputs(source); first = source.forward(x, a['budget']); second = source.adjoint(c, a['budget'])
    r = first.receipt
    assert 0 < r.multipole_far_blocks < r.multipole_images
    assert r.multipole_translations > r.multipole_far_blocks and r.multipole_tree_nodes > r.multipole_far_blocks
    for name in ('multipole_far_blocks', 'multipole_images', 'multipole_translations', 'multipole_tree_nodes', 'coupling_error_bound'):
        assert getattr(r, name) == getattr(second.receipt, name) == getattr(source.forward(0*x, a['budget']).receipt, name)
    assert not source.receipt.coupling_error_evaluated and not source.metric().receipt.coupling_error_evaluated
    a['options'].coupling_multipole_hierarchy = False
    ungrouped = manta.PeriodicGDFTiles(**a)
    assert source.metric_source_identity == ungrouped.metric_source_identity
    assert source.three_center_source_identity != ungrouped.three_center_source_identity
    np.testing.assert_array_equal(source.forward(x, a['budget']).values, first.values)
    # Zero approximation ignores the optional tree selection entirely.
    a['options'].coupling_multipole_error = 0.; exact = manta.PeriodicGDFTiles(**a)
    a['options'].coupling_multipole_hierarchy = True; zero = manta.PeriodicGDFTiles(**a)
    assert zero.source_identity == exact.source_identity
    np.testing.assert_array_equal(zero.forward(x, a['budget']).values, exact.forward(x, a['budget']).values)


@pytest.mark.parametrize('always_near', [False, True])
def test_strict_or_always_near_support_refines_to_direct_integrals(enable, always_near):
    a = leaf.multipole_case(False); o = a['options']; o.coupling_multipole_error = 1e-30
    if always_near:
        o.coupling_multipole_extents.maximum_radius = o.coupling_multipole_extents.minimum_radius = .1
    source = manta.PeriodicGDFTiles(**a); x, _ = inputs(source)
    r = source.forward(x, a['budget']); assert r.receipt.multipole_images == 0
    o.coupling_multipole_error = 0.; exact = manta.PeriodicGDFTiles(**a)
    np.testing.assert_allclose(assemble(source), assemble(exact), atol=1e-13, rtol=1e-14)


def test_typed_hierarchy_option():
    for value in (1, 'true', None):
        with pytest.raises(TypeError, match='boolean'):
            manta.PeriodicFittingOptions(.2, 8., coupling_multipole_hierarchy=value)


@pytest.mark.parametrize('method,species,limit', [('rhf', 'Be', 100), ('uhf', 'Li', 100), ('uhf', 'Li', 3)])
def test_public_hf_shared_fock_and_saved_evaluated_density(enable, monkeypatch, tmp_path, method, species, limit):
    original = leaf.multipole_job
    def job(*args, **kw):
        spec, a, k = original(*args, **kw)
        spec['periodic_fitting'] = dataclasses.replace(spec['periodic_fitting'], coupling_multipole_hierarchy=True)
        return spec, a, k
    monkeypatch.setattr(leaf, 'multipole_job', job)
    leaf.test_public_hf_uses_far_terms_and_keeps_unified_fock_and_saved_density(tmp_path, method, species, limit)


@pytest.mark.parametrize('spins', [1, 2])
def test_public_ks_shared_xc_and_fock_receipts(enable, monkeypatch, tmp_path, spins):
    original = leaf.multipole_ks_job
    def job(*args, **kw):
        spec, a, olda, k, old = original(*args, **kw)
        spec['periodic_fitting'] = dataclasses.replace(spec['periodic_fitting'], coupling_multipole_hierarchy=True)
        return spec, a, olda, k, old
    monkeypatch.setattr(leaf, 'multipole_ks_job', job)
    leaf.test_public_ks_retains_multipole_receipt_and_same_xc(tmp_path, spins)



def test_empty_selected_auxiliary_support_is_not_replaced_by_a_periodic_pair_domain(enable):
    from vibeqc import _vibeqc_core as core
    a = leaf.multipole_case(False); o = a['options']
    auxiliary_molecule = vq.Molecule([vq.Atom(2, [.97, 1.43, 1.21])])
    a['auxiliary'] = vq.BasisSet(auxiliary_molecule,
        [core.ShellInfo(0, 0, True, [5.], [1.], [.97, 1.43, 1.21])], 'off-center-auxiliary', True)
    o.pair_cutoff_bohr = .01; o.auxiliary_cutoff_bohr = .01
    source = manta.PeriodicGDFTiles(**a); x, _ = inputs(source)
    r = source.forward(x, a['budget'])
    assert r.receipt.multipole_tree_nodes == r.receipt.multipole_images == 0
    o.coupling_multipole_error = 0.; exact = manta.PeriodicGDFTiles(**a)
    np.testing.assert_array_equal(assemble(source), assemble(exact))


@pytest.mark.parametrize('order', [0, 2, 4])
def test_small_approximation_allowance_agrees_with_independent_exact_source(enable, order):
    a = leaf.multipole_case(False, order=order); a['options'].coupling_multipole_error = 1e-10
    source = manta.PeriodicGDFTiles(**a); _, exact = leaf.independent(a)
    np.testing.assert_allclose(assemble(source), exact, atol=1.05e-10, rtol=0)
    x, _ = inputs(source)
    assert source.forward(x, a['budget']).receipt.coupling_error_bound <= 1e-10


def test_hierarchy_stays_j_only_at_correlated_fit_export():
    from vibeqc.periodic.ccm import CCMSystem
    from vibeqc.periodic.ccm.fit_source import MANTAFitSource
    from vibeqc.periodic.ccm.neutral import ccm_neutral_cderi_fold
    from tests.test_ccm_manta_fit_source import _cell, _options, _budget, CUTOFF_HA
    ccm = CCMSystem(_cell(), (2, 1, 1), 'sto-3g'); o = _options()
    o.exchange_algorithm = 'exact'; o.fitting.coupling_multipole_error = .1
    o.fitting.coupling_multipole_hierarchy = True
    with pytest.raises(NotImplementedError, match='J-only'):
        ccm_neutral_cderi_fold(ccm, fit_source=MANTAFitSource(o, CUTOFF_HA, budget=_budget()))
