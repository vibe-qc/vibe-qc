"""Bounded source-owned Gaussian preparation across paired actions and SCF."""
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import threading

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta
from tests.test_manta_periodic_ri_multipole import multipole_case, multipole_hf_reference, multipole_ks_reference
from tests.test_manta_periodic_ri_hierarchy import hierarchy_job, hierarchy_ks_job
from tests.test_manta_periodic_fitting import CAPS
from tests.test_manta_periodic_gdf_actions import inputs
from tests.test_manta_periodic_gdf_tiles import assemble, plan as source_plan


def prepared_case(hierarchy=True, order=2):
    a = multipole_case(False, order=order)
    a['options'].coupling_multipole_hierarchy = hierarchy
    return a


def inventory(source):
    i = source.preparation
    return tuple(getattr(i, n) for n in ('pair_extents', 'auxiliary_extents', 'moment_panels', 'moment_elements', 'modeled_live_bytes'))


@pytest.mark.parametrize('hierarchy', [False, True])
@pytest.mark.parametrize('order', [0, 2, 4])
def test_cold_warm_and_dense_maps_are_identical_and_preparation_has_one_owner(hierarchy, order):
    a = prepared_case(hierarchy, order); source = manta.PeriodicGDFTiles(**a)
    initial = inventory(source); assert initial[:4] == (0, 0, 0, 0)
    info = source.preparation
    assert info.enabled and info.source_identity == source.three_center_source_identity
    assert 0 < info.modeled_live_bytes <= info.capacity_bytes == source.plan.preparation_cache_bytes
    source.metric(); assert inventory(source) == initial
    x, c = inputs(source); cold = source.forward(x, a['budget']); full = inventory(source)
    assert cold.receipt.multipole_moment_panels == full[2] > 0
    assert cold.receipt.multipole_envelope_evaluations > 0
    assert cold.receipt.multipole_reused_extents > 0 and cold.receipt.multipole_reused_moment_panels > 0
    assert full[0] <= source.plan.preparation_pair_slots and full[1] <= source.plan.preparation_auxiliary_slots
    assert full[4] <= source.preparation.capacity_bytes
    assert source.preparation.support_tiles == source.plan.preparation_support_slots
    warm = source.forward(x, a['budget']); adjoint = source.adjoint(c, a['budget'])
    for result in (warm, adjoint):
        assert result.receipt.multipole_moment_panels == result.receipt.multipole_envelope_evaluations == 0
        assert result.receipt.multipole_reused_extents > 0 and result.receipt.multipole_reused_moment_panels > 0
        assert result.receipt.coupling_error_bound == cold.receipt.coupling_error_bound
    np.testing.assert_array_equal(cold.values, warm.values)
    assert inventory(source) == full
    # Dense evaluation remains the uncached route through the same producers.
    dense = manta.compute_periodic_gdf(**a)
    np.testing.assert_array_equal(assemble(source), dense.three_center)
    assert dense.receipt.three_center_source_identity == source.three_center_source_identity
    assert dense.receipt.short_range.multipole_moment_panels > full[2]
    np.testing.assert_allclose(cold.values, np.einsum('kpij,kij->p', dense.three_center.conj(), x), atol=4e-12, rtol=3e-13)
    np.testing.assert_allclose(adjoint.values, np.einsum('kpij,p->kij', dense.three_center, c), atol=4e-12, rtol=3e-13)
    # Streaming is the same numerical source with a smaller storage bound.
    a['options'].coupling_multipole_reuse = False; streaming = manta.PeriodicGDFTiles(**a)
    assert not streaming.preparation.enabled and streaming.plan.preparation_cache_bytes == 0
    assert streaming.plan.modeled_peak_bytes < source.plan.modeled_peak_bytes
    assert streaming.three_center_source_identity == source.three_center_source_identity
    np.testing.assert_array_equal(streaming.forward(x, a['budget']).values, cold.values)
    a['options'].coupling_multipole_reuse = True
    # Separate instances do not inherit another source's prepared state.
    other = manta.PeriodicGDFTiles(**a); assert inventory(other)[:4] == (0, 0, 0, 0)
    with pytest.raises(AttributeError): source.preparation.moment_panels = 0
    # A saved descriptor is a snapshot, not a reference to mutable diagnostics.
    assert info.moment_panels == 0


@pytest.mark.parametrize('hierarchy', [False, True])
def test_concurrent_cold_actions_publish_each_moment_panel_once(hierarchy):
    a = prepared_case(hierarchy); source = manta.PeriodicGDFTiles(**a); x, c = inputs(source)
    t = manta.compute_periodic_gdf(**a).three_center; barrier = threading.Barrier(4)
    def run(index):
        barrier.wait()
        return source.forward((index+1)*x, a['budget']) if index%2 == 0 else source.adjoint((index+1)*c, a['budget'])
    with ThreadPoolExecutor(max_workers=4) as pool: results = list(pool.map(run, range(4)))
    assert sum(r.receipt.multipole_moment_panels for r in results) == source.preparation.moment_panels > 0
    for index, r in enumerate(results):
        expected = np.einsum('kpij,kij->p', t.conj(), (index+1)*x) if index%2 == 0 else np.einsum('kpij,p->kij', t, (index+1)*c)
        np.testing.assert_allclose(r.values, expected, atol=4e-12, rtol=3e-13)
    assert source.forward(x, a['budget']).receipt.multipole_moment_panels == 0
    assert source.preparation.modeled_live_bytes <= source.preparation.capacity_bytes


@pytest.mark.parametrize('field,quantity', CAPS)
def test_warm_cache_keeps_full_joint_admission_before_numerical_inputs(field, quantity):
    a = prepared_case(); source = manta.PeriodicGDFTiles(**a); x, _ = inputs(source)
    source.forward(x, a['budget']); snapshot = inventory(source); b = a['budget']; p = source.plan_coupling(b)
    for name, value in CAPS: setattr(b, name, getattr(p, value))
    source.forward(x, b)
    setattr(b, field, getattr(p, quantity)-1); x[:] = np.nan
    with pytest.raises((ValueError, RuntimeError), match='budget'): source.forward(x, b)
    assert inventory(source) == snapshot


def test_cache_storage_is_reserved_before_source_input_copy_and_no_bad_input_warms_it():
    a = prepared_case(); p = source_plan(a)
    assert p.preparation_cache_bytes > 0
    a['budget'].memory_bytes = p.modeled_peak_bytes-1; a['q'][:] = np.nan
    with pytest.raises((ValueError, RuntimeError), match='budget'): manta.PeriodicGDFTiles(**a)
    a = prepared_case(); source = manta.PeriodicGDFTiles(**a); x, _ = inputs(source); x[:] = np.nan
    with pytest.raises(ValueError, match='finite'): source.forward(x, a['budget'])
    assert inventory(source)[:4] == (0, 0, 0, 0)


def test_exact_and_reciprocal_only_sources_have_no_preparation_cache():
    a = prepared_case(); a['options'].coupling_multipole_error = 0.; exact = manta.PeriodicGDFTiles(**a)
    assert not exact.preparation.enabled and exact.plan.preparation_cache_bytes == 0
    a['options'].coupling_multipole_error = 1.; physical = manta.PeriodicGDFKernelOptions(); physical.kernel = 'erf'; physical.omega = .3
    reciprocal = manta.PeriodicGDFTiles(**a, physical_kernel=physical)
    assert not reciprocal.preparation.enabled and reciprocal.plan.preparation_cache_bytes == 0


def test_always_near_keeps_moments_lazy_and_direct_values_unchanged():
    a = prepared_case(); e = a['options'].coupling_multipole_extents;e.maximum_radius = e.minimum_radius = .1
    source = manta.PeriodicGDFTiles(**a); x, _ = inputs(source); source.forward(x, a['budget'])
    assert source.preparation.pair_extents > 0 and source.preparation.moment_panels == 0
    a['options'].coupling_multipole_error = 0.
    np.testing.assert_allclose(assemble(source), assemble(manta.PeriodicGDFTiles(**a)), atol=1e-13, rtol=1e-14)


def prepared_job(directory, method='uhf', species='Li', limit=100):
    job, args, k = hierarchy_job(directory, method, species, limit)
    args['options'].fitting.auxiliary_cutoff_bohr = 12.
    job['periodic_fitting'] = dataclasses.replace(job['periodic_fitting'], auxiliary_cutoff_bohr=12.)
    return job, args, k


def check_public_receipt(result):
    r = result.native.receipt.fitted_fock
    assert r.coupling_multipole_moment_panels == r.coupling_multipole_envelope_evaluations == 0
    assert r.coupling_multipole_reused_moment_panels > 0 and r.coupling_multipole_reused_extents > 0
    evidence = result.integral_provenance.get('coupling_approximation', result.integral_provenance.get('fitted_fock'))
    for field in ('coupling_multipole_moment_panels', 'coupling_multipole_envelope_evaluations', 'coupling_multipole_reused_moment_panels', 'coupling_multipole_reused_extents'):
        assert evidence[field] == getattr(r, field)


@pytest.mark.parametrize('method,species', [('rhf', 'Be'), ('uhf', 'Li')])
def test_public_hf_reuses_gaussian_preparation_in_complete_grouped_scf(tmp_path, method, species):
    from tests.test_manta_periodic_ri_domains import explicit
    from tests.test_manta_periodic_rhf import _nuclear_reference
    job, args, k = prepared_job(tmp_path, method, species)
    domains = getattr(manta, 'build_periodic_ri_'+method+'_domains')(**args); a = explicit(args, domains)
    _, _, build, _, _ = multipole_hf_reference(a, k, method)
    result = vq.run_periodic_job(**job); assert result.converged
    check_public_receipt(result); r = result.native.receipt.fitted_fock
    assert 0 < r.coupling_multipole_far_blocks < r.coupling_multipole_images
    f, e = build(result.native.density)
    np.testing.assert_allclose(result.native.fock, f, atol=8e-11, rtol=0)
    assert result.energy == pytest.approx(sum(_nuclear_reference(a)[0])+sum(e), abs=8e-11)


@pytest.mark.parametrize('spins', [1, 2])
def test_public_ks_reuses_same_preparation_with_shared_xc(tmp_path, spins):
    from tests.test_manta_periodic_ri_ks_domains import explicit
    job, args, a, k, old = hierarchy_ks_job(tmp_path, spins)
    explicit(args, a, old); _, _, build, _, _ = multipole_ks_reference(a, k, old)
    result = vq.run_periodic_job(**job); assert result.converged
    check_public_receipt(result)
    np.testing.assert_allclose(result.native.fock, build(result.native.density)[0], atol=8e-11, rtol=0)



def test_typed_reuse_control_keeps_streaming_available():
    for value in (1, 'true', None):
        with pytest.raises(TypeError, match='boolean'):
            manta.PeriodicFittingOptions(.2, 8., coupling_multipole_reuse=value)
    assert manta.PeriodicFittingOptions(.2, 8.).coupling_multipole_reuse is True
    assert manta.PeriodicFittingOptions(.2, 8., coupling_multipole_reuse=False).coupling_multipole_reuse is False
