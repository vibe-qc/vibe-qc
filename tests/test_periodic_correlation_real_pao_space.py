"""Explicit real PAO preparation on tiny synthetic mean-field eigenproblems."""

from __future__ import annotations

import gc
from decimal import Decimal, localcontext
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_correlation_density_factors import _virtual, _virtual_columns
from tests.test_periodic_correlation_pao_domain import _make as _domain, _options as _domain_options, _reference
from tests.test_periodic_correlation_pao_space import _geometry, _make as _complex_space, _options as _algebra
from tests.test_periodic_correlation_real_local_basis import _make as _real_basis
from tests.test_periodic_correlation_wannier import _make as _wannier


def _options(**changes):
    o = core._PeriodicCorrelationRealPAOSpaceOptions()
    o.algebra = _algebra()
    o.maximum_overlap_projection_error = o.maximum_fock_projection_error = 1e-12
    o.maximum_overlap_spectral_uncertainty = 1e-9
    o.maximum_original_metric_error = o.maximum_original_fock_error = 1e-8
    o.maximum_original_projector_relation_error = 1e-8
    o.coefficient_tr_absolute_tolerance = o.coefficient_tr_relative_tolerance = 1e-10
    for name,value in changes.items():
        setattr(o,name,value)
    return o


def _plan(reference,domain,r,options=None):
    return core._plan_periodic_correlation_real_pao_space(reference,domain,r,_options() if options is None else options)


def _caps(plan):
    c = core._PeriodicCorrelationRealPAOSpaceCaps()
    c.maximum_owned_numerical_bytes = plan.peak_owned_numerical_bytes
    c.maximum_work_units = plan.work_units
    return c


def _make(reference,domain,*,options=None,caps=None):
    options = _options() if options is None else options
    return core._make_periodic_correlation_real_pao_space(reference,domain,options,
        _caps(_plan(reference,domain,domain.domain_dimension,options)) if caps is None else caps)


def _decimal_bilinear_error(c,matrix,diagonal):
    n,r = c.shape
    squared = Decimal(0)
    for a,b in product(range(r),repeat=2):
        re,im = Decimal(0),Decimal(0)
        for mu,nu in product(range(n),repeat=2):
            weight = Decimal(float(c[mu,a].real))*Decimal(float(c[nu,b].real))
            re += weight*Decimal(float(matrix[mu,nu].real))
            im += weight*Decimal(float(matrix[mu,nu].imag))
        if a == b:
            re -= Decimal(float(diagonal[b]))
        squared += re*re+im*im
    return squared.sqrt()


@pytest.mark.parametrize("mesh",[(1,1,1),(2,1,1),(3,1,1),(2,2,1),(2,2,2)])
@pytest.mark.parametrize("reduced",[False,True])
def test_actual_real_coefficients_original_metric_fock_and_every_trim(mesh,reduced):
    reference,domain = _geometry(mesh,reduced=reduced)
    result = _make(reference,domain)
    space,diagnostic = result.space,result.diagnostics
    c,energies = space.coefficients_copy(),space.energies_copy()
    s,f = domain.overlap_copy(),domain.fock_copy()
    np.testing.assert_array_equal(c.imag,0)
    np.testing.assert_allclose(c.conj().T@s@c,np.eye(c.shape[1]),atol=3e-10)
    np.testing.assert_allclose(c.conj().T@f@c,np.diag(energies),atol=3e-10)
    with localcontext() as context:
        context.prec = 100
        assert _decimal_bilinear_error(c,s,np.ones(c.shape[1])) <= Decimal(diagnostic.original_metric_frobenius_error_upper_bound)
        assert _decimal_bilinear_error(c,f,energies) <= Decimal(diagnostic.original_fock_frobenius_error_upper_bound)
        g = Decimal(diagnostic.overlap_eigenvector_gram_error_upper_bound)
        assert Decimal(diagnostic.overlap_polar_distance_upper_bound) >= 1-(1-g).sqrt()
    b = SimpleNamespace(reference=reference,domain=domain,space=space)
    selected = _virtual(0,space.retained_dimension)
    cells = np.array(list(product(*(range(n) for n in mesh))))
    for k in range(reference.state.n_kpoints):
        bar = np.ravel_multi_index(tuple((-cells[k])%mesh),mesh)
        np.testing.assert_allclose(_virtual_columns(b,bar,selected),_virtual_columns(b,k,selected).conj(),atol=1e-10)
    assert diagnostic.inspected_kpoints == reference.state.n_kpoints
    assert diagnostic.inspected_trim_points == int(np.prod([2 if n%2 == 0 else 1 for n in mesh]))
    physical_lambda = np.linalg.eigvalsh(s)
    approximate = space.overlap_eigenvalues_copy()
    assert np.max(abs(physical_lambda-approximate)) <= diagnostic.overlap_eigenvalue_error_upper_bound
    assert space.pao_domain_identity_sha256 == domain.pao_domain_identity_sha256
    assert space.pao_space_identity_sha256 == result.identity_sha256
    assert not result.original_operators_unchanged_certified


def test_complex_degenerate_fixture_is_resolved_without_changing_default_complex_path():
    reference,domain = _geometry((3,1,1),reduced=True)
    complex_before = _complex_space(reference,domain)
    assert np.max(abs(complex_before.coefficients_copy().imag)) > 1e-3
    owner = _make(reference,domain)
    c = owner.space.coefficients_copy()
    np.testing.assert_array_equal(c.imag,0)
    # The exact same physical subspace is compared by its gauge-invariant
    # AO-domain metric projector, not by individual degenerate eigenvectors.
    z,s = complex_before.coefficients_copy(),domain.overlap_copy()
    np.testing.assert_allclose(c@c.conj().T@s,z@z.conj().T@s,atol=4e-10)
    complex_after = _complex_space(reference,domain)
    np.testing.assert_array_equal(complex_after.coefficients_copy(),z)
    assert complex_after.pao_space_identity_sha256 == complex_before.pao_space_identity_sha256
    gauge = np.ones((3,1,1),complex)
    b = SimpleNamespace(reference=reference,domain=domain,space=owner.space,gauge=gauge,wannier=_wannier(reference,gauge))
    certified = _real_basis(b,[[0,2],[0,0]],_virtual(0,2,1))
    assert certified.diagnostics.maximum_coefficient_tr_error < 1e-10


def test_prepared_space_flows_through_actual_gaussian_store_and_real_provider(tmp_path):
    from tests.test_periodic_correlation_local_orbital_factors import _masked_store
    from tests.test_periodic_correlation_real_local_provider import _make as _provider

    b = _masked_store(tmp_path,True)
    columns = np.array([b.domain.column(i) for i in range(b.domain.domain_dimension)],np.uint64)
    b.domain = _domain(b.reference,columns)
    owner = _make(b.reference,b.domain)
    b.space = owner.space
    b.gauge = np.ones_like(b.gauge)
    b.wannier = _wannier(b.reference,b.gauge)
    basis = _real_basis(b,[[0,1],[0,0]],_virtual(0,min(2,b.space.retained_dimension),1))
    result = _provider(b,basis)
    assert result.local_basis_identity_sha256 == basis.local_basis_identity_sha256
    assert result.integral(0,2,0,2).value >= 0
    assert not result.hf_hamiltonian_match_certified


def test_rank_cutoff_intervals_reject_boundary_instead_of_forcing_orbitals():
    reference,domain = _geometry((3,1,1),reduced=True)
    baseline = _make(reference,domain)
    eigenvalues = baseline.space.overlap_eigenvalues_copy()
    boundary = _options(algebra=_algebra(rank_absolute_cutoff=float(eigenvalues[1]),rank_relative_cutoff=0.))
    with pytest.raises(RuntimeError,match="rank is ambiguous"):
        _make(reference,domain,options=boundary)
    between = float((eigenvalues[0]+eigenvalues[1])/2)
    result = _make(reference,domain,options=_options(algebra=_algebra(rank_absolute_cutoff=between,rank_relative_cutoff=0.)))
    assert result.space.retained_dimension == 2
    assert result.diagnostics.minimum_rank_margin_lower_bound > 0
    with pytest.raises(RuntimeError,match="zero retained"):
        _make(reference,domain,options=_options(algebra=_algebra(rank_absolute_cutoff=100.,rank_relative_cutoff=0.)))


def test_relative_cutoff_is_propagated_through_largest_eigenvalue_interval():
    reference,domain = _geometry((3,1,1),reduced=True)
    result = _make(reference,domain,options=_options(algebra=_algebra(rank_absolute_cutoff=0.,rank_relative_cutoff=.01)))
    d = result.diagnostics
    exact = .01*np.linalg.eigvalsh(domain.overlap_copy())[-1]
    assert d.rank_cutoff_lower_bound <= exact <= d.rank_cutoff_upper_bound
    assert d.rank_cutoff_lower_bound < d.rank_cutoff_upper_bound


def test_polar_bound_is_nonzero_and_outward_when_one_minus_sqrt_would_cancel():
    reference,_ = _geometry((1,1,1),reduced=True)
    domain = _domain(reference,np.array([[0,2]],np.uint64))
    d = _make(reference,domain).diagnostics
    assert 0 < d.overlap_eigenvector_gram_error_upper_bound < 1e-100
    assert 1-np.sqrt(1-d.overlap_eigenvector_gram_error_upper_bound) == 0
    with localcontext() as context:
        context.prec = 2500
        g = Decimal(d.overlap_eigenvector_gram_error_upper_bound)
        assert Decimal(d.overlap_polar_distance_upper_bound) >= 1-(1-g).sqrt() > 0


@pytest.mark.parametrize("field,diagnostic",[
    ("maximum_overlap_projection_error","overlap_projection_frobenius_upper_bound"),
    ("maximum_fock_projection_error","fock_projection_frobenius_upper_bound"),
    ("maximum_overlap_spectral_uncertainty","overlap_eigenvalue_error_upper_bound"),
    ("maximum_original_fock_error","original_fock_frobenius_error_upper_bound"),
    ("maximum_original_projector_relation_error","original_projector_relation_frobenius_error_upper_bound"),
])
def test_measured_projection_and_original_relation_error_budgets(field,diagnostic):
    reference,domain = _geometry((3,1,1),reduced=True)
    baseline = _make(reference,domain)
    value = getattr(baseline.diagnostics,diagnostic)
    assert value > 0
    with pytest.raises(RuntimeError,match="error budget"):
        _make(reference,domain,options=_options(**{field:np.nextafter(value,0)}))


def test_original_metric_budget_includes_unrotated_and_final_coefficients():
    reference,domain = _geometry((3,1,1),reduced=True)
    baseline = _make(reference,domain)
    value = max(baseline.space.diagnostics.canonical_metric_frobenius_residual,
                baseline.diagnostics.original_metric_frobenius_error_upper_bound)
    with pytest.raises(RuntimeError,match="original metric.*error budget"):
        _make(reference,domain,options=_options(maximum_original_metric_error=np.nextafter(value,0)))


def test_domain_flags_are_not_enough_when_expanded_coefficients_fail_tighter_tr():
    reference,*_ = _reference((3,1,1),reduced=True,mixed=False,broken_projector_tr=True)
    domain = _domain(reference,np.array([[0,2],[0,3]],np.uint64),options=_domain_options(
        time_reversal_absolute_tolerance=.9,time_reversal_relative_tolerance=.9))
    np.testing.assert_array_equal(domain.overlap_copy().imag,0)
    np.testing.assert_array_equal(domain.fock_copy().imag,0)
    with pytest.raises(RuntimeError,match="actual expanded coefficients.*time reversal"):
        _make(reference,domain)


@pytest.mark.parametrize("gate",["require_time_reversal","require_real_matrices"])
def test_qualified_domain_controls_are_required(gate):
    reference,_ = _geometry()
    domain = _domain(reference,np.array([[0,2]],np.uint64),options=_domain_options(**{gate:False}))
    with pytest.raises(ValueError,match="proven domain"):
        _plan(reference,domain,1)


def test_exact_phase_inventory_and_stage_caps():
    reference,domain = _geometry((3,1,1),reduced=True)
    baseline = _make(reference,domain)
    p = _plan(reference,domain,baseline.space.retained_dimension)
    n,r,nao = domain.domain_dimension,baseline.space.retained_dimension,reference.state.n_basis
    assert p.compact.validation_phase_bytes == 32*n*r+40*n+8*r
    assert p.coefficient_tr_phase_bytes == 16*n*r+8*n+8*r+64*nao
    assert p.peak_owned_numerical_bytes == max(p.compact.peak_owned_numerical_bytes,p.coefficient_tr_phase_bytes)
    for field in ("maximum_owned_numerical_bytes","maximum_work_units"):
        caps = _caps(p)
        setattr(caps,field,getattr(caps,field)-1)
        with pytest.raises(ValueError,match="cap"):
            _make(reference,domain,caps=caps)
    result = _make(reference,domain,caps=_caps(p))
    np.testing.assert_array_equal(result.space.coefficients_copy(),baseline.space.coefficients_copy())
    d,b = reference.dimensions,reference.budget
    assert p.required_node_memory_bytes == (d.external_bytes+d.shared_bytes
        +b.mpi_ranks*(d.per_rank_bytes+d.localization_window_bytes_per_rank)
        +b.mpi_ranks*b.workers_per_rank*(p.compact.borrowed_domain_bytes+p.peak_owned_numerical_bytes))


def test_undersized_node_budget_is_rejected_before_real_pao_input_creation():
    reference,domain = _geometry()
    baseline = _make(reference,domain)
    p = _plan(reference,domain,baseline.space.retained_dimension)
    dims,budget = reference.dimensions,reference.budget
    dims.external_bytes -= reference.state_resident_bytes
    budget.memory_limit_bytes = p.required_node_memory_bytes-1
    # The static preflight is larger than this tiny PAO leaf. Its earlier
    # gate correctly prevents construction of an underbudget reference;
    # the leaf cannot bypass admission merely to exercise its later check.
    assert budget.memory_limit_bytes < reference.plan.required_memory_bytes
    with pytest.raises(RuntimeError,match="static resource planner returned MemoryExceeded"):
        core._make_periodic_correlation_admitted_reference(reference.state,dims,budget)


def test_distinct_equal_content_state_owner_is_rejected():
    _,domain = _geometry()
    other,_ = _geometry()
    with pytest.raises(ValueError,match="state owners"):
        _plan(other,domain,1)


@pytest.mark.parametrize("field",["maximum_overlap_projection_error","maximum_fock_projection_error",
    "maximum_overlap_spectral_uncertainty","maximum_original_metric_error","maximum_original_fock_error",
    "maximum_original_projector_relation_error","coefficient_tr_absolute_tolerance","coefficient_tr_relative_tolerance"])
@pytest.mark.parametrize("value",[-1.,np.inf,np.nan])
def test_invalid_explicit_controls(field,value):
    reference,domain = _geometry((1,1,1))
    with pytest.raises(ValueError,match="budget|tolerance"):
        _plan(reference,domain,1,_options(**{field:value}))


def test_space_view_pins_owner_and_detached_copies_cannot_change_proof():
    reference,domain = _geometry((3,1,1),reduced=True)
    owner = _make(reference,domain)
    space = owner.space
    expected = space.coefficients_copy()
    identity = space.pao_space_identity_sha256
    space.coefficients_copy()[:] = 0
    del owner,reference,domain
    gc.collect()
    np.testing.assert_array_equal(space.coefficients_copy(),expected)
    assert space.pao_space_identity_sha256 == identity
