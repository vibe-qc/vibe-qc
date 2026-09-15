"""Tiny selected-real-basis certificates; synthetic mean-field algebra only."""

from __future__ import annotations

import gc
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_correlation_density_factors import _occupied_columns, _virtual, _virtual_columns
from tests.test_periodic_correlation_pao_domain import _make as _make_domain, _options as _domain_options, _reference
from tests.test_periodic_correlation_pao_space import _geometry, _make as _make_space
from tests.test_periodic_correlation_wannier import _make as _make_wannier, _options as _wannier_options


def _options(**changes):
    value = core._PeriodicCorrelationRealLocalBasisOptions()
    for field in ("coefficient_tr_absolute_tolerance", "coefficient_tr_relative_tolerance",
                  "orthonormality_absolute_tolerance", "orthonormality_relative_tolerance",
                  "fock_absolute_tolerance", "fock_relative_tolerance", "maximum_fock_projection_error"):
        setattr(value, field, 1e-10)
    for field, setting in changes.items():
        setattr(value, field, setting)
    return value


def _caps(plan):
    value = core._PeriodicCorrelationRealLocalBasisCaps()
    value.maximum_owned_numerical_bytes = plan.peak_owned_numerical_bytes
    value.maximum_work_units = plan.work_units
    return value


def _bundle(mesh=(3, 1, 1), *, reduced=False, bad_gauge_k=None):
    reference, domain = _geometry(mesh, reduced=reduced)
    cells = np.array(list(product(*(range(n) for n in mesh))))
    phases = np.sin(2*np.pi*(cells/np.array(mesh)).sum(axis=1))
    gauge = np.exp(0.21j*phases).reshape(len(cells), 1, 1)
    if bad_gauge_k is not None:
        gauge[bad_gauge_k] *= np.exp(0.2j)
    return SimpleNamespace(reference=reference, domain=domain, space=_make_space(reference, domain), gauge=gauge,
        wannier=_make_wannier(reference, gauge, options=_wannier_options(
            require_time_reversal=False, require_real_home_coefficients=False)))


def _plan(b, count, selected):
    return core._plan_periodic_correlation_real_local_basis(b.reference, b.wannier, b.domain, b.space, count, selected)


def _make(b, rows, selected, *, options=None, caps=None, gauge=None):
    rows = np.asarray(rows, dtype=np.uint64)
    return core._make_periodic_correlation_real_local_basis(b.reference, b.wannier,
        b.gauge if gauge is None else gauge, b.domain, b.space, rows, selected,
        _options() if options is None else options, _caps(_plan(b, len(rows), selected)) if caps is None else caps)


def _oracle(b, rows, selected):
    n = len(rows) + selected.count
    s, f = np.zeros((n,n), complex), np.zeros((n,n), complex)
    for k in range(b.reference.state.n_kpoints):
        c = np.column_stack((_occupied_columns(b,k,rows), _virtual_columns(b,k,selected)))
        s += c.conj().T @ b.reference.state.overlap(k) @ c / b.reference.state.n_kpoints
        f += c.conj().T @ b.reference.state.fock(k) @ c / b.reference.state.n_kpoints
    return s, f


@pytest.mark.parametrize("mesh", [(1,1,1), (2,1,1), (3,1,1), (2,2,1), (2,2,2)])
@pytest.mark.parametrize("reduced", [False, True])
def test_actual_real_basis_metric_fock_and_every_trim_against_independent_oracle(mesh, reduced):
    b = _bundle(mesh, reduced=reduced)
    if reduced and mesh == (3,1,1):
        # The larger domain has a degenerate k/-k eigenspace whose valid
        # complex semicanonical gauge is not a real local basis (see the
        # negative regression below). This distinct two-column domain has
        # nondegenerate projected energies and actually real orbitals.
        b.domain = _make_domain(b.reference,np.array([[0,2],[0,3]],np.uint64))
        b.space = _make_space(b.reference,b.domain)
    nk = b.reference.state.n_kpoints
    rows = [[0,nk-1], [0,0]] if nk > 1 else [[0,0]]
    selected = _virtual(0, min(2,b.space.retained_dimension), nk//2)
    result = _make(b,rows,selected)
    metric, raw = _oracle(b,rows,selected)
    np.testing.assert_allclose(metric, np.eye(len(rows)+selected.count), atol=5e-12)
    np.testing.assert_allclose(raw.imag,0,atol=3e-12)
    np.testing.assert_allclose(result.fock_copy(), (raw.real+raw.real.T)/2, atol=5e-12)
    np.testing.assert_array_equal(result.fock_copy(),result.fock_copy().T)
    assert result.diagnostics.inspected_kpoints == nk
    assert result.diagnostics.inspected_trim_points == int(np.prod([2 if n%2 == 0 else 1 for n in mesh]))
    assert not result.hf_hamiltonian_match_certified
    assert result.diagnostics.maximum_fock_projection_error <= result.options.maximum_fock_projection_error


def test_real_domain_with_complex_degenerate_semicanonical_gauge_is_not_a_real_basis():
    b = _bundle((3,1,1),reduced=True)
    selected = _virtual(0,2,1)
    assert np.max(abs(b.domain.overlap_copy().imag)) < 1e-14
    assert np.max(abs(b.domain.fock_copy().imag)) < 1e-14
    assert np.max(abs(b.space.coefficients_copy().imag)) > 1e-3
    c1,c2 = _virtual_columns(b,1,selected),_virtual_columns(b,2,selected)
    assert np.max(abs(c2-c1.conj())) > 1e-3
    with pytest.raises(ValueError,match="actual orbital coefficients.*time reversal"):
        _make(b,[[0,2],[0,0]],selected)


@pytest.mark.parametrize("mesh,k", [((3,1,1),0), ((3,1,1),1), ((2,1,1),1), ((2,2,2),7)])
def test_actual_occupied_gauge_time_reversal_including_nyquist_is_mandatory(mesh,k):
    b = _bundle(mesh,bad_gauge_k=k)
    with pytest.raises(ValueError,match="actual orbital coefficients.*time reversal"):
        _make(b,[[0,0]],_virtual())


def test_real_domain_matrices_do_not_certify_actual_virtual_orbitals():
    reference, *_ = _reference((3,1,1), reduced=True, mixed=False, broken_projector_tr=True)
    domain = _make_domain(reference,np.array([[0,2],[0,3]],np.uint64),options=_domain_options(
        require_time_reversal=False,require_real_matrices=False))
    np.testing.assert_array_equal(domain.overlap_copy().imag,0)
    np.testing.assert_array_equal(domain.fock_copy().imag,0)
    gauge = np.ones((3,1,1),complex)
    b = SimpleNamespace(reference=reference,domain=domain,space=_make_space(reference,domain),gauge=gauge,
        wannier=_make_wannier(reference,gauge))
    with pytest.raises(ValueError,match="actual orbital coefficients.*time reversal"):
        _make(b,[[0,0]],_virtual(0,b.space.retained_dimension))


def test_direct_fock_reality_is_not_inferred_from_real_coefficients():
    b = _bundle()
    old = b.reference.state
    data = core._PeriodicRestrictedMeanFieldInput()
    for name in ("calculation_identity","reference_kind","normalization","periodic_dimension","mesh","is_shift",
                 "reciprocal_lattice","converged","n_basis","n_effective_orbitals","electrons_per_cell","reference_energy_per_cell"):
        setattr(data,name,getattr(old,name))
    data.minimum_band_gap_hartree = 0.1
    data.validation_tolerance_version = core._PERIODIC_MEAN_FIELD_VALIDATION_TOLERANCE_VERSION
    for k in range(old.n_kpoints):
        eps = np.array(old.orbital_energies(k))
        if k == 1:
            eps[:2] -= 0.1  # C remains TR-compatible; the occupied F bands do not.
        s,c = old.overlap(k),old.coefficients(k)
        f = s @ c @ np.diag(eps) @ c.conj().T @ s
        data.add_kpoint(old.kpoint_cartesian(k),old.uniform_weight,s,f,c,eps,old.occupations(k),
            old.frozen_core_mask(k),old.correlated_occupied_mask(k),old.virtual_mask(k))
    state = core._make_periodic_restricted_mean_field_state(data)
    dims = b.reference.dimensions
    dims.external_bytes -= b.reference.state_resident_bytes
    reference = core._make_periodic_correlation_admitted_reference(state,dims,b.reference.budget)
    columns = np.array([b.domain.column(i) for i in range(b.domain.domain_dimension)],np.uint64)
    domain = _make_domain(reference,columns,options=_domain_options(require_time_reversal=False,require_real_matrices=False))
    b.reference,b.domain,b.space = reference,domain,_make_space(reference,domain)
    b.wannier = _make_wannier(reference,b.gauge,options=_wannier_options(require_time_reversal=False,require_real_home_coefficients=False))
    with pytest.raises(ValueError,match="projected Fock matrix is not real"):
        _make(b,[[0,0],[0,1]],_virtual())


def test_explicit_fock_projection_bound_cannot_be_silently_exceeded():
    b = _bundle()
    baseline = _make(b,[[0,0],[0,1]],_virtual(0,2,1))
    bound = baseline.diagnostics.maximum_fock_projection_error
    assert bound > 0
    with pytest.raises(ValueError,match="Fock projection.*error budget"):
        _make(b,[[0,0],[0,1]],_virtual(0,2,1),options=_options(
            maximum_fock_projection_error=np.nextafter(bound,0)))


@pytest.mark.parametrize("field",["maximum_owned_numerical_bytes","maximum_work_units"])
def test_exact_caps_precede_invalid_label_and_gauge_scans(field):
    b = _bundle()
    caps = _caps(_plan(b,1,_virtual()))
    setattr(caps,field,getattr(caps,field)-1)
    with pytest.raises(ValueError,match="cap"):
        _make(b,[[2**64-1,0]],_virtual(),caps=caps,gauge=np.full_like(b.gauge,np.nan))


def test_memory_labels_selection_and_foreign_owner_are_sealed():
    b = _bundle()
    p = _plan(b,2,_virtual(0,2))
    o,v,n,nao = 2,2,4,b.reference.state.n_basis
    assert p.retained_output_bytes == 16*o+8*(o*o+v*v+o*v)
    assert p.projection_matrix_bytes == 64*n*n
    assert p.peak_owned_numerical_bytes == 16*o+64*n*n+32*nao*n+32*nao
    with pytest.raises(ValueError,match="duplicate"):
        _make(b,[[0,0],[0,0]],_virtual())
    with pytest.raises(IndexError,match="modular"):
        _make(b,[[0,3]],_virtual())
    other = _bundle()
    b.space = other.space
    with pytest.raises(ValueError,match="state owners"):
        _plan(b,1,_virtual())


@pytest.mark.parametrize("name",["coefficient_tr_absolute_tolerance","coefficient_tr_relative_tolerance",
    "orthonormality_absolute_tolerance","orthonormality_relative_tolerance","fock_absolute_tolerance",
    "fock_relative_tolerance","maximum_fock_projection_error"])
@pytest.mark.parametrize("value",[-1e-12,np.inf,np.nan])
def test_invalid_scientific_controls(name,value):
    b = _bundle((1,1,1))
    with pytest.raises(ValueError,match="tolerance|Fock projection"):
        _make(b,[[0,0]],_virtual(),options=_options(**{name:value}))


def test_detached_fock_copies_and_lifetime():
    b = _bundle()
    selected = _virtual(1,1,1)
    result = _make(b,[[0,2]],selected)
    expected = result.fock_copy()
    result.fock_copy()[:] = 42
    selected.begin = 0
    del b
    gc.collect()
    np.testing.assert_array_equal(result.fock_copy(),expected)
    assert result.occupied(0) == (0,2)
    assert result.virtual_selection.begin == 1
    with pytest.raises(IndexError):
        result.fock(2,0)
