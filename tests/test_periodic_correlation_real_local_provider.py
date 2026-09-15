"""Tiny finite-source real rows, not an HF-matched or production CC method.

The Gaussian AO/auxiliary store is native and physical; the separate admitted
mean-field eigenproblems are synthetic. Decimal tests concern the exact
represented inputs, not image-tail, integration, or SCF error estimates.
"""

from __future__ import annotations

import gc
import sys
from decimal import Decimal, localcontext
from itertools import product

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_aopair_fourier_panel import _basis as _gaussian_basis
from tests.test_periodic_correlation_density_factors import _virtual
from tests.test_periodic_correlation_local_factors import _native_store
from tests.test_periodic_correlation_local_orbital_factors import (
    _make as _panel, _masked_store, _oracle as _gaussian_oracle,
)
from tests.test_periodic_correlation_real_local_basis import _make as _basis, _options as _basis_options
from tests.test_periodic_correlation_wannier import _make as _wannier

pytestmark = pytest.mark.skipif(sys.platform not in ("darwin", "linux"), reason="native private POSIX store")


def _options(**changes):
    o = core._PeriodicCorrelationRealLocalProviderOptions()
    for name in ("reversal_absolute_tolerance", "reversal_relative_tolerance",
                 "conjugacy_absolute_tolerance", "conjugacy_relative_tolerance",
                 "self_q_absolute_tolerance", "self_q_relative_tolerance",
                 "maximum_eri_projection_error", "maximum_scalar_roundoff_error"):
        setattr(o,name,1e-10)
    for name,value in changes.items():
        setattr(o,name,value)
    return o


def _plan(b,basis):
    return core._plan_periodic_correlation_real_local_provider(
        b.reference,b.schedule,b.reader,b.wannier,b.domain,b.space,basis)


def _caps(p):
    c = core._PeriodicCorrelationRealLocalProviderCaps()
    for name,value in {
        "maximum_owned_numerical_bytes":p.peak_owned_numerical_bytes,
        "maximum_work_units":p.work_units,"maximum_factor_panels":p.factor_panels,
        "maximum_tile_visits":p.tile_visits,"maximum_reader_tile_bytes":p.maximum_reader_tile_bytes,
        "maximum_scalar_work_units":p.scalar_work_units,
    }.items():
        setattr(c,name,value)
    return c


def _make(b,basis,*,options=None,caps=None,gauge=None):
    return core._make_periodic_correlation_real_local_provider(
        b.reference,b.schedule,b.reader,b.wannier,b.gauge if gauge is None else gauge,
        b.domain,b.space,basis,_options() if options is None else options,
        _caps(_plan(b,basis)) if caps is None else caps)


def _real_gauge(mesh):
    cells = np.array(list(product(*(range(n) for n in mesh))))
    return np.exp(.21j*np.sin(2*np.pi*(cells/np.array(mesh)).sum(axis=1))).reshape(len(cells),1,1)


def _negative(q,mesh):
    return np.ravel_multi_index(tuple((-np.array(np.unravel_index(q,mesh)))%mesh),mesh)


def _expected_rows(panels,mesh):
    nk,aux,n,_ = panels.shape
    densities = [(i,j) for i in range(n) for j in range(i,n)]
    out = np.zeros((nk*aux,len(densities)))
    for q in range(nk):
        bar = _negative(q,mesh)
        if q > bar:
            continue
        for p in range(aux):
            for h,(i,j) in enumerate(densities):
                x = panels[q,p,i,j]
                if q == bar:
                    out[q*aux+p,h] = x.real
                else:
                    out[q*aux+p,h],out[bar*aux+p,h] = np.sqrt(2)*x.real,np.sqrt(2)*x.imag
    return out,densities


def _decimal_dot(a,b):
    return sum((Decimal(float(x))*Decimal(float(y)) for x,y in zip(a,b)),Decimal(0))


def _decimal_eri(panels,p,q,r,s):
    # General orientation: conj(L_qp)*L_rs, not the same-orientation Gram.
    left,right = panels[:,:,q,p].ravel(),panels[:,:,r,s].ravel()
    return (_decimal_dot(left.real,right.real)+_decimal_dot(left.imag,right.imag),
            _decimal_dot(left.real,right.imag)-_decimal_dot(left.imag,right.real))


@pytest.mark.parametrize("mesh",[(1,1,1),(2,1,1),(3,1,1),(2,2,1)])
def test_actual_panels_all_four_orientations_and_decimal_scalar_bounds(tmp_path,mesh):
    nk = int(np.prod(mesh))
    b = _native_store(tmp_path,mesh=mesh,cutoff=6.5,gauge=_real_gauge(mesh))
    labels = [[0,nk-1],[0,0]] if nk > 1 else [[0,0]]
    selected = _virtual(translation=nk//2)
    basis = _basis(b,labels,selected)
    result = _make(b,basis)
    panels = np.array([_panel(b,labels,selected,q).tensor_copy() for q in range(nk)])
    expected,densities = _expected_rows(panels,mesh)
    np.testing.assert_array_equal(result.rows_copy(),expected)
    bound = result.diagnostics.maximum_eri_projection_error_bound
    assert bound <= result.options.maximum_eri_projection_error
    with localcontext() as context:
        context.prec = 90
        for a,(p,q) in enumerate(densities):
            for bb,(r,s) in enumerate(densities):
                represented = _decimal_dot(expected[:,a],expected[:,bb])
                for reverse_left,reverse_right in product((False,True),repeat=2):
                    pp,qq = (q,p) if reverse_left else (p,q)
                    rr,ss = (s,r) if reverse_right else (r,s)
                    value = result.integral(pp,qq,rr,ss)
                    assert abs(Decimal(value.value)-represented) <= Decimal(value.roundoff_error_bound)
                    re,im = _decimal_eri(panels,pp,qq,rr,ss)
                    error = ((re-Decimal(value.value))**2+im**2).sqrt()
                    assert error <= Decimal(bound)+Decimal(value.roundoff_error_bound)
    assert result.diagnostics.factor_panels_built == nk*b.schedule.shape.auxiliary_tile_count
    assert result.diagnostics.tile_visits == b.schedule.shape.tile_count
    assert result.store_identity_sha256 == b.reader.storage_identity_sha256
    assert result.local_basis_identity_sha256 == basis.local_basis_identity_sha256
    assert result.basis_certificate_identity_sha256 == basis.identity_sha256
    assert result.finite_image_reference and not result.hf_hamiltonian_match_certified
    assert not result.ao_image_source_certified
    if nk == 3:
        assert np.max(abs(panels[1].imag)) > .1
        # A real-only q representative would lose a physically large sine row.
        assert np.linalg.norm(expected[2*b.auxiliary.nbasis:]) > .1
        assert result.diagnostics.maximum_real_row_conversion_norm > 0
    else:
        assert result.diagnostics.maximum_real_row_conversion_norm == 0


def test_actual_gaussian_reciprocal_oracle_for_full_complex_eri(tmp_path):
    mesh = (3,1,1)
    b = _native_store(tmp_path,mesh=mesh,cutoff=6.5,gauge=_real_gauge(mesh))
    labels,selected = [[0,0],[0,2]],_virtual(translation=1)
    result = _make(b,_basis(b,labels,selected))
    gaussian = np.array([_gaussian_oracle(b,labels,selected,q) for q in range(3)])
    n = result.memory.orbital_count
    for p,q,r,s in product(range(n),repeat=4):
        expected = np.vdot(gaussian[:,:,q,p],gaussian[:,:,r,s])
        np.testing.assert_allclose(result.integral(p,q,r,s).value,expected,atol=2e-10,rtol=5e-11)


@pytest.mark.parametrize("reduced",[False,True])
def test_frozen_and_reduced_native_store_selection_is_preserved(tmp_path,reduced):
    b = _masked_store(tmp_path,reduced)
    b.gauge = np.ones_like(b.gauge)
    b.wannier = _wannier(b.reference,b.gauge)
    labels,selected = [[0,1],[0,0]],_virtual(0,min(2,b.space.retained_dimension),1)
    basis = _basis(b,labels,selected)
    result = _make(b,basis)
    gaussian = np.array([_gaussian_oracle(b,labels,selected,q) for q in range(2)])
    n = result.memory.orbital_count
    for p,q,r,s in product(range(n),repeat=4):
        expected = np.vdot(gaussian[:,:,q,p],gaussian[:,:,r,s])
        np.testing.assert_allclose(result.integral(p,q,r,s).value,expected,atol=3e-10,rtol=5e-11)
    assert b.reference.state.n_frozen_core == 1
    assert b.reference.state.n_effective_orbitals == (3 if reduced else 4)


def test_ragged_auxiliary_storage_and_nonzero_virtual_slice(tmp_path):
    auxiliary = _gaussian_basis([
        (0,(0.,0.,0.),[.6],[1.],True), (0,(.3,-.2,.1),[.9],[1.],True),
        (0,(-.1,.4,.2),[1.3],[1.],True)])
    b = _native_store(tmp_path,mesh=(3,1,1),full_domain=True,auxiliary=auxiliary,auxiliary_block=2)
    labels,selected = [[0,1],[0,2]],_virtual(1,2,1)
    basis = _basis(b,labels,selected)
    result = _make(b,basis)
    panels = np.array([_panel(b,labels,selected,q).tensor_copy() for q in range(3)])
    expected,_ = _expected_rows(panels,(3,1,1))
    np.testing.assert_array_equal(result.rows_copy(),expected)
    assert result.diagnostics.factor_panels_built == 6
    assert result.diagnostics.tile_visits == b.schedule.shape.tile_count
    p,o,v,nao,ab = result.memory,2,2,2,2
    h,k = (o+v)*(o+v+1)//2,3*3
    assert p.retained_row_bytes == 8*k*h
    assert p.norm_workspace_bytes == 24*h
    assert p.building_panel_peak_bytes == 32*ab*(o+v)**2+32*nao*(o+v)+32*nao+16*o
    assert p.retained_partner_panel_bytes == 16*ab*(o+v)**2+16*o
    assert p.peak_owned_numerical_bytes == p.retained_row_bytes+p.norm_workspace_bytes+p.building_panel_peak_bytes+p.retained_partner_panel_bytes
    assert p.live_basis_bytes == basis.memory.retained_output_bytes
    assert p.basis_index_alias_bytes == basis.memory.retained_index_bytes
    assert result.provider_inventory(basis) == (p.retained_row_bytes+16*o,0)
    d,budget = b.reference.dimensions,b.reference.budget
    live = sum(getattr(p,name) for name in ("peak_owned_numerical_bytes","live_basis_bytes","caller_gauge_bytes",
        "live_wannier_bytes","live_domain_bytes","live_space_bytes","live_reader_numeric_bytes","live_reader_control_bytes"))
    assert p.required_node_memory_bytes == (d.external_bytes+d.shared_bytes
        +budget.mpi_ranks*(d.per_rank_bytes+d.localization_window_bytes_per_rank)
        +budget.mpi_ranks*budget.workers_per_rank*live)


@pytest.mark.parametrize("field",["maximum_owned_numerical_bytes","maximum_work_units","maximum_factor_panels",
    "maximum_tile_visits","maximum_reader_tile_bytes","maximum_scalar_work_units"])
@pytest.mark.parametrize("zero",[False,True])
def test_explicit_caps_precede_gauge_hash_and_ao_store_reads(tmp_path,field,zero):
    b = _native_store(tmp_path)
    basis = _basis(b,[[0,0]],_virtual())
    caps = _caps(_plan(b,basis))
    setattr(caps,field,0 if zero else getattr(caps,field)-1)
    with pytest.raises(ValueError,match="cap"):
        _make(b,basis,caps=caps,gauge=np.full_like(b.gauge,np.nan))


def test_live_node_budget_and_native_owner_provenance(tmp_path):
    b = _native_store(tmp_path)
    basis = _basis(b,[[0,0]],_virtual())
    p = _plan(b,basis)
    dims,budget = b.reference.dimensions,b.reference.budget
    dims.external_bytes -= b.reference.state_resident_bytes
    budget.memory_limit_bytes = p.required_node_memory_bytes-1
    b.reference = core._make_periodic_correlation_admitted_reference(b.reference.state,dims,budget)
    with pytest.raises(ValueError,match="node memory"):
        _make(b,basis,caps=_caps(p))
    foreign_directory = tmp_path/"foreign"
    foreign_directory.mkdir(mode=0o700)
    other = _native_store(foreign_directory)
    with pytest.raises(ValueError,match="state owners"):
        _plan(other,basis)


def test_gauge_payload_and_selection_are_bound_to_numerical_basis(tmp_path):
    b = _native_store(tmp_path)
    basis = _basis(b,[[0,0]],_virtual())
    with pytest.raises(ValueError,match="sealed Wannier gauge"):
        _make(b,basis,gauge=-b.gauge)
    b.gauge = -b.gauge
    b.wannier = _wannier(b.reference,b.gauge)
    with pytest.raises(ValueError,match="selected orbital payload"):
        _make(b,basis)


@pytest.mark.parametrize("gate",["reversal","conjugacy","self_q"])
def test_actual_factor_values_are_audited_not_inferred_from_basis_flags(tmp_path,gate):
    mesh = (3,1,1) if gate == "conjugacy" else (1,1,1)
    gauge = np.full((int(np.prod(mesh)),1,1),np.exp(.002j),complex)
    b = _native_store(tmp_path,mesh=mesh,gauge=gauge)
    # Deliberately allow a small non-real occupied basis in this NEGATIVE
    # fixture, so each independent actual factor-value gate is exercised.
    basis = _basis(b,[[0,0]],_virtual(),options=_basis_options(
        coefficient_tr_absolute_tolerance=.01,coefficient_tr_relative_tolerance=.01))
    options = _options(reversal_absolute_tolerance=.1,conjugacy_absolute_tolerance=.1,
                       self_q_absolute_tolerance=.1,maximum_eri_projection_error=1.)
    setattr(options,gate+"_absolute_tolerance",1e-12)
    setattr(options,gate+"_relative_tolerance",1e-12)
    match = {"reversal":"same-q density reversal","conjugacy":"original-row q conjugacy","self_q":"self-q imaginary"}[gate]
    with pytest.raises(ValueError,match=match):
        _make(b,basis,options=options)


def test_projection_and_scalar_error_budgets_cannot_be_silently_exceeded(tmp_path):
    b = _native_store(tmp_path,mesh=(3,1,1),gauge=_real_gauge((3,1,1)),cutoff=6.5)
    basis = _basis(b,[[0,0]],_virtual())
    baseline = _make(b,basis)
    bound = baseline.diagnostics.maximum_eri_projection_error_bound
    assert bound > 0
    with pytest.raises(ValueError,match="ERI projection.*error budget"):
        _make(b,basis,options=_options(maximum_eri_projection_error=np.nextafter(bound,0)))
    tight = _make(b,basis,options=_options(maximum_scalar_roundoff_error=np.nextafter(0.,1.)))
    with pytest.raises(ValueError,match="scalar integral roundoff"):
        tight.integral(0,0,0,0)


@pytest.mark.parametrize("field",["reversal_absolute_tolerance","reversal_relative_tolerance",
    "conjugacy_absolute_tolerance","conjugacy_relative_tolerance","self_q_absolute_tolerance",
    "self_q_relative_tolerance","maximum_eri_projection_error","maximum_scalar_roundoff_error"])
@pytest.mark.parametrize("value",[-1e-12,np.inf,np.nan])
def test_invalid_scientific_controls(tmp_path,field,value):
    b = _native_store(tmp_path,mesh=(1,1,1))
    basis = _basis(b,[[0,0]],_virtual())
    with pytest.raises(ValueError,match="tolerance|error budgets"):
        _make(b,basis,options=_options(**{field:value}))


def test_detached_rows_lifetime_and_different_basis_rejection(tmp_path):
    b = _native_store(tmp_path)
    basis = _basis(b,[[0,0]],_virtual())
    other_basis = _basis(b,[[0,1]],_virtual())
    result = _make(b,basis)
    with pytest.raises(ValueError,match="exact native basis certificate"):
        result.provider_inventory(other_basis)
    expected = result.integral(0,0,1,1).value
    copy = result.rows_copy()
    result.rows_copy()[:] = 100
    del b,basis,other_basis
    gc.collect()
    np.testing.assert_array_equal(result.rows_copy(),copy)
    assert result.integral(0,0,1,1).value == expected
    for indices in ((2,0,0,0),(0,0,0,2)):
        with pytest.raises(IndexError):
            result.integral(*indices)
    with pytest.raises(IndexError):
        result.row(result.memory.row_count,0,0)


def _dot(a,b,*,work=None,error=1e300):
    a,b = np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
    return core._diagnostic_real_local_dot_interval(a,b,256*len(a)+64 if work is None else work,error)


@pytest.mark.parametrize("a,b",[
    ([0.,-0.],[1.,-2.]),
    ([1e16,1.,-1e16],[1.,1.,1.]),
    ([1e150,-1e150,2.**-500,-2.**-500],[1e-150,1e-150,2.**-500,-2.**-500]),
    ([np.nextafter(0.,1.)],[.5]),
    ([np.nextafter(0.,1.)]*4,[.5,1.5,-.5,-1.5]),
    ([2.**-600,-2.**-600],[2.**-600,2.**-600]),
    (list(np.random.default_rng(19).normal(size=256)),list(np.random.default_rng(23).normal(size=256))),
])
def test_nonphysical_scalar_leaf_decimal_cancellation_signed_and_subnormal_oracle(a,b):
    result = _dot(a,b)
    with localcontext() as context:
        # This covers exact decimal expansions of binary64 products down
        # through denorm_min^2, not just their rounded binary64 products.
        context.prec = 2500
        exact = _decimal_dot(a,b)
        assert abs(Decimal(result.value)-exact) <= Decimal(result.roundoff_error_bound)
    assert np.isfinite(result.value) and np.isfinite(result.roundoff_error_bound)


@pytest.mark.parametrize("a,b",[
    ([np.finfo(float).max],[2.]),
    ([np.finfo(float).max],[1.]),  # Outward upper endpoint cannot be finite.
    ([1e308,1e308,-1e308],[1.,1.,1.]),
    ([np.inf],[1.]),([np.nan],[0.]),
])
def test_nonphysical_scalar_leaf_overflow_and_nonfinite_inputs_fail_closed(a,b):
    with pytest.raises(OverflowError,match="non-finite"):
        _dot(a,b)


def test_nonphysical_scalar_leaf_caps_layout_and_error_boundary():
    for work in (0,319):
        with pytest.raises(ValueError,match="work cap"):
            _dot([np.nan],[1.],work=work)
    with pytest.raises(ValueError,match="at least one lane"):
        _dot([],[])
    with pytest.raises(ValueError,match="tiny lane"):
        _dot(np.ones(257),np.ones(257))
    with pytest.raises(ValueError,match="float64"):
        core._diagnostic_real_local_dot_interval(np.ones(1,dtype=np.float32),np.ones(1),320,1.)
    with pytest.raises(ValueError,match="contiguous"):
        core._diagnostic_real_local_dot_interval(np.ones(4)[::2],np.ones(2),576,1.)
    unaligned = np.ndarray((1,),dtype=np.float64,buffer=np.zeros(9,np.uint8),offset=1)
    with pytest.raises(ValueError,match="aligned"):
        core._diagnostic_real_local_dot_interval(unaligned,np.ones(1),320,1.)
    baseline = _dot([1.,2.,-1.],[.3,-.1,.2])
    assert baseline.roundoff_error_bound > 0
    with pytest.raises(ValueError,match="roundoff.*error budget"):
        _dot([1.,2.,-1.],[.3,-.1,.2],error=np.nextafter(baseline.roundoff_error_bound,0))
