"""Actual selected domains -> pair-frame Eq.39 -> native coupled pair MP2."""

from __future__ import annotations

import gc

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_gaussian_pair_domain_builder import _case, _build, _pair_options, _pair_caps, _embedding
from tests.test_periodic_gaussian_selected_local_ccsd_t import _dense_case
from tests.test_periodic_gaussian_pair_mp2 import _controls as _legacy_controls, _run as _legacy_run
from tests.test_periodic_gaussian_embedded_pair_pnos import _options as _pno_options, _caps as _pno_caps


def _physical(nk=2, *, kind="he2", full=False, cut=.2, tail=.1):
    """Capture the independent AO-integral oracle before common owners move."""
    b = _case(nk,kind=kind,full=full,cut=cut,tail=tail,build=False)
    b.expected_dense = _dense_case(b)
    b.builder = _build(b)
    return b


def _controls(b, *, cutoff=0.0, maximum_iterations=None):
    old, live, oc = _legacy_controls(cutoff=cutoff)
    options = core._PeriodicGaussianDomainPairMP2Options()
    options.domain, options.pnos = _pair_options(), _pno_options(cutoff=cutoff)
    options.projection, options.solver = old.projection, old.solver
    options.maximum_occupied_virtual_fock_norm = old.maximum_occupied_virtual_fock_norm
    if maximum_iterations is not None:
        solver = options.solver
        solver.maximum_iterations = maximum_iterations
        options.solver = solver
    caps = core._PeriodicGaussianDomainPairMP2Caps()
    domain = _pair_caps(b)
    domain.maximum_owned_numerical_bytes = 2**21
    domain.maximum_control_storage_bytes = 2**23
    union = domain.pair_union
    union.maximum_control_storage_bytes = 2**23
    domain.pair_union = union
    embedding = domain.embedding
    embedding.maximum_control_storage_bytes_per_worker = 2**23
    domain.embedding = embedding
    caps.domain = domain
    pnos = _pno_caps()
    pnos.maximum_control_storage_bytes_per_worker = 2**23
    pnos.maximum_per_worker_inventoried_bytes = 2**25
    caps.pnos = pnos
    projection = oc.projection
    projection.maximum_per_worker_inventoried_bytes = 2**25
    caps.projection, caps.solver = projection, oc.solver
    caps.maximum_owned_numerical_bytes = 2**22
    caps.maximum_control_storage_bytes = 2**23
    caps.maximum_per_worker_inventoried_bytes, caps.maximum_node_inventoried_bytes = 2**25, 2**26
    caps.maximum_pair_count, caps.maximum_integral_calls = 32, 4096
    caps.maximum_progress_callbacks, caps.maximum_work_units = 512, 10**15
    return options, live, caps


def _plan(b, controls=None, **changes):
    args = dict(reference=b.reference,basis=b.basis,provider=b.provider,builder=b.builder)
    args.update(changes)
    o,l,c = _controls(b) if controls is None else controls
    return core._plan_periodic_gaussian_domain_pair_mp2(**args,options=o,live=l,caps=c)


def _run(b, controls=None, progress=None, **changes):
    args = dict(reference=b.reference,basis=b.basis,provider=b.provider,builder=b.builder)
    args.update(changes)
    o,l,c = _controls(b) if controls is None else controls
    return core._run_periodic_gaussian_domain_pair_mp2(**args,options=o,live=l,caps=c,progress=progress)


def _dense_pair_solution(b, result):
    """Independent linear Galerkin MP2 system in the exported pair frames."""
    o = len(b.rows)
    f = b.basis.fock_copy()[:o,:o]
    keys = [(i,j) for i in range(o) for j in range(i,o)]
    spaces = [result.pair(i,j) for i,j in keys]
    ranks = [s.memory.retained_dimension for s in spaces]
    offsets = np.cumsum([0]+[r*r for r in ranks])
    matrices = [s.coefficients_copy() for s in spaces]
    energies = [s.energies_copy() for s in spaces]
    g = [s.exchange_integrals_copy() for s in spaces]
    lookup = {key:a for a,key in enumerate(keys)}
    def amplitudes(vector,i,j):
        slot = lookup[tuple(sorted((i,j)))]
        t = vector[offsets[slot]:offsets[slot+1]].reshape(ranks[slot],ranks[slot])
        return slot,t if i<=j else t.T
    def action(vector):
        output = []
        for a,(i,j) in enumerate(keys):
            _,t = amplitudes(vector,i,j)
            residual = (energies[a][:,None]+energies[a][None,:])*t
            for k in range(o):
                source,tik = amplitudes(vector,i,k)
                overlap = matrices[a].T@matrices[source]
                residual -= f[k,j]*(overlap@tik@overlap.T)
                source,tkj = amplitudes(vector,k,j)
                overlap = matrices[a].T@matrices[source]
                residual -= f[i,k]*(overlap@tkj@overlap.T)
            output.extend(residual.ravel())
        return np.array(output)
    rhs = -np.concatenate([block.ravel() for block in g])
    if len(rhs):
        operator = np.column_stack([action(row) for row in np.eye(len(rhs))])
        expected = np.linalg.solve(operator,rhs)
    else:
        expected = rhs
    energy = 0.0
    for a,(i,j) in enumerate(keys):
        t = expected[offsets[a]:offsets[a+1]].reshape(ranks[a],ranks[a])
        energy += (1 if i==j else 2)*np.sum(g[a]*(2*t-t.T))
    return expected,energy


@pytest.mark.parametrize("kind,nk", [("he",1),("he",2),("he",3),("he2",1),("he2",2)])
@pytest.mark.parametrize("full", [False,True])
def test_selected_domains_feed_complete_coupled_equations(kind,nk,full):
    b = _case(nk,kind=kind,full=full)
    events = []
    result = _run(b,progress=events.append)
    assert result.converged and result.domain_generated and result.memory.domain_generated
    expected,energy = _dense_pair_solution(b,result)
    actual = np.concatenate([result.solver.pair_copy(i,j).ravel()
                             for i in range(len(b.rows)) for j in range(i,len(b.rows))])
    np.testing.assert_allclose(actual,expected,atol=3e-11,rtol=3e-9)
    assert result.solver.final_snapshot.correlation_energy == pytest.approx(energy,abs=3e-11)
    assert result.total_energy_per_cell == pytest.approx(b.hf.state.reference_energy_per_cell+energy/nk,abs=3e-11)
    d = result.diagnostics
    o,n = len(b.rows),b.basis.memory.virtual_count
    frames = [result.pair(i,j) for i in range(o) for j in range(i,o)]
    assert all(p.embedded_generation for p in frames)
    assert d.generation_dimension_sum == sum(p.memory.generation_dimension for p in frames)
    assert d.retained_pair_bytes == sum(p.memory.retained_output_bytes for p in frames)
    assert d.retained_generation_coefficient_bytes == sum(
        8*p.memory.generation_dimension*p.memory.retained_dimension for p in frames)
    diagonal = [result.diagonal_generation_embedding(i) for i in range(o)]
    assert d.diagonal_generation_dimension_sum == sum(e.memory.pair_dimension for e in diagonal)
    assert d.retained_generation_embedding_bytes == 8*(n+1)*d.diagonal_generation_dimension_sum
    assert d.retained_generation_embedding_bytes == sum(e.memory.output_numerical_bytes for e in diagonal)
    assert result.domain_builder_identity_sha256 == b.builder.identity_sha256
    assert not result.production_dlpno and not result.infinite_source_accuracy_certified
    stage = core._PeriodicGaussianPairMP2Stage
    assert [e.stage for e in events].count(stage.PAIR_DOMAIN_READY) == len(frames)
    assert [e.stage for e in events].count(stage.PAIR_COMPLETE) == len(frames)
    assert events[0].stage == stage.BEGIN and events[-1].stage == stage.FINISHED
    assert len(events) == d.completed_progress_callbacks <= result.memory.progress_callback_upper_bound
    if full:
        assert all(p.memory.generation_dimension == n for p in frames)
        old = _legacy_run(b,b.provider)
        assert result.correlation_energy_per_cell == pytest.approx(old.correlation_energy_per_cell,abs=3e-11)


def test_actual_rectangular_domains_and_positive_cut_empty_pnos_keep_distinct_dimensions():
    b = _case()
    full = _run(b)
    assert any(full.pair(i,i).memory.generation_dimension < b.basis.memory.virtual_count for i in range(len(b.rows)))
    empty = _run(b,_controls(b,cutoff=1.0))
    assert empty.converged and empty.diagnostics.zero_rank_pairs == empty.memory.pair_count
    assert empty.solver.final_snapshot.correlation_energy == 0
    assert empty.diagnostics.generation_dimension_sum == full.diagnostics.generation_dimension_sum
    assert empty.diagnostics.retained_generation_coefficient_bytes == 0
    for i in range(len(b.rows)):
        assert empty.diagonal_generation_embedding(i).identity_sha256 == full.diagonal_generation_embedding(i).identity_sha256


@pytest.mark.parametrize("field,reported", [
    ("maximum_owned_numerical_bytes","peak_owned_numerical_bytes"),
    ("maximum_control_storage_bytes","control_storage_reservation_bytes"),
    ("maximum_per_worker_inventoried_bytes","per_worker_inventoried_bytes"),
    ("maximum_node_inventoried_bytes","required_node_memory_bytes"),
    ("maximum_pair_count","pair_count"),("maximum_integral_calls","integral_calls_upper_bound"),
    ("maximum_progress_callbacks","progress_callback_upper_bound"),("maximum_work_units","work_units_upper_bound"),
])
def test_every_outer_cap_exact_and_minus_one_before_first_progress(field,reported):
    b,events = _case(1,kind="he"),[]
    controls = _controls(b)
    required = getattr(_plan(b,controls),reported)
    setattr(controls[2],field,required)
    _run(b,controls)
    setattr(controls[2],field,required-1)
    with pytest.raises((ValueError,RuntimeError),match="cap"):
        _run(b,controls,events.append)
    assert not events


def test_empty_selected_generation_domain_fails_without_forced_virtual():
    b = _case(cut=2.0)
    with pytest.raises((ValueError,RuntimeError),match="empty|zero|domain|rank"):
        _run(b)


def test_nonconvergence_cancellation_and_detached_diagonal_owner_lifetime():
    b = _case()
    result = _run(b,_controls(b,maximum_iterations=1))
    assert not result.converged and not result.periodic_energy_per_cell
    with pytest.raises(RuntimeError,match="converged"):
        _ = result.total_energy_per_cell
    def cancel(event):
        if event.stage == core._PeriodicGaussianPairMP2Stage.PAIR_DOMAIN_READY:
            raise RuntimeError("stop selected-domain diagnostic")
    with pytest.raises(RuntimeError,match="stop selected-domain diagnostic"):
        _run(b,progress=cancel)
    embedding = result.diagonal_generation_embedding(0)
    expected = embedding.coefficients_copy()
    del b,result
    gc.collect()
    np.testing.assert_array_equal(embedding.coefficients_copy(),expected)
