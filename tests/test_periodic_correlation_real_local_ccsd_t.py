"""Connected native finite-Gaussian provider/CCSD/(T) reference tests.

The large-gap mean-field eigenproblems are synthetic and intentionally NOT
authenticated to the Gaussian Hamiltonian. These are connection/oracle tests,
not chemical or per-cell reference energies. No runtime QC imports are added.
"""

from __future__ import annotations

import gc
from itertools import product
from types import SimpleNamespace
from functools import partial

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_correlation_real_local_provider import (
    _basis, _make as _provider, _native_store, _virtual,
)
from tests.test_periodic_correlation_local_factors import _orbital_objects
from tests.test_periodic_aopair_fourier_panel import _basis as _gaussians
from tests.test_periodic_correlation_reciprocal_metric import _CALCULATION_IDENTITY
from tests.test_bounded_restricted_ccsd_solver import _options as _ccsd_options, _run as _ccsd
from tests.test_bounded_restricted_triples_solver import (
    _options as _triples_options, _canonical_spin_energy, _moments, _linear_oracle, _energy,
)


def _large_gap_state(mesh, reciprocal_lattice, *, n_basis=3, rotated=False):
    assert n_basis == 3
    data = core._PeriodicRestrictedMeanFieldInput()
    data.calculation_identity = _CALCULATION_IDENTITY
    data.reference_kind = core._PeriodicMeanFieldReferenceKind.RESTRICTED_HARTREE_FOCK
    data.normalization = core._PeriodicMeanFieldNormalizationConvention.UNNORMALIZED_AO_BLOCH_SUMS_UNIFORM_FULL_BZ_WEIGHTS
    data.periodic_dimension = 3
    data.mesh, data.is_shift = mesh, (0, 0, 0)
    data.reciprocal_lattice = reciprocal_lattice
    data.converged = True
    data.n_basis = data.n_effective_orbitals = 3
    data.electrons_per_cell = 2
    data.reference_energy_per_cell = -5.0
    data.minimum_band_gap_hartree = 0.1
    data.validation_tolerance_version = core._PERIODIC_MEAN_FIELD_VALIDATION_TOLERANCE_VERSION
    eye = np.eye(3, dtype=np.complex128)
    coefficients = eye.copy()
    if rotated:
        c, s = np.cos(.23), np.sin(.23)
        coefficients[1:, 1:] = [[c, -s], [s, c]]
    for address in product(*(range(n) for n in mesh)):
        k = np.array(address) / np.array(mesh)
        energies = np.array([-100.0 + .3*np.cos(2*np.pi*k.sum()), 50., 70.])
        data.add_kpoint(reciprocal_lattice @ k, 1/int(np.prod(mesh)), eye,
            coefficients @ np.diag(energies) @ coefficients.conj().T,
            coefficients, energies, np.array([2., 0., 0.]),
            [0, 0, 0], [1, 0, 0], [0, 1, 1])
    return core._make_periodic_restricted_mean_field_state(data)


def _setup(tmp_path, monkeypatch, mesh=(3, 1, 1), *, rotated=False):
    from tests import test_periodic_correlation_three_center as three
    from tests import test_periodic_correlation_local_factors as local
    ao = _gaussians([
        (0, (.1, -.2, .15), [.55], [.7], True),
        (0, (.6, .3, -.25), [.7], [.8], True),
        (0, (-.15, .4, .2), [1.1], [.65], True),
    ])
    original_bundle = local._bundle
    with monkeypatch.context() as patch:
        patch.setattr(three, "_state", partial(_large_gap_state, rotated=rotated))
        patch.setattr(local, "_bundle", lambda **kw: original_bundle(ao=ao, **kw))
        b = _native_store(tmp_path, mesh=mesh, cutoff=.9)
    objects = _orbital_objects(b.reference, domain_columns=[[0, 1], [0, 2]])
    b.domain, b.space, b.wannier, b.gauge = objects.domain, objects.space, objects.wannier, objects.gauge
    labels = [[0, 0]] if int(np.prod(mesh)) == 1 else [[0, 0], [0, 1]]
    basis = _basis(b, labels, _virtual(0, 2))
    provider = _provider(b, basis)
    # All builder owners, reader, domains and temporary inputs leave scope.
    return SimpleNamespace(reference=b.reference, basis=basis, provider=provider)


def _options(**changes):
    options = core._PeriodicCorrelationRealLocalCCSDTOptions()
    options.ccsd, options.triples = _ccsd_options(), _triples_options()
    options.maximum_additional_fock_projection_norm = 1e-10
    for key, value in changes.items():
        setattr(options, key, value)
    return options


def _inventory(other=0):
    inventory = core._PeriodicCorrelationRealLocalCCSDTInventory()
    inventory.other_live_numerical_bytes_per_replica = other
    return inventory


def _plan(b, options=None, inventory=None):
    return core._plan_periodic_correlation_real_local_ccsd_t(b.reference, b.basis, b.provider,
        _options() if options is None else options, _inventory() if inventory is None else inventory)


def _caps(plan):
    caps = core._PeriodicCorrelationRealLocalCCSDTCaps()
    for name, value in {
        "maximum_owned_numerical_bytes": plan.peak_owned_numerical_bytes,
        "maximum_total_numerical_bytes": plan.total_live_numerical_bytes,
        "maximum_node_numerical_bytes": plan.required_node_numerical_bytes,
        "maximum_integral_calls": plan.integral_calls_upper_bound,
        "maximum_work_units": plan.work_units_upper_bound,
    }.items():
        setattr(caps, name, value)
    return caps


def _run(b, *, options=None, inventory=None, caps=None, progress=None):
    options = _options() if options is None else options
    inventory = _inventory() if inventory is None else inventory
    caps = _caps(_plan(b, options, inventory)) if caps is None else caps
    return core._solve_periodic_correlation_real_local_ccsd_t(
        b.reference, b.basis, b.provider, options, inventory, caps, progress)


def _case(b):
    o, v = b.basis.memory.occupied_count, b.basis.memory.virtual_count
    n = o + v
    fock = np.array([[b.basis.fock(i, j) for j in range(n)] for i in range(n)])
    fock[:o, o:] = fock[o:, :o] = 0
    fock[o:, o:] = np.diag(np.diag(fock[o:, o:]))
    eri = np.array([b.provider.integral(i, j, k, l).value
                   for i, j, k, l in product(range(n), repeat=4)]).reshape((n,)*4)
    rows = b.provider.rows_copy()
    factors = np.zeros((rows.shape[0], n, n))
    for h, (i, j) in enumerate((i, j) for i in range(n) for j in range(i, n)):
        factors[:, i, j] = factors[:, j, i] = rows[:, h]
    return dict(o=o, v=v, fock=fock, foo=np.ascontiguousarray(fock[:o, :o]),
        fvv=np.ascontiguousarray(fock[o:, o:]), fov=np.zeros((o, v)), eri=eri, factors=factors)


@pytest.mark.parametrize("mesh", [(1, 1, 1), (2, 1, 1), (3, 1, 1)])
def test_native_gaussian_rows_to_ccsd_and_coupled_triples_matches_independent_oracles(tmp_path, monkeypatch, mesh):
    b = _setup(tmp_path, monkeypatch, mesh)
    events = []
    result = _run(b, progress=events.append)
    assert result.converged and result.triples_evaluated
    assert result.ccsd.final_snapshot.converged
    case = _case(b)
    direct = _ccsd(case)
    np.testing.assert_allclose(result.ccsd.t1, direct.t1, atol=3e-14, rtol=1e-12)
    np.testing.assert_allclose(result.ccsd.t2, direct.t2, atol=3e-14, rtol=1e-12)
    assert result.ccsd.final_snapshot.correlation_energy == pytest.approx(direct.final_snapshot.correlation_energy, abs=3e-13)
    case.update(t1=result.ccsd.t1, t2=result.ccsd.t2)
    w, u = _moments(case)
    expected = _linear_oracle(case, w)
    np.testing.assert_allclose(result.triples.amplitudes, expected, atol=2e-14, rtol=1e-10)
    assert result.triples.final_snapshot.triples_energy == pytest.approx(_energy(expected, w, u), abs=1e-18, rel=1e-7)
    assert result.triples.final_snapshot.triples_energy == pytest.approx(_canonical_spin_energy(case), abs=1e-18, rel=1e-7)
    assert result.triples.ccsd_snapshot_id == result.ccsd.final_snapshot.iteration
    stages = [e.stage for e in events]
    stage = core._PeriodicCorrelationRealLocalCCSDTStage
    assert stages == [stage.CCSD]*result.ccsd.final_snapshot.iteration + [stage.TRIPLES]*result.triples.final_snapshot.iteration
    assert events[-1].ccsd.converged and events[-1].triples.converged
    assert not result.hf_hamiltonian_match_certified and not result.periodic_energy_per_cell
    assert result.provider_identity_sha256 == b.provider.identity_sha256
    assert result.basis_identity_sha256 == b.basis.identity_sha256
    assert result.total_fock_projection_norm_bound >= result.additional_fock_projection_norm_bound
    if mesh[0] == 3:
        assert abs(case["foo"][0, 1]) > .1
        assert abs(result.triples.final_snapshot.triples_energy) > 1e-13


def test_exact_sequential_memory_and_total_scalar_provider_work(tmp_path, monkeypatch):
    b = _setup(tmp_path, monkeypatch)
    p = _plan(b, inventory=_inventory(12345))
    c, t = p.ccsd, p.triples
    assert p.projected_fock_bytes == 8*(2**2 + 2**2 + 2*2)
    assert p.borrowed_basis_bytes == b.basis.memory.retained_output_bytes
    assert p.borrowed_factor_row_bytes == b.provider.memory.retained_row_bytes
    assert p.peak_owned_numerical_bytes == p.projected_fock_bytes + max(
        c.peak_owned_numerical_bytes, c.amplitude_snapshot_bytes + t.peak_owned_numerical_bytes)
    assert p.total_live_numerical_bytes == p.peak_owned_numerical_bytes + p.borrowed_basis_bytes + p.borrowed_factor_row_bytes + 12345
    assert p.required_node_numerical_bytes == p.external_node_numerical_bytes + p.numerical_replicas*p.total_live_numerical_bytes
    assert p.integral_calls_upper_bound == c.integral_calls_upper_bound + t.integral_calls_upper_bound
    assert p.provider_work_units_upper_bound == p.integral_calls_upper_bound*b.provider.memory.scalar_work_units
    assert p.work_units_upper_bound > c.kernel_work_units_upper_bound + t.kernel_work_units_upper_bound + p.provider_work_units_upper_bound
    assert p.retained_output_bytes_upper_bound == c.amplitude_snapshot_bytes + t.amplitude_snapshot_bytes
    result = _run(b, inventory=_inventory(12345))
    assert result.converged


@pytest.mark.parametrize("field", ["maximum_owned_numerical_bytes", "maximum_total_numerical_bytes",
    "maximum_node_numerical_bytes", "maximum_integral_calls", "maximum_work_units"])
@pytest.mark.parametrize("zero", [False, True])
def test_caps_for_both_stages_reject_before_progress(tmp_path, monkeypatch, field, zero):
    b = _setup(tmp_path, monkeypatch)
    caps = _caps(_plan(b))
    setattr(caps, field, 0 if zero else getattr(caps, field)-1)
    events = []
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        _run(b, caps=caps, progress=events.append)
    assert not events


def test_unconverged_ccsd_never_enters_triples_and_returns_evaluated_snapshot(tmp_path, monkeypatch):
    b = _setup(tmp_path, monkeypatch)
    options = _options(ccsd=_ccsd_options(maximum_iterations=1))
    events = []
    result = _run(b, options=options, progress=events.append)
    assert not result.converged and not result.triples_evaluated
    assert len(events) == 1 and not events[0].ccsd.converged
    assert result.ccsd.final_snapshot.correlation_energy == 0
    np.testing.assert_array_equal(result.ccsd.t1, 0)
    np.testing.assert_array_equal(result.ccsd.t2, 0)
    with pytest.raises(RuntimeError, match="not converge"):
        _ = result.triples


def test_triples_exhaustion_is_not_success_and_callback_errors_abort(tmp_path, monkeypatch):
    b = _setup(tmp_path, monkeypatch)
    result = _run(b, options=_options(triples=_triples_options(maximum_iterations=1)))
    assert result.triples_evaluated and not result.converged
    assert result.triples.final_snapshot.iteration == 1
    np.testing.assert_array_equal(result.triples.amplitudes, 0)
    def cancel(event):
        if event.stage == core._PeriodicCorrelationRealLocalCCSDTStage.TRIPLES:
            raise RuntimeError("cancel native connected triples")
    with pytest.raises(RuntimeError, match="cancel native"):
        _run(b, progress=cancel)
    assert _run(b).converged


def test_native_result_subowners_survive_inputs_and_parent_reference(tmp_path, monkeypatch):
    b = _setup(tmp_path, monkeypatch)
    result = _run(b)
    ccsd, triples = result.ccsd, result.triples
    saved1, saved3 = ccsd.t1, triples.amplitudes
    del result, b
    gc.collect()
    np.testing.assert_array_equal(ccsd.t1, saved1)
    np.testing.assert_array_equal(triples.amplitudes, saved3)
    saved1.fill(123)
    assert not np.all(ccsd.t1 == 123)


@pytest.mark.parametrize("bad", [-1., float("nan"), float("inf")])
def test_invalid_projection_and_downstream_controls_fail_before_ccsd(tmp_path, monkeypatch, bad):
    b = _setup(tmp_path, monkeypatch)
    events = []
    with pytest.raises(ValueError, match="audit controls"):
        _run(b, options=_options(maximum_additional_fock_projection_norm=bad), progress=events.append)
    with pytest.raises(ValueError, match="convergence controls"):
        _run(b, options=_options(triples=_triples_options(residual_tolerance=bad)), progress=events.append)
    assert not events


def test_foreign_native_reference_and_basis_are_rejected(tmp_path, monkeypatch):
    b = _setup(tmp_path, monkeypatch)
    other = _setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="same admitted native reference"):
        core._plan_periodic_correlation_real_local_ccsd_t(other.reference, b.basis, b.provider, _options(), _inventory())
    with pytest.raises(ValueError, match="same admitted native reference"):
        core._plan_periodic_correlation_real_local_ccsd_t(b.reference, other.basis, b.provider, _options(), _inventory())


def test_measured_nonzero_fock_projection_budget_and_same_projected_operator(tmp_path, monkeypatch):
    b = _setup(tmp_path, monkeypatch, rotated=True)
    result = _run(b)
    bound = result.additional_fock_projection_norm_bound
    assert 0 < bound < 1e-10
    o, v = b.basis.memory.occupied_count, b.basis.memory.virtual_count
    raw = np.array([[b.basis.fock(i, j) for j in range(o+v)] for i in range(o+v)])
    case = _case(b)
    assert np.linalg.norm(raw-case["fock"]) <= bound
    assert _run(b, options=_options(maximum_additional_fock_projection_norm=bound)).payload_sha256 == result.payload_sha256
    events = []
    with pytest.raises(ValueError, match="projection exceeds"):
        _run(b, options=_options(maximum_additional_fock_projection_norm=np.nextafter(bound, 0.)), progress=events.append)
    assert not events
    direct = _ccsd(case)
    np.testing.assert_allclose(result.ccsd.t2, direct.t2, atol=3e-14, rtol=1e-12)
    case.update(t1=direct.t1, t2=direct.t2)
    assert result.triples.final_snapshot.triples_energy == pytest.approx(_canonical_spin_energy(case), abs=1e-18, rel=1e-7)
