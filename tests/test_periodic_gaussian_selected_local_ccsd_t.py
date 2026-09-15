"""Actual finite-Gaussian HF through native selected-real CCSD and (T).

Only one or two He atoms, at most four normalized s functions and three k points.
Finite AO images and reciprocal/Ewald cuts are explicit Hamiltonian
choices, not converged bulk energies or production pair-specific DLPNO.
Dense tensors below are independent tiny test oracles, never factory inputs.
"""

from __future__ import annotations

import gc
from itertools import combinations, product

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from vibeqc.dlpno._ccsd_ref import so_energy, so_residuals
from tests.test_periodic_gaussian_rhf import (
    _bundle as _hf_bundle, _run as _hf_run, _dense_hamiltonian,
)
from tests.test_periodic_gaussian_one_electron import _normalized_basis, _options as _one_options
from tests.test_periodic_gaussian_nuclear import _options as _nuclear_options
from tests.test_periodic_gaussian_source_context import _make as _context, _exact_caps
from tests.test_periodic_correlation_pair_topology import _make_dimensions, _make_budget
from tests.test_periodic_gaussian_localization import _options as _local_options, _caps as _local_caps
from tests.test_periodic_gaussian_bloch_iao import _caps as _iao_caps
from tests.test_periodic_correlation_pao_domain import _make as _domain, _options as _domain_options
from tests.test_periodic_correlation_real_pao_space import _make as _real_space, _options as _space_options
from tests.test_periodic_correlation_real_local_basis import _make as _basis, _options as _basis_options
from tests.test_periodic_correlation_density_factors import _virtual, _occupied_columns, _virtual_columns
from tests.test_periodic_gaussian_local_orbital_factors import _caps as _panel_caps, _dense_panel
from tests.test_periodic_gaussian_fock import _q_data
from tests.test_periodic_correlation_real_local_provider import _options as _real_factor_options
from tests.test_periodic_correlation_real_local_ccsd_t import _options as _correlation_options
from tests.test_bounded_restricted_triples_solver import _moments, _linear_oracle, _energy, _canonical_spin_energy


def _bundle(mesh=(1, 1, 1), *, full=True):
    b = _hf_bundle(mesh)
    b.hf = _hf_run(b)
    assert b.hf.converged
    b.minimal = _normalized_basis([(0, (0, 0, 0), [1.8], [1.0], True)])
    return _admit(b, mesh, full=full)


def _admit(b, mesh, *, full=True):
    nk, nao, nocc = b.context.n_kpoints, b.ao.nbasis, b.mask.shape[1]
    dims = _make_dimensions(mesh, nocc, calculation_identity=b.hf.state.calculation_identity)
    dims.n_basis = dims.n_effective_orbitals = nao
    dims.n_home_virtual = nao-nocc
    dims.n_auxiliary = dims.domain_local_auxiliary_upper_bound = b.auxiliary.nbasis
    dims.domain_ao_support_upper_bound = dims.domain_pao_upper_bound = dims.domain_pno_upper_bound = nao*nk
    dims.domain_local_occupied_upper_bound = nocc*nk
    budget = _make_budget()
    budget.memory_limit_bytes = budget.scratch_limit_bytes = 2**26
    b.reference = core._make_periodic_correlation_admitted_reference(b.hf.state, dims, budget)
    b.columns = np.array([(cell, ao) for cell in range(nk if full else 1) for ao in range(nao)], np.uint64)
    b.rows = np.array([(i, cell) for cell in range(nk if full else 1) for i in range(nocc)], np.uint64)
    b.translation = 0
    b.panel_config = core._PeriodicGaussianLocalOrbitalFactorConfig()
    b.panel_config.ao_pair_block = 3
    b.panel_caps = _panel_caps(b)
    return b


def _he2_bundle(mesh=(1, 1, 1)):
    """Four actual electrons and a nontrivial localized occupied F block."""
    b = _hf_bundle(mesh)
    positions = ((0, 0, 0), (3, 0, 0))
    b.ao = _normalized_basis([(0, p, [a], [1.0], True) for p in positions for a in (1.8, .35)])
    b.auxiliary = _normalized_basis([(0, p, [a], [1.0], True) for p in positions for a in (.6, 1.3)])
    b.minimal = _normalized_basis([(0, p, [1.8], [1.0], True) for p in positions])
    b.system.unit_cell = [core.Atom(2, list(p)) for p in positions]
    source = b.context.options
    # Includes both intersite R=0 AO products, excludes nearest image5bohr.
    source.ao_pair_image_cutoff_bohr = 3.5
    b.context = _context(system=b.system, ao=b.ao, auxiliary=b.auxiliary,
        mesh=core._RegularKMesh(list(mesh)), options=source)
    metric, tile = b.config.metric, b.config.tile
    metric.basis_verification_caps = tile.basis_verification_caps = _exact_caps(b.context)
    b.config.metric, b.config.tile = metric, tile
    options = b.options
    options.one_electron = _one_options(b)
    options.nuclear = _nuclear_options(b, alpha=.6, rcut=3.0, gcut=2.0)
    b.options = options  # SCF controls remain those of the strict He fixture.
    caps, nuclear = b.caps, b.caps.nuclear
    nuclear.maximum_atom_count = 2
    caps.nuclear, b.caps = nuclear, caps
    b.mask = np.zeros((b.context.n_kpoints, 2), np.uint8)
    b.hf = _hf_run(b)
    assert b.hf.converged
    return _admit(b, mesh)


def _localization_options(b):
    return _local_options(dict(reference=b.reference, cutoff=b.context.options.ao_pair_image_cutoff_bohr),
        maximum_iterations=32, riemannian_gradient_tolerance=1e-11)


def _source_caps(b):
    atoms, shells = len(b.system.unit_cell), len(b.ao.shells())+len(b.minimal.shells())
    return _iao_caps(maximum_atom_count=atoms, maximum_shell_count=shells,
        maximum_contraction_count=shells, maximum_primitive_numeric_lanes=2*shells,
        maximum_basis_content_wire_bytes=4096, maximum_borrowed_active_numeric_bytes=256*atoms)


def _controls(b):
    o = core._PeriodicGaussianSelectedLocalCCSDTOptions()
    o.localization, o.domain = _localization_options(b), _domain_options()
    o.space, o.basis = _space_options(), _basis_options()
    factors = core._PeriodicGaussianRealLocalProviderConfig()
    factors.auxiliary_block, factors.panel = 1, b.panel_config
    o.factors, o.real_factors, o.correlation = factors, _real_factor_options(), _correlation_options()
    live = core._PeriodicGaussianSelectedLocalCCSDTLiveInventory()
    live.fixed_backend_margin_bytes_per_worker = 65536
    c = core._PeriodicGaussianSelectedLocalCCSDTCaps()
    c.iao_source, c.localization = _source_caps(b), _local_caps()
    c.maximum_domain_owned_numerical_bytes = 65536
    space = core._PeriodicCorrelationRealPAOSpaceCaps()
    space.maximum_owned_numerical_bytes, space.maximum_work_units = 65536, 10**9
    basis = core._PeriodicCorrelationRealLocalBasisCaps()
    basis.maximum_owned_numerical_bytes, basis.maximum_work_units = 65536, 10**9
    c.space, c.basis = space, basis
    factors = core._PeriodicGaussianRealLocalProviderCaps()
    resources = core._PeriodicGaussianMetricCaps()
    resources.maximum_owned_numeric_bytes = 2**20
    resources.maximum_per_replica_inventoried_bytes = 2**24
    resources.maximum_node_inventoried_bytes = 2**26
    resources.maximum_candidate_evaluations = 10**12
    resources.maximum_work_units = 10**15
    factors.resources, factors.metric, factors.panel = resources, b.config.metric_caps, b.panel_caps
    factors.maximum_factor_panels, factors.maximum_tile_calls = 64, 4096
    factors.maximum_image_candidate_evaluations = 10**15
    factors.maximum_progress_callbacks, factors.maximum_scalar_work_units = 512, 10**8
    c.factors = factors
    correlation = core._PeriodicCorrelationRealLocalCCSDTCaps()
    correlation.maximum_owned_numerical_bytes = 2**20
    correlation.maximum_total_numerical_bytes = 2**24
    correlation.maximum_node_numerical_bytes = 2**26
    correlation.maximum_integral_calls = 10**10
    correlation.maximum_work_units = 10**15
    c.correlation = correlation
    c.maximum_owned_numerical_bytes = 2**23
    c.maximum_per_worker_inventoried_bytes = 2**24
    c.maximum_node_inventoried_bytes = 2**26
    c.maximum_factor_control_storage_bytes = 2**20
    c.maximum_progress_callbacks, c.maximum_work_units = 65536, 10**17
    return o, live, c


def _plan(b, controls=None):
    return core._plan_periodic_gaussian_selected_local_ccsd_t(b.hf, b.reference,
        b.columns, b.rows, b.translation, *(_controls(b) if controls is None else controls))


def _run(b, *, controls=None, callback=None, **changes):
    args = dict(hf=b.hf, reference=b.reference, ao=b.ao, auxiliary=b.auxiliary,
        minimal=b.minimal, system=b.system, domain=b.columns, occupied=b.rows,
        virtual_translation_cell=b.translation)
    args.update(changes)
    options, live, caps = _controls(b) if controls is None else controls
    return core._run_periodic_gaussian_selected_local_ccsd_t(**args,
        options=options, live=live, caps=caps, callback=callback)


def _prepare_leaves(b, *, localization_options=None):
    """Independent leaf sequence exposes coefficients, never alters HF data."""
    localized = core._localize_periodic_gaussian_occupied(b.reference, b.ao,
        b.minimal, b.system, 65536,
        _localization_options(b) if localization_options is None else localization_options,
        _source_caps(b), _local_caps())
    assert localized.converged
    b.localization = localized
    b.gauge, b.wannier = localized.optimizer.gauges_copy(), localized.wannier
    b.domain = _domain(b.reference, b.columns)
    b.real_space = _real_space(b.reference, b.domain)
    b.space = b.real_space.space
    b.selected = _virtual(0, b.space.retained_dimension, b.translation)
    b.basis = _basis(b, b.rows, b.selected)
    return b


def _dense_case(b):
    """Actual AO factors and original F, independently transformed in NumPy.

    Nk^(-3/2) and reversed left density come from the finite-torus Coulomb
    convention. No provider rows, scalar ERI calls or factory F are read.
    """
    panels = np.asarray([_dense_panel(b, source, whitener)
        for source, _, whitener in (_q_data(b, q) for q in range(b.context.n_kpoints))])
    g = np.einsum("xAqp,xArs->pqrs", panels.conj(), panels)
    assert np.max(abs(g.imag)) < 3e-12
    np.testing.assert_allclose(g, g.swapaxes(0, 1), atol=3e-12, rtol=3e-12)
    np.testing.assert_allclose(g, g.swapaxes(2, 3), atol=3e-12, rtol=3e-12)
    o, v, nk = len(b.rows), b.selected.count, b.context.n_kpoints
    raw_f = np.zeros((o+v, o+v), complex)
    for k in range(nk):
        c = np.column_stack((_occupied_columns(b, k, b.rows), _virtual_columns(b, k, b.selected)))
        raw_f += c.conj().T @ b.reference.state.fock(k) @ c / nk
    assert np.max(abs(raw_f.imag)) < 1e-12
    # The connector explicitly budgets this semicanonical/Brillouin
    # projection. Check its magnitude before using the identical operator.
    f = (raw_f.real+raw_f.real.T)/2
    projected = f.copy()
    projected[:o, o:] = projected[o:, :o] = 0
    projected[o:, o:] = np.diag(np.diag(projected[o:, o:]))
    assert np.linalg.norm(projected-raw_f) < 1e-10
    return dict(o=o, v=v, fock=projected, foo=projected[:o, :o],
        fvv=projected[o:, o:], fov=projected[:o, o:], eri=g.real, panels=panels)


def _spin_orbital_check(case, t1, t2):
    """Spin-explicit Stanton residual, independent of the spatial C++ leaf."""
    o, v, g, f = (case[k] for k in ("o", "v", "eri", "fock"))
    n = o+v
    spin_g = np.zeros((2*n,)*4)
    for p, q, r, s in product(range(2*n), repeat=4):
        if p%2 == r%2 and q%2 == s%2:
            spin_g[p, q, r, s] += g[p//2, r//2, q//2, s//2]
        if p%2 == s%2 and q%2 == r%2:
            spin_g[p, q, r, s] -= g[p//2, s//2, q//2, r//2]
    fso = np.kron(f, np.eye(2))
    fod = fso.copy()
    np.fill_diagonal(fod, 0)
    eo, ev = np.diag(fso)[:2*o], np.diag(fso)[2*o:]
    d1 = eo[:, None]-ev[None, :]
    d2 = eo[:, None, None, None]+eo[None, :, None, None]-ev[None, None, :, None]-ev[None, None, None, :]
    one = np.kron(t1, np.eye(2))
    two = np.zeros((2*o, 2*o, 2*v, 2*v))
    for si, sj in product(range(2), repeat=2):
        two[si::2, sj::2, si::2, sj::2] += t2
        two[si::2, sj::2, sj::2, si::2] -= t2.transpose(0, 1, 3, 2)
    occupied, virtual = slice(0, 2*o), slice(2*o, 2*n)
    r1, r2 = so_residuals(fso, fod, spin_g, one, two, occupied, virtual, d1, d2)
    return max(np.max(abs(r1)), np.max(abs(r2))), so_energy(fso, spin_g, one, two, occupied, virtual)


def _original_hamiltonian_check(b, case):
    """Finite-torus normal ordering also checks the multi-k HF normalization."""
    physical = _dense_hamiltonian(b)
    o, n, nk = case["o"], case["o"]+case["v"], b.context.n_kpoints
    h = np.zeros((n, n), complex)
    for k in range(nk):
        c = np.column_stack((_occupied_columns(b, k, b.rows), _virtual_columns(b, k, b.selected)))
        h += c.conj().T @ physical.h[k] @ c / nk
    assert np.max(abs(h.imag)) < 1e-12
    g = case["eri"]
    f = h.real.copy()
    for i in range(o):
        f += 2*g[:, :, i, i]-g[:, i, i, :]
    np.testing.assert_allclose(f, case["fock"], atol=3e-12, rtol=3e-12)
    electronic = 2*np.trace(h[:o, :o]).real
    electronic += sum(2*g[i, i, j, j]-g[i, j, j, i] for i, j in product(range(o), repeat=2))
    assert electronic/nk+physical.nuclear == pytest.approx(b.hf.state.reference_energy_per_cell, abs=4e-12)
    return h.real, physical.nuclear


def _fci_two_electron(h, g):
    """Six determinant second-quantized oracle; no CC equations or amplitudes."""
    determinants = [sum(1 << p for p in occupied) for occupied in combinations(range(4), 2)]
    index = {bits: i for i, bits in enumerate(determinants)}
    matrix = np.zeros((6, 6))
    def apply(bits, operations):
        sign = 1
        for create, p in operations:
            occupied = bool(bits & (1 << p))
            if occupied == create:
                return None, 0
            sign *= -1 if (bits & ((1 << p)-1)).bit_count()%2 else 1
            bits ^= 1 << p
        return bits, sign
    for column, bits in enumerate(determinants):
        for p, q, spin in product(range(2), range(2), range(2)):
            target, sign = apply(bits, [(False, 2*q+spin), (True, 2*p+spin)])
            if sign:
                matrix[index[target], column] += sign*h[p, q]
        for p, q, r, s, sigma, tau in product(range(2), repeat=6):
            target, sign = apply(bits, [(False, 2*q+sigma), (False, 2*s+tau),
                (True, 2*r+tau), (True, 2*p+sigma)])
            if sign:
                matrix[index[target], column] += .5*sign*g[p, q, r, s]
    np.testing.assert_allclose(matrix, matrix.T, atol=2e-12)
    return np.linalg.eigvalsh(matrix)[0], matrix[0, 0]


@pytest.mark.parametrize("mesh", [(1, 1, 1), (2, 1, 1), (3, 1, 1)])
def test_actual_hf_full_torus_to_ccsd_t_and_independent_spin_residual(mesh):
    b = _bundle(mesh)
    events = []
    result = _run(b, callback=events.append)
    assert result.converged and result.correlation_evaluated
    assert result.diagnostics.complete_finite_torus_basis and result.periodic_energy_per_cell
    assert result.diagnostics.retained_virtual_count == b.context.n_kpoints
    correlation = result.correlation
    assert correlation.triples_evaluated and correlation.ccsd.final_snapshot.converged
    _prepare_leaves(b)
    case = _dense_case(b)
    _original_hamiltonian_check(b, case)
    residual, energy = _spin_orbital_check(case, correlation.ccsd.t1, correlation.ccsd.t2)
    assert residual < 4e-12
    assert energy == pytest.approx(correlation.ccsd.final_snapshot.correlation_energy, abs=3e-12)
    case.update(t1=correlation.ccsd.t1, t2=correlation.ccsd.t2)
    w, u = _moments(case)
    expected = _linear_oracle(case, w)
    np.testing.assert_allclose(correlation.triples.amplitudes, expected, atol=3e-12, rtol=3e-10)
    assert correlation.triples.final_snapshot.triples_energy == pytest.approx(_energy(expected, w, u), abs=3e-13)
    whole = energy+correlation.triples.final_snapshot.triples_energy
    assert result.correlation_energy_per_cell == pytest.approx(whole/b.context.n_kpoints, abs=3e-12)
    assert result.total_energy_per_cell == pytest.approx(b.hf.state.reference_energy_per_cell+whole/b.context.n_kpoints, abs=3e-12)
    assert result.hf_reference_source_identity_sha256 == b.hf.reference_source_identity_sha256
    assert result.matched_finite_gaussian_hf_recipe
    assert not result.production_dlpno and not result.infinite_source_accuracy_certified
    assert not result.bitwise_hf_factor_consumption_verified
    assert len(events) == result.diagnostics.completed_progress_callbacks
    assert len(events) <= result.plan.progress_callback_upper_bound
    stage = core._PeriodicGaussianSelectedLocalCCSDTStage
    assert events[0].stage == stage.BEGIN and events[-1].stage == stage.FINISHED
    assert stage.CORRELATION in [event.stage for event in events]
    if mesh[0] == 3:
        assert np.max(abs(case["panels"][1].imag)) > 1e-4


def test_gamma_two_electron_ccsd_matches_independent_determinant_hamiltonian():
    b = _bundle()
    result = _run(b)
    assert result.converged
    _prepare_leaves(b)
    case = _dense_case(b)
    h, nuclear = _original_hamiltonian_check(b, case)
    fci, reference = _fci_two_electron(h, case["eri"])
    assert result.correlation_energy_per_cell == pytest.approx(fci-reference, abs=4e-12)
    assert result.total_energy_per_cell == pytest.approx(fci+nuclear, abs=4e-12)
    assert abs(result.correlation.triples.final_snapshot.triples_energy) < 1e-15


def _he2_controls(b):
    controls = _controls(b)
    local = controls[0].localization
    optimizer = local.optimizer
    # Existing PM control, chosen explicitly only for this full-space
    # rotation oracle. No SCF, CC, energy or real-basis gate is changed.
    optimizer.riemannian_gradient_tolerance = 2e-6
    local.optimizer, controls[0].localization = optimizer, local
    return controls


@pytest.mark.parametrize("mesh", [(1, 1, 1), (2, 1, 1)])
def test_actual_he2_has_occupied_coupling_and_triples_match_full_linear_and_canonical_oracles(mesh):
    b = _he2_bundle(mesh)
    controls = _he2_controls(b)
    result = _run(b, controls=controls)
    assert result.converged and result.periodic_energy_per_cell
    _prepare_leaves(b, localization_options=controls[0].localization)
    optimizer = b.localization.optimizer
    assert optimizer.status == core._PeriodicCorrelationIAOOptimizerStatus.CONVERGED
    assert optimizer.diagnostics.riemannian_gradient_norm <= 2e-6
    case = _dense_case(b)
    _original_hamiltonian_check(b, case)
    assert abs(case["foo"][0, 1]) > 1e-3
    cc, triples = result.correlation.ccsd, result.correlation.triples
    if mesh[0] == 2:
        # Authenticated o=v=4 is the exact tiny result-copy boundary;
        # returned arrays are detached, not mutable native amplitudes.
        assert cc.t1.shape == (4, 4) and cc.t1.nbytes == 128
        assert cc.t2.shape == (4, 4, 4, 4) and cc.t2.nbytes == 2048
        assert triples.amplitudes.shape == (4,)*6 and triples.amplitudes.nbytes == 32768
        for getter in (lambda: cc.t1, lambda: cc.t2, lambda: triples.amplitudes):
            original = getter()
            modified = getter()
            modified.flat[0] += 1
            np.testing.assert_array_equal(getter(), original)
    residual, energy = _spin_orbital_check(case, cc.t1, cc.t2)
    assert residual < 4e-12
    assert energy == pytest.approx(cc.final_snapshot.correlation_energy, abs=3e-12)
    case.update(t1=cc.t1, t2=cc.t2)
    w, u = _moments(case)
    expected = _linear_oracle(case, w)
    np.testing.assert_allclose(triples.amplitudes, expected, atol=3e-12, rtol=3e-10)
    assert abs(triples.final_snapshot.triples_energy) > 1e-9
    assert triples.final_snapshot.triples_energy == pytest.approx(_energy(expected, w, u), abs=3e-13)
    # These self-inverse q factors are actually real; the independent
    # canonical spin oracle diagonalizes Foo and rotates both factors
    # and converged amplitudes, including the complete two-cell space.
    assert np.max(abs(case["panels"].imag)) < 1e-12
    n = case["o"]+case["v"]
    case["factors"] = case["panels"].real.reshape(-1, n, n)
    assert triples.final_snapshot.triples_energy == pytest.approx(_canonical_spin_energy(case), abs=3e-13)
    assert triples.final_snapshot.neighbour_visits > 0


def test_actual_he2_strict_pm_roundoff_stagnation_publishes_no_correlation():
    b = _he2_bundle()
    result = _run(b)  # Deliberately retains strict PM gradient1e-11.
    assert not result.converged and not result.correlation_evaluated
    assert not result.periodic_energy_per_cell
    local = result.unfinished_localization
    optimizer = local.optimizer
    assert optimizer.status == core._PeriodicCorrelationIAOOptimizerStatus.LINE_SEARCH_FAILED
    assert optimizer.diagnostics.accepted_steps > 0
    assert optimizer.diagnostics.riemannian_gradient_norm > 1e-11
    assert optimizer.diagnostics.riemannian_gradient_norm < 2e-6
    assert optimizer.diagnostics.final_objective > optimizer.diagnostics.initial_objective
    with pytest.raises(RuntimeError, match="localization.*not converge"):
        _ = result.correlation
    with pytest.raises(RuntimeError, match="complete|converged|per.cell|periodic"):
        _ = result.total_energy_per_cell


def test_partial_selection_never_adds_subsystem_correlation_to_periodic_hf():
    b = _bundle((2, 1, 1), full=False)
    result = _run(b)
    assert result.converged and result.correlation_evaluated
    assert not result.diagnostics.complete_finite_torus_basis and not result.periodic_energy_per_cell
    for field in ("correlation_energy_per_cell", "total_energy_per_cell"):
        with pytest.raises(RuntimeError, match="complete|per.cell|periodic"):
            getattr(result, field)


def test_same_content_is_not_same_hf_owner_and_source_cutoff_is_immutable():
    b, other = _bundle(), _bundle()
    assert b.hf.reference_source_identity_sha256 == other.hf.reference_source_identity_sha256
    with pytest.raises(ValueError, match="same|identical|owner"):
        _run(b, hf=other.hf)
    controls = _controls(b)
    local = controls[0].localization
    source = local.source
    source.image_cutoff_bohr = 1.9
    local.source, controls[0].localization = source, local
    with pytest.raises(ValueError, match="cutoff|source|image"):
        _run(b, controls=controls)


@pytest.mark.parametrize("field,reported", [
    ("maximum_owned_numerical_bytes", "owned_numerical_upper_bound"),
    ("maximum_per_worker_inventoried_bytes", "per_worker_inventoried_bytes_upper_bound"),
    ("maximum_node_inventoried_bytes", "node_inventoried_bytes_upper_bound"),
    ("maximum_progress_callbacks", "progress_callback_upper_bound"),
    ("maximum_work_units", "work_units_upper_bound"),
])
def test_exact_outer_caps_and_one_below_reject_before_callbacks(field, reported):
    b = _bundle()
    controls = _controls(b)
    planned = _plan(b, controls)
    setattr(controls[2], field, getattr(planned, reported))
    assert _run(b, controls=controls).converged
    setattr(controls[2], field, getattr(planned, reported)-1)
    events = []
    with pytest.raises((ValueError, OverflowError), match="cap|exceed"):
        _run(b, controls=controls, callback=events.append)
    assert not events


def test_progress_cancellation_and_mutated_actual_inputs_cannot_publish_success():
    b = _bundle((2, 1, 1))
    def cancel(event):
        raise RuntimeError("intentional selected-local cancellation")
    with pytest.raises(RuntimeError, match="intentional selected-local cancellation"):
        _run(b, callback=cancel)
    changed = b.rows.copy()
    def mutate(event):
        changed[0, 1] = 1
    with pytest.raises(ValueError, match="duplicate|changed|identity"):
        _run(b, occupied=changed, callback=mutate)


def test_finished_callback_still_rechecks_original_nuclear_and_electron_source():
    b = _bundle()
    reached = []
    def mutate_at_finish(event):
        if event.stage == core._PeriodicGaussianSelectedLocalCCSDTStage.FINISHED:
            reached.append(event.stage)
            b.system.charge = 2
    with pytest.raises(ValueError, match="input|source|nuclei|charge|electron|changed|identity"):
        _run(b, callback=mutate_at_finish)
    assert reached == [core._PeriodicGaussianSelectedLocalCCSDTStage.FINISHED]


def test_same_integral_gaussian_parameters_with_wrong_atom_map_are_not_an_iao_source():
    b = _bundle()
    # Reorder the separate construction molecule so the unchanged Gaussian
    # centers now map to atom1, absent in the actual one-atom HF cell.
    molecule = core.Molecule([core.Atom(2, [1.0, 0.0, 0.0]), core.Atom(2, [0.0, 0.0, 0.0])])
    wrong = core.BasisSet(molecule, b.ao.shells(), "same-Gaussians-wrong-atom-map", True)
    assert [shell.atom_index for shell in wrong.shells()] == [1, 1]
    for original, changed in zip(b.ao.shells(), wrong.shells()):
        assert (original.l, original.pure) == (changed.l, changed.pure)
        np.testing.assert_array_equal(original.origin, changed.origin)
        np.testing.assert_array_equal(original.exponents, changed.exponents)
        np.testing.assert_array_equal(original.coefficients, changed.coefficients)
    # HF source identity also seals these atom labels, even though the
    # represented Gaussian integral parameters themselves are unchanged.
    events = []
    with pytest.raises(ValueError, match="atom|label|map|basis content mismatch"):
        _run(b, ao=wrong, callback=events.append)
    assert not events


def test_grown_minimal_primitive_census_rejects_before_nonfinite_content_scan():
    b = _bundle()
    shell = core.ShellInfo(0, 0, True, [1.8]*8, [np.nan]*8, [0.0, 0.0, 0.0])
    larger = core.BasisSet(core.Molecule(b.system.unit_cell), [shell], "grown-minimal-test", True)
    events = []
    with pytest.raises(ValueError, match="primitive|coefficient|census|cap"):
        _run(b, minimal=larger, callback=events.append)
    assert not events


def test_unconverged_ccsd_is_not_a_periodic_energy_and_borrowed_result_is_pinned():
    b = _bundle()
    controls = _controls(b)
    corr = controls[0].correlation
    cc = corr.ccsd
    cc.maximum_iterations = 1
    corr.ccsd, controls[0].correlation = cc, corr
    result = _run(b, controls=controls)
    assert result.correlation_evaluated and not result.converged
    assert not result.correlation.triples_evaluated and not result.periodic_energy_per_cell
    with pytest.raises(RuntimeError, match="complete|converged|per.cell|periodic"):
        _ = result.total_energy_per_cell
    view = result.correlation
    identity = view.identity_sha256
    del result, b
    gc.collect()
    assert view.identity_sha256 == identity and not view.converged
