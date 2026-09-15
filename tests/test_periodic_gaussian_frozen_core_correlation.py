"""Actual finite-source He2 frozen-core correlation, Gamma and two k points.

Freeze one explicitly identified canonical occupied band per k. This is a
mask-contract test, not a chemical automatic core rule or a bulk benchmark.
Nejad2025 doi:10.1063/5.0290816 Sec.VII uses frozen-core correlation; the
inactive constant below is independently normal ordered from actual AO
integrals, never supplied to a native source or correlation constructor.
"""

from __future__ import annotations

from itertools import combinations, product

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_gaussian_selected_local_ccsd_t import (
    _he2_bundle, _he2_controls, _prepare_leaves, _dense_case,
    _run as _ccsdt_run, _spin_orbital_check, _moments, _linear_oracle, _energy,
)
from tests.test_periodic_gaussian_rhf import _run as _hf_run, _plan as _hf_plan, _dense_hamiltonian
from tests.test_periodic_correlation_pair_topology import _make_dimensions, _make_budget
from tests.test_periodic_gaussian_local_orbital_factors import _caps as _panel_caps
from tests.test_periodic_correlation_density_factors import _occupied_columns, _virtual_columns
from tests.test_periodic_gaussian_fock import _q_data, _physical_tile
from tests.test_periodic_gaussian_real_local_provider import _make as _provider
from tests.test_periodic_gaussian_pair_mp2 import (
    _run as _mp2_run, _controls as _mp2_controls, _projected_oracle,
)
from tests.test_periodic_correlation_wannier import _make as _wannier


def _active_dimensions(b, mesh):
    """Explicit active/total partition, unlike the all-correlated helper."""
    state, nk, nao = b.hf.state, b.context.n_kpoints, b.ao.nbasis
    assert (state.n_frozen_core, state.n_correlated_occupied, state.n_virtual) == (1, 1, 2)
    dims = _make_dimensions(mesh, state.n_correlated_occupied,
        calculation_identity=state.calculation_identity)
    dims.n_basis = dims.n_effective_orbitals = nao
    dims.n_home_total_occupied = state.n_frozen_core+state.n_correlated_occupied
    dims.n_home_virtual = state.n_virtual
    dims.n_auxiliary = dims.domain_local_auxiliary_upper_bound = b.auxiliary.nbasis
    dims.domain_ao_support_upper_bound = dims.domain_pao_upper_bound = dims.domain_pno_upper_bound = nao*nk
    dims.domain_local_occupied_upper_bound = state.n_correlated_occupied*nk
    # The all-correlated HF is retained solely as the unchanged-HF oracle.
    # Account for this second live state separately; the admitted factory
    # adds the masked state exactly once, as required by its contract.
    dims.external_bytes = b.unfrozen_hf.state.resident_bytes
    return dims


def _frozen_bundle(mesh=(1, 1, 1), *, frozen_band=0):
    b = _he2_bundle(mesh)
    b.unfrozen_hf = b.hf
    b.mask = np.zeros((b.context.n_kpoints, 2), np.uint8)
    b.mask[:, frozen_band] = 1
    # Same actual physical source, same four-electron HF solve, new exact
    # frozen/core partition captured natively. No synthetic MF relabeling.
    b.hf = _hf_run(b)
    assert b.hf.converged
    dims, budget = _active_dimensions(b, mesh), _make_budget()
    budget.memory_limit_bytes = budget.scratch_limit_bytes = 2**26
    b.reference = core._make_periodic_correlation_admitted_reference(b.hf.state, dims, budget)
    b.columns = np.array([(cell, ao) for cell in range(b.context.n_kpoints) for ao in range(b.ao.nbasis)], np.uint64)
    b.rows = np.array([(0, cell) for cell in range(b.context.n_kpoints)], np.uint64)
    b.translation = 0
    b.panel_config = core._PeriodicGaussianLocalOrbitalFactorConfig()
    b.panel_config.ao_pair_block = 3
    b.panel_caps = _panel_caps(b)
    return b


def _cc_controls(b):
    controls = _he2_controls(b)  # Existing explicit PM2e-6 only; no tuning.
    controls[1].other_live_bytes_per_worker = 2**20
    return controls


def _core_columns(b, k):
    state = b.reference.state
    cells = np.array(list(product(*(range(n) for n in state.mesh))))
    fraction = cells[k]/np.array(state.mesh)
    mask = np.asarray(state.frozen_core_mask(k), bool)
    canonical = state.coefficients(k)[:, mask]
    assert canonical.shape == (4, 1)
    return np.column_stack([canonical[:, 0]*np.exp(-2j*np.pi*fraction@cell) for cell in cells])


def _full_core_hamiltonian(b):
    """Independent full finite-torus AO transform and inactive normal ordering.

    Ordered orbitals are frozen K, active K, virtual2K. The active h is
    h_pq+sum_c[2(pq|cc)-(pc|cq)]; the retained inactive constant is
    2sum_c h_cc+sum_cd[2(cc|dd)-(cd|dc)]. Nuclear energy is per cell.
    """
    physical = _dense_hamiltonian(b)
    nk, nao = b.context.n_kpoints, b.ao.nbasis
    coefficients = [np.column_stack((_core_columns(b, k), _occupied_columns(b, k, b.rows),
        _virtual_columns(b, k, b.selected))) for k in range(nk)]
    n = coefficients[0].shape[1]
    assert n == 4*nk <= 8
    metric = sum(c.conj().T@physical.s[k]@c for k, c in enumerate(coefficients))/nk
    np.testing.assert_allclose(metric, np.eye(n), atol=5e-12)
    h = sum(c.conj().T@physical.h[k]@c for k, c in enumerate(coefficients))/nk
    panels = []
    for q in range(nk):
        source, _, whitener = _q_data(b, q)
        panel = np.zeros((b.auxiliary.nbasis, n, n), complex)
        for bra in range(nk):
            ket = b.context.ket_index(bra, q)
            tile = _physical_tile(b, source, whitener, bra).matrix.reshape(-1, nao, nao)
            panel += np.einsum("mi,nj,pmn->pij", coefficients[bra].conj(), coefficients[ket], tile)
        panels.append(panel/(nk*np.sqrt(nk)))
    g = np.einsum("xAqp,xArs->pqrs", np.asarray(panels).conj(), np.asarray(panels))
    assert np.max(abs(h.imag)) < 3e-12 and np.max(abs(g.imag)) < 3e-12
    h, g = h.real, g.real
    np.testing.assert_allclose(g, g.swapaxes(0, 1), atol=4e-12)
    np.testing.assert_allclose(g, g.swapaxes(2, 3), atol=4e-12)
    nc, no = nk, nk
    active_g = g[nc:, nc:, nc:, nc:]
    effective_h = h[nc:, nc:].copy()
    for c in range(nc):
        effective_h += 2*g[nc:, nc:, c, c]-g[nc:, c, c, nc:]
    inactive = 2*np.trace(h[:nc, :nc])
    inactive += sum(2*g[c, c, d, d]-g[c, d, d, c] for c, d in product(range(nc), repeat=2))
    f = effective_h.copy()
    for i in range(no):
        f += 2*active_g[:, :, i, i]-active_g[:, i, i, :]
    np.testing.assert_allclose(f, b.basis.fock_copy(), atol=5e-12, rtol=5e-12)
    active_reference = 2*np.trace(effective_h[:no, :no])
    active_reference += sum(2*active_g[i, i, j, j]-active_g[i, j, j, i]
                            for i, j in product(range(no), repeat=2))
    hf_per_cell = (inactive+active_reference)/nk+physical.nuclear
    assert hf_per_cell == pytest.approx(b.hf.state.reference_energy_per_cell, abs=8e-12)
    assert abs(inactive) > .1
    return dict(h=effective_h, eri=active_g, inactive=inactive, active_reference=active_reference,
        nuclear=physical.nuclear, hf_per_cell=hf_per_cell, coefficients=coefficients)


def _two_electron_fci(h, g):
    """Fifteen determinant Gamma oracle, independent of CC residuals."""
    n = h.shape[0]
    assert n == 3 and g.shape == (n,)*4
    determinants = [sum(1 << p for p in occupied) for occupied in combinations(range(2*n), 2)]
    index = {bits: i for i, bits in enumerate(determinants)}
    matrix = np.zeros((len(determinants),)*2)
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
        for p, q, spin in product(range(n), range(n), range(2)):
            target, sign = apply(bits, [(False, 2*q+spin), (True, 2*p+spin)])
            if sign:
                matrix[index[target], column] += sign*h[p, q]
        for p, q, r, s, sigma, tau in product(range(n), range(n), range(n), range(n), range(2), range(2)):
            target, sign = apply(bits, [(False, 2*q+sigma), (False, 2*s+tau),
                (True, 2*r+tau), (True, 2*p+sigma)])
            if sign:
                matrix[index[target], column] += .5*sign*g[p, q, r, s]
    np.testing.assert_allclose(matrix, matrix.T, atol=4e-12)
    return np.linalg.eigvalsh(matrix)[0], matrix[0, 0]


@pytest.fixture(scope="module", params=[(1, 1, 1), (2, 1, 1)])
def frozen_case(request):
    b = _frozen_bundle(request.param)
    _prepare_leaves(b, localization_options=_cc_controls(b)[0].localization)
    return b, _provider(b)


def test_actual_mask_changes_partition_receipts_but_not_four_electron_hf(frozen_case):
    b, _ = frozen_case
    state, original = b.hf.state, b.unfrozen_hf.state
    assert state.electrons_per_cell == original.electrons_per_cell == 4
    assert (state.n_frozen_core, state.n_correlated_occupied, state.n_virtual) == (1, 1, 2)
    assert state.reference_energy_per_cell == original.reference_energy_per_cell
    assert state.state_identity_sha256 != original.state_identity_sha256
    assert state.calculation_identity != original.calculation_identity
    assert b.hf.original_input_identity_sha256 != b.unfrozen_hf.original_input_identity_sha256
    assert b.hf.reference_source_identity_sha256 != b.unfrozen_hf.reference_source_identity_sha256
    # This wrapper receipt starts with the full original-input identity,
    # including the mask. It is not a density-independent operator digest.
    assert b.hf.one_electron_source_identity_sha256 != b.unfrozen_hf.one_electron_source_identity_sha256
    assert b.hf.final_fock_source_identity_sha256 == b.unfrozen_hf.final_fock_source_identity_sha256
    assert b.hf.context is b.unfrozen_hf.context is b.context
    for k in range(state.n_kpoints):
        np.testing.assert_array_equal(state.frozen_core_mask(k), [1, 0, 0, 0])
        np.testing.assert_array_equal(state.correlated_occupied_mask(k), [0, 1, 0, 0])
        np.testing.assert_array_equal(state.virtual_mask(k), [0, 0, 1, 1])
        for getter in ("overlap", "fock", "coefficients", "orbital_energies", "occupations"):
            np.testing.assert_array_equal(getattr(state, getter)(k), getattr(original, getter)(k))
    b.hf.verify_physical_inputs(b.ao, b.auxiliary, b.system)
    assert b.reference.dimensions.n_home_total_occupied == 2
    assert b.reference.dimensions.n_home_occupied == 1
    assert b.reference.dimensions.n_home_virtual == 2
    assert b.reference.dimensions.external_bytes == state.resident_bytes+original.resident_bytes


def test_localization_and_pao_exclude_core_without_mixed_gauge_escape(frozen_case):
    b, provider = frozen_case
    nk = b.context.n_kpoints
    assert b.localization.memory.n_active == b.wannier.n_home_occupied == 1
    assert b.gauge.shape == (nk, 1, 1)
    assert b.basis.memory.occupied_count == nk
    assert b.space.retained_dimension == b.basis.memory.virtual_count == 2*nk
    assert provider.memory.occupied_count == nk and provider.memory.virtual_count == 2*nk
    for k in range(nk):
        state = b.reference.state
        metric, orbitals = state.overlap(k), state.coefficients(k)
        active = _occupied_columns(b, k, b.rows)
        virtual = _virtual_columns(b, k, b.selected)
        frozen = orbitals[:, np.asarray(state.frozen_core_mask(k), bool)]
        occupied = orbitals[:, np.asarray(state.occupations(k)) == 2]
        np.testing.assert_allclose(frozen.conj().T@metric@active, 0, atol=4e-12)
        np.testing.assert_allclose(occupied.conj().T@metric@virtual, 0, atol=4e-12)
    mixed = np.tile(np.array([[1, 1], [-1, 1]], complex)/np.sqrt(2), (nk, 1, 1))
    with pytest.raises(ValueError, match="shape|gauge|occupied"):
        _wannier(b.reference, mixed)


def test_frozen_core_pair_pnos_and_coupled_mp2_match_active_dense_operator(frozen_case):
    b, provider = frozen_case
    result = _mp2_run(b, provider, _mp2_controls())
    assert result.converged and result.periodic_energy_per_cell
    nk = b.context.n_kpoints
    assert result.memory.occupied_count == nk and result.memory.common_virtual_dimension == 2*nk
    assert result.diagnostics.all_pairs_full_rank
    dense, _, energy = _projected_oracle(b, result)
    full = _full_core_hamiltonian(b)
    np.testing.assert_allclose(dense["eri"], full["eri"], atol=5e-12, rtol=5e-12)
    for i in range(nk):
        for j in range(i, nk):
            pair = result.pair(i, j)
            assert pair.memory.retained_dimension == 2*nk
            assert pair.occupied_i == (0, i) and pair.occupied_j == (0, j)
            assert pair.pno_options.occupation_cutoff == 0.0
    expected = (full["inactive"]+full["active_reference"]+energy)/nk+full["nuclear"]
    assert result.total_energy_per_cell == pytest.approx(expected, abs=8e-12)
    assert result.correlation_energy_per_cell == pytest.approx(energy/nk, abs=4e-12)
    assert result.hf_reference_source_identity_sha256 == b.hf.reference_source_identity_sha256


def test_frozen_core_ccsd_t_residual_triples_and_inactive_hf_constant(frozen_case):
    b, _ = frozen_case
    result = _ccsdt_run(b, controls=_cc_controls(b))
    assert result.converged and result.periodic_energy_per_cell
    case, full = _dense_case(b), _full_core_hamiltonian(b)
    cc, triples = result.correlation.ccsd, result.correlation.triples
    nk = b.context.n_kpoints
    assert cc.t1.shape == (nk, 2*nk) and cc.t2.shape == (nk, nk, 2*nk, 2*nk)
    np.testing.assert_allclose(case["eri"], full["eri"], atol=5e-12)
    residual, energy = _spin_orbital_check(case, cc.t1, cc.t2)
    assert residual < 5e-12
    assert energy == pytest.approx(cc.final_snapshot.correlation_energy, abs=4e-12)
    case.update(t1=cc.t1, t2=cc.t2)
    w, u = _moments(case)
    expected_t3 = _linear_oracle(case, w)
    np.testing.assert_allclose(triples.amplitudes, expected_t3, atol=4e-12, rtol=4e-10)
    triples_energy = _energy(expected_t3, w, u)
    assert triples.final_snapshot.triples_energy == pytest.approx(triples_energy, abs=4e-13)
    total = (full["inactive"]+full["active_reference"]+energy+triples_energy)/nk+full["nuclear"]
    assert result.total_energy_per_cell == pytest.approx(total, abs=8e-12)
    if nk == 1:
        fci, reference = _two_electron_fci(full["h"], full["eri"])
        assert reference == pytest.approx(full["active_reference"], abs=4e-12)
        assert energy == pytest.approx(fci-reference, abs=6e-12)
        assert abs(triples_energy) < 1e-15
        assert result.total_energy_per_cell == pytest.approx(full["inactive"]+fci+full["nuclear"], abs=8e-12)


def test_total_active_virtual_counts_and_occupied_labels_cannot_reintroduce_core(frozen_case):
    b, _ = frozen_case
    for field, value, message in (("n_home_total_occupied", 1, "total occupied"),
                                  ("n_home_occupied", 2, "correlated occupied"),
                                  ("n_home_virtual", 3, "virtual count")):
        dims = _active_dimensions(b, tuple(b.context.mesh))
        setattr(dims, field, value)
        with pytest.raises(ValueError, match=message):
            core._make_periodic_correlation_admitted_reference(b.hf.state, dims, b.reference.budget)
    rows = b.rows.copy()
    rows[0, 0] = 1  # Canonical occupied labels must not enter active-slot API.
    with pytest.raises((ValueError, IndexError), match="occupied|index"):
        _ccsdt_run(b, controls=_cc_controls(b), occupied=rows)
    with pytest.raises(ValueError, match="state|owner|reference"):
        _ccsdt_run(b, controls=_cc_controls(b), hf=b.unfrozen_hf)


def test_nonleading_frozen_band_is_exact_not_an_automatic_leading_core_rule():
    b = _frozen_bundle(frozen_band=1)
    state = b.hf.state
    np.testing.assert_array_equal(state.frozen_core_mask(0), [0, 1, 0, 0])
    np.testing.assert_array_equal(state.correlated_occupied_mask(0), [1, 0, 0, 0])
    _prepare_leaves(b, localization_options=_cc_controls(b)[0].localization)
    active = _occupied_columns(b, 0, b.rows)
    orbital = state.coefficients(0)[:, :1]
    np.testing.assert_allclose(active@active.conj().T, orbital@orbital.conj().T, atol=4e-12)
    assert b.space.retained_dimension == 2


def test_changing_frozen_rank_between_kpoints_is_rejected_before_hf_callbacks(frozen_case):
    b, _ = frozen_case
    if b.context.n_kpoints == 1:
        return
    bad = b.mask.copy()
    bad[1] = 0
    with pytest.raises(ValueError, match="equal frozen rank"):
        _hf_plan(b, mask=bad)
    events = []
    with pytest.raises(ValueError, match="equal frozen rank"):
        _hf_run(b, mask=bad, progress=events.append)
    assert not events
