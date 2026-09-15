"""Tiny native global-RI/local-orbital contractions, not SCF or local-aux DF.

The numerical producer is the actual Gaussian factor store. Its admitted
mean-field fixtures are synthetic exact matrix eigenproblems, not chemical
HF references. Independent analytic Gaussian Fourier integrals and tiny
reciprocal projected kernels supply the oracle, without the native whitener,
native factors, a Gamma adapter, or another QC program.
"""

from __future__ import annotations

import hashlib
import math
import struct
import sys
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_aopair_fourier_panel import _oracle as _ao_pair_oracle
from tests.test_periodic_correlation_factor_store import _populate, _writer
from tests.test_periodic_correlation_pao_domain import _make as _make_domain
from tests.test_periodic_correlation_pao_domain import _options as _domain_options
from tests.test_periodic_correlation_pao_space import _geometry, _make as _make_space
from tests.test_periodic_correlation_reciprocal_metric import (
    _accepted_metric_records,
    _analytic_s_fourier,
    _binary64_fma,
)
from tests.test_periodic_correlation_three_center import _bundle
from tests.test_periodic_correlation_wannier import _make as _make_wannier
from tests.test_periodic_correlation_wannier import _options as _wannier_options


pytestmark = pytest.mark.skipif(sys.platform not in ("darwin", "linux"), reason="native private store requires POSIX")


def _selection(**changes):
    result = core._PeriodicCorrelationLocalOrbitalSelection()
    result.virtual_count = 1
    for name, value in changes.items():
        setattr(result, name, value)
    return result


def _caps(plan):
    caps = core._PeriodicCorrelationLocalFactorCaps()
    caps.maximum_owned_numerical_bytes = plan.peak_owned_numerical_bytes
    caps.maximum_work_units = plan.work_units
    caps.maximum_tile_visits = plan.tile_visits
    caps.maximum_reader_tile_bytes = plan.maximum_reader_tile_bytes
    return caps


def _orbital_objects(reference, *, domain_columns=None, gauge=None):
    state = reference.state
    if gauge is None:
        gauge = np.tile(np.eye(state.n_correlated_occupied, dtype=complex), (state.n_kpoints, 1, 1))
    if domain_columns is None:
        virtual_ao = state.n_basis - 1
        domain_columns = np.array([[0, virtual_ao]], dtype=np.uint64)
    domain = _make_domain(reference, np.asarray(domain_columns, dtype=np.uint64), options=_domain_options(
        require_time_reversal=False, require_real_matrices=False))
    space = _make_space(reference, domain)
    wannier = _make_wannier(reference, gauge, options=_wannier_options(
        require_time_reversal=False, require_real_home_coefficients=False))
    return SimpleNamespace(reference=reference, domain=domain, space=space, wannier=wannier, gauge=gauge)


def _native_store(tmp_path, *, mesh=(2, 1, 1), full_domain=False, cutoff=0.9, gauge=None,
                  auxiliary=None, auxiliary_block=1):
    b = _bundle(q=0, mesh=mesh, pairs=3, auxiliary_block=auxiliary_block, block=2,
                auxiliary=auxiliary)
    b.whitener = None
    c = b.config
    c.publisher_mode = core._PeriodicCorrelationFactorPublisherMode.CANONICAL_SEQUENTIAL_EXACTLY_ONCE
    c.backing_mode = core._PeriodicCorrelationFactorBackingMode.DISK
    c.publisher_buffer_count = 0
    c.transpose_before_publish = False
    c.codec_identity_sha256 = core._periodic_correlation_private_factor_store_codec_identity_sha256()
    c.codec = core._periodic_correlation_private_factor_store_codec_inventory()
    backend = c.backend
    backend.exact_extra_retained_bytes = 16 * b.auxiliary.nbasis * 3 + (
        core._periodic_correlation_private_factor_store_receiver_bytes(b.schedule))
    backend.exact_extra_control_bytes = core._periodic_correlation_private_factor_store_fixed_control_bytes()
    c.backend = backend
    b.config = c
    b.census = core._make_periodic_correlation_factor_build_census(
        b.reference, b.schedule, c, [s.factor_build_q_record() for s in b.sources])
    b.store_caps = core._PeriodicCorrelationPrivateFactorStoreCaps()
    b.store_caps.maximum_file_bytes = 1048576
    b.store_caps.maximum_tile_bytes = 8192
    b.store_caps.maximum_tile_count = 128
    b.stream_caps = core._PeriodicCorrelationThreeCenterStreamCaps()
    b.stream_caps.maximum_tile_count = 128
    b.stream_caps.maximum_logical_bytes = 65536
    b.stream_caps.maximum_tile_bytes = 8192
    b.stream_caps.maximum_reciprocal_candidates_per_q = 65536
    b.stream_caps.maximum_image_candidates_per_tile = 20000
    b.stream_caps.receiver_retained_numeric_bytes = core._periodic_correlation_private_factor_store_receiver_bytes(b.schedule)
    writer = _writer(b, tmp_path, cutoff=cutoff)
    b.reader = writer.finish(_populate(b, writer, cutoff=cutoff))
    columns = [[cell, 1] for cell in range(math.prod(mesh))] if full_domain else [[0, 1]]
    objects = _orbital_objects(b.reference, domain_columns=columns, gauge=gauge)
    b.domain, b.space, b.wannier, b.gauge = objects.domain, objects.space, objects.wannier, objects.gauge
    return b


def _panel(b, ko, kv, *, selection=None, caps=None, gauge=None):
    selected = _selection() if selection is None else selection
    plan = core._plan_periodic_correlation_local_coefficient_panel(
        b.reference, b.wannier, b.domain, b.space, selected)
    return core._make_periodic_correlation_local_coefficient_panel(
        b.reference, b.wannier, b.gauge if gauge is None else gauge, b.domain, b.space, selected,
        ko, kv, _caps(plan) if caps is None else caps)


def _factor(b, q, *, vo=False, selection=None, begin=0, count=None, caps=None, gauge=None):
    selected = _selection() if selection is None else selection
    count = b.schedule.shape.n_auxiliary - begin if count is None else count
    plan = core._plan_periodic_correlation_local_factor_block(
        b.reference, b.schedule, b.reader, b.wannier, b.domain, b.space, selected, q, begin, count)
    orientation = (core._PeriodicCorrelationLocalFactorOrientation.VIRTUAL_OCCUPIED if vo else
                   core._PeriodicCorrelationLocalFactorOrientation.OCCUPIED_VIRTUAL)
    return core._build_periodic_correlation_local_factor_block(
        b.reference, b.schedule, b.reader, b.wannier, b.gauge if gauge is None else gauge,
        b.domain, b.space, selected, q, begin, count, orientation, _caps(plan) if caps is None else caps)


def _coefficient_oracle(b, ko, kv, selection):
    state = b.reference.state
    mesh = np.array(state.mesh)
    cells = np.array(list(product(*(range(n) for n in mesh))))
    fractions = cells / mesh
    active = np.asarray(state.correlated_occupied_mask(ko), dtype=bool)
    occupied = (state.coefficients(ko)[:, active] @ b.gauge[ko])[:, selection.occupied_index]
    occupied *= np.exp(-2j * np.pi * fractions[ko] @ cells[selection.occupied_cell])
    virtual = np.asarray(state.virtual_mask(kv), dtype=bool)
    c = state.coefficients(kv)[:, virtual]
    q = c @ c.conj().T @ state.overlap(kv)
    x = b.space.coefficients_copy()[:, selection.virtual_begin:selection.virtual_begin + selection.virtual_count]
    # Q is a tiny test-only matrix; native code never materializes it.
    selected = np.zeros((state.n_basis, b.domain.domain_dimension), dtype=complex)
    for d in range(b.domain.domain_dimension):
        cell, ao = b.domain.column(d)
        phase = np.exp(-2j * np.pi * fractions[kv] @ (
            cells[cell] + cells[selection.virtual_translation_cell]))
        selected[:, d] = phase * q[:, ao]
    return np.column_stack((occupied, selected @ x))


def _reciprocal_oracle(b, q, selection, *, vo=False):
    """Compute local density, then projected Coulomb kernel, independently.

    A[p,a] is the Fourier integral of the normalized finite-torus transition
    density. Native B and W are never read. Primitive analytic Gaussian
    transforms give the Coulomb auxiliary projector only in this tiny oracle.
    """
    records = _accepted_metric_records(b.sources[q])
    vectors = np.array([r[1] for r in records])
    weights = np.array([r[3] for r in records])  # 4*pi/(Omega*p^2)
    f = np.array([_analytic_s_fourier(b.auxiliary, r[1], r[2]) for r in records]).T
    metric = (f.conj() * weights) @ f.T
    eigenvalues, u = np.linalg.eigh(metric)
    retained = eigenvalues > b.config.metric_absolute_eigenvalue_threshold
    w = (u[:, retained] / np.sqrt(eigenvalues[retained])) @ u[:, retained].conj().T
    inverse = (u[:, retained] / eigenvalues[retained]) @ u[:, retained].conj().T
    nk = b.schedule.shape.n_kpoints
    density = np.zeros((len(records), selection.virtual_count), dtype=complex)
    individual = []
    lattice = 2 * np.pi * np.linalg.inv(b.reciprocal).T
    for k in range(nk):
        ket = b.schedule.descriptor(q * b.schedule.shape.tiles_per_q + k * b.schedule.shape.tiles_per_k_bra).k_ket_index
        ko, kv = (ket, k) if vo else (k, ket)
        c = _coefficient_oracle(b, ko, kv, selection)
        rho, _ = _ao_pair_oracle(b.ao, vectors, lattice,
            np.asarray(b.reference.state.kpoint_cartesian(ket)), b.reader.image_cutoff_bohr, 4)
        value = np.zeros_like(density)
        for mu in range(b.ao.nbasis):
            for nu in range(b.ao.nbasis):
                pair = c[mu, 1:].conj() * c[nu, 0] if vo else c[mu, 0].conj() * c[nu, 1:]
                value += rho[mu, nu, :, None] * pair
        individual.append(value)
        density += value / nk
    local = (w @ ((f.conj() * weights) @ density)) / np.sqrt(nk)
    kernel = (weights[:, None] * f.T) @ inverse @ (f.conj() * weights)
    return local, density, kernel, individual


@pytest.mark.parametrize("complex_case,reduced", [(False, False), (True, False), (False, True)])
def test_selected_coefficient_panel_has_independent_k_indices_and_frozen_virtual_projector(complex_case, reduced):
    reference, domain = _geometry(complex_case=complex_case, reduced=reduced)
    space = _make_space(reference, domain)
    gauge = np.ones((reference.state.n_kpoints, 1, 1), dtype=complex)
    gauge[:, 0, 0] = np.exp(1j * np.array([0.17, -0.24, 0.39]))
    b = SimpleNamespace(reference=reference, domain=domain, space=space, gauge=gauge,
                        wannier=_make_wannier(reference, gauge, options=_wannier_options(
                            require_time_reversal=False, require_real_home_coefficients=False)))
    s = _selection(occupied_cell=2, virtual_translation_cell=1, virtual_count=space.retained_dimension)
    for ko, kv in ((0, 0), (0, 2), (1, 0), (2, 1)):
        panel = _panel(b, ko, kv, selection=s)
        expected = _coefficient_oracle(b, ko, kv, s)
        np.testing.assert_allclose(panel.coefficients_copy(), expected, atol=3e-12, rtol=3e-12)
        assert panel.occupied_k_index == ko and panel.virtual_k_index == kv
        # Both frozen and active occupied columns are excluded from every PAO.
        occupied = np.asarray(reference.state.frozen_core_mask(kv), dtype=bool) | np.asarray(
            reference.state.correlated_occupied_mask(kv), dtype=bool)
        np.testing.assert_allclose(reference.state.coefficients(kv)[:, occupied].conj().T
            @ reference.state.overlap(kv) @ panel.coefficients_copy()[:, 1:], 0, atol=3e-12)


@pytest.mark.parametrize("mesh,q,vo", [((1, 1, 1), 0, False), ((2, 1, 1), 1, False),
                                      ((2, 1, 1), 1, True), ((3, 1, 1), 2, True)])
def test_actual_gaussian_store_local_factor_matches_independent_reciprocal_projection(tmp_path, mesh, q, vo):
    nk = math.prod(mesh)
    gauge = np.exp(1j * np.linspace(0.13, 0.71, nk)).reshape(nk, 1, 1)
    b = _native_store(tmp_path, mesh=mesh, gauge=gauge, cutoff=6.5)
    selected = _selection(occupied_cell=nk - 1, virtual_translation_cell=nk // 2)
    result = _factor(b, q, vo=vo, selection=selected)
    expected, _, _, _ = _reciprocal_oracle(b, q, selected, vo=vo)
    np.testing.assert_allclose(result.matrix_copy(), expected, atol=5e-12, rtol=5e-11)
    assert result.store_identity_sha256 == b.reader.storage_identity_sha256
    assert result.source_identity_sha256 == b.sources[q].source_identity_sha256
    assert result.finite_image_reference and not result.ao_image_source_certified


def test_complex_pair_uses_reverse_orientation_not_same_pair_gram(tmp_path):
    b = _native_store(tmp_path, gauge=np.exp(1j * np.array([0.31, 0.67])).reshape(2, 1, 1))
    expected = 0j
    actual = 0j
    naive = 0j
    selected = _selection()
    for q in range(2):
        ov, vo = _factor(b, q), _factor(b, q, vo=True)
        _, density_ov, kernel, _ = _reciprocal_oracle(b, q, selected)
        _, density_vo, _, _ = _reciprocal_oracle(b, q, selected, vo=True)
        expected += (density_vo.conj().T @ kernel @ density_ov)[0, 0] / 2
        actual += np.vdot(vo.matrix_copy(), ov.matrix_copy())
        naive += np.vdot(ov.matrix_copy(), ov.matrix_copy())
    np.testing.assert_allclose(actual, expected, atol=5e-12, rtol=5e-11)
    assert abs(actual.imag) > 1e-5
    assert abs(actual - naive) > 1e-5


def test_cross_k_sum_is_coherent_before_any_pair_product(tmp_path):
    b = _native_store(tmp_path, gauge=np.array([1, -1], dtype=complex).reshape(2, 1, 1))
    actual = _factor(b, 0).matrix_copy()
    expected, _, _, contributions = _reciprocal_oracle(b, 0, _selection())
    np.testing.assert_allclose(actual, expected, atol=1e-13)
    assert np.linalg.norm(actual) < 1e-13
    assert sum(np.linalg.norm(value)**2 for value in contributions) > 1e-3


def test_occupied_phase_covariance_and_common_translation(tmp_path):
    b = _native_store(tmp_path)
    original = {vo: _factor(b, 1, vo=vo).matrix_copy() for vo in (False, True)}
    phase = np.exp(0.37j)
    b.gauge *= phase
    b.wannier = _make_wannier(b.reference, b.gauge, options=_wannier_options(
        require_time_reversal=False, require_real_home_coefficients=False))
    for vo in (False, True):
        result = _factor(b, 1, vo=vo).matrix_copy()
        np.testing.assert_allclose(result, original[vo] * (phase if vo else phase.conjugate()), atol=3e-13)
        translated = _factor(b, 1, vo=vo, selection=_selection(
            occupied_cell=1, virtual_translation_cell=1)).matrix_copy()
        np.testing.assert_allclose(translated, -result, atol=3e-13)


def test_auxiliary_and_virtual_slices_match_complete_tiny_block(tmp_path):
    b = _native_store(tmp_path, full_domain=True)
    selected = _selection(virtual_count=b.space.retained_dimension)
    full = _factor(b, 1, selection=selected)
    by_aux = np.vstack([_factor(b, 1, begin=p, count=1, selection=selected).matrix_copy() for p in range(2)])
    by_virtual = np.hstack([_factor(b, 1, selection=_selection(virtual_begin=a)).matrix_copy()
                            for a in range(b.space.retained_dimension)])
    np.testing.assert_array_equal(by_aux, full.matrix_copy())
    np.testing.assert_array_equal(by_virtual, full.matrix_copy())
    assert full.memory.peak_owned_numerical_bytes == 32 * 2 * selected.virtual_count + (
        16 * b.ao.nbasis * (selected.virtual_count + 3))
    assert full.memory.live_reader_numeric_bytes == 65536 + full.memory.maximum_reader_tile_bytes
    assert full.memory.tile_visits == b.schedule.shape.tiles_per_q
    assert full.memory.reader_payload_bytes == b.schedule.shape.logical_bytes // b.schedule.shape.n_kpoints
    assert full.memory.live_space_bytes == b.space.memory.output_numerical_bytes
    copied = full.matrix_copy()
    copied[:] = 123
    assert not np.array_equal(copied, full.matrix_copy())


@pytest.mark.parametrize("field", ["maximum_owned_numerical_bytes", "maximum_work_units",
                                 "maximum_tile_visits", "maximum_reader_tile_bytes"])
def test_factor_caps_precede_gauge_hash_and_numerical_work(tmp_path, field):
    b = _native_store(tmp_path)
    plan = core._plan_periodic_correlation_local_factor_block(
        b.reference, b.schedule, b.reader, b.wannier, b.domain, b.space, _selection(), 0, 0, 2)
    caps = _caps(plan)
    setattr(caps, field, getattr(caps, field) - 1)
    invalid = np.full_like(b.gauge, np.nan)
    with pytest.raises((ValueError, OverflowError), match="cap"):
        _factor(b, 0, caps=caps, gauge=invalid)


def test_gauge_and_domain_provenance_fail_closed_before_tile_reads(tmp_path):
    b = _native_store(tmp_path)
    with pytest.raises(ValueError, match="sealed Wannier gauge"):
        _factor(b, 0, gauge=-b.gauge)
    with pytest.raises(ValueError, match="C-contiguous complex128"):
        _factor(b, 0, gauge=np.ones(b.gauge.shape, dtype=float))
    foreign = _orbital_objects(b.reference, domain_columns=[[1, 1]])
    b.space = foreign.space
    with pytest.raises(ValueError, match="provenance"):
        _factor(b, 0)


@pytest.mark.parametrize("changes", [{"occupied_index": 1}, {"occupied_cell": 2},
    {"virtual_translation_cell": 2}, {"virtual_begin": 1}, {"virtual_count": 0},
    {"virtual_count": 2**64 - 1}])
def test_selection_extents_and_modular_aliases_are_rejected(changes):
    reference, domain = _geometry((2, 1, 1))
    b = _orbital_objects(reference, domain_columns=[[0, 3]])
    with pytest.raises(IndexError, match="selection|translation"):
        _panel(b, 0, 1, selection=_selection(**changes))


def test_coefficient_payload_wire_is_independently_reproducible():
    reference, domain = _geometry()
    b = _orbital_objects(reference, domain_columns=[[0, 3]])
    panel = _panel(b, 0, 2)
    name = b"vibeqc.periodic.correlation.local-coefficients.payload"
    wire = struct.pack(">Q", len(name)) + name + struct.pack(">IQQQQ", 1, reference.state.n_basis, 1, 0, 2)
    array = panel.coefficients_copy().T.copy().reshape(-1)
    for z in array:
        wire += struct.pack(">dd", 0.0 if z.real == 0 else z.real, 0.0 if z.imag == 0 else z.imag)
    assert hashlib.sha256(wire).hexdigest() == panel.payload_sha256


@pytest.mark.parametrize("nk", [1, 2, 3])
def test_independent_bandlimited_torus_coulomb_normalization_and_complex_orientation(nk):
    # Exactly resolved plane waves, independent of Gaussian/DF code. The
    # Wannier envelope uses 1/Nk and unnormalized Bloch orbitals. |w| is
    # normalized on V=Nk*Omega; a one-sided transition has (ia|ia)=0,
    # whereas its Hermitian norm is positive. No native array-input seam.
    omega = 2 * np.pi
    length = nk * omega
    ngrid = 256
    x = np.arange(ngrid) * length / ngrid
    envelope = sum(np.exp(1j * k * x / nk) for k in range(nk)) / (nk * np.sqrt(omega))
    i = envelope
    a = envelope * np.exp(7j * x)
    np.testing.assert_allclose(length * np.mean(abs(i)**2), 1, atol=2e-14)
    rho = i.conj() * a
    fourier = np.fft.fft(rho) * length / ngrid
    p = 2 * np.pi * np.fft.fftfreq(ngrid, d=length / ngrid)
    weights = np.zeros(ngrid)
    weights[p != 0] = 4 * np.pi / (length * p[p != 0]**2)
    reverse = (-np.arange(ngrid)) % ngrid
    bilinear = np.sum(weights * fourier * fourier[reverse])
    norm = np.sum(weights * abs(fourier)**2)
    assert abs(bilinear) < 1e-14
    assert norm > 1e-3
    # A complex combination of opposite modes yields a genuinely complex
    # Coulomb bilinear. Conjugating the first same-pair factor loses it.
    a = envelope * (np.exp(7j * x) + 1j * np.exp(-7j * x)) / np.sqrt(2)
    rho = i.conj() * a
    fourier = np.fft.fft(rho) * length / ngrid
    value = np.sum(weights * fourier * fourier[reverse])
    assert abs(value.imag) > 1e-3
    assert abs(value.real) < 1e-14


def test_mixed_nyquist_skew_source_boundary_requires_paired_closure_audit():
    # A pure binary64 witness for the current declared radial policy, not a
    # loosened source cut or an unverified q/-q consumer shortcut.
    reciprocal = np.array([[1., 1., 0.], [0., 1., 0.], [0., 0., 1.]])
    q = np.array([-0.5, 1 / 3, 0.])
    qbar = np.array([-0.5, -1 / 3, 0.])
    def matvec(x):
        return np.array([_binary64_fma(reciprocal[row, 0], x[0],
            _binary64_fma(reciprocal[row, 1], x[1], reciprocal[row, 2] * x[2])) for row in range(3)])
    def norm2(x):
        return _binary64_fma(x[0], x[0], _binary64_fma(x[1], x[1], _binary64_fma(x[2], x[2], 0.)))
    p, pbar = matvec(q), matvec(np.array([1., 0., 0.]) + qbar)
    np.testing.assert_array_equal(pbar, -p)
    cutoff = 0.06944444444443253
    radius = math.sqrt(2 * cutoff)
    tol = lambda qf: 128 * np.finfo(float).eps * max(radius + np.linalg.norm(matvec(qf)), 1.)
    assert norm2(p) > (radius + tol(q))**2
    assert norm2(pbar) < (radius + tol(qbar))**2
