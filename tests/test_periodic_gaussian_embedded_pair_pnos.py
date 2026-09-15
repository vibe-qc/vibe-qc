"""Actual finite-source PNO generation INSIDE distinct PAO pair frames.

Nejad2025 doi:10.1063/5.0290816 IV.C Eqs.37-40, independently evaluated
from dense actual AO-factor integrals. These tiny He fixtures do not prove
infinite-source chemistry, scalable domains, or a coupled MP2 solution.
"""

from __future__ import annotations

import gc
import math
from types import SimpleNamespace

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_gaussian_selected_local_ccsd_t import (
    _bundle, _he2_bundle, _he2_controls, _prepare_leaves, _dense_case,
)
from tests.test_periodic_gaussian_frozen_core_correlation import _frozen_bundle
from tests.test_periodic_gaussian_real_local_provider import _make as _provider
from tests.test_periodic_gaussian_pair_pnos import _make as _old_pnos, _options as _pno_options
from tests.test_periodic_correlation_real_pao_embedding import (
    _make as _embedding, _torus, _domain, _real_space, _virtual,
)
from tests.test_periodic_correlation_reciprocal_metric import _CanonicalDigest
from tests.test_periodic_gaussian_pair_space import _accumulate


_CAPS = dict(maximum_common_dimension="common_virtual_dimension",
    maximum_generation_dimension="generation_dimension",
    maximum_owned_numerical_bytes="peak_owned_numerical_bytes",
    maximum_control_storage_bytes_per_worker="control_storage_reservation_bytes",
    maximum_per_worker_inventoried_bytes="per_worker_inventoried_bytes",
    maximum_node_inventoried_bytes="required_node_memory_bytes",
    maximum_integral_calls="integral_calls", maximum_work_units="work_units")
_PROJECTIONS = dict(maximum_diagonal_exchange_projection_norm="diagonal_exchange_projection_frobenius_upper_bound",
    maximum_fock_symmetry_projection_norm="fock_symmetry_projection_frobenius_upper_bound")
_EXPORTS = dict(maximum_exported_gram_error="exported_gram_frobenius_upper_bound",
    maximum_exported_fock_error="exported_fock_frobenius_upper_bound",
    maximum_exported_subspace_error="exported_subspace_frobenius_upper_bound")


def _options(*, cutoff=0.0, pno=None, **changes):
    options = core._PeriodicGaussianEmbeddedPairPNOOptions()
    options.pno = _pno_options(occupation_cutoff=cutoff) if pno is None else pno
    for field in _PROJECTIONS:
        setattr(options, field, 1e-10)
    for field in _EXPORTS:
        setattr(options, field, 1e-9)
    for field, value in changes.items():
        setattr(options, field, value)
    return options


def _live(**changes):
    result = core._PeriodicGaussianEmbeddedPairPNOLiveInventory()
    # Remaining tiny HF/Gaussian/localization/domain and NumPy oracle owners.
    # The native plan counts reference, basis, provider and embedding itself.
    result.other_live_numerical_bytes_per_worker = 2**20
    result.other_live_control_bytes_per_worker = result.fixed_backend_margin_bytes_per_worker = 65536
    for field, value in changes.items():
        setattr(result, field, value)
    return result


def _caps(plan=None):
    result = core._PeriodicGaussianEmbeddedPairPNOCaps()
    values = dict(maximum_common_dimension=8, maximum_generation_dimension=8,
        maximum_owned_numerical_bytes=2**20, maximum_control_storage_bytes_per_worker=2**22,
        maximum_per_worker_inventoried_bytes=2**24, maximum_node_inventoried_bytes=2**26,
        maximum_integral_calls=256, maximum_work_units=10**12)
    for field, value in values.items():
        setattr(result, field, value if plan is None else getattr(plan, _CAPS[field]))
    return result


def _plan(b, provider, embedding, i=0, j=0, *, options=None, live=None, caps=None):
    return core._plan_periodic_gaussian_embedded_pair_pnos(b.reference, b.basis, provider, embedding, i, j,
        _options() if options is None else options, _live() if live is None else live,
        _caps() if caps is None else caps)


def _make(b, provider, embedding, i=0, j=0, *, options=None, live=None, caps=None):
    return core._make_periodic_gaussian_embedded_pair_pnos(b.reference, b.basis, provider, embedding, i, j,
        _options() if options is None else options, _live() if live is None else live,
        _caps() if caps is None else caps)


def _frame(b, *, full=False, columns=None, translation=0):
    frame = SimpleNamespace(**vars(b))
    if full:
        frame.pair_domain, frame.pair_real_space = b.domain, b.real_space
        frame.pair_selected = _virtual(b.selected.begin, b.selected.count, b.selected.translation_cell)
    else:
        if columns is None:
            nk, n = b.reference.state.n_kpoints, b.selected.count
            columns = [[0, 0]] if n <= 2 else [[0, 0], [nk-1, b.reference.state.n_basis-2]]
        frame.pair_domain = _domain(b.reference, np.asarray(columns, np.uint64))
        frame.pair_real_space = _real_space(b.reference, frame.pair_domain)
        frame.pair_selected = _virtual(0, frame.pair_real_space.space.retained_dimension, translation)
    return _embedding(frame), frame


def _physical(kind="he2", nk=2, *, frozen=False):
    if frozen:
        b = _frozen_bundle((nk, 1, 1))
        _prepare_leaves(b, localization_options=_he2_controls(b)[0].localization)
    elif kind == "he2":
        b = _he2_bundle((nk, 1, 1))
        _prepare_leaves(b, localization_options=_he2_controls(b)[0].localization)
    else:
        b = _prepare_leaves(_bundle((nk, 1, 1)))
    return b, _provider(b), _dense_case(b)


@pytest.fixture(scope="module", params=[("he", 1), ("he", 2), ("he", 3), ("he2", 1), ("he2", 2)])
def physical(request):
    return _physical(*request.param)


@pytest.fixture(scope="module")
def rectangular():
    b, provider, dense = _physical()
    embedding, frame = _frame(b, translation=1)
    assert (embedding.memory.common_dimension, embedding.memory.pair_dimension) == (4, 2)
    return b, provider, dense, embedding, frame


def _oracle(b, dense, embedding, i=0, j=0):
    x = embedding.coefficients_copy()
    o = dense["o"]
    fock = b.basis.fock_copy()
    common_f, common_g = fock[o:, o:], dense["eri"][i, o:, j, o:]
    raw_g, raw_f = x.T@common_g@x, x.T@common_f@x
    g = (raw_g+raw_g.T)/2 if i == j else raw_g
    f = (raw_f+raw_f.T)/2
    np.testing.assert_allclose(f, np.diag(embedding.energies_copy()), atol=6e-11, rtol=3e-11)
    eps = np.diag(f)
    delta = eps[:, None]+eps[None, :]-fock[i, i]-fock[j, j]
    assert delta.min() > 1e-8
    t = -g/delta
    u = 2*t-t.T
    density = 2*(t@u.T+t.T@u)
    values, vectors = np.linalg.eigh(density)
    return dict(x=x, f=f, g=g, raw_g=raw_g, raw_f=raw_f, delta=delta, t=t, density=density,
        occupations=values[::-1], pnos=vectors[:, ::-1], common_f=common_f, common_g=common_g)


def _check(b, provider, embedding, result, expected, cutoff=0.0, i=0, j=0):
    values = expected["occupations"]
    kept = np.ones(len(values), bool) if cutoff == 0 else values > cutoff
    local = expected["pnos"][:, kept]
    desired = expected["x"]@local
    c, eps, occupations = result.coefficients_copy(), result.energies_copy(), result.original_pno_occupations_copy()
    generation = result.generation_coefficients_copy()
    n, m, t = expected["x"].shape[0], len(values), int(kept.sum())
    assert c.shape == (n, t) and occupations.shape == (m,) and eps.shape == (t,)
    assert generation.shape == (m, t) and not generation.flags.writeable
    np.testing.assert_allclose(generation@generation.T, local@local.T, atol=3e-9, rtol=3e-9)
    np.testing.assert_allclose(generation.T@generation, np.eye(t), atol=5e-11)
    np.testing.assert_allclose(generation.T@expected["f"]@generation, np.diag(eps), atol=5e-11, rtol=5e-11)
    np.testing.assert_allclose(c, expected["x"]@generation, atol=5e-11)
    np.testing.assert_allclose(occupations, values, atol=4e-12, rtol=3e-9)
    np.testing.assert_allclose(c@c.T, desired@desired.T, atol=3e-9, rtol=3e-9)
    np.testing.assert_allclose(c.T@c, np.eye(t), atol=5e-11)
    np.testing.assert_allclose(c.T@expected["common_f"]@c, np.diag(eps), atol=5e-11, rtol=5e-11)
    np.testing.assert_allclose(eps, np.linalg.eigvalsh(local.T@expected["f"]@local), atol=5e-11, rtol=5e-11)
    np.testing.assert_allclose(c, expected["x"]@expected["x"].T@c, atol=5e-11)
    assert result.diagnostics.retained_dimension == t
    assert result.complete_generation_pair_space == (t == m)
    assert result.complete_common_virtual_space == (t == n)
    assert result.diagnostics.generation.density_trace == pytest.approx(np.trace(expected["density"]), abs=4e-12)
    assert result.diagnostics.generation.minimum_denominator == pytest.approx(expected["delta"].min(), abs=4e-12)
    assert result.diagnostics.generation.maximum_denominator == pytest.approx(expected["delta"].max(), abs=4e-12)
    assert result.diagnostics.generation.maximum_initial_residual < 1e-12
    assert result.occupied_i == tuple(b.rows[i]) and result.occupied_j == tuple(b.rows[j])
    assert result.diagonal_pair == (i == j)
    assert result.state is b.reference.state and result.context is b.context
    assert result.basis_identity_sha256 == b.basis.identity_sha256
    assert result.provider_identity_sha256 == provider.identity_sha256
    assert result.hf_reference_source_identity_sha256 == b.hf.reference_source_identity_sha256
    assert result.embedding_identity_sha256 == embedding.identity_sha256
    assert not result.coupled_mp2_solution and not result.production_dlpno and not result.infinite_source_accuracy_certified
    assert not hasattr(result, "correlation_energy")
    for field in ("identity_sha256", "payload_sha256", "raw_exchange_identity_sha256", "projected_exchange_identity_sha256",
        "raw_fock_identity_sha256", "projected_fock_identity_sha256", "initial_amplitude_identity_sha256",
        "density_identity_sha256", "source_payload_receipt_sha256"):
        value = getattr(result, field)
        assert len(value) == 64 and int(value, 16) >= 0


def test_identity_embedding_matches_existing_actual_source_pair_pnos(physical):
    b, provider, dense = physical
    embedding, _ = _frame(b, full=True)
    np.testing.assert_allclose(embedding.coefficients_copy(), np.eye(dense["v"]), atol=5e-11)
    for i, j in [(0, 0)]+([(0, 1)] if dense["o"] > 1 else []):
        old, result = _old_pnos(b, provider, i, j), _make(b, provider, embedding, i, j)
        _check(b, provider, embedding, result, _oracle(b, dense, embedding, i, j), i=i, j=j)
        np.testing.assert_allclose(result.original_pno_occupations_copy(), old.original_pno_occupations_copy(), atol=4e-12, rtol=3e-9)
        np.testing.assert_allclose(result.coefficients_copy()@result.coefficients_copy().T,
            old.coefficients_copy()@old.coefficients_copy().T, atol=4e-11)
        np.testing.assert_allclose(result.energies_copy(), old.energies_copy(), atol=4e-11)
        assert result.complete_common_virtual_space


def test_rectangular_pair_density_and_original_fock_export_match_dense_oracle(physical):
    b, provider, dense = physical
    if dense["v"] == 1:
        return
    embedding, frame = _frame(b, translation=b.reference.state.n_kpoints-1)
    assert embedding.memory.pair_dimension < dense["v"]
    torus = _torus(frame)
    direct = torus.common.conj().T@torus.s@torus.pair
    np.testing.assert_allclose(embedding.coefficients_copy(), direct, atol=4e-11)
    for i, j in [(0, 0)]+([(0, 1), (1, 0)] if dense["o"] > 1 else []):
        expected = _oracle(b, dense, embedding, i, j)
        result = _make(b, provider, embedding, i, j)
        _check(b, provider, embedding, result, expected, i=i, j=j)
        assert result.complete_generation_pair_space and not result.complete_common_virtual_space
        if i == j:
            np.testing.assert_allclose(expected["density"], 4*expected["t"]@expected["t"].T, atol=3e-13)
        else:
            assert result.diagnostics.diagonal_exchange_projection_frobenius_upper_bound == 0


def test_pair_generation_is_not_posthoc_projection_of_common_density(rectangular):
    b, provider, dense, embedding, _ = rectangular
    expected = _oracle(b, dense, embedding)
    result = _make(b, provider, embedding)
    _check(b, provider, embedding, result, expected)
    eps = np.diag(expected["common_f"])
    foo = b.basis.fock_copy()[0, 0]
    t = -expected["common_g"]/(eps[:, None]+eps[None, :]-2*foo)
    u = 2*t-t.T
    wrong = expected["x"].T@(2*(t@u.T+t.T@u))@expected["x"]
    # Actual physical inputs, not supplied synthetic T: omitted internal
    # virtual directions affect the density if truncation happens too late.
    assert np.linalg.norm(wrong-expected["density"]) > 1e-10
    assert np.linalg.norm(np.linalg.eigvalsh(wrong)[::-1]-result.original_pno_occupations_copy()) > 1e-10


def test_explicit_positive_pno_cutoff_and_rank_zero_preserve_generation_dimension(rectangular):
    b, provider, dense, embedding, _ = rectangular
    expected = _oracle(b, dense, embedding)
    high, low = expected["occupations"]
    assert high > low and high > 0
    for cutoff in ((high+low)/2, 2*high):
        result = _make(b, provider, embedding, options=_options(cutoff=cutoff))
        _check(b, provider, embedding, result, expected, cutoff)
        n, m, t = result.memory.common_virtual_dimension, result.memory.generation_dimension, result.diagnostics.retained_dimension
        assert result.diagnostics.retained_output_bytes == 8*n*t+8*m*t+8*t+8*m
        assert result.diagnostics.retained_generation_coefficient_bytes == 8*m*t
        assert result.diagnostics.generation.retained_output_bytes == 8*m*t+8*t+8*m
        assert result.diagnostics.generation.completed_integral_calls == n*n
        assert result.diagnostics.generation.discarded_occupation_sum == pytest.approx(
            expected["occupations"][expected["occupations"] <= cutoff].sum(), abs=4e-12)
    assert t == 0 and result.diagnostics.retained_output_bytes == 8*m
    assert not result.complete_generation_pair_space and not result.complete_common_virtual_space


def test_offdiagonal_ordering_preserves_antisymmetry_and_transpose_covariance(rectangular):
    b, provider, dense, embedding, _ = rectangular
    # The two same-cell He orbitals give a symmetry-suppressed projected
    # antisymmetric block. A translated occupied partner is the nonzero
    # physical witness; keep the numerical assertion independent of noise.
    direct, reverse = _make(b, provider, embedding, 0, 2), _make(b, provider, embedding, 2, 0)
    expected = _oracle(b, dense, embedding, 0, 2)
    _check(b, provider, embedding, direct, expected, i=0, j=2)
    assert np.linalg.norm(expected["g"]-expected["g"].T) > 1e-10
    assert direct.diagnostics.diagonal_exchange_projection_frobenius_upper_bound == 0
    np.testing.assert_allclose(direct.original_pno_occupations_copy(), reverse.original_pno_occupations_copy(), atol=4e-12)
    np.testing.assert_allclose(direct.coefficients_copy()@direct.coefficients_copy().T,
        reverse.coefficients_copy()@reverse.coefficients_copy().T, atol=4e-11)
    assert direct.identity_sha256 != reverse.identity_sha256


def _ordered_projections(b, provider, embedding, i, j):
    """Tiny binary64 wire oracle, separate from the dense NumPy oracle."""
    x, o = embedding.coefficients_copy(), b.basis.memory.occupied_count
    n, m = x.shape
    f = b.basis.fock_copy()[o:, o:]
    matrices = []
    for role in ("G", "F"):
        values, correction = np.zeros((m, m)), np.zeros((m, m))
        for a in range(n):
            for b_index in range(n):
                g = (provider.provider.integral(i, o+a, j, o+b_index).value
                     if role == "G" else float(f[a, b_index]))
                for u in range(m):
                    for v in range(m):
                        term = (float(x[a, u])*g)*float(x[b_index, v])
                        values[u, v], correction[u, v] = _accumulate(
                            term, float(values[u, v]), float(correction[u, v]))
        matrices.append(values+correction)
    return matrices


def _symmetric(matrix):
    projected = matrix.copy()
    for a in range(len(matrix)):
        for b in range(a+1, len(matrix)):
            x, y = float(matrix[a, b]), float(matrix[b, a])
            if x == y:
                value = x
            else:
                exponent = math.frexp(max(abs(x), abs(y)))[1]
                value = math.ldexp(math.ldexp(x, -exponent)+math.ldexp(y, -exponent), exponent-1)
            projected[a, b] = projected[b, a] = value
    return projected


def _array_digest(role, matrix):
    h = _CanonicalDigest("vibeqc.periodic.gaussian-embedded-pair-pnos."+role, 1)
    h.u64(matrix.shape[0])
    h.u64(matrix.size)
    for value in matrix.ravel():
        h.binary64(value)
    return h.finish()


def test_raw_and_projected_payload_seals_pin_order_and_diagonal_only_repair(rectangular):
    b, provider, dense, embedding, _ = rectangular
    for i, j in ((0, 0), (0, 2)):
        result = _make(b, provider, embedding, i, j)
        o, n = dense["o"], dense["v"]
        common = np.array([[provider.provider.integral(i, o+a, j, o+c).value
            for c in range(n)] for a in range(n)])
        h = _CanonicalDigest("vibeqc.periodic.gaussian-pair-pnos.integrals", 1)
        h.u64(n)
        h.u64(n*n)
        for value in common.ravel():
            h.binary64(value)
        assert result.common_exchange_integral_identity_sha256 == h.finish()
        assert result.common_exchange_integral_identity_sha256 == _old_pnos(b, provider, i, j).exchange_integral_identity_sha256
        np.testing.assert_allclose(common, dense["eri"][i, o:, j, o:], atol=5e-12, rtol=3e-11)
        raw_g, raw_f = _ordered_projections(b, provider, embedding, i, j)
        expected = _oracle(b, dense, embedding, i, j)
        np.testing.assert_allclose(raw_g, expected["raw_g"], atol=5e-12, rtol=3e-11)
        np.testing.assert_allclose(raw_f, expected["raw_f"], atol=5e-12, rtol=3e-11)
        projected_g = _symmetric(raw_g) if i == j else raw_g
        projected_f = _symmetric(raw_f)
        assert result.raw_exchange_identity_sha256 == _array_digest("raw-G", raw_g)
        assert result.projected_exchange_identity_sha256 == _array_digest("G", projected_g)
        assert result.raw_fock_identity_sha256 == _array_digest("raw-F", raw_f)
        assert result.projected_fock_identity_sha256 == _array_digest("F", projected_f)
        if i == j:
            np.testing.assert_array_equal(projected_g, projected_g.T)
            assert np.linalg.norm(raw_g-projected_g) <= result.diagnostics.diagonal_exchange_projection_frobenius_upper_bound+1e-30
        else:
            assert np.linalg.norm(projected_g-projected_g.T) > 1e-10
            assert result.diagnostics.diagonal_exchange_projection_frobenius_upper_bound == 0
        assert np.linalg.norm(raw_f-projected_f) <= result.diagnostics.fock_symmetry_projection_frobenius_upper_bound+1e-30


@pytest.mark.parametrize("i,j", [(0, 0), (0, 2)])
def test_retained_generation_is_original_semicanonical_result_and_payload_v2(rectangular, i, j):
    """Exact move-source replay, separate from _check's NumPy density/Fock oracle.

    Replay the native scalar leaves from independently ordered projected G/F.
    This pins the original local D; X^T C would add a second numerical transform.
    """
    b, provider, _, embedding, _ = rectangular
    raw_g, raw_f = _ordered_projections(b, provider, embedding, i, j)
    g = _symmetric(raw_g) if i == j else raw_g
    f = _symmetric(raw_f)
    m, n = len(f), embedding.memory.common_dimension
    foo = b.basis.fock_copy()
    for cutoff in (0.0, 1.0):
        options = _options(cutoff=cutoff)
        p = options.pno
        result = _make(b, provider, embedding, i, j, options=options)
        initial = core._restricted_pair_semicanonical_mp2(g, np.diag(f).copy(),
            foo[i, i], foo[j, j], p.denominator_floor, 8*m*m)
        pnos = core.restricted_pair_pnos(initial.amplitudes_copy(), core.RestrictedPairKind.OffDiagonal,
            cutoff, core.plan_restricted_pair_pnos(m).peak_owned_numerical_bytes,
            p.pno_max_sweeps, p.pno_relative_eigensolver_tolerance)
        selected = pnos.coefficients_copy()
        rank = selected.shape[1]
        sc = core.restricted_pair_semicanonicalize(selected, f, p.semicanonical_orthonormality_tolerance,
            core.plan_restricted_pair_semicanonicalization(m, rank).peak_owned_numerical_bytes,
            p.semicanonical_max_sweeps, p.semicanonical_relative_eigensolver_tolerance)
        np.testing.assert_array_equal(result.generation_coefficients_copy(), sc.coefficients_copy())
        np.testing.assert_array_equal(result.energies_copy(), sc.energies_copy())
        h = _CanonicalDigest("vibeqc.periodic.gaussian-embedded-pair-pnos.payload-v2", 1)
        for dimension in (n, m, rank):
            h.u64(dimension)
        for values in (result.original_pno_occupations_copy(), result.coefficients_copy(),
                       result.generation_coefficients_copy(), result.energies_copy()):
            for value in values.ravel():
                h.binary64(value)
        assert result.payload_sha256 == h.finish()


def test_exact_projection_and_export_phase_inventory_with_all_live_owners(rectangular):
    b, provider, _, embedding, _ = rectangular
    live, p = _live(), _plan(b, provider, embedding)
    n, m = p.common_virtual_dimension, p.generation_dimension
    assert p.integral_calls == n*n and p.provider_work_units == n*n*provider.memory.scalar_work_units
    assert p.projection_phase_bytes == 24*m*m
    assert p.amplitude_phase_bytes == 24*m*m+8*m
    assert p.density_phase_bytes == 56*m*m+8*m
    assert p.semicanonical_phase_upper_bytes == 56*m*m+16*m
    assert p.export_phase_upper_bytes == 8*n*m+8*m*m+16*m
    assert p.peak_owned_numerical_bytes == max(p.projection_phase_bytes, p.amplitude_phase_bytes,
        p.density_phase_bytes, p.semicanonical_phase_upper_bytes, p.export_phase_upper_bytes)
    assert p.retained_output_upper_bytes == 8*n*m+8*m*m+16*m
    assert p.retained_generation_coefficient_upper_bytes == 8*m*m
    assert p.borrowed_basis_bytes == b.basis.memory.retained_output_bytes
    assert p.borrowed_provider_row_bytes == provider.memory.retained_row_bytes
    assert p.borrowed_embedding_bytes == embedding.memory.output_numerical_bytes
    assert p.complete_borrowed_numerical_bytes == p.borrowed_basis_bytes+p.borrowed_provider_row_bytes+p.borrowed_embedding_bytes
    assert p.per_worker_inventoried_bytes == (p.peak_owned_numerical_bytes+p.complete_borrowed_numerical_bytes
        +p.control_storage_reservation_bytes+live.other_live_numerical_bytes_per_worker+live.fixed_backend_margin_bytes_per_worker)
    assert p.required_node_memory_bytes == p.reference_base_node_bytes+p.replicas_per_node*p.per_worker_inventoried_bytes
    result = _make(b, provider, embedding)
    assert result.diagnostics.actual_semicanonical_phase_bytes <= p.semicanonical_phase_upper_bytes
    assert result.diagnostics.actual_export_phase_bytes <= p.export_phase_upper_bytes
    assert result.diagnostics.retained_output_bytes == result.diagnostics.actual_export_phase_bytes
    assert result.diagnostics.retained_generation_coefficient_bytes == result.generation_coefficients_copy().nbytes
    live.other_live_numerical_bytes_per_worker += 13
    live.other_live_control_bytes_per_worker += 17
    q = _plan(b, provider, embedding, live=live)
    assert q.per_worker_inventoried_bytes-p.per_worker_inventoried_bytes == 30
    assert q.required_node_memory_bytes-p.required_node_memory_bytes == 30*p.replicas_per_node


@pytest.mark.parametrize("field", _CAPS)
def test_exact_caps_and_one_below_every_native_preflight(rectangular, field):
    b, provider, _, embedding, _ = rectangular
    caps = _caps(_plan(b, provider, embedding))
    _make(b, provider, embedding, caps=caps)
    setattr(caps, field, getattr(caps, field)-1)
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        _make(b, provider, embedding, caps=caps)


@pytest.mark.parametrize("field,diagnostic", list(_PROJECTIONS.items())+list(_EXPORTS.items()))
def test_each_measured_projection_and_export_budget_is_enforced(rectangular, field, diagnostic):
    b, provider, _, embedding, _ = rectangular
    accepted = _make(b, provider, embedding)
    measured = getattr(accepted.diagnostics, diagnostic)
    # A exactly-zero represented correction needs no projection allowance.
    # Nonzero observations pin both the exact budget and strict cap-1 gate.
    if measured > 0:
        with pytest.raises(ValueError, match="projection|export|budget|Gram|Fock|subspace"):
            _make(b, provider, embedding, options=_options(**{field: np.nextafter(measured, 0)}))
    if field in _PROJECTIONS or measured > 0:
        repeat = _make(b, provider, embedding, options=_options(**{field: measured}))
        assert getattr(repeat.diagnostics, diagnostic) == measured


@pytest.mark.parametrize("field", list(_PROJECTIONS)+list(_EXPORTS))
@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf])
def test_explicit_projection_and_export_controls_reject_nonfinite_or_negative(rectangular, field, value):
    b, provider, _, embedding, _ = rectangular
    with pytest.raises(ValueError):
        _plan(b, provider, embedding, options=_options(**{field: value}))


@pytest.mark.parametrize("field", _EXPORTS)
def test_export_audits_need_positive_budgets(rectangular, field):
    b, provider, _, embedding, _ = rectangular
    with pytest.raises(ValueError):
        _plan(b, provider, embedding, options=_options(**{field: 0.0}))


@pytest.mark.parametrize("change", ["cutoff", "denominator", "fvv", "orthonormality", "sweeps", "margin", "occupied"])
def test_generation_controls_indices_and_backend_margin_are_not_silent_defaults(rectangular, change):
    b, provider, _, embedding, _ = rectangular
    options, live, i = _options(), _live(), 0
    pno = options.pno
    if change == "cutoff":
        pno.occupation_cutoff = np.nan
    elif change == "denominator":
        pno.denominator_floor = 100.0  # Admitted control, impossible physical denominators.
    elif change == "fvv":
        pno.maximum_initial_fvv_offdiagonal_norm = np.nan
    elif change == "orthonormality":
        pno.semicanonical_orthonormality_tolerance = 0.0
    elif change == "sweeps":
        pno.pno_max_sweeps = 0
    elif change == "margin":
        live.fixed_backend_margin_bytes_per_worker = 0
    else:
        i = len(b.rows)
    options.pno = pno
    with pytest.raises((ValueError, RuntimeError, IndexError)):
        _make(b, provider, embedding, i, options=options, live=live)


@pytest.mark.parametrize("field", ["reference", "basis", "provider", "embedding"])
def test_equal_content_distinct_state_owners_cannot_be_relabelled_as_actual_source(field):
    # Exact byte equality is a stronger fixture requirement than energy
    # parity: parallel integral reductions need not repeat in bitwise order.
    # Build BOTH native HF owners serially, then restore the caller's setting.
    threads = core.get_num_threads()
    try:
        core.set_num_threads(1)
        b, provider, _ = _physical()
        embedding, _ = _frame(b, translation=1)
        foreign, foreign_provider, _ = _physical()
        foreign_embedding, _ = _frame(foreign, translation=1)
    finally:
        core.set_num_threads(threads)
    assert b.reference.state.state_identity_sha256 == foreign.reference.state.state_identity_sha256
    target = SimpleNamespace(**vars(b))
    if field in ("reference", "basis"):
        setattr(target, field, getattr(foreign, field))
    elif field == "provider":
        provider = foreign_provider
    else:
        embedding = foreign_embedding
    with pytest.raises((ValueError, RuntimeError), match="state|source|owner|basis|allocation|identity"):
        _make(target, provider, embedding)


@pytest.mark.parametrize("nk", [1, 2])
def test_actual_explicit_frozen_core_pair_generation_never_reintroduces_core(nk):
    b, provider, dense = _physical("he2", nk, frozen=True)
    embedding, frame = _frame(b)
    assert b.reference.state.n_frozen_core == 1 and dense["o"] == nk and dense["v"] == 2*nk
    result = _make(b, provider, embedding)
    _check(b, provider, embedding, result, _oracle(b, dense, embedding))
    torus = _torus(frame)
    np.testing.assert_allclose(torus.pair.conj().T@torus.s@torus.pair,
        np.eye(embedding.memory.pair_dimension), atol=5e-11)
    assert result.state is b.hf.state and result.state is not b.unfrozen_hf.state


def test_detached_copies_readonly_receipts_and_owners_outlive_all_input_wrappers():
    b, provider, dense = _physical("he", 2)
    embedding, frame = _frame(b)
    result = _make(b, provider, embedding)
    c, eps, occupations = result.coefficients_copy(), result.energies_copy(), result.original_pno_occupations_copy()
    generation = result.generation_coefficients_copy()
    with pytest.raises(ValueError):
        generation[:] = 123
    detached = result.generation_coefficients_copy()
    detached.setflags(write=True)
    detached[:] = 123
    result.coefficients_copy()[:] = 123
    result.energies_copy()[:] = 456
    result.original_pno_occupations_copy()[:] = 789
    options = result.options
    options.maximum_exported_gram_error = 0
    pno = options.pno
    pno.occupation_cutoff = 1.0
    options.pno = pno
    assert result.options.pno.occupation_cutoff == 0
    assert result.options.maximum_exported_gram_error > 0
    state, context = result.state, result.context
    del b, provider, dense, embedding, frame
    gc.collect()
    assert result.state is state and result.context is context
    np.testing.assert_array_equal(result.coefficients_copy(), c)
    np.testing.assert_array_equal(result.energies_copy(), eps)
    np.testing.assert_array_equal(result.original_pno_occupations_copy(), occupations)
    np.testing.assert_array_equal(result.generation_coefficients_copy(), generation)
    for name in ("pnos", "embedding", "semicanonical", "generation_result"):
        assert not hasattr(result, name)
