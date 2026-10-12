"""Finite physical fitted-support tests with an independent Gaussian oracle.

Python-reference diagnostics do not certify a newly installed native runtime.
Native parametrizations require a core built from the matching source revision.
"""
from itertools import product
import json

import numpy as np
import pytest

from vibeqc import Atom, BasisSet, LatticeSumOptions, PeriodicSystem, ShellInfo
from vibeqc import _aopair_ft as ft
from vibeqc import aux_basis as aux


def _fixture(image=False, shift=None, permutation=False):
    lattice = np.array([[4.7, .8, -.3], [0., 5.2, .6], [0., 0., 5.8]])
    centers = np.array([[.2, .3, .4], [1.4, .6, .8]])
    exponents = [.7, 1.3]
    if image:
        centers[1] += lattice @ np.array([1, -1, 0])
    if shift is not None:
        centers += shift
    if permutation:
        centers = centers[::-1]
        exponents = exponents[::-1]
    system = PeriodicSystem(3, lattice, [Atom(1, c) for c in centers])
    basis = BasisSet(system.unit_cell_molecule(), [
        ShellInfo(i, 0, False, [a], [1.], c) for i, (a, c) in enumerate(zip(exponents, centers))
    ], "support-oracle", False)
    return system, basis


def _oracle(basis, lattice, vectors, k, radius):
    """Closed Gaussian-product integral over an oversized fixed integer cube."""
    result = np.zeros((2, 2, len(vectors)), dtype=complex)
    shells = basis.shells()
    for i, a in enumerate(shells):
        for j, b in enumerate(shells):
            A, B = np.asarray(a.origin), np.asarray(b.origin)
            alpha, beta = a.exponents[0], b.exponents[0]
            zeta = alpha + beta
            norm = a.coefficients[0] * b.coefficients[0] * (np.pi / zeta)**1.5
            for label in product(range(-4, 5), repeat=3):
                R = lattice @ label
                distance = A - B - R
                if distance @ distance > radius**2:
                    continue
                center = (alpha * A + beta * (B + R)) / zeta
                result[i, j] += norm * np.exp(-alpha * beta / zeta * (distance @ distance)) * np.exp(
                    -np.sum(vectors**2, axis=1) / (4 * zeta) - 1j * (vectors @ center) + 1j * (k @ R))
    return result


@pytest.mark.parametrize("backend", ["python", "cxx"])
@pytest.mark.parametrize("image,shift,permutation", [
    (False, None, False), (True, None, False), (False, [.31, -.27, .18], False),
    (False, None, True),
])
def test_physical_pair_ft_independent_gaussian_cube(monkeypatch, backend, image, shift, permutation):
    if backend == "cxx":
        from vibeqc._vibeqc_core import ao_pair_fourier_transform_bloch_cxx
        if "pair_cutoff_bohr" not in ao_pair_fourier_transform_bloch_cxx.__doc__:
            pytest.skip("matching native physical-support core is not installed")
    monkeypatch.setenv("VIBEQC_AOPAIR_FT_BACKEND", backend)
    system, basis = _fixture(image, shift, permutation)
    radius = 5.5
    cells = aux._classic_pair_cells(system, basis, radius, "physical_pairs", 100000)
    reciprocal = 2 * np.pi * np.linalg.inv(system.lattice).T
    q = reciprocal @ np.array([.23, -.17, .11])
    vectors = np.array([q, reciprocal[:, 0] + q, -reciprocal[:, 1] + q])
    k = reciprocal @ np.array([.31, .07, -.13])
    expected = _oracle(basis, system.lattice, vectors, k, radius)
    actual = ft.ao_pair_fourier_transform_bloch(basis, vectors, cells, k, pair_cutoff_bohr=radius)
    batch = ft.ao_pair_fourier_transform_bloch_multi(
        basis, vectors, cells, np.array([k, -k]), pair_cutoff_bohr=radius)
    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(batch[0], expected, atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(batch[1], _oracle(basis, system.lattice, vectors, -k, radius),
                               atol=2e-13, rtol=2e-13)


def test_covariance_and_historical_control(monkeypatch):
    monkeypatch.setenv("VIBEQC_AOPAIR_FT_BACKEND", "python")
    system, basis = _fixture()
    moved, moved_basis = _fixture(image=True)
    radius = 4.1
    vectors = np.array([[.1, .2, -.3], [.4, -.1, .2]])
    k = np.array([.23, -.11, .19])
    def evaluate(sys, bs, policy):
        cells = aux._classic_pair_cells(sys, bs, radius, policy)
        options = dict(pair_cutoff_bohr=radius) if policy == "physical_pairs" else {}
        return ft.ao_pair_fourier_transform_bloch(bs, vectors, cells, k, **options)
    initial = evaluate(system, basis, "physical_pairs")
    actual = evaluate(moved, moved_basis, "physical_pairs")
    displacement = system.lattice @ [1, -1, 0]
    shifts = np.array([[0., 0., 0.], displacement])
    for i in range(2):
        for j in range(2):
            phase = np.exp(1j * (k @ (shifts[i] - shifts[j])) - 1j * (vectors @ shifts[i]))
            np.testing.assert_allclose(actual[i, j], phase * initial[i, j], atol=2e-13, rtol=2e-13)
    assert np.max(np.abs(evaluate(moved, moved_basis, "historical_index_ball") - actual)) > .01


def _build(batched, system, basis, **kwargs):
    opts = LatticeSumOptions()
    opts.cutoff_bohr = 4.1
    common = dict(ke_cutoff=1.2, tail_ke_cutoff=2.1, tail_chunk_g=16, lat_opts=opts,
                  linear_dep_thr=1e-10, support_policy="physical_pairs", memory_byte_cap=32*1024**2,
                  image_candidate_cap=1000, reciprocal_candidate_cap=10000)
    common.update(kwargs)
    if batched:
        return aux.build_lpq_bloch_native_fft_shared_q(
            system, basis, basis, np.array([[0., 0., 0.], [.11, -.07, .03]]), np.zeros(3), **common)
    return aux.build_lpq_bloch_native_fft(system, basis, basis, np.zeros(3), np.zeros(3), **common)


@pytest.mark.parametrize("batched", [False, True])
def test_source_bound_cache_and_exact_records(monkeypatch, batched):
    monkeypatch.setenv("VIBEQC_AOPAIR_FT_BACKEND", "python")
    system, basis = _fixture()
    records, cache = [], {}
    factors = _build(batched, system, basis, _q_metric_cache=cache, _support_recorder=records.append)
    repeated = _build(batched, system, basis, _q_metric_cache=cache)
    np.testing.assert_array_equal(factors, repeated)
    record = json.loads(records[0])
    assert record["descriptor"]["support_policy"] == "physical_pairs"
    assert len(record["reciprocal_tail"]) > 0
    assert len(record["metric_eigenvalues"]) == 2
    assert sum(record["metric_keep"]) == record["rank"]
    assert record["pair_keep_ao"] is None
    entry = next(iter(cache.values()))
    for array in (entry.eigvals, entry.eigvecs, entry.keep):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)
    for changed in (dict(ke_cutoff=1.3), dict(tail_ke_cutoff=2.2),
                    dict(linear_dep_thr=2e-10), dict(support_policy="historical_index_ball"),
                    dict(omega_screen=.4)):
        with pytest.raises(ValueError, match="different source"):
            _build(batched, system, basis, _q_metric_cache=cache, **changed)
    moved, moved_basis = _fixture(image=True)
    with pytest.raises(ValueError, match="different source"):
        _build(batched, moved, moved_basis, _q_metric_cache=cache)


@pytest.mark.parametrize("batched", [False, True])
def test_explicit_capacity_and_unsupported_screen_refusal(monkeypatch, batched):
    monkeypatch.setenv("VIBEQC_AOPAIR_FT_BACKEND", "python")
    system, basis = _fixture()
    for changes in (dict(memory_byte_cap=10), dict(image_candidate_cap=1),
                    dict(reciprocal_candidate_cap=1)):
        with pytest.raises(MemoryError):
            _build(batched, system, basis, **changes)
    for changes in (dict(fit_screen_threshold=1e-8), dict(fit_pair_list=np.array([[0, 0]]))):
        with pytest.raises(NotImplementedError, match="complete AO pairs"):
            _build(batched, system, basis, **changes)
    with pytest.raises(MemoryError, match="record"):
        _build(batched, system, basis, _support_recorder=lambda _: None, _support_record_byte_cap=64)


def test_physical_refuses_stale_native(monkeypatch):
    import vibeqc._vibeqc_core as core
    monkeypatch.setenv("VIBEQC_AOPAIR_FT_BACKEND", "cxx")
    monkeypatch.setattr(core, "ao_pair_fourier_transform_bloch_cxx", lambda *a: None)
    _, basis = _fixture()
    with pytest.raises(RuntimeError, match="updated installed native core"):
        ft.ao_pair_fourier_transform_bloch(
            basis, np.zeros((1, 3)), np.zeros((1, 3)), np.zeros(3), pair_cutoff_bohr=3.)


def test_gamma_tail_boundary_is_an_exact_partition(monkeypatch):
    monkeypatch.setenv("VIBEQC_AOPAIR_FT_BACKEND", "python")
    system, basis = _fixture()
    reciprocal = 2 * np.pi * np.linalg.inv(system.lattice).T
    boundary = reciprocal[:, 0]
    energy = float(boundary @ boundary) / 2 * (1 - 8 * np.finfo(float).eps)
    base = aux._classic_reciprocal_mesh(system, np.zeros(3), energy, 1024**2, 10000)
    assert not any(np.linalg.norm(row - boundary) < 1e-13 for row in base)
    small = _build(False, system, basis, ke_cutoff=energy, tail_ke_cutoff=2.1)
    complete = _build(False, system, basis, ke_cutoff=2.1, tail_ke_cutoff=None)
    # Operator equality avoids arbitrary eigensystem phase choices.
    a, b = small.reshape(len(small), -1), complete.reshape(len(complete), -1)
    np.testing.assert_allclose(a.conj().T @ a, b.conj().T @ b, atol=3e-13, rtol=3e-13)


def test_cache_refuses_reciprocal_enumeration_convention_change(monkeypatch):
    monkeypatch.setenv("VIBEQC_AOPAIR_FT_BACKEND", "python")
    system, basis = _fixture()
    cache = {}
    _build(False, system, basis, support_policy="historical_index_ball", _q_metric_cache=cache)
    with pytest.raises(ValueError, match="different source"):
        _build(False, system, basis, support_policy="historical_index_ball", memory_byte_cap=None,
               _q_metric_cache=cache)


def test_sr_record_preserves_selector_and_actual_reciprocal_inputs(monkeypatch):
    """Metadata wiring only: no native SR acceptance is inferred from a stub."""
    import vibeqc._vibeqc_core as core
    system, basis = _fixture()
    seen = {}
    def integrals(orbital, auxiliary, system, q, k, vectors, *args):
        seen["vectors"] = vectors.copy()
        return (np.asfortranarray(np.diag([1., 2.]).astype(complex)),
                np.ones((len(k), 2, 2, 2), dtype=complex),
                int(np.count_nonzero(np.sum(vectors**2, axis=1) > 1e-24)))
    monkeypatch.setattr(core, "compute_gdf_range_separated_integrals", integrals)
    records = []
    batch = aux._build_lpq_range_separated_shared_q(
        system, basis, basis, np.zeros((1, 3)), np.zeros(3), omega=.7,
        pair_cutoff=4.1, auxiliary_cutoff=5.2, ke_cutoff=1.2, linear_dep_thr=1.5,
        memory_byte_cap=32*1024**2, native_workspace_byte_cap=1024**2,
        image_candidate_cap=1000, reciprocal_candidate_cap=10000, _support_recorder=records.append)
    assert batch.executed_support == records[0]
    record = json.loads(records[0])
    assert record["metric_keep"] == [False, True]
    assert record["metric_eigenvalues"] == [1., 2.]
    np.testing.assert_array_equal(record["reciprocal_vectors"], seen["vectors"])
    assert "not integral call trace" in record["selector_evidence"]
    assert "1e-24" in record["reciprocal_evidence"]


def _cube_fit(shift, policy):
    """The #438 H2 cube's Gamma classic fit, atom 0 moved by ``shift``."""
    import vibeqc as vq
    centers = [np.array([3.0, 3.0, 2.3]) + shift, np.array([3.3, 3.0, 3.7])]
    system = PeriodicSystem(3, np.eye(3) * 6.0, [Atom(1, c) for c in centers])
    basis = BasisSet(system.unit_cell_molecule(), "sto-3g")
    auxiliary = aux.make_modrho_aux_basis(
        aux.make_aux_basis_set(system.unit_cell_molecule(), aux_name="def2-svp-jk"),
        system.unit_cell_molecule())
    options = LatticeSumOptions()
    options.cutoff_bohr = 12.0
    L = aux.build_lpq_native_fft(
        system, basis, auxiliary, ke_cutoff=40.0, lat_opts=options,
        support_policy=policy, memory_byte_cap=(4 * 1024**3 if policy else None))
    flat = np.asarray(L).reshape(L.shape[0], -1)
    return flat.T @ flat


def test_gamma_fit_physical_support_is_label_invariant():
    """#466 on the Gamma builder (RIJCOSX's fit): the fitted four-centre
    tensor of physical pairs does not depend on which lattice image of an
    atom the input names; the historical index ball does."""
    shift = np.array([0.0, 0.0, -6.0])
    physical = _cube_fit(np.zeros(3), "physical_pairs")
    np.testing.assert_allclose(_cube_fit(shift, "physical_pairs"), physical,
                               rtol=0, atol=1e-11)
    historical = _cube_fit(np.zeros(3), None)
    assert np.max(np.abs(_cube_fit(shift, None) - historical)) > 1e-7
