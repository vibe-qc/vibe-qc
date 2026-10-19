"""Finite Bloch Fourier sums, explicit supports and independently checked phases."""
import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta, _vibeqc_core as core, _aopair_ft as reference
from tests.test_manta_fourier import _basis, _quadrature, VECTORS


LATTICE = np.array([[1.2, .3, -.1], [.1, 1.4, .2], [-.2, .1, 1.6]])
IMAGES = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
CELLS = np.ascontiguousarray(IMAGES@LATTICE.T)
MOMENTA = np.array([[0., 0., 0.], [.23, -.31, .17], [-.23, .31, -.17]])


def _oracle(basis, cutoff):
    shells = basis.shells()
    sizes = [2*s.l+1 if s.pure else (s.l+1)*(s.l+2)//2 for s in shells]
    offsets = np.cumsum([0]+sizes)
    out = np.zeros((len(MOMENTA), basis.nbasis, basis.nbasis, len(VECTORS)), complex)
    calls = 0
    for cell in CELLS:
        for ia, a in enumerate(shells):
            for ib, b in enumerate(basis.shells()):
                b.origin = (np.asarray(b.origin)+cell).tolist()
                if cutoff is not None and np.linalg.norm(np.asarray(a.origin)-b.origin)>cutoff:
                    continue
                tile = _quadrature(a, b, VECTORS).transpose(1, 2, 0)
                out[:, offsets[ia]:offsets[ia+1], offsets[ib]:offsets[ib+1]] += np.exp(1j*(MOMENTA@cell))[:, None, None, None]*tile
                calls += 1
    return out, calls


@pytest.mark.parametrize("pure", [False, True])
@pytest.mark.parametrize("cutoff", [None, 0., .9, 1.7])
def test_complete_finite_sum_against_independent_quadrature(pure, cutoff):
    basis = _basis(pure=pure)
    result = manta.compute_bloch_fourier_panel(basis, VECTORS, CELLS, MOMENTA, cutoff)
    expected, calls = _oracle(basis, cutoff)
    np.testing.assert_allclose(result.values, expected, atol=3e-12, rtol=3e-12)
    p = manta.plan_bloch_fourier(basis, len(VECTORS), len(CELLS), len(MOMENTA), cutoff)
    r = result.receipt
    assert result.shape == [len(MOMENTA), basis.nbasis, basis.nbasis, len(VECTORS)]
    assert r.candidate_shell_pairs == p.candidate_shell_pairs == len(CELLS)*basis.nshells**2
    assert r.shell_calls == calls
    assert r.skipped_shell_pairs == r.candidate_shell_pairs-calls
    assert r.primitive_pairs == 4*calls <= p.primitive_pairs
    assert r.vector_primitive_products == len(VECTORS)*r.primitive_pairs
    assert r.phase_products == p.phase_products == len(CELLS)*len(MOMENTA)
    assert r.phase_accumulations == len(MOMENTA)*r.integral_elements <= p.phase_accumulations
    assert r.modeled_peak_bytes == p.modeled_peak_bytes
    assert r.convention == "exp(+i*k.R)*exp(-i*q.r)"
    if pure:
        other = core.ao_pair_fourier_transform_bloch_multi_cxx(basis, VECTORS, CELLS, MOMENTA, 0., None, -1. if cutoff is None else cutoff)
        # The older raw FT table uses r^l Y_lm for l>0. Convert its AO
        # axes to the libint axial-shell convention before comparing.
        # The overlap test below independently fixes the physical convention.
        scale = np.concatenate([np.full(2*s.l+1, np.sqrt(4*np.pi/(2*s.l+1)) if s.l else 1.)
                                for s in basis.shells()])
        converted = np.asarray(other)*scale[None, :, None, None]*scale[None, None, :, None]
        np.testing.assert_allclose(result.values, converted, atol=3e-12, rtol=3e-12)


@pytest.mark.parametrize("pure", [False, True])
def test_zero_cell_zero_fourier_argument_is_physical_libint_overlap(pure):
    basis = _basis(pure=pure)
    result = manta.compute_bloch_fourier_panel(basis, np.zeros((1, 3)), np.zeros((1, 3)), MOMENTA)
    for overlap in result.values:
        np.testing.assert_allclose(overlap[:, :, 0], vq.compute_overlap(basis), atol=2e-13, rtol=2e-13)


def test_complex_phase_sign_reciprocal_periodicity_and_global_origin():
    basis = _basis(); shift = np.array([.3, -.2, .7])
    compute = manta.compute_bloch_fourier_panel
    result = compute(basis, VECTORS, CELLS, MOMENTA)
    np.testing.assert_allclose(compute(basis, -VECTORS, CELLS, -MOMENTA).values, result.values.conj(), atol=3e-12)
    reciprocal = 2*np.pi*np.linalg.inv(LATTICE).T[:, 1]
    np.testing.assert_allclose(compute(basis, VECTORS, CELLS, MOMENTA+reciprocal).values, result.values, atol=3e-12)
    shifted = compute(_basis(shift=shift), VECTORS, CELLS, MOMENTA)
    np.testing.assert_allclose(shifted.values, result.values*np.exp(-1j*(VECTORS@shift))[None, None, None, :], atol=3e-12)
    assert shifted.receipt.source_identity != result.receipt.source_identity
    assert np.max(np.abs(result.values.imag)) > .1


@pytest.mark.parametrize("pure", [False, True])
def test_maximum_angular_channel_and_momentum_transfer_adjoint(pure):
    basis = _basis((6,), pure)
    q = VECTORS[1:2]
    forward = manta.compute_bloch_fourier_panel(basis, q, CELLS, MOMENTA).values
    reverse = manta.compute_bloch_fourier_panel(basis, -q, CELLS, MOMENTA-q[0]).values
    np.testing.assert_allclose(forward.conj().transpose(0, 2, 1, 3), reverse, atol=3e-11, rtol=3e-11)
    overlap = manta.compute_bloch_fourier_panel(basis, np.zeros((1, 3)), CELLS, MOMENTA).values
    np.testing.assert_allclose(overlap.conj().transpose(0, 2, 1, 3), overlap, atol=3e-11, rtol=3e-11)


def test_momentum_and_vector_batching_share_primitive_work_and_preserve_domains():
    basis = _basis()
    result = manta.compute_bloch_fourier_panel(basis, VECTORS, CELLS, MOMENTA)
    separate = [manta.compute_bloch_fourier_panel(basis, VECTORS, CELLS, k[None, :]) for k in MOMENTA]
    np.testing.assert_allclose(np.concatenate([r.values for r in separate]), result.values, atol=2e-13)
    assert all(r.receipt.primitive_pairs == result.receipt.primitive_pairs for r in separate)
    panels = [manta.compute_bloch_fourier_panel(basis, q[None, :], CELLS, MOMENTA) for q in VECTORS]
    np.testing.assert_allclose(np.concatenate([r.values for r in panels], axis=-1), result.values, atol=2e-13)
    repeated = manta.compute_bloch_fourier_panel(basis, VECTORS, np.vstack([CELLS, CELLS]), MOMENTA)
    np.testing.assert_allclose(repeated.values, 2*result.values, atol=3e-12)
    assert repeated.receipt.source_identity != result.receipt.source_identity
    copy = result.values; copy[:] = 0
    assert np.max(np.abs(result.values)) > .1


@pytest.mark.parametrize("empty", [0, 1, 2])
def test_empty_axes_are_well_defined_zero_sums(empty):
    arrays = [VECTORS, CELLS, MOMENTA]; arrays[empty] = np.empty((0, 3))
    basis = _basis()
    r = manta.compute_bloch_fourier_panel(basis, *arrays)
    assert r.values.shape == (len(arrays[2]), basis.nbasis, basis.nbasis, len(arrays[0]))
    assert not np.any(r.values)
    assert r.receipt.shell_calls == r.receipt.phase_products == r.receipt.primitive_pairs == 0


def test_physical_cutoff_boundary_is_inclusive_and_not_a_precision_screen():
    basis = _basis((0,))
    cell = np.array([[3., 4., 0.]])
    q = k = np.zeros((1, 3))
    boundary = manta.compute_bloch_fourier_panel(basis, q, cell, k, 5.)
    outside = manta.compute_bloch_fourier_panel(basis, q, cell, k, np.nextafter(5., 0.))
    assert boundary.receipt.shell_calls == 1 and np.linalg.norm(boundary.values) > 0.
    assert outside.receipt.shell_calls == 0 and not np.any(outside.values)


@pytest.mark.parametrize("field,plan_field", [("memory_bytes", "modeled_peak_bytes"), ("shell_calls", "shell_calls"),
    ("primitive_quartets", "primitive_pairs"), ("scalar_terms", "scalar_terms"), ("kernel_terms", "kernel_terms")])
def test_aggregate_resource_boundaries_precede_vector_payload_copy(field, plan_field):
    basis = _basis(); budget = manta.ResourceBudget(); budget.other_live_bytes = 31937
    plan = manta.plan_bloch_fourier(basis, len(VECTORS), len(CELLS), len(MOMENTA), budget=budget)
    setattr(budget, field, getattr(plan, plan_field))
    result = manta.compute_bloch_fourier_panel(basis, VECTORS, CELLS, MOMENTA, budget=budget)
    assert result.receipt.modeled_peak_bytes == plan.modeled_peak_bytes
    setattr(budget, field, getattr(plan, plan_field)-1)
    with pytest.raises(ValueError, match="budget"):
        manta.compute_bloch_fourier_panel(basis, np.full_like(VECTORS, np.nan), CELLS, MOMENTA, budget=budget)


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("kind", ["float32", "complex", "strided", "shape", "nan"])
def test_strict_input_contract(axis, kind):
    arrays = [VECTORS, CELLS, MOMENTA]
    arrays[axis] = {"float32": np.ones((2, 3), np.float32), "complex": np.ones((2, 3), complex),
                    "strided": np.ones((4, 3))[::2], "shape": np.ones((2, 2)), "nan": np.ones((2, 3))*np.nan}[kind]
    with pytest.raises(ValueError):
        manta.compute_bloch_fourier_panel(_basis(), *arrays)


def test_fortran_layout_cutoff_identity_and_finite_input_errors():
    basis = _basis()
    reference = manta.compute_bloch_fourier_panel(basis, VECTORS, CELLS, MOMENTA)
    alternate = manta.compute_bloch_fourier_panel(basis, *[np.asfortranarray(a) for a in (VECTORS, CELLS, MOMENTA)])
    np.testing.assert_array_equal(reference.values, alternate.values)
    assert reference.receipt.source_identity == alternate.receipt.source_identity
    assert manta.compute_bloch_fourier_panel(basis, VECTORS, CELLS, MOMENTA, 0.).receipt.source_identity != reference.receipt.source_identity
    for bad in (-1., np.nan, np.inf):
        with pytest.raises(ValueError, match="cutoff"):
            manta.plan_bloch_fourier(basis, 1, 1, 1, bad)
    with pytest.raises(ValueError, match="overflow"):
        manta.compute_bloch_fourier_panel(basis, VECTORS, np.full((1, 3), 1e308), np.ones((1, 3))*10)


@pytest.mark.parametrize("cutoff", [None, 0., 1.7])
def test_existing_periodic_fourier_adapters_keep_the_same_source(cutoff, monkeypatch):
    from vibeqc import output
    basis = _basis(pure=False); engine = vq.IntegralEngineOptions(backend="manta")
    records = []
    monkeypatch.setattr(output, "record", lambda *args, **kwargs: records.append((args, kwargs)))
    expected = manta.compute_bloch_fourier_panel(basis, VECTORS, CELLS, MOMENTA, cutoff)
    actual = reference.ao_pair_fourier_transform_bloch_multi(basis, VECTORS, CELLS, MOMENTA,
        pair_cutoff_bohr=cutoff, integral_engine=engine)
    np.testing.assert_array_equal(actual, expected.values)
    single = reference.ao_pair_fourier_transform_bloch(basis, VECTORS, CELLS, MOMENTA[1],
        pair_cutoff_bohr=cutoff, integral_engine=engine)
    np.testing.assert_array_equal(single, actual[1])
    evidence = records[0][1]["integral_provenance"]
    assert evidence["source_identity"] == expected.receipt.source_identity
    assert evidence["citation_methods"] == ["manta_bloch_fourier"]
    for kwargs in ({"screen_tol": 1e-5}, {"pair_weights": np.eye(basis.nbasis)}):
        with pytest.raises(ValueError, match="do not yet qualify|does not yet qualify"):
            reference.ao_pair_fourier_transform_bloch_multi(basis, VECTORS, CELLS, MOMENTA, integral_engine=engine, **kwargs)
    assert len(records) == 2
