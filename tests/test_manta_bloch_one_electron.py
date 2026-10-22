"""Finite S/T supports, real-space blocks, Bloch phases and aggregate admission."""
import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta
from tests.test_manta_one_electron import _basis, _quadrature
from tests.test_manta_bloch_fourier import LATTICE, CELLS, MOMENTA, IMAGES


def _cells(system, translations=CELLS, indices=IMAGES):
    result = []
    for vector, index in zip(translations, indices, strict=True):
        cell = vq.LatticeCell(system, index)
        np.testing.assert_array_equal(cell.r_cart, vector)
        result.append(cell)
    return result


@pytest.mark.parametrize('operation', ['overlap', 'kinetic'])
@pytest.mark.parametrize('pure', [False, True])
@pytest.mark.parametrize('cutoff', [None, .9, 1.7])
def test_finite_cell_blocks_and_bloch_sum_against_libint(operation, pure, cutoff):
    basis = _basis(pure=pure)
    system = vq.PeriodicSystem(3, LATTICE, [vq.Atom(2, s.origin) for s in basis.shells()])
    reference = getattr(vq, f'compute_{operation}_lattice_explicit')(basis, system, _cells(system), 0. if cutoff is None else cutoff)
    expected = np.asarray(reference.blocks)
    result = manta.compute_bloch_one_electron(basis, operation, CELLS, MOMENTA, cutoff)
    np.testing.assert_allclose(result.cell_values, expected, atol=3e-12, rtol=3e-12)
    np.testing.assert_allclose(result.values, np.einsum('kr,rij->kij', np.exp(1j*MOMENTA@CELLS.T), expected), atol=4e-12, rtol=4e-12)
    np.testing.assert_allclose(result.values, result.values.conj().transpose(0, 2, 1), atol=3e-12)
    p, r = result.plan, result.receipt
    assert p.shape == [len(MOMENTA), basis.nbasis, basis.nbasis]
    assert p.cell_shape == list(expected.shape)
    assert r.candidate_shell_pairs == p.candidate_shell_pairs == len(CELLS)*basis.nshells**2
    assert r.skipped_shell_pairs + r.shell_calls == r.candidate_shell_pairs
    assert r.primitive_pairs == 4*r.shell_calls <= p.primitive_pairs
    assert r.phase_products == p.phase_products == len(CELLS)*len(MOMENTA)
    assert r.phase_accumulations == len(MOMENTA)*r.integral_elements <= p.phase_accumulations
    assert r.hermiticity_residual < 3e-12
    assert r.operation == operation and r.convention == 'exp(+i*k.R)'
    assert r.modeled_peak_bytes == p.modeled_peak_bytes


@pytest.mark.parametrize('operation', ['overlap', 'kinetic'])
@pytest.mark.parametrize('pure', [False, True])
def test_highest_angular_tiles_against_independent_polynomial_quadrature(operation, pure):
    basis = _basis((6,), pure, contracted=False)
    translations = CELLS[1:3]
    result = manta.compute_bloch_one_electron(basis, operation, translations, MOMENTA)
    for cell, actual in zip(translations, result.cell_values, strict=True):
        a, b = basis.shells()[0], basis.shells()[0]
        b.origin = (np.asarray(b.origin)+cell).tolist()
        expected = _quadrature(a, b, operation, np.zeros(3), np.zeros(3, int))
        np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize('operation', ['overlap', 'kinetic'])
def test_phases_support_permutation_duplicate_images_and_global_translation(operation):
    basis = _basis()
    compute = lambda b, cells, k: manta.compute_bloch_one_electron(b, operation, cells, k)
    result = compute(basis, CELLS, MOMENTA)
    np.testing.assert_array_equal(compute(basis, CELLS, -MOMENTA).values, result.values.conj())
    np.testing.assert_allclose(compute(basis, CELLS, MOMENTA+2*np.pi*np.linalg.inv(LATTICE).T[:, 0]).values, result.values, atol=3e-12)
    translated = compute(_basis(shift=(.4, -.2, .7)), CELLS, MOMENTA)
    np.testing.assert_allclose(translated.cell_values, result.cell_values, atol=3e-12)
    np.testing.assert_allclose(translated.values, result.values, atol=3e-12)
    assert translated.receipt.source_identity != result.receipt.source_identity
    permuted = compute(basis, np.ascontiguousarray(CELLS[::-1]), MOMENTA)
    np.testing.assert_array_equal(permuted.cell_values, result.cell_values[::-1])
    np.testing.assert_allclose(permuted.values, result.values, atol=3e-12)
    assert permuted.receipt.source_identity != result.receipt.source_identity
    doubled = compute(basis, np.vstack([CELLS, CELLS]), MOMENTA)
    np.testing.assert_array_equal(doubled.cell_values, np.concatenate([result.cell_values]*2))
    np.testing.assert_allclose(doubled.values, 2*result.values, atol=3e-12)
    single = [compute(basis, CELLS, k[None, :]) for k in MOMENTA]
    np.testing.assert_array_equal(np.concatenate([r.values for r in single]), result.values)
    assert all(r.receipt.primitive_pairs == result.receipt.primitive_pairs for r in single)
    assert np.max(np.abs(result.values.imag)) > .01


@pytest.mark.parametrize('operation', ['overlap', 'kinetic'])
def test_image_relabeling_covariance_on_pair_complete_support(operation):
    basis = _basis((0, 1)); lattice = np.diag([4., 5., 6.]); cutoff = 3.5
    def compute(b):
        system = vq.PeriodicSystem(3, lattice, [vq.Atom(2, s.origin) for s in b.shells()])
        cells = vq.pair_complete_lattice_cells(b, system, cutoff)
        return manta.compute_bloch_one_electron(b, operation, np.array([c.r_cart for c in cells]), MOMENTA, cutoff)
    shells = basis.shells(); shells[1].origin = (np.asarray(shells[1].origin)+lattice[:, 0]).tolist()
    moved = vq.BasisSet(vq.Molecule([vq.Atom(2, s.origin) for s in shells], 0, 1), shells, 'manta-relabel', True)
    expected = compute(basis).values
    shifts = np.array([[0., 0., 0.]]+[lattice[:, 0].tolist()]*3)
    u = np.exp(-1j*MOMENTA@shifts.T)
    expected = u.conj()[:, :, None]*expected*u[:, None, :]
    np.testing.assert_allclose(compute(moved).values, expected, atol=3e-12, rtol=3e-12)


@pytest.mark.parametrize('operation', ['overlap', 'kinetic'])
def test_scalar_chain_against_converged_analytic_gaussian_series(operation):
    basis = _basis((0,), contracted=False); alpha = basis.shells()[0].exponents[0]
    images = np.arange(-8, 9); translations = np.zeros((len(images), 3)); translations[:, 0] = 3*images
    # For normalized same-exponent s Gaussians, S_R=exp(-alpha*R^2/2)
    # and T_R=alpha/2*(3-alpha*R^2)*S_R. The omitted tails are below 1e-100.
    overlap = np.exp(-alpha*np.sum(translations**2, axis=1)/2)
    blocks = overlap if operation == 'overlap' else alpha/2*(3-alpha*np.sum(translations**2, axis=1))*overlap
    expected = np.exp(1j*MOMENTA@translations.T)@blocks
    actual = manta.compute_bloch_one_electron(basis, operation, translations, MOMENTA)
    np.testing.assert_allclose(actual.cell_values[:, 0, 0], blocks, atol=1e-14, rtol=2e-13)
    np.testing.assert_allclose(actual.values[:, 0, 0], expected, atol=2e-14, rtol=2e-13)


def test_asymmetric_support_is_reported_without_completion_or_matrix_repair():
    basis = _basis()
    result = manta.compute_bloch_one_electron(basis, 'overlap', CELLS[1:2], MOMENTA)
    expected = np.exp(1j*(MOMENTA@CELLS[1]))[:, None, None]*result.cell_values[0]
    np.testing.assert_array_equal(result.values, expected)
    residual = np.max(np.abs(result.values-result.values.conj().transpose(0, 2, 1)))
    assert residual > .1 and result.receipt.hermiticity_residual == pytest.approx(residual)


def test_zero_axes_keep_cell_blocks_and_owned_exports():
    basis = _basis()
    cell_only = manta.compute_bloch_one_electron(basis, 'overlap', CELLS, np.empty((0, 3)))
    full = manta.compute_bloch_one_electron(basis, 'overlap', CELLS, MOMENTA)
    np.testing.assert_array_equal(cell_only.cell_values, full.cell_values)
    assert cell_only.values.shape == (0, basis.nbasis, basis.nbasis)
    assert cell_only.receipt.shell_calls == full.receipt.shell_calls
    assert cell_only.receipt.phase_products == 0
    empty = manta.compute_bloch_one_electron(basis, 'overlap', np.empty((0, 3)), MOMENTA)
    assert empty.cell_values.shape == (0, basis.nbasis, basis.nbasis)
    assert not np.any(empty.values) and empty.receipt.shell_calls == 0
    for name in ('values', 'cell_values'):
        exported = getattr(full, name); saved = exported.copy(); exported[:] = 0
        np.testing.assert_array_equal(getattr(full, name), saved)


def test_physical_cutoff_exact_boundary_and_coincident_zero():
    basis = _basis((0,)); cell = np.array([[3., 4., 0.]])
    for cutoff, calls in [(None, 1), (5., 1), (np.nextafter(5., 0.), 0), (0., 0)]:
        result = manta.compute_bloch_one_electron(basis, 'overlap', cell, MOMENTA, cutoff)
        assert result.receipt.shell_calls == calls
        assert bool(np.any(result.cell_values)) == bool(calls)
    zero = manta.compute_bloch_one_electron(basis, 'overlap', np.zeros((1, 3)), MOMENTA, 0.)
    assert zero.receipt.shell_calls == 1


@pytest.mark.parametrize('field, plan_field', [('memory_bytes', 'modeled_peak_bytes'), ('shell_calls', 'shell_calls'),
    ('primitive_quartets', 'primitive_pairs'), ('scalar_terms', 'scalar_terms'), ('kernel_terms', 'kernel_terms')])
def test_aggregate_budget_boundary_before_numerical_input_scan(field, plan_field):
    basis = _basis(); budget = manta.ResourceBudget(); budget.other_live_bytes = 31793
    p = manta.plan_bloch_one_electron(basis, 'kinetic', len(CELLS), len(MOMENTA), budget=budget)
    setattr(budget, field, getattr(p, plan_field))
    r = manta.compute_bloch_one_electron(basis, 'kinetic', CELLS, MOMENTA, budget=budget)
    assert r.receipt.modeled_peak_bytes == p.modeled_peak_bytes
    setattr(budget, field, getattr(p, plan_field)-1)
    with pytest.raises(ValueError, match='budget'):
        manta.compute_bloch_one_electron(basis, 'kinetic', np.full_like(CELLS, np.nan), MOMENTA, budget=budget)


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('kind', ['float32', 'complex', 'strided', 'shape', 'nan'])
def test_strict_arrays(axis, kind):
    arrays = [CELLS, MOMENTA]
    arrays[axis] = {'float32': np.ones((2, 3), np.float32), 'complex': np.ones((2, 3), complex),
                    'strided': np.ones((4, 3))[::2], 'shape': np.ones((2, 2)), 'nan': np.full((2, 3), np.nan)}[kind]
    with pytest.raises(ValueError):
        manta.compute_bloch_one_electron(_basis(), 'overlap', *arrays)


def test_input_layout_identity_invalid_operation_and_finite_overflow():
    basis = _basis()
    result = manta.compute_bloch_one_electron(basis, 'overlap', CELLS, MOMENTA)
    alternate = manta.compute_bloch_one_electron(basis, 'overlap', np.asfortranarray(CELLS), np.asfortranarray(MOMENTA))
    np.testing.assert_array_equal(result.values, alternate.values)
    assert result.receipt.source_identity == alternate.receipt.source_identity
    assert result.receipt.source_identity != manta.compute_bloch_one_electron(basis, 'kinetic', CELLS, MOMENTA).receipt.source_identity
    assert result.receipt.source_identity != manta.compute_bloch_one_electron(basis, 'overlap', CELLS, MOMENTA, 0.).receipt.source_identity
    for bad in (-1., np.nan, np.inf):
        with pytest.raises(ValueError, match='cutoff'):
            manta.plan_bloch_one_electron(basis, 'overlap', 1, 1, bad)
    for operation in ('potential', 'moment', 'other'):
        with pytest.raises(ValueError, match='operation'):
            manta.plan_bloch_one_electron(basis, operation, 1, 1)
    with pytest.raises(ValueError, match='overflow'):
        manta.compute_bloch_one_electron(basis, 'overlap', np.full((1, 3), 1e308), np.full((1, 3), 10.))
    plus = manta.compute_bloch_one_electron(basis, 'overlap', np.zeros((1, 3)), MOMENTA, 0.)
    minus = manta.compute_bloch_one_electron(basis, 'overlap', -np.zeros((1, 3)), MOMENTA, -0.)
    assert plus.receipt.source_identity == minus.receipt.source_identity


@pytest.mark.parametrize('operation', ['overlap', 'kinetic'])
@pytest.mark.parametrize('dim', [1, 2, 3])
def test_explicit_lattice_adapter_preserves_pair_complete_domain_and_records_source(operation, dim, monkeypatch):
    from vibeqc import output
    from vibeqc.symmetry_integrals_reduced import one_electron_lattice_cells
    basis = _basis(); lattice = np.diag([3., 4., 5.])
    system = vq.PeriodicSystem(dim, lattice, [vq.Atom(2, s.origin) for s in basis.shells()])
    options = vq.LatticeSumOptions(); options.pair_complete_1e = True; options.cutoff_bohr = 3.5
    cells = one_electron_lattice_cells(basis, system, options)
    records = []; monkeypatch.setattr(output, 'record', lambda *args, **kwargs: records.append((args, kwargs)))
    engine = vq.IntegralEngineOptions(backend='manta')
    actual = manta.compute_one_electron_lattice(basis, cells, operation=operation, pair_cutoff_bohr=options.cutoff_bohr, integral_engine=engine)
    expected = getattr(vq, f'compute_{operation}_lattice')(basis, system, options)
    np.testing.assert_array_equal([c.index for c in actual.cells], [c.index for c in expected.cells])
    np.testing.assert_array_equal([c.r_cart for c in actual.cells], [c.r_cart for c in expected.cells])
    np.testing.assert_allclose(actual.blocks, expected.blocks, atol=3e-12, rtol=3e-12)
    r = records[0][1]['integral_provenance']
    native = manta.compute_bloch_one_electron(basis, operation, np.array([c.r_cart for c in cells]), np.empty((0, 3)), options.cutoff_bohr)
    assert r['source_identity'] == native.receipt.source_identity
    assert r['citation_methods'] == ['manta_bloch_one_electron']
    assert r['shell_calls'] == native.receipt.shell_calls
    assert engine.budget.other_live_bytes == 0


def test_lattice_adapter_refuses_unsupported_engine_and_preflights_before_copy(monkeypatch):
    from vibeqc import output
    monkeypatch.setattr(output, 'record', lambda *args, **kwargs: pytest.fail('refused build recorded'))
    basis = _basis(); engine = vq.IntegralEngineOptions(backend='manta')
    for bad in (object(), vq.IntegralEngineOptions(backend='libint'), vq.IntegralEngineOptions(backend='manta', channel_batch=2)):
        with pytest.raises((TypeError, ValueError)):
            manta.compute_one_electron_lattice(basis, [], integral_engine=bad)
    engine.budget.shell_calls = 0
    # Count-only admission must fail before attempting to read any cell payload.
    with pytest.raises(ValueError, match='budget'):
        manta.compute_one_electron_lattice(basis, [object()], integral_engine=engine)
    engine.budget.shell_calls = 1000
    with pytest.raises(TypeError, match='LatticeCell'):
        manta.compute_one_electron_lattice(basis, [object()], integral_engine=engine)
