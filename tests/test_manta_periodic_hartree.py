"""Finite same-source periodic J: quadrature, energy response and admission."""
import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta
from tests.test_manta_fourier import _basis, _quadrature


LATTICE = np.array([[2.5, .2, .1], [0., 2.7, .3], [.2, .1, 2.9]])
INDICES = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
CELLS = np.ascontiguousarray(INDICES@LATTICE.T)
G = np.ascontiguousarray(INDICES[1:]@(2*np.pi*np.linalg.inv(LATTICE)))
K = np.array([[0., 0., 0.], [.2, -.1, .07], [-.17, .12, .14]])
WEIGHTS = np.array([.2, .3, .5])


def _density(n, seed=37):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(len(K), n, n))+1j*rng.normal(size=(len(K), n, n))
    return np.ascontiguousarray(x+x.conj().transpose(0, 2, 1))


def _compute(basis, density=None, **kwargs):
    args = dict(lattice=LATTICE, wave_vectors=G, translations=CELLS, momenta=K,
                weights=WEIGHTS, densities=_density(basis.nbasis) if density is None else density,
                reciprocal_batch=2)
    args.update(kwargs)
    return manta.compute_periodic_hartree(basis, **args)


def _quadrature_reference(basis, density, cutoff):
    # Integrate AO products independently, then use the explicit finite
    # Fourier representation of the zero-average Coulomb Green function.
    offsets = np.cumsum([0]+[2*s.l+1 if s.pure else (s.l+1)*(s.l+2)//2 for s in basis.shells()])
    pair = np.zeros((len(CELLS), basis.nbasis, basis.nbasis, len(G)), complex)
    calls = 0
    for ir, r in enumerate(CELLS):
        for ia, a in enumerate(basis.shells()):
            for ib, b in enumerate(basis.shells()):
                b.origin = (np.asarray(b.origin)+r).tolist()
                if cutoff is not None and np.linalg.norm(np.asarray(a.origin)-b.origin) > cutoff:
                    continue
                pair[ir, offsets[ia]:offsets[ia+1], offsets[ib]:offsets[ib+1]] = _quadrature(a, b, G).transpose(1, 2, 0)
                calls += 1
    phase = np.exp(1j*K@CELLS.T)
    rho = np.einsum('k,kmn,kr,rmng->g', WEIGHTS, density, phase.conj(), pair)
    v = 4*np.pi/np.sum(G*G, axis=1)/abs(np.linalg.det(LATTICE))
    j_cells = np.einsum('g,g,rmng->rmn', v, rho, pair.conj())
    j_k = np.einsum('kr,rmn->kmn', phase, j_cells)
    energy = .5*np.dot(v, np.abs(rho)**2)
    return j_cells, j_k, energy, calls


@pytest.mark.parametrize('pure', [False, True])
@pytest.mark.parametrize('cutoff', [None, 0., 2.7])
def test_complex_whole_mesh_quadrature_and_energy(pure, cutoff):
    basis = _basis(pure=pure); density = _density(basis.nbasis)
    result = _compute(basis, density, pair_cutoff_bohr=cutoff)
    cells, momenta, energy, calls = _quadrature_reference(basis, density, cutoff)
    np.testing.assert_allclose(result.cell_values, cells, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(result.values, momenta, atol=2e-12, rtol=2e-12)
    assert result.receipt.energy == pytest.approx(energy, abs=2e-12, rel=2e-12)
    p, r = result.plan, result.receipt
    assert r.panel_calls == p.panel_calls == 2*(1+len(CELLS))
    assert r.shell_calls == 4*calls <= p.shell_calls
    assert r.primitive_pairs == 4*r.shell_calls <= p.primitive_pairs
    assert r.vector_primitive_products == 2*r.primitive_pairs
    assert r.contraction_terms == p.contraction_terms
    assert r.modeled_peak_bytes == p.modeled_peak_bytes
    assert r.cell_imaginary_residual < 1e-13
    assert r.response_hermiticity_residual < 2e-13
    assert r.boundary == '3d-zero-average-g0-omitted'
    if cutoff is None:
        assert np.max(np.abs(result.values.imag)) > 1e-3
    # Raw outputs remain complex and each export owns its storage.
    values = result.values; values[:] = 100.
    assert np.max(np.abs(result.values)) < 100.


def test_energy_gradient_linearity_and_one_k_changes_other_k_responses():
    basis = _basis(); d = _density(basis.nbasis); delta = _density(basis.nbasis, 8)
    delta[0] = delta[2] = 0
    original = _compute(basis, d); response = _compute(basis, delta)
    h = 2e-5
    plus, minus = _compute(basis, d+h*delta), _compute(basis, d-h*delta)
    slope = np.einsum('k,kmn,knm->', WEIGHTS, delta, original.values).real
    assert (plus.receipt.energy-minus.receipt.energy)/(2*h) == pytest.approx(slope, abs=2e-10)
    np.testing.assert_allclose(plus.values, original.values+h*response.values, atol=1e-13)
    assert np.linalg.norm(response.values[0]) > .01
    assert original.receipt.source_identity == plus.receipt.source_identity
    assert original.receipt.density_identity != plus.receipt.density_identity
    np.testing.assert_array_equal(_compute(basis, np.zeros_like(d)).values, 0.)


def test_batching_time_reversal_reciprocal_relabeling_and_origin_covariance():
    basis = _basis(); d = _density(basis.nbasis); expected = _compute(basis, d)
    for batch in (1, 4, 17):
        actual = _compute(basis, d, reciprocal_batch=batch)
        np.testing.assert_allclose(actual.values, expected.values, atol=2e-13)
        assert actual.receipt.source_identity == expected.receipt.source_identity
        assert actual.receipt.density_identity == expected.receipt.density_identity
    tr = _compute(basis, d.conj(), momenta=-K)
    np.testing.assert_allclose(tr.values, expected.values.conj(), atol=2e-13)
    shift = np.array([.3, -.2, .1])
    np.testing.assert_allclose(_compute(_basis(shift=shift), d).values, expected.values, atol=2e-13)
    relabeled = _compute(basis, d, momenta=K+G[0])
    np.testing.assert_allclose(relabeled.values, expected.values, atol=2e-13)
    assert relabeled.receipt.source_identity != expected.receipt.source_identity
    changed_weights = _compute(basis, d, weights=np.array([.3, .2, .5]))
    assert changed_weights.receipt.source_identity != expected.receipt.source_identity
    assert np.linalg.norm(changed_weights.values-expected.values) > .01


@pytest.mark.parametrize('axis', ['wave_vectors', 'translations'])
def test_empty_support_is_zero_but_still_validates_density(axis):
    basis = _basis(); result = _compute(basis, **{axis: np.empty((0, 3))})
    assert result.values.shape == (len(K), basis.nbasis, basis.nbasis)
    assert result.receipt.shell_calls == result.receipt.panel_calls == 0
    np.testing.assert_array_equal(result.values, 0.)
    d = _density(basis.nbasis); d[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        _compute(basis, d, **{axis: np.empty((0, 3))})


@pytest.mark.parametrize('field, plan_field', [
    ('memory_bytes', 'modeled_peak_bytes'), ('shell_calls', 'shell_calls'),
    ('primitive_quartets', 'primitive_pairs'), ('scalar_terms', 'scalar_terms'), ('kernel_terms', 'kernel_terms'),
])
def test_aggregate_budget_exact_boundary_and_refusal_before_payload_copy(field, plan_field):
    basis = _basis(); p = manta.plan_periodic_hartree(basis, len(G), len(CELLS), len(K), 2)
    budget = manta.ResourceBudget(); setattr(budget, field, getattr(p, plan_field))
    result = _compute(basis, budget=budget)
    assert result.receipt.modeled_peak_bytes == p.modeled_peak_bytes
    setattr(budget, field, getattr(p, plan_field)-1)
    bad = _density(basis.nbasis); bad[0, 0, 0] = np.nan
    with pytest.raises((ValueError, RuntimeError), match='budget'):
        _compute(basis, bad, budget=budget)


@pytest.mark.parametrize('change, message', [
    ({'wave_vectors': np.vstack([G, np.zeros((1, 3))])}, 'G=0'),
    ({'wave_vectors': G[:1]}, 'inversion'),
    ({'translations': CELLS[:2]}, 'inversion'),
    ({'wave_vectors': np.vstack([G, G[:1]])}, 'duplicate'),
    ({'translations': np.vstack([CELLS, CELLS[:1]])}, 'duplicate'),
    ({'wave_vectors': G+.01}, 'lattice vectors'),
    ({'translations': CELLS+.01}, 'lattice vectors'),
    ({'weights': np.array([.2, -.2, 1.])}, 'nonnegative'),
    ({'weights': np.array([.2, .3, .6])}, 'sum to one'),
    ({'weights': np.array([.2, np.nan, .8])}, 'finite'),
    ({'momenta': K*np.nan}, 'finite'),
    ({'lattice': np.zeros((3, 3))}, 'nonsingular'),
    ({'reciprocal_batch': 0}, 'positive'),
    ({'pair_cutoff_bohr': -1.}, 'cutoff'),
])
def test_invalid_physical_sources_refused(change, message):
    with pytest.raises((ValueError, RuntimeError), match=message):
        _compute(_basis(), **change)


def test_strict_shapes_layouts_types_and_raw_hermiticity():
    basis = _basis(); d = _density(basis.nbasis); expected = _compute(basis, d)
    f = _compute(basis, np.asfortranarray(d), lattice=np.asfortranarray(LATTICE),
                 wave_vectors=np.asfortranarray(G), translations=np.asfortranarray(CELLS), momenta=np.asfortranarray(K))
    np.testing.assert_array_equal(f.values, expected.values)
    for change in ({'densities': d.real}, {'weights': WEIGHTS.astype(np.float32)},
                   {'wave_vectors': G[:, ::-1]}, {'densities': d[:, :, :-1]},
                   {'momenta': K[:, :2]}):
        with pytest.raises((ValueError, TypeError), match='float64|contiguous|shape'):
            _compute(basis, **change)
    d[0, 0, 1] += .1j
    with pytest.raises(ValueError, match='Hermitian'):
        _compute(basis, d)
    overflow = np.ones_like(d)*complex(1.7e308, 1.7e308)
    with pytest.raises(ValueError, match='finite'):
        _compute(basis, overflow, wave_vectors=np.empty((0, 3)))


def test_gamma_limit_and_permuted_mesh_with_nonuniform_weights():
    basis = _basis(); d = _density(basis.nbasis)
    gamma = _compute(basis, d[:1].copy(), momenta=K[:1].copy(), weights=np.ones(1))
    repeated = _compute(basis, np.repeat(d[:1], 3, axis=0), momenta=np.zeros_like(K))
    np.testing.assert_allclose(repeated.values, np.repeat(gamma.values, 3, axis=0), atol=1e-13)
    expected = _compute(basis, d); order = [2, 0, 1]
    permuted = _compute(basis, d[order], momenta=K[order], weights=WEIGHTS[order])
    np.testing.assert_allclose(permuted.values, expected.values[order], atol=2e-13)
    assert permuted.receipt.energy == pytest.approx(expected.receipt.energy, abs=2e-13)


def test_scalar_gaussian_reciprocal_sum_and_cutoff_convergence():
    from vibeqc import _vibeqc_core as core
    mol = vq.Molecule([vq.Atom(2, [.2, -.1, .4])], 0, 1)
    shell = core.ShellInfo(); shell.atom_index = 0; shell.l = 0; shell.pure = True
    shell.origin = [.2, -.1, .4]; shell.exponents = [1.]; shell.coefficients = [1.]
    basis = vq.BasisSet(mol, [shell], 'manta-periodic-s', False)
    lattice = np.diag([3., 3., 3.]); k = np.zeros((1, 3)); d = np.array([[[2.+0j]]])
    energies = []
    for extent in (1, 2, 3, 4, 5):
        indices = np.array([(i, j, l) for i in range(-extent, extent+1)
                            for j in range(-extent, extent+1) for l in range(-extent, extent+1)
                            if (i, j, l) != (0, 0, 0)])
        q = np.ascontiguousarray(indices*(2*np.pi/3.)); q2 = np.sum(q*q, axis=1)
        # Normalized s AO squared has transform exp(-G²/(8 alpha)).
        # This closed form uses no AO Fourier or contraction implementation.
        expected_j = 2/27*np.sum(4*np.pi/q2*np.exp(-q2/4))
        result = _compute(basis, d, lattice=lattice, wave_vectors=q, translations=np.zeros((1, 3)),
                          momenta=k, weights=np.ones(1), reciprocal_batch=17)
        assert result.values[0, 0, 0] == pytest.approx(expected_j, abs=3e-14)
        assert result.receipt.energy == pytest.approx(expected_j, abs=3e-14)
        energies.append(result.receipt.energy)
    assert all(a < b for a, b in zip(energies[:-2], energies[1:-1]))
    assert abs(energies[-1]-energies[-2]) < 1e-12


def _adapter_fixture():
    basis = _basis()
    system = vq.PeriodicSystem(3, LATTICE, [vq.Atom(2, s.origin) for s in basis.shells()])
    cells = [vq.LatticeCell(system, idx) for idx in INDICES]
    return basis, system, cells


@pytest.mark.parametrize('cutoff', [None, 0., 2.7])
def test_existing_whole_mesh_interfaces_preserve_source_normalization_and_receipts(cutoff, monkeypatch):
    from vibeqc import output
    from vibeqc.ewald_composed import compute_j_ewald_3d_ft_k_density_to_cells as build_cells
    from vibeqc.periodic_fock_multi_k import build_periodic_j_ewald3d_k_from_k_density as build_k
    basis, system, cells = _adapter_fixture(); density = _density(basis.nbasis)
    opts = vq.LatticeSumOptions()
    opts.pair_complete_1e = cutoff is not None
    if cutoff is not None: opts.cutoff_bohr = cutoff
    common = dict(lattice_opts=opts, ke_cutoff=4., cell_chunk_size=1)
    engine = vq.IntegralEngineOptions(backend='manta')
    events = []
    monkeypatch.setattr(output, 'record', lambda *args, **kw: events.append(kw))
    reference = build_cells(basis, system, density, K, WEIGHTS, cells, .3, chunk_size=7, **common)
    actual = build_cells(basis, system, density, K, WEIGHTS, cells, .7,
                         chunk_size=7, integral_engine=engine, **common)
    np.testing.assert_allclose(actual, reference, atol=2e-12, rtol=2e-12)
    k = build_k(basis, system, density, K, WEIGHTS, cells, .2,
                reciprocal_chunk_size=7, integral_engine=engine, **common)
    expected = np.einsum('kr,rmn->kmn', np.exp(1j*K@CELLS.T), reference)
    np.testing.assert_allclose(k, expected, atol=2e-12, rtol=2e-12)
    assert len(events) == 2
    a, b = (e['integral_provenance'] for e in events)
    assert a['source_identity'] == b['source_identity']
    assert a['density_identity'] == b['density_identity']
    assert a['citation_methods'] == ['manta_periodic_hartree']
    assert a['cell_imaginary_residual'] < 1e-13
    assert a['modeled_peak_bytes'] <= engine.budget.memory_bytes


def test_python_mesh_preflight_refuses_before_allocating_grid_and_never_reports_execution(monkeypatch):
    from vibeqc import aux_basis, output
    from vibeqc.ewald_composed import compute_j_ewald_3d_ft_k_density_to_cells as build
    basis, system, cells = _adapter_fixture(); budget = manta.ResourceBudget(); budget.memory_bytes = 1
    engine = vq.IntegralEngineOptions(backend='manta', budget=budget)
    monkeypatch.setattr(aux_basis, 'rsgdf_dense_g_mesh', lambda *a: pytest.fail('unadmitted grid allocation'))
    monkeypatch.setattr(output, 'record', lambda *a, **kw: pytest.fail('false executed receipt'))
    with pytest.raises((ValueError, RuntimeError), match='budget'):
        build(basis, system, _density(basis.nbasis), K, WEIGHTS, cells, .5,
              ke_cutoff=4., integral_engine=engine)


@pytest.mark.parametrize('engine, overrides', [
    (vq.IntegralEngineOptions(backend='libint'), {}),
    (vq.IntegralEngineOptions(backend='manta', fallback='libint'), {}),
    (vq.IntegralEngineOptions(backend='manta', channel_batch=True), {}),
    (vq.IntegralEngineOptions(backend='manta'), {'chunk_size': True}),
    (vq.IntegralEngineOptions(backend='manta'), {'cell_chunk_size': 2}),
    (vq.IntegralEngineOptions(backend='manta'), {'ke_cutoff': np.nan}),
])
def test_unqualified_adapter_options_refuse(engine, overrides):
    from vibeqc.ewald_composed import compute_j_ewald_3d_ft_k_density_to_cells as build
    basis, system, cells = _adapter_fixture(); opts = dict(ke_cutoff=4.); opts.update(overrides)
    with pytest.raises((ValueError, TypeError)):
        build(basis, system, _density(basis.nbasis), K, WEIGHTS, cells, .5, integral_engine=engine, **opts)
