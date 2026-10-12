"""Finite 3D Ewald nuclear composition and independently resolved components."""
import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta
from vibeqc.aux_basis import rsgdf_dense_g_mesh
from tests.test_manta_one_electron import _basis
from tests.test_manta_fourier import _quadrature


K = np.array([[0., 0., 0.], [.21, -.17, .09], [-.21, .17, -.09]])
LATTICE = np.array([[3., .2, .1], [0., 3.2, .3], [.1, .2, 3.4]])


def _fixture(pure=True):
    basis = _basis((0, 2), pure)
    system = vq.PeriodicSystem(3, LATTICE, [vq.Atom(2, s.origin) for s in basis.shells()])
    options = vq.LatticeSumOptions(); options.cutoff_bohr = 2.4; options.nuclear_cutoff_bohr = 4.5
    options.pair_complete_1e = True
    cells = vq.pair_complete_lattice_cells(basis, system, options.cutoff_bohr)
    translations = np.array([c.r_cart for c in cells])
    g = rsgdf_dense_g_mesh(system, 3.); g = np.ascontiguousarray(g[np.any(g != 0., axis=1)])
    return basis, system, options, cells, translations, g


def _budget():
    b = manta.ResourceBudget(); b.kernel_terms = 100_000_000_000_000
    return b


def _compute(fixture, *, alpha=.7, batch=4, k=K, budget=None):
    basis, system, options, cells, translations, g = fixture
    return manta.compute_periodic_nuclear(basis, system, g, translations, k, alpha,
        options.nuclear_cutoff_bohr, batch, options.cutoff_bohr, _budget() if budget is None else budget)


def _reciprocal_quadrature(basis, system, g, cells, alpha, cutoff):
    shells = basis.shells()
    sizes = [2*s.l+1 if s.pure else (s.l+1)*(s.l+2)//2 for s in shells]
    offsets = np.cumsum([0]+sizes)
    out = np.zeros((len(cells), basis.nbasis, basis.nbasis), complex)
    rho = sum(atom.Z*np.exp(-1j*g@np.asarray(atom.xyz)) for atom in system.unit_cell)
    g2 = np.sum(g*g, axis=1)
    weight = -4*np.pi/abs(np.linalg.det(system.lattice))/g2*np.exp(-g2/(4*alpha**2))*rho
    for ic, cell in enumerate(cells):
        for ia, a in enumerate(shells):
            for ib, b in enumerate(basis.shells()):
                b.origin = (np.asarray(b.origin)+cell).tolist()
                if cutoff is not None and np.linalg.norm(np.asarray(a.origin)-b.origin)>cutoff:
                    continue
                tile = np.einsum('g,gij->ij', weight, _quadrature(a, b, g).conj())
                out[ic, offsets[ia]:offsets[ia+1], offsets[ib]:offsets[ib+1]] = tile
    return out


@pytest.mark.parametrize('pure', [False, True])
@pytest.mark.parametrize('alpha', [.35, .7, 1.3])
def test_short_reciprocal_constant_and_total_against_independent_references(pure, alpha):
    fixture = _fixture(pure); basis, system, opts, cells, translations, g = fixture
    result = _compute(fixture, alpha=alpha)
    short = vq.compute_nuclear_erfc_lattice(basis, system, alpha, opts)
    overlap = vq.compute_overlap_lattice(basis, system, opts)
    np.testing.assert_array_equal([c.index for c in short.cells], [c.index for c in cells])
    constant = np.pi*sum(a.Z for a in system.unit_cell)/(alpha**2*abs(np.linalg.det(system.lattice)))*np.asarray(overlap.blocks)
    reciprocal = _reciprocal_quadrature(basis, system, g, translations, alpha, opts.cutoff_bohr)
    np.testing.assert_allclose(result.short_range_cells, short.blocks, atol=5e-12, rtol=5e-12)
    np.testing.assert_allclose(result.constant_cells, constant, atol=3e-12, rtol=3e-12)
    np.testing.assert_allclose(result.reciprocal_cells, reciprocal, atol=3e-12, rtol=3e-12)
    expected = np.asarray(short.blocks)+reciprocal+constant
    np.testing.assert_allclose(result.cell_values, expected, atol=7e-12, rtol=7e-12)
    np.testing.assert_allclose(result.values, np.einsum('kr,rij->kij', np.exp(1j*K@translations.T), expected), atol=8e-12, rtol=8e-12)
    np.testing.assert_allclose(result.values, result.values.conj().transpose(0, 2, 1), atol=3e-12)
    r, p = result.receipt, result.plan
    assert r.nuclear_support == 'pair-midpoint-sphere-v1'
    assert r.boundary == '3d-zero-average-g0-omitted'
    assert r.nuclear_shell_calls+r.overlap_shell_calls+r.fourier_shell_calls <= p.shell_calls
    assert r.primitive_pairs <= p.primitive_pairs
    assert r.primitive_charge_products <= p.primitive_charge_products_upper_bound
    assert r.nuclear_images_retained <= p.nuclear_image_candidates_upper_bound
    assert r.vector_primitive_products <= p.vector_primitive_products
    assert r.contraction_terms == p.contraction_terms
    assert r.cell_imaginary_residual < 1e-12 and r.response_hermiticity_residual < 3e-12
    assert r.modeled_peak_bytes == p.modeled_peak_bytes


def test_batching_mesh_permutation_time_reversal_and_owned_components():
    fixture = _fixture(); result = _compute(fixture, batch=1)
    wider = _compute(fixture, batch=100)
    np.testing.assert_allclose(wider.values, result.values, atol=2e-13)
    assert wider.receipt.source_identity == result.receipt.source_identity
    assert wider.receipt.fourier_shell_calls < result.receipt.fourier_shell_calls
    assert wider.receipt.vector_primitive_products == result.receipt.vector_primitive_products
    reverse = _compute(fixture, k=-K)
    np.testing.assert_allclose(reverse.values, result.values.conj(), atol=3e-13)
    permutation = [2, 0, 1]
    np.testing.assert_allclose(_compute(fixture, k=K[permutation]).values, result.values[permutation], atol=3e-13)
    for name in ('values', 'cell_values', 'short_range_cells', 'constant_cells', 'reciprocal_cells'):
        exported = getattr(result, name); before = exported.copy(); exported[:] = 0
        np.testing.assert_array_equal(getattr(result, name), before)
    assert _compute(fixture, alpha=.8).receipt.source_identity != result.receipt.source_identity


def test_global_translation_and_individual_nuclear_image_relabeling():
    fixture = _fixture(); result = _compute(fixture)
    basis, system, opts, cells, translations, g = fixture
    shift = np.array([.31, -.27, .17])
    shifted_basis = _basis((0, 2), shift=shift)
    shifted_system = vq.PeriodicSystem(3, LATTICE, [vq.Atom(a.Z, (np.asarray(a.xyz)+shift).tolist()) for a in system.unit_cell])
    shifted = _compute((shifted_basis, shifted_system, opts, cells, translations, g))
    np.testing.assert_allclose(shifted.values, result.values, atol=4e-12, rtol=4e-12)
    atoms = list(system.unit_cell)
    atoms[0] = vq.Atom(atoms[0].Z, (np.asarray(atoms[0].xyz)+LATTICE[:, 1]).tolist())
    relabeled = _compute((basis, vq.PeriodicSystem(3, LATTICE, atoms), opts, cells, translations, g))
    np.testing.assert_allclose(relabeled.values, result.values, atol=4e-12, rtol=4e-12)
    assert relabeled.receipt.source_identity != result.receipt.source_identity


def test_basis_image_relabeling_covariance_with_regenerated_physical_support():
    fixture = _fixture(); basis, system, opts, cells, translations, g = fixture
    result = _compute(fixture)
    shells = basis.shells(); shells[1].origin = (np.asarray(shells[1].origin)+LATTICE[:, 0]).tolist()
    moved = vq.BasisSet(vq.Molecule([vq.Atom(2, s.origin) for s in shells], 0, 1), shells, 'manta-nuclear-relabel', True)
    moved_cells = vq.pair_complete_lattice_cells(moved, system, opts.cutoff_bohr)
    changed = _compute((moved, system, opts, moved_cells, np.array([c.r_cart for c in moved_cells]), g))
    shifts = np.array([[0., 0., 0.]]+[LATTICE[:, 0].tolist()]*5)
    u = np.exp(-1j*K@shifts.T)
    np.testing.assert_allclose(changed.values, u.conj()[:, :, None]*result.values*u[:, None, :], atol=7e-12, rtol=7e-12)


def test_converged_ewald_split_against_independent_scalar_plane_wave_series():
    basis = _basis((0,), contracted=False); a = basis.shells()[0].exponents[0]
    system = vq.PeriodicSystem(3, np.eye(3)*4.2, [vq.Atom(2, basis.shells()[0].origin)])
    g = rsgdf_dense_g_mesh(system, 100.); g = np.ascontiguousarray(g[np.any(g != 0., axis=1)])
    g2 = np.sum(g*g, axis=1)
    # Normalized same-center s product: rho_pair(G)=exp(-G^2/(8a)).
    # The omitted tail is below FP64 accuracy at this reciprocal cutoff.
    reference = -8*np.pi/abs(np.linalg.det(system.lattice))*np.sum(np.exp(-g2/(8*a))/g2)
    errors = []
    for cutoff in (4., 8., 18.):
        result = manta.compute_periodic_nuclear(basis, system, g, np.zeros((1, 3)), K, .4, cutoff, budget=_budget())
        errors.append(abs(result.values[0, 0, 0]-reference))
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 2e-12
    for alpha in (.7, 1.3):
        result = manta.compute_periodic_nuclear(basis, system, g, np.zeros((1, 3)), K, alpha, 18., budget=_budget())
        np.testing.assert_allclose(result.values[:, 0, 0], reference, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize('field, plan_field', [('memory_bytes', 'modeled_peak_bytes'), ('shell_calls', 'shell_calls'),
    ('primitive_quartets', 'primitive_pairs'), ('scalar_terms', 'scalar_terms'), ('kernel_terms', 'kernel_terms')])
def test_aggregate_admission_before_vector_copy_and_payload_scan(field, plan_field):
    fixture = _fixture(); basis, system, opts, cells, translations, g = fixture
    budget = _budget(); budget.other_live_bytes = 31937
    plan = manta.plan_periodic_nuclear(basis, system, len(g), len(cells), len(K), .7, opts.nuclear_cutoff_bohr, 4, opts.cutoff_bohr, budget)
    setattr(budget, field, getattr(plan, plan_field))
    result = _compute(fixture, budget=budget)
    assert result.receipt.modeled_peak_bytes == plan.modeled_peak_bytes
    setattr(budget, field, getattr(plan, plan_field)-1)
    with pytest.raises(ValueError, match='budget'):
        _compute((basis, system, opts, cells, translations, np.full_like(g, np.nan)), budget=budget)


@pytest.mark.parametrize('target', ['G', 'R'])
@pytest.mark.parametrize('error', ['zero', 'duplicate', 'open', 'fractional'])
def test_invalid_finite_supports_refused(target, error):
    fixture = list(_fixture()); i = 5 if target == 'G' else 4; a = fixture[i]
    if error == 'zero':
        if target == 'R': return  # The home AO cell is valid.
        a = np.vstack([a, np.zeros((1, 3))])
    elif error == 'duplicate': a = np.vstack([a, a[:1]])
    elif error == 'open': a = a[1:] if target == 'G' else a[2:]
    else: a = a.copy(); a[0] += .15
    fixture[i] = np.ascontiguousarray(a)
    with pytest.raises(ValueError, match='support|G=0'):
        _compute(fixture)


@pytest.mark.parametrize('alpha, radius', [(0., 4.), (-.1, 4.), (np.nan, 4.), (np.inf, 4.), (.7, -1.), (.7, np.nan), (.7, np.inf)])
def test_invalid_split_parameters(alpha, radius):
    basis, system, opts, cells, translations, g = _fixture()
    with pytest.raises(ValueError):
        manta.compute_periodic_nuclear(basis, system, g, translations, K, alpha, radius, budget=_budget())


def test_empty_axes_strict_arrays_and_three_dimensional_boundary():
    fixture = _fixture(); basis, system, opts, cells, translations, g = fixture
    base = _compute(fixture)
    cell_only = _compute(fixture, k=np.empty((0, 3)))
    np.testing.assert_array_equal(cell_only.cell_values, base.cell_values)
    empty = manta.compute_periodic_nuclear(basis, system, g, np.empty((0, 3)), K, .7, 4.5, budget=_budget())
    assert not np.any(empty.values) and empty.cell_values.shape == (0, basis.nbasis, basis.nbasis)
    no_g = _compute((basis, system, opts, cells, translations, np.empty((0, 3))))
    np.testing.assert_array_equal(no_g.cell_values, no_g.short_range_cells+no_g.constant_cells)
    assert no_g.receipt.fourier_shell_calls == 0
    for axis in [0, 1, 2]:
        for bad in (np.ones((2, 3), np.float32), np.ones((4, 3))[::2], np.ones((2, 2)), np.full((2, 3), np.nan)):
            arrays = [g, translations, K]; arrays[axis] = bad
            with pytest.raises(ValueError):
                manta.compute_periodic_nuclear(basis, system, *arrays, .7, 4.5, budget=_budget())
    other = vq.PeriodicSystem(2, LATTICE, list(system.unit_cell))
    with pytest.raises(ValueError, match='3D'):
        manta.compute_periodic_nuclear(basis, other, g, translations, K, .7, 4.5, budget=_budget())
    fortran = manta.compute_periodic_nuclear(basis, system, *[np.asfortranarray(a) for a in (g, translations, K)], .7, 4.5, 4, opts.cutoff_bohr, _budget())
    np.testing.assert_array_equal(fortran.values, base.values)
    assert fortran.receipt.source_identity == base.receipt.source_identity


@pytest.mark.parametrize('alpha', [.35, 1.3])
def test_existing_ewald_lattice_composition_at_the_same_finite_source(alpha, monkeypatch):
    from vibeqc.periodic_v_ne import compute_v_ne_ewald_3d_ft_lattice
    from vibeqc import output
    basis, system, opts, cells, translations, g = _fixture()
    records = []; monkeypatch.setattr(output, 'record', lambda *args, **kwargs: records.append((args, kwargs)))
    engine = vq.IntegralEngineOptions(backend='manta', budget=_budget())
    actual = manta.compute_nuclear_lattice(basis, system, cells, g, alpha=alpha,
        nuclear_cutoff_bohr=opts.nuclear_cutoff_bohr, pair_cutoff_bohr=opts.cutoff_bohr,
        reciprocal_batch=4, integral_engine=engine)
    ew = vq.EwaldOptions(); ew.alpha = alpha
    expected = compute_v_ne_ewald_3d_ft_lattice(basis, system, opts, ewald_options=ew, ke_cutoff=3., screen_rel=0., domain_rel=0.)
    np.testing.assert_array_equal([c.index for c in actual.cells], [c.index for c in expected.cells])
    np.testing.assert_allclose(actual.blocks, expected.blocks, atol=6e-12, rtol=6e-12)
    record = records[0][1]['integral_provenance']
    native = _compute(_fixture(), alpha=alpha, k=np.empty((0, 3)))
    assert record['source_identity'] == native.receipt.source_identity
    assert record['citation_methods'] == ['manta_periodic_nuclear']
    assert engine.budget.other_live_bytes == 0


def test_adapter_preflight_and_success_only_provenance(monkeypatch):
    from vibeqc import output
    monkeypatch.setattr(output, 'record', lambda *args, **kwargs: pytest.fail('refusal emitted a success receipt'))
    basis, system, opts, cells, translations, g = _fixture()
    engine = vq.IntegralEngineOptions(backend='manta', budget=_budget())
    kwargs = dict(alpha=.7, nuclear_cutoff_bohr=4.5, integral_engine=engine)
    engine.budget.shell_calls = 0
    with pytest.raises(ValueError, match='budget'):
        manta.compute_nuclear_lattice(basis, system, [object()], np.full_like(g, np.nan), **kwargs)
    engine.budget.shell_calls = 10_000
    with pytest.raises(TypeError, match='LatticeCell'):
        manta.compute_nuclear_lattice(basis, system, [object()], g, **kwargs)
    for bad in (0, True, 1.5):
        with pytest.raises(ValueError, match='batch'):
            manta.compute_nuclear_lattice(basis, system, cells, g, reciprocal_batch=bad, **kwargs)
    for bad in (object(), vq.IntegralEngineOptions(backend='libint')):
        with pytest.raises((TypeError, ValueError)):
            manta.compute_nuclear_lattice(basis, system, cells, g, **dict(kwargs, integral_engine=bad))


@pytest.mark.parametrize('pure', [False, True])
def test_l6_screened_component_against_optional_libcint(pure):
    gto = pytest.importorskip('pyscf.gto')
    basis = _basis((6,), pure, contracted=False); s = basis.shells()[0]
    system = vq.PeriodicSystem(3, np.eye(3)*20., [vq.Atom(2, s.origin)])
    mol = gto.M(atom=[('He', s.origin)], basis={'He': [[6, [s.exponents[0], 1.]]]}, unit='Bohr', cart=not pure, verbose=0)
    with mol.with_rinv_origin(s.origin):
        full = mol.intor('int1e_rinv')
        with mol.with_rinv_zeta(.7**2):
            long = mol.intor('int1e_rinv')
    expected = -2*(full-long)
    if not pure: expected *= 13/(4*np.pi)
    result = manta.compute_periodic_nuclear(basis, system, np.empty((0, 3)), np.zeros((1, 3)), K, .7, 1., budget=_budget())
    np.testing.assert_allclose(result.short_range_cells[0], expected, atol=2e-12, rtol=3e-11)


def test_executed_nuclear_image_and_primitive_charge_counts():
    from itertools import product
    fixture = _fixture(); basis, system, opts, cells, translations, g = fixture
    result = _compute(fixture)
    # A brute-force fixed box contains every physical image for this fixture;
    # it does not reuse the selector's fractional bounds or slack calculation.
    images = np.array(list(product(range(-4, 5), repeat=3)))@LATTICE.T
    points = np.concatenate([np.asarray(a.xyz)+images for a in system.unit_cell])
    count = 0
    for cell in translations:
        for a in basis.shells():
            for b in basis.shells():
                bra, ket = np.asarray(a.origin), np.asarray(b.origin)+cell
                if np.linalg.norm(bra-ket)>opts.cutoff_bohr: continue
                count += np.count_nonzero(np.linalg.norm(points-(bra+ket)/2, axis=1)<=opts.nuclear_cutoff_bohr)
    assert result.receipt.nuclear_images_retained == count
    assert result.receipt.primitive_charge_products == 4*count


def test_parameter_overflow_and_zero_radius_have_explicit_meanings():
    basis = _basis((0,), contracted=False)
    system = vq.PeriodicSystem(3, np.eye(3)*10., [vq.Atom(2, basis.shells()[0].origin)])
    zero = np.zeros((1, 3)); empty = np.empty((0, 3))
    result = manta.compute_periodic_nuclear(basis, system, empty, zero, K, .7, 0., pair_cutoff_bohr=0.)
    assert result.receipt.nuclear_images_retained == 1
    with pytest.raises(ValueError, match='finite'):
        manta.compute_periodic_nuclear(basis, system, empty, zero, K, 1e-300, 0.)
    with pytest.raises(ValueError, match='extent'):
        manta.plan_periodic_nuclear(basis, system, 1, 1, 1, .7, 1e30)
    with pytest.raises(ValueError, match='batch'):
        manta.plan_periodic_nuclear(basis, system, 1, 1, 1, .7, 1., 0)
