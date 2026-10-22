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


# Infinite-domain certificates use the same physical midpoint spheres as the
# finite RI Hamiltonian. References below expand Gaussian polynomials directly.
def nuclear_bound_fixture(*,compact=False,angular=0,pure=True,skew=True):
    from tests.test_manta_periodic_one_electron_images import fixture
    from tests.test_manta_periodic_ewald import cube
    c=fixture(angular=angular,pure=pure,skew=skew,cell_size=6. if compact else 3.8,
              exponent=4. if compact else .7,images=cube(0),k=K,weights=np.ones(len(K))/len(K))
    b=c['basis'];positions=[s.origin for s in b.shells()]
    system=vq.PeriodicSystem(3,c['cell'],[vq.Atom(2,x) for x in positions])
    g=np.ascontiguousarray(cube(8 if compact else 1,reciprocal=True)@(2*np.pi*np.linalg.inv(c['cell'])))
    cells=np.ascontiguousarray(c['images']@c['cell'].T)
    o=manta.PeriodicNuclearBoundOptions();o.ao_completion_cube_radius=2;o.real_completion_cube_radius=5
    o.reciprocal_completion_cube_radius=10 if compact else 6
    return dict(basis=b,system=system,g=g,cells=cells,k=K,alpha=.4 if compact else .65,
                cutoff=18. if compact else 2.4,paircut=None,options=o)


def nuclear_bound(c,budget=None):
    return manta.bound_periodic_nuclear(c['basis'],c['system'],c['g'],c['cells'],c['k'],c['alpha'],c['cutoff'],
        c['paircut'],c['options'],_budget() if budget is None else budget)


def nuclear_sphere_reference(basis,system,g,cells,k,alpha,cutoff,paircut=None,order=100):
    """Ordered polynomial/Gaussian erfc and analytic Fourier reference.

    Enumerates each midpoint sphere independently; no native source, integral,
    envelope coefficient or reconstructed canonical arm enters this reference.
    """
    from itertools import product
    from types import SimpleNamespace
    from tests.test_manta_lattice_gaussian import components,fourier
    from tests.test_manta_periodic_electrostatic import point_integral
    lattice=np.asarray(system.lattice);inverse=np.linalg.inv(lattice)
    shells=basis.shells();sizes=[2*s.l+1 if s.pure else (s.l+1)*(s.l+2)//2 for s in shells];offsets=np.cumsum([0,*sizes])
    positions=np.array([a.xyz for a in system.unit_cell]);charges=np.array([a.Z for a in system.unit_cell])
    g2=np.sum(g*g,axis=1);rho=np.exp(-1j*g@positions.T)@charges
    lam=4*np.pi/abs(np.linalg.det(lattice))*np.exp(-g2/(4*alpha**2))/g2
    out=np.zeros((3,len(k),basis.nbasis,basis.nbasis),complex)
    for shift in cells:
        for ia,a in enumerate(shells):
            for ib,b in enumerate(basis.shells()):
                b.origin=(np.asarray(b.origin)+shift).tolist()
                if paircut is not None and np.linalg.norm(np.array(a.origin)-b.origin)>paircut:continue
                midpoint=(np.array(a.origin)+b.origin)/2;selected=[]
                for x,z in zip(positions,charges):
                    fractional=inverse@(midpoint-x);reach=cutoff*np.linalg.norm(inverse,axis=1)
                    labels=np.array(list(product(*(range(int(np.floor(f-r))-1,int(np.ceil(f+r))+2) for f,r in zip(fractional,reach)))))
                    images=x+labels@lattice.T
                    selected.append((z,images[np.linalg.norm(images-midpoint,axis=1)<=cutoff+1e-12*(1+cutoff)]))
                pair=SimpleNamespace(shells=lambda a=a,b=b:[a,b])
                terms=components(dict(kind='pair',basis=pair,extent=SimpleNamespace(shells=(0,1))))
                ta=manta.cartesian_to_spherical(a.l) if a.pure else np.eye((a.l+1)*(a.l+2)//2)
                tb=manta.cartesian_to_spherical(b.l) if b.pure else np.eye((b.l+1)*(b.l+2)//2);transform=np.kron(ta,tb)
                real=transform@np.array([-sum(z*point_integral(t,images,alpha,order) for z,images in selected) for t in terms])
                panel=transform@np.array([fourier(t,g) for t in terms])
                reciprocal=-(panel@(lam*rho)).real
                overlap=transform@np.array([fourier(t,np.zeros((1,3)))[0].real for t in terms])
                gauge=np.pi*sum(charges)/(alpha**2*abs(np.linalg.det(lattice)))*overlap
                for index,value in enumerate((real,reciprocal,gauge)):
                    out[index,:,offsets[ia]:offsets[ia+1],offsets[ib]:offsets[ib+1]]+=np.exp(1j*k@shift)[:,None,None]*value.reshape(sizes[ia],sizes[ib])
    return out


def nuclear_point_reference(system,g,alpha,cutoff):
    from itertools import product
    from scipy.special import erfc
    lattice=np.asarray(system.lattice);inverse=np.linalg.inv(lattice);positions=np.array([a.xyz for a in system.unit_cell]);charges=np.array([a.Z for a in system.unit_cell])
    real=0.
    for a,origin in enumerate(positions):
        for c,x in enumerate(positions):
            f=inverse@(origin-x);reach=cutoff*np.linalg.norm(inverse,axis=1)
            labels=np.array(list(product(*(range(int(np.floor(v-r))-1,int(np.ceil(v+r))+2) for v,r in zip(f,reach)))))
            radius=np.linalg.norm(origin-x-labels@lattice.T,axis=1)
            mask=radius<=cutoff+1e-12*(1+cutoff)
            if a==c:mask&=np.any(labels!=0,axis=1)
            real+=.5*charges[a]*charges[c]*sum(erfc(alpha*radius[mask])/radius[mask])
    g2=np.sum(g*g,axis=1);rho=np.exp(1j*g@positions.T)@charges;volume=abs(np.linalg.det(lattice))
    reciprocal=2*np.pi/volume*np.sum(np.exp(-g2/(4*alpha**2))/g2*abs(rho)**2)
    self_=-alpha/np.sqrt(np.pi)*np.dot(charges,charges);background=-np.pi/(2*alpha**2*volume)*sum(charges)**2
    return np.array([real,reciprocal,self_,background])


@pytest.mark.parametrize('angular,pure,skew',[(0,True,False),(1,False,True),(2,True,True)])
def test_nuclear_completion_bounds_against_independent_sphere_and_fourier_reference(angular,pure,skew):
    from tests.test_manta_periodic_ewald import cube
    c=nuclear_bound_fixture(angular=angular,pure=pure,skew=skew);bound=nuclear_bound(c)
    b,s,g,cells,k,alpha,cutoff=[c[n] for n in ('basis','system','g','cells','k','alpha','cutoff')]
    finite=nuclear_sphere_reference(b,s,g,cells,k,alpha,cutoff)
    native=manta.compute_periodic_nuclear(b,s,g,cells,k,alpha,cutoff,budget=_budget())
    np.testing.assert_allclose(native.values,finite.sum(axis=0),atol=3e-11,rtol=3e-12)
    assert bound.receipt.nuclear_source_identity==native.receipt.source_identity
    # Isolate real and reciprocal omissions on the retained AO support.
    full_g=np.ascontiguousarray(cube(7,reciprocal=True)@(2*np.pi*np.linalg.inv(s.lattice)))
    enlarged=nuclear_sphere_reference(b,s,full_g,cells,k,alpha,16.)
    assert np.all(abs(enlarged[0]-finite[0])<=bound.real_error_bound+3e-11)
    assert np.all(abs(enlarged[1]-finite[1])<=bound.reciprocal_error_bound+3e-11)
    nn=nuclear_point_reference(s,g,alpha,cutoff);nn_full=nuclear_point_reference(s,full_g,alpha,16.)
    assert abs(nn_full[0]-nn[0])<=bound.receipt.nuclear_real_error_bound
    assert abs(nn_full[1]-nn[1])<=bound.receipt.nuclear_reciprocal_error_bound+3e-13
    assert abs(sum(nn_full)-sum(nn))<=bound.receipt.nuclear_energy_error_bound+3e-13
    np.testing.assert_allclose(bound.potential_error_bound,bound.real_error_bound+bound.reciprocal_error_bound+bound.pair_image_error_bound,rtol=8e-12,atol=0.)
    assert bound.receipt.real_terms<=bound.plan.real_terms
    assert bound.receipt.ordered_pairs<=bound.plan.maximum_ordered_pairs
    assert not bound.receipt.roundoff_included


def test_nuclear_image_completion_and_reciprocal_holes():
    from tests.test_manta_periodic_ewald import cube
    c=nuclear_bound_fixture();c['paircut']=.3;c['cells']=np.ascontiguousarray(cube(1)@c['system'].lattice.T)
    # A noncubic reciprocal inventory with interior holes and retained vectors
    # beyond the completion cube remains conservative (positive overcount).
    labels=cube(2,reciprocal=True);labels=labels[np.sum(labels*labels,axis=1)>2]
    c['g']=np.ascontiguousarray(labels@(2*np.pi*np.linalg.inv(c['system'].lattice)))
    c['options'].reciprocal_completion_cube_radius=1
    bound=nuclear_bound(c);b,s=[c[n] for n in ('basis','system')]
    finite=nuclear_sphere_reference(b,s,c['g'],c['cells'],K,c['alpha'],c['cutoff'],c['paircut']).sum(axis=0)
    full_cells=np.ascontiguousarray(cube(2)@s.lattice.T)
    full_g=np.ascontiguousarray(cube(7,reciprocal=True)@(2*np.pi*np.linalg.inv(s.lattice)))
    full=nuclear_sphere_reference(b,s,full_g,full_cells,K,c['alpha'],16.).sum(axis=0)
    assert np.all(abs(full-finite)<=bound.potential_error_bound+3e-11)
    assert np.max(abs(full-finite))>1e-3
    missing=nuclear_sphere_reference(b,s,c['g'],full_cells,K,c['alpha'],16.).sum(axis=0)
    keep=nuclear_sphere_reference(b,s,c['g'],c['cells'],K,c['alpha'],16.,c['paircut']).sum(axis=0)
    assert np.all(abs(missing-keep)<=bound.pair_image_error_bound+3e-11)


def test_nuclear_tight_compact_source_and_ewald_split():
    from tests.test_manta_periodic_ewald import cube
    c=nuclear_bound_fixture(compact=True,skew=False)
    # A single-center helium cell with two Gaussian basis centers.
    c['system']=vq.PeriodicSystem(3,c['system'].lattice,[vq.Atom(2,c['basis'].shells()[0].origin)])
    bound=nuclear_bound(c)
    assert 0<bound.receipt.maximum_potential_error<1e-8
    assert 0<bound.receipt.nuclear_energy_error_bound<1e-8
    assert bound.receipt.reciprocal_weight_bound>0
    b,s,g,cells,k,alpha,cutoff=[c[n] for n in ('basis','system','g','cells','k','alpha','cutoff')]
    finite=manta.compute_periodic_nuclear(b,s,g,cells,k,alpha,cutoff,budget=_budget())
    full_g=np.ascontiguousarray(cube(24,reciprocal=True)@(2*np.pi*np.linalg.inv(s.lattice)))
    full=nuclear_sphere_reference(b,s,full_g,cells,k,.6,24.).sum(axis=0)
    assert np.all(abs(full-finite.values)<=bound.potential_error_bound+3e-12)
    nn=nuclear_point_reference(s,g,alpha,cutoff);nn_full=nuclear_point_reference(s,full_g,.6,24.)
    assert abs(sum(nn_full)-sum(nn))<=bound.receipt.nuclear_energy_error_bound+3e-13


def test_nuclear_bound_translation_relabeling_owned_arrays_and_identity():
    c=nuclear_bound_fixture();a=nuclear_bound(c)
    atoms=list(c['system'].unit_cell);atoms[0]=vq.Atom(atoms[0].Z,(np.array(atoms[0].xyz)+3*c['system'].lattice[:,0]).tolist())
    moved=dict(c,system=vq.PeriodicSystem(3,c['system'].lattice,atoms));b=nuclear_bound(moved)
    np.testing.assert_allclose(b.potential_error_bound,a.potential_error_bound,rtol=1e-10,atol=1e-14)
    assert b.receipt.source_identity!=a.receipt.source_identity
    for name in ('potential_error_bound','real_error_bound','reciprocal_error_bound','pair_image_error_bound'):
        exported=getattr(a,name);before=exported.copy();assert not exported.flags.writeable
        exported.setflags(write=True);exported[:]=0;np.testing.assert_array_equal(getattr(a,name),before)
    options=a.options;options.real_completion_cube_radius=1;assert a.options.real_completion_cube_radius==5
    permuted=nuclear_bound(dict(c,k=np.ascontiguousarray(K[[2,0,1]]),g=np.ascontiguousarray(c['g'][::-1])))
    np.testing.assert_array_equal(permuted.potential_error_bound,a.potential_error_bound)
    assert permuted.receipt.nuclear_source_identity!=a.receipt.nuclear_source_identity


@pytest.mark.parametrize('field,plan_field',[('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),
    ('primitive_quartets','primitive_pairs'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_nuclear_bound_all_caps_before_payload_scan(field,plan_field):
    c=nuclear_bound_fixture();budget=_budget();budget.other_live_bytes=31937
    p=manta.plan_periodic_nuclear_bounds(c['basis'],c['system'],len(c['g']),len(c['cells']),len(K),c['alpha'],c['cutoff'],c['paircut'],c['options'],budget)
    setattr(budget,field,getattr(p,plan_field));result=nuclear_bound(c,budget)
    assert result.plan.modeled_peak_bytes==p.modeled_peak_bytes
    setattr(budget,field,getattr(p,plan_field)-1)
    with pytest.raises(ValueError,match='budget'):nuclear_bound(dict(c,g=np.full_like(c['g'],np.nan)),budget)


@pytest.mark.parametrize('error',['fractional','duplicate','open','zero','roundtrip','nan'])
def test_nuclear_bound_invalid_reciprocal_sources(error):
    c=nuclear_bound_fixture();g=c['g'].copy()
    if error=='fractional':g[0]+=.15
    elif error=='duplicate':g=np.vstack([g,g[:1]])
    elif error=='open':g=g[1:]
    elif error=='zero':g=np.vstack([g,np.zeros((1,3))])
    elif error=='roundtrip':g[0]*=1+1e-11;g[-1]*=1+1e-11
    else:g[:]=np.nan
    with pytest.raises(ValueError,match='support|G=0|finite|roundoff'):nuclear_bound(dict(c,g=np.ascontiguousarray(g)))


def test_nuclear_bound_empty_charges_insufficient_cube_and_coincident_points():
    c=nuclear_bound_fixture();empty=nuclear_bound(dict(c,system=vq.PeriodicSystem(3,c['system'].lattice,[])))
    assert not np.any(empty.potential_error_bound) and empty.receipt.nuclear_energy_error_bound==0
    c['options'].real_completion_cube_radius=0
    far=np.array(c['basis'].shells()[0].origin)+c['system'].lattice@np.array([.49,.49,.49])
    with pytest.raises(ValueError,match='not separated'):
        nuclear_bound(dict(c,system=vq.PeriodicSystem(3,c['system'].lattice,[vq.Atom(1,far.tolist())])))
    c['options'].real_completion_cube_radius=5
    x=c['system'].unit_cell[0].xyz
    with pytest.raises(ValueError,match='coincident'):nuclear_bound(dict(c,system=vq.PeriodicSystem(3,c['system'].lattice,[vq.Atom(1,x),vq.Atom(1,x)])))


def test_nuclear_bound_citation_route():
    from vibeqc.output.citations import load_default_database
    citations=load_default_database().assemble(method='manta_periodic_nuclear_bounds').citations
    assert {'ewald_lattice_sum_1921','mcmurchie_davidson_1978','sun_berkelbach_gdf_2017'}<={c.key for c in citations}



def nuclear_accuracy_targets(value=1e6,compact=False):
    o=manta.PeriodicNuclearAccuracyOptions()
    o.matrix_error=o.attraction_energy_error=o.nuclear_energy_error=value
    o.real_completion_cube_radius=5;o.reciprocal_completion_cube_radius=10 if compact else 6
    return o


@pytest.mark.parametrize('method,species,limit,automatic',[
    ('rhf','Be',100,False),('rhf','Be',100,True),('uhf','H',100,False),('uhf','Li',3,True),
    ('rks','Be',100,True),('uks','Li',3,False),('pbe0','Be',100,True),
    ('hse06','Li',100,True),('hse06-ri-k','Li',3,False)])
def test_complete_ri_nuclear_accuracy_shared_density_and_unchanged_state(tmp_path,method,species,limit,automatic):
    import dataclasses
    from vibeqc import _vibeqc_core as core
    from vibeqc.guess_read import _spin_density_k_from_qvf_bloch_payload
    from tests.test_manta_periodic_one_electron_images import ri_image_job
    from tests.test_bounded_periodic_rhf_solver import _density_envelope_caps
    job,a,k,scf=ri_image_job(tmp_path,method,species,limit,automatic)
    job['periodic_integrals']=dataclasses.replace(job['periodic_integrals'],nuclear_accuracy=nuclear_accuracy_targets())
    result=vq.run_periodic_job(**job);n=result.native;c=n.receipt.nuclear_accuracy;r=c.nuclear.receipt
    assert c.qualified and result.converged==(limit==100)
    assert n.plan.density_envelope and n.plan.nuclear_accuracy
    assert (n.receipt.accuracy_density is not None)==automatic
    ne=a['system'].n_electrons();nb=(ne-a['system'].multiplicity+1)//2;na=ne-nb
    caps,_=_density_envelope_caps(n.overlap,na,nb,scf)
    envelope=core._bound_bounded_periodic_density_envelope(n.overlap,na,nb,scf,caps)
    h=envelope.weighted_entry_bounds;b=c.nuclear.potential_error_bound
    entry=np.einsum('kji,ij->',h,b);l1=envelope.density_l1_limit*r.maximum_potential_error
    assert c.entry_attraction_energy_error_bound>=entry*(1-1e-14)
    assert c.entry_attraction_energy_error_bound==pytest.approx(entry,rel=8e-12)
    assert c.l1_attraction_energy_error_bound>=l1*(1-1e-14)
    assert c.l1_attraction_energy_error_bound==pytest.approx(l1,rel=8e-12)
    assert c.attraction_energy_error_bound==min(c.entry_attraction_energy_error_bound,c.l1_attraction_energy_error_bound)
    assert c.total_energy_error_bound>=c.attraction_energy_error_bound+r.nuclear_energy_error_bound
    assert c.density_envelope_identity==n.receipt.one_electron_accuracy.density_envelope_identity
    assert r.nuclear_source_identity==n.receipt.nuclear_identity
    assert r.pair_source_identity==n.receipt.one_electron_accuracy.images.receipt.pair_source_identity
    assert 0<c.shell_calls<=n.plan.nuclear_accuracy.shell_calls and 0<c.primitive_pairs<=n.plan.nuclear_accuracy.primitive_pairs
    saved=_spin_density_k_from_qvf_bloch_payload(job['output'].with_suffix('.qvf'))
    assert np.all(np.sum(abs(np.asarray(saved)),axis=0)/len(k)<=h+3e-12)
    certificate=result.integral_provenance['nuclear_accuracy']
    assert certificate['certificate_identity']==c.certificate_identity
    assert 'density-relaxation' in certificate['excludes'] and not certificate['nuclear']['roundoff_included']
    assert 'manta_periodic_nuclear_bounds' in result.integral_provenance['citation_numerics']
    assert not result.integral_provenance['symmetry_reduced']
    job['periodic_integrals']=dataclasses.replace(job['periodic_integrals'],nuclear_accuracy=None)
    job['output']=tmp_path/'disabled'/'job';control=vq.run_periodic_job(**job)
    for name in ('density','fock','overlap','hcore'):np.testing.assert_array_equal(getattr(n,name),getattr(control.native,name))
    for name in ('source_identity','overlap_identity','kinetic_identity','nuclear_identity','coulomb_identity','exchange_identity'):
        assert getattr(n.receipt,name)==getattr(control.native.receipt,name)
    assert result.energy==control.energy and control.native.receipt.nuclear_accuracy is None
    target=c.targets;target.matrix_error=0;assert c.targets.matrix_error==1e6
    export=c.nuclear.potential_error_bound;assert not export.flags.writeable
    export.setflags(write=True);export[:]=0;np.testing.assert_array_equal(c.nuclear.potential_error_bound,b)


@pytest.mark.parametrize('field,quantity',[('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),
    ('primitive_quartets','primitive_pairs'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_complete_nuclear_accuracy_all_resource_caps_before_payload(tmp_path,field,quantity):
    from vibeqc import _vibeqc_core as core
    from tests.test_manta_periodic_ri_preparation import prepared_job
    from tests.test_manta_periodic_ri_domains import explicit
    job,a,k=prepared_job(tmp_path,'rhf','Be',limit=1)
    a['options'].nuclear_accuracy=nuclear_accuracy_targets()
    domains=manta.build_periodic_ri_rhf_domains(**a);raw=explicit(a,domains)
    p=manta.plan_periodic_ri_rhf(raw['basis'],raw['auxiliary'],raw['system'],raw['mesh'],len(raw['reciprocal_vectors']),
        int(max(np.diff(raw['transfer_offsets']))),len(raw['translations']),raw['options'],raw['budget'])
    assert p.nuclear_accuracy and p.density_envelope and p.one_electron_accuracy is None and p.exchange_accuracy is None
    for key,value in [('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),('primitive_quartets','primitive_pairs'),
                      ('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')]:setattr(raw['budget'],key,getattr(p,value))
    result=core._manta.run_periodic_ri_rhf(**raw);assert result.receipt.nuclear_accuracy.qualified
    assert result.receipt.accuracy_density is None
    setattr(raw['budget'],field,getattr(p,quantity)-1);raw['translations']=np.full_like(raw['translations'],np.nan)
    with pytest.raises((ValueError,RuntimeError),match='budget'):core._manta.run_periodic_ri_rhf(**raw)


@pytest.mark.parametrize('field,reason',[('matrix_error','nuclear matrix'),('attraction_energy_error','nuclear attraction'),
    ('nuclear_energy_error','nuclear repulsion')])
def test_nuclear_accuracy_target_failure_and_invalid_controls(tmp_path,field,reason):
    import dataclasses
    from tests.test_manta_periodic_ri_preparation import prepared_job
    job,a,k=prepared_job(tmp_path,'rhf','Be',limit=1)
    target=nuclear_accuracy_targets();setattr(target,field,1e-300)
    job['periodic_integrals']=dataclasses.replace(job['periodic_integrals'],nuclear_accuracy=target)
    with pytest.raises(ValueError,match=reason):vq.run_periodic_job(**job)
    for value in (0.,-1.,float('nan'),float('inf')):
        setattr(target,field,value)
        with pytest.raises(ValueError,match='finite positive'):
            dataclasses.replace(job['periodic_integrals'],nuclear_accuracy=target)
        a['options'].nuclear_accuracy=target
        with pytest.raises(ValueError,match='nuclear accuracy'):
            manta.build_periodic_ri_rhf_domains(**a)
    with pytest.raises(TypeError,match='PeriodicNuclearAccuracyOptions'):
        dataclasses.replace(job['periodic_integrals'],nuclear_accuracy=object())


def test_nuclear_accuracy_exact_j_ri_k_and_nonri_guard(tmp_path):
    import dataclasses
    from tests.test_manta_periodic_ri_preparation import prepared_job
    job,a,k=prepared_job(tmp_path,'rhf','Be',limit=1)
    job['coulomb']=dataclasses.replace(job['coulomb'],algorithm='exact')
    job['exchange']=manta.ExchangeBuildOptions(algorithm='ri')
    job['periodic_fitting']=dataclasses.replace(job['periodic_fitting'],coupling_multipole_error=0.)
    job['periodic_integrals']=dataclasses.replace(job['periodic_integrals'],nuclear_accuracy=nuclear_accuracy_targets())
    assert vq.run_periodic_job(**job).native.receipt.nuclear_accuracy.qualified
    job['exchange']=manta.ExchangeBuildOptions(algorithm='exact');job['periodic_fitting']=None
    job['coulomb']=manta.CoulombBuildOptions(algorithm='exact')
    with pytest.raises(NotImplementedError,match='Nuclear accuracy currently requires'):vq.run_periodic_job(**job)


def test_complete_compact_ri_nuclear_meets_tight_targets(tmp_path):
    from benchmarks.manta_periodic_ri_nuclear_accuracy import accuracy_job
    job,c,scf=accuracy_job(tmp_path,'compact-tight')
    result=vq.run_periodic_job(**job);assert result.converged
    r=result.native.receipt.nuclear_accuracy
    assert 0<r.nuclear.receipt.maximum_potential_error<1e-8
    assert 0<r.nuclear.receipt.nuclear_energy_error_bound<1e-8
    assert 0<r.attraction_energy_error_bound<1e-8
    assert result.native.receipt.exchange_accuracy.qualified
