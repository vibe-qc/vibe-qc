"""Whole-mesh, multi-density finite reciprocal exchange and its energy derivative."""
import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta, _vibeqc_core as core
from tests.test_manta_fourier import _basis, _quadrature
from tests.test_manta_bloch_fourier import LATTICE, CELLS, MOMENTA


WEIGHTS = np.array([.2, .3, .5])
LABELS = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])


def _transfers(k=MOMENTA, lattice=LATTICE):
    reciprocal = 2*np.pi*np.linalg.inv(lattice).T
    base = LABELS@reciprocal.T
    chunks = []
    for target in k:
        for source in k:
            q = base+target-source
            chunks.append(q[np.any(q != 0., axis=1)])
    return np.ascontiguousarray(np.concatenate(chunks)), np.r_[0, np.cumsum([len(q) for q in chunks])].astype(np.int64)


def _density(basis, count=2):
    rng = np.random.default_rng(932)
    x = rng.normal(size=(count, len(MOMENTA), basis.nbasis, basis.nbasis))+1j*rng.normal(size=(count, len(MOMENTA), basis.nbasis, basis.nbasis))
    return np.ascontiguousarray(x+x.conj().transpose(0, 1, 3, 2))


def _compute(basis, density=None, *, k=MOMENTA, q=None, offsets=None, cells=CELLS, weights=WEIGHTS, cutoff=None, batch=4, budget=None):
    if density is None: density = _density(basis)
    if q is None: q, offsets = _transfers(k)
    return manta.compute_periodic_exchange(basis, LATTICE, q, offsets, cells, k, weights, density,
        batch, cutoff, manta.ResourceBudget() if budget is None else budget)


def _quadrature_bloch(basis, q, cells, momentum, cutoff):
    shells = basis.shells()
    sizes = [2*s.l+1 if s.pure else (s.l+1)*(s.l+2)//2 for s in shells]
    offsets = np.cumsum([0]+sizes)
    out = np.zeros((len(q), basis.nbasis, basis.nbasis), complex)
    for cell in cells:
        for ia, a in enumerate(shells):
            for ib, b in enumerate(basis.shells()):
                b.origin = (np.asarray(b.origin)+cell).tolist()
                if cutoff is not None and np.linalg.norm(np.asarray(a.origin)-b.origin)>cutoff: continue
                out[:, offsets[ia]:offsets[ia+1], offsets[ib]:offsets[ib+1]] += np.exp(1j*momentum@cell)*_quadrature(a, b, q)
    return out


@pytest.mark.parametrize('pure', [False, True])
@pytest.mark.parametrize('cutoff', [None, 1.7])
def test_all_density_channels_against_independent_fourier_quadrature(pure, cutoff):
    basis = _basis(pure=pure); density = _density(basis); q, offsets = _transfers()
    result = _compute(basis, density, cutoff=cutoff)
    expected = np.zeros_like(density)
    for target, k in enumerate(MOMENTA):
        for source in range(len(MOMENTA)):
            pair = target*len(MOMENTA)+source; vectors = q[offsets[pair]:offsets[pair+1]]
            b = _quadrature_bloch(basis, vectors, CELLS, k, cutoff)
            kernel = WEIGHTS[source]*4*np.pi/abs(np.linalg.det(LATTICE))/np.sum(vectors*vectors, axis=1)
            for state in range(len(density)):
                expected[state, target] += np.einsum('g,gai,ab,gbj->ij', kernel, b.conj(), density[state, source], b)
    np.testing.assert_allclose(result.values, expected, atol=2e-10, rtol=2e-11)
    energy = -.5*np.einsum('k,skij,skji->s', WEIGHTS, density, expected)
    np.testing.assert_allclose(result.receipt.energies, energy.real, atol=3e-10, rtol=2e-11)
    r, p = result.receipt, result.plan
    assert r.boundary == 'full-coulomb-q0-omitted-no-madelung'
    assert r.transfer_vectors == len(q)
    assert r.shell_calls <= p.shell_calls and r.primitive_pairs <= p.primitive_pairs
    assert r.primitive_pairs == 4*r.shell_calls
    assert r.matrix_products == 2*len(density)*len(q) <= p.matrix_products
    assert r.vector_primitive_products <= p.vector_primitive_products
    assert r.contraction_terms == r.matrix_products*basis.nbasis**3+density.size <= p.contraction_terms
    assert r.response_hermiticity_residual < 1e-11
    assert r.modeled_peak_bytes == p.modeled_peak_bytes


def test_existing_negative_source_phase_convention_converts_to_bdagger_d_b():
    from vibeqc.bipole_fock_ewald import _contract_K_long_range_channel_linear
    basis = _basis(); density = _density(basis, 1); q, offsets = _transfers()
    actual = _compute(basis, density).values[0]
    scale = np.concatenate([np.full(2*s.l+1, np.sqrt(4*np.pi/(2*s.l+1)) if s.l else 1.) for s in basis.shells()])
    expected = np.zeros_like(actual)
    for target in range(len(MOMENTA)):
        for source, k_source in enumerate(MOMENTA):
            pair = target*len(MOMENTA)+source; vectors = q[offsets[pair]:offsets[pair+1]]
            raw = core.ao_pair_fourier_transform_bloch_multi_cxx(basis, vectors, CELLS, -k_source[None, :], 0., None, -1.)[0]
            b = raw*scale[:, None, None]*scale[None, :, None]
            kernel = WEIGHTS[source]*4*np.pi/abs(np.linalg.det(LATTICE))/np.sum(vectors*vectors, axis=1)
            expected[target] += _contract_K_long_range_channel_linear(kernel, b, density[0, source])
    np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-11)


def test_exchange_energy_derivative_is_negative_k_and_perturbation_reaches_whole_mesh():
    basis = _basis((0, 1)); density = _density(basis)
    result = _compute(basis, density)
    direction = _density(basis, 1)[0, 0]
    h = 1e-5; plus = density.copy(); minus = density.copy()
    plus[0, 1] += h*direction; minus[0, 1] -= h*direction
    a, b = _compute(basis, plus), _compute(basis, minus)
    derivative = (a.receipt.energies[0]-b.receipt.energies[0])/(2*h)
    expected = -WEIGHTS[1]*np.trace(direction@result.values[0, 1]).real
    assert derivative == pytest.approx(expected, rel=2e-9, abs=2e-8)
    assert all(np.linalg.norm(x)>1e-8 for x in a.values[0]-b.values[0])
    np.testing.assert_array_equal(a.values[1], b.values[1])
    assert a.receipt.source_identity == b.receipt.source_identity
    assert a.receipt.density_identity != b.receipt.density_identity
    assert a.receipt.density_identities[1] == b.receipt.density_identities[1]
    # The weighted density-response pairing is self-adjoint across the mesh.
    left = np.einsum('k,kij,kji->', WEIGHTS, density[0], result.values[1])
    right = np.einsum('k,kij,kji->', WEIGHTS, density[1], result.values[0])
    assert abs(left-right) < 2e-11


def test_batch_and_density_sharing_source_identity_and_owned_exports():
    basis = _basis((0, 1)); density = _density(basis)
    result = _compute(basis, density, batch=1)
    wide = _compute(basis, density, batch=32)
    np.testing.assert_allclose(wide.values, result.values, atol=2e-12)
    assert wide.receipt.source_identity == result.receipt.source_identity
    assert wide.receipt.density_identity == result.receipt.density_identity
    assert wide.receipt.primitive_pairs < result.receipt.primitive_pairs
    assert wide.receipt.vector_primitive_products == result.receipt.vector_primitive_products
    for state in range(len(density)):
        separate = _compute(basis, density[state:state+1], batch=32)
        np.testing.assert_array_equal(separate.values[0], wide.values[state])
        assert separate.receipt.primitive_pairs == wide.receipt.primitive_pairs
        assert separate.receipt.density_identities[0] == wide.receipt.density_identities[state]
    exported = wide.values; saved = exported.copy(); exported[:] = 0.; density[:] = 0.
    np.testing.assert_array_equal(wide.values, saved)


def test_time_reversal_reciprocal_relabeling_origin_translation_and_mesh_permutation():
    basis = _basis((0, 1)); density = _density(basis); q, offsets = _transfers()
    result = _compute(basis, density)
    reverse = _compute(basis, density.conj().copy(), k=-MOMENTA, q=-q, offsets=offsets)
    np.testing.assert_allclose(reverse.values, result.values.conj(), atol=3e-12)
    shifted_k = MOMENTA.copy(); shifted_k[1] += 2*np.pi*np.linalg.inv(LATTICE).T[:, 0]
    relabeled = _compute(basis, density, k=shifted_k, q=q, offsets=offsets)
    np.testing.assert_allclose(relabeled.values, result.values, atol=5e-12)
    translated = _compute(_basis((0, 1), shift=(.4, -.2, .7)), density)
    np.testing.assert_allclose(translated.values, result.values, atol=5e-12)
    perm = [2, 0, 1]; reordered = []
    for target in perm:
        for source in perm:
            pair = target*len(MOMENTA)+source; reordered.append(q[offsets[pair]:offsets[pair+1]])
    moved_q = np.ascontiguousarray(np.concatenate(reordered))
    moved_offsets = np.r_[0, np.cumsum([len(a) for a in reordered])].astype(np.int64)
    moved = _compute(basis, density[:, perm].copy(), k=MOMENTA[perm], q=moved_q, offsets=moved_offsets, weights=WEIGHTS[perm])
    np.testing.assert_allclose(moved.values, result.values[:, perm], atol=3e-12)
    np.testing.assert_allclose(moved.receipt.energies, result.receipt.energies, atol=3e-12)


def test_basis_image_gauge_covariance_with_density_pushforward():
    basis = _basis((0, 1)); density = _density(basis); cutoff = 1.4
    system = vq.PeriodicSystem(3, LATTICE, [vq.Atom(2, s.origin) for s in basis.shells()])
    def cells(b): return np.array([c.r_cart for c in vq.pair_complete_lattice_cells(b, system, cutoff)])
    result = _compute(basis, density, cells=cells(basis), cutoff=cutoff)
    shells = basis.shells(); shells[1].origin = (np.asarray(shells[1].origin)+LATTICE[:, 0]).tolist()
    changed = vq.BasisSet(vq.Molecule([vq.Atom(2, s.origin) for s in shells], 0, 1), shells, 'manta-exchange-relabel', True)
    shifts = np.array([[0., 0., 0.]]+[LATTICE[:, 0].tolist()]*3)
    u = np.exp(-1j*MOMENTA@shifts.T)
    moved_density = np.ascontiguousarray(u.conj()[None, :, :, None]*density*u[None, :, None, :])
    moved = _compute(changed, moved_density, cells=cells(changed), cutoff=cutoff)
    expected = u.conj()[None, :, :, None]*result.values*u[None, :, None, :]
    np.testing.assert_allclose(moved.values, expected, atol=7e-12, rtol=3e-12)
    np.testing.assert_allclose(moved.receipt.energies, result.receipt.energies, atol=8e-12)


def test_scalar_gaussian_exchange_against_closed_form_and_gamma_reduction():
    from tests.test_manta_one_electron import _basis as scalar_basis
    basis = scalar_basis((0,), contracted=False); alpha = basis.shells()[0].exponents[0]
    density = np.array([[[[1.2]], [[-.7]], [[.3]]]], dtype=complex)
    q, offsets = _transfers(); expected = np.zeros(3)
    for target in range(3):
        for source in range(3):
            pair = target*3+source; q2 = np.sum(q[offsets[pair]:offsets[pair+1]]**2, axis=1)
            expected[target] += WEIGHTS[source]*density[0, source, 0, 0].real*4*np.pi/abs(np.linalg.det(LATTICE))*np.sum(np.exp(-q2/(4*alpha))/q2)
    actual = _compute(basis, density, cells=np.zeros((1, 3)))
    np.testing.assert_allclose(actual.values[0, :, 0, 0], expected, atol=1e-14, rtol=2e-13)
    gamma_k = np.zeros((1, 3)); gamma_d = density[:, :1].copy()
    g, o = _transfers(gamma_k)
    gamma = _compute(basis, gamma_d, cells=np.zeros((1, 3)), k=gamma_k, weights=np.ones(1), q=g, offsets=o)
    g2 = np.sum(g*g, axis=1)
    reference = 1.2*4*np.pi/abs(np.linalg.det(LATTICE))*np.sum(np.exp(-g2/(4*alpha))/g2)
    assert gamma.values[0, 0, 0, 0] == pytest.approx(reference, rel=2e-13, abs=1e-14)


def test_positive_density_produces_positive_exchange_and_zero_weight_channels_are_inactive():
    basis = _basis((0, 1)); x = _density(basis)
    density = np.ascontiguousarray(x@x.conj().transpose(0, 1, 3, 2))
    result = _compute(basis, density)
    assert np.min(np.linalg.eigvalsh(result.values)) > -1e-12
    weights = np.array([0., .4, .6])
    before = _compute(basis, density, weights=weights)
    density[:, 0] *= 7
    after = _compute(basis, density, weights=weights)
    np.testing.assert_array_equal(before.values, after.values)
    assert before.receipt.matrix_products < before.plan.matrix_products


@pytest.mark.parametrize('field, plan_field', [('memory_bytes', 'modeled_peak_bytes'), ('shell_calls', 'shell_calls'),
    ('primitive_quartets', 'primitive_pairs'), ('scalar_terms', 'scalar_terms'), ('kernel_terms', 'kernel_terms')])
def test_aggregate_budget_boundaries_precede_density_payload_copy(field, plan_field):
    basis = _basis((0, 1)); q, offsets = _transfers(); budget = manta.ResourceBudget(); budget.other_live_bytes = 31793
    plan = manta.plan_periodic_exchange(basis, int(np.max(np.diff(offsets))), len(CELLS), len(MOMENTA), 2, 4, budget=budget)
    setattr(budget, field, getattr(plan, plan_field))
    result = _compute(basis, budget=budget)
    assert result.receipt.modeled_peak_bytes == plan.modeled_peak_bytes
    setattr(budget, field, getattr(plan, plan_field)-1)
    with pytest.raises(ValueError, match='budget'):
        _compute(basis, np.full_like(_density(basis), np.nan), budget=budget)


@pytest.mark.parametrize('error', ['zero', 'duplicate', 'reverse_missing', 'fractional', 'bad_cells'])
def test_transfer_and_cell_source_contracts_refuse_ambiguous_domains(error):
    basis = _basis((0, 1)); q, offsets = _transfers(); cells = CELLS
    if error == 'zero': q[0] = 0
    elif error == 'duplicate': q[0] = q[1]
    elif error == 'reverse_missing':
        # Remove one vector from target=0/source=1 without its reversed partner.
        index = offsets[1]; q = np.delete(q, index, axis=0); offsets[2:] -= 1
    elif error == 'fractional': q[0] += .13
    else: cells = CELLS[1:2]
    with pytest.raises(ValueError, match='Q=0|duplicate|reverse|reversal|label'):
        _compute(basis, q=q, offsets=offsets, cells=cells)


@pytest.mark.parametrize('kind', ['weights', 'nan', 'nonhermitian', 'negative_weight', 'mask_by_other_channel'])
def test_each_density_block_and_weight_policy_is_validated(kind):
    basis = _basis((0, 1)); density = _density(basis); weights = WEIGHTS.copy()
    if kind == 'weights': weights *= .9
    elif kind == 'negative_weight': weights = np.array([-.1, .4, .7])
    elif kind == 'nan': density[0, 0, 0, 0] = np.nan
    else:
        density[0, 0, 0, 1] += .1j
        if kind == 'mask_by_other_channel': density[1] *= 1e15
    with pytest.raises(ValueError): _compute(basis, density, weights=weights)


def test_empty_transfer_support_is_zero_but_density_validation_still_runs():
    basis = _basis((0, 1)); density = _density(basis)
    q = np.empty((0, 3)); offsets = np.zeros(10, np.int64)
    result = _compute(basis, density, q=q, offsets=offsets)
    assert not np.any(result.values) and result.receipt.shell_calls == result.receipt.matrix_products == 0
    np.testing.assert_array_equal(result.receipt.energies, 0.)
    density[0, 1, 0, 0] = np.nan
    with pytest.raises(ValueError): _compute(basis, density, q=q, offsets=offsets)


def test_fortran_exports_strict_dtypes_and_offset_validation():
    basis = _basis((0, 1)); q, offsets = _transfers(); density = _density(basis)
    arguments = [LATTICE, q, offsets, CELLS, MOMENTA, WEIGHTS, density]
    result = manta.compute_periodic_exchange(basis, *arguments, 4)
    fortran = manta.compute_periodic_exchange(basis, *[np.asfortranarray(a) for a in arguments], 4)
    np.testing.assert_array_equal(fortran.values, result.values)
    assert fortran.receipt.density_identity == result.receipt.density_identity
    for index, dtype in [(0, np.float32), (1, np.complex128), (2, np.int32), (6, np.float64)]:
        changed = arguments.copy(); changed[index] = changed[index].real.astype(dtype)
        with pytest.raises(ValueError): manta.compute_periodic_exchange(basis, *changed, 4)
    for changed in (offsets[:-1], offsets+1, np.array([0, -1]+offsets[2:].tolist()), offsets*2):
        with pytest.raises(ValueError, match='offset'):
            _compute(basis, q=q, offsets=changed)
    with pytest.raises(ValueError, match='positive'):
        manta.plan_periodic_exchange(basis, 1, 1, 3, 0)


def test_selected_adapter_records_executed_epoch_and_respects_budget_without_mutation(monkeypatch):
    from vibeqc import output
    basis = _basis((0, 1)); q, offsets = _transfers(); density = _density(basis)
    engine = vq.IntegralEngineOptions(backend='manta')
    records = []; monkeypatch.setattr(output, 'record', lambda *args, **kwargs: records.append((args, kwargs)))
    args = (basis, LATTICE, q, offsets, CELLS, MOMENTA, WEIGHTS, density)
    result = manta.build_periodic_exchange(*args, reciprocal_batch=4, integral_engine=engine)
    expected = _compute(basis, density)
    np.testing.assert_array_equal(result.values, expected.values)
    evidence = records[0][1]['integral_provenance']
    assert evidence['density_identity'] == expected.receipt.density_identity
    assert evidence['source_identity'] == expected.receipt.source_identity
    assert evidence['citation_methods'] == ['manta_periodic_exchange']
    assert evidence['boundary'] == 'full-coulomb-q0-omitted-no-madelung'
    assert engine.budget.other_live_bytes == 0
    engine.budget.memory_bytes = 1
    with pytest.raises(ValueError, match='budget'):
        manta.build_periodic_exchange(*args, integral_engine=engine)
    for bad in (object(), vq.IntegralEngineOptions(backend='libint')):
        with pytest.raises((ValueError, TypeError)):
            manta.build_periodic_exchange(*args, integral_engine=bad)
    for bad in (True, 0, 1.5):
        with pytest.raises(ValueError, match='batch'):
            manta.build_periodic_exchange(*args, reciprocal_batch=bad, integral_engine=engine)
    assert len(records) == 1


@pytest.mark.parametrize('pure', [False, True])
def test_maximum_angular_exchange_channel_against_quadrature(pure):
    from tests.test_manta_one_electron import _basis as angular_basis
    basis = angular_basis((6,), pure, contracted=False); k = np.zeros((1, 3))
    q, offsets = _transfers(k)
    density = np.eye(basis.nbasis, dtype=complex)[None, None, :, :]
    result = _compute(basis, density, k=k, q=q, offsets=offsets, cells=np.zeros((1, 3)), weights=np.ones(1))
    b = _quadrature(basis.shells()[0], basis.shells()[0], q)
    kernel = 4*np.pi/abs(np.linalg.det(LATTICE))/np.sum(q*q, axis=1)
    expected = np.einsum('g,gai,gaj->ij', kernel, b.conj(), b)
    np.testing.assert_allclose(result.values[0, 0], expected, atol=3e-12, rtol=3e-11)


# The arithmetic reference starts at the producer's actual stored Fourier
# panels and stored reciprocal factors, independently contracted as Fractions.
def arithmetic_case(kernel='coulomb',omega=0.,kind='ordinary',batch=2):
    import math
    basis=_basis((0,0));lattice=np.diag([8.,8.,8.]);k=np.array([[0.,0.,0.],[math.pi/8,0.,0.]])
    base=np.array([[-math.pi/4,0.,0.],[0.,0.,0.],[math.pi/4,0.,0.]])
    qs=[]
    for target in k:
        for source in k:
            q=base+target-source
            if not (kernel=='erfc' and omega>0):q=q[np.any(q!=0.,axis=1)]
            qs.append(q)
    offsets=np.r_[0,np.cumsum([len(q) for q in qs])].astype(np.int64)
    rng=np.random.default_rng(736);d=rng.normal(size=(2,2,2,2))+1j*rng.normal(size=(2,2,2,2));d=np.ascontiguousarray(d+d.swapaxes(-1,-2).conj())
    if kind=='cancellation':d[...,0,0]+=2.**40;d[...,1,1]-=2.**40
    elif kind=='subnormal':d*=2.**-1030
    elif kind=='zero':d*=0.
    weights=np.array([.27,.73]) if kind!='zero-weight' else np.array([0.,1.])
    return dict(basis=basis,lattice=lattice,wave_vectors=np.ascontiguousarray(np.concatenate(qs)),
                transfer_offsets=offsets,translations=np.zeros((1,3)),momenta=k,weights=weights,densities=d,
                reciprocal_batch=batch,kernel=kernel,omega=omega)


def exact_stored_exchange(a):
    import math
    from fractions import Fraction as Q
    def z(v):return Q(float(v.real)),Q(float(v.imag))
    def plus(x,y):return x[0]+y[0],x[1]+y[1]
    def prod(x,y):return x[0]*y[0]-x[1]*y[1],x[0]*y[1]+x[1]*y[0]
    def factor(norm,w):
        kernel,omega=a['kernel'],a['omega']
        if kernel=='coulomb':return w*4*math.pi/512./norm
        if omega==0:value=0. if kernel=='erf' else 4*math.pi/norm
        elif norm==0:value=(math.pi/omega)/omega
        else:
            x=(.5*math.sqrt(norm))/omega;t=x*x
            if kernel=='erf':value=math.exp(math.log(4*math.pi)-math.log(norm)-t)
            elif t<=1:value=((math.pi/omega)/omega)*(1. if t==0 else -math.expm1(-t)/t)
            else:value=(4*math.pi/norm)*(-math.expm1(-t))
        return w*value/512.
    d=a['densities'];nd,nk,n,_=d.shape;out={index:(Q(0),Q(0)) for index in np.ndindex(d.shape)}
    batch=a['reciprocal_batch']
    for target in range(nk):
        for source in range(nk):
            pair=target*nk+source;begin,end=a['transfer_offsets'][pair:pair+2]
            for start in range(begin,end,batch):
                q=a['wave_vectors'][start:min(start+batch,end)]
                panel=manta.compute_bloch_fourier_panel(a['basis'],q,a['translations'],a['momenta'][target:target+1]).values[0]
                for iq,vector in enumerate(q):
                    # These supports have only an x component; volume is the
                    # exactly representable 8^3. Reproduce each stored factor
                    # before exact rational contraction, without bounding libm.
                    weight=Q(factor(float(vector[0])*float(vector[0]),float(a['weights'][source])))
                    for state,i,j in np.ndindex(nd,n,n):
                        value=(Q(0),Q(0))
                        for u,v in np.ndindex(n,n):
                            left=z(panel[u,i,iq].conjugate());right=z(panel[v,j,iq])
                            value=plus(value,prod(prod(left,z(d[state,source,u,v])),right))
                        key=state,target,i,j;out[key]=plus(out[key],(weight*value[0],weight*value[1]))
    energies=[]
    for state in range(nd):
        e=Q(0)
        for k,i,j in np.ndindex(nk,n,n):e-=Q(1,2)*Q(float(a['weights'][k]))*prod(z(d[state,k,j,i]),out[state,k,i,j])[0]
        energies.append(e)
    return out,energies


@pytest.mark.parametrize('kernel,omega',[('coulomb',0.),('erf',0.),('erf',.4),('erfc',0.),('erfc',.4)])
@pytest.mark.parametrize('kind',['ordinary','cancellation','subnormal','zero','zero-weight'])
@pytest.mark.parametrize('batch',[1,3])
def test_exchange_arithmetic_exact_stored_panel_graph(kernel,omega,kind,batch):
    from fractions import Fraction as Q
    a=arithmetic_case(kernel,omega,kind,batch)
    actual=manta.compute_periodic_exchange(**a,audit_roundoff=True)
    control=manta.compute_periodic_exchange(**a)
    np.testing.assert_array_equal(actual.values,control.values)
    assert actual.receipt.energies==control.receipt.energies
    assert actual.receipt.source_identity==control.receipt.source_identity
    assert actual.receipt.density_identity==control.receipt.density_identity
    assert actual.receipt.roundoff_evaluated and actual.plan.roundoff_requested
    assert not control.receipt.roundoff_evaluated and control.roundoff_bounds.shape[0]==0
    assert actual.receipt.roundoff_scope=='stored-bloch-panels-and-reciprocal-factors'
    for name in ('panel_calls','shell_calls','primitive_pairs','matrix_products','contraction_terms'):
        assert getattr(actual.receipt,name)==getattr(control.receipt,name)
    exact,energy=exact_stored_exchange(a)
    for key,(re,im) in exact.items():
        value=actual.values[key];error=abs(Q(float(value.real))-re)+abs(Q(float(value.imag))-im)
        assert error<=Q(float(actual.roundoff_bounds[key])),(key,error,actual.roundoff_bounds[key])
    for i,e in enumerate(energy):assert abs(Q(actual.receipt.energies[i])-e)<=Q(actual.receipt.energy_roundoff_bounds[i])
    assert actual.receipt.roundoff_bound==np.max(actual.roundoff_bounds)
    if kind=='zero' or (kernel=='erf' and omega==0):
        assert actual.receipt.roundoff_bound==0. and actual.receipt.energy_roundoff_bounds==[0.,0.]


@pytest.mark.parametrize('field,quantity',[('memory_bytes','modeled_peak_bytes'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_exchange_arithmetic_admits_all_resources_before_payload(field,quantity):
    a=arithmetic_case();b=manta.ResourceBudget();nq=max(np.diff(a['transfer_offsets']))
    p=manta.plan_periodic_exchange(a['basis'],int(nq),1,2,2,a['reciprocal_batch'],None,b,audit_roundoff=True)
    plain=manta.plan_periodic_exchange(a['basis'],int(nq),1,2,2,a['reciprocal_batch'],None,b)
    assert getattr(p,quantity)>getattr(plain,quantity)
    setattr(b,field,getattr(p,quantity)-1)
    with pytest.raises((ValueError,RuntimeError),match='budget'):
        manta.compute_periodic_exchange(**dict(a,densities=np.full_like(a['densities'],np.nan)),budget=b,audit_roundoff=True)
    setattr(b,field,getattr(p,quantity))
    assert manta.compute_periodic_exchange(**a,budget=b,audit_roundoff=True).receipt.roundoff_evaluated
