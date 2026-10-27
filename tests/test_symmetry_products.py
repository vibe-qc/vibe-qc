"""Independent partner, multiplicity and finite-source tensor oracles."""
from itertools import permutations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc.symmetry_shared import Budget, FiniteGroup
from vibeqc.symmetry_ao import molecular_point_group, build_molecular_space_action
from vibeqc.symmetry_products import IrrepMatrices, InvariantTensorMap, recoupling
from tests.test_symmetry_quartets import molecular_case, SPACE, dense_jk

BUDGET = Budget(1 << 30, 10**13)


def c3v():
    mol, basis, r, t = molecular_case()
    group = molecular_point_group(mol,r,t,identity_label='C3v product test',budget=BUDGET)
    reps = (np.ones((6,1,1),complex),
            np.array([1,1,1,-1,-1,-1],complex).reshape(6,1,1),
            np.ascontiguousarray(r[:,:2,:2],dtype=complex))
    return IrrepMatrices(group,reps,budget=BUDGET)


def molecular_pair(kind='ammonia',basis_name='sto-3g'):
    mol,basis,r,t = molecular_case(kind,basis_name=basis_name)
    irreps = c3v()
    s = np.asarray(vq.compute_overlap(basis))
    e,c = np.linalg.eigh(s); c = c/np.sqrt(e)
    inv = np.linalg.solve(c,np.eye(len(c)))
    actions = [build_molecular_space_action(mol,basis,rr,tt,source=SPACE,target=SPACE,budget=BUDGET)
               for rr,tt in zip(r,t)]
    u = np.array([inv@a.apply(np.ascontiguousarray(c,dtype=complex)) for a in actions])
    pairs = irreps.resolve(np.array([np.kron(x,x.conj()) for x in u]),budget=BUDGET)
    eri = np.asarray(vq.compute_eri(basis)).reshape((basis.nbasis,)*4)
    adapted = np.einsum('ap,bq,cr,ds,abcd->pqrs',c,c,c,c,eri,optimize=True)
    j = adapted.transpose(0,1,3,2).reshape(basis.nbasis**2,-1).astype(complex)
    k = adapted.transpose(0,3,1,2).reshape(j.shape).astype(complex)
    maps = tuple(InvariantTensorMap(x,target=pairs,source=pairs,
        source_tolerance=1e-10,budget=BUDGET) for x in (j,k))
    return mol,basis,c,inv,eri,pairs,maps,(j,k),actions


def test_c3v_product_against_published_partner_projectors():
    reps = c3v(); resolved = reps.product(2,2,budget=BUDGET)
    assert reps.products[2,2].tolist() == [1,1,1]
    assert resolved.multiplicities == (1,1,1)
    # Hellmann/Gauss Eqs. (5)-(8), in (xx,xy,yx,yy) coordinates.
    published = np.array([[1,0,1,0],[0,1,0,-1],[0,-1,0,-1],[1,0,-1,0]],complex)/np.sqrt(2)
    for lo,hi in ((0,1),(1,2),(2,4)):
        actual = resolved.basis[:,lo:hi]
        np.testing.assert_allclose(actual@actual.conj().T,
            published[:,lo:hi]@published[:,lo:hi].conj().T,atol=2e-14,rtol=0)
    swap = recoupling(resolved,resolved,tensor_shape=(2,2),permutation=(1,0),budget=BUDGET)
    np.testing.assert_allclose([b[0,0] for b in swap.blocks],[1,-1,1],atol=2e-14,rtol=0)


def test_multiplicity_gauge_recoupling_and_reduced_composition():
    reps = c3v()
    raw = np.array([np.kron(np.kron(e,e),e) for e in reps.matrices[2]])
    base = reps.resolve(raw,budget=BUDGET)
    assert base.multiplicities == (1,1,3)
    cycle = recoupling(base,base,tensor_shape=(2,2,2),permutation=(1,2,0),budget=BUDGET)
    # A three-dimensional E multiplicity space mixes through a matrix.
    assert np.count_nonzero(abs(cycle.blocks[2])>1e-10)>3
    triple = cycle.compose(cycle,budget=BUDGET).compose(cycle,budget=BUDGET)
    np.testing.assert_allclose(triple.to_full(budget=BUDGET),np.eye(8),atol=4e-14,rtol=0)
    assert cycle.stored_elements == 11 < 64
    assert triple.composition_products == 29 < 8**3
    rng = np.random.default_rng(450)
    x = rng.normal(size=(8,3))+1j*rng.normal(size=(8,3))
    got = cycle.apply(x,budget=BUDGET)
    np.testing.assert_allclose(got.values,cycle.to_full(budget=BUDGET)@x,atol=3e-14,rtol=0)
    assert got.scalar_products == 20*3 and got.transport_scalar_products == 2*64*3
    # An arbitrary unitary gauge across all raw coordinates must be carried
    # explicitly; matching characters do not make reduced arrays compatible.
    q,_ = np.linalg.qr(rng.normal(size=(8,8))+1j*rng.normal(size=(8,8)))
    changed = reps.resolve(np.array([q.conj().T@u@q for u in raw]),budget=BUDGET)
    bridge = InvariantTensorMap(q.conj().T.copy(),target=changed,source=base,source_tolerance=1e-10,budget=BUDGET)
    with pytest.raises(ValueError,match='intermediate gauge'):
        cycle.compose(bridge,budget=BUDGET)
    np.testing.assert_allclose(bridge.compose(cycle,budget=BUDGET).to_full(budget=BUDGET),
        q.conj().T@cycle.to_full(budget=BUDGET),atol=4e-14,rtol=0)


def test_finite_gram_uses_inverse_for_reconstruction_transport_and_composition(monkeypatch):
    reps = c3v()
    raw = np.array([np.kron(np.kron(e,e),e) for e in reps.matrices[2]])
    exact_frame = reps.resolve(raw,budget=BUDGET)
    eigh = np.linalg.eigh

    def finite_seeds(projector):
        values, vectors = eigh(projector)
        # Model a finite eigensolver frame within the existing 1e-10
        # admission. A common scale on each partner multiplet still
        # intertwines, but its Gram is measurably different from identity.
        vectors *= 1+np.linspace(1e-12,8e-12,len(values))
        return values, vectors

    with monkeypatch.context() as patch:
        patch.setattr(np.linalg,'eigh',finite_seeds)
        frame = reps.resolve(raw,budget=BUDGET)
    assert 1e-11 < frame.orthogonality_residual < reps.tolerance
    np.testing.assert_allclose(frame.basis_inverse@frame.basis,np.eye(8),atol=2e-15,rtol=0)
    assert np.linalg.norm(frame.basis_inverse-frame.basis.conj().T)>1e-11
    assert frame.retained_bytes >= frame.basis.nbytes+frame.basis_inverse.nbytes
    with pytest.raises(ValueError):frame.basis_inverse.setflags(write=True)

    identity = np.eye(8,dtype=complex)
    keep = InvariantTensorMap(identity,target=frame,source=frame,
        source_tolerance=1e-13,budget=BUDGET)
    bridge = InvariantTensorMap(identity,target=exact_frame,source=frame,
        source_tolerance=1e-13,budget=BUDGET)
    rng = np.random.default_rng(526)
    values = rng.normal(size=(8,3))+1j*rng.normal(size=(8,3))
    np.testing.assert_allclose(keep.apply(values,budget=BUDGET).values,values,atol=3e-14,rtol=0)
    repeated = keep
    for _ in range(12):repeated = repeated.compose(keep,budget=BUDGET)
    np.testing.assert_allclose(repeated.to_full(budget=BUDGET),identity,atol=3e-14,rtol=0)
    np.testing.assert_allclose(bridge.compose(repeated,budget=BUDGET).to_full(budget=BUDGET),
        identity,atol=3e-14,rtol=0)
    cycle = recoupling(frame,frame,tensor_shape=(2,2,2),permutation=(1,2,0),budget=BUDGET)
    restored = cycle.compose(cycle,budget=BUDGET).compose(cycle,budget=BUDGET).to_full(budget=BUDGET)
    np.testing.assert_allclose(restored,identity,atol=3e-14,rtol=0)


def test_tetrahedral_product_has_a_repeated_irrep_and_complex_conjugate_rows():
    # Proper tetrahedral rotations are the even vertex permutations. Build
    # the one-dimensional rows from the C3 quotient by the four-group.
    vertices = np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])
    even = [p for p in permutations(range(4))
            if sum(p[i]>p[j] for i in range(4) for j in range(i+1,4))%2 == 0]
    rotations = np.array([vertices[list(p)].T@vertices//4 for p in even],np.int64)
    group = FiniteGroup.from_integer_rotations(rotations,identity_label='tetrahedral A4',budget=BUDGET)
    normal = [g for g in range(12) if group.product(g,g)==group.identity]
    generator = next(g for g in range(12) if g not in normal)
    powers = [group.identity,generator,group.product(generator,generator)]
    coset = np.empty(12,np.int64)
    for i,g in enumerate(powers):
        for h in normal:coset[group.product(g,h)] = i
    chars = [np.exp(2j*np.pi*k*coset/3)[:,None,None] for k in range(3)]
    reps = IrrepMatrices(group,tuple(chars)+(rotations.astype(complex),),budget=BUDGET)
    assert reps.conjugates == (0,2,1,3)
    assert reps.products[3,3].tolist() == [1,1,1,2]
    resolved = reps.product(3,3,budget=BUDGET)
    assert resolved.multiplicities == (1,1,1,2)
    swap = recoupling(resolved,resolved,tensor_shape=(3,3),permutation=(1,0),budget=BUDGET)
    np.testing.assert_allclose(np.linalg.eigvalsh(swap.blocks[3]),[-1,1],atol=3e-14,rtol=0)
    np.testing.assert_allclose(swap.compose(swap,budget=BUDGET).to_full(budget=BUDGET),
        np.eye(9),atol=4e-14,rtol=0)


def test_complex_abelian_variance_is_not_realified_or_antiunitary():
    from tests.test_symmetry_abelian import cyclic
    chars = cyclic()
    reps = IrrepMatrices(chars.group,tuple(c[:,None,None].copy() for c in chars.characters),budget=BUDGET)
    assert reps.conjugates == (0,2,1)
    assert reps.product(1,1,budget=BUDGET).multiplicities == (0,0,1)
    assert reps.product(1,1,variance=(-1,1),budget=BUDGET).multiplicities == (1,0,0)
    with pytest.raises(ValueError,match='index variance'):
        reps.product(1,1,variance=(True,1),budget=BUDGET)
    group = FiniteGroup.from_table(np.array([[0,1],[1,0]],np.int64),np.array([0,1],np.uint8),
        identity=0,identity_label='antiunitary rejection',budget=BUDGET)
    with pytest.raises(ValueError,match='antiunitary'):
        IrrepMatrices(group,(np.ones((2,1,1),complex),np.array([1,-1],complex).reshape(2,1,1)),budget=BUDGET)


@pytest.mark.parametrize('kind',['ammonia','cartesian'])
def test_actual_integral_maps_arbitrary_density_metric_and_tensor_resort(kind):
    mol,basis,c,inv,eri,pairs,maps,full,actions = molecular_pair(kind)
    rng = np.random.default_rng(2014)
    d = rng.normal(size=(basis.nbasis,)*2)+1j*rng.normal(size=(basis.nbasis,)*2)
    def build(d):
        adapted = inv@d@inv.conj().T
        return [inv.conj().T@m.apply(adapted.reshape(-1,1),budget=BUDGET).values.reshape(d.shape)@inv
                for m in maps]
    computed = build(d)
    for got,wanted in zip(computed,dense_jk(eri,d)):
        np.testing.assert_allclose(got,wanted,atol=3e-11,rtol=0)
    for action in actions:
        for got,wanted in zip(build(action.push_density(d)),computed):
            np.testing.assert_allclose(action.pull_operator(got),wanted,atol=5e-11,rtol=0)
    n = basis.nbasis
    # Restore every partner before exchanging inner legs. Applying that
    # permutation directly to reduced blocks loses the missing elements.
    jfull = maps[0].to_full(budget=BUDGET).reshape((n,)*4)
    exchange = jfull.transpose(0,2,1,3).reshape(n*n,n*n).copy()
    resorted = InvariantTensorMap(exchange,target=pairs,source=pairs,source_tolerance=1e-10,budget=BUDGET)
    np.testing.assert_allclose(resorted.to_full(budget=BUDGET),full[1],atol=3e-11,rtol=0)
    contracted = maps[0].compose(maps[1],budget=BUDGET)
    np.testing.assert_allclose(contracted.to_full(budget=BUDGET),full[0]@full[1],atol=3e-10,rtol=0)
    assert contracted.composition_products < n**6
    for m in maps:
        assert m.stored_elements < n**4
        with pytest.raises(ValueError):m.blocks[0].setflags(write=True)
    # Perturb across inequivalent irreps; an arbitrary diagonal AO-pair
    # entry can itself be an invariant and would not test this refusal.
    bad = full[0]+.01*np.outer(pairs.basis[:,0],pairs.basis[:,pairs.offsets[2]].conj())
    with pytest.raises(ValueError,match='partner reconstruction'):
        InvariantTensorMap(bad,target=pairs,source=pairs,source_tolerance=1e-10,budget=BUDGET)


def test_representation_and_resource_refusals():
    reps = c3v()
    with pytest.raises(MemoryError):
        IrrepMatrices(reps.group,reps.matrices,budget=Budget(1,10**13))
    with pytest.raises(MemoryError):reps.product(2,2,budget=Budget(1,10**13))
    bad = reps.matrices[2].copy();bad[1,0,0]+=.01
    with pytest.raises(ValueError):reps.resolve(bad,budget=BUDGET)
    with pytest.raises(ValueError,match='reducible or equivalent'):
        IrrepMatrices(reps.group,(reps.matrices[0],reps.matrices[0],reps.matrices[2]),budget=BUDGET)
    p = reps.product(2,2,budget=BUDGET)
    with pytest.raises(MemoryError):
        InvariantTensorMap(np.eye(4,dtype=complex),target=p,source=p,source_tolerance=1e-10,budget=Budget(1,10**13))
    with pytest.raises(ValueError,match='invalid tensor recoupling'):
        recoupling(p,p,tensor_shape=(2,2),permutation=(0,0),budget=BUDGET)
    with pytest.raises(ValueError,match='partner reconstruction'):
        InvariantTensorMap(np.diag([1,2,3,4]).astype(complex),
            target=p,source=p,source_tolerance=1e-10,budget=BUDGET)


def test_error_propagation_rectangular_empty_irreps_and_citation_route():
    reps = c3v()
    left = reps.resolve(reps.matrices[2],budget=BUDGET)
    right = reps.resolve(np.array([np.kron(np.eye(2),g) for g in reps.matrices[2]]),budget=BUDGET)
    exact = np.hstack((np.eye(2),2j*np.eye(2)))
    supplied = exact.copy(); supplied[0,0]+=1e-12
    m = InvariantTensorMap(supplied,target=left,source=right,source_tolerance=1e-10,budget=BUDGET)
    assert m.blocks[0].shape == (0,0) and m.blocks[2].shape == (1,2)
    x = np.arange(12,dtype=complex).reshape(4,3)
    result = m.apply(x,budget=BUDGET)
    assert np.linalg.norm(result.values-supplied@x) <= result.error_bound+1e-13
    with pytest.raises(MemoryError):m.apply(x,budget=Budget(1,10**13))
    with pytest.raises(ValueError,match='nonfinite'):
        m.apply(np.full((4,1),np.nan+0j),budget=BUDGET)
    with pytest.raises(ValueError):left.basis.setflags(write=True)
    from vibeqc.output.citations.registry import load_default_database
    routes = load_default_database().routes()['numerics']
    assert set(routes[m.citation_numerics[0]]) == {'hellmann_gauss_direct_product_2026',
        'goodlett_molsym_2024','stanton_gauss_watts_bartlett_ccsd_1991'}
