"""Independent finite-integral oracles for complex Abelian block execution."""
import numpy as np
import pytest
from scipy.linalg import eigh

import vibeqc as vq
from vibeqc import _vibeqc_core as core
from vibeqc.symmetry_abelian import AbelianCharacters, AbelianERIPlan
from vibeqc.symmetry_ao import build_molecular_space_action
from vibeqc.symmetry_shared import Budget, FiniteGroup, OperatorContract, SpaceIdentity

BUDGET = Budget(512 << 20, 10**11)
SPACE = SpaceIdentity('finite molecule', 'test Gaussian basis', 'adapted AO', 'metric orthonormal')
CONTRACT = OperatorContract('molecular four-center Coulomb', 'native full ERI',
    'all molecular quartets', 'native source as supplied', '(p* q | r* s)')


def cyclic(order=3):
    table = (np.arange(order)[:, None] + np.arange(order)[None, :]) % order
    group = FiniteGroup.from_table(table.astype(np.int64), np.zeros(order, np.uint8),
        identity=0, identity_label='cyclic test group', budget=BUDGET)
    chars = np.ascontiguousarray(np.exp(2j*np.pi*np.outer(np.arange(order), np.arange(order))/order))
    return AbelianCharacters(group, chars, budget=BUDGET)


def plan(eri, labels, irreps, **kw):
    return AbelianERIPlan(np.ascontiguousarray(eri, dtype=complex),
        np.asarray(labels, np.int64), irreps, space=SPACE, contract=CONTRACT,
        source_tolerance=kw.pop('source_tolerance', 1e-10), budget=kw.pop('budget', BUDGET), **kw)


def dense_jk(eri, density):
    # Direct scalar-index Einstein contractions, no symmetry plan or averaging.
    return (np.einsum('pqrs,sr->pq', eri, density, optimize=False),
            np.einsum('prsq,rs->pq', eri, density, optimize=False))


def molecular_source(kind='ammonia', basis_name='sto-3g', stretch=0.):
    if kind == 'water':
        mol = vq.Molecule([vq.Atom(8, [0., 0., 0.]),
            vq.Atom(1, [1.5+stretch, 0., -1.2]), vq.Atom(1, [-1.5-stretch, 0., -1.2])])
        basis = vq.BasisSet(mol, basis_name)
        rotations = [np.eye(3), np.diag([-1., -1., 1.]),
                     np.diag([-1., 1., 1.]), np.diag([1., -1., 1.])]
        table = np.bitwise_xor.outer(np.arange(4), np.arange(4)).astype(np.int64)
        group = FiniteGroup.from_table(table, np.zeros(4, np.uint8), identity=0,
            identity_label='water C2v', budget=BUDGET)
        chars = np.array([[(-1)**((a & b).bit_count()) for b in range(4)] for a in range(4)], complex)
        irreps = AbelianCharacters(group, chars, budget=BUDGET)
    else:
        theta = 2*np.pi/3
        rotation = np.array([[np.cos(theta), -np.sin(theta), 0.],
                             [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])
        rotations = [np.eye(3), rotation, rotation @ rotation]
        irreps = cyclic()
        if kind == 'cartesian':
            mol = vq.Molecule([vq.Atom(2, [0., 0., 0.])])
            basis = vq.BasisSet(mol, [core.ShellInfo(0, l, False, [.8], [1.], [0., 0., 0.])
                for l in (0, 2)], 'Cartesian s+d', False)
        else:
            mol = vq.Molecule([vq.Atom(7, [0., 0., .3])] + [vq.Atom(1,
                [(1.7+stretch)*np.cos(t), (1.7+stretch)*np.sin(t), -.7])
                for t in (0., theta, 2*theta)])
            basis = vq.BasisSet(mol, basis_name)
    overlap = np.asarray(vq.compute_overlap(basis))
    values, vectors = eigh(overlap)
    orthogonal = vectors / np.sqrt(values)
    actions = [build_molecular_space_action(mol, basis, np.ascontiguousarray(r), np.zeros(3),
        source=SPACE, target=SPACE, budget=BUDGET) for r in rotations]
    unitary = [orthogonal.conj().T @ overlap @ a.apply(np.eye(basis.nbasis, dtype=complex)) @ orthogonal
               for a in actions]
    columns, labels = [], []
    for a, chars in enumerate(irreps.characters):
        projector = sum(c.conjugate()*u for c, u in zip(chars, unitary))/len(unitary)
        eigenvalues, eigvec = eigh(projector)
        columns.append(orthogonal @ eigvec[:, eigenvalues > .5])
        labels.extend([a]*np.count_nonzero(eigenvalues > .5))
    c = np.ascontiguousarray(np.column_stack(columns))
    np.testing.assert_allclose(c.conj().T @ overlap @ c, np.eye(basis.nbasis), atol=2e-13)
    for g, action in enumerate(actions):
        np.testing.assert_allclose(action.apply(c), c*irreps.characters[labels, g], atol=2e-13)
    ao = np.asarray(vq.compute_eri(basis)).reshape((basis.nbasis,)*4)
    eri = np.ascontiguousarray(np.einsum('up,vq,wr,xs,uvwx->pqrs',
        c.conj(), c, c.conj(), c, ao, optimize=True))
    h = np.asarray(vq.compute_kinetic(basis)) + np.asarray(vq.compute_nuclear(basis, mol))
    return mol, basis, irreps, labels, c, eri, ao, h, actions


def test_complex_selection_has_conjugates_and_variance_not_xor():
    irreps = cyclic()
    assert tuple(irreps.conjugates) == (0, 2, 1)
    assert irreps.allows((1, 2, 2, 1), (-1, 1, -1, 1))
    assert not irreps.allows((1, 1, 1, 1), (1, 1, 1, 1))
    assert irreps.allows((1, 1, 1, 1), (-1, 1, -1, 1))
    assert irreps.allows((2, 1), (1, -1), target=1)
    # Independent reorderings of operations and irreps, including the identity.
    order = [2, 0, 1]
    original = irreps.group
    inverse = np.argsort(order)
    table = np.array([[inverse[original.product(a, b)] for b in order] for a in order], np.int64)
    group = FiniteGroup.from_table(table, np.zeros(3, np.uint8), identity=1,
        identity_label='reordered C3', budget=BUDGET)
    reordered = AbelianCharacters(group, np.ascontiguousarray(irreps.characters[np.ix_(order, order)]), budget=BUDGET)
    assert reordered.trivial == 1
    assert reordered.allows((0, 2), (1, 1))
    with pytest.raises(ValueError):
        reordered.characters.setflags(write=True)


@pytest.mark.parametrize('kind', ['water', 'ammonia', 'cartesian'])
def test_actual_eri_jk_energy_and_nonsymmetric_density_equivariance(kind):
    mol, basis, irreps, labels, c, eri, ao, h, actions = molecular_source(kind)
    reduced = plan(eri, labels, irreps)
    rng = np.random.default_rng(6128)
    trial = rng.normal(size=(len(c),)*2) + 1j*rng.normal(size=(len(c),)*2)
    for density in (trial, trial+trial.conj().T):
        result = reduced.jk(density, budget=BUDGET)
        j, k = dense_jk(eri, density)
        np.testing.assert_allclose(result.j, j, atol=1e-11, rtol=0)
        np.testing.assert_allclose(result.k, k, atol=1e-11, rtol=0)
        off_diagonal_irreps = np.array(labels)[:, None] != np.array(labels)[None, :]
        assert np.linalg.norm(result.k[off_diagonal_irreps]) > 1e-3
        # Independent AO computation and covariant operator pullback, including
        # the non-Euclidean-unitary Cartesian action.
        da = c @ density @ c.conj().T
        ja, ka = dense_jk(ao, da)
        np.testing.assert_allclose(result.j, c.conj().T @ ja @ c, atol=1e-11, rtol=0)
        np.testing.assert_allclose(result.k, c.conj().T @ ka @ c, atol=1e-11, rtol=0)
        e = .5*np.einsum('pq,qp', density, 2*c.conj().T@h@c + result.j-.5*result.k)
        ea = .5*np.einsum('pq,qp', da, 2*h + ja-.5*ka)
        np.testing.assert_allclose(e, ea, atol=1e-10, rtol=0)
        for g in range(irreps.group.order):
            phase = irreps.characters[labels, g]
            moved = np.ascontiguousarray(phase[:, None]*density*phase.conj()[None, :])
            transported = reduced.jk(moved, budget=BUDGET)
            for new, old in ((transported.j, result.j), (transported.k, result.k)):
                np.testing.assert_allclose(new, phase[:, None]*old*phase.conj()[None, :], atol=1e-11, rtol=0)
    assert reduced.stored_elements < reduced.dense_elements
    assert result.scalar_products == 2*sum(b.size for _, b in reduced.blocks)
    assert not reduced.production_reduction_authorized


@pytest.mark.parametrize('kind', ['water', 'ammonia'])
def test_block_fock_at_independently_converged_rhf_density(kind):
    mol, basis, ir, labels, c, eri, ao, h, _ = molecular_source(kind)
    reference = vq.run_rhf(mol, basis)
    assert reference.converged
    overlap = np.asarray(vq.compute_overlap(basis))
    density = np.ascontiguousarray(c.conj().T @ overlap @ np.asarray(reference.density) @ overlap @ c)
    result = plan(eri, labels, ir).jk(density, budget=BUDGET)
    h_adapted = c.conj().T @ h @ c
    fock = h_adapted + result.j - .5*result.k
    # Native SCF convergence is a distinct, weaker witness than the arbitrary
    # density tests above; compare against the native Fock without averaging.
    np.testing.assert_allclose(fock, c.conj().T @ np.asarray(reference.fock) @ c, atol=1e-8, rtol=0)
    j, k = dense_jk(ao, np.asarray(reference.density))
    energy = .5*np.einsum('pq,qp', density, h_adapted+fock)
    full_energy = .5*np.einsum('pq,qp', np.asarray(reference.density), 2*h+j-.5*k)
    np.testing.assert_allclose(energy, full_energy, atol=1e-10, rtol=0)


def test_source_refusal_uses_all_forbidden_blocks_and_has_explicit_fallback():
    irreps, labels = cyclic(), np.array([0, 1, 2], np.int64)
    eri = np.zeros((3,)*4, complex)
    eri[0, 0, 0, 1] = 1e-3
    with pytest.raises(ValueError, match='use unreduced source'):
        plan(eri, labels, irreps)
    # A symmetric trial density hides this defect; the source scan does not.
    j, _ = dense_jk(eri, np.eye(3))
    assert not np.any(j)
    density = np.zeros((3, 3), complex)
    density[1, 0] = 1
    assert dense_jk(eri, density)[0][0, 0] == 1e-3
    eri[0, 0, 0, 1] = 7e-11
    eri[0, 0, 0, 2] = 8e-11
    with pytest.raises(ValueError, match='selection rules'):
        plan(eri, labels, irreps)  # aggregate, not per-entry tolerance


def test_omission_bound_and_snapshot_cover_arbitrary_complex_densities():
    irreps = cyclic()
    eri = np.zeros((3,)*4, complex)
    eri[0, 0, 0, 1] = 1e-12j
    eri[1, 1, 2, 2] = 2.3
    p = plan(eri, [0, 1, 2], irreps)
    d = np.arange(9, dtype=complex).reshape(3, 3)*(1+2j)
    result = p.jk(d, budget=BUDGET)
    for actual, expected in zip((result.j, result.k), dense_jk(eri, d)):
        assert np.linalg.norm(actual-expected) <= result.error_bound + 1e-15
    digest = p.source_digest
    eri[:] = 0
    assert plan(eri, [0, 1, 2], irreps).source_digest != digest
    np.testing.assert_array_equal(result.j, p.jk(d, budget=BUDGET).j)
    for a in (p.labels, p.indices[0], p.blocks[0][1]):
        with pytest.raises(ValueError):
            a.setflags(write=True)


def test_dense_degenerate_gauge_mixing_and_scalar_time_reversal_separate():
    _, _, irreps, labels, c, eri, _, _, _ = molecular_source()
    rng = np.random.default_rng(192)
    gauge = np.zeros((len(c), len(c)), complex)
    for label in set(labels):
        indices = np.flatnonzero(np.array(labels) == label)
        q = np.linalg.qr(rng.normal(size=(len(indices),)*2)+1j*rng.normal(size=(len(indices),)*2))[0]
        gauge[np.ix_(indices, indices)] = q
    rotated = np.einsum('up,vq,wr,xs,uvwx->pqrs', gauge.conj(), gauge, gauge.conj(), gauge, eri, optimize=True)
    p, q = plan(eri, labels, irreps), plan(rotated, labels, irreps)
    d = rng.normal(size=(len(c),)*2)+1j*rng.normal(size=(len(c),)*2)
    a = p.jk(d, budget=BUDGET)
    b = q.jk(np.ascontiguousarray(gauge.conj().T@d@gauge), budget=BUDGET)
    np.testing.assert_allclose(b.j, gauge.conj().T@a.j@gauge, atol=1e-11)
    np.testing.assert_allclose(b.k, gauge.conj().T@a.k@gauge, atol=1e-11)
    # K changes the complex basis too: conjugate both the tensor and labels.
    tr = plan(eri.conj(), irreps.conjugates[labels], irreps)
    b = tr.jk(d.conj(), budget=BUDGET)
    np.testing.assert_allclose(b.j, a.j.conj(), atol=1e-11)
    np.testing.assert_allclose(b.k, a.k.conj(), atol=1e-11)


def test_density_energy_directional_derivative_and_separate_spin_calls():
    _, _, ir, labels, c, eri, _, _, _ = molecular_source()
    p = plan(eri, labels, ir)
    rng = np.random.default_rng(295)
    a, b = [rng.normal(size=(len(c),)*2)+1j*rng.normal(size=(len(c),)*2) for _ in range(2)]
    density, delta = .1*(a+a.conj().T), .1*(b+b.conj().T)
    jk = p.jk(density, budget=BUDGET)
    direction = np.einsum('pq,qp', delta, jk.j-.5*jk.k)
    def energy(d):
        result = p.jk(d, budget=BUDGET)
        return .5*np.einsum('pq,qp', d, result.j-.5*result.k)
    for step in (2e-4, 1e-4):
        finite_difference = (energy(density+step*delta)-energy(density-step*delta))/(2*step)
        np.testing.assert_allclose(finite_difference, direction, atol=1e-8, rtol=0)
    da, db = .1*a@a.conj().T, .1*b@b.conj().T
    total = p.jk(da+db, budget=BUDGET)
    for ds in (da, db):
        spin = p.jk(ds, budget=BUDGET)
        fullj, _ = dense_jk(eri, da+db)
        _, fullk = dense_jk(eri, ds)
        np.testing.assert_allclose(total.j-spin.k, fullj-fullk, atol=1e-11, rtol=0)


def test_tiny_forbidden_components_are_not_squared_to_zero():
    eri = np.zeros((3,)*4, complex)
    eri[0, 0, 0, 1] = 1e-300j
    with pytest.raises(ValueError, match='selection rules'):
        plan(eri, [0, 1, 2], cyclic(), source_tolerance=0.)
    p = plan(eri, [0, 1, 2], cyclic())
    assert p.omitted_norm == 1e-300
    with pytest.raises(ValueError, match='error bound'):
        p.jk(np.eye(3, dtype=complex)*1e-300, budget=BUDGET)


def test_displaced_geometry_requires_new_source_admission():
    _, _, irreps, labels, c, eri, _, _, _ = molecular_source()
    mol = vq.Molecule([vq.Atom(7, [.02, 0., .3])] + [vq.Atom(1,
        [1.7*np.cos(t), 1.7*np.sin(t), -.7]) for t in (0, 2*np.pi/3, 4*np.pi/3)])
    b = vq.BasisSet(mol, 'sto-3g')
    ao = np.asarray(vq.compute_eri(b)).reshape((b.nbasis,)*4)
    displaced = np.einsum('up,vq,wr,xs,uvwx->pqrs', c.conj(), c, c.conj(), c, ao, optimize=True)
    with pytest.raises(ValueError, match='selection rules'):
        plan(displaced, labels, irreps)
    # Symmetry-preserving displacements are separately readmitted, not carried
    # over from the undisplaced source. No analytic derivative is claimed.
    for delta in (-1e-3, 1e-3):
        _, _, ir, lab, _, v, _, _, _ = molecular_source(stretch=delta)
        assert plan(v, lab, ir).source_digest != plan(eri, labels, irreps).source_digest


@pytest.mark.parametrize('kind', ['antiunitary', 'cocycle', 'nonabelian', 'characters'])
def test_unsupported_representations_fail_closed(kind):
    ir = cyclic()
    if kind == 'characters':
        chars = ir.characters.copy()
        chars[1, 1] = 2
        with pytest.raises(ValueError, match='unitary'):
            AbelianCharacters(ir.group, chars, budget=BUDGET)
        return
    if kind == 'nonabelian':
        from itertools import permutations
        perms = list(permutations(range(3)))
        table = np.array([[perms.index(tuple(a[b[i]] for i in range(3))) for b in perms] for a in perms], np.int64)
        anti, cocycle = np.zeros(6, np.uint8), np.zeros((6, 6, 3), np.int64)
    else:
        table = np.array([[0, 1], [1, 0]], np.int64)
        anti = np.array([0, kind == 'antiunitary'], np.uint8)
        cocycle = np.zeros((2, 2, 3), np.int64)
        if kind == 'cocycle':
            cocycle[1, 1, 0] = 1
    g = FiniteGroup.from_table(table, anti, identity=0, identity_label=kind, budget=BUDGET,
        rotations=np.tile(np.eye(3, dtype=np.int64), (len(table), 1, 1)), cocycle=cocycle)
    with pytest.raises(ValueError, match='antiunitary|cocycles|non-Abelian'):
        AbelianCharacters(g, np.ones((g.order, g.order), complex), budget=BUDGET)


def test_exact_budget_and_refusal_before_payload_scan(monkeypatch):
    ir = cyclic()
    eri = np.zeros((3,)*4, complex)
    p = plan(eri, [0, 1, 2], ir)
    assert plan(eri, [0, 1, 2], ir, budget=Budget(p.required_bytes, p.required_work)).stored_elements == 27
    def forbidden(*a, **kw):
        raise AssertionError('payload scanned before admission')
    monkeypatch.setattr(np, 'isfinite', forbidden)
    with pytest.raises(MemoryError):
        plan(eri, [0, 1, 2], ir, budget=Budget(p.required_bytes-1, p.required_work))
    with pytest.raises(ValueError, match='work budget'):
        plan(eri, [0, 1, 2], ir, budget=Budget(p.required_bytes, p.required_work-1))
