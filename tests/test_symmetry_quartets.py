"""Independent finite-source oracles for molecular quartet scheduling."""
from dataclasses import replace
from itertools import permutations, product
import sys

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import _vibeqc_core as core
from vibeqc.symmetry_ao import build_molecular_space_action, molecular_point_group
from vibeqc.symmetry_shared import Budget, BlockSpaceAction, FiniteGroup, SpaceIdentity
from vibeqc.symmetry_quartets import (
    QuartetTile, ShellQuartetPlan, manta_quartet_requirements, qualify_manta_quartets,
)
from vibeqc.symmetry_quartets_streaming import PairQuartetSchedule, StreamingShellQuartetPlan

BUDGET = Budget(512 << 20, 10**12)
SPACE = SpaceIdentity('test molecule', 'native Gaussian source', 'AO', 'nonorthogonal')
SCHEDULE = 'stored'


@pytest.fixture(params=['stored', 'streaming'], autouse=True)
def quartet_backend(request, monkeypatch):
    module = sys.modules[__name__]
    monkeypatch.setattr(module, 'SCHEDULE', request.param)
    if request.param == 'streaming':
        monkeypatch.setattr(module, 'ShellQuartetPlan', StreamingShellQuartetPlan)


class DenseSource:
    """Already evaluated oracle: reports no native calls or saved integrals."""
    def __init__(self, values, offsets):
        self.values = np.ascontiguousarray(values)
        self.shell_offsets = np.asarray(offsets, np.int64)
        self.source_identity = 'fixed dense test source'
        self.modeled_peak_bytes = self.values.nbytes
        self.requests = []
        self.precision = 0.

    def evaluate(self, key):
        self.requests.append(key)
        selection = tuple(slice(self.shell_offsets[s], self.shell_offsets[s+1]) for s in key)
        values = np.ascontiguousarray(self.values[selection])
        return QuartetTile(values, self.source_identity, 0, 0, 0, self.precision)


def molecular_case(kind='ammonia', basis_name='sto-3g', displacement=0.):
    theta = 2*np.pi/3
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0.],
                         [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])
    reflection = np.diag([1., -1., 1.])
    rotations = [np.eye(3), rotation, rotation@rotation,
                 reflection, rotation@reflection, rotation@rotation@reflection]
    if kind == 'water':
        mol = vq.Molecule([vq.Atom(8, [0., 0., 0.]),
            vq.Atom(1, [1.5, 0., -1.2]), vq.Atom(1, [-1.5, 0., -1.2])])
        basis = vq.BasisSet(mol, basis_name)
        rotations = [np.eye(3), np.diag([-1., -1., 1.]),
                     np.diag([-1., 1., 1.]), np.diag([1., -1., 1.])]
    elif kind == 'cartesian':
        mol = vq.Molecule([vq.Atom(2, [0., 0., 0.])])
        basis = vq.BasisSet(mol, [core.ShellInfo(0, l, False, [.8], [1.], [0., 0., 0.])
            for l in (0, 2)], 'Cartesian s+d', False)
    else:
        mol = vq.Molecule([vq.Atom(7, [displacement, 0., .3])] + [vq.Atom(1,
            [1.7*np.cos(t), 1.7*np.sin(t), -.7]) for t in (0., theta, 2*theta)])
        basis = vq.BasisSet(mol, basis_name)
    rotations = np.ascontiguousarray(rotations)
    translations = np.zeros((len(rotations), 3))
    return mol, basis, rotations, translations


def source_fixture(kind='ammonia', basis_name='sto-3g', displacement=0.):
    mol, basis, rotations, translations = molecular_case(kind, basis_name, displacement)
    group = molecular_point_group(mol, rotations, translations, identity_label=kind, budget=BUDGET)
    actions = tuple(build_molecular_space_action(mol, basis, r, t, source=SPACE, target=SPACE,
        budget=BUDGET) for r, t in zip(rotations, translations))
    eri = np.asarray(vq.compute_eri(basis)).reshape((basis.nbasis,)*4)
    source = DenseSource(eri, actions[0].offsets)
    return mol, basis, group, actions, source, rotations, translations


def dense_jk(values, d):
    return (np.einsum('pqrs,sr->pq', values, d, optimize=False),
            np.einsum('prsq,rs->pq', values, d, optimize=False))


@pytest.mark.parametrize('kind', ['water', 'ammonia', 'cartesian'])
@pytest.mark.parametrize('executor', ['numpy', 'native'])
def test_full_source_orbits_and_arbitrary_density_jk(kind, executor):
    _, basis, group, actions, source, _, _ = source_fixture(kind)
    plan = ShellQuartetPlan(group, actions, source, source_tolerance=1e-10, budget=BUDGET)
    ns, n = len(source.shell_offsets)-1, basis.nbasis
    assert plan.qualification_work.tile_requests == ns**4
    assert len(set(source.requests)) == ns**4
    assert plan.qualification_work.shell_calls == 0
    assert sum(len(orbit.members) for orbit in plan.orbits) == ns**4
    assert any(len(orbit.stabilizer) > 1 for orbit in plan.orbits)
    for orbit in plan.orbits:
        assert len(orbit.members)*len(orbit.stabilizer) == 8*group.order
    if kind == 'ammonia':
        assert any(group.product(g, h) != group.product(h, g)
                   for g, h in product(range(group.order), repeat=2))
        pairs = ns*(ns+1)//2
        assert len(plan.orbits) < pairs*(pairs+1)//2
    rng = np.random.default_rng(7932)
    raw = rng.normal(size=(n, n))+1j*rng.normal(size=(n, n))
    for density in (raw, raw+raw.conj().T):
        result = plan.jk(density, budget=BUDGET, executor=executor)
        assert result.executor == executor
        expected = dense_jk(source.values, density)
        for actual, reference in zip((result.j, result.k), expected):
            np.testing.assert_allclose(actual, reference, atol=2e-11, rtol=0)
            assert np.linalg.norm(actual-reference) <= result.error_bound+2e-13
        assert result.work.tile_requests == len(plan.orbits)
        assert result.work.shell_calls == 0
        assert result.scalar_products == 2*n**4
        assert result.block_contractions == 2*ns**4
        for action in actions:
            # Independent dense inverse law, including non-unitary Cartesian
            # coefficient actions. No symmetric-density or final-matrix repair.
            u = action.apply(np.eye(n, dtype=complex))
            moved = np.ascontiguousarray(u@density@u.conj().T)
            transformed = plan.jk(moved, budget=BUDGET, executor=executor)
            for target, original in ((transformed.j, result.j), (transformed.k, result.k)):
                np.testing.assert_allclose(u.conj().T@target@u, original, atol=3e-11, rtol=0)
    if kind == 'cartesian':
        u = actions[1].apply(np.eye(n, dtype=complex))
        assert np.linalg.norm(u.conj().T@u-np.eye(n)) > .1
    with pytest.raises(ValueError):
        result.j.setflags(write=True)
    assert not plan.production_reduction_authorized


def test_permutation_only_and_unreduced_fallback_counts():
    _, basis, group, actions, source, _, _ = source_fixture()
    ns = len(actions[0].destinations)
    counts = {}
    d = np.eye(basis.nbasis, dtype=complex)
    for mode in ('none', 'permutation', 'spatial'):
        plan = ShellQuartetPlan(group, actions, source, source_tolerance=1e-10, budget=BUDGET, mode=mode)
        counts[mode] = plan.jk(d, budget=BUDGET).work.tile_requests
    pairs = ns*(ns+1)//2
    assert counts['none'] == ns**4
    assert counts['permutation'] == pairs*(pairs+1)//2
    assert counts['spatial'] < counts['permutation'] < counts['none']


@pytest.mark.parametrize('executor', ['numpy', 'native'])
def test_audit_refuses_asymmetric_source_even_when_symmetric_density_hides_it(executor):
    _, _, group, actions, source, _, _ = source_fixture()
    # An antisymmetric q/s defect can be hidden by a diagonal trial density.
    source.values[0, 1, 0, 2] += 1e-3
    with pytest.raises(ValueError, match='not equivariant'):
        ShellQuartetPlan(group, actions, source, source_tolerance=1e-10, budget=BUDGET)
    fallback = ShellQuartetPlan(group, actions, source, source_tolerance=0., budget=BUDGET, mode='none')
    n = source.values.shape[0]
    d = np.arange(n*n, dtype=complex).reshape(n, n)*(1+.3j)
    result = fallback.jk(d, budget=BUDGET, executor=executor)
    for actual, expected in zip((result.j, result.k), dense_jk(source.values, d)):
        np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=0)


@pytest.mark.parametrize('executor', ['numpy', 'native'])
def test_source_and_representative_mutations_refuse_without_partial_result(executor):
    _, basis, group, actions, source, _, _ = source_fixture()
    plan = ShellQuartetPlan(group, actions, source, source_tolerance=1e-10, budget=BUDGET)
    d = np.eye(basis.nbasis, dtype=complex)
    source.source_identity = 'changed source'
    source.requests.clear()
    with pytest.raises(ValueError, match='identity changed'):
        plan.jk(d, budget=BUDGET, executor=executor)
    assert not source.requests
    source.source_identity = plan.source_identity
    source.values[0, 0, 0, 0] += 1e-5
    with pytest.raises(ValueError, match='representative changed'):
        plan.jk(d, budget=BUDGET, executor=executor)


def test_exact_budget_and_one_short_refusal_before_source_evaluation():
    _, basis, group, actions, source, _, _ = source_fixture('water')
    plan = ShellQuartetPlan(group, actions, source, source_tolerance=1e-10, budget=BUDGET)
    exact = Budget(plan.required_bytes, plan.required_work)
    repeat = ShellQuartetPlan(group, actions, source, source_tolerance=1e-10, budget=exact)
    repeat.jk(np.eye(basis.nbasis, dtype=complex), budget=exact)
    for cap, error in ((Budget(exact.maximum_bytes-1, exact.maximum_work), MemoryError),
                       (Budget(exact.maximum_bytes, exact.maximum_work-1), ValueError)):
        source.requests.clear()
        with pytest.raises(error):
            ShellQuartetPlan(group, actions, source, source_tolerance=1e-10, budget=cap)
        assert not source.requests
        with pytest.raises(error):
            plan.jk(np.eye(basis.nbasis, dtype=complex), budget=cap)
        assert not source.requests


def test_nonzero_screening_and_mismatched_shell_layout_are_refused():
    _, _, group, actions, source, _, _ = source_fixture('water')
    source.precision = 1e-12
    with pytest.raises(ValueError, match='zero primitive screening'):
        ShellQuartetPlan(group, actions, source, source_tolerance=1e-10, budget=BUDGET)
    source.precision = 0.
    source.shell_offsets = source.shell_offsets.copy()
    source.shell_offsets[1] += 1
    source.requests.clear()
    with pytest.raises(ValueError, match='layouts disagree'):
        ShellQuartetPlan(group, tuple(actions), source, source_tolerance=1e-10, budget=BUDGET)
    assert not source.requests


def test_bad_block_group_law_is_rejected_before_integrals():
    _, _, group, actions, source, _, _ = source_fixture('cartesian')
    bad = actions[1]
    matrices = bad.matrices.copy()
    matrices[-1] += .01
    actions = list(actions)
    actions[1] = BlockSpaceAction(SPACE, SPACE, bad.offsets, bad.destinations,
        matrices, antiunitary=False, budget=BUDGET)
    source.requests.clear()
    with pytest.raises(ValueError, match='group law'):
        ShellQuartetPlan(group, tuple(actions), source, source_tolerance=1e-10, budget=BUDGET)
    assert not source.requests


def test_dense_source_can_reuse_ephemeral_tile_buffer():
    _, _, group, actions, source, _, _ = source_fixture('water')
    class ReusingSource(DenseSource):
        def evaluate(self, key):
            tile = super().evaluate(key)
            size = tile.values.size
            if not hasattr(self, 'scratch'):
                self.scratch = np.empty(max(np.diff(self.shell_offsets))**4)
            self.scratch[:size] = tile.values.ravel()
            return replace(tile, values=self.scratch[:size].reshape(tile.values.shape))
    reuse = ReusingSource(source.values, source.shell_offsets)
    plan = ShellQuartetPlan(group, actions, reuse, source_tolerance=1e-10, budget=BUDGET)
    assert plan.source_residual_norm < 1e-11


@pytest.mark.parametrize('executor', ['numpy', 'native'])
def test_reordered_nonabelian_group_keeps_identity_and_stabilizers(executor):
    _, basis, group, actions, source, _, _ = source_fixture()
    order = [4, 2, 0, 5, 1, 3]
    inverse = np.argsort(order)
    table = np.array([[inverse[group.product(g, h)] for h in order] for g in order], np.int64)
    reordered = FiniteGroup.from_table(table, np.zeros(6, np.uint8), identity=2,
        identity_label='reordered C3v', budget=BUDGET)
    plan = ShellQuartetPlan(reordered, tuple(actions[g] for g in order), source,
        source_tolerance=1e-10, budget=BUDGET)
    d = np.eye(basis.nbasis, dtype=complex)
    expected = dense_jk(source.values, d)
    result = plan.jk(d, budget=BUDGET, executor=executor)
    np.testing.assert_allclose(result.j, expected[0], atol=1e-11, rtol=0)
    np.testing.assert_allclose(result.k, expected[1], atol=1e-11, rtol=0)


@pytest.mark.parametrize('kind', ['antiunitary', 'cocycle', 'complex_actions'])
def test_unsupported_actions_refuse_before_source_calls(kind):
    _, _, _, actions, source, _, _ = source_fixture('water')
    table = np.array([[0, 1], [1, 0]], np.int64)
    anti = np.array([0, kind == 'antiunitary'], np.uint8)
    cocycle = np.zeros((2, 2, 3), np.int64)
    if kind == 'cocycle':
        cocycle[1, 1, 0] = 1
    group = FiniteGroup.from_table(table, anti, identity=0, identity_label=kind, budget=BUDGET,
        rotations=np.tile(np.eye(3, dtype=np.int64), (2, 1, 1)), cocycle=cocycle)
    actions = actions[:2]
    if kind == 'complex_actions':
        a = actions[1]
        values = a.matrices.copy()
        values[0] += 1e-20j
        actions = (actions[0], BlockSpaceAction(SPACE, SPACE, a.offsets, a.destinations,
            values, antiunitary=False, budget=BUDGET))
    with pytest.raises(ValueError, match='antiunitary|periodic/projective|real AO'):
        ShellQuartetPlan(group, actions, source, source_tolerance=1e-10, budget=BUDGET)
    assert not source.requests


def test_displaced_source_is_not_readmitted_with_old_geometry_action():
    mol, basis, group, actions, source, _, _ = source_fixture()
    atoms = list(mol.atoms)
    atoms[0] = vq.Atom(7, [.02, 0., .3])
    displaced = vq.Molecule(atoms)
    changed_basis = vq.BasisSet(displaced, 'sto-3g')
    source.values = np.asarray(vq.compute_eri(changed_basis)).reshape((basis.nbasis,)*4)
    with pytest.raises(ValueError, match='not equivariant'):
        ShellQuartetPlan(group, actions, source, source_tolerance=1e-10, budget=BUDGET)


@pytest.mark.parametrize('executor', ['numpy', 'native'])
def test_accepted_source_residual_bounds_arbitrary_density_reconstruction(executor):
    _, basis, group, actions, source, _, _ = source_fixture()
    source.values[-1, 0, 0, 0] += 5e-12
    plan = ShellQuartetPlan(group, actions, source, source_tolerance=1e-10, budget=BUDGET)
    assert plan.source_residual_norm >= 4.9e-12
    rng = np.random.default_rng(731)
    density = 1000*(rng.normal(size=(basis.nbasis,)*2)+1j*rng.normal(size=(basis.nbasis,)*2))
    result = plan.jk(density, budget=BUDGET, executor=executor)
    for actual, expected in zip((result.j, result.k), dense_jk(source.values, density)):
        assert np.linalg.norm(actual-expected) <= result.error_bound+1e-10


@pytest.mark.parametrize('executor', ['numpy', 'native'])
def test_energy_and_density_directional_derivative_preserve_broken_spin_states(executor):
    mol, basis, group, actions, source, _, _ = source_fixture()
    plan = ShellQuartetPlan(group, actions, source, source_tolerance=1e-10, budget=BUDGET)
    rng = np.random.default_rng(549)
    a, b = [rng.normal(size=(basis.nbasis,)*2)+1j*rng.normal(size=(basis.nbasis,)*2) for _ in range(2)]
    da, db = .03*a@a.conj().T, .03*b@b.conj().T
    total = np.ascontiguousarray(da+db)
    h = np.asarray(vq.compute_kinetic(basis))+np.asarray(vq.compute_nuclear(basis, mol))
    result = plan.jk(total, budget=BUDGET, executor=executor)
    full_j, _ = dense_jk(source.values, total)
    for d in (da, db):
        spin = plan.jk(d, budget=BUDGET, executor=executor)
        _, full_k = dense_jk(source.values, d)
        np.testing.assert_allclose(h+result.j-spin.k, h+full_j-full_k, atol=2e-11, rtol=0)
    def energy(d):
        r = plan.jk(d, budget=BUDGET, executor=executor)
        return np.einsum('pq,qp', d, h+.5*r.j-.25*r.k)
    delta = np.ascontiguousarray(.01*(a+a.conj().T))
    direction = np.einsum('pq,qp', delta, h+result.j-.5*result.k)
    for step in (2e-4, 1e-4):
        measured = (energy(total+step*delta)-energy(total-step*delta))/(2*step)
        np.testing.assert_allclose(measured, direction, atol=2e-8, rtol=0)


@pytest.mark.parametrize('kind', ['water', 'ammonia', 'cartesian'])
@pytest.mark.parametrize('executor', ['numpy', 'native'])
@pytest.mark.parametrize('provider_kind', ['libint', 'manta'])
def test_native_manta_ordered_source_and_representative_integral_counts(kind, executor, provider_kind):
    native = getattr(core, '_manta', None)
    if native is None:
        pytest.skip('MANTA P0 provider is not present in this native build')
    mol, basis, rotations, translations = molecular_case(kind)
    provider = (native.ReferenceJKBuilder if provider_kind == 'libint' else native.DirectJKBuilder)(basis)
    plan = qualify_manta_quartets(mol, basis, rotations, translations, provider=provider,
        source_tolerance=1e-10, budget=BUDGET, schedule=SCHEDULE)
    n, ns = basis.nbasis, basis.nshells
    rng = np.random.default_rng(679)
    d = rng.normal(size=(n, n))+1j*rng.normal(size=(n, n))
    result = plan.jk(d, budget=BUDGET, executor=executor)
    # Build an independent dense ordered-tile reference, without orbit code or
    # real-only AO ERI permutation assumptions in the oracle.
    full = np.empty((n,)*4)
    offsets = plan.offsets
    primitive_count = 0
    for key in product(range(ns), repeat=4):
        tile = provider.evaluate(key)
        full[tuple(slice(offsets[s], offsets[s+1]) for s in key)] = tile['values']
        primitive_count += tile['primitive_quartets_requested']
    for actual, expected in zip((result.j, result.k), dense_jk(full, d)):
        np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=0)
    assert plan.qualification_work.shell_calls == ns**4
    assert plan.qualification_work.primitive_quartets_requested == primitive_count
    assert result.work.shell_calls == len(plan.orbits)
    assert result.work.shell_calls < ns**4
    assert result.work.primitive_quartets_requested < primitive_count
    assert result.scalar_products == 2*n**4


@pytest.mark.parametrize('kind', ['water', 'ammonia', 'cartesian'])
def test_native_md_fixed_mask_covariance_and_shared_fock(kind):
    from vibeqc.symmetry_quartets_screening import QuartetScreening
    from vibeqc.symmetry_quartets_ordered import ordered_quartet_jk, qualified_molecular_source
    from vibeqc.symmetry_molecular_hf import ScreenedMolecularJK, ScreenedMolecularHF
    from vibeqc.solvers.operators import OperatorFrame
    mol,basis,r,t = molecular_case(kind)
    provider = core._manta.DirectJKBuilder(basis)
    # Nonzero diagnostic omission tests whole-orbit masking, not a proposed
    # production accuracy threshold. Every source tile is still audited.
    cap = 2. if kind == 'cartesian' else .3
    options = dict(maximum_omitted_norm=cap) if SCHEDULE == 'streaming' else {}
    plan = qualify_manta_quartets(mol,basis,r,t,provider=provider,
        source_tolerance=1e-10,budget=BUDGET,schedule=SCHEDULE,**options)
    screen = plan.screening if SCHEDULE == 'streaming' else QuartetScreening(
        plan,maximum_omitted_norm=cap,budget=BUDGET)
    assert qualified_molecular_source(screen).evidence == 'native-MD-constructor-and-full-numerical-audit'
    assert plan.qualification_work.shell_calls == basis.nshells**4
    assert screen.omitted_ordered_tiles > 0
    full,masked = masked_source(plan,provider,screen)
    rng = np.random.default_rng(635)
    d = rng.normal(size=(basis.nbasis,basis.nbasis))+1j*rng.normal(size=(basis.nbasis,basis.nbasis))
    batch = d[None].copy()
    reduced = plan.jk_batch(batch,targets=('jk',),screening=screen,executor='native',budget=BUDGET)
    ordered = ordered_quartet_jk(screen,batch,targets=('jk',),budget=BUDGET)
    for i,(actual,reference) in enumerate(zip((reduced.j[0],reduced.k[0]),dense_jk(masked,d))):
        np.testing.assert_allclose(actual,reference,atol=2e-11,rtol=0)
        np.testing.assert_allclose(actual,(ordered.j[0],ordered.k[0])[i],atol=2e-11,rtol=0)
        assert np.linalg.norm(actual-dense_jk(full,d)[i]) <= reduced.error_bounds[0]+2e-11
    assert reduced.work.shell_calls < ordered.work.shell_calls
    for rotation,translation in zip(r,t):
        action = build_molecular_space_action(mol,basis,rotation,translation,
            source=SPACE,target=SPACE,budget=BUDGET)
        moved = plan.jk_batch(action.push_density(d)[None].copy(),targets=('jk',),
            screening=screen,executor='native',budget=BUDGET)
        for got,want in zip((moved.j[0],moved.k[0]),(reduced.j[0],reduced.k[0])):
            np.testing.assert_allclose(action.pull_operator(got),want,atol=3e-11,rtol=0)
    s = np.asarray(vq.compute_overlap(basis))
    values,vectors = np.linalg.eigh(s)
    frame = OperatorFrame(basis_id='native-MD test',n=basis.nbasis,ordering='native AO',
        image_gauge='molecular',overlap=s,X=vectors/np.sqrt(values))
    h = vq.compute_kinetic(basis)+vq.compute_nuclear(basis,mol)
    jk = ScreenedMolecularJK(screen,budget=BUDGET,capture_terminal=True)
    hf = ScreenedMolecularHF(jk,frame=frame,hcore=h,nuclear_energy=mol.nuclear_repulsion())
    assert hf.source.provider == 'screened-molecular-MD-HF-v1'
    ds = tuple(.01*x@x.T for x in rng.normal(size=(2,basis.nbasis,basis.nbasis)))
    # Independent spin densities are not projected to spatial symmetry.
    j = dense_jk(masked,sum(ds))[0]
    cold = hf.evaluate(ds,method='UHF',generation=0)
    for channel,density in zip(cold.build.channels,ds):
        np.testing.assert_array_equal(channel.density,density)
        np.testing.assert_allclose(channel.operator.matrix,h+j-dense_jk(masked,density)[1],atol=2e-11,rtol=0)
    warm = hf.evaluate(ds,method='UHF',generation=0,reuse_evaluated=True)
    assert warm.receipt.work.shell_calls == 0
    assert cold.build.identity == warm.build.identity
    jk.reset_state()
    again = hf.evaluate(ds,method='UHF',generation=0,reuse_evaluated=True)
    assert again.receipt.work.shell_calls > 0 and again.build.identity == cold.build.identity


def test_native_md_source_kernel_bound_and_budget_refusals(monkeypatch):
    from vibeqc.symmetry_quartets_bounds import MolecularCoulombBounds
    from vibeqc.symmetry_quartets_hierarchy import MolecularHierarchicalBounds
    import vibeqc.symmetry_quartets as module
    mol,basis,r,t = molecular_case('water')
    provider = core._manta.DirectJKBuilder(basis)
    def forbidden(*args,**kwargs):
        pytest.fail('invalid source must refuse before tile evaluation')
    monkeypatch.setattr(module._MantaSource,'evaluate',forbidden)
    request = core._manta.KernelRequest();request.kernel='erf';request.omega=.3
    other = core._manta.DirectJKBuilder(basis,request)
    with pytest.raises(ValueError,match='provider or kernel'):
        qualify_manta_quartets(mol,basis,r,t,provider=other,source_tolerance=1e-10,budget=BUDGET)
    with pytest.raises(TypeError,match='immutable native'):
        qualify_manta_quartets(mol,basis,r,t,provider=object(),source_tolerance=1e-10,budget=BUDGET)
    required = manta_quartet_requirements(mol,basis,provider,len(r),budget=BUDGET)
    with pytest.raises(MemoryError):
        qualify_manta_quartets(mol,basis,r,t,provider=provider,source_tolerance=1e-10,
            budget=Budget(required.maximum_bytes-1,required.maximum_work))
    # Native MD cannot inherit libint's pre-audit omission envelopes.
    for bound in (MolecularCoulombBounds,MolecularHierarchicalBounds):
        kwargs = dict(maximum_omitted_norm=1e-12) if bound is MolecularHierarchicalBounds else {}
        with pytest.raises(TypeError,match='P0'):
            bound(basis,provider,budget=BUDGET,**kwargs)
    reference = core._manta.ReferenceJKBuilder(basis)
    bounds = MolecularCoulombBounds(basis,reference,budget=BUDGET)
    with pytest.raises(ValueError,match='different numerical source'):
        qualify_manta_quartets(mol,basis,r,t,provider=provider,source_tolerance=1e-10,budget=BUDGET,
            schedule='streaming',maximum_omitted_norm=1e-12,bounds=bounds)


@pytest.mark.parametrize('mode', ['none', 'permutation', 'spatial'])
def test_compact_stabilizers_with_many_operations_fixing_every_shell(mode):
    # A nonfaithful shell representation is legal. This used to retain one
    # Python tuple per stabilizing (operation, permutation), beyond the cap.
    ng, ns = 24, 2
    table = np.array([[(g+h) % ng for h in range(ng)] for g in range(ng)], np.int64)
    group = FiniteGroup.from_table(table, np.zeros(ng, np.uint8), identity=0,
        identity_label='cyclic group with fixed scalar shells', budget=BUDGET)
    offsets = np.arange(ns+1, dtype=np.int64)
    action = BlockSpaceAction(SPACE, SPACE, offsets, np.arange(ns, dtype=np.int64),
        np.ones(ns, dtype=complex), antiunitary=False, budget=BUDGET)
    source = DenseSource(np.ones((ns,)*4), offsets)
    plan = ShellQuartetPlan(group, (action,)*ng, source, source_tolerance=0.,
        budget=BUDGET, mode=mode)
    operations = range(ng) if mode == 'spatial' else (0,)
    permutations = ((0, 1, 2, 3),) if mode == 'none' else (
        (0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2), (1, 0, 3, 2),
        (2, 3, 0, 1), (3, 2, 0, 1), (2, 3, 1, 0), (3, 2, 1, 0))
    covered = set()
    for orbit in plan.orbits:
        expected = {(g, p) for g in operations for p, axes in enumerate(permutations)
                    if tuple(orbit.representative[a] for a in axes) == orbit.representative}
        assert set(orbit.stabilizer) == expected
        assert len(orbit.stabilizer) == len(expected)
        assert len(orbit.stabilizer.masks) == ng
        assert isinstance(orbit.stabilizer.masks, bytes)
        assert len(orbit.members)*len(expected) == len(operations)*len(permutations)
        covered.update(member.shells for member in orbit.members)
    assert covered == set(product(range(ns), repeat=4))


def test_manta_factory_aggregate_exact_budget_and_pre_snapshot_refusal(monkeypatch):
    native = getattr(core, '_manta', None)
    if native is None:
        pytest.skip('MANTA P0 provider is not present in this native build')
    mol, basis, rotations, translations = molecular_case('cartesian')
    provider = native.ReferenceJKBuilder(basis)
    exact = manta_quartet_requirements(mol, basis, provider, len(rotations), budget=BUDGET, schedule=SCHEDULE)
    plan = qualify_manta_quartets(mol, basis, rotations, translations,
        provider=provider, source_tolerance=1e-10, budget=exact, schedule=SCHEDULE)
    assert exact.maximum_bytes > plan.required_bytes
    plan.jk(np.eye(basis.nbasis, dtype=complex), budget=exact)

    def forbidden(*args, **kwargs):
        pytest.fail('insufficient factory budget reached a snapshot or native engine')
    monkeypatch.setattr(type(basis), 'shells', forbidden)
    monkeypatch.setattr(native.ReferenceJKBuilder, 'shells', property(forbidden))
    monkeypatch.setattr(native.ReferenceJKBuilder, 'session', forbidden)
    monkeypatch.setattr(native.ReferenceJKBuilder, 'evaluate', forbidden)
    monkeypatch.setattr('vibeqc.symmetry_quartets._shell_signature', forbidden)
    monkeypatch.setattr('vibeqc.symmetry_ao.molecular_point_group', forbidden)
    monkeypatch.setattr('vibeqc.symmetry_ao.build_molecular_space_action', forbidden)
    for cap, error in ((Budget(exact.maximum_bytes-1, exact.maximum_work), MemoryError),
                       (Budget(exact.maximum_bytes, exact.maximum_work-1), ValueError)):
        with pytest.raises(error, match='budget exceeded'):
            qualify_manta_quartets(mol, basis, rotations, translations,
                provider=provider, source_tolerance=1e-10, budget=cap, schedule=SCHEDULE)


@pytest.mark.parametrize('provider_kind', ['libint', 'manta'])
def test_manta_factory_rejects_same_size_different_radial_source(provider_kind):
    native = getattr(core, '_manta', None)
    if native is None:
        pytest.skip('MANTA P0 provider is not present in this native build')
    mol, basis, rotations, translations = molecular_case('cartesian')
    changed = vq.BasisSet(mol, [core.ShellInfo(0, l, False, [.9], [1.], [0., 0., 0.])
        for l in (0, 2)], 'changed Cartesian s+d', False)
    provider = native.ReferenceJKBuilder if provider_kind == 'libint' else native.DirectJKBuilder
    with pytest.raises(ValueError, match='snapshots disagree'):
        qualify_manta_quartets(mol, basis, rotations, translations,
            provider=provider(changed), source_tolerance=1e-10, budget=BUDGET, schedule=SCHEDULE)


def permutation_group(degree, copies=1, fixed=0):
    # Independent S_n permutation model with reordered operation IDs.
    rows = list(permutations(range(degree)))
    np.random.default_rng(513).shuffle(rows)
    index = {p: i for i, p in enumerate(rows)}
    table = np.array([[index[tuple(g[h[s]] for s in range(degree))]
                       for h in rows] for g in rows], np.int64)
    group = FiniteGroup.from_table(table, np.zeros(len(rows), np.uint8),
        identity=index[tuple(range(degree))], identity_label='permutation reference', budget=BUDGET)
    destinations = np.array([[copy*degree+p[s] for copy in range(copies) for s in range(degree)]
        + list(range(copies*degree, copies*degree+fixed)) for p in rows], np.int64)
    return group, destinations


@pytest.mark.parametrize('degree,copies,fixed', [(1, 1, 2), (2, 2, 1), (3, 2, 0), (4, 1, 1)])
@pytest.mark.parametrize('mode', ['none', 'permutation', 'spatial'])
def test_streaming_pair_products_against_independent_complete_quartet_partition(degree, copies, fixed, mode):
    group, destinations = permutation_group(degree, copies, fixed)
    schedule = PairQuartetSchedule(group, destinations, budget=BUDGET, mode=mode)
    ns = destinations.shape[1]
    # The oracle enumerates all quartets, then all independent pair swaps.
    # It uses neither pair ranks, H_a orbits, Burnside nor the stored planner.
    expected = {}
    for q in product(range(ns), repeat=4):
        images = set()
        for g in range(group.order) if mode == 'spatial' else (group.identity,):
            a, b, c, d = (int(destinations[g, s]) for s in q)
            for left in ((a, b), (b, a)) if mode != 'none' else ((a, b),):
                for right in ((c, d), (d, c)) if mode != 'none' else ((c, d),):
                    images.add(left+right)
                    if mode != 'none':
                        images.add(right+left)
        expected.setdefault(min(images), set()).add(q)
    observed = {}
    for orbit in schedule:
        assert orbit.representative not in observed
        observed[orbit.representative] = {m.shells for m in orbit.members}
    assert observed == expected
    assert len(schedule) == len(observed)
    assert [o.representative for o in schedule] == list(observed)
    assert schedule.retained_array_bytes < schedule.required_bytes



@pytest.mark.parametrize('mode', ['none', 'permutation', 'spatial'])
def test_streaming_retained_orbits_skip_expansion_and_preserve_order(monkeypatch, mode):
    group, destinations = permutation_group(3, copies=2)
    schedule = PairQuartetSchedule(group, destinations, budget=BUDGET, mode=mode)
    complete = tuple(schedule)
    keep = tuple(i % 3 == 1 for i in range(len(schedule)))
    expected = tuple(orbit for orbit, selected in zip(complete, keep) if selected)
    allowed = {orbit.representative for orbit in expected}
    expand, expanded = PairQuartetSchedule._expand, []
    def checked(self, representative):
        assert representative in allowed, 'omitted orbit was expanded'
        expanded.append(representative)
        return expand(self, representative)
    monkeypatch.setattr(PairQuartetSchedule, '_expand', checked)
    assert tuple(schedule.retained(keep)) == expected
    assert expanded == [orbit.representative for orbit in expected]
    assert tuple(schedule.retained((False,)*len(schedule))) == ()
    assert len(expanded) == len(expected)
    with pytest.raises(ValueError, match='selection length'):
        tuple(schedule.retained(keep[:-1]))


def test_pair_schedule_exact_budget_and_invalid_group_action(monkeypatch):
    group, destinations = permutation_group(3, copies=2)
    schedule = PairQuartetSchedule(group, destinations, budget=BUDGET)
    exact = Budget(schedule.required_bytes, schedule.required_work)
    assert len(PairQuartetSchedule(group, destinations, budget=exact)) == len(schedule)
    bad = destinations.copy()
    bad[(group.identity+1) % group.order, :2] = bad[(group.identity+1) % group.order, 1::-1]
    with pytest.raises(ValueError, match='group law'):
        PairQuartetSchedule(group, bad, budget=BUDGET)
    with pytest.raises(ValueError):
        schedule.pair_maps.setflags(write=True)
    def forbidden(*args, **kwargs):
        pytest.fail('pair-array work occurred before admission')
    monkeypatch.setattr('vibeqc.symmetry_quartets_streaming.np.arange', forbidden)
    for cap, error in ((Budget(exact.maximum_bytes-1, exact.maximum_work), MemoryError),
                       (Budget(exact.maximum_bytes, exact.maximum_work-1), ValueError)):
        with pytest.raises(error, match='budget exceeded'):
            PairQuartetSchedule(group, destinations, budget=cap)


@pytest.mark.parametrize('degree,copies,fixed', [(2, 2, 1), (3, 2, 0), (4, 1, 1)])
@pytest.mark.parametrize('mode', ['none', 'permutation', 'spatial'])
def test_native_orbit_expansion_matches_explicit_images_and_dense_tensors(degree, copies, fixed, mode):
    group, destinations = permutation_group(degree, copies, fixed)
    schedule = PairQuartetSchedule(group, destinations, budget=BUDGET, mode=mode)
    ns = destinations.shape[1]
    rng = np.random.default_rng(638)
    densities = (rng.normal(size=(2,ns,ns))+1j*rng.normal(size=(2,ns,ns))).copy()
    memory = core._plan_symmetry_quartet_accumulator(ns,ns,group.order,ns,1,BUDGET.native(),2,1,1)
    def create():
        return core._SymmetryQuartetAccumulator(np.arange(ns+1,dtype=np.int64),
            np.ones((group.order,ns)),densities,1,Budget(memory.bytes,memory.work).native(),
            np.array([1,2],np.uint8))
    expanded, compiled = create(), create()
    full = np.zeros((ns,)*4)
    for index, orbit in enumerate(schedule):
        representative = np.array(orbit.representative,np.int64)
        tile = np.full((1,)*4,(index+1)/1000.)
        members = np.array([(m.operation,m.permutation,*m.shells) for m in orbit.members],np.int64)
        expanded.accumulate(representative,tile,members)
        compiled.accumulate_orbit(representative,tile,
            np.ascontiguousarray(destinations[:,orbit.representative]),group.identity,
            {'none':0,'permutation':1,'spatial':2}[mode])
        for member in orbit.members:
            full[member.shells] = tile.item()
    a,b = expanded.finish(),compiled.finish()
    np.testing.assert_array_equal(a[0],b[0]); np.testing.assert_array_equal(a[1],b[1])
    assert a[2] == b[2]
    np.testing.assert_allclose(b[0],dense_jk(full,densities[0])[0],atol=2e-11,rtol=0)
    np.testing.assert_allclose(b[1],dense_jk(full,densities[1])[1],atol=2e-11,rtol=0)


@pytest.mark.parametrize('failure', ['dtype','shape','row-count','stride','identity','identity-type',
    'mode','mode-type','mapped-index','identity-action','nonfinite-tile'])
def test_native_orbit_descriptor_failure_poisons_partial_targets(failure):
    accumulator = core._SymmetryQuartetAccumulator(np.array([0,1,2],np.int64),
        np.ones((2,2)),np.ones((2,2),complex),1,BUDGET.native())
    representative = np.zeros(4,np.int64)
    tile = np.ones((1,)*4)
    mapped = np.array([[0,0,0,0],[1,1,1,1]],np.int64)
    accumulator.accumulate_orbit(representative,tile,mapped,0,2)
    identity,mode = 0,2
    if failure == 'dtype': mapped = mapped.astype(float)
    if failure == 'shape': mapped = mapped.ravel()
    if failure == 'row-count': mapped = mapped[:1]
    if failure == 'stride': mapped = mapped[:,::-1]
    if failure == 'identity': identity = 2
    if failure == 'identity-type': identity = True
    if failure == 'mode': mode = 3
    if failure == 'mode-type': mode = 'spatial'
    if failure == 'mapped-index': mapped[1,0] = 2
    if failure == 'identity-action': mapped[0,0] = 1
    if failure == 'nonfinite-tile': tile[...] = np.nan
    with pytest.raises((ValueError,TypeError,RuntimeError)):
        accumulator.accumulate_orbit(representative,tile,mapped,identity,mode)
    with pytest.raises(ValueError,match='invalid|finished'):
        accumulator.finish()


def test_native_streaming_uses_compiled_images_with_checked_fallback(monkeypatch):
    mol,basis,r,t = molecular_case('ammonia')
    plan = qualify_manta_quartets(mol,basis,r,t,provider=core._manta.DirectJKBuilder(basis),
        source_tolerance=1e-10,budget=BUDGET,schedule='streaming',maximum_omitted_norm=.3)
    density = np.random.default_rng(635).normal(size=(1,basis.nbasis,basis.nbasis)).astype(complex)
    from vibeqc.symmetry_quartets_batch import quartet_batch_requirements
    exact = quartet_batch_requirements(plan,1,budget=BUDGET,executor='native',targets=('jk',),screening=plan.screening)
    expected = plan.jk_batch(density,targets=('jk',),screening=plan.screening,executor='numpy',budget=BUDGET)
    original = PairQuartetSchedule._expand
    def forbidden(*args,**kwargs):
        pytest.fail('compiled execution expanded Python image objects')
    assert hasattr(core._SymmetryQuartetAccumulator,'accumulate_orbit')
    monkeypatch.setattr(PairQuartetSchedule,'_expand',forbidden)
    actual = plan.jk_batch(density,targets=('jk',),screening=plan.screening,executor='native',budget=exact)
    np.testing.assert_allclose(actual.j,expected.j,atol=2e-11,rtol=0)
    np.testing.assert_allclose(actual.k,expected.k,atol=2e-11,rtol=0)
    monkeypatch.setattr(PairQuartetSchedule,'_expand',original)
    monkeypatch.setattr(core._SymmetryQuartetAccumulator,'accumulate_orbit',None)
    fallback = plan.jk_batch(density,targets=('jk',),screening=plan.screening,executor='native',budget=exact)
    np.testing.assert_array_equal(actual.j,fallback.j); np.testing.assert_array_equal(actual.k,fallback.k)
    assert actual.work == fallback.work and actual.transport_scalar_products == fallback.transport_scalar_products


def test_streamed_plan_preserves_stored_work_and_reconstruction():
    from vibeqc.symmetry_quartets import ShellQuartetPlan as StoredPlan
    _, basis, group, actions, source, _, _ = source_fixture('ammonia', 'cc-pvdz')
    stored = StoredPlan(group, actions, source, source_tolerance=1e-10, budget=BUDGET)
    streaming = StreamingShellQuartetPlan(group, actions, source, source_tolerance=1e-10, budget=BUDGET)
    assert streaming.required_bytes < stored.required_bytes
    assert streaming.qualification_work == stored.qualification_work
    assert len(streaming.representative_digest) == 64
    assert len(streaming.orbits) == len(stored.orbits)
    for a, b in zip(streaming.orbits, stored.orbits):
        assert a.representative == b.representative
        assert a.members == b.members
        assert set(a.stabilizer) == set(b.stabilizer)
    rng = np.random.default_rng(712)
    d = rng.normal(size=(basis.nbasis,)*2)+1j*rng.normal(size=(basis.nbasis,)*2)
    a, b = stored.jk(d, budget=BUDGET), streaming.jk(d, budget=BUDGET)
    assert a.work == b.work
    assert (a.scalar_products, a.block_contractions, a.transport_scalar_products) == (
        b.scalar_products, b.block_contractions, b.transport_scalar_products)
    np.testing.assert_array_equal(a.j, b.j)
    np.testing.assert_array_equal(a.k, b.k)


def test_large_pair_inventory_scales_quadratically_without_quartet_enumeration(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('schedule construction enumerated quartet products')
    monkeypatch.setattr(PairQuartetSchedule, '_expand', forbidden)
    payloads = []
    for copies in (20, 40):
        group, destinations = permutation_group(3, copies=copies)
        schedule = PairQuartetSchedule(group, destinations, budget=BUDGET)
        ns = 3*copies
        assert schedule.pair_maps.shape == (6, ns*(ns+1)//2)
        assert len(schedule) > 100_000
        assert schedule.retained_array_bytes < 1024*ns*ns
        payloads.append(schedule.retained_array_bytes)
    assert 3.8 < payloads[1]/payloads[0] < 4.1


def test_native_aggregate_budget_refuses_before_packing_or_source_calls(monkeypatch):
    from vibeqc.symmetry_quartets_native import native_quartet_requirements
    _, basis, group, actions, source, _, _ = source_fixture('water')
    plan = ShellQuartetPlan(group, actions, source, source_tolerance=1e-10, budget=BUDGET)
    exact = native_quartet_requirements(plan, budget=BUDGET)
    assert exact.maximum_bytes > plan.required_bytes
    d = np.eye(basis.nbasis, dtype=complex)
    native = plan.jk(d, budget=exact, executor='native')
    numpy = plan.jk(d, budget=BUDGET)
    np.testing.assert_allclose(native.j, numpy.j, atol=2e-12, rtol=0)
    assert (native.scalar_products, native.block_contractions, native.transport_scalar_products) == (
        numpy.scalar_products, numpy.block_contractions, numpy.transport_scalar_products)
    def forbidden(*args, **kwargs):
        pytest.fail('insufficient aggregate budget reached matrix packing or execution')
    monkeypatch.setattr('vibeqc.symmetry_quartets_batch.np.stack', forbidden)
    monkeypatch.setattr(core, '_SymmetryQuartetAccumulator', forbidden)
    for cap, error in ((Budget(exact.maximum_bytes-1, exact.maximum_work), MemoryError),
                       (Budget(exact.maximum_bytes, exact.maximum_work-1), ValueError)):
        source.requests.clear()
        with pytest.raises(error, match='budget exceeded'):
            plan.jk(d, budget=cap, executor='native')
        assert not source.requests


def test_native_four_leg_transport_permutations_and_owned_density():
    # Algebra oracle independent of group/source qualification and the
    # Python transport helper; nonorthogonal matrices and unequal axes.
    offsets = np.array([0, 1, 3, 6], np.int64)
    dimensions = np.diff(offsets)
    blocks = [np.array([[1.2]]), np.array([[.3, .8], [.5, 1.2]]),
              np.array([[.9, .1, -.2], [.3, 1.1, .4], [-.1, .2, .8]])]
    packed = np.stack([np.concatenate([np.eye(n).ravel() for n in dimensions]),
                       np.concatenate([m.ravel() for m in blocks])])
    rng = np.random.default_rng(8731)
    d = rng.normal(size=(6, 6))+1j*rng.normal(size=(6, 6))
    original_density = d.copy()
    budget = core._plan_symmetry_quartet_accumulator(6, 3, 2, 14, 3, BUDGET.native())
    accumulator = core._SymmetryQuartetAccumulator(offsets, packed, d, 3,
        Budget(budget.bytes, budget.work).native())
    d[:] = 0
    packed[:] = 0
    rep = np.array([0, 1, 2, 1], np.int64)
    tile = rng.normal(size=(1, 2, 3, 2))
    axes = [(0,1,2,3),(1,0,2,3),(0,1,3,2),(1,0,3,2),
            (2,3,0,1),(3,2,0,1),(2,3,1,0),(3,2,1,0)]
    members = np.array([(g, p, *(rep[list(order)])) for g in range(2)
                        for p, order in enumerate(axes)], np.int64)
    expected_j, expected_k = np.zeros((6, 6), complex), np.zeros((6, 6), complex)
    rotated = np.einsum('ap,bq,cr,ds,abcd->pqrs', *(blocks[s] for s in rep), tile, optimize=True)
    for g, p, *shells in members:
        value = (tile if g == 0 else rotated).transpose(axes[p])
        a, b, c, e = [slice(offsets[s], offsets[s+1]) for s in shells]
        expected_j[a, b] += np.einsum('pqrs,sr->pq', value, original_density[e, c])
        expected_k[a, e] += np.einsum('pqrs,qr->ps', value, original_density[b, c])
    accumulator.accumulate(rep, tile, members)
    tile[:] = 0
    j, k, counts = accumulator.finish()
    np.testing.assert_allclose(j, expected_j, atol=2e-13, rtol=0)
    np.testing.assert_allclose(k, expected_k, atol=2e-13, rtol=0)
    assert counts == dict(representatives=1, scalar_products=384,
                         block_contractions=32, transport_scalar_products=96)
    with pytest.raises(ValueError, match='finished'):
        accumulator.finish()


@pytest.mark.parametrize('failure', ['permutation', 'shell', 'overflow', 'nonfinite'])
def test_native_accumulator_failures_poison_partial_targets(failure):
    offsets = np.array([0, 1], np.int64)
    accumulator = core._SymmetryQuartetAccumulator(offsets, np.ones((1,1)),
        np.full((1,1), 2+1j, complex), 1, BUDGET.native())
    key, tile, members = np.zeros(4, np.int64), np.ones((1,)*4), np.zeros((1,6), np.int64)
    if failure == 'permutation':
        members[0,1] = 8
    elif failure == 'shell':
        members[0,2] = -1
    elif failure == 'overflow':
        tile.fill(1e308)
    else:
        tile.fill(np.nan)
    with pytest.raises(ValueError):
        accumulator.accumulate(key, tile, members)
    with pytest.raises(ValueError, match='failed'):
        accumulator.finish()


def test_native_count_plan_exact_caps_overflow_and_strict_arrays():
    p = core._plan_symmetry_quartet_accumulator(2, 2, 1, 2, 1, BUDGET.native())
    offsets, matrices, d = np.arange(3, dtype=np.int64), np.ones((1,2)), np.eye(2, dtype=complex)
    exact = Budget(p.bytes, p.work)
    core._SymmetryQuartetAccumulator(offsets, matrices, d, 1, exact.native())
    for cap in (Budget(p.bytes-1, p.work), Budget(p.bytes, p.work-1)):
        with pytest.raises(ValueError, match='budget exceeded'):
            core._SymmetryQuartetAccumulator(offsets, matrices, d, 1, cap.native())
    with pytest.raises(OverflowError):
        core._plan_symmetry_quartet_accumulator(2**32, 2**32, 1, 2**32, 1, BUDGET.native())
    with pytest.raises(ValueError, match='exact aligned'):
        core._SymmetryQuartetAccumulator(offsets.astype(np.int32), matrices, d, 1, exact.native())
    with pytest.raises(ValueError, match='exact aligned'):
        core._SymmetryQuartetAccumulator(offsets, matrices, d.real.copy(), 1, exact.native())
    with pytest.raises(ValueError, match='exact aligned'):
        core._SymmetryQuartetAccumulator(offsets, matrices, np.zeros((2,4), complex)[:,::2], 1, exact.native())


@pytest.mark.parametrize('failure', ['dtype', 'layout', 'rank', 'alignment', 'nonarray'])
def test_binding_rejection_poison_after_successful_partial_accumulation(failure):
    accumulator = core._SymmetryQuartetAccumulator(np.arange(3, dtype=np.int64),
        np.ones((1,2)), np.eye(2, dtype=complex), 1, BUDGET.native())
    key, tile, members = np.zeros(4, np.int64), np.ones((1,)*4), np.zeros((1,6), np.int64)
    accumulator.accumulate(key, tile, members)
    if failure == 'dtype':
        bad = tile.astype(np.float32)
    elif failure == 'layout':
        bad = np.ones((2,2,2,4))[:, :, :, ::2]
    elif failure == 'rank':
        bad = np.ones((1,1,1))
    elif failure == 'alignment':
        bad = np.ndarray((1,)*4, dtype=np.float64, buffer=np.zeros(9, np.uint8), offset=1)
    else:
        bad = tile.tolist()
    with pytest.raises((ValueError, TypeError)):
        accumulator.accumulate(key, bad, members)
    with pytest.raises(ValueError, match='failed'):
        accumulator.finish()
    with pytest.raises(ValueError, match='failed'):
        accumulator.accumulate(key, tile, members)


@pytest.mark.parametrize('executor', ['numpy', 'native'])
@pytest.mark.parametrize('kind', ['water', 'ammonia', 'cartesian'])
@pytest.mark.parametrize('mode', ['none', 'permutation', 'spatial'])
def test_batch_independent_complex_channels_covariance_and_shared_work(executor, kind, mode):
    _, basis, group, actions, source, _, _ = source_fixture(kind)
    plan = ShellQuartetPlan(group, actions, source, source_tolerance=1e-10, budget=BUDGET, mode=mode)
    n = basis.nbasis
    rng = np.random.default_rng(968)
    raw = rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
    densities = np.stack([raw, raw+raw.conj().T, np.zeros_like(raw)])
    source.requests.clear()
    batch = plan.jk_batch(densities, budget=BUDGET, executor=executor)
    assert len(source.requests) == len(plan.orbits)
    assert batch.j.shape == batch.k.shape == densities.shape
    assert batch.source_identity == plan.source_identity and batch.executor == executor
    separate = [plan.jk(d, budget=BUDGET, executor=executor) for d in densities]
    assert batch.work == separate[0].work
    assert batch.scalar_products == sum(r.scalar_products for r in separate)
    assert batch.block_contractions == sum(r.block_contractions for r in separate)
    assert batch.transport_scalar_products == separate[0].transport_scalar_products
    for channel, density in enumerate(densities):
        expected = dense_jk(source.values, density)
        for target, reference in zip((batch.j[channel], batch.k[channel]), expected):
            np.testing.assert_allclose(target, reference, atol=2e-11, rtol=0)
        np.testing.assert_allclose(batch.j[channel], separate[channel].j, atol=2e-11, rtol=0)
        assert batch.error_bounds[channel] == separate[channel].error_bound
    if mode == 'spatial':
        for action in actions:
            u = action.apply(np.eye(n, dtype=complex))
            moved = np.ascontiguousarray(u@densities@u.conj().T)
            transformed = plan.jk_batch(moved, budget=BUDGET, executor=executor)
            for target, original in ((transformed.j,batch.j), (transformed.k,batch.k)):
                np.testing.assert_allclose(u.conj().T@target@u, original, atol=3e-11, rtol=0)
    for a in (batch.j, batch.k, batch.error_bounds):
        with pytest.raises(ValueError):
            a.flags.writeable = True


@pytest.mark.parametrize('executor', ['numpy', 'native'])
def test_batch_exact_resources_and_early_refusals(monkeypatch, executor):
    from vibeqc.symmetry_quartets_batch import quartet_batch_requirements
    _, basis, group, actions, source, _, _ = source_fixture('water')
    plan = ShellQuartetPlan(group, actions, source, source_tolerance=1e-10, budget=BUDGET)
    d = np.stack([np.eye(basis.nbasis, dtype=complex)]*2)
    cap = quartet_batch_requirements(plan, 2, budget=BUDGET, executor=executor)
    result = plan.jk_batch(d, budget=cap, executor=executor)
    assert result.j.shape == d.shape
    smaller = quartet_batch_requirements(plan, 1, budget=BUDGET, executor=executor)
    assert cap.maximum_bytes > smaller.maximum_bytes and cap.maximum_work > smaller.maximum_work
    def forbidden(*args, **kwargs):
        pytest.fail('rejected batch reached allocation or source evaluation')
    monkeypatch.setattr(source, 'evaluate', forbidden)
    monkeypatch.setattr('vibeqc.symmetry_quartets_batch._freeze', forbidden)
    monkeypatch.setattr('vibeqc.symmetry_quartets_batch.np.stack', forbidden)
    monkeypatch.setattr(core, '_SymmetryQuartetAccumulator', forbidden)
    for insufficient in (Budget(cap.maximum_bytes-1, cap.maximum_work),
                         Budget(cap.maximum_bytes, cap.maximum_work-1)):
        with pytest.raises((ValueError, MemoryError), match='budget exceeded'):
            plan.jk_batch(d, budget=insufficient, executor=executor)
    for bad in (d.real.copy(), d[0], d[:, :, ::2], d[:0], np.full_like(d, np.nan),
                np.zeros((2,1,1), complex)):
        with pytest.raises((TypeError, ValueError)):
            plan.jk_batch(bad, budget=BUDGET, executor=executor)
    for count in (0, -1, True, 2**63):
        with pytest.raises((TypeError, ValueError)):
            quartet_batch_requirements(plan, count, budget=BUDGET, executor=executor)
    with pytest.raises((ValueError, MemoryError, OverflowError)):
        quartet_batch_requirements(plan, 2**62, budget=BUDGET, executor=executor)
    with pytest.raises(ValueError, match='unknown quartet executor'):
        plan.jk_batch(d, budget=BUDGET, executor='invalid')


@pytest.mark.parametrize('executor', ['numpy', 'native'])
def test_batch_snapshots_density_before_callbacks_and_checks_final_digest(monkeypatch, executor):
    _, basis, group, actions, source, _, _ = source_fixture('water')
    plan = ShellQuartetPlan(group, actions, source, source_tolerance=1e-10, budget=BUDGET)
    n = basis.nbasis
    d = np.stack([np.eye(n, dtype=complex), np.ones((n,n), complex)*(1+.2j)])
    expected = [dense_jk(source.values, channel) for channel in d]
    evaluate = source.evaluate
    def mutate(key):
        d[:] = 0
        return evaluate(key)
    monkeypatch.setattr(source, 'evaluate', mutate)
    batch = plan.jk_batch(d, budget=BUDGET, executor=executor)
    for channel, (j,k) in enumerate(expected):
        np.testing.assert_allclose(batch.j[channel], j, atol=2e-11, rtol=0)
        np.testing.assert_allclose(batch.k[channel], k, atol=2e-11, rtol=0)
    # A late representative change refuses the entire batch, including
    # channels whose accumulated values happen to be zero.
    last = tuple(plan.orbits)[-1].representative
    def corrupt_last(key):
        receipt = evaluate(key)
        if key == last:
            receipt.values.flat[0] += 1e-3
        return receipt
    monkeypatch.setattr(source, 'evaluate', corrupt_last)
    with pytest.raises(ValueError, match='representative changed'):
        plan.jk_batch(d, budget=BUDGET, executor=executor)
    monkeypatch.setattr(source, 'evaluate', evaluate)
    source.source_identity = 'different source'
    with pytest.raises(ValueError, match='identity changed'):
        plan.jk_batch(d, budget=BUDGET, executor=executor)


@pytest.mark.parametrize('executor', ['numpy', 'native'])
def test_batch_uhf_energy_response_and_per_channel_error_bounds(executor):
    mol, basis, group, actions, source, _, _ = source_fixture()
    # Distinct norms require distinct accepted-source error bounds.
    source.values[-1, 0, 0, 0] += 5e-12
    plan = ShellQuartetPlan(group, actions, source, source_tolerance=1e-10, budget=BUDGET)
    rng = np.random.default_rng(5984)
    raw = rng.normal(size=(2,basis.nbasis,basis.nbasis))+1j*rng.normal(size=(2,basis.nbasis,basis.nbasis))
    d = np.ascontiguousarray(.01*raw@raw.conj().transpose(0,2,1))
    h = np.asarray(vq.compute_kinetic(basis))+np.asarray(vq.compute_nuclear(basis,mol))
    result = plan.jk_batch(d, budget=BUDGET, executor=executor)
    refs = [dense_jk(source.values, channel) for channel in d]
    j_total = result.j.sum(axis=0)
    fock = h+j_total-result.k
    exact_j = refs[0][0]+refs[1][0]
    for i in range(2):
        np.testing.assert_allclose(fock[i], h+exact_j-refs[i][1], atol=2e-11, rtol=0)
        assert abs(result.error_bounds[i]-plan.source_residual_norm*np.linalg.norm(d[i])) < 1e-25
        assert np.linalg.norm(result.j[i]-refs[i][0]) <= result.error_bounds[i]+1e-12
    def energy(densities):
        r = plan.jk_batch(densities, budget=BUDGET, executor=executor)
        return sum(np.einsum('pq,qp', densities[i], h+.5*r.j.sum(axis=0)-.5*r.k[i]) for i in range(2))
    reference_energy = sum(np.einsum('pq,qp', d[i], h+.5*exact_j-.5*refs[i][1]) for i in range(2))
    assert abs(energy(d)-reference_energy) < 2e-11
    delta = np.ascontiguousarray(.03*(raw+raw.conj().transpose(0,2,1)))
    for channel in range(2):
        direction = np.zeros_like(d)
        direction[channel] = delta[channel]
        step = 1e-5
        finite_difference = (energy(d+step*direction)-energy(d-step*direction))/(2*step)
        expected = np.einsum('pq,qp', direction[channel], fock[channel])
        assert abs(finite_difference-expected) < 2e-9
    response = plan.jk_batch(delta, budget=BUDGET, executor=executor)
    plus = plan.jk_batch(d+1e-5*delta, budget=BUDGET, executor=executor)
    for target, base, derivative in ((plus.j,result.j,response.j), (plus.k,result.k,response.k)):
        np.testing.assert_allclose((target-base)/1e-5, derivative, atol=2e-10, rtol=0)


def test_native_batch_exact_budget_owned_channels_and_overflow_poison():
    from vibeqc.symmetry_shared import Budget
    offsets, matrices = np.arange(3, dtype=np.int64), np.ones((1,2))
    density = np.array([np.eye(2), np.ones((2,2))*(2+3j)], complex)
    plan = core._plan_symmetry_quartet_accumulator(2, 2, 1, 2, 1, BUDGET.native(), 2)
    exact = Budget(plan.bytes, plan.work)
    for cap in (Budget(plan.bytes-1, plan.work), Budget(plan.bytes, plan.work-1)):
        with pytest.raises(ValueError, match='budget exceeded'):
            core._SymmetryQuartetAccumulator(offsets, matrices, density, 1, cap.native())
    accumulator = core._SymmetryQuartetAccumulator(offsets, matrices, density, 1, exact.native())
    density[:] = 0
    key, tile, members = np.zeros(4,np.int64), np.ones((1,)*4), np.zeros((1,6),np.int64)
    accumulator.accumulate(key, tile, members)
    j,k,counts = accumulator.finish()
    assert j.shape == k.shape == (2,2,2)
    assert j[0,0,0] == k[0,0,0] == 1
    assert j[1,0,0] == k[1,0,0] == 2+3j
    assert counts['scalar_products'] == counts['block_contractions'] == 4
    for bad in (np.empty((0,2,2), complex), np.ones((2,2,1), complex), density.real.copy()):
        with pytest.raises(ValueError):
            core._SymmetryQuartetAccumulator(offsets, matrices, bad, 1, BUDGET.native())
    with pytest.raises(OverflowError):
        core._plan_symmetry_quartet_accumulator(2, 2, 1, 2, 1, BUDGET.native(), 2**63)
    density[0] = 1
    density[1] = 2+3j
    accumulator = core._SymmetryQuartetAccumulator(offsets, matrices, density, 1, exact.native())
    accumulator.accumulate(key, tile, members)
    tile[:] = 1e308  # First channel is finite; second channel overflows.
    with pytest.raises(ValueError, match='nonfinite'):
        accumulator.accumulate(key, tile, members)
    with pytest.raises(ValueError, match='failed'):
        accumulator.finish()


@pytest.mark.parametrize('executor', ['numpy', 'native'])
def test_batch_manta_receipts_integrals_and_singleton_shape(executor):
    from vibeqc.manta import ReferenceJKBuilder
    mol, basis, rotations, translations = molecular_case()
    provider = ReferenceJKBuilder(basis)
    plan = qualify_manta_quartets(mol, basis, rotations, translations, provider=provider,
        source_tolerance=1e-10, budget=BUDGET, schedule=SCHEDULE)
    rng = np.random.default_rng(679)
    d = rng.normal(size=(2,basis.nbasis,basis.nbasis))+1j*rng.normal(size=(2,basis.nbasis,basis.nbasis))
    batch = plan.jk_batch(d, budget=BUDGET, executor=executor)
    scalar = plan.jk(d[0], budget=BUDGET, executor=executor)
    singleton = plan.jk_batch(d[:1], budget=BUDGET, executor=executor)
    assert singleton.j.shape == singleton.k.shape == d[:1].shape
    assert singleton.error_bounds.shape == (1,)
    assert batch.work == scalar.work == singleton.work
    assert batch.work.shell_calls == len(plan.orbits) < basis.nshells**4
    assert batch.work.primitive_quartets_requested > 0
    assert batch.transport_scalar_products == scalar.transport_scalar_products
    assert batch.scalar_products == 2*scalar.scalar_products
    session, full = provider.session(), np.empty((basis.nbasis,)*4)
    for key in product(range(basis.nshells), repeat=4):
        receipt = session.evaluate(key)
        full[tuple(slice(plan.offsets[s], plan.offsets[s+1]) for s in key)] = receipt['values']
    for i, density in enumerate(d):
        for target, expected in zip((batch.j[i],batch.k[i]), dense_jk(full,density)):
            np.testing.assert_allclose(target, expected, atol=2e-11, rtol=0)


@pytest.mark.parametrize('executor', ['numpy', 'native'])
@pytest.mark.parametrize('kind', ['water', 'ammonia', 'cartesian'])
def test_selected_targets_preserve_channel_mapping_covariance_and_dense_values(executor, kind):
    _, basis, group, actions, source, _, _ = source_fixture(kind)
    plan = ShellQuartetPlan(group, actions, source, source_tolerance=1e-10, budget=BUDGET)
    rng = np.random.default_rng(3128)
    d = rng.normal(size=(3,basis.nbasis,basis.nbasis))+1j*rng.normal(size=(3,basis.nbasis,basis.nbasis))
    targets = ('k', 'j', 'jk')
    result = plan.jk_batch(d, budget=BUDGET, executor=executor, targets=targets)
    full = plan.jk_batch(d, budget=BUDGET, executor=executor)
    assert result.j_channels == (1,2) and result.k_channels == (0,2)
    assert result.work == full.work
    assert result.transport_scalar_products == full.transport_scalar_products
    assert result.scalar_products*3 == full.scalar_products*2
    assert result.block_contractions*3 == full.block_contractions*2
    assert result.j.nbytes+result.k.nbytes == 4*basis.nbasis**2*16
    for values, channels, which in ((result.j,result.j_channels,0), (result.k,result.k_channels,1)):
        for output, channel in enumerate(channels):
            np.testing.assert_allclose(values[output], dense_jk(source.values,d[channel])[which], atol=2e-11, rtol=0)
    np.testing.assert_array_equal(result.error_bounds, full.error_bounds)
    for action in actions:
        u = action.apply(np.eye(basis.nbasis,dtype=complex))
        moved = plan.jk_batch(np.ascontiguousarray(u@d@u.conj().T), budget=BUDGET,
            executor=executor, targets=targets)
        for value, reference in ((moved.j,result.j), (moved.k,result.k)):
            np.testing.assert_allclose(u.conj().T@value@u, reference, atol=3e-11, rtol=0)


@pytest.mark.parametrize('executor', ['numpy', 'native'])
@pytest.mark.parametrize('target', ['j', 'k'])
def test_only_requested_outputs_allocated_and_contracted(monkeypatch, executor, target):
    from vibeqc.symmetry_quartets_batch import quartet_batch_requirements
    _, basis, group, actions, source, _, _ = source_fixture('water')
    plan = ShellQuartetPlan(group, actions, source, source_tolerance=1e-10, budget=BUDGET)
    d = np.ones((2,basis.nbasis,basis.nbasis), complex)*(1+.5j)
    expected = dense_jk(source.values,d[0])[0 if target=='j' else 1]
    cap = quartet_batch_requirements(plan, 2, budget=BUDGET, executor=executor, targets=(target,)*2)
    full = quartet_batch_requirements(plan, 2, budget=BUDGET, executor=executor)
    assert cap.maximum_bytes == full.maximum_bytes-32*2*basis.nbasis**2
    assert cap.maximum_work < full.maximum_work
    einsum = np.einsum
    def checked(signature, *args, **kwargs):
        assert signature == ('pqrs,bsr->bpq' if target=='j' else 'pqrs,bqr->bps')
        return einsum(signature, *args, **kwargs)
    monkeypatch.setattr('vibeqc.symmetry_quartets_batch.np.einsum', checked)
    result = plan.jk_batch(d, budget=cap, executor=executor, targets=(target,)*2)
    present, absent = (result.j,result.k) if target=='j' else (result.k,result.j)
    assert absent.shape == (0,basis.nbasis,basis.nbasis) and absent.nbytes == 0
    assert result.scalar_products == 2*basis.nbasis**4
    for array in (present, absent):
        with pytest.raises(ValueError):
            array.flags.writeable = True
    for value in present:
        np.testing.assert_allclose(value, expected, atol=2e-11, rtol=0)


@pytest.mark.parametrize('executor', ['numpy', 'native'])
def test_selected_targets_exact_budgets_invalid_descriptors_and_late_source_failure(monkeypatch, executor):
    from vibeqc.symmetry_quartets_batch import quartet_batch_requirements
    _, basis, group, actions, source, _, _ = source_fixture('water')
    plan = ShellQuartetPlan(group, actions, source, source_tolerance=1e-10, budget=BUDGET)
    d = np.ones((3,basis.nbasis,basis.nbasis), complex)
    targets = ('j','k','k')
    cap = quartet_batch_requirements(plan, 3, budget=BUDGET, executor=executor, targets=targets)
    plan.jk_batch(d, budget=cap, executor=executor, targets=targets)
    def forbidden(*args, **kwargs):
        pytest.fail('invalid selection/cap reached source evaluation')
    evaluate = source.evaluate
    monkeypatch.setattr(source, 'evaluate', forbidden)
    with monkeypatch.context() as guards:
        guards.setattr(core, '_SymmetryQuartetAccumulator', forbidden)
        guards.setattr('vibeqc.symmetry_quartets_batch.np.stack', forbidden)
        guards.setattr('vibeqc.symmetry_quartets_batch._freeze', forbidden)
        for short in (Budget(cap.maximum_bytes-1,cap.maximum_work), Budget(cap.maximum_bytes,cap.maximum_work-1)):
            with pytest.raises((MemoryError, ValueError), match='budget exceeded'):
                plan.jk_batch(d, budget=short, executor=executor, targets=targets)
    for bad in ('j', ['j','k','k'], ('j',), ('j','k',''), ('j','k','kj'), ('j','k',1)):
        with pytest.raises((TypeError, ValueError)):
            plan.jk_batch(d, budget=BUDGET, executor=executor, targets=bad)
    final = tuple(plan.orbits)[-1].representative
    def mutate_final(key):
        receipt = evaluate(key)
        if key == final:
            receipt.values.flat[0] += 1e-3
        return receipt
    monkeypatch.setattr(source, 'evaluate', mutate_final)
    with pytest.raises(ValueError, match='representative changed'):
        plan.jk_batch(d, budget=cap, executor=executor, targets=targets)


@pytest.mark.parametrize('executor', ['numpy', 'native'])
def test_selected_manta_spin_targets_energy_and_density_response(executor):
    from vibeqc.manta import ReferenceJKBuilder
    mol, basis, rotations, translations = molecular_case()
    plan = qualify_manta_quartets(mol, basis, rotations, translations,
        provider=ReferenceJKBuilder(basis), source_tolerance=1e-10, budget=BUDGET, schedule=SCHEDULE)
    n = basis.nbasis
    rng = np.random.default_rng(888)
    raw = rng.normal(size=(2,n,n))+1j*rng.normal(size=(2,n,n))
    spins = np.ascontiguousarray(.01*raw@raw.conj().transpose(0,2,1))
    h = np.asarray(vq.compute_kinetic(basis))+np.asarray(vq.compute_nuclear(basis,mol))
    eri = np.asarray(vq.compute_eri(basis)).reshape((n,)*4)
    def selected(d):
        return plan.jk_batch(np.stack([d[0]+d[1],d[0],d[1]]), budget=BUDGET,
            executor=executor, targets=('j','k','k'))
    result = selected(spins)
    baseline = plan.jk_batch(spins, budget=BUDGET, executor=executor)
    assert result.j_channels == (0,) and result.k_channels == (1,2)
    assert result.work == baseline.work
    assert result.transport_scalar_products == baseline.transport_scalar_products
    assert result.scalar_products*4 == baseline.scalar_products*3
    assert result.block_contractions*4 == baseline.block_contractions*3
    expected_j = dense_jk(eri,spins.sum(axis=0))[0]
    expected_k = np.stack([dense_jk(eri,d)[1] for d in spins])
    np.testing.assert_allclose(result.j[0], expected_j, atol=2e-11, rtol=0)
    np.testing.assert_allclose(result.k, expected_k, atol=2e-11, rtol=0)
    np.testing.assert_allclose(result.j[0], baseline.j.sum(axis=0), atol=2e-11, rtol=0)
    def energy(d):
        r = selected(d)
        return sum(np.einsum('pq,qp',d[s],h+.5*r.j[0]-.5*r.k[s]) for s in range(2))
    expected_energy = sum(np.einsum('pq,qp',spins[s],h+.5*expected_j-.5*expected_k[s]) for s in range(2))
    assert abs(energy(spins)-expected_energy) < 2e-11
    direction = np.ascontiguousarray(.01*(raw+raw.conj().transpose(0,2,1)))
    step = 1e-5
    for spin in range(2):
        delta = np.zeros_like(spins)
        delta[spin] = direction[spin]
        fd = (energy(spins+step*delta)-energy(spins-step*delta))/(2*step)
        derivative = np.einsum('pq,qp',delta[spin],h+expected_j-expected_k[spin])
        assert abs(fd-derivative) < 2e-9
    # Empty beta and arbitrary non-Hermitian transition channels remain valid.
    spins[1] = 0
    zero = selected(spins)
    np.testing.assert_array_equal(zero.k[1], 0)
    transition = selected(raw)
    np.testing.assert_allclose(transition.j[0], dense_jk(eri,raw.sum(axis=0))[0], atol=2e-11, rtol=0)


@pytest.mark.parametrize('flags', [1,2])
def test_native_omitted_target_arithmetic_is_not_executed(flags):
    offsets, matrices = np.arange(3,dtype=np.int64), np.ones((1,2))
    density = np.zeros((1,2,2),complex)
    density[0,1,1] = 1e-308 if flags==1 else 2
    density[0,0,1] = 2 if flags==1 else 1e-308
    rep = np.array([0,0,1,1],np.int64)
    members = np.array([[0,0,*rep]],np.int64)
    tile = np.full((1,)*4,1e308)
    selectors = np.array([flags],np.uint8)
    selected = core._SymmetryQuartetAccumulator(offsets,matrices,density,1,BUDGET.native(),selectors)
    selectors[:] = 3  # Native target inventory is owned, not borrowed.
    selected.accumulate(rep,tile,members)
    j,k,counts = selected.finish()
    assert counts['scalar_products'] == counts['block_contractions'] == 1
    assert (k if flags==1 else j).shape == (0,2,2)
    assert abs((j[0,0] if flags==1 else k[0,1])-1) < 1e-15
    all_targets = core._SymmetryQuartetAccumulator(offsets,matrices,density,1,BUDGET.native())
    with pytest.raises(ValueError,match='nonfinite'):
        all_targets.accumulate(rep,tile,members)
    with pytest.raises(ValueError,match='failed'):
        all_targets.finish()


def test_native_target_planner_and_binding_refuse_invalid_requests():
    offsets, matrices, density = np.arange(3,dtype=np.int64), np.ones((1,2)), np.ones((2,2,2),complex)
    for nj,nk in ((0,0),(1,0),(3,0),(2,3)):
        with pytest.raises(ValueError,match='targets'):
            core._plan_symmetry_quartet_accumulator(2,2,1,2,1,BUDGET.native(),2,nj,nk)
    for bad in (np.array([0,1],np.uint8),np.array([1,4],np.uint8),np.array([1],np.uint8),
                np.ones(2,np.int64),np.ones((1,2),np.uint8),np.ones(4,np.uint8)[::2],[1,2]):
        with pytest.raises((ValueError,TypeError)):
            core._SymmetryQuartetAccumulator(offsets,matrices,density,1,BUDGET.native(),bad)
    selectors = np.array([1,2],np.uint8)
    p = core._plan_symmetry_quartet_accumulator(2,2,1,2,1,BUDGET.native(),2,1,1)
    core._SymmetryQuartetAccumulator(offsets,matrices,density,1,Budget(p.bytes,p.work).native(),selectors)
    for b in (Budget(p.bytes-1,p.work),Budget(p.bytes,p.work-1)):
        with pytest.raises(ValueError,match='budget exceeded'):
            core._SymmetryQuartetAccumulator(offsets,matrices,density,1,b.native(),selectors)


@pytest.mark.parametrize('flags', [(1,1,1,1), (2,2,2,2), (2,1,3,2)])
def test_native_selected_unequal_axes_all_permutations_and_accumulation_order(flags):
    # Independent four-leg oracle: unequal axes distinguish every ERI
    # permutation, while complex transition densities expose transposition
    # errors that symmetric densities can conceal. No source admission claim.
    offsets = np.array([0,1,3,6,10], np.int64)
    sizes = np.diff(offsets)
    rng = np.random.default_rng(1967)
    blocks = [np.eye(n)+.1*rng.normal(size=(n,n)) for n in sizes]
    packed = np.stack([np.concatenate([np.eye(n).ravel() for n in sizes]),
                       np.concatenate([b.ravel() for b in blocks])])
    density = rng.normal(size=(4,10,10))+1j*rng.normal(size=(4,10,10))
    owned = density.copy()
    selectors = np.array(flags, np.uint8)
    selected = core._SymmetryQuartetAccumulator(offsets,packed,density,4,BUDGET.native(),selectors)
    complete = core._SymmetryQuartetAccumulator(offsets,packed,density,4,BUDGET.native())
    density[:] = 0
    selectors[:] = 3
    packed[:] = 0
    orders = [(0,1,2,3),(1,0,2,3),(0,1,3,2),(1,0,3,2),
              (2,3,0,1),(3,2,0,1),(2,3,1,0),(3,2,1,0)]
    expected_j, expected_k = np.zeros_like(owned), np.zeros_like(owned)
    for key in ((0,1,2,3),(3,2,1,0),(1,3,0,2)):
        rep = np.array(key,np.int64)
        tile = rng.normal(size=tuple(sizes[rep]))
        rotated = np.einsum('ap,bq,cr,ds,abcd->pqrs', *(blocks[s] for s in rep), tile, optimize=True)
        members = np.array([(g,p,*(rep[list(order)])) for g in range(2)
                            for p,order in enumerate(orders)],np.int64)
        for g,p,*shells in members:
            value = (tile if g == 0 else rotated).transpose(orders[p])
            a,b,c,d = [slice(offsets[s],offsets[s+1]) for s in shells]
            expected_j[:,a,b] += np.einsum('pqrs,tsr->tpq',value,owned[:,d,c])
            expected_k[:,a,d] += np.einsum('pqrs,tqr->tps',value,owned[:,b,c])
        selected.accumulate(rep,tile,members)
        complete.accumulate(rep,tile,members)
        tile[:] = 0
    j,k,counts = selected.finish()
    all_j,all_k,_ = complete.finish()
    ji,ki = [i for i,f in enumerate(flags) if f&1], [i for i,f in enumerate(flags) if f&2]
    np.testing.assert_allclose(j,expected_j[ji],atol=2e-12,rtol=0)
    np.testing.assert_allclose(k,expected_k[ki],atol=2e-12,rtol=0)
    np.testing.assert_array_equal(j,all_j[ji])
    np.testing.assert_array_equal(k,all_k[ki])
    assert counts == dict(representatives=3, scalar_products=3*16*24*(len(ji)+len(ki)),
                         block_contractions=3*16*(len(ji)+len(ki)), transport_scalar_products=720)


@pytest.mark.parametrize('flags', [(1,2), (2,1), (1,3), (3,2)])
@pytest.mark.parametrize('size', [1,2])
def test_native_selected_late_channel_failure_poisons_all_targets(flags,size):
    density = np.stack([np.ones((size,size)),2*np.ones((size,size))]).astype(complex)
    accumulator = core._SymmetryQuartetAccumulator(np.array([0,size],dtype=np.int64),
        np.eye(size).reshape(1,-1),density,size,BUDGET.native(),np.array(flags,np.uint8))
    tile = np.zeros((size,)*4)
    tile.flat[0] = 1e308
    with pytest.raises(ValueError,match='nonfinite'):
        accumulator.accumulate(np.zeros(4,np.int64),tile,np.zeros((1,6),np.int64))
    with pytest.raises(ValueError,match='failed'):
        accumulator.finish()


def screening_fixture(kind='water'):
    """Stored immutable producer; streaming is deliberately a refusal target."""
    from vibeqc.manta import ReferenceJKBuilder
    mol, basis, rotations, translations = molecular_case(kind)
    provider = ReferenceJKBuilder(basis)
    plan = qualify_manta_quartets(mol, basis, rotations, translations,
        provider=provider, source_tolerance=1e-10, budget=BUDGET, schedule=SCHEDULE)
    return plan, provider, mol, basis, rotations, translations


def masked_source(plan, provider, screening):
    """Independent ordered provider calls, with the identical fixed mask."""
    n, ns = int(plan.offsets[-1]), len(plan.offsets)-1
    full, masked = np.empty((n,)*4), np.empty((n,)*4)
    kept_keys = {member.shells for keep, orbit in zip(screening.keep, plan.orbits)
                 if keep for member in orbit.members}
    session = provider.session()
    for key in product(range(ns), repeat=4):
        selection = tuple(slice(plan.offsets[s], plan.offsets[s+1]) for s in key)
        tile = session.evaluate(key)['values']
        full[selection] = tile
        masked[selection] = tile if key in kept_keys else 0.
    return full, masked


@pytest.mark.parametrize('kind', ['water', 'ammonia', 'cartesian'])
@pytest.mark.parametrize('executor', ['numpy', 'native'])
def test_fixed_orbit_screening_same_source_covariance_bounds_and_actual_work(kind, executor):
    from vibeqc.symmetry_quartets_screening import QuartetScreening
    plan, provider, mol, basis, rotations, translations = screening_fixture(kind)
    if SCHEDULE == 'streaming':
        with pytest.raises(TypeError, match='stored spatial'):
            QuartetScreening(plan, maximum_omitted_norm=.3, budget=BUDGET)
        return
    # Deliberately large diagnostic cuts exercise nonzero omitted integrals.
    # These are not proposed production accuracy settings.
    screen = QuartetScreening(plan, maximum_omitted_norm=2. if kind=='cartesian' else .3, budget=BUDGET)
    assert 0 < screen.omitted_representatives < len(plan.orbits)
    full, masked = masked_source(plan, provider, screen)
    np.testing.assert_allclose(screen.omitted_norm, np.linalg.norm(full-masked), atol=1e-14, rtol=1e-14)
    assert screen.omitted_norm <= screen.maximum_omitted_norm
    rng = np.random.default_rng(826)
    n = basis.nbasis
    # Fully general complex transition densities, including broken state symmetry.
    d = rng.normal(size=(3,n,n)) + 1j*rng.normal(size=(3,n,n))
    targets = ('j','k','jk')
    result = plan.jk_batch(d, targets=targets, screening=screen, budget=BUDGET, executor=executor)
    assert result.j_channels == (0,2) and result.k_channels == (1,2)
    assert result.screening is screen and result.source_identity == screen.source_identity
    assert result.source_identity != plan.source_identity
    exact, uncut = [dense_jk(masked,x) for x in d], [dense_jk(full,x) for x in d]
    for which, (array, channels) in enumerate(((result.j,result.j_channels), (result.k,result.k_channels))):
        for out, channel in enumerate(channels):
            np.testing.assert_allclose(array[out], exact[channel][which], atol=2e-11, rtol=0)
            assert np.linalg.norm(array[out]-exact[channel][which]) <= screen.retained_residual_norm*np.linalg.norm(d[channel])+2e-13
            assert np.linalg.norm(array[out]-uncut[channel][which]) <= result.error_bounds[channel]+2e-13
    for r,t in zip(rotations, translations):
        action = build_molecular_space_action(mol,basis,r,t,source=SPACE,target=SPACE,budget=BUDGET)
        u = action.apply(np.eye(n,dtype=complex))
        moved = np.ascontiguousarray(u@d@u.conj().T)
        transformed = plan.jk_batch(moved, targets=targets, screening=screen,budget=BUDGET,executor=executor)
        assert transformed.source_identity == result.source_identity
        for after,before in ((transformed.j,result.j),(transformed.k,result.k)):
            np.testing.assert_allclose(u.conj().T@after@u,before,atol=3e-11,rtol=0)
    active = [o for keep,o in zip(screen.keep,plan.orbits) if keep]
    assert result.work.tile_requests == result.work.shell_calls == len(active)
    expected_primitives = sum(provider.session().evaluate(o.representative)['primitive_quartets_requested'] for o in active)
    assert result.work.primitive_quartets_requested == expected_primitives
    assert result.scalar_products == 4*(n**4-screen.omitted_integral_values)
    assert result.block_contractions == 4*((len(plan.offsets)-1)**4-screen.omitted_ordered_tiles)
    assert result.work.shell_calls < plan.jk_batch(d,targets=targets,budget=BUDGET,executor=executor).work.shell_calls
    assert not plan.production_reduction_authorized


@pytest.mark.parametrize('executor', ['numpy','native'])
@pytest.mark.parametrize('cap', [0., 100.])
def test_screening_zero_or_all_omitted_and_empty_targets(executor, cap):
    from vibeqc.symmetry_quartets_screening import QuartetScreening
    if SCHEDULE == 'streaming':
        pytest.skip('fixed masks require the stored complete audit')
    plan, provider, _, basis, _, _ = screening_fixture()
    screen = QuartetScreening(plan,maximum_omitted_norm=cap,budget=BUDGET)
    d = np.ones((1,basis.nbasis,basis.nbasis),complex)
    result = plan.jk_batch(d,targets=('k',),screening=screen,budget=BUDGET,executor=executor)
    assert result.j.shape == (0,basis.nbasis,basis.nbasis)
    full,masked = masked_source(plan,provider,screen)
    np.testing.assert_allclose(result.k[0],dense_jk(masked,d[0])[1],atol=1e-12,rtol=0)
    if cap:
        assert not any(screen.keep)
        assert result.work.tile_requests == result.scalar_products == result.block_contractions == 0
        assert not np.any(result.k)
        assert np.linalg.norm(dense_jk(full,d[0])[1]) <= result.error_bounds[0]+1e-12
    else:
        assert screen.omitted_norm == 0
        assert screen.omitted_representatives > 0
        np.testing.assert_allclose(result.k[0],dense_jk(full,d[0])[1],atol=1e-12,rtol=0)


@pytest.mark.parametrize('executor', ['numpy','native'])
def test_screening_budget_identity_ownership_and_invalid_domains(executor):
    from vibeqc.symmetry_quartets_screening import QuartetScreening, quartet_screening_requirements
    from vibeqc.symmetry_quartets_batch import quartet_batch_requirements
    if SCHEDULE == 'streaming':
        pytest.skip('fixed masks require the stored complete audit')
    plan, provider, mol, basis, rotations, translations = screening_fixture()
    required = quartet_screening_requirements(plan,budget=BUDGET)
    screen = QuartetScreening(plan,maximum_omitted_norm=.3,budget=required)
    repeat = QuartetScreening(plan,maximum_omitted_norm=.3,budget=required)
    assert screen.receipt == repeat.receipt
    import json
    json.dumps(screen.receipt,allow_nan=False)
    receipt = screen.receipt
    receipt['source_identity'] = 'untrusted edit'
    assert screen.receipt['source_identity'] == screen.source_identity
    with pytest.raises((AttributeError,TypeError)):
        screen.keep = ()
    same_mask = QuartetScreening(plan,maximum_omitted_norm=screen.omitted_norm,budget=required)
    assert same_mask.source_identity == screen.source_identity
    assert same_mask.policy_identity != screen.policy_identity
    zero = QuartetScreening(plan,maximum_omitted_norm=0.,budget=BUDGET)
    assert zero.source_identity != screen.source_identity
    d = np.ones((1,basis.nbasis,basis.nbasis),complex)
    build = quartet_batch_requirements(plan,1,budget=BUDGET,executor=executor,screening=screen)
    plan.jk_batch(d,budget=build,executor=executor,screening=screen)
    # Count-only refusal precedes any integral callback or density inspection.
    invalid = d.copy()
    invalid[:] = np.nan
    for short,error in ((Budget(build.maximum_bytes-1,build.maximum_work),MemoryError),
                        (Budget(build.maximum_bytes,build.maximum_work-1),ValueError)):
        with pytest.raises(error,match='budget'):
            plan.jk_batch(invalid,budget=short,executor=executor,screening=screen)
    for short,error in ((Budget(required.maximum_bytes-1,required.maximum_work),MemoryError),
                        (Budget(required.maximum_bytes,required.maximum_work-1),ValueError)):
        with pytest.raises(error,match='budget'):
            QuartetScreening(plan,maximum_omitted_norm=.3,budget=short)
    other = qualify_manta_quartets(mol,basis,rotations,translations,provider=provider,
        source_tolerance=1e-10,budget=BUDGET)
    assert other.audit_identity == plan.audit_identity
    with pytest.raises(ValueError,match='different source audit'):
        other.jk_batch(d,budget=BUDGET,executor=executor,screening=screen)
    for cap in (np.nan, np.inf, -1., True, '0.1'):
        with pytest.raises((TypeError,ValueError)):
            QuartetScreening(plan,maximum_omitted_norm=cap,budget=BUDGET)
    _,_,g,a,mutable,_,_ = source_fixture('water')
    callback = ShellQuartetPlan(g,a,mutable,source_tolerance=1e-10,budget=BUDGET)
    with pytest.raises(TypeError,match='immutable MANTA'):
        QuartetScreening(callback,maximum_omitted_norm=.3,budget=BUDGET)
    permutation = qualify_manta_quartets(mol,basis,rotations,translations,provider=provider,
        source_tolerance=1e-10,budget=BUDGET,mode='permutation')
    with pytest.raises(TypeError,match='stored spatial'):
        QuartetScreening(permutation,maximum_omitted_norm=.3,budget=BUDGET)


@pytest.mark.parametrize('executor', ['numpy','native'])
def test_screened_fixed_source_broken_spin_energy_and_density_response(executor):
    from vibeqc.symmetry_quartets_screening import QuartetScreening
    if SCHEDULE == 'streaming':
        pytest.skip('fixed masks require the stored complete audit')
    plan,provider,_,basis,_,_ = screening_fixture('ammonia')
    screen = QuartetScreening(plan,maximum_omitted_norm=.3,budget=BUDGET)
    full,masked = masked_source(plan,provider,screen)
    rng = np.random.default_rng(32)
    raw = rng.normal(size=(2,basis.nbasis,basis.nbasis))+1j*rng.normal(size=(2,basis.nbasis,basis.nbasis))
    d = raw+raw.conj().transpose(0,2,1)
    delta = d*.17+np.eye(basis.nbasis)[None]
    def evaluate(spin):
        batch = np.stack((spin.sum(axis=0),spin[0],spin[1]))
        result = plan.jk_batch(batch,targets=('j','k','k'),screening=screen,budget=BUDGET,executor=executor)
        energy = .5*np.einsum('pq,qp',batch[0],result.j[0])-.5*np.einsum('apq,aqp',spin,result.k)
        return result,energy
    result,energy = evaluate(d)
    j = dense_jk(masked,d.sum(axis=0))[0]
    k = np.stack([dense_jk(masked,x)[1] for x in d])
    reference = .5*np.einsum('pq,qp',d.sum(axis=0),j)-.5*np.einsum('apq,aqp',d,k)
    assert abs(energy-reference) < 1e-12
    jfull = dense_jk(full,d.sum(axis=0))[0]
    kfull = np.stack([dense_jk(full,x)[1] for x in d])
    uncut = .5*np.einsum('pq,qp',d.sum(axis=0),jfull)-.5*np.einsum('apq,aqp',d,kfull)
    method_bound = .5*(np.linalg.norm(d.sum(axis=0))*result.error_bounds[0]
        +sum(np.linalg.norm(x)*b for x,b in zip(d,result.error_bounds[1:])))
    assert abs(energy-uncut) <= method_bound+1e-12
    h=1e-5
    derivative = (evaluate(d+h*delta)[1]-evaluate(d-h*delta)[1])/(2*h)
    expected = np.einsum('apq,aqp',delta,result.j[0][None]-result.k)
    np.testing.assert_allclose(derivative,expected,atol=3e-8,rtol=0)


def test_cartesian_screening_uses_all_member_norms_not_representative_multiplicity():
    from vibeqc.manta import ReferenceJKBuilder
    from vibeqc.symmetry_quartets_screening import QuartetScreening
    if SCHEDULE == 'streaming':
        pytest.skip('fixed masks require the stored complete audit')
    _,_,rotations,translations = molecular_case('cartesian')
    origins = [[1.7*np.cos(t),1.7*np.sin(t),0.] for t in (0.,2*np.pi/3,4*np.pi/3)]
    mol = vq.Molecule([vq.Atom(2,xyz) for xyz in origins])
    basis = vq.BasisSet(mol,[core.ShellInfo(i,2,False,[.8],[1.],xyz)
        for i,xyz in enumerate(origins)],'Cartesian triangle',False)
    provider = ReferenceJKBuilder(basis)
    plan = qualify_manta_quartets(mol,basis,rotations,translations,
        provider=provider,source_tolerance=1e-10,budget=BUDGET)
    # Cartesian coefficient actions are metric-unitary, not Euclidean-unitary.
    # Multiplying a representative's norm by sqrt(orbit size) underestimates
    # this complete orbit and would admit an invalid 0.52 tensor-error budget.
    smallest = min(plan.orbits,key=lambda o:o.source_norm)
    shortcut = np.linalg.norm(provider.evaluate(smallest.representative)['values'])*np.sqrt(len(smallest.members))
    assert shortcut < .52 < smallest.source_norm
    screen = QuartetScreening(plan,maximum_omitted_norm=.52,budget=BUDGET)
    assert screen.omitted_representatives == 0
    screen = QuartetScreening(plan,maximum_omitted_norm=.56,budget=BUDGET)
    assert screen.omitted_representatives == 1
    full,masked = masked_source(plan,provider,screen)
    assert np.linalg.norm(full-masked) <= .56
    np.testing.assert_allclose(screen.omitted_norm,np.linalg.norm(full-masked),atol=1e-14,rtol=0)
