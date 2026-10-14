"""Cutoff sources: independent membership, conjugacy, SCF and admission."""
import itertools

import numpy as np
import pytest
from scipy.linalg import eigh

import vibeqc as vq
from vibeqc import manta, _vibeqc_core as core
from tests.test_manta_periodic_rhf import _case, _components, _nuclear_reference


def _arguments(mesh=(1, 1, 1), shift=(0, 0, 0), cutoff=2.0):
    case, k = _case(mesh, shift)
    args = {name: case[name] for name in ('basis', 'system', 'mesh', 'options', 'budget')}
    args['reciprocal_energy_cutoff'] = cutoff
    args['budget'].memory_bytes = 2**30
    args['budget'].shell_calls = 10**9
    args['budget'].scalar_terms = 10**17
    args['budget'].kernel_terms = 10**18
    return args, k


def _explicit(args, domains):
    case = {name: args[name] for name in ('basis', 'system', 'mesh', 'options', 'budget')}
    case.update({name: getattr(domains, name) for name in (
        'reciprocal_vectors', 'exchange_vectors', 'transfer_offsets', 'translations')})
    return case


def _labels(vectors, inverse):
    values = np.asarray(vectors) @ inverse.T
    np.testing.assert_allclose(values, np.rint(values), atol=2e-12)
    return {tuple(int(x) for x in row) for row in np.rint(values)}


@pytest.mark.parametrize('shape,shift', [((1, 1, 1), (0, 0, 0)),
    ((2, 1, 1), (0, 0, 0)), ((2, 1, 1), (1, 0, 0)), ((3, 2, 1), (1, 1, 0))])
def test_skew_reciprocal_spheres_match_independent_integer_enumeration(shape, shift):
    args, _ = _arguments(shape, shift)
    lattice = np.array([[6., 3.5, .8], [0., 4.2, 1.2], [0., 0., 6.4]])
    args['system'] = vq.PeriodicSystem(3, lattice, args['system'].unit_cell)
    domains = manta.build_periodic_rhf_domains(**args)
    b = 2*np.pi*np.linalg.inv(lattice).T
    mesh = args['mesh']
    k = np.array([mesh.fractional(mesh.address(i)) for i in range(len(mesh))]) @ b.T
    labels = np.array(list(itertools.product(range(-8, 9), repeat=3)))
    base = labels @ b.T
    expected = labels[(np.sum(base*base, axis=1) <= 2*args['reciprocal_energy_cutoff']) & np.any(labels, axis=1)]
    assert _labels(domains.reciprocal_vectors, np.linalg.inv(b)) == {tuple(n) for n in expected}
    q, offsets = domains.exchange_vectors, domains.transfer_offsets
    for t, kt in enumerate(k):
        for s, ks in enumerate(k):
            pair = t*len(k)+s
            actual = q[offsets[pair]:offsets[pair+1]]
            trial = base+kt-ks
            keep = (np.sum(trial*trial, axis=1) <= 2*args['reciprocal_energy_cutoff']) & np.any(trial, axis=1)
            assert _labels(actual-kt+ks, np.linalg.inv(b)) == {tuple(n) for n in labels[keep]}
            other = s*len(k)+t
            reverse = q[offsets[other]:offsets[other+1]]
            assert len(actual) == len(reverse)
            assert max(min(np.linalg.norm(reverse+v, axis=1)) for v in actual) < 3e-14
    assert not np.any(np.all(q == 0, axis=1))
    assert domains.receipt.conjugacy_vectors == len(q)//len(k)
    assert domains.receipt.exchange_vectors <= domains.plan.exchange_candidates
    assert len(q) == offsets[-1] and offsets.dtype == np.int64


@pytest.mark.parametrize('shape,shift', [((1, 1, 1), (0, 0, 0)), ((2, 1, 1), (1, 0, 0))])
def test_generated_rhf_matches_explicit_source_and_independent_scf(shape, shift):
    args, k = _arguments(shape, shift)
    domains = manta.build_periodic_rhf_domains(**args)
    assert len(domains.translations) == 1
    case = _explicit(args, domains)
    result = core._manta.run_periodic_rhf_from_cutoffs(**args)
    explicit = core._manta.run_periodic_rhf(**case)
    assert result.converged and result.receipt.confirmed
    np.testing.assert_array_equal(result.density, explicit.density)
    np.testing.assert_array_equal(result.fock, explicit.fock)
    assert result.receipt.source_identity == explicit.receipt.source_identity
    assert result.receipt.domain_identity == domains.receipt.identity
    s, h, build = _components(case, k)
    def occupy(f):
        c = np.array([eigh(fp, sp)[1][:, :1] for fp, sp in zip(f, s)])
        return 2*c @ c.conj().transpose(0, 2, 1)
    density = occupy(h)
    for _ in range(150):
        updated = occupy(build(density))
        if np.max(abs(updated-density)) < 1e-12:
            break
        density = updated
    else:
        pytest.fail('Independent cutoff-source SCF failed to converge')
    nuclear, _ = _nuclear_reference(case)
    energy = sum(nuclear)+.5/len(k)*np.einsum('kij,kji->', density, h+build(density)).real
    assert result.receipt.energy == pytest.approx(energy, abs=3e-11)
    np.testing.assert_allclose(result.density, density, atol=3e-10)
    np.testing.assert_allclose(result.fock, build(result.density), atol=3e-12)
    assert result.plan.domain_live_bytes == domains.plan.domain_live_bytes
    assert result.plan.scalar_terms <= domains.plan.scalar_terms
    assert result.plan.kernel_terms <= domains.plan.kernel_terms
    assert result.receipt.modeled_peak_bytes <= domains.plan.modeled_peak_bytes


def test_physical_pair_images_cover_unwrapped_centers_and_preserve_bloch_gauge():
    args, k = _arguments((2, 1, 1), (1, 0, 0), cutoff=3.)
    args['options'].pair_cutoff_bohr = 7.
    args['options'].nuclear_cutoff_bohr = 9.
    args['options'].exchange_q0 = 'BvK-ewald'
    original = core._manta.run_periodic_rhf_from_cutoffs(**args)
    domains = manta.build_periodic_rhf_domains(**args)
    lattice = np.asarray(args['system'].lattice)
    atoms = list(args['system'].unit_cell)
    atoms[0] = vq.Atom(1, np.asarray(atoms[0].xyz)+lattice[:, 0])
    args['system'] = vq.PeriodicSystem(3, lattice, atoms)
    shells = list(args['basis'].shells())
    shells[0].origin = atoms[0].xyz
    args['basis'] = vq.BasisSet(args['system'].unit_cell_molecule(), shells, 'manta-domain-images', False)
    moved_domains = manta.build_periodic_rhf_domains(**args)
    translated = _labels(moved_domains.translations, np.linalg.inv(lattice))
    centers = np.array([s.origin for s in shells])
    for i, j in itertools.product(range(len(centers)), repeat=2):
        for label in itertools.product(range(-4, 5), repeat=3):
            if np.linalg.norm(centers[i]-centers[j]-lattice@label) <= 7.:
                assert label in translated
    moved = core._manta.run_periodic_rhf_from_cutoffs(**args)
    assert original.converged and moved.converged
    assert len(moved_domains.translations) > len(domains.translations) > 1
    phase = np.ones((len(k), 2), dtype=complex)
    phase[:, 0] = np.exp(1j*k@lattice[:, 0])
    # The AO source's exp(+ikR) Bloch basis fixes this gauge action on all
    # matrices; the shifted points give non-real phases and test its sign.
    action = phase[:, :, None]*phase[:, None, :].conj()
    np.testing.assert_allclose(moved.overlap, original.overlap*action, atol=3e-12)
    np.testing.assert_allclose(moved.hcore, original.hcore*action, atol=4e-12)
    np.testing.assert_allclose(moved.fock, original.fock*action, atol=1e-10)
    np.testing.assert_allclose(moved.density, original.density*action, atol=2e-9)
    assert moved.receipt.energy == pytest.approx(original.receipt.energy, abs=4e-11)


@pytest.mark.parametrize('field,planned', [('memory_bytes', 'modeled_peak_bytes'),
    ('shell_calls', 'shell_calls'), ('primitive_quartets', 'primitive_pairs'),
    ('scalar_terms', 'scalar_terms'), ('kernel_terms', 'kernel_terms')])
def test_complete_run_admission_exact_bound_and_one_below(field, planned):
    args, _ = _arguments((2, 1, 1), (1, 0, 0))
    args['budget'].other_live_bytes = 12345
    plan = manta.plan_periodic_rhf_from_cutoffs(**args)
    setattr(args['budget'], field, getattr(plan, planned))
    assert core._manta.run_periodic_rhf_from_cutoffs(**args).converged
    setattr(args['budget'], field, getattr(plan, planned)-1)
    with pytest.raises((ValueError, RuntimeError), match='budget'):
        core._manta.run_periodic_rhf_from_cutoffs(**args)


def test_complete_admission_precedes_empty_reciprocal_source_enumeration():
    args, _ = _arguments(cutoff=1e-5)
    plan = manta.plan_periodic_rhf_from_cutoffs(**args)
    with pytest.raises(ValueError, match='no nonzero Hartree'):
        manta.build_periodic_rhf_domains(**args)
    args['budget'].shell_calls = plan.shell_calls-1
    with pytest.raises((ValueError, RuntimeError), match='budget'):
        manta.build_periodic_rhf_domains(**args)


def test_owned_exports_and_identity_bind_cutoffs_mesh_and_centers():
    args, _ = _arguments()
    domains = manta.build_periodic_rhf_domains(**args)
    for name in ('translations', 'reciprocal_vectors', 'exchange_vectors', 'transfer_offsets'):
        original = getattr(domains, name).copy()
        getattr(domains, name)[:] = 999
        np.testing.assert_array_equal(getattr(domains, name), original)
    with pytest.raises(AttributeError):
        domains.receipt.identity = 'untrusted'
    args['options'].reciprocal_batch = 1
    assert manta.build_periodic_rhf_domains(**args).receipt.identity == domains.receipt.identity
    args['reciprocal_energy_cutoff'] += .1
    assert manta.build_periodic_rhf_domains(**args).receipt.identity != domains.receipt.identity


@pytest.mark.parametrize('field,value', [('reciprocal_energy_cutoff', 0.),
    ('reciprocal_energy_cutoff', float('nan')), ('reciprocal_energy_cutoff', float('inf')),
    ('reciprocal_energy_cutoff', 1e300), ('pair_cutoff_bohr', None),
    ('pair_cutoff_bohr', 0.), ('pair_cutoff_bohr', float('nan')), ('pair_cutoff_bohr', 1e300)])
def test_invalid_or_unbounded_cutoffs_refuse(field, value):
    args, _ = _arguments()
    if field == 'pair_cutoff_bohr':
        setattr(args['options'], field, value)
    else:
        args[field] = value
    with pytest.raises((ValueError, RuntimeError, OverflowError)):
        manta.build_periodic_rhf_domains(**args)


def test_selected_output_contains_executed_domains(monkeypatch):
    args, _ = _arguments((2, 1, 1), (1, 0, 0))
    records = []
    monkeypatch.setattr('vibeqc.output.record', lambda *a, **kw: records.append((a, kw)))
    engine = manta.IntegralEngineOptions(backend='manta', budget=args.pop('budget'))
    result = manta.run_periodic_rhf_from_cutoffs(**args, integral_engine=engine)
    assert result.converged and len(records) == 1
    evidence = records[0][1]['integral_provenance']
    assert evidence['domain_identity'] == result.receipt.domain_identity
    assert evidence['domain_selection'] == 'physical-pair-sphere;certified-reciprocal-energy-sphere'
    assert evidence['reciprocal_energy_cutoff'] == args['reciprocal_energy_cutoff']
    assert evidence['domain_translations'] == 1 and evidence['domain_exchange_vectors'] > 0
    assert evidence['domain_kernel_terms'] > 0


def test_exact_spherical_boundary_and_small_budget_for_large_mesh():
    args, _ = _arguments(cutoff=.5)
    args['system'] = vq.PeriodicSystem(3, 2*np.pi*np.eye(3), args['system'].unit_cell)
    domains = manta.build_periodic_rhf_domains(**args)
    assert len(domains.reciprocal_vectors) == 6
    np.testing.assert_array_equal(np.sum(domains.reciprocal_vectors**2, axis=1), np.ones(6))
    args['mesh'] = manta.RegularKMesh((10000, 10000, 1), (1, 1, 0))
    with pytest.raises((ValueError, RuntimeError, OverflowError), match='budget|overflow'):
        manta.plan_periodic_rhf_from_cutoffs(**args)


@pytest.mark.parametrize('shape,shift', [((1, 1, 1), (0, 0, 0)), ((2, 1, 1), (1, 0, 0))])
def test_refined_cutoff_rhf_against_optional_pyscf_aft(shape, shift):
    pytest.importorskip('pyscf.pbc')
    from pyscf.pbc import gto, df, scf
    args, k = _arguments(shape, shift, cutoff=60.)
    o = args['options']
    o.pair_cutoff_bohr = 9.; o.nuclear_cutoff_bohr = 12.
    o.exchange_q0 = 'BvK-ewald'; o.reciprocal_batch = 128
    o.scf.maximum_iterations = 100
    result = core._manta.run_periodic_rhf_from_cutoffs(**args)
    cell = gto.Cell()
    cell.atom = [('H', a.xyz) for a in args['system'].unit_cell]
    cell.a = args['system'].lattice.T
    cell.unit = 'Bohr'; cell.basis = {'H': [[0, [.8, 1.]]]}
    cell.precision = 1e-11; cell.verbose = 0; cell.build()
    mf = scf.KRHF(cell, k, exxdiv='ewald')
    mf.with_df = df.AFTDF(cell, k); mf.with_df.mesh = [25]*3
    mf.conv_tol = 1e-11; mf.max_cycle = 100
    energy = mf.kernel()
    assert mf.converged and result.converged
    assert result.receipt.energy == pytest.approx(energy, abs=3e-10)
    assert result.receipt.nuclear_energy == pytest.approx(cell.energy_nuc(), abs=3e-11)
    np.testing.assert_allclose(result.overlap, mf.get_ovlp(), atol=3e-11)
    np.testing.assert_allclose(result.hcore, mf.get_hcore(), atol=3e-10)
    np.testing.assert_allclose(result.density, mf.make_rdm1(), atol=2e-8)
    np.testing.assert_allclose(result.fock, mf.get_fock(dm=result.density), atol=3e-10)
