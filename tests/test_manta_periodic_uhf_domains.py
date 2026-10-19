"""Fixed-spin cutoff sources: independent geometry, SCF and resource admission."""
import itertools

import numpy as np
import pytest

from vibeqc import manta, _vibeqc_core as core
from tests.test_manta_periodic_uhf import _case, _reference, _scf_reference, _nuclear_reference
from tests.test_manta_periodic_domains import _explicit, _labels


def _arguments(mesh=(1, 1, 1), shift=(0, 0, 0), species='Li', correction='BvK-ewald', cutoff=2.):
    case, k = _case(mesh, shift, species, correction)
    args = {name: case[name] for name in ('basis', 'system', 'mesh', 'options', 'budget')}
    args['reciprocal_energy_cutoff'] = cutoff
    args['budget'].memory_bytes = 2**30
    args['budget'].shell_calls = 10**9
    args['budget'].scalar_terms = 10**17
    args['budget'].kernel_terms = 10**18
    return args, k


@pytest.mark.parametrize('species', ['H', 'Li'])
@pytest.mark.parametrize('mesh,shift', [((1, 1, 1), (0, 0, 0)), ((2, 1, 1), (1, 0, 0))])
@pytest.mark.parametrize('correction', ['strict-zero-mode', 'BvK-ewald'])
def test_generated_spin_scf_matches_explicit_and_independent_integrals(species, mesh, shift, correction):
    args, k = _arguments(mesh, shift, species, correction)
    domains = manta.build_periodic_uhf_domains(**args)
    case = _explicit(args, domains)
    result = core._manta.run_periodic_uhf_from_cutoffs(**args)
    explicit = core._manta.run_periodic_uhf(**case)
    assert result.converged and explicit.converged
    np.testing.assert_array_equal(result.density, explicit.density)
    np.testing.assert_array_equal(result.fock, explicit.fock)
    assert result.receipt.source_identity == explicit.receipt.source_identity
    assert result.receipt.domain_identity == domains.receipt.identity
    s, h, build, _ = _reference(case, k)
    density = _scf_reference(s, h, build, (result.receipt.n_alpha, result.receipt.n_beta))
    nuclear, _ = _nuclear_reference(case)
    assert result.receipt.energy == pytest.approx(sum(nuclear)+sum(build(density)[1]), abs=4e-11)
    np.testing.assert_allclose(result.density, density, atol=5e-10, rtol=0)
    np.testing.assert_allclose(result.fock, build(result.density)[0], atol=4e-12, rtol=0)
    assert result.plan.domain_live_bytes == domains.plan.domain_live_bytes
    for name in ('scalar_terms', 'kernel_terms', 'modeled_peak_bytes'):
        assert getattr(result.plan, name) <= getattr(domains.plan, name)
    assert domains.plan.uhf.n_alpha == result.receipt.n_alpha
    assert domains.plan.uhf.n_beta == result.receipt.n_beta


@pytest.mark.parametrize('mesh,shift', [((1, 1, 1), (0, 0, 0)), ((3, 2, 1), (1, 1, 0))])
def test_cutoff_membership_matches_independent_integer_enumeration(mesh, shift):
    args, k = _arguments(mesh, shift)
    domains = manta.build_periodic_uhf_domains(**args)
    b = 2*np.pi*np.linalg.inv(args['system'].lattice).T
    labels = np.array(list(itertools.product(range(-8, 9), repeat=3)))
    base = labels @ b.T
    q, offsets = domains.exchange_vectors, domains.transfer_offsets
    for target, kt in enumerate(k):
        for source, ks in enumerate(k):
            pair = target*len(k)+source
            actual = q[offsets[pair]:offsets[pair+1]]
            trial = base+kt-ks
            keep = (np.sum(trial*trial, axis=1) <= 2*args['reciprocal_energy_cutoff']) & np.any(trial, axis=1)
            assert _labels(actual-kt+ks, np.linalg.inv(b)) == {tuple(n) for n in labels[keep]}
    keep = (np.sum(base*base, axis=1) <= 2*args['reciprocal_energy_cutoff']) & np.any(labels, axis=1)
    assert _labels(domains.reciprocal_vectors, np.linalg.inv(b)) == {tuple(n) for n in labels[keep]}
    assert domains.receipt.conjugacy_vectors == len(q)//len(k)


@pytest.mark.parametrize('field,planned', [('memory_bytes', 'modeled_peak_bytes'),
    ('shell_calls', 'shell_calls'), ('primitive_quartets', 'primitive_pairs'),
    ('scalar_terms', 'scalar_terms'), ('kernel_terms', 'kernel_terms')])
def test_uhf_complete_admission_exact_and_one_below(field, planned):
    args, _ = _arguments((2, 1, 1), (1, 0, 0))
    args['budget'].other_live_bytes = 12345
    plan = manta.plan_periodic_uhf_from_cutoffs(**args)
    setattr(args['budget'], field, getattr(plan, planned))
    assert core._manta.run_periodic_uhf_from_cutoffs(**args).converged
    setattr(args['budget'], field, getattr(plan, planned)-1)
    with pytest.raises((ValueError, RuntimeError), match='budget'):
        manta.build_periodic_uhf_domains(**args)


def test_complete_spin_admission_precedes_empty_source_enumeration():
    args, _ = _arguments(cutoff=1e-5)
    plan = manta.plan_periodic_uhf_from_cutoffs(**args)
    with pytest.raises(ValueError, match='no nonzero Hartree'):
        manta.build_periodic_uhf_domains(**args)
    args['budget'].shell_calls = plan.shell_calls-1
    with pytest.raises((ValueError, RuntimeError), match='budget'):
        manta.build_periodic_uhf_domains(**args)
    args, _ = _arguments(cutoff=1e-5)
    args['system'].multiplicity = 1
    with pytest.raises(ValueError, match='spin populations'):
        manta.build_periodic_uhf_domains(**args)


def test_owned_exports_and_selected_spin_domain_provenance(monkeypatch):
    args, _ = _arguments((2, 1, 1), (1, 0, 0))
    domains = manta.build_periodic_uhf_domains(**args)
    for name in ('translations', 'reciprocal_vectors', 'exchange_vectors', 'transfer_offsets'):
        original = getattr(domains, name)
        getattr(domains, name)[:] = 999
        np.testing.assert_array_equal(getattr(domains, name), original)
    records = []
    monkeypatch.setattr('vibeqc.output.record', lambda *a, **kw: records.append((a, kw)))
    engine = manta.IntegralEngineOptions(backend='manta', budget=args.pop('budget'))
    result = manta.run_periodic_uhf_from_cutoffs(**args, integral_engine=engine)
    assert result.converged and len(records) == 1
    evidence = records[0][1]['integral_provenance']
    assert evidence['spin_electrons_per_cell'] == [2, 1]
    assert evidence['domain_identity'] == domains.receipt.identity
    assert evidence['domain_kernel_terms'] > 0
    assert evidence['reciprocal_energy_cutoff'] == args['reciprocal_energy_cutoff']


@pytest.mark.parametrize('correction', ['strict-zero-mode', 'BvK-ewald'])
def test_independent_shell_images_transport_both_spin_states(correction):
    import vibeqc as vq
    c, k = _arguments((2, 1, 1), (1, 0, 0), 'Li', correction)
    atoms = [vq.Atom(1, p) for p in ([0., 0., 0.], [.9, .2, 0.], [-.2, 1.1, .1])]
    c['system'] = vq.PeriodicSystem(3, c['system'].lattice, atoms)
    c['system'].multiplicity = 2
    shells = c['basis'].shells()
    for index, shell in enumerate(shells):
        shell.atom_index = index; shell.origin = atoms[index].xyz
    c['basis'] = vq.BasisSet(c['system'].unit_cell_molecule(), shells, 'manta-uhf-gauge', False)
    c['options'].scf.maximum_iterations = 3
    original = core._manta.run_periodic_uhf_from_cutoffs(**c)
    shifts = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]]) @ c['system'].lattice.T
    for shell, delta in zip(shells, shifts): shell.origin = np.array(shell.origin)+delta
    basis_cell = vq.PeriodicSystem(3, c['system'].lattice, [vq.Atom(1, shell.origin) for shell in shells])
    basis_cell.multiplicity = 2
    c['basis'] = vq.BasisSet(basis_cell.unit_cell_molecule(), shells, 'manta-uhf-image', False)
    moved = core._manta.run_periodic_uhf_from_cutoffs(**c)
    phase = np.exp(1j*k@shifts.T)
    transport = phase[:, :, None]*phase[:, None, :].conj()
    assert moved.solver_snapshot.evaluated_iteration == original.solver_snapshot.evaluated_iteration
    assert np.max(abs(moved.density-original.density)) > .01
    for field in ('density', 'fock', 'overlap', 'hcore'):
        np.testing.assert_allclose(getattr(moved, field), getattr(original, field)*transport, atol=4e-10, rtol=0)
    assert moved.receipt.energy == pytest.approx(original.receipt.energy, abs=4e-11)


@pytest.mark.parametrize("mesh,shift", [((1, 1, 1), (0, 0, 0)), ((2, 1, 1), (1, 0, 0))])
def test_refined_cutoff_uhf_against_optional_pyscf(mesh, shift):
    from tests.test_manta_periodic_uhf import _check_refined_pyscf
    _check_refined_pyscf(mesh, shift, from_cutoffs=True)


def test_geometric_identity_shared_with_rhf_but_spin_storage_admitted_separately():
    from tests.test_manta_periodic_domains import _arguments as rhf_arguments
    args, _ = rhf_arguments((2, 1, 1), (1, 0, 0))
    restricted = manta.build_periodic_rhf_domains(**args)
    old = args['options']
    options = manta.PeriodicUHFOptions()
    for name in ('alpha', 'nuclear_cutoff_bohr', 'pair_cutoff_bohr', 'reciprocal_batch', 'exchange_q0'):
        setattr(options, name, getattr(old, name))
    options.scf.maximum_iterations = old.scf.maximum_iterations
    args['options'] = options
    unrestricted = manta.build_periodic_uhf_domains(**args)
    assert unrestricted.receipt.identity == restricted.receipt.identity
    assert unrestricted.plan.domain_live_bytes == restricted.plan.domain_live_bytes
    assert unrestricted.plan.uhf.solver.peak_owned_numerical_bytes > restricted.plan.rhf.solver.peak_owned_numerical_bytes
    assert unrestricted.plan.modeled_peak_bytes > restricted.plan.modeled_peak_bytes
    args['budget'].memory_bytes = restricted.plan.modeled_peak_bytes
    with pytest.raises((ValueError, RuntimeError), match='budget'):
        manta.build_periodic_uhf_domains(**args)
