"""Fitted cutoff sources: independent geometry, spin SCF and joint admission."""
import itertools
import json

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta, _vibeqc_core as core
from tests.test_manta_periodic_domains import _labels
from tests.test_manta_periodic_ri_uhf import case, reference
from tests.test_manta_periodic_rhf import _nuclear_reference
from tests.test_manta_periodic_uhf import _scf_reference


def arguments(mesh=(1, 1, 1), shift=(0, 0, 0), species='Li', correction='strict-zero-mode', cutoff=6.):
    c, k = case(mesh, shift, species, correction)
    a = {name: c[name] for name in ('basis', 'auxiliary', 'system', 'mesh', 'options', 'budget')}
    return dict(a, reciprocal_energy_cutoff=cutoff), k


def explicit(a, domains):
    c = {name: a[name] for name in ('basis', 'auxiliary', 'system', 'mesh', 'options', 'budget')}
    return dict(c, **{name: getattr(domains, name) for name in (
        'reciprocal_vectors', 'pair_vectors', 'transfer_offsets', 'translations')})


@pytest.mark.parametrize('shape,shift', [((1, 1, 1), (0, 0, 0)), ((2, 1, 1), (0, 0, 0)),
    ((2, 1, 1), (1, 0, 0)), ((3, 2, 1), (1, 1, 0))])
def test_skew_spheres_match_independent_unwrapped_integer_enumeration(shape, shift):
    a, _ = arguments(shape, shift, cutoff=2.)
    lattice = np.array([[6., 3.5, .8], [0., 4.2, 1.2], [0., 0., 6.4]])
    a['system'] = vq.PeriodicSystem(3, lattice, a['system'].unit_cell)
    a['system'].multiplicity = 2
    domains = manta.build_periodic_ri_uhf_domains(**a)
    b = 2*np.pi*np.linalg.inv(lattice).T
    mesh = a['mesh']; nk = len(mesh)
    k = np.array([mesh.fractional(mesh.address(i)) for i in range(nk)]) @ b.T
    labels = np.array(list(itertools.product(range(-8, 9), repeat=3)))
    base = labels @ b.T
    keep = (np.sum(base*base, axis=1) <= 2*a['reciprocal_energy_cutoff']) & np.any(labels, axis=1)
    assert _labels(domains.reciprocal_vectors, np.linalg.inv(b)) == {tuple(n) for n in labels[keep]}
    q, offsets = domains.pair_vectors, domains.transfer_offsets
    for target, kt in enumerate(k):
        for source, ks in enumerate(k):
            pair = target*nk+source
            actual = q[offsets[pair]:offsets[pair+1]]
            if target == source:
                assert len(actual) == 0
                continue
            trial = base+ks-kt
            keep = np.sum(trial*trial, axis=1) <= 2*a['reciprocal_energy_cutoff']
            assert _labels(actual-ks+kt, np.linalg.inv(b)) == {tuple(n) for n in labels[keep]}
            opposite = source*nk+target
            reverse = q[offsets[opposite]:offsets[opposite+1]]
            assert len(actual) == len(reverse)
            assert max(min(np.linalg.norm(reverse+v, axis=1)) for v in actual) < 3e-14
    assert domains.receipt.conjugacy_vectors == len(q)//nk+len(domains.reciprocal_vectors)
    assert domains.receipt.exchange_vectors <= domains.plan.exchange_candidates
    assert domains.plan.uhf.fock.channels == 1+nk*(nk-1)
    assert offsets[-1] == len(q) and offsets.dtype == np.int64
    assert not np.any(np.all(q == 0., axis=1))
    if nk == 1:
        assert domains.plan.maximum_exchange_candidates == domains.plan.exchange_candidates == 0


@pytest.mark.parametrize('species', ['H', 'Li'])
@pytest.mark.parametrize('shape,shift', [((1, 1, 1), (0, 0, 0)), ((2, 1, 1), (1, 0, 0))])
@pytest.mark.parametrize('correction', ['strict-zero-mode', 'BvK-ewald'])
def test_cutoff_spin_scf_matches_explicit_and_independent_libint_reference(species, shape, shift, correction):
    a, k = arguments(shape, shift, species, correction)
    domains = manta.build_periodic_ri_uhf_domains(**a); c = explicit(a, domains)
    r = core._manta.run_periodic_ri_uhf_from_cutoffs(**a)
    direct = core._manta.run_periodic_ri_uhf(**c)
    assert r.converged and direct.converged and r.receipt.confirmed
    for name in ('density', 'fock', 'overlap', 'hcore', 'momenta'):
        np.testing.assert_array_equal(getattr(r, name), getattr(direct, name))
    assert r.receipt.source_identity == direct.receipt.source_identity
    assert r.receipt.final_density_identity == direct.receipt.final_density_identity
    assert r.receipt.domain_identity == domains.receipt.identity
    s, h, build, _ = reference(c, k)
    d = _scf_reference(s, h, build, (r.plan.n_alpha, r.plan.n_beta))
    np.testing.assert_allclose(r.overlap, s, atol=4e-12, rtol=0)
    np.testing.assert_allclose(r.hcore, h, atol=4e-12, rtol=0)
    np.testing.assert_allclose(r.density, d, atol=5e-9, rtol=0)
    f, energies = build(r.density)
    np.testing.assert_allclose(r.fock, f, atol=8e-11, rtol=0)
    np.testing.assert_allclose([r.receipt.one_electron_energy, r.receipt.coulomb_energy,
        r.receipt.uncorrected_exchange_energy, r.receipt.exchange_correction_energy], energies, atol=8e-11, rtol=0)
    nuclear = sum(_nuclear_reference(c)[0])
    assert r.receipt.energy == pytest.approx(nuclear+sum(build(d)[1]), abs=8e-11)
    assert r.plan.domain_live_bytes == domains.plan.domain_live_bytes
    assert r.receipt.fock_calls == r.solver_snapshot.fock_calls+1
    for field in ('modeled_peak_bytes', 'shell_calls', 'primitive_pairs', 'scalar_terms', 'kernel_terms'):
        assert getattr(r.plan, field) <= getattr(domains.plan, field)
    assert r.receipt.shell_calls <= r.plan.shell_calls
    assert r.receipt.primitive_pairs <= r.plan.primitive_pairs


@pytest.mark.parametrize('field,planned', [('memory_bytes', 'modeled_peak_bytes'),
    ('shell_calls', 'shell_calls'), ('primitive_quartets', 'primitive_pairs'),
    ('scalar_terms', 'scalar_terms'), ('kernel_terms', 'kernel_terms')])
def test_complete_fitted_admission_exact_and_one_short(field, planned):
    a, _ = arguments((2, 1, 1), (1, 0, 0))
    a['budget'].other_live_bytes = 12345
    p = manta.plan_periodic_ri_uhf_from_cutoffs(**a)
    setattr(a['budget'], field, getattr(p, planned))
    assert core._manta.run_periodic_ri_uhf_from_cutoffs(**a).converged
    setattr(a['budget'], field, getattr(p, planned)-1)
    for call in (manta.plan_periodic_ri_uhf_from_cutoffs, manta.build_periodic_ri_uhf_domains,
                 core._manta.run_periodic_ri_uhf_from_cutoffs):
        with pytest.raises((ValueError, RuntimeError), match='budget'):
            call(**a)


def test_complete_admission_before_empty_domain_and_auxiliary_ownership():
    a, _ = arguments(cutoff=1e-5)
    p = manta.plan_periodic_ri_uhf_from_cutoffs(**a)
    with pytest.raises(ValueError, match='no nonzero Hartree'):
        manta.build_periodic_ri_uhf_domains(**a)
    a['budget'].shell_calls = p.shell_calls-1
    with pytest.raises((ValueError, RuntimeError), match='budget'):
        manta.build_periodic_ri_uhf_domains(**a)
    a, _ = arguments((2, 1, 1), (1, 0, 0))
    original = manta.build_periodic_ri_uhf_domains(**a)
    shells = a['auxiliary'].shells()
    shells.append(core.ShellInfo(0, 0, True, [9.6], [1.], a['system'].unit_cell[0].xyz))
    a['auxiliary'] = vq.BasisSet(a['system'].unit_cell_molecule(), shells, 'ri-domain-larger-auxiliary', True)
    larger = manta.build_periodic_ri_uhf_domains(**a)
    # Domain receipt describes geometric supports. The method's fitted source
    # separately owns basis exponents, auxiliary images and metric controls.
    assert larger.receipt.identity == original.receipt.identity
    assert larger.plan.uhf.fock.retained_source_bytes > original.plan.uhf.fock.retained_source_bytes
    assert larger.plan.modeled_peak_bytes > original.plan.modeled_peak_bytes
    a['budget'].memory_bytes = original.plan.modeled_peak_bytes
    with pytest.raises((ValueError, RuntimeError), match='budget'):
        manta.build_periodic_ri_uhf_domains(**a)


def test_generated_fit_and_direct_domains_preserve_their_distinct_transfer_conventions():
    a, _ = arguments((3, 2, 1), (1, 1, 0), cutoff=2.)
    fitted = manta.build_periodic_ri_uhf_domains(**a)
    args = {key: value for key, value in a.items() if key != 'auxiliary'}
    direct = manta.build_periodic_uhf_domains(**args)
    nk = len(a['mesh'])
    for name in ('translations', 'reciprocal_vectors'):
        np.testing.assert_array_equal(getattr(fitted, name), getattr(direct, name))
    for i in range(nk*nk):
        flo, fhi = fitted.transfer_offsets[i:i+2]; dlo, dhi = direct.transfer_offsets[i:i+2]
        if i//nk == i%nk:
            assert fhi == flo and dhi > dlo
        else:
            np.testing.assert_array_equal(fitted.pair_vectors[flo:fhi], -direct.exchange_vectors[dlo:dhi])
    assert fitted.receipt.identity != direct.receipt.identity
    assert fitted.plan.exchange_candidates == direct.plan.exchange_candidates-nk*direct.plan.hartree_candidates


@pytest.mark.parametrize('correction', ['strict-zero-mode', 'BvK-ewald'])
def test_pair_complete_cells_and_independent_ao_auxiliary_image_gauges(correction):
    a, k = arguments((2, 1, 1), (1, 0, 0), correction=correction)
    atoms = [vq.Atom(1, p) for p in ([0., 0., 0.], [.9, .2, 0.], [-.2, 1.1, .1])]
    a['system'] = vq.PeriodicSystem(3, a['system'].lattice, atoms); a['system'].multiplicity = 2
    shells = a['basis'].shells()
    for i, shell in enumerate(shells):
        shell.atom_index = i; shell.origin = atoms[i].xyz
    a['basis'] = vq.BasisSet(a['system'].unit_cell_molecule(), shells, 'ri-domain-gauge', True)
    a['options'].scf.maximum_iterations = 3
    original = core._manta.run_periodic_ri_uhf_from_cutoffs(**a)
    old_domains = manta.build_periodic_ri_uhf_domains(**a)
    lattice = a['system'].lattice
    shifts = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]]) @ lattice.T
    for shell, delta in zip(shells, shifts):
        shell.origin = np.array(shell.origin)+delta
    image_system = vq.PeriodicSystem(3, lattice, [vq.Atom(1, s.origin) for s in shells]); image_system.multiplicity = 2
    a['basis'] = vq.BasisSet(image_system.unit_cell_molecule(), shells, 'ri-domain-images', True)
    auxiliary = a['auxiliary'].shells()
    for i, (shell, delta) in enumerate(zip(auxiliary, shifts[::-1])):
        shell.atom_index = i; shell.origin = np.array(shell.origin)+delta
    aux_system = vq.PeriodicSystem(3, lattice, [vq.Atom(1, s.origin) for s in auxiliary]); aux_system.multiplicity = 2
    a['auxiliary'] = vq.BasisSet(aux_system.unit_cell_molecule(), auxiliary, 'ri-domain-aux-images', True)
    domains = manta.build_periodic_ri_uhf_domains(**a)
    labels = _labels(domains.translations, np.linalg.inv(lattice))
    centers = [np.asarray(s.origin) for s in shells]
    for left, right in itertools.product(centers, repeat=2):
        for n in itertools.product(range(-3, 4), repeat=3):
            if np.linalg.norm(left-right-lattice@n) <= a['options'].pair_cutoff_bohr:
                assert n in labels
    assert len(domains.translations) > len(old_domains.translations)
    assert domains.receipt.identity != old_domains.receipt.identity
    moved = core._manta.run_periodic_ri_uhf_from_cutoffs(**a)
    phase = np.exp(1j*k@shifts.T); action = phase[:, :, None]*phase[:, None, :].conj()
    assert np.max(abs(moved.density.imag)) > .01
    for name in ('density', 'fock', 'overlap', 'hcore'):
        np.testing.assert_allclose(getattr(moved, name), getattr(original, name)*action, atol=8e-11, rtol=0)
    assert moved.receipt.energy == pytest.approx(original.receipt.energy, abs=8e-11)


def test_owned_exports_selected_provenance_and_python_budget(monkeypatch):
    a, _ = arguments((2, 1, 1), (1, 0, 0), correction='BvK-ewald')
    domains = manta.build_periodic_ri_uhf_domains(**a)
    for name in ('translations', 'reciprocal_vectors', 'pair_vectors', 'transfer_offsets'):
        original = getattr(domains, name)
        getattr(domains, name)[:] = 999
        np.testing.assert_array_equal(getattr(domains, name), original)
    a['options'].reciprocal_batch = 1
    assert manta.build_periodic_ri_uhf_domains(**a).receipt.identity == domains.receipt.identity
    records = []; monkeypatch.setattr('vibeqc.output.record', lambda *args, **kw: records.append(kw))
    budget = a.pop('budget'); engine = manta.IntegralEngineOptions(backend='manta', budget=budget)
    r = manta.run_periodic_ri_uhf_from_cutoffs(**a, integral_engine=engine)
    e = records[-1]['integral_provenance']; json.dumps(e, allow_nan=False)
    assert r.converged and len(records) == 1
    assert e['domain_identity'] == domains.receipt.identity
    assert e['domain_selection'].endswith('ri-ket-minus-bra;empty-pair-diagonal')
    assert e['domain_exchange_vectors'] == len(domains.pair_vectors)
    assert e['reciprocal_energy_cutoff'] == a['reciprocal_energy_cutoff']
    assert e['fitting']['metrics'][0]['projector_identity'] == r.receipt.fitting_info.metrics[0].projector_identity
    assert not e['symmetry_reduced'] and e['operator_kind'] == 'hf'
    assert budget.other_live_bytes == 0
    p = manta.plan_periodic_ri_uhf_from_cutoffs(**a, budget=budget)
    budget.memory_bytes = p.modeled_peak_bytes+e['python_reserved_bytes']
    assert manta.run_periodic_ri_uhf_from_cutoffs(**a, integral_engine=engine).converged
    budget.memory_bytes -= 1
    with pytest.raises(ValueError, match='memory budget'):
        manta.run_periodic_ri_uhf_from_cutoffs(**a, integral_engine=engine)


@pytest.mark.parametrize('field,value', [('reciprocal_energy_cutoff', 0.),
    ('reciprocal_energy_cutoff', float('nan')), ('reciprocal_energy_cutoff', float('inf')),
    ('reciprocal_energy_cutoff', 1e300), ('pair_cutoff_bohr', None), ('pair_cutoff_bohr', 0.),
    ('pair_cutoff_bohr', float('nan')), ('pair_cutoff_bohr', 1e300)])
def test_invalid_cutoffs_refuse(field, value):
    a, _ = arguments()
    if field == 'pair_cutoff_bohr':
        setattr(a['options'], field, value)
    else:
        a[field] = value
    with pytest.raises((ValueError, RuntimeError, OverflowError)):
        manta.build_periodic_ri_uhf_domains(**a)


def test_mismatched_physical_pair_cutoffs_refuse_before_empty_sphere():
    a, _ = arguments(cutoff=1e-5)
    a['options'].fitting.pair_cutoff_bohr += 1.
    with pytest.raises(ValueError, match='identical explicit HF and fitting pair cutoffs'):
        manta.build_periodic_ri_uhf_domains(**a)


def test_exact_sphere_boundary_empty_pair_and_large_mesh_admission():
    a, _ = arguments((2, 1, 1), (1, 0, 0), cutoff=.5)
    a['system'] = vq.PeriodicSystem(3, 2*np.pi*np.eye(3), a['system'].unit_cell)
    a['system'].multiplicity = 2
    domains = manta.build_periodic_ri_uhf_domains(**a)
    np.testing.assert_array_equal(np.sum(domains.reciprocal_vectors**2, axis=1), np.ones(6))
    assert len(domains.pair_vectors) == 4
    np.testing.assert_array_equal(np.sum(domains.pair_vectors**2, axis=1), np.full(4, .25))
    a['mesh'] = manta.RegularKMesh((10000, 10000, 1), (1, 1, 0))
    with pytest.raises((ValueError, RuntimeError, OverflowError), match='budget|overflow'):
        manta.plan_periodic_ri_uhf_from_cutoffs(**a)
    a['mesh'] = manta.RegularKMesh((2, 2, 1), (0, 0, 0))
    a['system'] = vq.PeriodicSystem(3, np.diag([10., 1., 1.]), a['system'].unit_cell)
    a['system'].multiplicity = 2
    with pytest.raises(ValueError, match='empty off-diagonal pair sphere'):
        manta.build_periodic_ri_uhf_domains(**a)
