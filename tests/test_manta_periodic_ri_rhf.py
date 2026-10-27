"""Restricted fitted SCF, independent finite-source oracle and public output."""
import dataclasses
import json
import tomllib

import numpy as np
import pytest
from scipy.linalg import eigh

import vibeqc as vq
from vibeqc import manta, _vibeqc_core as core
from vibeqc.guess_read import _spin_density_k_from_qvf_bloch_payload
from tests.test_manta_periodic_job import _qvf, _no_computed_artifacts
from tests.test_manta_periodic_ri_uhf import case as spin_case, reference as spin_reference
from tests.test_manta_periodic_ri_domains import explicit
from tests.test_manta_periodic_rhf import _nuclear_reference


def case(mesh=(1, 1, 1), shift=(0, 0, 0), species='He', correction='strict-zero-mode'):
    c, k = spin_case(mesh, shift, 'H' if species == 'He' else 'Li', correction)
    old = c['system']
    system = vq.PeriodicSystem(3, old.lattice, [vq.Atom(2 if species == 'He' else 4, old.unit_cell[0].xyz)])
    c['system'] = system
    c['basis'] = vq.BasisSet(system.unit_cell_molecule(), c['basis'].shells(), 'ri-rhf-orbital-test', False)
    c['auxiliary'] = vq.BasisSet(system.unit_cell_molecule(), c['auxiliary'].shells(), 'ri-rhf-auxiliary-test', True)
    o = manta.PeriodicRIRHFOptions()
    for name in ('alpha', 'nuclear_cutoff_bohr', 'pair_cutoff_bohr', 'reciprocal_batch', 'exchange_q0', 'fitting', 'ri'):
        setattr(o, name, getattr(c['options'], name))
    o.scf.maximum_iterations = 100
    c['options'] = o
    return c, k


def cutoff_case(*args, **kwargs):
    c, k = case(*args, **kwargs)
    return dict({key: c[key] for key in ('basis', 'auxiliary', 'system', 'mesh', 'options', 'budget')},
                reciprocal_energy_cutoff=6.), k


def job_case(directory, *args, **kwargs):
    a, k = cutoff_case(*args, **kwargs); o = a['options']
    job = dict(system=a['system'], basis=a['basis'], kpoints=a['mesh'], method='RHF',
        integral_engine=manta.IntegralEngineOptions(backend='manta', budget=a['budget']),
        coulomb=manta.CoulombBuildOptions(algorithm='ri', auxiliary_basis=a['auxiliary'], metric=o.ri),
        exchange=manta.ExchangeBuildOptions(algorithm='ri'),
        periodic_integrals=manta.PeriodicIntegralOptions(a['reciprocal_energy_cutoff'],
            o.pair_cutoff_bohr, o.nuclear_cutoff_bohr, alpha=o.alpha, reciprocal_batch=o.reciprocal_batch),
        periodic_fitting=manta.PeriodicFittingOptions(o.fitting.omega, o.fitting.auxiliary_cutoff_bohr),
        use_diis=False, output=directory/'selected'/'job', record_hostname=False, structured_log=True,
        max_iter=100, conv_tol_energy=1e-12,
        exchange_exxdiv='ewald' if o.exchange_q0 == 'BvK-ewald' else 'none')
    return job, a, k


def reference(c, k):
    # Libint metric/3c arrays, NumPy factors/contractions and independent H/S.
    # Restriction is imposed on the density, before the contraction and solve.
    s, h, spin_build, xi = spin_reference(c, k)
    def build(d):
        f, energies = spin_build(np.stack([.5*d, .5*d]))
        np.testing.assert_array_equal(f[0], f[1])
        return f[0], energies
    return s, h, build, xi


def occupy(f, s, occupied):
    c = np.array([eigh(fp, sp)[1][:, :occupied] for fp, sp in zip(f, s)])
    return 2*c@c.conj().swapaxes(-1, -2)


def scf_reference(s, h, build, occupied):
    d = occupy(h, s, occupied)
    for _ in range(250):
        updated = occupy(build(d)[0], s, occupied)
        if np.max(abs(updated-d)) < 1e-12:
            return updated
        d = updated
    raise AssertionError('Independent restricted finite-source SCF did not converge')


def plan(c):
    return manta.plan_periodic_ri_rhf(c['basis'], c['auxiliary'], c['system'], c['mesh'],
        len(c['reciprocal_vectors']), int(max(np.diff(c['transfer_offsets']))), len(c['translations']),
        c['options'], c['budget'])


@pytest.mark.parametrize('species', ['He', 'Be'])
@pytest.mark.parametrize('shape,shift', [((1, 1, 1), (0, 0, 0)), ((2, 1, 1), (0, 0, 0)), ((2, 1, 1), (1, 0, 0))])
@pytest.mark.parametrize('correction', ['strict-zero-mode', 'BvK-ewald'])
def test_restricted_solver_matches_independent_scf_and_every_energy(species, shape, shift, correction):
    c, k = case(shape, shift, species, correction)
    r = core._manta.run_periodic_ri_rhf(**c)
    assert r.converged and r.receipt.confirmed
    assert isinstance(r.solver_snapshot, core._BoundedPeriodicRHFSnapshot)
    s, h, build, xi = reference(c, k)
    d = scf_reference(s, h, build, r.plan.electrons//2)
    np.testing.assert_allclose(r.density, d, atol=5e-9, rtol=0)
    np.testing.assert_allclose(r.overlap, s, atol=4e-12, rtol=0)
    np.testing.assert_allclose(r.hcore, h, atol=4e-12, rtol=0)
    np.testing.assert_allclose(r.fock, build(r.density)[0], atol=8e-11, rtol=0)
    np.testing.assert_allclose([r.receipt.one_electron_energy, r.receipt.coulomb_energy,
        r.receipt.uncorrected_exchange_energy, r.receipt.exchange_correction_energy], build(r.density)[1], atol=8e-11, rtol=0)
    assert r.receipt.energy == pytest.approx(sum(_nuclear_reference(c)[0])+sum(build(d)[1]), abs=8e-11)
    assert r.receipt.madelung == pytest.approx(xi, abs=3e-12)
    np.testing.assert_allclose(r.density@s@r.density, 2*r.density, atol=5e-12, rtol=0)
    np.testing.assert_allclose(np.einsum('kij,kji->k', r.density, s), r.plan.electrons, atol=3e-12, rtol=0)
    assert r.receipt.fock_calls == r.solver_snapshot.fock_calls+1
    assert r.receipt.shell_calls <= r.plan.shell_calls and r.receipt.primitive_pairs <= r.plan.primitive_pairs
    assert len(r.receipt.fitting_info.metrics) == 1+len(k)*(len(k)-1)
    assert r.density.shape == (len(k), c['basis'].nbasis, c['basis'].nbasis)
    assert r.coefficients.shape == (len(k), c['basis'].nbasis, r.solver_memory.retained_rank)


@pytest.mark.parametrize('correction', ['strict-zero-mode', 'BvK-ewald'])
def test_capped_restricted_recurrence_and_immutable_fit(correction):
    c, k = case((2, 1, 1), (1, 0, 0), 'Be', correction)
    s, h, build, _ = reference(c, k); d = occupy(h, s, 2)
    c['options'].scf.maximum_iterations = 4
    final = core._manta.run_periodic_ri_rhf(**c)
    for iteration in range(1, 5):
        c['options'].scf.maximum_iterations = iteration
        r = core._manta.run_periodic_ri_rhf(**c)
        assert not r.converged and not r.receipt.confirmed
        np.testing.assert_allclose(r.density, d, atol=8e-11, rtol=0)
        np.testing.assert_allclose(r.fock, build(d)[0], atol=8e-11, rtol=0)
        assert r.receipt.fock_calls == iteration
        assert final.scf_history[iteration-1].energy_per_cell == r.solver_snapshot.energy_per_cell
        assert r.receipt.source_identity == final.receipt.source_identity
        assert [m.projector_identity for m in r.receipt.fitting_info.metrics] == [m.projector_identity for m in final.receipt.fitting_info.metrics]
        d = occupy(build(d)[0], s, 2)


CAPS = [('memory_bytes', 'modeled_peak_bytes'), ('shell_calls', 'shell_calls'),
        ('primitive_quartets', 'primitive_pairs'), ('scalar_terms', 'scalar_terms'), ('kernel_terms', 'kernel_terms')]


@pytest.mark.parametrize('field,planned', CAPS)
@pytest.mark.parametrize('correction', ['strict-zero-mode', 'BvK-ewald'])
@pytest.mark.parametrize('generated', [False, True])
def test_restricted_resource_admission_includes_spin_callback_and_real_solver(field, planned, correction, generated):
    if generated:
        c, _ = cutoff_case((2, 1, 1), (1, 0, 0), correction=correction)
        planner, run = manta.plan_periodic_ri_rhf_from_cutoffs, core._manta.run_periodic_ri_rhf_from_cutoffs
    else:
        c, _ = case((2, 1, 1), (1, 0, 0), correction=correction)
        planner, run = lambda **kw: plan(kw), core._manta.run_periodic_ri_rhf
    c['budget'].other_live_bytes = 12345
    p = planner(**c); setattr(c['budget'], field, getattr(p, planned))
    r = run(**c); assert r.converged
    assert r.solver_memory.provider_retained_bytes == r.plan.fock.retained_source_bytes
    assert r.solver_memory.density_bytes == 16*r.plan.momenta*r.plan.n_basis**2
    setattr(c['budget'], field, getattr(p, planned)-1)
    if not generated: c['reciprocal_vectors'][:] = np.nan
    with pytest.raises((ValueError, RuntimeError), match='budget'): run(**c)


@pytest.mark.parametrize('correction', ['strict-zero-mode', 'BvK-ewald'])
def test_truncated_and_empty_fit_have_independent_restricted_oracle(correction):
    c, k = case((2, 1, 1), (1, 0, 0), correction=correction)
    identities = []
    for cutoff, rank in ((1e-12, 4), (10., 1), (1e9, 0)):
        c['options'].ri.absolute_cutoff = cutoff
        r = core._manta.run_periodic_ri_rhf(**c); assert r.converged
        s, h, build, _ = reference(c, k); d = scf_reference(s, h, build, 1)
        np.testing.assert_allclose(r.density, d, atol=5e-9, rtol=0)
        np.testing.assert_allclose(r.fock, build(r.density)[0], atol=8e-11, rtol=0)
        assert all(m.rank == rank for m in r.receipt.fitting_info.metrics)
        identities.append(r.receipt.source_identity)
    assert len(set(identities)) == 3


@pytest.mark.parametrize('field,value,match', [('dim', 2, 'neutral'), ('charge', 1, 'neutral'), ('multiplicity', 3, 'singlet')])
def test_unqualified_restricted_profile_refuses_before_payload(field, value, match):
    c, _ = case(); setattr(c['system'], field, value); c['pair_vectors'][:] = np.nan
    with pytest.raises(ValueError, match=match): core._manta.run_periodic_ri_rhf(**c)


def test_closed_shell_bvk_factors_and_same_density_identity():
    c, _ = case((2, 1, 1), (1, 0, 0), 'Be')
    c['options'].scf.maximum_iterations = 1
    strict = core._manta.run_periodic_ri_rhf(**c)
    c['options'].exchange_q0 = 'BvK-ewald'; corrected = core._manta.run_periodic_ri_rhf(**c)
    np.testing.assert_array_equal(strict.density, corrected.density)
    xi = corrected.receipt.madelung
    np.testing.assert_allclose(corrected.fock-strict.fock, -.5*xi*strict.overlap@strict.density@strict.overlap, atol=2e-14, rtol=0)
    assert corrected.receipt.energy-strict.receipt.energy == pytest.approx(-.5*xi*strict.plan.electrons, abs=8e-12)
    assert strict.receipt.final_density_identity == corrected.receipt.final_density_identity
    assert strict.receipt.source_identity != corrected.receipt.source_identity
    assert strict.receipt.final_fock_identity != corrected.receipt.final_fock_identity
    for name in ('density', 'fock', 'coefficients', 'orbital_energies', 'overlap', 'hcore', 'momenta'):
        original = getattr(corrected, name).copy(); getattr(corrected, name)[:] = np.nan
        np.testing.assert_array_equal(getattr(corrected, name), original)


@pytest.mark.parametrize('species', ['He', 'Be'])
@pytest.mark.parametrize('shape,shift', [((1, 1, 1), (0, 0, 0)), ((2, 1, 1), (1, 0, 0))])
@pytest.mark.parametrize('correction', ['strict-zero-mode', 'BvK-ewald'])
def test_public_restricted_job_uses_same_cutoff_source_and_saves_evaluated_density(tmp_path, species, shape, shift, correction):
    from vibeqc.output import validate_qvf
    job, a, k = job_case(tmp_path, shape, shift, species, correction)
    domains = manta.build_periodic_ri_rhf_domains(**a); c = explicit(a, domains)
    r = vq.run_periodic_job(**job)
    assert isinstance(r, manta.PeriodicMANTAResult) and r.converged
    raw = core._manta.run_periodic_ri_rhf(**c)
    np.testing.assert_array_equal(raw.density, r.native.density)
    assert raw.receipt.source_identity == r.native.receipt.source_identity
    s, h, build, _ = reference(c, k); d = scf_reference(s, h, build, raw.plan.electrons//2)
    assert r.energy == pytest.approx(sum(_nuclear_reference(c)[0])+sum(build(d)[1]), abs=8e-11)
    e = r.integral_provenance; json.dumps(e, allow_nan=False)
    assert e['method'] == 'RHF' and e['coulomb_algorithm'] == e['exchange_algorithm'] == 'ri'
    assert e['spin_labels'] == ['restricted-total'] and e['operator_layout'] == 'k,ao,ao'
    assert e['spin_electrons_per_cell'] == [raw.plan.electrons]
    assert e['domain_identity'] == domains.receipt.identity and not e['symmetry_reduced']
    assert e['fitting_density_embedding'] == 'equal-spin-halves-of-restricted-total'
    path = job['output'].with_suffix('.qvf'); assert validate_qvf(path)['valid']
    saved = _spin_density_k_from_qvf_bloch_payload(path)
    np.testing.assert_array_equal(saved, np.stack([.5*r.native.density]*2))
    manifest = tomllib.loads(job['output'].with_suffix('.system').read_text())
    assert manifest['run']['jk_method_executed'] == 'finite-3d-ri-rhf'
    assert manifest['run']['fitting_source_identity'] == e['fitting']['source_identity']
    assert 'Finite periodic RI RHF: manta' in job['output'].with_suffix('.out').read_text()
    assert 'Berkelbach' in job['output'].with_suffix('.references').read_text()
    assert job['integral_engine'].budget.other_live_bytes == 0


@pytest.mark.parametrize('correction', ['strict-zero-mode', 'BvK-ewald'])
def test_restricted_exports_use_landed_fock_build_and_exact_capped_restart(tmp_path, correction):
    from tests.test_manta_periodic_ri_job import _as_fock_build
    identities = []
    for limit in (1, 3, 100):
        job, _, _ = job_case(tmp_path/str(limit), (2, 1, 1), (1, 0, 0), 'Be', correction)
        job['max_iter'] = limit
        r = vq.run_periodic_job(**job); native = r.native; state = _as_fock_build(r)
        identities.append(state.source.identity)
        assert state.energy.total == pytest.approx(r.energy, abs=1e-13)
        assert state.generation == native.solver_snapshot.evaluated_iteration
        np.testing.assert_allclose(state.metric_electron_counts, native.plan.electrons, atol=2e-12, rtol=0)
        for index, channel in enumerate(state.channels):
            assert channel.operator.metadata.spin == 'restricted-total'
            np.testing.assert_array_equal(channel.density, native.density[index])
            np.testing.assert_allclose(channel.operator.apply(np.eye(native.plan.n_basis)), native.fock[index], atol=0, rtol=0)
        saved = _spin_density_k_from_qvf_bloch_payload(job['output'].with_suffix('.qvf'))
        np.testing.assert_array_equal(saved, np.stack([.5*native.density]*2))
        if limit < 100:
            assert not r.converged
            c = native.coefficients[:, :, :native.plan.electrons//2]
            assert np.max(abs(2*c@c.conj().swapaxes(-1, -2)-native.density)) > 1e-5
            assert _qvf(job['output'].with_suffix('.qvf'))['provenance']['run_status'] == 'failed'
    assert len(set(identities)) == 1


@pytest.mark.parametrize('field,planned', CAPS)
def test_public_restricted_complete_output_and_evidence_caps(tmp_path, field, planned):
    job, a, _ = job_case(tmp_path, (2, 1, 1), (1, 0, 0), 'Be')
    r = vq.run_periodic_job(**job); e = r.integral_provenance
    a['budget'].other_live_bytes = e['python_reserved_bytes']+e['job_reserved_bytes']
    p = manta.plan_periodic_ri_rhf_from_cutoffs(**a)
    a['budget'].other_live_bytes = 0
    setattr(a['budget'], field, getattr(p, planned))
    assert vq.run_periodic_job(**job).converged
    setattr(a['budget'], field, getattr(p, planned)-1); job['output'] = tmp_path/'short'/'job'
    with pytest.raises((ValueError, RuntimeError), match='budget'): vq.run_periodic_job(**job)
    _no_computed_artifacts(job)


@pytest.mark.parametrize('fault', ['odd-electrons', 'no-virtual'])
def test_restricted_job_rejects_unqualified_choice_before_output(tmp_path, fault):
    job, _, _ = job_case(tmp_path)
    if fault == 'odd-electrons':
        job['system'] = vq.PeriodicSystem(3, job['system'].lattice, [vq.Atom(1, job['system'].unit_cell[0].xyz)])
    else:
        job['basis'] = vq.BasisSet(job['system'].unit_cell_molecule(), job['basis'].shells()[:1], 'occupied-only-test', False)
    with pytest.raises((ValueError, NotImplementedError)): vq.run_periodic_job(**job)
    _no_computed_artifacts(job)


@pytest.mark.parametrize('correction', ['strict-zero-mode', 'BvK-ewald'])
def test_restricted_and_singlet_unrestricted_share_fitting_source_without_sharing_solver(correction):
    a, _ = cutoff_case((2, 1, 1), (1, 0, 0), 'Be', correction)
    r = core._manta.run_periodic_ri_rhf_from_cutoffs(**a)
    o = manta.PeriodicRIUHFOptions()
    for name in ('alpha', 'nuclear_cutoff_bohr', 'pair_cutoff_bohr', 'reciprocal_batch', 'exchange_q0', 'fitting', 'ri'):
        setattr(o, name, getattr(a['options'], name))
    o.scf.maximum_iterations = 100
    u = core._manta.run_periodic_ri_uhf_from_cutoffs(**dict(a, options=o))
    assert r.converged and u.converged
    np.testing.assert_allclose(r.density, u.density.sum(axis=0), atol=5e-9, rtol=0)
    np.testing.assert_allclose(r.fock, u.fock[0], atol=5e-9, rtol=0)
    assert r.receipt.energy == pytest.approx(u.receipt.energy, abs=1e-11)
    assert r.receipt.fitting_info.source_identity == u.receipt.fitting_info.source_identity
    assert r.receipt.domain_identity == u.receipt.domain_identity
    assert r.receipt.source_identity != u.receipt.source_identity
    assert r.solver_memory.peak_owned_numerical_bytes < u.solver_memory.peak_owned_numerical_bytes
    assert r.plan.modeled_peak_bytes < u.plan.modeled_peak_bytes


def test_restricted_overlap_rank_reduction_preserves_projected_space_and_fock_contract(tmp_path):
    from tests.test_manta_periodic_ri_job import _as_fock_build
    job, a, k = job_case(tmp_path, (2, 1, 1), (1, 0, 0))
    shells = a['basis'].shells(); shells.append(shells[0])
    b = vq.BasisSet(a['system'].unit_cell_molecule(), shells, 'ri-rhf-redundant-orbital', False)
    a['basis'] = job['basis'] = b
    r = vq.run_periodic_job(**job); assert r.converged and r.native.solver_memory.retained_rank == 2
    c = explicit(a, manta.build_periodic_ri_rhf_domains(**a)); s, h, build, _ = reference(c, k)
    values, vectors = np.linalg.eigh(s); x = vectors[:, :, -2:]/np.sqrt(values[:, None, -2:])
    def projected(f):
        values, vectors = np.linalg.eigh(x.conj().swapaxes(-1, -2)@f@x)
        occupied = x@vectors[:, :, :1]
        return 2*occupied@occupied.conj().swapaxes(-1, -2)
    d = projected(h)
    for _ in range(100):
        updated = projected(build(d)[0])
        if np.max(abs(d-updated)) < 1e-12: break
        d = updated
    else: pytest.fail('Projected reference failed to converge')
    np.testing.assert_allclose(r.native.density, updated, atol=5e-9, rtol=0)
    np.testing.assert_allclose(r.native.fock, build(r.native.density)[0], atol=8e-11, rtol=0)
    state = _as_fock_build(r)
    assert all(ch.operator.frame.X.shape == (3, 2) for ch in state.channels)
    np.testing.assert_allclose(state.metric_electron_counts, 2., atol=2e-12, rtol=0)


@pytest.mark.parametrize('correction', ['strict-zero-mode', 'BvK-ewald'])
def test_multicenter_ao_auxiliary_gauges_and_nuclear_frame_output(tmp_path, correction):
    job, a, k = job_case(tmp_path/'base', (2, 1, 1), (1, 0, 0), 'Be', correction)
    atoms = [vq.Atom(z, xyz) for z, xyz in zip((2, 1, 1), ([0., 0., 0.], [.9, .2, 0.], [-.2, 1.1, .1]))]
    system = vq.PeriodicSystem(3, a['system'].lattice, atoms)
    shells = a['basis'].shells(); aux = a['auxiliary'].shells()
    for i, shell in enumerate(shells): shell.atom_index = i; shell.origin = atoms[i].xyz
    for shell in aux: shell.origin = atoms[0].xyz
    basis = vq.BasisSet(system.unit_cell_molecule(), shells, 'ri-rhf-multicenter', True)
    auxiliary = vq.BasisSet(system.unit_cell_molecule(), aux, 'ri-rhf-multicenter-aux', True)
    job.update(system=system, basis=basis, max_iter=3)
    job['coulomb'] = dataclasses.replace(job['coulomb'], auxiliary_basis=auxiliary)
    original = vq.run_periodic_job(**job)
    shifts = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]]) @ system.lattice.T
    for shell, delta in zip(shells, shifts): shell.origin = np.array(shell.origin)+delta
    images = vq.PeriodicSystem(3, system.lattice, [vq.Atom(z, shell.origin) for z, shell in zip((2, 1, 1), shells)])
    job['basis'] = vq.BasisSet(images.unit_cell_molecule(), shells, 'ri-rhf-ao-images', True)
    for i, (shell, delta) in enumerate(zip(aux, shifts[::-1])):
        shell.atom_index = i; shell.origin = np.array(shell.origin)+delta
    images = vq.PeriodicSystem(3, system.lattice, [vq.Atom(z, shell.origin) for z, shell in zip((2, 1, 1), aux)])
    job['coulomb'] = dataclasses.replace(job['coulomb'], auxiliary_basis=vq.BasisSet(images.unit_cell_molecule(), aux, 'ri-rhf-aux-images', True))
    job['output'] = tmp_path/'images'/'job'
    moved = vq.run_periodic_job(**job)
    phase = np.exp(1j*k@shifts.T); transport = phase[:, :, None]*phase[:, None, :].conj()
    assert np.max(abs(moved.native.density.imag)) > .01
    for name in ('density', 'fock', 'overlap', 'hcore'):
        np.testing.assert_allclose(getattr(moved.native, name), getattr(original.native, name)*transport, atol=2e-10, rtol=0)
    assert moved.energy == pytest.approx(original.energy, abs=1e-10)
    from tests.test_manta_periodic_ri_job import _as_fock_build
    state = _as_fock_build(moved)
    np.testing.assert_allclose(state.metric_electron_counts, 4., atol=5e-12, rtol=0)
    for ik, channel in enumerate(state.channels):
        np.testing.assert_allclose(channel.operator.apply(np.eye(3, dtype=complex)),
                                   moved.native.fock[ik], atol=1e-13, rtol=0)
    assert moved.integral_provenance['qvf_density_gauge'] == 'nuclear-centers'
    saved = _spin_density_k_from_qvf_bloch_payload(job['output'].with_suffix('.qvf'))
    np.testing.assert_allclose(saved, np.stack([.5*original.native.density]*2), atol=1e-10, rtol=0)
    assert moved.native.receipt.fitting_info.frame_identity != original.native.receipt.fitting_info.frame_identity


def test_selected_explicit_entry_has_restricted_citations_and_admitted_evidence(monkeypatch):
    from vibeqc import output
    c, _ = case((2, 1, 1), (1, 0, 0), correction='BvK-ewald')
    budget = c.pop('budget'); records = []
    monkeypatch.setattr(output, 'record', lambda *args, **kwargs: records.append(kwargs))
    engine = manta.IntegralEngineOptions(backend='manta', budget=budget)
    r = manta.run_periodic_ri_rhf(**c, integral_engine=engine); assert r.converged
    e = records[-1]['integral_provenance']; json.dumps(e, allow_nan=False)
    assert e['citation_methods'] == ['manta_periodic_ri_rhf', 'manta_periodic_madelung']
    assert e['python_reserved_bytes'] == 32768+12288*3
    assert e['fitting']['source_identity'] == r.receipt.fitting_info.source_identity
    assert budget.other_live_bytes == 0
    budget.memory_bytes = e['python_reserved_bytes']-1
    with pytest.raises(ValueError, match='provenance memory budget'):
        manta.run_periodic_ri_rhf(**c, integral_engine=engine)


def test_restricted_fit_rejects_inconsistent_pair_cutoff_and_negative_metric():
    import itertools
    c, _ = case()
    c['options'].fitting.pair_cutoff_bohr = 2.
    with pytest.raises(ValueError, match='identical explicit'): core._manta.run_periodic_ri_rhf(**c)
    c['options'].fitting.pair_cutoff_bohr = 3.; c['options'].fitting.auxiliary_cutoff_bohr = 5.
    base = np.array(list(itertools.product(range(-1, 2), repeat=3))) @ c['system'].reciprocal_lattice().T
    c['reciprocal_vectors'] = base[np.any(base != 0., axis=1)].copy()
    with pytest.raises(RuntimeError, match='significant negative eigenvalue'): core._manta.run_periodic_ri_rhf(**c)


def _ledger_evidence(one=3.9e-13, nuc=1.9e-17, exc=3.0e-10, envelope='env-a', nuclear_envelope=None, qualified=True):
    return dict(
        one_electron_accuracy=dict(qualified=qualified, kinetic_energy_error_bound=one, density_envelope_identity=envelope,
            certificate_identity='c1', excludes=['nuclear', 'two-electron'], images=dict(maximum_kinetic_error=2e-13, maximum_overlap_error=1e-13)),
        nuclear_accuracy=dict(qualified=True, total_energy_error_bound=nuc, density_envelope_identity=nuclear_envelope or envelope,
            certificate_identity='c2', excludes=['kinetic', 'two-electron'], nuclear=dict(maximum_potential_error=2.5e-12)),
        exchange_accuracy=dict(qualified=True, energy_error_bound=exc, maximum_error=8.9e-12, certificate_identity='c3',
            excludes=['madelung-numerics', 'fitting-metric'], exchange=dict(density_envelope_identity='occ-a')),
        fitting=dict(accuracy=dict(enabled=False)))


def test_fixed_density_ledger_sums_disjoint_components_and_refuses_mixed_envelopes():
    ledger = manta.compose_fixed_density_ledger(_ledger_evidence(), method='uhf')
    assert ledger['qualified'] and ledger['claimed_components'] == ['exact-exchange', 'kinetic', 'nuclear-attraction-and-repulsion']
    assert ledger['fixed_density_energy_error_bound'] == pytest.approx(3.9e-13+1.9e-17+3.0e-10, rel=1e-12)
    assert 'density-relaxation' in ledger['excludes'] and 'madelung-numerics' in ledger['excludes']
    assert 'hartree-omitted-domains' in ledger['excludes']
    assert not {'kinetic', 'nuclear', 'two-electron'} & set(ledger['excludes'])
    assert ledger['relaxation_delimiter']['qualification'] == 'not-evaluated'
    assert [c['requested'] for c in ledger['components']] == [True, True, True, False]
    json.dumps(ledger, allow_nan=False)
    mixed = manta.compose_fixed_density_ledger(_ledger_evidence(nuclear_envelope='env-b'), method='uhf')
    assert not mixed['qualified'] and mixed['fixed_density_energy_error_bound'] is None
    assert any('different canonical density envelopes' in r for r in mixed['reasons'])
    unqualified = manta.compose_fixed_density_ledger(_ledger_evidence(qualified=False), method='uhf')
    assert not unqualified['qualified'] and 'one_electron_accuracy is not qualified' in unqualified['reasons']
    empty = manta.compose_fixed_density_ledger(dict(fitting=dict(accuracy=dict(enabled=False))), method='uhf')
    assert not empty['qualified'] and empty['fixed_density_energy_error_bound'] is None
    fitted = manta.compose_fixed_density_ledger(dict(_ledger_evidence(), fitting=dict(accuracy=dict(
        enabled=True, qualified=True, energy_bound=4e-9, certificate_identity='f'))), method='uhf')
    assert fitted['qualified'] and fitted['fixed_density_energy_error_bound'] == pytest.approx(3.9e-13+1.9e-17+3.0e-10+4e-9, rel=1e-12)


@pytest.mark.parametrize('terms', [
    (1., 2.**-53, 2.**-53, 2.**-53),
    (2.**900, 2.**847, 2.**-1022, 0.),
    (5e-324, 5e-324, 5e-324, 5e-324),
    (0., 0., 0., 0.),
    (float.fromhex('0x1.fffffffffffffp+1023'), 0., 0., 0.),
])
def test_fixed_density_ledger_encloses_exact_stored_sum(terms):
    from fractions import Fraction
    import itertools
    for values in set(itertools.permutations(terms)):
        evidence = _ledger_evidence(*values[:3])
        evidence['fitting']['accuracy'] = dict(enabled=True, qualified=True, energy_bound=values[3])
        ledger = manta.compose_fixed_density_ledger(evidence, method='uhf')
        assert ledger['qualified']
        exact = sum(map(Fraction.from_float, values), Fraction())
        assert Fraction.from_float(ledger['fixed_density_energy_error_bound']) >= exact
        if sum(v != 0. for v in values) <= 1:
            assert ledger['fixed_density_energy_error_bound'] == float(exact)
        json.dumps(ledger, allow_nan=False)


@pytest.mark.parametrize('certificate,key', [
    ('one_electron_accuracy', 'kinetic_energy_error_bound'),
    ('nuclear_accuracy', 'total_energy_error_bound'),
    ('exchange_accuracy', 'energy_error_bound'),
    ('fitting_accuracy', 'energy_bound'),
])
@pytest.mark.parametrize('invalid', [-1., float('nan'), float('inf'), -float('inf'), None])
def test_fixed_density_ledger_refuses_invalid_energy_bounds(certificate, key, invalid):
    evidence = _ledger_evidence()
    if certificate == 'fitting_accuracy':
        cert = evidence['fitting']['accuracy'] = dict(enabled=True, qualified=True)
    else:
        cert = evidence[certificate]
    cert[key] = invalid
    ledger = manta.compose_fixed_density_ledger(evidence, method='rhf',
        orbital_energies=[[-1., 1.]], overlaps=[np.eye(2)], n_occupied=1, retained_rank=2)
    assert not ledger['qualified'] and ledger['fixed_density_energy_error_bound'] is None
    assert any(certificate in reason and 'finite and nonnegative' in reason for reason in ledger['reasons'])
    assert ledger['relaxation_delimiter']['qualification'] == 'not-evaluated'
    json.dumps(ledger, allow_nan=False)


def test_fixed_density_ledger_refuses_overflow_and_invalid_matrix_bound():
    ledger = manta.compose_fixed_density_ledger(_ledger_evidence(1e308, 1e308), method='uhf')
    assert not ledger['qualified'] and ledger['fixed_density_energy_error_bound'] is None
    assert 'fixed-density component sum has no finite outward bound' in ledger['reasons']
    json.dumps(ledger, allow_nan=False)
    evidence = _ledger_evidence()
    evidence['exchange_accuracy']['maximum_error'] = -1.
    ledger = manta.compose_fixed_density_ledger(evidence, method='rhf',
        orbital_energies=[[-1., 1.]], overlaps=[np.eye(2)], n_occupied=1, retained_rank=2)
    assert not ledger['qualified'] and ledger['fixed_density_energy_error_bound'] is None
    assert ledger['relaxation_delimiter']['qualification'] == 'not-evaluated'
    json.dumps(ledger, allow_nan=False)


def test_relaxation_delimiter_matches_the_explicit_closed_shell_formula():
    # Two k points, three retained functions, one occupied orbital; a plain
    # overlap with smallest eigenvalue 0.5 and distinct gaps per k point.
    energies = [np.array([-1.0, 0.2, 0.9]), np.array([-0.8, 0.5, 1.1])]
    s = np.diag([1.0, 0.75, 0.5]); overlaps = [s, s]
    bounds = dict(kinetic=2e-13, nuclear=2.5e-12, exchange=8.9e-12)
    d = manta.relaxation_delimiter(bounds, energies, overlaps, n_occupied=1, retained_rank=3)
    entry = sum(bounds.values()); frob = 3*entry; mo = frob/0.5; gap = min(1.2, 1.3)
    assert d['applies'] and d['minimum_gap'] == pytest.approx(gap) and d['per_k_gaps'] == pytest.approx([1.2, 1.3])
    assert d['overlap_retained_minimum'] == pytest.approx(0.5) and d['fock_mo_frobenius_bound'] == pytest.approx(mo)
    assert d['energy_estimate'] == pytest.approx(2*mo*mo/gap, rel=1e-12) and not d['is_certificate']
    # No virtual orbital in the retained frame: nothing to estimate.
    none = manta.relaxation_delimiter(bounds, [np.array([-1.0])], [np.eye(1)], n_occupied=1, retained_rank=1)
    assert not none['applies'] and none['energy_estimate'] is None
    # A perturbation comparable to the gap leaves the perturbative regime.
    big = manta.relaxation_delimiter(dict(exchange=1.0), energies, overlaps, n_occupied=1, retained_rank=3)
    assert not big['applies'] and big['energy_estimate'] is not None
    json.dumps(d, allow_nan=False)


def test_compact_profile_ledger_composes_its_certificates_and_delimits_relaxation(tmp_path):
    from benchmarks.manta_release_controls import make_job
    job = make_job(tmp_path, 'compact-tight')
    r = vq.run_periodic_job(**job)
    assert r.converged
    e = r.integral_provenance; json.dumps(e, allow_nan=False)
    ledger = e['component_ledger']
    assert ledger['qualified'] and not ledger['reasons']
    expected = (e['one_electron_accuracy']['kinetic_energy_error_bound'] + e['nuclear_accuracy']['total_energy_error_bound']
                + e['exchange_accuracy']['energy_error_bound'])
    assert ledger['fixed_density_energy_error_bound'] == pytest.approx(expected, rel=1e-12)
    assert ledger['claimed_components'] == ['exact-exchange', 'kinetic', 'nuclear-attraction-and-repulsion']
    assert ledger['components'][3] == dict(certificate='fitting_accuracy', component='coulomb-fitting', requested=False,
        note='exact finite coupling or no coupling target; metric inverse certificates feed this component only')
    d = ledger['relaxation_delimiter']
    assert d['qualification'] == 'uncoupled-second-order-estimate' and not d['is_certificate']
    assert d['applies'] and d['minimum_gap'] == pytest.approx(e['global_band_gap'], rel=1e-9)
    assert d['energy_estimate'] < ledger['fixed_density_energy_error_bound']
    assert set(d['fock_entry_error_bounds']) == {'kinetic', 'nuclear-attraction-and-repulsion', 'exact-exchange'}
    assert 'density-relaxation' in ledger['excludes'] and ledger['scope'].endswith('not-a-relaxed-energy-tolerance')
    assert 'hartree-omitted-domains' in ledger['excludes'] and 'kinetic' not in ledger['excludes']
