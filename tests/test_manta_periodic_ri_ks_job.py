"""Public fitted KS: functional state, joint admission and durable output."""
import dataclasses
import json
import tomllib

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta, _vibeqc_core as core
from vibeqc.guess_read import _spin_density_k_from_qvf_bloch_payload
from tests.test_manta_periodic_ri_ks_domains import arguments, explicit, function, PROFILES, ARRAYS
from tests.test_manta_periodic_ri_ks import reference, independent_scf, components, CAPS, as_fock_build
from tests.test_manta_periodic_global_hybrid_job import output_evidence
from tests.test_manta_periodic_job import _no_computed_artifacts
from tests.test_manta_periodic_rhf import _nuclear_reference


def job_case(directory, profile='pure', spins=1, gamma=False, empty=False, stratmann=False, correction='BvK-ewald'):
    kw = dict(mesh=(1, 1, 1), shift=(0, 0, 0)) if gamma else {}
    args, a, k, old = arguments(profile, spins, True, species='H' if empty else None, **kw)
    o, g = args['options'].ks, args['grid_options']; g.stratmann = stratmann
    if profile != 'pure': args['options'].exchange_q0 = correction
    j = 'exact' if profile == 'K' else 'ri'; exchange = 'none' if profile == 'pure' else 'exact' if profile == 'J' else 'ri'
    fit = args['options'].fitting
    job = dict(system=args['system'], basis=args['basis'], kpoints=args['mesh'],
        method='RKS' if spins == 1 else 'UKS', functional=args['functional'].name,
        integral_engine=manta.IntegralEngineOptions(backend='manta', budget=args['budget']),
        coulomb=manta.CoulombBuildOptions(algorithm=j, auxiliary_basis=args['auxiliary'], metric=args['options'].ri),
        exchange=manta.ExchangeBuildOptions(algorithm=exchange),
        periodic_integrals=manta.PeriodicIntegralOptions(args['reciprocal_energy_cutoff'], o.pair_cutoff_bohr,
            o.nuclear_cutoff_bohr, alpha=o.alpha, reciprocal_batch=o.reciprocal_batch),
        periodic_fitting=manta.PeriodicFittingOptions(fit.omega, fit.auxiliary_cutoff_bohr, fit.integral_screen_error, fit.image_candidate_cap),
        periodic_xc=manta.PeriodicXCOptions(args['xc_image_cutoff_bohr'], g.radial_points, g.lebedev_order,
            g.partition_radius_bohr, g.stratmann, o.grid_batch),
        max_iter=o.scf.maximum_iterations, conv_tol_energy=o.scf.energy_change_tolerance,
        conv_tol_grad=o.scf.commutator_tolerance, use_diis=False,
        exchange_exxdiv='ewald' if correction == 'BvK-ewald' else 'none',
        output=directory/'selected'/'job', record_hostname=False, structured_log=True)
    return job, args, a, k, old


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('spins', [1, 2])
@pytest.mark.parametrize('gamma', [False, True])
@pytest.mark.parametrize('correction', ['strict-zero-mode', 'BvK-ewald'])
def test_public_scf_matches_independent_functional_and_native_sources(tmp_path, profile, spins, gamma, correction):
    from vibeqc.output import validate_qvf
    job, args, a, k, old = job_case(tmp_path, profile, spins, gamma, stratmann=spins == 2, correction=correction)
    domains, grid = explicit(args, a, old)
    r = vq.run_periodic_job(**job); n = r.native; other = function(args)(**args)
    assert r.converged and n.receipt.confirmed
    assert isinstance(r, manta.PeriodicMANTARKSResult if spins == 1 else manta.PeriodicMANTAUKSResult)
    for field in ARRAYS: np.testing.assert_array_equal(getattr(n, field), getattr(other, field))
    assert n.receipt.energy == other.receipt.energy
    s, h, build = reference(a, k, old); pop = (n.plan.n_alpha,) if spins == 1 else (n.plan.n_alpha, n.plan.n_beta)
    density = independent_scf(s, h, build, pop, spins)
    np.testing.assert_allclose(n.density, density, atol=5e-9, rtol=0)
    f, energies, populations = build(n.density)
    np.testing.assert_allclose(n.fock, f, atol=8e-11, rtol=0)
    np.testing.assert_allclose(components(n), energies, atol=8e-11, rtol=0)
    assert r.energy == pytest.approx(sum(_nuclear_reference(old)[0])+sum(build(density)[1]), abs=8e-11)
    np.testing.assert_allclose(n.receipt.xc.electrons_grid, populations, atol=8e-11, rtol=0)
    path = job['output'].with_suffix('.qvf'); assert validate_qvf(path)['valid']
    expected = n.density if spins == 2 else np.repeat(.5*n.density, 2, axis=0)
    np.testing.assert_array_equal(_spin_density_k_from_qvf_bloch_payload(path), expected)
    qvf, evidence, history = output_evidence(job)
    assert evidence == r.integral_provenance and history == r.scf_history
    assert qvf['provenance']['method'].upper() == job['method']
    assert qvf['provenance']['run_status'] == 'converged'
    assert qvf['provenance']['run_fields'] == r.run_fields
    assert evidence['fitted_fock']['component_only'] and evidence['operator_kind'] == 'ks'
    assert evidence['domain_identity'] == domains.receipt.identity
    assert evidence['quadrature']['source_identity'] == grid.receipt.source_identity
    assert evidence['coulomb_algorithm'] == job['coulomb'].algorithm
    assert evidence['exchange_algorithm'] == job['exchange'].algorithm
    assert evidence['basis_frame_identity'] == n.receipt.overlap_identity
    assert not evidence['symmetry_reduced']
    if profile == 'pure':
        assert r.exchange_q0 == 'not-applicable' and evidence['exchange_q0_applicability'] == 'inactive'
    else:
        assert r.exchange_q0 == correction
        assert r.run_fields['exchange_source_identity'] == n.receipt.exchange_identity
        assert ('exchange' in evidence) == (profile == 'J')
    manifest = tomllib.loads(job['output'].with_suffix('.system').read_text())
    for key in ('fitting_source_identity', 'fitting_frame_identity', 'coulomb_source_identity', 'exchange_source_identity',
                'quadrature_source_identity', 'xc_source_identity', 'density_identity', 'fock_identity'):
        assert manifest['run'][key] == r.run_fields[key]
    text = job['output'].with_suffix('.out').read_text()
    assert 'XC='+job['functional'] in text and 'auxiliary='+args['auxiliary'].name in text
    if profile in ('K', 'JK'): assert 'RI exchange (scaled, uncorrected)' in text
    refs = job['output'].with_suffix('.references').read_text()
    for name in ('Sun', 'Berkelbach', 'Becke', 'Lebedev', 'Treutler'): assert name in refs
    if spins == 2: assert 'Stratmann' in refs
    events = [json.loads(x) for x in job['output'].with_suffix('.scf.jsonl').read_text().splitlines()]
    assert next(e['integral_provenance'] for e in events if e.get('event') == 'manta_periodic_ks') == evidence
    assert job['integral_engine'].budget.other_live_bytes == 0
    state = as_fock_build(n, k)
    assert state.energy.total == pytest.approx(r.energy, abs=1e-13)
    for channel, density, fock in zip(state.channels, n.density.reshape(-1, 3, 3), n.fock.reshape(-1, 3, 3)):
        np.testing.assert_array_equal(channel.density, density)
        np.testing.assert_allclose(channel.operator.apply(np.eye(3, dtype=complex)), fock, atol=1e-13, rtol=0)
        assert channel.operator.metadata.kind == 'ks' and channel.operator.metadata.symmetry_block is None


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('spins', [1, 2])
@pytest.mark.parametrize('rank_cutoff', [0., 10., 1e9])
def test_capped_public_epoch_and_frozen_ownership(tmp_path, profile, spins, rank_cutoff):
    job, args, a, k, old = job_case(tmp_path, profile, spins)
    job['coulomb'].metric.absolute_cutoff = rank_cutoff; job['max_iter'] = 3
    args['options'].ks.scf.maximum_iterations = 3; explicit(args, a, old)
    r = vq.run_periodic_job(**job); n = r.native; _, _, build = reference(a, k, old)
    assert not r.converged and r.iterations == 3
    np.testing.assert_allclose(n.fock, build(n.density)[0], atol=8e-11, rtol=0)
    np.testing.assert_allclose(components(n), build(n.density)[1], atol=8e-11, rtol=0)
    pop = (n.plan.n_alpha,) if spins == 1 else (n.plan.n_alpha, n.plan.n_beta)
    projected = np.array([(2 if spins == 1 else 1)*n.coefficients[spin, :, :, :count]@
                          n.coefficients[spin, :, :, :count].conj().swapaxes(-1, -2) for spin, count in enumerate(pop)])
    assert np.max(abs(projected-n.density)) > 1e-9
    expected = n.density if spins == 2 else np.repeat(.5*n.density, 2, axis=0)
    np.testing.assert_array_equal(_spin_density_k_from_qvf_bloch_payload(job['output'].with_suffix('.qvf')), expected)
    qvf, evidence, history = output_evidence(job)
    assert qvf['provenance']['run_status'] == 'failed' and len(history) == 3
    r.integral_provenance['fitting']['metrics'][0]['projector_identity'] = 'changed'
    assert r.integral_provenance == evidence
    old_density = np.array(r.density_k); r.density_k[0][:] = np.nan
    np.testing.assert_array_equal(r.density_k, old_density)
    with pytest.raises(dataclasses.FrozenInstanceError): r.mesh = (1, 1, 1)


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('gamma', [False, True])
def test_empty_spin_public_grid_and_evaluated_qvf(tmp_path, profile, gamma):
    job, args, a, k, old = job_case(tmp_path, profile, 2, gamma, empty=True)
    explicit(args, a, old); r = vq.run_periodic_job(**job); _, _, build = reference(a, k, old)
    assert r.converged and r.native.plan.n_beta == 0
    np.testing.assert_allclose(r.native.fock, build(r.native.density)[0], atol=8e-11, rtol=0)
    np.testing.assert_array_equal(_spin_density_k_from_qvf_bloch_payload(job['output'].with_suffix('.qvf')), r.native.density)
    assert r.integral_provenance['spin_scf_history'][-1]['spin_homo'][1] is None


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('spins', [1, 2])
@pytest.mark.parametrize('field,quantity', CAPS)
def test_joint_public_caps_before_generation_or_artifacts(tmp_path, monkeypatch, profile, spins, field, quantity):
    job, args, *_ = job_case(tmp_path, profile, spins); job.update(max_iter=1, dry_run=True)
    budget = job['integral_engine'].budget; budget.other_live_bytes = 12345
    name = function(args, 'plan').__name__; native_plan = getattr(core._manta, name); seen = []
    def capture(*a, **kw):
        p = native_plan(*a, **kw); seen.append((p, a[-1].other_live_bytes)); return p
    with monkeypatch.context() as patch:
        patch.setattr(core._manta, name, capture); assert vq.run_periodic_job(**job) is None
    p, live = seen[-1]; assert len(seen) == 2 and live > 12345
    for key, bound in CAPS: setattr(budget, key, getattr(p, bound))
    job.update(dry_run=False, output=tmp_path/'exact'/'job'); r = vq.run_periodic_job(**job)
    assert r.iterations == 1 and not r.converged and budget.other_live_bytes == 12345
    assert live == 12345+r.run_fields['job_reserved_bytes']+r.integral_provenance['python_reserved_bytes']
    setattr(budget, field, getattr(p, quantity)-1); job['output'] = tmp_path/'short'/'job'
    def fail(*a, **kw): raise AssertionError('Unadmitted public job generated a numerical source')
    monkeypatch.setattr(core._manta, function(args).__name__, fail)
    with pytest.raises((ValueError, RuntimeError), match='budget'): vq.run_periodic_job(**job)
    _no_computed_artifacts(job)


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('fault', ['no-fitting', 'no-aux', 'no-metric', 'incomplete-range', 'meta', 'symmetry', 'bipole', 'pure-k', 'ri-bupo'])
def test_unqualified_public_selections_refuse_before_work(tmp_path, monkeypatch, profile, fault):
    job, args, *_ = job_case(tmp_path, profile)
    if fault == 'no-fitting': job['periodic_fitting'] = None
    if fault == 'no-aux': job['coulomb'] = dataclasses.replace(job['coulomb'], auxiliary_basis=None)
    if fault == 'no-metric': job['coulomb'] = dataclasses.replace(job['coulomb'], metric=None)
    if fault == 'incomplete-range': job['functional'] = 'wb97x-v'
    if fault == 'meta': job['functional'] = 'scan'
    if fault == 'symmetry': job['symmetry_reduce_k'] = True
    if fault == 'bipole': job['bipole_cutoff_bohr'] = 7.
    if fault == 'pure-k': job['functional'] = 'pbe'; job['exchange'] = manta.ExchangeBuildOptions(algorithm='ri')
    if fault == 'ri-bupo': job['coulomb'] = dataclasses.replace(job['coulomb'], algorithm='ri-bupo')
    def fail(*a, **kw): raise AssertionError('Unqualified source reached numerical generation')
    for source in ('ri_ks', 'ri_global_hybrid_ks', 'ri_range_separated_ks', 'ks', 'global_hybrid_ks', 'range_separated_ks'):
        monkeypatch.setattr(core._manta, 'run_periodic_'+source+'_with_quadrature', fail)
    with pytest.raises((ValueError, TypeError, NotImplementedError)): vq.run_periodic_job(**job)
    _no_computed_artifacts(job)


@pytest.mark.parametrize('profile', PROFILES)
def test_exchange_defaults_do_not_enable_ri_k(tmp_path, profile):
    job, args, *_ = job_case(tmp_path, profile); job['exchange'] = None
    if profile == 'K':
        # No fitted component remains; the supplied fitting controls refuse.
        with pytest.raises(ValueError, match='periodic_fitting'): vq.run_periodic_job(**job)
        return
    r = vq.run_periodic_job(**job)
    assert r.integral_provenance['exchange_algorithm'] == ('none' if profile == 'pure' else 'exact')
    assert not r.native.plan.fitting.has_k


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('spins', [1, 2])
def test_public_image_gauge_and_nuclear_frame_qvf(tmp_path, profile, spins):
    from vibeqc.symmetry_shared import BlockSpaceAction, Budget, SpaceIdentity
    job, args, _, k, _ = job_case(tmp_path/'original', profile, spins); job['max_iter'] = 3
    charges = (2, 1, 1) if spins == 1 else (1, 1, 1)
    atoms = [vq.Atom(z, xyz) for z, xyz in zip(charges, ([0., 0., 0.], [.9, .2, 0.], [-.2, 1.1, .1]))]
    system = vq.PeriodicSystem(3, job['system'].lattice, atoms); system.multiplicity = spins
    shells = job['basis'].shells(); aux = job['coulomb'].auxiliary_basis.shells()
    for i, shell in enumerate(shells): shell.atom_index = i; shell.origin = atoms[i].xyz
    for shell in aux: shell.origin = atoms[0].xyz
    job.update(system=system, basis=vq.BasisSet(system.unit_cell_molecule(), shells, 'public-ri-ks', True))
    job['coulomb'] = dataclasses.replace(job['coulomb'], auxiliary_basis=vq.BasisSet(system.unit_cell_molecule(), aux, 'public-ri-ks-aux', True))
    original = vq.run_periodic_job(**job); before = original.native
    shifts = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])@system.lattice.T
    for shell, delta in zip(shells, shifts): shell.origin = np.array(shell.origin)+delta
    for i, (shell, delta) in enumerate(zip(aux, shifts[::-1])):
        shell.atom_index = i; shell.origin = np.array(shell.origin)+delta
    moved_bases = []
    for values in (shells, aux):
        images = vq.PeriodicSystem(3, system.lattice, [vq.Atom(z, shell.origin) for z, shell in zip(charges, values)])
        images.multiplicity = spins
        moved_bases.append(vq.BasisSet(images.unit_cell_molecule(), values, 'public-ri-ks-images', True))
    job.update(basis=moved_bases[0], output=tmp_path/'moved'/'job')
    job['coulomb'] = dataclasses.replace(job['coulomb'], auxiliary_basis=moved_bases[1])
    moved = vq.run_periodic_job(**job); after = moved.native; phase = np.exp(1j*k@shifts.T)
    transform = phase[:, :, None]*phase[:, None, :].conj()
    for name in ('density', 'fock', 'overlap', 'hcore'):
        np.testing.assert_allclose(getattr(after, name), getattr(before, name)*transform, atol=2e-10, rtol=0)
    assert moved.energy == pytest.approx(original.energy, abs=1e-10)
    for ik in range(len(k)):
        source = SpaceIdentity('public-ri-ks-test', before.receipt.overlap_identity, 'AO', f'original-k{ik}')
        target = SpaceIdentity('public-ri-ks-test', after.receipt.overlap_identity, 'AO', f'shifted-k{ik}')
        action = BlockSpaceAction(source, target, np.arange(4, dtype=np.int64), np.arange(3, dtype=np.int64),
            np.ascontiguousarray(phase[ik]), antiunitary=False, budget=Budget(2**20, 10**6))
        for spin in range(spins):
            np.testing.assert_allclose(action.push_density(before.density[spin, ik]), after.density[spin, ik], atol=2e-10, rtol=0)
            np.testing.assert_allclose(action.pull_operator(after.fock[spin, ik]), before.fock[spin, ik], atol=2e-10, rtol=0)
    expected = before.density if spins == 2 else np.repeat(.5*before.density, 2, axis=0)
    np.testing.assert_allclose(_spin_density_k_from_qvf_bloch_payload(job['output'].with_suffix('.qvf')), expected, atol=2e-10, rtol=0)
    assert moved.integral_provenance['qvf_density_gauge'] == 'nuclear-centers'
    state = as_fock_build(after, k)
    assert state.energy.total == pytest.approx(original.energy, abs=1e-10)


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('spins', [1, 2])
def test_public_rectangular_overlap_exports_actual_density(tmp_path, profile, spins):
    job, args, a, k, old = job_case(tmp_path, profile, spins); job['max_iter'] = 3
    args['options'].ks.scf.maximum_iterations = 3
    shells = job['basis'].shells(); shells.append(shells[0])
    job['basis'] = args['basis'] = a['basis'] = old['basis'] = vq.BasisSet(job['system'].unit_cell_molecule(), shells, 'public-ri-ks-rank', True)
    explicit(args, a, old); r = vq.run_periodic_job(**job); _, _, build = reference(a, k, old)
    np.testing.assert_allclose(r.native.fock, build(r.native.density)[0], atol=8e-11, rtol=0)
    np.testing.assert_allclose(components(r.native), build(r.native.density)[1], atol=8e-11, rtol=0)
    assert r.native.coefficients.shape == (spins, len(k), 4, 3)
    state = as_fock_build(r.native, k)
    for channel in state.channels: assert channel.operator.frame.X.shape == (4, 3)
    expected = r.native.density if spins == 2 else np.repeat(.5*r.native.density, 2, axis=0)
    np.testing.assert_array_equal(_spin_density_k_from_qvf_bloch_payload(job['output'].with_suffix('.qvf')), expected)


@pytest.mark.parametrize('profile', PROFILES)
def test_public_dry_run_and_delivery_failure_preserve_lifecycle(tmp_path, monkeypatch, profile):
    from vibeqc.output import OutputWriter
    from vibeqc.periodic_runner import _PERIODIC_OUTPUT_WRITER
    job, args, *_ = job_case(tmp_path/'dry', profile); job['dry_run'] = True
    def fail(*a, **kw): raise AssertionError('Dry run generated or solved a numerical source')
    with monkeypatch.context() as patch:
        patch.setattr(core._manta, function(args).__name__, fail)
        assert vq.run_periodic_job(**job) is None
    assert list(job['output'].parent.iterdir()) == [job['output'].with_suffix('.system')]
    job, *_ = job_case(tmp_path/'delivery', profile); dispatch = OutputWriter.dispatch_role
    def fail_delivery(self, role, **kwargs):
        if role == 'qvf': raise OSError('synthetic fitted KS delivery failure')
        return dispatch(self, role, **kwargs)
    monkeypatch.setattr(OutputWriter, 'dispatch_role', fail_delivery)
    before = _PERIODIC_OUTPUT_WRITER.get()
    with pytest.raises(OSError, match='synthetic fitted KS delivery'): vq.run_periodic_job(**job)
    assert _PERIODIC_OUTPUT_WRITER.get() is before
    manifest = tomllib.loads(job['output'].with_suffix('.system').read_text())
    assert manifest['outputs']['status'] == 'crashed' and manifest['outputs']['result_status'] == 'converged'


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('spins', [1, 2])
def test_public_functional_descriptor_reaches_fitted_solver(tmp_path, profile, spins):
    job, args, a, k, old = job_case(tmp_path, profile, spins, gamma=True)
    job['functional'] = 'lda' if profile == 'pure' else 'b3lyp'
    args['functional'] = a['functional'] = vq.Functional(job['functional'], spin=spins)
    explicit(args, a, old); r = vq.run_periodic_job(**job); _, _, build = reference(a, k, old)
    assert r.converged
    np.testing.assert_allclose(r.native.fock, build(r.native.density)[0], atol=8e-11, rtol=0)
    np.testing.assert_allclose(components(r.native), build(r.native.density)[1], atol=8e-11, rtol=0)
    assert r.native.receipt.exact_exchange_fraction == pytest.approx(0. if profile == 'pure' else .2)
