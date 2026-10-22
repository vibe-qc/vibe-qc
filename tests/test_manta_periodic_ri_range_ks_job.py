"""Public fitted range KS: actual kernel sources and evaluated QVF state."""
import dataclasses
import json
import tomllib

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta, _vibeqc_core as core
from vibeqc.guess_read import _spin_density_k_from_qvf_bloch_payload
from tests.test_manta_periodic_ri_range_ks_domains import (
    arguments, explicit, function, check_reference, ARRAYS, PROFILES,
    reference, as_fock_build, CAPS, components, MIXED_RANGE, EMPTY_RANGE,
)
from tests.test_manta_periodic_global_hybrid_job import output_evidence
from tests.test_manta_periodic_job import _no_computed_artifacts


def job_case(directory, profile='JK', spins=1, gamma=False, empty=False, stratmann=False, correction='BvK-ewald', name='cam-b3lyp'):
    kw = dict(mesh=(1, 1, 1), shift=(0, 0, 0)) if gamma else {}
    args, a, k, old = arguments(profile, spins, True, EMPTY_RANGE if empty else name, species='H' if empty else None, **kw)
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
@pytest.mark.parametrize('name', ['hse06', 'cam-b3lyp', MIXED_RANGE])
@pytest.mark.parametrize('correction', ['strict-zero-mode', 'BvK-ewald'])
def test_public_physical_scf_and_lossless_kernel_output(tmp_path, profile, spins, gamma, name, correction):
    from vibeqc.output import validate_qvf
    job, args, a, k, old = job_case(tmp_path, profile, spins, gamma, name=name,
                                   stratmann=spins == 2, correction=correction)
    domains, grid = explicit(args, a, old)
    result = vq.run_periodic_job(**job); r = result.native; other = function(args)(**args)
    assert r.converged and r.receipt.confirmed
    for key in ARRAYS: np.testing.assert_array_equal(getattr(r, key), getattr(other, key))
    assert r.receipt.energy == other.receipt.energy
    check_reference(r, a, k, old)
    path = job['output'].with_suffix('.qvf'); assert validate_qvf(path)['valid']
    expected = r.density if spins == 2 else np.repeat(.5*r.density, 2, axis=0)
    np.testing.assert_array_equal(_spin_density_k_from_qvf_bloch_payload(path), expected)
    qvf, evidence, history = output_evidence(job)
    assert evidence == result.integral_provenance and history == result.scf_history
    assert qvf['provenance']['run_status'] == 'converged'
    assert qvf['provenance']['run_fields'] == result.run_fields
    assert evidence['range_separated'] and not evidence['symmetry_reduced']
    assert evidence['operator_kind'] == 'ks' and evidence['basis_frame_identity'] == r.receipt.overlap_identity
    assert evidence['domain_identity'] == domains.receipt.identity
    assert evidence['quadrature']['source_identity'] == grid.receipt.source_identity
    actual_q0 = correction if r.receipt.long_range_exchange_fraction else 'not-applicable'
    assert result.exchange_q0 == evidence['exchange_q0'] == actual_q0
    assert evidence['exchange_q0_requested'] == correction
    assert evidence['exchange_q0_applicability'] == ('active' if r.receipt.long_range_exchange_fraction else 'inactive')
    assert ('fitting' in evidence) == (profile != 'K')
    fits = ([r.receipt.fitting_info] if r.receipt.fitting_info else [])
    fits += [c.fitting_info for c in r.receipt.exchange_components if c.fitting_info]
    assert result.run_fields['fitting_source_identities'] == [f.source_identity for f in fits]
    assert result.run_fields['fitting_frame_identities'] == [f.frame_identity for f in fits]
    assert 'fitting_source_identity' not in result.run_fields
    sources = []
    for c, e in zip(r.receipt.exchange_components, evidence['exchange_components']):
        sources.append((c.fitted_fock if c.fitted_fock else c.exchange).source_identity)
        assert e['kernel'] == c.kernel and e['omega'] == c.omega and e['coefficient'] == c.coefficient
        if profile != 'J':
            assert e['fitting']['source_identity'] == c.fitting_info.source_identity
            assert e['fitted_fock']['component_only'] and e['energy_scale'] == 1.
            assert e['energy'] == c.fitted_fock.exchange_energy
        else:
            assert e['exchange']['source_identity'] == c.exchange.source_identity
    assert result.run_fields['exchange_source_identities'] == sources
    manifest = tomllib.loads(job['output'].with_suffix('.system').read_text())
    for key, value in result.run_fields.items(): assert manifest['run'][key] == value
    text = job['output'].with_suffix('.out').read_text()
    assert 'physical range-separated' in text and 'auxiliary='+args['auxiliary'].name in text
    assert 'long-range fraction=' in text
    if profile != 'J': assert 'RI exchange (scaled, uncorrected)' in text
    refs = job['output'].with_suffix('.references').read_text()
    for name in ('Sun', 'Berkelbach', 'Becke', 'Lebedev', 'Treutler'): assert name in refs
    if spins == 2: assert 'Stratmann' in refs
    events = [json.loads(line) for line in job['output'].with_suffix('.scf.jsonl').read_text().splitlines()]
    assert next(e['integral_provenance'] for e in events if e.get('event') == 'manta_periodic_ks') == evidence
    state = as_fock_build(r, k)
    assert state.energy.total == pytest.approx(result.energy, abs=1e-13)
    assert job['integral_engine'].budget.other_live_bytes == 0


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('spins', [1, 2])
@pytest.mark.parametrize('rank_cutoff', [0., 10., 1e9])
def test_capped_public_epoch_and_owned_per_kernel_evidence(tmp_path, profile, spins, rank_cutoff):
    job, args, a, k, old = job_case(tmp_path, profile, spins, name=MIXED_RANGE)
    job['coulomb'].metric.absolute_cutoff = rank_cutoff; job['max_iter'] = 3
    args['options'].ks.scf.maximum_iterations = 3; explicit(args, a, old)
    result = vq.run_periodic_job(**job); r = result.native; _, _, build = reference(a, k, old)
    assert not result.converged and result.iterations == 3
    np.testing.assert_allclose(r.fock, build(r.density)[0], atol=8e-11, rtol=0)
    np.testing.assert_allclose(components(r), build(r.density)[1], atol=8e-11, rtol=0)
    pop = (r.plan.n_alpha,) if spins == 1 else (r.plan.n_alpha, r.plan.n_beta)
    projected = np.array([(2 if spins == 1 else 1)*r.coefficients[spin, :, :, :count]@
                          r.coefficients[spin, :, :, :count].conj().swapaxes(-1, -2) for spin, count in enumerate(pop)])
    assert np.max(abs(projected-r.density)) > 1e-9
    expected = r.density if spins == 2 else np.repeat(.5*r.density, 2, axis=0)
    np.testing.assert_array_equal(_spin_density_k_from_qvf_bloch_payload(job['output'].with_suffix('.qvf')), expected)
    qvf, evidence, history = output_evidence(job)
    assert qvf['provenance']['run_status'] == 'failed' and len(history) == 3
    changed = result.integral_provenance
    fit = changed['fitting'] if profile == 'J' else changed['exchange_components'][0]['fitting']
    fit['metrics'][0]['projector_identity'] = 'changed'
    assert result.integral_provenance == evidence
    old_density = np.array(result.density_k); result.density_k[0][:] = np.nan
    np.testing.assert_array_equal(result.density_k, old_density)
    with pytest.raises(dataclasses.FrozenInstanceError): result.mesh = (1, 1, 1)


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('gamma', [False, True])
def test_empty_beta_public_range_state(tmp_path, profile, gamma):
    job, args, a, k, old = job_case(tmp_path, profile, 2, gamma, empty=True)
    explicit(args, a, old); result = vq.run_periodic_job(**job)
    assert result.converged and result.native.plan.n_beta == 0
    check_reference(result.native, a, k, old)
    np.testing.assert_array_equal(_spin_density_k_from_qvf_bloch_payload(job['output'].with_suffix('.qvf')), result.native.density)
    assert result.integral_provenance['spin_scf_history'][-1]['spin_homo'][1] is None


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('spins', [1, 2])
@pytest.mark.parametrize('field,quantity', CAPS)
def test_joint_public_caps_before_generation_and_artifacts(tmp_path, monkeypatch, profile, spins, field, quantity):
    job, args, *_ = job_case(tmp_path, profile, spins, name=MIXED_RANGE); job.update(max_iter=1, dry_run=True)
    budget = job['integral_engine'].budget; budget.other_live_bytes = 12345
    name = function(args, 'plan').__name__; native_plan = getattr(core._manta, name); seen = []
    def capture(*a, **kw):
        p = native_plan(*a, **kw); seen.append((p, a[-1].other_live_bytes)); return p
    with monkeypatch.context() as patch:
        patch.setattr(core._manta, name, capture); assert vq.run_periodic_job(**job) is None
    p, live = seen[-1]; assert len(seen) == 2 and live > 12345
    for key, bound in CAPS: setattr(budget, key, getattr(p, bound))
    job.update(dry_run=False, output=tmp_path/'exact'/'job'); result = vq.run_periodic_job(**job)
    assert result.iterations == 1 and not result.converged and budget.other_live_bytes == 12345
    assert live == 12345+result.run_fields['job_reserved_bytes']+result.integral_provenance['python_reserved_bytes']
    setattr(budget, field, getattr(p, quantity)-1); job['output'] = tmp_path/'short'/'job'
    def fail(*a, **kw): raise AssertionError('Unadmitted public job generated a numerical source')
    monkeypatch.setattr(core._manta, function(args).__name__, fail)
    with pytest.raises((ValueError, RuntimeError), match='budget'): vq.run_periodic_job(**job)
    _no_computed_artifacts(job)


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('fault', ['no-fitting', 'no-aux', 'no-metric', 'incomplete-range', 'meta', 'symmetry', 'bipole', 'ri-bupo'])
def test_unqualified_sources_refuse_before_work(tmp_path, monkeypatch, profile, fault):
    job, args, *_ = job_case(tmp_path, profile)
    if fault == 'no-fitting': job['periodic_fitting'] = None
    if fault == 'no-aux': job['coulomb'] = dataclasses.replace(job['coulomb'], auxiliary_basis=None)
    if fault == 'no-metric': job['coulomb'] = dataclasses.replace(job['coulomb'], metric=None)
    if fault == 'incomplete-range': job['functional'] = 'wb97x-v'
    if fault == 'meta': job['functional'] = 'scan'
    if fault == 'symmetry': job['symmetry_reduce_k'] = True
    if fault == 'bipole': job['bipole_cutoff_bohr'] = 7.
    if fault == 'ri-bupo': job['coulomb'] = dataclasses.replace(job['coulomb'], algorithm='ri-bupo')
    def fail(*a, **kw): raise AssertionError('Unqualified source reached numerical generation')
    for source in ('ri_ks', 'ri_global_hybrid_ks', 'ri_range_separated_ks', 'ks', 'global_hybrid_ks', 'range_separated_ks'):
        monkeypatch.setattr(core._manta, 'run_periodic_'+source+'_with_quadrature', fail)
    with pytest.raises((ValueError, TypeError, NotImplementedError)): vq.run_periodic_job(**job)
    _no_computed_artifacts(job)


@pytest.mark.parametrize('profile', PROFILES)
def test_default_exchange_stays_exact(tmp_path, profile):
    job, *_ = job_case(tmp_path, profile); job['exchange'] = None
    if profile == 'K':
        with pytest.raises(ValueError, match='periodic_fitting'): vq.run_periodic_job(**job)
    else:
        result = vq.run_periodic_job(**job)
        assert result.integral_provenance['exchange_algorithm'] == 'exact'
        assert all(c.fitted_fock is None for c in result.native.receipt.exchange_components)



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
    job['functional'] = 'lc-wpbe'
    args['functional'] = a['functional'] = vq.Functional(job['functional'], spin=spins)
    explicit(args, a, old); r = vq.run_periodic_job(**job); _, _, build = reference(a, k, old)
    assert r.converged
    np.testing.assert_allclose(r.native.fock, build(r.native.density)[0], atol=8e-11, rtol=0)
    np.testing.assert_allclose(components(r.native), build(r.native.density)[1], atol=8e-11, rtol=0)
    assert r.native.receipt.long_range_exchange_fraction == 1.
    assert r.native.receipt.short_range_exchange_fraction == 0.
