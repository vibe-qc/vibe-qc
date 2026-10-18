"""Complete public range-separated jobs, physical epochs and lossless artifacts."""
import dataclasses
import json
import tomllib
import zipfile

import numpy as np
import pytest
from scipy.linalg import eigh

import vibeqc as vq
from vibeqc import manta, _vibeqc_core as core
from vibeqc.guess_read import _spin_density_k_from_qvf_bloch_payload
from tests.test_manta_periodic_job import _qvf, _no_computed_artifacts
from tests.test_manta_periodic_range_separated_domains import arguments, explicit, CAPS
from tests.test_manta_periodic_range_separated_ks import oracle, components, EMPTY_RANGE, MIXED_RANGE, TERMS
from tests.test_manta_periodic_ks import independent_scf
from tests.test_manta_periodic_rhf import _nuclear_reference


def job_case(tmp_path, species='H2', spins=1, correction='BvK-ewald', stratmann=False, name=None):
    args, c, k, old = arguments(species, spins, correction, grid=True, name=name)
    o, g = args['options'].ks, args['grid_options']
    g.stratmann = stratmann
    job = dict(system=args['system'], basis=args['basis'], kpoints=args['mesh'],
        method='RKS' if spins == 1 else 'UKS', functional=c['functional'].name,
        integral_engine=manta.IntegralEngineOptions(backend='manta', budget=args['budget']),
        periodic_integrals=manta.PeriodicIntegralOptions(args['reciprocal_energy_cutoff'],
            o.pair_cutoff_bohr, o.nuclear_cutoff_bohr, alpha=o.alpha, reciprocal_batch=o.reciprocal_batch),
        periodic_xc=manta.PeriodicXCOptions(args['xc_image_cutoff_bohr'], g.radial_points,
            g.lebedev_order, g.partition_radius_bohr, g.stratmann, o.grid_batch),
        max_iter=o.scf.maximum_iterations, conv_tol_energy=o.scf.energy_change_tolerance,
        conv_tol_grad=o.scf.commutator_tolerance, use_diis=False,
        exchange_exxdiv='ewald' if correction == 'BvK-ewald' else 'none',
        output=tmp_path/'selected'/'job', record_hostname=False, structured_log=True)
    return job, args, c, k, old


def output_evidence(job):
    path = job['output'].with_suffix('.qvf')
    manifest = _qvf(path)
    section = next(s for s in manifest['sections'] if s['kind'] == 'x_vibeqc.periodic_integral_engine')
    with zipfile.ZipFile(path) as archive:
        evidence = json.loads(archive.read(next(iter(section['members'].values()))['path']))
        history = json.loads(archive.read('scf_history/iterations.json'))['iterations']
    return manifest, evidence, history


@pytest.mark.parametrize('species,spins,stratmann,name', [
    ('H2',1,False,'hse06'),('H2',1,True,'cam-b3lyp'),('H2',1,False,'lc-wpbe'),
    ('H2',1,True,'wb97x'),('H2',2,True,'hse06'),('HeH',2,False,'cam-b3lyp'),
    ('HeH',2,True,MIXED_RANGE),('H',2,True,EMPTY_RANGE)])
@pytest.mark.parametrize('correction', ['strict-zero-mode', 'BvK-ewald'])
def test_complete_job_independent_scf_and_all_energy_components(tmp_path, species, spins, stratmann, name, correction):
    from vibeqc.output import validate_qvf
    job, args, c, k, old = job_case(tmp_path, species, spins, correction, stratmann, name)
    domains, grid = explicit(args, c, old)
    s, h, build, xi = oracle(c, k, old)
    result = vq.run_periodic_job(**job); r = result.native
    assert result.converged and r.receipt.confirmed
    assert isinstance(result, manta.PeriodicMANTARKSResult if spins == 1 else manta.PeriodicMANTAUKSResult)
    pop = (r.plan.n_alpha,) if spins == 1 else (r.plan.n_alpha, r.plan.n_beta)
    d = independent_scf(s, h, build, pop, spins); f, energy, populations = build(r.density)
    np.testing.assert_allclose(r.density, d, atol=3e-9, rtol=0)
    np.testing.assert_allclose(r.fock, f, atol=2e-11, rtol=0)
    np.testing.assert_allclose(r.overlap, s, atol=3e-12, rtol=0)
    np.testing.assert_allclose(r.hcore, h, atol=3e-12, rtol=0)
    np.testing.assert_allclose(components(r), energy, atol=2e-11, rtol=0)
    np.testing.assert_allclose(r.receipt.xc.electrons_grid, populations, atol=2e-12, rtol=0)
    nuclear, _ = _nuclear_reference(old)
    assert result.energy == pytest.approx(sum(nuclear)+sum(energy), abs=2e-11)
    direct = manta.run_periodic_range_separated_ks_with_quadrature(**args)
    np.testing.assert_array_equal(r.density, direct.density)
    assert r.receipt.source_identity == direct.receipt.source_identity

    expected = r.density if spins == 2 else np.repeat(.5*r.density, 2, axis=0)
    path = job['output'].with_suffix('.qvf')
    np.testing.assert_array_equal(_spin_density_k_from_qvf_bloch_payload(path), expected)
    assert validate_qvf(path)['valid']
    qvf, evidence, history = output_evidence(job)
    assert evidence == result.integral_provenance and history == result.scf_history
    assert qvf['provenance']['method'].upper() == job['method']
    assert qvf['provenance']['functional'] == job['functional']
    actual_q0=correction if r.receipt.long_range_exchange_fraction else 'not-applicable'
    assert qvf['provenance']['exchange_q0'] == result.exchange_q0 == actual_q0
    assert qvf['provenance']['run_status'] == 'converged'
    assert qvf['provenance']['run_fields'] == result.run_fields
    assert evidence['operator_kind'] == 'ks' and evidence['operator_layout'] == 'spin,k,ao,ao'
    assert evidence['operator_generation'] == result.iterations
    assert evidence['basis_frame_identity'] == r.receipt.overlap_identity
    assert evidence['accuracy_qualification'] == 'finite-source-only'
    assert evidence['quadrature']['source_identity'] == grid.receipt.source_identity
    assert evidence['domain_identity'] == domains.receipt.identity
    assert evidence['domain_exchange_vectors'] == len(domains.exchange_vectors)
    assert 'exchange' not in evidence and 'exchange_energy_scale' not in evidence
    assert evidence['range_separated'] and evidence['exact_exchange_fraction']==0.
    assert evidence['exchange_q0_requested']==correction
    assert evidence['exchange_q0_applicability']==('active' if r.receipt.long_range_exchange_fraction else 'inactive')
    assert evidence['coulomb_algorithm']==evidence['exchange_algorithm']=='exact'
    assert evidence['exchange_zero_mode_convention']=='canonical-coulomb-plus-erfc;finite-screened-q0'
    assert evidence['xc']['semilocal_component_only']
    assert evidence['xc']['required_exchange_components']==[dict(kernel=a.kernel,coefficient=a.coefficient,omega=a.omega) for a in r.receipt.exchange_components]
    for child,native,(kernel,coefficient,omega) in zip(evidence['exchange_components'],r.receipt.exchange_components,TERMS[name]):
        assert child['kernel']==kernel and child['coefficient']==pytest.approx(coefficient) and child['omega']==pytest.approx(omega)
        assert child['exchange']['source_identity']==native.exchange.source_identity
        assert child['exchange']['density_identities']==native.exchange.density_identities
        assert child['exchange']['zero_vectors']==(len(k) if kernel=='erfc' else 0)
        assert child['energy_scale']==pytest.approx(coefficient/(2 if spins==1 else 1))
        assert child['energy']==pytest.approx(child['energy_scale']*sum(child['exchange']['energies']),abs=2e-15)
    assert len(evidence['exchange_components'])==len(TERMS[name])
    assert evidence['uncorrected_exchange_energy']==pytest.approx(sum(c['energy'] for c in evidence['exchange_components']),abs=2e-15)
    assert evidence['exchange_energy']==pytest.approx(sum(energy[-2:]),abs=2e-11)
    assert evidence['madelung']==pytest.approx(xi,abs=4e-12)
    assert bool(evidence['exchange_correction_identity'])==(actual_q0=='BvK-ewald')
    assert ('manta_periodic_madelung' in evidence['citation_methods'])==(actual_q0=='BvK-ewald')
    assert 'manta_periodic_range_separated_ks' in evidence['citation_methods']
    if spins == 2 and species == 'H':
        assert evidence['spin_scf_history'][-1]['spin_homo'][1] is None
        assert evidence['spin_band_gaps'][1] is None
    manifest = tomllib.loads(job['output'].with_suffix('.system').read_text())
    for key, value in result.run_fields.items():
        assert manifest['run'][key] == value
    text = job['output'].with_suffix('.out').read_text()
    assert 'Exact exchange (scaled, uncorrected)' in text and 'Exchange zero-mode correction (scaled)' in text
    references = job['output'].with_suffix('.references').read_text()
    for name in ('Sundararaman', 'Becke', 'Treutler', 'Lebedev'):
        assert name in references
    if stratmann: assert 'Stratmann' in references
    records = [json.loads(line) for line in job['output'].with_suffix('.scf.jsonl').read_text().splitlines()]
    assert next(row['integral_provenance'] for row in records if row.get('event') == 'manta_periodic_ks') == evidence
    energies = next(row for row in records if row.get('event') == 'scf_energy_components')
    assert energies['density_identity'] == evidence['final_density_identity']
    assert sum(energies[key] for key in ('nuclear_energy', 'one_electron_energy', 'coulomb_energy',
                                       'xc_energy', 'exchange_energy')) == pytest.approx(result.energy, abs=3e-14)
    assert len([row for row in records if row.get('event') == 'scf_iteration']) == result.iterations
    assert job['integral_engine'].budget.other_live_bytes == 0
    assert 'exchange' not in job  # defaulting must not mutate caller kwargs
    result.integral_provenance['exchange_components'][0]['exchange']['energies'][0] = 999
    result.run_fields['exchange_source_identities'].clear()
    assert result.run_fields['exchange_source_identities']
    assert result.integral_provenance == evidence
    for name in ('density_alpha_k', 'density_beta_k', 'overlap_k', 'hcore_k'):
        before = np.array(getattr(result, name)); getattr(result, name)[0][:] = 777
        np.testing.assert_array_equal(getattr(result, name), before)
    with pytest.raises(dataclasses.FrozenInstanceError): result.mesh = (1, 1, 1)


@pytest.mark.parametrize('species,spins', [('H2', 1), ('HeH', 2), ('H', 2)])
@pytest.mark.parametrize('correction', ['strict-zero-mode', 'BvK-ewald'])
def test_capped_job_keeps_evaluated_state_and_separate_next_projector(tmp_path, species, spins, correction):
    job, args, c, k, old = job_case(tmp_path, species, spins, correction)
    if spins == 1:
        shells = job['basis'].shells(); shells[1].exponents = [.6]
        job['basis'] = vq.BasisSet(job['system'].unit_cell_molecule(), shells, 'hybrid-job-asymmetric', False)
        args['basis'] = c['basis'] = old['basis'] = job['basis']
    explicit(args, c, old); _, _, build, _ = oracle(c, k, old)
    previous_id = None
    for cap in (1, 2, 3):
        job.update(max_iter=cap, output=tmp_path/f'epoch-{cap}'/'job')
        result = vq.run_periodic_job(**job); r = result.native
        assert not result.converged and not r.receipt.confirmed
        assert len(result.scf_history) == result.iterations == cap
        f, energy, _ = build(r.density)
        np.testing.assert_allclose(r.fock, f, atol=3e-11, rtol=0)
        np.testing.assert_allclose(components(r), energy, atol=3e-11, rtol=0)
        assert result.energy == pytest.approx(r.receipt.nuclear_energy+sum(energy), abs=3e-11)
        counts = (r.plan.n_alpha,) if spins == 1 else (r.plan.n_alpha, r.plan.n_beta)
        next_density = []
        for spin, count in enumerate(counts):
            coeff = r.coefficients[spin, :, :, :count]
            next_density.append((2 if spins == 1 else 1)*coeff@coeff.conj().swapaxes(-1, -2))
            for point in range(len(k)):
                eps, _ = eigh(r.fock[spin, point], r.overlap[point])
                np.testing.assert_allclose(r.orbital_energies[spin, point], eps, atol=3e-11, rtol=0)
                cc = r.coefficients[spin, point]
                np.testing.assert_allclose(cc.conj().T@r.overlap[point]@cc, np.eye(cc.shape[1]), atol=3e-12, rtol=0)
        assert np.max(abs(np.array(next_density)-r.density)) > 1e-9
        expected = r.density if spins == 2 else np.repeat(.5*r.density, 2, axis=0)
        np.testing.assert_array_equal(_spin_density_k_from_qvf_bloch_payload(job['output'].with_suffix('.qvf')), expected)
        qvf, evidence, history = output_evidence(job)
        assert qvf['provenance']['run_status'] == 'failed'
        assert evidence == result.integral_provenance and history == result.scf_history
        assert evidence['final_density_identity'] != previous_id
        previous_id = evidence['final_density_identity']


@pytest.mark.parametrize('spins', [1, 2])
@pytest.mark.parametrize('correction', ['strict-zero-mode', 'BvK-ewald'])
def test_source_frame_and_nuclear_qvf_frame_remain_distinct(tmp_path, spins, correction):
    job, *_ = job_case(tmp_path/'original', spins=spins, correction=correction)
    job['periodic_integrals'] = dataclasses.replace(job['periodic_integrals'], pair_cutoff_bohr=7., nuclear_cutoff_bohr=9.)
    original = vq.run_periodic_job(**job)
    shells = job['basis'].shells(); shells[0].origin = np.array(shells[0].origin)+job['system'].lattice[:, 0]
    job['basis'] = vq.BasisSet(vq.Molecule([vq.Atom(1, s.origin) for s in shells], 0, 1), shells, 'hybrid-job-image', False)
    job['output'] = tmp_path/'moved'/'job'; moved = vq.run_periodic_job(**job)
    assert moved.energy == pytest.approx(original.energy, abs=3e-11)
    phase = np.ones((len(job['kpoints']), 2), complex)
    phase[:, 0] = np.exp(2j*np.pi*moved.kpoints_fractional[:, 0])
    action = phase[:, :, None]*phase.conj()[:, None, :]
    for name in ('overlap', 'hcore', 'density', 'fock'):
        np.testing.assert_allclose(getattr(moved.native, name), getattr(original.native, name)*action, atol=3e-9, rtol=0)
    p, q = original.integral_provenance, moved.integral_provenance
    assert p['basis_frame_identity'] != q['basis_frame_identity']
    assert [a['exchange']['source_identity'] for a in p['exchange_components']] != [a['exchange']['source_identity'] for a in q['exchange_components']]
    assert p['quadrature']['source_identity'] == q['quadrature']['source_identity']
    assert q['source_ao_image_shifts'] == [[1, 0, 0], [0, 0, 0]]
    restored = _spin_density_k_from_qvf_bloch_payload(job['output'].with_suffix('.qvf'))
    expected = original.native.density if spins == 2 else np.repeat(.5*original.native.density, 2, axis=0)
    np.testing.assert_allclose(restored, expected, atol=3e-9, rtol=0)


def test_madelung_controls_and_convention_reach_actual_source(tmp_path):
    job, args, c, k, old = job_case(tmp_path,name='cam-b3lyp')
    job['periodic_integrals'] = dataclasses.replace(job['periodic_integrals'], madelung_alpha=.55, madelung_tolerance=1e-10)
    args['options'].madelung_alpha = .55; args['options'].madelung_tolerance = 1e-10
    explicit(args, c, old); _, _, build, xi = oracle(c, k, old)
    r = vq.run_periodic_job(**job); p = r.integral_provenance
    assert p['madelung_alpha_requested'] == p['madelung_plan']['alpha'] == .55
    assert p['madelung_tolerance'] == 1e-10
    assert p['madelung'] == pytest.approx(xi, abs=1e-10)
    np.testing.assert_allclose(components(r.native), build(r.native.density)[1], atol=2e-11, rtol=0)
    strict = vq.run_periodic_job(**dict(job, exchange_exxdiv='none', output=tmp_path/'strict'/'job'))
    assert r.exchange_q0 == 'BvK-ewald' and strict.exchange_q0 == 'strict-zero-mode'
    assert strict.integral_provenance['exchange_correction_energy'] == 0.
    assert [a['exchange']['source_identity'] for a in strict.integral_provenance['exchange_components']]==[a['exchange']['source_identity'] for a in p['exchange_components']]
    assert strict.integral_provenance['source_identity'] != p['source_identity']
    assert r.energy-strict.energy == pytest.approx(-.65*xi, abs=2e-10)


@pytest.mark.parametrize('field,value', [
    ('functional', 'scan'), ('functional', 'tpssh'), ('functional', 'b2plyp'),
    ('functional', 'wb97x-v'),
    ('periodic_xc', None), ('exchange', manta.ExchangeBuildOptions(algorithm='none')),
    ('exchange', manta.ExchangeBuildOptions(algorithm='ri')), ('exchange_exxdiv', 'automatic'),
    ('coulomb', manta.CoulombBuildOptions(algorithm='ri')),
    ('symmetry', 'attach'), ('symmetry_reduce_k', True), ('reduce_to_primitive', True),
    ('bipole_cutoff_bohr', 7.), ('bipole_historical_domain', True),
    ('use_diis', True), ('smearing', .01), ('solver', 'davidson'), ('method', 'ROKS')])
def test_unqualified_components_and_sources_refuse_before_generation(tmp_path, monkeypatch, field, value):
    job, *_ = job_case(tmp_path); job[field] = value
    def fail(*a, **kw): raise AssertionError('Unadmitted request reached numerical generation')
    for method in ('ks', 'global_hybrid_ks', 'range_separated_ks'):
        monkeypatch.setattr(core._manta, f'run_periodic_{method}_with_quadrature', fail)
    with pytest.raises((ValueError, TypeError, NotImplementedError)): vq.run_periodic_job(**job)
    _no_computed_artifacts(job)


@pytest.mark.parametrize('field,quantity', CAPS)
@pytest.mark.parametrize('name',['hse06',MIXED_RANGE])
def test_simultaneous_job_caps_and_one_short_include_python_owners(tmp_path, monkeypatch, field, quantity,name):
    job, *_ = job_case(tmp_path, 'HeH', 2,name=name); job.update(max_iter=1, dry_run=True)
    budget = job['integral_engine'].budget; budget.other_live_bytes = 12345
    plan = core._manta.plan_periodic_range_separated_ks_with_quadrature; seen = []
    def capture(*a, **kw):
        p = plan(*a, **kw); seen.append((p, a[-1].other_live_bytes)); return p
    with monkeypatch.context() as patch:
        patch.setattr(core._manta, 'plan_periodic_range_separated_ks_with_quadrature', capture)
        assert vq.run_periodic_job(**job) is None
    p, live = seen[-1]
    assert len(seen) == 2 and live > budget.other_live_bytes
    for cap, name in CAPS: setattr(budget, cap, getattr(p, name))
    job.update(dry_run=False, output=tmp_path/'exact'/'job')
    r = vq.run_periodic_job(**job)
    assert not r.converged and r.iterations == 1 and budget.other_live_bytes == 12345
    assert live == 12345+r.run_fields['job_reserved_bytes']+r.integral_provenance['python_reserved_bytes']
    setattr(budget, field, getattr(p, quantity)-1)
    job['output'] = tmp_path/'short'/'job'
    def fail(*a, **kw): raise AssertionError('Resource refusal reached numerical generation')
    monkeypatch.setattr(core._manta, 'run_periodic_range_separated_ks_with_quadrature', fail)
    with pytest.raises((ValueError, RuntimeError), match='budget'): vq.run_periodic_job(**job)
    _no_computed_artifacts(job)


def test_dry_run_and_delivery_failure_preserve_lifecycle(tmp_path, monkeypatch):
    from vibeqc.output import OutputWriter
    from vibeqc.periodic_runner import _PERIODIC_OUTPUT_WRITER
    job, *_ = job_case(tmp_path/'dry'); job['dry_run'] = True
    def fail(*a, **kw): raise AssertionError('Dry run must not generate or solve')
    with monkeypatch.context() as patch:
        patch.setattr(core._manta, 'run_periodic_range_separated_ks_with_quadrature', fail)
        assert vq.run_periodic_job(**job) is None
    assert list(job['output'].parent.iterdir()) == [job['output'].with_suffix('.system')]
    job, *_ = job_case(tmp_path/'delivery')
    dispatch = OutputWriter.dispatch_role
    def fail_delivery(self, role, **kwargs):
        if role == 'qvf': raise OSError('synthetic hybrid delivery failure')
        return dispatch(self, role, **kwargs)
    monkeypatch.setattr(OutputWriter, 'dispatch_role', fail_delivery)
    before = _PERIODIC_OUTPUT_WRITER.get()
    with pytest.raises(OSError, match='synthetic hybrid delivery'): vq.run_periodic_job(**job)
    assert _PERIODIC_OUTPUT_WRITER.get() is before
    manifest = tomllib.loads(job['output'].with_suffix('.system').read_text())
    assert manifest['outputs']['status'] == 'crashed'
    assert manifest['outputs']['result_status'] == 'converged'


@pytest.mark.parametrize('spins',[1,2])
def test_screened_only_reports_actual_inactive_correction(tmp_path,spins):
    job,*_=job_case(tmp_path/'ewald',spins=spins,name='hse06')
    a=vq.run_periodic_job(**job)
    job.update(exchange_exxdiv='none',output=tmp_path/'strict'/'job')
    job['periodic_integrals']=dataclasses.replace(job['periodic_integrals'],madelung_alpha=.55,madelung_tolerance=1e-10)
    b=vq.run_periodic_job(**job)
    for field in ('density','fock'):np.testing.assert_array_equal(getattr(a.native,field),getattr(b.native,field))
    assert a.energy==b.energy and a.exchange_q0==b.exchange_q0=='not-applicable'
    pa,pb=a.integral_provenance,b.integral_provenance
    assert pa['source_identity']==pb['source_identity']
    assert pa['exchange_q0_requested']=='BvK-ewald' and pb['exchange_q0_requested']=='strict-zero-mode'
    assert pa['madelung_plan']['kernel_terms']==pb['madelung_plan']['kernel_terms']==0
    assert pa['exchange_correction_energy']==pb['exchange_correction_energy']==0
    assert 'manta_periodic_madelung' not in pb['citation_methods']
    qvf,p,_=output_evidence(job)
    assert p==pb and qvf['provenance']['exchange_q0']=='not-applicable'
    assert qvf['provenance']['run_fields']['exchange_q0_requested']=='strict-zero-mode'


@pytest.mark.parametrize('fraction',[0.,.2])
def test_cancelled_components_select_complete_descriptor_route(tmp_path,fraction):
    name='manta-job-cancelled-range-'+str(fraction)
    vq.define_functional(name,[('HYB_GGA_XC_HSE06',1.),('HYB_GGA_XC_HSE06',-1.),
        ('GGA_X_B88',.8),('GGA_C_LYP',1.)],hf_exchange_fraction=fraction)
    job,args,c,k,old=job_case(tmp_path,name=name);explicit(args,c,old)
    r=vq.run_periodic_job(**job);terms=[('coulomb',fraction,0.)] if fraction else []
    s,h,build,_=oracle(c,k,old,terms=terms)
    assert r.converged and r.native.plan.range_separated
    np.testing.assert_allclose(r.native.density,independent_scf(s,h,build,(1,),1),atol=3e-9,rtol=0)
    f,e,_=build(r.native.density)
    np.testing.assert_allclose(r.native.fock,f,atol=2e-11,rtol=0)
    np.testing.assert_allclose(components(r.native),e,atol=2e-11,rtol=0)
    p=r.integral_provenance
    assert len(p['exchange_components'])==len(terms)
    assert p['long_range_exchange_fraction']==fraction
    assert r.exchange_q0==('BvK-ewald' if fraction else 'not-applicable')
    assert output_evidence(job)[1]==p


@pytest.mark.parametrize('spins',[1,2])
@pytest.mark.parametrize('name',['lc-wpbe','wb97x'])
def test_gamma_range_functionals_keep_spin_and_kernel_contract(tmp_path,spins,name):
    job,args,c,_,old=job_case(tmp_path,spins=spins,name=name)
    job['kpoints']=None
    args['mesh']=c['mesh']=old['mesh']=manta.RegularKMesh((1,1,1),(0,0,0))
    explicit(args,c,old);s,h,build,_=oracle(c,np.zeros((1,3)),old)
    r=vq.run_periodic_job(**job);assert r.converged and r.mesh==(1,1,1)
    np.testing.assert_allclose(r.native.density,independent_scf(s,h,build,(1,) if spins==1 else (1,1),spins),atol=3e-9,rtol=0)
    f,e,_=build(r.native.density)
    np.testing.assert_allclose(r.native.fock,f,atol=2e-11,rtol=0)
    np.testing.assert_allclose(components(r.native),e,atol=2e-11,rtol=0)


def test_yukawa_descriptor_refuses_before_numerical_generation(tmp_path,monkeypatch):
    vq.define_functional('manta-job-yukawa-test',[('HYB_GGA_XC_CAMY_B3LYP',1.)])
    job,*_=job_case(tmp_path);job['functional']='manta-job-yukawa-test'
    def fail(*a,**kw):raise AssertionError('Unsupported kernel reached generation')
    monkeypatch.setattr(core._manta,'run_periodic_range_separated_ks_with_quadrature',fail)
    with pytest.raises(ValueError,match='range-separated'):vq.run_periodic_job(**job)
    _no_computed_artifacts(job)
