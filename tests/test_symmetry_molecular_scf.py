"""Public HF method calls preserve the screened source and evaluated state."""
import json

import numpy as np
import pytest

import vibeqc as vq
from vibeqc.symmetry_shared import Budget
from vibeqc.symmetry_molecular_scf import _density_orbitals
from vibeqc.solvers.operators import OperatorFrame
from vibeqc.symmetry_molecular_hf import ScreenedMolecularHF, ScreenedMolecularJK
from tests.test_symmetry_molecular_routing import case as source_case
from tests.test_symmetry_quartets import molecular_case, masked_source, dense_jk

BUDGET = Budget(512 << 20, 10**12)


def config(rotations, translations, **kwargs):
    return vq.MolecularSymmetryOptions(rotations, translations,
        source_tolerance=1e-10, maximum_omitted_norm=1e-12, budget=BUDGET, **kwargs)


@pytest.mark.parametrize('method', ['rhf', 'uhf'])
@pytest.mark.parametrize('max_iter', [0, 1, 80])
@pytest.mark.parametrize('executor', ['native', 'ordered'])
def test_public_method_owned_state_and_source(method, max_iter, executor):
    mol, basis, rotations, translations = molecular_case('water')
    opts = vq.RHFOptions() if method == 'rhf' else vq.UHFOptions()
    opts.initial_guess = vq.InitialGuess.HCORE
    opts.max_iter = max_iter
    opts.conv_tol_energy, opts.conv_tol_grad = 1e-11, 1e-9
    opts.level_shift = .2 if max_iter == 1 else 0.
    reduction = config(rotations, translations, executor=executor)
    result = getattr(vq, 'run_'+method)(mol, basis, opts, symmetry_reduction=reduction)
    assert result.converged == (max_iter == 80)
    assert result.n_iter <= max_iter
    assert result.orbital_kind == 'natural'
    channels = result.fock_build.channels
    h = vq.compute_kinetic(basis)+vq.compute_nuclear(basis, mol)
    eri = np.asarray(vq.compute_eri(basis)).reshape((basis.nbasis,)*4)
    ds = [c.density for c in channels]
    j = dense_jk(eri, sum(ds))[0]
    exchange = .5 if method == 'rhf' else 1.
    energy = mol.nuclear_repulsion()
    for suffix, d, channel in zip(('',) if method == 'rhf' else ('_alpha', '_beta'), ds, channels):
        k = dense_jk(eri, d)[1]
        np.testing.assert_allclose(channel.operator.matrix, h+j-exchange*k, atol=2e-11, rtol=0)
        energy += np.einsum('pq,qp', d, h+.5*j-.5*exchange*k)
        c, occ = getattr(result, 'mo_coeffs'+suffix), getattr(result, 'occupations'+suffix)
        np.testing.assert_allclose((c*occ)@c.T, d, atol=2e-11, rtol=0)
        np.testing.assert_allclose(c.T@result.overlap@c, np.eye(len(occ)), atol=2e-11, rtol=0)
        np.testing.assert_allclose(getattr(result, 'mo_energies'+suffix),
            np.diag(c.T@channel.operator.matrix@c), atol=2e-11, rtol=0)
        with pytest.raises(ValueError):
            c.setflags(write=True)
    assert abs(energy-result.energy) < 2e-11
    assert result.fock_build.energy.total == result.energy
    assert result.run_fields['evaluated_density_id'] == result.fock_build.density_id
    assert result.run_fields['scf_stop_reason'] == ('converged' if max_iter == 80 else 'noiter' if max_iter == 0 else 'iteration_limit')
    provenance = result.symmetry_provenance
    terminal = provenance['terminal_work']
    assert terminal['reused_from_builds']
    assert terminal['work']['shell_calls'] == 0
    assert terminal['scalar_products'] == terminal['block_contractions'] == 0
    assert provenance['terminal_capture']['captures'] > 0
    provenance['scf_work']['builds'] = -1
    assert result.symmetry_provenance['scf_work']['builds'] > 0
    assert result.ecp_provenance_verified and not result.ecp_operator_applied
    json.dumps(result.run_fields, allow_nan=False)
    with pytest.raises(NotImplementedError, match='derivatives'):
        (vq.compute_gradient if method == 'rhf' else vq.compute_gradient_uhf)(mol, basis, result)


def test_fractional_read_noiter_preserves_evaluated_density_and_occupations():
    mol, basis, rotations, translations = molecular_case('water')
    s = vq.compute_overlap(basis)
    val, vec = np.linalg.eigh(s)
    x = vec/np.sqrt(val)
    occ = np.array([2., 2., 2., 1.5, 1.5, .5, .5])
    d = (x*occ)@x.T
    opts = vq.RHFOptions(); opts.initial_guess = vq.InitialGuess.READ
    opts.max_iter = 0; opts.read_density = d
    result = vq.run_rhf(mol, basis, opts, symmetry_reduction=config(rotations, translations, screening='audit'))
    np.testing.assert_allclose(result.density, d, atol=2e-12, rtol=0)
    np.testing.assert_allclose(result.occupations, occ, atol=2e-12, rtol=0)
    np.testing.assert_allclose((result.mo_coeffs*result.occupations)@result.mo_coeffs.T, d, atol=2e-11, rtol=0)
    assert not result.converged
    np.testing.assert_array_equal(opts.read_density, d)


@pytest.mark.parametrize('executor,expected', [('ordered', None), ('auto', 1), ('auto', 2)])
def test_method_ordered_qualification_and_short_route_skip_setup(executor, expected, monkeypatch):
    import vibeqc.symmetry_quartets_streaming as stream
    mol, basis, rotations, translations = molecular_case('water')
    def forbidden(*args, **kwargs):
        raise AssertionError('ordered producer qualification must avoid integral audit')
    monkeypatch.setattr(stream, '_read', forbidden)
    opts = vq.RHFOptions(); opts.initial_guess = vq.InitialGuess.HCORE
    opts.conv_tol_energy, opts.conv_tol_grad = 1e-11, 1e-9
    result = vq.run_rhf(mol, basis, opts, symmetry_reduction=config(rotations,
        translations, executor=executor, expected_builds=expected))
    assert result.converged
    p = result.symmetry_provenance
    assert p['source_audit_work']['shell_calls'] == 0
    assert p['screening']['qualification_scope'] == 'ordered-source'
    assert all(r['samples'] == [] and r['calibration_seconds'] == 0 for r in p['routes'])
    eri = np.asarray(vq.compute_eri(basis)).reshape((basis.nbasis,)*4)
    j, k = dense_jk(eri, result.density)
    h = vq.compute_kinetic(basis)+vq.compute_nuclear(basis, mol)
    np.testing.assert_allclose(result.fock, h+j-.5*k, atol=2e-11, rtol=0)
    energy = mol.nuclear_repulsion()+np.einsum('pq,qp', result.density, h+.5*j-.25*k)
    assert abs(energy-result.energy) < 2e-11


def test_complete_degenerate_retained_space_and_leakage_refusal():
    screen, _, mol, basis = source_case('cartesian')
    s = vq.compute_overlap(basis)
    val, vec = np.linalg.eigh(s)
    # Retain an arbitrary rotated degenerate space explicitly. This is an
    # output-frame algebra test, not admission of a symmetry-truncated SCF.
    x = vec[:, 1:]/np.sqrt(val[1:])
    q, _ = np.linalg.qr(np.random.default_rng(78).normal(size=(x.shape[1],)*2))
    frame = OperatorFrame(basis_id='retained-test', n=basis.nbasis,
        ordering='native AO', image_gauge='molecular', overlap=s, X=x@q)
    hf = ScreenedMolecularHF(ScreenedMolecularJK(screen, budget=BUDGET), frame=frame,
        hcore=vq.compute_kinetic(basis)+vq.compute_nuclear(basis, mol), nuclear_energy=0.)
    d = 2*x@x.T
    channel = hf.evaluate((d,), method='RHF', generation=0).build.channels[0]
    c, _, occ = _density_orbitals(channel)
    np.testing.assert_allclose((c*occ)@c.T, d, atol=2e-11, rtol=0)
    leaked = hf.evaluate((np.eye(basis.nbasis),), method='RHF', generation=1).build.channels[0]
    with pytest.raises(ValueError, match='retained metric'):
        _density_orbitals(leaked)


def test_public_refusals_precede_source_reads(monkeypatch):
    import vibeqc.symmetry_molecular_scf as module
    mol, basis, rotations, translations = molecular_case('water')
    reduction = config(rotations, translations)
    def forbidden(*args, **kwargs):
        pytest.fail('source qualification reached on a refused request')
    monkeypatch.setattr(module, 'qualify_manta_quartets', forbidden)
    for name, value in [('density_fit', True), ('cosx', True), ('aux_basis', 'aux'),
                        ('initial_guess', vq.InitialGuess.SAD)]:
        opts = vq.RHFOptions(); opts.initial_guess = vq.InitialGuess.HCORE
        setattr(opts, name, value)
        with pytest.raises(ValueError):
            vq.run_rhf(mol, basis, opts, symmetry_reduction=reduction)
    with pytest.raises(ValueError, match='another engine'):
        vq.run_rhf(mol, basis, symmetry_reduction=reduction, integral_engine=object())
    with pytest.raises(TypeError, match='MolecularSymmetryOptions'):
        vq.run_rhf(mol, basis, symmetry_reduction=True)
    with pytest.raises(TypeError, match='expected builds'):
        config(rotations, translations, executor='auto')
    displaced = vq.Molecule([vq.Atom(a.Z, np.array(a.xyz)+[.1, 0, 0]) for a in mol.atoms])
    with pytest.raises(ValueError, match='geometry'):
        vq.run_rhf(displaced, basis, symmetry_reduction=reduction)


def test_public_default_and_orbital_output_metadata(tmp_path):
    from vibeqc.output.formats.qvf import qvf_wf_data, write_qvf
    from vibeqc.output.plan import OutputPlan
    import zipfile
    mol, basis, rotations, translations = molecular_case('water')
    result = vq.run_rhf(mol, 'sto-3g', symmetry_reduction=config(rotations, translations))
    assert result.converged
    data = qvf_wf_data(result, basis, mol)
    assert data['mo_metadata']['orbital_kind'] == 'natural'
    c = data['mo_coefficients'].T
    np.testing.assert_allclose((c*np.array(data['mo_metadata']['occupations']))@c.T,
        result.density, atol=2e-7, rtol=0)
    plan = OutputPlan.from_run_job_kwargs(output=tmp_path/'symmetry', method='rhf',
        basis='sto-3g', functional=None)
    path = write_qvf(plan.stem, plan, molecule=mol, basis=basis, result=result, wf_data=data)
    with zipfile.ZipFile(path) as zf:
        manifest = json.loads(zf.read('manifest.json'))
        fields = manifest['provenance']['run_fields']
        assert fields['evaluated_density_id'] == result.fock_build.density_id
        assert fields['fock_source_id'] == result.fock_build.source.identity
        assert fields['orbital_kind'] == 'natural'


def test_one_electron_operator_must_match_geometry_action(monkeypatch):
    import vibeqc.symmetry_molecular_scf as module
    mol, basis, rotations, translations = molecular_case('water')
    nuclear = module.core.compute_nuclear
    def incompatible(basis, molecule):
        h = nuclear(basis, molecule)
        h[-1, -1] += .01
        return h
    monkeypatch.setattr(module.core, 'compute_nuclear', incompatible)
    with pytest.raises(ValueError, match='operator equivariance'):
        vq.run_rhf(mol, basis, symmetry_reduction=config(rotations, translations))


@pytest.mark.parametrize('executor', ['native', 'ordered', 'auto'])
def test_prepared_source_reuses_audit_but_not_electronic_state(monkeypatch, executor):
    import vibeqc.symmetry_molecular_scf as module
    mol, basis, rotations, translations = molecular_case('water')
    reduction = config(rotations, translations, executor=executor,
        **({'expected_builds': 30} if executor == 'auto' else {}))
    source = vq.prepare_molecular_symmetry(mol, basis, symmetry_reduction=reduction)
    first_receipt = source.preparation_receipt
    assert first_receipt['source_audit_work']['shell_calls'] == (0 if executor == 'ordered' else basis.nshells**4)
    assert first_receipt['persistent_matrix_bytes'] == 16*basis.nbasis**2
    first_receipt['identity'] = 'caller mutation'
    assert source.preparation_receipt['identity'] == source.identity
    # Same nuclear/basis source, a different charge and open-shell state.
    cation = vq.Molecule(mol.atoms, charge=1, multiplicity=2)
    results = []
    def forbidden(*args, **kwargs):
        pytest.fail('prepared source repeated qualification or one-electron work')
    with monkeypatch.context() as patch:
        patch.setattr(module, 'qualify_manta_quartets', forbidden)
        for name in ('compute_overlap', 'compute_kinetic', 'compute_nuclear'):
            patch.setattr(module.core, name, forbidden)
        for method, molecule in [('rhf', mol), ('uhf', cation), ('rhf', mol)]:
            opts = vq.RHFOptions() if method == 'rhf' else vq.UHFOptions()
            opts.initial_guess = vq.InitialGuess.HCORE
            opts.conv_tol_energy, opts.conv_tol_grad = 1e-11, 1e-9
            result = getattr(vq, 'run_'+method)(molecule, basis, opts, symmetry_reduction=source)
            assert result.converged
            provenance = result.symmetry_provenance
            assert provenance['source_audit_work']['shell_calls'] == 0
            assert provenance['source_setup_seconds'] == 0
            assert provenance['one_electron_qualification']['one_electron_build_calls'] == 0
            assert provenance['prepared_source']['reused']
            results.append(result)
    assert results[0].fock_build.source.identity == results[-1].fock_build.source.identity
    np.testing.assert_allclose(results[0].density, results[-1].density, atol=2e-12, rtol=0)
    for result, count in zip(results, (10, 9, 10)):
        assert abs(np.trace(result.overlap@result.density)-count) < 1e-11
        # The independently generated tensor is still the full source.
        eri = np.asarray(vq.compute_eri(basis)).reshape((basis.nbasis,)*4)
        ds = [c.density for c in result.fock_build.channels]
        j = dense_jk(eri, sum(ds))[0]
        h = vq.compute_kinetic(basis)+vq.compute_nuclear(basis, mol)
        for d, channel in zip(ds, result.fock_build.channels):
            k = dense_jk(eri, d)[1]
            np.testing.assert_allclose(channel.operator.matrix,
                h+j-(.5 if len(ds) == 1 else 1.)*k, atol=3e-11, rtol=0)
    if executor == 'auto':
        assert len(results[0].symmetry_provenance['routes']) <= 4
        assert len(results[1].symmetry_provenance['routes']) >= 1
        assert results[2].symmetry_provenance['routes'] == []
        assert results[2].symmetry_provenance['reused_routes']
        assert len(source._jk.route_calibrations) <= 5
        for result in results:
            p = result.symmetry_provenance
            assert p['route_calibration'] == 'on-demand'
            assert p['scf_seconds'] >= 0
            assert p['scf_seconds']+p['scf_calibration_seconds'] == pytest.approx(p['scf_elapsed_seconds'])
            assert p['scf_calibration_seconds']+p['terminal_calibration_seconds'] == pytest.approx(
                sum(r['calibration_seconds'] for r in p['routes']))
    # The reusable source holds only calibration receipts, never a state.
    assert source._jk.totals.builds == 0 and source._jk.last_build is None
    assert source._jk.terminal_capture['snapshot_bytes'] == 0
    assert not source._jk.terminal_capture['enabled']
    assert (results[0].symmetry_provenance['scf_work']['builds']
            == results[-1].symmetry_provenance['scf_work']['builds'])


def test_prepared_source_rejects_changed_domain_before_reads(monkeypatch):
    from vibeqc.symmetry_quartets import _MantaSource
    mol, basis, rotations, translations = molecular_case('water')
    source = vq.prepare_molecular_symmetry(mol, basis,
        symmetry_reduction=config(rotations, translations))
    def forbidden(*args, **kwargs):
        pytest.fail('changed prepared domain reached an integral read')
    monkeypatch.setattr(_MantaSource, 'evaluate', forbidden)
    atoms = [vq.Atom(a.Z, np.array(a.xyz)+[.1, 0., 0.]) for a in mol.atoms]
    displaced = vq.Molecule(atoms)
    changed_basis = vq.BasisSet(displaced, 'sto-3g')
    shells = basis.shells()
    radial = [vq._vibeqc_core.ShellInfo(s.atom_index, s.l, s.pure,
        [x*(1.01 if i == 0 else 1.) for x in s.exponents], s.coefficients,
        s.origin) for i, s in enumerate(shells)]
    changed_radial = vq.BasisSet(mol, radial, 'changed radial snapshot', True)
    nuclei = vq.Molecule([vq.Atom(10 if i == 0 else a.Z, a.xyz)
        for i, a in enumerate(mol.atoms)])
    for molecule, candidate in [(displaced, basis), (displaced, changed_basis),
                                (mol, vq.BasisSet(mol, '6-31g')),
                                (mol, changed_radial), (nuclei, basis)]:
        with pytest.raises(ValueError, match='(geometry|basis)'):
            vq.run_rhf(molecule, candidate, symmetry_reduction=source)
    with pytest.raises(ValueError, match='another engine'):
        vq.run_rhf(mol, basis, symmetry_reduction=source, integral_engine=object())


def test_prepared_source_lease_releases_after_failure_and_is_process_local(monkeypatch):
    import pickle
    import vibeqc.symmetry_molecular_scf as module
    mol, basis, rotations, translations = molecular_case('water')
    source = vq.prepare_molecular_symmetry(mol, basis,
        symmetry_reduction=config(rotations, translations))
    with source._lease(mol, basis):
        with pytest.raises(RuntimeError, match='already in use'):
            vq.run_rhf(mol, basis, symmetry_reduction=source)
    with pytest.raises(TypeError, match='process-local'):
        pickle.dumps(source)
    with monkeypatch.context() as patch:
        patch.setattr(module.os, 'getpid', lambda: source._pid+1)
        with pytest.raises(ValueError, match='another process'):
            vq.run_rhf(mol, basis, symmetry_reduction=source)
    opts = vq.RHFOptions(); opts.initial_guess = vq.InitialGuess.READ; opts.max_iter = 0
    # Bad READ must not leave a live lease or contaminate the next run.
    opts.read_density = np.zeros((1, 1))
    with pytest.raises((ValueError, RuntimeError)):
        vq.run_rhf(mol, basis, opts, symmetry_reduction=source)
    assert vq.run_rhf(mol, basis, symmetry_reduction=source).converged


def test_terminal_capture_admission_precedes_new_route_probes(monkeypatch):
    mol, basis, rotations, translations = molecular_case('water')
    source = vq.prepare_molecular_symmetry(mol, basis,
        symmetry_reduction=config(rotations, translations, executor='auto', expected_builds=30))
    source._jk._budget = source._jk.requirements(('jk',))
    def forbidden(*args, **kwargs):
        raise AssertionError('terminal capture budget refused after calibration work')
    monkeypatch.setattr(ScreenedMolecularJK, 'calibrate_routes', forbidden)
    with pytest.raises(MemoryError, match='budget'):
        source._method_builder('rhf')
    assert not source._jk.route_calibrations and source._jk.totals.builds == 0


def ri_config(rotations, translations, characters, **kwargs):
    return vq.MolecularRISymmetryOptions(rotations, translations, characters,
        auxiliary_basis=kwargs.pop('auxiliary_basis', 'sto-3g'),
        source_tolerance=1e-10, budget=BUDGET, **kwargs)


@pytest.mark.parametrize('kind,method', [('water', 'rhf'), ('ammonia', 'uhf')])
@pytest.mark.parametrize('max_iter', [0, 1, 80])
def test_ri_public_method_evaluated_state_and_preparation(kind, method, max_iter):
    from tests.test_symmetry_ri import case, dense
    source, mol, basis, auxiliary, r, t = case(kind)
    reduction = ri_config(r, t, source.irreps.characters)
    prepared = vq.prepare_molecular_symmetry(mol, basis, symmetry_reduction=reduction)
    assert prepared.preparation_receipt['source_identity'] == source.source_identity
    opts = vq.RHFOptions() if method == 'rhf' else vq.UHFOptions()
    opts.initial_guess, opts.max_iter = vq.InitialGuess.HCORE, max_iter
    opts.conv_tol_energy, opts.conv_tol_grad = 1e-11, 1e-9
    result = getattr(vq, 'run_'+method)(mol, basis, opts, symmetry_reduction=prepared)
    assert result.converged == (max_iter == 80)
    h = vq.compute_kinetic(basis)+vq.compute_nuclear(basis, mol)
    # Independent three-center integrals with the exact native retained factor.
    factors = np.einsum('PA,Puv->Auv', source._native.factor, vq.compute_3c_eri(basis, auxiliary))
    channels = result.fock_build.channels
    j = dense(factors, sum(c.density for c in channels))[0]
    energy = mol.nuclear_repulsion()
    for suffix, channel in zip(('',) if method == 'rhf' else ('_alpha', '_beta'), channels):
        d = channel.density
        f = h+j-(.5 if method == 'rhf' else 1.)*dense(factors, d)[1]
        np.testing.assert_allclose(channel.operator.matrix, f, atol=3e-11, rtol=0)
        energy += .5*np.einsum('pq,qp', d, h+f)
        c, occ = getattr(result, 'mo_coeffs'+suffix), getattr(result, 'occupations'+suffix)
        np.testing.assert_allclose((c*occ)@c.T, d, atol=2e-11, rtol=0)
    assert abs(result.energy-energy) < 3e-11
    provenance = result.symmetry_provenance
    assert provenance['source_kind'] == 'abelian-ri'
    assert not any(provenance['fitting_setup_work'].values())
    assert provenance['prepared_source']['reused']
    assert 'source_audit_work' not in provenance and 'screening' not in provenance
    assert provenance['terminal_capture']['enabled']
    assert provenance['terminal_capture']['captures'] > 0
    assert provenance['terminal_work']['scalar_products'] == 0
    assert provenance['terminal_work']['transport_scalar_products'] == 0
    assert provenance['terminal_work']['reused_from_builds']
    assert provenance['scf_work']['work']['shell_calls'] == 0
    with pytest.raises((TypeError, NotImplementedError), match='qualified|symmetry|gradient'):
        (vq.compute_gradient(mol, basis, result) if method == 'rhf'
         else vq.compute_gradient_uhf(mol, basis, result))


def test_ri_prepared_reuse_routes_and_geometry_guard():
    from tests.test_symmetry_ri import case
    source, mol, basis, _, r, t = case('water')
    reduction = ri_config(r, t, source.irreps.characters, executor='auto', expected_builds=2)
    prepared = vq.prepare_molecular_symmetry(mol, basis, symmetry_reduction=reduction)
    a = vq.run_rhf(mol, basis, symmetry_reduction=prepared)
    b = vq.run_rhf(mol, basis, symmetry_reduction=prepared)
    assert a.energy == b.energy
    assert a.symmetry_provenance['routes'] and not b.symmetry_provenance['routes']
    assert b.symmetry_provenance['reused_routes']
    assert a.symmetry_provenance['scf_work']['builds'] == b.symmetry_provenance['scf_work']['builds']
    assert not prepared._jk.terminal_capture['enabled']
    assert prepared._jk.terminal_capture['snapshot_bytes'] == 0
    assert prepared._jk.totals.builds == 0
    moved = vq.Molecule([vq.Atom(a.Z, np.asarray(a.xyz)+[.01, 0., 0.]) for a in mol.atoms])
    moved_basis = vq.BasisSet(moved, 'sto-3g')
    with pytest.raises(ValueError, match='geometry or basis changed'):
        vq.run_rhf(moved, moved_basis, symmetry_reduction=prepared)


def test_ri_one_electron_rejects_unqualified_hamiltonian(monkeypatch):
    from tests.test_symmetry_ri import case
    source, mol, basis, _, r, t = case('water')
    reduction = ri_config(r, t, source.irreps.characters)
    from vibeqc import _vibeqc_core as core
    original = core.compute_nuclear
    def broken(*args):
        value = np.array(original(*args)); value[-1, -1] += .01
        return value
    monkeypatch.setattr(core, 'compute_nuclear', broken)
    with pytest.raises(ValueError, match='one-electron source fails'):
        vq.prepare_molecular_symmetry(mol, basis, symmetry_reduction=reduction)


def test_ri_terminal_capture_admission_precedes_new_route_probes(monkeypatch):
    from tests.test_symmetry_ri import case
    from vibeqc.symmetry_ri import MolecularRIJK
    source, mol, basis, _, r, t = case('water')
    prepared = vq.prepare_molecular_symmetry(mol, basis, symmetry_reduction=
        ri_config(r, t, source.irreps.characters, executor='auto', expected_builds=30))
    prepared._jk._budget = prepared._jk.requirements(('jk',))
    def forbidden(*args, **kwargs):
        pytest.fail('capture budget refused after calibration work')
    monkeypatch.setattr(MolecularRIJK, 'calibrate_routes', forbidden)
    with pytest.raises(MemoryError, match='budget'):
        prepared._method_builder('rhf')
    assert not prepared._jk.route_calibrations and prepared._jk.totals.builds == 0


def demand_source(family):
    if family == 'ri':
        from tests.test_symmetry_ri import case
        source, mol, basis, _, r, t = case('water')
        options = ri_config(r, t, source.irreps.characters, executor='auto', expected_builds=30)
    else:
        mol, basis, r, t = molecular_case('water')
        options = config(r, t, executor='auto', expected_builds=30)
    return mol, basis, vq.prepare_molecular_symmetry(mol, basis, symmetry_reduction=options)


@pytest.mark.parametrize('family', ['quartets', 'ri'])
def test_method_routes_follow_actual_requests_and_reuse_only_scalar_profiles(family, monkeypatch):
    mol, basis, prepared = demand_source(family)
    child, measured, reused = prepared._method_builder('rhf')
    assert measured == reused == [] and child.route_calibrations == ()
    calls = []
    original = child.calibrate_routes
    def calibrate(**kw):
        calls.append(kw['targets'])
        return original(**kw)
    monkeypatch.setattr(child, 'calibrate_routes', calibrate)
    d = np.random.default_rng(8102).normal(size=(basis.nbasis, basis.nbasis))
    child.build_J(d); child.build_J(d*2)
    assert calls == [('j',)] and len(measured) == 1
    assert set(dict(prepared._jk.route_calibrations)) == {('j',)}
    child.build_K(d)
    assert calls == [('j',), ('k',)] and len(measured) == 2
    assert [r.targets for r in measured] == calls
    assert prepared._jk._route_request is None
    assert prepared._jk.totals.builds == 0 and prepared._jk.terminal_capture['snapshot_bytes'] == 0
    other, new, used = prepared._method_builder('uhf')
    def forbidden(**kw):
        pytest.fail('a used scalar profile was calibrated again')
    monkeypatch.setattr(other, 'calibrate_routes', forbidden)
    other.build_K(d); other.build_K(d.T.copy())
    assert new == [] and used == [('k',)]
    assert other.totals.builds == 2 and child.totals.builds == 3


@pytest.mark.parametrize('family', ['quartets', 'ri'])
@pytest.mark.parametrize('method', ['rhf', 'uhf'])
def test_native_noiter_skips_probes_without_poisoning_future_workload(family, method, monkeypatch):
    mol, basis, prepared = demand_source(family)
    options = vq.RHFOptions() if method == 'rhf' else vq.UHFOptions()
    options.initial_guess = vq.InitialGuess.HCORE
    options.max_iter = 0
    result = getattr(vq, 'run_'+method)(mol, basis, options, symmetry_reduction=prepared)
    p = result.symmetry_provenance
    assert result.n_iter == 0 and p['single_evaluation_bound']
    assert p['scf_work']['builds'] == 1
    assert len(p['routes']) == 1 and p['routes'][0]['expected_builds'] == 1
    assert p['routes'][0]['targets'] == (['jk'] if method == 'rhf' else ['j', 'k', 'k'])
    assert p['routes'][0]['samples'] == [] and p['scf_calibration_seconds'] == 0
    assert p['terminal_work']['scalar_products'] == 0
    assert prepared._jk.route_calibrations == ()
    # A later iterative invocation still uses the declared workload, rather
    # than inheriting NOITER's zero-probe baseline decision.
    child, measured, _ = prepared._method_builder(method)
    d = np.eye(basis.nbasis)
    if method == 'rhf':
        child.build_g_rhf(d)
    else:
        child.build_jk_uhf(.5*d, .5*d)
    assert len(measured) == 1 and measured[0].expected_builds == 30
    assert measured[0].samples and prepared._jk.route_calibrations
    def forbidden(*args, **kwargs):
        pytest.fail('NOITER discarded an already paid valid route profile')
    monkeypatch.setattr(type(child), 'calibrate_routes', forbidden)
    repeated = getattr(vq, 'run_'+method)(mol, basis, options, symmetry_reduction=prepared)
    assert repeated.symmetry_provenance['routes'] == []
    assert repeated.symmetry_provenance['reused_routes']
    assert abs(repeated.energy-result.energy) < 1e-10


@pytest.mark.parametrize('family', ['quartets', 'ri'])
def test_demand_calibration_rejects_invalid_inputs_and_failed_probes_before_publication(family, monkeypatch):
    _, basis, prepared = demand_source(family)
    child, measured, reused = prepared._method_builder('rhf')
    def forbidden(**kw):
        pytest.fail('invalid density started calibration')
    with monkeypatch.context() as patch:
        patch.setattr(child, 'calibrate_routes', forbidden)
        with pytest.raises(ValueError, match='density shape or values'):
            child.build_J(np.full((basis.nbasis, basis.nbasis), np.nan))
        with pytest.raises(ValueError, match='dimensions disagree'):
            child.build_channels((), targets=('j',))
    execute = child._execute
    def corrupt(densities, targets, executor):
        j, k, receipt = execute(densities, targets, executor)
        if executor in ('native', 'blocked'):
            j += .01
        return j, k, receipt
    monkeypatch.setattr(child, '_execute', corrupt)
    with pytest.raises(ValueError, match='disagree'):
        child.build_J(np.eye(basis.nbasis))
    assert measured == reused == [] and prepared._jk.route_calibrations == ()
    assert child.totals.builds == 0
    monkeypatch.setattr(child, '_execute', execute)
    child.build_J(np.eye(basis.nbasis))
    assert len(measured) == 1 and len(prepared._jk.route_calibrations) == 1
