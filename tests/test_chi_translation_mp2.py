"""Same-source finite algebra checks, separate from vq scientific acceptance."""
from dataclasses import asdict, replace
from itertools import product
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import eigh

from vibeqc.periodic.chi import local_mp2 as local, reference as refs
from vibeqc.periodic.chi import checkpoint as cp
from vibeqc.periodic.chi.canonical_mp2 import evaluate_same_source_mp2
from vibeqc.periodic_fitted_support import ChiCoulombControls, ChiFittedOperator
from vibeqc.periodic_fit_source import RSGDFFitSource, factor_digest
from tests.test_chi_canonical_mp2 import spin_orbital_oracle
from tests.test_dlpno_source import _automatic_translation_source


def fixture(monkeypatch, mesh=(4, 1, 1), frozen=True):
    """Fourier-transform an independently prescribed real finite Hamiltonian.

    Canonical Bloch phases are deliberately complex, including at TRIM.
    No production adapter participates in constructing the input arrays.
    """
    model, active, core, home = _automatic_translation_source(mesh, frozen)
    cells = np.array(list(product(*(range(n) for n in mesh))))
    nk, nao = len(cells), 4
    k = cells / np.asarray(mesh)
    phase = np.exp(2j*np.pi*(k @ cells.T))
    per_k = []
    for matrix in (model.overlap, model.reference.fock):
        per_k.append(np.array([sum(phase[i,r]*matrix[:nao,r*nao:(r+1)*nao]
                                      for r in range(nk)) for i in range(nk)]))
    s, f = per_k
    energies, coeff = zip(*(eigh(fk, sk) for fk,sk in zip(f,s)))
    c = np.array(coeff)*np.exp(1j*np.arange(nk*nao).reshape(nk,nao)*.23)[:,None,:]
    e = np.array(energies)
    factors = {(i,j): sum(phase[i,r].conjugate()*phase[j,t]*home[:,r*nao:(r+1)*nao,t*nao:(t+1)*nao]
                           for r in range(nk) for t in range(nk))
               for i in range(nk) for j in range(nk)}
    groups = tuple((tuple(k[q]),tuple((i,j) for i,j in factors
        if np.array_equal((cells[j]-cells[i]) % mesh, cells[q]))) for q in range(nk))
    lattice = np.eye(3)*2*np.pi
    records = [dict(q=q,lattice=lattice.tolist(),metric_eigenvalues=[1.,2.],
                    metric_keep=[True,True],rank=2) for q,_ in groups]
    system = SimpleNamespace(dim=3,lattice=lattice,charge=0,multiplicity=1,
        unit_cell=[SimpleNamespace(Z=2,xyz=[0.,0.,0.]),SimpleNamespace(Z=2,xyz=[1.,0.,0.])],
        n_electrons=lambda:4)
    shells = [SimpleNamespace(atom_index=i,l=0,pure=True,exponents=[1.],coefficients=[1.],
                              origin=system.unit_cell[i].xyz) for i in (0,1,0,1)]
    basis = SimpleNamespace(name='finite-model',nbasis=nao,shells=lambda:shells)
    controls = ChiCoulombControls(6.2,6.2,10.,8.,5.,.6)
    desc = RSGDFFitSource('native-chi-classic-finite-contract',1,'classic','model',
        basis.name,cp._basis_record(basis)['sha256'],'model-aux','aux-hash',tuple(sorted(asdict(controls).items())),
        1e-9,'canonical',groups,tuple(map(tuple,k)),10.,None,
        tuple((p,a.shape) for p,a in sorted(factors.items())),factor_digest(factors),
        executed_domains_json=json.dumps(dict(q_records=records,controls=asdict(controls),
                                             schema='vibeqc.chi.fitted-domains/v1')))
    operator = ChiFittedOperator(factors,desc,np.ones(nk)/nk)
    hf = SimpleNamespace(energy=-4.,converged=True,overlap=s,fock=f,mo_coeffs=c,mo_energies=e,
        density=2*c[:,:,:2] @ c[:,:,:2].conj().transpose(0,2,1),
        occupations=np.tile([2.,2.,0.,0.],(nk,1)),kpoints_cart=k,kpoint_weights=np.ones(nk)/nk,
        aiccm2026dev_b=cp._AttributeRecord(dict(electronic_method='RHF',backend='ri')),
        exact_exchange_assembly=cp._AttributeRecord(dict(c_full=1.,c_sr=0.)),
        finite_torus_convention=cp._AttributeRecord(dict(exchange_q0='bvk-ewald',coulomb_kernel='3d-periodic-g0',
                                                         character_mesh_shape=mesh)),
        fitted_operator=operator,executed_fit_source=replace(desc,executed_domains_json=json.dumps(
            dict(fitted_operator_identity=operator.identity))),process_provenance={'model':1},
        chi_scf_acceptance=dict(final_rebuild=True,stationary=True,energy_residual=0.,
            energy_consistency_error=0.,orbital_residual=0.,density_refill_residual=0.),
        _chi_hf_request_identity='model-request')
    cp.bind_chi_restart_inputs(hf,system,basis,mesh,None,'RHF',dict(
        fit_source='classic',backend='ri',coulomb_controls=controls))
    monkeypatch.setattr(refs,'loaded_process_provenance',lambda:{'model':1})
    return refs._seal_chi_hf_result(hf,'RHF'), model, home


@pytest.mark.parametrize('mesh,core', [((1,1,1),0),((3,1,1),0),((4,1,1),1),((2,2,1),0)])
@pytest.mark.parametrize('provider', ['dense', 'streamed'])
@pytest.mark.parametrize('compact_storage', [False, True])
def test_raw_full_space_matches_independent_canonical(monkeypatch,mesh,core,provider,compact_storage,record_property):
    ref, model, home = fixture(monkeypatch,mesh, bool(core))
    source = local.build_chi_translation_source(ref,frozen_core=core,factor_provider=provider)
    np.testing.assert_allclose(source.overlap.blocks,
        model.overlap[:4].reshape(4,len(home[0])//4,4).transpose(1,0,2),atol=1e-13)
    if provider == 'dense':
        np.testing.assert_allclose(source.factors.home_factors,home,atol=1e-13)
    else:
        assert not hasattr(source.factors, 'home_factors')
        assert source.provenance['storage']['home_factor_bytes'] == 0
    canonical = evaluate_same_source_mp2(ref,frozen_core=core)
    hf = ref.hf_result
    # Independent dense finite-system oracle, expanded only in this test.
    nk = int(np.prod(mesh)); nao = 4; nocc = 2*nk
    indices = np.arange(nk*nao).reshape(*mesh,nao)
    expanded = np.concatenate([home[:,np.roll(indices,cell,axis=(0,1,2)).ravel()][:,:,
        np.roll(indices,cell,axis=(0,1,2)).ravel()] for cell in product(*(range(n) for n in mesh))])
    eps, can = eigh(model.reference.fock, model.overlap)
    ov = np.einsum('Pmn,mi,na->Pia',expanded,can[:,core*nk:nocc],can[:,nocc:],optimize=True)
    integrals = np.einsum('Pia,Pjb->ijab',ov,ov)
    eo,ev=eps[core*nk:nocc],eps[nocc:]
    denominator=eo[:,None,None,None]+eo[None,:,None,None]-ev[None,None,:,None]-ev[None,None,None,:]
    independent=float(np.sum(integrals*(2*integrals-integrals.swapaxes(2,3))/denominator)/nk)
    if nk == 1:
        assert independent == pytest.approx(spin_orbital_oracle(hf.fitted_operator,
            [(hf.mo_coeffs,hf.mo_energies,2)]*2,frozen=(core,core)),abs=1e-13)
    result = local.run_chi_local_mp2(ref,frozen_core=core,max_iter=250,factor_provider=provider,
                                    compact_storage=compact_storage,
                                    conv_tol_energy=1e-13,conv_tol_residual=1e-12)
    assert result.converged
    assert result.raw_e_corr_per_cell == pytest.approx(independent,abs=2e-11)
    assert canonical.e_corr_per_cell == pytest.approx(independent,abs=1e-12)
    assert result.pno_correction_per_cell == pytest.approx(0.,abs=2e-12)
    assert result.translation.solution.e_hf_per_cell == -4.
    storage = result.translation.pair_build.provenance['coefficient_storage']
    assert storage['selected_occupied_bytes'] == storage['dense_selected_occupied_bytes'] // (nk if compact_storage else 1)
    assert result.translation.solution.provenance['transport_preparation']['recipe_only'] == compact_storage
    if compact_storage:
        from vibeqc.dlpno.translation_storage import ProjectedPairCoefficients
        spaces = result.translation.pair_build.spaces
        assert type(spaces.pair_coefficients) is ProjectedPairCoefficients
        assert spaces._occupied is result.translation.occupied_selection.occupied
        assert spaces._frozen_occupied is result.translation.occupied_selection.frozen_occupied
        assert storage['pair_ao_bytes'] == 0
        assert storage['maximum_pair_reconstruction_error'] < 1e-10
    occupied_fock = result.translation.pair_build.spaces.occupied_fock
    if np.prod(mesh) > 1:
        assert np.max(abs(occupied_fock-np.diag(np.diag(occupied_fock)))) > 1e-3
        assert result.translation.solution.n_iter > 2
    if any(n % 2 == 0 for n in mesh):
        spaces = result.translation.pair_build.spaces
        assert spaces.stabilizers
        for row in spaces.stabilizers:
            assert spaces.plan.multiplicities[row] == nk // 2
            assert spaces.energy_weights[row] == 1.
    assert sum(result.translation.pair_build.spaces.energy_weights) == pytest.approx((nocc-core*nk)**2/nk)
    for name,value in dict(raw=result.raw_e_corr_per_cell,canonical=independent,
        error=abs(result.raw_e_corr_per_cell-independent),residual=result.translation.solution.max_residual,
        iterations=result.translation.solution.n_iter).items(): record_property(name,value)


def test_no_dense_matrix_reconstruction_or_scf(monkeypatch):
    from vibeqc.periodic.chi import posthf, scf
    ref,_,_ = fixture(monkeypatch)
    def forbidden(*args,**kwargs): pytest.fail('unexpected dense reconstruction or reference rebuild')
    for name in ('_build_b_real_reference','_periodic_matrix_to_real_supercell','_selected_lpq_cache'):
        monkeypatch.setattr(posthf,name,forbidden)
    monkeypatch.setattr(scf,'run_aiccm2026dev_b_rhf',forbidden)
    original=np.linalg.eigh
    def bounded(matrix,*args,**kwargs):
        # Adapter can diagonalize small occupied Gram matrices, never AO S/F.
        assert matrix.shape != (16,16)
        return original(matrix,*args,**kwargs)
    monkeypatch.setattr(np.linalg,'eigh',bounded)
    original_einsum = np.einsum
    def bounded_einsum(*args, **kwargs):
        answer = original_einsum(*args, **kwargs)
        assert answer.shape != (16,16), 'dense AO matrix materialized before slicing'
        return answer
    monkeypatch.setattr(np, 'einsum', bounded_einsum)
    monkeypatch.setattr(local, '_home_factors', forbidden)
    from vibeqc import _vibeqc_core
    monkeypatch.setattr(_vibeqc_core, 'aiccm2026dev_b_lpq_to_real_home_auxiliary', forbidden)
    source=local.build_chi_translation_source(ref,frozen_core=1)
    from vibeqc.dlpno.translation_operators import CyclicAOOperator
    assert type(source.overlap) is type(source.reference.fock) is CyclicAOOperator
    assert source.overlap.blocks.nbytes == 4*4*4*8


def test_adapted_operator_actions_and_factors_are_translation_covariant(monkeypatch):
    from vibeqc.dlpno.translation_driver import run_translation_dlpno_mp2
    ref, _, _ = fixture(monkeypatch)
    source = local.build_chi_translation_source(ref, frozen_core=1)
    columns = np.arange(16*3, dtype=float).reshape(16,3)/10.
    for operator in (source.overlap, source.reference.fock):
        for shift in range(4):
            translated = np.roll(columns.reshape(4,4,3), shift, axis=0).reshape(16,3)
            action = np.roll(operator.apply(columns).reshape(4,4,3), shift, axis=0).reshape(16,3)
            np.testing.assert_allclose(operator.apply(translated), action, atol=1e-13)
    # Audit actual pair integrals at every origin, independently of the
    # compact provider's constructive covariance certificate used by default.
    result = run_translation_dlpno_mp2(source, (4,1,1), n_frozen=4,
        factor_audit='all-origins', conv_tol_residual=1e-12, conv_tol_energy=1e-13)
    assert result.pair_build.max_translation_error < 1e-12
    assert result.solution.converged


@pytest.mark.parametrize('fault', ['mutated','source','sr','fractional','symmetry','mesh','complex_fock','complex_factors','basis'])
def test_refused_inputs(monkeypatch,fault):
    ref,_,_=fixture(monkeypatch,(3,1,1))
    hf=ref.hf_result
    if fault=='mutated': hf.fock[0,0,0] += .1
    elif fault=='source': hf.executed_fit_source=replace(hf.executed_fit_source,executed_domains_json=json.dumps(dict(fitted_operator_identity='different')))
    elif fault=='sr':
        hf.fitted_operator=ChiFittedOperator(dict(hf.fitted_operator),replace(hf.fitted_operator.descriptor,source='sr'),hf.kpoint_weights)
    elif fault=='fractional': hf.occupations[:,:2]=1.9
    elif fault=='symmetry': hf._chi_restart_physical['options']={'symmetry_reduce_k':True}
    elif fault=='mesh': hf._chi_restart_physical['mesh']=[2,1,1]
    elif fault=='complex_fock':
        hf.fock[0,0,1] += .1j; hf.fock[0,1,0] -= .1j
        e,c=eigh(hf.fock[0],hf.overlap[0]);hf.mo_coeffs[0]=c;hf.mo_energies[0]=e
        hf.density[0]=2*c[:,:2]@c[:,:2].conj().T
    elif fault=='complex_factors':
        changed={p:v*1j for p,v in hf.fitted_operator.items()}
        d=replace(hf.fitted_operator.descriptor,factor_sha256=factor_digest(changed))
        hf.fitted_operator=ChiFittedOperator(changed,d,hf.kpoint_weights)
    elif fault=='basis': hf._chi_restart_physical['basis']['sha256']='different'
    if fault != 'mutated':
        if fault in ('sr','complex_factors'):
            hf.executed_fit_source=replace(hf.fitted_operator.descriptor,executed_domains_json=json.dumps(
                dict(fitted_operator_identity=hf.fitted_operator.identity)))
        ref=refs._seal_chi_hf_result(hf,'RHF')
    with pytest.raises((ValueError,NotImplementedError)):
        local.build_chi_translation_source(ref)


@pytest.mark.parametrize('method',['UHF','ROHF'])
def test_unsupported_spin_before_transform(monkeypatch,method):
    ref,_,_=fixture(monkeypatch)
    with pytest.raises(NotImplementedError,match='restricted RHF'):
        local.build_chi_translation_source(replace(ref,method=method))


def test_output_nonconvergence_and_transport_budget(monkeypatch):
    from vibeqc.output.chi_correlation import chi_correlation_record, write_chi_correlation
    ref,_,_=fixture(monkeypatch)
    result=local.run_chi_local_mp2(ref,frozen_core=1,max_iter=1)
    assert not result.converged
    record=chi_correlation_record(result)
    assert record['source_identity']==ref.hf_result.fitted_operator.identity
    assert record['reference_identity']==ref.state_identity
    assert not record['converged']
    assert record['energy_raw_correlation_per_cell']==result.raw_e_corr_per_cell
    assert record['settings']['pno_norm']=='mp2'
    assert record['electron_counts']['frozen_explicit_electrons']==2
    assert record['domain_correction']=='not computed'
    output=[];write_chi_correlation(output.append,result)
    assert 'E(raw local MP2)' in ''.join(output) and 'False' in ''.join(output)
    json.dumps(record,allow_nan=False)
    with pytest.raises((ValueError,MemoryError),match='budget|max_bytes|memory|bytes'):
        local.run_chi_local_mp2(ref,frozen_core=1,max_transport_bytes=1)


def test_realification_checks_projector_not_imaginary_columns():
    phase=np.ones((1,1),complex)
    overlap=np.eye(3)[None]
    good=np.array([[[1j],[0.],[0.]]])
    frame,audit=local._real_occupied(good,overlap,phase,'active')
    np.testing.assert_allclose(frame@frame.T,np.diag([1.,0.,0.]),atol=1e-14)
    bad=np.array([[[1.],[1j],[0.]]])/np.sqrt(2)
    with pytest.raises(ValueError,match='real frame'):
        local._real_occupied(bad,overlap,phase,'frozen')


def test_saved_output_and_citation_route(monkeypatch,tmp_path):
    import tomllib
    ref,_,_=fixture(monkeypatch,(1,1,1),False)
    result=local.run_chi_local_mp2(ref,output=tmp_path/'local')
    record=tomllib.loads((tmp_path/'local.system').read_text())
    assert record['run']['chi_correlation']['energy_raw_correlation_per_cell']==result.raw_e_corr_per_cell
    assert record['run']['chi_correlation']['source_identity']==ref.hf_result.fitted_operator.identity
    assert '10.1063/5.0290816' in (tmp_path/'local.bibtex').read_text()
    assert '10.1063/1.4773581' in (tmp_path/'local.bibtex').read_text()
    assert 'E(raw local MP2)' in (tmp_path/'local.out').read_text()


def test_domain_and_pno_cutoffs_are_separate_experiments(monkeypatch,record_property):
    ref,_,_=fixture(monkeypatch,(3,1,1),False)
    exact=evaluate_same_source_mp2(ref).e_corr_per_cell
    rows=[]
    for domain,pno in ((0.,0.),(.05,0.),(.01,0.),(0.,1e-4),(0.,1e-6)):
        result=local.run_chi_local_mp2(ref,tcut_mkn=domain,tcut_pno=pno,
            max_iter=250,conv_tol_energy=1e-13,conv_tol_residual=1e-12)
        assert result.converged
        built=result.translation.pair_build
        if pno==0:
            assert tuple(built.pno_ranks)==tuple(built.pao_ranks)
            assert abs(result.pno_correction_per_cell)<2e-12
        if domain==0:
            assert all(len(d)==12 for d in result.translation.domain_selection.pair_ao_domains)
        rows.append(dict(domain=domain,pno=pno,raw_error=result.raw_e_corr_per_cell-exact,
            pno_estimate=result.pno_correction_per_cell,pao_ranks=list(built.pao_ranks),
            pno_ranks=list(built.pno_ranks),residual=result.translation.solution.max_residual))
    # This is a measured threshold ladder, not a monotonicity assertion.
    assert any(min(r['pao_ranks'])<min(rows[0]['pao_ranks']) for r in rows[1:3])
    assert any(min(r['pno_ranks'])<min(rows[0]['pno_ranks']) for r in rows[3:])
    record_property('independent_threshold_ladder',json.dumps(rows))


def test_ecp_metadata_and_explicit_electron_count_are_preserved(monkeypatch):
    from tests.test_chi_ecp_checkpoint import ecp_fixture
    ref,_,_=ecp_fixture(monkeypatch,nk=1)
    hf=ref.hf_result
    desc=replace(hf.fitted_operator.descriptor,engine='native-chi-classic-finite-contract')
    hf.fitted_operator=ChiFittedOperator(dict(hf.fitted_operator),desc,hf.kpoint_weights)
    domains=json.loads(hf.executed_fit_source.executed_domains_json)
    domains['fitted_operator_identity']=hf.fitted_operator.identity
    hf.executed_fit_source=replace(desc,executed_domains_json=json.dumps(domains))
    hf.finite_torus_convention=cp._AttributeRecord(dict(exchange_q0='bvk-ewald',
        coulomb_kernel='3d-periodic-g0',character_mesh_shape=(1,1,1)))
    ref=refs._seal_chi_hf_result(hf,'RHF')
    result=local.run_chi_local_mp2(ref)
    from vibeqc.output.chi_correlation import chi_correlation_record
    record=chi_correlation_record(result)
    assert record['ecp']==hf.chi_ecp
    counts=record['electron_counts']
    assert counts['physical_electrons']==4 and counts['ecp_core_electrons']==2
    assert counts['explicit_electrons']==2 and counts['frozen_explicit_electrons']==0
    assert record['quantitative_status']=='not-qualified'


def test_character_order_and_orbital_phases_do_not_define_the_real_frame(monkeypatch):
    ref,_,home=fixture(monkeypatch)
    hf=ref.hf_result
    permutation=[2,0,3,1]
    old=hf.fitted_operator
    factors={(i,j):old[permutation[i],permutation[j]] for i in range(4) for j in range(4)}
    old_to_new={p:i for i,p in enumerate(permutation)}
    groups=tuple((q,tuple((old_to_new[i],old_to_new[j]) for i,j in pairs))
                 for q,pairs in old.descriptor.q_grouping)
    d=replace(old.descriptor,kpoints_cart=tuple(old.descriptor.kpoints_cart[p] for p in permutation),
              q_grouping=groups,factor_sha256=factor_digest(factors))
    hf.fitted_operator=ChiFittedOperator(factors,d,hf.kpoint_weights)
    hf.executed_fit_source=replace(d,executed_domains_json=json.dumps(
        dict(fitted_operator_identity=hf.fitted_operator.identity)))
    for field in ('fock','overlap','mo_coeffs','mo_energies','occupations','density','kpoints_cart'):
        setattr(hf,field,np.asarray(getattr(hf,field))[permutation])
    ref=refs._seal_chi_hf_result(hf,'RHF')
    source=local.build_chi_translation_source(ref,frozen_core=1,factor_provider='dense')
    np.testing.assert_allclose(source.factors.home_factors,home,atol=1e-13)
    result=local.run_chi_local_mp2(ref,frozen_core=1,conv_tol_energy=1e-13,conv_tol_residual=1e-12)
    assert result.raw_e_corr_per_cell==pytest.approx(evaluate_same_source_mp2(ref,frozen_core=1).e_corr_per_cell,abs=2e-11)


@pytest.mark.parametrize('mesh', [(1,1,1), (3,1,1), (4,1,1), (2,2,1)])
@pytest.mark.parametrize('tile,cache', [(1,0), (3,1024**2)])
def test_streamed_contractions_against_dense_and_retained_operator(monkeypatch, mesh, tile, cache):
    from vibeqc.periodic.chi.canonical_mp2 import CanonicalFactorAdapter
    from vibeqc.dlpno.translation_bloch_factors import TranslationBlochFactors
    from vibeqc.dlpno.translation_factors import TranslationHomeFactors
    ref, _, home = fixture(monkeypatch, mesh)
    adapter = CanonicalFactorAdapter(ref.hf_result.fitted_operator)
    provider = TranslationBlochFactors(adapter, mesh, auxiliary_tile=tile, cache_bytes=cache)
    dense = TranslationHomeFactors(home, mesh)
    rng = np.random.default_rng(123)
    n = len(adapter.k_fractional)
    # Strides, dtype conversion and arbitrary complex orbital gauges.
    left = rng.normal(size=(n*4,4))[:,::2]
    right = rng.normal(size=(n*4,6))[:,::2]
    real = provider.mo_transform(left, right)
    np.testing.assert_allclose(real, dense.mo_transform(left,right), atol=2e-12, rtol=2e-13)
    for cl, cr in ((left.astype(np.float32),right),
                   (left*np.exp(.37j),right*np.exp(-.23j))):
        np.testing.assert_allclose(provider.mo_transform(cl,cr), dense.mo_transform(cl,cr),
                                   atol=2e-12, rtol=2e-13)
    exchange = provider.pair_exchange(left,right)
    np.testing.assert_allclose(exchange, dense.pair_exchange(left,right), atol=2e-11, rtol=2e-13)
    # Independent four-momentum operator contraction uses the canonical
    # (i,a)/(b,j) source convention, not provider q grouping or Parseval.
    cells = np.array(list(product(*(range(m) for m in mesh))))
    phase = np.exp(-2j*np.pi*(adapter.k_fractional @ cells.T))/np.sqrt(n)
    occupied = np.einsum('kr,rmi->kmi',phase,left.reshape(n,4,2))
    virtual = np.einsum('kr,rma->kma',phase,right.reshape(n,4,3))
    independent = np.zeros((3,3),complex)
    for i,j,a in product(range(n),repeat=3):
        b = adapter.conserved[i,j,a]
        independent += adapter.integral((i,a),(b,j),occupied[i,:,0],virtual[a],
                                         virtual[b],occupied[j,:,1],1)/n
    np.testing.assert_allclose(exchange, independent, atol=2e-11, rtol=2e-13)
    for shift in product(*(range(m) for m in mesh)):
        moved = [np.roll(c.reshape(*mesh,4,-1),shift,axis=(0,1,2)).reshape(c.shape)
                 for c in (left,right)]
        np.testing.assert_allclose(provider.pair_exchange(*moved),exchange,atol=2e-11,rtol=2e-13)
    assert provider.source_identity == adapter.source_identity
    assert provider._operator is adapter.operator
    with pytest.raises(AttributeError): provider._mesh = (1,1,1)
    with pytest.raises(TypeError):
        class Invalid(TranslationBlochFactors): pass
    record = provider.provenance
    assert record['peak_accounted_bytes'] <= record['max_factor_bytes']
    assert max(record['audits'].values()) < 1e-10


@pytest.mark.parametrize('domain,pno', [(0.,0.), (.05,0.), (0.,1e-4), (0.,1e-6)])
def test_both_providers_use_the_same_iterative_solver(monkeypatch,domain,pno):
    ref,_,_ = fixture(monkeypatch,(3,1,1),False)
    results = [local.run_chi_local_mp2(ref,factor_provider=provider,
        tcut_mkn=domain,tcut_pno=pno,max_iter=250,conv_tol_energy=1e-13,
        conv_tol_residual=1e-12) for provider in ('dense','streamed')]
    dense, streamed = results
    assert dense.converged and streamed.converged
    assert streamed.raw_e_corr_per_cell == pytest.approx(dense.raw_e_corr_per_cell,abs=2e-12)
    assert streamed.pno_correction_per_cell == pytest.approx(dense.pno_correction_per_cell,abs=2e-12)
    assert streamed.translation.solution.max_residual == pytest.approx(
        dense.translation.solution.max_residual,abs=2e-12)
    assert streamed.translation.pair_build.pno_ranks == dense.translation.pair_build.pno_ranks
    record = json.loads(streamed.source_record_json)['factor_provider_execution']
    assert record['calls'] > 0 and record['retained_home_factor_bytes'] == 0
    assert record['max_contraction_imaginary'] < 1e-10


def test_factor_budget_cache_and_allocation_boundaries(monkeypatch):
    from vibeqc.periodic.chi.canonical_mp2 import CanonicalFactorAdapter
    from vibeqc.dlpno.translation_bloch_factors import TranslationBlochFactors as Provider
    ref,_,_ = fixture(monkeypatch,(3,1,1))
    adapter = CanonicalFactorAdapter(ref.hf_result.fitted_operator)
    n = 3
    base = Provider._iterator_bytes()+64*n*n
    minimum = base+96*n*n
    original_empty = np.empty
    def forbidden(*args,**kwargs): pytest.fail('allocation before budget rejection')
    with monkeypatch.context() as patch:
        patch.setattr(np,'empty',forbidden)
        with pytest.raises(MemoryError,match='required minimum'):
            Provider(adapter,(3,1,1),max_bytes=minimum-1)
    for cache in (0,32*n*n-1,32*n*n):
        provider = Provider(adapter,(3,1,1),max_bytes=minimum,cache_bytes=cache)
        assert provider.provenance['max_auxiliary_tile'] == 1
        assert provider.provenance['cache_bytes'] == (32*n*n if cache==32*n*n else 0)
        if provider._phase is not None:
            with pytest.raises(ValueError): provider._phase.setflags(write=True)
        assert provider.provenance['peak_accounted_bytes'] == minimum
        # The larger output and Fourier inputs must be refused before any
        # coefficient conversions, phase, output or transform allocations.
        occupied, virtual = original_empty((12,2)), original_empty((12,12))
        with monkeypatch.context() as patch:
            patch.setattr(np,'empty',forbidden)
            patch.setattr(np,'array',forbidden)
            patch.setattr(np,'zeros',forbidden)
            with pytest.raises(MemoryError,match='factor workspace budget'):
                provider.pair_exchange(occupied,virtual)
            with pytest.raises(MemoryError,match='factor workspace budget'):
                provider.mo_transform(virtual,virtual)
    # Exact contraction boundary; the requested auxiliary tile shrinks.
    left,right = np.ones((12,2)),np.ones((12,3))
    fixed = 64*12*5+48*3*3
    per_aux = 64*2*3+32*2*4
    provider = Provider(adapter,(3,1,1),max_bytes=base+fixed+per_aux)
    provider.pair_exchange(left,right)
    assert provider.provenance['peak_accounted_bytes'] == base+fixed+per_aux
    assert provider.provenance['cache_hits'] == 1
    assert provider.provenance['max_output_bytes'] == 8*3*3


@pytest.mark.parametrize('option,value', [('max_bytes',0),('max_bytes',True),
    ('auxiliary_tile',0),('auxiliary_tile',1.5),('cache_bytes',-1)])
def test_invalid_factor_settings(monkeypatch,option,value):
    from vibeqc.periodic.chi.canonical_mp2 import CanonicalFactorAdapter
    from vibeqc.dlpno.translation_bloch_factors import TranslationBlochFactors
    ref,_,_ = fixture(monkeypatch,(1,1,1))
    with pytest.raises(ValueError):
        TranslationBlochFactors(CanonicalFactorAdapter(ref.hf_result.fitted_operator),
                                (1,1,1),**{option:value})


def test_full_factor_audit_rejects_single_entry_errors(monkeypatch):
    from vibeqc.periodic.chi.canonical_mp2 import CanonicalFactorAdapter
    from vibeqc.dlpno.translation_bloch_factors import TranslationBlochFactors
    ref,_,_ = fixture(monkeypatch,(3,1,1))
    original = ref.hf_result.fitted_operator
    for error in (1e-5j,1e-5):
        factors = {key:value.copy() for key,value in original.items()}
        factors[2,1][-1,-1,0] += error
        descriptor = replace(original.descriptor,factor_sha256=factor_digest(factors))
        operator = ChiFittedOperator(factors,descriptor,np.ones(3)/3)
        with pytest.raises(ValueError,match='source audit'):
            TranslationBlochFactors(CanonicalFactorAdapter(operator),(3,1,1))


def test_null_modes_source_identity_and_digest_compatibility():
    from hashlib import sha256
    from benchmarks.chi_streamed_factors import model
    from vibeqc.dlpno.translation_bloch_factors import TranslationBlochFactors
    adapter,_,_,mesh = model(3,b=2,a=3)
    provider = TranslationBlochFactors(adapter,mesh,auxiliary_tile=1,cache_bytes=0)
    assert provider.n_aux == 9  # Canonical frame includes all three null modes.
    assert adapter.retained_ranks == (2,2,2)
    assert provider.source_identity == adapter.operator.identity
    assert provider._operator.descriptor == adapter.operator.descriptor
    adapter.source_identity = 'mismatched-source'
    with pytest.raises(ValueError,match='source mutated'):
        TranslationBlochFactors(adapter,mesh)
    for factors in ({(0,0):np.ones((3,2,2),complex)},
                    {(0,0):np.arange(36).reshape(3,3,4)[:,:,::2]},
                    {(0,0):np.empty((0,2,2),complex)}):
        expected = sha256()
        for pair in sorted(factors):
            value = np.asarray(factors[pair],dtype='<c16')
            expected.update(np.asarray(pair,dtype='<i8').tobytes())
            expected.update(np.asarray(value.shape,dtype='<i8').tobytes())
            expected.update(value.tobytes())
        assert factor_digest(factors) == expected.hexdigest()
