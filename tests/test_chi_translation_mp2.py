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


@pytest.mark.parametrize("transport_backend", ["prepared", "bounded"])
def test_ecp_metadata_and_explicit_electron_count_are_preserved(monkeypatch,transport_backend):
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
    result=local.run_chi_local_mp2(ref,transport_backend=transport_backend)
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
    base = Provider._iterator_bytes()+128*n
    minimum = base+192*n
    original_empty = np.empty
    def forbidden(*args,**kwargs): pytest.fail('allocation before budget rejection')
    with monkeypatch.context() as patch:
        patch.setattr(np,'empty',forbidden)
        with pytest.raises(MemoryError,match='required minimum'):
            Provider(adapter,(3,1,1),max_bytes=minimum-1)
    for cache in (0,32*n-1,32*n):
        provider = Provider(adapter,(3,1,1),max_bytes=minimum,cache_bytes=cache)
        assert provider.provenance['max_auxiliary_tile'] == 1
        assert provider.provenance['cache_bytes'] == (32*n if cache==32*n else 0)
        if provider._phase is not None:
            with pytest.raises(ValueError): provider._phase[1].setflags(write=True)
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
    assert provider.provenance['cache_misses'] > 0
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


@pytest.mark.parametrize('mesh,core', [((1,1,1),0),((3,1,1),0),((4,1,1),1),((2,2,1),1)])
@pytest.mark.parametrize('seed_basis', ['ao', 'cosine'])
def test_cyclic_selection_matches_dense_seed_gauge_and_complete_audits(monkeypatch, mesh, core, seed_basis):
    from vibeqc.dlpno.translation_selection import select_translation_occupied
    from vibeqc.dlpno.translation_source import admit_translation_source
    from vibeqc.dlpno.translation_storage import occupied_projector_rows, occupied_orthogonality_error
    ref, _, _ = fixture(monkeypatch, mesh, bool(core))
    dense_source = local.build_chi_translation_source(ref, frozen_core=core)
    source = local.build_chi_translation_source(ref, frozen_core=core, compact_occupied=True)
    nk = int(np.prod(mesh))
    dense = select_translation_occupied(dense_source, mesh, n_frozen=core*nk, seed_basis=seed_basis)
    selected = select_translation_occupied(source, mesh, n_frozen=core*nk, compact_storage=True, seed_basis=seed_basis)
    assert selected.active_seed_aos == dense.active_seed_aos
    assert selected.frozen_seed_aos == dense.frozen_seed_aos
    for key in ('active_seed_minima','frozen_seed_minima'):
        np.testing.assert_allclose(selected.provenance[key], dense.provenance[key], atol=3e-14)
    for actual, expected in ((selected.occupied,dense.occupied), (selected.frozen_occupied,dense.frozen_occupied)):
        np.testing.assert_allclose(np.asarray(actual), expected, atol=3e-13)
        assert occupied_orthogonality_error(actual, source.overlap) < 1e-13
        for start in range(0,nk*4,4):
            rows = np.arange(start,start+4)
            np.testing.assert_allclose(occupied_projector_rows(actual,source.overlap,rows),
                occupied_projector_rows(expected,source.overlap,rows), atol=3e-13)
    admitted = admit_translation_source(source,mesh,selected.occupied,frozen_occupied=selected.frozen_occupied)
    assert admitted.provenance['active_projector_error'] < 1e-13
    assert admitted.provenance['frozen_projector_error'] < 1e-13
    if core:
        from vibeqc.dlpno.translation_storage import TranslationOccupiedColumns
        mixed = TranslationOccupiedColumns.from_home((selected.occupied.home+selected.frozen_occupied.home)/np.sqrt(2),mesh)
        other = TranslationOccupiedColumns.from_home((selected.occupied.home-selected.frozen_occupied.home)/np.sqrt(2),mesh)
        with pytest.raises(ValueError, match='projector'):
            admit_translation_source(source,mesh,mixed,frozen_occupied=other)


@pytest.mark.parametrize('seed_basis', ['ao', 'cosine'])
def test_compact_preparation_forbids_full_frames_and_translated_gram(monkeypatch, seed_basis):
    from vibeqc.dlpno import translation_selection as selection
    from vibeqc.dlpno.translation_storage import TranslationOccupiedColumns
    from vibeqc.dlpno.translation_preparation import selection_workspace
    ref, _, _ = fixture(monkeypatch, (4,1,1), True)
    def forbidden(*args, **kwargs):
        raise AssertionError('compact preparation reconstructed a complete occupied frame/seed Gram')
    monkeypatch.setattr(TranslationOccupiedColumns, '__array__', forbidden)
    monkeypatch.setattr(selection, '_projected_frame', forbidden)
    monkeypatch.setattr(local, '_real_occupied', forbidden)
    original_eigh, original_eigvalsh = np.linalg.eigh, np.linalg.eigvalsh
    def small_eigh(value, *args, **kwargs):
        assert value.shape[-1] <= 2*4
        return original_eigh(value,*args,**kwargs)
    def small_eigvalsh(value, *args, **kwargs):
        assert value.shape[-1] <= 4
        return original_eigvalsh(value,*args,**kwargs)
    with monkeypatch.context() as guard:
        guard.setattr(np.linalg,'eigh',small_eigh)
        guard.setattr(np.linalg,'eigvalsh',small_eigvalsh)
        source = local.build_chi_translation_source(ref,frozen_core=1,compact_occupied=True)
        bound = selection_workspace((4,1,1),4,2,10**7,seed_basis=seed_basis)['accounted_peak_bytes']
        selected = selection.select_translation_occupied(source,(4,1,1),n_frozen=4,
            compact_storage=True,max_selection_bytes=bound,seed_basis=seed_basis)
        assert selected.provenance['preparation_workspace']['accounted_peak_bytes'] == bound
        with pytest.raises(MemoryError,match='max_selection_bytes'):
            selection.select_translation_occupied(source,(4,1,1),n_frozen=4,
                compact_storage=True,max_selection_bytes=bound-1,seed_basis=seed_basis)
    result = local.run_chi_local_mp2(ref,frozen_core=1,conv_tol_energy=1e-13,conv_tol_residual=1e-12,seed_basis=seed_basis)
    assert result.converged
    execution=result.translation.provenance['occupied_audit_execution']
    assert execution['scalar_evaluations'] == 5 and execution['scalar_cache_hits'] >= 5
    assert execution['symbol_transforms'] == 3
    assert execution['cached_array_bytes'] <= execution['cache_limit_bytes']
    from vibeqc.output.chi_correlation import chi_correlation_record
    assert chi_correlation_record(result)['occupied_audit_execution'] == dict(execution)
    transport = result.translation.solution.provenance['transport_preparation']
    assert transport['occupied_home_input']
    assert transport['occupied_input_bytes'] == 8*16*2


@pytest.mark.parametrize('bad_budget', [0,-1,True,1.5])
def test_compact_selection_budget_refuses_before_transform(monkeypatch,bad_budget):
    ref, _, _ = fixture(monkeypatch,(3,1,1),False)
    with pytest.raises(ValueError,match='max_selection_bytes'):
        local.build_chi_translation_source(ref,compact_occupied=True,max_selection_bytes=bad_budget)


@pytest.mark.parametrize('defect', ['normalization','complex_trim','complex_pair'])
def test_character_real_span_preserves_dense_refusals(defect):
    from vibeqc.dlpno.translation_preparation import real_occupied_projector
    mesh=(3,1,1); n=3
    phase=np.exp(2j*np.pi*np.arange(n)[:,None]*np.arange(n)[None,:]/n)
    c=np.zeros((n,2,1),complex); c[:,0]=1
    if defect=='normalization': c[1]*=1+1e-6
    elif defect=='complex_trim': c[0,:,0]=[1/np.sqrt(2),1j/np.sqrt(2)]
    else: c[1,:,0]=[1/np.sqrt(2),1j/np.sqrt(2)]
    overlap=np.tile(np.eye(2),(n,1,1))
    with pytest.raises(ValueError): local._real_occupied(c,overlap,phase,'active')
    with pytest.raises(ValueError): real_occupied_projector(c,overlap,phase,mesh,'active',1e-10)


@pytest.mark.parametrize('mesh,core,domain,pno', [((3,1,1),0,.05,0.),
    ((3,1,1),0,0.,1e-4),((4,1,1),1,0.,1e-6),((2,2,1),0,0.,1e-6)])
def test_cyclic_preparation_preserves_domains_pnos_raw_residual_and_corrections(monkeypatch,mesh,core,domain,pno):
    ref,_,_=fixture(monkeypatch,mesh,bool(core))
    options=dict(frozen_core=core,tcut_mkn=domain,tcut_pno=pno,
                 conv_tol_energy=1e-13,conv_tol_residual=1e-12)
    dense=local.run_chi_local_mp2(ref,compact_storage=False,**options)
    compact=local.run_chi_local_mp2(ref,compact_storage=True,**options)
    assert dense.converged and compact.converged
    assert compact.raw_e_corr_per_cell == pytest.approx(dense.raw_e_corr_per_cell,abs=2e-12)
    assert compact.pno_correction_per_cell == pytest.approx(dense.pno_correction_per_cell,abs=2e-12)
    left,right=compact.translation,dense.translation
    np.testing.assert_allclose(left.domain_selection.home_populations,right.domain_selection.home_populations,atol=3e-13)
    for actual,expected in zip(left.domain_selection.pair_ao_domains,right.domain_selection.pair_ao_domains):
        np.testing.assert_array_equal(actual,expected)
    assert left.pair_build.pao_ranks == right.pair_build.pao_ranks
    assert left.pair_build.pno_ranks == right.pair_build.pno_ranks
    for row,(a,b) in enumerate(zip(left.pair_build.spaces.pair_coefficients,right.pair_build.spaces.pair_coefficients)):
        np.testing.assert_allclose(a@a.T,b@b.T,atol=3e-11)
        np.testing.assert_allclose(a@left.solution.residuals[row]@a.T,
                                   b@right.solution.residuals[row]@b.T,atol=3e-12)
        np.testing.assert_allclose(a@left.solution.amplitudes[row]@a.T,
                                   b@right.solution.amplitudes[row]@b.T,atol=3e-11)


@pytest.mark.parametrize('mesh', [(1,1,1),(3,1,1),(4,1,1),(2,3,1),(3,2,2)])
@pytest.mark.parametrize('perturb', [0., .8e-10, 1.2e-10])
def test_all_cyclic_defects_match_original_entry_norms(mesh, perturb):
    from vibeqc.dlpno.translation_operators import CyclicAOOperator
    from vibeqc.dlpno.translation_storage import (TranslationOccupiedColumns,
        occupied_orthogonality_error, occupied_gauge_error, occupied_projector_rows)
    from vibeqc.dlpno.translation_audit import projector_defect
    n, b = int(np.prod(mesh)), 4
    rng = np.random.default_rng(509)
    blocks = rng.normal(size=(n,b,b))*.015
    for d, cell in enumerate(np.ndindex(mesh)):
        neg = np.ravel_multi_index(tuple((-np.array(cell)) % mesh),mesh)
        if neg >= d:
            blocks[d] = (blocks[d]+blocks[neg].T)*.5
            blocks[neg] = blocks[d].T
    blocks[0] += np.eye(b)
    # Deliberate admitted nonsymmetry distinguishes C C.T S from C C.T S.T.
    blocks[-1,0,1] += perturb
    S = CyclicAOOperator(blocks,mesh)
    home = rng.normal(size=(n*b,2))*.2
    active = TranslationOccupiedColumns.from_home(home,mesh)
    core = TranslationOccupiedColumns.from_home(home[:,:1]*.3,mesh)
    dense, frozen = np.asarray(active), np.asarray(core)
    for right, dright in ((None,None),(core,frozen)):
        np.testing.assert_allclose(occupied_orthogonality_error(active,S,right),
            occupied_orthogonality_error(dense,S,dright),rtol=0,atol=5e-15)
    target_blocks = occupied_projector_rows(dense,S,np.arange(b)).reshape(b,n,b).transpose(1,0,2)
    target_blocks[-1,1,0] += max(perturb, 3e-11)
    target = CyclicAOOperator(target_blocks,mesh)
    expected = max(np.max(abs(occupied_projector_rows(dense,S,np.arange(r*b,(r+1)*b))
        -target.block(np.arange(r*b,(r+1)*b),np.arange(n*b)))) for r in range(n))
    assert projector_defect(active,S,target) == pytest.approx(expected,abs=3e-15)
    assert occupied_gauge_error(active,mesh) == occupied_gauge_error(dense,mesh) == 0.


@pytest.mark.parametrize('mesh', [(1,1,1),(3,1,1),(4,1,1),(2,3,1),(2,2,2)])
@pytest.mark.parametrize('reorder', [False,True])
def test_source_cyclic_audits_match_exhaustive_real_orbitals(mesh,reorder):
    from vibeqc.dlpno.translation_preparation import real_occupied_projector
    rng=np.random.default_rng(508)
    n,b,m=int(np.prod(mesh)),5,2
    cells=np.asarray(list(np.ndindex(mesh)))
    fractions=cells/np.asarray(mesh)
    raw=rng.normal(size=(b,b))
    s=np.eye(b)+.03*raw@raw.T
    w,v=np.linalg.eigh(s)
    c=(v/np.sqrt(w))@v.T@np.linalg.qr(rng.normal(size=(b,b)))[0][:,:m]
    coeff=np.tile(c,(n,1,1)).astype(complex)*np.exp(.37j*np.arange(n*m).reshape(n,1,m))
    overlap=np.tile(s,(n,1,1))
    if reorder:
        order=rng.permutation(n)
        coeff,overlap,fractions=coeff[order],overlap[order],fractions[order]
    phase=np.exp(2j*np.pi*(fractions@cells.T))
    results=[real_occupied_projector(coeff,overlap,phase,mesh,'active',1e-10,
                                    kfrac=fractions,audit=mode) for mode in ('auto','exhaustive')]
    np.testing.assert_allclose(results[0][0].blocks,results[1][0].blocks,atol=3e-15,rtol=0)
    for name in ('orthogonality','projector_error','translation_error','imaginary'):
        assert results[0][1][name] == pytest.approx(results[1][1][name],abs=4e-15)
    assert results[0][1]['complete_column_replays'] == 0
    assert results[1][1]['complete_column_replays'] > 0


@pytest.mark.parametrize('amplitude', [.8e-10, 1e-10, 1.2e-10])
def test_cyclic_admission_near_threshold_and_frozen_binding(amplitude):
    from benchmarks.chi_occupied_preparation import model
    from vibeqc.dlpno.translation_selection import select_translation_occupied
    from vibeqc.dlpno.translation_storage import TranslationOccupiedColumns, occupied_orthogonality_error
    from vibeqc.dlpno.translation_source import admit_translation_source
    from vibeqc.dlpno.translation_projectors import BlochOccupiedProjectors
    from vibeqc.dlpno.translation_operators import CyclicAOOperator
    from vibeqc.dlpno.translation_audit import occupied_audit_context, projector_defect
    source,mesh,nf,_,_=model(3)
    selection=select_translation_occupied(source,mesh,n_frozen=nf,compact_storage=True)
    active,core=selection.occupied,selection.frozen_occupied
    changed=TranslationOccupiedColumns.from_home(active.home*(1+amplitude/2),mesh)
    dense=np.asarray(changed)
    with occupied_audit_context() as state:
        fast=occupied_orthogonality_error(changed,source.overlap,tolerance=1e-10)
        from vibeqc.dlpno.translation_storage import _streamed_orthogonality_error
        slow=_streamed_orthogonality_error(changed,source.overlap)
        assert fast == pytest.approx(slow,abs=8e-16)
        if amplitude == 1e-10:
            assert fast == slow  # boundary replays the original compact arithmetic
            assert state['boundary_replays'] == 1
        assert (fast > 1e-10) == (slow > 1e-10)
    original=source.reference.occupied_projectors
    bad=original.frozen.blocks.copy()
    bad[-1,0,1]+=amplitude  # a non-home defect, active space remains correct
    target=CyclicAOOperator(bad,mesh)
    # Bind the deliberately defective fixture before probing its numerical
    # threshold; mutation of an already bound source must refuse earlier.
    source=replace(source,reference=replace(source.reference,
        occupied_projectors=BlochOccupiedProjectors(original.active,target,*original[2:])))
    expected=projector_defect(core,source.overlap,target,1e-10)
    for a,f in ((active,core),(np.asarray(active),np.asarray(core))):
        if expected > 1e-10:
            with pytest.raises(ValueError,match='frozen occupied projector'):
                admit_translation_source(source,mesh,a,frozen_occupied=f)
        else:
            result=admit_translation_source(source,mesh,a,frozen_occupied=f)
            assert result.provenance['frozen_projector_error'] <= 1e-10


def test_unverified_translation_and_mutable_inputs_retain_full_checks(monkeypatch):
    from vibeqc.dlpno.translation_preparation import real_occupied_projector
    from vibeqc.dlpno.translation_storage import occupied_orthogonality_error, occupied_gauge_error
    from vibeqc.dlpno import translation_audit as audit
    mesh=(3,1,1); cells=np.array(list(np.ndindex(mesh)))
    fractions=cells/np.array(mesh)
    phase=np.exp(2j*np.pi*(fractions@cells.T))
    c=np.zeros((3,3,1),complex); c[:,0]=1
    overlap=np.tile(np.eye(3),(3,1,1))
    phase[1,2] *= 1+1e-5
    with pytest.raises(ValueError):
        real_occupied_projector(c,overlap,phase,mesh,'active',1e-10)
    c=np.zeros((9,3)); c[::3]=np.eye(3)
    c[6,1]=2e-9
    assert occupied_gauge_error(c,mesh) > 1e-10
    monkeypatch.setattr(audit,'metric_defect',lambda *args: pytest.fail('unverified reduction'))
    assert occupied_orthogonality_error(c,np.eye(9)) > 1e-10


def test_cyclic_audits_bound_operations_cache_and_source_identities(monkeypatch):
    from benchmarks.chi_occupied_preparation import model
    from vibeqc.dlpno.translation_selection import select_translation_occupied
    from vibeqc.dlpno.translation_source import admit_translation_source
    from vibeqc.dlpno.translation_storage import TranslationOccupiedColumns
    from vibeqc.dlpno.translation_audit import occupied_audit_context
    from vibeqc.dlpno.translation_operators import CyclicAOOperator
    source,mesh,nf,_,_=model(4)
    selected=select_translation_occupied(source,mesh,n_frozen=nf,compact_storage=True)
    def forbidden(*args,**kwargs):
        pytest.fail('complete column replay or AO action in cyclic audit')
    monkeypatch.setattr(TranslationOccupiedColumns,'cell',forbidden)
    with occupied_audit_context() as state:
        for _ in range(2):
            result=admit_translation_source(source,mesh,selected.occupied,frozen_occupied=selected.frozen_occupied)
        assert state['evaluations'] == 5 and state['hits'] == 5
        record=result.provenance['occupied_audit']
        assert record['cached_array_bytes'] == record['cache_limit_bytes'] == 16*4*(16*16+16*6)
        assert record['symbol_cache_hits'] > 0
        assert result.provenance['occupied_audit']['displacement_blocks'] == 4
        source.identity += '-independent-binding'
        with pytest.raises(ValueError,match='changed after construction'):
            admit_translation_source(source,mesh,selected.occupied,frozen_occupied=selected.frozen_occupied)
        assert state['evaluations'] == 5  # refusal precedes cache lookup/reuse
        source=replace(source)
        admit_translation_source(source,mesh,selected.occupied,frozen_occupied=selected.frozen_occupied)
        assert state['evaluations'] == 10  # a different source cannot reuse the measurements
        source.overlap=CyclicAOOperator(source.overlap.blocks*1.001,mesh)
        with pytest.raises(ValueError,match='changed after construction'):
            admit_translation_source(source,mesh,selected.occupied,frozen_occupied=selected.frozen_occupied)
        assert state['evaluations'] == 10
        source=replace(source)
        with pytest.raises(ValueError,match='mesh/overlap/Fock'):
            admit_translation_source(source,mesh,selected.occupied,frozen_occupied=selected.frozen_occupied)
    assert all(isinstance(x,float) for x in state['values'].values())
    assert state['arrays'] == {} and state['array_bytes'] == 0
    with pytest.raises(ValueError):
        selected.occupied.home.flags.writeable=True
    with pytest.raises(AttributeError,match='reinitialized'):
        selected.occupied.__init__(selected.occupied.home,mesh)
    with occupied_audit_context() as fresh:
        assert fresh['values'] == {}


@pytest.mark.parametrize('amplitude', [.8e-10, 1e-10, 1.2e-10])
def test_source_real_character_metric_threshold_is_original_norm(amplitude):
    from vibeqc.dlpno.translation_preparation import real_occupied_projector
    mesh=(3,1,1); cells=np.array(list(np.ndindex(mesh)))
    fractions=cells/np.array(mesh)
    phase=np.exp(2j*np.pi*(fractions@cells.T))
    c=np.zeros((3,2,1),complex); c[:,0]=1
    overlap=np.tile(np.eye(2,dtype=complex),(3,1,1))
    overlap[1,0,0]+=1j*amplitude
    overlap[2,0,0]-=1j*amplitude
    outcomes=[]
    for mode in ('auto','exhaustive'):
        try:
            _,record=real_occupied_projector(c,overlap,phase,mesh,'active',1e-10,audit=mode)
            outcomes.append(record['orthogonality'])
        except ValueError:
            outcomes.append(None)
    if outcomes[1] is None:
        assert outcomes[0] is None
    else:
        assert outcomes[0] == pytest.approx(outcomes[1],abs=1e-15)


@pytest.mark.parametrize('amplitude', [.8e-10, 1e-10, 1.2e-10])
def test_source_projector_defect_away_from_home_at_threshold(amplitude):
    from vibeqc.dlpno.translation_preparation import real_occupied_projector
    mesh=(3,1,1); cells=np.array(list(np.ndindex(mesh)))
    phase=np.exp(2j*np.pi*((cells/np.array(mesh))@cells.T))
    c=np.zeros((3,2,1),complex); c[:,0]=1
    theta=amplitude*2*np.sqrt(3)
    c[1,:,0]=[np.cos(theta),np.sin(theta)]
    overlap=np.tile(np.eye(2),(3,1,1))
    measured=[]; admitted=[]
    for mode in ('auto','exhaustive'):
        _,record=real_occupied_projector(c,overlap,phase,mesh,'active',1e-8,audit=mode)
        measured.append(record['projector_error'])
        try:
            real_occupied_projector(c,overlap,phase,mesh,'active',1e-10,audit=mode)
            admitted.append(True)
        except ValueError:
            admitted.append(False)
    assert measured[0] == pytest.approx(measured[1],abs=1e-15)
    assert measured[0] == pytest.approx(amplitude,abs=1e-15)
    assert admitted[0] == admitted[1]
    # Even tiny noncharacter changes use replay, not a trusted declaration.
    phase[1,2]+=1e-14
    _,record=real_occupied_projector(c,overlap,phase,mesh,'active',1e-8)
    assert record['metric_audit'] == 'exhaustive-real-column-replay'
    assert record['cyclic_structure_verified'] is False


def test_source_real_pair_norm_is_not_complex_symbol_magnitude():
    from vibeqc.dlpno.translation_preparation import real_occupied_projector
    mesh=(3,1,1); cells=np.array(list(np.ndindex(mesh)))
    phase=np.exp(2j*np.pi*((cells/np.array(mesh))@cells.T))
    c=np.tile(np.eye(3,dtype=complex)[:,:2],(3,1,1))
    overlap=np.tile(np.eye(3,dtype=complex),(3,1,1))
    a=.8e-10
    overlap[1,0,1]=a+1j*a
    overlap[1,1,0]=-a+1j*a
    overlap[2]=overlap[1].conj()
    errors=[]
    for mode in ('auto','exhaustive'):
        _,record=real_occupied_projector(c,overlap,phase,mesh,'active',1e-10,audit=mode)
        errors.append(record['orthogonality'])
    assert errors[0] == pytest.approx(errors[1],abs=2e-15)
    assert errors[0] == pytest.approx(a,abs=2e-15)
    assert abs(a+1j*a) > 1e-10


@pytest.mark.parametrize('amplitude', [.8e-10,1e-10,1.2e-10])
def test_active_frozen_cross_norm_at_threshold(amplitude):
    from vibeqc.dlpno.translation_operators import CyclicAOOperator
    from vibeqc.dlpno.translation_storage import (TranslationOccupiedColumns,
        occupied_orthogonality_error, _streamed_orthogonality_error)
    mesh=(2,2,1);blocks=np.zeros((4,3,3));blocks[0]=np.eye(3)
    S=CyclicAOOperator(blocks,mesh)
    a=np.zeros((12,1));f=a.copy();a[0]=1.;f[10]=1.;f[9]=amplitude
    active=TranslationOccupiedColumns.from_home(a,mesh)
    frozen=TranslationOccupiedColumns.from_home(f,mesh)
    expected=_streamed_orthogonality_error(active,S,frozen)
    actual=occupied_orthogonality_error(active,S,frozen,tolerance=1e-10)
    assert actual == expected == amplitude
    assert (actual > 1e-10) == (amplitude > 1e-10)


def _complete_factor_audit(adapter, mesh):
    """D13 complete-table oracle, intentionally independent of production tiles."""
    import cmath
    n = adapter.nk
    cells = tuple(product(*(range(m) for m in mesh)))
    phase = np.array([[cmath.exp(2j*np.pi*sum(x*y for x,y in zip(k,cell)))
                       for cell in cells] for k in adapter.k_fractional])
    conjugate = phase.conj()
    a,b,_ = next(iter(adapter.operator.values())).shape
    errors = dict(imaginary=0.,symmetry=0.,round_trip=0.,translation_covariance=0.)
    for mu in range(b):
        for nu in range(b):
            def inverse(x,y):
                src = np.array([[adapter.operator[i,j][:,x,y] for j in range(n)] for i in range(n)])
                half = np.einsum('ir,ijp->rjp',phase,src,optimize=False)/n
                return np.einsum('js,rjp->rsp',conjugate,half,optimize=False)/n
            home, reverse = inverse(mu,nu), inverse(nu,mu)
            errors['imaginary'] = max(errors['imaginary'],float(np.max(abs(home.imag))))
            errors['symmetry'] = max(errors['symmetry'],float(np.max(abs(home.real-reverse.transpose(1,0,2).real))))
            average = ((home.real+reverse.transpose(1,0,2).real)*.5).astype(complex)
            half = np.einsum('ir,rsp->isp',conjugate,average,optimize=False)
            back = np.einsum('js,isp->ijp',phase,half,optimize=False)
            for (i,j),value in adapter.operator.items():
                errors['round_trip'] = max(errors['round_trip'],float(np.max(abs(back[i,j]-value[:,mu,nu]))))
    for i,t,r in product(range(n),repeat=3):
        shifted = cells.index(tuple((x-y)%m for x,y,m in zip(cells[r],cells[t],mesh)))
        errors['translation_covariance'] = max(errors['translation_covariance'],
            abs(phase[i,shifted]-phase[i,r]*conjugate[i,t]))
    return errors


def _factor_adapter_with_values(adapter, values, permutation=None):
    from vibeqc.periodic.chi.canonical_mp2 import CanonicalFactorAdapter
    d = replace(adapter.operator.descriptor, factor_sha256=factor_digest(values))
    if permutation is not None:
        inverse = {old:new for new,old in enumerate(permutation)}
        d = replace(d,kpoints_cart=tuple(d.kpoints_cart[i] for i in permutation),
                    q_grouping=tuple((q,tuple((inverse[i],inverse[j]) for i,j in pairs))
                                     for q,pairs in d.q_grouping))
    return CanonicalFactorAdapter(ChiFittedOperator(values,d,np.ones(adapter.nk)/adapter.nk))


@pytest.mark.parametrize('mesh', [(3,1,1),(4,1,1),(3,2,1),(2,2,2)])
def test_tiled_complete_factor_audit_and_order(monkeypatch, mesh):
    from vibeqc.periodic.chi.canonical_mp2 import CanonicalFactorAdapter
    from vibeqc.dlpno.translation_bloch_factors import TranslationBlochFactors as Provider
    ref,_,_ = fixture(monkeypatch,mesh)
    adapter = CanonicalFactorAdapter(ref.hf_result.fitted_operator)
    order = tuple(reversed(range(adapter.nk)))
    values = {(i,j):adapter.operator[order[i],order[j]]
              for i in range(adapter.nk) for j in range(adapter.nk)}
    adapter = _factor_adapter_with_values(adapter,values,order)
    expected = _complete_factor_audit(adapter,mesh)
    minimum = Provider._iterator_bytes()+320*adapter.nk
    for budget in (minimum,2**22):
        provider = Provider(adapter,mesh,max_bytes=budget,auxiliary_tile=1)
        for key,value in expected.items():
            assert provider.audit[key] == pytest.approx(value,abs=3e-15)
        record = provider.provenance
        assert record['recovered_entries'] == adapter.nk**2*2*4**2
        assert record['covariance_entries'] == adapter.nk**3
        assert record['audit_character_tile'] < adapter.nk
        assert record['peak_accounted_bytes'] <= budget


@pytest.mark.parametrize('n', [1,2,5,8,17])
def test_factor_audit_reuses_home_and_reverse_slabs_without_changing_recovery(n):
    from benchmarks.chi_streamed_factors import model
    from vibeqc.dlpno.translation_bloch_factors import TranslationBlochFactors as Provider
    adapter,_,_,mesh = model(n,b=2,a=3)
    replay = Provider(adapter,mesh,auxiliary_tile=2,cache_bytes=0)
    d = replay.provenance['audit_character_tile']
    first_only = Provider(adapter,mesh,auxiliary_tile=2,cache_bytes=32*n*d+16*n*d*2)
    reused = Provider(adapter,mesh,auxiliary_tile=2)
    assert reused.audit == first_only.audit == replay.audit
    expected = _complete_factor_audit(adapter,mesh)
    for key in expected:
        assert reused.audit[key] == pytest.approx(expected[key],abs=3e-15)
    r,s = replay.provenance,reused.provenance
    tiles = (n+s['audit_character_tile']-1)//s['audit_character_tile']
    assert first_only.provenance['audit_reverse_reuse_stores'] == 0
    assert first_only.provenance['inverse_blocks'] == r['inverse_blocks']-s['audit_reuse_hits']*(1+tiles)
    assert s['inverse_blocks'] == first_only.provenance['inverse_blocks']-s['audit_reverse_reuse_hits']
    assert s['inverse_entries'] <= first_only.provenance['inverse_entries']
    assert s['audit_reuse_hits'] == s['audit_reuse_stores']*(tiles-1)
    assert s['audit_reuse_stores'] == (0 if n==1 else 8)  # Four AO pairs, two aux tiles.
    assert s['audit_reuse_workspace_bytes'] == (0 if n==1 else 16*n*s['audit_character_tile']*2)
    assert s['audit_reverse_reuse_workspace_bytes'] == s['audit_reuse_workspace_bytes']
    assert s['audit_reverse_reuse_stores'] == s['audit_reuse_stores']
    assert s['audit_reverse_reuse_hits'] == s['audit_reuse_stores']*(tiles*tiles-tiles)
    assert s['audit_reuse_binding'] == reused.identity
    assert s['retained_home_factor_bytes'] == 0
    assert s['recovered_entries'] == r['recovered_entries'] == n*n*3*4
    assert s['covariance_entries'] == r['covariance_entries'] == n**3
    rng = np.random.default_rng(583)
    left,right = rng.normal(size=(n*2,2)),rng.normal(size=(n*2,3))
    np.testing.assert_array_equal(reused.pair_exchange(left,right),replay.pair_exchange(left,right))
    np.testing.assert_array_equal(reused.mo_transform(left*np.exp(.3j),right*np.exp(-.2j)),
                                  replay.mo_transform(left*np.exp(.3j),right*np.exp(-.2j)))


def test_factor_audit_reuse_exact_budget_and_cache_ceiling():
    from benchmarks.chi_streamed_factors import model
    from vibeqc.dlpno.translation_bloch_factors import TranslationBlochFactors as Provider
    adapter,_,_,mesh = model(5,b=2,a=3)
    # The original d=2,p=1 plan is unchanged; only spare space can hold reuse.
    base = Provider._iterator_bytes()+320*5*2
    extra = 16*5*2
    cache = 32*5*2+extra
    exact = Provider(adapter,mesh,auxiliary_tile=1,max_bytes=base+extra,cache_bytes=cache)
    record = exact.provenance
    assert record['peak_accounted_bytes'] == base+extra
    assert record['audit_reuse_workspace_bytes'] == extra
    assert record['audit_reverse_reuse_workspace_bytes'] == 0
    for budget,ceiling in ((base+extra-1,cache),(base+extra,cache-1),(base+extra,0)):
        replay = Provider(adapter,mesh,auxiliary_tile=1,max_bytes=budget,cache_bytes=ceiling)
        assert replay.provenance['audit_reuse_workspace_bytes'] == 0
        assert replay.provenance['audit_reuse_stores'] == 0
        assert replay.provenance['peak_accounted_bytes'] == base
        assert replay.audit == exact.audit
    both = Provider(adapter,mesh,auxiliary_tile=1,max_bytes=base+2*extra,cache_bytes=cache+extra)
    assert both.audit == exact.audit
    assert both.provenance['peak_accounted_bytes'] == base+2*extra
    assert both.provenance['audit_reverse_reuse_workspace_bytes'] == extra
    assert both.provenance['audit_reverse_reuse_stores'] > 0
    for budget,ceiling in ((base+2*extra-1,cache+extra),(base+2*extra,cache+extra-1)):
        first_only = Provider(adapter,mesh,auxiliary_tile=1,max_bytes=budget,cache_bytes=ceiling)
        assert first_only.audit == both.audit
        assert first_only.provenance['audit_reuse_workspace_bytes'] == extra
        assert first_only.provenance['audit_reverse_reuse_workspace_bytes'] == 0
        assert first_only.provenance['audit_reverse_reuse_stores'] == 0
        assert first_only.provenance['peak_accounted_bytes'] == base+extra
    minimum = Provider._iterator_bytes()+320*5
    tight = Provider(adapter,mesh,max_bytes=minimum)
    assert tight.provenance['peak_accounted_bytes'] == minimum
    assert tight.provenance['audit_reuse_workspace_bytes'] == 0
    assert tight.provenance['audit_reverse_reuse_workspace_bytes'] == 0


@pytest.mark.parametrize('admitted', [True,False])
def test_factor_audit_snapshots_are_immutable_and_released(monkeypatch,admitted):
    import gc
    import weakref
    from benchmarks.chi_streamed_factors import model
    from vibeqc.dlpno.translation_bloch_factors import TranslationBlochFactors as Provider
    adapter,_,_,mesh = model(5,b=2,a=3)
    if not admitted:
        values = {key:value.copy() for key,value in adapter.operator.items()}
        values[4,3][-1,-1,0] = 1e-5j
        adapter = _factor_adapter_with_values(adapter,values)
    before = factor_digest(adapter.operator)
    snapshots = []
    original = np.frombuffer
    def track(*args,**kwargs):
        value = original(*args,**kwargs)
        with pytest.raises(ValueError): value.setflags(write=True)
        snapshots.append(weakref.ref(value))
        return value
    with monkeypatch.context() as patch:
        patch.setattr(np,'frombuffer',track)
        if admitted:
            provider = Provider(adapter,mesh,auxiliary_tile=1)
            assert provider.provenance['audit_reuse_stores'] > 0
            assert provider.provenance['audit_reverse_reuse_stores'] > 0
        else:
            with pytest.raises(ValueError,match='source audit'):
                Provider(adapter,mesh,auxiliary_tile=1)
    # Only the final phase block can survive preparation. Both cached slabs
    # cannot leak into later contractions or the next AO/auxiliary tile.
    gc.collect()
    assert sum(reference() is not None for reference in snapshots) == int(admitted)
    assert factor_digest(adapter.operator) == before


@pytest.mark.parametrize('fraction', [.99999,np.nextafter(1.,0.),1.,np.nextafter(1.,2.),1.00001])
@pytest.mark.parametrize('error', [1.,1j])
def test_tiled_factor_threshold_and_late_null_component(fraction,error):
    from benchmarks.chi_streamed_factors import model
    from vibeqc.dlpno.translation_bloch_factors import TranslationBlochFactors as Provider
    adapter,_,_,mesh = model(5,b=2,a=3)
    values = {key:value.copy() for key,value in adapter.operator.items()}
    # Last character pair, last AO pair, last canonical auxiliary (a null mode).
    values[4,3][-1,-1,0] = error
    probe = _factor_adapter_with_values(adapter,values)
    scale = max(_complete_factor_audit(probe,mesh).values())
    values[4,3][-1,-1,0] *= fraction*1e-10/scale
    adapter = _factor_adapter_with_values(adapter,values)
    expected = _complete_factor_audit(adapter,mesh)
    for budget in (Provider._iterator_bytes()+320*5,2**22):
        if max(expected.values()) > 1e-10:
            import ast
            with pytest.raises(ValueError,match='source audit') as failure:
                Provider(adapter,mesh,max_bytes=budget,auxiliary_tile=1)
            actual = ast.literal_eval(str(failure.value).split('source audit: ')[1])
            for key,value in expected.items():
                assert actual[key] == pytest.approx(value,abs=3e-15)
        else:
            provider = Provider(adapter,mesh,max_bytes=budget,auxiliary_tile=1)
            for key,value in expected.items():
                assert provider.audit[key] == pytest.approx(value,abs=3e-15)


def test_tiled_provider_never_allocates_complete_tables(monkeypatch):
    from benchmarks.chi_streamed_factors import model
    from vibeqc.dlpno.translation_bloch_factors import TranslationBlochFactors as Provider
    adapter,_,_,mesh = model(9,b=2,a=3)
    n = adapter.nk
    empty,zeros,empty_like = np.empty,np.zeros,np.empty_like
    def checked(allocate):
        def call(shape,*args,**kwargs):
            dims = (shape,) if isinstance(shape,int) else tuple(shape)
            assert dims.count(n) < 2, 'complete quadratic character/displacement allocation'
            return allocate(shape,*args,**kwargs)
        return call
    left,right = np.ones((n*2,2)),np.ones((n*2,3))
    with monkeypatch.context() as patch:
        patch.setattr(np,'empty',checked(empty))
        patch.setattr(np,'zeros',checked(zeros))
        def checked_like(value,*args,**kwargs):
            assert value.shape.count(n) < 2
            return empty_like(value,*args,**kwargs)
        patch.setattr(np,'empty_like',checked_like)
        provider = Provider(adapter,mesh,cache_bytes=2**20)
        provider.pair_exchange(left,right)
        provider.mo_transform(left,right)
    assert provider.provenance['phase_table_bytes'] == 0
    assert provider.provenance['cache_bytes'] <= 32*n*provider.provenance['max_character_tile']
    # Mesh/source immutable cache, and no unbudgeted full-table private request.
    with pytest.raises(ValueError,match='tile'):
        provider._phases(0,n,0,n)
    with pytest.raises(ValueError): provider._phase[1].setflags(write=True)


def test_phase_covariance_audit_checks_late_displacement(monkeypatch):
    from benchmarks.chi_streamed_factors import model
    from vibeqc.dlpno.translation_bloch_factors import TranslationBlochFactors as Provider
    adapter,_,_,mesh = model(5,b=1,a=2)
    import cmath
    original = cmath.exp
    def broken(z):
        return original(z)*(1+2e-9 if abs(z-2j*np.pi*3.2)<1e-14 else 1)
    monkeypatch.setattr(cmath,'exp',broken)
    with pytest.raises(ValueError,match='source audit'):
        Provider(adapter,mesh)


@pytest.mark.parametrize('budget_extra', [12000,200000])
@pytest.mark.parametrize('cache_bytes', [0,2**20])
def test_factor_budget_preserves_downstream_science(monkeypatch,budget_extra,cache_bytes):
    from vibeqc.dlpno.translation_bloch_factors import TranslationBlochFactors as Provider
    ref,_,_ = fixture(monkeypatch,(3,1,1),True)
    options = dict(frozen_core=1,max_iter=250,conv_tol_energy=1e-13,conv_tol_residual=1e-12)
    dense = local.run_chi_local_mp2(ref,factor_provider='dense',**options)
    streamed = local.run_chi_local_mp2(ref,max_factor_bytes=Provider._iterator_bytes()+budget_extra,
                                     factor_auxiliary_tile=1,factor_cache_bytes=cache_bytes,**options)
    canonical = evaluate_same_source_mp2(ref,frozen_core=1)
    assert streamed.converged and dense.converged
    assert streamed.raw_e_corr_per_cell == pytest.approx(canonical.e_corr_per_cell,abs=2e-11)
    assert streamed.raw_e_corr_per_cell == pytest.approx(dense.raw_e_corr_per_cell,abs=2e-12)
    assert streamed.pno_correction_per_cell == pytest.approx(dense.pno_correction_per_cell,abs=2e-12)
    assert streamed.translation.solution.max_residual == pytest.approx(dense.translation.solution.max_residual,abs=2e-12)
    assert streamed.translation.pair_build.pno_ranks == dense.translation.pair_build.pno_ranks


@pytest.mark.parametrize('amplitude', [.99999e-10,1.00001e-10])
@pytest.mark.parametrize('component', ['imaginary','symmetry'])
@pytest.mark.parametrize('origin', [0,5])
def test_factor_defects_in_reused_and_late_real_space_tiles(amplitude,component,origin):
    from benchmarks.chi_streamed_factors import model
    from vibeqc.dlpno.translation_bloch_factors import TranslationBlochFactors as Provider
    adapter,phase,_,mesh = model(6,b=2,a=3)
    values = {key:value.copy() for key,value in adapter.operator.items()}
    # At origin 0, the reversed AO audit reuses this raw first-S slab at a
    # late R column. At origin 5, the defect lies outside both cached slabs.
    delta = amplitude*(1j if component=='imaginary' else 1)
    for (i,j),value in values.items():
        value[-1,1,0] += delta*phase[i,origin].conjugate()*phase[j,4]
    adapter = _factor_adapter_with_values(adapter,values)
    expected = _complete_factor_audit(adapter,mesh)
    assert expected[component] == pytest.approx(amplitude,rel=1e-13)
    first_only_cache = 32*6*3+16*6*3
    observed = []
    for budget,cache in ((Provider._iterator_bytes()+320*6,2**20),
                         (2**22,first_only_cache),(2**22,2**20)):
        if max(expected.values()) > 1e-10:
            import ast
            with pytest.raises(ValueError,match='source audit') as failure:
                Provider(adapter,mesh,max_bytes=budget,auxiliary_tile=1,cache_bytes=cache)
            actual = ast.literal_eval(str(failure.value).split('source audit: ')[1])
        else:
            actual = Provider(adapter,mesh,max_bytes=budget,auxiliary_tile=1,cache_bytes=cache).audit
        for key in expected:
            assert actual[key] == pytest.approx(expected[key],abs=3e-15)
        observed.append(actual)
    assert observed[1] == observed[2]  # Adding raw reverse reuse changes no defect.


def test_tiled_factor_exact_mo_budget_and_output(monkeypatch):
    from vibeqc.periodic.chi.canonical_mp2 import CanonicalFactorAdapter
    from vibeqc.dlpno.translation_bloch_factors import TranslationBlochFactors as Provider
    from vibeqc.dlpno.translation_factors import TranslationHomeFactors
    ref,_,home = fixture(monkeypatch,(3,1,1))
    adapter = CanonicalFactorAdapter(ref.hf_result.fitted_operator)
    rng = np.random.default_rng(611)
    left,right = rng.normal(size=(12,2))*np.exp(.23j),rng.normal(size=(12,3))*np.exp(-.37j)
    expected = TranslationHomeFactors(home,(3,1,1)).mo_transform(left,right)
    cap = Provider._iterator_bytes()+128*3+64*12*5+32*6*2*3+64*2*3+32*2*4
    provider = Provider(adapter,(3,1,1),max_bytes=cap)
    np.testing.assert_allclose(provider.mo_transform(left,right),expected,atol=2e-12,rtol=2e-13)
    assert provider.provenance['peak_accounted_bytes'] == cap
    assert provider.provenance['max_output_bytes'] == expected.nbytes
    smaller = Provider(adapter,(3,1,1),max_bytes=cap-1)
    def forbidden(*args,**kwargs): pytest.fail('allocation before MO budget refusal')
    with monkeypatch.context() as patch:
        patch.setattr(np,'empty',forbidden)
        patch.setattr(np,'array',forbidden)
        patch.setattr(np,'zeros',forbidden)
        with pytest.raises(MemoryError,match='required minimum'):
            smaller.mo_transform(left,right)


@pytest.mark.parametrize('mesh,core',[((1,1,1),0),((3,1,1),0),((4,1,1),1),((2,2,1),1)])
@pytest.mark.parametrize('pno',[0.,1e-4])
def test_bounded_transport_preserves_chi_canonical_limit_and_corrections(monkeypatch,mesh,core,pno,record_property):
    ref,_,_ = fixture(monkeypatch,mesh,bool(core))
    options = dict(frozen_core=core,tcut_pno=pno,conv_tol_energy=1e-13,conv_tol_residual=1e-12)
    prepared = local.run_chi_local_mp2(ref,transport_backend='prepared',**options)
    bounded = local.run_chi_local_mp2(ref,transport_backend='bounded',**options)
    a,b = prepared.translation.solution,bounded.translation.solution
    assert a.converged and b.converged
    for x,y in zip(a.amplitudes,b.amplitudes):
        np.testing.assert_array_equal(x,y)
    for x,y in zip(a.residuals,b.residuals):
        np.testing.assert_array_equal(x,y)
    assert a.e_pair_per_cell == b.e_pair_per_cell
    assert a.e_pno_correction_per_cell == b.e_pno_correction_per_cell
    assert prepared.translation.pair_build.pno_ranks == bounded.translation.pair_build.pno_ranks
    if pno == 0:
        canonical = evaluate_same_source_mp2(ref,frozen_core=core)
        error = abs(b.e_pair_per_cell-canonical.e_corr_per_cell)
        assert error < 2e-11
        record_property('bounded_canonical_error',error)
    record_property('bounded_raw_residual',b.max_residual)
    assert b.provenance['transport_preparation']['numerical_link_bytes'] == 0


@pytest.mark.parametrize('provider', ['dense', 'streamed'])
@pytest.mark.parametrize('compact', [False, True])
@pytest.mark.parametrize('core', [0, 1])
def test_cosine_seed_full_space_preserves_raw_canonical_and_partitions(monkeypatch, provider, compact, core):
    ref, _, _ = fixture(monkeypatch, (4,1,1), bool(core))
    expected = evaluate_same_source_mp2(ref, frozen_core=core)
    result = local.run_chi_local_mp2(ref, frozen_core=core, seed_basis='cosine',
        factor_provider=provider, compact_storage=compact, max_iter=250,
        conv_tol_energy=1e-13, conv_tol_residual=1e-11)
    assert result.converged
    assert result.raw_e_corr_per_cell == pytest.approx(expected.e_corr_per_cell, abs=2e-9)
    selection = result.translation.occupied_selection
    assert selection.provenance['seed_basis'] == 'cosine'
    assert not selection.active_seed_aos and not selection.frozen_seed_aos
    assert len(selection.provenance['active_seed_indices']) == (1 if core else 2)
    assert len(selection.provenance['frozen_seed_indices']) == core
    inputs = selection.provenance['source_inputs']
    for key in ('active_projector_error', 'frozen_projector_error', 'cross_partition_error'):
        assert inputs[key] < 1e-10


def test_cosine_seeds_span_disjoint_character_support_without_changing_default():
    from tests.test_dlpno_source import _translation_selection_projector_source
    from vibeqc.dlpno.translation_selection import select_translation_occupied
    occupied = np.array([[1.,0.], [0.,1.], [1.,0.], [0.,-1.]])/np.sqrt(2)
    source = _translation_selection_projector_source(occupied)
    with pytest.raises(ValueError, match='common-seed'):
        select_translation_occupied(source, (2,1,1))
    result = select_translation_occupied(source, (2,1,1), seed_basis='cosine')
    np.testing.assert_allclose(result.occupied.T @ result.occupied, np.eye(2), atol=2e-14)
    np.testing.assert_allclose(result.occupied @ result.occupied.T, occupied @ occupied.T, atol=2e-14)
    np.testing.assert_array_equal(source.reference.mo_coeffs, occupied)
    assert result.provenance['active_seed_indices'] == (0,)
    assert result.active_seed_aos == ()


def test_cosine_seed_workspace_refuses_before_mixture_allocation(monkeypatch):
    from vibeqc.dlpno.translation_preparation import selection_workspace
    from vibeqc.dlpno import translation_preparation as prep
    from vibeqc.dlpno.translation_selection import select_translation_occupied
    ref, _, _ = fixture(monkeypatch, (4,1,1), True)
    source = local.build_chi_translation_source(ref, frozen_core=1, compact_occupied=True)
    bound = selection_workspace((4,1,1),4,2,10**7,seed_basis='cosine')['accounted_peak_bytes']
    def forbidden(*args): pytest.fail('mixed seeds allocated before budget refusal')
    monkeypatch.setattr(prep, 'normalized_seed_weights', forbidden)
    with pytest.raises(MemoryError, match='occupied selection'):
        select_translation_occupied(source,(4,1,1),n_frozen=4,compact_storage=True,
                                    seed_basis='cosine',max_selection_bytes=bound-1)
    with pytest.raises(ValueError, match='seed_basis'):
        select_translation_occupied(source,(4,1,1),seed_basis='automatic-fallback')


def test_bounded_transport_composes_with_explicit_cosine_seeds(monkeypatch):
    ref,_,_ = fixture(monkeypatch,(4,1,1),True)
    options = dict(frozen_core=1,seed_basis='cosine',compact_storage=True,
                   conv_tol_energy=1e-13,conv_tol_residual=1e-11)
    prepared = local.run_chi_local_mp2(ref,transport_backend='prepared',**options)
    bounded = local.run_chi_local_mp2(ref,transport_backend='bounded',**options)
    assert prepared.converged and bounded.converged
    assert prepared.raw_e_corr_per_cell == bounded.raw_e_corr_per_cell
    for a,b in zip(prepared.translation.solution.amplitudes,bounded.translation.solution.amplitudes):
        np.testing.assert_array_equal(a,b)
