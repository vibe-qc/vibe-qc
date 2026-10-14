"""Independent finite-space MP2 proof, not native chemistry acceptance."""
from dataclasses import dataclass, replace
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from vibeqc.periodic.chi import canonical_mp2 as mp
from vibeqc.periodic.chi import reference as refs
from vibeqc.periodic_fit_source import RSGDFFitSource, factor_digest
from vibeqc.periodic_fitted_support import ChiFittedOperator


@dataclass(frozen=True)
class Diagnostic:
    electronic_method: str
    backend: str = "ri"


@dataclass(frozen=True)
class Assembly:
    c_full: float = 1.
    c_sr: float = 0.


def fixture(monkeypatch, nk=3, unrestricted=False, source='sr', nao=3):
    rng = np.random.default_rng(851)
    k = np.zeros((nk,3)); k[:,0] = np.arange(nk)/nk
    b = rng.normal(size=(2,nao,nao))*.1
    b = b+b.transpose(0,2,1)
    t = rng.normal(size=b.shape)*.1 + 1j*rng.normal(size=b.shape)*.1
    blocks = {0:b}
    for q in range(1,(nk+1)//2):
        blocks[q] = t
        blocks[nk-q] = t.conj().transpose(0,2,1)
    factors = {(i,a):blocks[(a-i)%nk] for i in range(nk) for a in range(nk)}
    groups = tuple(((q/nk,0.,0.),tuple((i,a) for i,a in factors if (a-i)%nk == q)) for q in range(nk))
    records = [dict(q=q, lattice=(2*np.pi*np.eye(3)).tolist(), metric_eigenvalues=[1.,2.], metric_keep=[True,True], rank=2) for q,pairs in groups]
    domains = json.dumps(dict(q_records=records))
    desc = RSGDFFitSource('synthetic-finite',1,source,'model','orb','orb-hash','aux','aux-hash',(),1e-9,'canonical',groups,tuple(map(tuple,k)),10.,None,tuple((p,a.shape) for p,a in sorted(factors.items())),factor_digest(factors),executed_domains_json=domains)
    op = ChiFittedOperator(factors,desc,np.ones(nk)/nk)
    # Arbitrary complex orbital frames expose missing AO conjugations.
    c = np.array([np.linalg.qr(rng.normal(size=(nao,nao))+1j*rng.normal(size=(nao,nao)))[0] for _ in range(nk)])
    z = np.array([np.linalg.qr(rng.normal(size=(nao,nao))+1j*rng.normal(size=(nao,nao)))[0] for _ in range(nk)]) if unrestricted else c.copy()
    energies = np.tile(np.arange(nao)*1.2-2., (nk,1))
    channels = [(c,energies,2 if unrestricted else 1),(z,energies+.17 if unrestricted else energies.copy(),1)]
    method = 'UHF' if unrestricted else 'RHF'
    state = dict(energy=-2., converged=True, overlap=np.tile(np.eye(nao),(nk,1,1)),
        kpoint_weights=np.ones(nk)/nk,kpoints_cart=k,smearing_temperature=0.,
        aiccm2026dev_b=Diagnostic(method), exact_exchange_assembly=Assembly(), process_provenance={'fixture':1},
        fitted_operator=op, executed_fit_source=replace(desc,executed_domains_json=json.dumps(dict(fitted_operator_identity=op.identity))),
        chi_scf_acceptance=dict(final_rebuild=True,stationary=True,energy_residual=0.,energy_consistency_error=0.,orbital_residual=0.,density_refill_residual=0.))
    for spin,(coeff,e,n) in enumerate(channels):
        suffix = ('_alpha','_beta')[spin] if unrestricted else ''
        multiplicity = 1 if unrestricted else 2
        occ = np.zeros_like(e); occ[:,:n] = multiplicity
        state.update({
            'mo_coeffs'+suffix:coeff, 'mo_energies'+suffix:e,
            'density'+suffix:multiplicity*coeff[:,:,:n]@coeff[:,:,:n].conj().transpose(0,2,1),
            'fock'+suffix:(coeff*e[:,None,:])@coeff.conj().transpose(0,2,1),
            'occupations'+suffix:occ})
    hf = SimpleNamespace(**state)
    monkeypatch.setattr(refs,'loaded_process_provenance',lambda:{'fixture':1})
    ref = refs.ChiHFReference(hf,os.getpid(),'request',refs._hf_state_identity(hf),method)
    ref = replace(ref,seal=refs._digest((ref.owner_pid,ref.request_identity,ref.state_identity,ref.method,ref.quantitative_status)))
    return ref, channels


def spin_orbital_oracle(operator, channels, frozen=(0,0), real_space=False):
    """Full AO four-index tensor, then explicit spin-orbital antisymmetrization.

    No production momentum, batching, AO/MO, active-space or energy helper.
    All four character indices are traversed; nonconserving elements are zero.
    """
    nk = len(channels[0][0]); nao = channels[0][0].shape[1]
    lattice=np.asarray(json.loads(operator.descriptor.executed_domains_json)["q_records"][0]["lattice"])
    momenta=np.asarray(operator.descriptor.kpoints_cart)@lattice/(2*np.pi)
    eri = np.zeros((nk,nao,nk,nao,nk,nao,nk,nao),complex)
    for ki in range(nk):
        for ka in range(nk):
            for kj in range(nk):
                for kb in range(nk):
                    transfer=momenta[ki]+momenta[kj]-momenta[ka]-momenta[kb]
                    if any(abs(x-round(x))>1e-10 for x in transfer): continue
                    for p in range(nao):
                        for q in range(nao):
                            for r in range(nao):
                                for s in range(nao):
                                    eri[ki,p,ka,q,kj,r,kb,s] = sum(operator[ki,ka][P,p,q]*operator[kb,kj][P,s,r].conjugate() for P in range(operator[ki,ka].shape[0]))/nk
    eri = eri.reshape((nk*nao,)*4)
    transform = np.kron(np.exp(2j*np.pi*np.outer(np.arange(nk),np.arange(nk))/nk)/np.sqrt(nk),np.eye(nao))
    if real_space:
        eri = np.einsum("up,vq,wr,xs,pqrs->uvwx",transform,transform.conj(),transform,transform.conj(),eri,optimize=True)
    occ,vir = [],[]
    for spin,(c,e,n) in enumerate(channels):
        for k in range(nk):
            for p in range(c.shape[-1]):
                vector = np.zeros(nk*nao,complex); vector[k*nao:(k+1)*nao] = c[k,:,p]
                if real_space: vector = transform @ vector
                item = (spin,vector,e[k,p])
                if frozen[spin] <= p < n: occ.append(item)
                elif p >= n: vir.append(item)
    energy = 0.
    for si,i,ei in occ:
        for sj,j,ej in occ:
            for sa,a,ea in vir:
                for sb,b,eb in vir:
                    g = np.einsum('p,q,r,s,pqrs->',i.conj(),a,j.conj(),b,eri) if si == sa and sj == sb else 0.
                    x = np.einsum('p,q,r,s,pqrs->',i.conj(),b,j.conj(),a,eri) if si == sb and sj == sa else 0.
                    energy += .25*abs(g-x)**2/(ei+ej-ea-eb)
    return float(energy/nk)


@pytest.mark.parametrize('nk',[1,3])
@pytest.mark.parametrize('source',['classic','sr'])
@pytest.mark.parametrize('unrestricted',[False,True])
def test_independent_spin_orbital_oracle(monkeypatch,nk,source,unrestricted):
    ref,channels = fixture(monkeypatch,nk,unrestricted,source)
    expected = spin_orbital_oracle(ref.hf_result.fitted_operator,channels)
    for batch in (1,2,7):
        result = mp.evaluate_same_source_mp2(ref,auxiliary_batch_size=batch)
        assert result.e_corr_per_cell == pytest.approx(expected,abs=2e-13)
        assert result.energy == pytest.approx(-2.+expected)
        assert result.quantitative_status == 'not-qualified'


def test_closed_shell_collapse_and_frozen_core(monkeypatch):
    ref,channels = fixture(monkeypatch)
    adapter = mp.CanonicalFactorAdapter(ref.hf_result.fitted_operator)
    aa,bb,ab = mp._contract(adapter,channels,(0,0),1,1e-12)
    assert aa == pytest.approx(bb,abs=1e-14)
    assert aa+bb+ab == pytest.approx(spin_orbital_oracle(adapter.operator,channels),abs=2e-13)
    result = mp.evaluate_same_source_mp2(ref,frozen_core=1)
    assert result.e_corr_per_cell == 0
    assert result.active_spaces == ((1,0,2),(1,0,2))


@pytest.mark.parametrize('change', ['phase','energies','density','occupations','source'])
def test_mutated_reference_refused(monkeypatch,change):
    ref,_ = fixture(monkeypatch)
    h = ref.hf_result
    if change == 'phase': h.mo_coeffs[1,0,0] *= 1j
    elif change == 'energies': h.mo_energies[0,0] += .1
    elif change == 'density': h.density[0,0,0] += .1
    elif change == 'occupations': h.occupations[0,0] = .5
    else: h.executed_fit_source = replace(h.executed_fit_source,source='classic')
    with pytest.raises(ValueError,match='changed'):
        mp.evaluate_same_source_mp2(ref)


def test_rank_and_factor_corruption_refused(monkeypatch):
    ref,_ = fixture(monkeypatch)
    op = ref.hf_result.fitted_operator
    domains=json.loads(op.descriptor.executed_domains_json)
    domains['q_records'][0]['rank']=1
    bad = ChiFittedOperator(op,replace(op.descriptor,executed_domains_json=json.dumps(domains)),op._weights)
    with pytest.raises(ValueError,match='rank'):
        mp.CanonicalFactorAdapter(bad)
    bad = ChiFittedOperator({p:a*1j for p,a in op.items()},op.descriptor,op._weights)
    with pytest.raises(ValueError,match='identity'):
        mp.CanonicalFactorAdapter(bad)


def test_denominator_refusal_and_phase_covariance(monkeypatch):
    ref,channels = fixture(monkeypatch)
    adapter = mp.CanonicalFactorAdapter(ref.hf_result.fitted_operator)
    expected = mp._contract(adapter,channels,(0,0),1,1e-12)
    rng=np.random.default_rng(65)
    rotated=[(c*np.exp(1j*rng.normal(size=e.shape))[:,None,:],e,n) for c,e,n in channels]
    assert mp._contract(adapter,rotated,(0,0),1,1e-12) == pytest.approx(expected,abs=1e-13)
    with pytest.raises(ValueError,match='denominator'):
        mp._contract(adapter,[(c,np.zeros_like(e),n) for c,e,n in channels],(0,0),1,1e-12)


def test_finite_real_space_character_equivalence(monkeypatch):
    ref,channels = fixture(monkeypatch,unrestricted=True)
    expected = spin_orbital_oracle(ref.hf_result.fitted_operator,channels,real_space=True)
    assert mp.evaluate_same_source_mp2(ref).e_corr_per_cell == pytest.approx(expected,abs=2e-13)


def test_virtual_permutations_and_degenerate_rotations(monkeypatch):
    ref,channels = fixture(monkeypatch)
    adapter = mp.CanonicalFactorAdapter(ref.hf_result.fitted_operator)
    channels = [(c,np.array([[-2.,1.,1.]]*3),n) for c,e,n in channels]
    expected = mp._contract(adapter,channels,(0,0),1,1e-12)
    rotation = np.array([[1,1j],[1j,1]])/np.sqrt(2)
    rotated=[]
    for c,e,n in channels:
        x=c.copy(); x[:,:,1:]=x[:,:,1:]@rotation
        rotated.append((x,e,n))
    assert mp._contract(adapter,rotated,(0,0),3,1e-12) == pytest.approx(expected,abs=2e-13)
    permuted=[(c[:,:,[0,2,1]],e[:,[0,2,1]],n) for c,e,n in channels]
    assert mp._contract(adapter,permuted,(0,0),2,1e-12) == pytest.approx(expected,abs=2e-13)


def reseal(ref):
    updated = replace(ref,state_identity=refs._hf_state_identity(ref.hf_result))
    return replace(updated,seal=refs._digest((updated.owner_pid,updated.request_identity,updated.state_identity,updated.method,updated.quantitative_status)))


def modified_operator(op,factors,records=None):
    domains=json.loads(op.descriptor.executed_domains_json)
    if records is not None: domains['q_records']=records
    descriptor=replace(op.descriptor,factor_sha256=factor_digest(factors),factor_shapes=tuple((p,a.shape) for p,a in sorted(factors.items())),executed_domains_json=json.dumps(domains))
    return ChiFittedOperator(factors,descriptor,op._weights)


def test_nontrivial_unrestricted_frozen_core(monkeypatch):
    ref,channels=fixture(monkeypatch,unrestricted=True)
    result=mp.evaluate_same_source_mp2(ref,frozen_core=(1,0))
    oracle=spin_orbital_oracle(ref.hf_result.fitted_operator,channels,frozen=(1,0))
    assert abs(oracle)>1e-8
    assert result.e_corr_per_cell == pytest.approx(oracle,abs=2e-13)
    assert result.active_spaces == ((1,1,1),(0,1,2))


def test_independent_q_metric_gauges(monkeypatch):
    ref,channels=fixture(monkeypatch,unrestricted=True)
    op=ref.hf_result.fitted_operator
    rng=np.random.default_rng(1934)
    factors={}
    for q,pairs in op.descriptor.q_grouping:
        u=np.linalg.qr(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))[0]
        for pair in pairs: factors[pair]=np.einsum('PQ,Qij->Pij',u,op[pair])
    adapter=mp.CanonicalFactorAdapter(modified_operator(op,factors))
    assert mp._contract(adapter,channels,(0,0),1,1e-12) == pytest.approx(mp._contract(mp.CanonicalFactorAdapter(op),channels,(0,0),2,1e-12),abs=2e-13)


def test_retained_metric_nullspace(monkeypatch):
    ref,channels=fixture(monkeypatch)
    op=ref.hf_result.fitted_operator
    factors={p:a.copy() for p,a in op.items()}
    for a in factors.values(): a[0]=0
    records=json.loads(op.descriptor.executed_domains_json)['q_records']
    for r in records: r.update(metric_eigenvalues=[0.,2.],metric_keep=[False,True],rank=1)
    op=modified_operator(op,factors,records)
    adapter=mp.CanonicalFactorAdapter(op)
    assert adapter.retained_ranks == (1,1,1)
    assert sum(mp._contract(adapter,channels,(0,0),1,1e-12)) == pytest.approx(spin_orbital_oracle(op,channels),abs=2e-13)


@pytest.mark.parametrize('kind',['energy_residual','energy_consistency_error','orbital_residual','density_refill_residual','smearing','fractional','noncanonical','weights'])
def test_resealed_invalid_state_is_not_admitted(monkeypatch,kind):
    ref,_=fixture(monkeypatch)
    h=ref.hf_result
    if kind in h.chi_scf_acceptance: h.chi_scf_acceptance[kind]=1e-4
    elif kind=='smearing': h.smearing_temperature=.1
    elif kind=='fractional': h.occupations[0,0]=1.5
    elif kind=='noncanonical': h.mo_energies[0,0]+=.1
    else: h.kpoint_weights=np.array([.2,.3,.5])
    with pytest.raises(ValueError): mp.evaluate_same_source_mp2(reseal(ref))


def test_separated_noninteracting_fragments_are_additive(monkeypatch):
    ref,channels=fixture(monkeypatch,nk=1)
    op=ref.hf_result.fitted_operator
    b=op[0,0]
    big=np.zeros((4,6,6),complex); big[:2,:3,:3]=b; big[2:,3:,3:]=b
    records=json.loads(op.descriptor.executed_domains_json)['q_records']
    records[0].update(metric_eigenvalues=[1.,2.,1.,2.],metric_keep=[True]*4,rank=4)
    doubled=modified_operator(op,{(0,0):big},records)
    states=[]
    order=[0,3,1,2,4,5]
    for c,e,n in channels:
        coef=np.zeros((1,6,6),complex);coef[0,:3,:3]=c[0];coef[0,3:,3:]=c[0]
        states.append((coef[:,:,order],np.concatenate((e,e),axis=1)[:,order],2))
    energy=sum(mp._contract(mp.CanonicalFactorAdapter(doubled),states,(0,0),1,1e-12))
    assert energy == pytest.approx(2*mp.evaluate_same_source_mp2(ref).e_corr_per_cell,abs=2e-13)


def test_output_record_has_correlation_and_admission(monkeypatch):
    from vibeqc.output.chi_correlation import chi_correlation_record
    ref,_=fixture(monkeypatch)
    result=mp.evaluate_same_source_mp2(ref)
    record=chi_correlation_record(result)
    assert record['energy_total_per_cell'] == pytest.approx(record['energy_hf_per_cell']+record['energy_correlation_per_cell'])
    assert record['source_identity']==ref.hf_result.fitted_operator.identity
    assert record['citation_route']=='chi-same-source-mp2'
    assert record['scf_acceptance']['stationary']
    assert record['public_admission'] == 'not-requested: developer evaluation'
    assert record['active_spaces'][0]['occupied_per_character']==1


@pytest.mark.parametrize('source', ['classic', 'sr'])
@pytest.mark.parametrize('unrestricted', [False, True])
def test_public_library_requires_native_source_contract(monkeypatch, source, unrestricted):
    from vibeqc.periodic.chi import correlation
    ref, _ = fixture(monkeypatch, source=source, unrestricted=unrestricted)
    if source == 'sr' and unrestricted:
        with pytest.raises(NotImplementedError, match='SR UMP2 remains gated'):
            correlation.run_chi_mp2(ref)
        return
    monkeypatch.setattr(mp, 'evaluate_same_source_mp2',
                        lambda *a, **k: pytest.fail('public refusal must precede correlation'))
    with pytest.raises(ValueError, match='native finite-source'):
        correlation.run_chi_mp2(ref)


@pytest.mark.parametrize('method', ['RKS', 'UKS'])
def test_public_library_unsupported_methods_stay_gated(method):
    from vibeqc.periodic.chi.correlation import require_public_mp2_admission
    with pytest.raises(NotImplementedError, match='only HF'):
        require_public_mp2_admission('mp2', method, 'classic')
    with pytest.raises(NotImplementedError, match=r'CCSD\(T\)'):
        require_public_mp2_admission('ccsd', 'RHF', 'sr')


@pytest.mark.parametrize('unrestricted', [False, True])
def test_job_adapter_keeps_sealed_hf_and_writes_actual_correlation(monkeypatch, unrestricted):
    from vibeqc.periodic.chi.correlation import _evaluate_job
    from vibeqc.output.chi_correlation import chi_correlation_record, write_chi_correlation
    ref, channels = fixture(monkeypatch, unrestricted=unrestricted)
    ref.hf_result._chi_hf_request_identity = ref.request_identity
    sealed = refs._seal_chi_hf_result(ref.hf_result, ref.method)
    assert sealed.state_identity == ref.state_identity
    result = _evaluate_job(sealed, auxiliary_batch_size=1)
    assert result.hf_result is ref.hf_result
    assert result.energy == ref.hf_result.energy
    assert result.e_correlation == pytest.approx(spin_orbital_oracle(ref.hf_result.fitted_operator, channels), abs=2e-13)
    assert result.e_total_correlated == pytest.approx(result.energy + result.e_correlation)
    assert sealed.validate(sealed.request_identity, sealed.method) is result.hf_result
    assert result.correlation.backend == ('chi-same-source-ump2' if unrestricted else 'chi-same-source-mp2')
    output = []
    write_chi_correlation(output.append, result.correlation)
    text = ''.join(output)
    assert 'union-and-weight' not in text
    assert 'not-qualified' in text and 'not-requested' in text
    assert 'E(corr)' in text and 'E(SCF+corr)' in text
    assert f'{result.e_correlation:.10f}' in text
    record = chi_correlation_record(result.correlation)
    assert record['reference_identity'] == sealed.state_identity
    assert record['source_identity'] == ref.hf_result.fitted_operator.identity
    # The runner stamps guess metadata on its wrapper without breaking the HF seal.
    selection = object()
    stamped = replace(result, guess_selection=selection)
    assert stamped.guess_selection is selection
    assert stamped.hf_result is result.hf_result
    assert sealed.validate(sealed.request_identity, sealed.method) is result.hf_result


@pytest.mark.parametrize('kind',['hybrid','cosx','rohf','ecp','screened'])
def test_unsupported_reference_theories_refuse(monkeypatch,kind):
    ref,_=fixture(monkeypatch)
    h=ref.hf_result
    if kind=='hybrid': h.exact_exchange_assembly=Assembly(.5,0.)
    elif kind=='cosx': h.aiccm2026dev_b=Diagnostic('RHF','rijcosx')
    elif kind=='rohf': h.aiccm2026dev_b=Diagnostic('ROHF')
    elif kind=='ecp': h.aiccm2026dev_b=ECPDiagnostic('RHF')
    else:
        op=h.fitted_operator
        h.fitted_operator=ChiFittedOperator(op,replace(op.descriptor,numerical_support=(('physical_kernel','erfc'),)),op._weights)
    with pytest.raises((ValueError,NotImplementedError)):
        mp.evaluate_same_source_mp2(reseal(ref))


@dataclass(frozen=True)
class ECPDiagnostic(Diagnostic):
    ecp_total_ncore: int = 2


def test_restricted_unrestricted_closed_shell_evaluator_collapse(monkeypatch):
    ref,_=fixture(monkeypatch)
    expected=mp.evaluate_same_source_mp2(ref)
    h=ref.hf_result
    for spin in ('alpha','beta'):
        for field in ('mo_coeffs','mo_energies','fock','density','occupations'):
            value=np.asarray(getattr(h,field)).copy()
            if field in ('density','occupations'): value/=2
            setattr(h,field+'_'+spin,value)
    h.aiccm2026dev_b=Diagnostic('UHF')
    uhf=mp.evaluate_same_source_mp2(reseal(replace(ref,method='UHF')))
    assert uhf.e_corr_per_cell == pytest.approx(expected.e_corr_per_cell,abs=1e-14)
    assert uhf.e_aa_per_cell == pytest.approx(expected.e_aa_per_cell,abs=1e-14)
    assert uhf.e_ab_per_cell == pytest.approx(expected.e_ab_per_cell,abs=1e-14)


def test_occupied_permutation_and_wrong_momentum_control(monkeypatch):
    ref,channels=fixture(monkeypatch,unrestricted=True)
    adapter=mp.CanonicalFactorAdapter(ref.hf_result.fitted_operator)
    expected=mp._contract(adapter,channels,(0,0),1,1e-12)
    c,e,n=channels[0]
    changed=[(c[:,:,[1,0,2]],e[:,[1,0,2]],n),channels[1]]
    assert mp._contract(adapter,changed,(0,0),2,1e-12) == pytest.approx(expected,abs=2e-13)
    with pytest.raises(ValueError,match='momentum'):
        adapter.integral((0,1),(0,0),c[0,:,0],c[1,:,n:],c[0,:,n:],c[0,:,0],1)


@pytest.mark.parametrize('source', ['classic', 'sr'])
@pytest.mark.parametrize('method', ['RHF', 'UHF'])
def test_public_capabilities_require_explicit_finite_controls(source, method):
    from vibeqc.periodic.chi.correlation import require_public_mp2_request
    from vibeqc.periodic.chi.fitted import ChiCoulombControls
    controls = ChiCoulombControls(6.2, 6.2, 18., 8., 5., .6,
        **(dict(auxiliary_cutoff_bohr=10., omega_bohr_inv=.7) if source == 'sr' else {}))
    request = dict(dimension=3, backend='ri', controls=controls,
                   conv_tol_energy=1e-11, conv_tol_grad=1e-9)
    if method == 'UHF' and source == 'sr':
        with pytest.raises(NotImplementedError, match='SR UMP2 remains gated'):
            require_public_mp2_request('mp2', method, source, **request)
        return
    require_public_mp2_request('mp2', method, source, **request)
    for change in (dict(backend='four_center'), dict(dimension=2), dict(controls=None),
                   dict(conv_tol_energy=1e-7), dict(conv_tol_grad=None),
                   dict(conv_tol_grad=float('nan')), dict(conv_tol_energy=True)):
        with pytest.raises((ValueError, TypeError, NotImplementedError)):
            require_public_mp2_request('mp2', method, source, **(request | change))
    for field in ('tail_reciprocal_cutoff_ha', 'integral_screen_error', 'fit_screen_threshold'):
        # Constructors also reject controls that do not describe this source.
        with pytest.raises((ValueError, NotImplementedError)):
            require_public_mp2_request('mp2', method, source,
                **(request | dict(controls=replace(controls, **{field: 20. if 'tail' in field else 1e-10}))))


@pytest.mark.parametrize('frozen,batch', [(True, 1), (-1, 1), ((0, 1), 1), (0, 0), (0, True), (0, 1.5)])
def test_public_core_panel_options_refuse_before_evaluation(frozen, batch):
    from vibeqc.periodic.chi.correlation import validate_public_mp2_options
    with pytest.raises(ValueError):
        validate_public_mp2_options('RHF', 'classic', frozen, batch)


@pytest.mark.parametrize('method,source', [('RHF','sr'), ('UHF','classic'), ('UHF','sr')])
def test_public_frozen_spaces_require_measured_source_and_spin(method, source):
    from vibeqc.periodic.chi.correlation import validate_public_mp2_options
    assert validate_public_mp2_options(method, source, 0, 7) == ((0, 0), 7)
    with pytest.raises(NotImplementedError, match='only for explicit classic RHF'):
        validate_public_mp2_options(method, source, 1, 7)
    assert validate_public_mp2_options('RHF', 'classic', 1, 7) == ((1, 1), 7)


def test_public_object_mesh_requires_complete_unshifted_uniform_characters():
    from vibeqc.periodic.chi.correlation import require_public_mp2_mesh
    lattice = 2*np.pi*np.eye(3)
    k = np.array([[0.,0.,0.], [1/3,0.,0.], [-1/3,0.,0.]])
    require_public_mp2_mesh(lattice, (3,1,1), k, [1/3]*3)
    for points, weights in ((k+.1, [1/3]*3), (k[:2], [.5]*2),
                            (k[[0,0,2]], [1/3]*3), (k, [.2,.3,.5])):
        with pytest.raises(NotImplementedError):
            require_public_mp2_mesh(lattice, (3,1,1), points, weights)


def test_public_cc_options_preserve_independent_frozen_spin_counts():
    from vibeqc.periodic.chi.correlation import validate_public_cc_options
    assert validate_public_cc_options('UHF', 'classic', (1, 0), 3) == ((1, 0), 3)
    assert validate_public_cc_options('RHF', 'classic', 1, 3) == ((1, 1), 3)
    for frozen in (True, (-1, 0), (1, False), (1,), (1.5, 0)):
        with pytest.raises(ValueError):
            validate_public_cc_options('UHF', 'classic', frozen, 3)
    with pytest.raises(ValueError, match='restricted frozen cores'):
        validate_public_cc_options('RHF', 'classic', (1, 0), 3)
    for method, source in [('UHF', 'sr'), ('RHF', 'sr'), ('ROHF', 'classic'),
                           ('UKS', 'classic'), ('RKS', 'classic'), ('UHF', 'auto')]:
        with pytest.raises(NotImplementedError):
            validate_public_cc_options(method, source, 0, 3)


def test_public_uhf_cc_requires_native_contract_before_evaluation(monkeypatch):
    from vibeqc.periodic.chi import canonical_cc, correlation
    ref, _ = fixture(monkeypatch, source='classic', unrestricted=True)
    monkeypatch.setattr(canonical_cc, 'evaluate_same_source_cc',
                        lambda *a, **k: pytest.fail('public refusal must precede correlation'))
    with pytest.raises(ValueError, match='native finite-source'):
        correlation.run_chi_ccsd(ref, frozen_core=(1, 0))


@pytest.mark.parametrize('compute_triples', [False, True])
def test_public_uhf_cc_wrapper_and_output_preserve_spin_source_contract(monkeypatch, compute_triples):
    """Synthetic public/output contract only; no native chemistry acceptance."""
    from dataclasses import asdict
    from vibeqc.periodic.chi.canonical_cc import ChiCCOptions
    from vibeqc.periodic.chi.correlation import run_chi_ccsd
    from vibeqc.periodic.chi.fitted import ChiCoulombControls
    from vibeqc.output.chi_correlation import chi_correlation_record, write_chi_correlation
    ref, _ = fixture(monkeypatch, nk=1, source='classic', unrestricted=True, nao=4)
    hf = ref.hf_result
    controls = ChiCoulombControls(6.2, 6.2, 18., 8., 5., .6)
    op = hf.fitted_operator
    # Exercise public schema admission with synthetic factors; this is not a
    # native source calculation and cannot qualify the scientific method.
    domains = json.loads(op.descriptor.executed_domains_json)
    domains.update(schema='vibeqc.chi.fitted-domains/v1', controls=asdict(controls))
    descriptor = replace(op.descriptor, engine='native-chi-classic-finite-contract',
                         executed_domains_json=json.dumps(domains))
    hf.fitted_operator = ChiFittedOperator(op, descriptor, op._weights)
    hf.executed_fit_source = replace(descriptor, executed_domains_json=json.dumps(
        dict(fitted_operator_identity=hf.fitted_operator.identity)))
    hf.e_nuclear = .2
    ref = reseal(ref)
    result = run_chi_ccsd(ref, frozen_core=(1, 0), auxiliary_batch_size=1,
                         options=ChiCCOptions(compute_triples=compute_triples))
    cc = result.correlation
    assert result.hf_result is hf
    assert ref.validate(ref.request_identity, 'UHF') is hf
    assert cc.method == ('UCCSD(T)' if compute_triples else 'UCCSD')
    assert cc.active_spaces == ((1, 1, 2), (0, 1, 3))
    assert cc.converged and cc.residual_norm < 1e-8
    assert cc.public_admission.startswith('admitted: experimental')
    assert cc.quantitative_status == 'not-qualified'
    record = chi_correlation_record(cc)
    assert record['reference_method'] == 'UHF'
    assert record['citation_route'] == 'chi-same-source-' + cc.method.lower()
    assert record['triples_variant'] == (cc.triples_variant if compute_triples else 'not computed')
    assert record['source_identity'] == hf.fitted_operator.identity
    assert record['reference_identity'] == ref.state_identity
    assert record['retained_metric_ranks'] == (2,)
    assert record['canonical_residuals'][0]['spin'] == 'alpha'
    assert record['canonical_residuals'][1]['spin'] == 'beta'
    assert max(record['canonical_residuals'][0][key] for key in
               ('metric_orthogonality', 'fock_eigenpair', 'density_refill')) < 1e-12
    assert record['energy_correlation_per_cell'] == pytest.approx(
        record['energy_ccsd_per_cell'] + record['energy_triples_per_cell'])
    assert record['energy_total_per_cell'] == pytest.approx(hf.energy + cc.e_corr_per_cell)
    assert record['energy_total_electronic_per_cell'] == pytest.approx(cc.energy - .2)
    output = []
    write_chi_correlation(output.append, cc)
    rendered = ''.join(output)
    assert cc.method in rendered and cc.backend in rendered and cc.source_identity in rendered
    assert 'not-qualified' in rendered and 'UHF' in rendered
    assert 'E(CCSD)' in rendered and 'E(triples)' in rendered
    assert "'virtual_per_character': 3" in rendered


def test_public_uhf_cc_frozen_controls_pass_runner_preflight(tmp_path):
    import tomllib
    import vibeqc as vq
    from vibeqc.periodic.chi.canonical_cc import ChiCCOptions
    from vibeqc.periodic.chi.fitted import ChiCoulombControls
    cell = vq.PeriodicSystem(3, np.eye(3)*8, [vq.Atom(3, [1., 1., 1.])], 0, 2)
    basis = vq.BasisSet(cell.unit_cell_molecule(), 'sto-3g')
    stem = tmp_path / 'uhf-cc-preflight'
    vq.run_periodic_job(cell, basis, method='aiccm', variant='chi', correlation='ccsd',
        scf_reference='uhf', fit_source='classic', aiccm_backend='ri',
        chi_coulomb_controls=ChiCoulombControls(6.2, 6.2, 18., 8., 5., .6),
        chi_cc_options=ChiCCOptions(compute_triples=False), chi_frozen_core=(1, 0),
        conv_tol_energy=1e-11, conv_tol_grad=1e-9, kpoints=(1, 1, 1),
        dry_run=True, output=stem, output_qvf=False, record_hostname=False)
    record = tomllib.loads(stem.with_suffix('.system').read_text())
    assert record['outputs']['status'] == 'dry_run'
