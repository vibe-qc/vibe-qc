"""Independent finite orbital-space checks; no native scientific acceptance."""
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import expm

from vibeqc.periodic.chi import rohf as r
from vibeqc.periodic.chi import reference as refs


class DenseOperator:
    """AO four-index J/K contractions, independent of production fit helpers."""
    def __init__(self, eri):
        self.eri = eri
    def coulomb(self, ds):
        return np.asarray([np.einsum('pqrs,sr->pq', self.eri, np.mean(ds,axis=0))]*len(ds))
    def exchange(self, ds):
        return np.asarray([np.einsum('prsq,rs->pq', self.eri, d) for d in ds])


def model(n=4, strength=.07):
    rng = np.random.default_rng(612)
    b = rng.normal(size=(3,n,n))*strength
    b += b.transpose(0,2,1)
    eri = np.einsum('Ppq,Prs->pqrs', b, b)
    h = np.diag(np.arange(n)*1.2-2.)
    h += .08*(np.ones((n,n))-np.eye(n))
    return h[None].astype(complex), np.eye(n)[None], DenseOperator(eri)


def energy_oracle(c, h, eri, na, nb):
    """Explicit occupied spin-orbital determinant expectation in AO tensors."""
    occupied = [(spin, c[:,i]) for spin, count in ((0,na),(1,nb)) for i in range(count)]
    e = 0j
    for spin, i in occupied:
        e += np.vdot(i, h@i)
        for other, j in occupied:
            for p,q,r0,s in np.ndindex(eri.shape):
                e += .5*i[p].conjugate()*i[q]*j[r0].conjugate()*j[s]*eri[p,q,r0,s]
                if spin == other:
                    e -= .5*i[p].conjugate()*j[q]*j[r0].conjugate()*i[s]*eri[p,q,r0,s]
    assert abs(e.imag)<1e-12
    return e.real


def test_complex_gradient_finite_difference_and_chart_chain_rule():
    h, s, op = model()
    rng = np.random.default_rng(178)
    c = np.linalg.qr(rng.normal(size=(4,4))+1j*rng.normal(size=(4,4)))[0][None]
    state = r._physical_state(c,h,s,np.ones(1),op,3,1,0.,0.)
    grad = r.rohf_gradient(c,state['fock_alpha'],state['fock_beta'],3,1)[0]
    for p,q in ((1,0),(3,0),(3,1)):
        for phase in (1.,1j):
            k = np.zeros((4,4),complex);k[p,q]=phase;k[q,p]=-np.conjugate(phase)
            step=1e-5
            fd=(energy_oracle(c[0]@expm(step*k),h[0],op.eri,3,1)
                -energy_oracle(c[0]@expm(-step*k),h[0],op.eri,3,1))/(2*step)
            assert fd==pytest.approx(np.vdot(grad,k).real,abs=2e-9)
    chart=r._RotationChart(c,3,1)
    x=rng.normal(size=chart.size)*.2
    cc,cache=chart.rotate(x)
    st=r._physical_state(cc,h,s,np.ones(1),op,3,1,0.,0.)
    jac=chart.pullback(st,cache,3,1,np.ones(1))
    for p in range(chart.size):
        xp=x.copy();xm=x.copy();xp[p]+=1e-5;xm[p]-=1e-5
        fd=(energy_oracle(chart.rotate(xp)[0][0],h[0],op.eri,3,1)
            -energy_oracle(chart.rotate(xm)[0][0],h[0],op.eri,3,1))/2e-5
        assert jac[p]==pytest.approx(fd,abs=3e-9)


def test_high_spin_stationary_shared_orbitals_and_physical_rebuild():
    h,s,op=model()
    result=r.optimize_rohf(h,s,[1.],op,3,1,max_iter=160,gradient_tolerance=1e-8)
    assert result.converged, result.chi_scf_acceptance
    assert result.energy==pytest.approx(energy_oracle(result.mo_coeffs[0],h[0],op.eri,3,1),abs=2e-12)
    assert result.s_squared==2.
    assert np.array_equal(result.mo_coeffs_alpha,result.mo_coeffs_beta)
    assert np.array_equal(result.occupations,[[2.,1.,1.,0.]])
    # Individual UHF gradients remain nonzero at a true ROHF stationary point.
    fa,fb=result.fock_alpha[0],result.fock_beta[0]
    da,db=result.density_alpha[0],result.density_beta[0]
    assert max(np.linalg.norm(fa@da-da@fa),np.linalg.norm(fb@db-db@fb))>1e-4


def test_closed_shell_reduces_to_independent_rhf_fixed_point():
    h,s,op=model(strength=.03)
    result=r.optimize_rohf(h,s,[1.],op,2,2,max_iter=160,gradient_tolerance=1e-8)
    d=np.diag([2.,2.,0.,0.]).astype(complex)
    for _ in range(80):
        f=h[0]+op.coulomb(d[None])[0]-.5*op.exchange(d[None])[0]
        _,c=np.linalg.eigh(f)
        new=2*c[:,:2]@c[:,:2].conj().T
        if np.linalg.norm(new-d)<1e-13:break
        d=new
    assert result.converged
    assert np.max(abs(result.density[0]-new))<2e-8
    assert np.max(abs(result.fock_alpha-result.fock_beta))<1e-13


def test_subspace_rotations_preserve_density_and_canonical_frame():
    h,s,op=model(n=6)
    result=r.optimize_rohf(h,s,[1.],op,4,2,max_iter=160,gradient_tolerance=1e-8)
    assert result.converged
    rng=np.random.default_rng(899)
    u=np.zeros((6,6),complex)
    for lo,hi in ((0,2),(2,4),(4,6)):
        u[lo:hi,lo:hi]=np.linalg.qr(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))[0]
    rotated=result.mo_coeffs@u
    for got,expected in zip(r.rohf_densities(rotated,4,2),(result.density_alpha,result.density_beta)):
        assert np.max(abs(got-expected))<1e-13
    rec,eps=r.canonicalize_rohf(rotated,result.fock_alpha,result.fock_beta,4,2)
    assert np.max(abs(eps-result.mo_energies))<1e-12
    assert abs(energy_oracle(rec[0],h[0],op.eri,4,2)-result.energy)<1e-12


def test_complex_character_phase_covariance():
    h,s,op=model()
    phases=np.exp(1j*np.arange(4)*.37)
    u=np.diag(phases)
    transformed=np.einsum('ap,bq,cr,ds,abcd->pqrs',u.conj(),u,u.conj(),u,op.eri,optimize=True)
    result=r.optimize_rohf(h,s,[1.],op,2,1,max_iter=160,gradient_tolerance=1e-8)
    complex_result=r.optimize_rohf((u.conj().T@h@u),s,[1.],DenseOperator(transformed),2,1,max_iter=160,gradient_tolerance=1e-8)
    assert result.converged and complex_result.converged
    assert complex_result.energy==pytest.approx(result.energy,abs=2e-12)
    assert np.max(abs(u@complex_result.density_alpha@u.conj().T-result.density_alpha))<1e-8


@pytest.mark.parametrize('counts',[(2,3),(5,0),(-1,0),(1.5,0),(True,0)])
def test_invalid_counts_refuse(counts):
    h,s,op=model()
    with pytest.raises(ValueError,match='integer'):
        r.optimize_rohf(h,s,[1.],op,*counts)


@dataclass(frozen=True)
class Diagnostic:
    electronic_method: str='ROHF'


def accepted(monkeypatch):
    h,s,op=model()
    result=r.optimize_rohf(h,s,[1.],op,2,1,max_iter=160)
    assert result.converged, result.chi_scf_acceptance
    result.aiccm2026dev_b=Diagnostic()
    result.executed_support={'source':'independent-finite-model'}
    result.process_provenance={'source':'fixture'}
    result._chi_hf_request_identity='fixture'
    monkeypatch.setattr(refs,'loaded_process_provenance',lambda:result.process_provenance)
    return refs._seal_chi_hf_result(result,'ROHF')


@pytest.mark.parametrize('mutation',[
    lambda x:x.rohf_subspaces.update(open=(0,)),
    lambda x:setattr(x,'n_closed',0),
    lambda x:setattr(x,'rohf_canonicalization','different'),
    lambda x:x.mo_coeffs_alpha.__setitem__((0,0,0),.1),
    lambda x:x.fock_beta.__setitem__((0,0,2),.1),
])
def test_subspace_and_physical_state_mutations_invalidate_seal(monkeypatch,mutation):
    reference=accepted(monkeypatch)
    assert reference.validate('fixture','ROHF') is reference.hf_result
    mutation(reference.hf_result)
    with pytest.raises(ValueError,match='state/source/factors'):
        reference.validate('fixture','ROHF')


@pytest.mark.parametrize('dimension,backend,source',[(2,'ri','classic'),(3,'four_center','classic'),(3,'ri','sr'),(3,'ri','auto')])
def test_unsupported_sources_fail_before_integrals(dimension,backend,source):
    with pytest.raises(NotImplementedError,match='3D classic'):
        r.require_rohf_controls(dimension=dimension,backend=backend,fit_source=source,controls=None)


def test_noiter_never_accepted_and_workspace_guard():
    h,s,op=model()
    assert not r.optimize_rohf(h,s,[1.],op,2,1,max_iter=0).converged
    with pytest.raises(MemoryError):
        r.optimize_rohf(h,s,[1.],op,2,1,memory_byte_cap=1)


def synthetic_driver_case(monkeypatch, nk=1):
    """Replace the native source boundary only, retaining public driver wiring."""
    from dataclasses import asdict
    import json
    import vibeqc as vq
    from vibeqc.periodic.chi.fitted import ChiCoulombControls
    from vibeqc.periodic.chi import scf
    from vibeqc import periodic_k_gdf
    from vibeqc.periodic_fit_source import RSGDFFitSource, factor_digest
    from vibeqc.periodic_fitted_support import ChiFittedOperator
    system=vq.PeriodicSystem(3,np.eye(3)*8,[vq.Atom(3,[1.,1.,1.])],0,2)
    basis=vq.BasisSet(system.unit_cell_molecule(),'sto-3g')
    controls=ChiCoulombControls(6.2,6.2,18.,8.,5.,.6)
    options=vq.PeriodicRHFOptions()
    options.conv_tol_energy=1e-11;options.conv_tol_grad=1e-9;options.max_iter=160
    def source(system,basis,kpoints,options,controls,auxiliary_name):
        n=int(basis.nbasis);nk=len(kpoints.weights)
        rng=np.random.default_rng(389)
        b=rng.normal(size=(2,n,n))*.03;b+=b.transpose(0,2,1)
        u=np.asarray([np.diag(np.exp(1j*np.arange(n)*.43*k)) for k in range(nk)])
        factors={(i,j):u[i].conj().T@b@u[j] for i in range(nk) for j in range(nk)}
        groups=tuple((tuple(kpoints.kpoints_cart[q]),tuple((i,j) for i,j in factors if (j-i)%nk==q)) for q in range(nk))
        records=[dict(q=q,lattice=np.asarray(system.lattice).tolist(),metric_eigenvalues=[1.,2.],
                      metric_keep=[True,True],rank=2) for q,pairs in groups]
        # Emulate the boundary's schema, while explicitly marking its arrays
        # synthetic. This is wiring evidence, never a native source verdict.
        descriptor=RSGDFFitSource('native-chi-classic-finite-contract',1,'classic','synthetic-model','orb','orb-hash',
            'aux','aux-hash',tuple(sorted(asdict(controls).items())),1e-9,'canonical',groups,
            tuple(map(tuple,kpoints.kpoints_cart)),18.,None,
            tuple((p,a.shape) for p,a in sorted(factors.items())),factor_digest(factors),
            executed_domains_json=json.dumps(dict(q_records=records,
                schema='vibeqc.chi.fitted-domains/v1',controls=asdict(controls),synthetic_fixture=True)))
        operator=ChiFittedOperator(factors,descriptor,kpoints.weights)
        h=np.diag(np.arange(n)*1.2-2.)+.08*(np.ones((n,n))-np.eye(n))
        h=np.asarray([uk.conj().T@h@uk for uk in u])
        return h,np.tile(np.eye(n),(nk,1,1)),0.,operator,'{}',SimpleNamespace(nbasis=2),'synthetic-aux',options.lattice_opts
    monkeypatch.setattr(r,'_build_source',source)
    monkeypatch.setattr(periodic_k_gdf,'_madelung_for_kmesh',lambda *a:0.)
    monkeypatch.setattr(scf,'_inverse_bloch_transform_native',lambda *a:None)
    return system,basis,controls,options


@pytest.mark.parametrize('nk',[1,3])
def test_public_driver_builds_seals_and_reuses_same_source(monkeypatch,nk):
    from vibeqc.periodic.chi.scf import run_aiccm2026dev_b_rohf
    from vibeqc.periodic.chi.correlation import run_chi_mp2
    system,basis,controls,options=synthetic_driver_case(monkeypatch,nk)
    kwargs=dict(backend='ri',fit_source='classic',coulomb_controls=controls)
    ref=refs.prepare_chi_hf_reference(system,basis,(nk,1,1),options,method='rohf',**kwargs)
    h=ref.hf_result
    assert h.converged
    assert h.aiccm2026dev_b.electronic_method=='ROHF'
    assert 'shared-spatial' in h.aiccm2026dev_b.variational_space
    assert h.finite_torus_convention.ccm_construction=='finite-translation-group-character'
    assert h.fitted_operator.identity in h.executed_fit_source.executed_domains_json
    assert h.rohf_canonicalization==r.ROHF_CANONICALIZATION
    assert run_aiccm2026dev_b_rohf(system,basis,(nk,1,1),options,hf_reference=ref,**kwargs) is h
    correlated=run_chi_mp2(ref,frozen_core=0)
    assert correlated.correlation.backend=='chi-same-source-rohf-rmp2'
    assert correlated.correlation.method=='ROHF-RMP2'
    assert correlated.correlation.e_corr_per_cell<0


def test_analytic_orbital_gradient_response_matches_independent_differences():
    h,s,op=model()
    rng=np.random.default_rng(859)
    c=np.linalg.qr(rng.normal(size=(4,4))+1j*rng.normal(size=(4,4)))[0][None]
    chart=r._RotationChart(c,3,1)
    direction=rng.normal(size=chart.size)
    state=r._physical_state(c,h,s,np.ones(1),op,3,1,0.,0.)
    response=r._rohf_gradient_response(chart,direction,state,s,[1.],op,3,1,0.)
    independent=[]
    for sign in (-1,1):
        rotated,_=chart.rotate(direction*sign*1e-5)
        ca=rotated[0,:,:3];cb=rotated[0,:,:1]
        da=ca@ca.conj().T;db=cb@cb.conj().T
        # Build physical operators directly from the AO 4-index Hamiltonian.
        j=np.einsum('pqrs,sr->pq',op.eri,da+db)
        fa=h[0]+j-np.einsum('prsq,rs->pq',op.eri,da)
        fb=h[0]+j-np.einsum('prsq,rs->pq',op.eri,db)
        ma=rotated[0].conj().T@fa@rotated[0]
        mb=rotated[0].conj().T@fb@rotated[0]
        a=(np.arange(4)<3).astype(float);b=(np.arange(4)<1).astype(float)
        lower=np.array([ma[p,q]*(a[q]-a[p])+mb[p,q]*(b[q]-b[p])
                        for p,q in zip(chart.p,chart.q)])
        independent.append(2*np.r_[lower.real,lower.imag])
    assert np.max(abs(response-(independent[1]-independent[0])/2e-5))<3e-8


def test_reference_method_must_match_executed_method_even_after_reseal(monkeypatch):
    from dataclasses import replace
    reference=accepted(monkeypatch)
    altered=replace(reference,method='UHF')
    altered=replace(altered,seal=refs._digest((altered.owner_pid,altered.request_identity,
        altered.state_identity,altered.method,altered.quantitative_status)))
    with pytest.raises(ValueError,match='spin/method mismatch'):
        altered.validate(altered.request_identity,'UHF')
