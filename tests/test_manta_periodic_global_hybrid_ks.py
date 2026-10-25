"""Complete finite global hybrids: independent AO contractions and SCF."""
import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta
from vibeqc.bipole_fock_ewald import probe_charge_madelung_supercell
from tests.test_manta_periodic_ks import case as pure_case, oracle as semilocal_oracle, occupy, independent_scf
from tests.test_manta_periodic_exchange import _quadrature_bloch
from tests.test_manta_periodic_rhf import _nuclear_reference
from tests.test_manta_periodic_semilocal_xc import bloch_values_gradients

EMPTY_HYBRID='manta-periodic-blyp-global-test'
vq.define_functional(EMPTY_HYBRID,[('GGA_X_B88',.8),('GGA_C_LYP',1.)],hf_exchange_fraction=.2)


def case(species='H2', name='pbe0', spins=1, correction='strict-zero-mode', **kwargs):
    c,k,old=pure_case(species,name,spins,**kwargs)
    o=manta.PeriodicGlobalHybridKSOptions();o.ks=c['options'];o.exchange_q0=correction
    c.update(options=o,exchange_vectors=old['exchange_vectors'],transfer_offsets=old['transfer_offsets'])
    return c,k,old


def oracle(c,k,old,fraction=None):
    # PBE0: Adamo & Barone, JCP 110, 6158 (1999). B3LYP: Becke, JCP
    # 98, 5648 (1993). These reference fractions do not read the native plan.
    if fraction is None:fraction={'pbe0':.25,'b3lyp':.2,EMPTY_HYBRID:.2}[c['functional'].name.lower()]
    s,h,semilocal=semilocal_oracle(c,k,old)
    weights=np.ones(len(k))/len(k);factor=4*np.pi/abs(np.linalg.det(c['system'].lattice))
    transfers=[]
    for target,kt in enumerate(k):
        for source in range(len(k)):
            lo,hi=c['transfer_offsets'][target*len(k)+source:target*len(k)+source+2]
            q=c['exchange_vectors'][lo:hi]
            transfers.append((_quadrature_bloch(c['basis'],q,c['translations'],kt,None),
                weights[source]*factor/np.sum(q*q,axis=1)))
    o=c['options']
    xi=0. if o.exchange_q0=='strict-zero-mode' else probe_charge_madelung_supercell(
        c['system'],tuple(c['mesh'].mesh),alpha=o.madelung_alpha,precision=1e-14)
    scale=fraction/(2 if c['functional'].spin==1 else 1)
    def build(d):
        f,energies,pop=semilocal(d);exchange=np.zeros_like(d)
        for spin in range(len(d)):
            for target in range(len(k)):
                for source in range(len(k)):
                    b,kernel=transfers[target*len(k)+source]
                    exchange[spin,target]+=np.einsum('g,gai,ab,gbj->ij',kernel,b.conj(),d[spin,source],b)
        correction=-scale*xi*(s@d@s)
        ek=-.5*scale*np.einsum('k,skij,skji->',weights,d,exchange).real
        ec=.5*np.einsum('k,skij,skji->',weights,d,correction).real
        return f-scale*exchange+correction,(*energies,ek,ec),pop
    return s,h,build,xi


def plan(c):
    return manta.plan_periodic_global_hybrid_ks(c['basis'],c['system'],c['mesh'],c['functional'],
        len(c['reciprocal_vectors']),int(max(np.diff(c['transfer_offsets']))),len(c['translations']),
        len(c['xc_images']),len(c['points']),c['options'],c['budget'])


def components(r):
    return (r.receipt.one_electron_energy,r.receipt.coulomb_energy,r.receipt.xc_energy,
        r.receipt.uncorrected_exchange_energy,r.receipt.exchange_correction_energy)


@pytest.mark.parametrize('species,name,spins',[
    ('H2','pbe0',1),('H2','b3lyp',1),('H2','pbe0',2),('HeH','pbe0',2),('H','manta-periodic-blyp-global-test',2)])
@pytest.mark.parametrize('correction',['strict-zero-mode','BvK-ewald'])
def test_complete_hybrid_matches_independent_scf(species,name,spins,correction):
    c,k,old=case(species,name,spins,correction);r=manta.run_periodic_global_hybrid_ks(**c)
    assert r.converged and r.receipt.confirmed
    s,h,build,xi=oracle(c,k,old)
    populations=(r.plan.n_alpha,) if spins==1 else (r.plan.n_alpha,r.plan.n_beta)
    d=independent_scf(s,h,build,populations,spins);f,energies,pop=build(r.density)
    np.testing.assert_allclose(r.density,d,atol=3e-9,rtol=0)
    np.testing.assert_allclose(r.fock,f,atol=1e-11,rtol=0)
    np.testing.assert_allclose(r.overlap,s,atol=4e-12,rtol=0)
    np.testing.assert_allclose(r.hcore,h,atol=4e-12,rtol=0)
    np.testing.assert_allclose(components(r),energies,atol=1e-11,rtol=0)
    np.testing.assert_allclose(r.receipt.xc.electrons_grid,pop,atol=1e-11,rtol=0)
    nuclear,_=_nuclear_reference(old)
    assert r.receipt.energy==pytest.approx(sum(nuclear)+sum(energies),abs=2e-11)
    assert r.receipt.madelung==pytest.approx(xi,abs=4e-12)
    assert r.receipt.exchange_energy==pytest.approx(sum(energies[-2:]),abs=1e-11)
    fraction={'pbe0':.25,'b3lyp':.2,EMPTY_HYBRID:.2}[name]
    assert r.receipt.exact_exchange_fraction==pytest.approx(fraction)
    assert r.receipt.xc.semilocal_component_only
    assert r.receipt.xc.required_exact_exchange_fraction==pytest.approx(fraction)
    assert r.receipt.exchange.kernel=='coulomb' and r.plan.exchange.densities==spins
    assert r.receipt.exchange_q0==correction
    assert r.receipt.fock_calls==r.solver_snapshot.fock_calls+1
    assert r.receipt.shell_calls<=r.plan.shell_calls and r.receipt.primitive_pairs<=r.plan.primitive_pairs
    assert r.receipt.primitive_points<=r.plan.primitive_points
    assert r.receipt.confirmation_fock_error<1e-10 and r.receipt.confirmation_energy_error<1e-11


@pytest.mark.parametrize('species,spins',[('H2',1),('HeH',2),('H',2)])
def test_correction_spin_factors_and_no_extra_k_weight(species,spins):
    c,_,_=case(species,name=EMPTY_HYBRID if species=='H' else 'pbe0',spins=spins)
    strict=manta.run_periodic_global_hybrid_ks(**c)
    c['options'].exchange_q0='BvK-ewald';r=manta.run_periodic_global_hybrid_ks(**c)
    a=r.receipt.exact_exchange_fraction;xi=r.receipt.madelung
    np.testing.assert_allclose(r.density,strict.density,atol=4e-9,rtol=0)
    assert r.receipt.energy-strict.receipt.energy==pytest.approx(-a*xi*r.plan.electrons/2,abs=2e-10)
    for spin,occupied in enumerate((r.plan.n_alpha,) if spins==1 else (r.plan.n_alpha,r.plan.n_beta)):
        shifted=strict.orbital_energies[spin].copy();shifted[:,:occupied]-=a*xi
        np.testing.assert_allclose(r.orbital_energies[spin],shifted,atol=2e-10,rtol=0)
    assert r.receipt.source_identity!=strict.receipt.source_identity
    assert r.receipt.exchange.source_identity==strict.receipt.exchange.source_identity
    assert r.receipt.exchange_correction_identity and not strict.receipt.exchange_correction_identity


@pytest.mark.parametrize('species,spins',[('H2',1),('HeH',2)])
@pytest.mark.parametrize('correction',['strict-zero-mode','BvK-ewald'])
def test_native_fock_is_derivative_of_independent_total_energy(species,spins,correction):
    c,k,old=case(species,spins=spins,correction=correction);c['options'].ks.scf.maximum_iterations=1
    r=manta.run_periodic_global_hybrid_ks(**c);_,_,build,_=oracle(c,k,old)
    rng=np.random.default_rng(817);a=rng.normal(size=r.density.shape)+1j*rng.normal(size=r.density.shape)
    d=r.density;adjoint=a.conj().swapaxes(-1,-2);delta=a@d+d@adjoint
    # Congruence paths remain positive semidefinite at both signs.
    step=1e-5;eye=np.eye(c['basis'].nbasis)
    plus=(eye+step*a)@d@(eye+step*adjoint);minus=(eye-step*a)@d@(eye-step*adjoint)
    difference=(sum(build(plus)[1])-sum(build(minus)[1]))/(2*step)
    expected=np.einsum('skij,skji->',delta,r.fock).real/len(k)
    assert difference==pytest.approx(expected,abs=3e-8)


@pytest.mark.parametrize('species,spins',[('H2',1),('HeH',2),('H',2)])
def test_capped_evaluated_epochs_and_next_projector(species,spins):
    c,k,old=case(species,name=EMPTY_HYBRID if species=='H' else 'pbe0',spins=spins,correction='BvK-ewald')
    if species=='H2':
        shells=c['basis'].shells();shells[1].exponents=[.6]
        c['basis']=vq.BasisSet(c['system'].unit_cell_molecule(),shells,'asymmetric-hybrid-h2',False);old['basis']=c['basis']
    s,h,build,_=oracle(c,k,old)
    electrons=sum(a.Z for a in c['system'].unit_cell);beta=(electrons-c['system'].multiplicity+1)//2
    pop=(electrons//2,) if spins==1 else (electrons-beta,beta)
    d=occupy(np.repeat(h[None],spins,axis=0),s,pop,spins)
    c['options'].ks.scf.maximum_iterations=4;whole=manta.run_periodic_global_hybrid_ks(**c)
    for iteration in range(1,5):
        c['options'].ks.scf.maximum_iterations=iteration;r=manta.run_periodic_global_hybrid_ks(**c)
        assert not r.converged and not r.receipt.confirmed and r.receipt.fock_calls==iteration
        f,energy,_=build(d)
        np.testing.assert_allclose(r.density,d,atol=1e-10,rtol=0)
        np.testing.assert_allclose(r.fock,f,atol=1e-10,rtol=0)
        assert r.receipt.energy==pytest.approx(r.receipt.nuclear_energy+sum(energy),abs=1e-10)
        assert whole.scf_history[iteration-1].energy_per_cell==r.solver_snapshot.energy_per_cell
        next_d=occupy(f,s,pop,spins)
        coeff=r.coefficients
        projected=np.array([(2 if spins==1 else 1)*coeff[i,:,:,:n]@coeff[i,:,:,:n].conj().swapaxes(-1,-2) for i,n in enumerate(pop)])
        np.testing.assert_allclose(projected,next_d,atol=1e-10,rtol=0)
        if iteration==1:assert np.max(abs(d-next_d))>1e-4
        d=next_d


@pytest.mark.parametrize('spins',[1,2])
@pytest.mark.parametrize('field,planned',[('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),
    ('primitive_quartets','primitive_work'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_exact_joint_caps_and_one_short_before_nan_payload(field,planned,spins):
    c,_,_=case(spins=spins,correction='BvK-ewald',mesh=(1,1,1));c['options'].ks.scf.maximum_iterations=1
    c['budget'].other_live_bytes=123456;p=plan(c)
    setattr(c['budget'],field,getattr(p,planned));r=manta.run_periodic_global_hybrid_ks(**c)
    assert r.receipt.fock_calls==1 and r.solver_memory.total_numerical_bytes==p.modeled_peak_bytes
    setattr(c['budget'],field,getattr(p,planned)-1);c['points'][:]=np.nan;c['exchange_vectors'][:]=np.nan
    with pytest.raises((ValueError,RuntimeError),match='budget'):manta.run_periodic_global_hybrid_ks(**c)


@pytest.mark.parametrize('correction',['strict-zero-mode','BvK-ewald'])
def test_closed_shell_limit_layout_and_owned_exports(correction):
    c,_,_=case(correction=correction);r=manta.run_periodic_global_hybrid_ks(**c)
    c['functional']=vq.Functional('pbe0',spin=2);u=manta.run_periodic_global_hybrid_ks(**c)
    np.testing.assert_allclose(u.density.sum(axis=0),r.density[0],atol=4e-10,rtol=0)
    np.testing.assert_allclose(u.fock,np.repeat(r.fock,2,axis=0),atol=4e-10,rtol=0)
    np.testing.assert_allclose(components(u),components(r),atol=4e-11,rtol=0)
    assert u.receipt.source_identity!=r.receipt.source_identity
    for field in ('reciprocal_vectors','exchange_vectors','translations','xc_images','points'):c[field]=np.asfortranarray(c[field])
    c['options'].ks.grid_batch=7;c['options'].ks.reciprocal_batch=3
    other=manta.run_periodic_global_hybrid_ks(**c)
    np.testing.assert_allclose(other.fock,u.fock,atol=4e-10,rtol=0)
    assert other.receipt.source_identity==u.receipt.source_identity
    other.density[:]=0;other.fock[:]=0;other.scf_history.clear();other.receipt.exchange.energies.clear()
    assert np.any(other.density) and np.any(other.fock) and other.scf_history and other.receipt.exchange.energies


@pytest.mark.parametrize('spins',[1,2])
def test_independent_shell_image_gauge_preserves_hybrid(spins):
    c,k,_=case(spins=spins,correction='BvK-ewald');c['translations']=c['xc_images'].copy()
    r=manta.run_periodic_global_hybrid_ks(**c);assert r.converged
    shells=c['basis'].shells();displacement=c['system'].lattice[:,0]
    shells[0].origin=(np.array(shells[0].origin)+displacement).tolist()
    c['basis']=vq.BasisSet(vq.Molecule([vq.Atom(1,s.origin) for s in shells],0,1),shells,'shifted-hybrid-shell',True)
    shifted=manta.run_periodic_global_hybrid_ks(**c);assert shifted.converged
    phase=np.ones((len(k),2),complex);phase[:,0]=np.exp(1j*k@displacement)
    action=phase[:,:,None]*phase.conj()[:,None,:]
    for name in ('overlap','hcore','density','fock'):
        np.testing.assert_allclose(getattr(shifted,name),action*getattr(r,name),atol=4e-10,rtol=0)
    np.testing.assert_allclose(components(shifted),components(r),atol=5e-11,rtol=0)
    assert shifted.receipt.source_identity!=r.receipt.source_identity


def test_resolved_custom_lda_hybrid_and_refused_pure_api():
    name='manta-periodic-lda-hybrid-test'
    vq.define_functional(name,[('LDA_X',.63),('LDA_C_VWN',1.)],hf_exchange_fraction=.37)
    c,k,old=case(name=name,spins=2);r=manta.run_periodic_global_hybrid_ks(**c)
    s,h,build,_=oracle(c,k,old,fraction=.37)
    d=independent_scf(s,h,build,(1,1),2)
    np.testing.assert_allclose(r.density,d,atol=3e-9,rtol=0)
    np.testing.assert_allclose(r.fock,build(r.density)[0],atol=1e-11,rtol=0)
    assert r.converged and not r.plan.xc.gradients
    assert r.receipt.exact_exchange_fraction==pytest.approx(.37)
    with pytest.raises(ValueError,match='complete native pure'):
        manta.plan_periodic_ks(c['basis'],c['system'],c['mesh'],c['functional'],26,1,27,512,c['options'].ks,c['budget'])


@pytest.mark.parametrize('name',['pbe','scan','tpssh','b2plyp','wb97x-v','cam-b3lyp','hse06'])
def test_incomplete_or_non_global_functional_refused_before_payload(name):
    c,_,_=case(name=name);c['points'][:]=np.nan
    with pytest.raises(ValueError,match='global'):manta.run_periodic_global_hybrid_ks(**c)


@pytest.mark.parametrize('field,value',[('exchange_q0','automatic'),('madelung_alpha',0.),('madelung_tolerance',1.)])
def test_unsupported_correction_controls(field,value):
    c,_,_=case();setattr(c['options'],field,value);c['points'][:]=np.nan
    with pytest.raises(ValueError,match='exchange_q0|Madelung'):manta.run_periodic_global_hybrid_ks(**c)


@pytest.mark.parametrize('mutation',['first','order','last','dtype'])
def test_invalid_transfer_offsets(mutation):
    c,_,_=case();c['options'].ks.scf.maximum_iterations=1;c['points'][:]=np.nan
    if mutation=='first':c['transfer_offsets'][0]=1
    if mutation=='order':c['transfer_offsets'][2]=-1
    if mutation=='last':c['transfer_offsets'][-1]-=1
    if mutation=='dtype':c['transfer_offsets']=c['transfer_offsets'].astype(np.int32)
    with pytest.raises(ValueError,match='offsets'):manta.run_periodic_global_hybrid_ks(**c)


def test_metadata_admission_precedes_offset_traversal():
    c,_,_=case();c['budget'].memory_bytes=1;c['transfer_offsets'][:]=-1
    with pytest.raises((ValueError,RuntimeError),match='budget'):manta.run_periodic_global_hybrid_ks(**c)


def test_exchange_support_and_functional_identity():
    c,_,_=case();c['options'].ks.scf.maximum_iterations=1;r=manta.run_periodic_global_hybrid_ks(**c)
    # Remove an inversion-closed shell from every pair while retaining valid Q.
    q=[];offsets=[0]
    for lo,hi in zip(c['transfer_offsets'][:-1],c['transfer_offsets'][1:]):
        block=c['exchange_vectors'][lo:hi];block=block[np.linalg.norm(block,axis=1)<1.5]
        q.extend(block);offsets.append(len(q))
    other=dict(c,exchange_vectors=np.array(q),transfer_offsets=np.array(offsets,dtype=np.int64))
    changed=manta.run_periodic_global_hybrid_ks(**other)
    assert changed.receipt.exchange.source_identity!=r.receipt.exchange.source_identity
    assert changed.receipt.source_identity!=r.receipt.source_identity
    np.testing.assert_array_equal(changed.density,r.density)
    assert changed.receipt.final_density_identity==r.receipt.final_density_identity
    assert changed.receipt.final_fock_identity!=r.receipt.final_fock_identity
    changed=manta.run_periodic_global_hybrid_ks(**dict(c,functional=vq.Functional('b3lyp')))
    assert changed.receipt.source_identity!=r.receipt.source_identity


def test_checked_count_overflow():
    c,_,_=case()
    with pytest.raises((ValueError,RuntimeError,OverflowError),match='overflow|budget'):
        manta.plan_periodic_global_hybrid_ks(c['basis'],c['system'],c['mesh'],c['functional'],26,2**63,1,27,512,c['options'],c['budget'])


@pytest.mark.parametrize('name',['pbe0','b3lyp'])
def test_empty_spin_endpoint_conditioning_is_measured_separately(name):
    c,k,old=case('H',name,2,correction='BvK-ewald');r=manta.run_periodic_global_hybrid_ks(**c)
    assert r.converged
    _,_,build,_=oracle(c,k,old);f,energy,_=build(r.density)
    np.testing.assert_allclose(r.fock[0],f[0],atol=1e-11,rtol=0)
    np.testing.assert_allclose(components(r),energy,atol=1e-11,rtol=0)
    assert not np.any(r.density[1])
    b,g=bloch_values_gradients(dict(basis=c['basis'],lattice=c['system'].lattice,images=c['xc_images'],momenta=k,points=c['points']))
    d=r.density;weights=np.ones(len(k))/len(k);qw=c['quadrature_weights']
    rho=np.einsum('k,kgi,skij,kgj->sg',weights,b,d,b.conj()).real
    grad=(np.einsum('k,akgi,skij,kgj->sag',weights,g,d,b.conj())
          +np.einsum('k,kgi,skij,akgj->sag',weights,b,d,g.conj())).real
    aa=np.sum(grad[0]**2,axis=0);zero=np.zeros_like(aa)
    def minority(alpha):
        _,_,vb,_,vab,_=c['functional'].eval_polarised(alpha,zero,aa,zero,zero)
        flow=vab*grad[0]
        return (np.einsum('g,kgi,g,kgj->kij',qw,b.conj(),vb,b)
            +np.einsum('g,akgi,ag,kgj->kij',qw,g.conj(),flow,b)
            +np.einsum('g,kgi,ag,akgj->kij',qw,b.conj(),flow,g))
    base=minority(rho[0])
    sensitivity=max(np.max(abs(minority(rho[0]*(1+sign*np.finfo(float).eps))-base)) for sign in (-1,1))
    if name == 'pbe0':
        # PBE correlation uses stable spin fractions in the vendored backend.
        assert sensitivity <= 1e-11
        np.testing.assert_allclose(r.fock[1],f[1],atol=1e-11,rtol=0)
    else:
        # Other kernels retain their independently measured endpoint limit.
        assert np.max(abs(r.fock[1]-f[1]))<=max(1e-11,2*sensitivity)


@pytest.mark.parametrize('species,spins',[('H2',1),('HeH',2)])
def test_rank_truncation_retains_physical_fock_and_exact_joint_admission(species,spins):
    from scipy.linalg import eigh
    c,k,old=case(species,spins=spins,correction='BvK-ewald')
    shells=c['basis'].shells();shells.append(shells[0])
    c['basis']=vq.BasisSet(c['system'].unit_cell_molecule(),shells,'duplicate-global-hybrid-shell',True);old['basis']=c['basis']
    c['budget'].other_live_bytes=123456;p=plan(c)
    for cap,field in [('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),
        ('primitive_quartets','primitive_work'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')]:
        setattr(c['budget'],cap,getattr(p,field))
    r=manta.run_periodic_global_hybrid_ks(**c);assert r.converged
    s,h,build,_=oracle(c,k,old);rank=c['basis'].nbasis-1;x=[]
    for sp in s:
        values,vectors=eigh(sp);keep=values>max(c['options'].ks.scf.overlap_rank_absolute_floor,
            c['options'].ks.scf.overlap_rank_relative_floor*values[-1]);x.append(vectors[:,keep]/np.sqrt(values[keep]))
    x=np.array(x);pop=(r.plan.n_alpha,) if spins==1 else (r.plan.n_alpha,r.plan.n_beta)
    def project(f):
        reduced=x.conj().swapaxes(-1,-2)@f@x
        return x@occupy(reduced,np.repeat(np.eye(rank)[None],len(k),axis=0),pop,spins)@x.conj().swapaxes(-1,-2)
    d=project(np.repeat(h[None],spins,axis=0))
    for _ in range(250):
        updated=project(build(d)[0])
        if np.max(abs(updated-d))<1e-12:break
        d=updated
    else:raise AssertionError('Independent reduced-rank global hybrid did not converge')
    np.testing.assert_allclose(r.density,updated,atol=3e-9,rtol=0)
    np.testing.assert_allclose(r.fock,build(r.density)[0],atol=1e-11,rtol=0)
    assert r.coefficients.shape==(spins,len(k),rank+1,rank)
    assert r.solver_memory.total_numerical_bytes<=p.modeled_peak_bytes
