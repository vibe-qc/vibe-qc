"""Complete finite range-separated KS: independent contractions and SCF."""
import gc

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta
from vibeqc.bipole_fock_ewald import probe_charge_madelung_supercell
from tests.test_manta_periodic_ks import case as pure_case, oracle as semilocal_oracle, occupy, independent_scf
from tests.test_manta_periodic_exchange import _quadrature_bloch
from tests.test_manta_periodic_global_hybrid_ks import components
from tests.test_manta_periodic_global_hybrid_domains import CAPS
from tests.test_manta_periodic_rhf import _nuclear_reference

EMPTY_RANGE='manta-periodic-lc-blyp-test'
MIXED_RANGE='manta-periodic-mixed-ranges-test'
vq.define_functional(EMPTY_RANGE,[('HYB_GGA_XC_LC_BLYP',1.)])
vq.define_functional(MIXED_RANGE,[('HYB_GGA_XC_HSE06',.4),('HYB_GGA_XC_CAM_B3LYP',.6)])
# HSE06: Krukau et al., JCP 125,224106 (2006); CAM-B3LYP: Yanai et al.,
# CPL 393,51 (2004). Coefficients below are independent reference data, not
# extracted from the implementation's resolved method plan.
TERMS={
    'hse06':[('erfc',.25,.11)],
    'cam-b3lyp':[('coulomb',.65,0.),('erfc',-.46,.33)],
    'lc-wpbe':[('coulomb',1.,0.),('erfc',-1.,.4)],
    'wb97x':[('coulomb',1.,0.),('erfc',-.842294,.3)],
    EMPTY_RANGE:[('coulomb',1.,0.),('erfc',-1.,.33)],
    MIXED_RANGE:[('coulomb',.39,0.),('erfc',.1,.11),('erfc',-.276,.33)],
}


def case(species='H2',name='cam-b3lyp',spins=1,correction='BvK-ewald',**kwargs):
    c,k,old=pure_case(species,name,spins,**kwargs)
    o=manta.PeriodicRangeSeparatedKSOptions();o.ks=c['options'];o.exchange_q0=correction
    c.update(options=o,exchange_vectors=old['exchange_vectors'],transfer_offsets=old['transfer_offsets'])
    c['budget'].memory_bytes=2**30;c['budget'].shell_calls=10**9
    c['budget'].scalar_terms=10**17;c['budget'].kernel_terms=10**18
    return c,k,old


def plan(c):
    return manta.plan_periodic_range_separated_ks(c['basis'],c['system'],c['mesh'],c['functional'],
        len(c['reciprocal_vectors']),int(max(np.diff(c['transfer_offsets']))),len(c['translations']),
        len(c['xc_images']),len(c['points']),c['options'],c['budget'])


def oracle(c,k,old,terms=None,include_zero=True):
    s,h,semilocal=semilocal_oracle(c,k,old)
    terms=TERMS[c['functional'].name.lower()] if terms is None else terms
    weights=np.ones(len(k))/len(k);volume=abs(np.linalg.det(c['system'].lattice))
    transfers=[]
    for kernel,coefficient,omega in terms:
        component=[]
        for target,kt in enumerate(k):
            for source in range(len(k)):
                lo,hi=c['transfer_offsets'][target*len(k)+source:target*len(k)+source+2]
                q=c['exchange_vectors'][lo:hi]
                if kernel=='erfc' and target==source and include_zero:q=np.vstack((q,np.zeros((1,3))))
                q2=np.sum(q*q,axis=1);v=np.empty(len(q))
                nonzero=q2!=0
                if kernel=='coulomb':v[:]=4*np.pi/q2
                else:
                    v[nonzero]=-4*np.pi*np.expm1(-q2[nonzero]/(4*omega**2))/q2[nonzero]
                    v[~nonzero]=np.pi/omega**2
                b=_quadrature_bloch(c['basis'],q,c['translations'],kt,None)
                component.append((b,weights[source]*v/volume))
        transfers.append((coefficient,component))
    long=sum(t[1] for t in terms if t[0]=='coulomb');o=c['options']
    xi=0. if long==0 or o.exchange_q0=='strict-zero-mode' else probe_charge_madelung_supercell(
        c['system'],tuple(c['mesh'].mesh),alpha=o.madelung_alpha,precision=1e-14)
    factor=2 if c['functional'].spin==1 else 1
    def build(d):
        f,e,pop=semilocal(d);energies=[]
        for coefficient,component in transfers:
            exchange=np.zeros_like(d)
            for spin in range(len(d)):
                for target in range(len(k)):
                    for source in range(len(k)):
                        b,v=component[target*len(k)+source]
                        exchange[spin,target]+=np.einsum('g,gai,ab,gbj->ij',v,b.conj(),d[spin,source],b)
            f-=coefficient/factor*exchange
            energies.append(-.5*coefficient/factor*np.einsum('k,skij,skji->',weights,d,exchange).real)
        correction=-long/factor*xi*(s@d@s)
        ec=.5*np.einsum('k,skij,skji->',weights,d,correction).real
        return f+correction,(*e,sum(energies),ec),pop
    return s,h,build,xi


@pytest.mark.parametrize('species,name,spins',[
    ('H2','hse06',1),('H2','cam-b3lyp',1),('H2','lc-wpbe',1),('H2','wb97x',1),
    ('H2','hse06',2),('HeH','hse06',2),('HeH','cam-b3lyp',2),('HeH','lc-wpbe',2),
    ('HeH',MIXED_RANGE,2),('H',EMPTY_RANGE,2)])
@pytest.mark.parametrize('correction',['strict-zero-mode','BvK-ewald'])
def test_complete_range_separated_scf_and_component_receipts(species,name,spins,correction):
    c,k,old=case(species,name,spins,correction);r=manta.run_periodic_range_separated_ks(**c)
    assert r.converged and r.receipt.confirmed and r.plan.range_separated and r.receipt.range_separated
    s,h,build,xi=oracle(c,k,old);pop=(r.plan.n_alpha,) if spins==1 else (r.plan.n_alpha,r.plan.n_beta)
    d=independent_scf(s,h,build,pop,spins);f,e,n=build(r.density)
    np.testing.assert_allclose(r.density,d,atol=3e-9,rtol=0)
    np.testing.assert_allclose(r.fock,f,atol=2e-11,rtol=0)
    np.testing.assert_allclose(components(r),e,atol=2e-11,rtol=0)
    np.testing.assert_allclose(r.overlap,s,atol=4e-12,rtol=0)
    np.testing.assert_allclose(r.hcore,h,atol=4e-12,rtol=0)
    np.testing.assert_allclose(r.receipt.xc.electrons_grid,n,atol=2e-11,rtol=0)
    nuclear,_=_nuclear_reference(old)
    assert r.receipt.energy==pytest.approx(sum(nuclear)+sum(e),abs=2e-11)
    assert r.receipt.madelung==pytest.approx(xi,abs=4e-12)
    assert r.receipt.exchange is None and r.plan.exchange is None
    assert r.receipt.xc.semilocal_component_only and r.receipt.exact_exchange_fraction==0.
    actual=r.receipt.exchange_components;expected=TERMS[name]
    assert len(actual)==len(expected)
    assert r.receipt.xc.required_exchange_components==[(a.kernel,a.coefficient,a.omega) for a in actual]
    for a,p,(kernel,coefficient,omega) in zip(actual,r.plan.exchange_components,expected):
        assert a.kernel==p.kernel==a.exchange.kernel==kernel
        assert a.coefficient==pytest.approx(coefficient) and a.omega==pytest.approx(omega)
        assert a.energy==pytest.approx(coefficient/(2 if spins==1 else 1)*sum(a.exchange.energies),abs=2e-14)
        assert a.exchange.zero_vectors==(len(k) if kernel=='erfc' else 0)
        assert a.exchange.transfer_vectors==len(c['exchange_vectors'])+(len(k) if kernel=='erfc' else 0)
    long=sum(x[1] for x in expected if x[0]=='coulomb')
    assert r.receipt.long_range_exchange_fraction==pytest.approx(long)
    assert r.receipt.short_range_exchange_fraction==pytest.approx(sum(x[1] for x in expected))
    assert r.receipt.exchange_q0==(correction if long else 'not-applicable')
    assert r.receipt.exchange_q0_requested==correction
    assert r.receipt.uncorrected_exchange_energy==pytest.approx(sum(a.energy for a in actual),abs=2e-14)
    assert r.receipt.fock_calls==r.solver_snapshot.fock_calls+1
    assert r.receipt.shell_calls<=r.plan.shell_calls and r.receipt.primitive_pairs<=r.plan.primitive_pairs


@pytest.mark.parametrize('spins',[1,2])
def test_screened_zero_mode_is_finite_weighted_and_has_no_madelung(spins):
    c,k,old=case('H2','hse06',spins);r=manta.run_periodic_range_separated_ks(**c)
    c['options'].exchange_q0='strict-zero-mode';c['options'].madelung_alpha=.4
    strict=manta.run_periodic_range_separated_ks(**c)
    for name in ('density','fock'):np.testing.assert_array_equal(getattr(r,name),getattr(strict,name))
    assert r.receipt.source_identity==strict.receipt.source_identity
    assert r.receipt.madelung==r.receipt.exchange_correction_energy==0.
    assert not r.receipt.exchange_correction_identity and r.plan.madelung.kernel_terms==0
    assert r.plan.exchange_correction_kernel_terms_per_call==0
    _,_,without,_=oracle(c,k,old,include_zero=False)
    missing=-.25/(2 if spins==1 else 1)*np.pi/.11**2/abs(np.linalg.det(c['system'].lattice))/len(k)*(r.overlap@r.density@r.overlap)
    f,e,_=without(r.density)
    np.testing.assert_allclose(r.fock-f,missing,atol=2e-11,rtol=0)
    de=.5*np.einsum('skij,skji->',r.density,missing).real/len(k)
    assert sum(components(r))-sum(e)==pytest.approx(de,abs=2e-11)


@pytest.mark.parametrize('species,spins',[('H2',1),('HeH',2),('H',2)])
def test_long_range_coefficient_controls_correction_without_extra_k_weight(species,spins):
    name=EMPTY_RANGE if species=='H' else 'cam-b3lyp'
    c,_,_=case(species,name,spins,correction='strict-zero-mode');strict=manta.run_periodic_range_separated_ks(**c)
    c['options'].exchange_q0='BvK-ewald';r=manta.run_periodic_range_separated_ks(**c)
    long=1. if species=='H' else .65
    np.testing.assert_allclose(r.density,strict.density,atol=4e-9,rtol=0)
    assert r.receipt.energy-strict.receipt.energy==pytest.approx(-long*r.receipt.madelung*r.plan.electrons/2,abs=3e-10)
    for spin,n in enumerate((r.plan.n_alpha,) if spins==1 else (r.plan.n_alpha,r.plan.n_beta)):
        shifted=strict.orbital_energies[spin].copy();shifted[:,:n]-=long*r.receipt.madelung
        np.testing.assert_allclose(r.orbital_energies[spin],shifted,atol=3e-10,rtol=0)
    assert r.receipt.source_identity!=strict.receipt.source_identity
    assert [a.exchange.source_identity for a in r.receipt.exchange_components]==[a.exchange.source_identity for a in strict.receipt.exchange_components]


@pytest.mark.parametrize('name',['hse06','cam-b3lyp',MIXED_RANGE])
@pytest.mark.parametrize('spins',[1,2])
def test_energy_derivative_at_complex_nonstationary_density(name,spins):
    c,k,old=case('H2' if spins==1 else 'HeH',name,spins);c['options'].ks.scf.maximum_iterations=1
    r=manta.run_periodic_range_separated_ks(**c);_,_,build,_=oracle(c,k,old)
    rng=np.random.default_rng(281);a=rng.normal(size=r.density.shape)+1j*rng.normal(size=r.density.shape)
    d=r.density;adjoint=a.conj().swapaxes(-1,-2);delta=a@d+d@adjoint;step=1e-5;eye=np.eye(c['basis'].nbasis)
    plus=(eye+step*a)@d@(eye+step*adjoint);minus=(eye-step*a)@d@(eye-step*adjoint)
    derivative=(sum(build(plus)[1])-sum(build(minus)[1]))/(2*step)
    expected=np.einsum('skij,skji->',delta,r.fock).real/len(k)
    assert derivative==pytest.approx(expected,abs=5e-8)


@pytest.mark.parametrize('name',['hse06','cam-b3lyp'])
@pytest.mark.parametrize('spins',[1,2])
def test_capped_epochs_and_independent_next_projectors(name,spins):
    c,k,old=case('H2' if spins==1 else 'HeH',name,spins)
    if spins==1:
        shells=c['basis'].shells();shells[1].exponents=[.6]
        c['basis']=old['basis']=vq.BasisSet(c['system'].unit_cell_molecule(),shells,'asymmetric-rsh-h2',False)
    s,h,build,_=oracle(c,k,old);pop=(1,) if spins==1 else (2,1)
    d=occupy(np.repeat(h[None],spins,axis=0),s,pop,spins)
    c['options'].ks.scf.maximum_iterations=4;whole=manta.run_periodic_range_separated_ks(**c)
    for iteration in range(1,5):
        c['options'].ks.scf.maximum_iterations=iteration;r=manta.run_periodic_range_separated_ks(**c)
        assert not r.converged and r.receipt.fock_calls==iteration
        f,e,_=build(d)
        np.testing.assert_allclose(r.density,d,atol=2e-10,rtol=0)
        np.testing.assert_allclose(r.fock,f,atol=2e-10,rtol=0)
        np.testing.assert_allclose(components(r),e,atol=2e-10,rtol=0)
        assert whole.scf_history[iteration-1].energy_per_cell==r.solver_snapshot.energy_per_cell
        next_d=occupy(f,s,pop,spins);coeff=r.coefficients
        projected=np.array([(2 if spins==1 else 1)*coeff[i,:,:,:n]@coeff[i,:,:,:n].conj().swapaxes(-1,-2) for i,n in enumerate(pop)])
        np.testing.assert_allclose(projected,next_d,atol=2e-10,rtol=0)
        if iteration==1:assert np.max(abs(d-next_d))>1e-4
        d=next_d


@pytest.mark.parametrize('field,quantity',CAPS)
@pytest.mark.parametrize('name',['hse06',MIXED_RANGE])
def test_all_joint_caps_and_one_short_before_invalid_arrays(field,quantity,name):
    c,_,_=case('HeH',name,2);c['options'].ks.scf.maximum_iterations=1;c['budget'].other_live_bytes=123456
    p=plan(c)
    for cap,attr in CAPS:setattr(c['budget'],cap,getattr(p,attr))
    r=manta.run_periodic_range_separated_ks(**c)
    assert r.receipt.fock_calls==1 and r.solver_memory.total_numerical_bytes==p.modeled_peak_bytes
    setattr(c['budget'],field,getattr(p,quantity)-1);c['points'][:]=np.nan;c['exchange_vectors'][:]=np.nan
    with pytest.raises((ValueError,RuntimeError),match='budget'):manta.run_periodic_range_separated_ks(**c)


@pytest.mark.parametrize('name',['hse06','cam-b3lyp'])
@pytest.mark.parametrize('spins',[1,2])
def test_image_gauge_component_frames_and_ownership(name,spins):
    c,k,_=case(name=name,spins=spins);c['translations']=c['xc_images'].copy()
    r=manta.run_periodic_range_separated_ks(**c);shells=c['basis'].shells();displacement=c['system'].lattice[:,0]
    shells[0].origin=(np.array(shells[0].origin)+displacement).tolist()
    c['basis']=vq.BasisSet(vq.Molecule([vq.Atom(1,s.origin) for s in shells],0,1),shells,'shifted-rsh-shell',True)
    shifted=manta.run_periodic_range_separated_ks(**c);assert r.converged and shifted.converged
    phase=np.ones((len(k),2),complex);phase[:,0]=np.exp(1j*k@displacement);action=phase[:,:,None]*phase.conj()[:,None,:]
    for field in ('overlap','hcore','density','fock'):
        np.testing.assert_allclose(getattr(shifted,field),action*getattr(r,field),atol=4e-10,rtol=0)
    np.testing.assert_allclose(components(shifted),components(r),atol=5e-11,rtol=0)
    assert shifted.receipt.source_identity!=r.receipt.source_identity
    for before,after in zip(r.receipt.exchange_components,shifted.receipt.exchange_components):
        assert before.exchange.source_identity!=after.exchange.source_identity
    r.density[:]=999;r.fock[:]=999;r.receipt.exchange_components.clear();r.receipt.xc.required_exchange_components.clear()
    assert r.receipt.exchange_components and r.receipt.xc.required_exchange_components
    leaf=r.receipt.exchange_components[-1];old=leaf.exchange.energies.copy();leaf.exchange.energies.clear()
    assert leaf.exchange.energies==old
    del r;gc.collect();assert leaf.exchange.energies==old


@pytest.mark.parametrize('name',['pbe','pbe0','scan','tpssh','b2plyp','wb97x-v','HYB_GGA_XC_CAMY_B3LYP'])
def test_incomplete_components_and_wrong_route_refuse_before_payload(name):
    if name.startswith('HYB_'):
        vq.define_functional('manta-rsh-yukawa-test',[(name,1.)]);name='manta-rsh-yukawa-test'
    c,_,_=case(name=name);c['points'][:]=np.nan
    with pytest.raises(ValueError,match='range-separated'):manta.run_periodic_range_separated_ks(**c)


@pytest.mark.parametrize('mutation',['zero','offsets','dtype','overflow'])
def test_support_metadata_and_singular_zero_refusals(mutation):
    c,_,_=case();c['points'][:]=np.nan
    if mutation=='zero':
        c['exchange_vectors'][0]=0
        with pytest.raises(ValueError,match='Q=0'):manta.run_periodic_range_separated_ks(**c)
    elif mutation=='overflow':
        with pytest.raises((ValueError,RuntimeError,OverflowError),match='overflow|budget'):
            manta.plan_periodic_range_separated_ks(c['basis'],c['system'],c['mesh'],c['functional'],26,2**63,1,27,512,c['options'],c['budget'])
    else:
        if mutation=='offsets':c['transfer_offsets'][0]=1
        else:c['transfer_offsets']=c['transfer_offsets'].astype(np.int32)
        with pytest.raises(ValueError,match='offsets'):manta.run_periodic_range_separated_ks(**c)


@pytest.mark.parametrize('spins',[1,2])
def test_gamma_multiple_ranges_and_batch_invariant_source(spins):
    c,k,old=case(name=MIXED_RANGE,spins=spins,mesh=(1,1,1),shift=(0,0,0));r=manta.run_periodic_range_separated_ks(**c)
    s,h,build,_=oracle(c,k,old);pop=(1,) if spins==1 else (1,1)
    np.testing.assert_allclose(r.density,independent_scf(s,h,build,pop,spins),atol=3e-9,rtol=0)
    for name in ('reciprocal_vectors','exchange_vectors','translations','xc_images','points'):c[name]=np.asfortranarray(c[name])
    c['options'].ks.grid_batch=7;c['options'].ks.reciprocal_batch=3
    other=manta.run_periodic_range_separated_ks(**c)
    np.testing.assert_allclose(other.fock,r.fock,atol=4e-10,rtol=0)
    assert r.receipt.source_identity==other.receipt.source_identity


@pytest.mark.parametrize('name',['hse06','cam-b3lyp'])
def test_closed_shell_limit_is_spin_consistent(name):
    c,_,_=case(name=name);r=manta.run_periodic_range_separated_ks(**c)
    c['functional']=vq.Functional(name,spin=2);u=manta.run_periodic_range_separated_ks(**c)
    np.testing.assert_allclose(u.density.sum(axis=0),r.density[0],atol=4e-10,rtol=0)
    np.testing.assert_allclose(u.fock,np.repeat(r.fock,2,axis=0),atol=4e-10,rtol=0)
    np.testing.assert_allclose(components(u),components(r),atol=4e-11,rtol=0)
    assert u.receipt.source_identity!=r.receipt.source_identity


def test_cancelled_exchange_still_admits_transfer_source_traversal():
    name='manta-rsh-zero-exchange-work-test'
    vq.define_functional(name,[('HYB_GGA_XC_HSE06',1.),('HYB_GGA_XC_HSE06',-1.),
        ('GGA_X_B88',.8),('GGA_C_LYP',1.)])
    c,_,_=case(name=name);c['options'].ks.scf.maximum_iterations=1
    narrow=plan(c);assert not narrow.exchange_components
    c['budget'].scalar_terms=narrow.scalar_terms
    with pytest.raises((ValueError,RuntimeError),match='scalar-work budget'):
        manta.plan_periodic_range_separated_ks(c['basis'],c['system'],c['mesh'],c['functional'],
            len(c['reciprocal_vectors']),100000,len(c['translations']),len(c['xc_images']),
            len(c['points']),c['options'],c['budget'])


@pytest.mark.parametrize('species,spins',[('H2',1),('HeH',2)])
@pytest.mark.parametrize('name',['hse06',MIXED_RANGE])
def test_rank_truncation_preserves_physical_source_and_budget(species,spins,name):
    from scipy.linalg import eigh
    c,k,old=case(species,name,spins);shells=c['basis'].shells();shells.append(shells[0])
    c['basis']=old['basis']=vq.BasisSet(c['system'].unit_cell_molecule(),shells,'duplicate-rsh-shell',True)
    c['budget'].other_live_bytes=123456;p=plan(c)
    for cap,field in CAPS:setattr(c['budget'],cap,getattr(p,field))
    r=manta.run_periodic_range_separated_ks(**c);assert r.converged
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
    else:raise AssertionError('Independent reduced-rank RSH did not converge')
    np.testing.assert_allclose(r.density,updated,atol=3e-9,rtol=0)
    np.testing.assert_allclose(r.fock,build(r.density)[0],atol=2e-11,rtol=0)
    assert r.coefficients.shape==(spins,len(k),rank+1,rank)
    assert r.solver_memory.total_numerical_bytes<=p.modeled_peak_bytes


@pytest.mark.parametrize('fraction',[0.,.2])
def test_cancelled_screened_components_are_not_replaced_by_aggregate_cam(fraction):
    name=f'manta-rsh-cancelled-{fraction}'
    vq.define_functional(name,[('HYB_GGA_XC_HSE06',1.),('HYB_GGA_XC_HSE06',-1.),
                              ('GGA_X_B88',.8),('GGA_C_LYP',1.)],hf_exchange_fraction=fraction)
    c,k,old=case(name=name);r=manta.run_periodic_range_separated_ks(**c)
    expected=[('coulomb',fraction,0.)] if fraction else []
    s,h,build,_=oracle(c,k,old,terms=expected)
    np.testing.assert_allclose(r.density,independent_scf(s,h,build,(1,),1),atol=3e-9,rtol=0)
    np.testing.assert_allclose(r.fock,build(r.density)[0],atol=2e-11,rtol=0)
    assert r.converged and len(r.receipt.exchange_components)==len(expected)
    assert r.receipt.long_range_exchange_fraction==fraction
    if not fraction:assert r.receipt.exchange_energy==0. and r.receipt.exchange_q0=='not-applicable'
