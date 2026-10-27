"""Finite physical KS: independent Gaussian/libxc contractions and SciPy SCF."""
import itertools

import numpy as np
import pytest
from scipy.linalg import eigh

import vibeqc as vq
from vibeqc import manta, _vibeqc_core as core
from tests.test_manta_periodic_rhf import _case as rhf_case, _components, _nuclear_reference
from tests.test_manta_periodic_exchange import _quadrature_bloch
from tests.test_manta_periodic_semilocal_xc import bloch_values_gradients


def case(species='H2', name='pbe', spins=1, mesh=(3,1,1), shift=(1,0,0), order=8):
    old,k=rhf_case(mesh,shift);system=old['system']
    if species!='H2':
        atoms=([vq.Atom(2,[-.7,0.,0.]),vq.Atom(1,[.7,0.,0.])]
               if species=='HeH' else [vq.Atom(1,[.1,-.15,.2])])
        system=vq.PeriodicSystem(3,system.lattice,atoms);system.multiplicity=2
        specs=[(0,1.1),(0,.3),(1,.6)] if species=='HeH' else [(0,.8),(0,.25)]
        shells=[]
        for atom,exponent in specs:
            shell=core.ShellInfo();shell.atom_index=atom;shell.origin=atoms[atom].xyz
            shell.l=0;shell.pure=True;shell.exponents=[exponent];shell.coefficients=[1.];shells.append(shell)
        old['basis']=vq.BasisSet(system.unit_cell_molecule(),shells,'manta-ks-test',False)
    o=manta.PeriodicKSOptions()
    for field in ('alpha','nuclear_cutoff_bohr','pair_cutoff_bohr','reciprocal_batch'):
        setattr(o,field,getattr(old['options'],field))
    o.grid_batch=64;o.scf.maximum_iterations=100
    x,w=np.polynomial.legendre.leggauss(order);indices=np.array(list(itertools.product(range(order),repeat=3)))
    points=np.ascontiguousarray(((x[indices]+1)/2)@system.lattice.T)
    qw=np.prod(w[indices]/2,axis=1)*abs(np.linalg.det(system.lattice))
    images=np.array(list(itertools.product(range(-1,2),repeat=3)),float)@system.lattice.T
    c=dict(basis=old['basis'],system=system,mesh=old['mesh'],functional=vq.Functional(name,spin=spins),
        reciprocal_vectors=old['reciprocal_vectors'],translations=old['translations'],xc_images=images,
        points=points,quadrature_weights=qw,options=o,budget=old['budget'])
    # Legacy oracle consumes Q only while preparing its unused HF response.
    old.update(basis=c['basis'],system=system,options=o)
    return c,k,old


def oracle(c,k,old):
    s,h,_=_components(old,k);f=c['functional'];kw=np.ones(len(k))/len(k)
    fj=np.stack([_quadrature_bloch(c['basis'],c['reciprocal_vectors'],c['translations'],-point,None) for point in k])
    kernel=4*np.pi/abs(np.linalg.det(c['system'].lattice))/np.sum(c['reciprocal_vectors']**2,axis=1)
    b,g=bloch_values_gradients(dict(basis=c['basis'],lattice=c['system'].lattice,images=c['xc_images'],momenta=k,points=c['points']))
    qw=c['quadrature_weights']
    def build(d):
        rho_g=np.einsum('k,skij,kgij->g',kw,d,fj)
        j=np.einsum('g,g,kgij->kij',kernel,rho_g,fj.conj())
        rho=np.einsum('k,kgi,skij,kgj->sg',kw,b,d,b.conj()).real
        grad=(np.einsum('k,akgi,skij,kgj->sag',kw,g,d,b.conj())
              +np.einsum('k,kgi,skij,akgj->sag',kw,b,d,g.conj())).real
        flow=np.zeros_like(grad)
        if len(d)==1:
            exc,vr,vs=f.eval_unpolarised(rho[0],np.sum(grad[0]**2,axis=0));vr=vr[None,:]
            if len(vs):flow=2*vs[None,None,:]*grad
        else:
            exc,va,vb,aa,ab,bb=f.eval_polarised(rho[0],rho[1],np.sum(grad[0]**2,axis=0),
                np.sum(grad[0]*grad[1],axis=0),np.sum(grad[1]**2,axis=0));vr=np.array([va,vb])
            if len(aa):flow=np.array([2*aa*grad[0]+ab*grad[1],2*bb*grad[1]+ab*grad[0]])
        v=np.einsum('g,kgi,sg,kgj->skij',qw,b.conj(),vr,b)
        v+=np.einsum('g,akgi,sag,kgj->skij',qw,g.conj(),flow,b)+np.einsum('g,kgi,sag,akgj->skij',qw,b.conj(),flow,g)
        e1=np.einsum('k,skij,kji->',kw,d,h).real
        ej=.5*np.einsum('k,skij,kji->',kw,d,j).real
        return h+j+v,(e1,ej,float(qw@exc)),rho@qw
    return s,h,build


def occupy(f,s,populations,spins):
    out=[]
    for spin,n in enumerate(populations):
        coeff=np.array([eigh(fp,sp)[1][:,:n] for fp,sp in zip(f[spin],s)])
        out.append((2 if spins==1 else 1)*(coeff@coeff.conj().swapaxes(-1,-2)))
    return np.array(out)


def independent_scf(s,h,build,populations,spins):
    d=occupy(np.repeat(h[None],spins,axis=0),s,populations,spins)
    for _ in range(250):
        updated=occupy(build(d)[0],s,populations,spins)
        if np.max(abs(updated-d))<1e-12:return updated
        d=updated
    raise AssertionError('Independent KS fixture did not converge')


@pytest.mark.parametrize('species,name,spins',[('H2','lda',1),('H2','pbe',1),('H2','pbe',2),('HeH','pbe',2),('H','blyp',2)])
def test_complete_physical_ks_matches_independent_scf(species,name,spins):
    c,k,old=case(species,name,spins);r=manta.run_periodic_ks(**c)
    assert r.converged and r.receipt.confirmed
    s,h,build=oracle(c,k,old)
    populations=(r.plan.n_alpha,) if spins==1 else (r.plan.n_alpha,r.plan.n_beta)
    d=independent_scf(s,h,build,populations,spins)
    np.testing.assert_allclose(r.overlap,s,atol=4e-12,rtol=0)
    np.testing.assert_allclose(r.hcore,h,atol=4e-12,rtol=0)
    np.testing.assert_allclose(r.density,d,atol=2e-9,rtol=0)
    f,energies,pop=build(r.density)
    np.testing.assert_allclose(r.fock,f,atol=8e-12,rtol=0)
    np.testing.assert_allclose([r.receipt.one_electron_energy,r.receipt.coulomb_energy,r.receipt.xc_energy],energies,atol=8e-12,rtol=0)
    np.testing.assert_allclose(r.receipt.xc.electrons_grid,pop,atol=8e-12,rtol=0)
    nuclear,count=_nuclear_reference(old)
    assert r.receipt.nuclear_energy==pytest.approx(sum(nuclear),abs=4e-12)
    assert r.receipt.nuclear_pair_images==count
    assert r.receipt.energy==pytest.approx(sum(nuclear)+sum(energies),abs=9e-12)
    wrong=sum(nuclear)+.5*np.einsum('skij,skji->',r.density,r.hcore+r.fock).real/len(k)
    assert abs(wrong-r.receipt.energy)>.05
    assert r.receipt.fock_calls==r.solver_snapshot.fock_calls+1
    assert r.receipt.confirmation_fock_error<1e-10 and r.receipt.confirmation_energy_error<1e-11
    assert len(r.scf_history)==r.solver_snapshot.evaluated_iteration
    assert r.receipt.shell_calls<=r.plan.shell_calls and r.receipt.primitive_pairs<=r.plan.primitive_pairs
    assert r.receipt.primitive_points<=r.plan.primitive_points
    expected=np.array([r.plan.electrons]) if spins==1 else np.array(populations)
    assert r.receipt.maximum_grid_population_error==pytest.approx(max(abs(pop-expected)),abs=8e-12)


@pytest.mark.parametrize('species,name,spins',[('H2','pbe',1),('HeH','pbe',2),('H','blyp',2)])
def test_capped_epochs_are_same_density_functional_snapshots(species,name,spins):
    c,k,old=case(species,name,spins)
    if species=='H2':
        shells=c['basis'].shells();shells[1].exponents=[.6]
        c['basis']=vq.BasisSet(c['system'].unit_cell_molecule(),shells,'asymmetric-h2-ks',False)
        old['basis']=c['basis']
    s,h,build=oracle(c,k,old)
    electrons=sum(a.Z for a in c['system'].unit_cell);beta=(electrons-c['system'].multiplicity+1)//2
    pop=(electrons//2,) if spins==1 else (electrons-beta,beta)
    d=occupy(np.repeat(h[None],spins,axis=0),s,pop,spins)
    c['options'].scf.maximum_iterations=4;whole=manta.run_periodic_ks(**c)
    for iteration in range(1,5):
        c['options'].scf.maximum_iterations=iteration;r=manta.run_periodic_ks(**c)
        assert not r.converged and not r.receipt.confirmed and r.receipt.fock_calls==iteration
        f,energies,_=build(d)
        np.testing.assert_allclose(r.density,d,atol=1e-10,rtol=0)
        np.testing.assert_allclose(r.fock,f,atol=1e-10,rtol=0)
        assert r.receipt.energy==pytest.approx(r.receipt.nuclear_energy+sum(energies),abs=1e-10)
        assert whole.scf_history[iteration-1].energy_per_cell==r.solver_snapshot.energy_per_cell
        d=occupy(f,s,pop,spins)


@pytest.mark.parametrize('spins',[1,2])
@pytest.mark.parametrize('field,planned',[('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),
    ('primitive_quartets','primitive_work'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_exact_aggregate_caps_and_one_short_before_poisoned_payload(field,planned,spins):
    c,_,_=case(spins=spins,mesh=(1,1,1));c['options'].scf.maximum_iterations=1
    p=manta.plan_periodic_ks(c['basis'],c['system'],c['mesh'],c['functional'],len(c['reciprocal_vectors']),
        len(c['translations']),len(c['xc_images']),len(c['points']),c['options'],c['budget'])
    setattr(c['budget'],field,getattr(p,planned));r=manta.run_periodic_ks(**c)
    assert r.receipt.fock_calls==1
    assert r.solver_memory.total_numerical_bytes==p.modeled_peak_bytes
    setattr(c['budget'],field,getattr(p,planned)-1);c['points'][:]=np.nan
    with pytest.raises((ValueError,RuntimeError),match='budget'):manta.run_periodic_ks(**c)


def test_restricted_limit_layout_history_and_owned_exports():
    c,_,_=case();r=manta.run_periodic_ks(**c);c['functional']=vq.Functional('pbe',spin=2);u=manta.run_periodic_ks(**c)
    np.testing.assert_allclose(u.density.sum(axis=0),r.density[0],atol=2e-10,rtol=0)
    np.testing.assert_allclose(u.fock,np.repeat(r.fock,2,axis=0),atol=2e-10,rtol=0)
    assert u.receipt.energy==pytest.approx(r.receipt.energy,abs=2e-11)
    assert u.receipt.source_identity!=r.receipt.source_identity
    for field in ('reciprocal_vectors','translations','xc_images','points'):
        c[field]=np.asfortranarray(c[field])
    c['options'].grid_batch=7;other=manta.run_periodic_ks(**c)
    np.testing.assert_allclose(other.fock,u.fock,atol=2e-10,rtol=0)
    assert other.receipt.source_identity==u.receipt.source_identity
    other.density[:]=0;other.fock[:]=0;other.scf_history.clear()
    assert np.any(other.density) and np.any(other.fock) and other.scf_history


@pytest.mark.parametrize('name',['pbe0','scan','b2plyp','wb97x-v'])
def test_incomplete_functional_refusal_precedes_payload(name):
    c,_,_=case();c['functional']=vq.Functional(name);c['points'][:]=np.nan
    with pytest.raises(ValueError,match='complete native pure LDA/GGA'):manta.run_periodic_ks(**c)


@pytest.mark.parametrize('field,value',[('dim',2),('charge',1),('multiplicity',2)])
def test_refused_physical_profiles(field,value):
    c,_,_=case();setattr(c['system'],field,value);c['points'][:]=np.nan
    with pytest.raises(ValueError,match='neutral|populations'):manta.run_periodic_ks(**c)


@pytest.mark.parametrize('spins',[1,2])
def test_independent_shell_image_gauge_with_physical_pair_images(spins):
    c,k,_=case(spins=spins)
    c['translations']=c['xc_images'].copy()
    r=manta.run_periodic_ks(**c)
    assert r.converged
    shells=c['basis'].shells();displacement=c['system'].lattice[:,0]
    shells[0].origin=(np.array(shells[0].origin)+displacement).tolist()
    shifted_atoms=[vq.Atom(1,s.origin) for s in shells]
    c['basis']=vq.BasisSet(vq.Molecule(shifted_atoms,0,1),shells,'shifted-ks-shell',True)
    shifted=manta.run_periodic_ks(**c)
    assert shifted.converged
    phase=np.ones((len(k),2),complex);phase[:,0]=np.exp(1j*k@displacement)
    action=phase[:,:,None]*phase.conj()[:,None,:]
    for name in ('overlap','hcore','fock','density'):
        np.testing.assert_allclose(getattr(shifted,name),action*getattr(r,name),atol=3e-10,rtol=0)
    assert shifted.receipt.energy==pytest.approx(r.receipt.energy,abs=4e-11)
    assert shifted.receipt.source_identity!=r.receipt.source_identity


def test_source_controls_and_owned_receipts():
    c,_,_=case();c['options'].scf.maximum_iterations=1;r=manta.run_periodic_ks(**c)
    for name in ('points','quadrature_weights'):
        changed=dict(c,**{name:c[name]*1.001});q=manta.run_periodic_ks(**changed)
        assert q.receipt.source_identity!=r.receipt.source_identity
        np.testing.assert_array_equal(q.density,r.density)
        assert q.receipt.final_density_identity==r.receipt.final_density_identity
        assert q.receipt.final_fock_identity!=r.receipt.final_fock_identity
    q=manta.run_periodic_ks(**dict(c,functional=vq.Functional('lda')))
    assert q.receipt.source_identity!=r.receipt.source_identity
    assert q.receipt.xc.component_ids!=r.receipt.xc.component_ids
    old=r.receipt.xc.electrons_grid;r.receipt.xc.electrons_grid.clear()
    assert r.receipt.xc.electrons_grid==old


@pytest.mark.parametrize('field,value,match',[
    ('points',np.full((1,3),np.nan),'extent'),
    ('quadrature_weights',-np.ones(512),'weight'),
    ('xc_images',np.array([[.123,0.,0.]]),'lattice'),
    ('reciprocal_vectors',np.zeros((1,3)),'G=0'),
])
def test_invalid_numerical_sources(field,value,match):
    c,_,_=case();c['options'].scf.maximum_iterations=1;c[field]=value
    with pytest.raises(ValueError,match=match):manta.run_periodic_ks(**c)


def test_solver_failure_status_never_claims_physical_confirmation():
    c,_,_=case();c['options'].scf.minimum_band_gap_hartree=100.
    r=manta.run_periodic_ks(**c)
    assert not r.converged and not r.receipt.confirmed
    assert not r.solver_snapshot.converged
    assert r.receipt.fock_calls==r.solver_snapshot.fock_calls


def test_snapshot_lifetimes_are_independent_of_result():
    import gc
    c,_,_=case('HeH',spins=2);r=manta.run_periodic_ks(**c)
    snapshot=r.solver_snapshot;memory=r.solver_memory;history=r.scf_history
    energy=r.receipt.energy;iterations=snapshot.evaluated_iteration
    del r;gc.collect()
    assert snapshot.energy_per_cell==pytest.approx(energy,abs=2e-11)
    assert len(history)==iterations and memory.n_alpha==2
    assert history[-1].energy_per_cell==snapshot.energy_per_cell


@pytest.mark.parametrize('species,spins',[('H2',1),('HeH',2)])
def test_rank_truncation_joint_exact_caps_and_external_live_memory(species,spins):
    c,k,old=case(species,spins=spins)
    shells=c['basis'].shells();shells.append(shells[0])
    c['basis']=vq.BasisSet(c['system'].unit_cell_molecule(),shells,'duplicate-ks-shell',True);old['basis']=c['basis']
    c['budget'].other_live_bytes=123456
    p=manta.plan_periodic_ks(c['basis'],c['system'],c['mesh'],c['functional'],len(c['reciprocal_vectors']),
        len(c['translations']),len(c['xc_images']),len(c['points']),c['options'],c['budget'])
    for cap,field in [('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),
                      ('primitive_quartets','primitive_work'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')]:
        setattr(c['budget'],cap,getattr(p,field))
    r=manta.run_periodic_ks(**c);assert r.converged
    s,h,build=oracle(c,k,old);rank=c['basis'].nbasis-1
    assert r.coefficients.shape==(spins,len(k),rank+1,rank)
    assert r.orbital_energies.shape==(spins,len(k),rank)
    assert r.solver_memory.total_numerical_bytes<=p.modeled_peak_bytes
    transforms=[]
    for sp in s:
        values,vectors=eigh(sp);keep=values>max(c['options'].scf.overlap_rank_absolute_floor,
                                               c['options'].scf.overlap_rank_relative_floor*values[-1])
        transforms.append(vectors[:,keep]/np.sqrt(values[keep]))
    x=np.array(transforms);pop=(r.plan.n_alpha,) if spins==1 else (r.plan.n_alpha,r.plan.n_beta)
    def project_occupy(f):
        reduced=x.conj().swapaxes(-1,-2)@f@x
        return x@occupy(reduced,np.repeat(np.eye(rank)[None],len(k),axis=0),pop,spins)@x.conj().swapaxes(-1,-2)
    d=project_occupy(np.repeat(h[None],spins,axis=0))
    for _ in range(250):
        updated=project_occupy(build(d)[0])
        if np.max(abs(updated-d))<1e-12:break
        d=updated
    else:raise AssertionError('Independent reduced-rank KS did not converge')
    np.testing.assert_allclose(r.density,updated,atol=2e-9,rtol=0)
    np.testing.assert_allclose(r.fock,build(r.density)[0],atol=1e-11,rtol=0)


def test_failed_terminal_confirmation_preserves_numerical_verdict():
    c,_,_=case('HeH',spins=2)
    c['options'].confirmation_fock_tolerance=1e-30
    c['options'].confirmation_energy_tolerance=1e-30
    r=manta.run_periodic_ks(**c)
    assert r.solver_snapshot.converged and not r.converged and not r.receipt.confirmed
    assert r.receipt.fock_calls==r.solver_snapshot.fock_calls+1
    assert max(r.receipt.confirmation_fock_error,r.receipt.confirmation_energy_error)>1e-30
