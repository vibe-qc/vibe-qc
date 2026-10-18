"""Pure-KS cutoff sources: independent supports, SCF, gauges and admission."""
import itertools

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta
from tests.test_manta_periodic_ks import case,oracle,independent_scf
from tests.test_manta_periodic_domains import _labels


def arguments(species='H2',spins=1,mesh=(3,1,1),shift=(1,0,0)):
    c,k,old=case(species,name='blyp' if species=='H' else 'pbe',spins=spins,mesh=mesh,shift=shift)
    c['budget'].memory_bytes=2**30;c['budget'].shell_calls=10**9
    c['budget'].scalar_terms=10**17;c['budget'].kernel_terms=10**18
    args={name:c[name] for name in ('basis','system','mesh','functional','options','budget')}
    args.update(reciprocal_energy_cutoff=2.,xc_image_cutoff_bohr=11.,grid_points=len(c['points']))
    return args,c,k,old


def run_args(args,c):
    result=dict(args);result.pop('grid_points')
    result.update(points=c['points'],quadrature_weights=c['quadrature_weights']);return result


@pytest.mark.parametrize('shape,shift',[((1,1,1),(0,0,0)),((3,1,1),(1,0,0)),((3,2,1),(1,1,0))])
def test_independent_integer_spheres_and_mesh_independent_domains(shape,shift):
    args,_,_,_=arguments(mesh=shape,shift=shift)
    lattice=np.array([[6.,3.5,.8],[0.,4.2,1.2],[0.,0.,6.4]])
    args['system']=vq.PeriodicSystem(3,lattice,args['system'].unit_cell)
    d=manta.build_periodic_ks_domains(**args);b=2*np.pi*np.linalg.inv(lattice).T
    labels=np.array(list(itertools.product(range(-8,9),repeat=3)));g=labels@b.T;images=labels@lattice.T
    expected_g=labels[(np.sum(g*g,axis=1)<=2*args['reciprocal_energy_cutoff']) & np.any(labels,axis=1)]
    assert _labels(d.reciprocal_vectors,np.linalg.inv(b))=={tuple(n) for n in expected_g}
    for field,radius in [('translations',d.plan.translation_radius_bohr),('xc_images',args['xc_image_cutoff_bohr'])]:
        selected=_labels(getattr(d,field),np.linalg.inv(lattice))
        assert selected=={tuple(n) for n in labels[np.sum(images*images,axis=1)<=radius**2]}
        assert {tuple(-np.array(n)) for n in selected}==selected
    assert d.receipt.conjugacy_vectors==len(d.reciprocal_vectors)
    assert d.receipt.hartree_vectors<=d.plan.hartree_candidates
    assert d.receipt.translations<=d.plan.translation_candidates
    assert d.receipt.xc_images<=d.plan.xc_image_candidates
    other=manta.build_periodic_ks_domains(**dict(args,mesh=manta.RegularKMesh((1,1,1),(0,0,0))))
    assert other.receipt.identity==d.receipt.identity
    assert other.plan.domain_live_bytes==d.plan.domain_live_bytes
    assert other.plan.candidate_visits==d.plan.candidate_visits


@pytest.mark.parametrize('species,spins',[('H2',1),('H2',2),('HeH',2),('H',2)])
def test_generated_source_matches_explicit_and_independent_scf(species,spins):
    args,c,k,old=arguments(species,spins);d=manta.build_periodic_ks_domains(**args)
    assert len(d.translations)==1
    for name in ('reciprocal_vectors','translations','xc_images'):c[name]=getattr(d,name)
    old.update(reciprocal_vectors=d.reciprocal_vectors,translations=d.translations)
    r=manta.run_periodic_ks_from_cutoffs(**run_args(args,c));explicit=manta.run_periodic_ks(**c)
    assert r.converged and r.receipt.confirmed
    np.testing.assert_array_equal(r.density,explicit.density)
    np.testing.assert_array_equal(r.fock,explicit.fock)
    assert r.receipt.source_identity==explicit.receipt.source_identity
    assert r.receipt.domain_identity==d.receipt.identity
    assert r.receipt.domain_xc_images==len(d.xc_images)
    s,h,build=oracle(c,k,old);pop=(r.plan.n_alpha,) if spins==1 else (r.plan.n_alpha,r.plan.n_beta)
    density=independent_scf(s,h,build,pop,spins)
    np.testing.assert_allclose(r.density,density,atol=2e-9,rtol=0)
    np.testing.assert_allclose(r.fock,build(r.density)[0],atol=1e-11,rtol=0)
    assert r.receipt.energy==pytest.approx(r.receipt.nuclear_energy+sum(build(r.density)[1]),abs=1e-11)
    assert r.plan.domain_live_bytes==d.plan.domain_live_bytes
    assert r.plan.modeled_peak_bytes<=d.plan.modeled_peak_bytes
    assert r.plan.scalar_terms<=d.plan.scalar_terms and r.plan.kernel_terms<=d.plan.kernel_terms


@pytest.mark.parametrize('spins',[1,2])
def test_pair_complete_shell_image_gauge_preserves_physical_source(spins):
    args,c,k,_=arguments(spins=spins);args['options'].pair_cutoff_bohr=7.;args['options'].nuclear_cutoff_bohr=9.
    original=manta.run_periodic_ks_from_cutoffs(**run_args(args,c));before=manta.build_periodic_ks_domains(**args)
    lattice=args['system'].lattice;shells=args['basis'].shells()
    shells[0].origin=(np.array(shells[0].origin)+lattice[:,0]).tolist()
    atoms=[vq.Atom(1,s.origin) for s in shells]
    args['basis']=vq.BasisSet(vq.Molecule(atoms,0,1),shells,'ks-domain-gauge',True)
    after=manta.build_periodic_ks_domains(**args);r=manta.run_periodic_ks_from_cutoffs(**run_args(args,c))
    assert original.converged and r.converged
    assert len(after.translations)>len(before.translations)>1
    np.testing.assert_array_equal(after.xc_images,before.xc_images)
    selected=_labels(after.translations,np.linalg.inv(lattice));origins=np.array([s.origin for s in shells])
    for i,j in itertools.product(range(len(shells)),repeat=2):
        for label in itertools.product(range(-4,5),repeat=3):
            if np.linalg.norm(origins[i]-origins[j]-lattice@label)<=7.:assert label in selected
    phase=np.ones((len(k),2),complex);phase[:,0]=np.exp(1j*k@lattice[:,0]);action=phase[:,:,None]*phase.conj()[:,None,:]
    for field in ('overlap','hcore','density','fock'):
        np.testing.assert_allclose(getattr(r,field),getattr(original,field)*action,atol=2e-9,rtol=0)
    assert r.receipt.energy==pytest.approx(original.receipt.energy,abs=4e-11)


@pytest.mark.parametrize('field,planned',[('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),
    ('primitive_quartets','primitive_work'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_allocation_and_full_scf_admission_before_payloads(field,planned):
    args,c,_,_=arguments();args['budget'].other_live_bytes=12345;args['options'].scf.maximum_iterations=2
    p=manta.plan_periodic_ks_from_cutoffs(**args);setattr(args['budget'],field,getattr(p,planned))
    d=manta.build_periodic_ks_domains(**args);r=manta.run_periodic_ks_from_cutoffs(**run_args(args,c))
    assert r.receipt.fock_calls>=2 and d.receipt.hartree_vectors
    setattr(args['budget'],field,getattr(p,planned)-1);c['points'][:]=np.nan
    for call,kwargs in [(manta.plan_periodic_ks_from_cutoffs,args),(manta.build_periodic_ks_domains,args),
                        (manta.run_periodic_ks_from_cutoffs,run_args(args,c))]:
        with pytest.raises((ValueError,RuntimeError),match='budget'):call(**kwargs)


def test_admission_precedes_empty_reciprocal_enumeration():
    args,_,_,_=arguments();args['reciprocal_energy_cutoff']=1e-5
    p=manta.plan_periodic_ks_from_cutoffs(**args)
    with pytest.raises(ValueError,match='no nonzero Hartree'):manta.build_periodic_ks_domains(**args)
    args['budget'].shell_calls=p.shell_calls-1
    with pytest.raises((ValueError,RuntimeError),match='budget'):manta.build_periodic_ks_domains(**args)


@pytest.mark.parametrize('field,value',[('reciprocal_energy_cutoff',0.),('reciprocal_energy_cutoff',float('inf')),
    ('xc_image_cutoff_bohr',-1.),('xc_image_cutoff_bohr',float('nan'))])
def test_invalid_cutoffs(field,value):
    args,_,_,_=arguments();args[field]=value
    with pytest.raises(ValueError,match='cutoff|radius'):manta.plan_periodic_ks_from_cutoffs(**args)


def test_identity_controls_owned_exports_and_explicit_origin_only_xc_support():
    args,_,_,_=arguments();d=manta.build_periodic_ks_domains(**args)
    for name in ('translations','reciprocal_vectors','xc_images'):
        expected=getattr(d,name).copy();getattr(d,name)[:]=999;np.testing.assert_array_equal(getattr(d,name),expected)
    args['options'].grid_batch=1;args['options'].reciprocal_batch=1
    assert manta.build_periodic_ks_domains(**args).receipt.identity==d.receipt.identity
    args['xc_image_cutoff_bohr']=11.001
    assert manta.build_periodic_ks_domains(**args).receipt.identity!=d.receipt.identity
    args['xc_image_cutoff_bohr']=0.
    np.testing.assert_array_equal(manta.build_periodic_ks_domains(**args).xc_images,np.zeros((1,3)))
    args['options'].pair_cutoff_bohr=None
    with pytest.raises(ValueError,match='positive'):manta.plan_periodic_ks_from_cutoffs(**args)


def test_empty_beta_pbe_backend_conditioning_is_separate_from_domain_correctness():
    from tests.test_manta_periodic_semilocal_xc import bloch_values_gradients,reference
    args,c,k,_=arguments('H',2);args['functional']=c['functional']=vq.Functional('pbe',spin=2)
    domains=manta.build_periodic_ks_domains(**args)
    r=manta.run_periodic_ks_from_cutoffs(**run_args(args,c))
    for name in ('translations','reciprocal_vectors','xc_images'):c[name]=getattr(domains,name)
    explicit=manta.run_periodic_ks(**c)
    np.testing.assert_array_equal(r.fock,explicit.fock)
    np.testing.assert_array_equal(r.density,explicit.density)
    leaf=dict(basis=c['basis'],lattice=c['system'].lattice,images=domains.xc_images,momenta=k,
        weights=np.ones(len(k))/len(k),points=c['points'],quadrature_weights=c['quadrature_weights'],densities=r.density,grid_batch=64)
    native=manta.compute_periodic_semilocal_xc(**leaf,functional=c['functional'])
    energy,v,_=reference(leaf,c['functional'])
    assert native.receipt.energy==pytest.approx(energy,abs=1e-12)
    np.testing.assert_allclose(native.values[0],v[0],atol=1e-11,rtol=0)
    # Isolate the observed libxc endpoint conditioning: a one-epsilon relative
    # change in rho_alpha alone can move the empty-beta potential by micro-Ha.
    # This is a sensitivity witness, not a uniform absolute-accuracy guarantee.
    b,g=bloch_values_gradients(leaf);d=leaf['densities'];qw=leaf['quadrature_weights']
    rho=np.einsum('k,kgi,skij,kgj->sg',leaf['weights'],b,d,b.conj()).real
    grad=(np.einsum('k,akgi,skij,kgj->sag',leaf['weights'],g,d,b.conj())
          +np.einsum('k,kgi,skij,akgj->sag',leaf['weights'],b,d,g.conj())).real
    aa=np.sum(grad[0]**2,axis=0);zero=np.zeros_like(aa)
    def minority(alpha):
        _,_,vb,_,vab,_=c['functional'].eval_polarised(alpha,zero,aa,zero,zero)
        flow=vab*grad[0]
        result=np.einsum('g,kgi,g,kgj->kij',qw,b.conj(),vb,b)
        result+=np.einsum('g,akgi,ag,kgj->kij',qw,g.conj(),flow,b)+np.einsum('g,kgi,ag,akgj->kij',qw,b.conj(),flow,g)
        return result
    base=minority(rho[0]);sensitivity=max(np.max(abs(minority(rho[0]*(1+sign*np.finfo(float).eps))-base)) for sign in (-1,1))
    error=np.max(abs(native.values[1]-v[1]))
    assert error<=max(1e-10,2*sensitivity)
    # An interior point restores the ordinary absolute comparison without
    # changing runtime regularization or the actually empty-spin calculation.
    interior=d.copy();interior[1]=1e-6*interior[0];leaf['densities']=interior
    native=manta.compute_periodic_semilocal_xc(**leaf,functional=c['functional'])
    _,v,_=reference(leaf,c['functional'])
    np.testing.assert_allclose(native.values,v,atol=1e-11,rtol=0)


def test_exact_gamma_reciprocal_boundary_and_one_ulp_below():
    args,_,_,_=arguments(mesh=(1,1,1),shift=(0,0,0))
    args['system']=vq.PeriodicSystem(3,2*np.pi*np.eye(3),args['system'].unit_cell)
    args['reciprocal_energy_cutoff']=.5
    exact=manta.build_periodic_ks_domains(**args)
    np.testing.assert_allclose(np.linalg.norm(exact.reciprocal_vectors,axis=1),1.,atol=0,rtol=0)
    assert len(exact.reciprocal_vectors)==6
    args['reciprocal_energy_cutoff']=np.nextafter(.5,1.)
    np.testing.assert_array_equal(manta.build_periodic_ks_domains(**args).reciprocal_vectors,exact.reciprocal_vectors)
    args['reciprocal_energy_cutoff']=np.nextafter(.5,0.)
    with pytest.raises(ValueError,match='no nonzero Hartree'):manta.build_periodic_ks_domains(**args)


@pytest.mark.parametrize('field,value',[('grid_points',2**63),('xc_image_cutoff_bohr',1e20),('reciprocal_energy_cutoff',1e20)])
def test_unbounded_source_requests_refuse_during_planning(field,value):
    args,_,_,_=arguments();args[field]=value
    with pytest.raises((ValueError,RuntimeError,OverflowError)):
        manta.plan_periodic_ks_from_cutoffs(**args)


def test_joint_exact_caps_include_retained_domains_and_external_live_storage():
    args,c,_,_=arguments('HeH',2);args['budget'].other_live_bytes=123456
    args['options'].scf.maximum_iterations=3;p=manta.plan_periodic_ks_from_cutoffs(**args)
    for field,planned in [('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),
                          ('primitive_quartets','primitive_work'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')]:
        setattr(args['budget'],field,getattr(p,planned))
    r=manta.run_periodic_ks_from_cutoffs(**run_args(args,c))
    assert r.receipt.fock_calls==3 and not r.converged
    assert r.plan.modeled_peak_bytes<=p.modeled_peak_bytes
    assert r.solver_memory.other_live_numerical_bytes>=p.domain_live_bytes+123456
