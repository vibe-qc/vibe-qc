"""Range-separated cutoff spheres and joint grid/domain/SCF acceptance."""
import itertools

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta
from tests.test_manta_periodic_range_separated_ks import case,oracle,components,EMPTY_RANGE,MIXED_RANGE
from tests.test_manta_periodic_ks import independent_scf
from tests.test_manta_periodic_domains import _labels

CAPS=[('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),
      ('primitive_quartets','primitive_work'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')]


def arguments(species='H2',spins=1,correction='BvK-ewald',mesh=(3,1,1),shift=(1,0,0),grid=False,order=8,name=None):
    name=name or {'H2':'hse06','HeH':'cam-b3lyp','H':EMPTY_RANGE}[species]
    c,k,old=case(species,name,spins,correction,mesh=mesh,shift=shift,order=order)
    c['budget'].memory_bytes=2**30;c['budget'].shell_calls=10**9
    c['budget'].scalar_terms=10**17;c['budget'].kernel_terms=10**18
    args={name:c[name] for name in ('basis','system','mesh','functional','options','budget')}
    args.update(reciprocal_energy_cutoff=2.,xc_image_cutoff_bohr=11.)
    if grid:
        q=manta.PeriodicQuadratureOptions();q.radial_points=12;q.lebedev_order=17;q.partition_radius_bohr=8.
        args['grid_options']=q
    else:args['grid_points']=len(c['points'])
    return args,c,k,old


def run_args(args,c):
    out=dict(args);out.pop('grid_points')
    out.update(points=c['points'],quadrature_weights=c['quadrature_weights']);return out


def explicit(args,c,old):
    da=dict(args);grid=None
    if 'grid_options' in args:
        grid=manta.build_periodic_quadrature(args['system'],args['grid_options'],args['budget'])
        da.pop('grid_options');da['grid_points']=grid.plan.grid_points
        c.update(points=grid.points,quadrature_weights=grid.weights)
    domains=manta.build_periodic_range_separated_ks_domains(**da)
    for name in ('translations','reciprocal_vectors','exchange_vectors','transfer_offsets','xc_images'):
        c[name]=getattr(domains,name)
        if name!='xc_images':old[name]=c[name]
    return domains,grid


@pytest.mark.parametrize('shape,shift',[((1,1,1),(0,0,0)),((3,1,1),(1,0,0)),((3,2,1),(1,1,0))])
def test_integer_membership_conjugacy_and_distinct_image_sources(shape,shift):
    args,_,_,_=arguments(mesh=shape,shift=shift)
    lattice=np.array([[6.,3.5,.8],[0.,4.2,1.2],[0.,0.,6.4]])
    args['system']=vq.PeriodicSystem(3,lattice,args['system'].unit_cell)
    domains=manta.build_periodic_range_separated_ks_domains(**args);b=2*np.pi*np.linalg.inv(lattice).T
    labels=np.array(list(itertools.product(range(-8,9),repeat=3)));g=labels@b.T;images=labels@lattice.T
    fractions=np.array([args['mesh'].fractional(args['mesh'].address(i)) for i in range(len(args['mesh']))]);k=fractions@b.T
    for target,kt in enumerate(k):
        for source,ks in enumerate(k):
            pair=target*len(k)+source;lo,hi=domains.transfer_offsets[pair:pair+2]
            actual=domains.exchange_vectors[lo:hi];trial=g+kt-ks
            keep=(np.sum(trial*trial,axis=1)<=2*args['reciprocal_energy_cutoff']) & np.any(trial,axis=1)
            assert _labels(actual-kt+ks,np.linalg.inv(b))=={tuple(n) for n in labels[keep]}
    keep=(np.sum(g*g,axis=1)<=2*args['reciprocal_energy_cutoff']) & np.any(labels,axis=1)
    assert _labels(domains.reciprocal_vectors,np.linalg.inv(b))=={tuple(n) for n in labels[keep]}
    for field,radius in [('translations',domains.plan.translation_radius_bohr),('xc_images',args['xc_image_cutoff_bohr'])]:
        selected=_labels(getattr(domains,field),np.linalg.inv(lattice))
        assert selected=={tuple(n) for n in labels[np.sum(images*images,axis=1)<=radius**2]}
        assert {tuple(-np.array(n)) for n in selected}==selected
    assert domains.receipt.conjugacy_vectors==len(domains.exchange_vectors)//len(k)
    assert domains.receipt.exchange_vectors<=domains.plan.exchange_candidates
    assert max(np.diff(domains.transfer_offsets))<=domains.plan.maximum_exchange_candidates
    other=manta.build_periodic_range_separated_ks_domains(**dict(args,mesh=manta.RegularKMesh((1,1,1),(0,0,0))))
    assert other.receipt.xc_image_identity==domains.receipt.xc_image_identity
    np.testing.assert_array_equal(other.xc_images,domains.xc_images)


@pytest.mark.parametrize('species,spins,name',[('H2',1,'hse06'),('H2',1,'cam-b3lyp'),
    ('H2',2,'hse06'),('HeH',2,'cam-b3lyp'),('HeH',2,MIXED_RANGE),('H',2,EMPTY_RANGE)])
@pytest.mark.parametrize('correction',['strict-zero-mode','BvK-ewald'])
def test_generated_cutoff_scf_matches_explicit_and_independent(species,spins,name,correction):
    args,c,k,old=arguments(species,spins,correction,name=name);domains,_=explicit(args,c,old)
    assert len(domains.translations)==1 and isinstance(domains,manta.PeriodicRangeSeparatedKSDomains)
    assert isinstance(domains.plan,manta.PeriodicRangeSeparatedKSDomainPlan)
    assert isinstance(domains.receipt,manta.PeriodicRangeSeparatedKSDomainReceipt)
    r=manta.run_periodic_range_separated_ks_from_cutoffs(**run_args(args,c));separate=manta.run_periodic_range_separated_ks(**c)
    assert r.converged and r.receipt.confirmed and r.plan.range_separated
    assert all(a.exchange.zero_vectors==(len(k) if a.kernel=='erfc' else 0) for a in r.receipt.exchange_components)
    np.testing.assert_array_equal(r.density,separate.density);np.testing.assert_array_equal(r.fock,separate.fock)
    assert r.receipt.source_identity==separate.receipt.source_identity
    assert r.receipt.domain_identity==domains.receipt.identity
    assert r.receipt.domain_exchange_vectors==len(domains.exchange_vectors)
    assert r.receipt.domain_xc_images==len(domains.xc_images)
    s,h,build,xi=oracle(c,k,old);pop=(r.plan.n_alpha,) if spins==1 else (r.plan.n_alpha,r.plan.n_beta)
    d=independent_scf(s,h,build,pop,spins);f,e,n=build(r.density)
    np.testing.assert_allclose(r.density,d,atol=3e-9,rtol=0)
    np.testing.assert_allclose(r.fock,f,atol=2e-11,rtol=0)
    np.testing.assert_allclose(components(r),e,atol=2e-11,rtol=0)
    assert r.receipt.madelung==pytest.approx(xi,abs=4e-12)
    np.testing.assert_allclose(r.receipt.xc.electrons_grid,n,atol=2e-12,rtol=0)
    assert r.plan.domain_live_bytes==domains.plan.domain_live_bytes
    for field in ('modeled_peak_bytes','shell_calls','primitive_work','scalar_terms','kernel_terms'):
        assert getattr(r.plan,field)<=getattr(domains.plan,field)


@pytest.mark.parametrize('field,quantity',CAPS)
@pytest.mark.parametrize('name',['hse06',MIXED_RANGE])
def test_joint_cutoff_caps_before_poisoned_payload(field,quantity,name):
    args,c,_,_=arguments('HeH',2,name=name);args['options'].ks.scf.maximum_iterations=1;args['budget'].other_live_bytes=12345
    p=manta.plan_periodic_range_separated_ks_from_cutoffs(**args)
    for cap,name in CAPS:setattr(args['budget'],cap,getattr(p,name))
    r=manta.run_periodic_range_separated_ks_from_cutoffs(**run_args(args,c));assert r.receipt.fock_calls==1
    setattr(args['budget'],field,getattr(p,quantity)-1);c['points'][:]=np.nan
    for fn,kwargs in [(manta.plan_periodic_range_separated_ks_from_cutoffs,args),(manta.build_periodic_range_separated_ks_domains,args),
        (manta.run_periodic_range_separated_ks_from_cutoffs,run_args(args,c))]:
        with pytest.raises((ValueError,RuntimeError),match='budget'):fn(**kwargs)


def test_full_hybrid_admission_before_empty_reciprocal_enumeration():
    args,_,_,_=arguments();args['reciprocal_energy_cutoff']=1e-5
    p=manta.plan_periodic_range_separated_ks_from_cutoffs(**args)
    with pytest.raises(ValueError,match='no nonzero Hartree'):manta.build_periodic_range_separated_ks_domains(**args)
    args['budget'].shell_calls=p.shell_calls-1
    with pytest.raises((ValueError,RuntimeError),match='budget'):manta.build_periodic_range_separated_ks_domains(**args)


def test_shared_hf_geometry_identity_and_owned_source_exports():
    args,_,_,_=arguments();d=manta.build_periodic_range_separated_ks_domains(**args)
    h=manta.PeriodicRHFOptions();h.pair_cutoff_bohr=args['options'].ks.pair_cutoff_bohr
    hf={key:args[key] for key in ('basis','system','mesh','reciprocal_energy_cutoff','budget')};hf['options']=h
    other=manta.build_periodic_rhf_domains(**hf)
    assert d.receipt.pair_identity==other.receipt.identity
    for name in ('translations','reciprocal_vectors','exchange_vectors','transfer_offsets'):
        np.testing.assert_array_equal(getattr(d,name),getattr(other,name))
    for name in ('translations','reciprocal_vectors','exchange_vectors','transfer_offsets','xc_images'):
        old=getattr(d,name);getattr(d,name)[:]=999;np.testing.assert_array_equal(getattr(d,name),old)
    args['options'].ks.grid_batch=1;args['options'].ks.reciprocal_batch=1
    assert manta.build_periodic_range_separated_ks_domains(**args).receipt.identity==d.receipt.identity
    args['xc_image_cutoff_bohr']+=.001;changed=manta.build_periodic_range_separated_ks_domains(**args)
    assert changed.receipt.identity!=d.receipt.identity and changed.receipt.xc_image_identity!=d.receipt.xc_image_identity
    assert changed.receipt.pair_identity==d.receipt.pair_identity
    args['xc_image_cutoff_bohr']=0.;np.testing.assert_array_equal(manta.build_periodic_range_separated_ks_domains(**args).xc_images,np.zeros((1,3)))


@pytest.mark.parametrize('spins',[1,2])
def test_pair_complete_shell_image_gauge(spins):
    args,c,k,_=arguments(spins=spins);args['options'].ks.pair_cutoff_bohr=7.;args['options'].ks.nuclear_cutoff_bohr=9.
    before=manta.build_periodic_range_separated_ks_domains(**args);r=manta.run_periodic_range_separated_ks_from_cutoffs(**run_args(args,c))
    a=args['system'].lattice;shells=args['basis'].shells();shells[0].origin=(np.array(shells[0].origin)+a[:,0]).tolist()
    args['basis']=vq.BasisSet(vq.Molecule([vq.Atom(1,s.origin) for s in shells],0,1),shells,'hybrid-cutoff-gauge',True)
    after=manta.build_periodic_range_separated_ks_domains(**args);moved=manta.run_periodic_range_separated_ks_from_cutoffs(**run_args(args,c))
    assert r.converged and moved.converged and len(after.translations)>len(before.translations)>1
    assert after.receipt.xc_image_identity==before.receipt.xc_image_identity
    phase=np.ones((len(k),2),complex);phase[:,0]=np.exp(1j*k@a[:,0]);transport=phase[:,:,None]*phase.conj()[:,None,:]
    for name in ('overlap','hcore','density','fock'):
        np.testing.assert_allclose(getattr(moved,name),getattr(r,name)*transport,atol=3e-9,rtol=0)
    np.testing.assert_allclose(components(moved),components(r),atol=5e-11,rtol=0)


@pytest.mark.parametrize('field,value',[('reciprocal_energy_cutoff',0.),('reciprocal_energy_cutoff',float('inf')),
    ('xc_image_cutoff_bohr',-1.),('xc_image_cutoff_bohr',float('nan')),('xc_image_cutoff_bohr',1e100)])
def test_invalid_support_controls(field,value):
    args,_,_,_=arguments();args[field]=value
    with pytest.raises((ValueError,OverflowError),match='cutoff|radius|extent'):manta.plan_periodic_range_separated_ks_from_cutoffs(**args)


def test_exact_reciprocal_boundary_and_ulp_below():
    args,_,_,_=arguments(mesh=(1,1,1),shift=(0,0,0));args['system']=vq.PeriodicSystem(3,2*np.pi*np.eye(3),args['system'].unit_cell)
    args['reciprocal_energy_cutoff']=.5;exact=manta.build_periodic_range_separated_ks_domains(**args)
    assert len(exact.reciprocal_vectors)==6
    np.testing.assert_array_equal(exact.exchange_vectors,exact.reciprocal_vectors)
    args['reciprocal_energy_cutoff']=np.nextafter(.5,1.)
    np.testing.assert_array_equal(manta.build_periodic_range_separated_ks_domains(**args).exchange_vectors,exact.exchange_vectors)
    args['reciprocal_energy_cutoff']=np.nextafter(.5,0.)
    with pytest.raises(ValueError,match='no nonzero Hartree'):manta.build_periodic_range_separated_ks_domains(**args)


@pytest.mark.parametrize('species,spins,name,stratmann',[('H2',1,'hse06',False),('H2',1,'cam-b3lyp',True),
    ('H2',2,'hse06',True),('HeH',2,'cam-b3lyp',False),('HeH',2,MIXED_RANGE,True),('H',2,EMPTY_RANGE,True)])
@pytest.mark.parametrize('correction',['strict-zero-mode','BvK-ewald'])
def test_joint_grid_matches_explicit_and_independent_scf(species,spins,name,stratmann,correction):
    args,c,k,old=arguments(species,spins,correction,grid=True,name=name);args['grid_options'].stratmann=stratmann
    domains,grid=explicit(args,c,old);p=manta.plan_periodic_range_separated_ks_with_quadrature(**args)
    assert isinstance(p,manta.PeriodicRangeSeparatedKSGridPlan)
    r=manta.run_periodic_range_separated_ks_with_quadrature(**args);separate=manta.run_periodic_range_separated_ks(**c)
    assert r.converged and r.receipt.confirmed and r.plan.range_separated
    assert all(a.exchange.zero_vectors==(len(k) if a.kernel=='erfc' else 0) for a in r.receipt.exchange_components)
    for name in ('density','fock','overlap','hcore','coefficients','orbital_energies'):
        np.testing.assert_array_equal(getattr(r,name),getattr(separate,name))
    assert r.receipt.energy==separate.receipt.energy and r.receipt.source_identity==separate.receipt.source_identity
    assert r.receipt.domain_identity==domains.receipt.identity
    assert r.receipt.quadrature.source_identity==grid.receipt.source_identity
    assert r.receipt.grid_stratmann==stratmann and r.receipt.grid_radial_points==12 and r.receipt.grid_lebedev_order==17
    s,h,build,_=oracle(c,k,old);pop=(r.plan.n_alpha,) if spins==1 else (r.plan.n_alpha,r.plan.n_beta)
    d=independent_scf(s,h,build,pop,spins);f,e,n=build(r.density)
    np.testing.assert_allclose(r.density,d,atol=3e-9,rtol=0);np.testing.assert_allclose(r.fock,f,atol=2e-11,rtol=0)
    np.testing.assert_allclose(components(r),e,atol=2e-11,rtol=0)
    np.testing.assert_allclose(r.receipt.xc.electrons_grid,n,atol=3e-12,rtol=0)
    assert r.plan.quadrature_live_bytes==p.grid_live_bytes
    assert r.plan.xc.grid_points==grid.plan.grid_points==r.plan.quadrature.grid_points
    assert r.receipt.modeled_peak_bytes==r.plan.modeled_peak_bytes<=p.modeled_peak_bytes
    assert r.plan.scalar_terms<=p.scalar_terms and r.plan.kernel_terms<=p.kernel_terms


@pytest.mark.parametrize('spins',[1,2])
def test_joint_capped_state_is_the_evaluated_density(spins):
    args,c,k,old=arguments('HeH' if spins==2 else 'H2',spins,grid=True);args['options'].ks.scf.maximum_iterations=1
    _,grid=explicit(args,c,old);r=manta.run_periodic_range_separated_ks_with_quadrature(**args)
    assert not r.converged and r.receipt.fock_calls==1 and len(r.scf_history)==1
    _,_,build,_=oracle(c,k,old);f,e,_=build(r.density)
    np.testing.assert_allclose(r.fock,f,atol=2e-11,rtol=0);np.testing.assert_allclose(components(r),e,atol=2e-11,rtol=0)
    assert r.receipt.quadrature.source_identity==grid.receipt.source_identity


@pytest.mark.parametrize('field,quantity',CAPS)
@pytest.mark.parametrize('name',['hse06',MIXED_RANGE])
def test_joint_grid_all_caps_exact_and_one_short(field,quantity,name):
    args,_,_,_=arguments('HeH',2,grid=True,name=name);args['options'].ks.scf.maximum_iterations=1;args['budget'].other_live_bytes=54321
    p=manta.plan_periodic_range_separated_ks_with_quadrature(**args)
    for cap,name in CAPS:setattr(args['budget'],cap,getattr(p,name))
    r=manta.run_periodic_range_separated_ks_with_quadrature(**args);assert r.receipt.fock_calls==1
    setattr(args['budget'],field,getattr(p,quantity)-1)
    for fn in (manta.plan_periodic_range_separated_ks_with_quadrature,manta.run_periodic_range_separated_ks_with_quadrature):
        with pytest.raises((ValueError,RuntimeError),match='budget'):fn(**args)


def test_grid_owners_and_generation_work_remain_reserved():
    args,_,_,_=arguments(grid=True);p=manta.plan_periodic_range_separated_ks_with_quadrature(**args);q=p.quadrature
    assert p.grid_live_bytes>=q.output_bytes
    assert p.modeled_peak_bytes==max(q.modeled_peak_bytes,p.domains.modeled_peak_bytes)
    assert p.scalar_terms==q.scalar_terms+p.domains.scalar_terms and p.kernel_terms==q.kernel_terms+p.domains.kernel_terms
    args['budget'].other_live_bytes=98765;external=manta.plan_periodic_range_separated_ks_with_quadrature(**args)
    assert external.modeled_peak_bytes==p.modeled_peak_bytes+98765
    args['budget'].other_live_bytes=0;args['budget'].scalar_terms=p.domains.scalar_terms
    with pytest.raises((ValueError,RuntimeError),match='budget'):manta.run_periodic_range_separated_ks_with_quadrature(**args)


def test_aggregate_and_functional_refusal_before_invalid_partition_generation():
    args,_,_,_=arguments(grid=True);s=args['system'];atom=s.unit_cell[0]
    args['system']=vq.PeriodicSystem(3,s.lattice,[atom,atom]);shells=args['basis'].shells()
    for shell in shells:shell.origin=atom.xyz
    args['basis']=vq.BasisSet(args['system'].unit_cell_molecule(),shells,'joint-hybrid-refusal',True)
    with pytest.raises(ValueError,match='Coincident'):manta.build_periodic_quadrature(args['system'],args['grid_options'],args['budget'])
    args['budget'].shell_calls=0
    with pytest.raises((ValueError,RuntimeError),match='budget'):manta.run_periodic_range_separated_ks_with_quadrature(**args)
    args['budget'].shell_calls=10**9;args['functional']=vq.Functional('pbe0')
    with pytest.raises(ValueError,match='range-separated'):manta.run_periodic_range_separated_ks_with_quadrature(**args)


def test_grid_receipt_and_arrays_survive_input_and_export_mutation():
    args,_,_,_=arguments(grid=True);r=manta.run_periodic_range_separated_ks_with_quadrature(**args)
    receipt=r.receipt.quadrature;plan=r.plan.quadrature;identity=receipt.source_identity;points=plan.grid_points
    r.density[:]=np.nan;r.fock[:]=np.nan;args['grid_options'].radial_points=1;del args
    assert np.isfinite(r.density).all() and np.isfinite(r.fock).all()
    del r
    assert receipt.source_identity==identity and plan.grid_points==points


@pytest.mark.parametrize('name',['cam-b3lyp',MIXED_RANGE])
def test_geometry_identity_is_separate_from_resolved_exchange(name):
    args,c,_,_=arguments();first=manta.build_periodic_range_separated_ks_domains(**args)
    a=manta.run_periodic_range_separated_ks_from_cutoffs(**run_args(args,c))
    args['functional']=vq.Functional(name)
    other=manta.build_periodic_range_separated_ks_domains(**args)
    b=manta.run_periodic_range_separated_ks_from_cutoffs(**run_args(args,c))
    assert first.receipt.identity==other.receipt.identity
    assert first.receipt.pair_identity==other.receipt.pair_identity
    assert first.receipt.xc_image_identity==other.receipt.xc_image_identity
    assert a.receipt.source_identity!=b.receipt.source_identity
    assert a.receipt.exchange_q0=='not-applicable' and b.receipt.exchange_q0=='BvK-ewald'


def test_joint_grid_with_cancelled_exchange_retains_source_admission():
    name='manta-rsh-cancelled-grid-test'
    vq.define_functional(name,[('HYB_GGA_XC_HSE06',1.),('HYB_GGA_XC_HSE06',-1.),
        ('GGA_X_B88',.8),('GGA_C_LYP',1.)])
    args,c,k,old=arguments(grid=True,name=name);args['options'].ks.scf.maximum_iterations=1
    explicit(args,c,old);p=manta.plan_periodic_range_separated_ks_with_quadrature(**args)
    assert not p.domains.ks.exchange_components
    for field,quantity in CAPS:setattr(args['budget'],field,getattr(p,quantity))
    r=manta.run_periodic_range_separated_ks_with_quadrature(**args)
    assert not r.receipt.exchange_components and r.receipt.exchange_q0=='not-applicable'
    assert r.receipt.domain_exchange_vectors>0 and r.receipt.exchange_energy==0
    _,_,build,_=oracle(c,k,old,terms=[]);f,e,_=build(r.density)
    np.testing.assert_allclose(r.fock,f,atol=2e-11,rtol=0)
    np.testing.assert_allclose(components(r),e,atol=2e-11,rtol=0)
