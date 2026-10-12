"""Joint admission of quadrature, cutoff domains and physical pure KS."""
import numpy as np
import pytest

from vibeqc import manta
from tests.test_manta_periodic_ks_domains import arguments
from tests.test_manta_periodic_ks import oracle,independent_scf


def case(species='H2',spins=1,stratmann=False):
    args,c,k,old=arguments(species,spins)
    args.pop('grid_points');o=manta.PeriodicQuadratureOptions();o.radial_points=12;o.lebedev_order=17
    o.partition_radius_bohr=8.;o.stratmann=stratmann;args['grid_options']=o
    return args,c,k,old


def explicit(args,c,old):
    grid=manta.build_periodic_quadrature(args['system'],args['grid_options'],args['budget'])
    da={k:v for k,v in args.items() if k!='grid_options'};da['grid_points']=grid.plan.grid_points
    domains=manta.build_periodic_ks_domains(**da)
    c.update(points=grid.points,quadrature_weights=grid.weights)
    for name in ('translations','reciprocal_vectors','xc_images'):c[name]=getattr(domains,name)
    old.update(reciprocal_vectors=domains.reciprocal_vectors,translations=domains.translations)
    return grid,domains


@pytest.mark.parametrize('species,spins,stratmann',[('H2',1,False),('H2',2,True),('HeH',2,False),('H',2,True)])
def test_complete_joint_run_matches_explicit_source_and_independent_scf(species,spins,stratmann):
    args,c,k,old=case(species,spins,stratmann);grid,domains=explicit(args,c,old)
    plan=manta.plan_periodic_ks_with_quadrature(**args);r=manta.run_periodic_ks_with_quadrature(**args)
    separate=manta.run_periodic_ks(**c)
    assert r.converged and r.receipt.confirmed
    for field in ('density','fock','overlap','hcore','coefficients','orbital_energies'):
        np.testing.assert_array_equal(getattr(r,field),getattr(separate,field))
    assert r.receipt.energy==separate.receipt.energy
    assert r.receipt.source_identity==separate.receipt.source_identity
    assert r.receipt.domain_identity==domains.receipt.identity
    assert r.receipt.quadrature.source_identity==grid.receipt.source_identity
    assert separate.receipt.quadrature is None and separate.plan.quadrature is None
    assert r.receipt.grid_radial_points==12 and r.receipt.grid_lebedev_order==17
    assert r.receipt.grid_partition_radius_bohr==8 and r.receipt.grid_stratmann==stratmann
    s,h,build=oracle(c,k,old);pop=(r.plan.n_alpha,) if spins==1 else (r.plan.n_alpha,r.plan.n_beta)
    d=independent_scf(s,h,build,pop,spins)
    np.testing.assert_allclose(r.density,d,atol=3e-9,rtol=0)
    f,e,n=build(r.density)
    np.testing.assert_allclose(r.fock,f,atol=2e-11,rtol=0)
    assert r.receipt.energy==pytest.approx(r.receipt.nuclear_energy+sum(e),abs=2e-11)
    np.testing.assert_allclose(r.receipt.xc.electrons_grid,n,atol=2e-12,rtol=0)
    assert r.plan.quadrature_live_bytes==plan.grid_live_bytes
    assert r.plan.quadrature.grid_points==grid.plan.grid_points==r.plan.xc.grid_points
    assert r.plan.modeled_peak_bytes<=plan.modeled_peak_bytes
    assert r.receipt.modeled_peak_bytes==r.plan.modeled_peak_bytes
    assert r.plan.scalar_terms<=plan.scalar_terms and r.plan.kernel_terms<=plan.kernel_terms


@pytest.mark.parametrize('spins',[1,2])
def test_capped_epoch_retains_same_density_physical_energy_and_grid_receipt(spins):
    args,c,k,old=case('HeH' if spins==2 else 'H2',spins);args['options'].scf.maximum_iterations=1
    grid,_=explicit(args,c,old);r=manta.run_periodic_ks_with_quadrature(**args)
    assert not r.converged
    _,_,build=oracle(c,k,old);f,e,n=build(r.density)
    np.testing.assert_allclose(r.fock,f,atol=2e-11,rtol=0)
    assert r.receipt.energy==pytest.approx(r.receipt.nuclear_energy+sum(e),abs=2e-11)
    assert r.receipt.quadrature.source_identity==grid.receipt.source_identity
    assert len(r.scf_history)==1


@pytest.mark.parametrize('field,quantity',[
    ('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),('primitive_quartets','primitive_work'),
    ('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_simultaneous_exact_caps_and_each_one_short(field,quantity):
    args,_,_,_=case();args['options'].scf.maximum_iterations=1
    p=manta.plan_periodic_ks_with_quadrature(**args);b=args['budget']
    for name,q in [('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),
        ('primitive_quartets','primitive_work'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')]:
        setattr(b,name,getattr(p,q))
    r=manta.run_periodic_ks_with_quadrature(**args)
    assert r.plan.modeled_peak_bytes<=p.modeled_peak_bytes
    setattr(b,field,getattr(p,quantity)-1)
    for fn in (manta.plan_periodic_ks_with_quadrature,manta.run_periodic_ks_with_quadrature):
        with pytest.raises((ValueError,RuntimeError),match='budget'):fn(**args)


def test_retained_grid_and_work_charged_before_domains_and_solver():
    args,_,_,_=case();p=manta.plan_periodic_ks_with_quadrature(**args);q=p.quadrature
    assert p.grid_live_bytes>=q.output_bytes
    assert p.modeled_peak_bytes==max(q.modeled_peak_bytes,p.domains.modeled_peak_bytes)
    assert p.scalar_terms==q.scalar_terms+p.domains.scalar_terms
    assert p.kernel_terms==q.kernel_terms+p.domains.kernel_terms
    args['budget'].other_live_bytes=98765
    external=manta.plan_periodic_ks_with_quadrature(**args)
    assert external.modeled_peak_bytes==p.modeled_peak_bytes+98765
    assert external.grid_live_bytes==p.grid_live_bytes
    args['budget'].other_live_bytes=0
    # A budget sufficient for the domain/SCF alone must not silently omit
    # grid production work. Admission refuses before numerical generation.
    args['budget'].scalar_terms=p.domains.scalar_terms
    with pytest.raises((ValueError,RuntimeError),match='budget'):manta.run_periodic_ks_with_quadrature(**args)


def test_aggregate_refusal_precedes_invalid_partition_generation():
    import vibeqc as vq
    args,_,_,_=case();s=args['system'];atom=s.unit_cell[0]
    args['system']=vq.PeriodicSystem(3,s.lattice,[atom,atom]);args['budget'].shell_calls=0
    shells=args['basis'].shells()
    for shell in shells:shell.origin=atom.xyz
    args['basis']=vq.BasisSet(args['system'].unit_cell_molecule(),shells,'joint-grid-refusal',True)
    with pytest.raises(ValueError,match='Coincident'):
        manta.build_periodic_quadrature(args['system'],args['grid_options'],args['budget'])
    with pytest.raises((ValueError,RuntimeError),match='budget'):manta.run_periodic_ks_with_quadrature(**args)


def test_receipt_and_arrays_survive_inputs_and_exports():
    args,_,_,_=case();r=manta.run_periodic_ks_with_quadrature(**args)
    receipt=r.receipt.quadrature;plan=r.plan.quadrature;points=plan.grid_points;identity=receipt.source_identity
    density=r.density;density[:]=np.nan
    args['grid_options'].radial_points=1;del args
    assert np.isfinite(r.density).all() and r.receipt.quadrature.source_identity==identity
    assert r.plan.quadrature.grid_points==points
    del r
    assert receipt.source_identity==identity and plan.grid_points==points
