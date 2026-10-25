"""Full Coulomb fitting enclosures without finite-metric projection assumptions."""
import dataclasses
import numpy as np
import pytest
from vibeqc import manta
from tests.test_manta_periodic_ri_pair_tree import pair_actions_case
from tests.test_manta_periodic_ri_paired import paired_case
from tests.test_manta_periodic_ri_q import densities,reference as ri_reference
from tests.test_manta_periodic_gdf import independent,gaussian_fourier
from tests.test_manta_periodic_ewald import cube
from tests.test_manta_periodic_pairs import options as pair_options
from tests.test_manta_periodic_bubbles import bubble_options
from tests.test_manta_periodic_bupo import options as bupo_options
from tests.test_manta_periodic_fitting import CAPS


def unfitted_source(a,weights,*,radius=2,full=True,images=1):
    pairs=manta.PeriodicPairSource(a['orbital'],a['system'].lattice,cube(images),a['ket_kpoints'],weights,
        pair_options(0,a['options'].pair_cutoff_bohr,.2),a['budget'])
    tree=manta.PeriodicBubbleTree(pairs,bubble_options(),a['budget'])
    direct=manta.PeriodicEwaldSource(pairs,.4,cube(radius),cube(radius+2,reciprocal=True),64,a['budget'])
    opts=bupo_options(0.);opts.bound_infinite_domain=opts.bound_pair_images=full
    return manta.PeriodicBupoSource(tree,direct,cube(0),opts,a['budget'])


def case(*,component=True,weights=(.27,.73),rank=6,diffuse=False):
    a=paired_case() if diffuse else pair_actions_case(True,fitted=True,order=0);a['budget'].memory_bytes=4*2**30
    weights=np.asarray(weights,dtype=float);src=manta.PeriodicGDFTiles(**a);values=np.linalg.eigvalsh(src.metric().values)
    opts=manta.RIOptions();opts.relative_cutoff=0.;opts.absolute_cutoff=1e-12 if rank==6 else (2*values[-1] if rank==0 else .5*(values[-rank-1]+values[-rank]))
    ri=manta.PeriodicRIQBuilder(src,2,opts,a['budget']);ref=unfitted_source(a,weights)
    o=manta.PeriodicRIFittingBoundOptions();o.density_l1_limit=12.
    if component:o.density_entry_limits=np.full((len(weights),a['orbital'].nbasis,a['orbital'].nbasis),2.).ravel().tolist()
    return a,ri,ref,o,weights


def bare_reference(a,d,weights,cutoff=12):
    # Independent scalar Gaussian product theorem and full reciprocal Coulomb.
    # Enlarge AO support separately from the RI finite fit. Tight compact
    # Gaussians make the finite image/shell remainder negligible here.
    options=a['options'];previous=options.pair_cutoff_bohr;options.pair_cutoff_bohr=10.
    vectors=cube(cutoff,reciprocal=True)@a['system'].reciprocal_lattice().T
    try:_,panel=gaussian_fourier(a,vectors)
    finally:options.pair_cutoff_bohr=previous
    kernel=4*np.pi/abs(np.linalg.det(a['system'].lattice))/np.sum(vectors*vectors,axis=1)
    rho=np.einsum('k,skij,kijg->g',weights,d,panel.conj())
    j=np.einsum('g,g,kijg->kij',kernel,rho,panel)
    return j,.5*np.dot(kernel,abs(rho)**2)


@pytest.mark.parametrize('rank',[0,2,6])
@pytest.mark.parametrize('component',[False,True])
def test_fitting_bound_covers_unfitted_coulomb_on_complex_indefinite_family(rank,component):
    a,ri,ref,o,w=case(rank=rank,component=component)
    bound=manta.PeriodicRIFittingBoundSource(ri,ref,o,a['budget']);c=bound.receipt
    assert c.probes==bound.plan.maximum_probes==8 and c.reference_remainder_j_bound>0.
    assert not c.roundoff_included and c.ri_source_identity==ri.metric_info.source_identity
    for seed in range(3):
        d=densities(seed=seed)*.07;actual=bound.build(d,w);truth,energy=bare_reference(a,d,w)
        assert np.all(abs(actual.J-truth)<=bound.j_error_bounds+1e-10)
        assert abs(actual.receipt.coulomb_energy-energy)<=c.energy_error_bound+1e-10
        finite=ref.build(np.sum(d,axis=0)).receipt.energy
        assert abs(actual.receipt.coulomb_energy-finite)<=c.finite_difference_energy_bound+1e-10
        assert abs(finite-energy)<=c.reference_remainder_energy_bound+1e-10
        control=ri.build(d,w,make_k=False);np.testing.assert_array_equal(actual.J,control.J)
        assert actual.receipt.snapshot_identity==control.receipt.snapshot_identity
    assert c.j_error_bound==np.max(bound.j_error_bounds)
    assert c.finite_difference_energy_bound<=.5*c.density_l1_limit*c.finite_difference_j_bound*(1+1e-12)
    assert c.reference_remainder_energy_bound<=.5*c.density_l1_limit*c.reference_remainder_j_bound*(1+1e-12)
    old=bound.j_error_bounds;mutated=bound.j_error_bounds;mutated[:]=0.;np.testing.assert_array_equal(bound.j_error_bounds,old)
    # Input options are captured; changing the original cannot broaden them.
    o.density_l1_limit=1e12
    with pytest.raises(ValueError,match='envelope'):bound.build(densities()*100,w)


def test_independent_full_fourier_reference_is_converged():
    a,ri,ref,o,w=case(diffuse=True);d=densities()*.07
    x,e=bare_reference(a,d,w,12);y,f=bare_reference(a,d,w,16)
    np.testing.assert_allclose(x,y,atol=3e-11,rtol=0);assert abs(e-f)<3e-11
    assert np.max(abs(d.imag))>0 and np.max(abs(x.imag))>1e-8
    bound=manta.PeriodicRIFittingBoundSource(ri,ref,o,a['budget']);actual=bound.build(d,w)
    assert np.all(abs(actual.J-x)<=bound.j_error_bounds+1e-10)
    assert abs(actual.receipt.coulomb_energy-e)<=bound.receipt.energy_error_bound+1e-10


@pytest.mark.parametrize('weights',[(0.,1.),(.27,.73)])
def test_zero_entries_weights_and_density_scale(weights):
    a,ri,ref,o,w=case(weights=weights);o.density_entry_limits=[0.]*8
    zero=manta.PeriodicRIFittingBoundSource(ri,ref,o,a['budget']);assert zero.receipt.probes==0
    assert zero.receipt.j_error_bound==zero.receipt.energy_error_bound==zero.receipt.finite_difference_energy_bound==zero.receipt.reference_remainder_energy_bound==0.
    np.testing.assert_array_equal(zero.build(np.zeros((2,2,2,2),complex),w).J,0.)
    o.density_entry_limits=[];first=manta.PeriodicRIFittingBoundSource(ri,ref,o,a['budget'])
    o.density_l1_limit*=2;second=manta.PeriodicRIFittingBoundSource(ri,ref,o,a['budget'])
    assert first.receipt.probes==(4 if weights[0]==0 else 8)
    assert second.receipt.j_error_bound==pytest.approx(2*first.receipt.j_error_bound,rel=2e-12)
    assert second.receipt.energy_error_bound==pytest.approx(4*first.receipt.energy_error_bound,rel=3e-12)
    for name in ('finite_difference_energy_bound','reference_remainder_energy_bound'):
        assert getattr(second.receipt,name)==pytest.approx(4*getattr(first.receipt,name),rel=3e-12)
    with pytest.raises(ValueError,match='weights'):first.build(densities()*.01,w[::-1].copy())


def test_source_target_and_family_refusals():
    a,ri,ref,o,w=case()
    o.maximum_probes=7
    with pytest.raises(ValueError,match='probe limit'):manta.PeriodicRIFittingBoundSource(ri,ref,o,a['budget'])
    o.maximum_probes=8;o.j_error=1.
    with pytest.raises(ValueError,match='both'):manta.PeriodicRIFittingBoundSource(ri,ref,o,a['budget'])
    o.energy_error=o.j_error=1e-20
    with pytest.raises(RuntimeError,match='targets unmet'):manta.PeriodicRIFittingBoundSource(ri,ref,o,a['budget'])
    o.energy_error=o.j_error=1e6
    bound=manta.PeriodicRIFittingBoundSource(ri,ref,o,a['budget']);assert bound.receipt.targets_requested
    partial=unfitted_source(a,w,full=False)
    with pytest.raises(ValueError,match='full Ewald'):manta.PeriodicRIFittingBoundSource(ri,partial,o,a['budget'])
    other=dict(a,ket_kpoints=a['ket_kpoints']+.01);different=unfitted_source(other,w)
    with pytest.raises(ValueError,match='frame mismatch'):manta.PeriodicRIFittingBoundSource(ri,different,o,a['budget'])
    o.density_entry_limits=[-1.]*8
    with pytest.raises(ValueError,match='nonnegative'):manta.PeriodicRIFittingBoundSource(ri,ref,o,a['budget'])


@pytest.mark.parametrize('field,quantity',CAPS)
def test_joint_source_and_probe_resource_caps(field,quantity):
    a,ri,ref,o,w=case();b=a['budget'];p=manta.plan_periodic_ri_fitting_bound(ri,ref,o,b)
    setattr(b,field,getattr(p,quantity)-1)
    with pytest.raises(ValueError,match='budget'):manta.PeriodicRIFittingBoundSource(ri,ref,o,b)
    setattr(b,field,getattr(p,quantity));manta.PeriodicRIFittingBoundSource(ri,ref,o,b)


def fitting_job(directory,method='rhf'):
    from tests.test_manta_periodic_ri_metric_bounds import metric_job
    job,args,k=metric_job(directory,method,truncate=False);job['max_iter']=100;args['options'].scf.maximum_iterations=100
    # Match the native method's frame exactly, including reciprocal-lattice rounding.
    k=np.array([args['system'].reciprocal_lattice()@args['mesh'].fractional(args['mesh'].address(i)) for i in range(len(args['mesh']))])
    job['periodic_fitting']=dataclasses.replace(job['periodic_fitting'],accuracy=None,coupling_multipole_error=1e-12)
    args['options'].fitting.coupling_multipole_pair_tree=True;args['options'].fitting.coupling_multipole_error=1e-12
    args['options'].ri=job['coulomb'].metric;args['budget'].memory_bytes=4*2**30
    domains=getattr(manta,'build_periodic_ri_'+method+'_domains')(**args)
    a=dict(orbital=args['basis'],auxiliary=args['auxiliary'],system=args['system'],q=np.zeros(3),ket_kpoints=k,
        vectors=domains.reciprocal_vectors,options=args['options'].fitting,budget=args['budget'])
    ri=manta.PeriodicRIQBuilder(manta.PeriodicGDFTiles(**a),2,args['options'].ri,args['budget']);w=np.full(len(k),1/len(k))
    ref=unfitted_source(a,w,radius=3)
    o=manta.PeriodicRIFittingBoundOptions();o.density_l1_limit=1000.
    return job,args,a,ri,ref,o,w


def test_certificate_retains_sources_and_honors_existing_ri_accuracy_family():
    import gc
    from tests.test_manta_periodic_ri_pair_tree import automatic_accuracy
    a,_,ref,o,w=case();accuracy=automatic_accuracy(1e12,1e12,32.)
    ri=manta.PeriodicRIQBuilder(manta.PeriodicGDFTiles(**a),2,manta.RIOptions(),a['budget'],accuracy)
    bound=manta.PeriodicRIFittingBoundSource(ri,ref,o,a['budget'])
    d=densities()*.01;expected=ri.build(d,w,make_k=False)
    del ri,ref;gc.collect();actual=bound.build(d,w)
    np.testing.assert_array_equal(actual.J,expected.J)
    assert actual.receipt.accuracy_certified
    with pytest.raises(ValueError,match='float64'):bound.build(d,np.ones(200))


def prepared_fitting_reference(args):
    source = manta.PeriodicBupoSourceOptions()
    source.pairs = pair_options(0, args['options'].pair_cutoff_bohr, .2)
    source.bubbles = bubble_options()
    source.bupo = bupo_options(0.)
    source.bupo.bound_infinite_domain = source.bupo.bound_pair_images = True
    source.pair_cube_radius = 1
    source.real_cube_radius = 3
    source.reciprocal_cube_radius = 5
    source.near_cube_radius = 0
    source.eta = .4
    return manta.prepare_periodic_bupo_source(args['basis'], args['system'], args['mesh'], source, 64, args['budget'])


@pytest.mark.parametrize('method', ['rhf', 'uhf'])
def test_fitting_consumer_reuses_shared_prepared_source_and_budget(tmp_path, method):
    _, args, a, ri, ref, options, weights = fitting_job(tmp_path, method)
    prepared = prepared_fitting_reference(args)
    assert prepared.source.source_identity == ref.source_identity
    actual = manta.PeriodicRIFittingBoundSource(ri, prepared.source, options, prepared.remaining_budget)
    control = manta.PeriodicRIFittingBoundSource(ri, ref, options, args['budget'])
    assert actual.receipt.certificate_identity == control.receipt.certificate_identity
    np.testing.assert_array_equal(actual.j_error_bounds, control.j_error_bounds)
    assert actual.plan.modeled_peak_bytes == control.plan.modeled_peak_bytes + prepared.plan.source_live_bytes
    n, nk = a['orbital'].nbasis, len(weights)
    d = np.ascontiguousarray(np.ones((2, nk, n, n), complex)*.01)
    np.testing.assert_array_equal(actual.build(d, weights).J, control.build(d, weights).J)
