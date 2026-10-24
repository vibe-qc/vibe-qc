"""Bounded shared actions preserve scalar physics, envelopes and identities."""
import numpy as np
import pytest
from vibeqc import manta
from tests.test_manta_periodic_ewald import make_basis, cube
from tests.test_manta_periodic_pairs import source, density, IMAGES
from tests.test_manta_periodic_bupo import fixture, options
from tests.test_manta_periodic_ri_fitting_bounds import case


def budget():
    b=manta.ResourceBudget();b.memory_bytes=4*2**30
    for name in ('shell_calls','primitive_quartets','scalar_terms','kernel_terms'):setattr(b,name,10**18)
    return b


def same_receipt(a,b):
    for name in dir(a):
        if name.startswith('_'):continue
        x,y=getattr(a,name),getattr(b,name)
        if callable(x):continue
        if x is None or isinstance(x,(bool,int,float,str)):assert x==y,name
        else:same_receipt(x,y)


@pytest.mark.parametrize('angular,pure,empty_g',[(False,True,False),(True,True,False),(True,False,True)])
@pytest.mark.parametrize('layout',['C','F'])
def test_ewald_batch_preserves_scalar_actions_bitwise(angular,pure,empty_g,layout):
    b=make_basis((0,1) if angular else (0,0),pure,contracted=False)
    s=source(b,order=0,images=IMAGES[:3],weights=np.array([0.,.3,.7]))
    e=manta.PeriodicEwaldSource(s,.63,cube(0),cube(0 if empty_g else 1,reciprocal=True),7)
    d=np.array([density(b.nbasis,seed) for seed in (17,83,12)],order=layout)
    result=e.build_batch(d,budget());assert result.receipt.source_passes==1
    assert result.receipt.source_identity==e.source_identity
    for actual,dd in zip(result.results,d):
        expected=e.build(np.ascontiguousarray(dd));np.testing.assert_array_equal(actual.values,expected.values)
        same_receipt(actual.receipt,expected.receipt)
    assert result.receipt.real_shell_calls==result.results[0].receipt.real_shell_calls
    assert result.receipt.coefficient_panels==sum(x.receipt.coefficient_panels for x in result.results)
    view=result.results[0].values;view[:]=0
    assert np.any(result.results[0].values!=0)
    np.testing.assert_array_equal(e.build_batch(d,budget()).results[-1].values,result.results[-1].values)


@pytest.mark.parametrize('full,approximate',[(False,False),(True,False),(True,True)])
def test_bupo_batch_bounds_match_scalar_and_count_actual_work(full,approximate):
    b,s,t,e,_,_=fixture(0,tolerance=10. if approximate else 0.,exponent=3200.,images=False)
    o=options(10. if approximate else 0.);o.bound_infinite_domain=o.bound_pair_images=full
    c=manta.PeriodicBupoSource(t,e,cube(0),o,budget())
    assert bool(c.partition.approximate_blocks)==approximate
    d=np.stack([density(b.nbasis,seed) for seed in (17,83,12)])
    batch=c.build_batch(d,budget());assert batch.receipt.source_passes==(3 if approximate else 1)
    for actual,dd in zip(batch.results,d):
        expected=c.build(dd);np.testing.assert_array_equal(actual.values,expected.values)
        for name in ('j_error_bound','domain_j_error_bound','pair_image_j_error_bound','retained_pair_mass_bound','omitted_pair_mass_bound'):
            np.testing.assert_array_equal(getattr(actual,name),getattr(expected,name))
        same_receipt(actual.receipt,expected.receipt)
    if full:
        assert batch.receipt.domain_majorant_pairs_evaluated==(3 if approximate else 1)*batch.results[0].receipt.domain.majorant_pairs_evaluated


@pytest.mark.parametrize('kind',['ewald','bupo'])
@pytest.mark.parametrize('cap,field',[('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),('primitive_quartets','primitive_products'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_batch_admission_rejects_each_cap_before_build(kind,cap,field):
    b,s,t,e,c,_=fixture(0,tolerance=0.,images=False);source=e if kind=='ewald' else c
    d=np.stack([density(b.nbasis)]*2);limits=budget();plan=source.plan_batch(2,limits)
    setattr(limits,cap,getattr(plan,field)-1)
    with pytest.raises((ValueError,RuntimeError),match='budget'):source.build_batch(d,limits)
    limits=budget();limits.other_live_bytes=10**12
    with pytest.raises((ValueError,RuntimeError),match='memory'):source.plan_batch(2,limits)


@pytest.mark.parametrize('count',[0,33])
def test_batch_count_shape_and_all_density_validation(count):
    b,s,t,e,c,_=fixture(0,tolerance=0.,images=False)
    for source in (e,c):
        with pytest.raises(ValueError,match='1 to 32'):source.build_batch(np.zeros((count,3,b.nbasis,b.nbasis),complex),budget())
        with pytest.raises(ValueError,match='shape|dtype'):source.build_batch(np.zeros((2,3,b.nbasis,b.nbasis)),budget())
        d=np.stack([density(b.nbasis)]*2);d[-1,0,0,1]+=1j
        with pytest.raises(ValueError,match='Hermitian'):source.build_batch(d,budget())
        d[-1]=np.nan
        with pytest.raises((ValueError,RuntimeError),match='finite'):source.build_batch(d,budget())


@pytest.mark.parametrize('batch_size',[2,3,8,32])
@pytest.mark.parametrize('weights',[(.27,.73),(0.,1.)])
def test_fitting_batches_preserve_certificate_including_split_pairs(batch_size,weights):
    a,ri,ref,o,w=case(weights=weights);o.probe_batch_size=1
    scalar=manta.PeriodicRIFittingBoundSource(ri,ref,o,a['budget'])
    o.probe_batch_size=batch_size;batch=manta.PeriodicRIFittingBoundSource(ri,ref,o,a['budget'])
    np.testing.assert_array_equal(batch.j_error_bounds,scalar.j_error_bounds)
    for name in ('certificate_identity','j_error_bound','energy_error_bound','finite_difference_j_bound','reference_remainder_j_bound','finite_difference_energy_bound','reference_remainder_energy_bound'):
        assert getattr(batch.receipt,name)==getattr(scalar.receipt,name)
    count=batch.receipt.probes;passes=(count+batch_size-1)//batch_size
    assert batch.receipt.reference_batches==batch.receipt.reference_source_passes==passes
    assert batch.receipt.reference_shell_calls*count==scalar.receipt.reference_shell_calls*passes
    assert batch.receipt.reference_majorant_pairs*count==scalar.receipt.reference_majorant_pairs*passes
    o.density_entry_limits=[0.]*8;zero=manta.PeriodicRIFittingBoundSource(ri,ref,o,a['budget'])
    assert zero.receipt.reference_batches==zero.receipt.probes==0


@pytest.mark.parametrize('batch_size',[0,33])
def test_fitting_rejects_invalid_batch_capacity(batch_size):
    a,ri,ref,o,w=case();o.probe_batch_size=batch_size
    with pytest.raises(ValueError,match='probe batch'):manta.plan_periodic_ri_fitting_bound(ri,ref,o,a['budget'])
