"""Bound-admitted periodic BUPO with a complete source-matched direct fallback."""
import gc
import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta, _vibeqc_core as core
from tests.test_manta_periodic_pairs import K, WEIGHTS, IMAGES, density, options as pair_options
from tests.test_manta_periodic_bubbles import bubble_options
from tests.test_manta_periodic_ewald import reference, cube


def options(tolerance=1e-5, opening=.7):
    o = manta.PeriodicBupoOptions(); o.integral_tolerance=tolerance; o.opening=opening; return o


def fixture(order=6, *, angular=False, pure=True, skew=True, tolerance=1e-5, cap=2,
            near=None, images=True, exponent=3200., radius=.07, weights=WEIGHTS):
    cell = np.array([[6., .3, -.2], [0., 6.2, .2], [0., 0., 5.8]]) if skew else np.eye(3)*6
    positions = [[.02, .0, .0], [.06, .0, .0], [3., .02, .01], [3.04, .02, .01]]
    if images: positions += [[5.96, 0., 0.], [5.99, 0., 0.]]
    molecule = vq.Molecule([vq.Atom(2, p) for p in positions], 0, 1); shells=[]
    for i, center in enumerate(positions):
        s=core.ShellInfo(); s.atom_index=i; s.origin=center; s.l=1 if angular and i in (1,3) else 0; s.pure=pure
        s.exponents=[exponent*(1+.05*i), 2.1*exponent*(1+.05*i)]; s.coefficients=[.8, -.12]; shells.append(s)
    basis=vq.BasisSet(molecule, shells, 'periodic-bupo-reference', False)
    po=pair_options(order, cutoff=.2, radius=radius)
    pairs=manta.PeriodicPairSource(basis, cell, IMAGES[:3].copy() if images else cube(0), K, weights, po)
    tree=manta.PeriodicBubbleTree(pairs, bubble_options(cap=cap))
    direct=manta.PeriodicEwaldSource(pairs, .4, cube(1), cube(1, reciprocal=True), 16)
    near=cube(0) if near is None else near
    consumer=manta.PeriodicBupoSource(tree, direct, near, options(tolerance))
    return basis, pairs, tree, direct, consumer, cell


def calculation(order=6, *, angular=False, pure=True, skew=True, tolerance=1e-5, cap=2,
                near=None, delta=0., independent=False, **kwargs):
    b,s,t,exact,c,cell=fixture(order,angular=angular,pure=pure,skew=skew,tolerance=tolerance,cap=cap,near=near,**kwargs)
    d=density(b.nbasis)+delta*density(b.nbasis,83); out=c.build(d); direct=exact.build(d); r=out.receipt
    row=dict(values=out.values,direct_values=direct.values,j_error_bound=out.j_error_bound,
        energy=r.energy,direct_reference_energy=direct.receipt.energy,trace_energy=r.trace_energy,
        energy_bound=r.energy_error_bound,direct_energy=r.direct_energy,far_energy=r.far_energy,
        matrix_error=float(np.max(abs(out.values-direct.values))),energy_error=abs(r.energy-direct.receipt.energy),
        source_identity=r.source_identity,pair_source_identity=r.pair_source_identity,density_identity=r.density_identity,
        partition_identity=r.partition.identity,far_blocks=r.partition.approximate_blocks,direct_blocks=r.partition.direct_blocks,
        far_pairs=r.partition.approximate_pairs,direct_pairs=r.partition.direct_pairs,pairs=s.receipt.pairs,nodes=len(t.nodes),
        maximum_integral_error=r.partition.maximum_integral_error,refinements=r.partition.traversal.refined_candidates,
        d=d,weights=kwargs.get('weights',WEIGHTS),direction=density(b.nbasis,83),cell=cell)
    if independent:
        ref=reference(b,s,d,cell,.4,cube(1),cube(1,reciprocal=True),quadrature=160,weights=row['weights'])
        row.update(reference_values=ref['values'],reference_energy=ref['energy'],reference_components=ref['energies'],
                   component_matrices=ref['component_matrices'],coefficients=ref['coefficients'])
    return row


@pytest.mark.parametrize('pure', [False,True])
@pytest.mark.parametrize('skew', [False,True])
def test_complete_complex_response_and_component_bounds(pure,skew):
    row=calculation(angular=True,pure=pure,skew=skew)
    assert row['far_blocks'] > 0 and row['direct_blocks'] > 0
    assert row['far_pairs']+row['direct_pairs'] == row['pairs']*(row['pairs']+1)//2
    assert np.all(abs(row['values']-row['direct_values']) <= row['j_error_bound'][None,:,:]+2e-11)
    assert row['energy_error'] <= row['energy_bound']+2e-11
    assert row['energy'] == pytest.approx(row['trace_energy'], abs=3e-11)
    assert row['maximum_integral_error'] <= 1e-5
    assert np.max(abs(row['values'].imag)) > 1e-3


@pytest.mark.parametrize('order,tolerance', [(0,10.),(4,.01),(6,1e-5),(8,1e-7)])
def test_independent_signed_gaussian_complete_energy(order,tolerance):
    row=calculation(order,tolerance=tolerance,independent=True)
    assert row['far_pairs'] > 0
    np.testing.assert_allclose(row['direct_values'],row['reference_values'],atol=2e-11,rtol=2e-12)
    assert np.all(abs(row['values']-row['reference_values'])<=row['j_error_bound'][None,:,:]+2e-11)
    assert abs(row['energy']-row['reference_energy'])<=row['energy_bound']+2e-11


@pytest.mark.parametrize('weights', [WEIGHTS,np.array([0.,.4,.6])])
def test_fixed_partition_weighted_derivative_linearity_and_density_scaled_bounds(weights):
    b,s,t,e,c,cell=fixture(angular=True,weights=weights)
    d=density(b.nbasis); h=density(b.nbasis,83); base=c.build(d); response=c.build(h)
    slope=np.einsum('k,kij,kji->',weights,h,base.values).real
    assert base.receipt.partition.approximate_blocks>0
    for step in (1e-4,5e-5):
        plus,minus=c.build(d+step*h),c.build(d-step*h)
        assert (plus.receipt.energy-minus.receipt.energy)/(2*step)==pytest.approx(slope,abs=2e-8)
        np.testing.assert_allclose(plus.values,base.values+step*response.values,atol=3e-12,rtol=3e-12)
        assert plus.receipt.partition.identity==minus.receipt.partition.identity==base.receipt.partition.identity==c.partition.identity
    scaled=c.build(-2*d)
    np.testing.assert_allclose(scaled.values,-2*base.values,atol=2e-12)
    np.testing.assert_allclose(scaled.j_error_bound,2*base.j_error_bound,atol=0.,rtol=3e-12)
    assert scaled.receipt.energy_error_bound==pytest.approx(4*base.receipt.energy_error_bound,rel=3e-12)
    assert scaled.receipt.energy==pytest.approx(4*base.receipt.energy,abs=2e-11)
    zero=c.build(np.zeros_like(d));np.testing.assert_array_equal(zero.values,0.);np.testing.assert_array_equal(zero.j_error_bound,0.)
    assert zero.receipt.energy==zero.receipt.energy_error_bound==0
    assert c.source_identity==base.receipt.source_identity==scaled.receipt.source_identity
    assert base.receipt.density_identity==s.contract(d).receipt.density_identity


def test_zero_tolerance_exact_fallback_and_periodic_image_overlap_refinement():
    b,s,t,e,c,cell=fixture(tolerance=0.)
    d=density(b.nbasis);actual=c.build(d);exact=e.build(d)
    assert c.partition.approximate_blocks==0 and c.partition.traversal.refined_candidates>0
    np.testing.assert_array_equal(actual.values,exact.values)
    assert actual.receipt.energy==exact.receipt.energy and actual.receipt.energy_error_bound==0
    assert actual.receipt.forward_shell_calls==actual.receipt.adjoint_shell_calls==0
    np.testing.assert_array_equal(actual.j_error_bound,0.)
    loose=manta.PeriodicBupoSource(t,e,cube(0),options(1e6))
    assert loose.partition.approximate_blocks>0 and loose.partition.traversal.refined_candidates>0
    # Refine central-cell candidates whose periodic representatives overlap.
    geometric=sum(x.pair_interactions for x in t.traversal(.7) if x.far_candidate)
    assert loose.partition.approximate_pairs<geometric


def test_finite_ewald_bound_refuses_wide_extents_at_strict_target():
    b,s,t,e,c,cell=fixture(6,exponent=800.,radius=.25,tolerance=1e-5)
    assert c.partition.approximate_blocks==0 and c.partition.traversal.refined_candidates>0
    d=density(b.nbasis);actual=c.build(d);exact=e.build(d)
    np.testing.assert_array_equal(actual.values,exact.values)
    assert actual.receipt.energy==exact.receipt.energy


def test_order_twelve_and_tiny_density_keep_full_moment_action():
    b,s,t,e,c,cell=fixture(12,images=False,tolerance=1e-7)
    d=density(b.nbasis);actual=c.build(d);exact=e.build(d)
    assert c.partition.approximate_blocks>0
    assert actual.receipt.direct.source_identity!=e.source_identity
    assert actual.receipt.direct.qualification=='finite-ewald-direct-partition-subset'
    assert actual.receipt.direct_energy==pytest.approx(actual.receipt.direct.trace_energy,abs=3e-11)
    assert np.all(abs(actual.values-exact.values)<=actual.j_error_bound[None,:,:]+2e-11)
    tiny=c.build(d*1e-280)
    np.testing.assert_allclose(tiny.values/1e-280,actual.values,atol=3e-11,rtol=3e-12)
    np.testing.assert_allclose(tiny.j_error_bound/1e-280,actual.j_error_bound,atol=0.,rtol=2e-11)
    assert tiny.receipt.energy_error_bound>0


def test_near_partition_transfer_and_trace_moments_keep_one_full_kernel():
    b,s,t,e,c,cell=fixture(6,tolerance=.1)
    d=density(b.nbasis);exact=e.build(d)
    results=[]
    for near in (cube(0),IMAGES[:3].copy(),cube(1)):
        x=manta.PeriodicBupoSource(t,e,near,options(.1));a=x.build(d);results.append(a)
        assert x.partition.approximate_blocks>0
        assert abs(a.receipt.energy-exact.receipt.energy)<=a.receipt.energy_error_bound+2e-11
        assert np.all(abs(a.values-exact.values)<=a.j_error_bound[None,:,:]+2e-11)
    for a in results[1:]: np.testing.assert_allclose(a.values,results[0].values,atol=2e-11,rtol=2e-12)
    assert len({a.receipt.source_identity for a in results})==3
    assert len({a.receipt.kernel_identity for a in results})==1


@pytest.mark.parametrize('field,planned',[('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),
    ('primitive_quartets','primitive_products'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_joint_caps_before_near_payload_reads(field,planned):
    b,s,t,e,c,cell=fixture(4,tolerance=.01)
    p=manta.plan_periodic_bupo(t,e,1,options(.01)); budget=manta.ResourceBudget();setattr(budget,field,getattr(p,planned))
    admitted=manta.PeriodicBupoSource(t,e,cube(0),options(.01),budget);r=admitted.build(density(b.nbasis)).receipt
    assert r.modeled_peak_bytes==p.modeled_peak_bytes
    assert r.direct.real_shell_calls+r.direct.fourier_shell_calls+r.direct.overlap_shell_calls+r.forward_shell_calls+r.adjoint_shell_calls<=p.shell_calls
    setattr(budget,field,getattr(p,planned)-1)
    with pytest.raises((ValueError,RuntimeError),match='budget'):
        manta.PeriodicBupoSource(t,e,np.array([[2**40,0,0]],np.int64),options(.01),budget)


def test_source_ownership_no_density_partition_changes_or_array_aliases():
    b,s,t,e,c,cell=fixture(4,tolerance=.01);d=density(b.nbasis)
    a=c.build(d); options_copy=options(.01);near=cube(0);other=manta.PeriodicBupoSource(t,e,near,options_copy)
    options_copy.integral_tolerance=0;near[:]=17
    assert other.source_identity==c.source_identity
    changed=manta.PeriodicPairSource(b,cell,IMAGES[:3].copy(),K*1.1,WEIGHTS,pair_options(4,cutoff=.2,radius=.25))
    with pytest.raises(ValueError,match='must match'):
        manta.PeriodicBupoSource(t,manta.PeriodicEwaldSource(changed,.4,cube(1),cube(1,reciprocal=True)),cube(0))
    for array in (a.values,a.j_error_bound):array[:]=0
    del b,s,t,e;gc.collect();np.testing.assert_array_equal(c.build(d).values,a.values)
    unaligned=np.ndarray(d.shape,dtype=d.dtype,buffer=bytearray(d.nbytes+1),offset=1);unaligned[:]=d
    np.testing.assert_array_equal(c.build(unaligned).values,a.values)
    for bad,message in [(d.astype(np.complex64),'dtype'),(d[:1].copy(),'shape'),(d.copy(),'Hermitian'),(d.copy(),'finite')]:
        if message=='Hermitian':bad[0,0,1]+=.1j
        if message=='finite':bad[0,0,0]=np.nan
        with pytest.raises(ValueError,match=message):c.build(bad)
    assert not a.receipt.roundoff_included and a.receipt.qualification=='finite-periodic-bupo-j'


@pytest.mark.parametrize('near,message',[(np.zeros((2,3),np.int64),'count'),(np.zeros((3,3),np.int64),'unique'),
    (np.array([[1,0,0]],np.int64),'origin'),(np.array([[0,0,0],[2,0,0],[-2,0,0]],np.int64),'contained'),
    (np.zeros((1,3)),'dtype'),(np.zeros((1,2),np.int64),'shape')])
def test_invalid_near_sets(near,message):
    b,s,t,e,c,cell=fixture(0,tolerance=10.)
    with pytest.raises(ValueError,match=message):manta.PeriodicBupoSource(t,e,near)
