"""Whole-source infinite-Ewald bounds, with independent complete Coulomb limits."""
from types import SimpleNamespace
import numpy as np
import pytest

from vibeqc import manta
from tests.test_manta_periodic_bupo import fixture, options, cube, density, IMAGES, WEIGHTS, K
from tests.test_manta_periodic_pairs import coefficients, options as pair_options
from tests.test_manta_periodic_bubbles import bubble_options
from tests.test_manta_periodic_ewald import reference
from tests.test_manta_lattice_gaussian import components, fourier
from tests.test_manta_gaussian_domain import make_shell


def bounded(order=6, *, real_radius=1, reciprocal_radius=1, eta=.4, tolerance=1e-5, **kwargs):
    b,s,t,_,_,cell=fixture(order,tolerance=tolerance,**kwargs)
    direct=manta.PeriodicEwaldSource(s,eta,cube(real_radius),cube(reciprocal_radius,reciprocal=True),16)
    o=options(tolerance);o.bound_infinite_domain=True
    c=manta.PeriodicBupoSource(t,direct,cube(0),o)
    return b,s,t,direct,c,cell


def atomic_source(radius,eta=.4):
    b=make_shell();cell=np.eye(3)*3
    s=manta.PeriodicPairSource(b,cell,IMAGES[:3],K,WEIGHTS,pair_options(0,radius=.25))
    t=manta.PeriodicBubbleTree(s,bubble_options())
    e=manta.PeriodicEwaldSource(s,eta,cube(radius),cube(radius,reciprocal=True),16)
    o=options(0.);o.bound_infinite_domain=True
    return b,s,t,e,manta.PeriodicBupoSource(t,e,cube(0),o),cell


def infinite_reference(basis,source,d,cell,cutoff=12,weights=WEIGHTS):
    """Bare reciprocal Coulomb from analytical Gaussian characteristic functions."""
    g=cube(cutoff,reciprocal=True)@(2*np.pi*np.linalg.inv(cell))
    panels=[]
    for e in source.entries:
        a,b=[basis.shells()[i] for i in e.shells]
        b.origin=(np.array(b.origin)+e.translation).tolist()
        pair=SimpleNamespace(shells=lambda a=a,b=b:[a,b])
        raw=components(dict(kind='pair',basis=pair,extent=SimpleNamespace(shells=(0,1))))
        ta=manta.cartesian_to_spherical(a.l) if a.pure else np.eye((a.l+1)*(a.l+2)//2)
        tb=manta.cartesian_to_spherical(b.l) if b.pure else np.eye((b.l+1)*(b.l+2)//2)
        panels.append(np.kron(ta,tb)@np.array([fourier(component,g) for component in raw]))
    f=np.concatenate(panels);c=np.concatenate([coefficients(e,d,weights=weights).ravel() for e in source.entries])
    rho=c@f;lam=4*np.pi/abs(np.linalg.det(cell))/np.sum(g*g,axis=1)
    potential=(f.conj()@(lam*rho)).real;j=np.zeros_like(d);start=0
    for e in source.entries:
        i,k=e.offsets;ni,nk=e.sizes;v=potential[start:start+ni*nk].reshape(ni,nk);start+=ni*nk
        phase=np.exp(1j*K@np.array(e.translation))
        j[:,i:i+ni,k:k+nk]+=phase[:,None,None]*v
        if e.paired:j[:,k:k+nk,i:i+ni]+=phase.conj()[:,None,None]*v.T
    return dict(values=j,energy=float(.5*np.dot(lam,abs(rho)**2)),rho=rho,lambda_=lam)


def aggregate_reference(basis,source,d,cell,eta,real,reciprocal,weights=WEIGHTS):
    entries=source.entries;n=basis.nbasis
    norms=np.array([np.sum(abs(coefficients(e,d,weights=weights))) for e in entries])
    majorants=[manta.compute_pair_gaussian_majorant(basis,e.shells,e.translation) for e in entries]
    real_bounds=np.empty((len(entries),len(entries)));reciprocal_bounds=real_bounds.copy();total=real_bounds.copy()
    for p,a in enumerate(majorants):
        for q,b in enumerate(majorants):
            r=manta.lattice_gaussian_domain_remainder(cell,eta,real,reciprocal,cube(0),a,b)
            real_bounds[p,q]=r.real_remainder;reciprocal_bounds[p,q]=r.reciprocal_remainder;total[p,q]=r.total
    potential=total@norms;j=np.zeros((n,n))
    for p,e in enumerate(entries):
        i,k=e.offsets;ni,nk=e.sizes;j[i:i+ni,k:k+nk]+=potential[p]
        if e.paired:j[k:k+nk,i:i+ni]+=potential[p]
    return dict(j=j,real_energy=float(.5*norms@real_bounds@norms),
                reciprocal_energy=float(.5*norms@reciprocal_bounds@norms),norms=norms,total=total)


@pytest.mark.parametrize('angular',[False,True])
@pytest.mark.parametrize('pure',[False,True])
@pytest.mark.parametrize('skew',[False,True])
def test_complete_retained_source_against_bare_gaussian_fourier_limit(angular,pure,skew):
    b,s,t,e,c,cell=bounded(4,angular=angular,pure=pure,skew=skew,exponent=.7,tolerance=0.)
    d=density(b.nbasis);a=c.build(d);r=a.receipt;ref=infinite_reference(b,s,d,cell)
    assert np.all(abs(a.values-ref['values'])<=a.j_error_bound[None,:,:]+3e-11)
    assert abs(r.energy-ref['energy'])<=r.energy_error_bound+3e-11
    assert r.domain.real_cube_radius==r.domain.reciprocal_cube_radius==1
    assert r.domain.pair_interactions==len(s.entries)*(len(s.entries)+1)//2
    assert r.domain.kernel_identity==e.kernel_identity and r.domain.pair_source_identity==s.receipt.source_identity
    assert r.domain.density_identity==r.density_identity and not r.domain.pair_image_omission_included
    assert r.finite_energy_error_bound==0 and not r.domain.roundoff_included
    np.testing.assert_array_equal(a.values,e.build(d).values)
    assert abs(r.energy-r.trace_energy)<3e-11


def test_independent_reciprocal_limit_is_converged_for_angular_source():
    b,s,t,e,c,cell=bounded(4,angular=True,exponent=.7,tolerance=0.)
    d=density(b.nbasis);a=infinite_reference(b,s,d,cell,12);z=infinite_reference(b,s,d,cell,14)
    np.testing.assert_allclose(a['values'],z['values'],atol=3e-12,rtol=3e-12)
    assert a['energy']==pytest.approx(z['energy'],abs=3e-12)


@pytest.mark.parametrize('order,tolerance',[(0,10.),(6,1e-5),(8,1e-7)])
def test_mixed_bupo_plus_domain_error_against_enlarged_independent_ewald(order,tolerance):
    b,s,t,e,c,cell=bounded(order,tolerance=tolerance)
    d=density(b.nbasis);a=c.build(d)
    o=options(tolerance);finite=manta.PeriodicBupoSource(t,e,cube(0),o);f=finite.build(d)
    ref=reference(b,s,d,cell,.4,cube(5),cube(7,reciprocal=True))
    assert c.partition.approximate_blocks>0 and c.partition.identity==finite.partition.identity
    np.testing.assert_array_equal(a.values,f.values)
    assert a.receipt.energy==f.receipt.energy and a.receipt.finite_energy_error_bound==f.receipt.energy_error_bound
    assert np.all(a.j_error_bound>=a.domain_j_error_bound+f.j_error_bound)
    assert a.receipt.energy_error_bound>=a.receipt.domain.energy_error_bound+f.receipt.energy_error_bound
    assert np.all(abs(a.values-ref['values'])<=a.j_error_bound[None,:,:]+3e-11)
    assert abs(a.receipt.energy-ref['energy'])<=a.receipt.energy_error_bound+3e-11
    assert finite.source_identity!=c.source_identity and f.receipt.domain is None and f.domain_j_error_bound is None


@pytest.mark.parametrize('diagonal_images',[False,True])
def test_component_norm_aggregation_half_diagonal_and_both_scatter_arms(diagonal_images):
    if diagonal_images:
        b,s,t,e,c,cell=atomic_source(2);radius=2
        assert any(x.paired and x.shells[0]==x.shells[1] for x in s.entries)
    else:
        b,s,t,e,c,cell=bounded(4,angular=True,exponent=.7,tolerance=0.);radius=1
    d=density(b.nbasis);a=c.build(d);r=a.receipt.domain
    ref=aggregate_reference(b,s,d,cell,.4,cube(radius),cube(radius,reciprocal=True))
    np.testing.assert_allclose(a.domain_j_error_bound,ref['j'],atol=0.,rtol=2e-9)
    assert r.real_energy_error_bound==pytest.approx(ref['real_energy'],rel=2e-9)
    assert r.reciprocal_energy_error_bound==pytest.approx(ref['reciprocal_energy'],rel=2e-9)
    assert r.energy_error_bound>=ref['real_energy']+ref['reciprocal_energy']


@pytest.mark.parametrize('eta',[.4,.7])
def test_domain_enlargement_reaches_a_declared_whole_source_error(eta):
    d=density(1);previous=np.inf;reached=False
    b,s,t,e,c,cell=atomic_source(1,eta);ref=infinite_reference(b,s,d,cell)
    for radius in (1,4,8,10,12):
        b,s,t,e,c,cell=atomic_source(radius,eta);a=c.build(d);r=a.receipt
        assert r.energy_error_bound<previous;previous=r.energy_error_bound
        assert abs(r.energy-ref['energy'])<=r.energy_error_bound+3e-11
        assert np.all(abs(a.values-ref['values'])<=a.j_error_bound[None,:,:]+3e-11)
        if r.energy_error_bound<1e-8 and r.maximum_j_error_bound<1e-8:
            reached=True;break
    assert reached


def test_density_scaling_zero_weights_tiny_bounds_and_owned_exports():
    weights=np.array([0.,.4,.6]);b,s,t,e,c,cell=bounded(weights=weights)
    d=density(b.nbasis);a=c.build(d);scaled=c.build(-2*d);tiny=c.build(d*1e-280);zero=c.build(np.zeros_like(d))
    np.testing.assert_allclose(scaled.domain_j_error_bound,2*a.domain_j_error_bound,rtol=3e-11,atol=0.)
    assert scaled.receipt.domain.energy_error_bound==pytest.approx(4*a.receipt.domain.energy_error_bound,rel=3e-11)
    np.testing.assert_allclose(tiny.domain_j_error_bound/1e-280,a.domain_j_error_bound,rtol=3e-11,atol=0.)
    assert tiny.receipt.domain.energy_error_bound>0
    assert zero.receipt.energy_error_bound==0;np.testing.assert_array_equal(zero.j_error_bound,0.)
    assert a.receipt.domain.source_identity==scaled.receipt.domain.source_identity
    exported=a.domain_j_error_bound;exported[:]=0
    assert np.max(a.domain_j_error_bound)>0
    assert a.receipt.domain.density_identity!=scaled.receipt.domain.density_identity


@pytest.mark.parametrize('field,planned',[('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),
    ('primitive_quartets','primitive_products'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_joint_domain_caps_before_payloads(field,planned):
    b,s,t,e,c,cell=bounded(4,tolerance=.01)
    o=options(.01);o.bound_infinite_domain=True;p=manta.plan_periodic_bupo(t,e,1,o)
    limits=manta.ResourceBudget();setattr(limits,field,getattr(p,planned))
    good=manta.PeriodicBupoSource(t,e,cube(0),o,limits);a=good.build(density(b.nbasis))
    assert a.plan.domain_pair_interactions==a.receipt.domain.pair_interactions
    assert a.plan.domain_majorant_terms>=a.receipt.domain.majorant_terms
    setattr(limits,field,getattr(p,planned)-1)
    with pytest.raises((ValueError,RuntimeError),match='budget'):
        manta.PeriodicBupoSource(t,e,np.array([[2**40,0,0]],np.int64),o,limits)


def test_domain_source_near_invariance_and_explicit_cube_separation_refusals():
    b,s,t,e,c,cell=bounded();d=density(b.nbasis);a=c.build(d)
    o=options();o.bound_infinite_domain=True
    other=manta.PeriodicBupoSource(t,e,IMAGES[:3],o).build(d)
    assert other.receipt.domain.source_identity==a.receipt.domain.source_identity
    np.testing.assert_array_equal(other.domain_j_error_bound,a.domain_j_error_bound)
    for real,message in ((IMAGES[:3],'complete centered'),(cube(0),'not separated')):
        direct=manta.PeriodicEwaldSource(s,.4,real,cube(1,reciprocal=True),16)
        with pytest.raises(ValueError,match=message):manta.PeriodicBupoSource(t,direct,cube(0),o)
