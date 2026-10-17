"""Full-image/full-reciprocal exchange enclosures with independent Fourier sums."""
from decimal import Decimal,localcontext
import numpy as np
import pytest
from vibeqc import manta,_vibeqc_core as core
from tests.test_manta_periodic_one_electron_images import fixture
from tests.test_manta_periodic_ewald import cube
from tests.test_manta_fourier import _cart,_quadrature

_NODES,_WEIGHTS=np.polynomial.hermite.hermgauss(8)


def transfers(cell,k,radius,inner_radius=None):
    labels=cube(radius)
    if inner_radius is not None:labels=labels[np.max(abs(labels),axis=1)>inner_radius]
    base=labels@(2*np.pi*np.linalg.inv(cell));chunks=[]
    for target in k:
        for source in k:
            q=base+target-source;chunks.append(q[np.any(q!=0,axis=1)])
    return np.ascontiguousarray(np.concatenate(chunks)),np.r_[0,np.cumsum([len(q) for q in chunks])].astype(np.int64)


def exact_tile(a,b,q):
    """Direct polynomial quadrature on the shifted Gaussian contour, degree <=12.

    Unlike oscillatory real-axis quadrature this remains exact at large Q;
    neither native Fourier recurrences nor envelope coefficients are used.
    """
    nodes,weights=_NODES,_WEIGHTS;ca,cb=_cart(a.l),_cart(b.l)
    A,B=np.array(a.origin),np.array(b.origin);out=np.zeros((len(q),len(ca),len(cb)),complex)
    for alpha,ac in zip(a.exponents,a.coefficients):
        for beta,bc in zip(b.exponents,b.coefficients):
            p=alpha+beta;center=(alpha*A+beta*B)/p
            value=np.ones_like(out)
            for axis in range(3):
                x=nodes[None,:]/np.sqrt(p)+center[axis]-1j*q[:,axis,None]/(2*p)
                pa=(x[:,None,:]-A[axis])**ca[None,:,axis,None]
                pb=(x[:,None,:]-B[axis])**cb[None,:,axis,None]
                value*=np.einsum('qin,qjn,n->qij',pa,pb,weights)/np.sqrt(p)
            factor=ac*bc*np.exp(-alpha*beta/p*np.dot(A-B,A-B)-np.sum(q*q,axis=1)/(4*p)-1j*q@center)
            out+=factor[:,None,None]*value
    if a.pure:out=np.einsum('ai,qij->qaj',manta.cartesian_to_spherical(a.l),out)
    if b.pure:out=np.einsum('qij,bj->qib',out,manta.cartesian_to_spherical(b.l))
    return out


def ordered_fourier(basis,q,cells,k,cutoff):
    shells=basis.shells();sizes=[2*s.l+1 if s.pure else (s.l+1)*(s.l+2)//2 for s in shells]
    offsets=np.cumsum([0,*sizes]);out=np.zeros((len(q),basis.nbasis,basis.nbasis),complex)
    for shift in cells:
        for i,a in enumerate(shells):
            for j,b in enumerate(basis.shells()):
                b.origin=(np.array(b.origin)+shift).tolist()
                if cutoff is not None and np.linalg.norm(np.array(a.origin)-b.origin)>cutoff:continue
                out[:,offsets[i]:offsets[i+1],offsets[j]:offsets[j+1]]+=np.exp(1j*k@shift)*exact_tile(a,b,q)
    return out


def reference(c,d,q,offsets,images,cutoff=None):
    out=np.zeros_like(d);nk=len(c['k']);cells=images@c['cell'].T
    for target,k in enumerate(c['k']):
        for source in range(nk):
            pair=target*nk+source
            for start in range(offsets[pair],offsets[pair+1],4096):
                wave=q[start:min(start+4096,offsets[pair+1])];panel=ordered_fourier(c['basis'],wave,cells,k,cutoff)
                weight=c['weights'][source]*4*np.pi/abs(np.linalg.det(c['cell']))/np.sum(wave*wave,axis=1)
                out[:,target]+=np.einsum('g,gai,sab,gbj->sij',weight,panel.conj(),d[:,source],panel)
    return out


def setup(*,angular=0,pure=True,skew=True,image_radius=0,cutoff=None,reciprocal_radius=2,weights=None):
    k=np.array([[.11,-.07,.03],[-.2,.04,.09]])
    w=np.array([.3,.7]) if weights is None else weights
    c=fixture(angular=angular,pure=pure,skew=skew,images=cube(image_radius),cutoff=cutoff,
              cell_size=6.,exponent=.35,k=k,weights=w)
    q,offsets=transfers(c['cell'],k,reciprocal_radius)
    source=manta.PeriodicExchangeBoundSource(c['source'],q,offsets,reciprocal_radius)
    rng=np.random.default_rng(519);n=c['basis'].nbasis
    d=rng.normal(size=(2,len(k),n,n))+1j*rng.normal(size=(2,len(k),n,n));d=np.ascontiguousarray(d+d.swapaxes(-1,-2).conj())
    return c,q,offsets,source,d


def calculation(**kwargs):
    c,q,o,s,d=setup(**kwargs);bound=s.build(d)
    native=manta.compute_periodic_exchange(c['basis'],c['cell'],q,o,np.ascontiguousarray(c['images']@c['cell'].T),c['k'],c['weights'],d,64,c['cutoff'])
    finite=reference(c,d,q,o,c['images'],c['cutoff'])
    # Separate full-image and reciprocal enlargements as well as the joint limit.
    image=reference(c,d,q,o,cube(2))
    full_q,full_o=transfers(c['cell'],c['k'],12)
    full=reference(c,d,full_q,full_o,cube(2))
    tail_q,tail_o=transfers(c['cell'],c['k'],14,inner_radius=12)
    reciprocal_refinement=reference(c,d,tail_q,tail_o,cube(2))
    probes=np.array([[0.,0.,0.],[.3,-.7,.4],[2.,-1.,.7],[8.,3.,-2.]])
    image_refinement=max(float(np.max(abs(ordered_fourier(c['basis'],probes,cube(3)@c['cell'].T,k,None)
        -ordered_fourier(c['basis'],probes,cube(2)@c['cell'].T,k,None)))) for k in c['k'])
    return dict(context=c,q=q,offsets=o,source=s,d=d,bound=bound,native=native.values,finite=finite,image=image,full=full,
                reciprocal_refinement=reciprocal_refinement,image_refinement=image_refinement)


@pytest.mark.parametrize('angular,pure,skew',[(0,True,False),(1,False,True)])
def test_full_exchange_against_unreduced_independent_limit(angular,pure,skew):
    r=calculation(angular=angular,pure=pure,skew=skew,cutoff=.3 if angular else None)
    assert np.max(abs(r['reciprocal_refinement']))<3e-12 and r['image_refinement']<3e-12
    np.testing.assert_allclose(r['native'],r['finite'],atol=3e-11,rtol=1e-12)
    assert np.all(abs(r['image']-r['native'])<=r['bound'].pair_image_error_bound+3e-11)
    assert np.all(abs(r['full']-r['image'])<=r['bound'].reciprocal_error_bound+3e-11)
    assert np.all(abs(r['full']-r['native'])<=r['bound'].error_bound+3e-11)
    error=-.5*np.einsum('k,skij,skji->s',r['context']['weights'],r['d'],r['full']-r['native']).real
    assert np.all(abs(error)<=np.array(r['bound'].receipt.energy_error_bounds)+3e-11)


@pytest.mark.parametrize('angular,pure',[(0,True),(2,False),(4,False),(6,True)])
def test_fourier_contour_and_angular_image_envelopes(angular,pure):
    c,q,o,s,d=setup(angular=angular,pure=pure,cutoff=.3)
    wave=np.array([[0.,0.,0.],[.3,-.7,.4],[2.,-1.,.7],[8.,3.,-2.]])
    a,b=c['basis'].shells()
    np.testing.assert_allclose(exact_tile(a,b,wave[:3]),_quadrature(a,b,wave[:3]),atol=3e-11,rtol=1e-11)
    finite=ordered_fourier(c['basis'],wave,c['images']@c['cell'].T,c['k'][0],c['cutoff'])
    full=ordered_fourier(c['basis'],wave,cube(2)@c['cell'].T,c['k'][0],None)
    decay=np.exp(-s.receipt.fourier_decay*np.sum(wave*wave,axis=1))[:,None,None]
    assert np.all(abs(finite)<=s.retained_amplitudes[None]*decay+3e-12)
    assert np.all(abs(full-finite)<=s.omitted_amplitudes[None]*decay+3e-12)


def test_density_scaling_zero_weights_ownership_and_positive_underflow():
    c,q,o,s,d=setup(weights=np.array([0.,1.]))
    a=s.build(d);b=s.build(np.ascontiguousarray(-2*d))
    np.testing.assert_allclose(b.error_bound,2*a.error_bound,rtol=2e-11,atol=0)
    np.testing.assert_allclose(b.receipt.energy_error_bounds,4*np.array(a.receipt.energy_error_bounds),rtol=2e-11,atol=0)
    zero=s.build(np.zeros_like(d));assert np.max(zero.error_bound)==0 and max(zero.receipt.energy_error_bounds)==0
    tiny=s.build(np.ascontiguousarray(d*1e-310));assert np.max(tiny.error_bound)>0 and min(tiny.receipt.energy_error_bounds)>0
    unused=d.copy();unused[:,0]*=100;np.testing.assert_array_equal(s.build(unused).error_bound,a.error_bound)
    saved=a.error_bound;a.error_bound[:]=0;np.testing.assert_array_equal(a.error_bound,saved)
    saved=s.omitted_amplitudes;s.omitted_amplitudes[:]=0;np.testing.assert_array_equal(s.omitted_amplitudes,saved)
    with pytest.raises(AttributeError):s.receipt.fourier_decay=1.


@pytest.mark.parametrize('field,planned',[('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),
    ('primitive_quartets','primitive_pairs'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_all_constructor_caps_before_payload_reads(field,planned):
    c,q,o,s,d=setup();b=manta.ResourceBudget();p=s.plan_source(c['source'],len(q),2,1,b)
    setattr(b,field,getattr(p,planned));same=manta.PeriodicExchangeBoundSource(c['source'],q,o,2,b)
    assert same.receipt.source_identity==s.receipt.source_identity
    setattr(b,field,getattr(p,planned)-1)
    with pytest.raises((ValueError,RuntimeError),match='budget'):manta.PeriodicExchangeBoundSource(c['source'],np.full_like(q,np.nan),o,2,b)


@pytest.mark.parametrize('field,planned',[('memory_bytes','modeled_peak_bytes'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_density_caps_before_payload_reads(field,planned):
    c,q,o,s,d=setup();b=manta.ResourceBudget();p=s.plan_build(len(d),b);setattr(b,field,getattr(p,planned));s.build(d,b)
    setattr(b,field,getattr(p,planned)-1)
    with pytest.raises((ValueError,RuntimeError),match='budget'):s.build(np.full_like(d,np.nan),b)


def test_domain_shape_reversal_and_density_refusals():
    c,q,o,s,d=setup()
    for bad in (q.astype(np.float32),q[::2]):
        with pytest.raises((ValueError,TypeError)):manta.PeriodicExchangeBoundSource(c['source'],bad,o,2)
    with pytest.raises(ValueError,match='cube'):manta.PeriodicExchangeBoundSource(c['source'],q,o,1)
    with pytest.raises((ValueError,RuntimeError,OverflowError),match='budget|overflow'):manta.PeriodicExchangeBoundSource(c['source'],q,o,10**6)
    bad=q.copy();bad[0]=bad[1]
    with pytest.raises(ValueError,match='unique'):manta.PeriodicExchangeBoundSource(c['source'],bad,o,2)
    bad=q.copy();bad[0]+=2*np.pi*np.linalg.inv(c['cell']).T[:,0]
    with pytest.raises(ValueError):manta.PeriodicExchangeBoundSource(c['source'],bad,o,3)
    bad=d.copy();bad[0,0,0,1]+=.1
    with pytest.raises(ValueError,match='Hermitian'):s.build(bad)


def hf_case(*,spin='restricted',z=2,shifted=False,q0='strict-zero-mode',radius=30,iterations=40,
            image_radius=1,cutoff=1.,eta=.6):
    from tests.test_manta_periodic_bupo_hf import case
    a,c=case(spin=spin,z=z,shifted=shifted,q0=q0,iterations=iterations,cell_size=6.,
             shell_exponents=[4.,6.] if z!=3 else [4.,6.,9.],exchange_radius=radius,
             cutoff=cutoff,image_radius=image_radius,eta=eta,real_radius=4,reciprocal_radius=10)
    a['options'].bound_exchange=True;a['options'].exchange_bound_cube_radius=radius
    return a,c


def test_cached_reference_preserves_fock_and_orthonormal_scf():
    from tests.test_manta_periodic_bupo_hf import case,independent
    a,c=case(spin='unrestricted',z=3,shifted=True,q0='BvK-ewald',exchange_radius=2)
    direct=independent(a,c,fourier_reference=ordered_fourier)
    cached=independent(a,c,fourier_reference=ordered_fourier,cache_exchange=True,density_metric='orthonormal')
    rng=np.random.default_rng(359);d=rng.normal(size=direct['density'].shape)+1j*rng.normal(size=direct['density'].shape)
    d=d+d.conj().swapaxes(-1,-2)
    original=direct['build'](d);rebuilt=cached['build'](d)
    np.testing.assert_allclose(rebuilt['fock'],original['fock'],atol=3e-11,rtol=1e-12)
    assert abs(rebuilt['energy']-original['energy'])<3e-11
    np.testing.assert_allclose(cached['density'],direct['density'],atol=3e-9,rtol=2e-10)
    assert abs(cached['converged']['energy']-direct['converged']['energy'])<3e-11
    assert cached['density_metric']=='orthonormal' and cached['density_change']<1e-13


@pytest.mark.parametrize('spin,z,q0',[('restricted',2,'strict-zero-mode'),('unrestricted',1,'BvK-ewald')])
def test_complete_hf_explicit_exchange_admission_and_fixed_hamiltonian(spin,z,q0):
    a,c=hf_case(iterations=1,spin=spin,z=z,q0=q0);r=core._manta.run_periodic_bupo_hf(**a)
    assert r.receipt.exchange_accuracy_passed and not r.converged
    assert r.receipt.maximum_observed_exchange_matrix_error<=1e-8
    assert r.receipt.maximum_observed_exchange_energy_error<=1e-8
    assert r.receipt.exchange_bounds.exchange_scale==(.5 if spin=='restricted' else 1.)
    a['options'].bound_exchange=False;old=core._manta.run_periodic_bupo_hf(**a)
    np.testing.assert_array_equal(r.density,old.density);np.testing.assert_array_equal(r.fock,old.fock)
    assert r.receipt.energy==old.receipt.energy and old.receipt.exchange_bounds is None
    for name in ('maximum_exchange_matrix_error','maximum_exchange_energy_error'):
        a,c=hf_case(radius=2,iterations=1);setattr(a['options'],name,1e-300)
        with pytest.raises(ValueError,match='exchange accuracy target'):core._manta.run_periodic_bupo_hf(**a)
