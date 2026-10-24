"""Stored-matrix spectral enclosures, independent Decimal and perturbation witnesses."""
from decimal import Decimal, localcontext
import numpy as np
import pytest
from vibeqc import manta


def options(absolute=.1, relative=0.):
    o=manta.RIOptions();o.absolute_cutoff=absolute;o.relative_cutoff=relative
    return o


def certify(a, values=None, vectors=None, o=None, budget=None):
    a=np.ascontiguousarray(a,dtype=np.complex128)
    if values is None:values,vectors=np.linalg.eigh(a)
    return manta.certify_hermitian_spectrum(a,np.ascontiguousarray(values),
        np.ascontiguousarray(vectors),o or options(),budget or manta.ResourceBudget())


@pytest.mark.parametrize('scale',[1e-150,1.,1e150])
@pytest.mark.parametrize('relative',[0.,.2])
def test_two_by_two_decimal_eigenvalues_and_complex_residual(scale,relative):
    # Decimal computes eigenvalues of the actual binary64 entries, not the
    # rounded matrix used to synthesize the fixture or another float64 solve.
    a=np.array([[.4, .13+.29j],[.13-.29j,2.7]],complex)*scale
    w,q=np.linalg.eigh(a);q[0,1]+=complex(2e-10,-3e-10)
    c=certify(a,w,q,options(.1*scale,relative))
    assert c.spectrum_certified and c.rank_certified
    with localcontext() as ctx:
        ctx.prec=90;d=Decimal.from_float
        aa,bb=d(a[0,0].real),d(a[1,1].real)
        z=d(a[0,1].real)**2+d(a[0,1].imag)**2
        gap=((aa-bb)**2+4*z).sqrt()
        exact=[(aa+bb-gap)/2,(aa+bb+gap)/2]
        assert all(abs(x-d(float(y)))<=d(c.eigenvalue_error_bound) for x,y in zip(exact,w))
        cut=max(d(.1*scale),d(relative)*max(Decimal(0),exact[-1]))
        assert d(c.cutoff_lower)<=cut<=d(c.cutoff_upper)
        assert c.rank==sum(x>cut for x in exact)
        safe=min(abs(x-cut) for x in exact)/(1+d(relative))
        assert d(c.rank_stability_radius)<=safe


@pytest.mark.parametrize('seed',range(6))
def test_clusters_and_hermitian_perturbations(seed):
    rng=np.random.default_rng(seed);z=rng.normal(size=(6,6))+1j*rng.normal(size=(6,6))
    q,_=np.linalg.qr(z);a=q@np.diag([.01,.01,.7,.7,2.,3.])@q.conj().T
    a=np.tril(a)+np.tril(a,-1).conj().T;a[np.diag_indices(6)]=a.diagonal().real
    w,u=np.linalg.eigh(a);u*=1.00001
    o=options(.05,.1);c=certify(a,w,u,o)
    assert c.spectrum_certified and c.rank_certified and c.rank==4
    assert np.max(abs(np.linalg.eigvalsh(a)-w))<=c.eigenvalue_error_bound
    assert np.linalg.norm(a@u-u*w,2)<=c.residual_bound
    assert np.linalg.norm(u.conj().T@u-np.eye(6),2)<=c.orthogonality_bound
    for _ in range(8):
        z=rng.normal(size=(6,6))+1j*rng.normal(size=(6,6));h=z+z.conj().T
        h*=.99*c.rank_stability_radius/np.linalg.norm(h,2)
        perturbed=np.linalg.eigvalsh(a+h)
        assert np.count_nonzero(perturbed>max(o.absolute_cutoff,o.relative_cutoff*max(0.,perturbed[-1])))==c.rank


def test_cutoff_boundary_and_bad_eigenvectors_never_claim_resolved_rank():
    a=np.diag([.1,2.]).astype(complex)
    c=certify(a)
    assert c.spectrum_certified and not c.rank_certified and c.rank_stability_radius==0
    for q in (np.zeros((2,2),complex),np.array([[1,1],[0,0]],complex)):
        c=certify(a,np.array([.1,2.]),q)
        assert not c.spectrum_certified and not c.rank_certified and c.rank_stability_radius==0


@pytest.mark.parametrize('cutoff,rank',[(0.,2),(3.,0)])
def test_empty_or_full_retained_space(cutoff,rank):
    c=certify(np.diag([.2,2.]),o=options(cutoff))
    assert c.rank_certified and c.rank==rank and c.rank_stability_radius>0


@pytest.mark.parametrize('name,field',[('memory_bytes','modeled_peak_bytes'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_admission_before_payload_and_exact_plan_boundary(name,field):
    o=options();b=manta.ResourceBudget();p=manta.plan_hermitian_spectrum(2,o,b)
    setattr(b,name,getattr(p,field)-1)
    with pytest.raises((ValueError,RuntimeError),match='budget'):
        certify(np.full((2,2),np.nan),np.zeros(2),np.eye(2,dtype=complex),o,b)
    setattr(b,name,getattr(p,field))
    assert certify(np.eye(2),o=o,budget=b).rank_certified


def test_input_contract_sorted_finite_hermitian_contiguous():
    a=np.eye(2,dtype=complex);w=np.array([1.,1.])
    for bad,values,q,pattern in [(a,np.array([2.,1.]),a,'sorted'),
        (np.array([[1,1j],[1j,2]]),w,a,'Hermitian'),(a,np.array([np.nan,1.]),a,'finite')]:
        with pytest.raises(ValueError,match=pattern):certify(bad,values,q)
    with pytest.raises(ValueError,match='complex128'):
        manta.certify_hermitian_spectrum(a.real,w,a)
    with pytest.raises(ValueError,match='contiguous'):
        manta.certify_hermitian_spectrum(np.eye(4,dtype=complex)[::2,::2],w,a)
    f=manta.certify_hermitian_spectrum(np.asfortranarray(a),w,np.asfortranarray(a))
    assert f.rank_certified


@pytest.mark.parametrize('gamma',[True,False])
def test_actual_periodic_owner_exports_certificate_without_changing_factor(gamma):
    from tests.test_manta_periodic_gdf import case
    from tests.test_manta_periodic_ri_q import builder
    b=builder(case(gamma));c=b.metric_info.spectrum
    assert c.spectrum_certified and c.rank_certified and c.rank==b.metric_info.rank
    w=np.linalg.eigvalsh(b.metric)
    assert abs(w[0]-b.metric_info.minimum_eigenvalue)<=c.eigenvalue_error_bound
    assert abs(w[-1]-b.metric_info.maximum_eigenvalue)<=c.eigenvalue_error_bound
    np.testing.assert_allclose(b.factor.conj().T@b.metric@b.factor,np.eye(c.rank),atol=3e-14,rtol=0)


def test_underflow_and_unrepresentable_intervals_are_not_zero_error_certificates():
    tiny=np.nextafter(0.,1.)
    c=certify(np.array([[tiny]],complex),o=options(0.))
    assert c.spectrum_certified and c.eigenvalue_error_bound>0. and not c.rank_certified
    with pytest.raises(OverflowError,match='overflow'):
        certify(np.array([[np.finfo(float).max]],complex),np.array([np.finfo(float).max]),np.ones((1,1),complex))
