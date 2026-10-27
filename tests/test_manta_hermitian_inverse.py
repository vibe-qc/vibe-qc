"""Stored-factor inverse bounds against Decimal spectra and independent limits."""
from decimal import Decimal, localcontext
import numpy as np
import pytest
from vibeqc import manta
from tests.test_manta_hermitian_spectrum import options


def arrays(a, cutoff=.1):
    a=np.ascontiguousarray(a,dtype=np.complex128)
    w,q=np.linalg.eigh(a);keep=w>cutoff
    return a,w,np.ascontiguousarray(q),np.ascontiguousarray(q[:,keep]/np.sqrt(w[keep]))


def certify(a,w,q,f,o=None,b=None):
    return manta.certify_hermitian_inverse(a,w,q,f,o or options(),b or manta.ResourceBudget())


def decimal_inverse_error(a,f,cutoff):
    # Exact binary64 input entries promoted to 100-digit Decimal. Analytic
    # two-by-two spectral projectors; no second floating-point eigensolver.
    with localcontext() as ctx:
        ctx.prec=100;d=Decimal.from_float
        aa,bb=d(float(a[0,0].real)),d(float(a[1,1].real))
        zr,zi=d(float(a[0,1].real)),d(float(a[0,1].imag))
        gap=((aa-bb)**2+4*(zr*zr+zi*zi)).sqrt()
        lo,hi=(aa+bb-gap)/2,(aa+bb+gap)/2
        dl=1/lo if lo>d(cutoff) else Decimal(0)
        dh=1/hi if hi>d(cutoff) else Decimal(0)
        if gap:
            slope=(dh-dl)/gap
            c0,c1=dl+slope*(aa-lo),dl+slope*(bb-lo)
            cr,ci=slope*zr,slope*zi
        else:c0=c1=dl;cr=ci=Decimal(0)
        b0=b1=br=bi=Decimal(0)
        for k in range(f.shape[1]):
            x,y=d(float(f[0,k].real)),d(float(f[0,k].imag))
            z,t=d(float(f[1,k].real)),d(float(f[1,k].imag))
            b0+=x*x+y*y;b1+=z*z+t*t;br+=x*z+y*t;bi+=y*z-x*t
        x,y,z,t=b0-c0,b1-c1,br-cr,bi-ci
        radius=((x-y)**2+4*(z*z+t*t)).sqrt()
        error=max(abs((x+y-radius)/2),abs((x+y+radius)/2))
        norm=max(abs(dl),abs(dh))
        return error,norm


@pytest.mark.parametrize('scale',[1e-100,1.,1e100])
@pytest.mark.parametrize('cut',[0.,1.,5.])
@pytest.mark.parametrize('perturbed',[False,True])
def test_decimal_actual_factor_full_truncated_empty(scale,cut,perturbed):
    a,w,q,f=arrays(np.array([[.4,.13+.29j],[.13-.29j,2.7]])*scale,cut*scale)
    if perturbed:
        q[0,1]+=complex(2e-8,-3e-8)
        if f.size:f[0,-1]*=complex(1+4e-7,-2e-7)
    c=certify(a,w,q,f,options(cut*scale))
    assert c.qualified and c.spectrum.rank_certified
    error,norm=decimal_inverse_error(a,f,cut*scale)
    assert error<=Decimal.from_float(c.inverse_error_bound)
    assert norm<=Decimal.from_float(c.inverse_norm_bound)
    if not f.size:assert c.inverse_error_bound==0.
    else:assert c.factor_product_bound>0. and c.inverse_error_bound>0.


@pytest.mark.parametrize('cutoff',[0.,.1,4.])
def test_complex_clusters_scaled_vectors_and_arbitrary_factor(cutoff):
    rng=np.random.default_rng(88);z=rng.normal(size=(6,6))+1j*rng.normal(size=(6,6))
    u,_=np.linalg.qr(z);a=u@np.diag([.01,.01,.7,.7,2.,3.])@u.conj().T
    a=np.tril(a)+np.tril(a,-1).conj().T;a[np.diag_indices(6)]=a.diagonal().real
    a,w,q,f=arrays(a,cutoff);exact=(q[:,w>cutoff]/w[w>cutoff])@q[:,w>cutoff].conj().T
    q*=1.00001
    if f.size:f+=(rng.normal(size=f.shape)+1j*rng.normal(size=f.shape))*.003
    c=certify(a,w,q,f,options(cutoff))
    assert c.qualified
    assert np.linalg.norm(f@f.conj().T-exact,2)<=c.inverse_error_bound+1e-13
    assert c.residual_frobenius_bound>=np.linalg.norm(a@q-q*w,'fro')


def test_rank_crossing_bad_vectors_and_factor_contract():
    a,w,q,f=arrays(np.diag([.1,2.]))
    assert not certify(a,w,q,f).qualified
    a,w,q,f=arrays(np.diag([.2,2.]))
    assert not certify(a,w,np.zeros_like(q),f).qualified
    for bad in (f[:,:1].copy(),np.full_like(f,np.nan)):
        with pytest.raises(ValueError,match='factor'):certify(a,w,q,bad)
    with pytest.raises(ValueError,match='complex128'):certify(a,w,q,f.real)
    assert certify(np.asfortranarray(a),w,np.asfortranarray(q),np.asfortranarray(f)).qualified


@pytest.mark.parametrize('name,field',[('memory_bytes','modeled_peak_bytes'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_pre_payload_admission_and_exact_boundaries(name,field):
    o=options();b=manta.ResourceBudget();plan=manta.plan_hermitian_inverse(2,o,b)
    a,w,q,f=arrays(np.eye(2));setattr(b,name,getattr(plan,field)-1)
    with pytest.raises((RuntimeError,ValueError),match='budget'):certify(np.full_like(a,np.nan),w,q,f,o,b)
    setattr(b,name,getattr(plan,field));assert certify(a,w,q,f,o,b).qualified


@pytest.mark.parametrize('gamma',[True,False])
def test_periodic_owner_uses_actual_stored_factor_and_shared_spectrum(gamma):
    from tests.test_manta_periodic_gdf import case
    from tests.test_manta_periodic_ri_q import builder
    b=builder(case(gamma));c=b.metric_info.inverse_certificate
    assert c.qualified and c.spectrum.rank==b.metric_info.rank
    assert c.spectrum.eigenvalue_error_bound==b.metric_info.spectrum.eigenvalue_error_bound
    a,w,q,f=arrays(b.metric,b.metric_info.cutoff)
    exact=f@f.conj().T
    assert np.linalg.norm(b.factor@b.factor.conj().T-exact,2)<=c.inverse_error_bound
    np.testing.assert_allclose(b.factor.conj().T@b.metric@b.factor,np.eye(c.spectrum.rank),atol=3e-14,rtol=0)


def test_nonrepresentable_inverse_bound_refuses():
    a,w,q,f=arrays(np.array([[1e-310]]),0.)
    # The spectrum resolves this positive scalar, but its inverse is too large.
    with pytest.raises(OverflowError,match='overflow'):certify(a,w,q,f,options(0.))



def _decimal_spectral_function(a,cutoff,projector=False):
    d=Decimal.from_float
    aa,bb=d(float(a[0,0].real)),d(float(a[1,1].real));zr,zi=d(float(a[0,1].real)),d(float(a[0,1].imag))
    gap=((aa-bb)**2+4*(zr*zr+zi*zi)).sqrt();lo,hi=(aa+bb-gap)/2,(aa+bb+gap)/2
    def f(x):return (Decimal(1) if projector else 1/x) if x>d(cutoff) else Decimal(0)
    fl,fh=f(lo),f(hi)
    slope=(fh-fl)/gap if gap else Decimal(0)
    return fl+slope*(aa-lo),fl+slope*(bb-lo),slope*zr,slope*zi


def _decimal_hermitian_norm(entries):
    a,b,r,i=entries;gap=((a-b)**2+4*(r*r+i*i)).sqrt()
    return max(abs((a+b-gap)/2),abs((a+b+gap)/2))


@pytest.mark.parametrize('scale',[1e-80,1.,1e80])
@pytest.mark.parametrize('cut',[0.,1.,5.])
def test_finite_source_response_decimal_includes_projector_rotation(scale,cut):
    a,w,q,f=arrays(np.array([[.4,.13+.29j],[.13-.29j,2.7]])*scale,cut*scale)
    if f.size:f[0,-1]*=complex(1.00001,-.00002)
    base=certify(a,w,q,f,options(cut*scale));t=.01*scale;c=base.bound_perturbation(t)
    assert c.qualified and c.rank==base.spectrum.rank and c.dimension==2
    b=a+np.array([[.001,.002-.003j],[.002+.003j,-.002]])*scale
    with localcontext() as ctx:
        ctx.prec=250
        def diff(fn):return _decimal_hermitian_norm([y-x for x,y in zip(fn(a),fn(b))])
        observed=diff(lambda z:_decimal_spectral_function(z,cut*scale))
        rotation=diff(lambda z:_decimal_spectral_function(z,cut*scale,True))
        assert observed<=Decimal.from_float(c.inverse_error_bound)
        assert rotation<=Decimal.from_float(c.projector_error_bound)
        error,_=decimal_inverse_error(b,f,cut*scale)
        assert error<=Decimal.from_float(c.factor_inverse_error_bound)
        if cut==1.:assert rotation>0 and c.gap_lower>0
        else:assert c.projector_error_bound==0
    zero=base.bound_perturbation(0.)
    assert zero.qualified and zero.projector_error_bound==zero.inverse_error_bound==0
    assert zero.factor_inverse_error_bound==base.inverse_error_bound


@pytest.mark.parametrize('relative',[0.,.1,.9,1.1])
def test_finite_source_response_clusters_and_adaptive_cutoff(relative):
    rng=np.random.default_rng(257);z=rng.normal(size=(6,6))+1j*rng.normal(size=(6,6));u,_=np.linalg.qr(z)
    a=u@np.diag([.01,.01,.7,.7,2.,3.])@u.conj().T
    a=np.tril(a)+np.tril(a,-1).conj().T;a[np.diag_indices(6)]=a.diagonal().real
    cutoff=max(.001,relative*3.);a,w,q,f=arrays(a,cutoff);o=options(.001);o.relative_cutoff=relative
    base=certify(a,w,q,f,o);t=min(.002,base.spectrum.rank_stability_radius/3);c=base.bound_perturbation(t)
    assert c.qualified
    keep=w>max(o.absolute_cutoff,relative*w[-1]);pa=q[:,keep]@q[:,keep].conj().T;ia=(q[:,keep]/w[keep])@q[:,keep].conj().T
    for _ in range(8):
        x=rng.normal(size=(6,6))+1j*rng.normal(size=(6,6));e=x+x.conj().T;e*=t/(2*np.linalg.norm(e,2))
        v,r=np.linalg.eigh(a+e);kb=v>max(o.absolute_cutoff,relative*v[-1]);assert kb.sum()==keep.sum()
        pb=r[:,kb]@r[:,kb].conj().T;ib=(r[:,kb]/v[kb])@r[:,kb].conj().T
        assert np.linalg.norm(pb-pa,2)<=c.projector_error_bound+2e-14
        assert np.linalg.norm(ib-ia,2)<=c.inverse_error_bound+2e-12
        assert np.linalg.norm(f@f.conj().T-ib,2)<=c.factor_inverse_error_bound+2e-12


def test_finite_source_response_strict_refusals_and_subnormal_perturbation():
    a,w,q,f=arrays(np.diag([.2,2.]));base=certify(a,w,q,f)
    for t in (base.spectrum.rank_stability_radius,2*base.spectrum.rank_stability_radius):
        assert not base.bound_perturbation(t).qualified
    for t in (-1.,np.inf,np.nan):
        with pytest.raises(ValueError,match='nonnegative finite'):base.bound_perturbation(t)
    a,w,q,f=arrays(np.diag([.1,2.]))
    assert not certify(a,w,q,f).bound_perturbation(0.).qualified
    a,w,q,f=arrays(np.array([[.5]]),0.);base=certify(a,w,q,f,options(0.));t=np.nextafter(0.,1.)
    c=base.bound_perturbation(t);assert c.qualified and c.inverse_error_bound>0
    with localcontext() as ctx:
        ctx.prec=400
        observed=abs(1/(Decimal('.5')+Decimal.from_float(t))-2)
        assert 0<observed<=Decimal.from_float(c.inverse_error_bound)


@pytest.mark.parametrize('name,field',[('memory_bytes','modeled_peak_bytes'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_finite_source_response_exact_caps_before_payload(name,field):
    base=certify(*arrays(np.eye(2)));b=manta.ResourceBudget();p=manta.plan_hermitian_inverse_perturbation(b)
    setattr(b,name,getattr(p,field)-1)
    with pytest.raises((ValueError,RuntimeError),match='budget'):base.bound_perturbation(np.nan,b)
    setattr(b,name,getattr(p,field));assert base.bound_perturbation(.001,b).qualified
