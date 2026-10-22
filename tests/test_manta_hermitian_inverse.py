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
