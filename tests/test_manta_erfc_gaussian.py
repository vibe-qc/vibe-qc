"""Gauge-free erfc Taylor interactions against independent Gaussian integrals."""
from itertools import product

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.special import erfc

from vibeqc import manta
from tests.test_manta_lattice_gaussian import source, moments, components, gaussian_integral


def reference(a, b, omega, quadrature=96):
    aa, bb = components(a), components(b)
    x, w = leggauss(quadrature); t = (x+1)/2
    s, weights = omega+t/(1-t), .5*w/(1-t)**2
    out = np.zeros((len(aa), len(bb)))
    for i, ai in enumerate(aa):
        for j, bj in enumerate(bb):
            for ta, tb in product(ai, bj):
                out[i, j] += 2/np.sqrt(np.pi)*np.dot(weights, gaussian_integral(ta, tb, np.zeros(3), s))
    return out


@pytest.mark.parametrize('kinds', [('aux', 'aux'), ('aux', 'pair'), ('pair', 'aux'), ('pair', 'pair')])
@pytest.mark.parametrize('order', [0, 2, 6, 12])
def test_signed_angular_sources_bound_independent_erfc_integrals(kinds, order):
    a = source(kinds[0], [3.1, -.2, .1], exponent=30., radius=1.3, angular=2, signed=True, spread=True, strict=True)
    b = source(kinds[1], [0., .1, -.1], exponent=40., radius=1.1, angular=1, signed=True, spread=True, strict=True)
    omega = .23
    result = manta.erfc_gaussian_interaction(omega, a['extent'], b['extent'], order)
    value = moments(a, order) @ result.matrix.values @ moments(b, order).T
    exact = reference(a, b, omega)
    np.testing.assert_allclose(exact, reference(a, b, omega, 128), atol=2e-20, rtol=2e-11)
    assert np.max(abs(value-exact)) <= result.error.total+2e-18
    assert result.matrix.gauge_constant == result.error.bounded_kernel_sup == 0.
    assert result.matrix.boundary == 'isolated-erfc'
    assert result.error.qualification == 'isolated-erfc-gaussian-approximation'
    assert not result.error.roundoff_included
    assert result.matrix.source_identity == result.compact.expansion_identity
    assert 0 < result.error.compact_moment_scale <= 1.
    assert result.error.compact_remainder <= result.compact.total
    assert result.error.total >= result.error.compact_remainder+result.error.tail_penetration+result.error.full_moment_tail
    reverse = manta.erfc_gaussian_interaction(omega, b['extent'], a['extent'], order)
    np.testing.assert_array_equal(result.matrix.values, reverse.matrix.values.T)


@pytest.mark.parametrize('omega', [.01, .5, 4.])
def test_no_gauge_cancellation_and_nonharmonic_trace(omega):
    a = source('aux', [3., -.2, .1], exponent=40., radius=1.1, strict=True)
    b = source('pair', [0., 0., 0.], exponent=40., radius=1.1, strict=True)
    result = manta.erfc_gaussian_interaction(omega, a['extent'], b['extent'], 2)
    distance = np.linalg.norm(np.array(a['extent'].center)-b['extent'].center)
    matrix, indices = result.matrix.values, list(map(tuple, result.matrix.indices))
    np.testing.assert_allclose(matrix[0, 0], erfc(omega*distance)/distance, atol=0., rtol=5e-14)
    trace = sum(2*matrix[indices.index(index), 0] for index in [(2, 0, 0), (0, 2, 0), (0, 0, 2)])
    assert trace == pytest.approx(4*omega**3/np.sqrt(np.pi)*np.exp(-(omega*distance)**2), rel=3e-11, abs=1e-19)


@pytest.mark.parametrize('field,quantity', [('memory_bytes','modeled_peak_bytes'), ('scalar_terms','scalar_terms'), ('kernel_terms','kernel_terms')])
def test_joint_budget_before_evaluation(field, quantity):
    budget = manta.ResourceBudget(); p = manta.plan_erfc_gaussian_interaction(4, budget)
    a = source('aux', [3., 0., 0.]); b = source('pair', [0., 0., 0.])
    for cap, name in [('memory_bytes','modeled_peak_bytes'), ('scalar_terms','scalar_terms'), ('kernel_terms','kernel_terms')]:
        setattr(budget, cap, getattr(p, name))
    manta.erfc_gaussian_interaction(.3, a['extent'], b['extent'], 4, budget)
    setattr(budget, field, getattr(p, quantity)-1)
    with pytest.raises((RuntimeError, ValueError), match='budget'):
        manta.erfc_gaussian_interaction(.3, a['extent'], b['extent'], 4, budget)


def test_native_envelope_and_support_requirements():
    a = source('aux', [0., 0., 0.]); b = source('pair', [0., 0., 0.])
    with pytest.raises(ValueError, match='singularity'):
        manta.erfc_gaussian_interaction(.3, a['extent'], b['extent'], 4)
    for omega in [0., -1., float('inf'), float('nan')]:
        with pytest.raises(ValueError):
            manta.erfc_gaussian_interaction(omega, a['extent'], b['extent'], 4)
    b = source('pair', [3., 0., 0.], retain=False)
    with pytest.raises(ValueError, match='retained'):
        manta.erfc_gaussian_interaction(.3, a['extent'], b['extent'], 4)
    b = source('pair', [3., 0., 0.], radius=.01, strict=True)
    with pytest.raises(ValueError, match='always-near'):
        manta.erfc_gaussian_interaction(.3, a['extent'], b['extent'], 4)



def joint_compact_reference(a,b,omega,order):
    """Independent quadrature of the directional polynomial/Gaussian supremum.

    The native implementation uses positive upper-gamma recurrences and logs.
    This reference numerically integrates both positive branches and combines
    each with the older separate polynomial/decay envelope.
    """
    from math import factorial
    from scipy.integrate import quad
    ea,eb=a['extent'],b['extent'];n=order+1
    rho=ea.radius+eb.radius
    distance=np.linalg.norm(np.array(ea.center)-eb.center);d=distance-rho;upper=distance+rho
    total=0.;separate=0.
    for k in range(n//2+1):
        power=n-2*k;m=n-k
        def integrand(s):
            z=max(d,np.sqrt(power/2)/s) if power else d
            return s**(2*m)*z**power*np.exp(-(s*z)**2)
        cut=max(omega,np.sqrt(power/2)/d) if power else omega
        joint=quad(integrand,omega,cut,epsabs=1e-24,epsrel=2e-12)[0]
        joint+=quad(integrand,cut,np.inf,epsabs=1e-24,epsrel=2e-12)[0]
        old=upper**power*quad(lambda s:s**(2*m)*np.exp(-(d*s)**2),omega,np.inf,epsabs=1e-24,epsrel=2e-12)[0]
        factor=2**(power+1)*rho**n/(np.sqrt(np.pi)*factorial(k)*factorial(power))
        total+=factor*min(joint,old);separate+=factor*old
    charge=ea.full_moment_bounds[0]*eb.full_moment_bounds[0]
    return total*charge,separate*charge


@pytest.mark.parametrize('order',[0,1,2,4,7,12])
@pytest.mark.parametrize('omega',[.01,.2,1.1])
def test_joint_radial_bound_covers_independent_supremum_and_compact_charges(order,omega):
    from tests.test_manta_lattice_tail import point_moments
    a=source('aux',[3.2,.1,-.2],exponent=25.,radius=.6)
    b=source('pair',[0.,0.,0.],exponent=30.,radius=.5)
    result=manta.erfc_gaussian_interaction(omega,a['extent'],b['extent'],order)
    expected,previous=joint_compact_reference(a,b,omega,order)
    assert result.compact.total>=expected*(1-2e-12)
    # The positive upper-gamma approximation can exceed its exact integral;
    # its changed radial dependence must still yield a useful improvement.
    assert result.compact.total<previous
    assert result.compact.radial_integrals_evaluated==result.compact.plan.radial_integrals
    pa=.6*np.array([[.7,.2,-.1],[-.3,.4,.6],[.1,-.7,.2]])
    pb=.5*np.array([[.2,-.4,.7],[-.6,.3,.2],[.4,.1,-.5]])
    qa=np.array([.4,-.3,.3])*a['extent'].full_moment_bounds[0]
    qb=np.array([.2,-.4,.4])*b['extent'].full_moment_bounds[0]
    displacement=np.array(a['extent'].center)-b['extent'].center
    distances=np.linalg.norm(displacement+pa[:,None,:]-pb[None,:,:],axis=2)
    exact=qa@(erfc(omega*distances)/distances)@qb
    ma=point_moments(result.matrix.indices,pa,qa);mb=point_moments(result.matrix.indices,pb,qb)
    assert abs(ma@result.matrix.values@mb-exact)<=result.compact.total+1e-17


@pytest.mark.parametrize('order',[0,2,4,8,12])
@pytest.mark.parametrize('diffuse',[False,True])
def test_interior_moment_scale_uses_native_full_moments_without_dropping_tails(order,diffuse):
    from math import comb
    exponent=.03 if diffuse else 30.
    a=source('aux',[5.,0.,0.],exponent=exponent,radius=1.3,angular=2,signed=True)
    b=source('pair',[0.,0.,0.],exponent=1.3*exponent,radius=1.1,angular=1,signed=True,spread=True)
    ea,eb=a['extent'],b['extent'];n=order+1
    def interior(e,j):
        return min(e.radius**(j-k)*e.full_moment_bounds[k] for k in range(min(j,order)+1))
    moment=sum(comb(n,j)*interior(ea,j)*interior(eb,n-j) for j in range(n+1))
    expected=min(1.,moment/((ea.radius+eb.radius)**n*ea.full_moment_bounds[0]*eb.full_moment_bounds[0]))
    result=manta.erfc_gaussian_interaction(.2,ea,eb,order)
    assert result.error.compact_moment_scale>=expected*(1-2e-13)
    assert result.error.compact_moment_scale==pytest.approx(expected,rel=4e-11)
    assert result.error.compact_remainder==pytest.approx(result.compact.total*expected,rel=5e-11)
    if order==0:assert result.error.compact_moment_scale==1.
    elif not diffuse:assert result.error.compact_moment_scale<.15
    penetration=ea.tail_moment_bounds[0]*eb.potential_bound+eb.tail_moment_bounds[0]*ea.potential_bound
    assert result.error.tail_penetration>=penetration
    assert result.error.tail_penetration==pytest.approx(penetration,rel=5e-11)
    if diffuse:assert result.error.full_moment_tail>0.
    actual=moments(a,order)@result.matrix.values@moments(b,order).T
    exact=reference(a,b,.2,128)
    assert np.max(abs(actual-exact))<=result.error.total+2e-12


def test_joint_radial_branch_boundary_and_positive_underflow():
    a=source('aux',[3.,0.,0.],exponent=10000.,radius=.2)
    b=source('pair',[0.,0.,0.],exponent=12000.,radius=.2)
    for order in (0,2,4,12):
        split=np.sqrt((order+1)/2)/2.6
        values=[]
        for omega in (np.nextafter(split,0.),split,np.nextafter(split,np.inf)):
            r=manta.erfc_gaussian_interaction(omega,a['extent'],b['extent'],order)
            values.append(r.compact.total)
            assert r.compact.total>0 and np.isfinite(r.error.total)
        np.testing.assert_allclose(values,values[1],rtol=3e-12,atol=0.)
        r=manta.erfc_gaussian_interaction(1000.,a['extent'],b['extent'],order)
        assert r.compact.total>0 and r.error.compact_remainder>0
