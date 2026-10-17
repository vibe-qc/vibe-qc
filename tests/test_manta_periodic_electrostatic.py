"""Matched electron/point Ewald components, charge policy and full-source errors."""
from math import comb
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.special import erf, erfc, factorial2

from vibeqc import manta
from tests.test_manta_periodic_pairs import K, WEIGHTS, IMAGES, density, coefficients
from tests.test_manta_periodic_image_bounds import fixture, ordered_reference
from tests.test_manta_periodic_ewald import cube
from tests.test_manta_lattice_gaussian import components, fourier

POSITIONS=np.array([[.12,.19,-.08],[.65,-.11,.3]])
CHARGES=np.array([1.25,-.3])


def options(policy='uniform-background'):
    o=manta.PeriodicElectrostaticOptions();o.charge_policy=policy;return o


def point_ewald(cell,positions,charges,eta,real_radius,reciprocal_radius):
    """Independent ordered point Ewald sum, including only point self removal."""
    shifts=cube(real_radius)@cell.T;real=0.
    for a,za in enumerate(charges):
        for b,zb in enumerate(charges):
            r=np.linalg.norm(positions[a]-positions[b]+shifts,axis=1)
            mask=r!=0 if a==b else np.ones(len(r),bool)
            real+=.5*za*zb*np.sum(erfc(eta*r[mask])/r[mask])
    g=cube(reciprocal_radius,reciprocal=True)@(2*np.pi*np.linalg.inv(cell))
    g2=np.sum(g*g,axis=1);rho=np.exp(1j*g@positions.T)@charges
    reciprocal=.5*np.dot(4*np.pi/abs(np.linalg.det(cell))*np.exp(-g2/(4*eta**2))/g2,abs(rho)**2)
    self_=-eta/np.sqrt(np.pi)*np.dot(charges,charges)
    gauge=-np.pi/(2*eta**2*abs(np.linalg.det(cell)))*sum(charges)**2
    return np.array([real,reciprocal,self_,gauge])


def point_integral(terms,points,eta,order=80):
    """Independent polynomial/Gaussian Laplace quadrature of erfc(eta*r)/r."""
    nodes,weights=np.polynomial.legendre.leggauss(order)
    t=(nodes+1)/2;u=eta+t/(1-t);w=weights/2/(1-t)**2
    result=0.
    for p,center,coefficient,polys in terms:
        delta=center-points;radius=np.linalg.norm(delta,axis=1)
        if all(len(poly)==1 for poly in polys):
            effective=(eta**-2+1/p)**-.5;value=np.empty(len(radius));nz=radius!=0
            value[nz]=(erf(np.sqrt(p)*radius[nz])-erf(effective*radius[nz]))/radius[nz]
            value[~nz]=2/np.sqrt(np.pi)*(np.sqrt(p)-effective)
            result+=coefficient*(np.pi/p)**1.5*np.prod([q[0] for q in polys])*sum(value)
        else:
            a=p+u*u;shift=-u[None,:]**2/a[None,:]
            value=(np.pi/a)[None,:]**1.5*np.exp(-p*u[None,:]**2/a[None,:]*np.sum(delta*delta,axis=1)[:,None])
            for axis,poly in enumerate(polys):
                mean=delta[:,axis,None]*shift;expectation=np.zeros_like(mean)
                for n,c in enumerate(poly):
                    for j in range(n//2+1):
                        expectation+=c*comb(n,2*j)*(factorial2(2*j-1) if j else 1)*mean**(n-2*j)/(2*a[None,:])**j
                value*=expectation
            result+=coefficient*2/np.sqrt(np.pi)*np.sum(value@w)
    return result


def finite_nuclear_reference(basis,pairs,cell,positions,charges,*,eta=.45,real_radius=2,reciprocal_radius=3,order=80,k=K):
    """Resolve native canonical component ordering without using native integrals."""
    g=cube(reciprocal_radius,reciprocal=True)@(2*np.pi*np.linalg.inv(cell));g2=np.sum(g*g,axis=1)
    rho=np.exp(1j*g@positions.T)@charges;volume=abs(np.linalg.det(cell))
    lam=4*np.pi/volume*np.exp(-g2/(4*eta**2))/g2
    shifts=cube(real_radius)@cell.T;rows=[];out=np.zeros((3,len(k),basis.nbasis,basis.nbasis),complex)
    overlap=np.zeros_like(out[0])
    for entry in pairs.entries:
        a,b=[basis.shells()[i] for i in entry.shells];b.origin=(np.array(b.origin)+entry.translation).tolist()
        pair=SimpleNamespace(shells=lambda a=a,b=b:[a,b])
        terms=components(dict(kind='pair',basis=pair,extent=SimpleNamespace(shells=(0,1))))
        ta=manta.cartesian_to_spherical(a.l) if a.pure else np.eye((a.l+1)*(a.l+2)//2)
        tb=manta.cartesian_to_spherical(b.l) if b.pure else np.eye((b.l+1)*(b.l+2)//2);transform=np.kron(ta,tb)
        q=transform@np.array([fourier(t,np.zeros((1,3)))[0].real for t in terms])
        real=transform@np.array([-sum(z*point_integral(t,x-shifts,eta,order) for x,z in zip(positions,charges)) for t in terms])
        f=transform@np.array([fourier(t,g) for t in terms])
        reciprocal=-np.real(f.conj()@(lam*rho));gauge=np.pi*sum(charges)/(eta**2*volume)*q
        rows.append(np.array([real,reciprocal,gauge]))
        i,j=entry.offsets;ni,nj=entry.sizes;phase=np.exp(1j*k@entry.translation)
        for target,value in zip([*out,overlap],[real,reciprocal,gauge,q]):
            value=value.reshape(ni,nj);target[:,i:i+ni,j:j+nj]+=phase[:,None,None]*value
            if entry.paired:target[:,j:j+nj,i:i+ni]+=phase.conj()[:,None,None]*value.T
    return dict(values=out.sum(axis=0),components=out,overlap=overlap,component_rows=np.concatenate(rows,axis=1))


def finite_calculation(*,angular=0,pure=True,skew=True,delta=0.,weights=WEIGHTS,order=80):
    b,p,t,e,c,cell=fixture(angular=angular,pure=pure,skew=skew,weights=weights)
    engine=manta.PeriodicElectrostaticSource(c,POSITIONS,CHARGES,options());d=density(b.nbasis)+delta*density(b.nbasis,83)
    out=engine.build(d);r=out.receipt
    ref=finite_nuclear_reference(b,p,cell,POSITIONS,CHARGES,order=order)
    nn=point_ewald(cell,POSITIONS,CHARGES,.45,2,3)
    coeff=np.concatenate([coefficients(entry,d,K,weights).ravel() for entry in p.entries])
    return dict(values=out.values,nuclear_values=engine.nuclear_values,reference_nuclear=ref['values'],
        attraction=np.array([r.attraction_real,r.attraction_reciprocal,r.attraction_gauge]),
        reference_attraction=np.einsum('xkij,k,kji->x',ref['components'],weights,d).real,
        nuclear=np.array([r.nuclear_real,r.nuclear_reciprocal,r.nuclear_self,r.nuclear_gauge]),reference_nuclear_components=nn,
        energy=r.energy,electronic_energy=r.electronic_energy,trace_energy=r.trace_energy,
        overlap=engine.overlap,reference_overlap=ref['overlap'],d=d,weights=weights,cell=cell,
        energy_error_bound=r.energy_error_bound,j_error_bound=out.j_error_bound,
        nuclear_j_error_bound=out.nuclear_j_error_bound,nuclear_domain_error_bound=r.nuclear_domain_error_bound,
        attraction_domain_error_bound=r.attraction_domain_error_bound,attraction_image_error_bound=r.attraction_image_error_bound,
        charge=r.electron_charge,net_charge=r.retained_net_charge,charge_uncertainty=r.charge_uncertainty_bound,
        background_charge=r.background_charge,total_gauge=r.total_gauge,electronic_gauge=r.electronic_gauge,
        electronic_values=out.electronic.values,source_identity=r.source_identity,density_identity=r.density_identity,
        component_rows=ref['component_rows'],coefficients=coeff,positions=POSITIONS,charges=CHARGES)


def tight_fixture(*,eta=.4,image_radius=1,policy='neutral',scale=1.):
    b,p,t,e,c,cell=fixture(two_shells=False,exponent=4.,skew=False,cell_size=6.,images=cube(image_radius),
        real_radius=4,reciprocal_radius=12 if eta>.5 else 8,eta=eta)
    positions=np.array([b.shells()[0].origin]);charges=np.array([1.])
    engine=manta.PeriodicElectrostaticSource(c,positions,charges,options(policy))
    d=np.ones((len(K),1,1),complex);d*=scale/np.einsum('k,kij,kji->',WEIGHTS,d,engine.overlap).real
    return b,engine,d,cell,positions,charges


def tight_reference(basis,d,cell,positions,charges,*,image_radius=2,reciprocal_radius=28):
    """Bare reciprocal electronic terms; separately converged point Ewald sum.

    Closed s-Gaussian characteristic functions over every ordered image,
    streamed in reciprocal batches. No canonical source/integral is used.
    """
    shell=basis.shells()[0];alpha=shell.exponents[0];center=np.array(shell.origin)
    translations=cube(image_radius)@cell.T
    masses=(np.pi/(2*alpha))**1.5*np.exp(-alpha/2*np.sum(translations*translations,axis=1))
    phase=np.exp(-1j*K@translations.T);centers=center+translations/2
    vectors=cube(reciprocal_radius,reciprocal=True)@(2*np.pi*np.linalg.inv(cell))
    j=np.zeros_like(d);v=np.zeros_like(d);ee=en=0.;rho_rows=[];point_rows=[];lambdas=[]
    for start in range(0,len(vectors),4096):
        g=vectors[start:start+4096];g2=np.sum(g*g,axis=1)
        panel=phase@(masses[:,None]*np.exp(1j*centers@g.T))*np.exp(-g2/(8*alpha))[None,:]
        rho=np.einsum('k,k,kq->q',WEIGHTS,d[:,0,0],panel);point=np.exp(1j*g@positions.T)@charges
        lam=4*np.pi/abs(np.linalg.det(cell))/g2
        ee+=.5*np.dot(lam,abs(rho)**2);en-=np.dot(lam,np.real(rho.conj()*point))
        j[:,0,0]+=panel.conj()@(lam*rho);v[:,0,0]-=panel.conj()@(lam*point)
        rho_rows.append(rho);point_rows.append(point);lambdas.append(lam)
    nn=point_ewald(cell,positions,charges,.7,5,14)
    return dict(values=j+v,electronic_values=j,nuclear_values=v,energy=float(ee+en+sum(nn)),
        electronic_energy=float(ee),attraction_energy=float(en),nuclear_components=nn,
        rho=np.concatenate(rho_rows),point_rho=np.concatenate(point_rows),lambda_=np.concatenate(lambdas))


def tight_calculation(*,eta=.4,image_radius=1,policy='neutral',scale=1.,reciprocal_radius=28):
    b,engine,d,cell,positions,charges=tight_fixture(eta=eta,image_radius=image_radius,policy=policy,scale=scale)
    out=engine.build(d);r=out.receipt;ref=tight_reference(b,d,cell,positions,charges,reciprocal_radius=reciprocal_radius)
    return dict(values=out.values,energy=r.energy,trace_energy=r.trace_energy,d=d,weights=WEIGHTS,
        electronic_values=out.electronic.values,electronic_energy=r.electronic_energy,
        nuclear_values=engine.nuclear_values,attraction_energy=r.attraction_energy,nuclear_energy=r.nuclear_energy,
        nuclear=np.array([r.nuclear_real,r.nuclear_reciprocal,r.nuclear_self,r.nuclear_gauge]),
        reference_values=ref['values'],reference_energy=ref['energy'],reference_electronic=ref['electronic_energy'],
        reference_attraction=ref['attraction_energy'],reference_nuclear=sum(ref['nuclear_components']),
        rho=ref['rho'],point_rho=ref['point_rho'],lambda_=ref['lambda_'],
        energy_error_bound=r.energy_error_bound,j_error_bound=out.j_error_bound,
        charge=r.electron_charge,net_charge=r.retained_net_charge,charge_uncertainty=r.charge_uncertainty_bound,
        background_charge=r.background_charge,total_gauge=r.total_gauge,source_identity=r.source_identity,
        density_identity=r.density_identity)


@pytest.mark.parametrize('angular,pure',[(0,True),(1,False),(2,True)])
@pytest.mark.parametrize('skew',[False,True])
def test_finite_components_signed_points_and_contractions(angular,pure,skew):
    row=finite_calculation(angular=angular,pure=pure,skew=skew)
    for a,b in [('nuclear_values','reference_nuclear'),('overlap','reference_overlap'),
                ('attraction','reference_attraction'),('nuclear','reference_nuclear_components')]:
        np.testing.assert_allclose(row[a],row[b],atol=8e-12,rtol=3e-12)
    assert row['energy']==pytest.approx(row['trace_energy'],abs=2e-11)
    assert row['total_gauge']==pytest.approx(row['electronic_gauge']+row['attraction'][2]+row['nuclear'][3],abs=2e-12)
    assert row['background_charge']==-row['net_charge']


@pytest.mark.parametrize('eta',[.4,.7])
def test_complete_neutral_source_against_infinite_bare_reference(eta):
    row=tight_calculation(eta=eta)
    assert row['energy_error_bound']<1e-8 and np.max(row['j_error_bound'])<1e-8
    assert abs(row['energy']-row['reference_energy'])<=row['energy_error_bound']+3e-11
    assert np.all(abs(row['values']-row['reference_values'])<=row['j_error_bound'][None]+3e-11)
    assert abs(row['net_charge'])<1e-14 and row['charge_uncertainty']<1e-10 and row['background_charge']==0


def test_independent_reference_refinement_and_non_neutral_background_limit():
    a=tight_calculation(policy='uniform-background',scale=1.3,reciprocal_radius=28)
    b,engine,d,cell,positions,charges=tight_fixture(policy='uniform-background',scale=1.3)
    refined=tight_reference(b,d,cell,positions,charges,image_radius=3,reciprocal_radius=30)
    np.testing.assert_allclose(a['reference_values'],refined['values'],atol=3e-12,rtol=1e-12)
    assert a['reference_energy']==pytest.approx(refined['energy'],abs=3e-12)
    assert a['energy']==pytest.approx(refined['energy'],abs=3e-11)
    assert a['background_charge']==pytest.approx(.3,abs=1e-14)


def test_neutral_policy_includes_missing_image_charge_and_rejects_non_neutral():
    b,s,d,*_=tight_fixture(image_radius=0)
    assert abs(np.einsum('k,kij,kji->',WEIGHTS,d,s.overlap).real-1)<1e-14
    with pytest.raises(ValueError,match='omitted-image uncertainty'):s.build(d)
    b,s,d,*_=tight_fixture()
    with pytest.raises(ValueError,match='net charge'):s.build(1.01*d)
    row=tight_calculation(image_radius=0,policy='uniform-background')
    assert row['charge_uncertainty']>1e-10
    assert abs(row['energy']-row['reference_energy'])<=row['energy_error_bound']+3e-11
    assert np.all(abs(row['values']-row['reference_values'])<=row['j_error_bound'][None]+3e-11)


def test_weighted_derivative_is_j_plus_v_and_partition_stays_fixed():
    weights=np.array([0.,.4,.6]);b,p,t,e,c,cell=fixture(angular=1,weights=weights)
    s=manta.PeriodicElectrostaticSource(c,POSITIONS,CHARGES,options());d=density(b.nbasis);h=density(b.nbasis,83)
    a=s.build(d);expected=np.einsum('k,kij,kji->',weights,h,a.values).real
    for step in (1e-4,5e-5):
        plus,minus=s.build(d+step*h),s.build(d-step*h)
        assert (plus.receipt.energy-minus.receipt.energy)/(2*step)==pytest.approx(expected,abs=2e-8)
        assert plus.electronic.receipt.partition.identity==a.electronic.receipt.partition.identity
        assert plus.receipt.source_identity==a.receipt.source_identity
    np.testing.assert_allclose(a.values,a.electronic.values+s.nuclear_values,atol=0,rtol=0)


def test_empty_point_source_has_no_gaussian_point_self_correction():
    b,p,t,e,c,cell=fixture(two_shells=False);d=density(1)
    s=manta.PeriodicElectrostaticSource(c,np.empty((0,3)),np.empty(0),options());a=s.build(d);ee=c.build(d)
    np.testing.assert_array_equal(a.values,ee.values)
    assert a.receipt.energy==ee.receipt.energy and a.receipt.nuclear_self==0 and a.receipt.attraction_energy==0
    assert a.receipt.nuclear_domain_error_bound==0


@pytest.mark.parametrize('field,planned',[('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),
    ('primitive_quartets','primitive_products'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_all_joint_caps_precede_position_or_charge_payload_read(field,planned):
    b,p,t,e,c,cell=fixture(two_shells=False);plan=manta.plan_periodic_electrostatic(c,2,options())
    budget=manta.ResourceBudget();setattr(budget,field,getattr(plan,planned))
    s=manta.PeriodicElectrostaticSource(c,POSITIONS,CHARGES,options(),budget);a=s.build(density(1))
    assert a.receipt.modeled_peak_bytes==plan.modeled_peak_bytes
    assert a.receipt.nuclear_shell_calls<=plan.nuclear_shell_calls
    assert a.receipt.fourier_shell_calls<=plan.fourier_shell_calls
    assert a.receipt.point_pair_terms<=plan.point_pair_terms
    setattr(budget,field,getattr(plan,planned)-1)
    with pytest.raises((ValueError,RuntimeError),match='budget'):
        manta.PeriodicElectrostaticSource(c,np.full_like(POSITIONS,np.nan),np.full_like(CHARGES,np.nan),options(),budget)


def test_invalid_charge_options_point_geometry_and_owned_arrays():
    b,p,t,e,c,cell=fixture(two_shells=False)
    for key,value in [('charge_policy','isolated'),('charge_tolerance',0.),('charge_tolerance',np.nan)]:
        o=options();setattr(o,key,value)
        with pytest.raises(ValueError):manta.plan_periodic_electrostatic(c,2,o)
    for positions in (POSITIONS*100, np.array([POSITIONS[0],POSITIONS[0]+cell[:,0]]),np.full_like(POSITIONS,np.nan)):
        with pytest.raises(ValueError):manta.PeriodicElectrostaticSource(c,positions,CHARGES,options())
    for positions,charges in [(POSITIONS.astype(np.float32),CHARGES),(POSITIONS,CHARGES.astype(np.float32)),
                              (POSITIONS[:1],CHARGES),(POSITIONS,np.array([1.,np.inf]))]:
        with pytest.raises(ValueError):manta.PeriodicElectrostaticSource(c,positions,charges,options())
    s=manta.PeriodicElectrostaticSource(c,np.asfortranarray(POSITIONS),CHARGES,options());a=s.build(np.asfortranarray(density(1)))
    for owner,names in [(s,['overlap','nuclear_values']),(a,['values','j_error_bound','nuclear_j_error_bound'])]:
        for name in names:
            x=getattr(owner,name);copy=x.copy();x[:]=123;np.testing.assert_array_equal(getattr(owner,name),copy)
    assert not a.receipt.roundoff_included
    assert a.receipt.electronic_source_identity==c.source_identity
    with pytest.raises(AttributeError):a.receipt.energy=0


def ordered_nuclear_reference(basis,d,cell,positions,charges,*,images=None,reciprocal_radius=12):
    """Unreduced ordered Bloch potential from the bare nonzero-G kernel."""
    images=cube(2) if images is None else images
    g=cube(reciprocal_radius,reciprocal=True)@(2*np.pi*np.linalg.inv(cell))
    lam=4*np.pi/abs(np.linalg.det(cell))/np.sum(g*g,axis=1)
    rho=np.exp(1j*g@positions.T)@charges;result=np.zeros_like(d)
    shells=basis.shells();sizes=[2*s.l+1 if s.pure else (s.l+1)*(s.l+2)//2 for s in shells]
    offsets=np.cumsum([0]+sizes)
    for index in images:
        translation=cell@index;phase=np.exp(1j*K@translation)
        for ia in range(len(shells)):
            for ib in range(len(shells)):
                a,b=basis.shells()[ia],basis.shells()[ib];b.origin=(np.array(b.origin)+translation).tolist()
                pair=SimpleNamespace(shells=lambda a=a,b=b:[a,b])
                raw=components(dict(kind='pair',basis=pair,extent=SimpleNamespace(shells=(0,1))))
                ta=manta.cartesian_to_spherical(a.l) if a.pure else np.eye((a.l+1)*(a.l+2)//2)
                tb=manta.cartesian_to_spherical(b.l) if b.pure else np.eye((b.l+1)*(b.l+2)//2)
                f=np.kron(ta,tb)@np.array([fourier(t,g) for t in raw])
                v=-(f.conj()@(lam*rho)).reshape(sizes[ia],sizes[ib])
                result[:,offsets[ia]:offsets[ia+1],offsets[ib]:offsets[ib+1]]+=phase[:,None,None]*v
    return dict(values=result,energy=float(np.einsum('k,kij,kji->',WEIGHTS,d,result).real))


def test_attraction_full_missing_source_and_target_errors_and_point_domain():
    b,p,t,e,c,cell=fixture(angular=1,cutoff=.3,real_radius=1,reciprocal_radius=1)
    d=density(b.nbasis);s=manta.PeriodicElectrostaticSource(c,POSITIONS,CHARGES,options());a=s.build(d)
    ref=ordered_nuclear_reference(b,d,cell,POSITIONS,CHARGES)
    assert np.max(abs(s.nuclear_values-ref['values']))>1e-3
    assert np.all(abs(s.nuclear_values-ref['values'])<=a.nuclear_j_error_bound[None]+3e-11)
    assert abs(a.receipt.attraction_energy-ref['energy'])<=a.receipt.attraction_domain_error_bound+a.receipt.attraction_image_error_bound+3e-11
    nn=sum(point_ewald(cell,POSITIONS,CHARGES,.7,5,14))
    assert abs(a.receipt.nuclear_energy-nn)>100*3e-11  # Resolve a domain error above roundoff.
    assert abs(a.receipt.nuclear_energy-nn)<=a.receipt.nuclear_domain_error_bound+3e-11
    # Off-diagonal AO target was entirely excluded from the retained source.
    assert np.max(abs(ref['values'][:,0,1:]))>1e-3
    assert np.all(a.nuclear_j_error_bound[0,1:]>0)
    small=ordered_nuclear_reference(b,d,cell,POSITIONS,CHARGES,images=cube(0),reciprocal_radius=12)
    refined=ordered_nuclear_reference(b,d,cell,POSITIONS,CHARGES,images=cube(0),reciprocal_radius=14)
    np.testing.assert_allclose(small['values'],refined['values'],atol=3e-11,rtol=1e-12)


def test_zero_charges_permutation_and_quadrature_refinement():
    b,p,t,e,c,cell=fixture(angular=2);d=density(b.nbasis)
    ref=finite_nuclear_reference(b,p,cell,POSITIONS,CHARGES,order=112)
    base=finite_nuclear_reference(b,p,cell,POSITIONS,CHARGES,order=64)
    np.testing.assert_allclose(base['values'],ref['values'],atol=3e-12,rtol=1e-12)
    original=manta.PeriodicElectrostaticSource(c,POSITIONS,CHARGES,options()).build(d)
    for positions,charges in [(POSITIONS[::-1].copy(),CHARGES[::-1].copy()),
                              (np.vstack([POSITIONS,POSITIONS[0],POSITIONS[0]*1e4]),np.append(CHARGES,[0.,0.]))]:
        a=manta.PeriodicElectrostaticSource(c,positions,charges,options()).build(d)
        np.testing.assert_allclose(a.values,original.values,atol=3e-13,rtol=1e-12)
        assert a.receipt.energy==pytest.approx(original.receipt.energy,abs=3e-13)


def mixed_calculation():
    from tests.test_manta_periodic_bupo import fixture as mixed_fixture, options as bupo_options
    b,p,t,e,finite,cell=mixed_fixture();o=bupo_options();o.bound_infinite_domain=True;o.bound_pair_images=True
    c=manta.PeriodicBupoSource(t,e,cube(0),o);positions=np.array([x.origin for x in b.shells()]);charges=np.ones(len(positions))
    s=manta.PeriodicElectrostaticSource(c,positions,charges,options());d=density(b.nbasis);a=s.build(d)
    ref=finite_nuclear_reference(b,p,cell,positions,charges,eta=.4,real_radius=1,reciprocal_radius=1)
    nn=point_ewald(cell,positions,charges,.4,1,1)
    baseline=finite.build(d);np.testing.assert_array_equal(baseline.values,a.electronic.values)
    assert baseline.receipt.energy==a.receipt.electronic_energy
    return dict(values=a.values,energy=a.receipt.energy,electronic_energy=a.receipt.electronic_energy,
        electronic_values=a.electronic.values,nuclear_values=s.nuclear_values,reference_nuclear=ref['values'],
        attraction=np.array([a.receipt.attraction_real,a.receipt.attraction_reciprocal,a.receipt.attraction_gauge]),
        reference_attraction=np.einsum('xkij,k,kji->x',ref['components'],WEIGHTS,d).real,
        nuclear=np.array([a.receipt.nuclear_real,a.receipt.nuclear_reciprocal,a.receipt.nuclear_self,a.receipt.nuclear_gauge]),
        reference_nuclear_components=nn,d=d,weights=WEIGHTS,trace_energy=a.receipt.trace_energy,
        far_blocks=a.electronic.receipt.partition.approximate_blocks,
        source_identity=a.receipt.source_identity,density_identity=a.receipt.density_identity)


def test_accepted_far_partition_combines_with_matched_nuclear_kernel():
    row=mixed_calculation();assert row['far_blocks']>0
    np.testing.assert_allclose(row['nuclear_values'],row['reference_nuclear'],atol=2e-11,rtol=2e-12)
    np.testing.assert_allclose(row['nuclear'],row['reference_nuclear_components'],atol=3e-11,rtol=1e-12)
    np.testing.assert_allclose(row['attraction'],row['reference_attraction'],atol=3e-11,rtol=2e-12)
    assert row['energy']==pytest.approx(row['trace_energy'],abs=3e-11)
