"""Direct complex Bloch local exchange, independent polynomials and libxc."""
import itertools
import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta, _vibeqc_core as core
from tests.test_manta_fourier import _basis, _cart

LATTICE=np.array([[3.8,.4,.2],[0.,4.1,.3],[0.,0.,4.3]])
K=np.array([[.13,-.21,.07],[-.08,.17,.11],[.19,.04,-.12]])
KW=np.array([.2,.3,.5])


def grid(order=6):
    x,w=np.polynomial.legendre.leggauss(order)
    indices=np.array(list(itertools.product(range(order),repeat=3)))
    return np.ascontiguousarray(((x[indices]+1)/2)@LATTICE.T),np.prod(w[indices]/2,axis=1)*np.linalg.det(LATTICE)


def case(spins=2,pure=True,angular=(0,1,2),order=6,reach=1):
    b=_basis(angular,pure); n=b.nbasis
    rng=np.random.default_rng(731)
    z=rng.normal(size=(spins,3,n,n))+1j*rng.normal(size=(spins,3,n,n))
    d=(z@z.conj().swapaxes(-1,-2))/n
    points,weights=grid(order)
    images=np.array(list(itertools.product(range(-reach,reach+1),repeat=3)),float)@LATTICE.T
    return dict(basis=b,lattice=LATTICE.copy(),images=images,momenta=K.copy(),weights=KW.copy(),
                points=points,quadrature_weights=weights,densities=np.ascontiguousarray(d),grid_batch=17)


def polynomial_bloch(c):
    """Full-grid oracle; no native AO evaluator, panels or density folds."""
    chunks=[]
    for shell in c['basis'].shells():
        origin=np.array(shell.origin)
        frac=np.linalg.solve(c['lattice'],origin)
        frac=np.where(abs(frac-np.round(frac))<1e-10,np.round(frac),frac)
        shift=c['lattice']@np.floor(frac)
        powers=_cart(shell.l)
        x=c['points'][:,None,:]-origin-c['images'][None,:,:]+shift
        radius=np.sum(x*x,axis=-1)
        radial=sum(a*np.exp(-z*radius) for z,a in zip(shell.exponents,shell.coefficients))
        polynomial=np.prod(x[:,:,None,:]**powers[None,None,:,:],axis=-1)*radial[:,:,None]
        if shell.pure:
            polynomial=np.einsum('gti,ai->gta',polynomial,manta.cartesian_to_spherical(shell.l))
        phase=np.exp(1j*c['momenta']@(c['images']-shift).T)
        chunks.append(np.einsum('kt,gti->kgi',phase,polynomial))
    return np.concatenate(chunks,axis=-1)


def reference(c):
    b=polynomial_bloch(c); d=c['densities']; qw=c['quadrature_weights']
    rho=np.einsum('k,kgi,skij,kgj->sg',c['weights'],b,d,b.conj()).real
    # Independent libxc implementation of the same local exchange functional.
    func=vq.Functional('slater',spin=len(d))
    if len(d)==1:
        energy,v,*_=func.eval_unpolarised(rho[0],np.empty(0))
        v=np.array([v])
    else:
        energy,va,vb,*_=func.eval_polarised(rho[0],rho[1],np.empty(0),np.empty(0),np.empty(0))
        v=np.array([va,vb])
    potential=np.einsum('g,kgi,sg,kgj->skij',qw,b.conj(),v,b)
    return float(qw@energy),potential,rho@qw


@pytest.mark.parametrize('spins',[1,2])
@pytest.mark.parametrize('pure',[False,True])
def test_polynomial_ao_and_independent_libxc(spins,pure):
    c=case(spins,pure); e,v,ne=reference(c)
    result=manta.compute_periodic_local_exchange(**c)
    assert result.receipt.energy==pytest.approx(e,abs=3e-11)
    np.testing.assert_allclose(result.values,v,atol=3e-12,rtol=3e-12)
    np.testing.assert_allclose(result.receipt.electrons_grid,ne,atol=3e-11)
    r,p=result.receipt,result.plan
    assert r.shell_calls==p.shell_calls
    assert r.primitive_points==p.primitive_points
    assert r.panel_calls==(len(c['points'])+16)//17
    assert r.rounded_negative_points==0
    bloch=polynomial_bloch(c)
    rho=np.einsum('k,kgi,skij,kgj->sg',c['weights'],bloch,c['densities'],bloch.conj()).real
    assert r.minimum_density==pytest.approx(float(rho.min()),abs=1e-12)
    assert r.minimum_density>0


@pytest.mark.parametrize('spins',[1,2])
def test_complex_direction_energy_derivative_and_k_weights(spins):
    c=case(spins,angular=(0,1)); r=manta.compute_periodic_local_exchange(**c)
    rng=np.random.default_rng(511)
    x=rng.normal(size=c['densities'].shape)+1j*rng.normal(size=c['densities'].shape)
    h=x+x.conj().swapaxes(-1,-2)
    for s in range(spins):
        for k in range(3):
            delta=np.zeros_like(h); delta[s,k]=h[s,k]; eps=1e-5
            a=manta.compute_periodic_local_exchange(**dict(c,densities=c['densities']+eps*delta))
            b=manta.compute_periodic_local_exchange(**dict(c,densities=c['densities']-eps*delta))
            expected=c['weights'][k]*np.trace(delta[s,k]@r.values[s,k]).real
            assert (a.receipt.energy-b.receipt.energy)/(2*eps)==pytest.approx(expected,abs=2e-8,rel=2e-8)
    # Imaginary coherences matter at these non-time-reversal momenta.
    real=manta.compute_periodic_local_exchange(**dict(c,densities=c['densities'].real.astype(complex)))
    assert abs(real.receipt.energy-r.receipt.energy)>1e-5


def test_restricted_limit_spin_swap_and_empty_spin():
    c=case(1,angular=(0,1)); r=manta.compute_periodic_local_exchange(**c)
    d=np.repeat(c['densities']/2,2,axis=0)
    u=manta.compute_periodic_local_exchange(**dict(c,densities=d))
    np.testing.assert_allclose(u.values,np.repeat(r.values,2,axis=0),atol=2e-12)
    assert u.receipt.energy==pytest.approx(r.receipt.energy,abs=2e-12)
    d[1]=0
    u=manta.compute_periodic_local_exchange(**dict(c,densities=d))
    swap=manta.compute_periodic_local_exchange(**dict(c,densities=d[::-1].copy()))
    np.testing.assert_allclose(swap.values,u.values[::-1],atol=2e-12)
    np.testing.assert_array_equal(u.values[1],0.)
    assert swap.receipt.energy==u.receipt.energy


def test_independent_shell_image_gauge():
    c=case(angular=(0,1,2)); r=manta.compute_periodic_local_exchange(**c)
    shifts=np.array([[1,-1,0],[-2,0,1],[0,1,1]])@LATTICE.T
    shells=c['basis'].shells(); atoms=[]; phases=[]
    for i,shell in enumerate(shells):
        shell.origin=(np.array(shell.origin)+shifts[i]).tolist()
        atoms.append(vq.Atom(2,shell.origin))
        phases.extend([np.exp(1j*K@shifts[i])]*(2*shell.l+1))
    b=vq.BasisSet(vq.Molecule(atoms,0,1),shells,'shifted-local-exchange',True)
    p=np.array(phases).T
    d=p[None,:,:,None]*c['densities']*p.conj()[None,:,None,:]
    shifted=manta.compute_periodic_local_exchange(**dict(c,basis=b,densities=np.ascontiguousarray(d)))
    expected=p[None,:,:,None]*r.values*p.conj()[None,:,None,:]
    np.testing.assert_allclose(shifted.values,expected,atol=3e-11,rtol=3e-12)
    assert shifted.receipt.energy==pytest.approx(r.receipt.energy,abs=4e-11)


def test_source_identity_ownership_layout_and_batch_invariance():
    c=case(angular=(0,1)); r=manta.compute_periodic_local_exchange(**c); expected=r.values
    for batch in (1,64,1000):
        a=manta.compute_periodic_local_exchange(**dict(c,grid_batch=batch))
        assert a.receipt.source_identity==r.receipt.source_identity
        np.testing.assert_allclose(a.values,expected,atol=2e-12)
    for name in ('images','momenta','points','lattice','weights','quadrature_weights','densities'):
        a=manta.compute_periodic_local_exchange(**dict(c,**{name:np.asfortranarray(c[name])}))
        np.testing.assert_allclose(a.values,expected,atol=2e-12)
    a=manta.compute_periodic_local_exchange(**dict(c,densities=2*c['densities']))
    assert a.receipt.source_identity==r.receipt.source_identity
    assert a.receipt.density_identity!=r.receipt.density_identity
    for name in ('points','momenta','quadrature_weights'):
        a=manta.compute_periodic_local_exchange(**dict(c,**{name:c[name]*1.01}))
        assert a.receipt.source_identity!=r.receipt.source_identity
    c['densities'][:]=0; expected[:]=0
    assert np.linalg.norm(r.values)>1


@pytest.mark.parametrize('field,plan_field',[('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),
    ('primitive_quartets','primitive_points'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_exact_resource_admission_and_one_short(field,plan_field):
    c=case(angular=(0,1)); budget=manta.ResourceBudget(); budget.other_live_bytes=117
    p=manta.plan_periodic_local_exchange(c['basis'],len(c['images']),3,len(c['points']),2,17,budget)
    setattr(budget,field,getattr(p,plan_field))
    manta.compute_periodic_local_exchange(**c,budget=budget)
    setattr(budget,field,getattr(p,plan_field)-1)
    # Poisoned values demonstrate that count-only caps precede payload scans.
    c['densities'][:]=np.nan
    with pytest.raises(ValueError,match='budget'):
        manta.compute_periodic_local_exchange(**c,budget=budget)


@pytest.mark.parametrize('bad',['weights','nan','negative','nonhermitian','image','duplicate','lattice','dtype','layout'])
def test_invalid_source_refusals(bad):
    c=case(angular=(0,))
    if bad=='weights': c['weights']*=2
    if bad=='nan': c['densities'][0,0,0,0]=np.nan
    if bad=='negative': c['densities']*=-1
    if bad=='nonhermitian': c['densities'][0,0,0,0]+=1j
    if bad=='image': c['images'][0,0]+=.01
    if bad=='duplicate': c['images'][0]=c['images'][1]
    if bad=='lattice': c['lattice'][:]=0
    if bad=='dtype': c['densities']=c['densities'].astype(np.complex64)
    if bad=='layout': c['points']=c['points'][::2];c['quadrature_weights']=c['quadrature_weights'][::2]
    with pytest.raises((ValueError,TypeError)):
        manta.compute_periodic_local_exchange(**c)


def test_count_overflow_and_batch_memory_bound():
    b=_basis((0,))
    with pytest.raises(ValueError,match='not addressable'):
        manta.plan_periodic_local_exchange(b,2**63,3,8,2,8)
    with pytest.raises(OverflowError):
        manta.plan_periodic_local_exchange(b,2**32,3,8,2,8)
    small=manta.plan_periodic_local_exchange(b,27,3,1000,2,16)
    large=manta.plan_periodic_local_exchange(b,27,3,2000,2,16)
    assert small.panel_bytes==large.panel_bytes
    assert large.input_bytes-small.input_bytes==64000
    assert large.modeled_peak_bytes-small.modeled_peak_bytes==64000


@pytest.mark.parametrize('pure',[False,True])
def test_highest_admitted_angular_channel(pure):
    c=case(1,pure,angular=(4,),order=4)
    e,v,_=reference(c);r=manta.compute_periodic_local_exchange(**c)
    assert r.receipt.energy==pytest.approx(e,abs=2e-10)
    np.testing.assert_allclose(r.values,v,atol=2e-11,rtol=2e-11)
    with pytest.raises(ValueError,match='angular'):
        manta.plan_periodic_local_exchange(_basis((5,),pure),27,3,64,1,16)


def test_grid_and_image_refinement():
    # Smooth Gaussian source in a cell, asymmetric k mesh and complex D.
    c=case(1,angular=(0,),order=24,reach=3)
    exact=manta.compute_periodic_local_exchange(**c).receipt
    image_errors=[]
    for reach in (0,1,2):
        a=case(1,angular=(0,),order=24,reach=reach)
        image_errors.append(abs(manta.compute_periodic_local_exchange(**a).receipt.energy-exact.energy))
    assert image_errors[0]>1e-3
    assert image_errors[2]<image_errors[1]<image_errors[0]
    assert image_errors[2]<1e-10
    grid_errors=[]
    for order in (6,10,16):
        a=case(1,angular=(0,),order=order,reach=2)
        grid_errors.append(abs(manta.compute_periodic_local_exchange(**a).receipt.energy-exact.energy))
    assert grid_errors[2]<grid_errors[1]<grid_errors[0]
    assert grid_errors[2]<1e-7


def test_zero_density_and_declared_roundoff_extension():
    c=case(2,angular=(0,),order=4)
    c['densities'][:]=0
    zero=manta.compute_periodic_local_exchange(**c)
    assert zero.receipt.energy==0
    assert zero.receipt.minimum_density==0
    np.testing.assert_array_equal(zero.values,0.)
    c['densities'][:]=-1e-15
    tiny=manta.compute_periodic_local_exchange(**c)
    assert tiny.receipt.minimum_density<0
    assert tiny.receipt.rounded_negative_points==2*len(c['points'])
    np.testing.assert_array_equal(tiny.values,0.)
    assert tiny.receipt.energy==0


def test_reciprocal_momentum_relabeling():
    c=case(angular=(0,1));r=manta.compute_periodic_local_exchange(**c)
    reciprocal=2*np.pi*np.linalg.inv(LATTICE).T
    k=c['momenta']+np.array([[1,-2,0],[0,1,1],[-1,0,2]])@reciprocal.T
    shifted=manta.compute_periodic_local_exchange(**dict(c,momenta=k))
    np.testing.assert_allclose(shifted.values,r.values,atol=2e-11)
    assert shifted.receipt.energy==pytest.approx(r.receipt.energy,abs=3e-11)


def test_analytic_single_gaussian_integral_and_euler_identity():
    # Independent all-space Gaussian limit of the explicit finite AO source.
    # The integration grid is deliberately supplied, not a periodic-grid claim.
    a=.8; occupation=1.7; normal=(2*a/np.pi)**.75
    shell=core.ShellInfo();shell.atom_index=0;shell.l=0;shell.pure=True
    shell.origin=[0.,0.,0.];shell.exponents=[a];shell.coefficients=[normal]
    b=vq.BasisSet(vq.Molecule([vq.Atom(2,[0.,0.,0.])],0,1),[shell],'analytic-local-exchange',True)
    x,w=np.polynomial.hermite.hermgauss(24);idx=np.array(list(itertools.product(range(24),repeat=3)))
    exponent=8*a/3
    points=np.ascontiguousarray(x[idx]/np.sqrt(exponent))
    qw=np.prod(w[idx]*np.exp(x[idx]**2)/np.sqrt(exponent),axis=1)
    d=np.full((1,1,1,1),occupation,dtype=complex)
    r=manta.compute_periodic_local_exchange(b,np.eye(3)*50,np.zeros((1,3)),np.zeros((1,3)),np.ones(1),points,qw,d,64)
    expected=-.75*(3/np.pi)**(1/3)*occupation**(4/3)*normal**(8/3)*(np.pi/exponent)**1.5
    assert r.receipt.energy==pytest.approx(expected,abs=2e-13)
    assert r.values[0,0,0,0].real==pytest.approx((4/3)*expected/occupation,abs=2e-13)
    assert r.receipt.electrons_grid[0]==pytest.approx(occupation,abs=2e-11)
