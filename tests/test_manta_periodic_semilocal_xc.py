"""Complex finite-source LDA/GGA with independent Gaussian gradient algebra."""
import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta
from tests.test_manta_periodic_local_exchange import case, _cart, LATTICE, K


def bloch_values_gradients(c):
    values=[];gradients=[[],[],[]]
    for shell in c['basis'].shells():
        origin=np.array(shell.origin);frac=np.linalg.solve(c['lattice'],origin)
        frac=np.where(abs(frac-np.round(frac))<1e-10,np.round(frac),frac)
        shifts=c['lattice']@np.floor(frac)
        x=c['points'][:,None,:]-origin-c['images'][None,:,:]+shifts
        r2=np.sum(x*x,axis=-1);powers=_cart(shell.l)
        radial=sum(a*np.exp(-z*r2) for z,a in zip(shell.exponents,shell.coefficients))
        radial_derivative=sum(-2*z*a*np.exp(-z*r2) for z,a in zip(shell.exponents,shell.coefficients))
        monomial=np.prod(x[:,:,None,:]**powers[None,None,:,:],axis=-1)
        ao=monomial*radial[:,:,None];derivative=[]
        for axis in range(3):
            lowered=powers.copy();lowered[:,axis]=np.maximum(0,lowered[:,axis]-1)
            polynomial=powers[:,axis]*np.prod(x[:,:,None,:]**lowered[None,None,:,:],axis=-1)
            derivative.append(polynomial*radial[:,:,None]+monomial*x[:,:,axis,None]*radial_derivative[:,:,None])
        if shell.pure:
            transform=manta.cartesian_to_spherical(shell.l)
            ao=np.einsum('gti,ai->gta',ao,transform)
            derivative=[np.einsum('gti,ai->gta',g,transform) for g in derivative]
        phase=np.exp(1j*c['momenta']@(c['images']-shifts).T)
        values.append(np.einsum('kt,gti->kgi',phase,ao))
        for axis in range(3):gradients[axis].append(np.einsum('kt,gti->kgi',phase,derivative[axis]))
    return np.concatenate(values,axis=-1),np.array([np.concatenate(x,axis=-1) for x in gradients])


def reference(c,f):
    b,g=bloch_values_gradients(c);d=c['densities'];kw=c['weights'];qw=c['quadrature_weights']
    rho=np.einsum('k,kgi,skij,kgj->sg',kw,b,d,b.conj()).real
    grad=(np.einsum('k,akgi,skij,kgj->sag',kw,g,d,b.conj())
          +np.einsum('k,kgi,skij,akgj->sag',kw,b,d,g.conj())).real
    if len(d)==1:
        e,v,vs=f.eval_unpolarised(rho[0],np.sum(grad[0]**2,axis=0))
        flow=2*vs[None,None,:]*grad if len(vs) else np.zeros_like(grad)
        v=np.array([v])
    else:
        aa=np.sum(grad[0]**2,axis=0);ab=np.sum(grad[0]*grad[1],axis=0);bb=np.sum(grad[1]**2,axis=0)
        e,va,vb,vaa,vab,vbb=f.eval_polarised(rho[0],rho[1],aa,ab,bb)
        v=np.array([va,vb]);flow=np.zeros_like(grad)
        if len(vaa):
            flow[0]=2*vaa*grad[0]+vab*grad[1]
            flow[1]=2*vbb*grad[1]+vab*grad[0]
    potential=np.einsum('g,kgi,sg,kgj->skij',qw,b.conj(),v,b)
    potential+=np.einsum('g,akgi,sag,kgj->skij',qw,g.conj(),flow,b)
    potential+=np.einsum('g,kgi,sag,akgj->skij',qw,b.conj(),flow,g)
    return float(qw@e),potential,rho@qw


@pytest.mark.parametrize('name',['lda','pbe','blyp','pbesol','bpw91'])
@pytest.mark.parametrize('spins',[1,2])
@pytest.mark.parametrize('pure',[False,True])
def test_independent_complex_ao_gradient_contraction(name,spins,pure):
    c=case(spins,pure,angular=(0,1,2),order=5);f=vq.Functional(name,spin=spins)
    e,v,ne=reference(c,f);r=manta.compute_periodic_semilocal_xc(**c,functional=f)
    assert r.receipt.energy==pytest.approx(e,abs=4e-11)
    np.testing.assert_allclose(r.values,v,atol=5e-12,rtol=4e-12)
    np.testing.assert_allclose(r.receipt.electrons_grid,ne,atol=3e-11)
    assert r.plan.gradients==(name!='lda')
    assert r.receipt.functional_batches==r.receipt.panel_calls
    assert r.receipt.component_ids and r.receipt.backend_version
    assert r.receipt.shell_calls==r.plan.shell_calls
    assert r.receipt.primitive_points==r.plan.primitive_points


@pytest.mark.parametrize('name',['lda','pbe','blyp'])
@pytest.mark.parametrize('spins',[1,2])
def test_joint_energy_real_and_imaginary_derivatives(name,spins):
    c=case(spins,angular=(0,1),order=5);f=vq.Functional(name,spin=spins)
    r=manta.compute_periodic_semilocal_xc(**c,functional=f)
    for spin in range(spins):
        for imaginary in (False,True):
            delta=np.zeros_like(c['densities']);delta[spin,1,0,2]=1j if imaginary else 1
            delta[spin,1,2,0]=delta[spin,1,0,2].conjugate();h=2e-5
            energies=[]
            for shift in (-2,-1,1,2):
                q=manta.compute_periodic_semilocal_xc(**dict(c,densities=c['densities']+shift*h*delta),functional=f)
                energies.append(q.receipt.energy)
            slope=(energies[0]-8*energies[1]+8*energies[2]-energies[3])/(12*h)
            expected=c['weights'][1]*np.trace(delta[spin,1]@r.values[spin,1]).real
            assert slope==pytest.approx(expected,abs=4e-9,rel=3e-8)


@pytest.mark.parametrize('name',['lda','pbe','blyp'])
def test_spin_restricted_limit_exchange_and_empty_spin(name):
    c=case(1,angular=(0,1));f=vq.Functional(name,spin=1)
    restricted=manta.compute_periodic_semilocal_xc(**c,functional=f)
    d=np.repeat(c['densities']/2,2,axis=0);f=vq.Functional(name,spin=2)
    u=manta.compute_periodic_semilocal_xc(**dict(c,densities=d),functional=f)
    assert u.receipt.energy==pytest.approx(restricted.receipt.energy,abs=5e-11)
    np.testing.assert_allclose(u.values,np.repeat(restricted.values,2,axis=0),atol=3e-11)
    d[1]=0;u=manta.compute_periodic_semilocal_xc(**dict(c,densities=d),functional=f)
    swapped=manta.compute_periodic_semilocal_xc(**dict(c,densities=d[::-1].copy()),functional=f)
    assert swapped.receipt.energy==pytest.approx(u.receipt.energy,abs=2e-12)
    np.testing.assert_allclose(swapped.values,u.values[::-1],atol=2e-12)
    assert np.all(np.isfinite(u.values))


def test_dirac_limit_preserves_original_component():
    c=case(2);f=vq.Functional('slater',spin=2)
    old=manta.compute_periodic_local_exchange(**c)
    new=manta.compute_periodic_semilocal_xc(**c,functional=f)
    np.testing.assert_allclose(new.values,old.values,atol=4e-12)
    assert new.receipt.energy==pytest.approx(old.receipt.energy,abs=4e-12)


@pytest.mark.parametrize('name',['lda','pbe'])
def test_independent_shell_image_gauge_and_reciprocal_relabel(name):
    c=case(2,angular=(0,1,2),order=5);f=vq.Functional(name,spin=2)
    r=manta.compute_periodic_semilocal_xc(**c,functional=f)
    shifts=np.array([[1,-1,0],[-2,0,1],[0,1,1]])@LATTICE.T
    shells=c['basis'].shells();atoms=[];phases=[]
    for i,s in enumerate(shells):
        s.origin=(np.array(s.origin)+shifts[i]).tolist();atoms.append(vq.Atom(2,s.origin))
        phases.extend([np.exp(1j*K@shifts[i])]*(2*s.l+1))
    b=vq.BasisSet(vq.Molecule(atoms,0,1),shells,'shifted-semilocal',True);p=np.array(phases).T
    action=p[None,:,:,None]*p.conj()[None,:,None,:]
    shifted=manta.compute_periodic_semilocal_xc(**dict(c,basis=b,densities=np.ascontiguousarray(action*c['densities'])),functional=f)
    np.testing.assert_allclose(shifted.values,action*r.values,atol=5e-11)
    assert shifted.receipt.energy==pytest.approx(r.receipt.energy,abs=5e-11)
    momenta=c['momenta']+np.array([[1,-1,0],[0,1,1],[-1,0,2]])@(2*np.pi*np.linalg.inv(LATTICE))
    shifted=manta.compute_periodic_semilocal_xc(**dict(c,momenta=momenta),functional=f)
    np.testing.assert_allclose(shifted.values,r.values,atol=4e-11)


def test_identity_batch_layout_and_ownership():
    c=case(2,angular=(0,1),order=5);f=vq.Functional('pbe',spin=2)
    r=manta.compute_periodic_semilocal_xc(**c,functional=f);v=r.values
    for batch in (1,64,1000):
        a=manta.compute_periodic_semilocal_xc(**dict(c,grid_batch=batch),functional=f)
        assert a.receipt.source_identity==r.receipt.source_identity
        np.testing.assert_allclose(a.values,v,atol=3e-12)
    for key in ('points','images','momenta','lattice','weights','quadrature_weights','densities'):
        a=manta.compute_periodic_semilocal_xc(**dict(c,**{key:np.asfortranarray(c[key])}),functional=f)
        np.testing.assert_allclose(a.values,v,atol=3e-12)
    a=manta.compute_periodic_semilocal_xc(**c,functional=vq.Functional('pbesol',spin=2))
    assert a.receipt.source_identity!=r.receipt.source_identity
    a=manta.compute_periodic_semilocal_xc(**dict(c,densities=1.01*c['densities']),functional=f)
    assert a.receipt.source_identity==r.receipt.source_identity
    assert a.receipt.density_identity!=r.receipt.density_identity
    c['densities'][:]=0;v[:]=0;assert np.linalg.norm(r.values)>1


@pytest.mark.parametrize('name',['lda','pbe'])
@pytest.mark.parametrize('budget_key,plan_key',[('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),
    ('primitive_quartets','primitive_points'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_aggregate_resource_boundaries(name,budget_key,plan_key):
    c=case(2,angular=(0,1),order=4);f=vq.Functional(name,spin=2)
    budget=manta.ResourceBudget();budget.other_live_bytes=223
    p=manta.plan_periodic_semilocal_xc(c['basis'],f,len(c['images']),3,len(c['points']),17,budget)
    setattr(budget,budget_key,getattr(p,plan_key));manta.compute_periodic_semilocal_xc(**c,functional=f,budget=budget)
    setattr(budget,budget_key,getattr(p,plan_key)-1);c['densities'][:]=np.nan
    with pytest.raises(ValueError,match='budget'):
        manta.compute_periodic_semilocal_xc(**c,functional=f,budget=budget)


@pytest.mark.parametrize('name',['pbe0','b3lyp','wb97x','scan','b2plyp','wb97x-v'])
def test_incomplete_functional_refusal_before_density_copy(name):
    c=case(2,angular=(0,),order=3);c['densities'][:]=np.nan;f=vq.Functional(name,spin=2)
    with pytest.raises(ValueError,match='complete native pure'):
        manta.compute_periodic_semilocal_xc(**c,functional=f)
    with pytest.raises(ValueError,match='complete native pure'):
        manta.plan_periodic_semilocal_xc(c['basis'],f,27,3,27,17)


def test_spin_mismatch_and_bounded_workspace():
    c=case(1,angular=(0,),order=3);f=vq.Functional('pbe',spin=2)
    with pytest.raises(ValueError,match='spin'):
        manta.compute_periodic_semilocal_xc(**c,functional=f)
    a=manta.plan_periodic_semilocal_xc(c['basis'],f,27,3,1000,16)
    b=manta.plan_periodic_semilocal_xc(c['basis'],f,27,3,2000,16)
    assert a.panel_bytes==b.panel_bytes and a.functional_workspace_bytes==b.functional_workspace_bytes
    assert b.modeled_peak_bytes-a.modeled_peak_bytes==64000


@pytest.mark.parametrize('name',['50','19'])
def test_kinetic_and_low_dimensional_functionals_are_not_relabelled(name):
    # libxc IDs: Thomas-Fermi kinetic and 2D local exchange, respectively.
    c=case(1,angular=(0,),order=3);f=vq.Functional(name)
    with pytest.raises(ValueError,match='3D exchange/correlation'):
        manta.compute_periodic_semilocal_xc(**c,functional=f)


def test_external_provider_is_refused_without_evaluation():
    def provider(_):
        pytest.fail('Full-grid provider must not execute in a pointwise source')
    vq.define_external_functional('test-manta-semilocal-full-grid-refusal',provider)
    f=vq.Functional('test-manta-semilocal-full-grid-refusal',spin=2)
    c=case(2,angular=(0,),order=3)
    with pytest.raises(ValueError,match='complete native pure'):
        manta.compute_periodic_semilocal_xc(**c,functional=f)


@pytest.mark.parametrize('name',['lda','pbe','blyp'])
def test_zero_density_and_negative_roundoff_chain_rule(name):
    c=case(2,angular=(0,),order=3);f=vq.Functional(name,spin=2)
    c['densities'][:]=0
    r=manta.compute_periodic_semilocal_xc(**c,functional=f)
    assert r.receipt.energy==0
    np.testing.assert_array_equal(r.values,0.)
    c['densities'][:]=-1e-15
    r=manta.compute_periodic_semilocal_xc(**c,functional=f)
    assert r.receipt.energy==0 and r.receipt.rounded_negative_points==2*len(c['points'])
    np.testing.assert_array_equal(r.values,0.)


@pytest.mark.parametrize('pure',[False,True])
def test_highest_gradient_angular_channel(pure):
    c=case(2,pure,angular=(4,),order=4);f=vq.Functional('pbe',spin=2)
    e,v,_=reference(c,f);r=manta.compute_periodic_semilocal_xc(**c,functional=f)
    assert r.receipt.energy==pytest.approx(e,abs=2e-10)
    np.testing.assert_allclose(r.values,v,atol=2e-11,rtol=2e-11)


@pytest.mark.parametrize('name',['lda','pbe'])
def test_image_and_cell_grid_refinement(name):
    f=vq.Functional(name,spin=1)
    budget=manta.ResourceBudget();budget.kernel_terms=10**14;budget.scalar_terms=10**13
    def energy(order,reach):
        c=case(1,angular=(0,),order=order,reach=reach);c['grid_batch']=64
        return manta.compute_periodic_semilocal_xc(**c,functional=f,budget=budget).receipt.energy
    target=energy(28,3)
    image_errors=[abs(energy(28,reach)-target) for reach in (0,1,2)]
    assert image_errors[0]>1e-3 and image_errors[2]<image_errors[1]<image_errors[0]
    assert image_errors[2]<1e-9
    if name=='pbe':
        # Gradient-sensitive PBE needs a finer cell tensor rule for this source.
        # Verify the reference itself stabilizes, rather than relaxing accuracy.
        target=energy(96,2)
        assert abs(energy(80,2)-target)<5e-8
        orders=(12,28,64)
    else:
        orders=(6,12,20)
    grid_errors=[abs(energy(order,2)-target) for order in orders]
    assert grid_errors[2]<grid_errors[1]<grid_errors[0]
    assert grid_errors[2]<1e-7


def test_custom_mixture_epoch_and_owned_functional_capture():
    name='test-manta-semilocal-mixture'
    components=[('LDA_X',.3),('LDA_C_VWN',.2),('GGA_X_PBE',.7),('GGA_C_PBE',.8)]
    vq.define_functional(name,components)
    old=vq.Functional(name,spin=2);c=case(2,angular=(0,1),order=4)
    result=manta.compute_periodic_semilocal_xc(**c,functional=old)
    e,v,_=reference(c,old)
    assert result.receipt.energy==pytest.approx(e,abs=3e-12)
    np.testing.assert_allclose(result.values,v,atol=3e-12)
    vq.define_functional(name,[('LDA_X',.4),*components[1:]])
    new=manta.compute_periodic_semilocal_xc(**c,functional=vq.Functional(name,spin=2))
    again=manta.compute_periodic_semilocal_xc(**c,functional=old)
    assert new.receipt.source_identity!=result.receipt.source_identity
    assert abs(new.receipt.energy-result.receipt.energy)>1e-5
    assert again.receipt.snapshot_identity==result.receipt.snapshot_identity
    np.testing.assert_array_equal(again.values,result.values)


def test_nonfinite_functional_weight_refused_in_plan():
    name='test-manta-semilocal-invalid-weight'
    vq.define_functional(name,[('LDA_X',float('nan'))])
    f=vq.Functional(name);c=case(1,angular=(0,),order=3)
    with pytest.raises(ValueError,match='finite'):
        manta.plan_periodic_semilocal_xc(c['basis'],f,27,3,27,17)
