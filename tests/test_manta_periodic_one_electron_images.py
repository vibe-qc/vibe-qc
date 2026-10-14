"""Full missing-image S/T enclosures against unreduced Gaussian quadrature."""
from decimal import Decimal,localcontext
import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta,_vibeqc_core as core
from tests.test_manta_periodic_pairs import K,WEIGHTS,IMAGES,options,density
from tests.test_manta_periodic_ewald import cube
from tests.test_manta_one_electron import _cart


def fixture(*,angular=0,pure=True,skew=True,images=None,cutoff=None,exponent=.8,cell_size=4.,weights=WEIGHTS,k=K):
    cell=np.array([[1.,.08,-.03],[0.,1.03,.06],[0.,0.,.97]])*cell_size if skew else np.eye(3)*cell_size
    centers=[[.1,.2,-.1],[.7,-.15,.25]];atoms=[vq.Atom(2,p) for p in centers];mol=vq.Molecule(atoms,0,1);shells=[]
    for i,center in enumerate(centers):
        s=core.ShellInfo();s.origin=center;s.atom_index=i;s.l=angular if i else 0;s.pure=pure
        s.exponents=[exponent*(1+.2*i),2.1*exponent*(1+.2*i)];s.coefficients=[1.,-.23];shells.append(s)
    basis=vq.BasisSet(mol,shells,'one-electron-images-reference',True)
    images=IMAGES[:3].copy() if images is None else images
    source=manta.PeriodicPairSource(basis,cell,images,k,weights,options(0,cutoff,.2))
    return dict(basis=basis,source=source,cell=cell,images=images,cutoff=cutoff,k=k,weights=weights)


def tile(a,b):
    """Independent separable Gauss-Hermite integration of polynomials/gradients."""
    nodes,weights=np.polynomial.hermite.hermgauss(8);ca,cb=_cart(a.l),_cart(b.l)
    result=np.zeros((2,len(ca),len(cb)))
    delta=np.array(a.origin)-np.array(b.origin)
    for alpha,ac in zip(a.exponents,a.coefficients):
        for beta,bc in zip(b.exponents,b.coefficients):
            p=alpha+beta;center=(alpha*np.array(a.origin)+beta*np.array(b.origin))/p
            xyz=nodes[None,:]/np.sqrt(p)+center[:,None]
            xa=xyz-np.array(a.origin)[:,None];xb=xyz-np.array(b.origin)[:,None];s=[];g=[]
            for axis in range(3):
                la,lb=ca[:,axis,None],cb[:,axis,None];pa=xa[axis]**la;pb=xb[axis]**lb
                ga=la*xa[axis]**np.maximum(la-1,0)-2*alpha*xa[axis]*pa
                gb=lb*xb[axis]**np.maximum(lb-1,0)-2*beta*xb[axis]*pb
                s.append((pa*weights)@pb.T);g.append((ga*weights)@gb.T)
            factor=ac*bc*np.exp(-alpha*beta/p*np.dot(delta,delta))/p**1.5
            result[0]+=factor*s[0]*s[1]*s[2]
            result[1]+=.5*factor*(g[0]*s[1]*s[2]+s[0]*g[1]*s[2]+s[0]*s[1]*g[2])
    ta=manta.cartesian_to_spherical(a.l) if a.pure else np.eye(len(ca))
    tb=manta.cartesian_to_spherical(b.l) if b.pure else np.eye(len(cb))
    return np.array([ta@x@tb.T for x in result])


def ordered(c,images,cutoff=None):
    b=c['basis'];sizes=[2*s.l+1 if s.pure else (s.l+1)*(s.l+2)//2 for s in b.shells()]
    offsets=np.cumsum([0,*sizes]);out=np.zeros((2,len(c['k']),b.nbasis,b.nbasis),complex)
    for image in images:
        shift=c['cell']@image;phase=np.exp(1j*c['k']@shift)
        for i,a in enumerate(b.shells()):
            for j,bb in enumerate(b.shells()):
                bb.origin=(np.array(bb.origin)+shift).tolist()
                if cutoff is not None and np.linalg.norm(np.array(a.origin)-bb.origin)>cutoff:continue
                value=tile(a,bb)
                out[:,:,offsets[i]:offsets[i+1],offsets[j]:offsets[j+1]]+=phase[None,:,None,None]*value[:,None]
    return out


def calculation(**kwargs):
    c=fixture(**kwargs);bounds=manta.bound_periodic_one_electron_images(c['source'])
    finite=ordered(c,c['images'],c['cutoff']);full=ordered(c,cube(3 if kwargs.get('angular',0)>=4 else 2))
    native=np.array([manta.compute_bloch_one_electron(c['basis'],operation,
        np.ascontiguousarray(c['images']@c['cell'].T),c['k'],c['cutoff']).values
        for operation in ('overlap','kinetic')])
    np.testing.assert_allclose(native,finite,atol=3e-12,rtol=1e-12)
    d=density(c['basis'].nbasis)
    return dict(context=c,bounds=bounds,finite=finite,full=full,native=native,d=d)


@pytest.mark.parametrize('angular,pure',[(0,True),(1,False),(2,True)])
@pytest.mark.parametrize('skew',[False,True])
def test_full_ordered_overlap_and_kinetic_against_native_enclosures(angular,pure,skew):
    r=calculation(angular=angular,pure=pure,skew=skew,cutoff=1. if angular==2 else None)
    b=r['bounds'];c=r['context'];errors=abs(r['full']-r['finite'])
    assert np.max(errors)>1e-6
    for i,matrix in enumerate((b.overlap_error_bound,b.kinetic_error_bound)):
        assert np.all(errors[i]<=matrix[None]+3e-12)
        weighted=np.einsum('k,kij->ij',c['weights'],abs(r['d']))
        difference=abs(np.einsum('k,kij,kji->',c['weights'],r['d'],r['full'][i]-r['finite'][i]).real)
        assert difference<=np.sum(weighted*matrix)+3e-12
    assert b.receipt.retained_ordered_pairs==c['source'].receipt.ordered_pairs
    assert b.receipt.retained_ordered_pairs+b.receipt.omitted_ordered_pairs_in_cube==c['basis'].nshells**2*b.receipt.cube_cells
    assert b.receipt.shell_pairs_evaluated<=b.plan.shell_calls
    assert b.receipt.primitive_pairs_evaluated<=b.plan.primitive_pairs
    assert not b.receipt.roundoff_included


def test_independent_reference_image_refinement_and_native_finite_values():
    from tests.test_manta_periodic_bupo_hf import case,independent
    c=fixture(angular=1);np.testing.assert_allclose(ordered(c,cube(2)),ordered(c,cube(3)),atol=3e-12,rtol=1e-12)
    a,c=case(shifted=True);ref=independent(a,c);native=core._manta.run_periodic_bupo_hf(**a)
    b=manta.bound_periodic_one_electron_images(c['pairs'])
    np.testing.assert_allclose(native.overlap,ref['overlap'],atol=3e-13,rtol=1e-12)
    np.testing.assert_array_equal(native.one_electron_images.overlap_error_bound,b.overlap_error_bound)
    np.testing.assert_array_equal(native.one_electron_images.kinetic_error_bound,b.kinetic_error_bound)


@pytest.mark.parametrize('angular,pure',[(4,False),(6,True)])
def test_high_angular_gradient_majorants(angular,pure):
    r=calculation(angular=angular,pure=pure,images=cube(0),cell_size=3.)
    assert np.max(abs(r['full'][1]-r['finite'][1]))>1e-5
    assert np.all(abs(r['full'][1]-r['finite'][1])<=r['bounds'].kinetic_error_bound[None]+3e-11)


@pytest.mark.parametrize('angular,pure',[(4,False),(6,True)])
def test_high_angular_full_reference_image_convergence(angular,pure):
    c=fixture(angular=angular,pure=pure,cell_size=3.)
    np.testing.assert_allclose(ordered(c,cube(3)),ordered(c,cube(4)),atol=3e-12,rtol=1e-12)


def test_mesh_weights_do_not_change_bounds_and_exports_are_owned():
    a=fixture(angular=1);b=fixture(angular=1,k=K[::-1].copy(),weights=np.array([0.,.4,.6]))
    x=manta.bound_periodic_one_electron_images(a['source']);y=manta.bound_periodic_one_electron_images(b['source'])
    assert x.receipt.source_identity!=y.receipt.source_identity
    for name in ('overlap_error_bound','kinetic_error_bound'):
        np.testing.assert_array_equal(getattr(x,name),getattr(y,name))
        value=getattr(x,name);saved=value.copy();value[:]=0;np.testing.assert_array_equal(getattr(x,name),saved)
    with pytest.raises(AttributeError):x.receipt.maximum_kinetic_error=0.


@pytest.mark.parametrize('field,planned',[('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),
    ('primitive_quartets','primitive_pairs'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_all_five_exact_and_one_short_limits(field,planned):
    c=fixture();budget=manta.ResourceBudget();p=manta.plan_periodic_one_electron_images(c['source'],budget)
    setattr(budget,field,getattr(p,planned));r=manta.bound_periodic_one_electron_images(c['source'],budget)
    assert r.plan.modeled_peak_bytes==p.modeled_peak_bytes
    setattr(budget,field,getattr(p,planned)-1)
    with pytest.raises((ValueError,RuntimeError),match='budget'):manta.bound_periodic_one_electron_images(c['source'],budget)


def test_sparse_completion_cube_cannot_escape_admission():
    c=fixture(images=np.array([[0,0,0],[10**6,0,0],[-10**6,0,0]],dtype=np.int64),cutoff=1.)
    with pytest.raises((ValueError,RuntimeError,OverflowError),match='budget|overflow'):manta.bound_periodic_one_electron_images(c['source'])


def test_bupo_hf_enforces_matrix_and_density_weighted_kinetic_targets():
    from tests.test_manta_periodic_bupo_hf import case,mixed_case
    for name in ('maximum_overlap_image_error','maximum_kinetic_image_error','maximum_kinetic_energy_error'):
        a,c=case();setattr(a['options'],name,1e-300)
        with pytest.raises(ValueError,match='accuracy target'):core._manta.run_periodic_bupo_hf(**a)
        setattr(a['options'],name,0.)
        with pytest.raises(ValueError,match='finite positive'):core._manta.run_periodic_bupo_hf(**a)
    for spin in ('restricted','unrestricted'):
        a,c=mixed_case(spin=spin,shifted=True,q0='BvK-ewald');r=core._manta.run_periodic_bupo_hf(**a)
        w=np.einsum('k,kij->ij',c['weights'],abs(r.density.sum(axis=0)))
        assert r.converged and r.receipt.one_electron_accuracy_passed
        assert r.receipt.kinetic_energy_error_bound>0
        # Positive subnormal bounds require outward, not nearest, products.
        with localcontext() as ctx:
            ctx.prec=90
            expected=sum((Decimal(float(x))*Decimal(float(y)) for x,y in
                zip(w.ravel(),r.one_electron_images.kinetic_error_bound.ravel())),Decimal(0))
            assert Decimal(r.receipt.kinetic_energy_error_bound)>=expected
        assert r.receipt.maximum_observed_kinetic_energy_error<=a['options'].maximum_kinetic_energy_error


def test_missing_central_pair_and_larger_declared_cutoff_domain():
    r=calculation(angular=1,cutoff=.3)
    for index,name in enumerate(('overlap_error_bound','kinetic_error_bound')):
        assert np.max(abs(r['full'][index,:,0,1:]-r['finite'][index,:,0,1:]))>.01
        assert np.all(abs(r['full'][index]-r['finite'][index])<=getattr(r['bounds'],name)[None]+3e-12)
    small=fixture(images=cube(0),cutoff=3.,exponent=4.,cell_size=6.)
    large=fixture(images=cube(1),cutoff=3.,exponent=4.,cell_size=6.)
    a=manta.bound_periodic_one_electron_images(small['source']);b=manta.bound_periodic_one_electron_images(large['source'])
    np.testing.assert_array_equal(ordered(small,small['images'],3.),ordered(large,large['images'],3.))
    assert b.receipt.maximum_overlap_error<a.receipt.maximum_overlap_error
    assert b.receipt.maximum_kinetic_error<a.receipt.maximum_kinetic_error
