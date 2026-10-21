"""Omitted AO-image Coulomb bounds against unreduced Gaussian Bloch sums."""
from types import SimpleNamespace
import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta, _vibeqc_core as core
from tests.test_manta_periodic_pairs import K, WEIGHTS, IMAGES, density, options as pair_options
from tests.test_manta_periodic_bupo import options, fixture as mixed_fixture
from tests.test_manta_periodic_bubbles import bubble_options
from tests.test_manta_periodic_ewald import cube
from tests.test_manta_lattice_gaussian import components, fourier
from tests.test_manta_periodic_bupo_domain import infinite_reference


def fixture(*, angular=0, pure=True, skew=True, images=None, cutoff=None, exponent=.8,
            two_shells=True, cell_size=4., real_radius=2, reciprocal_radius=3, eta=.45, weights=WEIGHTS):
    cell=np.array([[1.,.08,-.03],[0.,1.03,.06],[0.,0.,.97]])*cell_size if skew else np.eye(3)*cell_size
    centers=[[.1,.2,-.1],[.7,-.15,.25]][:2 if two_shells else 1]
    mol=vq.Molecule([vq.Atom(2,p) for p in centers],0,1); shells=[]
    for i,center in enumerate(centers):
        s=core.ShellInfo();s.origin=center;s.atom_index=i;s.l=angular if i else 0;s.pure=pure
        s.exponents=[exponent*(1+.2*i)];s.coefficients=[1.]
        if two_shells:s.exponents+=[exponent*2.1*(1+.2*i)];s.coefficients+=[-.23]
        shells.append(s)
    basis=vq.BasisSet(mol,shells,'pair-image-bound-reference',True)
    images=IMAGES[:3] if images is None else images
    pairs=manta.PeriodicPairSource(basis,cell,images,K,weights,pair_options(0,cutoff,.2))
    tree=manta.PeriodicBubbleTree(pairs,bubble_options(cap=2))
    exact=manta.PeriodicEwaldSource(pairs,eta,cube(real_radius),cube(reciprocal_radius,reciprocal=True),16)
    opts=options(0.);opts.bound_infinite_domain=True;opts.bound_pair_images=True
    bounded=manta.PeriodicBupoSource(tree,exact,cube(0),opts)
    return basis,pairs,tree,exact,bounded,cell


def ordered_reference(basis,d,cell,*,image_radius=2,reciprocal_radius=9,weights=WEIGHTS,images=None,cutoff=None):
    """Every ordered (a,b,R), analytical Gaussian Fourier integrals, bare 4pi/G².

    Neither the native canonical inventory nor native integral/bound routines
    enter the reference. Both Fourier and image radii can be refined separately.
    """
    g=cube(reciprocal_radius,reciprocal=True)@(2*np.pi*np.linalg.inv(cell))
    images=cube(image_radius) if images is None else images
    shells=basis.shells();sizes=[2*s.l+1 if s.pure else (s.l+1)*(s.l+2)//2 for s in shells]
    offset=np.cumsum([0]+sizes);n=basis.nbasis;panel=np.zeros((len(K),n,n,len(g)),complex)
    for index in images:
        translation=cell@index;phase=np.exp(-1j*K@translation)
        for ia in range(len(shells)):
            for ib in range(len(shells)):
                a,b=basis.shells()[ia],basis.shells()[ib];b.origin=(np.array(b.origin)+translation).tolist()
                if cutoff is not None and np.linalg.norm(np.array(a.origin)-b.origin)>cutoff:continue
                pair=SimpleNamespace(shells=lambda a=a,b=b:[a,b])
                raw=components(dict(kind='pair',basis=pair,extent=SimpleNamespace(shells=(0,1))))
                ta=manta.cartesian_to_spherical(a.l) if a.pure else np.eye((a.l+1)*(a.l+2)//2)
                tb=manta.cartesian_to_spherical(b.l) if b.pure else np.eye((b.l+1)*(b.l+2)//2)
                f=(np.kron(ta,tb)@np.array([fourier(c,g) for c in raw])).reshape(sizes[ia],sizes[ib],len(g))
                panel[:,offset[ia]:offset[ia+1],offset[ib]:offset[ib+1],:]+=phase[:,None,None,None]*f
    rho=np.einsum('k,kij,kijg->g',weights,d,panel)
    lam=4*np.pi/abs(np.linalg.det(cell))/np.sum(g*g,axis=1)
    j=np.einsum('kijg,g->kij',panel.conj(),lam*rho)
    return dict(values=j,energy=float(.5*np.dot(lam,abs(rho)**2)),rho=rho,lambda_=lam)


def calculation(**kwargs):
    b,s,t,e,c,cell=fixture(**kwargs);d=density(b.nbasis);out=c.build(d)
    weights=kwargs.get('weights',WEIGHTS)
    full=ordered_reference(b,d,cell,weights=weights)
    kept=ordered_reference(b,d,cell,weights=weights,images=kwargs.get('images',IMAGES[:3]),cutoff=kwargs.get('cutoff'))
    r=out.receipt;im=r.pair_images
    return dict(values=out.values,energy=r.energy,trace_energy=r.trace_energy,d=d,weights=weights,cell=cell,
        reference_values=full['values'],reference_energy=full['energy'],rho=full['rho'],lambda_=full['lambda_'],
        retained_values=kept['values'],retained_energy=kept['energy'],retained_rho=kept['rho'],
        j_error_bound=out.j_error_bound,energy_error_bound=r.energy_error_bound,
        image_j_bound=out.pair_image_j_error_bound,image_energy_bound=im.energy_error_bound,
        domain_j_bound=out.domain_j_error_bound,domain_energy_bound=r.domain.energy_error_bound,
        retained_mass=out.retained_pair_mass_bound,omitted_mass=out.omitted_pair_mass_bound,
        qr=im.retained_density_mass_bound,qm=im.omitted_density_mass_bound,kernel_bound=im.kernel_bound,
        source_identity=r.source_identity,pair_source_identity=r.pair_source_identity,density_identity=r.density_identity,
        envelope_identity=im.source_identity,partition_identity=r.partition.identity)


@pytest.mark.parametrize('angular,pure',[(0,True),(1,False),(2,True)])
@pytest.mark.parametrize('skew',[False,True])
def test_missing_source_and_target_images_against_unreduced_bloch_coulomb(angular,pure,skew):
    row=calculation(angular=angular,pure=pure,skew=skew,cutoff=1. if angular==2 else None)
    assert np.max(abs(row['reference_values']-row['retained_values']))>1e-6
    assert np.all(abs(row['reference_values']-row['retained_values'])<=row['image_j_bound'][None]+3e-11)
    assert abs(row['reference_energy']-row['retained_energy'])<=row['image_energy_bound']+3e-11
    assert np.all(abs(row['reference_values']-row['values'])<=row['j_error_bound'][None]+3e-11)
    assert abs(row['reference_energy']-row['energy'])<=row['energy_error_bound']+3e-11
    assert row['energy']==pytest.approx(row['trace_energy'],abs=3e-11)
    assert .5*np.einsum('k,kij,kji->',row['weights'],row['d'],row['reference_values']).real==pytest.approx(row['reference_energy'],abs=3e-11)


def test_reference_independent_image_and_fourier_refinement_and_canonical_limit():
    b,s,t,e,c,cell=fixture(angular=1)
    d=density(b.nbasis)
    base=ordered_reference(b,d,cell)
    more_images=ordered_reference(b,d,cell,image_radius=3)
    more_g=ordered_reference(b,d,cell,reciprocal_radius=11)
    for ref in (more_images,more_g):
        np.testing.assert_allclose(base['values'],ref['values'],atol=3e-11,rtol=3e-12)
        assert base['energy']==pytest.approx(ref['energy'],abs=3e-11)
    ordered=ordered_reference(b,d,cell,images=IMAGES[:3]);canonical=infinite_reference(b,s,d,cell,cutoff=9)
    np.testing.assert_allclose(ordered['values'],canonical['values'],atol=3e-11,rtol=3e-12)


def test_aggregation_ordered_multiplicities_scaling_and_no_phase_cancellation():
    weights=np.array([0.,.4,.6]);b,s,t,e,c,cell=fixture(angular=1,weights=weights)
    d=density(b.nbasis);a=c.build(d);r=a.receipt.pair_images
    w=np.einsum('k,kij->ij',weights,abs(d));keep=a.retained_pair_mass_bound;omit=a.omitted_pair_mass_bound
    qr=np.sum(w*keep);qm=np.sum(w*omit)
    assert r.retained_density_mass_bound==pytest.approx(qr,rel=2e-10)
    assert r.omitted_density_mass_bound==pytest.approx(qm,rel=2e-10)
    np.testing.assert_allclose(a.pair_image_j_error_bound,r.kernel_bound*(qm*keep+(qr+qm)*omit),atol=0,rtol=2e-10)
    assert r.energy_error_bound==pytest.approx(r.kernel_bound*(qr*qm+.5*qm*qm),rel=2e-10)
    assert r.retained_ordered_pairs==s.receipt.ordered_pairs
    assert r.retained_ordered_pairs+r.omitted_ordered_pairs_in_cube==b.nshells**2*r.cube_cells
    for scale in (-2.,1e-280):
        scaled=c.build(scale*d)
        np.testing.assert_allclose(scaled.pair_image_j_error_bound/abs(scale),a.pair_image_j_error_bound,rtol=3e-10,atol=0.)
        assert scaled.receipt.pair_images.energy_error_bound>0
        assert scaled.receipt.pair_images.source_identity==r.source_identity
    zero=c.build(np.zeros_like(d));assert zero.receipt.energy_error_bound==0
    np.testing.assert_array_equal(zero.pair_image_j_error_bound,0.)
    assert not r.roundoff_included and r.density_identity==a.receipt.density_identity
    for name in ('pair_image_j_error_bound','retained_pair_mass_bound','omitted_pair_mass_bound'):
        x=getattr(a,name);x[:]=0;assert np.max(getattr(a,name))>0


def test_image_inventory_enlargement_reaches_total_energy_and_j_target():
    previous=np.inf
    for radius in (0,1):
        b,s,t,e,c,cell=fixture(two_shells=False,exponent=4.,skew=False,cell_size=6.,images=cube(radius),
            real_radius=4,reciprocal_radius=8,eta=.4)
        a=c.build(density(1));r=a.receipt
        assert r.energy_error_bound<previous;previous=r.energy_error_bound
        ref=ordered_reference(b,density(1),cell,image_radius=2,reciprocal_radius=16)
        assert abs(ref['energy']-r.energy)<=r.energy_error_bound+3e-11
        assert np.max(abs(ref['values']-a.values))<=r.maximum_j_error_bound+3e-11
    assert r.energy_error_bound<1e-8 and r.maximum_j_error_bound<1e-8


def test_sparse_outer_images_do_not_hide_interior_holes_or_cutoff_pairs():
    images=np.array([[0,0,0],[-2,0,0],[2,0,0]],np.int64)
    b,s,t,e,c,cell=fixture(two_shells=False,exponent=4.,skew=False,cell_size=6.,images=images,real_radius=3)
    a=c.build(density(1));r=a.receipt.pair_images
    all_images=cube(4);missing=np.array([n for n in all_images if not np.any(np.all(images==n,axis=1))])
    exact_mass=np.sum((np.pi/8)**1.5*np.exp(-2*np.sum((missing@cell.T)**2,axis=1)))
    assert a.omitted_pair_mass_bound[0,0]>=exact_mass>0
    assert r.cube_radius==2 and r.omitted_ordered_pairs_in_cube==125-3
    b,s,t,e,c,cell=fixture(cutoff=.3)
    a=c.build(density(b.nbasis));assert a.retained_pair_mass_bound[0,1]==0
    # The omitted central off-diagonal pair must contribute to the target
    # response even though it never occurs in the retained canonical source.
    assert a.omitted_pair_mass_bound[0,1]>1 and a.pair_image_j_error_bound[0,1]>0


def test_sparse_huge_completion_refuses_before_domain_work_or_payload_read():
    b,s,t,e,c,cell=fixture(two_shells=False)
    images=np.array([[0,0,0],[-1000,0,0],[1000,0,0]],np.int64)
    source=manta.PeriodicPairSource(b,cell,images,K,WEIGHTS,pair_options(0,None,.2))
    tree=manta.PeriodicBubbleTree(source,bubble_options())
    direct=manta.PeriodicEwaldSource(source,.45,cube(1),cube(1,reciprocal=True),16)
    o=options(0.);o.bound_infinite_domain=True;o.bound_pair_images=True
    with pytest.raises((ValueError,RuntimeError),match='budget'):
        manta.PeriodicBupoSource(tree,direct,np.array([[2**40,0,0]],np.int64),o)


@pytest.mark.parametrize('field,planned',[('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),
    ('primitive_quartets','primitive_products'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_joint_image_caps_before_near_payloads(field,planned):
    b,s,t,e,c,cell=fixture(two_shells=False)
    o=options(0.);o.bound_infinite_domain=True;o.bound_pair_images=True
    p=manta.plan_periodic_bupo(t,e,1,o);limit=manta.ResourceBudget();setattr(limit,field,getattr(p,planned))
    a=manta.PeriodicBupoSource(t,e,cube(0),o,limit).build(density(b.nbasis))
    assert a.plan.image_cells==a.receipt.pair_images.cube_cells
    assert a.plan.image_primitive_products>=a.receipt.pair_images.primitive_pairs_evaluated
    setattr(limit,field,getattr(p,planned)-1)
    with pytest.raises((ValueError,RuntimeError),match='budget'):
        manta.PeriodicBupoSource(t,e,np.array([[2**40,0,0]],np.int64),o,limit)


def test_mixed_partition_unchanged_opt_in_and_missing_domain_refusal():
    b,s,t,e,c,cell=mixed_fixture();d=density(b.nbasis);finite=c.build(d)
    o=options();o.bound_pair_images=True
    with pytest.raises(ValueError,match='require.*infinite-domain'):manta.PeriodicBupoSource(t,e,cube(0),o)
    o.bound_infinite_domain=True;bound=manta.PeriodicBupoSource(t,e,cube(0),o);a=bound.build(d)
    assert bound.partition.identity==c.partition.identity and bound.partition.approximate_blocks>0
    np.testing.assert_array_equal(finite.values,a.values)
    assert finite.receipt.energy==a.receipt.energy and finite.receipt.pair_images is None
    assert finite.pair_image_j_error_bound is finite.retained_pair_mass_bound is finite.omitted_pair_mass_bound is None
    assert a.receipt.qualification=='periodic-bupo-j-with-full-pair-image-ewald-bound'
    assert a.receipt.pair_images.kernel_identity==e.kernel_identity
    assert a.receipt.pair_images.pair_source_identity==s.receipt.source_identity
    assert a.receipt.domain.pair_image_omission_included is False
    np.testing.assert_array_less(a.domain_j_error_bound+a.pair_image_j_error_bound+finite.j_error_bound,a.j_error_bound)


@pytest.mark.parametrize('skew',[False,True])
@pytest.mark.parametrize('cutoff',[None,.3])
def test_s_pair_masses_keep_admitted_full_exponents_and_signed_primitive_sums(skew,cutoff):
    b,source,_,_,bounded,cell=fixture(skew=skew,cutoff=cutoff)
    out=bounded.build(density(b.nbasis));shells=b.shells()
    images=IMAGES[:3];large=cube(7);larger=cube(8)
    ceiling=max(x for s in shells for x,c in zip(s.exponents,s.coefficients) if c!=0)
    def primitive_mass(a,b,labels,exact=False):
        distance=np.sum((np.asarray(a.origin)-b.origin-labels@cell.T)**2,axis=1)
        mass=np.zeros(len(labels))
        for x,cx in zip(a.exponents,a.coefficients):
            for y,cy in zip(b.exponents,b.coefficients):
                divisor=1 if exact else max(1,(x+y)/ceiling)
                exponent=(x+y)/divisor;mu=x*y/(x+y)/divisor
                mass+=abs(cx*cy)*(np.pi/exponent)**1.5*np.exp(-mu*distance)
        return mass,distance
    for i,a in enumerate(shells):
        for j,b in enumerate(shells):
            assert a.l==b.l==0 and any(c<0 for c in a.coefficients)
            keep,dist=primitive_mass(a,b,images)
            if cutoff is not None:keep=keep[dist<=cutoff**2]
            np.testing.assert_allclose(out.retained_pair_mass_bound[i,j],sum(keep),rtol=3e-10,atol=0)
            missing=[]
            for labels in (large,larger):
                mass,dist=primitive_mass(a,b,labels,exact=True)
                present=np.any(np.all(labels[:,None,:]==images[None,:,:],axis=2),axis=1)
                if cutoff is not None:present &= dist<=cutoff**2
                missing.append(sum(mass[~present]))
            assert missing[0]==pytest.approx(missing[1],rel=2e-12,abs=1e-30)
            assert out.omitted_pair_mass_bound[i,j]>=missing[1]
    expected=ceiling
    assert out.receipt.pair_images.maximum_gaussian_exponent==pytest.approx(expected,rel=3e-10)


@pytest.mark.parametrize('cutoff',[None,.3])
def test_mixed_angular_image_bounds_do_not_exceed_original_half_gaussians(cutoff):
    basis,_,_,_,bounded,cell=fixture(angular=1,pure=False,cutoff=cutoff)
    d=density(basis.nbasis);out=bounded.build(d);receipt=out.receipt.pair_images
    shells=basis.shells();sizes=[1,3];offset=np.cumsum([0]+sizes)
    keep=np.zeros((basis.nbasis,basis.nbasis));omit=keep.copy();labels=cube(receipt.cube_radius)
    def h(l,x):return (l/(x*np.e))**(l/2) if l else 1.
    for i,a in enumerate(shells):
        for j,b in enumerate(shells):
            distance=np.sum((np.asarray(a.origin)-b.origin-labels@cell.T)**2,axis=1)
            included=np.any(np.all(labels[:,None,:]==IMAGES[None,:3,:],axis=2),axis=1)
            if cutoff is not None:included &= distance<=cutoff**2
            kr=ko=0.
            for x,cx in zip(a.exponents,a.coefficients):
                for y,cy in zip(b.exponents,b.coefficients):
                    mu=x*y/(2*(x+y));amp=abs(cx*cy)*h(a.l,x)*h(b.l,y)*(2*np.pi/(x+y))**1.5
                    values=amp*np.exp(-mu*distance);kr+=sum(values[included]);ko+=sum(values[~included])
                    decay=mu*receipt.real_metric_lower**2;theta=2+np.sqrt(np.pi/decay)
                    u=receipt.cube_radius+1-np.linalg.norm(np.asarray(a.origin)-b.origin)/receipt.real_metric_lower
                    tail=theta**3
                    if u>0:tail=min(tail,6*theta**2*np.exp(-decay*u*u)*(1+1/(2*decay*u)))
                    ko+=amp*tail
            keep[offset[i]:offset[i+1],offset[j]:offset[j+1]]=kr
            omit[offset[i]:offset[i+1],offset[j]:offset[j+1]]=ko
    assert receipt.maximum_gaussian_exponent==pytest.approx(max(max(s.exponents) for s in shells),rel=3e-10)
    weights=np.einsum('k,kij->ij',WEIGHTS,abs(d));qr=np.sum(weights*keep);qm=np.sum(weights*omit)
    old_j=receipt.kernel_bound*(qm*keep+(qr+qm)*omit)
    old_energy=receipt.kernel_bound*(qr*qm+.5*qm*qm)
    assert np.all(out.pair_image_j_error_bound<=old_j*(1+3e-10))
    assert receipt.energy_error_bound<=old_energy*(1+3e-10)
    assert out.pair_image_j_error_bound[0,0]<old_j[0,0]
