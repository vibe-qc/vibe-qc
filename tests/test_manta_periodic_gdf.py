"""Complete finite Coulomb fitting sources, independently matched in each term."""
import itertools

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta, _vibeqc_core as core
from tests.test_manta_periodic_fitting import case as sr_case, analytic_metric, CAPS
from tests.test_aux_eri_sr_lattice_enumeration import _sr_three_center_oracle, _weighted_gradient_fixture


def case(gamma=False,shift=False):
    a=sr_case(shift)
    if gamma:a['q'][:]=0.
    integers=np.array(list(itertools.product(range(-1,2),repeat=3)))
    a['vectors']=integers@a['system'].reciprocal_lattice().T+a['q']
    return a


def plan(a):
    args=dict(a);args.pop('q');args['vectors']=len(args['vectors']);args['momenta']=len(args.pop('ket_kpoints'))
    return manta.plan_periodic_gdf(**args)


def legacy(a):
    o=a['options']
    return core.compute_gdf_range_separated_integrals(a['orbital'],a['auxiliary'],a['system'],a['q'],a['ket_kpoints'],a['vectors'],
        o.omega,o.pair_cutoff_bohr,o.auxiliary_cutoff_bohr,2**28,2**28,o.image_candidate_cap,o.integral_screen_error,o.compute_metric)


def gaussian_fourier(a,vectors):
    """Independent Gaussian product theorem on the admitted physical pair set."""
    squared=np.sum(vectors**2,axis=1)
    aft=np.array([(np.pi/s.exponents[0])**1.5*np.exp(-squared/(4*s.exponents[0])-1j*vectors@np.array(s.origin))
        for s in a['auxiliary'].shells()])
    b=a['orbital'];nk=len(a['ket_kpoints']);pair=np.zeros((nk,b.nbasis,b.nbasis,len(vectors)),complex)
    translations=np.array(list(itertools.product(range(-5,6),repeat=3)))@a['system'].lattice.T
    for mu,sm in enumerate(b.shells()):
        alpha,A=sm.exponents[0],np.array(sm.origin)
        for nu,sn in enumerate(b.shells()):
            beta=sn.exponents[0];B=np.array(sn.origin)+translations;rho=alpha+beta
            d2=np.sum((A-B)**2,axis=1);keep=d2<=a['options'].pair_cutoff_bohr**2
            product=(alpha*A+beta*B[keep])/rho
            value=(np.pi/rho)**1.5*np.exp(-alpha*beta/rho*d2[keep,None]-squared[None]/(4*rho)-1j*product@vectors.T)
            pair[:,mu,nu]=np.exp(1j*a['ket_kpoints']@translations[keep].T)@value
    return aft,pair


def fourier_blocks(a,vectors,weights):
    aft,pair=gaussian_fourier(a,vectors)
    return (aft.conj()*weights)@aft.T,np.einsum('pg,kmng,g->kpmn',aft.conj(),pair,weights)


def independent(a):
    o=a['options'];v=a['vectors'];squared=np.sum(v*v,axis=1);keep=squared>1e-24
    volume=abs(np.linalg.det(a['system'].lattice))
    lr_m,lr_t=fourier_blocks(a,v[keep],4*np.pi/volume/squared[keep]*np.exp(-squared[keep]/(4*o.omega**2)))
    m=analytic_metric(a)+lr_m
    t=np.array([_sr_three_center_oracle(a['system'],a['orbital'],a['auxiliary'],a['q'],k,o.omega,
        o.pair_cutoff_bohr,o.auxiliary_cutoff_bohr) for k in a['ket_kpoints']])+lr_t
    frac=a['system'].lattice.T@a['q']/(2*np.pi)
    if np.linalg.norm(frac-np.round(frac))<1e-12:
        zm,zt=fourier_blocks(a,np.zeros((1,3)),np.array([np.pi/volume/o.omega**2]))
        m-=zm;t-=zt
    return m,t


def quadrature_long_range(a):
    """Gauss-Hermite polynomial integrals, including Cartesian/high-l shells."""
    from tests.test_manta_fourier import _quadrature
    vectors=a['vectors'];squared=np.sum(vectors*vectors,axis=1);keep=squared>1e-24
    vectors=vectors[keep];o=a['options'];volume=abs(np.linalg.det(a['system'].lattice))
    weights=4*np.pi/volume/squared[keep]*np.exp(-squared[keep]/(4*o.omega**2))
    frac=a['system'].lattice.T@a['q']/(2*np.pi)
    if np.linalg.norm(frac-np.round(frac))<1e-12:
        vectors=np.vstack((vectors,np.zeros((1,3))));weights=np.append(weights,-np.pi/volume/o.omega**2)
    aux=a['auxiliary'];orb=a['orbital'];nk=len(a['ket_kpoints'])
    aft=np.concatenate([_quadrature(s,None,vectors) for s in aux.shells()],axis=1).T
    pairs=np.zeros((nk,orb.nbasis,orb.nbasis,len(vectors)),complex)
    translations=np.array(list(itertools.product(range(-5,6),repeat=3)))@a['system'].lattice.T
    row=0
    for sm in orb.shells():
        col=0;nm=2*sm.l+1 if sm.pure else (sm.l+1)*(sm.l+2)//2
        for sn in orb.shells():
            nn=2*sn.l+1 if sn.pure else (sn.l+1)*(sn.l+2)//2
            origin=np.array(sn.origin)
            for R in translations:
                if np.linalg.norm(np.array(sm.origin)-origin-R)>o.pair_cutoff_bohr:continue
                sn.origin=origin+R
                tile=_quadrature(sm,sn,vectors).transpose(1,2,0)
                pairs[:,row:row+nm,col:col+nn]+=np.exp(1j*a['ket_kpoints']@R)[:,None,None,None]*tile[None]
            col+=nn
        row+=nm
    return (aft.conj()*weights)@aft.T,np.einsum('pg,kmng,g->kpmn',aft.conj(),pairs,weights)


@pytest.mark.parametrize('gamma',[False,True])
@pytest.mark.parametrize('screen',[0.,1e-4])
def test_same_source_blocks_general_complex_responses_and_actual_work(gamma,screen):
    a=case(gamma);a['options'].integral_screen_error=screen
    r=manta.compute_periodic_gdf(**a);m,t,ng=legacy(a)
    np.testing.assert_allclose(r.metric,m,atol=2e-12,rtol=2e-13)
    np.testing.assert_allclose(r.three_center,t,atol=2e-12,rtol=2e-13)
    expected_m,expected_t=independent(a)
    np.testing.assert_allclose(r.metric,expected_m,atol=screen+3e-12,rtol=2e-13)
    np.testing.assert_allclose(r.three_center,expected_t,atol=screen+3e-12,rtol=2e-13)
    rng=np.random.default_rng(911);flat=t.reshape(len(t),len(m),-1)
    trial=rng.normal(size=flat.shape[-1])+1j*rng.normal(size=flat.shape[-1])
    for b,c in zip(r.three_center.reshape(flat.shape),flat):
        actual=b.conj().T@np.linalg.solve(r.metric,b@trial)
        expected=c.conj().T@np.linalg.solve(m,c@trial)
        np.testing.assert_allclose(actual,expected,atol=5e-10,rtol=2e-12)
        assert np.vdot(trial,actual).real==pytest.approx(np.vdot(trial,expected).real,abs=1e-9)
    assert r.receipt.finite_zero_removed==gamma and r.receipt.reciprocal_vectors==ng
    assert r.receipt.shell_calls<=r.plan.shell_calls and r.receipt.primitive_products<=r.plan.primitive_products
    assert r.receipt.vector_primitive_products<=r.plan.vector_primitive_products
    assert r.receipt.threads==1 and r.receipt.kernel=='coulomb'
    assert r.plan.output_bytes==r.metric.nbytes+r.three_center.nbytes


def test_zero_list_membership_is_separate_from_finite_zero_subtraction():
    a=case(True);r=manta.compute_periodic_gdf(**a)
    v=a['vectors'][np.linalg.norm(a['vectors'],axis=1)>1e-12]
    omitted=manta.compute_periodic_gdf(**dict(a,vectors=v))
    np.testing.assert_allclose(omitted.metric,r.metric,atol=3e-13,rtol=0)
    np.testing.assert_allclose(omitted.three_center,r.three_center,atol=3e-13,rtol=0)
    assert omitted.receipt.finite_zero_removed
    assert omitted.receipt.source_identity!=r.receipt.source_identity
    # A reciprocal Gamma representative still needs the finite-zero term.
    G=a['system'].reciprocal_lattice()@np.array([2,-1,1])
    shifted=manta.compute_periodic_gdf(**dict(a,q=G,ket_kpoints=a['ket_kpoints']+G,vectors=v))
    np.testing.assert_allclose(shifted.metric,r.metric,atol=2e-12,rtol=0)
    np.testing.assert_allclose(shifted.three_center,r.three_center,atol=2e-12,rtol=0)
    assert shifted.receipt.finite_zero_removed


@pytest.mark.parametrize('gamma',[False,True])
def test_converged_splits_against_independent_bare_coulomb_reciprocal_sum(gamma):
    from vibeqc.aux_basis import _rsgdf_shifted_dense_g_mesh
    a=case(gamma);o=a['options'];o.pair_cutoff_bohr=12.;o.auxiliary_cutoff_bohr=18.
    dense=_rsgdf_shifted_dense_g_mesh(a['system'],a['q'],80.)
    squared=np.sum(dense*dense,axis=1);keep=squared>1e-24
    expected_m,expected_t=fourier_blocks(a,dense[keep],4*np.pi/abs(np.linalg.det(a['system'].lattice))/squared[keep])
    results=[]
    for omega in (.45,.8):
        o.omega=omega;a['vectors']=_rsgdf_shifted_dense_g_mesh(a['system'],a['q'],2*omega**2*np.log(1e15))
        r=manta.compute_periodic_gdf(**a);results.append(r)
        np.testing.assert_allclose(r.metric,expected_m,atol=3e-11,rtol=2e-12)
        np.testing.assert_allclose(r.three_center,expected_t,atol=3e-11,rtol=2e-12)
    assert results[0].receipt.source_identity!=results[1].receipt.source_identity


@pytest.mark.parametrize('angular,pure',[(1,True),(2,True),(3,False)])
def test_signed_long_contractions_pure_and_cartesian_full_source(angular,pure):
    a=case();s,b,aux=_weighted_gradient_fixture(angular,orbital_angular=1)
    def change_representation(basis):
        shells=basis.shells()
        for sh in shells:sh.pure=pure
        return vq.BasisSet(s.unit_cell_molecule(),shells,'gdf-angular',True)
    integers=np.array(list(itertools.product(range(-1,2),repeat=3)))
    a.update(system=s,orbital=change_representation(b),auxiliary=change_representation(aux),vectors=integers@s.reciprocal_lattice().T+a['q'])
    a['options'].pair_cutoff_bohr=4.3;a['options'].auxiliary_cutoff_bohr=5.1
    r=manta.compute_periodic_gdf(**a)
    if pure:m,t,_=legacy(a)
    else:
        from tests.test_manta_periodic_fitting import libint_reference
        m,t=libint_reference(a);lr_m,lr_t=quadrature_long_range(a);m+=lr_m;t+=lr_t
    np.testing.assert_allclose(r.metric,m,atol=3e-11,rtol=2e-12)
    np.testing.assert_allclose(r.three_center,t,atol=3e-11,rtol=2e-12)


def test_image_gauge_transfer_conjugacy_and_shared_k_primitives():
    a=case();b=case(shift=True);r=manta.compute_periodic_gdf(**a);moved=manta.compute_periodic_gdf(**b)
    L=a['system'].lattice[:,0];q=a['q'];ap=np.exp(1j*np.array([0,q@L]))
    np.testing.assert_allclose(moved.metric,r.metric*ap[:,None]*ap.conj()[None],atol=2e-12,rtol=0)
    for ik,k in enumerate(a['ket_kpoints']):
        bra=np.exp(1j*np.array([0,(k-q)@L]));ket=np.exp(-1j*np.array([0,k@L]))
        np.testing.assert_allclose(moved.three_center[ik],r.three_center[ik]*ap[:,None,None]*bra[None,:,None]*ket[None,None,:],atol=2e-12,rtol=0)
        single=manta.compute_periodic_gdf(**dict(a,ket_kpoints=k[None]))
        np.testing.assert_array_equal(single.three_center[0],r.three_center[ik])
        assert single.receipt.shell_calls==r.receipt.shell_calls and single.receipt.primitive_products==r.receipt.primitive_products
        assert single.receipt.metric_source_identity==r.receipt.metric_source_identity
    negative=manta.compute_periodic_gdf(**dict(a,q=-q,ket_kpoints=-a['ket_kpoints'],vectors=-a['vectors']))
    np.testing.assert_array_equal(negative.metric,r.metric.conj())
    np.testing.assert_array_equal(negative.three_center,r.three_center.conj())
    assert moved.receipt.source_identity!=r.receipt.source_identity


@pytest.mark.parametrize('field,quantity',CAPS)
@pytest.mark.parametrize('metric',[True,False])
def test_joint_sr_lr_caps_exact_and_one_short_before_numerical_copy(field,quantity,metric):
    a=case(True);a['options'].compute_metric=metric;a['budget'].other_live_bytes=23456;p=plan(a)
    for cap,value in CAPS:setattr(a['budget'],cap,getattr(p,value))
    r=manta.compute_periodic_gdf(**a);assert r.plan.modeled_peak_bytes==p.modeled_peak_bytes
    setattr(a['budget'],field,getattr(p,quantity)-1)
    for key in ('q','ket_kpoints','vectors'):a[key][:]=np.nan
    with pytest.raises((ValueError,RuntimeError),match='budget'):plan(a)
    with pytest.raises((ValueError,RuntimeError),match='budget'):manta.compute_periodic_gdf(**a)


def test_owned_exports_optional_metric_and_layout_independent_identities():
    a=case(True);r=manta.compute_periodic_gdf(**a);m,t=r.metric,r.three_center
    r.metric[:]=np.nan;r.three_center[:]=np.nan
    np.testing.assert_array_equal(r.metric,m);np.testing.assert_array_equal(r.three_center,t)
    for key in ('q','ket_kpoints','vectors'):
        same=manta.compute_periodic_gdf(**dict(a,**{key:np.asfortranarray(a[key])}))
        assert same.receipt.source_identity==r.receipt.source_identity
        np.testing.assert_array_equal(same.three_center,t)
    p=plan(a);a['budget'].other_live_bytes+=1234
    assert plan(a).modeled_peak_bytes==p.modeled_peak_bytes+1234
    a['options'].image_candidate_cap*=2
    assert manta.compute_periodic_gdf(**a).receipt.source_identity==r.receipt.source_identity
    p=plan(a);a['options'].compute_metric=False;only=manta.compute_periodic_gdf(**a)
    assert only.metric.shape==(0,0) and only.receipt.metric_source_identity==''
    assert only.receipt.three_center_source_identity==r.receipt.three_center_source_identity
    np.testing.assert_array_equal(only.three_center,t)
    assert only.plan.modeled_peak_bytes<p.modeled_peak_bytes


@pytest.mark.parametrize('fault',['lattice','nonfinite','empty','dtype','strides','shape','omega_overflow','omega_underflow'])
def test_reciprocal_source_refusals(fault):
    a=case()
    if fault=='lattice':a['vectors'][0,1]+=.01
    elif fault=='nonfinite':a['vectors'][0,0]=np.nan
    elif fault=='empty':a['vectors']=np.empty((0,3))
    elif fault=='dtype':a['vectors']=a['vectors'].astype(np.float32)
    elif fault=='strides':a['vectors']=a['vectors'][::-1]
    elif fault=='shape':a['vectors']=np.zeros((2,4))
    elif fault=='omega_overflow':a['options'].omega=1e300
    else:a['options'].omega=1e-300
    with pytest.raises((ValueError,RuntimeError,TypeError,OverflowError)):manta.compute_periodic_gdf(**a)


def test_count_overflow_without_allocating_reciprocal_domain():
    a=case()
    with pytest.raises((ValueError,RuntimeError,OverflowError)):
        manta.plan_periodic_gdf(a['orbital'],a['auxiliary'],a['system'],2**63,2,a['options'],a['budget'])


def test_small_nonzero_transfer_requires_its_shifted_coulomb_vector():
    a=case(True);a['q']=a['system'].reciprocal_lattice()@np.array([1e-10,0.,0.])
    with pytest.raises(ValueError,match='G\\+q'):
        manta.compute_periodic_gdf(**a)


def test_unrepresentable_cell_volume_cannot_silently_zero_the_kernel():
    a=case(True);a['system']=vq.PeriodicSystem(3,np.eye(3)*1e103,a['system'].unit_cell)
    a['vectors']=np.zeros((1,3))
    with pytest.raises((ValueError,RuntimeError),match='volume|cell|image'):
        manta.compute_periodic_gdf(**a)


@pytest.mark.parametrize('orbital_l,auxiliary_l',[(0,7),(6,0)])
def test_high_angular_fourier_assembly_against_separable_quadrature(orbital_l,auxiliary_l):
    a=case();system=vq.PeriodicSystem(3,20*np.eye(3),[vq.Atom(2,[0.,0.,0.])])
    def basis(l):
        return vq.BasisSet(system.unit_cell_molecule(),[core.ShellInfo(0,l,True,[.8],[1.],[0.,0.,0.])],'gdf-high-angular',False)
    indices=np.array([[0,0,0],[1,0,0],[-1,0,0],[1,1,0],[0,-1,1]])
    a.update(system=system,orbital=basis(orbital_l),auxiliary=basis(auxiliary_l),vectors=indices@system.reciprocal_lattice().T+a['q'])
    a['options'].pair_cutoff_bohr=a['options'].auxiliary_cutoff_bohr=1.
    complete=manta.compute_periodic_gdf(**a)
    sr=manta.compute_periodic_gdf_short_range(**{k:v for k,v in a.items() if k!='vectors'})
    m,t=quadrature_long_range(a)
    # The raw high-l source has separate radial/Fourier witnesses. Isolate
    # this new reciprocal assembly to compare with direct polynomial integrals.
    np.testing.assert_allclose(complete.metric-sr.metric,m,atol=2e-12,rtol=2e-12)
    np.testing.assert_allclose(complete.three_center-sr.three_center,t,atol=2e-12,rtol=2e-12)


def test_metric_identity_reuse_and_reciprocal_order_are_explicit():
    a=case();r=manta.compute_periodic_gdf(**a)
    a['options'].pair_cutoff_bohr=4.9
    reused=manta.compute_periodic_gdf(**a)
    np.testing.assert_array_equal(reused.metric,r.metric)
    assert reused.receipt.metric_source_identity==r.receipt.metric_source_identity
    assert reused.receipt.three_center_source_identity!=r.receipt.three_center_source_identity
    permuted=manta.compute_periodic_gdf(**dict(a,vectors=a['vectors'][::-1].copy()))
    np.testing.assert_allclose(permuted.metric,reused.metric,atol=3e-13,rtol=0)
    np.testing.assert_allclose(permuted.three_center,reused.three_center,atol=3e-13,rtol=0)
    assert permuted.receipt.metric_source_identity!=reused.receipt.metric_source_identity
    lattice=a['system'].lattice.copy();lattice[:,[0,1]]=lattice[:,[1,0]]
    left=manta.compute_periodic_gdf(**dict(a,system=vq.PeriodicSystem(3,lattice,a['system'].unit_cell)))
    np.testing.assert_allclose(left.metric,reused.metric,atol=2e-12,rtol=0)
    np.testing.assert_allclose(left.three_center,reused.three_center,atol=2e-12,rtol=0)


def test_engaged_screening_changes_only_the_retained_short_range_source():
    a=case(True);a['system'],a['orbital'],a['auxiliary']=_weighted_gradient_fixture(1)
    a['vectors']=np.array(list(itertools.product(range(-1,2),repeat=3)))@a['system'].reciprocal_lattice().T
    a['options'].pair_cutoff_bohr=8.;a['options'].auxiliary_cutoff_bohr=20.
    exact=manta.compute_periodic_gdf(**a);a['options'].integral_screen_error=1e-3
    r=manta.compute_periodic_gdf(**a);m,t,_=legacy(a)
    np.testing.assert_allclose(r.metric,m,atol=2e-11,rtol=2e-12)
    np.testing.assert_allclose(r.three_center,t,atol=2e-11,rtol=2e-12)
    assert np.max(abs(r.metric-exact.metric))<1e-3 and np.max(abs(r.three_center-exact.three_center))<1e-3
    assert r.receipt.short_range.three_center_shell_calls<exact.receipt.short_range.three_center_shell_calls
    assert r.receipt.pair_fourier_shell_calls==exact.receipt.pair_fourier_shell_calls
    assert r.receipt.source_identity!=exact.receipt.source_identity


def test_complete_source_citation_route(tmp_path):
    from vibeqc.output.citations import assemble
    from tests.test_citations import _plan
    citations=assemble(_plan(tmp_path,basis='sto-3g'),numerics=['manta_periodic_gdf'])
    keys={c.bibtex_key for c in citations.citations}
    assert {'mcmurchie_davidson_1978','sun_berkelbach_gdf_2017','ye_berkelbach_rsgdf_2021'}<=keys
