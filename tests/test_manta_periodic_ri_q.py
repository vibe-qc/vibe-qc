"""Independent finite-source acceptance of streamed, fixed-projector q channels."""
import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta, _vibeqc_core as core
from tests.test_manta_periodic_gdf import case, independent, legacy
from tests.test_manta_periodic_fitting import CAPS


def densities(nk=2, n=2, spins=2, seed=754):
    rng=np.random.default_rng(seed)
    z=rng.normal(size=(spins,nk,n,n))+1j*rng.normal(size=(spins,nk,n,n))
    return np.ascontiguousarray((z+z.swapaxes(-1,-2).conj())/2)


def builder(a, spins=2, block=1, **cutoffs):
    o=manta.RIOptions();o.mode_block=block
    for name,value in cutoffs.items():setattr(o,name,value)
    return manta.PeriodicRIQBuilder(manta.PeriodicGDFTiles(**a),spins,o,a['budget'])


def reference(m,t,d,w,absolute=1e-12,relative=1e-10):
    # Independent NumPy solve, lower-triangle canonicalization and contractions.
    m=np.tril(m)+np.tril(m,-1).conj().T;m[np.diag_indices(len(m))]=m.diagonal().real
    e,u=np.linalg.eigh(m);keep=e>max(absolute,relative*max(0.,e[-1]))
    f=u[:,keep]/np.sqrt(e[keep]);p=u[:,keep]@u[:,keep].conj().T
    b=np.einsum('pl,kpmn->klmn',f.conj(),t)
    rho=np.einsum('k,klmn,kmn->l',w,b.conj(),d.sum(axis=0))
    j=np.einsum('klmn,l->kmn',b,rho)
    k=np.einsum('k,klmp,skpq,klnq->skmn',w,b,d,b.conj())
    rhs=np.einsum('k,kpmn,kmn->p',w,t.conj(),d.sum(axis=0))
    residual=np.linalg.norm(m.conj()@(f.conj()@rho)-rhs)
    return j,k,.5*np.vdot(rho,rho).real,p,residual


@pytest.mark.parametrize('gamma',[False,True])
@pytest.mark.parametrize('spins',[1,2])
@pytest.mark.parametrize('block',[1,2,8])
def test_complex_indefinite_spin_densities_against_two_independent_sources(gamma,spins,block):
    a=case(gamma);b=builder(a,spins,block);d=densities(spins=spins);w=np.array([.23,.77])
    assert np.linalg.eigvalsh(d).min()<0
    r=b.build(d,w,make_j=gamma)
    for m,t in (independent(a),legacy(a)[:2]):
        j,k,energy,p,residual=reference(m,t,d,w)
        np.testing.assert_allclose(r.K,k,atol=4e-12,rtol=3e-12)
        np.testing.assert_allclose(b.projector,p,atol=2e-13,rtol=0)
        if gamma:
            np.testing.assert_allclose(r.J,j,atol=4e-12,rtol=3e-12)
            assert r.receipt.coulomb_energy==pytest.approx(energy,abs=3e-12)
            assert r.receipt.metric_residual_norm==pytest.approx(residual,abs=4e-12)
    assert r.receipt.has_j==gamma and r.receipt.has_k
    assert r.receipt.metric_residual_evaluated==gamma
    assert r.receipt.coupling_passes==(b.metric_info.rank+b.plan.mode_block-1)//b.plan.mode_block
    assert r.receipt.shell_calls+b.metric_info.shell_calls<=b.plan.shell_calls
    assert r.receipt.primitive_products+b.metric_info.primitive_products<=b.plan.primitive_products
    assert r.receipt.hermiticity_residual<1e-12
    assert (r.receipt.source_frame_identity==r.receipt.target_frame_identity)==gamma
    if not gamma:assert r.J.shape==(0,2,2)


@pytest.mark.parametrize('angular,pure',[(1,True),(2,False),(3,True)])
def test_signed_angular_contractions_against_libint_and_independent_fourier_quadrature(angular,pure):
    import itertools
    from tests.test_aux_eri_sr_lattice_enumeration import _weighted_gradient_fixture
    from tests.test_manta_periodic_gdf import quadrature_long_range
    from tests.test_manta_periodic_fitting import libint_reference
    a=case();s,orb,aux=_weighted_gradient_fixture(angular,orbital_angular=1)
    def change(basis):
        shells=basis.shells()
        for sh in shells:sh.pure=pure
        return vq.BasisSet(s.unit_cell_molecule(),shells,'ri-q-angular',True)
    a.update(system=s,orbital=change(orb),auxiliary=change(aux),
        vectors=np.array(list(itertools.product(range(-1,2),repeat=3)))@s.reciprocal_lattice().T+a['q'])
    a['options'].pair_cutoff_bohr=4.3;a['options'].auxiliary_cutoff_bohr=5.1
    d=densities(n=a['orbital'].nbasis);w=np.array([.32,.68]);b=builder(a,block=4)
    r=b.build(d,w,make_j=False);sm,st=libint_reference(a);lm,lt=quadrature_long_range(a)
    references=[(sm+lm,st+lt)]
    if pure:
        references.append(legacy(a)[:2])
    else:
        # The old auxiliary FT route has no Cartesian L>0 implementation.
        # Libint SR plus independent polynomial quadrature still checks it.
        with pytest.raises(ValueError,match='Cartesian'):
            legacy(a)
    for m,t in references:
        _,k,_,_,_=reference(m,t,d,w)
        np.testing.assert_allclose(r.K,k,atol=4e-11,rtol=3e-12)


@pytest.mark.parametrize('rank',[0,1,2])
def test_fixed_projector_truncation_residual_and_zero_rank(rank):
    a=case(True);m,t=independent(a);e=np.linalg.eigvalsh(m)
    cutoff={0:e[-1]*2,1:float(e.mean()),2:1e-12}[rank]
    b=builder(a,absolute_cutoff=cutoff,relative_cutoff=0.)
    d=densities();w=np.array([.41,.59]);r=b.build(d,w)
    j,k,energy,p,residual=reference(m,t,d,w,cutoff,0.)
    assert b.metric_info.rank==rank and b.factor.shape==(2,rank)
    np.testing.assert_allclose(r.J,j,atol=3e-12,rtol=0)
    np.testing.assert_allclose(r.K,k,atol=3e-12,rtol=0)
    np.testing.assert_allclose(b.projector,p,atol=2e-13,rtol=0)
    np.testing.assert_allclose(b.factor.conj().T@b.metric@b.factor,np.eye(rank),atol=3e-14,rtol=0)
    assert r.receipt.coulomb_energy==pytest.approx(energy,abs=3e-12)
    assert r.receipt.metric_residual_norm==pytest.approx(residual,abs=4e-12)
    assert r.receipt.coupling_passes==max(rank,1)
    only_k=b.build(d,w,make_j=False)
    np.testing.assert_array_equal(only_k.K,r.K)
    assert only_k.receipt.coupling_passes==rank
    if not rank:
        assert r.receipt.metric_rhs_norm>0
        assert r.receipt.metric_rhs_norm==pytest.approx(r.receipt.metric_residual_norm)
        assert only_k.receipt.shell_calls==0


def test_redundant_auxiliary_space_and_strict_cutoff_boundary():
    a=case(True);shells=a['auxiliary'].shells()
    a['auxiliary']=vq.BasisSet(a['system'].unit_cell_molecule(),shells+[shells[0]],'redundant-fit',True)
    b=builder(a);assert b.metric_info.rank==2
    d=densities();w=np.array([.36,.64]);r=b.build(d,w)
    j,k,_,p,_=reference(*independent(a),d,w)
    np.testing.assert_allclose(r.J,j,atol=3e-12,rtol=0)
    np.testing.assert_allclose(r.K,k,atol=3e-12,rtol=0)
    np.testing.assert_allclose(b.projector,p,atol=2e-13,rtol=0)
    assert np.linalg.norm(b.projector-np.eye(3))>0.9
    boundary=builder(a,absolute_cutoff=b.metric_info.maximum_eigenvalue,relative_cutoff=0.)
    assert boundary.metric_info.rank==0  # Retain strictly above the cutoff.


def test_significant_negative_metric_is_refused_without_repair():
    a=case(True);a['options'].auxiliary_cutoff_bohr=.5;a['vectors']=np.zeros((1,3))
    m,_=independent(a)
    assert np.linalg.eigvalsh(m)[0]<-3.
    with pytest.raises(RuntimeError,match='significant negative eigenvalue'):
        builder(a)


def test_nonhermitian_coulomb_response_from_unqualified_reciprocal_list_is_refused():
    a=case(True);a['vectors']=a['vectors'][1:].copy()
    with pytest.raises(RuntimeError,match='qualify the reciprocal domain'):
        builder(a).build(densities(),np.array([.3,.7]))


def test_hermiticity_overflow_cannot_admit_invalid_density_even_at_zero_rank():
    a=case(True);b=builder(a,absolute_cutoff=1e9);d=densities()
    d[0,0,0,0]=complex(np.finfo(float).max,np.finfo(float).max)
    with pytest.raises(OverflowError,match='Hermiticity norm'):
        b.build(d,np.array([.3,.7]),make_j=False)


@pytest.mark.parametrize('gamma',[False,True])
def test_converged_splits_match_independent_bare_coulomb_fit(gamma):
    from vibeqc.aux_basis import _rsgdf_shifted_dense_g_mesh
    from tests.test_manta_periodic_gdf import fourier_blocks
    a=case(gamma);a['options'].pair_cutoff_bohr=12.;a['options'].auxiliary_cutoff_bohr=18.
    vectors=_rsgdf_shifted_dense_g_mesh(a['system'],a['q'],80.)
    squared=np.sum(vectors*vectors,axis=1);keep=squared>1e-24
    m,t=fourier_blocks(a,vectors[keep],4*np.pi/abs(np.linalg.det(a['system'].lattice))/squared[keep])
    d=densities();w=np.array([.17,.83]);j,k,energy,_,_=reference(m,t,d,w)
    for omega in (.45,.8):
        a['options'].omega=omega
        a['vectors']=_rsgdf_shifted_dense_g_mesh(a['system'],a['q'],2*omega**2*np.log(1e15))
        r=builder(a).build(d,w,make_j=gamma)
        np.testing.assert_allclose(r.K,k,atol=2e-10,rtol=2e-12)
        if gamma:
            np.testing.assert_allclose(r.J,j,atol=2e-10,rtol=2e-12)
            assert r.receipt.coulomb_energy==pytest.approx(energy,abs=2e-10)


def test_metric_projector_is_fixed_across_builds_and_block_choices_and_exports_are_owned():
    a=case(True);d=densities();w=np.array([.1,.9]);b=builder(a);r=b.build(d,w)
    identity=b.metric_info.source_identity;projector=b.projector;factor=b.factor;metric=b.metric
    for block in (2,8):
        other=builder(a,block=block);s=other.build(d,w)
        assert other.metric_info.source_identity==identity
        assert s.receipt.snapshot_identity==r.receipt.snapshot_identity
        np.testing.assert_allclose(s.J,r.J,atol=1e-14,rtol=0)
        np.testing.assert_allclose(s.K,r.K,atol=1e-14,rtol=0)
    for value in (b.projector,b.factor,b.metric,r.J,r.K):value[:]=np.nan
    np.testing.assert_array_equal(b.projector,projector);np.testing.assert_array_equal(b.factor,factor)
    np.testing.assert_array_equal(b.metric,metric)
    r2=b.build(2*d,w);r3=b.build(d,w[::-1].copy())
    assert r2.receipt.source_identity==r3.receipt.source_identity==identity
    assert r2.receipt.projector_identity==r.receipt.projector_identity
    assert r2.receipt.density_identity!=r.receipt.density_identity
    assert r3.receipt.density_identity==r.receipt.density_identity
    assert r3.receipt.weight_identity!=r.receipt.weight_identity
    assert len({r.receipt.snapshot_identity,r2.receipt.snapshot_identity,r3.receipt.snapshot_identity})==3
    np.testing.assert_allclose(r2.J,2*r.J,atol=1e-13,rtol=0)
    np.testing.assert_allclose(r2.K,2*r.K,atol=1e-13,rtol=0)
    c=b.build(np.asfortranarray(d),np.asfortranarray(w))
    assert c.receipt.snapshot_identity==r.receipt.snapshot_identity
    np.testing.assert_array_equal(c.J,r.J);np.testing.assert_array_equal(c.K,r.K)


@pytest.mark.parametrize('gamma',[False,True])
@pytest.mark.parametrize('truncated',[False,True])
def test_atom_image_gauge_transports_density_from_ket_and_exchange_to_bra(gamma,truncated):
    a=case(gamma);shift=case(gamma,shift=True);d=densities();w=np.array([.21,.79])
    cutoff=float(np.linalg.eigvalsh(independent(a)[0]).mean()) if truncated else 1e-12
    original_fit=builder(a,absolute_cutoff=cutoff);moved_fit=builder(shift,absolute_cutoff=cutoff)
    assert original_fit.metric_info.rank==(1 if truncated else 2)
    r=original_fit.build(d,w,make_j=gamma);moved_d=d.copy();L=a['system'].lattice[:,0]
    auxiliary_phase=np.exp(1j*np.array([0.,a['q']@L]))
    np.testing.assert_allclose(moved_fit.projector,original_fit.projector*auxiliary_phase[:,None]*auxiliary_phase.conj()[None,:],atol=3e-13,rtol=0)
    for ik,k in enumerate(a['ket_kpoints']):
        phase=np.exp(1j*np.array([0.,k@L]));moved_d[:,ik]*=phase[None,:,None]*phase.conj()[None,None,:]
    moved=moved_fit.build(moved_d,w,make_j=gamma)
    for ik,k in enumerate(a['ket_kpoints']):
        phase=np.exp(1j*np.array([0.,(k-a['q'])@L]))
        expected=r.K[:,ik]*phase[None,:,None]*phase.conj()[None,None,:]
        np.testing.assert_allclose(moved.K[:,ik],expected,atol=3e-12,rtol=0)
        if gamma:np.testing.assert_allclose(moved.J[ik],r.J[ik]*phase[:,None]*phase.conj()[None,:],atol=3e-12,rtol=0)
    assert r.receipt.source_frame_identity!=moved.receipt.source_frame_identity
    if gamma:assert moved.receipt.coulomb_energy==pytest.approx(r.receipt.coulomb_energy,abs=3e-12)


def test_reciprocal_gamma_representative_preserves_j_k_without_reusing_frame_identity():
    a=case(True);d=densities();w=np.array([.45,.55]);r=builder(a).build(d,w)
    G=a['system'].reciprocal_lattice()@np.array([2,-1,1])
    shifted=builder(dict(a,q=G,ket_kpoints=a['ket_kpoints']+G)).build(d,w)
    np.testing.assert_allclose(shifted.J,r.J,atol=3e-12,rtol=0)
    np.testing.assert_allclose(shifted.K,r.K,atol=3e-12,rtol=0)
    assert shifted.receipt.source_frame_identity!=r.receipt.source_frame_identity


def test_frame_identity_includes_lattice_separately_from_local_orbital_basis():
    a=case(True);s=manta.PeriodicGDFTiles(**a);d=densities();w=np.array([.45,.55])
    r=manta.PeriodicRIQBuilder(s,budget=a['budget']).build(d,w)
    a['system'].lattice=1.1*a['system'].lattice;a['vectors']=a['vectors']/1.1
    changed=manta.PeriodicGDFTiles(**a)
    assert changed.orbital_basis_identity==s.orbital_basis_identity
    moved=manta.PeriodicRIQBuilder(changed,budget=a['budget']).build(d,w)
    assert moved.receipt.source_frame_identity!=r.receipt.source_frame_identity


def test_coulomb_energy_directional_derivative_and_weighted_reciprocity():
    a=case(True);b=builder(a);d=densities();delta=densities(seed=231);w=np.array([.37,.63]);h=2e-5
    r=b.build(d,w,make_k=False);probe=b.build(delta,w,make_k=False)
    finite_difference=(b.build(d+h*delta,w,make_k=False).receipt.coulomb_energy-
        b.build(d-h*delta,w,make_k=False).receipt.coulomb_energy)/(2*h)
    action=np.einsum('k,skij,kji->',w,delta,r.J).real
    assert finite_difference==pytest.approx(action,abs=2e-9)
    assert action==pytest.approx(np.einsum('k,skij,kji->',w,d,probe.J).real,abs=2e-12)
    assert r.K.shape==(0,2,2,2)
    assert r.receipt.coulomb_energy==pytest.approx(.5*np.einsum('k,skij,kji->',w,d,r.J).real,abs=2e-12)


def test_partial_exchange_channels_have_source_weights_and_conjugate_transfer_energy_closure():
    a=case(True);d=densities();delta=densities(seed=864);w=np.array([.27,.73]);h=2e-5
    diag=builder(a);pairs={}
    for i,j in ((0,1),(1,0)):
        q=a['ket_kpoints'][j]-a['ket_kpoints'][i]
        pairs[i,j]=builder(dict(a,q=q,ket_kpoints=a['ket_kpoints'][j:j+1],vectors=a['vectors']+q))
    def evaluate(d):
        r=diag.build(d,w);K=r.K
        for (i,j),b in pairs.items():K[:,i]+=b.build(d[:,j:j+1].copy(),w[j:j+1],make_j=False).K[:,0]
        energy=r.receipt.coulomb_energy-.5*np.einsum('k,skij,skji->',w,d,K).real
        return energy,r.J[None]-K
    energy,fock=evaluate(d);derivative=(evaluate(d+h*delta)[0]-evaluate(d-h*delta)[0])/(2*h)
    assert derivative==pytest.approx(np.einsum('k,skij,skji->',w,delta,fock).real,abs=4e-9)
    r=pairs[0,1].build(d[:,1:].copy(),np.array([0.]),make_j=False)
    np.testing.assert_array_equal(r.K,np.zeros_like(r.K))
    assert not r.receipt.metric_residual_evaluated


@pytest.mark.parametrize('field,quantity',CAPS)
def test_constructor_and_repeated_coupling_work_caps_exact_and_one_short(field,quantity):
    a=case(True);s=manta.PeriodicGDFTiles(**a);o=manta.RIOptions();o.mode_block=1
    budget=a['budget'];budget.other_live_bytes=321;p=manta.plan_periodic_ri_q(s,2,o,budget)
    for cap,value in CAPS:setattr(budget,cap,getattr(p,value))
    b=manta.PeriodicRIQBuilder(s,2,o,budget);b.build(densities(),np.array([.4,.6]))
    setattr(budget,field,getattr(p,quantity)-1)
    with pytest.raises((ValueError,RuntimeError),match='budget'):manta.plan_periodic_ri_q(s,2,o,budget)
    with pytest.raises((ValueError,RuntimeError),match='budget'):manta.PeriodicRIQBuilder(s,2,o,budget)


def test_joint_external_memory_is_charged_once_and_block_plan_changes_work():
    a=case(True);s=manta.PeriodicGDFTiles(**a);o=manta.RIOptions();o.mode_block=1;b=a['budget']
    p=manta.plan_periodic_ri_q(s,2,o,b);b.other_live_bytes+=987
    assert manta.plan_periodic_ri_q(s,2,o,b).modeled_peak_bytes==p.modeled_peak_bytes+987
    s2=manta.PeriodicGDFTiles(**a)
    assert manta.plan_periodic_ri_q(s2,2,o,b).modeled_peak_bytes==p.modeled_peak_bytes+987
    o.mode_block=2;p2=manta.plan_periodic_ri_q(s,2,o,b)
    assert p.shell_calls-s.plan.metric_shell_calls==2*(p2.shell_calls-s.plan.metric_shell_calls)


def test_plan_excludes_full_three_center_and_full_fitted_tensors():
    a=case(True);s=vq.PeriodicSystem(3,8*np.eye(3),[vq.Atom(2,[0.,0.,0.])])
    def basis(count):
        shells=[core.ShellInfo(0,0,True,[.3+.02*i],[1.],[0.,0.,0.]) for i in range(count)]
        return vq.BasisSet(s.unit_cell_molecule(),shells,'ri-memory-census',True)
    a.update(system=s,orbital=basis(80),auxiliary=basis(256),vectors=np.zeros((1,3)),ket_kpoints=np.zeros((4,3)))
    for cap,_ in CAPS:setattr(a['budget'],cap,2**64-1)
    source=manta.PeriodicGDFTiles(**a);p=manta.plan_periodic_ri_q(source,budget=a['budget'])
    # Count-only admission witness; the large metric/traversal is not executed.
    assert p.modeled_peak_bytes<16*4*256*80**2
    assert p.mode_block==8 and p.maximum_coupling_passes==32


@pytest.mark.parametrize('fault',['dtype','shape','strides','nonfinite','nonhermitian','weights','negative','partial','none'])
def test_input_refusals(fault):
    a=case(True);b=builder(a);d=densities();w=np.array([.5,.5]);kw={}
    if fault=='dtype':d=d.real.copy()
    elif fault=='shape':d=d[0]
    elif fault=='strides':d=d[:,::-1]
    elif fault=='nonfinite':d[0,0,0,0]=np.nan
    elif fault=='nonhermitian':d[0,0,0,1]+=1j
    elif fault=='weights':w=np.array([.5,np.nan])
    elif fault=='negative':w=np.array([-.5,1.5])
    elif fault=='partial':w=np.array([.2,.5])
    elif fault=='none':kw=dict(make_j=False,make_k=False)
    with pytest.raises((ValueError,RuntimeError,TypeError)):b.build(d,w,**kw)


def test_metric_required_options_validated_and_gamma_target_refuses_before_density_payload():
    a=case();b=builder(a)
    with pytest.raises(ValueError,match='Gamma'):b.build(np.full((2,2,2,2),np.nan+0j),np.zeros(2))
    a['options'].compute_metric=False;s=manta.PeriodicGDFTiles(**a)
    with pytest.raises(ValueError,match='metric'):manta.plan_periodic_ri_q(s,budget=a['budget'])
    with pytest.raises(ValueError,match='metric'):manta.PeriodicRIQBuilder(s,budget=a['budget'])
    a['options'].compute_metric=True;s=manta.PeriodicGDFTiles(**a)
    for spins in (0,3):
        with pytest.raises(ValueError,match='spin'):manta.plan_periodic_ri_q(s,spins,budget=a['budget'])
    o=manta.RIOptions();o.mode_block=0
    with pytest.raises(ValueError):manta.PeriodicRIQBuilder(s,options=o,budget=a['budget'])


def test_component_citation_routes():
    from vibeqc.output.citations import load_default_database
    db=load_default_database()
    keys={c.key for c in db.assemble(numerics=['manta_periodic_ri_q']).citations}
    assert {'mcmurchie_davidson_1978','sun_berkelbach_gdf_2017','ye_berkelbach_rsgdf_2021','neese_shark_2023'}<=keys
    assert {'mcmurchie_davidson_1978','sun_berkelbach_gdf_2017','ye_berkelbach_rsgdf_2021'}<={
        c.key for c in db.assemble(numerics=['manta_periodic_gdf_tiles']).citations}


def rational_retained(f,t,d,w):
    """Exact binary-rational stored-source expansion, independent of intervals."""
    from fractions import Fraction as Q
    def z(x):return Q(float(x.real)),Q(float(x.imag))
    def plus(x,y):return x[0]+y[0],x[1]+y[1]
    def prod(x,y):return x[0]*y[0]-x[1]*y[1],x[0]*y[1]+x[1]*y[0]
    def conj(x):return x[0],-x[1]
    def total(xs):
        out=Q(0),Q(0)
        for x in xs:out=plus(out,x)
        return out
    ns,nk,n,_=d.shape;rank=f.shape[1]
    b={(k,l,i,j):total(prod(conj(z(f[p,l])),z(t[k,p,i,j])) for p in range(f.shape[0]))
       for k in range(nk) for l in range(rank) for i in range(n) for j in range(n)}
    rho={l:total(prod((Q(w[k]),Q(0)),prod(conj(b[k,l,i,j]),z(d[s,k,i,j])))
                  for s,k,i,j in np.ndindex(d.shape)) for l in range(rank)}
    j={(k,i,q):total(prod(b[k,l,i,q],rho[l]) for l in range(rank)) for k,i,q in np.ndindex((nk,n,n))}
    k={(s,h,i,jj):total(prod((Q(w[h]),Q(0)),prod(prod(b[h,l,i,a],z(d[s,h,a,bb])),conj(b[h,l,jj,bb])))
       for l in range(rank) for a in range(n) for bb in range(n)) for s,h,i,jj in np.ndindex(d.shape)}
    energy=sum((prod(conj(x),x)[0]*Q(1,2) for x in rho.values()),Q(0))
    return j,k,energy


def assert_retained_rational(result,reference,j=True,k=True):
    from fractions import Fraction as Q
    ej,ek,energy=reference;r=result.receipt
    assert r.retained_j_roundoff_evaluated==j and r.retained_k_roundoff_evaluated==k
    for enabled,actual,bounds,exact in ((j,result.J,result.J_roundoff_bounds,ej),(k,result.K,result.K_roundoff_bounds,ek)):
        if not enabled:
            assert bounds.size==0
            continue
        assert bounds.shape==actual.shape and np.all(np.isfinite(bounds))
        for q in np.ndindex(actual.shape):
            value=actual[q];x,y=exact[q]
            assert abs(Q(float(value.real))-x)+abs(Q(float(value.imag))-y)<=Q(bounds[q])
    if j:
        assert abs(Q(r.coulomb_energy)-energy)<=Q(r.retained_energy_roundoff_bound)
        assert r.retained_j_roundoff_bound==np.max(result.J_roundoff_bounds)
    if k:assert r.retained_k_roundoff_bound==np.max(result.K_roundoff_bounds)


@pytest.mark.parametrize('gamma',[False,True])
@pytest.mark.parametrize('spins',[1,2])
@pytest.mark.parametrize('block',[1,8])
@pytest.mark.parametrize('kind',['ordinary','cancellation','subnormal','zero'])
def test_retained_contraction_exact_rational_enclosures(gamma,spins,block,kind):
    from tests.test_manta_periodic_gdf_tiles import assemble
    a=case(gamma);source=manta.PeriodicGDFTiles(**a);b=builder(a,spins,block)
    d=densities(spins=spins);w=np.array([.27,.73])
    if kind=='cancellation':d*=2.**40;d[...,1,1]=-d[...,0,0]
    elif kind=='subnormal':d*=2.**-1030
    elif kind=='zero':d[:]=0.
    out=b.build(d,w,make_j=gamma,audit_metric_arithmetic=True)
    plain=b.build(d,w,make_j=gamma)
    assert_retained_rational(out,rational_retained(b.factor,assemble(source),d,w),gamma,True)
    for name in ('J','K'):np.testing.assert_array_equal(getattr(out,name),getattr(plain,name))
    assert out.receipt.snapshot_identity==plain.receipt.snapshot_identity
    assert out.receipt.coupling_passes==plain.receipt.coupling_passes
    assert out.receipt.shell_calls==plain.receipt.shell_calls
    assert out.receipt.coulomb_energy==plain.receipt.coulomb_energy
    assert not plain.receipt.retained_k_roundoff_evaluated and plain.K_roundoff_bounds.size==0
    if kind=='zero':
        assert np.all(out.K_roundoff_bounds==0.) and np.all(out.J_roundoff_bounds==0.)


@pytest.mark.parametrize('rank',[0,1,2])
@pytest.mark.parametrize('make_k',[False,True])
def test_retained_rank_and_j_only_exact_rational(rank,make_k):
    from tests.test_manta_periodic_gdf_tiles import assemble
    a=case(True);source=manta.PeriodicGDFTiles(**a);e=np.linalg.eigvalsh(builder(a).metric)
    cutoff={0:float(e[-1]*2),1:float(e.mean()),2:0.}[rank]
    b=builder(a,block=8,absolute_cutoff=cutoff,relative_cutoff=0.)
    d=densities();w=np.array([0.,1.]);out=b.build(d,w,make_k=make_k,audit_metric_arithmetic=True)
    assert b.metric_info.rank==rank and out.receipt.coulomb_contraction=='retained-mode'
    assert_retained_rational(out,rational_retained(b.factor,assemble(source),d,w),True,make_k)
    if rank==0:assert out.receipt.retained_j_roundoff_bound==out.receipt.retained_k_roundoff_bound==out.receipt.retained_energy_roundoff_bound==0.


@pytest.mark.parametrize('gamma',[False,True])
def test_metric_assembly_rank_uses_shared_spectrum_without_changing_fock(gamma):
    a=case(gamma);control=builder(a);a['options'].metric_assembly_rank=True;b=builder(a)
    info=b.metric_info
    assert info.assembly_rank_requested and info.assembly_rank_qualified
    assert 0<info.assembly_perturbation_bound<info.spectrum.rank_stability_radius
    assert info.assembly_certificate_identity
    response=info.assembly_response
    assert response.qualified and response.rank==info.rank
    assert response.perturbation_bound==info.assembly_perturbation_bound
    assert response.factor_inverse_error_bound>=response.inverse_error_bound>0
    assert response.factor_inverse_error_bound>=info.inverse_certificate.inverse_error_bound
    raw_response=info.inverse_certificate.bound_perturbation(info.assembly_perturbation_bound)
    assert response.inverse_error_bound==raw_response.inverse_error_bound
    assert not control.metric_info.assembly_response.qualified
    raw=manta.PeriodicGDFTiles(**a).metric()
    radii=np.array(raw.metric_assembly_error_bounds).reshape(2,2)
    radii=np.tril(radii)+np.tril(radii,-1).T
    assert info.assembly_perturbation_bound>=np.sum(radii,axis=1).max()
    for name in ('canonical_metric_identity','factor_identity','projector_identity','source_identity'):
        assert getattr(info,name)==getattr(control.metric_info,name)
    for name in ('metric','factor','projector'):np.testing.assert_array_equal(getattr(b,name),getattr(control,name))
    d=densities();w=np.array([.3,.7]);result=b.build(d,w,make_j=gamma);expected=control.build(d,w,make_j=gamma)
    for name in ('J','K'):np.testing.assert_array_equal(getattr(result,name),getattr(expected,name))
    with pytest.raises(ValueError,match='assembly rank is unresolved'):
        builder(a,absolute_cutoff=info.maximum_eigenvalue,relative_cutoff=0.)
