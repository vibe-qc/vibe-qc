"""Full-image/full-reciprocal exchange enclosures with independent Fourier sums."""
from decimal import Decimal,localcontext
import numpy as np
import pytest
import vibeqc as vq
from vibeqc import manta,_vibeqc_core as core
from tests.test_manta_periodic_one_electron_images import fixture
from tests.test_manta_periodic_ewald import cube
from tests.test_manta_fourier import _cart,_quadrature

_NODES,_WEIGHTS=np.polynomial.hermite.hermgauss(8)


def transfers(cell,k,radius,inner_radius=None,screened_zero=False):
    labels=cube(radius)
    if inner_radius is not None:labels=labels[np.max(abs(labels),axis=1)>inner_radius]
    base=labels@(2*np.pi*np.linalg.inv(cell));chunks=[]
    for target in k:
        for source in k:
            q=base+target-source;chunks.append(q if screened_zero else q[np.any(q!=0,axis=1)])
    return np.ascontiguousarray(np.concatenate(chunks)),np.r_[0,np.cumsum([len(q) for q in chunks])].astype(np.int64)


def exact_tile(a,b,q):
    """Direct polynomial quadrature on the shifted Gaussian contour, degree <=12.

    Unlike oscillatory real-axis quadrature this remains exact at large Q;
    neither native Fourier recurrences nor envelope coefficients are used.
    """
    nodes,weights=_NODES,_WEIGHTS;ca,cb=_cart(a.l),_cart(b.l)
    A,B=np.array(a.origin),np.array(b.origin);out=np.zeros((len(q),len(ca),len(cb)),complex)
    for alpha,ac in zip(a.exponents,a.coefficients):
        for beta,bc in zip(b.exponents,b.coefficients):
            p=alpha+beta;center=(alpha*A+beta*B)/p
            value=np.ones_like(out)
            for axis in range(3):
                x=nodes[None,:]/np.sqrt(p)+center[axis]-1j*q[:,axis,None]/(2*p)
                pa=(x[:,None,:]-A[axis])**ca[None,:,axis,None]
                pb=(x[:,None,:]-B[axis])**cb[None,:,axis,None]
                value*=np.einsum('qin,qjn,n->qij',pa,pb,weights)/np.sqrt(p)
            factor=ac*bc*np.exp(-alpha*beta/p*np.dot(A-B,A-B)-np.sum(q*q,axis=1)/(4*p)-1j*q@center)
            out+=factor[:,None,None]*value
    if a.pure:out=np.einsum('ai,qij->qaj',manta.cartesian_to_spherical(a.l),out)
    if b.pure:out=np.einsum('qij,bj->qib',out,manta.cartesian_to_spherical(b.l))
    return out


def ordered_fourier(basis,q,cells,k,cutoff):
    shells=basis.shells();sizes=[2*s.l+1 if s.pure else (s.l+1)*(s.l+2)//2 for s in shells]
    offsets=np.cumsum([0,*sizes]);out=np.zeros((len(q),basis.nbasis,basis.nbasis),complex)
    for shift in cells:
        for i,a in enumerate(shells):
            for j,b in enumerate(basis.shells()):
                b.origin=(np.array(b.origin)+shift).tolist()
                if cutoff is not None and np.linalg.norm(np.array(a.origin)-b.origin)>cutoff:continue
                out[:,offsets[i]:offsets[i+1],offsets[j]:offsets[j+1]]+=np.exp(1j*k@shift)*exact_tile(a,b,q)
    return out


def reference(c,d,q,offsets,images,cutoff=None,kernel="coulomb",omega=0.):
    out=np.zeros_like(d);nk=len(c['k']);cells=images@c['cell'].T
    for target,k in enumerate(c['k']):
        for source in range(nk):
            pair=target*nk+source
            for start in range(offsets[pair],offsets[pair+1],4096):
                wave=q[start:min(start+4096,offsets[pair+1])];panel=ordered_fourier(c['basis'],wave,cells,k,cutoff)
                squared=np.sum(wave*wave,axis=1);nonzero=squared>0
                radial=np.zeros_like(squared)
                radial[nonzero]=4*np.pi/squared[nonzero]
                if kernel=='erf':radial*=np.exp(-squared/(4*omega**2)) if omega else 0.
                elif kernel=='erfc' and omega:
                    radial[nonzero]*=-np.expm1(-squared[nonzero]/(4*omega**2))
                    radial[~nonzero]=np.pi/omega**2
                weight=c['weights'][source]/abs(np.linalg.det(c['cell']))*radial
                out[:,target]+=np.einsum('g,gai,sab,gbj->sij',weight,panel.conj(),d[:,source],panel)
    return out


def setup(*,angular=0,pure=True,skew=True,image_radius=0,cutoff=None,reciprocal_radius=2,weights=None):
    k=np.array([[.11,-.07,.03],[-.2,.04,.09]])
    w=np.array([.3,.7]) if weights is None else weights
    c=fixture(angular=angular,pure=pure,skew=skew,images=cube(image_radius),cutoff=cutoff,
              cell_size=6.,exponent=.35,k=k,weights=w)
    q,offsets=transfers(c['cell'],k,reciprocal_radius)
    source=manta.PeriodicExchangeBoundSource(c['source'],q,offsets,reciprocal_radius)
    rng=np.random.default_rng(519);n=c['basis'].nbasis
    d=rng.normal(size=(2,len(k),n,n))+1j*rng.normal(size=(2,len(k),n,n));d=np.ascontiguousarray(d+d.swapaxes(-1,-2).conj())
    return c,q,offsets,source,d


def calculation(**kwargs):
    c,q,o,s,d=setup(**kwargs);bound=s.build(d)
    native=manta.compute_periodic_exchange(c['basis'],c['cell'],q,o,np.ascontiguousarray(c['images']@c['cell'].T),c['k'],c['weights'],d,64,c['cutoff'])
    finite=reference(c,d,q,o,c['images'],c['cutoff'])
    # Separate full-image and reciprocal enlargements as well as the joint limit.
    image=reference(c,d,q,o,cube(2))
    full_q,full_o=transfers(c['cell'],c['k'],12)
    full=reference(c,d,full_q,full_o,cube(2))
    tail_q,tail_o=transfers(c['cell'],c['k'],14,inner_radius=12)
    reciprocal_refinement=reference(c,d,tail_q,tail_o,cube(2))
    probes=np.array([[0.,0.,0.],[.3,-.7,.4],[2.,-1.,.7],[8.,3.,-2.]])
    image_refinement=max(float(np.max(abs(ordered_fourier(c['basis'],probes,cube(3)@c['cell'].T,k,None)
        -ordered_fourier(c['basis'],probes,cube(2)@c['cell'].T,k,None)))) for k in c['k'])
    return dict(context=c,q=q,offsets=o,source=s,d=d,bound=bound,native=native.values,finite=finite,image=image,full=full,
                reciprocal_refinement=reciprocal_refinement,image_refinement=image_refinement)


@pytest.mark.parametrize('angular,pure,skew',[(0,True,False),(1,False,True)])
def test_full_exchange_against_unreduced_independent_limit(angular,pure,skew):
    r=calculation(angular=angular,pure=pure,skew=skew,cutoff=.3 if angular else None)
    assert np.max(abs(r['reciprocal_refinement']))<3e-12 and r['image_refinement']<3e-12
    np.testing.assert_allclose(r['native'],r['finite'],atol=3e-11,rtol=1e-12)
    assert np.all(abs(r['image']-r['native'])<=r['bound'].pair_image_error_bound+3e-11)
    assert np.all(abs(r['full']-r['image'])<=r['bound'].reciprocal_error_bound+3e-11)
    assert np.all(abs(r['full']-r['native'])<=r['bound'].error_bound+3e-11)
    error=-.5*np.einsum('k,skij,skji->s',r['context']['weights'],r['d'],r['full']-r['native']).real
    assert np.all(abs(error)<=np.array(r['bound'].receipt.energy_error_bounds)+3e-11)


@pytest.mark.parametrize('angular,pure',[(0,True),(2,False),(4,False),(6,True)])
def test_fourier_contour_and_angular_image_envelopes(angular,pure):
    c,q,o,s,d=setup(angular=angular,pure=pure,cutoff=.3)
    wave=np.array([[0.,0.,0.],[.3,-.7,.4],[2.,-1.,.7],[8.,3.,-2.]])
    a,b=c['basis'].shells()
    np.testing.assert_allclose(exact_tile(a,b,wave[:3]),_quadrature(a,b,wave[:3]),atol=3e-11,rtol=1e-11)
    finite=ordered_fourier(c['basis'],wave,c['images']@c['cell'].T,c['k'][0],c['cutoff'])
    full=ordered_fourier(c['basis'],wave,cube(2)@c['cell'].T,c['k'][0],None)
    decay=np.exp(-s.receipt.fourier_decay*np.sum(wave*wave,axis=1))[:,None,None]
    assert np.all(abs(finite)<=s.retained_amplitudes[None]*decay+3e-12)
    assert np.all(abs(full-finite)<=s.omitted_amplitudes[None]*decay+3e-12)


def test_density_scaling_zero_weights_ownership_and_positive_underflow():
    c,q,o,s,d=setup(weights=np.array([0.,1.]))
    a=s.build(d);b=s.build(np.ascontiguousarray(-2*d))
    np.testing.assert_allclose(b.error_bound,2*a.error_bound,rtol=2e-11,atol=0)
    np.testing.assert_allclose(b.receipt.energy_error_bounds,4*np.array(a.receipt.energy_error_bounds),rtol=2e-11,atol=0)
    zero=s.build(np.zeros_like(d));assert np.max(zero.error_bound)==0 and max(zero.receipt.energy_error_bounds)==0
    tiny=s.build(np.ascontiguousarray(d*1e-310));assert np.max(tiny.error_bound)>0 and min(tiny.receipt.energy_error_bounds)>0
    unused=d.copy();unused[:,0]*=100;np.testing.assert_array_equal(s.build(unused).error_bound,a.error_bound)
    saved=a.error_bound;a.error_bound[:]=0;np.testing.assert_array_equal(a.error_bound,saved)
    saved=s.omitted_amplitudes;s.omitted_amplitudes[:]=0;np.testing.assert_array_equal(s.omitted_amplitudes,saved)
    with pytest.raises(AttributeError):s.receipt.fourier_decay=1.


@pytest.mark.parametrize('field,planned',[('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),
    ('primitive_quartets','primitive_pairs'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_all_constructor_caps_before_payload_reads(field,planned):
    c,q,o,s,d=setup();b=manta.ResourceBudget();p=s.plan_source(c['source'],len(q),2,1,b)
    setattr(b,field,getattr(p,planned));same=manta.PeriodicExchangeBoundSource(c['source'],q,o,2,b)
    assert same.receipt.source_identity==s.receipt.source_identity
    setattr(b,field,getattr(p,planned)-1)
    with pytest.raises((ValueError,RuntimeError),match='budget'):manta.PeriodicExchangeBoundSource(c['source'],np.full_like(q,np.nan),o,2,b)


@pytest.mark.parametrize('field,planned',[('memory_bytes','modeled_peak_bytes'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_density_caps_before_payload_reads(field,planned):
    c,q,o,s,d=setup();b=manta.ResourceBudget();p=s.plan_build(len(d),b);setattr(b,field,getattr(p,planned));s.build(d,b)
    setattr(b,field,getattr(p,planned)-1)
    with pytest.raises((ValueError,RuntimeError),match='budget'):s.build(np.full_like(d,np.nan),b)


def test_domain_shape_reversal_and_density_refusals():
    c,q,o,s,d=setup()
    for bad in (q.astype(np.float32),q[::2]):
        with pytest.raises((ValueError,TypeError)):manta.PeriodicExchangeBoundSource(c['source'],bad,o,2)
    with pytest.raises(ValueError,match='cube'):manta.PeriodicExchangeBoundSource(c['source'],q,o,1)
    with pytest.raises((ValueError,RuntimeError,OverflowError),match='budget|overflow'):manta.PeriodicExchangeBoundSource(c['source'],q,o,10**6)
    bad=q.copy();bad[0]=bad[1]
    with pytest.raises(ValueError,match='unique'):manta.PeriodicExchangeBoundSource(c['source'],bad,o,2)
    bad=q.copy();bad[0]+=2*np.pi*np.linalg.inv(c['cell']).T[:,0]
    with pytest.raises(ValueError):manta.PeriodicExchangeBoundSource(c['source'],bad,o,3)
    bad=d.copy();bad[0,0,0,1]+=.1
    with pytest.raises(ValueError,match='Hermitian'):s.build(bad)


def hf_case(*,spin='restricted',z=2,shifted=False,q0='strict-zero-mode',radius=30,iterations=40,
            image_radius=1,cutoff=1.,eta=.6):
    from tests.test_manta_periodic_bupo_hf import case
    a,c=case(spin=spin,z=z,shifted=shifted,q0=q0,iterations=iterations,cell_size=6.,
             shell_exponents=[4.,6.] if z!=3 else [4.,6.,9.],exchange_radius=radius,
             cutoff=cutoff,image_radius=image_radius,eta=eta,real_radius=4,reciprocal_radius=10)
    a['options'].bound_exchange=True;a['options'].exchange_bound_cube_radius=radius
    return a,c


def test_cached_reference_preserves_fock_and_orthonormal_scf():
    from tests.test_manta_periodic_bupo_hf import case,independent
    a,c=case(spin='unrestricted',z=3,shifted=True,q0='BvK-ewald',exchange_radius=2)
    direct=independent(a,c,fourier_reference=ordered_fourier)
    cached=independent(a,c,fourier_reference=ordered_fourier,cache_exchange=True,density_metric='orthonormal')
    rng=np.random.default_rng(359);d=rng.normal(size=direct['density'].shape)+1j*rng.normal(size=direct['density'].shape)
    d=d+d.conj().swapaxes(-1,-2)
    original=direct['build'](d);rebuilt=cached['build'](d)
    np.testing.assert_allclose(rebuilt['fock'],original['fock'],atol=3e-11,rtol=1e-12)
    assert abs(rebuilt['energy']-original['energy'])<3e-11
    np.testing.assert_allclose(cached['density'],direct['density'],atol=3e-9,rtol=2e-10)
    assert abs(cached['converged']['energy']-direct['converged']['energy'])<3e-11
    assert cached['density_metric']=='orthonormal' and cached['density_change']<1e-13


@pytest.mark.parametrize('spin,z,q0',[('restricted',2,'strict-zero-mode'),('unrestricted',1,'BvK-ewald')])
def test_complete_hf_explicit_exchange_admission_and_fixed_hamiltonian(spin,z,q0):
    a,c=hf_case(iterations=1,spin=spin,z=z,q0=q0);r=core._manta.run_periodic_bupo_hf(**a)
    assert r.receipt.exchange_accuracy_passed and not r.converged
    assert r.receipt.maximum_observed_exchange_matrix_error<=1e-8
    assert r.receipt.maximum_observed_exchange_energy_error<=1e-8
    assert r.receipt.exchange_bounds.exchange_scale==(.5 if spin=='restricted' else 1.)
    a['options'].bound_exchange=False;old=core._manta.run_periodic_bupo_hf(**a)
    np.testing.assert_array_equal(r.density,old.density);np.testing.assert_array_equal(r.fock,old.fock)
    assert r.receipt.energy==old.receipt.energy and old.receipt.exchange_bounds is None
    for name in ('maximum_exchange_matrix_error','maximum_exchange_energy_error'):
        a,c=hf_case(radius=2,iterations=1);setattr(a['options'],name,1e-300)
        with pytest.raises(ValueError,match='exchange accuracy target'):core._manta.run_periodic_bupo_hf(**a)



def occupied_envelope_case(angular=0, image_radius=0, reciprocal_radius=2):
    from tests.test_bounded_periodic_rhf_solver import _options
    c,q,offsets,source,_=setup(angular=angular,weights=np.array([.5,.5]),
        image_radius=image_radius,reciprocal_radius=reciprocal_radius)
    s=manta.compute_bloch_one_electron(c['basis'],'overlap',
        np.ascontiguousarray(c['images']@c['cell'].T),c['k'],c['cutoff']).values
    return c,q,offsets,source,np.ascontiguousarray(s),_options()


def occupied_samples(s,populations,count=12,seed=708):
    rng=np.random.default_rng(seed);e,u=np.linalg.eigh(s);rank=np.sum(e>1e-8,axis=1)
    assert np.all(rank==rank[0]);r=int(rank[0]);x=u[:,:,-r:]/np.sqrt(e[:,-r:])[:,None,:]
    result=np.zeros((count,2,*s.shape),complex)
    for sample in result:
        for k in range(len(s)):
            for spin,occupied in enumerate(populations):
                v=np.linalg.qr(rng.normal(size=(r,r))+1j*rng.normal(size=(r,r)))[0][:,:occupied]
                c=x[k]@v;sample[spin,k]=c@c.conj().T
    return result


@pytest.mark.parametrize('angular,populations',[(0,(1,1)),(0,(1,0)),(0,(0,0)),(1,(2,1))])
def test_occupied_exchange_enclosure_covers_all_spin_subspaces(angular,populations):
    c,q,o,source,s,scf=occupied_envelope_case(angular)
    result=source.bound_occupied_subspaces(s,*populations,scf)
    shared=source.bound_density_envelope(result.density)
    assert shared.certificate_identity==result.certificate_identity
    np.testing.assert_array_equal(shared.error_bound,result.error_bound)
    assert result.source_identity==source.receipt.source_identity
    assert not result.roundoff_included
    assert result.energy_error_bound==min(result.entry_energy_error_bound,result.l1_energy_error_bound)
    for d in occupied_samples(s,populations):
        assert np.all(np.sum(abs(d),axis=0)/len(s)<=result.density.weighted_entry_bounds+3e-12)
        assert np.all(np.sum(abs(d),axis=(0,2,3))<=result.density.per_k_l1+3e-12)
        individual=source.build(d)
        assert np.all(individual.error_bound<=result.error_bound[None]*(1+3e-11)+3e-13)
        assert sum(individual.receipt.energy_error_bounds)<=result.energy_error_bound*(1+3e-11)+3e-13
    if populations==(0,0):
        assert np.max(result.error_bound)==result.energy_error_bound==0.
    else:
        assert result.maximum_error>0 and result.energy_error_bound>0
    for name in ('error_bound','pair_image_error_bound','reciprocal_error_bound'):
        a=getattr(result,name);assert a.shape==s.shape and a.flags.owndata and not a.flags.writeable
        a.flags.writeable=True;a[:]=0
        np.testing.assert_array_equal(getattr(result,name),getattr(shared,name))


@pytest.mark.parametrize('angular',[0,1])
def test_occupied_exchange_independent_fourier_limit(angular):
    c,q,o,source,s,scf=occupied_envelope_case(angular)
    populations=(1,0) if angular==0 else (2,1)
    result=source.bound_occupied_subspaces(s,*populations,scf)
    d=occupied_samples(s,populations,count=2).reshape((4,*s.shape))
    finite=reference(c,d,q,o,c['images'])
    full_q,full_o=transfers(c['cell'],c['k'],12)
    full=reference(c,d,full_q,full_o,cube(2))
    native=manta.compute_periodic_exchange(c['basis'],c['cell'],q,o,
        np.ascontiguousarray(c['images']@c['cell'].T),c['k'],c['weights'],d,64,c['cutoff']).values
    np.testing.assert_allclose(native,finite,atol=3e-11,rtol=1e-12)
    assert np.all(abs(full-native)<=result.error_bound[None]+3e-11)
    energy=.5*abs(np.einsum('k,skij,skji->s',c['weights'],d,full-native)).reshape(2,2).sum(axis=1)
    assert np.all(energy<=result.energy_error_bound+3e-11)


@pytest.mark.parametrize('field,planned',[('memory_bytes','modeled_peak_bytes'),
    ('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
@pytest.mark.parametrize('reuse',[False,True])
def test_occupied_exchange_admission_before_numerical_payloads(field,planned,reuse):
    c,q,o,source,s,scf=occupied_envelope_case();b=manta.ResourceBudget();b.other_live_bytes=27341
    density=source.bound_occupied_subspaces(s,1,0,scf).density
    p=source.plan_density_envelope(b) if reuse else source.plan_occupied_subspaces(1,0,scf,b)
    setattr(b,field,getattr(p,planned))
    run=lambda value:source.bound_density_envelope(value,b) if reuse else source.bound_occupied_subspaces(value,1,0,scf,b)
    run(density if reuse else s)
    setattr(b,field,getattr(p,planned)-1)
    with pytest.raises((ValueError,RuntimeError),match='budget'):run(density if reuse else np.full_like(s,np.nan))
    # Existing source construction has already paid shell/primitive work.
    assert p.shell_calls==p.primitive_pairs==0


def test_occupied_exchange_rank_order_and_contract_refusals():
    c,q,o,source,s,scf=occupied_envelope_case(1)
    e,u=np.linalg.eigh(s);e[:,:2]=0;s=np.ascontiguousarray((u*e[:,None,:])@u.conj().swapaxes(-1,-2))
    result=source.bound_occupied_subspaces(s,1,1,scf);assert result.density.memory.retained_rank==2
    other=source.bound_occupied_subspaces(np.asfortranarray(s),1,1,scf)
    assert other.certificate_identity==result.certificate_identity
    assert source.bound_occupied_subspaces(s,1,0,scf).certificate_identity!=result.certificate_identity
    with pytest.raises(ValueError,match='occupied-plus-virtual'):source.bound_occupied_subspaces(s,2,1,scf)
    with pytest.raises(ValueError):source.bound_occupied_subspaces(s.real,1,1,scf)
    with pytest.raises(ValueError,match='shape'):source.bound_occupied_subspaces(s[:1].copy(),1,1,scf)
    c,q,o,unequal,d=setup(angular=1)
    with pytest.raises(ValueError,match='uniform'):unequal.bound_density_envelope(result.density)
    c,q,o,small,s,scf=occupied_envelope_case()
    with pytest.raises(ValueError,match='matching'):small.bound_density_envelope(result.density)
    with pytest.raises(TypeError):source.bound_density_envelope(result.density.weighted_entry_bounds)


def ri_exchange_envelope_job(directory,method='rhf',species='Be',limit=100):
    from tests.test_manta_periodic_ri_preparation import prepared_job
    job,a,k=prepared_job(directory,method,species,limit)
    domains=getattr(manta,'build_periodic_ri_'+method+'_domains')(**a)
    labels=np.rint(domains.translations@np.linalg.inv(a['system'].lattice).T).astype(np.int64)
    po=manta.PeriodicPairOptions();po.order=0;po.pair_cutoff_bohr=a['options'].pair_cutoff_bohr
    pairs=manta.PeriodicPairSource(a['basis'],a['system'].lattice,labels,k,np.full(len(k),1./len(k)),po,a['budget'])
    # The actual exact-K cutoff sphere is contained in this declared cube.
    source=manta.PeriodicExchangeBoundSource(pairs,domains.pair_vectors,domains.transfer_offsets,6,a['budget'])
    return job,a,k,domains,source


@pytest.mark.parametrize('method,species,limit',[('rhf','Be',100),('uhf','H',100),('uhf','Li',3)])
def test_uniform_exchange_certificate_at_complete_ri_job_state(tmp_path,method,species,limit):
    job,a,k,domains,source=ri_exchange_envelope_job(tmp_path,method,species,limit)
    result=vq.run_periodic_job(**job);assert result.converged==(limit==100)
    n=result.native;ne=a['system'].n_electrons();nb=(ne-a['system'].multiplicity+1)//2;na=ne-nb
    uniform=source.bound_occupied_subspaces(n.overlap,na,nb,a['options'].scf)
    d=np.array([n.density/2,n.density/2]) if method=='rhf' else n.density
    fixed=source.build(d)
    assert np.all(fixed.error_bound<=uniform.error_bound[None]*(1+3e-11))
    assert sum(fixed.receipt.energy_error_bounds)<=uniform.energy_error_bound*(1+3e-11)
    assert uniform.source_identity==source.receipt.source_identity



def test_uniform_exchange_preserves_original_mesh_order():
    c,q,o,source,s,scf=occupied_envelope_case();first=source.bound_occupied_subspaces(s,1,0,scf)
    k=c['k'][::-1].copy()
    from tests.test_manta_periodic_pairs import options
    pairs=manta.PeriodicPairSource(c['basis'],c['cell'],c['images'],k,c['weights'],options(0,c['cutoff'],.2))
    q,o=transfers(c['cell'],k,2);reordered=manta.PeriodicExchangeBoundSource(pairs,q,o,2)
    second=reordered.bound_occupied_subspaces(s[::-1].copy(),1,0,scf)
    np.testing.assert_allclose(first.error_bound[::-1],second.error_bound,atol=0,rtol=3e-13)
    assert first.source_identity!=second.source_identity and first.certificate_identity!=second.certificate_identity
    assert first.energy_error_bound==pytest.approx(second.energy_error_bound,rel=3e-13)



def compact_occupied_envelope_case():
    a,c=hf_case()
    source=manta.PeriodicExchangeBoundSource(c['pairs'],a['exchange_vectors'],a['transfer_offsets'],30,a['budget'])
    s=manta.compute_bloch_one_electron(c['basis'],'overlap',
        np.ascontiguousarray(c['images']@c['cell'].T),c['k'],c['cutoff']).values
    return a,c,source,np.ascontiguousarray(s)


def test_compact_uniform_exchange_certificate_meets_tight_target():
    a,c,source,s=compact_occupied_envelope_case()
    result=source.bound_occupied_subspaces(s,1,1,a['options'].scf,a['budget'])
    assert 0<result.maximum_error<1e-8 and 0<result.energy_error_bound<1e-8
    for d in occupied_samples(s,(1,1),count=24):
        individual=source.build(d,a['budget'])
        assert np.all(individual.error_bound<=result.error_bound[None]*(1+3e-11))
        assert sum(individual.receipt.energy_error_bounds)<=result.energy_error_bound*(1+3e-11)



def exchange_accuracy_targets(value=1e12,ao_radius=2,reciprocal_radius=6):
    o=manta.PeriodicExchangeAccuracyOptions();o.matrix_error=o.energy_error=value
    o.ao_completion_cube_radius=ao_radius;o.reciprocal_completion_cube_radius=reciprocal_radius
    return o


def ri_exchange_accuracy_job(directory,method='rhf',species='Be',limit=100,automatic=False,bvk=True):
    import dataclasses
    from tests.test_manta_periodic_one_electron_images import ri_image_job
    job,a,k,scf=ri_image_job(directory,method,species,limit,automatic)
    target=exchange_accuracy_targets()
    job['periodic_integrals']=dataclasses.replace(job['periodic_integrals'],exchange_accuracy=target)
    job['exchange_exxdiv']='ewald' if bvk else 'none'
    if method in ('rhf','uhf'):
        a['options'].exchange_accuracy=target;a['options'].exchange_q0='BvK-ewald' if bvk else 'strict-zero-mode'
    return job,a,k,scf


def compact_ri_exchange_accuracy_job(directory,cutoff=512.,target=1e-8,bvk=True):
    import dataclasses
    from tests.test_manta_periodic_ri_preparation import prepared_job
    from tests.test_manta_periodic_ri_paired import auxiliary
    base,c=hf_case();job,a,k=prepared_job(directory,'rhf','He')
    a.update(system=base['system'],basis=c['basis'],mesh=base['mesh'],reciprocal_energy_cutoff=cutoff)
    a['auxiliary']=auxiliary(a['system'],[atom.xyz for atom in a['system'].unit_cell]);k=c['k']
    o=a['options'];o.alpha=.6;o.pair_cutoff_bohr=o.fitting.pair_cutoff_bohr=1.;o.nuclear_cutoff_bohr=4.;o.reciprocal_batch=64
    o.fitting.auxiliary_cutoff_bohr=12.1;o.fitting.coupling_multipole_error=0.
    o.exchange_q0='BvK-ewald' if bvk else 'strict-zero-mode';o.exchange_accuracy=exchange_accuracy_targets(target,2,40)
    a['budget'].memory_bytes=4*2**30
    job.update(system=a['system'],basis=a['basis'],kpoints=a['mesh'],exchange_exxdiv='ewald' if bvk else 'none')
    job['coulomb']=dataclasses.replace(job['coulomb'],auxiliary_basis=a['auxiliary'])
    job['periodic_integrals']=dataclasses.replace(job['periodic_integrals'],reciprocal_energy_cutoff=cutoff,pair_cutoff_bohr=1.,
        nuclear_cutoff_bohr=4.,alpha=.6,reciprocal_batch=64,exchange_accuracy=o.exchange_accuracy)
    job['periodic_fitting']=dataclasses.replace(job['periodic_fitting'],auxiliary_cutoff_bohr=12.1,coupling_multipole_error=0.)
    return job,a,k,o.scf


@pytest.mark.parametrize('method,species,limit,automatic,bvk',[
    ('rhf','Be',100,False,False),('rhf','Be',100,True,True),('uhf','H',100,False,True),
    ('uhf','Li',3,True,True),('pbe0','Be',100,True,True),('pbe0','Li',3,False,False)])
def test_complete_ri_exchange_admission_and_unchanged_state(tmp_path,method,species,limit,automatic,bvk):
    import dataclasses
    from vibeqc.guess_read import _spin_density_k_from_qvf_bloch_payload
    job,a,k,scf=ri_exchange_accuracy_job(tmp_path,method,species,limit,automatic,bvk)
    result=vq.run_periodic_job(**job);n=result.native;c=n.receipt.exchange_accuracy
    assert c.qualified and c.bvk_correction==bvk and result.converged==(limit==100)
    assert (n.receipt.accuracy_density is not None)==automatic
    assert n.plan.density_envelope and n.plan.exchange_accuracy
    assert (c.images is not None)==bvk
    coefficient=.25 if method=='pbe0' else 1.
    assert c.method_scale==coefficient/(2 if species=='Be' else 1)
    assert c.maximum_error<=c.targets.matrix_error and c.energy_error_bound<=c.targets.energy_error
    np.testing.assert_allclose(c.error_bound,c.method_scale*c.exchange.error_bound+c.correction_error_bound,rtol=1e-10,atol=0)
    assert c.exchange.pair_source_identity==n.receipt.one_electron_accuracy.images.receipt.pair_source_identity
    total=n.density[0] if n.density.ndim==4 and n.density.shape[0]==1 else n.density
    d=np.array([total/2,total/2]) if species=='Be' else n.density
    h=c.exchange.density.weighted_entry_bounds
    assert np.all(np.sum(abs(d),axis=0)/len(k)<=h+3e-12)
    saved=_spin_density_k_from_qvf_bloch_payload(job['output'].with_suffix('.qvf'));np.testing.assert_array_equal(saved,d)
    evidence=result.integral_provenance['exchange_accuracy'];assert evidence['certificate_identity']==c.certificate_identity
    assert evidence['excludes'] and not evidence['exchange']['roundoff_included']
    assert 'manta_periodic_exchange_bounds' in result.integral_provenance['citation_numerics']
    target=c.targets;target.matrix_error=0.;assert c.targets.matrix_error>0
    job['periodic_integrals']=dataclasses.replace(job['periodic_integrals'],exchange_accuracy=None)
    job['output']=tmp_path/'disabled'/'job';old=vq.run_periodic_job(**job)
    for name in ('density','fock','hcore','overlap'):np.testing.assert_array_equal(getattr(n,name),getattr(old.native,name))
    assert result.energy==old.energy and n.receipt.source_identity==old.native.receipt.source_identity
    assert old.native.receipt.exchange_accuracy is None


@pytest.mark.parametrize('field,quantity',[('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),
    ('primitive_quartets','primitive_pairs'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_complete_exchange_accuracy_all_resource_caps_before_payload(tmp_path,field,quantity):
    from tests.test_manta_periodic_ri_domains import explicit
    job,a,k,scf=ri_exchange_accuracy_job(tmp_path,limit=1)
    domains=manta.build_periodic_ri_rhf_domains(**a);raw=explicit(a,domains)
    p=manta.plan_periodic_ri_rhf(raw['basis'],raw['auxiliary'],raw['system'],raw['mesh'],len(raw['reciprocal_vectors']),
        int(max(np.diff(raw['transfer_offsets']))),len(raw['translations']),raw['options'],raw['budget'])
    assert p.exchange_accuracy and p.density_envelope
    for key,value in [('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),('primitive_quartets','primitive_pairs'),
                      ('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')]:setattr(raw['budget'],key,getattr(p,value))
    result=core._manta.run_periodic_ri_rhf(**raw);assert result.receipt.exchange_accuracy.qualified
    setattr(raw['budget'],field,getattr(p,quantity)-1);raw['translations']=np.full_like(raw['translations'],np.nan)
    with pytest.raises((ValueError,RuntimeError),match='budget'):core._manta.run_periodic_ri_rhf(**raw)


def test_exchange_accuracy_targets_and_unqualified_methods_refuse(tmp_path):
    import dataclasses
    job,a,_,_=ri_exchange_accuracy_job(tmp_path,limit=1)
    for field in ('matrix_error','energy_error'):
        target=exchange_accuracy_targets();setattr(target,field,1e-300)
        job['periodic_integrals']=dataclasses.replace(job['periodic_integrals'],exchange_accuracy=target)
        with pytest.raises(ValueError,match='exchange accuracy target'):vq.run_periodic_job(**job)
        for value in (0.,-1.,float('inf'),float('nan')):
            setattr(target,field,value)
            with pytest.raises(ValueError,match='finite positive'):dataclasses.replace(job['periodic_integrals'],exchange_accuracy=target)
    with pytest.raises(TypeError,match='PeriodicExchangeAccuracyOptions'):
        dataclasses.replace(job['periodic_integrals'],exchange_accuracy=object())
    for method in ('rks','hse06','hse06-ri-k'):
        job,a,_,_=ri_exchange_accuracy_job(tmp_path/method,method,limit=1)
        with pytest.raises(ValueError,match='exchange accuracy requires'):vq.run_periodic_job(**job)
    job,a,_,_=ri_exchange_accuracy_job(tmp_path/'ri-k',limit=1)
    job['exchange']=manta.ExchangeBuildOptions(algorithm='ri')
    with pytest.raises(ValueError,match='exchange accuracy requires'):vq.run_periodic_job(**job)
    job['exchange']=manta.ExchangeBuildOptions(algorithm='exact');job['coulomb']=manta.CoulombBuildOptions(algorithm='exact')
    job['periodic_fitting']=None
    job['periodic_integrals']=dataclasses.replace(job['periodic_integrals'],one_electron_accuracy=None)
    with pytest.raises(NotImplementedError,match='requires a periodic RI method'):vq.run_periodic_job(**job)


@pytest.mark.parametrize('radius',[0,1,2])
def test_explicit_exchange_image_cube_covers_retained_support(radius):
    c,q,o,source,s,scf=occupied_envelope_case(image_radius=1)
    if radius==0:
        with pytest.raises(ValueError,match='contain every retained image'):
            manta.PeriodicExchangeBoundSource(c['source'],q,o,2,ao_completion_cube_radius=radius)
        return
    new=manta.PeriodicExchangeBoundSource(c['source'],q,o,2,ao_completion_cube_radius=radius)
    assert new.plan.ao_cube_radius==radius
    if radius==1:assert new.receipt.source_identity==source.receipt.source_identity
    result=new.bound_occupied_subspaces(s,1,0,scf)
    for d in occupied_samples(s,(1,0),count=4):assert np.all(new.build(d).error_bound<=result.error_bound[None]*(1+3e-11))


def test_complete_compact_ri_exchange_meets_tight_targets(tmp_path):
    job,a,k,scf=compact_ri_exchange_accuracy_job(tmp_path)
    result=vq.run_periodic_job(**job);assert result.converged
    c=result.native.receipt.exchange_accuracy
    assert 0<c.maximum_error<=1e-8 and 0<c.energy_error_bound<=1e-8
    assert 0<c.maximum_correction_error<c.maximum_error and c.bvk_correction


def fitting_case(nk=2,cutoff=1e-12,component=True,kernel=None):
    from tests.test_manta_periodic_ri_fock import case
    from tests.test_manta_periodic_ri_multipole import compact_basis
    from tests.test_manta_periodic_pairs import options
    a=case(nk,cutoff=cutoff)
    for name in ('orbital','auxiliary'):a[name]=compact_basis(a[name],a['system'])
    if kernel is not None:
        from tests.test_manta_periodic_range_fitting import physical
        a['make_j']=False;a['make_k']=True;a['exchange_kernel']=physical(*kernel)
    ri=manta.PeriodicRIFockBuilder(**a);images=cube(0)
    pair=manta.PeriodicPairSource(a['orbital'],a['system'].lattice,images,a['momenta'],a['weights'],options(0,None,.2),a['budget'])
    q,offsets=transfers(a['system'].lattice,a['momenta'],2)
    o=manta.PeriodicRIExchangeFittingOptions();o.density_l1_limit=12.;o.reciprocal_completion_cube_radius=2;o.ao_completion_cube_radius=1
    if component:o.density_entry_limits=np.full((nk,a['orbital'].nbasis,a['orbital'].nbasis),2.).ravel().tolist()
    c=dict(basis=a['orbital'],cell=a['system'].lattice,k=a['momenta'],weights=a['weights'])
    return a,ri,pair,q,offsets,o,c


@pytest.mark.parametrize('nk',[1,2])
@pytest.mark.parametrize('cutoff',[1e-12,.5,1e9])
@pytest.mark.parametrize('component',[False,True])
def test_exchange_fitting_family_encloses_independent_finite_and_full_sources(nk,cutoff,component):
    from tests.test_manta_periodic_ri_q import densities
    from tests.test_manta_periodic_ri_fock import oracle
    a,ri,pair,q,offsets,o,c=fitting_case(nk,cutoff,component)
    assert ri.info.metrics[0].rank==({1e-12:2,.5:1,1e9:0}[cutoff])
    result=manta.certify_periodic_ri_exchange_fitting(ri,pair,q,offsets,o,a['budget'])
    assert result.probes==nk*a['orbital'].nbasis**2==result.plan.maximum_probes
    assert result.qualified and not result.targets_requested
    assert result.probe_roundoff_included and not result.source_construction_included
    assert result.ri_source_identity==ri.info.source_identity and len(result.certificate_identity)==64
    assert result.maximum_error==np.max(result.error_bounds)
    assert result.ri_shell_calls+result.reference_shell_calls<=result.plan.shell_calls
    full_q,full_offsets=transfers(c['cell'],c['k'],8)
    for seed in [412,902]:
        d=densities(nk,seed=seed)*.08;result.validate_density(d)
        actual=ri.build_exchange(d,True)
        independent=oracle(a,np.zeros_like(d[0]),d)[1]
        np.testing.assert_allclose(actual.K,independent,atol=4e-12,rtol=2e-12)
        finite=reference(c,d,q,offsets,cube(0))
        full=reference(c,d,full_q,full_offsets,cube(1))
        assert np.all(abs(independent-finite)<=result.finite_difference_bounds+5e-12)
        assert np.all(abs(finite-full)<=result.reference_remainder_bounds+5e-12)
        assert np.all(abs(independent-full)<=result.error_bounds+5e-12)
        energy=.5*abs(np.einsum('k,skij,skji->',c['weights'],d,independent-full).real)
        assert energy<=result.energy_error_bound+1e-11
        # A method's executed response adds its separate arithmetic radius.
        assert np.all(abs(actual.K-full)<=result.error_bounds+actual.K_roundoff_bounds+5e-12)
    if cutoff==1e9:assert result.finite_difference_maximum>0.
    plain=ri.build(np.zeros_like(d[0]),d)
    np.testing.assert_array_equal(plain.K,actual.K)
    assert not actual.receipt.has_j and actual.J.size==0
    assert plain.receipt.source_identity!=actual.receipt.source_identity
    assert [x.projector_identity for x in plain.components]==[x.projector_identity for x in actual.components]


@pytest.mark.parametrize('field,quantity',[('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),('primitive_quartets','primitive_products'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_exchange_fitting_joint_caps_precede_payload(field,quantity):
    a,ri,pair,q,offsets,o,_=fitting_case(1)
    p=manta.plan_periodic_ri_exchange_fitting(ri,pair,len(q),o,a['budget'])
    b=manta.ResourceBudget()
    for f,v in [('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),('primitive_quartets','primitive_products'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')]:setattr(b,f,getattr(p,v))
    assert manta.certify_periodic_ri_exchange_fitting(ri,pair,q,offsets,o,b).qualified
    setattr(b,field,getattr(p,quantity)-1)
    with pytest.raises((ValueError,RuntimeError),match='budget'):
        manta.certify_periodic_ri_exchange_fitting(ri,pair,np.full_like(q,np.nan),offsets,o,b)


@pytest.mark.parametrize('fault',['l1','shape','asymmetric','negative','nan','probes','target'])
def test_exchange_fitting_family_options_refuse(fault):
    a,ri,pair,q,offsets,o,_=fitting_case(1)
    if fault=='l1':o.density_l1_limit=0.
    elif fault=='shape':o.density_entry_limits=[1.]
    elif fault=='asymmetric':o.density_entry_limits=[1.,2.,1.,1.]
    elif fault=='negative':o.density_entry_limits=[-1.]*4
    elif fault=='nan':o.density_l1_limit=float('nan')
    elif fault=='probes':o.maximum_probes=1
    else:o.matrix_error=1.
    with pytest.raises(ValueError):manta.certify_periodic_ri_exchange_fitting(ri,pair,q,offsets,o,a['budget'])


def test_exchange_fitting_targets_family_and_owned_results():
    from tests.test_manta_periodic_ri_q import densities
    a,ri,pair,q,offsets,o,_=fitting_case(1)
    r=manta.certify_periodic_ri_exchange_fitting(ri,pair,q,offsets,o,a['budget'])
    expected=r.error_bounds.copy();r.error_bounds[:]=0.;np.testing.assert_array_equal(r.error_bounds,expected)
    d=densities(1)*.01;r.validate_density(d)
    with pytest.raises(ValueError,match='family'):r.validate_density(d*1e9)
    d[0,0,0,1]+=1j
    with pytest.raises(ValueError,match='Hermitian'):r.validate_density(d)
    o.matrix_error=np.nextafter(r.maximum_error,np.inf);o.energy_error=np.nextafter(r.energy_error_bound,np.inf)
    assert manta.certify_periodic_ri_exchange_fitting(ri,pair,q,offsets,o,a['budget']).qualified
    o.matrix_error=o.energy_error=1e-100
    with pytest.raises(RuntimeError,match='targets unmet'):manta.certify_periodic_ri_exchange_fitting(ri,pair,q,offsets,o,a['budget'])


def test_exchange_fitting_rejects_other_physical_frames():
    from tests.test_manta_periodic_pairs import options
    from tests.test_manta_periodic_ri_range_fock import case
    a,ri,pair,q,offsets,o,_=fitting_case()
    other=manta.PeriodicPairSource(a['orbital'],a['system'].lattice,cube(0),a['momenta'],np.array([.5,.5]),options(0,None,.2),a['budget'])
    with pytest.raises(ValueError,match='weight mismatch'):manta.certify_periodic_ri_exchange_fitting(ri,other,q,offsets,o,a['budget'])
    only_j=manta.PeriodicRIFockBuilder(**dict(a,make_k=False,pair_vectors=np.empty((0,3)),transfer_offsets=np.zeros(5,dtype=np.int64)))
    with pytest.raises(ValueError,match='K owner'):only_j.build_exchange(np.zeros((2,2,2,2),complex))



def test_exchange_fitting_cites_the_component_and_arithmetic_sources():
    from vibeqc.output.citations import load_default_database
    keys={c.key for c in load_default_database().assemble(numerics=['manta_periodic_ri_exchange_fitting']).citations}
    assert {'sun_berkelbach_gdf_2017','sundararaman_arias_exx_2013','rump_verification_2010'}<=keys


@pytest.mark.parametrize('nk',[1,2])
@pytest.mark.parametrize('cutoff',[1e-12,.5,1e9])
@pytest.mark.parametrize('screened',[False,True])
def test_exchange_fitting_capacity_dominates_actual_preparation(nk,cutoff,screened):
    from tests.test_manta_periodic_pairs import options
    from tests.test_manta_periodic_ri_fock import plan
    a,ri,_,q,offsets,o,_=fitting_case(nk,cutoff)
    po=options(0,0. if screened else None,.2)
    images=cube(1)
    capacity=manta.plan_periodic_ri_exchange_fitting_capacity(a['orbital'],plan(a),len(images),len(q),po,o,budget=a['budget'])
    pairs=manta.PeriodicPairSource(a['orbital'],a['system'].lattice,images,a['momenta'],a['weights'],po,a['budget'])
    actual=manta.plan_periodic_ri_exchange_fitting(ri,pairs,len(q),o,a['budget'])
    preparation=pairs.plan
    for field,setup in [('shell_calls','shell_calls'),('primitive_products','primitive_pairs'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms'),('modeled_peak_bytes','modeled_peak_bytes')]:
        assert getattr(capacity,field)>=getattr(actual,field)+getattr(preparation,setup)
    assert capacity.maximum_probes==nk*a['orbital'].nbasis**2
    b=manta.ResourceBudget()
    for f,v in [('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),('primitive_quartets','primitive_products'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')]:setattr(b,f,getattr(capacity,v))
    result=manta.certify_periodic_ri_exchange_fitting(ri,pairs,q,offsets,o,b)
    assert result.qualified and result.probes==capacity.maximum_probes


@pytest.mark.parametrize('field,quantity',[('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),('primitive_quartets','primitive_products'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_exchange_fitting_capacity_exact_and_one_short_before_sources(field,quantity):
    from tests.test_manta_periodic_pairs import options
    from tests.test_manta_periodic_ri_fock import case,plan
    a=case(2);po=options(0,None,.2);o=manta.PeriodicRIExchangeFittingOptions();o.density_l1_limit=1.
    rp=plan(a)
    def capacity(b):return manta.plan_periodic_ri_exchange_fitting_capacity(a['orbital'],rp,27,500,po,o,12345,b)
    p=capacity(a['budget']);b=manta.ResourceBudget()
    for f,v in [('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),('primitive_quartets','primitive_products'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')]:setattr(b,f,getattr(p,v))
    assert getattr(capacity(b),quantity)==getattr(p,quantity)
    setattr(b,field,getattr(p,quantity)-1)
    with pytest.raises((ValueError,RuntimeError),match='budget'):capacity(b)


def test_exchange_fitting_capacity_retained_j_lifetime_and_invalid_counts():
    from tests.test_manta_periodic_pairs import options
    from tests.test_manta_periodic_ri_fock import case,plan
    a=case(1);rp=plan(a);po=options(0,None,.2);o=manta.PeriodicRIExchangeFittingOptions();o.density_l1_limit=1.
    def capacity(extra=0,basis=None,cells=1,vecs=124):
        return manta.plan_periodic_ri_exchange_fitting_capacity(basis or a['orbital'],rp,cells,vecs,po,o,extra,a['budget'])
    p=capacity();assert capacity(12345).modeled_peak_bytes==p.modeled_peak_bytes+12345
    from vibeqc import _vibeqc_core as core
    wrong=vq.BasisSet(a['system'].unit_cell_molecule(),[core.ShellInfo(0,0,True,[.8],[1.],a['orbital'].shells()[0].origin)],'single-function',False)
    with pytest.raises(ValueError,match='matching basis'):capacity(basis=wrong)
    with pytest.raises(ValueError,match='image count'):capacity(cells=0)
    with pytest.raises((ValueError,RuntimeError,OverflowError)):capacity(vecs=2**64-1)
    o.maximum_probes=1
    with pytest.raises(ValueError,match='probe'):capacity()


def exchange_admission_job(directory,method='rhf',joint=True):
    from dataclasses import replace
    from tests.test_manta_inverse_action import fitted_exchange_job
    job,_=fitted_exchange_job(directory,method)
    accuracy=manta.PeriodicRIExchangeAccuracyOptions();accuracy.matrix_error=1000.;accuracy.energy_error=1e6
    job['periodic_fitting']=replace(job['periodic_fitting'],full_coulomb_exchange_accuracy=accuracy,
        full_coulomb_accuracy=job['periodic_fitting'].full_coulomb_accuracy if joint else None)
    return job,accuracy


@pytest.mark.parametrize('method',['rhf','uhf'])
@pytest.mark.parametrize('joint',[False,True])
@pytest.mark.parametrize('capped',[False,True])
def test_exchange_admission_preserves_complete_and_stopped_hf(tmp_path,method,joint,capped):
    from dataclasses import replace
    from vibeqc.output import validate_qvf
    job,accuracy=exchange_admission_job(tmp_path/'selected',method,joint)
    if capped:job['max_iter']=1
    selected=vq.run_periodic_job(**job)
    control=vq.run_periodic_job(**dict(job,output=tmp_path/'control'/'job',
        periodic_fitting=replace(job['periodic_fitting'],full_coulomb_exchange_accuracy=None)))
    assert selected.converged==control.converged==(not capped)
    assert selected.energy==control.energy and selected.iterations==control.iterations
    for field in ('density','fock','coefficients','orbital_energies'):
        np.testing.assert_array_equal(getattr(selected.native,field),getattr(control.native,field))
    r=selected.native.receipt;c=r.exchange_fitting_accuracy
    assert c.qualified and c.targets_requested and c.probe_roundoff_included and not c.source_construction_included
    assert c.ri_source_identity==r.fitting_info.source_identity==control.native.receipt.fitting_info.source_identity
    assert c.maximum_error<=accuracy.matrix_error and c.energy_error_bound<=accuracy.energy_error
    assert r.fitted_fock.fitted_exchange_roundoff_evaluated
    d=selected.native.density;spin=np.stack([.5*d,.5*d]) if method=='rhf' else d;c.validate_density(np.ascontiguousarray(spin))
    nk=len(job['kpoints']);assert c.density_l1_limit>=nk*r.accuracy_density.density_l1_limit
    assert np.all(np.array(c.density_entry_limits).reshape(d.shape[-3:])>=nk*np.array(r.accuracy_density.weighted_entry_bounds))
    p=selected.integral_provenance['exchange_fitting_accuracy'];assert p['certificate_identity']==c.certificate_identity
    from fractions import Fraction
    ledger = selected.integral_provenance['component_ledger']
    assert ledger['qualified']
    family = c.canonical_density_envelope_identity
    assert len(family) == 64 and p['canonical_density_envelope_identity'] == family
    if joint:
        assert family == selected.integral_provenance['fitting']['full_coulomb_accuracy']['density_envelope_identity']
    term = next(t for t in ledger['components'] if t['certificate'] == 'exchange_fitting_accuracy')
    assert term['energy_error_bound'] == c.energy_error_bound and term['source_identity'] == c.ri_source_identity
    exact = sum((Fraction.from_float(t['energy_error_bound']) for t in ledger['components'] if t['requested']), Fraction())
    assert Fraction.from_float(ledger['fixed_density_energy_error_bound']) >= exact
    assert {'exchange-source-correction', 'exchange-response-and-assembly-arithmetic', 'restricted-density-embedding',
            'integral-roundoff', 'rank-stability', 'density-relaxation'} <= set(ledger['excludes'])
    assert not any(t.get('certificate') == 'exchange_fitting_accuracy' for t in control.integral_provenance['component_ledger']['components'])
    assert not ledger['relaxation_delimiter']['is_certificate']
    assert p['response_roundoff_evaluated'] and 'BvK/source correction' in p['exclusions']
    assert 'manta_periodic_ri_exchange_fitting' in selected.integral_provenance['citation_numerics']
    assert validate_qvf(str(job['output'].with_suffix('.qvf')))['valid']


@pytest.mark.parametrize('field,quantity',[('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),('primitive_quartets','primitive_pairs'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_exchange_admission_joint_caps_before_numerical_setup(tmp_path,field,quantity):
    from tests.test_manta_periodic_ri_fitting_accuracy import accuracy_job
    _,args,_,_=accuracy_job(tmp_path)
    o=manta.PeriodicRIExchangeAccuracyOptions();o.matrix_error=1000.;o.energy_error=1e6
    args['options'].exchange_algorithm='ri';args['options'].exchange_fitting_accuracy=o
    args['options'].fitting.coupling_multipole_error=0.;args['options'].fitting.coupling_multipole_pair_tree=False
    args['options'].ri_accuracy=manta.PeriodicRIAccuracyOptions()
    p=manta.plan_periodic_ri_rhf_from_cutoffs(**args)
    assert p.rhf.exchange_fitting_accuracy.maximum_probes==len(args['mesh'])*args['basis'].nbasis**2
    setattr(args['budget'],field,getattr(p,quantity)-1)
    with pytest.raises((ValueError,RuntimeError,OverflowError),match='budget'):manta.plan_periodic_ri_rhf_from_cutoffs(**args)
    setattr(args['budget'],field,getattr(p,quantity));manta.plan_periodic_ri_rhf_from_cutoffs(**args)


def test_exchange_admission_unmet_targets_refuse_before_scf(tmp_path):
    job,o=exchange_admission_job(tmp_path,joint=False);o.matrix_error=o.energy_error=1e-30
    assert vq.run_periodic_job(**dict(job,dry_run=True)) is None
    with pytest.raises(RuntimeError,match='exchange fitting targets unmet'):vq.run_periodic_job(**job)
    assert not job['output'].with_suffix('.qvf').exists()


@pytest.mark.parametrize('fault',['exact','probes','completion','nan','one_target'])
def test_exchange_admission_invalid_policies_refuse(tmp_path,fault):
    from dataclasses import replace
    job,o=exchange_admission_job(tmp_path)
    if fault=='exact':job['exchange']=manta.ExchangeBuildOptions(algorithm='exact')
    elif fault=='probes':o.maximum_probes=1
    elif fault=='completion':o.ao_completion_cube_radius=0
    elif fault=='nan':o.energy_error=float('nan')
    else:o.energy_error=0.
    with pytest.raises((ValueError,RuntimeError),match='target|completion|probe|K plan'):
        job['periodic_fitting']=replace(job['periodic_fitting'],full_coulomb_exchange_accuracy=o)
        vq.run_periodic_job(**dict(job,dry_run=True))


@pytest.mark.parametrize('profile',['hse06-rks','cam-b3lyp-uks'])
def test_exchange_admission_unsupported_ks_is_not_silently_ignored(tmp_path,profile):
    job,_=exchange_admission_job(tmp_path,profile)
    with pytest.raises(NotImplementedError,match='global-hybrid KS'):vq.run_periodic_job(**dict(job,dry_run=True))


@pytest.mark.parametrize('method',['rhf','uhf'])
def test_exchange_admission_partial_metric_retains_the_actual_owner(tmp_path,method):
    job,o=exchange_admission_job(tmp_path,method,joint=False)
    job['coulomb'].metric.absolute_cutoff=.01;job['max_iter']=1
    result=vq.run_periodic_job(**job)
    c=result.native.receipt.exchange_fitting_accuracy;info=result.native.receipt.fitting_info
    assert 0<info.metrics[0].rank<job['coulomb'].auxiliary_basis.nbasis
    assert c.ri_source_identity==info.source_identity and c.qualified


@pytest.mark.parametrize('shape,shift',[((1,1,1),(0,0,0)),((2,1,1),(1,0,0))])
def test_exchange_admission_gamma_and_shifted_reference_frames(tmp_path,shape,shift):
    from dataclasses import replace
    from tests.test_manta_periodic_ri_rhf import job_case
    job,_,_=job_case(tmp_path,shape,shift)
    o=manta.PeriodicRIExchangeAccuracyOptions();o.matrix_error=o.energy_error=1e12
    job['periodic_fitting']=replace(job['periodic_fitting'],full_coulomb_exchange_accuracy=o);job['max_iter']=1
    result=vq.run_periodic_job(**job);c=result.native.receipt.exchange_fitting_accuracy
    assert c.plan.momenta==len(job['kpoints']) and c.qualified
    assert c.probes==len(job['kpoints'])*job['basis'].nbasis**2


def test_exchange_admission_runs_at_joint_native_caps(tmp_path):
    from tests.test_manta_periodic_ri_fitting_accuracy import accuracy_job
    _,args,_,_=accuracy_job(tmp_path)
    o=args['options'];o.fitting.coupling_multipole_error=0.;o.fitting.coupling_multipole_pair_tree=False
    o.ri_accuracy=manta.PeriodicRIAccuracyOptions();o.exchange_algorithm='ri';o.scf.maximum_iterations=1
    x=manta.PeriodicRIExchangeAccuracyOptions();x.matrix_error=1000.;x.energy_error=1e6;o.exchange_fitting_accuracy=x
    p=manta.plan_periodic_ri_rhf_from_cutoffs(**args)
    for f,v in [('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),('primitive_quartets','primitive_pairs'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')]:setattr(args['budget'],f,getattr(p,v))
    r=core._manta.run_periodic_ri_rhf_from_cutoffs(**args)
    assert r.receipt.exchange_fitting_accuracy.qualified and r.receipt.shell_calls<=p.shell_calls


@pytest.mark.parametrize('profile',['pbe0-rks','pbe0-uks'])
@pytest.mark.parametrize('joint',[False,True])
@pytest.mark.parametrize('capped',[False,True])
def test_exchange_ks_admission_preserves_states_and_scales(tmp_path,profile,joint,capped):
    from dataclasses import replace
    from vibeqc.output import validate_qvf
    job,o=exchange_admission_job(tmp_path/'selected',profile,joint)
    if capped:job['max_iter']=1
    r=vq.run_periodic_job(**job)
    control=vq.run_periodic_job(**dict(job,output=tmp_path/'control'/'job',periodic_fitting=replace(job['periodic_fitting'],full_coulomb_exchange_accuracy=None)))
    assert r.converged==control.converged==(not capped)
    assert r.energy==control.energy and r.iterations==control.iterations
    for field in ('density','fock','coefficients','orbital_energies'):
        np.testing.assert_array_equal(getattr(r.native,field),getattr(control.native,field))
    c=r.native.receipt.exchange_fitting_accuracy
    assert c.qualified and c.ri_source_identity==r.native.receipt.fitting_info.source_identity
    density=r.native.density;spin=np.repeat(.5*density,2,axis=0) if profile.endswith('-rks') else density
    c.validate_density(np.ascontiguousarray(spin))
    p=r.integral_provenance['exchange_fitting_accuracy'];scale=r.native.receipt.exact_exchange_fraction
    assert scale==.25 and p['exchange_fraction']==scale
    assert p['scaled_component_matrix_bound']>=scale*c.maximum_error
    assert p['scaled_component_energy_bound']>=scale*c.energy_error_bound
    assert p['response_roundoff_evaluated'] and c.maximum_error<=o.matrix_error and c.energy_error_bound<=o.energy_error
    ledger = r.integral_provenance['component_ledger']
    assert ledger['qualified'] and 'xc' in ledger['excludes']
    term = next(t for t in ledger['components'] if t['certificate'] == 'exchange_fitting_accuracy')
    assert term['energy_error_bound'] == p['scaled_component_energy_bound']
    assert term['method_coefficient'] == scale
    if joint:
        assert term['density_envelope_identity'] == r.integral_provenance['fitting']['full_coulomb_accuracy']['density_envelope_identity']
    assert not ledger['relaxation_delimiter']['is_certificate']
    assert validate_qvf(str(job['output'].with_suffix('.qvf')))['valid']


@pytest.mark.parametrize('field,quantity',[('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),('primitive_quartets','primitive_work'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_exchange_ks_admission_resource_caps(field,quantity):
    from tests.test_manta_periodic_ri_ks import case,plan
    a,_,_=case('JK',spins=2,order=2);o=manta.PeriodicRIExchangeAccuracyOptions();o.matrix_error=o.energy_error=1e12
    a['options'].exchange_fitting_accuracy=o;p=plan(a)
    assert p.exchange_fitting_accuracy is not None
    setattr(a['budget'],field,getattr(p,quantity)-1)
    with pytest.raises((ValueError,RuntimeError,OverflowError),match='budget'):plan(a)
    setattr(a['budget'],field,getattr(p,quantity));plan(a)


def test_exchange_ks_admission_targets_refuse_before_scf(tmp_path):
    job,o=exchange_admission_job(tmp_path,'pbe0-uks',False);o.matrix_error=o.energy_error=1e-30
    assert vq.run_periodic_job(**dict(job,dry_run=True)) is None
    with pytest.raises(RuntimeError,match='exchange fitting targets unmet'):vq.run_periodic_job(**job)
    assert not job['output'].with_suffix('.qvf').exists()


@pytest.mark.parametrize('profile',['pure','J'])
def test_exchange_ks_admission_native_unsupported_paths_refuse(profile):
    from tests.test_manta_periodic_ri_ks import case,plan
    a,_,_=case(profile,order=2);o=manta.PeriodicRIExchangeAccuracyOptions();o.matrix_error=o.energy_error=1e12
    a['options'].exchange_fitting_accuracy=o
    with pytest.raises(ValueError,match='global-hybrid RI K'):plan(a)


@pytest.mark.parametrize('spins',[1,2])
@pytest.mark.parametrize('profile',['K','JK'])
def test_exchange_ks_admission_runs_at_joint_native_caps(spins,profile):
    from tests.test_manta_periodic_ri_ks import case,plan,run,CAPS
    a,_,_=case(profile,spins=spins,order=2)
    o=manta.PeriodicRIExchangeAccuracyOptions();o.matrix_error=o.energy_error=1e12
    a['options'].exchange_fitting_accuracy=o;a['options'].ks.scf.maximum_iterations=1
    p=plan(a)
    for field,quantity in CAPS:setattr(a['budget'],field,getattr(p,quantity))
    r=run(a)
    assert r.receipt.exchange_fitting_accuracy.qualified
    assert r.receipt.shell_calls<=p.shell_calls and r.receipt.primitive_pairs<=p.primitive_work


@pytest.mark.parametrize('profile',['pbe0-rks','pbe0-uks'])
def test_exchange_ks_admission_signed_component(tmp_path,profile):
    from benchmarks.manta_metric_spectrum import exchange_ks_admission_case
    result=exchange_ks_admission_case(tmp_path,profile,joint=False,capped=True,fraction=-.2)
    assert result['certificate']['exchange_fraction']==-.2


@pytest.mark.parametrize('kernel,omega',[('coulomb',0.),('erf',.43),('erfc',.43)])
@pytest.mark.parametrize('nk',[1,2])
@pytest.mark.parametrize('cutoff',[1e-12,.5,1e9])
def test_range_exchange_fitting_component_bounds(kernel,omega,nk,cutoff):
    from tests.test_manta_periodic_ri_q import densities
    from tests.test_manta_periodic_ri_range_fock import independent
    a,ri,pair,q,offsets,o,c=fitting_case(nk,cutoff=cutoff,kernel=(kernel,omega))
    zero=kernel=='erfc' and omega>0
    q,offsets=transfers(c['cell'],c['k'],2,screened_zero=zero)
    bound=manta.certify_periodic_ri_exchange_fitting(ri,pair,q,offsets,o,a['budget'])
    assert bound.kernel==kernel and bound.physical_omega==omega and bound.qualified
    assert bound.ri_source_identity==ri.info.source_identity
    d=densities(nk,seed=412)*.04;bound.validate_density(d)
    fitted=ri.build_exchange(d,True)
    from tests.test_manta_periodic_ri_fock import oracle
    independent_fit=(oracle(a,np.zeros_like(d[0]),d,1.) if kernel=='coulomb' else independent(a,np.zeros_like(d[0]),d,1.))[1]
    np.testing.assert_allclose(fitted.K,independent_fit,atol=5e-12,rtol=2e-12)
    full_q,full_offsets=transfers(c['cell'],c['k'],5,screened_zero=zero)
    finite=reference(c,d,q,offsets,cube(0),kernel=kernel,omega=omega)
    full=reference(c,d,full_q,full_offsets,cube(1),kernel=kernel,omega=omega)
    assert np.all(abs(independent_fit-finite)<=bound.finite_difference_bounds+8e-12)
    assert np.all(abs(finite-full)<=bound.reference_remainder_bounds+8e-12)
    assert np.all(abs(fitted.K-full)<=bound.error_bounds+fitted.K_roundoff_bounds+8e-12)
    energy=.5*abs(np.einsum('k,skij,skji->',c['weights'],d,fitted.K-full).real)
    assert energy<=bound.energy_error_bound+8e-12


@pytest.mark.parametrize('omega',[.04,.43,4.3])
def test_range_exchange_zero_is_kept_or_bounded_when_omitted(omega):
    a,ri,pair,q,offsets,o,c=fitting_case(1,kernel=('erfc',omega))
    qzero,ozero=transfers(c['cell'],c['k'],2,screened_zero=True)
    omitted=manta.PeriodicExchangeBoundSource(pair,q,offsets,2,kernel='erfc',omega=omega,ao_completion_cube_radius=1)
    included=manta.PeriodicExchangeBoundSource(pair,qzero,ozero,2,kernel='erfc',omega=omega,ao_completion_cube_radius=1)
    base=manta.PeriodicExchangeBoundSource(pair,q,offsets,2,ao_completion_cube_radius=1)
    assert omitted.receipt.source_identity!=included.receipt.source_identity!=base.receipt.source_identity
    assert omitted.receipt.kernel=='erfc' and included.receipt.physical_omega==omega
    # Move exactly the finite-zero majorant between retained and omitted sums.
    delta_keep=included.reciprocal_coefficients[:,0]-base.reciprocal_coefficients[:,0]
    delta_omit=omitted.reciprocal_coefficients[:,1]-base.reciprocal_coefficients[:,1]
    expected=np.pi/omega**2/abs(np.linalg.det(c['cell']))
    assert np.all(delta_keep>=expected) and np.all(delta_omit>=expected)
    np.testing.assert_allclose(delta_keep,delta_omit,rtol=2e-12,atol=2e-12)
    from tests.test_manta_periodic_ri_q import densities
    d=densities(1)*.04
    zero_error=reference(c,d,qzero,ozero,cube(0),kernel='erfc',omega=omega)-reference(c,d,q,offsets,cube(0),kernel='erfc',omega=omega)
    assert np.all(abs(zero_error)<=omitted.build(d).reciprocal_error_bound+5e-12)


@pytest.mark.parametrize('kernel,omega',[('bad',.3),('coulomb',.3),('erfc',-1.),('erf',float('nan')),('erfc',float('inf'))])
def test_range_exchange_bound_invalid_kernel_refuses(kernel,omega):
    a,ri,pair,q,offsets,o,c=fitting_case(1)
    with pytest.raises(ValueError,match='range'):manta.PeriodicExchangeBoundSource(pair,q,offsets,2,kernel=kernel,omega=omega)


@pytest.mark.parametrize('kernel,omega',[('coulomb',0.),('erfc',0.),('erf',.43)])
def test_range_exchange_singular_zero_still_refuses(kernel,omega):
    a,ri,pair,q,offsets,o,c=fitting_case(1)
    q,offsets=transfers(c['cell'],c['k'],2,screened_zero=True)
    with pytest.raises(ValueError,match='nonzero Q'):manta.PeriodicExchangeBoundSource(pair,q,offsets,2,kernel=kernel,omega=omega)


def test_range_exchange_zero_range_limits_and_default_identity():
    a,ri,pair,q,offsets,o,c=fitting_case(1)
    base=manta.PeriodicExchangeBoundSource(pair,q,offsets,2)
    screened=manta.PeriodicExchangeBoundSource(pair,q,offsets,2,kernel='erfc',omega=0.)
    zero=manta.PeriodicExchangeBoundSource(pair,q,offsets,2,kernel='erf',omega=0.)
    explicit=manta.PeriodicExchangeBoundSource(pair,q,offsets,2,kernel='coulomb',omega=0.)
    np.testing.assert_array_equal(screened.reciprocal_coefficients,base.reciprocal_coefficients)
    assert np.all(zero.reciprocal_coefficients==0.)
    assert explicit.receipt.source_identity==base.receipt.source_identity
    from tests.test_manta_periodic_ri_q import densities
    assert np.all(zero.build(densities(1)).error_bound==0.)


@pytest.mark.parametrize('kernel',['erf','erfc'])
@pytest.mark.parametrize('field,quantity',[('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),('primitive_quartets','primitive_products'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_range_exchange_certificate_joint_caps(kernel,field,quantity):
    a,ri,pair,q,offsets,o,c=fitting_case(1,kernel=(kernel,.43))
    q,offsets=transfers(c['cell'],c['k'],2,screened_zero=kernel=='erfc')
    plan=manta.plan_periodic_ri_exchange_fitting(ri,pair,len(q),o,a['budget'])
    b=manta.ResourceBudget()
    for key,value in [('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),('primitive_quartets','primitive_products'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')]:setattr(b,key,getattr(plan,value))
    assert manta.certify_periodic_ri_exchange_fitting(ri,pair,q,offsets,o,b).qualified
    setattr(b,field,getattr(plan,quantity)-1)
    with pytest.raises((ValueError,RuntimeError),match='budget'):manta.certify_periodic_ri_exchange_fitting(ri,pair,np.full_like(q,np.nan),offsets,o,b)



def range_admission_job(directory,profile='hse06-rks',joint=True):
    from dataclasses import replace
    from tests.test_manta_inverse_action import fitted_exchange_job
    job,_=fitted_exchange_job(directory,profile)
    options=manta.PeriodicRIExchangeAccuracyOptions();options.matrix_error=1000.;options.energy_error=1e6
    job['periodic_fitting']=replace(job['periodic_fitting'],range_exchange_accuracy=options,
        full_coulomb_accuracy=job['periodic_fitting'].full_coulomb_accuracy if joint else None)
    return job,options


@pytest.mark.parametrize('profile',['hse06-rks','hse06-uks','cam-b3lyp-rks','cam-b3lyp-uks'])
@pytest.mark.parametrize('joint',[False,True])
@pytest.mark.parametrize('capped',[False,True])
def test_range_admission_states_and_signed_composition(tmp_path,profile,joint,capped):
    from dataclasses import replace
    from vibeqc.output import validate_qvf
    job,options=range_admission_job(tmp_path/'selected',profile,joint)
    if capped:job['max_iter']=1
    r=vq.run_periodic_job(**job)
    c=vq.run_periodic_job(**dict(job,output=tmp_path/'control'/'job',periodic_fitting=replace(job['periodic_fitting'],range_exchange_accuracy=None)))
    assert r.energy==c.energy and r.iterations==c.iterations and r.converged==c.converged==(not capped)
    for field in ('density','fock','coefficients','orbital_energies'):np.testing.assert_array_equal(getattr(r.native,field),getattr(c.native,field))
    b=r.native.receipt.range_exchange_accuracy;assert b.qualified
    assert b.maximum_error<=options.matrix_error and b.energy_error_bound<=options.energy_error
    spin=np.ascontiguousarray(np.repeat(.5*r.native.density,2,axis=0) if profile.endswith('-rks') else r.native.density)
    assert len(b.components)==len(r.native.receipt.exchange_components)
    errors=np.zeros_like(b.error_bounds);energy=0.
    for certificate,part,coefficient in zip(b.components,r.native.receipt.exchange_components,b.coefficients):
        certificate.validate_density(spin)
        assert certificate.ri_source_identity==part.fitting_info.source_identity
        assert certificate.kernel==part.kernel and certificate.physical_omega==part.omega
        assert coefficient==part.coefficient and not certificate.targets_requested and certificate.qualified
        assert part.fitted_fock.fitted_exchange_roundoff_evaluated
        errors+=abs(coefficient)*certificate.error_bounds;energy+=abs(coefficient)*certificate.energy_error_bound
    assert np.all(b.error_bounds>=errors) and b.energy_error_bound>=energy
    if profile.startswith('cam'):assert any(a<0 for a in b.coefficients)
    assert r.integral_provenance['range_exchange_accuracy']['qualified']
    from fractions import Fraction
    ledger = r.integral_provenance['component_ledger']
    assert ledger['qualified'] and 'xc' in ledger['excludes']
    term = next(t for t in ledger['components'] if t['certificate'] == 'range_exchange_accuracy')
    assert term['energy_error_bound'] == b.energy_error_bound
    assert len(term['component_certificate_identities']) == len(b.components)
    requested = [t for t in ledger['components'] if t['requested']]
    assert len([t for t in requested if t['component'] == 'exact-exchange']) == 1
    assert Fraction.from_float(ledger['fixed_density_energy_error_bound']) >= sum(Fraction.from_float(t['energy_error_bound']) for t in requested)
    families = [c.canonical_density_envelope_identity for c in b.components]
    assert len(set(families)) == 1 and families[0] == term['density_envelope_identity']
    if joint:
        assert families[0] == r.integral_provenance['fitting']['full_coulomb_accuracy']['density_envelope_identity']
    assert {'exchange-source-correction', 'exchange-response-and-assembly-arithmetic', 'restricted-density-embedding',
            'integral-roundoff', 'rank-stability', 'density-relaxation'} <= set(ledger['excludes'])
    assert not ledger['relaxation_delimiter']['is_certificate']
    assert all(x['targets'] is None for x in r.integral_provenance['range_exchange_accuracy']['components'])
    assert validate_qvf(str(job['output'].with_suffix('.qvf')))['valid']


@pytest.mark.parametrize('field,quantity',[('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),('primitive_quartets','primitive_work'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_range_admission_resource_caps(field,quantity):
    from tests.test_manta_periodic_ri_range_ks import case,plan
    a,_,_=case(order=2);o=manta.PeriodicRIExchangeAccuracyOptions();o.matrix_error=o.energy_error=1e12
    a['options'].range_exchange_accuracy=o;p=plan(a)
    assert p.range_exchange_accuracy_requested and all(c.exchange_fitting_accuracy is not None for c in p.exchange_components)
    setattr(a['budget'],field,getattr(p,quantity)-1)
    with pytest.raises((ValueError,RuntimeError,OverflowError),match='budget'):plan(a)
    setattr(a['budget'],field,getattr(p,quantity));plan(a)


@pytest.mark.parametrize('spins',[1,2])
@pytest.mark.parametrize('profile',['K','JK'])
def test_range_admission_runs_mixed_ranges_at_native_caps(spins,profile):
    from tests.test_manta_periodic_ri_range_ks import case,plan,run,CAPS,MIXED_RANGE
    a,_,_=case(profile,spins=spins,name=MIXED_RANGE,order=2)
    o=manta.PeriodicRIExchangeAccuracyOptions();o.matrix_error=o.energy_error=1e12
    a['options'].range_exchange_accuracy=o;a['options'].ks.scf.maximum_iterations=1;p=plan(a)
    for field,quantity in CAPS:setattr(a['budget'],field,getattr(p,quantity))
    r=run(a);b=r.receipt.range_exchange_accuracy
    assert b.qualified and len(b.components)==3
    from vibeqc.manta_periodic import periodic_ks_provenance
    evidence = periodic_ks_provenance(r, a['mesh'], a['options'], 0)
    ledger = evidence['component_ledger']
    assert ledger['qualified']
    term = next(t for t in ledger['components'] if t['certificate'] == 'range_exchange_accuracy')
    assert term['energy_error_bound'] == b.energy_error_bound
    assert len(term['component_certificate_identities']) == 3
    assert len(set(c.canonical_density_envelope_identity for c in b.components)) == 1
    assert r.receipt.shell_calls<=p.shell_calls and r.receipt.primitive_pairs<=p.primitive_work


def test_range_admission_strict_target_refuses_before_scf(tmp_path):
    job,o=range_admission_job(tmp_path,'cam-b3lyp-uks',False);o.matrix_error=o.energy_error=1e-30
    assert vq.run_periodic_job(**dict(job,dry_run=True)) is None
    with pytest.raises(RuntimeError,match='range exchange fitting targets unmet'):vq.run_periodic_job(**job)
    assert not job['output'].with_suffix('.qvf').exists()


@pytest.mark.parametrize('profile',['rhf','pbe0-rks'])
def test_range_admission_unsupported_method_refuses(tmp_path,profile):
    job,_=range_admission_job(tmp_path,profile)
    with pytest.raises(NotImplementedError,match='range-separated KS'):vq.run_periodic_job(**job)


def test_range_admission_exact_exchange_refuses(tmp_path):
    job,_=range_admission_job(tmp_path,'hse06-rks',False);job['exchange']=manta.ExchangeBuildOptions(algorithm='exact')
    with pytest.raises(ValueError,match='fitted physical range K'):vq.run_periodic_job(**job)


def test_range_admission_policy_validation(tmp_path):
    from dataclasses import replace
    job,o=range_admission_job(tmp_path)
    with pytest.raises(ValueError,match='mutually exclusive'):replace(job['periodic_fitting'],full_coulomb_exchange_accuracy=o)
    with pytest.raises(TypeError,match='range_exchange_accuracy'):replace(job['periodic_fitting'],range_exchange_accuracy=object())
    for value in (0.,-1.,float('nan'),float('inf')):
        o.matrix_error=value
        with pytest.raises(ValueError,match='positive finite'):replace(job['periodic_fitting'],range_exchange_accuracy=o)


@pytest.mark.parametrize('angular,pure,kernel,omega',[(0,True,'coulomb',0.),(0,True,'erfc',.31),(1,False,'erf',.43),(2,True,'coulomb',0.)])
def test_decay_groups_enclose_independent_complex_fourier_remainders(angular,pure,kernel,omega):
    c=fixture(angular=angular,pure=pure,skew=True,images=cube(0),exponent=.55,cell_size=5.)
    q,offsets=transfers(c['cell'],c['k'],2,screened_zero=kernel=='erfc')
    args=(c['source'],q,offsets,2)
    old=manta.PeriodicExchangeBoundSource(*args,ao_completion_cube_radius=1,kernel=kernel,omega=omega)
    grouped=manta.PeriodicExchangeBoundSource(*args,ao_completion_cube_radius=1,kernel=kernel,omega=omega,fourier_decay_bins=4)
    rng=np.random.default_rng(873);n=c['basis'].nbasis
    d=rng.normal(size=(2,len(c['k']),n,n))+1j*rng.normal(size=(2,len(c['k']),n,n));d=np.ascontiguousarray(.1*(d+d.swapaxes(-1,-2).conj()))
    legacy=old.build(d);bound=grouped.build(d)
    assert bound.plan.fourier_decay_bins==4 and grouped.receipt.source_identity!=old.receipt.source_identity
    assert np.all(bound.error_bound<=legacy.error_bound)
    finite=reference(c,d,q,offsets,cube(0),kernel=kernel,omega=omega)
    image=reference(c,d,q,offsets,cube(1),kernel=kernel,omega=omega)
    full_q,full_offsets=transfers(c['cell'],c['k'],4,screened_zero=kernel=='erfc')
    full=reference(c,d,full_q,full_offsets,cube(1),kernel=kernel,omega=omega)
    assert np.all(abs(image-finite)<=bound.pair_image_error_bound+8e-11)
    assert np.all(abs(full-image)<=bound.reciprocal_error_bound+8e-11)
    assert np.all(abs(full-finite)<=bound.error_bound+8e-11)
    np.testing.assert_array_equal(grouped.build(np.zeros_like(d)).error_bound,0.)


@pytest.mark.parametrize('field,quantity',[('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),('primitive_quartets','primitive_pairs'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_decay_groups_admit_all_caps_before_vector_payload(field,quantity):
    c,q,offsets,_,d=setup(reciprocal_radius=2)
    p=manta.PeriodicExchangeBoundSource.plan_source(c['source'],len(q),2,fourier_decay_bins=4)
    b=manta.ResourceBudget();setattr(b,field,getattr(p,quantity)-1)
    poison=np.full_like(q,np.nan)
    with pytest.raises((ValueError,RuntimeError,OverflowError),match='budget'):
        manta.PeriodicExchangeBoundSource(c['source'],poison,offsets,2,b,fourier_decay_bins=4)
    setattr(b,field,getattr(p,quantity))
    s=manta.PeriodicExchangeBoundSource(c['source'],q,offsets,2,b,fourier_decay_bins=4)
    assert s.plan.fourier_decay_bins==4
    qplan=s.plan_build(2);build_budget=manta.ResourceBudget()
    for cap,name in [('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),('primitive_quartets','primitive_pairs'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')]:setattr(build_budget,cap,getattr(qplan,name))
    s.build(d,build_budget)


@pytest.mark.parametrize('bins',[0,9])
def test_decay_groups_reject_invalid_partition_before_reference(bins):
    a,ri,pair,q,offsets,o,c=fitting_case(1);o.fourier_decay_bins=bins
    with pytest.raises(ValueError,match='bins'):manta.plan_periodic_ri_exchange_fitting(ri,pair,len(q),o,a['budget'])
    with pytest.raises(ValueError,match='bins'):manta.PeriodicExchangeBoundSource(pair,q,offsets,2,fourier_decay_bins=bins)


def test_decay_groups_empty_bins_and_single_group_legacy_identity():
    c,q,offsets,old,d=setup(reciprocal_radius=2)
    single=manta.PeriodicExchangeBoundSource(c['source'],q,offsets,2,fourier_decay_bins=1)
    assert single.receipt.source_identity==old.receipt.source_identity
    np.testing.assert_array_equal(single.build(d).error_bound,old.build(d).error_bound)
    eight=manta.PeriodicExchangeBoundSource(c['source'],q,offsets,2,fourier_decay_bins=8)
    assert np.isfinite(eight.build(d).error_bound).all()
    assert np.all(eight.build(d).error_bound<=single.build(d).error_bound)


@pytest.mark.parametrize('spins',[1,2])
@pytest.mark.parametrize('profile',['K','JK'])
def test_decay_groups_mixed_range_execution_at_exact_native_caps(spins,profile):
    from tests.test_manta_periodic_ri_range_ks import case,plan,run,CAPS,MIXED_RANGE
    a,_,_=case(profile,spins=spins,name=MIXED_RANGE,order=2)
    o=manta.PeriodicRIExchangeAccuracyOptions();o.matrix_error=o.energy_error=1e12;o.fourier_decay_bins=4
    a['options'].range_exchange_accuracy=o;a['options'].ks.scf.maximum_iterations=1;p=plan(a)
    for field,quantity in CAPS:setattr(a['budget'],field,getattr(p,quantity))
    r=run(a);assert r.receipt.range_exchange_accuracy.qualified
    assert r.receipt.shell_calls<=p.shell_calls and r.receipt.primitive_pairs<=p.primitive_work


def test_decay_groups_admit_a_fixed_target_that_single_group_refuses(tmp_path):
    job,policy=range_admission_job(tmp_path/'selected','hse06-rks',False)
    policy.matrix_error=50.;policy.energy_error=5000.;policy.fourier_decay_bins=4
    r=vq.run_periodic_job(**job);assert r.converged and r.native.receipt.range_exchange_accuracy.qualified
    assert r.integral_provenance['range_exchange_accuracy']['targets']['fourier_decay_bins']==4
    policy.fourier_decay_bins=1
    with pytest.raises(RuntimeError,match='range exchange fitting targets unmet'):
        vq.run_periodic_job(**dict(job,output=tmp_path/'refused'/'job'))


def fitted_exchange_ledger_evidence():
    from tests.test_manta_periodic_ri_fitting_accuracy import ledger_full_evidence
    e = ledger_full_evidence()
    e.pop('exchange_accuracy')
    e['fitting'].update(has_k=True, source_identity='whole-mesh-ri')
    e['exchange_fitting_accuracy'] = dict(targets_requested=True, qualified=True,
        kernel='coulomb', physical_omega=0., exchange_fraction=1.,
        qualification='full-coulomb-exchange-on-unweighted-hermitian-spin-density-family;zero-mode-omitted;stored-integrals',
        ri_source_identity='whole-mesh-ri', reference_source_identity='reference-k',
        certificate_identity='fitted-k', canonical_density_envelope_identity='env-a',
        maximum_error=.125, energy_error_bound=.25)
    return e


def test_fitted_exchange_ledger_composes_disjoint_energy_and_keeps_diagnostic_separate():
    from fractions import Fraction
    e = fitted_exchange_ledger_evidence()
    ledger = manta.compose_fixed_density_ledger(e, method='uhf')
    assert ledger['qualified']
    terms = [c for c in ledger['components'] if c['requested']]
    assert len([c for c in terms if c['component'] == 'exact-exchange']) == 1
    assert Fraction.from_float(ledger['fixed_density_energy_error_bound']) >= sum(Fraction.from_float(c['energy_error_bound']) for c in terms)
    assert 'exchange-source-correction' in ledger['excludes']
    assert 'exact-exchange-domains' not in ledger['excludes']
    assert 'hartree-omitted-domains' not in ledger['excludes']
    e['exchange_fitting_accuracy']['targets_requested'] = False
    diagnostic = manta.compose_fixed_density_ledger(e, method='uhf')
    assert diagnostic['qualified'] and diagnostic['fixed_density_energy_error_bound'] < ledger['fixed_density_energy_error_bound']
    assert 'exact-exchange' not in diagnostic['claimed_components']


@pytest.mark.parametrize('fault', ['source', 'missing-source', 'not-fitted-k', 'family', 'missing-family',
    'unqualified', 'energy', 'matrix', 'range', 'omega', 'reference', 'scaled', 'overlap', 'ks'])
def test_fitted_exchange_ledger_refuses_incompatible_claims(fault):
    import json
    from tests.test_manta_periodic_ri_rhf import _ledger_evidence
    e = fitted_exchange_ledger_evidence(); c = e['exchange_fitting_accuracy']
    if fault == 'source': c['ri_source_identity'] = 'other'
    elif fault == 'missing-source': e['fitting'].pop('source_identity')
    elif fault == 'not-fitted-k': e['fitting']['has_k'] = False
    elif fault == 'family': c['canonical_density_envelope_identity'] = 'other'
    elif fault == 'missing-family': c.pop('canonical_density_envelope_identity')
    elif fault == 'unqualified': c['qualified'] = False
    elif fault == 'energy': c['energy_error_bound'] = float('nan')
    elif fault == 'matrix': c['maximum_error'] = -1.
    elif fault == 'range': c['kernel'] = 'erf'
    elif fault == 'omega': c['physical_omega'] = .2
    elif fault == 'reference': c['qualification'] = 'finite-only'
    elif fault == 'scaled': c['exchange_fraction'] = .25
    elif fault == 'overlap': e['exchange_accuracy'] = _ledger_evidence()['exchange_accuracy']
    ledger = manta.compose_fixed_density_ledger(e, method='rks' if fault == 'ks' else 'uhf')
    assert not ledger['qualified'] and ledger['fixed_density_energy_error_bound'] is None
    assert ledger['reasons'] and not ledger['relaxation_delimiter']['is_certificate']
    json.dumps(ledger, allow_nan=False)


@pytest.mark.parametrize('coefficient', [.25, -.2, 1e-300])
def test_ks_ledger_scales_global_exchange_once(coefficient):
    from fractions import Fraction
    e = fitted_exchange_ledger_evidence()
    e.update(exact_exchange_fraction=coefficient, range_separated=False)
    c = e['exchange_fitting_accuracy']; c['exchange_fraction'] = coefficient
    # A duplicate display field cannot replace scaling of the raw certificate.
    c['scaled_component_energy_bound'] = 0.
    for method in ('rks', 'uks'):
        ledger = manta.compose_fixed_density_ledger(e, method=method)
        assert ledger['qualified'] and 'xc' in ledger['excludes']
        term = next(t for t in ledger['components'] if t['certificate'] == 'exchange_fitting_accuracy')
        exact = abs(Fraction.from_float(coefficient)) * Fraction.from_float(c['energy_error_bound'])
        assert Fraction.from_float(term['energy_error_bound']) >= exact
        assert term['energy_error_bound'] == np.nextafter(abs(coefficient)*c['energy_error_bound'], np.inf)
        assert not ledger['relaxation_delimiter']['is_certificate']


@pytest.mark.parametrize('fault', ['coefficient', 'missing', 'nonfinite', 'zero', 'overflow', 'range'])
def test_ks_ledger_refuses_invalid_global_scaling(fault):
    import json
    e = fitted_exchange_ledger_evidence(); e.update(exact_exchange_fraction=.25, range_separated=False)
    c = e['exchange_fitting_accuracy']; c['exchange_fraction'] = .25
    if fault == 'coefficient': c['exchange_fraction'] = -.25
    elif fault == 'missing': e.pop('exact_exchange_fraction')
    elif fault == 'nonfinite': c['exchange_fraction'] = e['exact_exchange_fraction'] = float('inf')
    elif fault == 'zero': c['exchange_fraction'] = e['exact_exchange_fraction'] = 0.
    elif fault == 'overflow':
        c['exchange_fraction'] = e['exact_exchange_fraction'] = 1e308
        c['energy_error_bound'] = 1e308
    else: e['range_separated'] = True
    ledger = manta.compose_fixed_density_ledger(e, method='rks')
    assert not ledger['qualified'] and ledger['fixed_density_energy_error_bound'] is None
    json.dumps(ledger, allow_nan=False)


def range_ledger_evidence():
    import copy
    e = fitted_exchange_ledger_evidence(); base = e.pop('exchange_fitting_accuracy')
    e['range_separated'] = True
    children, physical, required = [], [], []
    for kernel, omega, coefficient, source in [('coulomb', 0., .65, 'full-k'), ('erfc', .33, -.46, 'screened-k')]:
        c = copy.deepcopy(base); c.update(kernel=kernel, physical_omega=omega, exchange_fraction=coefficient,
            ri_source_identity=source, certificate_identity=source+'-certificate', targets_requested=False)
        if kernel != 'coulomb': c['qualification'] = 'physical-erfc-exchange-on-unweighted-hermitian-spin-density-family;screened-zero-accounted;stored-integrals;no-madelung'
        children.append(c)
        physical.append(dict(kernel=kernel, omega=omega, coefficient=coefficient,
            fitting=dict(has_k=True, source_identity=source, exchange_kernel=kernel, physical_omega=omega)))
        required.append(dict(kernel=kernel, omega=omega, coefficient=coefficient))
    e['exchange_components'] = physical; e['xc'] = dict(required_exchange_components=required)
    e['range_exchange_accuracy'] = dict(qualified=True,
        qualification='signed-physical-range-exchange-on-canonical-spin-family;component-sum;no-madelung-or-scf-relaxation',
        coefficients=[.65, -.46], components=children, maximum_error=.2, energy_error_bound=.3,
        certificate_identity='signed-sum')
    return e


def test_ks_ledger_counts_signed_range_aggregate_once():
    from fractions import Fraction
    for method in ('rks', 'uks'):
        e = range_ledger_evidence(); ledger = manta.compose_fixed_density_ledger(e, method=method)
        assert ledger['qualified']
        terms = [t for t in ledger['components'] if t['requested']]
        exchange = [t for t in terms if t['component'] == 'exact-exchange']
        assert len(exchange) == 1 and exchange[0]['energy_error_bound'] == .3
        assert exchange[0]['component_certificate_identities'] == ['full-k-certificate', 'screened-k-certificate']
        assert Fraction.from_float(ledger['fixed_density_energy_error_bound']) >= sum(Fraction.from_float(t['energy_error_bound']) for t in terms)
        assert {'xc', 'exchange-source-correction', 'exchange-response-and-assembly-arithmetic'} <= set(ledger['excludes'])


@pytest.mark.parametrize('fault', ['count', 'empty', 'required-count', 'coefficient', 'sign', 'kernel', 'omega',
    'source', 'missing-source', 'not-fitted', 'family', 'missing-family', 'child-unqualified', 'aggregate-unqualified',
    'targets', 'qualification', 'child-qualification', 'energy', 'matrix', 'underbound', 'scale-overflow',
    'child-nan', 'overlap', 'global-overlap', 'not-range', 'hf'])
def test_ks_ledger_refuses_incompatible_range_components(fault):
    import json
    from tests.test_manta_periodic_ri_rhf import _ledger_evidence
    e = range_ledger_evidence(); r = e['range_exchange_accuracy']; c = r['components'][1]
    if fault == 'count': r['coefficients'].pop()
    elif fault == 'empty': r['components'] = []
    elif fault == 'required-count': e['xc']['required_exchange_components'].pop()
    elif fault == 'coefficient': e['exchange_components'][1]['coefficient'] = .5
    elif fault == 'sign': r['coefficients'][1] = .46
    elif fault == 'kernel': c['kernel'] = 'erf'
    elif fault == 'omega': c['physical_omega'] = .2
    elif fault == 'source': c['ri_source_identity'] = 'other'
    elif fault == 'missing-source': e['exchange_components'][1]['fitting'].pop('source_identity')
    elif fault == 'not-fitted': e['exchange_components'][1]['fitting']['has_k'] = False
    elif fault == 'family': c['canonical_density_envelope_identity'] = 'other'
    elif fault == 'missing-family': c.pop('canonical_density_envelope_identity')
    elif fault == 'child-unqualified': c['qualified'] = False
    elif fault == 'aggregate-unqualified': r['qualified'] = False
    elif fault == 'targets': c['targets_requested'] = True
    elif fault == 'qualification': r['qualification'] = 'finite-only'
    elif fault == 'child-qualification': c['qualification'] = 'finite-only'
    elif fault == 'energy': r['energy_error_bound'] = float('nan')
    elif fault == 'matrix': r['maximum_error'] = -1.
    elif fault == 'underbound': r['energy_error_bound'] = .1
    elif fault == 'scale-overflow':
        r['coefficients'][1] = c['exchange_fraction'] = e['exchange_components'][1]['coefficient'] = e['xc']['required_exchange_components'][1]['coefficient'] = 1e308
        c['energy_error_bound'] = 1e308
    elif fault == 'child-nan': c['energy_error_bound'] = float('nan')
    elif fault == 'overlap': e['exchange_accuracy'] = _ledger_evidence()['exchange_accuracy']
    elif fault == 'global-overlap': e['exchange_fitting_accuracy'] = fitted_exchange_ledger_evidence()['exchange_fitting_accuracy']
    elif fault == 'not-range': e['range_separated'] = False
    ledger = manta.compose_fixed_density_ledger(e, method='rhf' if fault == 'hf' else 'rks')
    assert not ledger['qualified'] and ledger['fixed_density_energy_error_bound'] is None and ledger['reasons']
    json.dumps(ledger, allow_nan=False)


@pytest.mark.parametrize('bound,coefficient', [(0., np.finfo(float).max), (1e-308, np.finfo(float).max),
    (np.nextafter(0., 1.), -.5), (.25, 0.), (1e-300, -1e-300)])
def test_ks_ledger_extreme_stored_scaling_encloses_exact_product(bound, coefficient):
    from fractions import Fraction
    from vibeqc.manta_periodic import _ledger_scaled_bound
    scaled = _ledger_scaled_bound(bound, coefficient)
    assert scaled is not None and np.isfinite(scaled)
    assert Fraction.from_float(scaled) >= Fraction.from_float(bound) * abs(Fraction.from_float(coefficient))
    if not bound or not coefficient: assert scaled == 0.
    e = fitted_exchange_ledger_evidence()
    e.update(exact_exchange_fraction=coefficient, range_separated=False)
    e['exchange_fitting_accuracy'].update(exchange_fraction=coefficient, energy_error_bound=bound, maximum_error=0.)
    ledger = manta.compose_fixed_density_ledger(e, method='rks')
    assert ledger['qualified'] == (coefficient != 0.)


def metric_exchange_case(nk=2,cutoff=1e-12,component=True,kernel=None,zero_weight=False):
    a,_,pair,q,offsets,o,c=fitting_case(nk,cutoff,component,kernel)
    a['options'].metric_assembly_rank=True
    if zero_weight:
        from tests.test_manta_periodic_pairs import options
        a['weights']=np.array([0.,1.]);c['weights']=a['weights']
        pair=manta.PeriodicPairSource(a['orbital'],a['system'].lattice,cube(0),a['momenta'],a['weights'],options(0,None,.2),a['budget'])
    if kernel:q,offsets=transfers(c['cell'],c['k'],2,screened_zero=kernel[0]=='erfc')
    o.metric_assembly_matrix_error=o.metric_assembly_energy_error=1e20
    return a,manta.PeriodicRIFockBuilder(**a),pair,q,offsets,o,c


def metric_exchange_decimal(a,ri,o,d):
    """Independent exact-source perturbation contraction and envelope, 180 digits."""
    from tests.test_manta_periodic_gdf_tiles import assemble
    from tests.test_manta_inverse_action import dc,add,mul,conj,dot
    from tests.test_manta_hermitian_inverse import _decimal_spectral_function
    nk=len(a['weights']);n=a['orbital'].nbasis;z=(Decimal(0),Decimal(0));D=Decimal.from_float
    with localcontext() as ctx:
        ctx.prec=180
        bounds=np.zeros((nk,n,n),object);bounds[:]=Decimal(0)
        maximum=bounds.copy();delta=np.empty(d.shape,object)
        for x in np.ndindex(d.shape):delta[x]=z
        metrics=iter(ri.info.metrics);gamma_metric=next(metrics)
        for k in range(nk):
            for l in range(nk):
                kw=dict(orbital=a['orbital'],auxiliary=a['auxiliary'],system=a['system'],options=a['options'],budget=a['budget'])
                if 'exchange_kernel' in a:kw['physical_kernel']=a['exchange_kernel']
                if k==l:
                    kw.update(q=np.zeros(3),ket_kpoints=a['momenta'],vectors=a['gamma_vectors']);metric=gamma_metric;ki=k
                else:
                    lo,hi=a['transfer_offsets'][k*nk+l:k*nk+l+2]
                    kw.update(q=a['momenta'][l]-a['momenta'][k],ket_kpoints=a['momenta'][l:l+1],vectors=a['pair_vectors'][lo:hi]);metric=next(metrics);ki=0
                src=manta.PeriodicGDFTiles(**kw);t=assemble(src)[ki];norm=[]
                for mu in range(n):
                    norm.append([sum(abs(dc(v)[0])+abs(dc(v)[1]) for v in t[:,mu,aa]) for aa in range(n)])
                eta=D(metric.assembly_response.inverse_error_bound);weight=D(float(a['weights'][l]))
                # Independently perturb the actual canonical lower-triangle metric.
                m=src.metric().values.copy();m=np.tril(m)+np.tril(m,-1).conj().T;m[np.diag_indices(2)]=m.diagonal().real
                other=m.copy();other[0,0]+=metric.assembly_perturbation_bound*.25
                assert abs(other[0,0]-m[0,0])<=metric.assembly_perturbation_bound
                ca,cb=(_decimal_spectral_function(x,a['ri_options'].absolute_cutoff) for x in (m,other))
                aa,bb,re,im=[y-x for x,y in zip(ca,cb)];change=[[(aa,Decimal(0)),(re,im)],[(re,-im),(bb,Decimal(0))]]
                for mu in range(n):
                    for nu in range(n):
                        for aa in range(n):
                            for bb in range(n):
                                coefficient=eta*weight*norm[mu][aa]*norm[nu][bb]
                                maximum[k,mu,nu]=max(maximum[k,mu,nu],coefficient)
                                if o.density_entry_limits:bounds[k,mu,nu]+=coefficient*D(o.density_entry_limits[l*n*n+aa*n+bb])
                                # T_mu,a^T conj(delta C) conj(T_nu,b), consistent
                                # with B=F* T and B D B*. Spectral response is Hermitian.
                                value=z
                                for ip in range(2):
                                    for jp in range(2):value=add(value,mul(mul(dc(t[ip,mu,aa]),conj(change[ip][jp])),conj(dc(t[jp,nu,bb]))))
                                for s in range(d.shape[0]):delta[s,k,mu,nu]=add(delta[s,k,mu,nu],mul((weight,Decimal(0)),mul(value,dc(d[s,l,aa,bb]))))
        for x in np.ndindex(bounds.shape):bounds[x]=min(bounds[x],D(o.density_l1_limit)*maximum[x]) if o.density_entry_limits else D(o.density_l1_limit)*maximum[x]
        energy=z
        for s,k,mu,nu in np.ndindex(d.shape):energy=add(energy,mul((D(float(a['weights'][k]))/2,Decimal(0)),mul(dc(d[s,k,nu,mu]),delta[s,k,mu,nu])))
        observed=np.empty(d.shape,object)
        for x in np.ndindex(d.shape):observed[x]=(delta[x][0]**2+delta[x][1]**2).sqrt()
        return bounds,observed,abs(energy[0])


@pytest.mark.parametrize('nk,cutoff,component,kernel,zero_weight',[
    (1,1e-12,False,None,False),(2,1e-12,True,None,False),
    (2,.5,True,None,False),(2,1e9,True,None,False),
    (2,1e-12,True,None,True),(2,1e-12,True,('erf',.43),False),
    (2,1e-12,True,('erfc',.43),False)])
def test_metric_exchange_decimal_channels_and_fixed_source(nk,cutoff,component,kernel,zero_weight):
    from tests.test_manta_periodic_ri_q import densities
    a,ri,pair,q,offsets,o,c=metric_exchange_case(nk,cutoff,component,kernel,zero_weight)
    r=manta.certify_periodic_ri_exchange_fitting(ri,pair,q,offsets,o,a['budget']);d=densities(nk)*.04;r.validate_density(d)
    expected,observed,energy=metric_exchange_decimal(a,ri,o,d)
    actual=np.asarray(r.metric_assembly_error_bounds).reshape(nk,2,2)
    assert r.metric_assembly_requested and r.metric_assembly_qualified and r.metric_assembly_channels==1+nk*(nk-1)
    assert len(r.metric_assembly_identity)==64
    np.testing.assert_allclose(actual,np.asarray(expected,float),rtol=1e-13,atol=0)
    for x in np.ndindex(observed.shape):assert observed[x]<=Decimal.from_float(actual[x[1:]])
    assert energy<=Decimal.from_float(r.metric_assembly_energy_bound)
    if cutoff==1e9:assert not actual.any() and r.metric_assembly_energy_bound==0.
    o.metric_assembly_matrix_error=o.metric_assembly_energy_error=0.
    control=manta.certify_periodic_ri_exchange_fitting(ri,pair,q,offsets,o,a['budget'])
    assert control.certificate_identity==r.certificate_identity
    np.testing.assert_array_equal(control.error_bounds,r.error_bounds)
    assert not control.metric_assembly_requested


@pytest.mark.parametrize('field,quantity',[('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),('primitive_quartets','primitive_products'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_metric_exchange_caps(field,quantity):
    a,ri,pair,q,offsets,o,c=metric_exchange_case();b=a['budget']
    p=manta.plan_periodic_ri_exchange_fitting(ri,pair,len(q),o,b)
    setattr(b,field,getattr(p,quantity)-1)
    with pytest.raises((ValueError,RuntimeError),match='budget'):manta.certify_periodic_ri_exchange_fitting(ri,pair,np.full_like(q,np.nan),offsets,o,b)
    setattr(b,field,getattr(p,quantity));assert manta.certify_periodic_ri_exchange_fitting(ri,pair,q,offsets,o,b).metric_assembly_qualified


def test_metric_exchange_strict_targets_and_missing_response():
    a,ri,pair,q,offsets,o,c=metric_exchange_case()
    r=manta.certify_periodic_ri_exchange_fitting(ri,pair,q,offsets,o,a['budget'])
    o.metric_assembly_matrix_error=r.metric_assembly_matrix_bound;o.metric_assembly_energy_error=r.metric_assembly_energy_bound
    assert manta.certify_periodic_ri_exchange_fitting(ri,pair,q,offsets,o,a['budget']).metric_assembly_qualified
    o.metric_assembly_matrix_error=np.nextafter(r.metric_assembly_matrix_bound,0.)
    with pytest.raises(RuntimeError,match='assembly exchange targets unmet'):manta.certify_periodic_ri_exchange_fitting(ri,pair,q,offsets,o,a['budget'])
    o.metric_assembly_matrix_error=r.metric_assembly_matrix_bound;o.metric_assembly_energy_error=np.nextafter(r.metric_assembly_energy_bound,0.)
    with pytest.raises(RuntimeError,match='assembly exchange targets unmet'):manta.certify_periodic_ri_exchange_fitting(ri,pair,q,offsets,o,a['budget'])
    a['options'].metric_assembly_rank=False;other=manta.PeriodicRIFockBuilder(**a)
    with pytest.raises(RuntimeError,match='qualified metric_assembly_rank'):manta.certify_periodic_ri_exchange_fitting(other,pair,q,offsets,o,a['budget'])
    for bad in (0.,-1.,float('nan'),float('inf')):
        o.metric_assembly_matrix_error=bad
        with pytest.raises(ValueError):manta.plan_periodic_ri_exchange_fitting(ri,pair,len(q),o,a['budget'])


def metric_exchange_job(directory,profile='rhf'):
    from dataclasses import replace
    job,policy=(range_admission_job(directory,profile) if profile.startswith(('hse','cam'))
                else exchange_admission_job(directory,profile))
    policy.metric_assembly_matrix_error=policy.metric_assembly_energy_error=1e20
    job['periodic_fitting']=replace(job['periodic_fitting'],metric_assembly_rank=True)
    return job,policy


@pytest.mark.parametrize('profile',['rhf','uhf','pbe0-rks','pbe0-uks','hse06-uks','cam-b3lyp-rks'])
def test_metric_exchange_complete_states_and_scope(tmp_path,profile):
    from dataclasses import replace
    job,policy=metric_exchange_job(tmp_path,profile);result=vq.run_periodic_job(**job)
    assert result.converged;e=result.integral_provenance
    parts=e['range_exchange_accuracy']['components'] if profile.startswith(('hse','cam')) else [e['exchange_fitting_accuracy']]
    for part in parts:
        c=part['metric_assembly'];assert c['requested'] and c['qualified'] and c['identity']
        assert c['matrix_bound']>0 and c['energy_bound']>0
        assert c['channels']==1+len(job['mesh'])*(len(job['mesh'])-1)
    policy.metric_assembly_matrix_error=policy.metric_assembly_energy_error=0.
    control=vq.run_periodic_job(**dict(job,output=tmp_path/'control/job'))
    assert control.converged and control.energy==result.energy
    for name in ('density','fock','hcore','overlap','coefficients','orbital_energies'):
        np.testing.assert_array_equal(getattr(result.native,name),getattr(control.native,name))
    assert e['source_identity']==control.integral_provenance['source_identity']
    assert e['component_ledger']==control.integral_provenance['component_ledger']


def test_metric_exchange_complete_refusals(tmp_path):
    from dataclasses import replace
    job,policy=metric_exchange_job(tmp_path)
    policy.metric_assembly_matrix_error=policy.metric_assembly_energy_error=1e-100
    with pytest.raises(RuntimeError,match='metric assembly exchange targets unmet'):vq.run_periodic_job(**job)
    assert not job['output'].with_suffix('.qvf').exists()
    with pytest.raises(ValueError,match='metric_assembly_rank'):replace(job['periodic_fitting'],metric_assembly_rank=False)
    policy.metric_assembly_matrix_error=-1.
    with pytest.raises(ValueError,match='paired finite'):replace(job['periodic_fitting'])
