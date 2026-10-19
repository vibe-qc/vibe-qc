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


def transfers(cell,k,radius,inner_radius=None):
    labels=cube(radius)
    if inner_radius is not None:labels=labels[np.max(abs(labels),axis=1)>inner_radius]
    base=labels@(2*np.pi*np.linalg.inv(cell));chunks=[]
    for target in k:
        for source in k:
            q=base+target-source;chunks.append(q[np.any(q!=0,axis=1)])
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


def reference(c,d,q,offsets,images,cutoff=None):
    out=np.zeros_like(d);nk=len(c['k']);cells=images@c['cell'].T
    for target,k in enumerate(c['k']):
        for source in range(nk):
            pair=target*nk+source
            for start in range(offsets[pair],offsets[pair+1],4096):
                wave=q[start:min(start+4096,offsets[pair+1])];panel=ordered_fourier(c['basis'],wave,cells,k,cutoff)
                weight=c['weights'][source]*4*np.pi/abs(np.linalg.det(c['cell']))/np.sum(wave*wave,axis=1)
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
