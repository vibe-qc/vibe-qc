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


@pytest.mark.parametrize('radius',[0,1,2,3])
def test_explicit_completion_cube_can_be_smaller_than_retained_inventory(radius):
    c=fixture(images=cube(2),cutoff=4.5)
    result=manta.bound_periodic_one_electron_images(c['source'],completion_cube_radius=radius)
    p=manta.plan_periodic_one_electron_images(c['source'],radius)
    assert p.cube_cells==(2*radius+1)**3==result.receipt.cube_cells
    finite=ordered(c,c['images'],c['cutoff']);full=ordered(c,cube(3))
    for index,matrix in enumerate((result.overlap_error_bound,result.kinetic_error_bound)):
        assert np.all(abs(full[index]-finite[index])<=matrix[None]+3e-12)
    if radius==2:
        original=manta.bound_periodic_one_electron_images(c['source'])
        assert original.receipt.source_identity==result.receipt.source_identity
        np.testing.assert_array_equal(original.kinetic_error_bound,result.kinetic_error_bound)


def test_sparse_far_inventory_can_use_a_bounded_explicit_completion_cube():
    c=fixture(images=np.array([[0,0,0],[10**6,0,0],[-10**6,0,0]],dtype=np.int64),cutoff=1.)
    b=manta.bound_periodic_one_electron_images(c['source'],2)
    assert b.receipt.cube_radius==2 and b.receipt.maximum_overlap_error>0
    with pytest.raises((ValueError,RuntimeError,OverflowError),match='budget|overflow'):
        manta.plan_periodic_one_electron_images(c['source'],2**63)


def image_accuracy_targets(value=1e6,radius=2):
    o=manta.PeriodicOneElectronAccuracyOptions()
    o.overlap_error=o.kinetic_error=o.kinetic_energy_error=value
    o.completion_cube_radius=radius
    return o


def ri_image_job(directory,method='rhf',species='Be',limit=100,automatic=False):
    import dataclasses
    from tests.test_manta_periodic_ri_preparation import prepared_job
    from tests.test_manta_periodic_ri_hierarchy import hierarchy_ks_job
    from tests.test_manta_periodic_ri_pair_tree import automatic_accuracy
    if method in ('rhf','uhf'):
        job,args,k=prepared_job(directory,method,species,limit);scf=args['options'].scf
    else:
        job,args,_,k,_=hierarchy_ks_job(directory,1 if species=='Be' else 2,limit);scf=args['options'].ks.scf
        if method in ('pbe0','hse06','hse06-ri-k'):
            job.update(functional='hse06' if method.startswith('hse06') else method,
                       exchange=manta.ExchangeBuildOptions(algorithm='exact'),exchange_exxdiv='ewald')
        if method=='hse06-ri-k':
            job['coulomb']=dataclasses.replace(job['coulomb'],algorithm='exact')
            job['exchange']=manta.ExchangeBuildOptions(algorithm='ri')
            job['periodic_fitting']=dataclasses.replace(job['periodic_fitting'],coupling_multipole_error=0.,
                coupling_multipole_hierarchy=False,coupling_multipole_pair_tree=False)
    if automatic:
        job['periodic_fitting']=dataclasses.replace(job['periodic_fitting'],coupling_multipole_pair_tree=True,
            accuracy=automatic_accuracy(1e-6,1e-6,0.))
    job['periodic_integrals']=dataclasses.replace(job['periodic_integrals'],one_electron_accuracy=image_accuracy_targets())
    return job,args,k,scf


@pytest.mark.parametrize('method,species,limit,automatic',[
    ('rhf','Be',100,False),('rhf','Be',100,True),('uhf','H',100,False),('uhf','Li',3,True),
    ('rks','Be',100,True),('uks','Li',3,False),('pbe0','Be',100,True),
    ('hse06','Li',100,True),('hse06-ri-k','Li',3,False)])
def test_complete_ri_image_certificate_and_unchanged_public_state(tmp_path,method,species,limit,automatic):
    import dataclasses
    from tests.test_bounded_periodic_rhf_solver import _density_envelope_caps
    from vibeqc.guess_read import _spin_density_k_from_qvf_bloch_payload
    job,args,k,scf=ri_image_job(tmp_path,method,species,limit,automatic)
    result=vq.run_periodic_job(**job);n=result.native;c=n.receipt.one_electron_accuracy
    assert result.converged==(limit==100) and c.qualified
    assert n.plan.one_electron_accuracy and n.plan.density_envelope
    assert (n.receipt.accuracy_density is not None)==automatic
    ne=args['system'].n_electrons();nb=(ne-args['system'].multiplicity+1)//2;na=ne-nb
    caps,_=_density_envelope_caps(n.overlap,na,nb,scf)
    envelope=core._bound_bounded_periodic_density_envelope(n.overlap,na,nb,scf,caps)
    expected=min(np.sum(envelope.weighted_entry_bounds*c.images.kinetic_error_bound[None]),
                 envelope.density_l1_limit*c.images.receipt.maximum_kinetic_error)
    assert c.kinetic_energy_error_bound>=expected*(1-1e-14)
    assert c.kinetic_energy_error_bound==pytest.approx(expected,rel=2e-12)
    assert c.density_l1_limit==envelope.density_l1_limit
    assert len(c.certificate_identity)==len(c.density_envelope_identity)==64
    certificate=result.integral_provenance['one_electron_accuracy']
    assert certificate['certificate_identity']==c.certificate_identity
    assert certificate['images']['roundoff_included'] is False
    assert 'density-relaxation' in certificate['excludes']
    assert not result.integral_provenance['symmetry_reduced']
    d=np.asarray(_spin_density_k_from_qvf_bloch_payload(job['output'].with_suffix('.qvf'))).sum(axis=0)
    assert np.all(abs(d)/len(k)<=envelope.weighted_entry_bounds)
    control=dict(job,output=tmp_path/'without-images'/'job')
    control['periodic_integrals']=dataclasses.replace(job['periodic_integrals'],one_electron_accuracy=None)
    other=vq.run_periodic_job(**control)
    assert other.native.receipt.one_electron_accuracy is None
    for name in ('density','fock','overlap','hcore'):np.testing.assert_array_equal(getattr(n,name),getattr(other.native,name))
    assert result.energy==other.energy and result.converged==other.converged
    for name in ('source_identity','overlap_identity','kinetic_identity','coulomb_identity','exchange_identity'):
        assert result.integral_provenance[name]==other.integral_provenance[name]
    job['periodic_integrals'].one_electron_accuracy.overlap_error=1.
    exported=c.targets;exported.overlap_error=2.
    assert c.targets.overlap_error==1e6


@pytest.mark.parametrize('field',['overlap_error','kinetic_error','kinetic_energy_error'])
def test_ri_image_targets_refuse_unqualified_physical_domain(tmp_path,field):
    job,_,_,_=ri_image_job(tmp_path);setattr(job['periodic_integrals'].one_electron_accuracy,field,1e-300)
    with pytest.raises(ValueError,match='accuracy target'):vq.run_periodic_job(**job)


@pytest.mark.parametrize('field,quantity',[('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),
    ('primitive_quartets','primitive_pairs'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_ri_image_complete_method_admission_precedes_poisoned_arrays(tmp_path,field,quantity):
    from tests.test_manta_periodic_ri_domains import explicit
    _,args,_,_=ri_image_job(tmp_path);args['options'].one_electron_accuracy=image_accuracy_targets()
    args['options'].one_electron_accuracy.overlap_image_rank=True
    domains=manta.build_periodic_ri_rhf_domains(**args);a=explicit(args,domains)
    p=manta.plan_periodic_ri_rhf(a['basis'],a['auxiliary'],a['system'],a['mesh'],len(a['reciprocal_vectors']),
        max(np.diff(a['transfer_offsets'])),len(a['translations']),a['options'],a['budget'])
    setattr(a['budget'],field,getattr(p,quantity));result=core._manta.run_periodic_ri_rhf(**a)
    assert result.receipt.one_electron_accuracy.qualified
    setattr(a['budget'],field,getattr(p,quantity)-1);a['translations'][:]=np.nan
    with pytest.raises((ValueError,RuntimeError),match='budget'):core._manta.run_periodic_ri_rhf(**a)


def test_ri_image_options_reject_invalid_and_non_ri_controls(tmp_path):
    import dataclasses
    job,args,_,_=ri_image_job(tmp_path)
    with pytest.raises(TypeError,match='PeriodicOneElectronAccuracyOptions'):
        dataclasses.replace(job['periodic_integrals'],one_electron_accuracy=object())
    for value in (0.,-1.,float('inf'),float('nan')):
        o=image_accuracy_targets();o.kinetic_error=value
        with pytest.raises(ValueError,match='finite positive'):
            dataclasses.replace(job['periodic_integrals'],one_electron_accuracy=o)
        args['options'].one_electron_accuracy=o
        with pytest.raises(ValueError,match='accuracy'):
            manta.plan_periodic_ri_rhf_from_cutoffs(**args)
    job['coulomb']=manta.CoulombBuildOptions(algorithm='exact')
    job['periodic_fitting']=None
    with pytest.raises(NotImplementedError,match='requires a periodic RI method'):vq.run_periodic_job(**job)


@pytest.mark.parametrize('radius',[1,2,3])
def test_expanded_physical_support_meets_tight_image_targets(tmp_path,radius):
    import dataclasses
    from tests.test_manta_periodic_ri_domains import explicit
    job,args,k,_=ri_image_job(tmp_path)
    target=image_accuracy_targets(1e-6,radius)
    job['periodic_integrals']=dataclasses.replace(job['periodic_integrals'],pair_cutoff_bohr=10.,one_electron_accuracy=target)
    args['options'].pair_cutoff_bohr=args['options'].fitting.pair_cutoff_bohr=10.
    # The completion radius is only an enclosure control. A small cube may
    # overcount retained exterior blocks and legitimately fail a tight target.
    if radius==1:
        with pytest.raises(ValueError,match='accuracy target'):vq.run_periodic_job(**job)
        return
    result=vq.run_periodic_job(**job);n=result.native;c=n.receipt.one_electron_accuracy
    assert result.converged and 0<c.kinetic_energy_error_bound<1e-6
    domains=manta.build_periodic_ri_rhf_domains(**args)
    labels=np.rint(domains.translations@np.linalg.inv(args['system'].lattice).T).astype(np.int64)
    context=dict(basis=args['basis'],cell=args['system'].lattice,k=k)
    finite=ordered(context,labels,10.);full=ordered(context,cube(3))
    np.testing.assert_allclose(n.overlap,finite[0],atol=3e-12,rtol=1e-12)
    kinetic=manta.compute_bloch_one_electron(args['basis'],'kinetic',domains.translations,k,10.).values
    np.testing.assert_allclose(kinetic,finite[1],atol=3e-12,rtol=1e-12)
    for i,matrix in enumerate((c.images.overlap_error_bound,c.images.kinetic_error_bound)):
        assert np.all(abs(full[i]-finite[i])<=matrix[None]+3e-12)
    difference=abs(np.einsum('kij,kji->',n.density,full[1]-finite[1]).real/len(k))
    assert difference<=c.kinetic_energy_error_bound+3e-12
    if radius==2:
        from tests.test_manta_periodic_rhf import _nuclear_reference
        from tests.test_manta_periodic_ri_rhf import scf_reference
        a=explicit(args,domains);s,h,build=ri_image_hf_reference(a,k,finite)
        f,energy=build(n.density);reference=scf_reference(s,h,build,args['system'].n_electrons()//2)
        np.testing.assert_allclose(n.hcore,h,atol=8e-12,rtol=0)
        np.testing.assert_allclose(n.fock,f,atol=8e-11,rtol=0)
        np.testing.assert_allclose(n.density,reference,atol=5e-9,rtol=0)
        assert abs(result.energy-sum(_nuclear_reference(a)[0])-sum(energy))<8e-11




def ri_image_hf_reference(a,k,finite):
    """Extend the origin-only HF fixture reference to actual Bloch S/T/V."""
    from tests.test_manta_periodic_ri_multipole import multipole_hf_reference
    from tests.test_manta_periodic_nuclear import _reciprocal_quadrature
    from vibeqc.bipole_fock_ewald import probe_charge_madelung_supercell
    old_s,old_h,old_build,_,_=multipole_hf_reference(a,k,'rhf')
    o=a['options'];system=a['system'];lattice_options=vq.LatticeSumOptions()
    lattice_options.cutoff_bohr=o.pair_cutoff_bohr
    lattice_options.nuclear_cutoff_bohr=o.nuclear_cutoff_bohr;lattice_options.pair_complete_1e=True
    short=vq.compute_nuclear_erfc_lattice(a['basis'],system,o.alpha,lattice_options)
    translations=np.array([c.r_cart for c in short.cells])
    reciprocal=_reciprocal_quadrature(a['basis'],system,a['reciprocal_vectors'],translations,o.alpha,o.pair_cutoff_bohr)
    s=finite[0];h=finite[1]+np.einsum('kr,rij->kij',np.exp(1j*k@translations.T),np.asarray(short.blocks)+reciprocal)
    h+=np.pi*sum(atom.Z for atom in system.unit_cell)/(o.alpha**2*abs(np.linalg.det(system.lattice)))*s
    xi=0. if o.exchange_q0=='strict-zero-mode' else probe_charge_madelung_supercell(
        system,tuple(a['mesh'].mesh),alpha=o.madelung_alpha,precision=1e-14)
    def build(d):
        f,energy=old_build(d)
        old_correction=-.5*xi*(old_s@d@old_s);correction=-.5*xi*(s@d@s)
        f=f+h-old_h+correction-old_correction
        terms=(np.einsum('kij,kji->',d,h).real/len(k),energy[1],energy[2],
               .5*np.einsum('kij,kji->',d,correction).real/len(k))
        return f,terms
    return s,h,build


@pytest.mark.parametrize('scale',[1e-100,1.,1e100])
@pytest.mark.parametrize('relative',[0.,.2])
def test_overlap_image_rank_independent_perturbed_spectra(scale,relative):
    from decimal import Decimal,localcontext
    from tests.test_bounded_periodic_rhf_solver import _options
    scf=_options(overlap_rank_absolute_floor=.1*scale,overlap_rank_relative_floor=relative)
    a=np.array([[.4,.13+.29j],[.13-.29j,2.7]],complex)*scale
    errors=np.full((2,2),.02*scale)
    c=manta.bound_periodic_overlap_image_rank(a[None],errors,1 if relative else 2,scf)
    assert c.qualified and c.maximum_perturbation_bound<c.minimum_stability_radius
    with localcontext() as ctx:
        ctx.prec=90;d=Decimal.from_float
        for sign in (-1,1):
            # Exact Decimal eigenvalues for the stored inputs plus a signed
            # Hermitian perturbation, independent of native Jacobi/intervals.
            aa=d(a[0,0].real)+sign*d(errors[0,0]);bb=d(a[1,1].real)-sign*d(errors[1,1])
            re=d(a[0,1].real)+sign*d(errors[0,1]);im=d(a[0,1].imag)
            gap=((aa-bb)**2+4*(re*re+im*im)).sqrt()
            eig=[(aa+bb-gap)/2,(aa+bb+gap)/2]
            cutoff=max(d(scf.overlap_rank_absolute_floor),d(relative)*max(Decimal(0),eig[-1]))
            assert sum(x>cutoff for x in eig)==c.spectra[0].rank
        assert d(c.maximum_perturbation_bound)>=sum(map(d,errors[0]))


@pytest.mark.parametrize('relative',[0.,.2])
def test_overlap_image_rank_clusters_crossings_and_projection(relative):
    from tests.test_bounded_periodic_rhf_solver import _options
    scf=_options(overlap_rank_absolute_floor=.1,overlap_rank_relative_floor=relative)
    a=np.diag([.01,.01,.7,.7,2.,3.]).astype(complex)
    e=np.full((6,6),1e-4)
    c=manta.bound_periodic_overlap_image_rank(a[None],e,4,scf)
    assert c.qualified and c.spectra[0].rank==4
    rng=np.random.default_rng(625)
    for _ in range(12):
        z=rng.uniform(-.5,.5,(6,6))+1j*rng.uniform(-.5,.5,(6,6));h=(z+z.conj().T)/2*e
        w=np.linalg.eigvalsh(a+h)
        assert np.count_nonzero(w>max(.1,relative*max(0.,w[-1])))==4
    assert not manta.bound_periodic_overlap_image_rank(a[None],e,5,scf).qualified
    boundary=max(.1,relative*3.)
    a[0,0]=boundary
    assert not manta.bound_periodic_overlap_image_rank(a[None],e,4,scf).qualified
    # Deliberately non-Hermitian stored lanes: displacement from the SAME
    # canonical projection is included, including imaginary diagonals.
    a=np.array([[1.+1e-5j,.2+2e-5j],[.2-1e-5j,2.]],complex)
    c=manta.bound_periodic_overlap_image_rank(a[None],np.zeros((2,2)),2,scf)
    projected=(a+a.conj().T)/2
    assert c.qualified and c.projection_bounds[0]>=np.linalg.norm(projected-a,2)>0


@pytest.mark.parametrize('field,quantity',[('memory_bytes','modeled_peak_bytes'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_overlap_image_rank_exact_caps_before_poison(field,quantity):
    from tests.test_bounded_periodic_rhf_solver import _options
    scf=_options();b=manta.ResourceBudget();p=manta.plan_periodic_overlap_image_rank(2,1,scf,b)
    a=np.eye(2,dtype=complex)[None];e=np.zeros((2,2))
    setattr(b,field,getattr(p,quantity))
    assert manta.bound_periodic_overlap_image_rank(a,e,2,scf,b).qualified
    setattr(b,field,getattr(p,quantity)-1)
    with pytest.raises(ValueError,match='budget'):
        manta.bound_periodic_overlap_image_rank(a*np.nan,e,2,scf,b)


def test_overlap_image_rank_strict_radius_and_invalid_entries():
    from tests.test_bounded_periodic_rhf_solver import _options
    scf=_options(overlap_rank_absolute_floor=.1,overlap_rank_relative_floor=0.)
    a=np.eye(2,dtype=complex)[None];e=np.zeros((2,2))
    c=manta.bound_periodic_overlap_image_rank(a,e,2,scf)
    assert c.qualified
    e[0,0]=c.minimum_stability_radius
    assert not manta.bound_periodic_overlap_image_rank(a,e,2,scf).qualified
    for bad in (-1.,float('nan'),float('inf')):
        e[0,0]=bad
        with pytest.raises(ValueError,match='nonnegative'):
            manta.bound_periodic_overlap_image_rank(a,e,2,scf)


def overlap_rank_job(directory,method='rhf',species='Be'):
    import dataclasses
    job,args,k,scf=ri_image_job(directory,method,species,100)
    target=image_accuracy_targets(1e6,3);target.overlap_image_rank=True
    job['periodic_integrals']=dataclasses.replace(job['periodic_integrals'],pair_cutoff_bohr=10.,one_electron_accuracy=target)
    job['periodic_fitting']=dataclasses.replace(job['periodic_fitting'],auxiliary_cutoff_bohr=12.1)
    physical=args['options'].ks if hasattr(args['options'],'ks') else args['options']
    physical.pair_cutoff_bohr=10.
    args['options'].fitting.pair_cutoff_bohr=10.
    args['options'].fitting.auxiliary_cutoff_bohr=12.1
    return job,args,k,scf


@pytest.mark.parametrize('method,species',[('rhf','Be'),('uhf','H'),('rks','Be'),('pbe0','Be'),('hse06','Li')])
def test_overlap_image_rank_complete_methods_preserve_state(tmp_path,method,species):
    import dataclasses
    job,_,_,_=overlap_rank_job(tmp_path,method,species)
    result=vq.run_periodic_job(**job);rank=result.native.receipt.one_electron_accuracy.overlap_rank
    assert result.converged and rank.qualified
    assert rank.maximum_perturbation_bound<rank.minimum_stability_radius
    assert len(rank.spectra)==len(result.native.overlap)
    assert all(c.rank==result.native.solver_memory.retained_rank for c in rank.spectra)
    evidence=result.integral_provenance['one_electron_accuracy']['overlap_image_rank']
    assert evidence['qualified'] and 'integral-construction-arithmetic' in evidence['excludes']
    target=image_accuracy_targets(1e6,3)
    control=vq.run_periodic_job(**dict(job,output=tmp_path/'control/job',
        periodic_integrals=dataclasses.replace(job['periodic_integrals'],one_electron_accuracy=target)))
    assert control.converged and control.energy==result.energy
    assert control.native.receipt.source_identity==result.native.receipt.source_identity
    assert control.native.receipt.one_electron_accuracy.overlap_rank is None
    for name in ('density','fock','hcore','overlap','coefficients','orbital_energies'):
        np.testing.assert_array_equal(getattr(result.native,name),getattr(control.native,name))


def test_overlap_image_rank_fixed_source_refuses_near_cutoff_before_scf(tmp_path):
    _,args,_,_=ri_image_job(tmp_path)
    args['options'].one_electron_accuracy=image_accuracy_targets()
    args['options'].one_electron_accuracy.overlap_image_rank=True
    args['options'].scf.overlap_rank_absolute_floor=.0089
    with pytest.raises(ValueError,match='overlap image rank is unresolved'):
        core._manta.run_periodic_ri_rhf_from_cutoffs(**args)
