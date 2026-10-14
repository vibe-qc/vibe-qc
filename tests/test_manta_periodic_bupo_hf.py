"""Complete matched-electrostatic BUPO HF with independently integrated SCF."""
import json
from math import fsum
import numpy as np
import pytest
from scipy.linalg import eigh

import vibeqc as vq
from vibeqc import manta,_vibeqc_core as core
from tests.test_manta_periodic_pairs import options as pair_options,coefficients
from tests.test_manta_periodic_bubbles import bubble_options
from tests.test_manta_periodic_bupo import options as bupo_options
from tests.test_manta_periodic_ewald import cube,reference as ee_reference
from tests.test_manta_periodic_electrostatic import finite_nuclear_reference,point_ewald
from tests.test_manta_periodic_exchange import _quadrature_bloch


def case(*,spin='restricted',z=2,shifted=False,q0='strict-zero-mode',iterations=40,eta=.2,
         image_radius=1,real_radius=4,reciprocal_radius=8,cutoff=3.,redundant=False,basis_image=False,two_atoms=False,
         cell_size=12.,shell_exponents=None,exchange_radius=5):
    cell=np.eye(3)*cell_size;atoms=[vq.Atom(z,[.1,.2,-.1])]
    if two_atoms:atoms=[vq.Atom(1,[.1,.2,-.1]),vq.Atom(1,[1.5,.2,-.1])]
    system=vq.PeriodicSystem(3,cell,atoms);system.multiplicity=1 if z==2 else 2
    shells=[]
    exponents=[4.,1.2,1.2] if redundant else ([8.,4.,1.2] if z==3 else [4.,1.2])
    if shell_exponents is not None:exponents=shell_exponents
    for i,exponent in enumerate(exponents):
        s=core.ShellInfo();s.atom_index=i if two_atoms else 0;s.origin=atoms[s.atom_index].xyz;s.l=0;s.pure=True
        s.exponents=[exponent];s.coefficients=[1.];shells.append(s)
    if basis_image:shells[1].origin=(np.array(shells[1].origin)+cell[:,0]).tolist()
    basis_system=system
    if basis_image:
        basis_system=vq.PeriodicSystem(3,cell,[vq.Atom(atoms[s.atom_index].Z,s.origin) for s in shells])
        for i,s in enumerate(shells):s.atom_index=i
    basis=vq.BasisSet(basis_system.unit_cell_molecule(),shells,'bupo-hf-reference',False)
    mesh=manta.RegularKMesh((2,1,1) if shifted else (1,1,1),(1,0,0) if shifted else (0,0,0))
    k=np.array([system.reciprocal_lattice()@mesh.fractional(mesh.address(i)) for i in range(len(mesh))])
    weights=np.full(len(mesh),1./len(mesh));images=cube(image_radius)
    pairs=manta.PeriodicPairSource(basis,cell,images,k,weights,pair_options(0,cutoff,.2))
    tree=manta.PeriodicBubbleTree(pairs,bubble_options());direct=manta.PeriodicEwaldSource(pairs,eta,cube(real_radius),cube(reciprocal_radius,reciprocal=True),64)
    o=bupo_options(0.);o.bound_infinite_domain=True;o.bound_pair_images=True
    source=manta.PeriodicBupoSource(tree,direct,cube(0),o)
    base=cube(exchange_radius)@(2*np.pi*np.linalg.inv(cell));chunks=[]
    for target in k:
        for origin in k:
            q=base+target-origin;chunks.append(q[np.any(q!=0.,axis=1)])
    q=np.ascontiguousarray(np.concatenate(chunks));offsets=np.r_[0,np.cumsum([len(x) for x in chunks])].astype(np.int64)
    options=manta.PeriodicBupoHFOptions();options.spin_mode=spin;options.exchange_q0=q0;options.scf.maximum_iterations=iterations
    budget=manta.ResourceBudget();budget.memory_bytes=2<<30;budget.kernel_terms=10**17;budget.scalar_terms=10**16
    budget.shell_calls=10**12;budget.primitive_quartets=10**14
    args=dict(electronic=source,system=system,mesh=mesh,exchange_vectors=q,transfer_offsets=offsets,options=options,budget=budget)
    context=dict(basis=basis,pairs=pairs,cell=cell,k=k,weights=weights,images=images,eta=eta,
                 real_radius=real_radius,reciprocal_radius=reciprocal_radius,cutoff=cutoff)
    return args,context


def mixed_case(**kwargs):
    """Two neutral centers with admitted far blocks and unchanged 1e-8 targets."""
    a,c=case(**kwargs);atoms=[vq.Atom(2,[0.,0.,0.]),vq.Atom(2,[3.,0.,0.])]
    system=vq.PeriodicSystem(3,c['cell'],atoms);shells=[]
    for i,atom in enumerate(atoms):
        for exponent in (3200.,960.):
            s=core.ShellInfo();s.atom_index=i;s.origin=atom.xyz;s.l=0;s.pure=True
            s.exponents=[exponent];s.coefficients=[1.];shells.append(s)
    b=vq.BasisSet(system.unit_cell_molecule(),shells,'bupo-mixed-hf-reference',False)
    p=manta.PeriodicPairSource(b,c['cell'],c['images'],c['k'],c['weights'],pair_options(8,.2,.16))
    t=manta.PeriodicBubbleTree(p,bubble_options(cap=2))
    e=manta.PeriodicEwaldSource(p,c['eta'],cube(c['real_radius']),cube(c['reciprocal_radius'],reciprocal=True),64)
    o=bupo_options(1e-9);o.bound_infinite_domain=True;o.bound_pair_images=True
    a.update(electronic=manta.PeriodicBupoSource(t,e,cube(0),o),system=system)
    c.update(basis=b,pairs=p,cutoff=.2)
    return a,c


def independent(args,context,fourier_reference=None,*,cache_exchange=False,density_metric='ao'):
    if density_metric not in ('ao','orthonormal'):raise ValueError('Unknown reference density metric')
    b=context['basis'];pairs=context['pairs'];cell=context['cell'];k=context['k'];weights=context['weights'];o=args['options']
    nk,n=len(k),b.nbasis;spins=1 if o.spin_mode=='restricted' else 2
    positions=np.array([a.xyz for a in args['system'].unit_cell]);charges=np.array([a.Z for a in args['system'].unit_cell])
    nr=finite_nuclear_reference(b,pairs,cell,positions,charges,eta=context['eta'],real_radius=context['real_radius'],
        reciprocal_radius=context['reciprocal_radius'],k=k)
    zero=np.zeros((nk,n,n),complex)
    er=ee_reference(b,pairs,zero,cell,context['eta'],cube(context['real_radius']),cube(context['reciprocal_radius'],reciprocal=True),k=k,weights=weights)
    coulomb_matrix=sum(er['component_matrices']);t=np.zeros_like(zero);s=np.zeros_like(zero)
    cells=context['images']@cell.T
    # Independent analytic primitive s overlap/kinetic over the original ordered images.
    for translation in cells:
        for i,a in enumerate(b.shells()):
            for j,bb in enumerate(b.shells()):
                distance=np.linalg.norm(np.array(a.origin)-np.array(bb.origin)-translation)
                if context['cutoff'] is not None and distance>context['cutoff']:continue
                pa,pb=a.exponents[0],bb.exponents[0];mu=pa*pb/(pa+pb)
                overlap=a.coefficients[0]*bb.coefficients[0]*(np.pi/(pa+pb))**1.5*np.exp(-mu*distance**2)
                phase=np.exp(1j*k@translation);s[:,i,j]+=phase*overlap;t[:,i,j]+=phase*mu*(3-2*mu*distance**2)*overlap
    h=t+nr['values'];nn=float(sum(point_ewald(cell,positions,charges,context['eta'],context['real_radius'],context['reciprocal_radius'])))
    exchange=[];q=args['exchange_vectors'];offsets=args['transfer_offsets']
    for target,kt in enumerate(k):
        row=[]
        for source in range(nk):
            ch=target*nk+source;vectors=q[offsets[ch]:offsets[ch+1]]
            panel=(_quadrature_bloch if fourier_reference is None else fourier_reference)(b,vectors,cells,kt,context['cutoff'])
            weight=weights[source]*4*np.pi/abs(np.linalg.det(cell))/np.sum(vectors*vectors,axis=1)
            # Tiny independent-reference systems only: integrate the fixed
            # four-index operator once instead of every reference SCF step.
            # Compensated scalar sums avoid losing small Q terms whose error
            # is amplified by large AO density coefficients.
            if cache_exchange:
                tensor=np.empty((n,n,n,n),complex)
                for aa,bb,i,j in np.ndindex(tensor.shape):
                    values=weight*panel[:,aa,i].conj()*panel[:,bb,j]
                    tensor[aa,bb,i,j]=complex(fsum(values.real),fsum(values.imag))
                row.append(tensor)
            else:row.append((panel,weight))
        exchange.append(row)
    xi=0.
    if o.exchange_q0=='BvK-ewald':
        supercell=cell@np.diag(args['mesh'].mesh)
        alpha=o.madelung_alpha or np.sqrt(np.pi)/abs(np.linalg.det(supercell))**(1/3)
        xi=-2*sum(point_ewald(supercell,np.zeros((1,3)),np.ones(1),alpha,7,7))
    def build(d):
        total=np.sum(d,axis=0);c=np.concatenate([coefficients(e,total,k,weights).ravel() for e in pairs.entries])
        potential=coulomb_matrix@c;j=np.zeros_like(total);offset=0
        for e in pairs.entries:
            a,b=e.offsets;na,nb=e.sizes;v=potential[offset:offset+na*nb].reshape(na,nb);offset+=na*nb
            phase=np.exp(1j*k@e.translation);j[:,a:a+na,b:b+nb]+=phase[:,None,None]*v
            if e.paired:j[:,b:b+nb,a:a+na]+=phase.conj()[:,None,None]*v.T
        kk=np.zeros_like(d)
        for spin in range(spins):
            for target in range(nk):
                for source in range(nk):
                    if cache_exchange:
                        kk[spin,target]+=np.einsum('abij,ab->ij',exchange[target][source],d[spin,source])
                    else:
                        panel,weight=exchange[target][source]
                        kk[spin,target]+=np.einsum('g,gai,ab,gbj->ij',weight,panel.conj(),d[spin,source],panel)
        scale=.5 if spins==1 else 1.
        delta=-scale*xi*np.einsum('kij,skjl,klm->skim',s,d,s)
        f=h[None]+j[None]-scale*kk+delta
        one=float(np.einsum('k,kij,kji->',weights,total,h).real);ee=float(.5*c@coulomb_matrix@c)
        ex=float(-.5*scale*np.einsum('k,skij,skji->',weights,d,kk).real)
        correction=float(.5*np.einsum('k,skij,skji->',weights,d,delta).real)
        return dict(fock=f,energy=nn+one+ee+ex+correction,components=np.array([one,ee,ex,correction,nn]),
                    hartree=j,exchange=kk,correction=delta,coulomb_matrix=coulomb_matrix,canonical_coefficients=c)
    ne=int(sum(charges));nb=(ne-(args['system'].multiplicity-1))//2;na=ne-nb
    counts=[ne//2] if spins==1 else [na,nb];occupancy=2 if spins==1 else 1
    def occupy(f):
        result=np.zeros_like(f)
        for spin,count in enumerate(counts):
            for ik in range(nk):
                ev,u=np.linalg.eigh(s[ik]);keep=ev>1e-10;x=u[:,keep]/np.sqrt(ev[keep])
                _,v=eigh(x.conj().T@f[spin,ik]@x);co=x@v[:,:count]
                result[spin,ik]=occupancy*co@co.conj().T
        return result
    initial=occupy(np.broadcast_to(h,(spins,*h.shape)));d=initial.copy();root=None
    if density_metric=='orthonormal':
        # Canonical orthogonalization (Helgaker et al., Ch.10): compare the
        # occupied projector sqrt(lambda) U^H D U sqrt(lambda). Raw AO
        # coefficients amplify eigensolver roundoff for nearly dependent AOs.
        eigenvalues,vectors=np.linalg.eigh(s)
        root=vectors*np.sqrt(np.where(eigenvalues>1e-10,eigenvalues,0.))[:,None,:]
    for _ in range(200):
        row=build(d);new=occupy(row['fock'])
        delta=d-new if root is None else root[None].conj().swapaxes(-1,-2)@(d-new)@root[None]
        density_change=float(np.max(abs(delta)))
        if density_change<1e-13:break
        d=new
    else:raise AssertionError('Independent BUPO finite-source SCF did not converge')
    return dict(overlap=s,hcore=h,density=d,initial=initial,build=build,converged=row,madelung=xi,
                density_metric=density_metric,density_change=density_change,iterations=_+1)


def calculation(**kwargs):
    args,context=case(**kwargs);r=core._manta.run_periodic_bupo_hf(**args);reference=independent(args,context)
    same=reference['build'](r.density)
    return args,context,r,reference,same


@pytest.mark.parametrize('spin,z',[('restricted',2),('unrestricted',2),('unrestricted',1),('unrestricted',3)])
@pytest.mark.parametrize('shifted',[False,True])
def test_complete_scf_against_independent_gaussian_hf(spin,z,shifted):
    args,context,r,ref,same=calculation(spin=spin,z=z,shifted=shifted)
    assert r.converged and r.receipt.confirmed and r.receipt.electrostatic_accuracy_passed
    np.testing.assert_allclose(r.overlap,ref['overlap'],atol=3e-13,rtol=1e-12)
    np.testing.assert_allclose(r.hcore,ref['hcore'],atol=3e-12,rtol=1e-12)
    np.testing.assert_allclose(r.fock,same['fock'],atol=3e-11,rtol=1e-12)
    np.testing.assert_allclose(r.density,ref['density'],atol=2e-9,rtol=2e-10)
    assert r.receipt.energy==pytest.approx(same['energy'],abs=3e-11)
    assert r.receipt.energy==pytest.approx(ref['converged']['energy'],abs=3e-11)
    np.testing.assert_allclose([r.receipt.one_electron_energy,r.receipt.coulomb_energy,r.receipt.uncorrected_exchange_energy,
        r.receipt.exchange_correction_energy,r.receipt.nuclear_energy],same['components'],atol=3e-11,rtol=1e-12)
    assert r.receipt.electrostatic.energy_error_bound<=args['options'].maximum_electrostatic_energy_error
    assert r.receipt.fock_calls==r.solver_snapshot.fock_calls+1
    assert r.receipt.shell_calls<=r.plan.shell_calls and r.receipt.primitive_pairs<=r.plan.primitive_pairs


@pytest.mark.parametrize('spin',['restricted','unrestricted'])
def test_mixed_near_far_scf_and_converged_madelung(spin):
    a,c=mixed_case(spin=spin,shifted=True,q0='BvK-ewald')
    r=core._manta.run_periodic_bupo_hf(**a);ref=independent(a,c);same=ref['build'](r.density)
    assert r.converged and r.receipt.coulomb.partition.approximate_pairs>0
    assert r.receipt.coulomb.partition.direct_pairs>0
    np.testing.assert_allclose(r.fock,same['fock'],atol=3e-11,rtol=1e-12)
    assert abs(r.receipt.energy-same['energy'])<=r.receipt.electrostatic.energy_error_bound+3e-11
    assert r.receipt.energy==pytest.approx(ref['converged']['energy'],abs=3e-10)
    assert r.receipt.maximum_observed_electrostatic_energy_error<1e-8
    assert r.receipt.maximum_observed_electrostatic_matrix_error<1e-8
    assert r.receipt.madelung==pytest.approx(ref['madelung'],abs=3e-12)


def test_madelung_and_capped_evaluated_density_contract():
    args,context,r,ref,same=calculation(spin='unrestricted',z=1,shifted=True,q0='BvK-ewald',iterations=1)
    assert not r.converged and not r.receipt.confirmed and r.receipt.fock_calls==1
    np.testing.assert_allclose(r.density,ref['initial'],atol=3e-13,rtol=1e-12)
    np.testing.assert_allclose(r.fock,same['fock'],atol=3e-11,rtol=1e-12)
    assert r.receipt.energy==pytest.approx(same['energy'],abs=3e-11)
    assert r.receipt.madelung==pytest.approx(ref['madelung'],abs=3e-12)
    assert np.max(abs(r.density[0]-np.einsum('kio,kjo->kij',r.coefficients[0,:,:,:1],r.coefficients[0,:,:,:1].conj())))>1e-3


@pytest.mark.parametrize('field,planned',[('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),
    ('primitive_quartets','primitive_pairs'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_exact_caps_and_pre_payload_refusal(field,planned):
    args,context=case(iterations=1);nq=int(np.max(np.diff(args['transfer_offsets'])))
    p=manta.plan_periodic_bupo_hf(args['electronic'],args['system'],args['mesh'],nq,args['options'],args['budget'])
    setattr(args['budget'],field,getattr(p,planned));r=core._manta.run_periodic_bupo_hf(**args)
    assert r.plan.modeled_peak_bytes==p.modeled_peak_bytes
    setattr(args['budget'],field,getattr(p,planned)-1);args['exchange_vectors']=np.full_like(args['exchange_vectors'],np.nan)
    with pytest.raises((ValueError,RuntimeError),match='budget'):core._manta.run_periodic_bupo_hf(**args)


def test_charge_source_domain_and_accuracy_refusals():
    args,context=case();args['options'].maximum_electrostatic_energy_error=1e-300
    with pytest.raises(ValueError,match='accuracy target'):core._manta.run_periodic_bupo_hf(**args)
    args,context=case(image_radius=0)
    with pytest.raises(ValueError,match='uncertainty|accuracy target'):core._manta.run_periodic_bupo_hf(**args)
    args,context=case();args['mesh']=manta.RegularKMesh((1,1,1),(1,0,0))
    with pytest.raises(ValueError,match='source momenta'):core._manta.run_periodic_bupo_hf(**args)
    args,context=case();args['system']=vq.PeriodicSystem(3,np.eye(3)*13,args['system'].unit_cell)
    with pytest.raises(ValueError,match='original source lattice'):core._manta.run_periodic_bupo_hf(**args)


def test_original_cutoff_image_domain_controls_errors_without_changing_integrals():
    objects=[]
    for radius in (0,1):
        a,c=case(image_radius=radius);objects.append((a,c))
    d=np.ones((1,2,2),complex)
    first=objects[0][0]['electronic'].build(d);second=objects[1][0]['electronic'].build(d)
    np.testing.assert_array_equal(first.values,second.values)
    assert first.receipt.energy==second.receipt.energy
    assert first.receipt.pair_images.cube_radius==0 and second.receipt.pair_images.cube_radius==1
    assert second.receipt.pair_images.omitted_density_mass_bound<first.receipt.pair_images.omitted_density_mass_bound*1e-10
    assert second.receipt.pair_images.energy_error_bound<1e-8


def test_individual_basis_image_covariance_with_complex_shifted_mesh():
    a,c=case(shifted=True,image_radius=3,real_radius=6,two_atoms=True)
    moved,mc=case(shifted=True,image_radius=3,real_radius=6,two_atoms=True,basis_image=True)
    r=core._manta.run_periodic_bupo_hf(**a);s=core._manta.run_periodic_bupo_hf(**moved)
    assert r.converged and s.converged
    u=np.ones((len(c['k']),2),complex);u[:,1]=np.exp(-1j*c['k']@c['cell'][:,0])
    for name in ('fock','density'):
        expected=u.conj()[None,:,:,None]*getattr(r,name)*u[None,:,None,:]
        np.testing.assert_allclose(getattr(s,name),expected,atol=4e-9,rtol=2e-10)
    assert np.max(abs(s.density.imag))>1e-3
    assert s.receipt.energy==pytest.approx(r.receipt.energy,abs=4e-11)
    ref=independent(moved,mc);np.testing.assert_allclose(s.fock,ref['build'](s.density)['fock'],atol=3e-11,rtol=2e-12)


def test_redundant_basis_retains_the_solver_projector_and_evaluated_state():
    args,c,r,ref,same=calculation(redundant=True,iterations=1)
    assert r.solver_memory.retained_rank==2 and r.coefficients.shape[-2:]==(3,2)
    np.testing.assert_allclose(r.density,ref['initial'],atol=3e-12,rtol=1e-12)
    np.testing.assert_allclose(r.fock,same['fock'],atol=3e-11,rtol=1e-12)
    assert r.receipt.energy==pytest.approx(same['energy'],abs=3e-11)


def as_fock_build(r,k):
    from vibeqc.fock_state import FockBuild,EnergyComponents,SourceProvenance
    from vibeqc.solvers.operators import OperatorFrame
    receipt=r.receipt;nk=r.plan.momenta;n=r.plan.n_basis;ns=r.plan.spins;frames=[]
    for i,s in enumerate(r.overlap):
        ev,u=np.linalg.eigh(s);rank=r.solver_memory.retained_rank
        frames.append(OperatorFrame(basis_id=receipt.overlap_identity,n=n,ordering='libint-shell-axial-v1',
            image_gauge=f'native-exp(+ikR):k-index={i}',overlap=s,X=u[:,-rank:]/np.sqrt(ev[-rank:])))
    energies=EnergyComponents(electronic=receipt.one_electron_energy+receipt.coulomb_energy+receipt.uncorrected_exchange_energy,
        nuclear=receipt.nuclear_energy,components=(('one_electron',receipt.one_electron_energy),('coulomb',receipt.coulomb_energy),
        ('uncorrected_exchange',receipt.uncorrected_exchange_energy)),corrections=(('exchange_zero_mode',receipt.exchange_correction_energy),))
    source=SourceProvenance.capture('manta',native_source_identity=receipt.source_identity,
        coulomb_algorithm='bupo',exchange_algorithm='exact-reciprocal',partition_identity=receipt.coulomb.partition.identity,
        electrostatic_identity=receipt.electrostatic.source_identity,accuracy_qualification=receipt.accuracy_qualification)
    labels=['restricted-total'] if ns==1 else ['alpha','beta']
    return FockBuild.from_dense(method='RHF' if ns==1 else 'UHF',generation=r.solver_snapshot.evaluated_iteration,
        source=source,energy=energies,densities=r.density.reshape(-1,n,n),focks=r.fock.reshape(-1,n,n),frames=frames*ns,
        spins=[s for s in labels for _ in k],kpoints=[tuple(x) for x in k]*ns,weights=[1./nk]*(nk*ns))


@pytest.mark.parametrize('spin,z',[('restricted',2),('unrestricted',1)])
def test_selected_output_and_shared_fock_keep_complete_method_energy(monkeypatch,spin,z):
    from vibeqc import output
    a,c=case(spin=spin,z=z,shifted=True,q0='BvK-ewald',iterations=1)
    events=[];monkeypatch.setattr(output,'record',lambda *args,**kw:events.append(kw['integral_provenance']))
    budget=a.pop('budget');a['integral_engine']=manta.IntegralEngineOptions(backend='manta',budget=budget)
    r=manta.run_periodic_bupo_hf(**a);assert budget.other_live_bytes==0
    evidence=events[-1];json.dumps(evidence,allow_nan=False)
    assert evidence['electrostatic_accuracy_passed'] and not evidence['symmetry_reduced']
    assert evidence['accuracy_qualification']=='electrostatic-and-one-electron-images-at-evaluated-density;finite-exchange'
    state=as_fock_build(r,c['k']);assert state.energy.total==pytest.approx(r.receipt.energy,abs=3e-12)
    for channel,d,f in zip(state.channels,r.density.reshape(-1,2,2),r.fock.reshape(-1,2,2)):
        np.testing.assert_array_equal(channel.density,d)
        np.testing.assert_allclose(channel.operator.apply(np.eye(2,dtype=complex)),f,atol=1e-13,rtol=0)
        assert channel.operator.metadata.symmetry_block is None and channel.operator.metadata.kind=='fock'
    for name in ('density','fock','overlap','hcore','coefficients','orbital_energies'):
        x=getattr(r,name);copy=x.copy();x[:]=123;np.testing.assert_array_equal(getattr(r,name),copy)
