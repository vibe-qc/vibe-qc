"""Complete finite-source MANTA RHF: independent SCF and terminal contracts."""
import itertools
import math

import numpy as np
import pytest
from scipy.linalg import eigh

import vibeqc as vq
from vibeqc import manta,_vibeqc_core as core
from tests.test_manta_periodic_exchange import _quadrature_bloch


def _case(mesh_shape=(1,1,1),shift=(0,0,0)):
    lattice=np.array([[6.,.2,.1],[0.,6.2,.15],[0.,0.,6.4]])
    atoms=[vq.Atom(1,[-.7,0.,0.]),vq.Atom(1,[.7,0.,0.])]
    system=vq.PeriodicSystem(3,lattice,atoms); shells=[]
    for i,atom in enumerate(atoms):
        s=core.ShellInfo(); s.atom_index=i; s.origin=atom.xyz; s.l=0; s.pure=True
        s.exponents=[.8]; s.coefficients=[1.]; shells.append(s)
    basis=vq.BasisSet(system.unit_cell_molecule(),shells,'manta-periodic-rhf-test',False)
    mesh=manta.RegularKMesh(mesh_shape,shift)
    fractions=np.array([mesh.fractional(mesh.address(i)) for i in range(len(mesh))])
    k=fractions@(2*np.pi*np.linalg.inv(lattice))
    labels=np.array(list(itertools.product(range(-1,2),repeat=3)),dtype=float)
    base=labels@(2*np.pi*np.linalg.inv(lattice)); g=base[np.any(base!=0.,axis=1)].copy()
    chunks=[base+t-s for t in k for s in k]; chunks=[q[np.any(q!=0.,axis=1)] for q in chunks]
    q=np.ascontiguousarray(np.concatenate(chunks)); offsets=np.r_[0,np.cumsum([len(a) for a in chunks])].astype(np.int64)
    options=manta.PeriodicRHFOptions(); options.alpha=.6; options.nuclear_cutoff_bohr=4.; options.reciprocal_batch=16
    options.pair_cutoff_bohr=3.; options.scf.maximum_iterations=40
    budget=manta.ResourceBudget(); budget.kernel_terms=10**15; budget.scalar_terms=10**15
    return dict(basis=basis,system=system,mesh=mesh,reciprocal_vectors=g,exchange_vectors=q,
        transfer_offsets=offsets,translations=np.zeros((1,3)),options=options,budget=budget),k


def _components(case,k):
    b=case['basis']; system=case['system']; g=case['reciprocal_vectors']; cells=case['translations']; o=case['options']
    volume=abs(np.linalg.det(system.lattice)); alpha=o.alpha
    s=np.asarray(vq.compute_overlap(b)); t=np.asarray(vq.compute_kinetic(b))
    # No nonzero AO images are in this fixture. The shared physical nuclear
    # sphere contains the two unit-cell atoms and no translated nuclei.
    lattice_options=vq.LatticeSumOptions(); lattice_options.cutoff_bohr=3.
    lattice_options.nuclear_cutoff_bohr=o.nuclear_cutoff_bohr
    lattice_options.pair_complete_1e=True
    short=np.asarray(vq.compute_nuclear_erfc_lattice(b,system,alpha,lattice_options).blocks)[0]
    fourier=_quadrature_bloch(b,g,cells,np.zeros(3),None)
    positions=np.array([a.xyz for a in system.unit_cell]); charges=np.array([a.Z for a in system.unit_cell])
    structure=np.exp(-1j*g@positions.T)@charges; norm=np.sum(g*g,axis=1)
    potential=-np.einsum('g,g,gij->ij',4*np.pi/volume*np.exp(-norm/(4*alpha**2))/norm,structure,fourier.conj())
    h=t+short+potential+np.pi*sum(charges)/(alpha**2*volume)*s
    h=np.stack([h]*len(k)); overlap=np.stack([s]*len(k)).astype(complex)
    weights=np.ones(len(k))/len(k)
    fj=np.stack([_quadrature_bloch(b,g,cells,-point,None) for point in k])
    exchange=[]; q=case['exchange_vectors']; offsets=case['transfer_offsets']
    for target,kt in enumerate(k):
        channels=[]
        for source in range(len(k)):
            pair=target*len(k)+source; vectors=q[offsets[pair]:offsets[pair+1]]
            channels.append((_quadrature_bloch(b,vectors,cells,kt,None),weights[source]*4*np.pi/volume/np.sum(vectors*vectors,axis=1)))
        exchange.append(channels)
    def build(d):
        rho=np.einsum('k,kij,kgij->g',weights,d,fj)
        j=np.einsum('g,g,kgij->kij',4*np.pi/volume/norm,rho,fj.conj())
        response=np.zeros_like(d)
        for target in range(len(k)):
            for source in range(len(k)):
                fourier,factor=exchange[target][source]
                response[target]+=np.einsum('g,gai,ab,gbj->ij',factor,fourier.conj(),d[source],fourier)
        return h+j-.5*response
    return overlap,h,build


def _nuclear_reference(case):
    system=case['system']; a=case['options'].alpha; radius=case['options'].nuclear_cutoff_bohr
    lattice=np.asarray(system.lattice); positions=np.array([atom.xyz for atom in system.unit_cell]); charges=np.array([atom.Z for atom in system.unit_cell])
    real=0.; count=0
    for i,p in enumerate(positions):
        for j,q in enumerate(positions):
            for label in itertools.product(range(-3,4),repeat=3):
                if i==j and label==(0,0,0): continue
                distance=np.linalg.norm(q+lattice@label-p)
                if distance<=radius+1e-12*(1+radius):
                    real+=.5*charges[i]*charges[j]*math.erfc(a*distance)/distance; count+=1
    g=case['reciprocal_vectors']; norm=np.sum(g*g,axis=1); volume=abs(np.linalg.det(lattice))
    structure=np.exp(1j*g@positions.T)@charges
    reciprocal=2*np.pi/volume*np.sum(abs(structure)**2*np.exp(-norm/(4*a*a))/norm)
    self_term=-a/np.sqrt(np.pi)*sum(charges**2); background=-np.pi*sum(charges)**2/(2*a*a*volume)
    return (real,reciprocal,self_term,background),count


@pytest.mark.parametrize('mesh,shift',[((1,1,1),(0,0,0)),((2,1,1),(0,0,0)),((2,1,1),(1,0,0))])
def test_complete_rhf_matches_independent_scf_and_same_density_components(mesh,shift):
    case,k=_case(mesh,shift); result=core._manta.run_periodic_rhf(**case)
    assert result.converged and result.receipt.confirmed
    s,h,build=_components(case,k)
    np.testing.assert_allclose(result.overlap,s,atol=2e-13)
    np.testing.assert_allclose(result.hcore,h,atol=3e-12)
    coefficients=[eigh(hp,sp)[1][:,:1] for hp,sp in zip(h,s)]
    density=np.array([2*c@c.conj().T for c in coefficients])
    for _ in range(100):
        f=build(density); coefficients=[eigh(fp,sp)[1][:,:1] for fp,sp in zip(f,s)]
        updated=np.array([2*c@c.conj().T for c in coefficients])
        if np.max(abs(updated-density))<1e-12: break
        density=updated
    else: pytest.fail('Independent finite-source RHF did not converge')
    nuclear,count=_nuclear_reference(case)
    expected=sum(nuclear)+.5/len(k)*np.einsum('kij,kji->',density,h+build(density)).real
    assert result.receipt.energy==pytest.approx(expected,abs=3e-11)
    np.testing.assert_allclose(result.density,density,atol=3e-10)
    np.testing.assert_allclose(result.fock,build(result.density),atol=3e-12)
    np.testing.assert_allclose([result.receipt.nuclear_real,result.receipt.nuclear_reciprocal,result.receipt.nuclear_self,result.receipt.nuclear_background],nuclear,atol=2e-14)
    assert result.receipt.nuclear_pair_images==count
    assert result.receipt.fock_calls==result.solver_snapshot.fock_calls+1
    assert result.receipt.shell_calls<=result.plan.shell_calls
    assert result.receipt.primitive_pairs<=result.plan.primitive_pairs
    assert result.receipt.confirmation_fock_error<1e-11 and result.receipt.confirmation_energy_error<1e-11
    assert np.max(abs(np.einsum('kij,kji->k',result.density,result.overlap)-2))<1e-10


def test_iteration_limit_retains_evaluated_state_without_confirmation():
    case,k=_case(); case['options'].scf.maximum_iterations=1
    result=core._manta.run_periodic_rhf(**case)
    assert not result.converged and not result.receipt.confirmed
    assert result.receipt.fock_calls==1 and result.solver_snapshot.evaluated_iteration==1
    _,_,build=_components(case,k)
    np.testing.assert_allclose(result.fock,build(result.density),atol=3e-12)


def test_iteration_trace_matches_independently_capped_evaluations_and_owns_copies():
    case,k=_case((2,1,1),(1,0,0))
    shells=case['basis'].shells(); shells[0].exponents=[.4]
    case['basis']=vq.BasisSet(case['system'].unit_cell_molecule(),shells,'manta-trace',False)
    case['options'].scf.maximum_iterations=4
    result=core._manta.run_periodic_rhf(**case)
    history=result.scf_history
    assert len(history)==result.solver_snapshot.evaluated_iteration==4
    assert not history[0].has_energy_change
    for iteration,step in enumerate(history,1):
        case['options'].scf.maximum_iterations=iteration
        stopped=core._manta.run_periodic_rhf(**case)
        assert step.evaluated_iteration==iteration
        assert step.fock_calls==iteration
        for field in ('energy_per_cell','commutator_frobenius_rms',
                      'maximum_density_closure_frobenius','global_band_gap'):
            assert getattr(step,field)==getattr(stopped.solver_snapshot,field)
        if iteration>1:
            assert step.has_energy_change
            assert step.energy_change==step.energy_per_cell-history[iteration-2].energy_per_cell
    with pytest.raises(AttributeError):
        history[0].energy_per_cell=0.
    history.clear()
    assert len(result.scf_history)==4
    copied=result.scf_history[-1]
    del result
    assert copied.evaluated_iteration==4


def test_converged_trace_excludes_confirmation_and_reserves_maximum_before_payloads():
    case,_=_case()
    result=core._manta.run_periodic_rhf(**case)
    assert result.scf_history[-1].converged and result.converged
    assert len(result.scf_history)==result.solver_snapshot.fock_calls==result.receipt.fock_calls-1
    p=result.plan
    assert p.history_capacity==case['options'].scf.maximum_iterations
    assert p.history_live_bytes>=p.history_capacity*512
    case['options'].scf.maximum_iterations*=2
    changed=manta.plan_periodic_rhf(case['basis'],case['system'],case['mesh'],len(case['reciprocal_vectors']),
        int(np.max(np.diff(case['transfer_offsets']))),len(case['translations']),case['options'],case['budget'])
    assert changed.history_live_bytes==2*p.history_live_bytes
    assert changed.modeled_peak_bytes-p.modeled_peak_bytes==p.history_live_bytes
    case['budget'].memory_bytes=changed.modeled_peak_bytes-changed.history_live_bytes
    case['exchange_vectors'][:]=np.nan
    with pytest.raises((ValueError,RuntimeError),match='memory budget'):
        core._manta.run_periodic_rhf(**case)


@pytest.mark.parametrize('field,plan_field',[('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),
    ('primitive_quartets','primitive_pairs'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_entire_scf_resource_bound_is_admitted_before_vector_snapshot(field,plan_field):
    case,_=_case((2,1,1))
    p=manta.plan_periodic_rhf(case['basis'],case['system'],case['mesh'],len(case['reciprocal_vectors']),
        int(np.max(np.diff(case['transfer_offsets']))),len(case['translations']),case['options'],case['budget'])
    setattr(case['budget'],field,getattr(p,plan_field))
    assert core._manta.run_periodic_rhf(**case).converged
    setattr(case['budget'],field,getattr(p,plan_field)-1); case['exchange_vectors'][:]=np.nan
    with pytest.raises((ValueError,RuntimeError),match='budget'):
        core._manta.run_periodic_rhf(**case)


def test_nuclear_energy_real_images_and_individual_image_relabeling():
    case,_=_case(); case['options'].nuclear_cutoff_bohr=7.
    result=core._manta.run_periodic_rhf(**case); reference,count=_nuclear_reference(case)
    assert count>2 and result.receipt.nuclear_pair_images==count
    assert result.receipt.nuclear_energy==pytest.approx(sum(reference),abs=2e-14)
    atoms=list(case['system'].unit_cell); atom=atoms[0]
    atoms[0]=vq.Atom(atom.Z,np.array(atom.xyz)+np.asarray(case['system'].lattice)[:,0])
    changed=dict(case,system=vq.PeriodicSystem(3,case['system'].lattice,atoms))
    relabeled=core._manta.run_periodic_rhf(**changed)
    assert relabeled.converged
    assert relabeled.receipt.nuclear_energy==pytest.approx(result.receipt.nuclear_energy,abs=2e-14)
    assert relabeled.receipt.energy==pytest.approx(result.receipt.energy,abs=3e-12)
    np.testing.assert_allclose(relabeled.hcore,result.hcore,atol=3e-12)
    np.testing.assert_allclose(relabeled.density,result.density,atol=3e-11)


def test_source_batch_identity_owned_results_and_no_options_mutation():
    case,_=_case((2,1,1)); original=core._manta.run_periodic_rhf(**case)
    density=original.density.copy(); hcore=original.hcore.copy(); coefficients=original.coefficients.copy()
    case['options'].reciprocal_batch=1
    narrow=core._manta.run_periodic_rhf(**case)
    assert narrow.receipt.source_identity==original.receipt.source_identity
    np.testing.assert_allclose(narrow.density,density,atol=1e-12)
    assert narrow.receipt.energy==pytest.approx(original.receipt.energy,abs=2e-13)
    case['options'].alpha=.9; case['translations'][:]=17.; case['reciprocal_vectors'][:]=0.
    for array in (original.density,original.fock,original.coefficients,original.orbital_energies,original.hcore,original.overlap): array[:]=0.
    np.testing.assert_array_equal(original.density,density); np.testing.assert_array_equal(original.hcore,hcore)
    np.testing.assert_array_equal(original.coefficients,coefficients)
    with pytest.raises(AttributeError): original.receipt.confirmed=True


def test_gap_failure_and_unsupported_solver_controls_never_claim_convergence():
    case,_=_case(); case['options'].scf.minimum_band_gap_hartree=100.
    result=core._manta.run_periodic_rhf(**case)
    assert not result.converged and not result.receipt.confirmed
    assert result.solver_snapshot.status==core._BoundedPeriodicRHFStatus.NON_INSULATING
    case,_=_case(); case['options'].scf.maximum_diis_history=4
    with pytest.raises(ValueError,match='DIIS'): core._manta.run_periodic_rhf(**case)
    case,_=_case(); case['options'].scf.maximum_iterations=0
    with pytest.raises(ValueError,match='iteration'): core._manta.run_periodic_rhf(**case)


def test_physical_domain_and_kernel_support_refusals():
    for field,value in [('dim',2),('charge',1),('multiplicity',3)]:
        case,_=_case(); setattr(case['system'],field,value)
        with pytest.raises(ValueError,match='neutral'): core._manta.run_periodic_rhf(**case)
    case,_=_case(); atoms=list(case['system'].unit_cell); atoms[0]=vq.Atom(1,[-.6,0.,0.])
    case['system']=vq.PeriodicSystem(3,case['system'].lattice,atoms)
    with pytest.raises(ValueError,match='centers'): core._manta.run_periodic_rhf(**case)
    case,_=_case(); case['exchange_vectors'][0]=0.
    with pytest.raises(ValueError,match='Q=0'): core._manta.run_periodic_rhf(**case)
    case,_=_case(); case['reciprocal_vectors'][0]=0.
    with pytest.raises(ValueError,match='G=0'): core._manta.run_periodic_rhf(**case)


def test_selected_route_emits_execution_and_unconverged_status_without_mutating_budget(monkeypatch):
    from vibeqc import output
    case,_=_case(); budget=case.pop('budget'); engine=vq.IntegralEngineOptions(backend='manta',budget=budget)
    records=[]; monkeypatch.setattr(output,'record',lambda *args,**kwargs:records.append(kwargs))
    result=manta.run_periodic_rhf(**case,integral_engine=engine)
    assert result.converged
    evidence=records[-1]['integral_provenance']
    assert evidence['confirmed'] and evidence['converged'] and evidence['status']=='CONVERGED'
    assert evidence['source_identity']==result.receipt.source_identity
    assert evidence['citation_methods']==['manta_periodic_rhf']
    assert engine.budget.other_live_bytes==0
    case['options'].scf.maximum_iterations=1
    limited=manta.run_periodic_rhf(**case,integral_engine=engine)
    assert not limited.converged and not records[-1]['integral_provenance']['confirmed']
    assert records[-1]['integral_provenance']['status']=='ITERATION_LIMIT'
    engine.budget.memory_bytes=1
    with pytest.raises(ValueError,match='budget'): manta.run_periodic_rhf(**case,integral_engine=engine)
    assert len(records)==2


def test_overlap_rank_truncation_has_the_same_energy_as_the_independent_basis():
    case,_=_case(); system=vq.PeriodicSystem(3,case['system'].lattice,[vq.Atom(2,[0.,0.,0.])])
    def basis(duplicate):
        shells=[]
        for l in ([0,0,1] if duplicate else [0,1]):
            s=core.ShellInfo(); s.atom_index=0; s.origin=[0.,0.,0.]; s.l=l; s.pure=True
            s.exponents=[.9]; s.coefficients=[1.]; shells.append(s)
        return vq.BasisSet(system.unit_cell_molecule(),shells,'manta-rhf-rank-test',False)
    ordinary=core._manta.run_periodic_rhf(**dict(case,system=system,basis=basis(False)))
    redundant=core._manta.run_periodic_rhf(**dict(case,system=system,basis=basis(True)))
    assert ordinary.converged and redundant.converged
    assert redundant.solver_memory.retained_rank==ordinary.solver_memory.retained_rank==4
    assert redundant.coefficients.shape==(1,5,4)
    assert redundant.receipt.energy==pytest.approx(ordinary.receipt.energy,abs=4e-12)
    assert redundant.solver_snapshot.maximum_electron_count_error<1e-10
    case['options'].scf.overlap_rank_relative_floor=.99
    with pytest.raises((ValueError,RuntimeError),match='rank|occupied|virtual'):
        core._manta.run_periodic_rhf(**dict(case,system=system,basis=basis(True)))


def test_truncated_indefinite_overlap_is_rejected_instead_of_repaired():
    case,_=_case(); vector=np.asarray(case['system'].lattice)[:,0]
    case['translations']=np.array([vector,-vector]); case['options'].pair_cutoff_bohr=None
    with pytest.raises((ValueError,RuntimeError),match='negative|positive|overlap'):
        core._manta.run_periodic_rhf(**case)


def test_failed_terminal_confirmation_cannot_report_convergence():
    case,_=_case()
    case['options'].confirmation_fock_tolerance=1e-30
    result=core._manta.run_periodic_rhf(**case)
    assert result.solver_snapshot.status.name=='CONVERGED'
    assert result.receipt.confirmation_fock_error>case['options'].confirmation_fock_tolerance
    assert not result.receipt.confirmed and not result.converged
