"""Full reciprocal Coulomb plus the separately admitted BvK Ewald seam."""
import itertools
import numpy as np
import pytest
from scipy.linalg import eigh

from vibeqc import manta, _vibeqc_core as core
from vibeqc.bipole_fock_ewald import probe_charge_madelung_supercell
from tests.test_manta_periodic_rhf import _case, _components, _nuclear_reference


@pytest.mark.parametrize('mesh,shift', [((1,1,1),(0,0,0)),((2,1,1),(0,0,0)),((2,1,1),(1,0,0))])
def test_corrected_rhf_matches_independent_scf_and_expected_seam(mesh,shift):
    case,k=_case(mesh,shift)
    strict=core._manta.run_periodic_rhf(**case)
    case['options'].exchange_q0='BvK-ewald'
    case['options'].scf.maximum_iterations=100
    result=core._manta.run_periodic_rhf(**case)
    assert result.converged and result.receipt.confirmed
    xi=probe_charge_madelung_supercell(case['system'],mesh,precision=1e-14)
    assert result.receipt.madelung==pytest.approx(xi,abs=3e-12)
    s,h,uncorrected=_components(case,k)
    def fock(d): return uncorrected(d)-.5*xi*(s@d@s)
    def occupy(f):
        c=np.array([eigh(fp,sp)[1][:,:1] for fp,sp in zip(f,s)])
        return 2*c@c.conj().transpose(0,2,1)
    d=occupy(h)
    for _ in range(150):
        updated=occupy(fock(d))
        if np.max(abs(updated-d))<1e-12: break
        d=updated
    else: pytest.fail('independent corrected SCF did not converge')
    nuclear,_=_nuclear_reference(case)
    energy=sum(nuclear)+.5/len(k)*np.einsum('kij,kji->',d,h+fock(d)).real
    np.testing.assert_allclose(result.density,d,atol=3e-9)
    np.testing.assert_allclose(result.fock,fock(result.density),atol=4e-12)
    assert result.receipt.energy==pytest.approx(energy,abs=4e-11)
    # At an idempotent RHF solution the seam leaves density unchanged,
    # lowers occupied levels by xi, and shifts E by -Ne*xi/2. No extra Nk.
    np.testing.assert_allclose(result.density,strict.density,atol=3e-9)
    np.testing.assert_allclose(result.orbital_energies[:,0],strict.orbital_energies[:,0]-xi,atol=2e-9)
    np.testing.assert_allclose(result.orbital_energies[:,1],strict.orbital_energies[:,1],atol=2e-9)
    assert result.receipt.energy-strict.receipt.energy==pytest.approx(-xi,abs=4e-11)
    r=result.receipt
    correction=-.25*xi/len(k)*np.einsum('kij,kji->',result.density,s@result.density@s).real
    assert r.exchange_correction_energy==pytest.approx(correction,abs=2e-12)
    assert r.exchange_energy==pytest.approx(r.uncorrected_exchange_energy+correction,abs=2e-12)
    assert r.source_identity!=strict.receipt.source_identity
    assert r.final_density_identity!=strict.receipt.final_density_identity
    assert r.final_fock_identity!=strict.receipt.final_fock_identity
    assert r.exchange_q0=='BvK-ewald' and len(r.exchange_correction_identity)==64
    assert r.madelung_real_vectors<=result.plan.madelung.real_candidates
    assert r.madelung_reciprocal_vectors<=result.plan.madelung.reciprocal_candidates


@pytest.mark.parametrize('alpha',[.2,.6,1.1])
def test_madelung_alpha_invariance_and_last_evaluated_epoch(alpha):
    case,k=_case((2,1,1),(1,0,0)); case['options'].scf.maximum_iterations=1
    strict=core._manta.run_periodic_rhf(**case)
    case['options'].exchange_q0='BvK-ewald'; case['options'].madelung_alpha=alpha
    result=core._manta.run_periodic_rhf(**case)
    xi=probe_charge_madelung_supercell(case['system'],(2,1,1),alpha=alpha,precision=1e-14)
    assert not result.converged and not result.receipt.confirmed
    assert result.receipt.madelung==pytest.approx(xi,abs=3e-12)
    np.testing.assert_array_equal(result.density,strict.density)
    np.testing.assert_allclose(result.fock-strict.fock,-.5*xi*(strict.overlap@strict.density@strict.overlap),atol=3e-12)
    assert result.plan.madelung.alpha==alpha


@pytest.mark.parametrize('field,planned',[('memory_bytes','modeled_peak_bytes'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_correction_aggregate_budget_before_vector_payload(field,planned):
    case,_=_case(); case['options'].exchange_q0='BvK-ewald'
    p=manta.plan_periodic_rhf(case['basis'],case['system'],case['mesh'],len(case['reciprocal_vectors']),
        max(np.diff(case['transfer_offsets'])),len(case['translations']),case['options'],case['budget'])
    setattr(case['budget'],field,getattr(p,planned))
    assert core._manta.run_periodic_rhf(**case).converged
    setattr(case['budget'],field,getattr(p,planned)-1); case['reciprocal_vectors'][:]=np.nan
    with pytest.raises((ValueError,RuntimeError),match='budget'):
        core._manta.run_periodic_rhf(**case)


@pytest.mark.parametrize('field,value', [('exchange_q0','ewald'),('madelung_tolerance',0.),('madelung_tolerance',1.),
    ('madelung_tolerance',float('nan')),('madelung_alpha',0.),('madelung_alpha',float('inf')),('madelung_alpha',1e9)])
def test_correction_refuses_invalid_or_unbounded_settings(field,value):
    case,_=_case(); case['options'].exchange_q0='BvK-ewald'; setattr(case['options'],field,value)
    case['exchange_vectors'][:]=np.nan
    with pytest.raises((ValueError,RuntimeError),match='Madelung|exchange_q0'):
        core._manta.run_periodic_rhf(**case)


def test_selected_output_records_executed_correction_and_owned_result(monkeypatch):
    import vibeqc.output as output
    from vibeqc.periodic.exchange_convention import executed_exchange_q0
    case,_=_case(); case['options'].exchange_q0='BvK-ewald'
    engine=manta.IntegralEngineOptions(backend='manta',budget=case.pop('budget'))
    records=[]; monkeypatch.setattr(output,'record',lambda *args,**kw:records.append(kw))
    result=manta.run_periodic_rhf(**case,integral_engine=engine)
    evidence=records[-1]['integral_provenance']
    assert result.converged and evidence['exchange_q0']=='BvK-ewald'
    assert 'manta_periodic_madelung' in evidence['citation_methods']
    assert evidence['madelung_plan']['alpha']==result.plan.madelung.alpha
    before=result.fock.copy(); result.fock[:]=0
    case['options'].exchange_q0='strict-zero-mode'; case['options'].madelung_alpha=.9
    np.testing.assert_array_equal(result.fock,before)
    assert result.receipt.exchange_q0=='BvK-ewald'
    assert executed_exchange_q0(result,None)=='BvK-ewald'
    assert engine.budget.other_live_bytes==0


def _periodic_reference_case(mesh,shift=(0,0,0),span=12):
    case,k=_case(mesh,shift); system=case['system']; o=case['options']
    o.exchange_q0='BvK-ewald'; o.scf.maximum_iterations=100
    o.reciprocal_batch=128; o.pair_cutoff_bohr=9.; o.nuclear_cutoff_bohr=12.
    labels=np.array(list(itertools.product(range(-span,span+1),repeat=3)),dtype=float)
    base=labels@(2*np.pi*np.linalg.inv(system.lattice))
    case['reciprocal_vectors']=base[np.any(labels,axis=1)].copy()
    chunks=[base+t-s for t in k for s in k]; chunks=[q[np.any(q!=0.,axis=1)] for q in chunks]
    case['exchange_vectors']=np.concatenate(chunks)
    case['transfer_offsets']=np.r_[0,np.cumsum([len(q) for q in chunks])].astype(np.int64)
    images=np.array(list(itertools.product(range(-1,2),repeat=3)),dtype=float)
    case['translations']=np.ascontiguousarray(images@system.lattice.T)
    return case,k


@pytest.mark.parametrize('mesh,shift',[((1,1,1),(0,0,0)),((2,1,1),(0,0,0)),((2,1,1),(1,0,0))])
def test_complete_periodic_rhf_against_optional_pyscf_aft(mesh,shift):
    pytest.importorskip('pyscf.pbc')
    from pyscf.pbc import gto,df,scf,tools
    case,k=_periodic_reference_case(mesh,shift); result=core._manta.run_periodic_rhf(**case)
    system=case['system']; cell=gto.Cell()
    cell.atom=[('H',a.xyz) for a in system.unit_cell]; cell.a=system.lattice.T
    cell.unit='Bohr'; cell.basis={'H':[[0,[.8,1.]]]}; cell.precision=1e-11; cell.verbose=0; cell.build()
    mf=scf.KRHF(cell,k,exxdiv='ewald'); mf.with_df=df.AFTDF(cell,k); mf.with_df.mesh=[25]*3
    mf.conv_tol=1e-11; mf.max_cycle=100; energy=mf.kernel()
    assert mf.converged and result.converged
    assert result.receipt.madelung==pytest.approx(tools.pbc.madelung(cell,k),abs=3e-12)
    assert result.receipt.nuclear_energy==pytest.approx(cell.energy_nuc(),abs=3e-11)
    assert result.receipt.energy==pytest.approx(energy,abs=3e-10)
    np.testing.assert_allclose(result.overlap,mf.get_ovlp(),atol=3e-11)
    np.testing.assert_allclose(result.hcore,mf.get_hcore(),atol=3e-10)
    np.testing.assert_allclose(result.density,mf.make_rdm1(),atol=2e-8)
    np.testing.assert_allclose(result.fock,mf.get_fock(dm=result.density),atol=3e-10)


def test_corrected_source_reciprocal_and_nuclear_split_convergence():
    coarse,_=_periodic_reference_case((2,1,1),span=8)
    fine,_=_periodic_reference_case((2,1,1),span=12)
    a=core._manta.run_periodic_rhf(**coarse); b=core._manta.run_periodic_rhf(**fine)
    fine['options'].alpha=.9
    c=core._manta.run_periodic_rhf(**fine)
    assert a.converged and b.converged and c.converged
    np.testing.assert_allclose([a.receipt.energy,c.receipt.energy],b.receipt.energy,atol=3e-10,rtol=0)
    assert a.receipt.source_identity!=b.receipt.source_identity!=c.receipt.source_identity
    assert a.receipt.exchange_correction_identity==b.receipt.exchange_correction_identity==c.receipt.exchange_correction_identity


def test_probe_depends_on_full_mesh_and_lattice_but_not_atoms_or_shift():
    import vibeqc as vq
    case,_=_case(); case['options'].exchange_q0='BvK-ewald'; case['options'].scf.maximum_iterations=1
    gamma=core._manta.run_periodic_rhf(**case)
    system=vq.PeriodicSystem(3,case['system'].lattice,[vq.Atom(1,[-.6,.1,0.]),vq.Atom(1,[.8,0.,.1])])
    altered=core._manta.run_periodic_rhf(**dict(case,system=system,basis=vq.BasisSet(system.unit_cell_molecule(),'sto-3g')))
    assert altered.receipt.exchange_correction_identity==gamma.receipt.exchange_correction_identity
    assert altered.receipt.source_identity!=gamma.receipt.source_identity
    probes=[]
    for shift in [(0,0,0),(1,1,1)]:
        doubled,_=_case((2,2,2),shift)
        doubled['options'].exchange_q0='BvK-ewald'; doubled['options'].scf.maximum_iterations=1
        probes.append(core._manta.run_periodic_rhf(**doubled))
    assert probes[0].receipt.madelung==pytest.approx(gamma.receipt.madelung/2,abs=2e-12)
    assert probes[0].receipt.exchange_correction_identity==probes[1].receipt.exchange_correction_identity
