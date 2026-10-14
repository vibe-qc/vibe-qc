"""Immutable finite-source spin Fock snapshots and their variational energy."""
import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta
from tests.test_manta_periodic_exchange import _basis, _density, _transfers, _quadrature_bloch, LABELS, LATTICE, CELLS, MOMENTA, WEIGHTS


G = np.ascontiguousarray(LABELS[1:]@(2*np.pi*np.linalg.inv(LATTICE)))


def _case(pure=True, kernel='coulomb', omega=0., scale=1.):
    basis = _basis((0, 1), pure=pure); density = _density(basis)
    hcore = np.ascontiguousarray(_density(basis, 1)[0]*.17)
    q, offsets = _transfers()
    if kernel=='erfc' and omega>0:
        base = LABELS@(2*np.pi*np.linalg.inv(LATTICE))
        q = np.concatenate([base+t-s for t in MOMENTA for s in MOMENTA])
        offsets = np.arange(10, dtype=np.int64)*len(LABELS)
    return dict(basis=basis,lattice=LATTICE.copy(),hartree_vectors=G.copy(),exchange_vectors=q,
        transfer_offsets=offsets,translations=CELLS.copy(),momenta=MOMENTA.copy(),weights=WEIGHTS.copy(),
        hcore=hcore,spin_densities=density,reciprocal_batch=4,kernel=kernel,omega=omega,exchange_scale=scale)


def _reference(case):
    b,g,k,d,h=case['basis'],case['hartree_vectors'],case['momenta'],case['spin_densities'],case['hcore']
    cells,w=case['translations'],case['weights']; volume=abs(np.linalg.det(case['lattice']))
    negative = np.stack([_quadrature_bloch(b,g,cells,-momentum,None) for momentum in k])
    rho = np.einsum('k,kmn,kgmn->g', w, d.sum(axis=0), negative)
    j = np.einsum('g,g,kgmn->kmn',4*np.pi/volume/np.sum(g*g,axis=1),rho,negative.conj())
    exchange=np.zeros_like(d); q,o=case['exchange_vectors'],case['transfer_offsets']
    for t,kt in enumerate(k):
        for s in range(len(k)):
            vectors=q[o[t*len(k)+s]:o[t*len(k)+s+1]]
            q2=np.sum(vectors*vectors,axis=1); coefficient=np.zeros(len(q2)); nz=q2>0
            coefficient[nz]=4*np.pi/q2[nz]
            if case['kernel']=='erf': coefficient[nz]*=np.exp(-q2[nz]/(4*case['omega']**2))
            if case['kernel']=='erfc':
                coefficient[nz]*=-np.expm1(-q2[nz]/(4*case['omega']**2))
                coefficient[~nz]=np.pi/case['omega']**2
            fourier=_quadrature_bloch(b,vectors,cells,kt,None)
            exchange[:,t]+=w[s]/volume*np.einsum('g,gai,dab,gbj->dij',coefficient,fourier.conj(),d[:,s],fourier)
    f=h[None]+j[None]-case['exchange_scale']*exchange
    one=np.einsum('k,kmn,knm->',w,d.sum(axis=0),h).real
    ej=.5*np.einsum('k,kmn,knm->',w,d.sum(axis=0),j).real
    ek=-.5*case['exchange_scale']*np.einsum('k,skmn,sknm->',w,d,exchange).real
    return f,j,exchange,(one,ej,ek)


@pytest.mark.parametrize('pure', [False, True])
@pytest.mark.parametrize('kernel,omega,scale', [('coulomb',0.,1.),('erf',.7,.25),('erfc',.7,.35)])
def test_spin_fock_and_all_energy_components_against_quadrature(pure,kernel,omega,scale):
    case=_case(pure,kernel,omega,scale); result=manta.compute_periodic_fock(**case)
    f,j,k,energy=_reference(case)
    np.testing.assert_allclose(result.values,f,atol=2e-11,rtol=2e-11)
    np.testing.assert_allclose(result.hartree_values,j,atol=3e-12,rtol=2e-11)
    np.testing.assert_allclose(result.exchange_values,k,atol=2e-11,rtol=2e-11)
    r=result.receipt
    np.testing.assert_allclose([r.one_electron_energy,r.coulomb_energy,r.exchange_energy],energy,atol=4e-11,rtol=2e-11)
    assert r.electronic_energy==pytest.approx(sum(energy),abs=5e-11,rel=2e-11)
    assert result.hartree_receipt.shell_calls+result.exchange_receipt.shell_calls<=result.plan.shell_calls
    assert result.hartree_receipt.primitive_pairs+result.exchange_receipt.primitive_pairs<=result.plan.primitive_pairs
    assert r.fock_hermiticity_residual<1e-11


@pytest.mark.parametrize('kernel,omega', [('coulomb',0.),('erf',.8),('erfc',.8)])
def test_electronic_energy_derivative_matches_each_spin_k_fock(kernel,omega):
    case=_case(kernel=kernel,omega=omega,scale=.6); result=manta.compute_periodic_fock(**case)
    rng=np.random.default_rng(643); raw=rng.normal(size=case['spin_densities'].shape)+1j*rng.normal(size=case['spin_densities'].shape)
    direction=raw+raw.conj().transpose(0,1,3,2); h=1e-5
    for spin,point in [(0,1),(1,2)]:
        delta=np.zeros_like(direction); delta[spin,point]=direction[spin,point]
        a=manta.compute_periodic_fock(**dict(case,spin_densities=case['spin_densities']+h*delta))
        b=manta.compute_periodic_fock(**dict(case,spin_densities=case['spin_densities']-h*delta))
        actual=(a.receipt.electronic_energy-b.receipt.electronic_energy)/(2*h)
        expected=WEIGHTS[point]*np.trace(delta[spin,point]@result.values[spin,point]).real
        assert actual==pytest.approx(expected,rel=2e-8,abs=3e-8)
        # Hartree couples both spins across the whole mesh; K remains spin-diagonal.
        np.testing.assert_array_equal(a.exchange_values[1-spin],b.exchange_values[1-spin])
        assert all(np.linalg.norm(block)>1e-9 for block in (a.values-b.values)[1-spin])


def test_restricted_density_factors_and_spin_interchange():
    case=_case(); d=case['spin_densities'][0].copy()
    restricted=manta.compute_periodic_fock(**dict(case,spin_densities=np.stack([d/2,d/2])))
    j=manta.compute_periodic_hartree(case['basis'],LATTICE,G,CELLS,MOMENTA,WEIGHTS,d,4)
    k=manta.compute_periodic_exchange(case['basis'],LATTICE,case['exchange_vectors'],case['transfer_offsets'],CELLS,MOMENTA,WEIGHTS,d[None],4)
    expected=case['hcore']+j.values-.5*k.values[0]
    np.testing.assert_allclose(restricted.values,np.stack([expected,expected]),atol=2e-12)
    assert restricted.receipt.exchange_energy==pytest.approx(.5*k.receipt.energies[0],abs=1e-12)
    result=manta.compute_periodic_fock(**case)
    swapped=manta.compute_periodic_fock(**dict(case,spin_densities=case['spin_densities'][::-1].copy()))
    np.testing.assert_allclose(swapped.values,result.values[::-1],atol=2e-12)
    assert swapped.receipt.electronic_energy==pytest.approx(result.receipt.electronic_energy,abs=1e-12)


def test_snapshot_owns_matrices_and_binds_hcore_density_kernel_and_scale():
    case=_case(); result=manta.compute_periodic_fock(**case); expected=result.values.copy()
    changed_h=case['hcore'].copy(); changed_h[:,0,0]+=.01
    changed=manta.compute_periodic_fock(**dict(case,hcore=changed_h))
    assert changed.receipt.one_electron_identity!=result.receipt.one_electron_identity
    assert changed.receipt.source_identity!=result.receipt.source_identity
    changed=manta.compute_periodic_fock(**dict(case,spin_densities=2*case['spin_densities']))
    assert changed.receipt.source_identity==result.receipt.source_identity
    assert changed.receipt.density_identity!=result.receipt.density_identity
    assert changed.receipt.snapshot_identity!=result.receipt.snapshot_identity
    changed=manta.compute_periodic_fock(**dict(case,exchange_scale=.2))
    assert changed.receipt.source_identity!=result.receipt.source_identity
    changed=manta.compute_periodic_fock(**dict(case,reciprocal_batch=32))
    assert changed.receipt.source_identity==result.receipt.source_identity
    assert changed.receipt.density_identity==result.receipt.density_identity
    np.testing.assert_allclose(changed.values,expected,atol=2e-12)
    for name in ('hcore','spin_densities','exchange_vectors','hartree_vectors'): case[name][:]=0
    for value in (result.values,result.hartree_values,result.exchange_values): value[:]=0
    np.testing.assert_array_equal(result.values,expected)
    with pytest.raises(AttributeError): result.receipt.source_identity='changed'


@pytest.mark.parametrize('field', ['memory_bytes','shell_calls','primitive_quartets','scalar_terms','kernel_terms'])
def test_combined_resource_admission_exact_and_one_below_before_nan_copy(field):
    case=_case(kernel='erfc',omega=.7)
    p=manta.plan_periodic_fock(case['basis'],len(G),len(LABELS),len(CELLS),len(MOMENTA),4,kernel='erfc',omega=.7)
    plan_field={'memory_bytes':'modeled_peak_bytes','primitive_quartets':'primitive_pairs'}.get(field,field)
    budget=manta.ResourceBudget(); setattr(budget,field,getattr(p,plan_field))
    manta.compute_periodic_fock(**dict(case,budget=budget))
    setattr(budget,field,getattr(p,plan_field)-1); case['hcore'][:]=np.nan
    with pytest.raises((ValueError,RuntimeError),match='budget'): manta.compute_periodic_fock(**dict(case,budget=budget))


def test_raw_invalid_hcore_and_each_density_block_fail_without_repair():
    case=_case(); case['hcore'][1,0,1]+=1e-4
    with pytest.raises(ValueError,match='Hermitian'): manta.compute_periodic_fock(**case)
    case=_case(); case['spin_densities'][0,0]*=1e10; case['spin_densities'][1,2,0,1]+=1e-4
    with pytest.raises(ValueError,match='Hermitian'): manta.compute_periodic_fock(**case)
    for key in ('hcore','spin_densities'):
        case=_case(); case[key].flat[0]=np.nan
        with pytest.raises(ValueError,match='finite'): manta.compute_periodic_fock(**case)
    with pytest.raises(ValueError,match='finite'): manta.compute_periodic_fock(**dict(_case(),exchange_scale=np.inf))


def test_layout_strict_shapes_and_empty_sources():
    case=_case(); expected=manta.compute_periodic_fock(**case)
    f={key:np.asfortranarray(value) if isinstance(value,np.ndarray) else value for key,value in case.items()}
    actual=manta.compute_periodic_fock(**f)
    np.testing.assert_array_equal(actual.values,expected.values)
    assert actual.receipt.snapshot_identity==expected.receipt.snapshot_identity
    for key in ('hcore','spin_densities'):
        wrong=dict(case); wrong[key]=wrong[key].real.copy()
        with pytest.raises(ValueError): manta.compute_periodic_fock(**wrong)
    with pytest.raises(ValueError): manta.compute_periodic_fock(**dict(case,spin_densities=case['spin_densities'][:1]))
    result=manta.compute_periodic_fock(**dict(case,hartree_vectors=np.empty((0,3)),exchange_vectors=np.empty((0,3)),transfer_offsets=np.zeros(10,dtype=np.int64)))
    np.testing.assert_array_equal(result.values,np.stack([case['hcore'],case['hcore']]))
    assert result.receipt.coulomb_energy==0 and result.receipt.exchange_energy==0


def test_selected_snapshot_records_executed_components_and_does_not_mutate_engine(monkeypatch):
    from vibeqc import output
    records=[]; monkeypatch.setattr(output,'record',lambda *args,**kwargs: records.append(kwargs))
    case=_case(kernel='erfc',omega=.6,scale=.25); engine=vq.IntegralEngineOptions(backend='manta')
    result=manta.build_periodic_fock(**case,integral_engine=engine)
    receipt=records[0]['integral_provenance']
    assert receipt['snapshot_identity']==result.receipt.snapshot_identity
    assert receipt['exchange']['kernel']=='erfc' and receipt['exchange']['zero_vectors']==3
    assert receipt['hartree']['boundary']=='3d-zero-average-g0-omitted'
    assert receipt['one_electron_source']=='supplied-matrix'
    assert receipt['spin_labels']==['alpha','beta']
    assert engine.budget.other_live_bytes==0
    engine.budget.memory_bytes=1
    with pytest.raises(ValueError,match='budget'): manta.build_periodic_fock(**case,integral_engine=engine)
    assert len(records)==1


def test_time_reversal_and_shared_atom_image_gauge_preserve_complete_snapshot_energy():
    case=_case(); result=manta.compute_periodic_fock(**case)
    reverse=dict(case,momenta=-case['momenta'],exchange_vectors=-case['exchange_vectors'],
        hcore=case['hcore'].conj().copy(),spin_densities=case['spin_densities'].conj().copy())
    actual=manta.compute_periodic_fock(**reverse)
    np.testing.assert_allclose(actual.values,result.values.conj(),atol=3e-12)
    assert actual.receipt.electronic_energy==pytest.approx(result.receipt.electronic_energy,abs=3e-12)
    basis=case['basis']; cutoff=1.4
    system=vq.PeriodicSystem(3,LATTICE,[vq.Atom(2,s.origin) for s in basis.shells()])
    cells=lambda b:np.array([c.r_cart for c in vq.pair_complete_lattice_cells(b,system,cutoff)])
    case.update(translations=cells(basis),pair_cutoff_bohr=cutoff)
    original=manta.compute_periodic_fock(**case)
    shells=basis.shells(); shells[1].origin=(np.array(shells[1].origin)+LATTICE[:,0]).tolist()
    moved_basis=vq.BasisSet(vq.Molecule([vq.Atom(2,s.origin) for s in shells],0,1),shells,'manta-spin-fock-image',True)
    shifts=np.array([[0.,0.,0.]]+[LATTICE[:,0].tolist()]*3); u=np.exp(-1j*MOMENTA@shifts.T)
    push=lambda a:np.ascontiguousarray(u.conj()[...,None]*a*u[:,None,:])
    changed=manta.compute_periodic_fock(**dict(case,basis=moved_basis,translations=cells(moved_basis),
        hcore=push(case['hcore']),spin_densities=np.stack([push(d) for d in case['spin_densities']])))
    np.testing.assert_allclose(changed.values,np.stack([push(f) for f in original.values]),atol=8e-12)
    assert changed.receipt.electronic_energy==pytest.approx(original.receipt.electronic_energy,abs=1e-11)
