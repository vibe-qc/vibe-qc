"""MANTA envelope omission checked against independently evaluated quartets."""
from itertools import product

import numpy as np
import pytest

import vibeqc as vq
from vibeqc.manta import ReferenceJKBuilder
from vibeqc.symmetry_ao import build_molecular_space_action
from vibeqc.symmetry_quartets import qualify_manta_quartets, manta_quartet_requirements
from vibeqc.symmetry_quartets_bounds import (
    MolecularCoulombBounds, molecular_coulomb_bounds_requirements, _upper_product,
)
from vibeqc.symmetry_quartets_batch import quartet_batch_requirements
from vibeqc.symmetry_molecular_hf import ScreenedMolecularJK
from vibeqc.symmetry_shared import Budget
from tests.test_symmetry_quartets import BUDGET, SPACE, molecular_case, masked_source, dense_jk


def separated_hydrogens(separation=30.):
    xyz = np.array([[x,0.,z] for x in (-separation/2,separation/2) for z in (-.7,.7)])
    mol = vq.Molecule([vq.Atom(1,p) for p in xyz])
    basis = vq.BasisSet(mol,'sto-3g')
    rotations = np.array([np.diag(x) for x in product((1.,-1.),repeat=3)])
    return mol,basis,rotations,np.zeros((8,3))


def bound_case(kind='separated', cap=1e-12):
    mol,basis,r,t = separated_hydrogens() if kind=='separated' else molecular_case(kind)
    provider = ReferenceJKBuilder(basis)
    bounds = MolecularCoulombBounds(basis,provider,budget=BUDGET)
    plan = qualify_manta_quartets(mol,basis,r,t,provider=provider,source_tolerance=1e-10,
        schedule='streaming',maximum_omitted_norm=cap,bounds=bounds,budget=BUDGET)
    return plan,bounds,provider,mol,basis,r,t


@pytest.mark.parametrize('kind', ['water','ammonia','cartesian','separated'])
def test_native_envelope_covers_actual_ordered_tiles(kind):
    plan,bounds,provider,_,basis,_,_ = bound_case(kind)
    ns = basis.nshells
    session = provider.session()
    for key in product(range(ns),repeat=4):
        actual = session.evaluate(key)['values']
        assert np.linalg.norm(actual) <= bounds.tile_frobenius_bound(key)
    assert bounds.receipt['integral_shell_calls'] == 0
    assert bounds.receipt['shell_pairs'] == ns*(ns+1)//2
    assert bounds.envelope_evaluations == 2*bounds.shell_pairs
    assert not bounds.receipt['operator_equivariance_certified']
    with pytest.raises(ValueError):
        bounds._q.setflags(write=True)
    with pytest.raises((AttributeError,TypeError)):
        bounds.source_identity = 'foreign'


@pytest.mark.parametrize('executor', ['numpy','native'])
def test_skip_before_integral_reads_same_mask_complex_covariance(executor, monkeypatch):
    plan,bounds,provider,mol,basis,r,t = bound_case()
    screen = plan.screening
    ns,n = basis.nshells,basis.nbasis
    assert 0 < screen.omitted_ordered_tiles < ns**4
    assert plan.qualification_work.shell_calls+screen.omitted_ordered_tiles == ns**4
    assert plan.bound_queries == ns**4
    full,masked = masked_source(plan,provider,screen)
    assert np.linalg.norm(full-masked) <= screen.omitted_norm <= 1e-12
    assert screen.receipt['omitted_norm_kind'] == 'upper-bound'
    assert screen.receipt['qualification_scope'] == 'retained-orbits'
    # Observe actual audit requests, not just the final counters.
    import vibeqc.symmetry_quartets_streaming as stream
    read = stream._read
    omitted = {m.shells for keep,o in zip(screen.keep,plan.orbits) if not keep for m in o.members}
    visited = []
    def observe(source,key,*args):
        assert key not in omitted
        visited.append(key)
        return read(source,key,*args)
    monkeypatch.setattr(stream,'_read',observe)
    repeat = qualify_manta_quartets(mol,basis,r,t,provider=provider,source_tolerance=1e-10,
        schedule='streaming',maximum_omitted_norm=1e-12,bounds=bounds,budget=BUDGET)
    assert len(visited) == len(set(visited)) == ns**4-len(omitted)
    assert repeat.screening.source_identity == screen.source_identity
    rng=np.random.default_rng(828)
    d=rng.normal(size=(2,n,n))+1j*rng.normal(size=(2,n,n))
    result=plan.jk_batch(d,targets=('jk','k'),screening=screen,executor=executor,budget=BUDGET)
    for array,channels,which in ((result.j,result.j_channels,0),(result.k,result.k_channels,1)):
        for value,channel in zip(array,channels):
            np.testing.assert_allclose(value,dense_jk(masked,d[channel])[which],atol=1e-12,rtol=0)
            assert np.linalg.norm(value-dense_jk(full,d[channel])[which]) <= result.error_bounds[channel]+1e-13
    for rotation,translation in zip(r,t):
        action=build_molecular_space_action(mol,basis,rotation,translation,source=SPACE,target=SPACE,budget=BUDGET)
        u=action.apply(np.eye(n,dtype=complex))
        after=plan.jk_batch(np.ascontiguousarray(u@d@u.conj().T),targets=('jk','k'),
            screening=screen,executor=executor,budget=BUDGET)
        np.testing.assert_allclose(u.conj().T@after.j@u,result.j,atol=1e-12,rtol=0)
        np.testing.assert_allclose(u.conj().T@after.k@u,result.k,atol=1e-12,rtol=0)
    assert result.work.shell_calls == len(plan.orbits)-screen.omitted_representatives
    assert result.scalar_products == 3*(n**4-screen.omitted_integral_values)
    assert result.block_contractions == 3*(ns**4-screen.omitted_ordered_tiles)
    # The same certificate enters the existing native HF bridge unchanged.
    jk=ScreenedMolecularJK(screen,budget=BUDGET,executor=executor)
    np.testing.assert_allclose(jk.build_J(d[0].real.copy()),dense_jk(masked,d[0].real)[0],atol=1e-12,rtol=0)


@pytest.mark.parametrize('executor', ['numpy','native'])
def test_partial_plan_refuses_unmasked_and_foreign_source(executor):
    plan,bounds,provider,mol,basis,r,t = bound_case()
    d=np.ones((1,basis.nbasis,basis.nbasis),complex)
    with pytest.raises(ValueError,match='partially audited'):
        plan.jk_batch(d,budget=BUDGET,executor=executor)
    with pytest.raises(ValueError,match='partially audited'):
        plan.jk(d[0],budget=BUDGET,executor=executor)
    displaced,other_basis,_,_=separated_hydrogens(30.02)
    other=ReferenceJKBuilder(other_basis)
    with pytest.raises(ValueError,match='different numerical source'):
        qualify_manta_quartets(displaced,other_basis,r,t,provider=other,source_tolerance=1e-10,
            schedule='streaming',maximum_omitted_norm=1e-12,bounds=bounds,budget=BUDGET)
    with pytest.raises(ValueError,match='snapshot'):
        MolecularCoulombBounds(other_basis,provider,budget=BUDGET)
    with pytest.raises(TypeError,match='owned native'):
        qualify_manta_quartets(mol,basis,r,t,provider=provider,source_tolerance=1e-10,
            schedule='streaming',maximum_omitted_norm=1e-12,bounds=object(),budget=BUDGET)
    with pytest.raises(ValueError,match='explicit spatial streaming mask'):
        qualify_manta_quartets(mol,basis,r,t,provider=provider,source_tolerance=1e-10,
            schedule='streaming',bounds=bounds,budget=BUDGET)
    # Explicit full requalification/unreduced source remains available.
    complete=qualify_manta_quartets(mol,basis,r,t,provider=provider,source_tolerance=1e-10,
        schedule='streaming',budget=BUDGET)
    assert complete.qualification_work.shell_calls == basis.nshells**4
    complete.jk_batch(d,budget=BUDGET,executor=executor)


def test_bound_budget_admission_and_extreme_positive_products(monkeypatch):
    mol,basis,r,t=separated_hydrogens()
    provider=ReferenceJKBuilder(basis)
    size=molecular_coulomb_bounds_requirements(basis,provider,budget=BUDGET)
    bounds=MolecularCoulombBounds(basis,provider,budget=size)
    from vibeqc import _vibeqc_core as core
    def no_compute(*args):
        raise AssertionError('budget refusal must precede bound generation')
    monkeypatch.setattr(core._manta,'compute_pair_extents',no_compute)
    for short,error in ((Budget(size.maximum_bytes-1,size.maximum_work),MemoryError),
                        (Budget(size.maximum_bytes,size.maximum_work-1),ValueError)):
        with pytest.raises(error,match='budget'):
            MolecularCoulombBounds(basis,provider,budget=short)
    assert _upper_product(1e-300,1e-300)>0.
    assert _upper_product(0.,1e300)==0.
    with pytest.raises(ValueError,match='nonrepresentable'):
        _upper_product(1e300,1e300)
    required=manta_quartet_requirements(mol,basis,provider,len(r),budget=BUDGET,
        schedule='streaming',maximum_omitted_norm=1e-12,bounds=bounds)
    plan=qualify_manta_quartets(mol,basis,r,t,provider=provider,source_tolerance=1e-10,
        schedule='streaming',maximum_omitted_norm=1e-12,bounds=bounds,budget=required)
    for executor in ('numpy','native'):
        needed=quartet_batch_requirements(plan,1,budget=BUDGET,executor=executor,screening=plan.screening)
        with pytest.raises(MemoryError,match='budget'):
            plan.jk_batch(np.full((1,basis.nbasis,basis.nbasis),np.nan,complex),
                budget=Budget(needed.maximum_bytes-1,needed.maximum_work),executor=executor,screening=plan.screening)
