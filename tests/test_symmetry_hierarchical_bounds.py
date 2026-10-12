"""Native hierarchical candidates must retain complete source-qualified orbits."""
from itertools import product

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from vibeqc.manta import ReferenceJKBuilder
from vibeqc.symmetry_ao import build_molecular_space_action
from vibeqc.symmetry_quartets import qualify_manta_quartets, manta_quartet_requirements
from vibeqc.symmetry_quartets_hierarchy import (
    MolecularHierarchicalBounds, molecular_hierarchical_bounds_requirements,
)
from vibeqc.symmetry_shared import Budget
from tests.test_symmetry_producer_bounds import separated_hydrogens
from tests.test_symmetry_quartets import BUDGET, SPACE, molecular_case, masked_source, dense_jk


def hierarchy_case(kind='separated', cap=1e-12, leaf_capacity=1):
    mol,basis,r,t = separated_hydrogens() if kind=='separated' else molecular_case(kind)
    provider=ReferenceJKBuilder(basis)
    bounds=MolecularHierarchicalBounds(basis,provider,maximum_omitted_norm=cap,
        leaf_capacity=leaf_capacity,budget=BUDGET)
    plan=qualify_manta_quartets(mol,basis,r,t,provider=provider,source_tolerance=1e-10,
        schedule='streaming',maximum_omitted_norm=cap,bounds=bounds,budget=BUDGET)
    return plan,bounds,provider,mol,basis,r,t


@pytest.mark.parametrize('kind', ['water','ammonia','cartesian','separated'])
def test_hierarchy_bounds_cover_independently_evaluated_ordered_integrals(kind):
    plan,bounds,provider,_,basis,_,_=hierarchy_case(kind,cap=1e6)
    session=provider.session()
    assert bounds.receipt['integral_shell_calls']==0
    assert not bounds.receipt['operator_equivariance_certified']
    # Large cap accepts every native partition block. Check the hierarchical
    # bound itself, not a minimum that could hide it behind the old Q/V bound.
    assert bounds.receipt['candidate_integral_values']==basis.nbasis**4
    for key in product(range(basis.nshells),repeat=4):
        covered,probes=bounds.covered_bound(key)
        assert covered is not None and probes>0
        actual=np.linalg.norm(session.evaluate(key)['values'])
        assert actual<=covered
        assert actual<=bounds.tile_frobenius_bound(key)
    if kind=='separated':
        assert bounds.receipt['far_bound_queries']>0
        assert bounds.receipt['tightened_node_bounds']>0
    with pytest.raises(TypeError):
        bounds._coverage[(0,0)]=0.
    with pytest.raises(ValueError):
        bounds._q.setflags(write=True)


@pytest.mark.parametrize('kind', ['separated','cartesian','ammonia'])
@pytest.mark.parametrize('executor', ['numpy','native'])
def test_hierarchical_orbit_closure_arbitrary_density_and_actual_reads(kind, executor, monkeypatch):
    plan,bounds,provider,mol,basis,r,t=hierarchy_case(kind)
    screen=plan.screening
    ns,n=basis.nshells,basis.nbasis
    full,masked=masked_source(plan,provider,screen)
    assert np.linalg.norm(full-masked)<=screen.omitted_norm<=1e-12
    assert plan.qualification_work.shell_calls+screen.omitted_ordered_tiles==ns**4
    # A generic tree is not required to map to itself. No member may escape
    # the accepted native candidate blocks when an orbit is actually omitted.
    for keep,orbit in zip(screen.keep,plan.orbits):
        if not keep:
            assert all(bounds.covered_bound(m.shells)[0] is not None for m in orbit.members)
    if kind=='separated':
        assert 0<screen.omitted_ordered_tiles<ns**4
        assert bounds.receipt['node_bound_queries']<ns**4
        import vibeqc.symmetry_quartets_streaming as stream
        read=stream._read
        omitted={m.shells for keep,o in zip(screen.keep,plan.orbits) if not keep for m in o.members}
        seen=[]
        def observe(source,key,*args):
            assert key not in omitted
            seen.append(key)
            return read(source,key,*args)
        monkeypatch.setattr(stream,'_read',observe)
        again=qualify_manta_quartets(mol,basis,r,t,provider=provider,source_tolerance=1e-10,
            schedule='streaming',maximum_omitted_norm=1e-12,bounds=bounds,budget=BUDGET)
        assert len(seen)==len(set(seen))==ns**4-len(omitted)
        assert again.screening.source_identity==screen.source_identity
    rng=np.random.default_rng(881)
    d=np.ascontiguousarray(rng.normal(size=(2,n,n))+1j*rng.normal(size=(2,n,n)))
    result=plan.jk_batch(d,targets=('jk','k'),executor=executor,screening=screen,budget=BUDGET)
    assert np.linalg.norm(result.k)>0.
    for arrays,channels,which in ((result.j,result.j_channels,0),(result.k,result.k_channels,1)):
        for a,ch in zip(arrays,channels):
            np.testing.assert_allclose(a,dense_jk(masked,d[ch])[which],atol=1e-11,rtol=0)
    for rotation,translation in zip(r,t):
        action=build_molecular_space_action(mol,basis,rotation,translation,source=SPACE,target=SPACE,budget=BUDGET)
        u=action.apply(np.eye(n,dtype=complex))
        after=plan.jk_batch(np.ascontiguousarray(u@d@u.conj().T),targets=('jk','k'),
            executor=executor,screening=screen,budget=BUDGET)
        np.testing.assert_allclose(u.conj().T@after.j@u,result.j,atol=1e-10,rtol=0)
        np.testing.assert_allclose(u.conj().T@after.k@u,result.k,atol=1e-10,rtol=0)
    assert result.work.shell_calls==len(plan.orbits)-screen.omitted_representatives
    with pytest.raises(ValueError,match='partially audited'):
        plan.jk_batch(d,budget=BUDGET,executor=executor)


def test_source_policy_and_aggregate_budget_refusals_precede_integrals(monkeypatch):
    plan,bounds,provider,mol,basis,r,t=hierarchy_case()
    required=manta_quartet_requirements(mol,basis,provider,len(r),budget=BUDGET,
        schedule='streaming',maximum_omitted_norm=1e-12,bounds=bounds)
    qualify_manta_quartets(mol,basis,r,t,provider=provider,source_tolerance=1e-10,
        schedule='streaming',maximum_omitted_norm=1e-12,bounds=bounds,budget=required)
    with pytest.raises(ValueError,match='candidate budget changed'):
        qualify_manta_quartets(mol,basis,r,t,provider=provider,source_tolerance=1e-10,
            schedule='streaming',maximum_omitted_norm=2e-12,bounds=bounds,budget=BUDGET)
    displaced,other,_,_=separated_hydrogens(30.02)
    with pytest.raises(ValueError,match='different numerical source'):
        qualify_manta_quartets(displaced,other,r,t,provider=ReferenceJKBuilder(other),source_tolerance=1e-10,
            schedule='streaming',maximum_omitted_norm=1e-12,bounds=bounds,budget=BUDGET)
    with pytest.raises(ValueError,match='snapshot'):
        MolecularHierarchicalBounds(other,provider,maximum_omitted_norm=1e-12,budget=BUDGET)
    needed=molecular_hierarchical_bounds_requirements(basis,provider,budget=BUDGET)
    def no_compute(*args): raise AssertionError('budget refusal must precede envelopes')
    monkeypatch.setattr(core._manta,'compute_pair_extents',no_compute)
    for short,error in ((Budget(needed.maximum_bytes-1,needed.maximum_work),MemoryError),
                        (Budget(needed.maximum_bytes,needed.maximum_work-1),ValueError)):
        with pytest.raises(error,match='budget'):
            MolecularHierarchicalBounds(basis,provider,maximum_omitted_norm=1e-12,budget=short)
    with pytest.raises(ValueError,match='opening'):
        molecular_hierarchical_bounds_requirements(basis,provider,opening=1.,budget=BUDGET)


def separated_h8():
    import vibeqc as vq
    xyz=np.array([[x,y,z] for x in (-15.,15.) for y in (-15.,15.) for z in (-.7,.7)])
    mol=vq.Molecule([vq.Atom(1,p) for p in xyz])
    basis=vq.BasisSet(mol,'sto-3g')
    r=np.array([np.diag(x) for x in product((1.,-1.),repeat=3)])
    return mol,basis,r,np.zeros((8,3))


def test_accepted_parent_blocks_compress_actual_pair_interactions():
    mol,basis,r,t=separated_h8()
    provider=ReferenceJKBuilder(basis)
    bounds=MolecularHierarchicalBounds(basis,provider,maximum_omitted_norm=1e-12,budget=BUDGET)
    nodes=bounds._tree.nodes
    pair_interactions=0
    internal_blocks=0
    for a,b in bounds._coverage:
        na,nb=nodes[a],nodes[b]
        n,m=na.end-na.begin,nb.end-nb.begin
        pair_interactions+=n*(n+1)//2 if a==b else n*m
        internal_blocks+=bool(na.children or nb.children)
    assert internal_blocks>0
    assert pair_interactions>bounds.receipt['accepted_candidate_blocks']
    plan=qualify_manta_quartets(mol,basis,r,t,provider=provider,source_tolerance=1e-10,
        schedule='streaming',maximum_omitted_norm=1e-12,bounds=bounds,budget=BUDGET)
    full,masked=masked_source(plan,provider,plan.screening)
    assert 0<np.linalg.norm(full-masked)<=plan.screening.omitted_norm<=1e-12
    assert plan.qualification_work.shell_calls<basis.nshells**4


def test_partial_candidate_orbits_are_retained_without_completing_the_mask():
    # Deliberately tight caps cut through a generic native tree partition.
    # The mathematical group is unchanged: numerical candidates may be mixed.
    mixed=0
    for cap in (1e-35,1e-34,1e-33):
        plan,bounds,provider,_,_,_,_=hierarchy_case(cap=cap)
        for keep,orbit in zip(plan.screening.keep,plan.orbits):
            covered=[bounds.covered_bound(m.shells)[0] is not None for m in orbit.members]
            if any(covered) and not all(covered):
                mixed+=1
                assert keep
        full,masked=masked_source(plan,provider,plan.screening)
        assert np.linalg.norm(full-masked)<=plan.screening.omitted_norm<=cap
    assert mixed>0, 'fixture must expose an actually asymmetric candidate mask'
