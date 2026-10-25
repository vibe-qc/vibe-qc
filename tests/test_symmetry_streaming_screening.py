"""Compact masks against independently evaluated ordered molecular sources."""
from dataclasses import FrozenInstanceError
import json

import numpy as np
import pytest

from vibeqc.manta import ReferenceJKBuilder
from vibeqc.symmetry_ao import build_molecular_space_action
from vibeqc.symmetry_quartets import qualify_manta_quartets, manta_quartet_requirements
from vibeqc.symmetry_quartets_batch import quartet_batch_requirements
from vibeqc.symmetry_quartets_streaming import StreamingShellQuartetPlan
from vibeqc.symmetry_shared import Budget
from tests.test_symmetry_quartets import (
    BUDGET, SPACE, dense_jk, masked_source, molecular_case, source_fixture,
)
import vibeqc as vq
from vibeqc import _vibeqc_core as core


def screened_case(kind='water', cap=.3):
    mol, basis, rotations, translations = molecular_case(kind)
    provider = ReferenceJKBuilder(basis)
    plan = qualify_manta_quartets(mol, basis, rotations, translations,
        provider=provider, source_tolerance=1e-10, budget=BUDGET,
        schedule='streaming', maximum_omitted_norm=cap)
    return plan, provider, mol, basis, rotations, translations


@pytest.mark.parametrize('kind', ['water', 'ammonia', 'cartesian'])
@pytest.mark.parametrize('executor', ['numpy', 'native'])
def test_compact_mask_ordered_oracle_covariance_and_work(kind, executor):
    plan, provider, mol, basis, rotations, translations = screened_case(
        kind, 2. if kind == 'cartesian' else .3)
    screen = plan.screening
    n, ns = basis.nbasis, basis.nshells
    assert 0 < screen.omitted_representatives < len(plan.orbits)
    assert len(screen.keep.data) == (len(plan.orbits)+7)//8
    assert plan.qualification_work.tile_requests == ns**4
    full, masked = masked_source(plan, provider, screen)
    np.testing.assert_allclose(screen.omitted_norm, np.linalg.norm(full-masked), atol=1e-14, rtol=1e-14)
    assert screen.omitted_norm <= screen.maximum_omitted_norm
    rng = np.random.default_rng(72)
    d = rng.normal(size=(3,n,n))+1j*rng.normal(size=(3,n,n))
    targets = ('j', 'k', 'jk')
    result = plan.jk_batch(d, targets=targets, screening=screen, executor=executor, budget=BUDGET)
    for which, (values, channels) in enumerate(((result.j, result.j_channels), (result.k, result.k_channels))):
        for value, channel in zip(values, channels):
            np.testing.assert_allclose(value, dense_jk(masked, d[channel])[which], atol=2e-11, rtol=0)
            assert np.linalg.norm(value-dense_jk(full,d[channel])[which]) <= result.error_bounds[channel]+2e-13
    # Raw operator covariance, before any converged state or symmetrization.
    for r,t in zip(rotations, translations):
        action = build_molecular_space_action(mol,basis,r,t,source=SPACE,target=SPACE,budget=BUDGET)
        u = action.apply(np.eye(n, dtype=complex))
        moved = plan.jk_batch(np.ascontiguousarray(u@d@u.conj().T), targets=targets,
            screening=screen, executor=executor, budget=BUDGET)
        for after,before in ((moved.j,result.j),(moved.k,result.k)):
            np.testing.assert_allclose(u.conj().T@after@u, before, atol=3e-11, rtol=0)
    active = [o for keep,o in zip(screen.keep,plan.orbits) if keep]
    assert result.work.shell_calls == result.work.tile_requests == len(active)
    session = provider.session()
    assert result.work.primitive_quartets_requested == sum(
        session.evaluate(o.representative)['primitive_quartets_requested'] for o in active)
    assert result.scalar_products == 4*(n**4-screen.omitted_integral_values)
    assert result.block_contractions == 4*(ns**4-screen.omitted_ordered_tiles)
    assert not plan.production_reduction_authorized
    # An explicit unmasked build still uses the full digest and source.
    uncut = plan.jk_batch(d, targets=targets, executor=executor, budget=BUDGET)
    np.testing.assert_allclose(uncut.j[0],dense_jk(full,d[0])[0],atol=2e-11,rtol=0)
    assert uncut.source_identity == plan.source_identity != screen.source_identity


@pytest.mark.parametrize('executor', ['numpy', 'native'])
@pytest.mark.parametrize('cap', [0., 100.])
def test_compact_zero_and_all_omitted(executor, cap):
    plan,provider,_,basis,_,_ = screened_case(cap=cap)
    screen = plan.screening
    d = np.ones((1,basis.nbasis,basis.nbasis), complex)
    result = plan.jk_batch(d,targets=('k',),screening=screen,executor=executor,budget=BUDGET)
    full,masked = masked_source(plan,provider,screen)
    assert result.j.shape == (0,basis.nbasis,basis.nbasis)
    np.testing.assert_allclose(result.k[0],dense_jk(masked,d[0])[1],atol=1e-12,rtol=0)
    if cap:
        assert not any(screen.keep)
        assert result.work.shell_calls == result.scalar_products == result.block_contractions == 0
    else:
        assert screen.omitted_norm == 0.
        np.testing.assert_allclose(result.k[0],dense_jk(full,d[0])[1],atol=1e-12,rtol=0)


@pytest.mark.parametrize('executor', ['numpy', 'native'])
def test_compact_budget_identity_and_retained_digest(executor, monkeypatch):
    plan,provider,mol,basis,r,t = screened_case()
    screen = plan.screening
    kwargs = dict(provider=provider,source_tolerance=1e-10,schedule='streaming',maximum_omitted_norm=.3)
    required = manta_quartet_requirements(mol,basis,provider,len(r),budget=BUDGET,
        schedule='streaming',maximum_omitted_norm=.3)
    repeat = qualify_manta_quartets(mol,basis,r,t,budget=required,**kwargs)
    assert repeat.screening.receipt == screen.receipt
    json.dumps(screen.receipt,allow_nan=False)
    with pytest.raises(FrozenInstanceError):
        screen.keep.data = b''
    same = qualify_manta_quartets(mol,basis,r,t,budget=BUDGET,
        **{**kwargs,'maximum_omitted_norm':np.nextafter(.3,1.)})
    assert same.screening.source_identity == screen.source_identity
    assert same.screening.policy_identity != screen.policy_identity
    d = np.ones((1,basis.nbasis,basis.nbasis),complex)
    with pytest.raises(ValueError,match='different source audit'):
        repeat.jk_batch(d,screening=screen,executor=executor,budget=BUDGET)
    needed = quartet_batch_requirements(plan,1,budget=BUDGET,executor=executor,screening=screen)
    plan.jk_batch(d,screening=screen,executor=executor,budget=needed)
    for short,error in ((Budget(needed.maximum_bytes-1,needed.maximum_work),MemoryError),
                        (Budget(needed.maximum_bytes,needed.maximum_work-1),ValueError)):
        with pytest.raises(error,match='budget'):
            plan.jk_batch(np.full_like(d,np.nan),screening=screen,executor=executor,budget=short)
    import vibeqc.symmetry_quartets_batch as batch
    original_read = batch._read
    def changed(*args):
        tile = original_read(*args).copy()
        tile.flat[0] += .01
        return tile
    monkeypatch.setattr(batch,'_read',changed)
    with pytest.raises(ValueError,match='representative changed'):
        plan.jk_batch(d,screening=screen,executor=executor,budget=BUDGET)


def test_compact_admission_refuses_callbacks_and_invalid_domains(monkeypatch):
    _,_,g,a,source,_,_ = source_fixture('water')
    with pytest.raises(TypeError,match='immutable MANTA'):
        StreamingShellQuartetPlan(g,a,source,source_tolerance=1e-10,
            maximum_omitted_norm=.3,budget=BUDGET)
    assert not source.requests
    mol,basis,r,t = molecular_case('water')
    provider = ReferenceJKBuilder(basis)
    for cap in (-1.,np.inf,np.nan,True,'0.1'):
        with pytest.raises((TypeError,ValueError)):
            qualify_manta_quartets(mol,basis,r,t,provider=provider,source_tolerance=1e-10,
                schedule='streaming',maximum_omitted_norm=cap,budget=BUDGET)
    for schedule,mode in (('stored','spatial'),('streaming','permutation'),('streaming','none')):
        with pytest.raises(ValueError,match='spatial streaming'):
            qualify_manta_quartets(mol,basis,r,t,provider=provider,source_tolerance=1e-10,
                schedule=schedule,mode=mode,maximum_omitted_norm=.3,budget=BUDGET)
    import vibeqc.symmetry_quartets_streaming as stream
    def no_read(*args):
        raise AssertionError('budget refusal must precede all source reads')
    monkeypatch.setattr(stream,'_read',no_read)
    required = manta_quartet_requirements(mol,basis,provider,len(r),budget=BUDGET,
        schedule='streaming',maximum_omitted_norm=.3)
    for short,error in ((Budget(required.maximum_bytes-1,required.maximum_work),MemoryError),
                        (Budget(required.maximum_bytes,required.maximum_work-1),ValueError)):
        with pytest.raises(error,match='budget'):
            qualify_manta_quartets(mol,basis,r,t,provider=provider,source_tolerance=1e-10,
                schedule='streaming',maximum_omitted_norm=.3,budget=short)


def test_compact_cartesian_member_norms_and_hf_density_response():
    from vibeqc.symmetry_molecular_hf import ScreenedMolecularJK
    from tests.test_symmetry_molecular_hf import method
    _,_,r,t = molecular_case('cartesian')
    origins = [[1.7*np.cos(x),1.7*np.sin(x),0.] for x in (0.,2*np.pi/3,4*np.pi/3)]
    mol = vq.Molecule([vq.Atom(2,xyz) for xyz in origins])
    basis = vq.BasisSet(mol,[core.ShellInfo(i,2,False,[.8],[1.],xyz)
        for i,xyz in enumerate(origins)],'Cartesian triangle',False)
    provider = ReferenceJKBuilder(basis)
    plan = qualify_manta_quartets(mol,basis,r,t,provider=provider,source_tolerance=1e-10,
        schedule='streaming',maximum_omitted_norm=.52,budget=BUDGET)
    # A representative norm times sqrt(multiplicity) would omit an orbit
    # here and exceed the true bound in these nonunitary AO coordinates.
    assert plan.screening.omitted_representatives == 0
    plan = qualify_manta_quartets(mol,basis,r,t,provider=provider,source_tolerance=1e-10,
        schedule='streaming',maximum_omitted_norm=.56,budget=BUDGET)
    assert plan.screening.omitted_representatives == 1
    _,masked = masked_source(plan,provider,plan.screening)
    jk = ScreenedMolecularJK(plan.screening,budget=BUDGET)
    hf,_,h = method(jk,mol,basis)
    raw = np.random.default_rng(174).normal(size=(2,basis.nbasis,basis.nbasis))
    spin = raw+raw.transpose(0,2,1)
    delta = .1*spin+np.eye(basis.nbasis)[None]
    evaluated = hf.evaluate(tuple(spin),method='UHF',generation=1)
    total = spin.sum(axis=0)
    j = dense_jk(masked,total)[0]
    k = np.stack([dense_jk(masked,d)[1] for d in spin])
    expected = np.einsum('pq,qp',total,h+.5*j)-.5*np.einsum('apq,aqp',spin,k)
    np.testing.assert_allclose(evaluated.build.energy.electronic,expected,atol=2e-11,rtol=0)
    step = 1e-5
    def energy(d):
        return hf.evaluate(tuple(d),method='UHF',generation=2).build.energy.electronic
    derivative = (energy(spin+step*delta)-energy(spin-step*delta))/(2*step)
    np.testing.assert_allclose(derivative,np.einsum('apq,aqp',delta,h[None]+j[None]-k),atol=3e-8,rtol=0)
