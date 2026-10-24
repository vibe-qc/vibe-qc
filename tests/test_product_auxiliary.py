"""Shared product basis ranges, nesting, polynomial space and admission."""
import gc
import math
import numpy as np
import pytest
import vibeqc as vq
from vibeqc._vibeqc_core import ShellInfo
from vibeqc.auxiliary_generation import (
    ProductAuxiliaryOptions, ProductAuxiliaryLimits, plan_product_auxiliary_basis,
    prepare_product_auxiliary_basis, product_auxiliary_provenance,
)


def basis(molecule, channels, *, normalized=False):
    shells = [ShellInfo(atom_index=atom, l=l, pure=pure, exponents=exponents,
        coefficients=coefficients, origin=molecule.atoms[atom].xyz)
        for atom,l,pure,exponents,coefficients in channels]
    return vq.BasisSet(molecule, shells, 'fixture', coefficients_pre_normalized=normalized)


def fixture():
    molecule = vq.Molecule([vq.Atom(4,[0.,0.,0.]), vq.Atom(2,[2.,1.,-.4])])
    orbital = basis(molecule, [(0,0,True,[.2,2.4],[.7,-.2]),
        (0,1,True,[.7],[1.]),(1,0,True,[.3],[1.])])
    seed = basis(molecule, [(0,0,True,[.4],[1.]), (0,1,False,[.9,1.6],[.3,-.4])])
    return molecule,orbital,seed


def shell_data(b):
    return [(s.atom_index,s.l,s.pure,tuple(s.origin),tuple(s.exponents),tuple(s.coefficients)) for s in b.shells()]


def test_independent_product_ranges_and_preserved_seed():
    mol,ao,seed=fixture();source=prepare_product_auxiliary_basis(mol,ao,seed)
    assert shell_data(source.basis)[:seed.nshells] == shell_data(seed)
    expected={(0,0):(.4,4.8),(0,1):(.9,3.1),(0,2):(1.4,1.4),(1,0):(.6,.6)}
    for span in source.receipt.spans:
        lo,hi=expected[(span.atom,span.angular_momentum)]
        assert span.minimum_exponent == pytest.approx(lo)
        assert span.maximum_product_exponent == pytest.approx(hi)
        count=math.ceil(math.log(hi/lo,2))
        assert span.coarse_intervals == count
        assert span.upper_grid_exponent == pytest.approx(lo*2**count)
    assert source.receipt.duplicate_seed_shells==1
    assert source.plan.maximum_shells==source.basis.nshells+1
    assert source.plan.maximum_functions>=source.basis.nbasis
    assert source.receipt.added_functions==source.basis.nbasis-seed.nbasis
    assert source.matches_orbital(ao)
    other=basis(mol,[(0,0,False,[.21],[1.])]);assert not source.matches_orbital(other)


def test_refinement_nesting_and_captured_owned_metadata():
    mol,ao,seed=fixture();o=ProductAuxiliaryOptions();sources=[]
    for level in range(3):
        o.refinement=level;sources.append(prepare_product_auxiliary_basis(mol,ao,seed,o))
    for lower,upper in zip(sources,sources[1:]):
        assert set(shell_data(lower.basis)) <= set(shell_data(upper.basis))
        assert lower.receipt.identity!=upper.receipt.identity
    assert sources[0].receipt.options.refinement==0
    receipt=sources[0].receipt;receipt.options.beta=7.
    assert sources[0].receipt.options.beta==2.
    assert product_auxiliary_provenance(sources[0])==product_auxiliary_provenance(prepare_product_auxiliary_basis(mol,ao,seed))
    held=sources[0].basis;expected=shell_data(held);del sources;gc.collect()
    assert shell_data(held)==expected


@pytest.mark.parametrize('pure',[False,True])
@pytest.mark.parametrize('angular',[0,1,2,3])
def test_cartesian_channel_spans_same_center_primitive_products(pure,angular):
    mol=vq.Molecule([vq.Atom(4,[.3,-.1,.2])]);ao=basis(mol,[(0,angular,pure,[.7],[1.])])
    aux=prepare_product_auxiliary_basis(mol,ao).basis
    assert aux.nshells==1 and not aux.shells()[0].pure
    rng=np.random.default_rng(441);fit=rng.normal(size=(180,3));check=rng.normal(size=(160,3))
    a=vq.evaluate_ao(ao,fit);b=vq.evaluate_ao(aux,fit)
    target=np.einsum('pi,pj->pij',a,a).reshape(len(fit),-1)
    coefficients=np.linalg.lstsq(b,target,rcond=1e-13)[0]
    x=vq.evaluate_ao(ao,check);truth=np.einsum('pi,pj->pij',x,x).reshape(len(check),-1)
    np.testing.assert_allclose(vq.evaluate_ao(aux,check)@coefficients,truth,atol=3e-13,rtol=3e-12)
    # A separately normalized raw shell fixes the shared normalization convention.
    raw=basis(mol,[(0,2*angular,False,[1.4],[1.])])
    np.testing.assert_array_equal(vq.evaluate_ao(aux,check),vq.evaluate_ao(raw,check))


def test_pure_d_seed_does_not_replace_cartesian_trace_components():
    mol=vq.Molecule([vq.Atom(4,[0.,0.,0.])]);ao=basis(mol,[(0,1,True,[.7],[1.])])
    pure=basis(mol,[(0,2,True,[1.4],[1.])]);cart=basis(mol,[(0,2,False,[1.4],[1.])])
    a=prepare_product_auxiliary_basis(mol,ao,pure);b=prepare_product_auxiliary_basis(mol,ao,cart)
    assert a.receipt.added_shells==1 and a.basis.nbasis==11
    assert b.receipt.added_shells==0 and b.basis.nbasis==6


@pytest.mark.parametrize('field,quantity', [('memory_bytes','modeled_peak_bytes'),('work_units','work_units'),
    ('maximum_shells','maximum_shells'),('maximum_functions','maximum_functions')])
def test_limits_admit_before_construction(field,quantity):
    mol,ao,seed=fixture();plan=plan_product_auxiliary_basis(mol,ao,seed);limit=ProductAuxiliaryLimits()
    setattr(limit,field,getattr(plan,quantity)-1)
    with pytest.raises((ValueError,OverflowError),match='budget'):prepare_product_auxiliary_basis(mol,ao,seed,limits=limit)
    setattr(limit,field,getattr(plan,quantity));source=prepare_product_auxiliary_basis(mol,ao,seed,limits=limit)
    assert source.plan.modeled_peak_bytes==plan.modeled_peak_bytes


@pytest.mark.parametrize('field,value',[('beta',1.),('beta',float('inf')),('beta',float('nan')),
    ('refinement',9),('maximum_angular_momentum',33),('maximum_angular_momentum',-1)])
def test_invalid_controls(field,value):
    mol,ao,seed=fixture();o=ProductAuxiliaryOptions();setattr(o,field,value)
    with pytest.raises(ValueError,match='controls'):plan_product_auxiliary_basis(mol,ao,seed,o)


def test_angular_omission_center_mismatch_and_tiny_spacing_refuse():
    mol,ao,seed=fixture();o=ProductAuxiliaryOptions();o.maximum_angular_momentum=1
    with pytest.raises(ValueError,match='omit'):prepare_product_auxiliary_basis(mol,ao,seed,o)
    o=ProductAuxiliaryOptions();o.beta=np.nextafter(1.,2.)
    with pytest.raises(ValueError,match='budget'):prepare_product_auxiliary_basis(mol,ao,seed,o)
    moved=vq.Molecule([vq.Atom(4,[.001,0.,0.]),vq.Atom(2,[2.,1.,-.4])])
    with pytest.raises(ValueError,match='centers'):prepare_product_auxiliary_basis(moved,ao,seed)


def test_common_translation_preserves_values_and_changes_identity():
    mol,ao,seed=fixture();shift=np.array([.123,-.31,.78])
    moved=vq.Molecule([vq.Atom(a.Z,np.array(a.xyz)+shift) for a in mol.atoms])
    def transport(b):
        return basis(moved,[(s.atom_index,s.l,s.pure,s.exponents,s.coefficients) for s in b.shells()],normalized=True)
    a=prepare_product_auxiliary_basis(mol,ao,seed);b=prepare_product_auxiliary_basis(moved,transport(ao),transport(seed))
    points=np.random.default_rng(6).normal(size=(20,3))
    np.testing.assert_allclose(vq.evaluate_ao(a.basis,points),vq.evaluate_ao(b.basis,points+shift),atol=2e-15,rtol=2e-14)
    assert a.receipt.identity!=b.receipt.identity


@pytest.mark.parametrize('alpha',[1e-300,1e300])
def test_unrepresentable_normalization_refuses(alpha):
    mol=vq.Molecule([vq.Atom(2,[0.,0.,0.])]);ao=basis(mol,[(0,0,False,[alpha],[1.])],normalized=True)
    with pytest.raises((ValueError,OverflowError),match='normalization'):prepare_product_auxiliary_basis(mol,ao)



def test_metadata_angular_ceiling_and_integer_overflow_refuse_safely():
    mol=vq.Molecule([vq.Atom(2,[0.,0.,0.])]);ao=basis(mol,[(0,16,False,[1.],[1.])],normalized=True)
    options=ProductAuxiliaryOptions();options.maximum_angular_momentum=32
    source=prepare_product_auxiliary_basis(mol,ao,options=options)
    assert source.basis.nbasis==561 and source.basis.shells()[0].l==32
    assert math.isfinite(source.basis.shells()[0].coefficients[0])
    ao=basis(mol,[(0,0,False,[1e-100,1e100],[1.,1.])],normalized=True)
    options.beta=np.nextafter(1.,2.);options.refinement=8
    limits=ProductAuxiliaryLimits()
    for field in ('memory_bytes','work_units','maximum_shells','maximum_functions'):setattr(limits,field,2**64-1)
    with pytest.raises(OverflowError,match='overflow'):plan_product_auxiliary_basis(mol,ao,options=options,limits=limits)



def test_sequence_preserves_shared_candidates_and_source_identity():
    from vibeqc.auxiliary_generation import prepare_product_auxiliary_sequence
    mol,ao,seed=fixture();initial=ProductAuxiliaryOptions()
    sequence=prepare_product_auxiliary_sequence(mol,ao,seed,initial,2)
    assert sequence.plan.candidates==3 and sequence.matches_sources(ao,seed)
    assert not sequence.matches_sources(ao,ao)
    for i in range(3):
        initial.refinement=i;single=prepare_product_auxiliary_basis(mol,ao,seed,initial)
        assert sequence.candidate(i).receipt.identity==single.receipt.identity
        assert shell_data(sequence.candidate(i).basis)==shell_data(single.basis)
        assert single.matches_seed(seed) and single.matches_basis(sequence.candidate(i).basis)
    with pytest.raises(IndexError):sequence.candidate(3)
    held=sequence.candidate(2).basis;before=shell_data(held);del sequence;gc.collect()
    assert shell_data(held)==before


@pytest.mark.parametrize('field,quantity',[('memory_bytes','modeled_peak_bytes'),('work_units','work_units'),
    ('maximum_shells','maximum_shells'),('maximum_functions','maximum_functions')])
def test_sequence_aggregate_limits(field,quantity):
    from vibeqc.auxiliary_generation import plan_product_auxiliary_sequence,prepare_product_auxiliary_sequence
    mol,ao,seed=fixture();plan=plan_product_auxiliary_sequence(mol,ao,seed)
    limits=ProductAuxiliaryLimits();setattr(limits,field,getattr(plan,quantity)-1)
    with pytest.raises((ValueError,OverflowError),match='budget'):prepare_product_auxiliary_sequence(mol,ao,seed,limits=limits)
    setattr(limits,field,getattr(plan,quantity));assert prepare_product_auxiliary_sequence(mol,ao,seed,limits=limits).plan.candidates==3


def test_sequence_refinement_limit():
    from vibeqc.auxiliary_generation import plan_product_auxiliary_sequence
    mol,ao,seed=fixture();o=ProductAuxiliaryOptions();o.refinement=7
    with pytest.raises(ValueError,match='level eight'):plan_product_auxiliary_sequence(mol,ao,seed,o,2)
