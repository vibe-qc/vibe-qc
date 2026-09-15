"""Exact physical product domains through native LR and overlap components.

Tiny source-development witnesses, not production HF/correlation validation.
Geometry decisions, polynomial FT and native one-electron overlap have separate
references. Numerical production integrals and LR contractions remain native.
"""
import itertools

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from vibeqc._bipole_physical_support import build_product_support, plan_product_support
from vibeqc.symmetry_shared import Budget
from tests.test_bipole_ewald_gram import (
    _cell_call, _cell_oracle, _gram, _gram_bundle, _gram_records, _mixed_basis,
)
from tests.test_bipole_finite_source import (
    _exact_phase, _hf_overlap, _mapped, _mapped_bundle, _mapped_controls,
    _physical_centers, _physical_support,
)


def _product(lattice, centers, radius=2.1, *, plan=False, **overrides):
    kw = dict(pair_radius=radius, budget=Budget(8 << 20, 10**10),
              maximum_candidates=100000, maximum_cells=4096)
    kw.update(overrides)
    fn = plan_product_support if plan else build_product_support
    return fn(np.ascontiguousarray(lattice, dtype=float),
              np.ascontiguousarray(centers, dtype=float), **kw)


def _labels(cells):
    return {tuple(int(x) for x in row) for row in cells}


def _product_gram(b, left, right, *, plan=False):
    fn = core._plan_bipole_ewald_product_gram if plan else core._make_bipole_ewald_product_gram
    return fn(b.basis, b.system, b.mesh, left, right,
              b.selection, b.options, b.inventory, b.caps)


def _product_gram_oracle(b, left_cells, right_cells):
    p, weights = _gram_records(b)
    reciprocal = 2*np.pi*np.linalg.inv(np.asarray(b.system.lattice)).T
    s = b.selection
    left = _cell_oracle(b.basis, p, np.asarray(b.system.lattice),
                        -reciprocal @ b.mesh.fractional_at(s.left_k_index), left_cells)
    right = _cell_oracle(b.basis, p, np.asarray(b.system.lattice),
                         -reciprocal @ b.mesh.fractional_at(s.right_k_index), right_cells)
    return (left[s.left_pair_begin:s.left_pair_begin+s.left_pair_count].conj()*weights) @ (
        right[s.right_pair_begin:s.right_pair_begin+s.right_pair_count].T)


@pytest.mark.parametrize('lattice', [np.diag([3.,4.,5.]),
    np.array([[3.,.5,-.25],[0.,4.,.5],[.25,0.,5.]]),
    np.array([[-3.,.5,-.25],[0.,4.,.5],[.25,0.,5.]])])
def test_product_exact_cells_match_independent_exhaustive_cartesian_cube(lattice):
    centers = np.array([[.25,-.5,.75],[-.5,.25,-.25]])
    result = _product(lattice, centers, 3.5)
    expected = set()
    for label in itertools.product(range(-3,4), repeat=3):
        delta = centers[0]-centers[1]-lattice@label
        if np.dot(delta,delta) <= 3.5**2:
            expected.add(label)
    assert _labels(result.cells) == expected
    assert len(result.cells) == len(expected) > 1
    assert list(map(tuple,result.cells)) == sorted(expected)
    assert not result.physical_hamiltonian_certified and not result.symmetry_certified


def test_product_reversal_and_large_reanchoring_without_wrapping():
    lattice = np.diag([4.,5.,3.])
    centers = np.array([[.25,.5,0.],[-.25,-.5,1.5]])
    original = _product(lattice, centers)
    reverse = _product(lattice, centers[::-1])
    assert _labels(reverse.cells) == _labels(-original.cells)
    assert _labels(original.cells) != _labels(-original.cells)
    shifts = np.array([[1000,-2000,3000],[-13,7,6]])
    moved = _product(lattice, centers+shifts@lattice.T)
    assert _labels(moved.cells) == _labels(original.cells+shifts[0]-shifts[1])


def test_product_cutoff_boundary_and_zero_radius_are_exact():
    a, xyz = np.eye(3), np.zeros((2,3))
    assert len(_product(a,xyz,0).cells) == 1
    assert len(_product(a,xyz,np.nextafter(1.,0)).cells) == 1
    assert len(_product(a,xyz,1).cells) == 7
    assert len(_product(a,xyz,np.nextafter(1.,2)).cells) == 7
    xyz[1,0] = .5
    assert len(_product(a,xyz,0).cells) == 0


def test_quartet_axes_use_the_same_product_predicate():
    b = _mapped_bundle()
    for quartet in itertools.product(range(2),repeat=4):
        xyz = _physical_centers(b,quartet)
        left = _product(b.system.lattice,xyz[:2])
        right = _product(b.system.lattice,xyz[2:])
        images = _physical_support(b.system.lattice,xyz,midpoint_radius=4.0).images
        assert _labels(images[:,0]) <= _labels(left.cells)
        assert _labels(images[:,2]-images[:,1]) <= _labels(right.cells)
        # Radius 4 exceeds this orthorhombic cell's covering radius,
        # so every allowed g,h has at least one retained p.
        assert {(tuple(g),tuple(s-p)) for g,p,s in images} == {
            (tuple(g),tuple(h)) for g in left.cells for h in right.cells}


@pytest.mark.parametrize('which',['bytes','work','candidates','cells'])
def test_product_exact_and_minus_one_resource_caps(which):
    a, xyz = np.eye(3)*2, np.zeros((2,3))
    plan = _product(a,xyz,plan=True)
    kw = dict(budget=Budget(plan.inventoried_bytes,plan.work_units),
              maximum_candidates=plan.candidate_count,maximum_cells=plan.cell_count)
    exact = _product(a,xyz,**kw)
    assert exact.memory == plan
    if which == 'bytes': kw['budget'] = Budget(plan.inventoried_bytes-1,plan.work_units)
    elif which == 'work': kw['budget'] = Budget(plan.inventoried_bytes,plan.work_units-1)
    elif which == 'candidates': kw['maximum_candidates'] -= 1
    else: kw['maximum_cells'] -= 1
    with pytest.raises((MemoryError,ValueError)):
        _product(a,xyz,**kw)


def test_product_identity_snapshot_and_execution_caps():
    a, xyz = np.eye(3)*3, np.zeros((2,3))
    first = _product(a,xyz)
    second = _product(a,xyz,maximum_cells=100)
    assert first.memory.support_identity_sha256 == second.memory.support_identity_sha256
    with pytest.raises(ValueError): first.cells.flags.writeable = True
    xyz[1,0] = .25
    changed = _product(a,xyz)
    assert first.memory.support_identity_sha256 != changed.memory.support_identity_sha256
    np.testing.assert_array_equal(first.cells,[[0,0,0]])


@pytest.mark.parametrize('change',['shape','list','float32','strided','unaligned','bool_radius','bool_cap','nan','singular'])
def test_product_refuses_invalid_inputs(change):
    a, xyz = np.eye(3)*3, np.zeros((2,3))
    kw = dict(pair_radius=2.1,budget=Budget(8<<20,10**10),maximum_candidates=100000,maximum_cells=4096)
    if change == 'shape': xyz = np.zeros((4,3))
    if change == 'list': xyz = xyz.tolist()
    if change == 'float32': xyz = xyz.astype(np.float32)
    if change == 'strided': a = a.T
    if change == 'unaligned': xyz = np.ndarray((2,3),dtype=float,buffer=bytearray(49),offset=1)
    if change == 'bool_radius': kw['pair_radius'] = True
    if change == 'bool_cap': kw['maximum_cells'] = True
    if change == 'nan': xyz[0,0] = np.nan
    if change == 'singular': a[:] = 0
    with pytest.raises((ValueError,TypeError)):
        build_product_support(a,xyz,**kw)


@pytest.mark.parametrize('q,shift,mixed',[(0,(0,0,0),False),(1,(1,0,0),False),(2,(1,0,0),True)])
def test_native_independent_product_gram_matches_polynomial_oracle(q,shift,mixed):
    b = _gram_bundle(q=q,shift=shift,basis=_mixed_basis() if mixed else None)
    b.options.require_cell_inversion_closure = False
    if mixed:
        b.selection.left_pair_begin,b.selection.left_pair_count = 5,7
        b.selection.right_pair_begin,b.selection.right_pair_count = 13,5
    left = np.array([[0,0,0],[1,0,0]],np.int64)
    right = np.array([[0,0,0],[0,-1,0],[0,0,1]],np.int64)
    result = _product_gram(b,left,right)
    np.testing.assert_allclose(result.values_copy(),_product_gram_oracle(b,left,right),atol=5e-13,rtol=3e-12)
    assert (result.memory.left_cell_count,result.memory.right_cell_count) == (2,3)
    assert result.memory.borrowed_cell_bytes == 5*24
    assert not result.diagnostics.cell_inversion_closed
    assert not result.physical_hamiltonian_certified and not result.symmetry_certified
    # A shared union silently changes the operator.
    b.cells = np.array(sorted(_labels(left)|_labels(right)),np.int64)
    assert np.max(np.abs(_gram(b).values_copy()-result.values_copy())) > 1e-5


def test_native_shared_adapter_preserves_identity_values_and_alias_inventory():
    b = _gram_bundle()
    legacy = _gram(b)
    alias = _product_gram(b,b.cells,b.cells)
    detached = _product_gram(b,b.cells,b.cells.copy())
    for result in [alias,detached]:
        np.testing.assert_array_equal(result.values_copy(),legacy.values_copy())
        assert result.input_identity_sha256 == legacy.input_identity_sha256
        assert result.payload_identity_sha256 == legacy.payload_identity_sha256
    assert alias.memory.shared_cell_storage and not detached.memory.shared_cell_storage
    assert detached.memory.borrowed_cell_bytes == 2*alias.memory.borrowed_cell_bytes
    assert detached.memory.work_units_upper_bound > alias.memory.work_units_upper_bound


def test_native_right_product_changes_scientific_identity_and_values():
    b = _gram_bundle()
    b.options.require_cell_inversion_closure = False
    left, right = b.cells, b.cells[:2].copy()
    first = _product_gram(b,left,right)
    right[1] = [0,0,1]
    second = _product_gram(b,left,right)
    assert first.input_identity_sha256 != second.input_identity_sha256
    assert first.reciprocal_source_identity_sha256 == second.reciprocal_source_identity_sha256
    assert np.max(np.abs(first.values_copy()-second.values_copy())) > 1e-5
    assert first.diagnostics.left_cell_inversion_closed
    assert not first.diagnostics.right_cell_inversion_closed
    b.options.require_cell_inversion_closure = True
    with pytest.raises(ValueError,match='inversion'):
        _product_gram(b,left,right)


@pytest.mark.parametrize('side',[0,1])
@pytest.mark.parametrize('bad',['duplicate','overflow','float','strided','unaligned'])
def test_native_both_product_payloads_validated_even_for_empty_output(side,bad):
    b = _gram_bundle()
    b.selection.left_pair_count = 0
    b.options.require_cell_inversion_closure = False
    cells = [b.cells.copy(),b.cells.copy()]
    if bad == 'duplicate': cells[side][1] = cells[side][0]
    elif bad == 'overflow': cells[side][0,0] = 2**53+1
    elif bad == 'float': cells[side] = cells[side].astype(float)
    elif bad == 'strided': cells[side] = cells[side][::2]
    else:
        cells[side] = np.ndarray(b.cells.shape,dtype=np.int64,buffer=bytearray(b.cells.nbytes+1),offset=1)
    with pytest.raises((ValueError,TypeError)):
        _product_gram(b,*cells)


@pytest.mark.parametrize('side',[0,1])
def test_native_empty_product_gives_zero_without_substitution(side):
    b = _gram_bundle()
    cells = [b.cells,np.empty((0,3),np.int64)]
    if side == 0: cells.reverse()
    result = _product_gram(b,*cells)
    np.testing.assert_array_equal(result.values_copy(),np.zeros((4,4)))
    assert result.diagnostics.accepted_vectors > 0


@pytest.mark.parametrize('field,measured',[
    ('maximum_borrowed_numerical_bytes','borrowed_numerical_bytes'),
    ('maximum_owned_numerical_bytes','peak_owned_numerical_bytes'),
    ('maximum_control_storage_bytes','control_storage_bytes'),
    ('maximum_per_replica_inventoried_bytes','per_replica_inventoried_bytes'),
    ('maximum_node_inventoried_bytes','required_node_inventoried_bytes'),
    ('maximum_work_units','work_units_upper_bound'),
])
def test_native_product_outer_caps_exact_and_minus_one_precede_payload(field,measured):
    b = _gram_bundle()
    b.options.require_cell_inversion_closure = False
    left,right = b.cells,b.cells[:2].copy()
    plan = _product_gram(b,left,right,plan=True)
    setattr(b.caps,field,getattr(plan,measured))
    _product_gram(b,left,right)
    setattr(b.caps,field,getattr(plan,measured)-1)
    right[0,0] = 2**53+1
    with pytest.raises((ValueError,RuntimeError),match='cap'):
        _product_gram(b,left,right)


@pytest.mark.parametrize('side',[0,1])
def test_native_each_product_leaf_cap_precedes_invalid_label(side):
    b = _gram_bundle()
    b.options.require_cell_inversion_closure = False
    cells = [np.zeros((1,3),np.int64),np.array([[0,0,0],[0,0,1]],np.int64)]
    if side == 0: cells.reverse()
    b.caps.ao_panel.maximum_cells = 1
    cells[side][0,0] = 2**53+1
    with pytest.raises((ValueError,RuntimeError),match='cap'):
        _product_gram(b,*cells)


def _screw_product_fixture(mesh,shift):
    mapped = _mapped_bundle(mesh=mesh,shift=shift)
    b = _gram_bundle(mesh=mesh,shift=shift,q=0,left=0,right=0,basis=mapped.basis)
    b.system = mapped.system
    b.options.omega = .6
    b.options.require_cell_inversion_closure = False
    b.selection.left_pair_count = b.selection.right_pair_count = 1
    supports = {}
    for a,c in itertools.product(range(2),repeat=2):
        xyz = _physical_centers(mapped,(a,c))
        supports[a,c] = _product(b.system.lattice,xyz).cells
    return mapped,b,supports


@pytest.mark.parametrize('mesh,shift',[
    ((1,1,1),(0,0,0)),((1,1,2),(0,0,0)),((1,1,3),(0,0,0)),
    ((1,1,3),(0,0,1)),((2,2,2),(1,1,1)),
])
def test_physical_product_lr_screw_covariance_and_independent_fourier_oracle(mesh,shift):
    mapped,b,supports = _screw_product_fixture(mesh,shift)
    nk = len(b.mesh)
    rotation = np.diag([-1,-1,1])
    moduli = b.mesh.doubled_modulus
    def moved_index(index,transfer=False):
        raw = b.mesh.transfer_address(index) if transfer else b.mesh.address(index)
        label = [(-raw[0])%moduli[0],(-raw[1])%moduli[1],raw[2]]
        return b.mesh.index_of_transfer(label) if transfer else b.mesh.index(label)
    mappings = {}
    for quartet in itertools.product(range(2),repeat=4):
        receipt = _mapped(mapped,controls=_mapped_controls(quartet))
        dest = tuple(receipt.destination_shells)
        ell = np.array(receipt.atom_shifts).reshape(4,3)
        for i,j in [(0,1),(2,3)]:
            assert _labels(supports[quartet[i],quartet[j]]@rotation+ell[i]-ell[j]) == _labels(supports[dest[i],dest[j]])
        mappings[quartet] = dest,ell[0]-ell[1:]
    # All q,kL,kR for Nk<=3; all q and one selected kL,kR for the 8-point
    # multiaxis witness. Neither fixture establishes a production operator.
    channels = list(itertools.product(range(nk),repeat=3)) if nk<=3 else [(q,0,nk-1) for q in range(nk)]
    reciprocal = 2*np.pi*np.linalg.inv(np.asarray(b.system.lattice)).T
    oracle = {}
    for q in range(nk):
        b.selection.q_index = q
        vectors,weights = _gram_records(b)
        factors = {}
        for k in range(nk):
            for pair,cells in supports.items():
                factors[k,pair] = _cell_oracle(b.basis,vectors,np.asarray(b.system.lattice),
                    -reciprocal@b.mesh.fractional_at(k),cells)[2*pair[0]+pair[1]]
        oracle[q] = weights,factors
    max_imaginary = 0
    for q,kl,kr in channels:
        weights,factors = oracle[q]
        for quartet,(dest,offset) in mappings.items():
            b.selection.q_index,b.selection.left_k_index,b.selection.right_k_index = q,kl,kr
            b.selection.left_pair_begin = 2*quartet[0]+quartet[1]
            b.selection.right_pair_begin = 2*quartet[2]+quartet[3]
            result = _product_gram(b,supports[quartet[:2]],supports[quartet[2:]])
            value = result.element(0,0)
            expected = np.dot(factors[kl,quartet[:2]].conj()*weights,factors[kr,quartet[2:]])
            np.testing.assert_allclose(value,expected,atol=6e-13,rtol=4e-12)
            qt,klt,krt = moved_index(q,True),moved_index(kl),moved_index(kr)
            b.selection.q_index,b.selection.left_k_index,b.selection.right_k_index = qt,klt,krt
            b.selection.left_pair_begin,b.selection.right_pair_begin = 2*dest[0]+dest[1],2*dest[2]+dest[3]
            transformed = _product_gram(b,supports[dest[:2]],supports[dest[2:]]).element(0,0)
            phase = complex(_exact_phase(offset,mesh,shift,qt,klt,krt))
            np.testing.assert_allclose(transformed,phase*value,atol=6e-13,rtol=4e-12)
            max_imaginary = max(max_imaginary,abs(phase.imag))
    if mesh == (1,1,3): assert max_imaginary > .8


@pytest.mark.parametrize('mesh,shift',[
    ((1,1,1),(0,0,0)),((1,1,2),(0,0,0)),((1,1,3),(0,0,0)),
    ((1,1,3),(0,0,1)),((2,2,2),(1,1,1)),
])
def test_physical_product_fourier_zero_matches_pair_selected_native_hf_overlap(mesh,shift):
    mapped,b,supports = _screw_product_fixture(mesh,shift)
    reciprocal = 2*np.pi*np.linalg.inv(np.asarray(b.system.lattice)).T
    nk = len(b.mesh)
    overlap = np.empty((nk,2,2),complex)
    fourier = np.empty_like(overlap)
    for k in range(nk):
        for (a,c),cells in supports.items():
            overlap[k,a,c] = _hf_overlap(mapped,k,cells=cells)[a,c]
            fourier[k,a,c] = _cell_call(b.basis,[[0,0,0]],cells,lattice=b.system.lattice,
                k=-reciprocal@b.mesh.fractional_at(k),begin=2*a+c,count=1)['values'][0,0]
    # Pair reversal, rather than individual-list inversion, yields both
    # overlap identities needed by the G=0 J/K subtraction on these domains.
    np.testing.assert_allclose(overlap,fourier.conj(),atol=6e-13,rtol=4e-12)
    np.testing.assert_allclose(overlap,fourier.transpose(0,2,1),atol=6e-13,rtol=4e-12)
    coefficient = -np.pi/(abs(np.linalg.det(b.system.lattice))*b.options.omega**2*nk)
    for k,c,d,imaginary in itertools.product(range(nk),range(2),range(2),[False,True]):
        density = np.zeros_like(overlap)
        density[k,c,d] = 1j if imaginary else 1
        # This tests the factor identities, not a production zero-mode API.
        j_factors = coefficient*fourier.conj()*np.einsum('kab,kab->',fourier,density)
        j_overlap = coefficient*overlap*np.einsum('kab,kba->',density,overlap)
        k_factors = coefficient*np.einsum('kac,kbd,kcd->kab',fourier.conj(),fourier,density)
        k_overlap = coefficient*np.einsum('kac,kcd,kdb->kab',overlap,density,overlap)
        np.testing.assert_allclose(j_factors,j_overlap,atol=8e-13,rtol=5e-12)
        np.testing.assert_allclose(k_factors,k_overlap,atol=8e-13,rtol=5e-12)


def test_native_product_swap_is_adjoint_and_streaming_preserves_identity():
    b = _gram_bundle(shift=(1,0,0))
    b.options.require_cell_inversion_closure = False
    left,right = b.cells[:2].copy(),b.cells[2:].copy()
    original = _product_gram(b,left,right)
    b.options.reciprocal_block_size = 1
    tiled = _product_gram(b,left,right)
    np.testing.assert_array_equal(tiled.values_copy(),original.values_copy())
    assert tiled.input_identity_sha256 == original.input_identity_sha256
    assert tiled.payload_identity_sha256 == original.payload_identity_sha256
    b.selection.left_k_index,b.selection.right_k_index = b.selection.right_k_index,b.selection.left_k_index
    swapped = _product_gram(b,right,left)
    np.testing.assert_allclose(swapped.values_copy(),original.values_copy().conj().T,atol=5e-13,rtol=3e-12)


def test_native_overlapping_product_views_are_conservatively_inventoried():
    b = _gram_bundle()
    b.options.require_cell_inversion_closure = False
    left,right = b.cells[:3],b.cells[2:]
    plan = _product_gram(b,left,right,plan=True)
    assert not plan.shared_cell_storage
    assert plan.borrowed_cell_bytes == left.nbytes+right.nbytes
    result = _product_gram(b,left,right)
    copied = _product_gram(b,left.copy(),right.copy())
    assert result.input_identity_sha256 == copied.input_identity_sha256
    np.testing.assert_array_equal(result.values_copy(),copied.values_copy())


# Composed native finite sources with immutable left/right AO-domain ownership.
def _finite_product_fixture(quartet=(0,1,1,0), *, mesh=(1,1,3), shift=(0,0,0)):
    b = _mapped_bundle(mesh=mesh,shift=shift)
    xyz = _physical_centers(b,quartet)
    b.images = _physical_support(b.system.lattice,xyz).images
    b.cells = _product(b.system.lattice,xyz[:2]).cells
    b.right_cells = _product(b.system.lattice,xyz[2:]).cells
    b.selection.left_pair_begin,b.selection.left_pair_count = 2*quartet[0]+quartet[1],1
    b.selection.right_pair_begin,b.selection.right_pair_count = 2*quartet[2]+quartet[3],1
    b.options.omega = .6
    b.options.require_image_permutation_closure = False
    b.options.require_cell_inversion_closure = False
    return b


def _finite_domain(b):
    domain = core._BipoleFiniteProductDomain()
    for field in ['left_pair_begin','left_pair_count','right_pair_begin','right_pair_count']:
        setattr(domain,field,getattr(b.selection,field))
    return domain


def _finite_product_source(b, *, domain=None, plan=False):
    d = _finite_domain(b) if domain is None else domain
    args = (b.basis,b.mesh,b.images,b.cells,b.right_cells,d,b.options,b.source_caps)
    if plan: return core._plan_bipole_finite_product_source(*args)
    return core._make_bipole_finite_product_source(b.basis,b.system,*args[1:])


def _finite_product_panel(b, source=None, *, plan=False):
    fn = core._plan_bipole_finite_product_panel if plan else core._make_bipole_finite_product_panel
    return fn(_finite_product_source(b) if source is None else source,b.basis,
              b.images,b.cells,b.right_cells,b.selection,b.inventory,b.caps)


def _finite_product_components(b, raw=None):
    from tests.test_bipole_finite_source import _erfc_oracle
    s = b.selection
    left = (s.left_pair_begin,s.left_pair_count)
    right = (s.right_pair_begin,s.right_pair_count)
    if raw is None:
        raw = _erfc_oracle(b.basis,b.images,lattice=np.asarray(b.system.lattice),
                           left=left,right=right,omega=b.options.omega)
    sr = np.zeros((left[1],right[1]),complex)
    for image,value in zip(b.images,raw):
        phase = complex(_exact_phase(image,tuple(b.mesh.mesh),tuple(b.mesh.is_shift),
                                      s.q_index,s.left_k_index,s.right_k_index))
        sr += phase*value
    lr = _product_gram_oracle(b,b.cells,b.right_cells)
    zero = np.zeros_like(sr)
    if s.q_index == 0:
        a = np.asarray(b.system.lattice)
        reciprocal = 2*np.pi*np.linalg.inv(a).T
        bl = _cell_oracle(b.basis,np.zeros((1,3)),a,
            -reciprocal@b.mesh.fractional_at(s.left_k_index),b.cells)[left[0]:sum(left),0]
        br = _cell_oracle(b.basis,np.zeros((1,3)),a,
            -reciprocal@b.mesh.fractional_at(s.right_k_index),b.right_cells)[right[0]:sum(right),0]
        zero = np.pi/(abs(np.linalg.det(a))*b.options.omega**2)*np.outer(bl.conj(),br)
    return sr,lr,zero


@pytest.mark.parametrize('mesh,shift',[
    ((1,1,1),(0,0,0)),((1,1,2),(0,0,0)),((1,1,3),(0,0,0)),
    ((1,1,3),(0,0,1)),((2,2,2),(1,1,1)),
])
def test_composed_product_source_full_components_and_screw_covariance(mesh,shift):
    from tests.test_bipole_finite_source import _erfc_oracle
    mapped = _mapped_bundle(mesh=mesh,shift=shift)
    nk = len(mapped.mesh)
    fixtures = {}
    for quartet in itertools.product(range(2),repeat=4):
        b = _finite_product_fixture(quartet,mesh=mesh,shift=shift)
        source = _finite_product_source(b)
        assert source.product_resolved
        assert not source.physical_hamiltonian_certified and not source.symmetry_certified
        raw = _erfc_oracle(b.basis,b.images,lattice=np.asarray(b.system.lattice),
            left=(b.selection.left_pair_begin,1),right=(b.selection.right_pair_begin,1),omega=.6)
        receipt = _mapped(mapped,controls=_mapped_controls(quartet))
        ell = np.array(receipt.atom_shifts).reshape(4,3)
        fixtures[quartet] = b,source,raw,tuple(receipt.destination_shells),ell[0]-ell[1:]
    moduli = mapped.mesh.doubled_modulus
    def moved_index(index,transfer=False):
        raw = mapped.mesh.transfer_address(index) if transfer else mapped.mesh.address(index)
        label = [(-raw[0])%moduli[0],(-raw[1])%moduli[1],raw[2]]
        return mapped.mesh.index_of_transfer(label) if transfer else mapped.mesh.index(label)
    channels = list(itertools.product(range(nk),repeat=3)) if nk<=3 else [(q,0,nk-1) for q in range(nk)]
    for q,kl,kr in channels:
        for b,source,raw,dest,offset in fixtures.values():
            b.selection.q_index,b.selection.left_k_index,b.selection.right_k_index = q,kl,kr
            result = _finite_product_panel(b,source)
            sr,lr,zero = _finite_product_components(b,raw)
            np.testing.assert_allclose(result.values_copy(),sr+lr-zero,atol=2e-12,rtol=6e-12)
            assert result.memory.subtract_zero_mode == (q == 0)
            assert result.diagnostics.zero_mode_pair_values == (2 if q==0 else 0)
            target,target_source,*_ = fixtures[dest]
            qt,klt,krt = moved_index(q,True),moved_index(kl),moved_index(kr)
            target.selection.q_index,target.selection.left_k_index,target.selection.right_k_index = qt,klt,krt
            transformed = _finite_product_panel(target,target_source).element(0,0)
            phase = complex(_exact_phase(offset,mesh,shift,qt,klt,krt))
            np.testing.assert_allclose(transformed,phase*result.element(0,0),atol=2e-12,rtol=6e-12)


def test_composed_source_binds_domains_supports_and_preserves_source_lifetime():
    import gc
    b = _finite_product_fixture()
    d = _finite_domain(b)
    source = _finite_product_source(b,domain=d)
    identity = source.source_identity_sha256
    d.left_pair_begin = 0
    copy = source.product_domain
    copy.right_pair_begin = 0
    assert source.product_domain.left_pair_begin == 1
    assert source.product_domain.right_pair_begin == 2
    b.selection.left_pair_begin = 0
    changed = _finite_product_source(b)
    assert changed.source_identity_sha256 != identity
    b.selection.left_pair_begin = 1
    result = _finite_product_panel(b,source)
    values = result.values_copy()
    del source,b
    gc.collect()
    assert result.source.source_identity_sha256 == identity
    np.testing.assert_array_equal(result.values_copy(),values)
    with pytest.raises(ValueError): values[0,0] = 0


@pytest.mark.parametrize('axis',['left','right'])
@pytest.mark.parametrize('change',['before','after','overrun','overflow'])
def test_composed_source_refuses_out_of_domain_tile_before_payload(axis,change):
    b = _finite_product_fixture()
    source = _finite_product_source(b)
    begin = getattr(b.selection,axis+'_pair_begin')
    if change == 'before': setattr(b.selection,axis+'_pair_begin',begin-1)
    if change == 'after': setattr(b.selection,axis+'_pair_begin',begin+1)
    if change == 'overrun': setattr(b.selection,axis+'_pair_count',2)
    if change == 'overflow': setattr(b.selection,axis+'_pair_begin',2**64-1)
    b.right_cells = np.array([[2**53+1,0,0]],np.int64)
    with pytest.raises(ValueError,match='domain'):
        _finite_product_panel(b,source)


@pytest.mark.parametrize('axis',['left','right'])
@pytest.mark.parametrize('change',['zero','outside','overflow'])
def test_composed_source_refuses_invalid_domain(axis,change):
    b = _finite_product_fixture()
    domain = _finite_domain(b)
    if change == 'zero': setattr(domain,axis+'_pair_count',0)
    if change == 'outside': setattr(domain,axis+'_pair_begin',4)
    if change == 'overflow': setattr(domain,axis+'_pair_count',2**64-1)
    with pytest.raises(ValueError,match='domain'):
        _finite_product_source(b,domain=domain)


@pytest.mark.parametrize('change',['left','right','images','basis'])
@pytest.mark.parametrize('empty',[False,True])
def test_composed_source_revalidates_every_payload_even_with_empty_output(change,empty):
    from tests.test_periodic_aopair_fourier_panel import _basis
    b = _finite_product_fixture()
    source = _finite_product_source(b)
    if change == 'left': b.cells = np.array([[0,0,0]],np.int64)
    if change == 'right': b.right_cells = np.array([[0,0,0]],np.int64)
    if change == 'images': b.images = np.zeros((1,3,3),np.int64)
    if change == 'basis':
        b.basis = _basis([(0,(.2,.4,0.),[.21],[.7],True),(0,(-.2,-.4,1.5),[.2],[.7],True)])
    if empty: b.selection.left_pair_count = 0
    with pytest.raises(ValueError,match='immutable source'):
        _finite_product_panel(b,source)


@pytest.mark.parametrize('field,measured',[
    ('maximum_context_storage_bytes','context_storage_bytes'),
    ('maximum_borrowed_numerical_bytes','borrowed_numerical_bytes'),
    ('maximum_work_units','work_units_upper_bound'),
])
def test_composed_source_exact_and_minus_one_caps_precede_payload(field,measured):
    b = _finite_product_fixture()
    plan = _finite_product_source(b,plan=True)
    setattr(b.source_caps,field,getattr(plan,measured))
    _finite_product_source(b)
    setattr(b.source_caps,field,getattr(plan,measured)-1)
    b.right_cells = b.right_cells.copy()
    b.right_cells[0,0] = 2**53+1
    with pytest.raises((ValueError,RuntimeError),match='cap'):
        _finite_product_source(b)


@pytest.mark.parametrize('field,measured',[
    ('maximum_borrowed_numerical_bytes','borrowed_numerical_bytes'),
    ('maximum_owned_numerical_bytes','peak_owned_numerical_bytes'),
    ('maximum_control_storage_bytes','control_storage_bytes'),
    ('maximum_worker_bytes','per_replica_inventoried_bytes'),
    ('maximum_node_bytes','required_node_inventoried_bytes'),
    ('maximum_work_units','work_units_upper_bound'),
])
def test_composed_panel_exact_and_minus_one_caps_precede_bad_right_support(field,measured):
    b = _finite_product_fixture()
    source = _finite_product_source(b)
    plan = _finite_product_panel(b,source,plan=True)
    setattr(b.caps,field,getattr(plan,measured))
    _finite_product_panel(b,source)
    setattr(b.caps,field,getattr(plan,measured)-1)
    b.right_cells = b.right_cells.copy();b.right_cells[0,0] = 2**53+1
    with pytest.raises((ValueError,RuntimeError),match='cap'):
        _finite_product_panel(b,source)


def test_composed_source_deduplicates_alias_storage_but_not_identity():
    b = _finite_product_fixture()
    b.right_cells = b.cells
    alias = _finite_product_source(b)
    b.right_cells = b.cells.copy()
    copied = _finite_product_source(b)
    assert copied.source_identity_sha256 == alias.source_identity_sha256
    assert copied.memory.borrowed_cell_bytes == 2*alias.memory.borrowed_cell_bytes
    np.testing.assert_array_equal(_finite_product_panel(b,alias).values_copy(),
                                   _finite_product_panel(b,copied).values_copy())


def test_composed_domain_allows_subtiles_with_distinct_product_lists():
    from tests.test_bipole_finite_source import _bundle
    b = _bundle()
    b.options.require_cell_inversion_closure = False
    b.cells = np.array([[0,0,0],[1,0,0]],np.int64)
    b.right_cells = np.array([[0,0,0],[0,0,1],[0,0,-1]],np.int64)
    source = _finite_product_source(b)
    whole = _finite_product_panel(b,source)
    b.selection.left_pair_begin,b.selection.left_pair_count = 1,2
    b.selection.right_pair_begin,b.selection.right_pair_count = 2,2
    selected = _finite_product_panel(b,source)
    np.testing.assert_allclose(selected.values_copy(),whole.values_copy()[1:3,2:4],atol=4e-13,rtol=3e-12)
    b.inventory.reciprocal_block_size = 1
    tiled = _finite_product_panel(b,source)
    np.testing.assert_array_equal(tiled.values_copy(),selected.values_copy())
    assert tiled.input_identity_sha256 == selected.input_identity_sha256
    assert tiled.payload_identity_sha256 == selected.payload_identity_sha256


@pytest.mark.parametrize('consumer',['panel','support','overlap','jk'])
def test_product_source_cannot_enter_common_support_consumers(consumer):
    from tests.test_bipole_finite_source import _panel,_support,_overlap_audit,_jk,_jk_bundle
    b = _finite_product_fixture()
    source = _finite_product_source(b)
    with pytest.raises(ValueError,match='product|common source'):
        if consumer == 'panel': _panel(b,source,plan=True)
        elif consumer == 'support': _support(b,source=source,plan=True)
        elif consumer == 'overlap': _overlap_audit(b,np.zeros((2,2),complex),source=source,plan=True)
        else:
            jk = _jk_bundle()
            _jk(jk,np.zeros((len(jk.mesh),2,2),complex),source=source,plan=True)


@pytest.mark.parametrize('axis',['left','right'])
def test_composed_empty_product_with_empty_sr_is_zero(axis):
    b = _finite_product_fixture()
    b.images = np.empty((0,3,3),np.int64)
    if axis == 'left': b.cells = np.empty((0,3),np.int64)
    else: b.right_cells = np.empty((0,3),np.int64)
    result = _finite_product_panel(b)
    assert result.element(0,0) == 0
    assert result.diagnostics.maximum_zero_mode_magnitude == 0


@pytest.mark.parametrize('q',[0,1])
def test_composed_zero_mode_uses_right_product_child_admission(q):
    b = _finite_product_fixture()
    b.cells = np.zeros((1,3),np.int64)
    b.right_cells = np.array([[0,0,0],[0,0,1]],np.int64)
    b.selection.q_index = q
    source = _finite_product_source(b)
    b.caps.zero_mode.maximum_cells = 1
    if q == 0:
        with pytest.raises((ValueError,RuntimeError),match='cap'):
            _finite_product_panel(b,source)
    else:
        result = _finite_product_panel(b,source)
        assert not result.memory.subtract_zero_mode
        assert result.memory.zero_mode_workspace_bytes == 0


def test_composed_multiple_replica_census_includes_both_supports_and_outer_output():
    b = _finite_product_fixture()
    source = _finite_product_source(b)
    first = _finite_product_panel(b,source,plan=True)
    b.inventory.numerical_replicas = 2
    b.inventory.external_node_bytes = 4567
    b.inventory.other_live_numerical_bytes_per_replica = 1234
    second = _finite_product_panel(b,source,plan=True)
    assert second.borrowed_numerical_bytes == first.borrowed_numerical_bytes
    assert second.source.borrowed_cell_bytes == b.cells.nbytes+b.right_cells.nbytes
    assert second.per_replica_inventoried_bytes == first.per_replica_inventoried_bytes+1234
    assert second.required_node_inventoried_bytes == 2*second.per_replica_inventoried_bytes+4567
    assert second.peak_owned_numerical_bytes >= second.retained_output_bytes+second.compensation_bytes+second.zero_mode_workspace_bytes


@pytest.mark.parametrize('kind',['list','float','strided','unaligned'])
def test_composed_source_right_binding_does_not_coerce_or_copy(kind):
    b = _finite_product_fixture()
    if kind == 'list': b.right_cells = b.right_cells.tolist()
    if kind == 'float': b.right_cells = b.right_cells.astype(float)
    if kind == 'strided': b.right_cells = np.zeros((4,3),np.int64)[::2]
    if kind == 'unaligned':
        b.right_cells = np.ndarray((2,3),dtype=np.int64,buffer=bytearray(49),offset=1)
    with pytest.raises((ValueError,TypeError)):
        _finite_product_source(b)


# One common basis/geometry policy over the complete ordered AO quartet space.
def _physical_owner(b=None, **overrides):
    from vibeqc._bipole_physical_source import make_physical_source
    b = _finite_product_fixture() if b is None else b
    kw = dict(pair_radius=2.1, midpoint_radius=1.6, maximum_candidates=100000,
              maximum_cells=64, maximum_images=64, maximum_quartets=b.basis.nbasis**4,
              budget=Budget(16<<20,10**11))
    kw.update(overrides)
    return make_physical_source(b.basis,b.system,b.mesh,b.options,b.source_caps,**kw)


def _owned_domain(owner, quartet, *, plan=False, **kw):
    budget = kw.pop('budget',Budget(16<<20,10**11))
    assert not kw
    fn = owner.plan_domain if plan else owner.build_domain
    return fn(quartet,budget=budget)


def _use_domain(b, domain):
    b.basis = domain.owner.basis
    b.images,b.cells,b.right_cells = domain.images,domain.left_cells,domain.right_cells
    b.options = domain.owner.declaration.options
    for field in ['left_pair_begin','left_pair_count','right_pair_begin','right_pair_count']:
        setattr(b.selection,field,getattr(domain.native_source.product_domain,field))


def test_physical_owner_restartable_complete_descriptor_walk_has_no_domain_cache(monkeypatch):
    owner = _physical_owner()
    budget = Budget(16<<20,10**11)
    expected = list(itertools.product(range(2),repeat=4))
    def forbidden(*a,**kw): raise AssertionError('descriptor walk must not allocate domains')
    monkeypatch.setattr(core,'_make_bipole_finite_product_source',forbidden)
    assert list(owner.iter_quartets(budget=budget)) == expected
    assert list(owner.iter_quartets(budget=budget,stop=5)) + list(
        owner.iter_quartets(budget=budget,start=5)) == expected
    assert list(owner.iter_quartets(budget=budget,start=16)) == []
    assert [owner.quartet_at(i) for i in range(16)] == expected
    assert not owner.physical_hamiltonian_certified and not owner.symmetry_certified
    assert owner.ao_centers.shape == (2,3)
    with pytest.raises(ValueError): owner.ao_centers.flags.writeable = True
    with pytest.raises(ValueError): owner.lattice.flags.writeable = True
    with pytest.raises(AttributeError): owner.pair_radius = 99.


@pytest.mark.parametrize('field',['bytes','work'])
def test_physical_owner_construction_and_domain_exact_minus_one_budgets(field):
    owner = _physical_owner()
    p = owner.memory
    kw = dict(budget=Budget(p.inventoried_bytes,p.work_units))
    assert _physical_owner(**kw).memory == p
    kw['budget'] = Budget(p.inventoried_bytes-(field=='bytes'),p.work_units-(field=='work'))
    with pytest.raises((MemoryError,ValueError)):
        _physical_owner(**kw)
    d = _owned_domain(owner,(0,1,1,0),plan=True)
    exact = _owned_domain(owner,(0,1,1,0),budget=Budget(d.inventoried_bytes,d.work_units))
    assert exact.memory == d
    assert d.resident_bytes < d.inventoried_bytes
    with pytest.raises((MemoryError,ValueError)):
        _owned_domain(owner,(0,1,1,0),budget=Budget(d.inventoried_bytes-(field=='bytes'),
                                                  d.work_units-(field=='work')))
    size, work = owner.memory.resident_bytes+4096,65536+256*16
    assert len(list(owner.iter_quartets(budget=Budget(size,work)))) == 16
    with pytest.raises((MemoryError,ValueError)):
        owner.iter_quartets(budget=Budget(size-(field=='bytes'),work-(field=='work')))


@pytest.mark.parametrize('change',['basis','lattice','mesh','omega','pair_radius','midpoint_radius'])
def test_physical_owner_binds_common_declaration_and_policy(change):
    from tests.test_periodic_aopair_fourier_panel import _basis
    b = _finite_product_fixture()
    first = _physical_owner(b)
    kw = {}
    if change == 'basis':
        b.basis = _basis([(0,(.2,.4,0.),[.21],[.7],True),(0,(-.2,-.4,1.5),[.2],[.7],True)])
    elif change == 'lattice': b.system.lattice = np.asarray(b.system.lattice)*1.01
    elif change == 'mesh': b.mesh = core._RegularKMesh([1,1,3],[0,0,1])
    elif change == 'omega': b.options.omega += .1
    else: kw[change] = getattr(first,change)+.00001
    second = _physical_owner(b,**kw)
    assert second.source_identity_sha256 != first.source_identity_sha256
    a,c = _owned_domain(first,(0,1,1,0)),_owned_domain(second,(0,1,1,0))
    assert a.domain_identity_sha256 != c.domain_identity_sha256
    if change in ('pair_radius','midpoint_radius'):
        # Same finite labels, but a different generating policy must not
        # be mistaken for the same all-domain source declaration.
        assert a.native_source.source_identity_sha256 == c.native_source.source_identity_sha256


def test_physical_owner_snapshots_caller_options_caps_and_geometry():
    import gc
    b = _finite_product_fixture()
    owner = _physical_owner(b)
    identity = owner.source_identity_sha256
    first = _owned_domain(owner,(0,1,1,0))
    b.options.omega = 99.
    b.source_caps.maximum_images = 1
    b.system.lattice = np.eye(3)
    copy = owner.declaration.options
    copy.omega = 12.
    second = _owned_domain(owner,(0,1,1,0))
    assert second.domain_identity_sha256 == first.domain_identity_sha256
    _use_domain(b,first)
    values = _finite_product_panel(b,first.native_source).values_copy()
    assert owner.source_identity_sha256 == identity
    del owner,second
    gc.collect()
    np.testing.assert_array_equal(_finite_product_panel(b,first.native_source).values_copy(),values)


@pytest.mark.parametrize('bad',[True,-1,16,2**64,1.5])
def test_physical_owner_refuses_invalid_descriptor_indices(bad):
    owner = _physical_owner()
    with pytest.raises((ValueError,TypeError)): owner.quartet_at(bad)
    with pytest.raises((ValueError,TypeError)):
        owner.iter_quartets(budget=Budget(16<<20,10**11),start=bad,stop=1)


@pytest.mark.parametrize('bad',[(0,0,0),[0,0,0,0],(0,0,0,True),(0,0,0,-1),(0,0,0,2)])
def test_physical_owner_refuses_invalid_quartets(bad):
    with pytest.raises((ValueError,TypeError)):
        _owned_domain(_physical_owner(),bad)


@pytest.mark.parametrize('change',['quartets','cells','images','candidates','negative_radius','bool_radius','nan_radius',
                                    'cell_closure','image_closure'])
def test_physical_owner_refuses_invalid_or_insufficient_caps_and_policy(change):
    b = _finite_product_fixture()
    kw = {}
    if change == 'quartets': kw['maximum_quartets'] = 15
    elif change == 'cells': kw['maximum_cells'] = 65
    elif change == 'images': kw['maximum_images'] = 65
    elif change == 'candidates': kw['maximum_candidates'] = 0
    elif change == 'negative_radius': kw['pair_radius'] = -1
    elif change == 'bool_radius': kw['pair_radius'] = True
    elif change == 'nan_radius': kw['midpoint_radius'] = float('nan')
    elif change == 'cell_closure': b.options.require_cell_inversion_closure = True
    else: b.options.require_image_permutation_closure = True
    with pytest.raises((ValueError,TypeError)):
        _physical_owner(b,**kw)


@pytest.mark.parametrize('cap',['maximum_candidates','maximum_cells','maximum_images'])
def test_physical_owner_geometry_limits_fail_before_native_source_hash(monkeypatch,cap):
    owner = _physical_owner(**{cap:1})
    def forbidden(*a,**kw): raise AssertionError('native source must not hash unadmitted geometry')
    monkeypatch.setattr(core,'_make_bipole_finite_product_source',forbidden)
    with pytest.raises(ValueError,match='cap'):
        _owned_domain(owner,(0,1,1,0))


def test_physical_owner_uses_actual_mixed_angular_AO_centers_and_matches_native_components():
    from tests.test_periodic_aopair_fourier_panel import _basis
    b = _finite_product_fixture()
    xyz = np.array([[.25,.5,0.],[-.25,-.5,1.5],[.5,0.,.25],[0.,.25,.5]])
    b.basis = _basis([(0,xyz[0],[.3],[.7],True),(1,xyz[1],[.2],[.6],True),
                      (2,xyz[2],[.4],[.5],True),(2,xyz[3],[.3],[.8],True)])
    owner = _physical_owner(b)
    assert owner.memory.n_basis == 14
    np.testing.assert_array_equal(owner.ao_centers,np.repeat(xyz,[1,3,5,5],axis=0))
    domain = _owned_domain(owner,(2,10,5,13))
    expected = xyz[[1,3,2,3]]
    np.testing.assert_array_equal(domain.left_cells,_product(b.system.lattice,expected[:2]).cells)
    np.testing.assert_array_equal(domain.right_cells,_product(b.system.lattice,expected[2:]).cells)
    np.testing.assert_array_equal(domain.images,_physical_support(b.system.lattice,expected).images)
    _use_domain(b,domain)
    b.selection.q_index,b.selection.left_k_index,b.selection.right_k_index = 1,0,2
    sr,lr,zero = _finite_product_components(b)
    np.testing.assert_allclose(_finite_product_panel(b,domain.native_source).values_copy(),sr+lr-zero,
                               atol=3e-12,rtol=8e-12)


@pytest.mark.parametrize('mesh,shift',[
    ((1,1,1),(0,0,0)),((1,1,2),(0,0,0)),((1,1,3),(0,0,0)),
    ((1,1,3),(0,0,1)),((2,2,2),(1,1,1)),
])
def test_physical_owner_all_quartet_complex_density_actions_match_components_and_overlap(mesh,shift):
    """Test-only dense recontraction; production HF still has its own traversal."""
    from tests.test_bipole_finite_source import _erfc_oracle
    b = _finite_product_fixture(mesh=mesh,shift=shift)
    owner = _physical_owner(b)
    nk = len(b.mesh)
    rng = np.random.default_rng(82031)
    density = rng.normal(size=(nk,2,2))+1j*rng.normal(size=(nk,2,2))
    assert np.max(np.abs(density-density.conj().transpose(0,2,1))) > .1
    native = np.zeros((2,nk,2,2),complex)
    expected = np.zeros_like(native)
    zero_action = np.zeros_like(native)
    overlap = np.empty((nk,2,2),complex)
    # At most one domain is built at a time; no all-quartet source cache.
    for quartet in owner.iter_quartets(budget=Budget(16<<20,10**11)):
        domain = _owned_domain(owner,quartet)
        _use_domain(b,domain)
        a,c,d,e = quartet
        raw = _erfc_oracle(b.basis,b.images,lattice=np.asarray(b.system.lattice),
            left=(2*a+c,1),right=(2*d+e,1),omega=b.options.omega)
        if (d,e) == (0,0):
            for k in range(nk):
                overlap[k,a,c] = _hf_overlap(b,k,cells=domain.left_cells)[a,c]
        for target,k in itertools.product(range(nk),repeat=2):
            # J_ac = I[ac,de] D_de/Nk; K_ad = I[ac,de] D_ce/Nk,
            # with the latter's exact q = target-k and kL=kR=k.
            for arm,q,kl,output,weight in (
                (0,0,target,(a,c),density[k,d,e]/nk),
                (1,b.mesh.transfer_index(k,target),k,(a,d),density[k,c,e]/nk)):
                b.selection.q_index,b.selection.left_k_index,b.selection.right_k_index = q,kl,k
                value = _finite_product_panel(b,domain.native_source).element(0,0)
                sr,lr,zero = (part[0,0] for part in _finite_product_components(b,raw))
                at = (arm,target,*output)
                native[at] += value*weight
                expected[at] += (sr+lr-zero)*weight
                zero_action[at] += (value-sr-lr)*weight
        del domain
    np.testing.assert_allclose(native,expected,atol=5e-12,rtol=8e-12)
    coefficient = -np.pi/(abs(np.linalg.det(b.system.lattice))*b.options.omega**2*nk)
    jzero = coefficient*overlap*np.einsum('kab,kba->',density,overlap)
    kzero = coefficient*np.einsum('kac,kcd,kdb->kab',overlap,density,overlap)
    np.testing.assert_allclose(zero_action[0],jzero,atol=5e-12,rtol=8e-12)
    np.testing.assert_allclose(zero_action[1],kzero,atol=5e-12,rtol=8e-12)


def test_physical_owner_execution_caps_do_not_change_scientific_identity():
    b = _finite_product_fixture()
    first = _physical_owner(b)
    b.source_caps.maximum_work_units //= 2
    second = _physical_owner(b,maximum_candidates=99999,maximum_cells=63,maximum_images=63,
                             maximum_quartets=20)
    assert first.source_identity_sha256 == second.source_identity_sha256
    assert _owned_domain(first,(0,1,1,0)).domain_identity_sha256 == (
        _owned_domain(second,(0,1,1,0)).domain_identity_sha256)


def test_physical_owner_preserves_native_cartesian_angular_refusal():
    from tests.test_periodic_aopair_fourier_panel import _basis
    b = _finite_product_fixture()
    b.basis = _basis([(2,(.25,.5,0.),[.3],[.7],False)])
    with pytest.raises(ValueError,match='pure L<=6 or Cartesian s'):
        _physical_owner(b)


# Native complete J/K reduction over the source-owned singleton panel stream.
def _product_jk_stream_caps():
    caps = core._BipoleProductJKStreamCaps()
    caps.maximum_panel_calls = 32768
    caps.maximum_density_bytes = 16<<20
    caps.maximum_state_bytes = 128<<20
    caps.maximum_panel_inventoried_bytes = 32<<20
    caps.maximum_panel_work_units = 10**8
    caps.maximum_node_bytes = 512<<20
    caps.maximum_work_units = 10**11
    return caps


def _product_jk_stream(b, owner, density, target=0, *, caps=None, plan=False, budget=None):
    fn = owner.plan_jk_stream if plan else owner.start_jk_stream
    return fn(density,target,inventory=b.inventory,caps=_product_jk_stream_caps() if caps is None else caps,
              budget=Budget(1<<30,10**11) if budget is None else budget)


def _product_jk_next_panel(b,owner,stream,domain=None):
    selection = stream.next_selection()
    n = owner.memory.n_basis
    quartet = (*divmod(selection.left_pair_begin,n),*divmod(selection.right_pair_begin,n))
    if domain is None or domain.memory.quartet != quartet:
        domain = _owned_domain(owner,quartet)
    _use_domain(b,domain)
    b.selection = selection
    return domain,_finite_product_panel(b,domain.native_source)


def _product_jk_finish(b,owner,stream):
    domain = None
    while not stream.complete:
        domain,panel = _product_jk_next_panel(b,owner,stream,domain)
        stream.consume(panel)
    stream.finalize()
    return stream


@pytest.mark.parametrize('mesh,shift',[
    ((1,1,1),(0,0,0)),((1,1,2),(0,0,0)),((1,1,3),(0,0,0)),
    ((1,1,3),(0,0,1)),((2,2,2),(1,1,1)),
])
def test_product_jk_stream_native_actions_match_independent_components(mesh,shift):
    from tests.test_bipole_finite_source import _erfc_oracle
    b = _finite_product_fixture(mesh=mesh,shift=shift)
    owner = _physical_owner(b)
    nk = len(b.mesh)
    rng = np.random.default_rng(230718)
    density = rng.normal(size=(nk,2,2))+1j*rng.normal(size=(nk,2,2))
    for target in range(nk):
        stream = _product_jk_stream(b,owner,density,target)
        assert stream.memory.panel_calls == 2*16*nk
        assert not stream.complete and not stream.finalized
        expected = np.zeros((2,2,2),complex)
        domain = None
        while not stream.complete:
            old = domain
            domain,panel = _product_jk_next_panel(b,owner,stream,domain)
            if domain is not old:
                raw = _erfc_oracle(b.basis,b.images,lattice=np.asarray(b.system.lattice),
                    left=(b.selection.left_pair_begin,1),right=(b.selection.right_pair_begin,1),omega=.6)
            a,c,d,e = domain.memory.quartet
            k = b.selection.right_k_index
            sr,lr,zero = (x[0,0] for x in _finite_product_components(b,raw))
            if stream.accepted_panels%2:
                expected[1,a,d] += (sr+lr-zero)*density[k,c,e]/nk
            else:
                expected[0,a,c] += (sr+lr-zero)*density[k,d,e]/nk
            stream.consume(panel)
        assert stream.accepted_panels == stream.memory.panel_calls
        stream.finalize()
        np.testing.assert_allclose(stream.values_copy().reshape(2,2,2),expected,atol=5e-12,rtol=8e-12)
        assert stream.finalized
        assert stream.declared_policy_identity_sha256 == owner.source_identity_sha256
        assert stream.citation_numerics == ['bipole_finite_jk']
        assert not stream.physical_hamiltonian_certified and not stream.symmetry_certified


@pytest.mark.parametrize('action',['finalize','values_copy','element','input','payload'])
def test_product_jk_stream_cannot_expose_partial_results(action):
    b = _finite_product_fixture()
    stream = _product_jk_stream(b,_physical_owner(b),np.zeros((3,2,2),complex))
    with pytest.raises(RuntimeError,match='incomplete|finalized'):
        if action == 'input': _ = stream.input_identity_sha256
        elif action == 'payload': _ = stream.payload_identity_sha256
        elif action == 'element': stream.element(0,0)
        else: getattr(stream,action)()
    assert stream.accepted_panels == 0


@pytest.mark.parametrize('change',['q','left_k','right_k','left_pair','right_pair'])
def test_product_jk_stream_refuses_wrong_schedule_and_recovers(change):
    b = _finite_product_fixture()
    owner = _physical_owner(b)
    density = np.ones((3,2,2),complex)*(1+.2j)
    stream = _product_jk_stream(b,owner,density)
    domain,valid = _product_jk_next_panel(b,owner,stream)
    if change.endswith('pair'):
        quartet = (0,1,0,0) if change == 'left_pair' else (0,0,0,1)
        wrong = _owned_domain(owner,quartet)
        _use_domain(b,wrong)
        invalid = _finite_product_panel(b,wrong.native_source)
    else:
        setattr(b.selection,{'q':'q_index','left_k':'left_k_index','right_k':'right_k_index'}[change],1)
        invalid = _finite_product_panel(b,domain.native_source)
    with pytest.raises(ValueError,match='schedule'): stream.consume(invalid)
    assert stream.accepted_panels == 0
    stream.consume(valid)
    assert stream.accepted_panels == 1
    result = _product_jk_finish(b,owner,stream)
    reference = _product_jk_finish(b,owner,_product_jk_stream(b,owner,density))
    assert result.payload_identity_sha256 == reference.payload_identity_sha256


def test_product_jk_stream_repeated_integral_may_fill_matching_jk_slots_but_not_skip_k():
    b = _finite_product_fixture()
    owner = _physical_owner(b)
    stream = _product_jk_stream(b,owner,np.ones((3,2,2),complex))
    _,panel = _product_jk_next_panel(b,owner,stream)
    # At target=k=0 the J/K descriptors coincide; the same immutable
    # integral legitimately contributes with their respective density entries.
    stream.consume(panel);stream.consume(panel)
    assert stream.accepted_panels == 2
    with pytest.raises(ValueError,match='schedule'): stream.consume(panel)
    assert stream.accepted_panels == 2
    _product_jk_finish(b,owner,stream)


@pytest.mark.parametrize('change',['basis','lattice','mesh','omega','cutoff','reciprocal_flag'])
def test_product_jk_stream_refuses_foreign_common_context(change):
    from tests.test_periodic_aopair_fourier_panel import _basis
    b = _finite_product_fixture()
    owner = _physical_owner(b)
    stream = _product_jk_stream(b,owner,np.ones((3,2,2),complex))
    if change == 'basis':
        b.basis = _basis([(0,(.2,.4,0.),[.21],[.7],True),(0,(-.2,-.4,1.5),[.2],[.7],True)])
    elif change == 'lattice': b.system.lattice = np.asarray(b.system.lattice)*1.01
    elif change == 'mesh': b.mesh = core._RegularKMesh([1,1,3],[0,0,1])
    elif change == 'omega': b.options.omega = .7
    elif change == 'cutoff': b.options.reciprocal_energy_cutoff *= 1.1
    else: b.options.require_reciprocal_conjugacy = False
    other = _physical_owner(b)
    domain = _owned_domain(other,(0,0,0,0));_use_domain(b,domain)
    b.selection = stream.next_selection()
    panel = _finite_product_panel(b,domain.native_source)
    with pytest.raises(ValueError,match='common declaration'): stream.consume(panel)
    assert stream.accepted_panels == 0


@pytest.mark.parametrize('change',['within_quartet','across_quartets','same_product_roles','broad_source'])
def test_product_jk_stream_locks_domain_and_product_identities(change):
    b = _finite_product_fixture()
    owner = _physical_owner(b)
    stream = _product_jk_stream(b,owner,np.ones((3,2,2),complex))
    if change == 'within_quartet':
        _,first = _product_jk_next_panel(b,owner,stream);stream.consume(first)
    if change == 'across_quartets':
        domain = None
        for _ in range(6):
            domain,panel = _product_jk_next_panel(b,owner,stream,domain);stream.consume(panel)
    domain,valid = _product_jk_next_panel(b,owner,stream)
    if change == 'within_quartet': b.images = np.empty((0,3,3),np.int64)
    else: b.cells = np.array([[0,0,1]],np.int64)
    declared = _finite_domain(b)
    if change == 'broad_source': declared.left_pair_count = 2
    changed = _finite_product_source(b,domain=declared)
    panel = _finite_product_panel(b,changed)
    before = stream.accepted_panels
    with pytest.raises(ValueError,match='quartet|supports|singleton'): stream.consume(panel)
    assert stream.accepted_panels == before
    stream.consume(valid)
    _product_jk_finish(b,owner,stream)


@pytest.mark.parametrize('field,measured',[
    ('maximum_panel_calls','panel_calls'),('maximum_density_bytes','density_bytes'),
    ('maximum_state_bytes','state_bytes'),('maximum_node_bytes','required_node_inventoried_bytes'),
    ('maximum_work_units','work_units_upper_bound'),
])
def test_product_jk_stream_outer_exact_minus_one_caps_precede_density(field,measured):
    b = _finite_product_fixture();owner = _physical_owner(b)
    density = np.ones((3,2,2),complex);caps = _product_jk_stream_caps()
    plan = _product_jk_stream(b,owner,density,caps=caps,plan=True)
    setattr(caps,field,getattr(plan,measured))
    _product_jk_stream(b,owner,density,caps=caps)
    setattr(caps,field,getattr(plan,measured)-1)
    density[0,0,0] = np.nan
    with pytest.raises((ValueError,RuntimeError),match='cap'):
        _product_jk_stream(b,owner,density,caps=caps)


@pytest.mark.parametrize('field',['maximum_panel_inventoried_bytes','maximum_panel_work_units'])
def test_product_jk_stream_incoming_panel_envelope_is_enforced_before_commit(field):
    b = _finite_product_fixture();owner = _physical_owner(b)
    density = np.ones((3,2,2),complex)
    first = _product_jk_stream(b,owner,density)
    _,panel = _product_jk_next_panel(b,owner,first)
    value = panel.memory.required_node_inventoried_bytes if field.endswith('bytes') else panel.memory.work_units_upper_bound
    caps = _product_jk_stream_caps();setattr(caps,field,value)
    exact = _product_jk_stream(b,owner,density,caps=caps);exact.consume(panel)
    setattr(caps,field,value-1)
    short = _product_jk_stream(b,owner,density,caps=caps)
    with pytest.raises((ValueError,RuntimeError),match='incoming panel'): short.consume(panel)
    assert short.accepted_panels == 0


@pytest.mark.parametrize('bad',['nan','inf','real','list','strided','unaligned','shape'])
def test_product_jk_stream_strict_full_density_snapshot(bad):
    b = _finite_product_fixture();owner = _physical_owner(b)
    d = np.zeros((3,2,2),complex)
    if bad == 'nan': d[2,1,1] = np.nan
    elif bad == 'inf': d[2,1,1] = np.inf*1j
    elif bad == 'real': d = d.real.copy()
    elif bad == 'list': d = d.tolist()
    elif bad == 'strided': d = np.zeros((6,2,2),complex)[::2]
    elif bad == 'unaligned': d = np.ndarray(d.shape,dtype=complex,buffer=bytearray(d.nbytes+1),offset=1)
    else: d = np.zeros((3,4),complex)
    with pytest.raises((ValueError,TypeError,OverflowError)):
        _product_jk_stream(b,owner,d)


def test_product_jk_stream_owns_density_and_final_state_survives_producers():
    import gc
    b = _finite_product_fixture();owner = _physical_owner(b)
    density = np.arange(12,dtype=float).reshape(3,2,2).astype(complex)*(1+.31j)
    saved = density.copy()
    stream = _product_jk_stream(b,owner,density)
    density[:] = np.nan
    _product_jk_finish(b,owner,stream)
    fresh = _product_jk_finish(b,owner,_product_jk_stream(b,owner,saved))
    assert stream.payload_identity_sha256 == fresh.payload_identity_sha256
    values = stream.values_copy()
    with pytest.raises(ValueError): values.flags.writeable = True
    with pytest.raises(RuntimeError,match='finalized'): stream.finalize()
    with pytest.raises(RuntimeError,match='complete'): stream.next_selection()
    del b,owner,fresh,density,saved
    gc.collect()
    np.testing.assert_array_equal(stream.values_copy(),values)
    with pytest.raises(IndexError): stream.element(2,0)


def test_product_jk_stream_execution_caps_preserve_identity_and_policy_changes_it():
    b = _finite_product_fixture();owner = _physical_owner(b)
    d = np.ones((3,2,2),complex)
    first = _product_jk_finish(b,owner,_product_jk_stream(b,owner,d))
    caps = _product_jk_stream_caps();caps.maximum_panel_work_units = 9*10**7
    b.inventory.reciprocal_block_size = 1
    second = _product_jk_finish(b,owner,_product_jk_stream(b,owner,d,caps=caps))
    assert second.payload_identity_sha256 == first.payload_identity_sha256
    changed = core._make_bipole_product_jk_stream(owner.declaration,d,0,'a'*64,b.inventory,caps)
    _product_jk_finish(b,owner,changed)
    assert changed.input_identity_sha256 != first.input_identity_sha256
    np.testing.assert_array_equal(changed.values_copy(),first.values_copy())


@pytest.mark.parametrize('policy',['','A'*64,'g'*64,'a'*63,'a'*65])
def test_product_jk_stream_refuses_malformed_declared_policy(policy):
    b = _finite_product_fixture();owner = _physical_owner(b)
    with pytest.raises(ValueError,match='policy|SHA256'):
        core._make_bipole_product_jk_stream(owner.declaration,np.zeros((3,2,2),complex),0,
                                           policy,b.inventory,_product_jk_stream_caps())


def test_product_jk_stream_replica_census_and_shared_outer_budget():
    b = _finite_product_fixture();owner = _physical_owner(b)
    d = np.ones((3,2,2),complex);caps = _product_jk_stream_caps()
    first = _product_jk_stream(b,owner,d,caps=caps,plan=True)
    b.inventory.numerical_replicas = 2
    b.inventory.external_node_bytes = 1234
    b.inventory.other_live_numerical_bytes_per_replica = 5678
    second = _product_jk_stream(b,owner,d,caps=caps,plan=True)
    assert second.per_replica_inventoried_bytes == first.per_replica_inventoried_bytes+5678
    assert second.required_node_inventoried_bytes == 1234+2*(second.per_replica_inventoried_bytes+caps.maximum_panel_inventoried_bytes)
    exact = second.required_node_inventoried_bytes+2*owner.memory.resident_bytes
    _product_jk_stream(b,owner,d,caps=caps,budget=Budget(exact,second.work_units_upper_bound))
    d[0,0,0] = np.nan
    with pytest.raises(MemoryError):
        _product_jk_stream(b,owner,d,caps=caps,budget=Budget(exact-1,second.work_units_upper_bound))


@pytest.mark.parametrize('kind',['nonempty_common','product'])
def test_product_jk_stream_requires_empty_common_anchor(kind):
    from tests.test_bipole_finite_source import _source
    b = _finite_product_fixture()
    source = _finite_product_source(b) if kind == 'product' else _source(b)
    with pytest.raises(ValueError,match='empty common declaration'):
        core._make_bipole_product_jk_stream(source,np.zeros((3,2,2),complex),0,'a'*64,
                                           b.inventory,_product_jk_stream_caps())


def test_product_jk_stream_arithmetic_overflow_does_not_advance_schedule():
    from tests.test_periodic_aopair_fourier_panel import _basis
    b = _finite_product_fixture()
    b.basis = _basis([(0,(.2,.4,0.),[.2],[70.],True),(0,(-.2,-.4,1.5),[.2],[70.],True)])
    owner = _physical_owner(b)
    stream = _product_jk_stream(b,owner,np.full((3,2,2),1e308,complex))
    _,panel = _product_jk_next_panel(b,owner,stream)
    assert abs(panel.element(0,0)) > 10
    with pytest.raises(OverflowError): stream.consume(panel)
    assert stream.accepted_panels == 0
    assert stream.next_selection().left_pair_begin == 0
    with pytest.raises(RuntimeError,match='incomplete'): stream.finalize()


def test_product_jk_stream_mixed_sp_full_ao_action_matches_independent_components():
    from tests.test_periodic_aopair_fourier_panel import _basis
    from tests.test_bipole_finite_source import _erfc_oracle
    b = _finite_product_fixture(mesh=(1,1,1))
    b.basis = _basis([(0,(.2,.4,0.),[.2],[.7],True),(1,(-.2,-.4,1.5),[.3],[.6],True)])
    owner = _physical_owner(b)
    rng = np.random.default_rng(30812)
    density = rng.normal(size=(1,4,4))+1j*rng.normal(size=(1,4,4))
    caps = _product_jk_stream_caps();caps.maximum_panel_work_units = 9*10**7
    stream = _product_jk_stream(b,owner,density,caps=caps)
    expected = np.zeros((2,4,4),complex)
    domain = None
    while not stream.complete:
        previous = domain
        domain,panel = _product_jk_next_panel(b,owner,stream,domain)
        if domain is not previous:
            raw = _erfc_oracle(b.basis,b.images,lattice=np.asarray(b.system.lattice),
                left=(b.selection.left_pair_begin,1),right=(b.selection.right_pair_begin,1),omega=.6)
        a,c,d,e = domain.memory.quartet
        sr,lr,zero = (part[0,0] for part in _finite_product_components(b,raw))
        if stream.accepted_panels%2: expected[1,a,d] += (sr+lr-zero)*density[0,c,e]
        else: expected[0,a,c] += (sr+lr-zero)*density[0,d,e]
        stream.consume(panel)
    stream.finalize()
    assert stream.accepted_panels == 512
    np.testing.assert_allclose(stream.values_copy().reshape(2,4,4),expected,atol=5e-12,rtol=8e-12)


# Complete geometry admission followed by native production/reduction replay.
def _physical_jk(b, owner, density, target=0, *, plan=False, budget=None, stream_caps=None):
    return (owner.plan_jk if plan else owner.contract_jk)(
        density,target,inventory=b.inventory,panel_caps=b.caps,
        stream_caps=_product_jk_stream_caps() if stream_caps is None else stream_caps,
        budget=Budget(1<<30,10**11) if budget is None else budget)


@pytest.mark.parametrize('mesh,shift',[
    ((1,1,1),(0,0,0)),((1,1,2),(0,0,0)),
    ((1,1,3),(0,0,1)),((2,2,2),(1,1,1)),
])
def test_physical_jk_complete_producer_matches_native_reference_stream(mesh,shift):
    b = _finite_product_fixture(mesh=mesh,shift=shift); owner = _physical_owner(b)
    nk = len(b.mesh)
    rng = np.random.default_rng(10942)
    density = rng.normal(size=(nk,2,2))+1j*rng.normal(size=(nk,2,2))
    for target in sorted({0,nk-1}):
        expected = _product_jk_finish(b,owner,_product_jk_stream(b,owner,density,target))
        plan = _physical_jk(b,owner,density,target,plan=True)
        actual = _physical_jk(b,owner,density,target,
                              budget=Budget(plan.inventoried_bytes,plan.work_units))
        np.testing.assert_array_equal(actual.values_copy(),expected.values_copy())
        assert actual.input_identity_sha256 == expected.input_identity_sha256
        assert actual.payload_identity_sha256 == expected.payload_identity_sha256
        assert actual.accepted_panels == plan.panel_calls == 2*16*nk
        assert actual.finalized and actual.complete
        assert not actual.physical_hamiltonian_certified and not actual.symmetry_certified
        assert actual.citation_numerics == ['bipole_finite_jk']


@pytest.mark.parametrize('which',['bytes','work'])
def test_physical_jk_global_minus_one_refuses_before_density_or_integrals(which,monkeypatch):
    b = _finite_product_fixture(); owner = _physical_owner(b)
    d = np.ones((3,2,2),complex)
    p = _physical_jk(b,owner,d,plan=True)
    budget = Budget(p.inventoried_bytes-(which=='bytes'),p.work_units-(which=='work'))
    d[:] = np.nan
    def forbidden(*args,**kw):
        pytest.fail('density snapshot or integral ran before complete admission')
    monkeypatch.setattr(core,'_make_bipole_product_jk_stream',forbidden)
    monkeypatch.setattr(core,'_make_bipole_finite_product_panel',forbidden)
    with pytest.raises((MemoryError,ValueError),match='budget'):
        _physical_jk(b,owner,d,budget=budget)


def test_physical_jk_late_panel_refusal_precedes_any_numerics(monkeypatch):
    b = _finite_product_fixture(); owner = _physical_owner(b)
    original = core._plan_bipole_finite_product_panel
    calls = []
    def plan(*args):
        s = args[5]; calls.append((s.left_pair_begin,s.right_pair_begin))
        if calls[-1] == (3,3):
            raise ValueError('late quartet sentinel')
        return original(*args)
    def forbidden(*args,**kw):
        pytest.fail('numerics ran before final-quartet admission')
    monkeypatch.setattr(core,'_plan_bipole_finite_product_panel',plan)
    monkeypatch.setattr(core,'_make_bipole_product_jk_stream',forbidden)
    monkeypatch.setattr(core,'_make_bipole_finite_product_panel',forbidden)
    with pytest.raises(ValueError,match='late quartet sentinel'):
        _physical_jk(b,owner,np.ones((3,2,2),complex))
    assert len(calls) == 15*6+1


def test_physical_jk_planning_never_scans_density_or_evaluates_panels(monkeypatch):
    b = _finite_product_fixture(); owner = _physical_owner(b)
    def forbidden(*args,**kw):
        pytest.fail('planning evaluated a panel or copied density')
    monkeypatch.setattr(core,'_make_bipole_product_jk_stream',forbidden)
    monkeypatch.setattr(core,'_make_bipole_finite_product_panel',forbidden)
    p = _physical_jk(b,owner,np.full((3,2,2),np.nan,complex),plan=True)
    assert p.quartet_count == 16 and p.panel_calls == 96


@pytest.mark.parametrize('kind',['real','shape','strided','unaligned','target_bool','target_high'])
def test_physical_jk_bad_inputs_refuse_before_geometry(kind,monkeypatch):
    from vibeqc._bipole_physical_source import PhysicalSource
    b = _finite_product_fixture(); owner = _physical_owner(b)
    d = np.ones((3,2,2),complex); target = 0
    if kind == 'real': d = d.real.copy()
    elif kind == 'shape': d = d[0]
    elif kind == 'strided': d = d.transpose(0,2,1)
    elif kind == 'unaligned': d = np.ndarray((3,2,2),complex,buffer=bytearray(193),offset=1)
    elif kind == 'target_bool': target = True
    elif kind == 'target_high': target = 3
    def forbidden(*args,**kw): pytest.fail('invalid density/target reached geometry')
    monkeypatch.setattr(PhysicalSource,'build_domain',forbidden)
    with pytest.raises((ValueError,TypeError)):
        _physical_jk(b,owner,d,target)


def test_physical_jk_releases_previous_domain_and_panel_before_next_allocation(monkeypatch):
    import weakref
    from vibeqc._bipole_physical_source import PhysicalSource
    b = _finite_product_fixture(); owner = _physical_owner(b)
    build = PhysicalSource.build_domain; make = core._make_bipole_finite_product_panel
    domain_refs, panel_refs = [], []
    def domain(*args,**kw):
        assert not domain_refs or domain_refs[-1]() is None
        assert not panel_refs or panel_refs[-1]() is None
        result = build(*args,**kw)
        domain_refs.append(weakref.ref(result.images))
        return result
    def panel(*args,**kw):
        assert not panel_refs or panel_refs[-1]() is None
        result = make(*args,**kw); panel_refs.append(weakref.ref(result))
        return result
    monkeypatch.setattr(PhysicalSource,'build_domain',domain)
    monkeypatch.setattr(core,'_make_bipole_finite_product_panel',panel)
    result = _physical_jk(b,owner,np.ones((3,2,2),complex))
    assert result.finalized and len(domain_refs) == 32 and len(panel_refs) == 96
    assert all(ref() is None for ref in domain_refs+panel_refs)


def test_physical_jk_controls_are_snapshotted_before_walk(monkeypatch):
    from vibeqc._bipole_physical_source import PhysicalSource
    b = _finite_product_fixture(); owner = _physical_owner(b)
    density = np.ones((3,2,2),complex)
    expected = _physical_jk(b,owner,density)
    caps = _product_jk_stream_caps(); build = PhysicalSource.build_domain
    def mutate_caller(*args,**kw):
        caps.maximum_panel_calls = 0
        b.inventory.reciprocal_block_size = 0
        b.caps.short_range.raw.maximum_images = 0
        b.caps.maximum_node_bytes = 0
        return build(*args,**kw)
    monkeypatch.setattr(PhysicalSource,'build_domain',mutate_caller)
    result = _physical_jk(b,owner,density,stream_caps=caps)
    assert result.payload_identity_sha256 == expected.payload_identity_sha256


def test_physical_jk_replay_digest_failure_cannot_finalize(monkeypatch):
    from vibeqc._bipole_physical_source import PhysicalSource
    b = _finite_product_fixture(); owner = _physical_owner(b)
    build = PhysicalSource.build_domain; calls = 0
    def changed_receipt(*args,**kw):
        nonlocal calls
        d = build(*args,**kw); calls += 1
        if calls == 32:
            # Simulate producer replay drift without forging a native value.
            object.__setattr__(d,'domain_identity_sha256','a'*64)
        return d
    monkeypatch.setattr(PhysicalSource,'build_domain',changed_receipt)
    with pytest.raises(RuntimeError,match='replay differs'):
        _physical_jk(b,owner,np.ones((3,2,2),complex))
    assert calls == 32


def test_physical_jk_replica_memory_and_cumulative_geometry_work():
    b = _finite_product_fixture(); owner = _physical_owner(b); density = np.zeros((3,2,2),complex)
    first = _physical_jk(b,owner,density,plan=True)
    b.inventory.numerical_replicas = 2; b.inventory.external_node_bytes = 987
    second = _physical_jk(b,owner,density,plan=True)
    # Caller external bytes also occur in each incoming-panel envelope, but
    # its reserved maximum stays fixed. Top-level external storage is once.
    assert second.inventoried_bytes == 2*first.inventoried_bytes+987
    assert second.work_units == first.work_units
    domains = [owner.plan_domain(owner.quartet_at(i),budget=Budget(16<<20,10**11)) for i in range(16)]
    assert first.geometry_work_units == sum(d.work_units for d in domains)
    assert first.maximum_domain_bytes == max(d.inventoried_bytes for d in domains)
    assert second.domain_walk_sha256 == first.domain_walk_sha256


@pytest.mark.parametrize('field,value',[
    ('maximum_panel_inventoried_bytes',(512<<20)+1),('maximum_panel_work_units',10**9+1),
])
def test_physical_jk_refuses_reservations_outside_native_limits(field,value):
    b = _finite_product_fixture(); owner = _physical_owner(b); c = _product_jk_stream_caps()
    setattr(c,field,value)
    with pytest.raises(ValueError,match='native diagnostic bounds'):
        _physical_jk(b,owner,np.zeros((3,2,2),complex),stream_caps=c)


@pytest.mark.parametrize('mixed',[False,True])
def test_physical_jk_empty_support_and_mixed_sp_ao_walk(mixed):
    from tests.test_periodic_aopair_fourier_panel import _basis
    b = _finite_product_fixture(mesh=(1,1,1))
    if mixed:
        b.basis = _basis([(0,(.2,.4,0.),[.2],[.7],True),(1,(-.2,-.4,1.5),[.3],[.6],True)])
    owner = _physical_owner(b,**({} if mixed else dict(pair_radius=0.,midpoint_radius=0.)))
    n = owner.memory.n_basis
    d = (np.arange(n*n).reshape(1,n,n)+1j).astype(complex)
    caps = _product_jk_stream_caps(); caps.maximum_panel_work_units = 9*10**7
    expected = _product_jk_finish(b,owner,_product_jk_stream(b,owner,d,caps=caps))
    result = _physical_jk(b,owner,d,stream_caps=caps)
    assert result.accepted_panels == 2*n**4
    np.testing.assert_array_equal(result.values_copy(),expected.values_copy())
    assert result.payload_identity_sha256 == expected.payload_identity_sha256


@pytest.mark.parametrize('field',['maximum_panel_inventoried_bytes','maximum_panel_work_units'])
def test_physical_jk_tight_panel_reservation_refuses_before_integrals(field,monkeypatch):
    b = _finite_product_fixture(); owner = _physical_owner(b); caps = _product_jk_stream_caps()
    setattr(caps,field,1)
    def forbidden(*args,**kw): pytest.fail('producer exceeded its admitted panel envelope')
    monkeypatch.setattr(core,'_make_bipole_finite_product_panel',forbidden)
    monkeypatch.setattr(core,'_make_bipole_product_jk_stream',forbidden)
    with pytest.raises((ValueError,MemoryError)):
        _physical_jk(b,owner,np.zeros((3,2,2),complex),stream_caps=caps)


def test_physical_jk_nonfinite_density_refuses_before_integrals(monkeypatch):
    b = _finite_product_fixture(); owner = _physical_owner(b)
    def forbidden(*args,**kw): pytest.fail('nonfinite density reached the integral producer')
    monkeypatch.setattr(core,'_make_bipole_finite_product_panel',forbidden)
    with pytest.raises((ValueError,OverflowError),match='nonfinite'):
        _physical_jk(b,owner,np.full((3,2,2),np.nan,complex))
