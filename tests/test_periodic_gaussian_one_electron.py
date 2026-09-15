"""Tiny finite-image S/T integrals; no SCF, nuclear source or target jobs."""

from __future__ import annotations

import gc
import hashlib
import itertools
import math
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_aopair_fourier_panel import (
    _basis, _call as _fourier, _mixed_basis, _polynomials, _primitive_pair, _system,
)
from tests.test_periodic_gaussian_source_context import (
    _CAP_FIELDS as _BASIS_CAP_FIELDS, _caps as _basis_caps, _exact_caps,
    _make as _context, _options as _source_options,
)


def _bundle(*, basis=None, lattice=None, mesh=(2,3,1), cutoff=3.4):
    ao = _basis([(0,(.1,-.2,.15),[.55,1.1],[.7,-.12],True),
                 (0,(.6,.3,-.25),[.7],[.8],True)]) if basis is None else basis
    system = _system(np.array([[2.7,.2,.1],[0,3.1,.25],[0,0,2.9]]) if lattice is None else lattice)
    auxiliary = _basis([(0,(.1,.2,.3),[.8],[.6],True)])
    options = _source_options()
    options.ao_pair_image_cutoff_bohr = cutoff
    context = _context(system=system,ao=ao,auxiliary=auxiliary,
                       mesh=core._RegularKMesh(mesh),options=options)
    return SimpleNamespace(ao=ao,auxiliary=auxiliary,system=system,context=context)


def _normalized_basis(specifications):
    # _basis intentionally preserves arbitrary pre-normalized coefficients.
    # Closed-form unit-normalized witnesses instead request libint's explicit
    # primitive/contracted normalization at basis construction, not in S/T.
    raw = _basis(specifications)
    centers = list(dict.fromkeys(tuple(s.origin) for s in raw.shells()))
    molecule = core.Molecule([core.Atom(2,list(center)) for center in centers])
    return core.BasisSet(molecule,raw.shells(),"normalized-analytic-witness",False)


def _options(b):
    o = core._PeriodicGaussianOneElectronOptions()
    o.basis_verification_caps = _exact_caps(b.context)
    o.structural_absolute_tolerance = o.structural_relative_tolerance = 2e-12
    return o


def _live():
    live = core._PeriodicGaussianOneElectronLiveInventory()
    live.replicas_per_node = 1
    live.fixed_backend_margin_bytes_per_replica = 4096
    return live


def _caps():
    c = core._PeriodicGaussianOneElectronCaps()
    c.maximum_owned_numeric_bytes = 1048576
    c.maximum_per_replica_inventoried_bytes = 2**22
    c.maximum_node_inventoried_bytes = 2**25
    c.maximum_pair_count = 256
    c.maximum_candidate_evaluations = 1000000
    c.maximum_work_units = 2000000000
    return c


def _call(b,k=0,begin=0,count=None,*,options=None,live=None,caps=None,plan=False):
    function = (core._plan_periodic_gaussian_one_electron_panel if plan
                else core._build_periodic_gaussian_one_electron_panel)
    return function(b.context,b.ao,b.auxiliary,b.system,k,begin,
                    b.ao.nbasis**2-begin if count is None else count,
                    _options(b) if options is None else options,
                    _live() if live is None else live,_caps() if caps is None else caps)


def _gradient(polynomial,alpha,axis):
    """Independent differentiated Cartesian polynomials, not MD tables."""
    output = {}
    for powers,coefficient in polynomial.items():
        if powers[axis]:
            lower = list(powers)
            lower[axis] -= 1
            output[tuple(lower)] = output.get(tuple(lower),0)+coefficient*powers[axis]
        raised = list(powers)
        raised[axis] += 1
        output[tuple(raised)] = output.get(tuple(raised),0)-2*alpha*coefficient
    return output


def _oracle(b,k,image_bound=2):
    orbitals = [(shell,p) for shell in b.ao.shells() for p in _polynomials(shell.l)]
    n = b.ao.nbasis
    output = np.zeros((2,n,n),complex)
    cutoff = b.context.options.ao_pair_image_cutoff_bohr
    kv = np.array(b.context.k_record(k).cartesian)
    retained = 0
    for mu,(a,pa) in enumerate(orbitals):
        for nu,(bb,pb) in enumerate(orbitals):
            for label in itertools.product(range(-image_bound,image_bound+1),repeat=3):
                translation = b.system.lattice@label
                ac,bc = np.array(a.origin),np.array(bb.origin)+translation
                if np.dot(ac-bc,ac-bc)>cutoff**2:
                    continue
                retained += 1
                phase = np.exp(1j*kv@translation)
                for alpha,ca in zip(a.exponents,a.coefficients):
                    for beta,cb in zip(bb.exponents,bb.coefficients):
                        common = phase*ca*cb
                        output[0,mu,nu] += common*_primitive_pair(alpha,beta,ac,bc,pa,pb,np.zeros(3))
                        output[1,mu,nu] += .5*common*sum(
                            _primitive_pair(alpha,beta,ac,bc,_gradient(pa,alpha,d),
                                            _gradient(pb,beta,d),np.zeros(3)) for d in range(3))
    return output,retained


@pytest.mark.parametrize("mesh,k",[((1,1,1),0),((2,1,1),1),((3,1,1),1),
                                  ((2,3,1),4),((2,2,2),7)])
def test_actual_gaussian_direct_images_fix_phase_normalization_skew_and_trim(mesh,k):
    b = _bundle(mesh=mesh)
    result = _call(b,k)
    expected,retained = _oracle(b,k)
    actual = result.values_copy().reshape(expected.shape)
    np.testing.assert_allclose(actual,expected,atol=4e-13,rtol=3e-12)
    ft = _fourier(b.ao,[[0,0,0]],system=b.system,ket=b.context.k_record(k).cartesian,
                  cutoff=b.context.options.ao_pair_image_cutoff_bohr)["values"]
    np.testing.assert_allclose(actual[0].ravel(),ft[:,0],atol=3e-14,rtol=3e-13)
    d,p = result.diagnostics,result.plan
    assert d.retained_primary_pair_images == retained
    assert d.retained_audit_pair_images == 2*retained
    assert d.completed_candidate_evaluations == p.candidate_evaluations
    assert d.completed_primitive_pair_evaluations <= p.primitive_pair_evaluations_upper_bound
    assert p.opposite_k_index == np.ravel_multi_index(tuple((-np.array(np.unravel_index(k,mesh)))%mesh),mesh)
    np.testing.assert_allclose(actual,actual.conj().transpose(0,2,1),atol=3e-14)
    assert d.maximum_overlap_hermitian_error < 1e-12
    assert d.maximum_kinetic_time_reversal_error < 1e-12
    if k != p.opposite_k_index:
        assert np.max(abs(actual.imag)) > 1e-5
    assert not result.nuclear_hcore_certified
    assert not result.infinite_image_tail_certified
    assert not result.ao_image_source_certified


def test_mixed_s_p_d_contracts_original_normalized_derivative_polynomials():
    b = _bundle(basis=_mixed_basis(),lattice=np.eye(3)*8,mesh=(1,1,1),cutoff=2.)
    actual = _call(b).values_copy().reshape(2,9,9)
    expected,_ = _oracle(b,0,image_bound=0)
    np.testing.assert_allclose(actual,expected,atol=2e-12,rtol=3e-12)
    np.testing.assert_allclose(actual[0].real,core.compute_overlap(b.ao),atol=3e-13,rtol=3e-13)
    np.testing.assert_allclose(actual[1].real,core.compute_kinetic(b.ao),atol=2e-12,rtol=3e-12)
    np.testing.assert_array_equal(actual.imag,0)


@pytest.mark.parametrize("angular",range(6))
def test_original_pure_shell_normalization_through_l5_matches_native_libint(angular):
    basis = _basis([(angular,(.1,-.2,.3),[.65],[.8],True),
                    (0,(-.1,.1,-.2),[.8],[.7],True)])
    b = _bundle(basis=basis,lattice=np.eye(3)*8,mesh=(1,1,1),cutoff=2.)
    reference = np.array([core.compute_overlap(basis),core.compute_kinetic(basis)])
    n = basis.nbasis
    for mu,nu in ((0,0),(angular,angular),(0,n-1),(n-1,angular)):
        actual = _call(b,begin=mu*n+nu,count=1).values_copy()[:,0]
        np.testing.assert_allclose(actual,reference[:,mu,nu],atol=4e-12,rtol=5e-12)


def test_l6_raised_derivatives_match_analytic_pure_shell_without_unsupported_engine():
    # The vendored libint overlap/kinetic Engine supports L<=5. The new
    # MD leaf reaches L6 without calling that unsupported Engine route.
    alpha = .65
    basis = _normalized_basis([(6,(.1,-.2,.3),[alpha],[1.],True)])
    b = _bundle(basis=basis,lattice=np.eye(3)*8,mesh=(1,1,1),cutoff=2.)
    n = basis.nbasis
    for mu,nu in ((0,0),(6,6),(12,12),(0,6),(6,12)):
        result = _call(b,begin=mu*n+nu,count=1).values_copy()[:,0]
        expected = [1.,alpha*(6+1.5)] if mu==nu else [0.,0.]
        np.testing.assert_allclose(result,expected,atol=6e-12,rtol=6e-12)
        ft = _fourier(basis,[[0,0,0]],system=b.system,begin=mu*n+nu,count=1,cutoff=2.)
        np.testing.assert_allclose(result[0],ft["values"][0,0],atol=3e-13,rtol=3e-13)


def test_closed_form_s_kinetic_has_half_laplacian_and_no_cell_average_factor():
    a,beta = .61,1.23
    basis = _normalized_basis([(0,(.1,.2,.3),[a],[1.],True),
                               (0,(.4,-.1,.5),[beta],[1.],True)])
    b = _bundle(basis=basis,lattice=np.eye(3)*8,mesh=(3,1,1),cutoff=2.)
    result = _call(b,1,begin=1,count=1).values_copy()[:,0]
    distance2 = .3**2+.3**2+.2**2
    gamma = a+beta
    expected_s = (4*a*beta/gamma**2)**.75*np.exp(-a*beta/gamma*distance2)
    expected_t = expected_s*a*beta/gamma*(3-2*a*beta/gamma*distance2)
    np.testing.assert_allclose(result,[expected_s,expected_t],atol=5e-15,rtol=2e-14)


def test_pair_tiling_is_bitwise_identical_and_input_names_do_not_change_content():
    b = _bundle()
    whole = _call(b,4)
    split = np.hstack([_call(b,4,begin=i,count=1).values_copy() for i in range(4)])
    np.testing.assert_array_equal(whole.values_copy().view(np.uint64),split.view(np.uint64))
    assert whole.payload_identity_sha256 == _call(b,4).payload_identity_sha256
    assert whole.payload_identity_sha256 != _call(b,1).payload_identity_sha256
    assert whole.operator_source_identity_sha256 != _call(b,4,begin=1,count=1).operator_source_identity_sha256


def test_exact_image_source_wire_covers_all_three_walks_and_payload():
    basis = _basis([(0,(0,0,0),[.7],[1.],True)])
    b = _bundle(basis=basis,lattice=np.eye(3)*2,mesh=(2,1,1),cutoff=2.)
    result = _call(b,1)
    pieces = []
    def u64(v):
        pieces.append(struct.pack(">Q",v%(2**64)))
    def text(value):
        value = value.encode()
        u64(len(value))
        pieces.append(value)
    text("vibeqc.periodic.gaussian-one-electron.images")
    pieces.append(struct.pack(">I",1))
    for n in (1,1,1,0,1):
        u64(n)
    retained = [label for label in itertools.product(range(-3,4),repeat=3)
                if sum(v*v for v in label)<=1]
    for role in range(3):
        u64(0)
        pieces.append(struct.pack(">I",role))
        for value in (-3,-3,-3,3,3,3,343):
            u64(value)
        for label in retained:
            for value in label:
                u64(value)
        u64(len(retained))
    wire = b"".join(pieces)
    assert result.image_source_identity_sha256 == hashlib.sha256(wire).hexdigest()
    assert result.diagnostics.image_identity_wire_bytes == len(wire)
    assert result.diagnostics.retained_primary_pair_images == 7
    assert result.plan.candidate_evaluations == 3*343
    assert result.plan.image_identity_wire_bytes_upper_bound == len(wire)+24*3*(343-7)


@pytest.mark.parametrize("direction",[-1,0,1])
def test_exact_cutoff_boundary_matches_existing_ao_pair_source(direction):
    cutoff = 2. if direction==0 else np.nextafter(2.,0. if direction<0 else np.inf)
    basis = _basis([(0,(0,0,0),[.8],[1.],True)])
    b = _bundle(basis=basis,lattice=np.eye(3)*2,mesh=(1,1,1),cutoff=cutoff)
    result = _call(b)
    ft = _fourier(basis,[[0,0,0]],system=b.system,cutoff=cutoff)
    assert result.diagnostics.retained_primary_pair_images == ft["retained_pair_image_count"]
    assert result.diagnostics.retained_primary_pair_images == (1 if direction<0 else 7)
    np.testing.assert_allclose(result.values_copy()[0,0],ft["values"][0,0],atol=5e-15)


def test_no_retained_image_is_exact_zero_without_false_nuclear_certificate():
    b = _bundle(cutoff=.01)
    result = _call(b,begin=1,count=1)
    np.testing.assert_array_equal(result.values_copy().view(np.uint64),0)
    assert result.diagnostics.retained_primary_pair_images == 0


def test_source_atoms_charge_and_spin_are_deliberately_not_hcore_inputs():
    b = _bundle()
    original = _call(b)
    b.system.unit_cell = [core.Atom(8,[4.,3.,2.])]
    b.system.charge,b.system.multiplicity = 6,9
    changed = _call(b)
    assert changed.operator_source_identity_sha256 == original.operator_source_identity_sha256
    assert changed.payload_identity_sha256 == original.payload_identity_sha256


def test_measured_roundoff_is_preserved_and_tighter_structure_gate_fails():
    b = _bundle()
    baseline = _call(b,4)
    d = baseline.diagnostics
    measured = max(d.maximum_overlap_hermitian_error,d.maximum_kinetic_hermitian_error,
                   d.maximum_overlap_time_reversal_error,d.maximum_kinetic_time_reversal_error,
                   d.maximum_diagonal_imaginary_magnitude)
    assert measured > 0
    options = _options(b)
    options.structural_relative_tolerance = 0.
    options.structural_absolute_tolerance = np.nextafter(measured,0.)
    with pytest.raises(RuntimeError,match="audit failed"):
        _call(b,4,options=options)
    np.testing.assert_array_equal(baseline.values_copy(),_call(b,4).values_copy())


def test_exact_memory_work_and_live_inventory_boundaries():
    b = _bundle()
    live = _live()
    live.replicas_per_node = 3
    live.other_retained_bytes_per_replica = 127
    live.other_transient_bytes_per_replica = 31
    live.external_node_bytes = 101
    p = _call(b,4,live=live,plan=True)
    assert p.output_numeric_bytes == 32*p.pair_count
    assert p.fixed_numeric_workspace_bytes == 23040
    assert p.owned_numeric_peak_bytes == 23040+32*p.pair_count
    assert p.borrowed_basis_active_numeric_bytes == b.context.inventory.combined_borrowed_active_numeric_bytes
    assert p.per_replica_inventoried_bytes == (p.owned_numeric_peak_bytes+p.borrowed_basis_active_numeric_bytes
        +p.fixed_inventoried_object_bytes+127+31+4096)
    assert p.node_inventoried_bytes == 101+3*p.per_replica_inventoried_bytes
    expected = (p.preflight_work_units+128*p.output_numeric_bytes+4096*p.candidate_evaluations
                +512*p.primitive_pair_evaluations_upper_bound+32*p.md_table_cells_upper_bound
                +512*p.cartesian_pair_terms_upper_bound)
    assert p.work_units_upper_bound == expected
    mapping = {"maximum_owned_numeric_bytes":"owned_numeric_peak_bytes",
               "maximum_per_replica_inventoried_bytes":"per_replica_inventoried_bytes",
               "maximum_node_inventoried_bytes":"node_inventoried_bytes",
               "maximum_pair_count":"pair_count","maximum_candidate_evaluations":"candidate_evaluations",
               "maximum_work_units":"work_units_upper_bound"}
    exact = _caps()
    for cap,field in mapping.items():
        setattr(exact,cap,getattr(p,field))
    baseline = _call(b,4,live=live,caps=exact)
    for cap,field in mapping.items():
        setattr(exact,cap,getattr(p,field)-1)
        with pytest.raises((ValueError,RuntimeError),match="cap"):
            _call(b,4,live=live,caps=exact)
        setattr(exact,cap,getattr(p,field))
    assert _call(b,4,live=live,caps=exact).payload_identity_sha256 == baseline.payload_identity_sha256


@pytest.mark.parametrize("field",_BASIS_CAP_FIELDS)
def test_caller_basis_caps_must_cover_exact_context_before_verification(field):
    b = _bundle()
    options = _options(b)
    caps = options.basis_verification_caps
    setattr(caps,field,getattr(caps,field)-1)
    options.basis_verification_caps = caps
    with pytest.raises((ValueError,RuntimeError),match="cap"):
        _call(b,options=options,plan=True)


def test_generous_basis_caps_are_clamped_before_oversized_substitution_scans():
    b = _bundle()
    options = _options(b)
    options.basis_verification_caps = _basis_caps()
    b.ao = _basis([(0,(.1,-.2,.15),[.55]*20,[.7]*20,True),
                   (0,(.6,.3,-.25),[.7],[.8],True)])
    with pytest.raises((ValueError,RuntimeError),match="source-context.*(lanes|wire|payload|work).*cap"):
        _call(b,count=1,options=options,plan=True)


def test_owned_and_work_admission_precede_mutated_basis_verification():
    b = _bundle()
    b.ao = _basis([(0,(.1,-.2,.15),[.8],[.4],True)])
    caps = _caps()
    caps.maximum_owned_numeric_bytes = 1
    with pytest.raises(ValueError,match="owned numerical byte cap"):
        _call(b,count=1,caps=caps,plan=True)
    caps = _caps()
    caps.maximum_work_units = 1
    with pytest.raises(ValueError,match="work cap"):
        _call(b,count=1,caps=caps,plan=True)


def test_changed_content_and_original_cell_fail_closed():
    b = _bundle()
    b.ao = _basis([(0,(.1,-.2,.15),[.56,1.1],[.7,-.12],True),
                   (0,(.6,.3,-.25),[.7],[.8],True)])
    with pytest.raises(ValueError,match="content mismatch"):
        _call(b)
    b = _bundle()
    b.system.lattice = b.system.lattice.T.copy()
    with pytest.raises(ValueError,match="original cell"):
        _call(b)


@pytest.mark.parametrize("field,value",[("structural_absolute_tolerance",np.nan),
    ("structural_relative_tolerance",np.inf),("structural_absolute_tolerance",-1.),
    ("structural_relative_tolerance",1.)])
def test_invalid_structure_controls(field,value):
    b = _bundle()
    options = _options(b)
    setattr(options,field,value)
    with pytest.raises(ValueError,match="tolerance"):
        _call(b,options=options)


@pytest.mark.parametrize("k,begin,count",[(6,0,1),(0,4,1),(0,0,0),(0,2,2**64-1)])
def test_invalid_k_and_pair_extents(k,begin,count):
    with pytest.raises(ValueError,match="interval"):
        _call(_bundle(),k,begin,count,plan=True)


def test_node_replica_overflow_and_zero_controls_fail_before_work():
    b = _bundle()
    live = _live()
    live.replicas_per_node = 2**64-1
    with pytest.raises(OverflowError,match="overflow"):
        _call(b,live=live,plan=True)
    live = _live()
    live.fixed_backend_margin_bytes_per_replica = 0
    with pytest.raises(ValueError,match="positive"):
        _call(b,live=live,plan=True)
    options = _options(b)
    options.structural_absolute_tolerance = options.structural_relative_tolerance = 0.
    with pytest.raises(ValueError,match="both be zero"):
        _call(b,options=options,plan=True)


def test_panel_and_context_survive_borrowed_basis_release_and_copies_detach():
    b = _bundle()
    result = _call(b,4)
    expected = result.values_copy()
    identity = result.payload_identity_sha256
    result.values_copy()[:] = 123
    del b
    gc.collect()
    np.testing.assert_array_equal(result.values_copy(),expected)
    assert result.payload_identity_sha256 == identity
    assert result.context.n_kpoints == 6
    assert result.element(1,2) == expected[1,2]
    with pytest.raises(IndexError):
        result.element(2,0)
