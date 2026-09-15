"""Tiny native reciprocal Ewald integrals, not a complete BIPOLE/CCSD run.

The independent polynomial Gaussian oracle uses the declared cell list, not
the production MD recurrence or a radial image cutoff. No target SCF runs.
"""

from __future__ import annotations

import gc
import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_aopair_fourier_panel import (
    _basis, _mixed_basis, _polynomials, _primitive_pair, _system,
)


def _cell_caps():
    caps = core._AOPairFourierCellPanelCaps()
    caps.maximum_cells = 128
    caps.maximum_pair_cell_visits = 65536
    caps.maximum_work_units = 2_000_000_000
    caps.maximum_output_bytes = 1 << 20
    return caps


def _cell_call(basis, vectors, cells, *, lattice=None, k=None, begin=0, count=None, caps=None):
    return core._ao_pair_gaussian_fourier_cell_panel(
        basis, _system(lattice), np.ascontiguousarray(vectors, dtype=float).reshape(-1, 3),
        np.zeros(3) if k is None else np.ascontiguousarray(k, dtype=float),
        begin, basis.nbasis**2 - begin if count is None else count,
        np.ascontiguousarray(cells, dtype=np.int64).reshape(-1, 3),
        _cell_caps() if caps is None else caps,
    )


def _cell_oracle(basis, vectors, lattice, k, cells):
    orbitals = [(shell, polynomial) for shell in basis.shells()
                for polynomial in _polynomials(shell.l)]
    out = np.zeros((basis.nbasis, basis.nbasis, len(vectors)), complex)
    for mu, (bra, pa) in enumerate(orbitals):
        for nu, (ket, pb) in enumerate(orbitals):
            for label in cells:
                r = lattice @ np.asarray(label)
                phase = np.exp(1j * np.dot(k, r))
                for alpha, ca in zip(bra.exponents, bra.coefficients):
                    for beta, cb in zip(ket.exponents, ket.coefficients):
                        for v, p in enumerate(vectors):
                            out[mu, nu, v] += phase * ca * cb * _primitive_pair(
                                alpha, beta, np.asarray(bra.origin),
                                np.asarray(ket.origin) + r, pa, pb, p,
                            )
    return out.reshape(basis.nbasis**2, len(vectors))


def _two_s():
    return _basis([(0, (0.1, -0.2, 0.15), [0.55, 1.1], [0.7, -0.12], True),
                   (0, (0.6, 0.3, -0.25), [0.7], [0.8], True)])


_LATTICE = np.array([[2.7, 0.2, 0.1], [0.0, 3.1, 0.25], [0.0, 0.0, 2.9]])
_CELLS = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], np.int64)


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("sign", [-1, 1])
def test_explicit_cell_fourier_matches_independent_polynomials(mixed, sign):
    basis = _mixed_basis() if mixed else _two_s()
    vectors = np.array([[0.21, -0.34, 0.17], [-0.71, 0.13, 0.42]])
    k = sign * np.array([0.25, 0.19, -0.11])
    begin, count = (5, 17) if mixed else (0, 4)
    actual = _cell_call(basis, vectors, _CELLS, lattice=_LATTICE, k=k,
                        begin=begin, count=count)["values"]
    expected = _cell_oracle(basis, vectors, _LATTICE, k, _CELLS)[begin:begin+count]
    np.testing.assert_allclose(actual, expected, atol=4e-13, rtol=2e-12)


def test_explicit_cell_domain_is_not_a_pair_separation_cutoff():
    basis = _two_s()
    p = np.array([[0.2, -0.1, 0.3]])
    # Deliberately asymmetric and missing home: exactly these two cells, no repair.
    cells = np.array([[1, 0, 0], [0, 1, 0]], np.int64)
    k = np.array([0.2, 0.1, -0.4])
    actual = _cell_call(basis, p, cells, lattice=_LATTICE, k=k)["values"]
    expected = _cell_oracle(basis, p, _LATTICE, k, cells)
    np.testing.assert_allclose(actual, expected, atol=4e-13, rtol=2e-12)
    home = _cell_call(basis, p, [[0, 0, 0]], lattice=_LATTICE, k=k)["values"]
    assert np.max(np.abs(actual - home)) > 0.01


def test_explicit_cell_tile_splits_and_time_reversal():
    b = _mixed_basis()
    p = np.array([[0.2, -0.1, 0.3], [0.4, 0.2, -0.1]])
    k = np.array([0.2, 0.1, -0.4])
    kw = dict(lattice=_LATTICE, k=k, begin=5, count=17)
    whole = _cell_call(b, p, _CELLS, **kw)["values"]
    columns = np.hstack([_cell_call(b, p[i:i+1], _CELLS, **kw)["values"] for i in range(2)])
    rows = np.vstack([
        _cell_call(b, p, _CELLS, lattice=_LATTICE, k=k, begin=5, count=7)["values"],
        _cell_call(b, p, _CELLS, lattice=_LATTICE, k=k, begin=12, count=10)["values"],
    ])
    np.testing.assert_array_equal(whole, columns)
    np.testing.assert_array_equal(whole, rows)
    tr = _cell_call(b, -p, _CELLS, lattice=_LATTICE, k=-k, begin=5, count=17)["values"]
    np.testing.assert_allclose(tr, whole.conj(), atol=2e-13, rtol=2e-13)


def test_explicit_cell_zero_vector_is_native_overlap():
    b = _mixed_basis()
    actual = np.vstack([
        _cell_call(b, [[0, 0, 0]], [[0, 0, 0]], begin=i, count=min(32, b.nbasis**2-i))["values"]
        for i in range(0, b.nbasis**2, 32)
    ]).reshape(b.nbasis, b.nbasis)
    np.testing.assert_allclose(actual, core.compute_overlap(b), atol=3e-13, rtol=3e-13)


def test_explicit_cell_lifetime_and_readonly_output():
    b = _two_s()
    result = _cell_call(b, [[0.2, 0.1, -0.1]], _CELLS, lattice=_LATTICE)
    actual, expected = result["values"], result["values"].copy()
    del result, b
    gc.collect()
    np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match="read-only"):
        actual[0, 0] = 0


@pytest.mark.parametrize("field,plan_field", [
    ("maximum_cells", "cell_count"),
    ("maximum_pair_cell_visits", "pair_cell_visits"),
    ("maximum_work_units", "work_units"),
    ("maximum_output_bytes", "output_bytes"),
])
def test_explicit_cell_exact_caps_and_minus_one_before_nan(field, plan_field):
    b = _two_s()
    caps = _cell_caps()
    p = core._plan_ao_pair_gaussian_fourier_cell_panel(b, 2, 0, 4, len(_CELLS), caps)
    setattr(caps, field, getattr(p, plan_field))
    _cell_call(b, [[0.1, 0.2, 0.3]] * 2, _CELLS, caps=caps)
    setattr(caps, field, getattr(p, plan_field) - 1)
    with pytest.raises((ValueError, RuntimeError, OverflowError), match="cap|work|byte|cell"):
        _cell_call(b, [[np.nan, 0, 0]] * 2, _CELLS, caps=caps)


@pytest.mark.parametrize("field", ["maximum_cells", "maximum_pair_cell_visits",
                                   "maximum_work_units", "maximum_output_bytes"])
def test_explicit_cell_caps_must_be_explicit(field):
    caps = _cell_caps()
    setattr(caps, field, 0)
    with pytest.raises(ValueError, match="positive"):
        _cell_call(_two_s(), [[0.1, 0.2, 0.3]], _CELLS, caps=caps)


@pytest.mark.parametrize("cells,match", [
    ([[0, 0, 0], [0, 0, 0]], "duplicate"),
    ([[2**53 + 1, 0, 0]], "exact|representable|label"),
])
def test_explicit_cell_bad_labels_rejected(cells, match):
    with pytest.raises(ValueError, match=match):
        _cell_call(_two_s(), [[0.1, 0.2, 0.3]], cells)


@pytest.mark.parametrize("kind", ["vectors", "k", "lattice"])
def test_explicit_cell_nonfinite_inputs_rejected(kind):
    values = dict(vectors=[[0.1, 0.2, 0.3]], k=np.zeros(3), lattice=_LATTICE.copy())
    values[kind] = np.full_like(values[kind], np.nan)
    with pytest.raises(ValueError, match="finite"):
        _cell_call(_two_s(), cells=_CELLS, **values)


@pytest.mark.parametrize("empty", ["cells", "pairs", "vectors"])
def test_explicit_cell_empty_sums(empty):
    r = _cell_call(_two_s(), [] if empty == "vectors" else [[0.1, 0.2, 0.3]],
                   [] if empty == "cells" else _CELLS, count=0 if empty == "pairs" else 4)
    assert np.all(r["values"] == 0)


def test_explicit_cell_adapter_rejects_implicit_conversion():
    with pytest.raises(TypeError):
        core._ao_pair_gaussian_fourier_cell_panel(
            _two_s(), _system(), np.zeros((1, 3)), np.zeros(3), 0, 4,
            _CELLS.astype(float), _cell_caps(),
        )


_GRAM_LATTICE = np.array([[3.4, 0.15, 0.1], [0.0, 3.8, 0.2], [0.0, 0.0, 4.1]])


def _gram_bundle(*, mesh=(3, 1, 1), shift=(0, 0, 0), q=1, left=1, right=2,
                 block=3, basis=None, cells=None):
    options = core._BipoleEwaldGramOptions()
    options.omega = 0.8
    options.reciprocal_energy_cutoff = 2.6
    options.reciprocal_block_size = block
    options.require_reciprocal_conjugacy = True
    options.require_cell_inversion_closure = True
    selection = core._BipoleEwaldGramSelection()
    selection.q_index, selection.left_k_index, selection.right_k_index = q, left, right
    basis = _two_s() if basis is None else basis
    selection.left_pair_count = selection.right_pair_count = min(4, basis.nbasis**2)
    inventory = core._BipoleEwaldGramInventory()
    inventory.numerical_replicas = 1
    inventory.backend_margin_bytes_per_replica = 65536
    caps = core._BipoleEwaldGramCaps()
    caps.ao_panel = _cell_caps()
    caps.maximum_kpoints = 4096
    caps.maximum_reciprocal_candidates = 65536
    caps.maximum_accepted_vectors = 65536
    caps.maximum_reciprocal_blocks = 65536
    caps.maximum_borrowed_numerical_bytes = 16 << 20
    caps.maximum_owned_numerical_bytes = 16 << 20
    caps.maximum_control_storage_bytes = 16 << 20
    caps.maximum_per_replica_inventoried_bytes = 128 << 20
    caps.maximum_node_inventoried_bytes = 128 << 20
    caps.maximum_work_units = 100_000_000_000
    return SimpleNamespace(basis=basis, system=_system(_GRAM_LATTICE),
                           mesh=core._RegularKMesh(list(mesh), list(shift)),
                           cells=_CELLS.copy() if cells is None else np.asarray(cells, np.int64),
                           selection=selection, options=options, inventory=inventory, caps=caps)


def _gram(b, *, plan=False):
    fn = core._plan_bipole_ewald_gram if plan else core._make_bipole_ewald_gram
    return fn(b.basis, b.system, b.mesh, b.cells, b.selection, b.options, b.inventory, b.caps)


def _gram_records(b):
    reciprocal = 2*np.pi*np.linalg.inv(np.asarray(b.system.lattice)).T
    # q belongs to the zero-shift transfer group, even for a shifted k mesh.
    mesh = core._RegularKMesh(list(b.mesh.mesh))
    q = np.asarray(mesh.fractional_at(b.selection.q_index))
    q = (q + 0.5) % 1.0 - 0.5
    vectors, weights = [], []
    for label in itertools.product(range(-4, 5), repeat=3):
        fractional = np.asarray(label) + q
        if np.all(fractional == 0):
            continue
        p = reciprocal @ fractional
        p2 = np.dot(p, p)
        if p2 <= 2*b.options.reciprocal_energy_cutoff:
            vectors.append(p)
            weights.append(4*np.pi/abs(np.linalg.det(b.system.lattice))
                           * np.exp(-p2/(4*b.options.omega**2))/p2)
    return np.asarray(vectors).reshape(-1, 3), np.asarray(weights)


def _gram_oracle(b):
    p, weights = _gram_records(b)
    reciprocal = 2*np.pi*np.linalg.inv(np.asarray(b.system.lattice)).T
    s = b.selection
    left = _cell_oracle(b.basis, p, np.asarray(b.system.lattice),
                        -reciprocal @ b.mesh.fractional_at(s.left_k_index), b.cells)
    right = _cell_oracle(b.basis, p, np.asarray(b.system.lattice),
                         -reciprocal @ b.mesh.fractional_at(s.right_k_index), b.cells)
    return (left[s.left_pair_begin:s.left_pair_begin+s.left_pair_count].conj()*weights) @ (
        right[s.right_pair_begin:s.right_pair_begin+s.right_pair_count].T)


@pytest.mark.parametrize("q", [0, 1, 2])
@pytest.mark.parametrize("shift", [(0, 0, 0), (1, 0, 0)])
def test_ewald_gram_matches_independent_multik_gaussians(q, shift):
    b = _gram_bundle(q=q, shift=shift)
    r = _gram(b)
    np.testing.assert_allclose(r.values_copy(), _gram_oracle(b), atol=3e-13, rtol=3e-12)
    assert r.diagnostics.accepted_vectors == len(_gram_records(b)[0])
    assert r.diagnostics.cell_inversion_closed
    assert r.diagnostics.reciprocal_conjugacy_audited
    assert not r.physical_hamiltonian_certified
    assert not r.symmetry_certified
    assert r.citation_numerics == ["bipole_ewald_gram"]


def test_ewald_gram_mixed_angular_rectangular_selection():
    b = _gram_bundle(basis=_mixed_basis(), cells=[[0, 0, 0]])
    b.selection.left_pair_begin, b.selection.left_pair_count = 7, 5
    b.selection.right_pair_begin, b.selection.right_pair_count = 18, 3
    np.testing.assert_allclose(_gram(b).values_copy(), _gram_oracle(b), atol=5e-13, rtol=3e-12)


@pytest.mark.parametrize("block", [1, 2, 7, 31])
def test_ewald_gram_streaming_blocks_do_not_change_integrals_or_scientific_identity(block):
    b = _gram_bundle()
    first = _gram(b)
    b.options.reciprocal_block_size = block
    second = _gram(b)
    np.testing.assert_array_equal(first.values_copy(), second.values_copy())
    assert first.input_identity_sha256 == second.input_identity_sha256
    assert first.reciprocal_source_identity_sha256 == second.reciprocal_source_identity_sha256
    assert first.payload_identity_sha256 == second.payload_identity_sha256


def test_ewald_gram_pair_slices_and_hermitian_metric():
    b = _gram_bundle(left=1, right=1)
    whole = _gram(b).values_copy()
    np.testing.assert_allclose(whole, whole.conj().T, atol=3e-14)
    assert np.linalg.eigvalsh(whole).min() > -3e-14
    b.selection.left_pair_begin, b.selection.left_pair_count = 1, 2
    b.selection.right_pair_begin, b.selection.right_pair_count = 2, 1
    selected = _gram(b).values_copy()
    np.testing.assert_array_equal(selected, whole[1:3, 2:3])


def test_ewald_gram_scalar_time_reversal_including_shifted_mesh():
    b = _gram_bundle(shift=(1, 0, 0))
    first = _gram(b).values_copy()
    b.selection.q_index = 2
    b.selection.left_k_index = b.mesh.negate_index(b.selection.left_k_index)
    b.selection.right_k_index = b.mesh.negate_index(b.selection.right_k_index)
    np.testing.assert_allclose(_gram(b).values_copy(), first.conj(), atol=3e-13, rtol=3e-12)


def test_ewald_gram_has_no_hidden_k_weight():
    a = _gram_bundle()
    b = _gram_bundle(mesh=(6, 1, 1), q=2, left=2, right=4)
    np.testing.assert_allclose(_gram(a).values_copy(), _gram(b).values_copy(), atol=3e-13, rtol=3e-12)


@pytest.mark.parametrize("material", ["mgo", "diamond"])
def test_project_basis_one_selected_integral_on_8_cubed_addressing_only(material):
    # One home-cell pair on each side, ONE q, no SCF, localization or CC.
    # This cannot qualify a target-system energy or finite-image accuracy.
    lattice_constant = (4.21 if material == "mgo" else 3.567)/0.529177210903
    lattice = lattice_constant/2*np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    fraction = np.full(3, 0.5 if material == "mgo" else 0.25)
    numbers = (12, 8) if material == "mgo" else (6, 6)
    system = core.PeriodicSystem(3, lattice, [core.Atom(numbers[0], [0, 0, 0]),
                                            core.Atom(numbers[1], lattice @ fraction)])
    basis = core.BasisSet(system.unit_cell_molecule(), "pob-tzvp-rev2")
    b = _gram_bundle(mesh=(8, 8, 8), q=1, left=319, right=227,
                     basis=basis, cells=[[0, 0, 0]], block=1)
    b.system = system
    b.options.reciprocal_energy_cutoff = 0.2
    b.selection.left_pair_count = b.selection.right_pair_count = 1
    b.selection.right_pair_begin = basis.nbasis**2-1
    r = _gram(b)
    assert r.memory.n_kpoints == 512
    assert r.memory.retained_output_bytes == 16
    assert r.memory.peak_owned_numerical_bytes < 32768
    assert r.diagnostics.accepted_vectors > 0
    assert np.isfinite(r.element(0, 0))
    assert not r.physical_hamiltonian_certified


def test_ewald_gram_inversion_covariance_of_actual_finite_source():
    center = np.array([0.3, -0.2, 0.1])
    basis = _basis([(0, center, [0.4], [0.8], True),
                    (0, -center, [0.4], [0.8], True)])
    b = _gram_bundle(basis=basis)
    original = _gram(b).values_copy()
    b.selection.q_index = 2
    b.selection.left_k_index, b.selection.right_k_index = 2, 1
    inverted = _gram(b).values_copy()
    pair_permutation = [3, 2, 1, 0]
    np.testing.assert_allclose(inverted, original[np.ix_(pair_permutation, pair_permutation)],
                               atol=3e-13, rtol=3e-12)


def test_ewald_gram_optional_conjugacy_does_not_publish_an_unperformed_audit():
    b = _gram_bundle()
    b.options.require_reciprocal_conjugacy = False
    r = _gram(b)
    assert not r.diagnostics.reciprocal_conjugacy_audited
    assert r.diagnostics.cell_inversion_closed


@pytest.mark.parametrize("field", ["q_index", "left_k_index", "right_k_index"])
def test_ewald_gram_rejects_out_of_mesh_indices(field):
    b = _gram_bundle()
    setattr(b.selection, field, len(b.mesh))
    with pytest.raises((IndexError, ValueError), match="index|mesh"):
        _gram(b)


def test_ewald_gram_known_count_cap_precedes_lattice_payload():
    b = _gram_bundle()
    b.caps.maximum_kpoints = 1
    b.system.lattice = np.full((3, 3), np.nan)
    with pytest.raises((ValueError, RuntimeError), match="kpoint|k-point|k point|mesh|cap"):
        _gram(b)


@pytest.mark.parametrize("field", ["numerical_replicas", "backend_margin_bytes_per_replica"])
def test_ewald_gram_requires_replica_and_backend_margin(field):
    b = _gram_bundle()
    setattr(b.inventory, field, 0)
    with pytest.raises(ValueError, match="positive|replica|margin"):
        _gram(b)


def test_ewald_gram_replica_inventory_overflow():
    b = _gram_bundle()
    b.inventory.numerical_replicas = 2**64-1
    with pytest.raises((OverflowError, ValueError, RuntimeError), match="overflow|cap"):
        _gram(b)


def test_ewald_gram_zero_mode_omitted_and_zero_weight_underflow_supported():
    b = _gram_bundle(q=0)
    b.options.omega = 1e-6
    r = _gram(b)
    assert r.diagnostics.accepted_vectors > 0
    assert r.diagnostics.zero_weight_vectors == r.diagnostics.accepted_vectors
    np.testing.assert_array_equal(r.values_copy(), np.zeros((4, 4)))


def test_ewald_gram_empty_gamma_envelope_does_not_insert_a_zero_mode():
    b = _gram_bundle(q=0)
    b.options.reciprocal_energy_cutoff = 0.001
    r = _gram(b)
    assert r.diagnostics.accepted_vectors == 0
    assert r.diagnostics.evaluated_blocks == 0
    np.testing.assert_array_equal(r.values_copy(), np.zeros((4, 4)))


@pytest.mark.parametrize("field", ["left_pair_count", "right_pair_count"])
def test_ewald_gram_empty_selected_interval(field):
    b = _gram_bundle()
    setattr(b.selection, field, 0)
    r = _gram(b)
    assert r.values_copy().size == 0
    assert r.diagnostics.accepted_vectors > 0
    assert not r.physical_hamiltonian_certified


@pytest.mark.parametrize("coefficient_exponent", [-270, 270])
def test_ewald_gram_weighted_product_avoids_premature_underflow_or_overflow(coefficient_exponent):
    from decimal import Decimal, localcontext

    coefficient = float(2.0**coefficient_exponent)
    basis = _basis([(0, (0, 0, 0), [0.5], [coefficient], True)])
    b = _gram_bundle(basis=basis, mesh=(64, 1, 1), q=1, left=0, right=0,
                     cells=[[0, 0, 0]])
    b.system.lattice = np.eye(3)*4
    b.options.reciprocal_energy_cutoff = 0.001
    if coefficient_exponent > 0:
        # Choose a small positive weight which brings an otherwise overflowing
        # B*B product back into finite range. No arbitrary factor injection.
        b.options.omega = 0.0014
    p, weights = _gram_records(b)
    assert len(p) == 1
    # Independent analytic s-product FT; Decimal postpones multiplication
    # underflow/overflow, unlike the buggy unweighted complex product.
    stored = basis.shells()[0].coefficients[0]
    with localcontext() as ctx:
        ctx.prec = 100
        rho = (Decimal.from_float(stored)**2
               * Decimal.from_float(np.pi**1.5)
               * Decimal.from_float(np.exp(-np.dot(p[0], p[0])/4)))
        expected = float(Decimal.from_float(weights[0]) * rho**2)
    assert np.isfinite(expected) and expected > 0
    actual = _gram(b).element(0, 0)
    assert actual.imag == 0
    # Subnormal results have a coarse relative grid; use a two-ULP absolute
    # allowance there, not a tolerance large enough to admit a silent zero.
    assert abs(actual.real - expected) <= max(3e-12*expected, 2*np.nextafter(0.0, 1.0))
    assert actual.real > 0


def test_ewald_weighted_product_preserves_a_restored_cancellation_residual():
    # Deliberately a scalar arithmetic witness, not an authenticated Gaussian
    # factor source. Rounded normal products cancel before weighting, whereas
    # the exact binary-input residual is representable after weighting.
    x, u, weight = 2.0**-510, 2.0**-52, 2.0**60
    left, right = complex(x, -x*(1+u)), complex(x, x*(1-u))
    assert (left.conjugate()*right*weight).real == 0
    actual = core._bipole_ewald_weighted_product_diagnostic(left, right, weight)
    assert actual.real == 2.0**-1064
    assert actual.imag == pytest.approx(2.0**-959, rel=3e-16)


@pytest.mark.parametrize("left,right,weight", [
    (complex(np.inf, 0), 1+0j, 1.0), (1+0j, complex(0, np.nan), 1.0),
    (1+0j, 1+0j, np.inf), (1+0j, 1+0j, -1.0),
])
def test_ewald_scalar_diagnostic_rejects_invalid_inputs(left, right, weight):
    with pytest.raises((ValueError, OverflowError), match="finite|weight|nonnegative"):
        core._bipole_ewald_weighted_product_diagnostic(left, right, weight)


def test_ewald_gram_cell_closure_is_measured_not_repaired():
    b = _gram_bundle(cells=[[0, 0, 0], [1, 0, 0]])
    with pytest.raises(ValueError, match="inversion|closure"):
        _gram(b)
    b.options.require_cell_inversion_closure = False
    r = _gram(b)
    assert not r.diagnostics.cell_inversion_closed
    np.testing.assert_allclose(r.values_copy(), _gram_oracle(b), atol=3e-13, rtol=3e-12)


def test_ewald_gram_scientific_inputs_and_actual_cells_change_identity():
    b = _gram_bundle()
    initial = _gram(b)
    b.options.omega *= 1.1
    changed = _gram(b)
    assert changed.input_identity_sha256 != initial.input_identity_sha256
    assert changed.reciprocal_source_identity_sha256 != initial.reciprocal_source_identity_sha256
    assert np.max(np.abs(changed.values_copy() - initial.values_copy())) > 1e-4
    b.cells = b.cells[:3].copy()
    changed_cells = _gram(b)
    assert changed_cells.input_identity_sha256 != changed.input_identity_sha256


_GRAM_CAP_MAP = [
    ("maximum_kpoints", "n_kpoints"),
    ("maximum_reciprocal_candidates", "reciprocal_candidates"),
    ("maximum_accepted_vectors", "accepted_vectors_upper_bound"),
    ("maximum_reciprocal_blocks", "reciprocal_blocks_upper_bound"),
    ("maximum_borrowed_numerical_bytes", "borrowed_numerical_bytes"),
    ("maximum_owned_numerical_bytes", "peak_owned_numerical_bytes"),
    ("maximum_control_storage_bytes", "control_storage_bytes"),
    ("maximum_per_replica_inventoried_bytes", "per_replica_inventoried_bytes"),
    ("maximum_node_inventoried_bytes", "required_node_inventoried_bytes"),
    ("maximum_work_units", "work_units_upper_bound"),
]


@pytest.mark.parametrize("field,plan_field", _GRAM_CAP_MAP)
def test_ewald_gram_exact_resource_caps_and_minus_one(field, plan_field):
    b = _gram_bundle(q=0)
    p = _gram(b, plan=True)
    setattr(b.caps, field, getattr(p, plan_field))
    _gram(b)
    setattr(b.caps, field, getattr(p, plan_field)-1)
    with pytest.raises((ValueError, RuntimeError, OverflowError), match="cap|bound|work|byte|storage"):
        _gram(b)


@pytest.mark.parametrize("field", [row[0] for row in _GRAM_CAP_MAP])
def test_ewald_gram_requires_explicit_resource_caps(field):
    b = _gram_bundle()
    setattr(b.caps, field, 0)
    with pytest.raises(ValueError, match="positive"):
        _gram(b)


def test_ewald_gram_node_inventory_and_replicas():
    b = _gram_bundle()
    a = _gram(b, plan=True)
    b.inventory.numerical_replicas = 3
    b.inventory.external_node_bytes = 12345
    b.inventory.other_live_numerical_bytes_per_replica = 23456
    b.inventory.other_live_control_bytes_per_replica = 789
    p = _gram(b, plan=True)
    assert p.per_replica_inventoried_bytes == a.per_replica_inventoried_bytes + 23456 + 789
    assert p.required_node_inventoried_bytes == 3*p.per_replica_inventoried_bytes + 12345
    assert p.peak_owned_numerical_bytes == a.peak_owned_numerical_bytes


def test_ewald_gram_copy_is_independent_and_result_owns_payload():
    b = _gram_bundle()
    r = _gram(b)
    a = r.values_copy()
    expected = a.copy()
    a.setflags(write=True)
    a[:] = 0
    np.testing.assert_array_equal(r.values_copy(), expected)
    del b, a
    gc.collect()
    assert r.element(1, 2) == expected[1, 2]
    with pytest.raises(IndexError):
        r.element(4, 0)


@pytest.mark.parametrize("omega", [0.0, -1.0, np.nan, np.inf])
def test_ewald_gram_invalid_omega(omega):
    b = _gram_bundle()
    b.options.omega = omega
    with pytest.raises(ValueError, match="omega|finite|positive"):
        _gram(b)


def test_ewald_gram_recontracts_bipole_jk_on_complete_hermitian_density_basis():
    from vibeqc.bipole_fock_ewald import (
        _build_j_long_range_cache, build_k_exchange_long_range_cache,
        compute_J_long_range_at_k, compute_K_long_range_at_k,
        compute_rho_hat_from_k_density,
    )

    # Exact halves avoid the old cache's decimal q rounding. This is a
    # matched-envelope numerical comparison, not equality of source owners.
    b = _gram_bundle(mesh=(2, 1, 1), q=0, left=0, right=0)
    n, nk = b.basis.nbasis, len(b.mesh)
    reciprocal = 2*np.pi*np.linalg.inv(np.asarray(b.system.lattice)).T
    ks = [reciprocal @ b.mesh.fractional_at(i) for i in range(nk)]
    weights = [1/nk]*nk
    jcache = _build_j_long_range_cache(
        b.basis, b.system, b.cells @ np.asarray(b.system.lattice).T,
        b.options.omega, 1e-8, K_max=np.sqrt(2*b.options.reciprocal_energy_cutoff),
    )
    xcache = build_k_exchange_long_range_cache(
        b.basis, b.system, jcache, K_max=np.sqrt(2*b.options.reciprocal_energy_cutoff),
    )
    # compute_J only needs the matching finite cell count when cache/rho are given.
    lattice_density = SimpleNamespace(nbf=n, cells=[SimpleNamespace(r_cart=r)
        for r in b.cells @ np.asarray(b.system.lattice).T])
    jkernels, kkernels = {}, {}
    for target in range(nk):
        for source in range(nk):
            b.selection.q_index = 0
            b.selection.left_k_index, b.selection.right_k_index = target, source
            jkernels[target, source] = _gram(b).values_copy().reshape(n, n, n, n)
            b.selection.q_index = b.mesh.transfer_index(source, target)
            b.selection.left_k_index = b.selection.right_k_index = source
            kkernels[target, source] = _gram(b).values_copy().reshape(n, n, n, n)
    hermitian = [np.diag([1, 0]), np.diag([0, 1]), np.array([[0, 1], [1, 0]]),
                 np.array([[0, 1j], [-1j, 0]])]
    density_basis, j_responses, k_responses = [], [], []
    for populated in range(nk):
        for component in hermitian:
            density = [np.zeros((n, n), complex) for _ in range(nk)]
            density[populated] = np.asarray(component, complex)
            density_basis.append(np.asarray(density))
            jr, kr = [], []
            rho = compute_rho_hat_from_k_density(density, ks, weights, jcache)
            for target in range(nk):
                native_j = sum(np.einsum("mnls,ls->mn", jkernels[target, source], density[source])/nk
                               for source in range(nk))
                native_k = sum(np.einsum("mlns,ls->mn", kkernels[target, source], density[source])/nk
                               for source in range(nk))
                expected_j = compute_J_long_range_at_k(
                    lattice_density, b.basis, b.system, b.options.omega, ks[target],
                    cache=jcache, rho_hat=rho,
                )
                expected_k = compute_K_long_range_at_k(xcache, ks[target], ks, weights, density)
                # The tested inversion-closed two-k fixture is Hermitian without
                # native repair. An arbitrary finite domain is not certified so.
                np.testing.assert_allclose(native_j, native_j.conj().T, atol=5e-13)
                np.testing.assert_allclose(native_k, native_k.conj().T, atol=5e-13)
                np.testing.assert_allclose(native_j, expected_j, atol=5e-13, rtol=4e-12)
                np.testing.assert_allclose(native_k, expected_k, atol=5e-13, rtol=4e-12)
                jr.append(native_j)
                kr.append(native_k)
            j_responses.append(np.asarray(jr))
            k_responses.append(np.asarray(kr))
    # Complete-density-basis adjoint check, independently of the old driver:
    # <D1,G[D2]> = <G[D1],D2> for the uniform-mesh trace inner product.
    # A single symmetric SCF fixed point would not establish this property.
    for responses in (j_responses, k_responses):
        matrix = np.array([[np.einsum("kmn,knm->", d, response).real/nk
                            for response in responses] for d in density_basis])
        np.testing.assert_allclose(matrix, matrix.T, atol=8e-13, rtol=5e-12)


# Shared HF arithmetic, with exact transfer addresses and no projection.
def _hf_lr_caches(b):
    from vibeqc.bipole_fock_ewald import (
        _build_j_long_range_cache, build_k_exchange_long_range_cache,
    )
    radius = np.sqrt(2 * b.options.reciprocal_energy_cutoff)
    j = _build_j_long_range_cache(
        b.basis, b.system, b.cells @ np.asarray(b.system.lattice).T,
        b.options.omega, 1e-8, K_max=radius,
    )
    return j, build_k_exchange_long_range_cache(b.basis, b.system, j, K_max=radius)


@pytest.mark.parametrize("mesh,shift", [
    ((1, 1, 1), (0, 0, 0)), ((2, 1, 1), (0, 0, 0)),
    ((3, 1, 1), (0, 0, 0)), ((3, 1, 1), (1, 0, 0)),
    ((2, 2, 1), (1, 1, 0)), ((3, 2, 1), (1, 1, 1)),
])
def test_shared_hf_lr_arithmetic_matches_native_on_complete_complex_density_basis(mesh, shift):
    from vibeqc.bipole_fock_ewald import (
        _compute_J_long_range_from_rho_linear, _contract_K_long_range_channel_linear,
        compute_rho_hat_from_k_density,
    )
    b = _gram_bundle(mesh=mesh, shift=shift, q=0, left=0, right=0)
    jcache, xcache = _hf_lr_caches(b)
    n, nk = b.basis.nbasis, len(b.mesh)
    reciprocal = 2 * np.pi * np.linalg.inv(np.asarray(b.system.lattice)).T
    ks = [reciprocal @ b.mesh.fractional_at(k) for k in range(nk)]
    kernels = [{}, {}]
    channels = {}
    for target, source in itertools.product(range(nk), repeat=2):
        channels[target, source] = xcache._channel_tables_on_mesh(b.mesh, target, source)
        for kind in range(2):
            b.selection.q_index = 0 if kind == 0 else b.mesh.transfer_index(source, target)
            b.selection.left_k_index = target if kind == 0 else source
            b.selection.right_k_index = source
            result = _gram(b)
            native = result.values_copy()
            np.testing.assert_allclose(native, _gram_oracle(b), atol=5e-13, rtol=4e-12)
            kernels[kind][target, source] = native.reshape((n,) * 4)
            assert not result.physical_hamiltonian_certified
    # Every real and imaginary matrix unit at every source k. Unlike an SCF
    # density, these include non-Hermitian and non-time-reversal directions.
    largest_unprojected = 0.0
    for source, row, column, factor in itertools.product(range(nk), range(n), range(n), (1, 1j)):
        density = np.zeros((nk, n, n), complex)
        density[source, row, column] = factor
        rho = compute_rho_hat_from_k_density(density, ks, [1 / nk] * nk, jcache)
        for target in range(nk):
            j = _compute_J_long_range_from_rho_linear(jcache, ks[target], rho)
            weights, pair = channels[target, source]
            k = _contract_K_long_range_channel_linear(weights, pair, density[source]) / nk
            for kind, actual, axes in ((0, j, "mnls,ls->mn"), (1, k, "mlns,ls->mn")):
                expected = np.einsum(axes, kernels[kind][target, source], density[source]) / nk
                np.testing.assert_allclose(actual, expected, atol=8e-13, rtol=5e-12)
                largest_unprojected = max(largest_unprojected, np.max(np.abs(actual - actual.conj().T)))
    assert largest_unprojected > 1e-3


def test_exact_mesh_hf_channel_avoids_decimal_rounding_of_thirds():
    b = _gram_bundle(mesh=(3, 1, 1), q=1, left=0, right=0)
    _, cache = _hf_lr_caches(b)
    reciprocal = 2 * np.pi * np.linalg.inv(np.asarray(b.system.lattice)).T
    exact = cache._channel_tables_on_mesh(b.mesh, 1, 0)
    legacy = cache.channel_tables(reciprocal @ b.mesh.fractional_at(1), np.zeros(3))
    # Record an existing numerical mismatch; the legacy SCF selection is
    # deliberately retained until the full operator/source contract is ready.
    assert np.max(np.abs(exact[0] - legacy[0])) > 1e-10
    expected_vectors, expected_weights = _gram_records(b)
    np.testing.assert_allclose(exact[0], expected_weights, atol=2e-14, rtol=2e-14)
    expected_pairs = _cell_oracle(b.basis, expected_vectors, np.asarray(b.system.lattice),
                                  np.zeros(3), b.cells).reshape(exact[1].shape)
    np.testing.assert_allclose(exact[1], expected_pairs, atol=3e-13, rtol=3e-12)
    assert len(cache.q_channels) == 2  # Different numerical sources cannot alias.


def test_exact_mesh_hf_channel_cache_reuses_transfers_and_separates_mesh_sizes():
    b = _gram_bundle(mesh=(3, 1, 1))
    jcache, cache = _hf_lr_caches(b)
    first = cache._channel_tables_on_mesh(b.mesh, 1, 0)
    repeated = cache._channel_tables_on_mesh(b.mesh, 1, 0)
    same_transfer = cache._channel_tables_on_mesh(b.mesh, 2, 1)
    assert first[0] is repeated[0] is same_transfer[0]
    assert first[1] is repeated[1]
    assert first[1] is not same_transfer[1]
    diagonal = cache._channel_tables_on_mesh(b.mesh, 2, 2)
    assert diagonal[0] is jcache.kernel
    other_mesh = core._RegularKMesh([5, 1, 1])
    other = cache._channel_tables_on_mesh(other_mesh, 1, 0)
    assert other[0] is not first[0]
    assert len(cache.q_channels) == 2
    shifted = core._RegularKMesh([3, 1, 1], [1, 0, 0])
    shifted_result = cache._channel_tables_on_mesh(shifted, 1, 0)
    assert shifted_result[0] is first[0]
    assert shifted_result[1] is not first[1]


@pytest.mark.parametrize("mesh,target,source,error", [
    (None, 0, 0, TypeError), ((3, 1, 1), 0, 0, TypeError),
    (core._RegularKMesh([3, 1, 1]), 3, 0, RuntimeError),
    (core._RegularKMesh([3, 1, 1]), 0, 3, RuntimeError),
])
def test_exact_mesh_hf_channel_rejects_bad_addresses_before_cache_population(mesh, target, source, error):
    _, cache = _hf_lr_caches(_gram_bundle())
    with pytest.raises(error):
        cache._channel_tables_on_mesh(mesh, target, source)
    assert not cache.q_channels
