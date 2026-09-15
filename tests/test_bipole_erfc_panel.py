"""Selected physical erfc quartets, not a complete periodic Hamiltonian.

The independent oracle integrates a six-dimensional Gaussian analytically
at each attenuation quadrature point. It does not use libint recurrences.
McMurchie/Davidson (1978), Secs. 2 and 3C, and Sun (2023), Eqs. 18-22,
fix the Gaussian product prefactor and the short-range interaction.
"""

from __future__ import annotations

import itertools
import math
from functools import lru_cache
from fractions import Fraction

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_aopair_fourier_panel import _basis, _polynomials, _system


_LATTICE = np.array([[2.7, 0.2, 0.1], [0.0, 3.1, 0.25], [0.0, 0.0, 2.9]])
_IMAGES = np.array([
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[1, 0, 0], [0, -1, 0], [1, 0, 0]],
], dtype=np.int64)
_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(48)


def _two_s():
    return _basis([
        (0, (0.1, -0.2, 0.15), [0.55, 1.1], [0.7, -0.12], True),
        (0, (0.6, 0.3, -0.25), [0.7], [0.8], True),
    ])


def _angular_basis():
    return _basis([
        (0, (0.1, -0.2, 0.15), [0.55], [0.7], True),
        (1, (0.6, 0.3, -0.25), [0.7], [0.8], True),
        (2, (-0.2, 0.1, 0.3), [0.9], [-0.6], True),
    ])


def _axis_polynomial_expectation(powers, centers, mean, covariance):
    """Joint Gaussian moments via integration by parts, not AO recursion."""
    @lru_cache(None)
    def moment(i, j):
        if i == j == 0:
            return np.ones_like(mean[0])
        if i:
            value = mean[0] * moment(i - 1, j)
            if i > 1:
                value = value + (i - 1) * covariance[0] * moment(i - 2, j)
            if j:
                value = value + j * covariance[1] * moment(i - 1, j - 1)
            return value
        value = mean[1] * moment(0, j - 1)
        if j > 1:
            value = value + (j - 1) * covariance[2] * moment(0, j - 2)
        return value

    result = np.zeros_like(mean[0])
    for degrees in itertools.product(*(range(n + 1) for n in powers)):
        coefficient = math.prod(
            math.comb(n, d) * (-center)**(n - d)
            for n, d, center in zip(powers, degrees, centers)
        )
        result += coefficient * moment(sum(degrees[:2]), sum(degrees[2:]))
    return result


def _primitive_erfc(exponents, centers, polynomials, omega):
    a, b, c, d = exponents
    p, q = a + b, c + d
    rho = p * q / (p + q)
    ca, cb, cc, cd = centers
    cp, cq = (a * ca + b * cb) / p, (c * cc + d * cd) / q
    lower = omega / math.sqrt(omega**2 + rho)
    u = lower + (1 - lower) * (_NODES + 1) / 2
    weights = _WEIGHTS * (1 - lower) / 2
    u2 = u * u
    covariance = ((1 - u2) / (2 * p) + u2 / (2 * (p + q)),
                  u2 / (2 * (p + q)),
                  (1 - u2) / (2 * q) + u2 / (2 * (p + q)))
    means = (cp[:, None] - q / (p + q) * (cp - cq)[:, None] * u2,
             cq[:, None] + p / (p + q) * (cp - cq)[:, None] * u2)
    polynomial = np.zeros_like(u)
    for terms in itertools.product(*(list(poly.items()) for poly in polynomials)):
        factor = math.prod(term[1] for term in terms)
        value = np.ones_like(u)
        for axis in range(3):
            value *= _axis_polynomial_expectation(
                [term[0][axis] for term in terms], centers[:, axis],
                (means[0][axis], means[1][axis]), covariance,
            )
        polynomial += factor * value
    prefactor = 2 * math.pi**2.5 / (p * q * math.sqrt(p + q))
    prefactor *= math.exp(-a * b / p * np.dot(ca - cb, ca - cb)
                         - c * d / q * np.dot(cc - cd, cc - cd))
    return prefactor * np.dot(weights, polynomial * np.exp(-rho * np.dot(cp - cq, cp - cq) * u2))


def _oracle(basis, images, *, lattice=_LATTICE, left=(0, 4), right=(0, 4), omega=0.6):
    orbitals = [(shell, poly) for shell in basis.shells() for poly in _polynomials(shell.l)]
    result = np.zeros((len(images), left[1], right[1]))
    for image, labels in enumerate(images):
        shifts = np.vstack([np.zeros(3), np.asarray(labels) @ lattice.T])
        for i in range(left[1]):
            mu, nu = divmod(left[0] + i, basis.nbasis)
            for j in range(right[1]):
                lam, sig = divmod(right[0] + j, basis.nbasis)
                selected = [orbitals[x] for x in (mu, nu, lam, sig)]
                centers = np.array([shell.origin for shell, _ in selected]) + shifts
                primitives = [list(zip(shell.exponents, shell.coefficients)) for shell, _ in selected]
                result[image, i, j] = math.fsum(
                    math.prod(x[1] for x in primitive) * _primitive_erfc(
                        [x[0] for x in primitive], centers,
                        [poly for _, poly in selected], omega,
                    ) for primitive in itertools.product(*primitives)
                )
    return result


def _controls(left=(0, 4), right=(0, 4), *, omega=0.6):
    selection = core._BipoleErfcPanelSelection()
    selection.left_pair_begin, selection.left_pair_count = left
    selection.right_pair_begin, selection.right_pair_count = right
    options = core._BipoleErfcPanelOptions()
    options.omega = omega
    inventory = core._BipoleErfcPanelInventory()
    inventory.numerical_replicas = 1
    inventory.backend_margin_bytes_per_replica = 8 << 20
    caps = core._BipoleErfcPanelCaps()
    caps.maximum_images = 64
    caps.maximum_shell_quartet_calls = 100_000
    caps.maximum_primitive_quartet_visits = 1_000_000
    caps.maximum_borrowed_numerical_bytes = 1 << 20
    caps.maximum_owned_numerical_bytes = 128 << 20
    caps.maximum_control_storage_bytes = 16 << 20
    caps.maximum_worker_bytes = 256 << 20
    caps.maximum_node_bytes = 512 << 20
    caps.maximum_work_units = 10**11
    return selection, options, inventory, caps


def _call(basis=None, images=_IMAGES, *, left=(0, 4), right=(0, 4), omega=0.6,
          lattice=_LATTICE, controls=None, plan=False):
    function = core._plan_bipole_erfc_panel if plan else core._make_bipole_erfc_panel
    return function(
        _two_s() if basis is None else basis, _system(lattice),
        np.ascontiguousarray(images, dtype=np.int64).reshape(-1, 3, 3),
        *(_controls(left, right, omega=omega) if controls is None else controls),
    )


def test_independent_zero_attenuation_oracle_matches_existing_eri():
    basis = _angular_basis()
    left, right = (3, 4), (13, 3)
    expected = _oracle(basis, _IMAGES[:1], left=left, right=right, omega=0)[0]
    eri = np.asarray(core.compute_eri(basis)).reshape(basis.nbasis**2, basis.nbasis**2)
    np.testing.assert_allclose(expected, eri[left[0]:left[0]+left[1], right[0]:right[0]+right[1]],
                               atol=2e-13, rtol=2e-12)


@pytest.mark.parametrize("omega", [0.1, 0.6, 2.0])
def test_erfc_panel_matches_independent_s_gaussian_oracle(omega):
    result = _call(omega=omega)
    expected = _oracle(_two_s(), _IMAGES, omega=omega)
    np.testing.assert_allclose(result.values_copy(), expected, atol=3e-13, rtol=3e-12)


def test_erfc_panel_matches_independent_angular_gaussian_moments():
    basis = _angular_basis()
    left, right = (3, 7), (13, 6)
    actual = _call(basis, left=left, right=right).values_copy()
    expected = _oracle(basis, _IMAGES, left=left, right=right)
    np.testing.assert_allclose(actual, expected, atol=3e-13, rtol=3e-12)


def test_erfc_image_and_pair_slabs_preserve_values_and_order():
    whole = _call().values_copy()
    by_image = np.concatenate([_call(images=_IMAGES[i:i+1]).values_copy() for i in range(2)])
    by_left = np.concatenate([_call(left=(i, 1)).values_copy() for i in range(4)], axis=1)
    by_right = np.concatenate([_call(right=(i, 1)).values_copy() for i in range(4)], axis=2)
    for tiled in (by_image, by_left, by_right):
        np.testing.assert_array_equal(tiled, whole)
    repeated = _call(images=_IMAGES[[1, 0, 1]]).values_copy()
    np.testing.assert_array_equal(repeated, whole[[1, 0, 1]])


def test_erfc_source_is_not_a_mutable_buffer_or_hamiltonian_certificate():
    images = _IMAGES.copy()
    result = _call(images=images)
    values = result.values_copy()
    assert not values.flags.writeable
    images[:] = 0
    again = result.values_copy()
    assert not np.shares_memory(values, again)
    np.testing.assert_array_equal(again, values)
    assert result.element(1, 2, 3) == values[1, 2, 3]
    for indices in ((2, 0, 0), (0, 4, 0), (0, 0, 4)):
        with pytest.raises((IndexError, ValueError)):
            result.element(*indices)
    assert result.citation_numerics == ["bipole_erfc_panel"]
    assert not result.physical_hamiltonian_certified
    assert not result.symmetry_certified
    assert result.diagnostics.primitive_screening_precision == 0
    assert result.diagnostics.shell_quartet_calls <= result.memory.shell_quartet_calls_upper_bound


def test_erfc_actual_source_identities_separate_input_payload_and_resources():
    base = _call()
    controls = _controls()
    controls[2].backend_margin_bytes_per_replica *= 2
    resources = _call(controls=controls)
    for name in ("input_identity_sha256", "source_identity_sha256", "payload_identity_sha256"):
        digest = getattr(base, name)
        assert len(digest) == 64 and int(digest, 16) >= 0
        assert digest == getattr(resources, name)
    changed_omega = _call(omega=0.7)
    changed_images = _call(images=_IMAGES[::-1])
    assert changed_omega.source_identity_sha256 != base.source_identity_sha256
    assert changed_images.source_identity_sha256 == base.source_identity_sha256
    for changed in (changed_omega, changed_images):
        assert changed.input_identity_sha256 != base.input_identity_sha256
        assert changed.payload_identity_sha256 != base.payload_identity_sha256


@pytest.mark.parametrize("permutation", ["left", "right", "electron"])
def test_physical_quartet_permutations_recenter_exact_images(permutation):
    original = _call().values_copy().reshape(2, 2, 2, 2, 2)
    labels = _IMAGES.copy()
    g, p, s = labels[:, 0].copy(), labels[:, 1].copy(), labels[:, 2].copy()
    if permutation == "left":
        labels[:] = np.stack([-g, p - g, s - g], axis=1)
        expected = original.swapaxes(1, 2)
    elif permutation == "right":
        labels[:] = np.stack([g, s, p], axis=1)
        expected = original.swapaxes(3, 4)
    else:
        labels[:] = np.stack([s - p, -p, g - p], axis=1)
        expected = original.transpose(0, 3, 4, 1, 2)
    actual = _call(images=labels).values_copy().reshape(2, 2, 2, 2, 2)
    np.testing.assert_allclose(actual, expected, atol=3e-13, rtol=3e-12)


_CAP_FIELDS = {
    "maximum_images": "image_count",
    "maximum_shell_quartet_calls": "shell_quartet_calls_upper_bound",
    "maximum_primitive_quartet_visits": "primitive_quartet_visits_upper_bound",
    "maximum_borrowed_numerical_bytes": "borrowed_numerical_bytes",
    "maximum_owned_numerical_bytes": "peak_owned_numerical_bytes",
    "maximum_control_storage_bytes": "control_storage_bytes",
    "maximum_worker_bytes": "per_replica_inventoried_bytes",
    "maximum_node_bytes": "required_node_inventoried_bytes",
    "maximum_work_units": "work_units_upper_bound",
}


@pytest.mark.parametrize("field", list(_CAP_FIELDS))
def test_erfc_exact_caps_and_cap_minus_one(field):
    plan = _call(plan=True)
    controls = _controls()
    amount = getattr(plan, _CAP_FIELDS[field])
    setattr(controls[3], field, amount)
    _call(controls=controls)
    setattr(controls[3], field, amount - 1)
    # An unsafe label must not be read until count/resource admission passes.
    bad = _IMAGES.copy()
    bad[0, 0, 0] = np.iinfo(np.int64).min
    with pytest.raises((ValueError, OverflowError), match="cap|bound|exceed|positive"):
        _call(images=bad, controls=controls)


@pytest.mark.parametrize("field", list(_CAP_FIELDS))
def test_erfc_caps_must_be_positive(field):
    controls = _controls()
    setattr(controls[3], field, 0)
    with pytest.raises(ValueError, match="positive|cap"):
        _call(controls=controls, plan=True)


@pytest.mark.parametrize("omega", [0, -1, float("nan"), float("inf")])
def test_erfc_refuses_invalid_attenuation(omega):
    with pytest.raises(ValueError):
        _call(omega=omega)


@pytest.mark.parametrize("value", [2**53 + 1, -(2**53 + 1), np.iinfo(np.int64).min])
def test_erfc_refuses_inexact_image_labels(value):
    bad = _IMAGES.copy()
    bad[0, 1, 2] = value
    with pytest.raises((ValueError, OverflowError), match="label|image|representable|integer"):
        _call(images=bad)


@pytest.mark.parametrize("change", ["lattice", "origin", "exponent", "coefficient"])
def test_erfc_refuses_nonfinite_physical_payload(change):
    center = [0.0, 0.0, 0.0]
    exponent, coefficient = [0.7], [0.8]
    lattice = _LATTICE.copy()
    if change == "lattice":
        lattice[0, 0] = float("nan")
    elif change == "origin":
        center[1] = float("nan")
    elif change == "exponent":
        exponent[0] = float("nan")
    else:
        coefficient[0] = float("nan")
    with pytest.raises((ValueError, OverflowError, RuntimeError)):
        basis = _basis([(0, center, exponent, coefficient, True)])
        _call(basis, left=(0, 1), right=(0, 1), lattice=lattice)


@pytest.mark.parametrize("field,value", [("numerical_replicas", 0),
    ("backend_margin_bytes_per_replica", 0), ("backend_margin_bytes_per_replica", 1)])
def test_erfc_requires_replica_and_known_backend_admission(field, value):
    controls = _controls()
    setattr(controls[2], field, value)
    with pytest.raises(ValueError, match="replica|backend|margin"):
        _call(controls=controls)


def test_erfc_replica_and_external_owners_are_inventoried():
    base = _call(plan=True)
    controls = _controls()
    inventory = controls[2]
    inventory.numerical_replicas = 3
    inventory.external_node_bytes = 12345
    inventory.other_live_numerical_bytes_per_replica = 2048
    inventory.other_live_control_bytes_per_replica = 1024
    plan = _call(controls=controls, plan=True)
    assert plan.per_replica_inventoried_bytes == base.per_replica_inventoried_bytes + 3072
    assert plan.required_node_inventoried_bytes == 3 * plan.per_replica_inventoried_bytes + 12345
    assert plan.engine_primitive_bytes > 0
    assert plan.engine_stack_bytes > 0
    assert plan.retained_output_bytes == 2 * 4 * 4 * 8


@pytest.mark.parametrize("images,left,right", [(np.empty((0, 3, 3), dtype=np.int64), (0, 4), (0, 4)),
    (_IMAGES, (4, 0), (0, 4)), (_IMAGES, (0, 4), (4, 0))])
def test_erfc_empty_selection_has_no_backend_calls(images, left, right):
    result = _call(images=images, left=left, right=right)
    assert result.values_copy().shape == (len(images), left[1], right[1])
    assert result.diagnostics.shell_quartet_calls == 0


def test_erfc_binding_requires_exact_contiguous_integer_image_slab():
    for images in (_IMAGES.astype(float), _IMAGES.astype(np.int32), _IMAGES[:, :, ::-1], _IMAGES.tolist()):
        with pytest.raises(TypeError):
            core._make_bipole_erfc_panel(_two_s(), _system(_LATTICE), images, *_controls())


def test_erfc_raw_quartets_recontract_the_existing_finite_sr_jk():
    basis, system = _two_s(), _system(_LATTICE)
    cells = list(core.direct_lattice_cells(system, 2.75))
    labels = [tuple(cell.index) for cell in cells]
    assert set(labels) == {(0, 0, 0), (1, 0, 0), (-1, 0, 0)}
    densities = [np.array([[0.8, 0.2], [0.2, 0.5]]) if label == (0, 0, 0)
                 else np.array([[0.1, 0.03], [0.03, -0.05]]) for label in labels]
    density = core.make_lattice_matrix_set(2, cells, densities)
    options = core.LatticeSumOptions()
    options.schwarz_threshold = 0
    options.sr_range_screening = False
    native = core.build_jk_2e_real_space_domains(
        basis, system, options, density, cells, list(range(3)), [], 0.6, True,
    )
    requests, positions = [], []
    for ig, ip, isig in itertools.product(range(3), repeat=3):
        h = tuple(np.array(labels[isig]) - np.array(labels[ip]))
        if h in labels:
            requests.append([labels[ig], labels[ip], labels[isig]])
            positions.append((ig, labels.index(h)))
    images = np.array(requests, np.int64)
    j_raw = _call(basis, images).values_copy().reshape(-1, 2, 2, 2, 2)
    k_raw = _call(basis, images[:, [1, 0, 2]]).values_copy().reshape(-1, 2, 2, 2, 2)
    j, k = np.zeros((3, 2, 2)), np.zeros((3, 2, 2))
    for index, (ig, ih) in enumerate(positions):
        j[ig] += np.einsum("abcd,cd->ab", j_raw[index], densities[ih])
        k[ig] += np.einsum("acbd,cd->ab", k_raw[index], densities[ih])
    np.testing.assert_allclose(j, np.asarray(native.J.blocks), atol=3e-13, rtol=3e-12)
    np.testing.assert_allclose(k, np.asarray(native.K.blocks), atol=3e-13, rtol=3e-12)


_BLOCH_IMAGES = np.array([
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[1, 0, 0], [1, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [-1, 0, 0], [1, 0, 0]],
    [[-1, 0, 0], [1, 0, 0], [1, 0, 0]],
], dtype=np.int64)


def _bloch_controls(*, q=1, kl=1, kr=2, left=(0, 4), right=(0, 4), omega=0.6):
    _, raw_options, raw_inventory, raw_caps = _controls(left, right, omega=omega)
    selection = core._BipoleEwaldGramSelection()
    selection.q_index, selection.left_k_index, selection.right_k_index = q, kl, kr
    selection.left_pair_begin, selection.left_pair_count = left
    selection.right_pair_begin, selection.right_pair_count = right
    options = core._BipoleErfcBlochOptions()
    options.raw = raw_options
    inventory = core._BipoleErfcBlochInventory()
    inventory.numerical_replicas = raw_inventory.numerical_replicas
    inventory.backend_margin_bytes_per_replica = raw_inventory.backend_margin_bytes_per_replica
    caps = core._BipoleErfcBlochCaps()
    caps.raw = raw_caps
    caps.maximum_kpoints = 4096
    caps.maximum_phase_evaluations = 64
    caps.maximum_borrowed_numerical_bytes = 1 << 20
    caps.maximum_owned_numerical_bytes = 128 << 20
    caps.maximum_control_storage_bytes = 16 << 20
    caps.maximum_worker_bytes = 256 << 20
    caps.maximum_node_bytes = 512 << 20
    caps.maximum_work_units = 10**11
    return selection, options, inventory, caps


def _bloch(basis=None, images=_BLOCH_IMAGES, *, mesh=(3, 1, 1), shift=(0, 0, 0),
           lattice=_LATTICE, controls=None, plan=False, **kwargs):
    function = core._plan_bipole_erfc_bloch if plan else core._make_bipole_erfc_bloch
    return function(
        _two_s() if basis is None else basis, _system(lattice), core._RegularKMesh(list(mesh), list(shift)),
        np.ascontiguousarray(images, dtype=np.int64).reshape(-1, 3, 3),
        *(_bloch_controls(**kwargs) if controls is None else controls),
    )


def _exact_phase(labels, mesh, shift, q, kl, kr):
    indices = [np.unravel_index(index, mesh) for index in (q, kl, kr)]
    turns = Fraction(0)
    for axis, n in enumerate(mesh):
        g, p, s = [int(row[axis]) for row in labels]
        aq = 2 * int(indices[0][axis])
        al = 2 * int(indices[1][axis]) + shift[axis]
        ar = 2 * int(indices[2][axis]) + shift[axis]
        turns += Fraction(al * g + (aq + ar) * p - ar * s, 2 * n)
    turns %= 1
    if turns in (0, Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)):
        return (1, 1j, -1, -1j)[int(4 * turns)]
    if turns > Fraction(1, 2):
        turns -= 1
    return complex(np.exp(2j * np.pi * float(turns)))


@pytest.mark.parametrize("shift", [(0, 0, 0), (1, 0, 0)])
@pytest.mark.parametrize("q", [0, 1, 2])
def test_multik_erfc_fold_matches_independent_gaussian_phase_oracle(shift, q):
    result = _bloch(shift=shift, q=q)
    raw = _oracle(_two_s(), _BLOCH_IMAGES)
    phases = [_exact_phase(v, (3, 1, 1), shift, q, 1, 2) for v in _BLOCH_IMAGES]
    expected = np.einsum("i,ilr->lr", phases, raw)
    np.testing.assert_allclose(result.values_copy(), expected, atol=5e-13, rtol=4e-12)
    assert np.max(np.abs(expected.imag)) > 1e-6
    assert not result.physical_hamiltonian_certified and not result.symmetry_certified
    assert result.citation_numerics == ["bipole_erfc_bloch"]


def test_multik_erfc_angular_phase_oracle():
    basis = _angular_basis()
    left, right = (3, 3), (13, 2)
    actual = _bloch(basis, left=left, right=right).values_copy()
    raw = _oracle(basis, _BLOCH_IMAGES, left=left, right=right)
    phases = [_exact_phase(v, (3, 1, 1), (0, 0, 0), 1, 1, 2) for v in _BLOCH_IMAGES]
    np.testing.assert_allclose(actual, np.einsum("i,ilr->lr", phases, raw), atol=5e-13, rtol=4e-12)


@pytest.mark.parametrize("shift", [(0, 0, 0), (1, 0, 0)])
def test_multik_erfc_time_reversal_is_a_numerical_phase_identity(shift):
    a = _bloch(shift=shift).values_copy()
    b = _bloch(shift=shift, q=2, kl=(-1-shift[0]) % 3, kr=(-2-shift[0]) % 3).values_copy()
    np.testing.assert_allclose(b, a.conj(), atol=4e-14, rtol=3e-13)


def test_multik_erfc_has_no_hidden_k_weight():
    a = _bloch().values_copy()
    b = _bloch(mesh=(6, 1, 1), q=2, kl=2, kr=4).values_copy()
    np.testing.assert_array_equal(a, b)


def test_multik_erfc_image_and_pair_tiling():
    whole = _bloch().values_copy()
    image_slabs = [_bloch(images=_BLOCH_IMAGES[i:i+1]).values_copy() for i in range(4)]
    np.testing.assert_allclose(sum(image_slabs), whole, atol=5e-15, rtol=5e-14)
    by_left = np.concatenate([_bloch(left=(i, 1)).values_copy() for i in range(4)], axis=0)
    by_right = np.concatenate([_bloch(right=(i, 1)).values_copy() for i in range(4)], axis=1)
    np.testing.assert_array_equal(by_left, whole)
    np.testing.assert_array_equal(by_right, whole)


def test_multik_erfc_large_exact_integer_phase_avoids_cartesian_dot_loss():
    labels = np.array([[[2**53, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=np.int64)
    lattice = np.eye(3) * 2.0**-53
    basis = _basis([(0, (0, 0, 0), [0.7], [0.8], True)])
    raw = _call(basis, labels, left=(0, 1), right=(0, 1), lattice=lattice).element(0, 0, 0)
    folded = _bloch(basis, labels, left=(0, 1), right=(0, 1), lattice=lattice).element(0, 0)
    assert raw > 0.01
    expected = raw * _exact_phase(labels[0], (3, 1, 1), (0, 0, 0), 1, 1, 2)
    assert folded == pytest.approx(expected, abs=2e-14)


def test_multik_erfc_gamma_and_half_turn_are_exactly_real():
    for mesh, kl in (((1, 1, 1), 0), ((2, 1, 1), 1)):
        actual = _bloch(mesh=mesh, q=0, kl=kl, kr=0).values_copy()
        np.testing.assert_array_equal(actual.imag, np.zeros((4, 4)))


def test_multik_erfc_resource_composition_does_not_double_spend_child_caps():
    plan = _bloch(plan=True)
    assert plan.retained_output_bytes == 16 * 4 * 4
    assert plan.compensation_bytes == plan.retained_output_bytes
    assert plan.peak_owned_numerical_bytes == (
        plan.retained_output_bytes + plan.compensation_bytes
        + plan.fixed_numeric_workspace_bytes + plan.raw.peak_owned_numerical_bytes
    )
    assert plan.required_node_inventoried_bytes >= plan.raw.required_node_inventoried_bytes
    raw_alone = _call(images=_BLOCH_IMAGES, plan=True)
    controls = _bloch_controls()
    child = controls[3].raw
    child.maximum_worker_bytes = raw_alone.per_replica_inventoried_bytes
    controls[3].raw = child
    with pytest.raises(ValueError, match="worker|cap"):
        _bloch(controls=controls)


_BLOCH_CAP_FIELDS = {
    "maximum_kpoints": "n_kpoints", "maximum_phase_evaluations": "phase_evaluations",
    "maximum_borrowed_numerical_bytes": "borrowed_numerical_bytes",
    "maximum_owned_numerical_bytes": "peak_owned_numerical_bytes",
    "maximum_control_storage_bytes": "control_storage_bytes",
    "maximum_worker_bytes": "per_replica_inventoried_bytes",
    "maximum_node_bytes": "required_node_inventoried_bytes",
    "maximum_work_units": "work_units_upper_bound",
}


@pytest.mark.parametrize("field", list(_BLOCH_CAP_FIELDS))
def test_multik_erfc_exact_composed_caps(field):
    plan = _bloch(plan=True)
    controls = _bloch_controls()
    amount = getattr(plan, _BLOCH_CAP_FIELDS[field])
    setattr(controls[3], field, amount)
    _bloch(controls=controls)
    setattr(controls[3], field, amount - 1)
    bad = _BLOCH_IMAGES.copy()
    bad[0, 0, 0] = np.iinfo(np.int64).min
    with pytest.raises((ValueError, OverflowError), match="cap|bound|exceed|positive"):
        _bloch(images=bad, controls=controls)


@pytest.mark.parametrize("field", ["q_index", "left_k_index", "right_k_index"])
def test_multik_erfc_rejects_invalid_addresses_before_integral_work(field):
    controls = _bloch_controls()
    setattr(controls[0], field, 3)
    with pytest.raises((ValueError, IndexError)):
        _bloch(controls=controls)


def test_multik_erfc_preserves_raw_receipts_and_immutable_payload():
    raw = _call(images=_BLOCH_IMAGES)
    result = _bloch()
    for role in ("input", "source", "payload"):
        assert getattr(result, f"raw_{role}_identity_sha256") == getattr(raw, f"{role}_identity_sha256")
    a, b = result.values_copy(), result.values_copy()
    assert not a.flags.writeable and not b.flags.writeable and not np.shares_memory(a, b)
    assert result.element(2, 3) == a[2, 3]
    assert result.diagnostics.evaluated_images == len(_BLOCH_IMAGES)
    assert result.diagnostics.folded_scalar_terms == len(_BLOCH_IMAGES) * 4 * 4
    changed = _bloch(q=2)
    assert changed.input_identity_sha256 != result.input_identity_sha256
    assert changed.payload_identity_sha256 != result.payload_identity_sha256


def test_target_mesh_addresses_do_not_materialize_full_k_integrals():
    for mesh in ((8, 8, 8), (6, 6, 6)):
        result = _bloch(mesh=mesh, images=_BLOCH_IMAGES[:1], left=(0, 1), right=(0, 1))
        assert result.memory.n_kpoints == math.prod(mesh)
        assert result.memory.output_elements == 1
        assert result.memory.retained_output_bytes == 16
        assert result.values_copy().shape == (1, 1)


def test_multik_erfc_recontracts_sr_jk_on_complete_tr_hermitian_density_basis():
    """All seven real degrees of freedom of a TR-compatible 3-k, 2-AO density.

    No converged-SCF density, projection or Hermitization is used. The J and
    crossed K supports are supplied separately, just as in the finite-domain
    builder; this does not certify their union as one correlation operator.
    """
    basis, system = _two_s(), _system(_LATTICE)
    cells = list(core.direct_lattice_cells(system, 2.75))
    labels = [tuple(cell.index) for cell in cells]
    assert set(labels) == {(0, 0, 0), (1, 0, 0), (-1, 0, 0)}
    images = np.array([
        [g, p, s] for g, p, s in itertools.product(labels, repeat=3)
        if tuple(np.subtract(s, p)) in labels
    ], dtype=np.int64)
    j_integrals, k_integrals = {}, {}
    for target, source in itertools.product(range(3), repeat=2):
        j_integrals[target, source] = _bloch(
            basis, images, q=0, kl=target, kr=source,
        ).values_copy().reshape(2, 2, 2, 2)
        k_integrals[target, source] = _bloch(
            basis, images[:, [1, 0, 2]], q=(target - source) % 3, kl=source, kr=source,
        ).values_copy().reshape(2, 2, 2, 2)
    real_basis = [np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]]),
                  np.array([[0, 1], [1, 0]])]
    complex_basis = real_basis + [np.array([[0, 1j], [-1j, 0]])]
    density_basis = []
    for value in real_basis:
        density = np.zeros((3, 2, 2), complex)
        density[0] = value
        density_basis.append(density)
    for value in complex_basis:
        density = np.zeros((3, 2, 2), complex)
        density[1], density[2] = value, value.conj()
        density_basis.append(density)
    phases = np.exp(2j * np.pi * np.outer(np.arange(3) / 3, np.asarray(labels)[:, 0]))
    options = core.LatticeSumOptions()
    options.schwarz_threshold = 0
    options.sr_range_screening = False
    for density in density_basis:
        real_density = np.einsum("kh,kab->hab", phases.conj(), density) / 3
        assert np.max(np.abs(real_density.imag)) < 3e-16
        native = core.build_jk_2e_real_space_domains(
            basis, system, options,
            core.make_lattice_matrix_set(2, cells, list(real_density.real)),
            cells, list(range(3)), [], 0.6, True,
        )
        for integrals, native_blocks, axes in (
            (j_integrals, native.J.blocks, "abcd,cd->ab"),
            (k_integrals, native.K.blocks, "acbd,cd->ab"),
        ):
            reconstructed = np.array([
                sum(np.einsum(axes, integrals[target, source], density[source])
                    for source in range(3)) / 3 for target in range(3)
            ])
            expected = np.einsum("kh,hab->kab", phases, np.asarray(native_blocks))
            np.testing.assert_allclose(reconstructed, expected, atol=5e-13, rtol=4e-12)


def test_finite_sr_support_is_not_an_eri_crossing_certificate():
    cells = {(0, 0, 0), (1, 0, 0), (-1, 0, 0)}

    def accepted(labels):
        g, p, s = map(tuple, labels)
        return g in cells and p in cells and s in cells and tuple(np.subtract(s, p)) in cells

    # chi(g,p,s) != chi(p,g,s): a J receipt is not automatically a K receipt.
    labels = np.array([[1, 0, 0], [0, 0, 0], [-1, 0, 0]], dtype=np.int64)
    assert accepted(labels) and not accepted(labels[[1, 0, 2]])
    # Likewise first-pair reversal plus home-cell reanchoring leaves the support.
    reversed_pair = np.array([-labels[0], labels[1] - labels[0], labels[2] - labels[0]])
    assert not accepted(reversed_pair)
    raw = _call(images=labels[None])
    assert np.max(np.abs(raw.values_copy())) > 1e-6
    assert not raw.physical_hamiltonian_certified and not raw.symmetry_certified
    folded = _bloch(images=labels[None])
    assert not folded.physical_hamiltonian_certified and not folded.symmetry_certified


@pytest.mark.parametrize("images,left,right", [
    (np.empty((0, 3, 3), dtype=np.int64), (0, 4), (0, 4)),
    (_BLOCH_IMAGES, (4, 0), (0, 4)), (_BLOCH_IMAGES, (0, 4), (4, 0)),
])
def test_multik_erfc_empty_panels_preserve_shape_and_receipts(images, left, right):
    result = _bloch(images=images, left=left, right=right)
    assert result.values_copy().shape == (left[1], right[1])
    assert np.count_nonzero(result.values_copy()) == 0
    assert result.diagnostics.raw.shell_quartet_calls == 0
    assert len(result.payload_identity_sha256) == 64


@pytest.mark.parametrize("angular,pure", [(1, False), (7, True)])
def test_erfc_unsupported_shell_conventions_fail_before_engine(angular, pure):
    with pytest.raises((ValueError, RuntimeError), match="angular|Cartesian|shell|limit|support"):
        basis = _basis([(angular, (0, 0, 0), [0.7], [0.8], pure)])
        _call(basis, images=_IMAGES[:1], left=(0, 1), right=(0, 1))


def _image_orbit(seed):
    """Generate closure from three independent algebraic generators."""
    def generators(row):
        g, p, s = np.asarray(row, dtype=np.int64).reshape(3, 3)
        return ((-g, p - g, s - g), (g, s, p), (s - p, -p, g - p))

    seen, todo = set(), [tuple(np.asarray(seed).reshape(-1))]
    while todo:
        row = todo.pop()
        if row in seen:
            continue
        seen.add(row)
        todo.extend(tuple(np.asarray(x).reshape(-1)) for x in generators(row))
    return np.array(sorted(seen), dtype=np.int64).reshape(-1, 3, 3)


_CLOSED_IMAGES = _image_orbit([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


def _certified_bloch(images=_CLOSED_IMAGES, *, controls=None, **kwargs):
    if controls is None:
        controls = _bloch_controls(**{
            k: kwargs.pop(k) for k in list(kwargs)
            if k in {"q", "kl", "kr", "left", "right", "omega"}
        })
        controls[1].require_image_permutation_closure = True
        controls[3].maximum_support_image_comparisons = 8 * max(1, len(images))**2
    return _bloch(images=images, controls=controls, **kwargs)


def test_closed_image_support_receipt_is_optional_and_does_not_repair_values():
    assert len(_CLOSED_IMAGES) == 8
    unchecked = _bloch(images=_CLOSED_IMAGES)
    checked = _certified_bloch()
    assert not unchecked.image_permutation_support_certified
    assert unchecked.image_permutation_support_identity_sha256 == ""
    assert checked.image_permutation_support_certified
    assert len(checked.image_permutation_support_identity_sha256) == 64
    assert not checked.physical_hamiltonian_certified and not checked.symmetry_certified
    assert checked.diagnostics.support_image_comparisons == 8 * 8**2
    np.testing.assert_array_equal(checked.values_copy(), unchecked.values_copy())
    assert checked.raw_payload_identity_sha256 == unchecked.raw_payload_identity_sha256
    assert checked.citation_numerics == unchecked.citation_numerics == ["bipole_erfc_bloch"]
    assert checked.input_identity_sha256 != unchecked.input_identity_sha256
    assert checked.source_identity_sha256 != unchecked.source_identity_sha256
    assert checked.payload_identity_sha256 != unchecked.payload_identity_sha256


@pytest.mark.parametrize("images", [_CLOSED_IMAGES[:-1],
    np.concatenate([_CLOSED_IMAGES, _CLOSED_IMAGES[:1]]),
    _BLOCH_IMAGES])
def test_image_support_refuses_missing_members_and_unequal_multiplicity(images):
    # The underlying selected contributions remain legal, without a certificate.
    _bloch(images=images)
    with pytest.raises(ValueError, match="multiset.*permutation/reanchoring"):
        _certified_bloch(images)


def test_image_support_accepts_complete_repetitions_and_stabilizers():
    # A generic orbit has eight members, an all-home quartet only one.
    images = np.concatenate([_CLOSED_IMAGES, _CLOSED_IMAGES, np.zeros((3, 3, 3), np.int64)])
    result = _certified_bloch(images)
    assert result.image_permutation_support_certified
    assert result.diagnostics.support_image_comparisons == 8 * len(images)**2
    raw = _oracle(_two_s(), images)
    phases = [_exact_phase(v, (3, 1, 1), (0, 0, 0), 1, 1, 2) for v in images]
    np.testing.assert_allclose(result.values_copy(), np.einsum("i,ilr->lr", phases, raw),
                               atol=5e-13, rtol=4e-12)
    reversed_result = _certified_bloch(images[::-1])
    assert reversed_result.image_permutation_support_certified
    assert (reversed_result.image_permutation_support_identity_sha256
            != result.image_permutation_support_identity_sha256)


@pytest.mark.parametrize("q", [0, 1, 2])
@pytest.mark.parametrize("shift", [0, 1])
def test_certified_multik_quartets_obey_reanchored_permutation_identities(q, shift):
    # These are NOT same-k or real-part comparisons. Each permutation changes
    # the physical momenta; for shifted grids m(-q-k) = -q-m(k)-shift mod N.
    kl, kr = 1, 2
    def tensor(q, kl, kr):
        return _certified_bloch(q=q % 3, kl=kl % 3, kr=kr % 3,
                                shift=(shift, 0, 0)).values_copy().reshape(2, 2, 2, 2)

    original = tensor(q, kl, kr)
    first = tensor(q, -q-kl-shift, kr).transpose(1, 0, 2, 3)
    second = tensor(q, kl, -q-kr-shift).transpose(0, 1, 3, 2)
    exchanged = tensor(-q, -kr-shift, -kl-shift).transpose(2, 3, 0, 1)
    for value in (first, second, exchanged):
        np.testing.assert_allclose(value, original, atol=5e-13, rtol=4e-12)


def test_unclosed_support_has_a_numerical_permutation_counterexample():
    images = _BLOCH_IMAGES[1:2]
    # Here -q-kl == kl modulo three; only the AO labels change in this test.
    original = _bloch(images=images, q=1, kl=1, kr=2).values_copy().reshape(2, 2, 2, 2)
    first = _bloch(images=images, q=1, kl=1, kr=2).values_copy().reshape(2, 2, 2, 2).transpose(1, 0, 2, 3)
    assert np.max(np.abs(original-first)) > 1e-5
    with pytest.raises(ValueError, match="multiset"):
        _certified_bloch(images)


def test_image_support_admission_is_count_only_and_precedes_bad_payload():
    controls = _bloch_controls()
    controls[1].require_image_permutation_closure = True
    controls[3].maximum_support_image_comparisons = 8 * 8**2
    plan = _certified_bloch(controls=controls, plan=True)
    assert plan.support_image_comparisons_upper_bound == 8 * 8**2
    assert plan.support_work_units == 4096 + 128 * 8 + 256 * 8 * 8**2
    baseline = _bloch(images=_CLOSED_IMAGES, plan=True)
    assert plan.fixed_numeric_workspace_bytes == baseline.fixed_numeric_workspace_bytes + 256
    assert plan.peak_owned_numerical_bytes == baseline.peak_owned_numerical_bytes + 256
    assert plan.work_units_upper_bound == baseline.work_units_upper_bound + plan.support_work_units
    assert plan.per_replica_inventoried_bytes == plan.raw.per_replica_inventoried_bytes
    bad = _CLOSED_IMAGES.copy()
    bad[0, 0, 0] = np.iinfo(np.int64).min
    # Planning uses only the extent; it does not inspect the label values.
    _certified_bloch(bad, controls=controls, plan=True)
    controls[3].maximum_support_image_comparisons -= 1
    with pytest.raises(ValueError, match="comparison cap"):
        _certified_bloch(bad, controls=controls)
    controls[3].maximum_support_image_comparisons += 1
    with pytest.raises(ValueError, match="representable image labels"):
        _certified_bloch(bad, controls=controls)
    controls[3].maximum_support_image_comparisons = 0
    with pytest.raises(ValueError, match="positive support"):
        _certified_bloch(controls=controls)


@pytest.mark.parametrize("cap_field,plan_field", [
    ("maximum_work_units", "work_units_upper_bound"),
    ("maximum_owned_numerical_bytes", "peak_owned_numerical_bytes"),
    ("maximum_worker_bytes", "per_replica_inventoried_bytes"),
    ("maximum_node_bytes", "required_node_inventoried_bytes"),
])
def test_image_support_is_charged_to_enclosing_caps(cap_field, plan_field):
    controls = _bloch_controls()
    controls[1].require_image_permutation_closure = True
    controls[3].maximum_support_image_comparisons = 10000
    plan = _certified_bloch(controls=controls, plan=True)
    amount = getattr(plan, plan_field)
    setattr(controls[3], cap_field, amount)
    _certified_bloch(controls=controls)
    setattr(controls[3], cap_field, amount - 1)
    with pytest.raises(ValueError, match="cap"):
        _certified_bloch(controls=controls)


def test_empty_image_support_has_vacuous_image_only_certificate():
    result = _certified_bloch(np.empty((0, 3, 3), np.int64))
    assert result.image_permutation_support_certified
    assert result.diagnostics.support_image_comparisons == 0
    assert result.diagnostics.raw.shell_quartet_calls == 0
    assert not result.physical_hamiltonian_certified and not result.symmetry_certified


def test_image_support_identity_ignores_resource_caps_and_mesh_but_not_images():
    first = _certified_bloch()
    changed_mesh = _certified_bloch(mesh=(8, 8, 8), q=7, kl=2, kr=3)
    assert first.image_permutation_support_identity_sha256 == changed_mesh.image_permutation_support_identity_sha256
    controls = _bloch_controls()
    controls[1].require_image_permutation_closure = True
    controls[3].maximum_support_image_comparisons = 10000
    controls[2].numerical_replicas = 2
    controls[2].external_node_bytes = 1234
    other_resources = _certified_bloch(controls=controls)
    assert first.payload_identity_sha256 == other_resources.payload_identity_sha256
    changed_images = _certified_bloch(np.zeros((1, 3, 3), np.int64))
    assert first.image_permutation_support_identity_sha256 != changed_images.image_permutation_support_identity_sha256


def test_large_exact_image_support_is_checked_without_integer_wrap():
    limit = 2**53
    images = _image_orbit([[limit, 0, 0], [0, 0, 0], [0, 0, 0]])
    result = _certified_bloch(images, lattice=np.eye(3) / limit)
    assert result.image_permutation_support_certified
    # Valid original labels can require an absent 2^54 reanchored member.
    invalid = np.array([[[limit, 0, 0], [-limit, 0, 0], [0, 0, 0]]], np.int64)
    with pytest.raises(ValueError, match="multiset"):
        _certified_bloch(invalid, lattice=np.eye(3) / limit)


def test_certified_angular_contribution_matches_independent_integral_oracle():
    basis = _angular_basis()
    left, right = (14, 1), (27, 1)  # Selected p/d/s-containing quartet.
    result = _certified_bloch(basis=basis, left=left, right=right, q=1, kl=0, kr=2)
    raw = _oracle(basis, _CLOSED_IMAGES, left=left, right=right)
    phases = [_exact_phase(v, (3, 1, 1), (0, 0, 0), 1, 0, 2) for v in _CLOSED_IMAGES]
    np.testing.assert_allclose(result.values_copy(), np.einsum("i,ilr->lr", phases, raw),
                               atol=5e-13, rtol=4e-12)


def test_incomplete_support_is_refused_before_integral_payload_evaluation():
    # A nonfinite shell center is already refused by BasisSet construction.
    # Instead use a malformed lattice accepted by the data carrier, and prove
    # that the raw payload error is reachable when support checking is off.
    lattice = _LATTICE.copy()
    lattice[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite original 3D lattice"):
        _bloch(images=_BLOCH_IMAGES, lattice=lattice)
    with pytest.raises(ValueError, match="multiset"):
        _certified_bloch(_BLOCH_IMAGES, lattice=lattice)
