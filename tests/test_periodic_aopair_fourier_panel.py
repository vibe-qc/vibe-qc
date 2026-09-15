"""Bounded C++ AO-pair Fourier panels against independent Gaussian moments.

These are small integral-only tests. No SCF or target-system calculation is
run. The numerical image policy is tested separately from source certification.
"""

from __future__ import annotations

import gc
import itertools
import math

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core


def _basis(specifications):
    centers = list(dict.fromkeys(tuple(item[1]) for item in specifications))
    molecule = core.Molecule([core.Atom(2, list(center)) for center in centers])
    shells = [
        core.ShellInfo(
            centers.index(tuple(center)), angular, pure, list(exponents),
            list(coefficients), list(center),
        )
        for angular, center, exponents, coefficients, pure in specifications
    ]
    return core.BasisSet(molecule, shells, "diagnostic-pair-ft", True)


def _system(lattice=None):
    system = core.PeriodicSystem()
    system.lattice = np.eye(3) * 8.0 if lattice is None else np.asarray(lattice)
    return system


def _call(basis, vectors, *, system=None, ket=None, begin=0, count=None,
          cutoff=2.0, candidates=50000, cap=None):
    vectors = np.ascontiguousarray(vectors, dtype=np.float64).reshape(-1, 3)
    ket = np.zeros(3) if ket is None else np.asarray(ket, dtype=np.float64)
    count = basis.nbasis**2 - begin if count is None else count
    return core._periodic_ao_pair_gaussian_fourier_panel(
        basis, _system() if system is None else system, vectors, ket,
        begin, count, cutoff, candidates,
        max(1, count * len(vectors) * 16) if cap is None else cap,
    )


def _full(basis, vectors, **kwargs):
    total = basis.nbasis**2
    return np.vstack([
        _call(basis, vectors, begin=begin, count=min(16, total - begin),
              **kwargs)["values"]
        for begin in range(0, total, 16)
    ]).reshape(basis.nbasis, basis.nbasis, -1)


def _polynomials(angular):
    # Normalized libint solid harmonics in m=-l..+l order, independent
    # explicit low-L tables. Radial primitive coefficients are separate.
    if angular == 0:
        return [{(0, 0, 0): 1.0}]
    if angular == 1:
        return [{(0, 1, 0): 1.0}, {(0, 0, 1): 1.0}, {(1, 0, 0): 1.0}]
    if angular == 2:
        root3 = math.sqrt(3.0)
        return [
            {(1, 1, 0): root3}, {(0, 1, 1): root3},
            {(0, 0, 2): 1.0, (2, 0, 0): -0.5, (0, 2, 0): -0.5},
            {(1, 0, 1): root3},
            {(2, 0, 0): root3 / 2, (0, 2, 0): -root3 / 2},
        ]
    raise ValueError("small oracle supports s, p and d")


def _moment(power, gamma, momentum):
    # Gaussian generating-function moments: derivatives of
    # sqrt(pi/gamma)*exp(-p^2/(4 gamma)), not the production MD recursion.
    moments = [math.sqrt(math.pi / gamma) * math.exp(-momentum**2 / (4 * gamma))]
    for order in range(1, power + 1):
        value = (-1j * momentum / (2 * gamma)) * moments[-1]
        if order > 1:
            value += (order - 1) / (2 * gamma) * moments[-2]
        moments.append(value)
    return moments[power]


def _primitive_pair(alpha, beta, a, b, pa, pb, momentum):
    gamma = alpha + beta
    center = (alpha * a + beta * b) / gamma
    scale = math.exp(-alpha * beta / gamma * np.dot(a - b, a - b))
    scale *= np.exp(-1j * np.dot(momentum, center))
    integral = 0j
    for apower, acoef in pa.items():
        for bpower, bcoef in pb.items():
            term = complex(acoef * bcoef)
            for axis in range(3):
                factor = 0j
                for ia in range(apower[axis] + 1):
                    for ib in range(bpower[axis] + 1):
                        factor += (
                            math.comb(apower[axis], ia) * math.comb(bpower[axis], ib)
                            * (center[axis] - a[axis]) ** (apower[axis] - ia)
                            * (center[axis] - b[axis]) ** (bpower[axis] - ib)
                            * _moment(ia + ib, gamma, momentum[axis])
                        )
                term *= factor
            integral += term
    return scale * integral


def _oracle(basis, vectors, lattice, ket, cutoff, image_bound=2):
    orbitals = [(shell, polynomial) for shell in basis.shells()
                for polynomial in _polynomials(shell.l)]
    output = np.zeros((basis.nbasis, basis.nbasis, len(vectors)), complex)
    retained = 0
    for mu, (bra, pa) in enumerate(orbitals):
        a = np.asarray(bra.origin)
        for nu, (ket_shell, pb) in enumerate(orbitals):
            for label in itertools.product(range(-image_bound, image_bound + 1), repeat=3):
                translation = lattice @ np.asarray(label)
                b = np.asarray(ket_shell.origin) + translation
                if np.linalg.norm(a - b) > cutoff:
                    continue
                retained += 1
                phase = np.exp(1j * np.dot(ket, translation))
                for alpha, ca in zip(bra.exponents, bra.coefficients):
                    for beta, cb in zip(ket_shell.exponents, ket_shell.coefficients):
                        for v, momentum in enumerate(vectors):
                            output[mu, nu, v] += phase * ca * cb * _primitive_pair(
                                alpha, beta, a, b, pa, pb, momentum,
                            )
    return output, retained


def _mixed_basis():
    return _basis([
        (0, (0.13, -0.21, 0.09), [0.6, 1.3], [0.7, -0.2], True),
        (1, (-0.23, 0.17, 0.31), [0.8], [0.8], True),
        (2, (0.13, -0.21, 0.09), [0.9], [0.6], True),
    ])


def test_mixed_s_p_d_zero_momentum_is_native_libint_overlap():
    basis = _mixed_basis()
    actual = _full(basis, [[0, 0, 0]])[:, :, 0]
    np.testing.assert_allclose(actual.real, core.compute_overlap(basis),
                               rtol=3e-13, atol=3e-13)
    np.testing.assert_array_equal(actual.imag, np.zeros_like(actual.imag))


def test_mixed_s_p_d_displaced_moments_are_independent_oracle():
    basis = _mixed_basis()
    vectors = np.array([[0.21, -0.34, 0.17], [-0.71, 0.13, 0.42]])
    actual = _full(basis, vectors)
    expected, _ = _oracle(basis, vectors, np.eye(3) * 8, np.zeros(3), 2.0, 0)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=3e-13)


def test_nonzero_q_and_ket_phase_on_skew_image_lattice():
    basis = _basis([
        (0, (0.1, -0.2, 0.15), [0.55, 1.1], [0.7, -0.12], True),
        (0, (0.6, 0.3, -0.25), [0.7], [0.8], True),
    ])
    lattice = np.array([[2.7, 0.2, 0.1], [0.0, 3.1, 0.25], [0.0, 0.0, 2.9]])
    reciprocal = 2 * np.pi * np.linalg.inv(lattice).T
    q = reciprocal @ np.array([-0.5, 1 / 3, 0.0])
    ket = reciprocal @ np.array([1 / 4, 1 / 3, 0.0])
    vectors = np.array([q, q + reciprocal[:, 0], q - reciprocal[:, 2]])
    result = _call(basis, vectors, system=_system(lattice), ket=ket, cutoff=3.4)
    expected, retained = _oracle(basis, vectors, lattice, ket, 3.4)
    np.testing.assert_allclose(result["values"].reshape(expected.shape), expected,
                               rtol=2e-12, atol=3e-13)
    assert result["retained_pair_image_count"] == retained
    assert result["image_candidate_count"] > retained
    assert result["output_bytes"] == 4 * 3 * 16
    other = _call(basis, vectors, system=_system(lattice), ket=np.zeros(3), cutoff=3.4)
    assert np.max(np.abs(other["values"] - result["values"])) > 1e-3


def test_pair_and_reciprocal_tile_splits_are_bit_identical():
    basis = _mixed_basis()
    vectors = np.array([[0.2, -0.3, 0.5], [-0.4, 0.1, 0.8]])
    whole = _call(basis, vectors, begin=5, count=21)["values"]
    split_pairs = np.vstack([
        _call(basis, vectors, begin=5, count=7)["values"],
        _call(basis, vectors, begin=12, count=14)["values"],
    ])
    split_vectors = np.hstack([
        _call(basis, vectors[:1], begin=5, count=21)["values"],
        _call(basis, vectors[1:], begin=5, count=21)["values"],
    ])
    np.testing.assert_array_equal(whole.view(np.uint64), split_pairs.view(np.uint64))
    np.testing.assert_array_equal(whole.view(np.uint64), split_vectors.view(np.uint64))


def test_time_reversal_and_lattice_relabeling_covariance():
    lattice = np.eye(3) * 2.7
    a, b = np.array([0.1, 0.2, -0.3]), np.array([0.7, -0.1, 0.3])
    basis = _basis([(0, a, [0.65], [0.8], True), (0, b, [0.8], [0.9], True)])
    shifted = _basis([
        (0, a, [0.65], [0.8], True),
        (0, b + lattice[:, 0], [0.8], [0.9], True),
    ])
    vectors = np.array([[0.4, -0.3, 0.2], [-0.1, 0.2, 0.5]])
    ket = np.array([0.2, -0.1, 0.3])
    options = dict(system=_system(lattice), begin=1, count=1, cutoff=3.3)
    actual = _call(basis, vectors, ket=ket, **options)["values"]
    reversed_value = _call(basis, -vectors, ket=-ket, **options)["values"]
    translated = _call(shifted, vectors, ket=ket, **options)["values"]
    np.testing.assert_allclose(reversed_value, actual.conj(), rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(translated, actual * np.exp(-1j * ket @ lattice[:, 0]),
                               rtol=2e-13, atol=2e-13)


def test_exact_workspace_and_array_lifetime():
    basis = _mixed_basis()
    result = _call(basis, [[0.1, 0.2, 0.3]], count=3)
    primitive_bytes = 3 * 7 * 7 * 13 * 8 + 3 * 13 * 16
    vector_tile_bytes = 32 * (4 * 8 + 3 * 13 * 16 + 4 * 8)
    assert result["fixed_numeric_workspace_bytes"] == primitive_bytes + vector_tile_bytes
    assert result["fixed_numeric_workspace_bytes"] == (
        core._PERIODIC_AOPAIR_FOURIER_FIXED_NUMERIC_WORKSPACE_BYTES
    )
    values = result["values"]
    expected = values.copy()
    del result, basis
    gc.collect()
    np.testing.assert_array_equal(values, expected)


@pytest.mark.parametrize("vector_count", [1, 31, 32, 33, 63])
def test_tiled_distance_source_is_bitwise_equal_to_same_explicit_images(vector_count):
    origin = (0.13, -0.21, 0.09)
    basis = _basis([(angular, origin, [0.6, 1.3], [0.7, -0.2], True)
                    for angular in [0, 1, 2]])
    system = _system(np.eye(3) * 2.7)
    vectors = np.random.default_rng(142).normal(size=(vector_count, 3))
    vectors[0] = 0.0
    if vector_count > 32:
        vectors[32] = vectors[0]
    ket = np.array([0.23, -0.17, 0.31])
    # Coincident centers make the same seven images apply to every AO pair.
    # Lexicographic integer labels preserve the distance source's image order.
    cells = np.array([label for label in itertools.product([-1, 0, 1], repeat=3)
                      if np.linalg.norm(np.asarray(label) * 2.7) < 3.0], dtype=np.int64)
    # Thirteen rows keep the 63-vector case within the diagnostic work cap.
    pair_count = 13
    distance = _call(basis, vectors, system=system, ket=ket, begin=5, count=pair_count,
                     cutoff=3.0, candidates=pair_count * 7**3)
    caps = core._AOPairFourierCellPanelCaps()
    caps.maximum_cells = 7
    caps.maximum_pair_cell_visits = 7 * pair_count
    caps.maximum_work_units = 2_000_000_000
    caps.maximum_output_bytes = 16 * pair_count * vector_count
    explicit = core._ao_pair_gaussian_fourier_cell_panel(
        basis, system, vectors, ket, 5, pair_count, cells, caps,
    )
    np.testing.assert_array_equal(distance["values"].view(np.uint64),
                                  explicit["values"].view(np.uint64))
    assert explicit["plan"].fixed_numeric_workspace_bytes < distance["fixed_numeric_workspace_bytes"]


@pytest.mark.parametrize("options,match", [
    ({"begin": 4, "count": 1}, "interval"),
    ({"cap": 15}, "byte"),
    ({"candidates": 1}, "source cap"),
    ({"cutoff": -1.0}, "cutoff"),
    ({"cutoff": float("nan")}, "finite"),
    ({"ket": [0.0, float("inf"), 0.0]}, "finite"),
])
def test_invalid_requests_fail_closed(options, match):
    basis = _basis([(0, (0, 0, 0), [0.8], [0.7], True)])
    with pytest.raises((ValueError, OverflowError), match=match):
        _call(basis, [[0.1, 0.2, 0.3]], **options)


@pytest.mark.parametrize("angular,pure", [(1, False), (7, True)])
def test_unsupported_shell_is_rejected_even_for_empty_vector_panel(angular, pure):
    basis = _basis([(angular, (0, 0, 0), [0.8], [0.7], pure)])
    with pytest.raises(ValueError, match="spherical"):
        _call(basis, np.empty((0, 3)), count=1)


def test_nonfinite_vector_and_non3d_lattice_are_rejected():
    basis = _basis([(0, (0, 0, 0), [0.8], [0.7], True)])
    with pytest.raises(ValueError, match="finite"):
        _call(basis, [[float("nan"), 0.0, 0.0]])
    system = _system()
    system.dim = 2
    with pytest.raises(ValueError, match="3D"):
        _call(basis, [[0.0, 0.0, 0.0]], system=system)


def test_empty_pair_and_vector_panels_allocate_no_payload():
    basis = _basis([(0, (0, 0, 0), [0.8], [0.7], True)])
    empty_pairs = _call(basis, [[0.0, 0.0, 0.0]], count=0)
    empty_vectors = _call(basis, np.empty((0, 3)))
    assert empty_pairs["values"].shape == (0, 1)
    assert empty_vectors["values"].shape == (1, 0)
    assert empty_pairs["output_bytes"] == empty_vectors["output_bytes"] == 0


def test_no_retained_images_produces_exact_zero_panel():
    basis = _basis([
        (0, (0, 0, 0), [0.8], [0.7], True),
        (0, (0.5, 0.1, 0.2), [0.6], [0.9], True),
    ])
    result = _call(basis, [[0.1, -0.2, 0.3]], begin=1, count=1, cutoff=0.0)
    assert result["retained_pair_image_count"] == 0
    np.testing.assert_array_equal(result["values"].view(np.uint64), [[0, 0]])
