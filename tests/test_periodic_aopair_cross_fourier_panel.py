"""Tiny two-basis Fourier integrals; explicit finite images, no SCF runs.

The independent s/p oracle uses Gaussian means and covariances, not the
production Hermite recurrence or its spherical transformation tables.
"""

from __future__ import annotations

import gc
import itertools
import math

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_aopair_fourier_panel import (
    _basis,
    _call as _single_call,
    _mixed_basis,
    _system,
)


def _bases():
    bra = _basis([
        (0, (0.13, -0.21, 0.09), [0.6, 1.3], [0.7, -0.2], True),
        (1, (-0.23, 0.17, 0.31), [0.8], [0.8], True),
    ])
    ket = _basis([
        (1, (0.37, -0.11, 0.24), [0.75, 1.2], [0.65, 0.17], True),
        (0, (-0.41, 0.18, -0.2), [0.9], [0.6], True),
        (0, (0.09, 0.28, -0.16), [0.55], [0.4], False),
    ])
    return bra, ket


def _call(bra, ket_basis, vectors, *, system=None, ket=None, begin=0,
          count=None, cutoff=2.0, candidates=50000, cap=None):
    vectors = np.ascontiguousarray(vectors, dtype=np.float64).reshape(-1, 3)
    ket = np.zeros(3) if ket is None else np.asarray(ket, dtype=np.float64)
    count = bra.nbasis * ket_basis.nbasis - begin if count is None else count
    return core._periodic_ao_pair_gaussian_cross_fourier_panel(
        bra, ket_basis, _system() if system is None else system, vectors, ket,
        begin, count, cutoff, candidates,
        max(1, count * len(vectors) * 16) if cap is None else cap,
    )


def _orbitals(basis):
    result = []
    for shell in basis.shells():
        assert shell.l in (0, 1)
        # Physical libint pure-p order m=-1,0,+1 is y,z,x.
        result.extend((shell, axis) for axis in ((-1,) if shell.l == 0 else (1, 2, 0)))
    return result


def _primitive_sp(alpha, beta, a, b, bra_axis, ket_axis, momentum):
    gamma = alpha + beta
    center = (alpha * a + beta * b) / gamma
    coefficient = (math.pi / gamma) ** 1.5
    coefficient *= math.exp(-alpha * beta / gamma * float((a - b) @ (a - b))
                            - float(momentum @ momentum) / (4 * gamma))
    coefficient *= np.exp(-1j * float(momentum @ center))
    mean = center - 1j * momentum / (2 * gamma)
    if bra_axis < 0 and ket_axis < 0:
        polynomial = 1.0
    elif bra_axis < 0:
        polynomial = mean[ket_axis] - b[ket_axis]
    elif ket_axis < 0:
        polynomial = mean[bra_axis] - a[bra_axis]
    else:
        polynomial = ((mean[bra_axis] - a[bra_axis]) * (mean[ket_axis] - b[ket_axis])
                      + (1.0 / (2 * gamma) if bra_axis == ket_axis else 0.0))
    return coefficient * polynomial


def _oracle(bra, ket_basis, vectors, lattice, ket, cutoff, image_bound):
    bra_orbitals, ket_orbitals = _orbitals(bra), _orbitals(ket_basis)
    values = np.zeros((bra.nbasis, ket_basis.nbasis, len(vectors)), dtype=complex)
    retained = 0
    for mu, (a_shell, a_axis) in enumerate(bra_orbitals):
        a = np.array(a_shell.origin)
        for nu, (b_shell, b_axis) in enumerate(ket_orbitals):
            for label in itertools.product(range(-image_bound, image_bound + 1), repeat=3):
                translation = lattice @ np.array(label)
                b = np.array(b_shell.origin) + translation
                if np.linalg.norm(a - b) > cutoff:
                    continue
                retained += 1
                phase = np.exp(1j * float(ket @ translation))
                for alpha, ca in zip(a_shell.exponents, a_shell.coefficients):
                    for beta, cb in zip(b_shell.exponents, b_shell.coefficients):
                        for v, momentum in enumerate(vectors):
                            values[mu, nu, v] += ca * cb * phase * _primitive_sp(
                                alpha, beta, a, b, a_axis, b_axis, momentum)
    return values, retained


@pytest.mark.parametrize("reverse", [False, True])
def test_rectangular_s_and_p_zero_momentum_matches_native_two_basis_overlap(reverse):
    bra, ket = _bases()
    if reverse:
        bra, ket = ket, bra
    result = _call(bra, ket, [[0, 0, 0]])
    actual = result["values"].reshape(bra.nbasis, ket.nbasis)
    assert bra.nbasis != ket.nbasis
    np.testing.assert_allclose(actual.real, core.compute_overlap_two_basis(bra, ket),
                               rtol=4e-13, atol=4e-13)
    np.testing.assert_array_equal(actual.imag, np.zeros_like(actual.imag))
    assert result["retained_pair_image_count"] == bra.nbasis * ket.nbasis


@pytest.mark.parametrize("vectors", [
    [[0, 0, 0]], [[0.21, -0.34, 0.17], [-0.71, 0.13, 0.42]],
])
def test_rectangular_sp_moments_match_independent_complex_gaussian_means(vectors):
    bra, ket = _bases()
    vectors = np.asarray(vectors)
    expected, retained = _oracle(bra, ket, vectors, np.eye(3) * 8, np.zeros(3), 2.0, 0)
    result = _call(bra, ket, vectors)
    np.testing.assert_allclose(result["values"].reshape(expected.shape), expected,
                               rtol=2e-12, atol=4e-13)
    assert result["retained_pair_image_count"] == retained


def test_skew_finite_image_cross_panel_has_positive_ket_phase_without_weights():
    bra, ket_basis = _bases()
    lattice = np.array([[2.7, 0.31, -0.16], [0.0, 3.1, 0.25], [0.0, 0.0, 2.9]])
    reciprocal = 2 * np.pi * np.linalg.inv(lattice).T
    q = reciprocal @ np.array([-1 / 3, 0.25, 0.0])
    ket = reciprocal @ np.array([0.25, 1 / 3, 0.2])
    vectors = np.array([q, q + reciprocal[:, 0], q - reciprocal[:, 2]])
    expected, retained = _oracle(bra, ket_basis, vectors, lattice, ket, 3.4, 3)
    result = _call(bra, ket_basis, vectors, system=_system(lattice), ket=ket, cutoff=3.4)
    np.testing.assert_allclose(result["values"].reshape(expected.shape), expected,
                               rtol=3e-12, atol=5e-13)
    assert result["retained_pair_image_count"] == retained > bra.nbasis * ket_basis.nbasis
    assert result["image_candidate_count"] > retained
    assert result["output_bytes"] == bra.nbasis * ket_basis.nbasis * len(vectors) * 16
    no_phase = _call(bra, ket_basis, vectors, system=_system(lattice), cutoff=3.4)
    assert np.max(np.abs(result["values"] - no_phase["values"])) > 1e-3
    assert not any("certif" in key for key in result)


@pytest.mark.parametrize("fractional", list(itertools.product((0.0, 0.5), repeat=3)))
def test_every_even_mesh_k_point_cross_overlap_adjoint_and_time_reversal(fractional):
    bra, ket_basis = _bases()
    lattice = np.array([[2.7, 0.31, -0.16], [0.0, 3.1, 0.25], [0.0, 0.0, 2.9]])
    reciprocal = 2 * np.pi * np.linalg.inv(lattice).T
    ket = reciprocal @ np.array(fractional)
    settings = dict(system=_system(lattice), cutoff=3.4)
    ab = _call(bra, ket_basis, [[0, 0, 0]], ket=ket, **settings)["values"].reshape(4, 5)
    ba = _call(ket_basis, bra, [[0, 0, 0]], ket=ket, **settings)["values"].reshape(5, 4)
    tr = _call(bra, ket_basis, [[0, 0, 0]], ket=-ket, **settings)["values"].reshape(4, 5)
    np.testing.assert_allclose(ab, ba.conj().T, rtol=2e-12, atol=4e-13)
    np.testing.assert_allclose(tr, ab.conj(), rtol=2e-13, atol=4e-13)
    # At all eight TRIMs a real local AO gauge gives a real overlap. This
    # statement does not infer anything about an independently supplied HF S.
    np.testing.assert_allclose(ab.imag, 0.0, atol=5e-13)


def test_cross_adjoint_at_nonzero_transfer_and_nonself_time_reversal():
    bra, ket_basis = _bases()
    lattice = np.array([[2.7, 0.31, -0.16], [0.0, 3.1, 0.25], [0.0, 0.0, 2.9]])
    reciprocal = 2 * np.pi * np.linalg.inv(lattice).T
    ka, kb = reciprocal @ np.array([1 / 3, 0.2, -0.25]), reciprocal @ np.array([0.0, -1 / 3, 0.4])
    q = kb - ka
    vectors = np.array([q, q + reciprocal[:, 1]])
    settings = dict(system=_system(lattice), cutoff=3.4)
    ab = _call(bra, ket_basis, vectors, ket=kb, **settings)["values"].reshape(4, 5, 2)
    ba = _call(ket_basis, bra, -vectors, ket=ka, **settings)["values"].reshape(5, 4, 2)
    tr = _call(bra, ket_basis, -vectors, ket=-kb, **settings)["values"].reshape(4, 5, 2)
    np.testing.assert_allclose(ab, ba.conj().transpose(1, 0, 2), rtol=3e-12, atol=5e-13)
    np.testing.assert_allclose(tr, ab.conj(), rtol=3e-13, atol=3e-13)
    wrong = _call(ket_basis, bra, -vectors, ket=kb, **settings)["values"].reshape(5, 4, 2)
    assert np.max(np.abs(ab - wrong.conj().transpose(1, 0, 2))) > 1e-4


@pytest.mark.parametrize("same_owner", [True, False])
def test_same_basis_delegation_is_bitwise_identical_in_values_and_inventory(same_owner):
    bra = _mixed_basis()
    ket = bra if same_owner else _mixed_basis()
    vectors = [[0.21, -0.34, 0.17], [-0.71, 0.13, 0.42]]
    kwargs = dict(ket=[0.2, -0.1, 0.3], begin=5, count=21, cutoff=3.3,
                  system=_system(np.eye(3) * 2.7))
    rectangular = _call(bra, ket, vectors, **kwargs)
    square = _single_call(bra, vectors, **kwargs)
    np.testing.assert_array_equal(rectangular["values"].view(np.uint64), square["values"].view(np.uint64))
    for key in ("pair_begin", "output_bytes", "image_candidate_count", "retained_pair_image_count",
                "fixed_numeric_workspace_bytes"):
        assert rectangular[key] == square[key]


def test_partial_rectangular_flattened_tiles_and_vector_splits_are_bit_identical():
    bra, ket = _bases()
    vectors = np.array([[0.2, -0.3, 0.5], [-0.4, 0.1, 0.8], [0.7, -0.2, 0.1]])
    whole = _call(bra, ket, vectors)["values"]
    selected = _call(bra, ket, vectors, begin=3, count=13)["values"]
    split_rows = np.vstack([_call(bra, ket, vectors, begin=begin, count=count)["values"]
                            for begin, count in ((3, 4), (7, 5), (12, 4))])
    split_vectors = np.hstack([_call(bra, ket, vectors[:1], begin=3, count=13)["values"],
                               _call(bra, ket, vectors[1:], begin=3, count=13)["values"]])
    for compared in (whole[3:16], split_rows, split_vectors):
        np.testing.assert_array_equal(selected.view(np.uint64), compared.view(np.uint64))
    assert whole.shape == (4 * 5, 3)
    # Flattening uses the KET width, not the bra width or a merged basis.
    one = _call(bra, ket, vectors, begin=2 * ket.nbasis + 3, count=1)["values"]
    np.testing.assert_array_equal(one[0], whole.reshape(4, 5, 3)[2, 3])


def test_exact_fixed_scratch_payload_and_moved_output_lifetime():
    bra, ket = _bases()
    result = _call(bra, ket, [[0.1, 0.2, 0.3]], begin=3, count=4)
    assert result["fixed_numeric_workspace_bytes"] == 15912 + 32 * 688
    assert result["fixed_numeric_workspace_bytes"] == core._PERIODIC_AOPAIR_FOURIER_FIXED_NUMERIC_WORKSPACE_BYTES
    assert result["output_bytes"] == 4 * 16
    values = result["values"]
    expected = values.copy()
    del result, bra, ket
    gc.collect()
    np.testing.assert_array_equal(values, expected)


@pytest.mark.parametrize("vector_count", [31, 32, 33, 63])
def test_reciprocal_tiles_preserve_repeated_vectors_and_arbitrary_partition_bits(vector_count):
    bra, ket = _bases()
    vectors = np.random.default_rng(758).normal(size=(vector_count, 3))
    repeated = [index for index in [0, 30, 31, 32, 62] if index < vector_count]
    vectors[repeated] = [0.2, -0.3, 0.5]
    options = dict(system=_system(np.eye(3) * 2.7), ket=[0.23, -0.17, 0.31],
                   begin=3, count=13, cutoff=3.3, candidates=7000)
    whole = _call(bra, ket, vectors, **options)
    chunks = [_call(bra, ket, piece, **options)
              for piece in np.split(vectors, [cut for cut in [1, 17, 33, 61]
                                             if cut < vector_count])]
    partitioned = np.hstack([chunk["values"] for chunk in chunks])
    np.testing.assert_array_equal(whole["values"].view(np.uint64),
                                  partitioned.view(np.uint64))
    for index in repeated:
        np.testing.assert_array_equal(whole["values"][:, index], whole["values"][:, 0])
    assert all(chunk["fixed_numeric_workspace_bytes"] == whole["fixed_numeric_workspace_bytes"]
               for chunk in chunks)


def test_exact_candidate_and_output_caps_and_early_admission_errors():
    bra, ket = _bases()
    vectors = [[0.1, 0.2, 0.3], [-0.2, 0.3, 0.1]]
    result = _call(bra, ket, vectors, begin=2, count=3)
    exact = _call(bra, ket, vectors, begin=2, count=3,
                  candidates=result["image_candidate_count"], cap=result["output_bytes"])
    np.testing.assert_array_equal(result["values"], exact["values"])
    with pytest.raises(ValueError, match="source cap"):
        _call(bra, ket, vectors, begin=2, count=3, candidates=result["image_candidate_count"] - 1)
    with pytest.raises(ValueError, match="byte"):
        _call(bra, ket, vectors, begin=2, count=3, cap=result["output_bytes"] - 1)
    # The native byte gate precedes any reciprocal lane validation and all
    # output allocation, even if supplied numerical content is invalid.
    with pytest.raises(ValueError, match="byte"):
        _call(bra, ket, [[np.nan, 0.0, 0.0]], begin=0, count=1, cap=0)
    for begin, count in ((20, 1), (21, 0), (19, 2)):
        with pytest.raises(ValueError, match="interval"):
            _call(bra, ket, [[0, 0, 0]], begin=begin, count=count)


@pytest.mark.parametrize("axis", ["bra", "ket"])
@pytest.mark.parametrize("angular,pure", [(1, False), (7, True)])
def test_both_basis_conventions_are_checked_for_empty_panels(axis, angular, pure):
    good = _basis([(0, (0, 0, 0), [0.8], [0.7], True)])
    bad = _basis([(angular, (0.1, 0, 0), [0.8], [0.7], pure)])
    pair = (bad, good) if axis == "bra" else (good, bad)
    with pytest.raises(ValueError, match="spherical"):
        _call(*pair, np.empty((0, 3)), count=0)


def test_strict_array_boundary_alignment_and_readonly_views():
    bra, ket = _bases()
    vectors = np.array([[0.1, 0.2, 0.3], [0.4, -0.1, 0.2]])
    momentum = np.zeros(3)

    def direct(v, k):
        return core._periodic_ao_pair_gaussian_cross_fourier_panel(
            bra, ket, _system(), v, k, 0, 1, 2.0, 50000, 32)

    for bad in (vectors.astype(np.float32), np.asfortranarray(vectors), vectors[:, :2]):
        with pytest.raises(ValueError, match="C-contiguous float64"):
            direct(bad, momentum)
    with pytest.raises(ValueError, match="C-contiguous float64"):
        direct(vectors, momentum.astype(np.float32))
    misaligned = np.ndarray(vectors.shape, dtype=float, buffer=bytearray(vectors.nbytes + 1), offset=1)
    misaligned[:] = vectors
    with pytest.raises(ValueError, match="aligned"):
        direct(misaligned, momentum)
    vectors.flags.writeable = momentum.flags.writeable = False
    assert direct(vectors, momentum)["values"].shape == (1, 2)


def test_empty_intervals_vectors_and_no_retained_images_are_explicit_zero_payloads():
    bra, ket = _bases()
    pairs = _call(bra, ket, [[0, 0, 0]], begin=20, count=0)
    vectors = _call(bra, ket, np.empty((0, 3)), begin=3, count=2)
    assert pairs["values"].shape == (0, 1) and pairs["output_bytes"] == 0
    assert vectors["values"].shape == (2, 0) and vectors["output_bytes"] == 0
    assert pairs["image_candidate_count"] == 0
    assert vectors["image_candidate_count"] > 0  # empty-vector view still audits images
    zero = _call(bra, ket, [[0.1, -0.2, 0.3]], cutoff=0.0)
    assert zero["retained_pair_image_count"] == 0
    np.testing.assert_array_equal(zero["values"].view(np.uint64), np.zeros((20, 2), dtype=np.uint64))


@pytest.mark.parametrize("changes,match", [
    ({"cutoff": -1.0}, "cutoff"), ({"cutoff": np.nan}, "finite"),
    ({"ket": [0.0, np.inf, 0.0]}, "finite"), ({"candidates": 0}, "positive"),
    ({"candidates": 65537}, "tiny"), ({"count": 65}, "tiny"),
])
def test_invalid_controls_and_tiny_diagnostic_limits(changes, match):
    bra, ket = _bases()
    with pytest.raises((ValueError, OverflowError), match=match):
        _call(bra, ket, [[0.1, 0.2, 0.3]], **changes)


def test_nonfinite_vectors_and_non3d_or_singular_lattice_fail_closed():
    bra, ket = _bases()
    with pytest.raises(ValueError, match="finite"):
        _call(bra, ket, [[np.nan, 0.0, 0.0]])
    system = _system()
    system.dim = 2
    with pytest.raises(ValueError, match="3D"):
        _call(bra, ket, [[0, 0, 0]], system=system)
    with pytest.raises(ValueError, match="nonsingular"):
        _call(bra, ket, [[0, 0, 0]], system=_system(np.zeros((3, 3))))
