"""Native bounded one-center auxiliary-Gaussian Fourier panels.

The existing Python RSGDF transform is the readable numerical reference.  The
native kernel is a lower-level production primitive: it consumes one already
bounded reciprocal panel, allocates exactly its capped ``A * n_vector`` output,
and performs no target calculation or SCF work.
"""

from __future__ import annotations

import hashlib
import math
import struct
from pathlib import Path

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from vibeqc.aux_basis import rsgdf_aux_fourier_transform


_DIGEST_DOMAIN = "vibeqc.periodic.auxiliary-basis-content"
_DIGEST_KAT = "872e456958fd60fb576db153c0f9c6ed339abe4a98eda4621ce8b5183af97bad"
_ROOT = Path(__file__).resolve().parents[1]


def _shell(
    atom_index: int,
    angular_momentum: int,
    pure: bool,
    exponents: list[float],
    coefficients: list[float],
    origin: tuple[float, float, float],
) -> object:
    return core.ShellInfo(
        atom_index,
        angular_momentum,
        pure,
        exponents,
        coefficients,
        list(origin),
    )


def _one_center_basis(
    angular_momenta: tuple[int, ...],
    *,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    name: str = "diagnostic-aux",
    pure: bool = True,
) -> object:
    molecule = core.Molecule([core.Atom(2, list(center))], 0, 1)
    shells = [
        _shell(
            0,
            angular_momentum,
            pure,
            [0.55 + 0.07 * angular_momentum, 1.35 + 0.11 * angular_momentum],
            [0.625, -0.175],
            center,
        )
        for angular_momentum in angular_momenta
    ]
    return core.BasisSet(
        molecule,
        shells,
        name,
        coefficients_pre_normalized=True,
    )


def _native_panel(basis: object, vectors: np.ndarray, cap: int | None = None):
    vectors = np.ascontiguousarray(vectors, dtype=np.float64)
    required = int(basis.nbasis) * vectors.shape[0] * 16
    return np.asarray(
        core._periodic_auxiliary_gaussian_fourier_panel(
            basis,
            vectors,
            required if cap is None else cap,
        )
    )


def _basis_identity_oracle(basis: object) -> str:
    """Independent implementation of the documented big-endian v1 wire."""
    digest = hashlib.sha256()

    def u8(value: int) -> None:
        digest.update(struct.pack(">B", value))

    def u32(value: int) -> None:
        digest.update(struct.pack(">I", value))

    def u64(value: int) -> None:
        digest.update(struct.pack(">Q", value))

    def binary64(value: float) -> None:
        assert math.isfinite(value)
        digest.update(struct.pack(">d", 0.0 if value == 0.0 else value))

    domain = _DIGEST_DOMAIN.encode("ascii")
    u64(len(domain))
    digest.update(domain)
    u32(1)
    u64(int(basis.nbasis))
    u64(int(basis.nshells))
    records = basis.shells()
    u64(len(records))
    for shell in records:
        # Every custom diagnostic shell has one contraction.  The native wire
        # records this boundary explicitly so generalized-shell partitions
        # cannot alias even when their flattened records are otherwise equal.
        u64(1)
        u64(int(shell.atom_index))
        u32(int(shell.l))
        u8(1 if shell.pure else 0)
        u64(len(shell.exponents))
        for exponent in shell.exponents:
            binary64(float(exponent))
        for coefficient in shell.coefficients:
            binary64(float(coefficient))
        for coordinate in shell.origin:
            binary64(float(coordinate))
    return digest.hexdigest()


@pytest.mark.parametrize("angular_momentum", [0, 1, 2])
def test_s_p_d_contracted_shifted_center_matches_python_reference(
    angular_momentum: int,
) -> None:
    basis = _one_center_basis(
        (angular_momentum,), center=(0.31, -0.27, 0.19)
    )
    vectors = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, -0.4, 0.7],
            [-1.1, 0.3, 0.5],
            [0.8, 1.2, -0.6],
        ],
        dtype=np.float64,
    )
    actual = _native_panel(basis, vectors)
    expected = rsgdf_aux_fourier_transform(basis, vectors)
    assert actual.shape == (basis.nbasis, vectors.shape[0])
    assert actual.dtype == np.complex128
    assert actual.flags.c_contiguous
    np.testing.assert_allclose(actual, expected, rtol=2.0e-12, atol=2.0e-13)


@pytest.mark.parametrize("angular_momentum", [3, 4, 5, 6])
def test_high_angular_momentum_generated_polynomials_match_python_reference(
    angular_momentum: int,
) -> None:
    basis = _one_center_basis((angular_momentum,))
    vectors = np.array(
        [
            [0.13, -0.29, 0.47],
            [-0.91, 0.38, 0.22],
            [0.64, 0.71, -0.53],
        ],
        dtype=np.float64,
    )
    actual = _native_panel(basis, vectors)
    expected = rsgdf_aux_fourier_transform(basis, vectors)
    np.testing.assert_allclose(actual, expected, rtol=5.0e-12, atol=5.0e-13)


def test_gamma_values_and_translation_covariance() -> None:
    angular_momenta = (0, 1, 2)
    center = (0.37, -0.21, 0.43)
    origin_basis = _one_center_basis(angular_momenta)
    shifted_basis = _one_center_basis(angular_momenta, center=center)
    vectors = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.31, -0.52, 0.27],
            [-0.81, 0.14, 0.63],
        ],
        dtype=np.float64,
    )
    at_origin = _native_panel(origin_basis, vectors)
    shifted = _native_panel(shifted_basis, vectors)
    phase = np.exp(-1j * (vectors @ np.asarray(center)))
    np.testing.assert_allclose(
        shifted, at_origin * phase[None, :], rtol=2.0e-13, atol=2.0e-13
    )

    exponents = np.asarray(origin_basis.shells()[0].exponents)
    coefficients = np.asarray(origin_basis.shells()[0].coefficients)
    expected_s_monopole = np.sum(coefficients * (np.pi / exponents) ** 1.5)
    np.testing.assert_allclose(at_origin[0, 0], expected_s_monopole, atol=1.0e-14)
    assert np.count_nonzero(at_origin[1:, 0]) == 0


def test_zero_copy_soa_entry_matches_interleaved_entry() -> None:
    basis = _one_center_basis((0, 1, 2), center=(0.2, 0.3, -0.4))
    vectors = np.ascontiguousarray(
        [[0.0, 0.0, 0.0], [0.2, -0.7, 0.5], [-0.4, 0.9, 0.1]],
        dtype=np.float64,
    )
    required = basis.nbasis * vectors.shape[0] * 16
    interleaved = _native_panel(basis, vectors, required)
    soa = np.asarray(
        core._periodic_auxiliary_gaussian_fourier_panel_soa(
            basis,
            np.ascontiguousarray(vectors[:, 0]),
            np.ascontiguousarray(vectors[:, 1]),
            np.ascontiguousarray(vectors[:, 2]),
            required,
        )
    )
    np.testing.assert_array_equal(soa, interleaved)


def test_output_cap_is_checked_before_vector_or_shell_work() -> None:
    basis = _one_center_basis((0, 1))
    vectors = np.array([[np.nan, 0.0, 0.0]], dtype=np.float64)
    required = basis.nbasis * vectors.shape[0] * 16

    with pytest.raises(ValueError, match="cap"):
        _native_panel(basis, vectors, required - 1)
    with pytest.raises(ValueError, match="vectors must be finite"):
        _native_panel(basis, vectors, required)
    with pytest.raises(ValueError, match="cap must be positive"):
        _native_panel(basis, np.zeros((0, 3)), 0)


def test_nonfinite_derived_vector_norm_fails_before_transcendentals() -> None:
    basis = _one_center_basis((0,))
    vectors = np.array([[1.0e308, 1.0e308, 0.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="squared vector norm"):
        _native_panel(basis, vectors)


def test_malformed_diagnostic_arrays_fail_closed() -> None:
    basis = _one_center_basis((0,))
    with pytest.raises(ValueError, match="shape"):
        core._periodic_auxiliary_gaussian_fourier_panel(
            basis, np.zeros(3, dtype=np.float64), 16
        )
    with pytest.raises(ValueError, match="equal lengths"):
        core._periodic_auxiliary_gaussian_fourier_panel_soa(
            basis,
            np.zeros(2, dtype=np.float64),
            np.zeros(1, dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            32,
        )
    too_many = np.zeros(
        core._PERIODIC_AUXILIARY_FOURIER_DIAGNOSTIC_MAX_VECTORS + 1,
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match="internal auxiliary Fourier diagnostic"):
        core._periodic_auxiliary_gaussian_fourier_panel_soa(
            basis, too_many, too_many, too_many, too_many.size * 16
        )


def test_diagnostic_bindings_never_forcecast_or_stage_inputs() -> None:
    basis = _one_center_basis((0,))
    vectors = np.zeros((3, 3), dtype=np.float64)
    with pytest.raises(TypeError):
        core._periodic_auxiliary_gaussian_fourier_panel(
            basis, vectors.astype(np.float32), 3 * 16
        )
    with pytest.raises(TypeError):
        core._periodic_auxiliary_gaussian_fourier_panel(
            basis, vectors[:, ::-1], 3 * 16
        )
    with pytest.raises(TypeError):
        core._periodic_auxiliary_gaussian_fourier_panel_soa(
            basis,
            vectors[:, 0],
            np.ascontiguousarray(vectors[:, 1]),
            np.ascontiguousarray(vectors[:, 2]),
            3 * 16,
        )


def test_unsupported_shell_conventions_fail_closed() -> None:
    vectors = np.zeros((1, 3), dtype=np.float64)
    cartesian_p = _one_center_basis((1,), pure=False)
    with pytest.raises(ValueError, match="Cartesian"):
        _native_panel(cartesian_p, vectors)

    beyond_table = _one_center_basis((7,))
    with pytest.raises(ValueError, match="table coverage"):
        _native_panel(beyond_table, vectors)


def _identity_fixture(
    *,
    name: str = "identity-a",
    signed_zero: float = -0.0,
    first_exponent: float = 1.25,
    first_coefficient: float = 0.75,
    first_pure: bool = False,
    second_origin: tuple[float, float, float] = (1.25, -0.5, 0.75),
    reverse_shells: bool = False,
) -> object:
    first = (0.0, signed_zero, 0.0)
    second = second_origin
    molecule = core.Molecule(
        [core.Atom(1, list(first)), core.Atom(1, list(second))], 0, 1
    )
    shells = [
        _shell(
            0,
            0,
            first_pure,
            [first_exponent, 0.5],
            [first_coefficient, -0.125],
            first,
        ),
        _shell(1, 1, True, [0.8], [-0.25], second),
    ]
    if reverse_shells:
        shells.reverse()
    return core.BasisSet(
        molecule, shells, name, coefficients_pre_normalized=True
    )


def test_basis_content_identity_matches_independent_wire_and_fixed_kat() -> None:
    basis = _identity_fixture()
    actual = core._auxiliary_basis_content_identity_sha256(basis)
    assert core._AUXILIARY_BASIS_CONTENT_DIGEST_VERSION == 1
    assert actual == _basis_identity_oracle(basis)
    assert actual == _DIGEST_KAT


def test_basis_content_identity_name_and_signed_zero_invariance() -> None:
    negative_zero = _identity_fixture(name="display-name-a", signed_zero=-0.0)
    positive_zero = _identity_fixture(name="renamed", signed_zero=0.0)
    assert negative_zero.name != positive_zero.name
    assert (
        core._auxiliary_basis_content_identity_sha256(negative_zero)
        == core._auxiliary_basis_content_identity_sha256(positive_zero)
    )


@pytest.mark.parametrize(
    ("keyword", "changed"),
    [
        ("first_exponent", 1.2500000000000002),
        ("first_coefficient", 0.7500000000000001),
    ],
)
def test_basis_content_identity_is_sensitive_to_numerical_content(
    keyword: str, changed: float
) -> None:
    reference = _identity_fixture()
    variant = _identity_fixture(**{keyword: changed})
    assert (
        core._auxiliary_basis_content_identity_sha256(reference)
        != core._auxiliary_basis_content_identity_sha256(variant)
    )


@pytest.mark.parametrize(
    "change",
    [
        {"first_pure": True},
        {"second_origin": (1.25, -0.5, 0.7500000000000001)},
        {"reverse_shells": True},
    ],
)
def test_basis_content_identity_is_sensitive_to_structure(change: dict) -> None:
    reference = _identity_fixture()
    variant = _identity_fixture(**change)
    assert (
        core._auxiliary_basis_content_identity_sha256(reference)
        != core._auxiliary_basis_content_identity_sha256(variant)
    )


def test_production_source_has_no_shell_inventory_or_vector_staging() -> None:
    source = (_ROOT / "cpp/src/periodic_auxiliary_fourier.cpp").read_text()
    header = (
        _ROOT / "cpp/include/vibeqc/periodic_auxiliary_fourier.hpp"
    ).read_text()
    cmake = (_ROOT / "cpp/CMakeLists.txt").read_text()
    bindings = (_ROOT / "cpp/src/bindings.cpp").read_text()
    assert "basis.shells()" not in source
    assert "AuxiliaryFourierVectorView" in header
    assert "src/periodic_auxiliary_fourier.cpp" in cmake
    assert '#include "periodic_auxiliary_fourier_bindings.cpp"' in bindings
    assert "bind_periodic_auxiliary_fourier(m);" in bindings
