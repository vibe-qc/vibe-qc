"""Contract tests for the native periodic restricted mean-field snapshot."""

from __future__ import annotations

import hashlib
import math
import re
import resource
import struct
import sys
from collections.abc import Callable

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core


_IDENTITY = "1" * 64
_RECIPROCAL = np.diag([2.0, 3.0, 4.0])


def _rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss * (1 if sys.platform == "darwin" else 1024))


def _expected_resident_bytes(
    mesh: tuple[int, int, int], n_basis: int, n_orbitals: int
) -> int:
    n_kpoints = math.prod(mesh)
    complex_values = 2 * n_basis * n_basis + n_basis * n_orbitals
    real_values = 2 * n_orbitals + 4
    mask_values = 3 * n_orbitals
    return 9 * 8 + n_kpoints * (
        complex_values * 16 + real_values * 8 + mask_values
    )


class _CanonicalDigest:
    """Independent oracle for the native state-digest wire format."""

    def __init__(self, domain: str) -> None:
        self._hasher = hashlib.sha256()
        self.string(domain)
        self.u32(core._PERIODIC_MEAN_FIELD_STATE_DIGEST_VERSION)

    def u32(self, value: int) -> None:
        self._hasher.update(struct.pack(">I", value))

    def u64(self, value: int) -> None:
        self._hasher.update(struct.pack(">Q", value))

    def double(self, value: float) -> None:
        number = float(value)
        if number == 0.0:
            number = 0.0
        self._hasher.update(struct.pack(">d", number))

    def string(self, value: str) -> None:
        encoded = value.encode("utf-8")
        self.u64(len(encoded))
        self._hasher.update(encoded)

    def vector(self, value: object) -> None:
        vector = np.asarray(value, dtype=float).reshape(-1)
        self.u64(len(vector))
        for element in vector:
            self.double(float(element))

    def real_matrix(self, value: object) -> None:
        matrix = np.asarray(value, dtype=float)
        self.u64(matrix.shape[0])
        self.u64(matrix.shape[1])
        for column in range(matrix.shape[1]):
            for row in range(matrix.shape[0]):
                self.double(float(matrix[row, column]))

    def complex_matrix(self, value: object) -> None:
        matrix = np.asarray(value, dtype=np.complex128)
        self.u64(matrix.shape[0])
        self.u64(matrix.shape[1])
        for column in range(matrix.shape[1]):
            for row in range(matrix.shape[0]):
                element = complex(matrix[row, column])
                self.double(element.real)
                self.double(element.imag)

    def mask(self, value: object) -> None:
        mask = bytes(int(element) for element in value)  # type: ignore[arg-type]
        self.u64(len(mask))
        self._hasher.update(mask)

    def finish(self) -> str:
        return self._hasher.hexdigest()


def _python_payload_sha256(state: object) -> str:
    digest = _CanonicalDigest(
        "vibeqc.periodic.restricted-mean-field-state.numerical-payload"
    )
    digest.u32(state.periodic_dimension)
    for value in state.mesh:
        digest.u64(value)
    for value in state.is_shift:
        digest.u64(value)
    digest.u64(state.n_kpoints)
    digest.u64(state.n_basis)
    digest.u64(state.n_effective_orbitals)
    digest.real_matrix(state.reciprocal_lattice)
    digest.double(state.reference_energy_per_cell)
    for kpoint in range(state.n_kpoints):
        digest.vector(state.kpoint_cartesian(kpoint))
        digest.double(state.weight(kpoint))
        digest.complex_matrix(state.overlap(kpoint))
        digest.complex_matrix(state.fock(kpoint))
        digest.complex_matrix(state.coefficients(kpoint))
        digest.vector(state.orbital_energies(kpoint))
        digest.vector(state.occupations(kpoint))
        digest.mask(state.frozen_core_mask(kpoint))
        digest.mask(state.correlated_occupied_mask(kpoint))
        digest.mask(state.virtual_mask(kpoint))
    return digest.finish()


def _python_state_identity_sha256(state: object) -> str:
    digest = _CanonicalDigest(
        "vibeqc.periodic.restricted-mean-field-state.identity"
    )
    digest.u32(state.contract_version)
    digest.string(state.calculation_identity)
    digest.string("restricted-hartree-fock")
    digest.string("unnormalized-ao-bloch-sums-uniform-full-bz-weights")
    digest.double(state.requested_minimum_band_gap_hartree)
    tolerances = state.validation_tolerances
    digest.u32(tolerances.version)
    digest.double(tolerances.matrix_absolute)
    digest.double(tolerances.matrix_relative)
    digest.double(tolerances.coordinate_absolute)
    digest.double(tolerances.coordinate_relative)
    digest.double(tolerances.scalar_absolute)
    digest.double(tolerances.scalar_relative)
    digest.double(tolerances.reciprocal_lattice_relative_volume_floor)
    digest.double(tolerances.overlap_eigenvalue_relative_floor)
    digest.string(state.numerical_payload_sha256)
    return digest.finish()


def _coordinates(shift: tuple[int, int, int]) -> list[np.ndarray]:
    # Explicit RegularKMesh order for a 2x1x1 mesh. The final mesh index is
    # the fastest-changing one; here its extent is one.
    fractions = [
        np.array([shift[0] / 4.0, shift[1] / 2.0, shift[2] / 2.0]),
        np.array([(2 + shift[0]) / 4.0, shift[1] / 2.0, shift[2] / 2.0]),
    ]
    return [_RECIPROCAL @ fractional for fractional in fractions]


def _point(
    index: int,
    *,
    shift: tuple[int, int, int] = (0, 0, 0),
    n_basis: int = 3,
    energies: tuple[float, ...] | None = None,
) -> dict[str, object]:
    if energies is None:
        energies = (-1.0, 0.5) if index == 0 else (-0.8, 0.7)
    n_orbitals = len(energies)

    coefficients = np.eye(n_basis, dtype=np.complex128)[:, :n_orbitals]
    if n_basis == 3 and n_orbitals == 2 and index == 1:
        coefficients = np.array(
            [[0.0, 1.0j], [1.0, 0.0], [0.0, 0.0]],
            dtype=np.complex128,
        )

    occupied_projector = coefficients @ coefficients.conj().T
    fock = coefficients @ np.diag(energies) @ coefficients.conj().T
    fock += (2.0 + 0.1 * index) * (
        np.eye(n_basis, dtype=np.complex128) - occupied_projector
    )

    return {
        "k_cartesian": _coordinates(shift)[index],
        "weight": 0.5,
        "overlap": np.eye(n_basis, dtype=np.complex128),
        "fock": fock,
        "coefficients": coefficients,
        "orbital_energies": np.asarray(energies, dtype=float),
        "occupations": np.array([2.0] + [0.0] * (n_orbitals - 1)),
        "frozen_core_mask": [0] * n_orbitals,
        "correlated_occupied_mask": [1] + [0] * (n_orbitals - 1),
        "virtual_mask": [0] + [1] * (n_orbitals - 1),
    }


def _points(
    shift: tuple[int, int, int] = (0, 0, 0),
) -> list[dict[str, object]]:
    return [_point(0, shift=shift), _point(1, shift=shift)]


def _input(
    *,
    mesh: tuple[int, int, int] = (2, 1, 1),
    shift: tuple[int, int, int] = (0, 0, 0),
    points: list[dict[str, object]] | None = None,
    add_count: int | None = None,
    n_basis: int = 3,
    n_orbitals: int = 2,
    **overrides: object,
) -> object:
    data = core._PeriodicRestrictedMeanFieldInput()
    data.calculation_identity = _IDENTITY
    data.reference_kind = core._PeriodicMeanFieldReferenceKind.RESTRICTED_HARTREE_FOCK
    data.normalization = (
        core._PeriodicMeanFieldNormalizationConvention
        .UNNORMALIZED_AO_BLOCH_SUMS_UNIFORM_FULL_BZ_WEIGHTS
    )
    data.periodic_dimension = 3
    data.mesh = mesh
    data.is_shift = shift
    data.reciprocal_lattice = _RECIPROCAL
    data.converged = True
    data.symmetry_reduced_input = False
    data.symmetry_reconstructed_input = False
    data.n_basis = n_basis
    data.n_effective_orbitals = n_orbitals
    data.electrons_per_cell = 2
    data.reference_energy_per_cell = -5.0
    data.minimum_band_gap_hartree = 0.1
    data.validation_tolerance_version = (
        core._PERIODIC_MEAN_FIELD_VALIDATION_TOLERANCE_VERSION
    )
    for name, value in overrides.items():
        setattr(data, name, value)

    blocks = _points(shift) if points is None else points
    count = len(blocks) if add_count is None else add_count
    for block in blocks[:count]:
        data.add_kpoint(**block)
    return data


def _make(**kwargs: object) -> object:
    return core._make_periodic_restricted_mean_field_state(_input(**kwargs))


def _replace_energies(
    block: dict[str, object], energies: tuple[float, ...]
) -> None:
    coefficients = np.asarray(block["coefficients"])
    overlap = np.asarray(block["overlap"])
    complement = np.eye(overlap.shape[0]) - coefficients @ coefficients.conj().T
    block["orbital_energies"] = np.asarray(energies, dtype=float)
    block["fock"] = coefficients @ np.diag(energies) @ coefficients.conj().T
    block["fock"] += 2.0 * complement


def test_full_complex_rectangular_state_round_trips_one_block_at_a_time() -> None:
    state = _make()

    assert core._PERIODIC_RESTRICTED_MEAN_FIELD_STATE_CONTRACT_VERSION == 1
    assert core._PERIODIC_MEAN_FIELD_STATE_DIGEST_VERSION == 2
    assert (
        state.contract_version
        == core._PERIODIC_RESTRICTED_MEAN_FIELD_STATE_CONTRACT_VERSION
    )
    assert state.digest_version == core._PERIODIC_MEAN_FIELD_STATE_DIGEST_VERSION
    assert re.fullmatch(r"[0-9a-f]{64}", state.numerical_payload_sha256)
    assert re.fullmatch(r"[0-9a-f]{64}", state.state_identity_sha256)
    assert state.calculation_identity == _IDENTITY
    assert state.converged is True
    assert (
        state.reference_kind
        == core._PeriodicMeanFieldReferenceKind.RESTRICTED_HARTREE_FOCK
    )
    assert state.normalization == (
        core._PeriodicMeanFieldNormalizationConvention
        .UNNORMALIZED_AO_BLOCH_SUMS_UNIFORM_FULL_BZ_WEIGHTS
    )
    assert state.periodic_dimension == 3
    assert state.mesh == [2, 1, 1]
    assert state.is_shift == [0, 0, 0]
    np.testing.assert_array_equal(state.reciprocal_lattice, _RECIPROCAL)
    assert state.n_kpoints == 2
    assert state.n_basis == 3
    assert state.n_effective_orbitals == 2
    assert (state.n_frozen_core, state.n_correlated_occupied, state.n_virtual) == (
        0,
        1,
        1,
    )
    assert state.uniform_weight == pytest.approx(0.5)
    assert state.reference_energy_per_cell == pytest.approx(-5.0)
    assert state.band_gap_hartree == pytest.approx(1.3)
    assert state.requested_minimum_band_gap_hartree == pytest.approx(0.1)

    np.testing.assert_array_equal(state.kpoint_cartesian(0), [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(state.kpoint_cartesian(1), [1.0, 0.0, 0.0])
    assert state.weight(1) == 0.5
    np.testing.assert_array_equal(state.overlap(1), np.eye(3))
    np.testing.assert_array_equal(state.fock(1), _point(1)["fock"])
    np.testing.assert_array_equal(state.orbital_energies(1), [-0.8, 0.7])
    np.testing.assert_array_equal(state.occupations(1), [2.0, 0.0])
    assert state.frozen_core_mask(1) == [0, 0]
    assert state.correlated_occupied_mask(1) == [1, 0]
    assert state.virtual_mask(1) == [0, 1]

    coefficients = state.coefficients(1)
    assert coefficients.shape == (3, 2)
    assert coefficients[0, 1] == 1.0j
    assert np.iscomplexobj(coefficients)
    coefficients[0, 1] = 0.0
    assert state.coefficients(1)[0, 1] == 1.0j
    with pytest.raises(IndexError):
        state.fock(2)
    assert state.validation_tolerances.version == (
        core._PERIODIC_MEAN_FIELD_VALIDATION_TOLERANCE_VERSION
    )
    assert state.diagnostics.maximum_roothaan_residual == pytest.approx(0.0)


def test_state_digests_match_the_independent_canonical_wire_format() -> None:
    state = _make()

    assert state.numerical_payload_sha256 == _python_payload_sha256(state)
    assert state.state_identity_sha256 == _python_state_identity_sha256(state)
    assert state.numerical_payload_sha256 == (
        "a624e529ae17fd36ba014cd7a18202855048ea00f81d6bc008ea6eaf7221f970"
    )
    assert state.state_identity_sha256 == (
        "317f1b92ed9528f0a6774101b8c659112468b39662701c20085de28fded9710c"
    )


def test_payload_and_state_identity_have_separate_mutation_boundaries() -> None:
    baseline = _make()
    repeated = _make()
    relabeled = _make(calculation_identity="2" * 64)
    stricter_gap = _make(minimum_band_gap_hartree=0.2)
    changed_energy = _make(
        reference_energy_per_cell=np.nextafter(-5.0, -np.inf)
    )
    changed_dimension = _make(periodic_dimension=2)

    assert repeated.numerical_payload_sha256 == baseline.numerical_payload_sha256
    assert repeated.state_identity_sha256 == baseline.state_identity_sha256
    assert relabeled.numerical_payload_sha256 == baseline.numerical_payload_sha256
    assert relabeled.state_identity_sha256 != baseline.state_identity_sha256
    assert stricter_gap.numerical_payload_sha256 == baseline.numerical_payload_sha256
    assert stricter_gap.state_identity_sha256 != baseline.state_identity_sha256
    assert changed_energy.numerical_payload_sha256 != baseline.numerical_payload_sha256
    assert changed_energy.state_identity_sha256 != baseline.state_identity_sha256
    assert (
        changed_dimension.numerical_payload_sha256
        != baseline.numerical_payload_sha256
    )
    assert changed_dimension.state_identity_sha256 != baseline.state_identity_sha256


def test_exact_payload_digest_is_orbital_gauge_sensitive() -> None:
    baseline = _make()
    points = _points()
    coefficients = np.asarray(points[0]["coefficients"]).copy()
    coefficients[:, 0] *= 1.0j
    points[0]["coefficients"] = coefficients

    phase_rotated = _make(points=points)

    assert (
        phase_rotated.numerical_payload_sha256
        != baseline.numerical_payload_sha256
    )
    assert phase_rotated.state_identity_sha256 != baseline.state_identity_sha256


def test_exact_payload_digest_includes_the_explicit_frozen_core_partition() -> None:
    points = [
        _point(0, energies=(-1.5, -1.0, 0.5)),
        _point(1, energies=(-1.4, -0.8, 0.7)),
    ]
    for point in points:
        point["occupations"] = np.array([2.0, 2.0, 0.0])
        point["frozen_core_mask"] = [1, 0, 0]
        point["correlated_occupied_mask"] = [0, 1, 0]
        point["virtual_mask"] = [0, 0, 1]
    baseline = _make(
        points=points,
        n_orbitals=3,
        electrons_per_cell=4,
    )

    swapped_points = [dict(point) for point in points]
    for point in swapped_points:
        point["frozen_core_mask"] = [0, 1, 0]
        point["correlated_occupied_mask"] = [1, 0, 0]
    swapped = _make(
        points=swapped_points,
        n_orbitals=3,
        electrons_per_cell=4,
    )

    assert swapped.numerical_payload_sha256 != baseline.numerical_payload_sha256
    assert swapped.state_identity_sha256 != baseline.state_identity_sha256


def test_digest_canonicalizes_signed_zero_and_ignores_array_layout() -> None:
    baseline = _make()

    signed_zero_points = _points()
    for point in signed_zero_points:
        coordinate = np.asarray(point["k_cartesian"]).copy()
        coordinate[coordinate == 0.0] = -0.0
        point["k_cartesian"] = coordinate
    signed_zero = _make(points=signed_zero_points)

    fortran_points = _points()
    for point in fortran_points:
        for field in ("overlap", "fock", "coefficients"):
            point[field] = np.asfortranarray(point[field])
    fortran_layout = _make(points=fortran_points)

    assert signed_zero.numerical_payload_sha256 == baseline.numerical_payload_sha256
    assert signed_zero.state_identity_sha256 == baseline.state_identity_sha256
    assert fortran_layout.numerical_payload_sha256 == baseline.numerical_payload_sha256
    assert fortran_layout.state_identity_sha256 == baseline.state_identity_sha256


def test_digest_uses_the_canonical_stored_uniform_weight() -> None:
    baseline = _make()
    points = _points()
    points[0]["weight"] = np.nextafter(0.5, 1.0)

    rounded_input = _make(points=points)

    assert rounded_input.weight(0) == 0.5
    assert rounded_input.numerical_payload_sha256 == baseline.numerical_payload_sha256
    assert rounded_input.state_identity_sha256 == baseline.state_identity_sha256


@pytest.mark.parametrize(
    ("shift", "expected"),
    [
        ((0, 0, 0), ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])),
        ((1, 0, 0), ([0.5, 0.0, 0.0], [1.5, 0.0, 0.0])),
    ],
    ids=["unshifted", "shifted"],
)
def test_exact_regular_mesh_coordinates(
    shift: tuple[int, int, int], expected: tuple[list[float], list[float]]
) -> None:
    state = _make(shift=shift)
    np.testing.assert_array_equal(state.kpoint_cartesian(0), expected[0])
    np.testing.assert_array_equal(state.kpoint_cartesian(1), expected[1])
    if shift[0]:
        assert not np.array_equal(state.kpoint_cartesian(0), np.zeros(3))


@pytest.mark.parametrize(
    ("shift", "expected"),
    [
        (
            (0, 0, 0),
            ([0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [1.0, 0.0, 0.0], [1.0, 0.0, 2.0]),
        ),
        (
            (1, 0, 1),
            ([0.5, 0.0, 1.0], [0.5, 0.0, 3.0], [1.5, 0.0, 1.0], [1.5, 0.0, 3.0]),
        ),
    ],
    ids=["unshifted", "shifted"],
)
def test_state_uses_exact_last_axis_fast_coordinate_order(
    shift: tuple[int, int, int],
    expected: tuple[list[float], ...],
) -> None:
    points = []
    for index, coordinate in enumerate(expected):
        block = _point(index % 2)
        block["k_cartesian"] = np.asarray(coordinate)
        block["weight"] = 0.25
        points.append(block)

    state = _make(mesh=(2, 1, 2), shift=shift, points=points)
    assert [state.kpoint_cartesian(i).tolist() for i in range(4)] == list(expected)


def test_resident_bytes_match_the_exact_owned_payload_formula() -> None:
    expected = _expected_resident_bytes((2, 1, 1), 3, 2)
    assert expected == 980
    state_bytes = _make().resident_bytes
    assert state_bytes == expected
    assert core._estimate_periodic_restricted_mean_field_resident_bytes(
        (2, 1, 1), 3, 2
    ) == expected


@pytest.mark.parametrize(
    ("mesh", "n_basis", "n_orbitals"),
    [
        ((8, 8, 8), 37, 37),
        ((6, 6, 6), 196, 196),
    ],
    ids=["eight-cubed", "six-cubed"],
)
def test_target_mesh_estimate_is_metadata_only(
    mesh: tuple[int, int, int], n_basis: int, n_orbitals: int
) -> None:
    before = _rss_bytes()
    estimated = core._estimate_periodic_restricted_mean_field_resident_bytes(
        mesh, n_basis, n_orbitals
    )
    after = _rss_bytes()

    assert estimated == _expected_resident_bytes(mesh, n_basis, n_orbitals)
    assert after - before < 64 * 1024 * 1024


def test_resident_byte_overflow_fails_before_allocation() -> None:
    before = _rss_bytes()
    with pytest.raises(OverflowError, match="resident-byte overflow"):
        core._estimate_periodic_restricted_mean_field_resident_bytes(
            (1, 1, 1), 2**32, 2**32
        )
    assert _rss_bytes() - before < 64 * 1024 * 1024


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"converged": False}, "converged SCF snapshot"),
        ({"symmetry_reduced_input": True}, "symmetry-reduced"),
        ({"symmetry_reconstructed_input": True}, "symmetry-reconstructed"),
        ({"calculation_identity": "A" * 64}, "lowercase SHA-256"),
        ({"periodic_dimension": 0}, "periodic_dimension"),
        ({"periodic_dimension": 4}, "periodic_dimension"),
        (
            {"periodic_dimension": 1, "mesh": (2, 2, 1)},
            "inactive mesh axes",
        ),
        (
            {"periodic_dimension": 1, "is_shift": (0, 1, 0)},
            "inactive mesh axes",
        ),
        ({"electrons_per_cell": 3}, "positive even integer"),
        ({"minimum_band_gap_hartree": 0.0}, "strictly positive"),
        ({"validation_tolerance_version": 2}, "tolerance version"),
    ],
    ids=[
        "nonconverged",
        "reduced",
        "reconstructed",
        "identity",
        "unset-periodic-dimension",
        "invalid-periodic-dimension",
        "inactive-axis-extent",
        "inactive-axis-shift",
        "odd-electron-count",
        "zero-gap-floor",
        "unknown-tolerance-version",
    ],
)
def test_rejected_snapshot_provenance(
    overrides: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _make(**overrides)


@pytest.mark.parametrize(
    ("field", "bad_value", "match"),
    [
        ("overlap", np.eye(2), "overlap matrix shape"),
        ("fock", np.eye(2), "Fock matrix shape"),
        ("coefficients", np.eye(3), "coefficient matrix shape"),
        ("orbital_energies", np.zeros(1), "orbital-energy vector length"),
        ("occupations", np.zeros(1), "occupation vector length"),
        ("frozen_core_mask", [0], "frozen-core mask length"),
    ],
)
def test_inconsistent_block_shapes_fail_closed(
    field: str, bad_value: object, match: str
) -> None:
    points = _points()
    points[1][field] = bad_value
    with pytest.raises(ValueError, match=match):
        _make(points=points)


def _nonfinite_reciprocal() -> object:
    reciprocal = _RECIPROCAL.copy()
    reciprocal[0, 0] = np.nan
    return reciprocal


def _mutate_coordinate(points: list[dict[str, object]]) -> None:
    points[0]["k_cartesian"] = np.array([np.nan, 0.0, 0.0])


def _mutate_overlap(points: list[dict[str, object]]) -> None:
    overlap = np.asarray(points[0]["overlap"]).copy()
    overlap[0, 0] = np.inf
    points[0]["overlap"] = overlap


def _mutate_energies(points: list[dict[str, object]]) -> None:
    points[0]["orbital_energies"] = np.array([-1.0, np.nan])


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (_mutate_coordinate, "Cartesian coordinate finiteness"),
        (_mutate_overlap, "matrix/vector finiteness"),
        (_mutate_energies, "matrix/vector finiteness"),
    ],
    ids=["coordinate", "matrix", "vector"],
)
def test_nonfinite_block_data_fail_closed(
    mutation: Callable[[list[dict[str, object]]], None], match: str
) -> None:
    points = _points()
    mutation(points)
    with pytest.raises(ValueError, match=match):
        _make(points=points)


def test_nonfinite_snapshot_metadata_fail_closed() -> None:
    with pytest.raises(ValueError, match="reciprocal lattice"):
        _make(reciprocal_lattice=_nonfinite_reciprocal())
    with pytest.raises(ValueError, match="reference_energy_per_cell"):
        _make(reference_energy_per_cell=np.inf)
    with pytest.raises(ValueError, match="minimum_band_gap_hartree"):
        _make(minimum_band_gap_hartree=np.nan)


def test_singular_reciprocal_lattice_cannot_alias_the_full_mesh_to_gamma() -> None:
    with pytest.raises(ValueError, match="three linearly independent vectors"):
        _make(reciprocal_lattice=np.zeros((3, 3)))


@pytest.mark.parametrize(
    ("matrix_name", "match"),
    [("overlap", "overlap Hermiticity"), ("fock", "Fock Hermiticity")],
)
def test_nonhermitian_matrices_fail_closed(matrix_name: str, match: str) -> None:
    points = _points()
    matrix = np.asarray(points[0][matrix_name]).copy()
    matrix[0, 1] = 0.2j
    points[0][matrix_name] = matrix
    with pytest.raises(ValueError, match=match):
        _make(points=points)


def test_finite_inputs_whose_derived_norm_overflows_still_fail_closed() -> None:
    points = _points()
    huge = np.finfo(float).max
    overlap = np.asarray(points[0]["overlap"]).copy()
    overlap[0, 1] = complex(huge, huge)
    points[0]["overlap"] = overlap
    with pytest.raises(ValueError, match="overlap Hermiticity"):
        _make(points=points)


def test_nonpositive_overlap_fails_closed() -> None:
    points = _points()
    points[0]["overlap"] = np.diag([1.0, 1.0, 0.0]).astype(complex)
    with pytest.raises(ValueError, match="positive-definite overlap metric"):
        _make(points=points)


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"frozen_core_mask": [2, 0]}, "mask value"),
        (
            {"frozen_core_mask": [1, 0], "correlated_occupied_mask": [1, 0]},
            "disjoint and complete",
        ),
        (
            {"correlated_occupied_mask": [0, 0]},
            "disjoint and complete",
        ),
        (
            {
                "correlated_occupied_mask": [0, 0],
                "frozen_core_mask": [1, 0],
            },
            "at least one correlated occupied and one virtual",
        ),
        (
            {
                "correlated_occupied_mask": [1, 1],
                "virtual_mask": [0, 0],
                "occupations": np.array([2.0, 2.0]),
            },
            "at least one correlated occupied and one virtual",
        ),
        (
            {"occupations": np.array([1.5, 0.0])},
            "closed-shell 2/0 occupations and masks",
        ),
    ],
    ids=[
        "nonbinary",
        "overlap",
        "incomplete",
        "no-correlated-occupied",
        "no-virtual",
        "fractional-occupation",
    ],
)
def test_masks_and_occupations_fail_closed(
    updates: dict[str, object], match: str
) -> None:
    points = _points()
    points[0].update(updates)
    with pytest.raises(ValueError, match=match):
        _make(points=points)


def test_per_kpoint_orbital_counts_must_be_identical() -> None:
    points = [
        _point(0, n_basis=4, energies=(-1.2, -0.8, 0.5, 0.9)),
        _point(1, n_basis=4, energies=(-1.1, -0.7, 0.6, 1.0)),
    ]
    for point in points:
        point["occupations"] = np.array([2.0, 2.0, 0.0, 0.0])
        point["correlated_occupied_mask"] = [1, 1, 0, 0]
        point["virtual_mask"] = [0, 0, 1, 1]
    points[1]["frozen_core_mask"] = [1, 0, 0, 0]
    points[1]["correlated_occupied_mask"] = [0, 1, 0, 0]

    with pytest.raises(ValueError, match="orbital-mask ranks"):
        _make(
            points=points,
            n_basis=4,
            n_orbitals=4,
            electrons_per_cell=4,
        )


def test_full_mesh_count_order_weight_and_electron_count_fail_closed() -> None:
    with pytest.raises(ValueError, match="requires all points"):
        _make(add_count=1)

    points = _points()
    points[0]["k_cartesian"], points[1]["k_cartesian"] = (
        points[1]["k_cartesian"],
        points[0]["k_cartesian"],
    )
    with pytest.raises(ValueError, match="RegularKMesh ordering"):
        _make(points=points)

    points = _points()
    points[0]["weight"] = 0.4
    with pytest.raises(ValueError, match="uniform full-BZ weight"):
        _make(points=points)

    with pytest.raises(ValueError, match="weighted electron count"):
        _make(electrons_per_cell=4)

    with pytest.raises(ValueError, match="positive even integer"):
        _make(electrons_per_cell=3)


def test_indirect_global_gap_must_exceed_the_requested_floor() -> None:
    points = _points()
    _replace_energies(points[0], (-0.2, 0.6))
    _replace_energies(points[1], (-1.0, 0.0))
    with pytest.raises(ValueError, match="global band gap"):
        _make(points=points, minimum_band_gap_hartree=0.3)


def _capture_request(
    *,
    n_basis: int = 3,
    masks: list[list[int]] | None = None,
    byte_limit: int = 10_000,
) -> object:
    request = core._PeriodicRHFStateCaptureRequest()
    request.calculation_identity = _IDENTITY
    request.maximum_retained_numerical_payload_bytes = byte_limit
    request.minimum_band_gap_hartree = 0.1
    request.frozen_core_mask_per_k = (
        [[0] * n_basis for _ in range(2)] if masks is None else masks
    )
    return request


def _capture_preflight(request: object, **overrides: object) -> object:
    arguments: dict[str, object] = {
        "request": request,
        "periodic_dimension": 3,
        "mesh": (2, 1, 1),
        "is_shift": (0, 0, 0),
        "reciprocal_lattice": _RECIPROCAL,
        "kpoints": _coordinates((0, 0, 0)),
        "weights": [0.5, 0.5],
        "symmetry_reduced_or_reconstructed": False,
        "n_basis": 3,
        "electrons_per_cell": 2,
    }
    arguments.update(overrides)
    return core._preflight_periodic_rhf_state_capture(**arguments)


def test_capture_preflight_is_exact_and_retained_payload_bounded() -> None:
    expected = _expected_resident_bytes((2, 1, 1), 3, 3)
    preflight = _capture_preflight(
        _capture_request(byte_limit=expected)
    )
    assert preflight.retained_numerical_payload_bytes == expected
    assert (
        preflight.n_frozen_core,
        preflight.n_correlated_occupied,
        preflight.n_virtual,
    ) == (0, 1, 2)

    with pytest.raises(RuntimeError, match="above the explicit limit"):
        _capture_preflight(_capture_request(byte_limit=expected - 1))


@pytest.mark.parametrize(
    ("mutate", "overrides", "match"),
    [
        (
            lambda request: setattr(request, "calculation_identity", "BAD"),
            {},
            "SHA-256",
        ),
        (
            lambda request: setattr(request, "minimum_band_gap_hartree", 0.0),
            {},
            "minimum_band_gap",
        ),
        (
            lambda request: None,
            {"periodic_dimension": 0},
            "periodic_dimension",
        ),
        (
            lambda request: None,
            {"periodic_dimension": 1, "mesh": (2, 2, 1)},
            "inactive mesh axes",
        ),
        (
            lambda request: None,
            {"periodic_dimension": 1, "is_shift": (0, 1, 0)},
            "inactive mesh axes",
        ),
        (
            lambda request: None,
            {"symmetry_reduced_or_reconstructed": True},
            "full Brillouin",
        ),
        (lambda request: None, {"weights": [0.4, 0.6]}, "uniform full-BZ"),
        (
            lambda request: None,
            {"kpoints": list(reversed(_coordinates((0, 0, 0))))},
            "RegularKMesh order",
        ),
        (
            lambda request: setattr(request, "frozen_core_mask_per_k", []),
            {},
            "one explicit frozen-core mask",
        ),
        (
            lambda request: setattr(
                request, "frozen_core_mask_per_k", [[0, 0], [0, 0]]
            ),
            {},
            "mask length",
        ),
        (
            lambda request: setattr(
                request, "frozen_core_mask_per_k", [[2, 0, 0], [2, 0, 0]]
            ),
            {},
            "masks must be binary",
        ),
        (
            lambda request: setattr(
                request, "frozen_core_mask_per_k", [[1, 0, 0], [1, 0, 0]]
            ),
            {},
            "at least one correlated occupied",
        ),
        (
            lambda request: setattr(
                request, "frozen_core_mask_per_k", [[0, 1, 0], [0, 1, 0]]
            ),
            {},
            "freeze a virtual band",
        ),
    ],
    ids=[
        "identity",
        "gap",
        "dimension",
        "inactive-axis-extent",
        "inactive-axis-shift",
        "symmetry",
        "weight",
        "order",
        "missing-masks",
        "mask-length",
        "mask-nonbinary",
        "all-occupied-frozen",
        "virtual-frozen",
    ],
)
def test_capture_preflight_rejects_ambiguous_inputs(
    mutate: Callable[[object], None], overrides: dict[str, object], match: str
) -> None:
    request = _capture_request()
    mutate(request)
    with pytest.raises(ValueError, match=match):
        _capture_preflight(request, **overrides)


def test_capture_preflight_requires_common_frozen_rank() -> None:
    request = _capture_request(
        n_basis=4,
        masks=[[1, 0, 0, 0], [0, 0, 0, 0]],
    )
    with pytest.raises(ValueError, match="rank must be identical"):
        _capture_preflight(
            request,
            n_basis=4,
            electrons_per_cell=4,
        )


def test_capture_preflight_accepts_an_explicit_nonzero_frozen_core() -> None:
    request = _capture_request(
        n_basis=4,
        masks=[[1, 0, 0, 0], [1, 0, 0, 0]],
    )
    preflight = _capture_preflight(
        request,
        n_basis=4,
        electrons_per_cell=4,
    )
    assert (
        preflight.n_frozen_core,
        preflight.n_correlated_occupied,
        preflight.n_virtual,
    ) == (1, 1, 2)


def test_physical_density_diagnostics_use_spin_summed_metric_idempotency() -> None:
    overlap = np.eye(2, dtype=np.complex128)
    coefficients = np.array(
        [[1.0j, 0.0], [0.0, 1.0]], dtype=np.complex128
    )
    density = 2.0 * coefficients[:, :1] @ coefficients[:, :1].conj().T
    fock = coefficients @ np.diag([-1.0, 0.5]) @ coefficients.conj().T

    diagnostics = core._periodic_rhf_physical_density_diagnostics(
        overlap, fock, density
    )
    assert diagnostics.commutator_frobenius == pytest.approx(0.0)
    assert diagnostics.metric_idempotency_frobenius == pytest.approx(0.0)
    assert diagnostics.metric_idempotency_relative == pytest.approx(0.0)
    closure = core._periodic_rhf_density_fixed_point_diagnostics(
        density, coefficients, 1
    )
    assert closure.relative == pytest.approx(0.0)

    mixed_density = 0.75 * density
    mixed = core._periodic_rhf_physical_density_diagnostics(
        overlap, fock, mixed_density
    )
    assert mixed.metric_idempotency_relative > 0.1

    nonstationary_fock = np.array(
        [[-1.0, 0.2j], [-0.2j, 0.5]], dtype=np.complex128
    )
    nonstationary = core._periodic_rhf_physical_density_diagnostics(
        overlap, nonstationary_fock, density
    )
    assert nonstationary.commutator_frobenius > 0.1


def test_physical_density_diagnostics_use_the_generalized_metric_order() -> None:
    overlap = np.array([[1.4, 0.2], [0.2, 0.8]], dtype=np.complex128)
    eigenvalues, eigenvectors = np.linalg.eigh(overlap)
    inverse_sqrt = (
        eigenvectors
        @ np.diag(eigenvalues ** -0.5)
        @ eigenvectors.conj().T
    )
    unitary = np.array([[1.0, 1.0j], [1.0j, 1.0]]) / np.sqrt(2.0)
    coefficients = inverse_sqrt @ unitary
    orbital_energies = np.diag([-0.7, 0.9])
    fock = (
        overlap
        @ coefficients
        @ orbital_energies
        @ coefficients.conj().T
        @ overlap
    )
    density = (
        2.0
        * coefficients[:, :1]
        @ coefficients[:, :1].conj().T
    )

    diagnostics = core._periodic_rhf_physical_density_diagnostics(
        overlap, fock, density
    )
    assert diagnostics.commutator_frobenius < 1.0e-13
    assert diagnostics.metric_idempotency_frobenius < 1.0e-13


def test_small_commutator_does_not_imply_projector_closure() -> None:
    theta = 2.0e-7
    coefficients = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ],
        dtype=np.complex128,
    )
    density = (
        2.0
        * coefficients[:, :1]
        @ coefficients[:, :1].conj().T
    )
    overlap = np.eye(2, dtype=np.complex128)
    fock = np.diag([-0.5, 0.5]).astype(np.complex128)

    diagnostics = core._periodic_rhf_physical_density_diagnostics(
        overlap, fock, density
    )
    closure = core._periodic_rhf_density_fixed_point_diagnostics(
        density, overlap, 1
    )
    assert diagnostics.commutator_frobenius < 1.0e-6
    assert closure.relative > 1.0e-8
