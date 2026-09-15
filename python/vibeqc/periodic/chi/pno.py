"""Periodic PAO, pair-orbit, and PNO primitives for AICCM2026DEV-B.

The functions here are deliberately independent of the molecular DLPNO
driver.  They define the finite Born-von Karman (BvK) objects consumed by
that driver and expose invariants that can be tested before a correlation
energy is trusted.

For an AO overlap ``S`` and an ``S``-orthonormal occupied coefficient matrix
``C_o``, the coefficient-space projector is

``Q = I - C_o C_o^H S``.

Columns of ``Q`` selected by a pair domain are redundant PAOs.  Canonical
orthogonalization of their Gram matrix removes the null space without
changing their span.  For a pair amplitude matrix ``T_ij`` the Hermitian
model pair density is

``D_ij = (T T^H + T^H T) / (1 + delta_ij)``.

Its eigenvectors are the PNOs and its eigenvalues are non-negative pair
occupations.  Tightening the occupation threshold therefore gives nested
ranks and the zero-threshold limit retains the complete PAO pair space.
The MP2 energy itself is not variational, so strict monotonicity of the
energy error remains a numerical validation criterion rather than a theorem.

The pair-density convention matches the real closed-shell construction in
Pinski et al., J. Chem. Phys. 143, 034108 (2015).  The finite-BvK pair-orbit
bookkeeping follows Nejad et al., J. Chem. Phys. 163, 214107 (2025),
doi:10.1063/5.0290816: simultaneous translations act on both occupied
indices, and optional space-group permutations enlarge those orbits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from itertools import product
from math import prod
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "AICCM2026DevBPAOSpace",
    "AICCM2026DevBPNOSpace",
    "AICCM2026DevBPairOrbit",
    "AICCM2026DevBPointPairPlan",
    "AICCM2026DevBTranslationPairPlan",
    "build_point_pair_plan",
    "build_translation_pair_plan",
    "build_pair_natural_orbitals",
    "enumerate_pair_orbits",
    "full_space_noncanonical_mp2_energy",
    "projected_atomic_orbitals",
    "translation_permutations",
]


@dataclass(frozen=True)
class AICCM2026DevBPAOSpace:
    """An orthonormal PAO basis for one finite-torus domain."""

    coefficients: np.ndarray
    orbital_energies: np.ndarray
    projector: np.ndarray
    gram_eigenvalues: np.ndarray
    source_ao_indices: np.ndarray
    orthonormality_error: float
    occupied_leakage_error: float
    discarded_rank: int

    @property
    def rank(self) -> int:
        return int(self.coefficients.shape[1])


@dataclass(frozen=True)
class AICCM2026DevBPNOSpace:
    """Pair natural orbitals obtained from one PAO pair space."""

    coefficients: np.ndarray
    occupations: np.ndarray
    retained_occupations: np.ndarray
    orbital_energies: np.ndarray | None
    retained_rank: int
    discarded_occupation: float
    hermiticity_error: float
    minimum_occupation: float


@dataclass(frozen=True)
class AICCM2026DevBPairOrbit:
    """One orbit of unordered occupied pairs under finite permutations."""

    representative: tuple[int, int]
    members: tuple[tuple[int, int], ...]
    n_cells: int

    @property
    def multiplicity(self) -> int:
        return len(self.members)

    @property
    def per_cell_weight(self) -> float:
        """Coefficient multiplying a representative pair energy per cell."""

        return self.multiplicity / self.n_cells


@dataclass(frozen=True)
class AICCM2026DevBTranslationPairPlan:
    """Compact exact plan for translation-unique occupied-pair work.

    Occupieds are labelled ``(band, cell)`` on the finite group
    ``Z_n0 x Z_n1 x Z_n2``.  Every row represents the unordered pair
    ``{(band_i, 0), (band_j, displacement)}``; simultaneous translation
    supplies its placed members.  Same-band displacements are canonical
    under ``L ~ -L``.  A nonzero self-inverse displacement on an even mesh
    has a two-element stabilizer and therefore multiplicity ``n_cells / 2``;
    all other rows have multiplicity ``n_cells``.

    The structure-of-arrays payload deliberately contains no orbit members
    and no ``(n_cells, n_occupied)`` permutation matrix.  It is therefore the
    workload description that a later streamed MPI/GPU executor can partition
    without first materialising the quadratic placed-pair space.  This class
    is a plan only: constructing it does not enable representative execution
    or change a correlation energy.
    """

    mesh: tuple[int, int, int]
    n_bands: int
    band_pairs: np.ndarray
    displacements: np.ndarray
    multiplicities: np.ndarray
    schema: str = field(
        init=False,
        default="vibeqc.aiccm2026dev-b.translation-pair-plan/v1",
    )
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        mesh = _validated_pair_mesh(self.mesh)
        n_bands = _validated_positive_integer(self.n_bands, "n_bands")
        n_cells, n_occupied, n_placed_pairs = _checked_pair_counts(
            mesh,
            n_bands,
        )

        raw_band_pairs = np.asarray(self.band_pairs)
        raw_displacements = np.asarray(self.displacements)
        raw_multiplicities = np.asarray(self.multiplicities)
        if any(
            array.dtype.kind not in "iu"
            for array in (raw_band_pairs, raw_displacements, raw_multiplicities)
        ):
            raise ValueError("translation-pair plan arrays must contain integers")
        if raw_band_pairs.ndim != 2 or raw_band_pairs.shape[1:] != (2,):
            raise ValueError("translation-pair plan band_pairs must have shape (n, 2)")
        n_representatives = int(raw_band_pairs.shape[0])
        if raw_displacements.shape != (n_representatives, 3):
            raise ValueError(
                "translation-pair plan displacements must have shape (n, 3)"
            )
        if raw_multiplicities.shape != (n_representatives,):
            raise ValueError(
                "translation-pair plan multiplicities must have shape (n,)"
            )
        maximum = np.iinfo(np.int64).max
        for name, array in (
            ("band_pairs", raw_band_pairs),
            ("displacements", raw_displacements),
            ("multiplicities", raw_multiplicities),
        ):
            if array.size and (
                int(np.min(array)) < 0 or int(np.max(array)) > maximum
            ):
                raise ValueError(f"translation-pair plan {name} is out of range")

        band_pairs = np.ascontiguousarray(
            raw_band_pairs,
            dtype=np.int64,
        )
        displacements = np.ascontiguousarray(
            raw_displacements,
            dtype=np.int64,
        )
        multiplicities = np.ascontiguousarray(
            raw_multiplicities,
            dtype=np.int64,
        )
        if band_pairs.size and int(np.max(band_pairs)) >= n_bands:
            raise ValueError("translation-pair plan band index is out of range")
        for axis, extent in enumerate(mesh):
            if displacements.size and int(np.max(displacements[:, axis])) >= extent:
                raise ValueError(
                    "translation-pair plan displacement is outside its finite mesh"
                )
        if multiplicities.size and int(np.min(multiplicities)) <= 0:
            raise ValueError("translation-pair plan multiplicities must be positive")

        rows = [
            (
                int(band_pairs[index, 0]),
                int(band_pairs[index, 1]),
                *(int(value) for value in displacements[index]),
            )
            for index in range(n_representatives)
        ]
        if rows != sorted(rows) or len(set(rows)) != len(rows):
            raise ValueError(
                "translation-pair plan representatives must be unique canonical order"
            )
        for index, row in enumerate(rows):
            band_i, band_j, *raw_displacement = row
            displacement = tuple(raw_displacement)
            inverse = tuple(
                (-displacement[axis]) % mesh[axis]
                for axis in range(3)
            )
            if band_i > band_j or (
                band_i == band_j and displacement > inverse
            ):
                raise ValueError(
                    "translation-pair plan representative is not canonical"
                )
            expected_multiplicity = n_cells
            if (
                band_i == band_j
                and displacement != (0, 0, 0)
                and displacement == inverse
            ):
                expected_multiplicity //= 2
            if int(multiplicities[index]) != expected_multiplicity:
                raise ValueError(
                    "translation-pair plan stabilizer multiplicity is inconsistent"
                )
        if sum(int(value) for value in multiplicities) != n_placed_pairs:
            raise ValueError("translation-pair plan placed-pair census is incomplete")

        expected_representatives = _translation_pair_representative_count(
            mesh,
            n_bands,
        )
        if n_representatives != expected_representatives:
            raise ValueError(
                "translation-pair plan representative census is inconsistent"
            )
        del n_cells, n_occupied

        # A merely read-only owning ndarray can be made writable again with
        # ``setflags(write=True)``.  Rebuild each canonical payload from a
        # Python ``bytes`` buffer so neither a caller alias nor a flag change
        # can invalidate the fingerprint after construction.
        band_pairs = np.frombuffer(
            band_pairs.tobytes(order="C"),
            dtype=np.int64,
        ).reshape(n_representatives, 2)
        displacements = np.frombuffer(
            displacements.tobytes(order="C"),
            dtype=np.int64,
        ).reshape(n_representatives, 3)
        multiplicities = np.frombuffer(
            multiplicities.tobytes(order="C"),
            dtype=np.int64,
        )
        object.__setattr__(self, "mesh", mesh)
        object.__setattr__(self, "n_bands", n_bands)
        object.__setattr__(self, "band_pairs", band_pairs)
        object.__setattr__(self, "displacements", displacements)
        object.__setattr__(self, "multiplicities", multiplicities)
        object.__setattr__(self, "fingerprint", self._build_fingerprint())

    def _build_fingerprint(self) -> str:
        digest = sha256()
        digest.update(self.schema.encode("ascii"))
        digest.update(b"\0")
        for value in (*self.mesh, self.n_bands):
            digest.update(int(value).to_bytes(8, "big", signed=False))
        digest.update(np.asarray(self.band_pairs, dtype=">i8").tobytes())
        digest.update(np.asarray(self.displacements, dtype=">i8").tobytes())
        digest.update(np.asarray(self.multiplicities, dtype=">i8").tobytes())
        return digest.hexdigest()

    @property
    def n_cells(self) -> int:
        return prod(self.mesh)

    @property
    def n_occupied(self) -> int:
        return self.n_cells * self.n_bands

    @property
    def n_representatives(self) -> int:
        return int(self.band_pairs.shape[0])

    @property
    def n_placed_pairs(self) -> int:
        return self.n_occupied * (self.n_occupied + 1) // 2

    @property
    def per_cell_weights(self) -> np.ndarray:
        return np.asarray(self.multiplicities, dtype=float) / self.n_cells

    @property
    def payload_nbytes(self) -> int:
        return sum(
            int(array.nbytes)
            for array in (
                self.band_pairs,
                self.displacements,
                self.multiplicities,
            )
        )


@dataclass(frozen=True, init=False)
class AICCM2026DevBPointPairPlan:
    """Compact point-symmetry quotient of a translation-pair plan.

    ``member_indices`` partitions the rows of ``translation_plan`` into
    orbits generated by caller-supplied occupied-index support permutations.
    The permutations are not retained: ``action_fingerprint`` binds their
    exact canonical support-generator set, while ``parent_fingerprint`` binds
    the D123 plan on which they acted.  ``combined_multiplicities`` therefore
    preserves the full placed-pair census without materialising a single
    placed pair.

    This factory-only immutable object is a workload plan only.  It carries
    no orbital signs or phases, does not attest an orbital gauge or a
    covariant PAO/PNO domain, and is not consumed by an energy path.
    """

    translation_plan: AICCM2026DevBTranslationPairPlan
    representative_indices: np.ndarray
    orbit_offsets: np.ndarray
    member_indices: np.ndarray
    combined_multiplicities: np.ndarray
    n_action_generators: int
    action_fingerprint: str
    schema: str = field(
        init=False,
        default="vibeqc.aiccm2026dev-b.point-pair-plan/v1",
    )
    parent_fingerprint: str = field(init=False)
    fingerprint: str = field(init=False)

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError(
            "AICCM2026DevBPointPairPlan is factory-only; "
            "use build_point_pair_plan(...)"
        )

    @classmethod
    def _from_payload(
        cls,
        *,
        translation_plan: AICCM2026DevBTranslationPairPlan,
        representative_indices: np.ndarray,
        orbit_offsets: np.ndarray,
        member_indices: np.ndarray,
        combined_multiplicities: np.ndarray,
        n_action_generators: int,
        action_fingerprint: str,
    ) -> AICCM2026DevBPointPairPlan:
        instance = object.__new__(cls)
        object.__setattr__(instance, "translation_plan", translation_plan)
        object.__setattr__(
            instance,
            "representative_indices",
            representative_indices,
        )
        object.__setattr__(instance, "orbit_offsets", orbit_offsets)
        object.__setattr__(instance, "member_indices", member_indices)
        object.__setattr__(
            instance,
            "combined_multiplicities",
            combined_multiplicities,
        )
        object.__setattr__(
            instance,
            "n_action_generators",
            n_action_generators,
        )
        object.__setattr__(instance, "action_fingerprint", action_fingerprint)
        instance._validate_freeze_and_fingerprint()
        return instance

    def _validate_freeze_and_fingerprint(self) -> None:
        parent = self.translation_plan
        if not isinstance(parent, AICCM2026DevBTranslationPairPlan):
            raise ValueError(
                "point-pair plan requires an AICCM2026DevBTranslationPairPlan"
            )
        generator_count = _validated_positive_integer(
            self.n_action_generators,
            "n_action_generators",
        )
        if generator_count > int(np.iinfo(np.int64).max):
            raise ValueError("point-pair plan generator count exceeds int64")
        action_fingerprint = _validated_sha256(
            self.action_fingerprint,
            "point-pair action fingerprint",
        )

        raw_representatives = np.asarray(self.representative_indices)
        raw_offsets = np.asarray(self.orbit_offsets)
        raw_members = np.asarray(self.member_indices)
        raw_multiplicities = np.asarray(self.combined_multiplicities)
        arrays = (
            ("representative_indices", raw_representatives),
            ("orbit_offsets", raw_offsets),
            ("member_indices", raw_members),
            ("combined_multiplicities", raw_multiplicities),
        )
        if any(array.dtype.kind not in "iu" for _, array in arrays):
            raise ValueError("point-pair plan arrays must contain integers")
        if raw_representatives.ndim != 1:
            raise ValueError(
                "point-pair plan representative_indices must be one-dimensional"
            )
        n_orbits = int(raw_representatives.size)
        if n_orbits < 1:
            raise ValueError("point-pair plan must contain at least one orbit")
        if raw_offsets.shape != (n_orbits + 1,):
            raise ValueError(
                "point-pair plan orbit_offsets must have shape (n_orbits + 1,)"
            )
        if raw_members.shape != (parent.n_representatives,):
            raise ValueError(
                "point-pair plan member_indices must partition every parent row"
            )
        if raw_multiplicities.shape != (n_orbits,):
            raise ValueError(
                "point-pair plan combined_multiplicities must have shape "
                "(n_orbits,)"
            )

        maximum = int(np.iinfo(np.int64).max)
        for name, array in arrays:
            if array.size and (
                int(np.min(array)) < 0 or int(np.max(array)) > maximum
            ):
                raise ValueError(f"point-pair plan {name} is out of range")
        representatives = np.ascontiguousarray(raw_representatives, dtype=np.int64)
        offsets = np.ascontiguousarray(raw_offsets, dtype=np.int64)
        members = np.ascontiguousarray(raw_members, dtype=np.int64)
        multiplicities = np.ascontiguousarray(raw_multiplicities, dtype=np.int64)

        if int(offsets[0]) != 0 or int(offsets[-1]) != parent.n_representatives:
            raise ValueError("point-pair plan orbit offsets do not cover parent rows")
        if np.any(offsets[1:] <= offsets[:-1]):
            raise ValueError("point-pair plan orbits must be nonempty")
        if not np.array_equal(
            np.sort(members),
            np.arange(parent.n_representatives, dtype=np.int64),
        ):
            raise ValueError(
                "point-pair plan members must be an exact partition of parent rows"
            )
        if np.any(representatives[1:] <= representatives[:-1]):
            raise ValueError(
                "point-pair plan representatives must be in canonical order"
            )
        if multiplicities.size and int(np.min(multiplicities)) <= 0:
            raise ValueError(
                "point-pair plan combined multiplicities must be positive"
            )

        for orbit_index in range(n_orbits):
            start = int(offsets[orbit_index])
            stop = int(offsets[orbit_index + 1])
            orbit_members = members[start:stop]
            if np.any(orbit_members[1:] <= orbit_members[:-1]):
                raise ValueError(
                    "point-pair plan members must be sorted within each orbit"
                )
            if int(representatives[orbit_index]) != int(orbit_members[0]):
                raise ValueError(
                    "point-pair plan representative must be its minimum member"
                )
            parent_multiplicities = parent.multiplicities[orbit_members]
            if np.any(parent_multiplicities != parent_multiplicities[0]):
                raise ValueError(
                    "point-pair plan orbit changes a translation stabilizer"
                )
            expected = sum(int(value) for value in parent_multiplicities)
            if int(multiplicities[orbit_index]) != expected:
                raise ValueError(
                    "point-pair plan combined multiplicity is inconsistent"
                )
        if sum(int(value) for value in multiplicities) != parent.n_placed_pairs:
            raise ValueError("point-pair plan placed-pair census is incomplete")

        # Rebuild the arrays from immutable bytes-backed storage.  This
        # detaches caller aliases and prevents changing WRITEABLE after the
        # plan fingerprint has been established.
        representatives = np.frombuffer(
            representatives.tobytes(order="C"),
            dtype=np.int64,
        )
        offsets = np.frombuffer(offsets.tobytes(order="C"), dtype=np.int64)
        members = np.frombuffer(members.tobytes(order="C"), dtype=np.int64)
        multiplicities = np.frombuffer(
            multiplicities.tobytes(order="C"),
            dtype=np.int64,
        )
        object.__setattr__(self, "representative_indices", representatives)
        object.__setattr__(self, "orbit_offsets", offsets)
        object.__setattr__(self, "member_indices", members)
        object.__setattr__(self, "combined_multiplicities", multiplicities)
        object.__setattr__(self, "n_action_generators", generator_count)
        object.__setattr__(self, "action_fingerprint", action_fingerprint)
        object.__setattr__(self, "parent_fingerprint", parent.fingerprint)
        object.__setattr__(self, "fingerprint", self._build_fingerprint())

    def _build_fingerprint(self) -> str:
        digest = sha256()
        digest.update(self.schema.encode("ascii"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(self.parent_fingerprint))
        digest.update(bytes.fromhex(self.action_fingerprint))
        digest.update(int(self.n_action_generators).to_bytes(8, "big"))
        for array in (
            self.representative_indices,
            self.orbit_offsets,
            self.member_indices,
            self.combined_multiplicities,
        ):
            digest.update(np.asarray(array, dtype=">i8").tobytes())
        return digest.hexdigest()

    @property
    def n_representatives(self) -> int:
        return int(self.representative_indices.size)

    @property
    def n_translation_representatives(self) -> int:
        return self.translation_plan.n_representatives

    @property
    def n_cells(self) -> int:
        return self.translation_plan.n_cells

    @property
    def n_occupied(self) -> int:
        return self.translation_plan.n_occupied

    @property
    def n_placed_pairs(self) -> int:
        return self.translation_plan.n_placed_pairs

    @property
    def per_cell_weights(self) -> np.ndarray:
        return np.asarray(self.combined_multiplicities, dtype=float) / self.n_cells

    @property
    def payload_nbytes(self) -> int:
        return sum(
            int(array.nbytes)
            for array in (
                self.representative_indices,
                self.orbit_offsets,
                self.member_indices,
                self.combined_multiplicities,
            )
        )


def _validated_positive_integer(value: object, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"translation-pair plan {label} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"translation-pair plan {label} must be positive")
    return result


def _validated_pair_mesh(mesh: Sequence[int]) -> tuple[int, int, int]:
    try:
        values = tuple(mesh)
    except TypeError as exc:
        raise ValueError(
            "translation-pair plan mesh must contain three positive integers"
        ) from exc
    if len(values) != 3:
        raise ValueError(
            "translation-pair plan mesh must contain three positive integers"
        )
    return tuple(
        _validated_positive_integer(value, f"mesh[{axis}]")
        for axis, value in enumerate(values)
    )


def _checked_pair_counts(
    mesh: tuple[int, int, int],
    n_bands: int,
) -> tuple[int, int, int]:
    maximum = int(np.iinfo(np.int64).max)
    n_cells = prod(mesh)
    if n_cells > maximum:
        raise OverflowError("translation-pair plan cell count exceeds int64")
    if n_cells > maximum // n_bands:
        raise OverflowError("translation-pair plan occupied count exceeds int64")
    n_occupied = n_cells * n_bands
    if n_occupied > (2 * maximum) // (n_occupied + 1):
        raise OverflowError("translation-pair plan pair count exceeds int64")
    n_placed_pairs = n_occupied * (n_occupied + 1) // 2
    return n_cells, n_occupied, n_placed_pairs


def _two_torsion_count(mesh: tuple[int, int, int]) -> int:
    """Number of finite translations satisfying ``2 L = 0``."""

    return prod(2 if extent % 2 == 0 else 1 for extent in mesh)


def _translation_pair_representative_count(
    mesh: tuple[int, int, int],
    n_bands: int,
) -> int:
    n_cells = prod(mesh)
    two_torsion = _two_torsion_count(mesh)
    same_band = n_bands * (n_cells + two_torsion) // 2
    cross_band = n_bands * (n_bands - 1) // 2 * n_cells
    return same_band + cross_band


def build_translation_pair_plan(
    n_bands: int,
    mesh: Sequence[int],
) -> AICCM2026DevBTranslationPairPlan:
    """Build the compact exact finite-translation pair-orbit plan.

    The returned rows are canonical under simultaneous translation and
    unordered-pair exchange.  Their multiplicities sum to the full placed
    pair count ``N_occ (N_occ + 1) / 2`` without allocating that space.
    """

    mesh_tuple = _validated_pair_mesh(mesh)
    bands = _validated_positive_integer(n_bands, "n_bands")
    n_cells, _, _ = _checked_pair_counts(mesh_tuple, bands)
    n_representatives = _translation_pair_representative_count(
        mesh_tuple,
        bands,
    )
    maximum_index = int(np.iinfo(np.intp).max)
    payload_bytes_per_row = 6 * np.dtype(np.int64).itemsize
    if n_representatives > maximum_index // payload_bytes_per_row:
        raise OverflowError("translation-pair plan payload exceeds addressable memory")

    band_pairs = np.empty((n_representatives, 2), dtype=np.int64)
    displacements = np.empty((n_representatives, 3), dtype=np.int64)
    multiplicities = np.empty(n_representatives, dtype=np.int64)
    cursor = 0
    for band_i in range(bands):
        for band_j in range(band_i, bands):
            for displacement in product(*(range(value) for value in mesh_tuple)):
                inverse = tuple(
                    (-displacement[axis]) % mesh_tuple[axis]
                    for axis in range(3)
                )
                if band_i == band_j and displacement > inverse:
                    continue
                multiplicity = n_cells
                if (
                    band_i == band_j
                    and displacement != (0, 0, 0)
                    and displacement == inverse
                ):
                    multiplicity //= 2
                band_pairs[cursor] = (band_i, band_j)
                displacements[cursor] = displacement
                multiplicities[cursor] = multiplicity
                cursor += 1
    if cursor != n_representatives:
        raise RuntimeError("translation-pair plan construction broke its census")
    return AICCM2026DevBTranslationPairPlan(
        mesh=mesh_tuple,
        n_bands=bands,
        band_pairs=band_pairs,
        displacements=displacements,
        multiplicities=multiplicities,
    )


def _validated_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validated_point_action_generators(
    permutations: Iterable[Sequence[int] | np.ndarray],
    *,
    n_occupied: int,
) -> tuple[np.ndarray, ...]:
    """Return unique occupied permutations in deterministic byte order."""

    try:
        supplied = tuple(permutations)
    except TypeError as exc:
        raise ValueError(
            "point-pair plan occupied_permutations must be an iterable"
        ) from exc
    if not supplied:
        raise ValueError(
            "point-pair plan requires at least one occupied permutation"
        )
    expected = np.arange(n_occupied, dtype=np.int64)
    canonical: dict[bytes, np.ndarray] = {}
    for supplied_permutation in supplied:
        raw = np.asarray(supplied_permutation)
        if raw.dtype.kind not in "iu":
            raise ValueError("point-pair occupied mappings must contain integers")
        if raw.shape != (n_occupied,):
            raise ValueError("point-pair occupied permutation shape mismatch")
        maximum = int(np.iinfo(np.int64).max)
        if raw.size and (
            int(np.min(raw)) < 0 or int(np.max(raw)) > maximum
        ):
            raise ValueError("point-pair occupied mapping is out of range")
        permutation = np.ascontiguousarray(raw, dtype=np.int64)
        if not np.array_equal(np.sort(permutation), expected):
            raise ValueError("point-pair occupied mapping is not a permutation")
        key = permutation.astype(">i8", copy=False).tobytes()
        canonical[key] = permutation
    return tuple(canonical[key] for key in sorted(canonical))


def _cell_coordinates(
    cell_index: int,
    mesh: tuple[int, int, int],
) -> tuple[int, int, int]:
    plane = mesh[1] * mesh[2]
    coordinate_0, remainder = divmod(cell_index, plane)
    coordinate_1, coordinate_2 = divmod(remainder, mesh[2])
    return coordinate_0, coordinate_1, coordinate_2


def _cell_index(
    coordinates: tuple[int, int, int],
    mesh: tuple[int, int, int],
) -> int:
    return (
        (coordinates[0] * mesh[1] + coordinates[1]) * mesh[2]
        + coordinates[2]
    )


def _occupied_translation_generator(
    plan: AICCM2026DevBTranslationPairPlan,
    axis: int,
) -> np.ndarray:
    """Build one axis generator, rather than the full translation table."""

    permutation = np.empty(plan.n_occupied, dtype=np.int64)
    for cell_index in range(plan.n_cells):
        cell = _cell_coordinates(cell_index, plan.mesh)
        target = tuple(
            (cell[component] + (1 if component == axis else 0))
            % plan.mesh[component]
            for component in range(3)
        )
        target_start = _cell_index(target, plan.mesh) * plan.n_bands
        source_start = cell_index * plan.n_bands
        permutation[source_start : source_start + plan.n_bands] = (
            target_start + np.arange(plan.n_bands, dtype=np.int64)
        )
    return permutation


def _validate_point_action_normalizer(
    plan: AICCM2026DevBTranslationPairPlan,
    permutation: np.ndarray,
) -> None:
    """Require conjugation of every axis translation to be a translation."""

    inverse = np.empty(plan.n_occupied, dtype=np.int64)
    inverse[permutation] = np.arange(plan.n_occupied, dtype=np.int64)
    occupied_indices = np.arange(plan.n_occupied, dtype=np.int64)
    source_cells = occupied_indices // plan.n_bands
    source_bands = occupied_indices % plan.n_bands
    source_coordinates = np.asarray(
        [_cell_coordinates(int(index), plan.mesh) for index in source_cells],
        dtype=np.int64,
    )
    mesh_array = np.asarray(plan.mesh, dtype=np.int64)

    for axis in range(3):
        translation = _occupied_translation_generator(plan, axis)
        conjugated = permutation[translation[inverse]]
        target_cells = conjugated // plan.n_bands
        target_bands = conjugated % plan.n_bands
        target_coordinates = np.asarray(
            [_cell_coordinates(int(index), plan.mesh) for index in target_cells],
            dtype=np.int64,
        )
        displacement = (target_coordinates[0] - source_coordinates[0]) % mesh_array
        if not np.array_equal(target_bands, source_bands) or not np.all(
            (target_coordinates - source_coordinates) % mesh_array == displacement
        ):
            raise ValueError(
                "point-pair occupied action does not normalize the finite "
                "translation subgroup"
            )


def _canonical_translation_pair_row(
    plan: AICCM2026DevBTranslationPairPlan,
    occupied_i: int,
    occupied_j: int,
) -> tuple[int, int, int, int, int]:
    cell_i, band_i = divmod(occupied_i, plan.n_bands)
    cell_j, band_j = divmod(occupied_j, plan.n_bands)
    coordinates_i = _cell_coordinates(cell_i, plan.mesh)
    coordinates_j = _cell_coordinates(cell_j, plan.mesh)
    displacement = tuple(
        (coordinates_j[axis] - coordinates_i[axis]) % plan.mesh[axis]
        for axis in range(3)
    )
    if band_i > band_j:
        band_i, band_j = band_j, band_i
        displacement = tuple(
            (-displacement[axis]) % plan.mesh[axis]
            for axis in range(3)
        )
    if band_i == band_j:
        inverse = tuple(
            (-displacement[axis]) % plan.mesh[axis]
            for axis in range(3)
        )
        displacement = min(displacement, inverse)
    return band_i, band_j, *displacement


def _induced_translation_row_action(
    plan: AICCM2026DevBTranslationPairPlan,
    permutation: np.ndarray,
    row_lookup: dict[tuple[int, int, int, int, int], int],
) -> np.ndarray:
    action = np.empty(plan.n_representatives, dtype=np.int64)
    for row_index in range(plan.n_representatives):
        band_i, band_j = (int(value) for value in plan.band_pairs[row_index])
        displacement = tuple(int(value) for value in plan.displacements[row_index])
        occupied_i = band_i
        occupied_j = _cell_index(displacement, plan.mesh) * plan.n_bands + band_j
        key = _canonical_translation_pair_row(
            plan,
            int(permutation[occupied_i]),
            int(permutation[occupied_j]),
        )
        try:
            action[row_index] = row_lookup[key]
        except KeyError as exc:
            raise RuntimeError(
                "point-pair action did not induce a parent-plan row"
            ) from exc
    if not np.array_equal(
        np.sort(action),
        np.arange(plan.n_representatives, dtype=np.int64),
    ):
        raise RuntimeError("point-pair action on parent rows is not a permutation")
    if np.any(plan.multiplicities[action] != plan.multiplicities):
        raise RuntimeError("point-pair action changed a translation stabilizer")
    return action


def _point_action_fingerprint(
    plan: AICCM2026DevBTranslationPairPlan,
    generators: tuple[np.ndarray, ...],
) -> str:
    digest = sha256()
    digest.update(b"vibeqc.aiccm2026dev-b.occupied-point-action/v1\0")
    digest.update(bytes.fromhex(plan.fingerprint))
    digest.update(len(generators).to_bytes(8, "big"))
    for generator in generators:
        digest.update(np.asarray(generator, dtype=">i8").tobytes())
    return digest.hexdigest()


def build_point_pair_plan(
    translation_plan: AICCM2026DevBTranslationPairPlan,
    occupied_permutations: Iterable[Sequence[int] | np.ndarray],
) -> AICCM2026DevBPointPairPlan:
    """Build an exact compact point quotient of D123 pair-plan rows.

    Each supplied support permutation maps a source occupied index to its
    target and must normalize the finite translation subgroup.  The induced
    support action is evaluated on D123 row anchors, never on their
    placed-pair members.  Orbit closure over those exact row permutations
    produces a second compact plan whose multiplicities retain the complete
    placed-pair census.

    The API is deliberately permutation-support-only.  It neither accepts nor
    fingerprints the signs/phases of a physical monomial representation.
    General unitary occupied actions cannot map an individual pair row to
    another row and are rejected by construction rather than approximated
    geometrically.
    """

    if not isinstance(translation_plan, AICCM2026DevBTranslationPairPlan):
        raise ValueError(
            "build_point_pair_plan requires an "
            "AICCM2026DevBTranslationPairPlan"
        )
    generators = _validated_point_action_generators(
        occupied_permutations,
        n_occupied=translation_plan.n_occupied,
    )
    row_lookup = {
        (
            int(translation_plan.band_pairs[row, 0]),
            int(translation_plan.band_pairs[row, 1]),
            *(int(value) for value in translation_plan.displacements[row]),
        ): row
        for row in range(translation_plan.n_representatives)
    }
    row_actions: list[np.ndarray] = []
    for generator in generators:
        _validate_point_action_normalizer(translation_plan, generator)
        row_actions.append(
            _induced_translation_row_action(
                translation_plan,
                generator,
                row_lookup,
            )
        )

    visited = np.zeros(translation_plan.n_representatives, dtype=bool)
    orbits: list[tuple[int, ...]] = []
    for seed in range(translation_plan.n_representatives):
        if visited[seed]:
            continue
        members = {seed}
        queue = [seed]
        cursor = 0
        while cursor < len(queue):
            member = queue[cursor]
            cursor += 1
            for action in row_actions:
                mapped = int(action[member])
                if mapped not in members:
                    members.add(mapped)
                    queue.append(mapped)
        orbit = tuple(sorted(members))
        visited[np.asarray(orbit, dtype=np.int64)] = True
        orbits.append(orbit)
    if not bool(np.all(visited)):
        raise RuntimeError("point-pair orbit closure broke its parent-row census")

    representative_indices = np.asarray(
        [orbit[0] for orbit in orbits],
        dtype=np.int64,
    )
    orbit_offsets = np.empty(len(orbits) + 1, dtype=np.int64)
    orbit_offsets[0] = 0
    for index, orbit in enumerate(orbits, start=1):
        orbit_offsets[index] = orbit_offsets[index - 1] + len(orbit)
    member_indices = np.asarray(
        [member for orbit in orbits for member in orbit],
        dtype=np.int64,
    )
    combined_multiplicities = np.asarray(
        [
            sum(int(translation_plan.multiplicities[member]) for member in orbit)
            for orbit in orbits
        ],
        dtype=np.int64,
    )
    return AICCM2026DevBPointPairPlan._from_payload(
        translation_plan=translation_plan,
        representative_indices=representative_indices,
        orbit_offsets=orbit_offsets,
        member_indices=member_indices,
        combined_multiplicities=combined_multiplicities,
        n_action_generators=len(generators),
        action_fingerprint=_point_action_fingerprint(
            translation_plan,
            generators,
        ),
    )


def _validate_occupied_metric(
    occupied: np.ndarray,
    overlap: np.ndarray,
    *,
    tolerance: float,
) -> None:
    metric = occupied.conj().T @ overlap @ occupied
    error = float(np.max(np.abs(metric - np.eye(metric.shape[0]))))
    if error > tolerance:
        raise ValueError(
            "AICCM2026DEV-B PAOs require S-orthonormal occupied orbitals; "
            f"maximum error is {error:.3e}"
        )


def projected_atomic_orbitals(
    occupied_coefficients: np.ndarray,
    overlap: np.ndarray,
    fock: np.ndarray,
    *,
    ao_indices: Sequence[int] | np.ndarray | None = None,
    linear_dependence_threshold: float = 1e-8,
    occupied_tolerance: float = 1e-8,
) -> AICCM2026DevBPAOSpace:
    """Project occupieds from selected AOs and canonically orthogonalize.

    ``ao_indices`` is the union of the atom-centred AOs in a pair domain.
    Passing ``None`` selects the full AO space and provides the exact virtual
    limit.  Eigenvalues are retained relative to the largest PAO Gram
    eigenvalue, which makes the threshold invariant to a uniform AO scaling.
    """

    occupied = np.asarray(occupied_coefficients)
    S = np.asarray(overlap)
    F = np.asarray(fock)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("AICCM2026DEV-B PAO overlap must be square")
    if occupied.ndim != 2 or occupied.shape[0] != S.shape[0]:
        raise ValueError("AICCM2026DEV-B PAO occupied/overlap shape mismatch")
    if F.shape != S.shape:
        raise ValueError("AICCM2026DEV-B PAO Fock/overlap shape mismatch")
    if linear_dependence_threshold <= 0.0:
        raise ValueError("PAO linear-dependence threshold must be positive")
    _validate_occupied_metric(
        occupied,
        S,
        tolerance=occupied_tolerance,
    )

    n_ao = S.shape[0]
    if ao_indices is None:
        selected = np.arange(n_ao, dtype=int)
    else:
        selected = np.unique(np.asarray(ao_indices, dtype=int))
        if selected.ndim != 1 or selected.size == 0:
            raise ValueError("AICCM2026DEV-B PAO domain must select at least one AO")
        if selected[0] < 0 or selected[-1] >= n_ao:
            raise ValueError("AICCM2026DEV-B PAO domain AO index out of range")

    projector = np.eye(n_ao, dtype=np.result_type(occupied, S)) - (
        occupied @ (occupied.conj().T @ S)
    )
    raw = projector[:, selected]
    gram = raw.conj().T @ S @ raw
    gram = 0.5 * (gram + gram.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.asarray(eigenvalues[order], dtype=float)
    eigenvectors = eigenvectors[:, order]
    scale = max(float(eigenvalues[0]), np.finfo(float).eps)
    keep = eigenvalues > linear_dependence_threshold * scale
    if not np.any(keep):
        raise RuntimeError("AICCM2026DEV-B PAO domain has zero virtual rank")

    coefficients = raw @ (
        eigenvectors[:, keep] / np.sqrt(eigenvalues[keep])[None, :]
    )
    pao_metric = coefficients.conj().T @ S @ coefficients
    orthogonality_error = float(
        np.max(np.abs(pao_metric - np.eye(pao_metric.shape[0])))
    )
    leakage = float(np.max(np.abs(occupied.conj().T @ S @ coefficients)))
    if max(orthogonality_error, leakage) > 10.0 * occupied_tolerance:
        raise RuntimeError(
            "AICCM2026DEV-B PAO invariant failure: "
            f"orthonormality={orthogonality_error:.3e}, "
            f"occupied leakage={leakage:.3e}"
        )

    fock_pao = coefficients.conj().T @ F @ coefficients
    fock_pao = 0.5 * (fock_pao + fock_pao.conj().T)
    orbital_energies, rotation = np.linalg.eigh(fock_pao)
    coefficients = coefficients @ rotation
    return AICCM2026DevBPAOSpace(
        coefficients=np.asarray(np.real_if_close(coefficients, tol=1000)),
        orbital_energies=np.asarray(orbital_energies, dtype=float),
        projector=projector,
        gram_eigenvalues=eigenvalues,
        source_ao_indices=selected,
        orthonormality_error=orthogonality_error,
        occupied_leakage_error=leakage,
        discarded_rank=int(np.count_nonzero(~keep)),
    )


def build_pair_natural_orbitals(
    pair_amplitudes: np.ndarray,
    virtual_coefficients: np.ndarray,
    *,
    diagonal_pair: bool,
    occupation_threshold: float,
    fock: np.ndarray | None = None,
    minimum_rank: int = 1,
    negative_occupation_tolerance: float = 1e-10,
) -> AICCM2026DevBPNOSpace:
    """Diagonalize the Hermitian MP2 model pair density.

    A zero threshold retains the complete supplied virtual space, including
    exactly zero-occupation directions, and is therefore the algebraic
    no-truncation limit.  Positive thresholds retain occupations strictly
    larger than the threshold, subject to ``minimum_rank``.
    """

    amplitudes = np.asarray(pair_amplitudes)
    virtuals = np.asarray(virtual_coefficients)
    if amplitudes.ndim != 2 or amplitudes.shape[0] != amplitudes.shape[1]:
        raise ValueError("AICCM2026DEV-B pair amplitudes must be square")
    if virtuals.ndim != 2 or virtuals.shape[1] != amplitudes.shape[0]:
        raise ValueError("AICCM2026DEV-B PNO virtual/amplitude shape mismatch")
    if occupation_threshold < 0.0:
        raise ValueError("PNO occupation threshold must be non-negative")
    if minimum_rank < 0 or minimum_rank > amplitudes.shape[0]:
        raise ValueError("PNO minimum rank is out of range")

    adjoint = amplitudes.conj().T
    divisor = 2.0 if diagonal_pair else 1.0
    pair_density = (amplitudes @ adjoint + adjoint @ amplitudes) / divisor
    hermiticity_error = float(
        np.max(np.abs(pair_density - pair_density.conj().T))
    )
    pair_density = 0.5 * (pair_density + pair_density.conj().T)
    occupations, eigenvectors = np.linalg.eigh(pair_density)
    order = np.argsort(occupations)[::-1]
    occupations = np.asarray(occupations[order], dtype=float)
    eigenvectors = eigenvectors[:, order]
    minimum_occupation = float(occupations[-1])
    if minimum_occupation < -negative_occupation_tolerance:
        raise RuntimeError(
            "AICCM2026DEV-B pair density is not positive semidefinite; "
            f"minimum occupation is {minimum_occupation:.3e}"
        )
    occupations = np.maximum(occupations, 0.0)

    if occupation_threshold == 0.0:
        keep = np.ones_like(occupations, dtype=bool)
    else:
        keep = occupations > occupation_threshold
    if int(np.count_nonzero(keep)) < minimum_rank:
        keep[:minimum_rank] = True
    retained_vectors = eigenvectors[:, keep]
    coefficients = virtuals @ retained_vectors

    orbital_energies: np.ndarray | None = None
    if fock is not None:
        F = np.asarray(fock)
        if F.shape != (virtuals.shape[0], virtuals.shape[0]):
            raise ValueError("AICCM2026DEV-B PNO Fock matrix shape mismatch")
        fock_pno = coefficients.conj().T @ F @ coefficients
        fock_pno = 0.5 * (fock_pno + fock_pno.conj().T)
        orbital_energies, rotation = np.linalg.eigh(fock_pno)
        coefficients = coefficients @ rotation

    return AICCM2026DevBPNOSpace(
        coefficients=np.asarray(np.real_if_close(coefficients, tol=1000)),
        occupations=occupations,
        retained_occupations=occupations[keep],
        orbital_energies=(
            None
            if orbital_energies is None
            else np.asarray(orbital_energies, dtype=float)
        ),
        retained_rank=int(np.count_nonzero(keep)),
        discarded_occupation=float(np.sum(occupations[~keep])),
        hermiticity_error=hermiticity_error,
        minimum_occupation=minimum_occupation,
    )


def _full_space_mp2_energy_python(
    factors_ov: np.ndarray,
    eps_occ: np.ndarray,
    eps_vir: np.ndarray,
    denominator_tolerance: float,
) -> float:
    """Reference tensor expression for the complete-domain MP2 audit."""

    integrals = np.einsum(
        "Pia,Pjb->ijab",
        factors_ov,
        factors_ov,
        optimize=True,
    )
    denominator = (
        eps_occ[:, None, None, None]
        + eps_occ[None, :, None, None]
        - eps_vir[None, None, :, None]
        - eps_vir[None, None, None, :]
    )
    if np.any(denominator >= -denominator_tolerance):
        raise RuntimeError(
            "AICCM2026DEV-B MP2 audit encountered a non-negative denominator"
        )
    amplitudes = integrals / denominator
    return float(
        np.einsum(
            "ijab,ijab->",
            amplitudes,
            2.0 * integrals - integrals.swapaxes(2, 3),
            optimize=True,
        )
    )


def _full_space_mp2_energy_native(
    factors: np.ndarray,
    occupied_canonical: np.ndarray,
    virtual_canonical: np.ndarray,
    eps_occ: np.ndarray,
    eps_vir: np.ndarray,
    denominator_tolerance: float,
) -> float | None:
    """Return native streaming MP2 audit energy, or ``None`` if unavailable."""

    try:
        from ... import _vibeqc_core
    except ImportError:
        return None
    transform = getattr(_vibeqc_core, "aiccm2026dev_b_3index_mo_transform", None)
    kernel = getattr(_vibeqc_core, "aiccm2026dev_b_real_mp2_energy_from_lov", None)
    if transform is None or kernel is None:
        return None

    factors_ov = transform(
        np.ascontiguousarray(factors, dtype=float),
        np.ascontiguousarray(occupied_canonical, dtype=float),
        np.ascontiguousarray(virtual_canonical, dtype=float),
    )
    try:
        return float(
            kernel(
                np.ascontiguousarray(factors_ov, dtype=float),
                np.ascontiguousarray(eps_occ, dtype=float),
                np.ascontiguousarray(eps_vir, dtype=float),
                float(denominator_tolerance),
            )
        )
    finally:
        del factors_ov


def full_space_noncanonical_mp2_energy(
    occupied_coefficients: np.ndarray,
    virtual_coefficients: np.ndarray,
    fock: np.ndarray,
    three_center_factors: np.ndarray,
    *,
    denominator_tolerance: float = 1e-12,
) -> float:
    """Return the closed-shell DF-MP2 energy in rotation-invariant form.

    The occupied and virtual inputs may be any real orthonormal gauges.  We
    diagonalize the two Fock blocks, rotate the fitted three-center factors,
    and evaluate the ordinary ordered-pair MP2 expression.  This is the
    complete-domain, zero-PNO-threshold limit used to audit the iterative
    local-pair equations; it is not used for a truncated calculation.
    """

    occupied = np.asarray(occupied_coefficients)
    virtual = np.asarray(virtual_coefficients)
    F = np.asarray(fock)
    factors = np.asarray(three_center_factors)
    if any(np.iscomplexobj(value) for value in (occupied, virtual, F, factors)):
        maximum_imaginary = max(
            float(np.max(np.abs(np.asarray(value).imag)))
            for value in (occupied, virtual, F, factors)
        )
        if maximum_imaginary > 1e-10:
            raise ValueError(
                "AICCM2026DEV-B real-torus MP2 audit requires real tensors"
            )
        occupied = occupied.real
        virtual = virtual.real
        F = F.real
        factors = factors.real
    if occupied.ndim != 2 or virtual.ndim != 2:
        raise ValueError("AICCM2026DEV-B MP2 audit coefficients must be matrices")
    if occupied.shape[0] != virtual.shape[0] or F.shape != (
        occupied.shape[0],
        occupied.shape[0],
    ):
        raise ValueError("AICCM2026DEV-B MP2 audit one-particle shape mismatch")
    if factors.ndim != 3 or factors.shape[1:] != F.shape:
        raise ValueError("AICCM2026DEV-B MP2 audit factor shape mismatch")

    f_occ = occupied.T @ F @ occupied
    f_vir = virtual.T @ F @ virtual
    eps_occ, rotation_occ = np.linalg.eigh(0.5 * (f_occ + f_occ.T))
    eps_vir, rotation_vir = np.linalg.eigh(0.5 * (f_vir + f_vir.T))
    occupied_canonical = occupied @ rotation_occ
    virtual_canonical = virtual @ rotation_vir
    native = _full_space_mp2_energy_native(
        factors,
        occupied_canonical,
        virtual_canonical,
        eps_occ,
        eps_vir,
        denominator_tolerance,
    )
    if native is not None:
        return native

    factors_ov = np.einsum(
        "mi,Pmn,na->Pia",
        occupied_canonical,
        factors,
        virtual_canonical,
        optimize=True,
    )
    return _full_space_mp2_energy_python(
        factors_ov,
        eps_occ,
        eps_vir,
        denominator_tolerance,
    )


def translation_permutations(
    n_bands: int,
    mesh: Sequence[int],
) -> tuple[np.ndarray, ...]:
    """Return occupied-index permutations for every finite translation.

    Occupied indices use ``index = cell_index * n_bands + band`` with cells
    in lexicographic product order.  Each permutation acts simultaneously on
    both indices of an occupied pair.
    """

    mesh_tuple = tuple(int(value) for value in mesh)
    if len(mesh_tuple) != 3 or any(value < 1 for value in mesh_tuple):
        raise ValueError("AICCM2026DEV-B pair mesh must contain three positive values")
    if n_bands < 1:
        raise ValueError("AICCM2026DEV-B pair orbit needs at least one band")
    native = _translation_permutations_native(n_bands, mesh_tuple)
    if native is not None:
        return native
    return _translation_permutations_python(n_bands, mesh_tuple)


def _translation_permutations_native(
    n_bands: int,
    mesh: tuple[int, int, int],
) -> tuple[np.ndarray, ...] | None:
    """Return native translation permutations, or ``None`` when unavailable."""

    try:
        from ... import _vibeqc_core
    except ImportError:
        return None
    kernel = getattr(_vibeqc_core, "aiccm2026dev_b_translation_permutations", None)
    if kernel is None:
        return None
    matrix = np.asarray(
        kernel(
            int(n_bands),
            np.ascontiguousarray(mesh, dtype=np.int64),
        ),
        dtype=int,
    )
    expected = int(np.prod(mesh))
    if matrix.shape != (expected, expected * int(n_bands)):
        raise RuntimeError(
            "AICCM2026DEV-B native translation-permutation kernel broke shape"
        )
    return tuple(np.array(row, dtype=int, copy=True) for row in matrix)


def _translation_permutations_python(
    n_bands: int,
    mesh_tuple: tuple[int, int, int],
) -> tuple[np.ndarray, ...]:
    """Python oracle for finite-translation occupied-index permutations."""

    cells = tuple(product(*(range(value) for value in mesh_tuple)))
    cell_index = {cell: index for index, cell in enumerate(cells)}
    permutations: list[np.ndarray] = []
    for shift in cells:
        permutation = np.empty(len(cells) * n_bands, dtype=int)
        for source_index, cell in enumerate(cells):
            target = tuple(
                (cell[axis] + shift[axis]) % mesh_tuple[axis]
                for axis in range(3)
            )
            target_index = cell_index[target]
            for band in range(n_bands):
                permutation[source_index * n_bands + band] = (
                    target_index * n_bands + band
                )
        permutations.append(permutation)
    return tuple(permutations)


def _canonical_pair(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i <= j else (j, i)


def _validate_permutations(
    permutations: Iterable[Sequence[int] | np.ndarray],
) -> tuple[np.ndarray, ...]:
    output = tuple(np.asarray(value, dtype=int) for value in permutations)
    if not output:
        raise ValueError("AICCM2026DEV-B pair orbits require at least one permutation")
    n_orbitals = int(output[0].size)
    expected = np.arange(n_orbitals)
    for permutation in output:
        if permutation.shape != (n_orbitals,):
            raise ValueError("AICCM2026DEV-B pair permutation shape mismatch")
        if not np.array_equal(np.sort(permutation), expected):
            raise ValueError("AICCM2026DEV-B pair mapping is not a permutation")
    return output


def _pair_orbits_from_native(
    permutations: tuple[np.ndarray, ...],
    *,
    n_cells: int,
) -> tuple[AICCM2026DevBPairOrbit, ...] | None:
    """Return native pair orbits, or ``None`` when the extension is older."""

    try:
        from ... import _vibeqc_core
    except ImportError:
        return None
    kernel = getattr(_vibeqc_core, "aiccm2026dev_b_pair_orbits", None)
    if kernel is None:
        return None

    permutation_matrix = np.ascontiguousarray(np.vstack(permutations), dtype=np.int64)
    representatives, offsets, members = kernel(permutation_matrix, int(n_cells))
    reps = np.asarray(representatives, dtype=int)
    starts = np.asarray(offsets, dtype=int)
    flat_members = np.asarray(members, dtype=int)
    if reps.ndim != 2 or reps.shape[1] != 2:
        raise RuntimeError("AICCM2026DEV-B native pair-orbit representatives broke shape")
    if starts.shape != (reps.shape[0] + 1,):
        raise RuntimeError("AICCM2026DEV-B native pair-orbit offsets broke shape")
    if flat_members.ndim != 2 or flat_members.shape[1] != 2:
        raise RuntimeError("AICCM2026DEV-B native pair-orbit members broke shape")

    orbits: list[AICCM2026DevBPairOrbit] = []
    for orbit_index, representative in enumerate(reps):
        start = int(starts[orbit_index])
        stop = int(starts[orbit_index + 1])
        if start < 0 or stop < start or stop > flat_members.shape[0]:
            raise RuntimeError("AICCM2026DEV-B native pair-orbit offsets are invalid")
        orbit_members = tuple(
            (int(i), int(j))
            for i, j in flat_members[start:stop]
        )
        if not orbit_members:
            raise RuntimeError("AICCM2026DEV-B native pair orbit is empty")
        orbits.append(
            AICCM2026DevBPairOrbit(
                representative=(int(representative[0]), int(representative[1])),
                members=orbit_members,
                n_cells=n_cells,
            )
        )
    return tuple(orbits)


def _enumerate_pair_orbits_python(
    permutations: tuple[np.ndarray, ...],
    *,
    n_cells: int,
) -> tuple[AICCM2026DevBPairOrbit, ...]:
    if n_cells < 1:
        raise ValueError("AICCM2026DEV-B pair orbit n_cells must be positive")
    n_orbitals = int(permutations[0].size)
    all_pairs = {
        (i, j)
        for i in range(n_orbitals)
        for j in range(i, n_orbitals)
    }
    unassigned = set(all_pairs)
    orbits: list[AICCM2026DevBPairOrbit] = []
    while unassigned:
        seed = min(unassigned)
        members = {seed}
        frontier = [seed]
        while frontier:
            i, j = frontier.pop()
            for permutation in permutations:
                mapped = _canonical_pair(
                    int(permutation[i]),
                    int(permutation[j]),
                )
                if mapped not in members:
                    members.add(mapped)
                    frontier.append(mapped)
        if not members <= all_pairs:
            raise RuntimeError("AICCM2026DEV-B pair orbit escaped pair space")
        unassigned -= members
        ordered = tuple(sorted(members))
        orbits.append(
            AICCM2026DevBPairOrbit(
                representative=ordered[0],
                members=ordered,
                n_cells=n_cells,
            )
        )
    flattened = [member for orbit in orbits for member in orbit.members]
    if len(flattened) != len(all_pairs) or set(flattened) != all_pairs:
        raise RuntimeError("AICCM2026DEV-B pair orbits do not partition pair space")
    return tuple(sorted(orbits, key=lambda orbit: orbit.representative))


def enumerate_pair_orbits(
    permutations: Iterable[Sequence[int] | np.ndarray],
    *,
    n_cells: int,
) -> tuple[AICCM2026DevBPairOrbit, ...]:
    """Partition all unordered occupied pairs under supplied generators.

    The breadth-first closure makes the result correct whether callers pass
    the full finite symmetry group or only a generating set.  The hard gate
    verifies that every unordered pair occurs in exactly one orbit.  Production
    uses the private native χ-CCM kernel when it is present; the Python closure
    remains as the reference path for source checkouts with an older extension.
    """

    group_generators = _validate_permutations(permutations)
    native = _pair_orbits_from_native(group_generators, n_cells=n_cells)
    if native is not None:
        return native
    return _enumerate_pair_orbits_python(group_generators, n_cells=n_cells)
