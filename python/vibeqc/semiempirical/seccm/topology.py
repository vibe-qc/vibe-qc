"""Hamiltonian-independent cyclic-cluster image topology.

This module generalizes the proven MSINDO ``WEIGHT_OLD`` Wigner-Seitz
ownership rule without attaching its image geometry to any particular
semiempirical Hamiltonian. Ordinary reduced cells retain the validated
MSINDO records and fractional ownership. Arbitrary accepted skew bases are
Minkowski-reduced before the closest-image search, and the resulting image
labels are mapped back through the integer unimodular transform. The record
contract is therefore independent of the basis used to represent the same
cyclic lattice.

The geometric closest-image topology may be bound separately to a finite
Born-von Karman lattice group and to a complete static-site translation
action.  Method-family Hamiltonians and dynamic electronic-state covariance
remain outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from decimal import Decimal
from enum import Enum
import hashlib
from itertools import product
import json
import math
import re
from typing import Literal, Never
import unicodedata

import numpy as np

from vibeqc.molecule import ANGSTROM_TO_BOHR

ImageShellLabel = tuple[int, int, int]
FiniteTranslationLabel = tuple[int, int, int]
AtomEquivalenceKey = tuple[str, ...]

_BUILDER_PROVENANCE = "seccm-exact-ws-minkowski-v2"
_TOPOLOGY_SCHEMA = "seccm-ws-topology-v1"
_GEOMETRY_SCHEMA = "seccm-reference-geometry-v1"
_FINITE_GROUP_SCHEMA = "seccm-finite-bvk-group-v1"
_FINITE_GROUP_PROVENANCE = "explicit-finite-bvk-v1"
_ATOM_ACTION_SCHEMA = "seccm-complete-static-site-action-v1"
_ATOM_ACTION_PROVENANCE = "complete-static-site-action-v1"
_VERSIONED_SCHEMA_PATTERN = re.compile(r".+-v[1-9][0-9]*$")
_SUPPORTED_LENGTH_UNITS = frozenset({"angstrom", "bohr"})
_REDUCED_IMAGE_SEARCH = 2
_WS_SCORE_ATOL = 1.0e-12
# Equidistant-image tie tolerance, a PHYSICAL length: 1.0e-5 bohr, converted
# to the caller's length unit through _bohr_per_length_unit (GitLab #669; an
# angstrom-tagged topology used to reuse the raw number and so judged ties in
# a window 1.89x wider than the bohr routes).  Every vibe-qc production route
# builds in bohr.  Bredow, Geudtner & Jug, J. Comput. Chem. 22, 89 (2001),
# pp. 90-91: in bulk MgO and NiO "there are always several neighbors N at the
# borders of the Wigner-Seitz unit cell around atom I with the same distance",
# and keeping only one of them "would lead to a non-symmetric region with a
# dipole moment", so all of them are retained with Evjen weight 1/n (also
# Peintinger & Bredow, J. Comput. Chem. 35, 839 (2014), Theory, two-center
# weights).  "The same distance" must be judged far above the precision of any
# real input geometry and far below any physical distinction: unit conversions
# with different CODATA constants disagree at 4.4e-10 relative (5e-9 bohr on
# MgO 2x2x2), a six-decimal Angstrom file at 1e-6 bohr.  MSINDO neighbors.f
# shares every image whose distance to the central atom is within 1.0E-5 bohr
# of the minimum; the pure 256-eps roundoff guard that replaced that rule in
# 65f6b0048 broke the 4-, 2- and 6-fold ties of MgO 2x2x2 at a 5e-9 bohr
# mismatch and moved the MSINDO SECCM energy by 6e-2 Ha (issues #592, #593).
# The native builder (seccm/topology.hpp, kEquidistantImageTolerance) carries
# the same constant; the two must stay bit-identical.
_EQUIDISTANT_IMAGE_TOLERANCE = 1.0e-5
_REVERSAL_DIAGNOSTICS = frozenset(
    {
        "missing_reverse_image",
        "ambiguous_reverse_image",
        "nonopposite_reverse_label",
        "inconsistent_reverse_displacement",
        "inconsistent_reverse_ownership",
        "duplicate_discrete_record",
    }
)


class SECCMTopologyError(ValueError):
    """Base error for malformed or unsupported SECCM topology operations."""


class SECCMTopologyProvenanceError(SECCMTopologyError):
    """Raised when an operation needs topology-builder provenance."""


class SECCMTopologyChangedError(SECCMTopologyError):
    """Raised when a fixed-topology derivative batch leaves its trust region."""


class TopologyDiagnosticCode(str, Enum):
    """Stable serialization values for SECCM topology diagnostics."""

    INVALID_TOTAL_ORIGIN_OWNERSHIP = "invalid_total_origin_ownership"
    MISSING_REVERSE_IMAGE = "missing_reverse_image"
    AMBIGUOUS_REVERSE_IMAGE = "ambiguous_reverse_image"
    NONOPPOSITE_REVERSE_LABEL = "nonopposite_reverse_label"
    INCONSISTENT_REVERSE_DISPLACEMENT = "inconsistent_reverse_displacement"
    INCONSISTENT_REVERSE_OWNERSHIP = "inconsistent_reverse_ownership"
    DUPLICATE_DISCRETE_RECORD = "duplicate_discrete_record"
    NEAR_SWITCHING_SURFACE = "near_switching_surface"
    MISSING_BUILDER_PROVENANCE = "missing_builder_provenance"
    FINITE_GROUP_DIMENSION_MISMATCH = "finite_group_dimension_mismatch"
    INVALID_PRIMITIVE_VECTORS = "invalid_primitive_vectors"
    INVALID_REPLICAS = "invalid_replicas"
    CYCLIC_TRANSLATION_MISMATCH = "cyclic_translation_mismatch"
    NONCANONICAL_CELL_LABEL = "noncanonical_cell_label"
    NONCONTIGUOUS_PRIMITIVE_SITE_IDS = (
        "noncontiguous_primitive_site_ids"
    )
    MISSING_CELL_COPY = "missing_cell_copy"
    DUPLICATE_CELL_COPY = "duplicate_cell_copy"
    COORDINATE_COVARIANCE_MISMATCH = "coordinate_covariance_mismatch"
    ATOM_ACTION_NOT_BOUND = "atom_action_not_bound"
    CURRENT_GEOMETRY_NOT_TRANSLATION_COVARIANT = (
        "current_geometry_not_translation_covariant"
    )
    MISSING_TRANSLATED_RECORD = "missing_translated_record"
    AMBIGUOUS_TRANSLATED_RECORD = "ambiguous_translated_record"
    TRANSLATED_RECORD_OWNERSHIP_MISMATCH = (
        "translated_record_ownership_mismatch"
    )
    TRANSLATED_RECORD_DISPLACEMENT_MISMATCH = (
        "translated_record_displacement_mismatch"
    )
    MISSING_ATOM_EQUIVALENCE_KEY = "missing_atom_equivalence_key"
    INVALID_ATOM_EQUIVALENCE_KEY = "invalid_atom_equivalence_key"
    ATOM_EQUIVALENCE_KEY_MISMATCH = "atom_equivalence_key_mismatch"
    INVALID_EQUIVALENCE_KEY_SCHEMA = "invalid_equivalence_key_schema"


@dataclass(frozen=True)
class TopologyDiagnostic:
    """One immutable topology diagnostic with optional record context."""

    code: TopologyDiagnosticCode
    message: str
    central: int | None = None
    record_index: int | None = None
    origin: int | None = None


@dataclass(frozen=True)
class SECCMFiniteGroup:
    """Finite Born-von Karman translation group for one cyclic cluster."""

    primitive_vectors: tuple[np.ndarray, ...]
    replicas: tuple[int, int, int]
    geometry_tolerance: float
    length_unit: str
    provenance: str = _FINITE_GROUP_PROVENANCE

    def __post_init__(self) -> None:
        primitive_vectors = _validated_primitive_vectors(
            self.primitive_vectors,
            len(self.primitive_vectors),
            self.geometry_tolerance,
        )
        replicas = _validated_replicas(
            self.replicas, len(primitive_vectors)
        )
        length_unit = _normalized_length_unit(self.length_unit)
        if self.provenance != _FINITE_GROUP_PROVENANCE:
            raise SECCMTopologyProvenanceError(
                "unsupported SECCM finite-group provenance"
            )
        object.__setattr__(self, "primitive_vectors", primitive_vectors)
        object.__setattr__(self, "replicas", replicas)
        object.__setattr__(
            self, "geometry_tolerance", float(self.geometry_tolerance)
        )
        object.__setattr__(self, "length_unit", length_unit)

    @property
    def dimensionality(self) -> int:
        return len(self.primitive_vectors)

    @property
    def order(self) -> int:
        return math.prod(self.replicas)

    @property
    def labels(self) -> tuple[FiniteTranslationLabel, ...]:
        """Return canonical group elements in deterministic lexicographic order."""
        return tuple(
            product(*(range(replica) for replica in self.replicas))
        )

    def add(
        self,
        left: FiniteTranslationLabel,
        right: FiniteTranslationLabel,
    ) -> FiniteTranslationLabel:
        left = self.require_canonical_label(left)
        right = self.require_canonical_label(right)
        return tuple(
            (left[axis] + right[axis]) % self.replicas[axis]
            for axis in range(3)
        )

    def subtract(
        self,
        left: FiniteTranslationLabel,
        right: FiniteTranslationLabel,
    ) -> FiniteTranslationLabel:
        left = self.require_canonical_label(left)
        right = self.require_canonical_label(right)
        return tuple(
            (left[axis] - right[axis]) % self.replicas[axis]
            for axis in range(3)
        )

    def inverse(
        self, label: FiniteTranslationLabel
    ) -> FiniteTranslationLabel:
        label = self.require_canonical_label(label)
        return tuple(
            (-label[axis]) % self.replicas[axis] for axis in range(3)
        )

    def require_canonical_label(
        self, label: FiniteTranslationLabel
    ) -> FiniteTranslationLabel:
        try:
            values = tuple(label)
        except TypeError as exc:
            _raise_topology_error(
                TopologyDiagnosticCode.NONCANONICAL_CELL_LABEL,
                "finite translation label must contain exactly three integers",
                cause=exc,
            )
        if len(values) != 3 or any(not _is_strict_integer(v) for v in values):
            _raise_topology_error(
                TopologyDiagnosticCode.NONCANONICAL_CELL_LABEL,
                "finite translation label must contain exactly three integers",
            )
        canonical = tuple(int(value) for value in values)
        if any(
            value < 0 or value >= self.replicas[axis]
            for axis, value in enumerate(canonical)
        ):
            _raise_topology_error(
                TopologyDiagnosticCode.NONCANONICAL_CELL_LABEL,
                "finite translation label is outside its canonical range",
            )
        return canonical

    def signed_representatives(
        self, label: FiniteTranslationLabel
    ) -> tuple[FiniteTranslationLabel, ...]:
        """Return geometric signed views without duplicating group elements."""
        label = self.require_canonical_label(label)
        axis_representatives: list[tuple[int, ...]] = []
        for value, replica in zip(label, self.replicas, strict=True):
            if replica % 2 == 0 and value == replica // 2:
                axis_representatives.append((-value, value))
            elif 2 * value > replica:
                axis_representatives.append((value - replica,))
            else:
                axis_representatives.append((value,))
        return tuple(product(*axis_representatives))

    def _identity_payload(self) -> dict:
        return {
            "schema": _FINITE_GROUP_SCHEMA,
            "provenance": self.provenance,
            "replicas": list(self.replicas),
            "dimensionality": self.dimensionality,
            "length_unit": self.length_unit,
        }


@dataclass(frozen=True)
class SECCMTranslationAction:
    """Complete static-site action of a bound finite translation group.

    The equivalence keys attest static site identity.  Family adapters must
    separately validate translation covariance of their dynamic Hamiltonian
    state before using orbit reduction.
    """

    atom_primitive_site_ids: tuple[int, ...]
    atom_cell_labels: tuple[FiniteTranslationLabel, ...]
    atom_equivalence_keys: tuple[AtomEquivalenceKey, ...]
    atom_permutations: tuple[tuple[int, ...], ...]
    equivalence_key_schema: str
    provenance: str = _ATOM_ACTION_PROVENANCE

    def __post_init__(self) -> None:
        if self.provenance != _ATOM_ACTION_PROVENANCE:
            raise SECCMTopologyProvenanceError(
                "unsupported SECCM translation-action provenance"
            )
        site_ids = _validated_site_ids(self.atom_primitive_site_ids)
        cell_labels = tuple(tuple(label) for label in self.atom_cell_labels)
        equivalence_keys = _validated_equivalence_keys(
            self.atom_equivalence_keys, expected_count=len(site_ids)
        )
        permutations = tuple(
            tuple(int(index) for index in permutation)
            for permutation in self.atom_permutations
        )
        schema = _validated_equivalence_key_schema(
            self.equivalence_key_schema
        )
        object.__setattr__(self, "atom_primitive_site_ids", site_ids)
        object.__setattr__(self, "atom_cell_labels", cell_labels)
        object.__setattr__(self, "atom_equivalence_keys", equivalence_keys)
        object.__setattr__(self, "atom_permutations", permutations)
        object.__setattr__(self, "equivalence_key_schema", schema)

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(self._identity_payload())

    def _identity_payload(self) -> dict:
        return {
            "schema": _ATOM_ACTION_SCHEMA,
            "provenance": self.provenance,
            "equivalence_key_schema": self.equivalence_key_schema,
            "atom_primitive_site_ids": list(self.atom_primitive_site_ids),
            "atom_cell_labels": [
                list(label) for label in self.atom_cell_labels
            ],
            "atom_equivalence_keys": [
                list(key) for key in self.atom_equivalence_keys
            ],
        }


@dataclass(frozen=True)
class SECCMCandidateClassification:
    """Reference WS-membership state for one enumerated image candidate."""

    central: int
    origin: int
    image_shell_label: ImageShellLabel
    prefiltered: bool
    legacy_membership_score: float | None
    rounded_legacy_score: float | None
    legacy_inside: bool
    signed_margin: float | None
    absolute_margin: float | None
    reference_tie: bool
    symmetric_ws_score: float | None


@dataclass
class SECCMImage:
    """One directed geometric image in a central atom's WS cell.

    The first three fields preserve the positional ``WSNeighbor`` constructor.
    ``image_shell_label`` identifies a geometric image in the original cyclic
    lattice basis; it is not a finite-group cell label. Components are
    unbounded integers because a strongly skew input basis can map a nearest
    reduced-basis image to a large original-basis label.
    """

    origin: int
    weight: float
    disp: np.ndarray
    image_shell_label: ImageShellLabel = (0, 0, 0)
    ownership_multiplicity: int = 1


@dataclass
class SECCMTopology:
    """Directed WS image cells plus their frozen reference classification."""

    cells: list[list[SECCMImage]]
    translations: list[np.ndarray] = field(default_factory=list)
    reference_coords: np.ndarray | None = field(default=None, repr=False)
    length_unit: str | None = None
    geometry_quantum: float | None = None
    builder_provenance: str | None = None
    _reference_candidates: tuple[SECCMCandidateClassification, ...] = field(
        default=(), repr=False
    )
    _reference_translations: tuple[np.ndarray, ...] = field(
        default=(), repr=False
    )
    finite_group: SECCMFiniteGroup | None = None
    translation_action: SECCMTranslationAction | None = None
    _current_lattice_group_compatible: bool = field(
        default=False, repr=False
    )
    _current_atom_geometry_group_covariant: bool | None = field(
        default=None, repr=False
    )

    def total_weight(self, i: int) -> float:
        return float(sum(image.weight for image in self.cells[i]))

    def is_valid(self, natoms: int) -> bool:
        """Preserve the MSINDO ``round(WTOT) == NATOMS - 1`` validity rule."""
        return all(
            round(self.total_weight(i)) == natoms - 1
            for i in range(len(self.cells))
        )

    @property
    def dimensionality(self) -> int:
        return len(self.translations)

    @property
    def finite_group_bound(self) -> bool:
        return self.finite_group is not None

    @property
    def complete_atom_action_bound(self) -> bool:
        return self.translation_action is not None

    @property
    def current_lattice_group_compatible(self) -> bool:
        return (
            self.finite_group_bound
            and self._current_lattice_group_compatible
        )

    @property
    def current_atom_geometry_group_covariant(self) -> bool | None:
        if not self.complete_atom_action_bound:
            return None
        return self._current_atom_geometry_group_covariant

    @property
    def current_geometry_group_covariant(self) -> bool:
        """Return whether the current atoms realize the complete group action."""
        return self.current_atom_geometry_group_covariant is True

    @property
    def record_orbits_currently_usable(self) -> bool:
        return (
            self.complete_atom_action_bound
            and self.current_lattice_group_compatible
            and self.current_atom_geometry_group_covariant is True
        )

    @property
    def candidate_classifications(
        self,
    ) -> tuple[SECCMCandidateClassification, ...]:
        self._require_builder_provenance()
        return self._reference_candidates

    @property
    def switching_margin(self) -> float:
        """Minimum absolute unrounded WS margin at the reference geometry."""
        self._require_builder_provenance()
        margins = [
            candidate.absolute_margin
            for candidate in self._reference_candidates
            if candidate.absolute_margin is not None
        ]
        return min(margins, default=math.inf)

    @property
    def has_reference_ties(self) -> bool:
        self._require_builder_provenance()
        return any(c.reference_tie for c in self._reference_candidates)

    @property
    def topology_fingerprint(self) -> str:
        """Hash the discrete records, their order, and builder provenance."""
        self._require_builder_provenance()
        records = []
        for central, cell in enumerate(self.cells):
            for image in cell:
                records.append(
                    [
                        central,
                        int(image.origin),
                        *[int(value) for value in image.image_shell_label],
                        int(image.ownership_multiplicity),
                    ]
                )
        payload = {
            "schema": _TOPOLOGY_SCHEMA,
            "builder_provenance": self.builder_provenance,
            "dimensionality": self.dimensionality,
            "records": records,
        }
        if self.finite_group is not None:
            payload["finite_group"] = self.finite_group._identity_payload()
        if self.translation_action is not None:
            payload["translation_action"] = (
                self.translation_action._identity_payload()
            )
        return _canonical_sha256(payload)

    @property
    def reference_geometry_fingerprint(self) -> str:
        """Hash quantized reference geometry under a cross-language rule."""
        self._require_builder_provenance()
        if self.reference_coords is None or not self._reference_translations:
            raise SECCMTopologyProvenanceError(
                "reference geometry fingerprint requires copied reference arrays"
            )
        if self.length_unit is None or self.geometry_quantum is None:
            raise SECCMTopologyProvenanceError(
                "reference geometry fingerprint requires length_unit and "
                "geometry_quantum"
            )
        coords = np.asarray(self.reference_coords, dtype=float)
        translations = np.asarray(self._reference_translations, dtype=float)
        payload = {
            "schema": _GEOMETRY_SCHEMA,
            "length_unit": self.length_unit,
            "geometry_quantum": _decimal_string(self.geometry_quantum),
            "coordinates": _quantized_array_payload(
                coords, self.geometry_quantum
            ),
            "translations": _quantized_array_payload(
                translations, self.geometry_quantum
            ),
        }
        if self.finite_group is not None:
            payload["finite_group"] = {
                "schema": _FINITE_GROUP_SCHEMA,
                "primitive_vectors": _quantized_array_payload(
                    np.asarray(self.finite_group.primitive_vectors),
                    self.geometry_quantum,
                ),
            }
        return _canonical_sha256(payload)

    def diagnostics(self, natoms: int | None = None) -> tuple[TopologyDiagnostic, ...]:
        """Return deterministic ownership, reversal, and switching diagnostics."""
        diagnostics: list[TopologyDiagnostic] = []
        count = len(self.cells) if natoms is None else int(natoms)
        for central, cell in enumerate(self.cells):
            totals: dict[int, float] = {}
            for image in cell:
                totals[image.origin] = totals.get(image.origin, 0.0) + image.weight
            for origin in range(count):
                expected = 0.0 if origin == central else 1.0
                actual = totals.get(origin, 0.0)
                if not math.isclose(actual, expected, abs_tol=1.0e-12):
                    diagnostics.append(
                        TopologyDiagnostic(
                            TopologyDiagnosticCode.INVALID_TOTAL_ORIGIN_OWNERSHIP,
                            "directed image ownership does not sum to its "
                            "expected origin coverage",
                            central=central,
                            origin=origin,
                        )
                    )

        if self.builder_provenance is None:
            diagnostics.append(_missing_provenance_diagnostic())
            return tuple(diagnostics)

        diagnostics.extend(self._duplicate_diagnostics())
        _, reversal_diagnostics = self._reversal_analysis()
        diagnostics.extend(reversal_diagnostics)
        diagnostics.extend(
            TopologyDiagnostic(
                TopologyDiagnosticCode.NEAR_SWITCHING_SURFACE,
                "reference image candidate lies on the legacy rounded WS "
                "switching surface",
                central=candidate.central,
                origin=candidate.origin,
            )
            for candidate in self._reference_candidates
            if candidate.reference_tie
        )
        if (
            self.translation_action is not None
            and not self.record_orbits_currently_usable
        ):
            diagnostics.append(
                TopologyDiagnostic(
                    TopologyDiagnosticCode.CURRENT_GEOMETRY_NOT_TRANSLATION_COVARIANT,
                    "current geometry cannot use translation-orbit reduction",
                )
            )
        return tuple(diagnostics)

    def reversal_map(
        self,
    ) -> dict[tuple[int, int], tuple[int, int]]:
        """Return exact directed reversal pairs, failing on malformed records."""
        self._require_builder_provenance()
        mapping, diagnostics = self._reversal_analysis()
        diagnostics = [
            diagnostic
            for diagnostic in (*self._duplicate_diagnostics(), *diagnostics)
            if diagnostic.code.value in _REVERSAL_DIAGNOSTICS
        ]
        if diagnostics:
            codes = ", ".join(sorted({d.code.value for d in diagnostics}))
            raise SECCMTopologyError(
                f"SECCM reversal map is incomplete or ambiguous: {codes}"
            )
        return mapping

    def rebuild_displacements(
        self,
        coords,
        translations=None,
        *,
        tie_policy: Literal["freeze_reference_ties"] = "freeze_reference_ties",
        max_tie_score_excursion: float | None = None,
    ) -> SECCMTopology:
        """Recompute displacements while preserving and validating topology.

        Exact reference ties use symmetric fractional-topology continuation.
        Callers must provide a positive finite score-excursion trust bound when
        such ties exist; the bound prevents reuse at an arbitrarily distant
        geometry.
        """
        self._require_builder_provenance()
        if tie_policy != "freeze_reference_ties":
            raise ValueError(
                "T1a supports only tie_policy='freeze_reference_ties'"
            )
        if self.reference_coords is None or not self._reference_candidates:
            raise SECCMTopologyProvenanceError(
                "fixed-topology rebuild requires reference candidate provenance"
            )
        new_coords = _validated_coords(coords, natoms=len(self.reference_coords))
        current_translations = (
            self.translations if translations is None else translations
        )
        new_translations = _validated_translations(current_translations)
        if len(new_translations) != self.dimensionality:
            raise ValueError("fixed-topology rebuild cannot change dimensionality")

        if self.has_reference_ties:
            if (
                max_tie_score_excursion is None
                or not math.isfinite(max_tie_score_excursion)
                or max_tie_score_excursion <= 0.0
            ):
                raise ValueError(
                    "reference ties require a positive finite "
                    "max_tie_score_excursion"
                )

        reduced_lattice, unimodular_map = _reduce_cyclic_lattice(
            new_translations
        )
        vecs, betrag = _first_shell(reduced_lattice)
        included_keys = {
            (central, image.origin, image.image_shell_label)
            for central, cell in enumerate(self.cells)
            for image in cell
        }
        for reference in self._reference_candidates:
            displaced = _classify_candidate(
                reference.central,
                reference.origin,
                reference.image_shell_label,
                new_coords,
                new_translations,
                vecs,
                betrag,
            )
            key = (
                reference.central,
                reference.origin,
                reference.image_shell_label,
            )
            included = key in included_keys
            if included and reference.reference_tie:
                if displaced.legacy_membership_score is None:
                    raise SECCMTopologyChangedError(
                        "reference tie left the score-defined continuation region"
                    )
                excursion = abs(
                    displaced.legacy_membership_score
                    - reference.legacy_membership_score
                )
                if excursion > max_tie_score_excursion:
                    raise SECCMTopologyChangedError(
                        "reference tie exceeded max_tie_score_excursion"
                    )
            elif included and not displaced.legacy_inside:
                raise SECCMTopologyChangedError(
                    "included non-tie image left the frozen WS assignment"
                )
            elif not included and displaced.legacy_inside:
                raise SECCMTopologyChangedError(
                    "previously absent image entered the frozen WS assignment"
                )

        current_candidates = _enumerate_candidate_classifications(
            new_coords,
            new_translations,
            reduced_lattice,
            unimodular_map,
            vecs,
            betrag,
            unit_scale=_bohr_per_length_unit(self.length_unit),
        )
        newly_included = {
            (
                candidate.central,
                candidate.origin,
                candidate.image_shell_label,
            )
            for candidate in current_candidates
            if candidate.legacy_inside
            and not (
                candidate.central == candidate.origin
                and candidate.image_shell_label == (0, 0, 0)
            )
        }
        if not newly_included <= included_keys:
            raise SECCMTopologyChangedError(
                "previously absent image entered the frozen WS assignment"
            )
        return _rebuild_from_labels(self, new_coords, new_translations)

    @property
    def finite_group_labels(self) -> tuple[FiniteTranslationLabel, ...]:
        if self.finite_group is None:
            raise NotImplementedError(
                "SECCM finite-group labels require an explicit lattice binding"
            )
        return self.finite_group.labels

    @property
    def translation_orbits(self) -> tuple[tuple[tuple[int, int], ...], ...]:
        return self.record_translation_orbits()

    def atom_translation_permutation(
        self, shift: FiniteTranslationLabel
    ) -> tuple[int, ...]:
        """Return the reference static-site permutation for one group shift."""
        group, action = self._require_translation_action()
        shift = group.require_canonical_label(shift)
        return action.atom_permutations[group.labels.index(shift)]

    def require_current_translation_covariance(self) -> None:
        """Fail before any consumer can use current-geometry orbit reduction."""
        self._require_translation_action()
        if not self.record_orbits_currently_usable:
            _raise_topology_error(
                TopologyDiagnosticCode.CURRENT_GEOMETRY_NOT_TRANSLATION_COVARIANT,
                "current geometry cannot use translation-orbit reduction",
            )

    def record_translation_map(
        self, shift: FiniteTranslationLabel
    ) -> dict[tuple[int, int], tuple[int, int]]:
        """Map every directed WS record under a simultaneous group shift."""
        self.require_current_translation_covariance()
        group, action = self._require_translation_action()
        shift = group.require_canonical_label(shift)
        permutation = self.atom_translation_permutation(shift)

        lookup: dict[
            tuple[int, int, ImageShellLabel], list[tuple[int, int]]
        ] = {}
        for central, cell in enumerate(self.cells):
            for record_index, image in enumerate(cell):
                lookup.setdefault(
                    (central, image.origin, image.image_shell_label), []
                ).append((central, record_index))

        mapping: dict[tuple[int, int], tuple[int, int]] = {}
        for central, cell in enumerate(self.cells):
            central_label = action.atom_cell_labels[central]
            translated_central = permutation[central]
            for record_index, image in enumerate(cell):
                origin_label = action.atom_cell_labels[image.origin]
                translated_origin = permutation[image.origin]
                translated_shell = tuple(
                    image.image_shell_label[axis]
                    + (origin_label[axis] + shift[axis])
                    // group.replicas[axis]
                    - (central_label[axis] + shift[axis])
                    // group.replicas[axis]
                    for axis in range(3)
                )
                matches = lookup.get(
                    (
                        translated_central,
                        translated_origin,
                        translated_shell,
                    ),
                    [],
                )
                if not matches:
                    _raise_topology_error(
                        TopologyDiagnosticCode.MISSING_TRANSLATED_RECORD,
                        "group shift has no translated directed WS record",
                    )
                if len(matches) != 1:
                    _raise_topology_error(
                        TopologyDiagnosticCode.AMBIGUOUS_TRANSLATED_RECORD,
                        "group shift has multiple translated directed WS records",
                    )
                target_key = matches[0]
                target = self.cells[target_key[0]][target_key[1]]
                if (
                    target.ownership_multiplicity
                    != image.ownership_multiplicity
                    or not math.isclose(
                        target.weight, image.weight, abs_tol=1.0e-12
                    )
                ):
                    _raise_topology_error(
                        TopologyDiagnosticCode.TRANSLATED_RECORD_OWNERSHIP_MISMATCH,
                        "translated directed WS record changed ownership",
                    )
                if not np.allclose(
                    target.disp,
                    image.disp,
                    rtol=0.0,
                    atol=group.geometry_tolerance,
                ):
                    _raise_topology_error(
                        TopologyDiagnosticCode.TRANSLATED_RECORD_DISPLACEMENT_MISMATCH,
                        "translated directed WS record changed displacement",
                    )
                mapping[(central, record_index)] = target_key
        return mapping

    def record_translation_orbits(
        self,
    ) -> tuple[tuple[tuple[int, int], ...], ...]:
        """Partition directed WS records into exact finite-group orbits."""
        self.require_current_translation_covariance()
        group, _ = self._require_translation_action()
        maps = [self.record_translation_map(shift) for shift in group.labels]
        all_records = {
            (central, record_index)
            for central, cell in enumerate(self.cells)
            for record_index in range(len(cell))
        }
        unassigned = set(all_records)
        orbits: list[tuple[tuple[int, int], ...]] = []
        while unassigned:
            representative = min(unassigned)
            orbit = tuple(sorted({mapping[representative] for mapping in maps}))
            orbit_set = set(orbit)
            if not orbit_set <= unassigned:
                _raise_topology_error(
                    TopologyDiagnosticCode.AMBIGUOUS_TRANSLATED_RECORD,
                    "translation orbits do not form a disjoint partition",
                )
            unassigned.difference_update(orbit_set)
            orbits.append(orbit)
        return tuple(orbits)

    def _require_translation_action(
        self,
    ) -> tuple[SECCMFiniteGroup, SECCMTranslationAction]:
        if self.finite_group is None or self.translation_action is None:
            _raise_topology_error(
                TopologyDiagnosticCode.ATOM_ACTION_NOT_BOUND,
                "operation requires a complete static-site translation action",
            )
        return self.finite_group, self.translation_action

    def _require_builder_provenance(self) -> None:
        if self.builder_provenance is None:
            diagnostic = _missing_provenance_diagnostic()
            raise SECCMTopologyProvenanceError(
                f"{diagnostic.code.value}: {diagnostic.message}"
            )

    def _duplicate_diagnostics(self) -> list[TopologyDiagnostic]:
        diagnostics: list[TopologyDiagnostic] = []
        for central, cell in enumerate(self.cells):
            seen: dict[tuple[int, ImageShellLabel], int] = {}
            for record_index, image in enumerate(cell):
                key = (image.origin, image.image_shell_label)
                if key in seen:
                    diagnostics.append(
                        TopologyDiagnostic(
                            TopologyDiagnosticCode.DUPLICATE_DISCRETE_RECORD,
                            "central cell contains a duplicate origin/image label",
                            central=central,
                            record_index=record_index,
                            origin=image.origin,
                        )
                    )
                else:
                    seen[key] = record_index
        return diagnostics

    def _reversal_analysis(
        self,
    ) -> tuple[
        dict[tuple[int, int], tuple[int, int]],
        list[TopologyDiagnostic],
    ]:
        lookup: dict[
            tuple[int, int, ImageShellLabel], list[tuple[int, int]]
        ] = {}
        by_pair: dict[tuple[int, int], list[tuple[int, int, ImageShellLabel]]] = {}
        for central, cell in enumerate(self.cells):
            for record_index, image in enumerate(cell):
                lookup.setdefault(
                    (central, image.origin, image.image_shell_label), []
                ).append((central, record_index))
                by_pair.setdefault((central, image.origin), []).append(
                    (central, record_index, image.image_shell_label)
                )

        mapping: dict[tuple[int, int], tuple[int, int]] = {}
        diagnostics: list[TopologyDiagnostic] = []
        for central, cell in enumerate(self.cells):
            for record_index, image in enumerate(cell):
                opposite = tuple(-value for value in image.image_shell_label)
                matches = lookup.get((image.origin, central, opposite), [])
                if not matches:
                    diagnostics.append(
                        TopologyDiagnostic(
                            TopologyDiagnosticCode.MISSING_REVERSE_IMAGE,
                            "directed image has no opposite-label reverse",
                            central=central,
                            record_index=record_index,
                            origin=image.origin,
                        )
                    )
                    if by_pair.get((image.origin, central)):
                        diagnostics.append(
                            TopologyDiagnostic(
                                TopologyDiagnosticCode.NONOPPOSITE_REVERSE_LABEL,
                                "reverse central/origin pair exists under a "
                                "non-opposite image label",
                                central=central,
                                record_index=record_index,
                                origin=image.origin,
                            )
                        )
                    continue
                if len(matches) != 1:
                    diagnostics.append(
                        TopologyDiagnostic(
                            TopologyDiagnosticCode.AMBIGUOUS_REVERSE_IMAGE,
                            "directed image has multiple opposite-label reverses",
                            central=central,
                            record_index=record_index,
                            origin=image.origin,
                        )
                    )
                    continue
                reverse_key = matches[0]
                reverse = self.cells[reverse_key[0]][reverse_key[1]]
                mapping[(central, record_index)] = reverse_key
                if not np.allclose(
                    np.asarray(reverse.disp, dtype=float),
                    -np.asarray(image.disp, dtype=float),
                    rtol=1.0e-12,
                    atol=1.0e-12,
                ):
                    diagnostics.append(
                        TopologyDiagnostic(
                            TopologyDiagnosticCode.INCONSISTENT_REVERSE_DISPLACEMENT,
                            "reverse image displacement is not opposite",
                            central=central,
                            record_index=record_index,
                            origin=image.origin,
                        )
                    )
                if (
                    reverse.ownership_multiplicity
                    != image.ownership_multiplicity
                ):
                    diagnostics.append(
                        TopologyDiagnostic(
                            TopologyDiagnosticCode.INCONSISTENT_REVERSE_OWNERSHIP,
                            "reverse image has a different ownership multiplicity",
                            central=central,
                            record_index=record_index,
                            origin=image.origin,
                        )
                    )
        return mapping, diagnostics


# Compatibility aliases retain the public MSINDO constructor names.
WSNeighbor = SECCMImage
WignerSeitzCells = SECCMTopology


def _rund(value: float, digit: int) -> float:
    """Faithful port of MSINDO ``rund.f``."""
    half = digit // 2
    rest = digit - half
    expo1 = 10.0**rest
    expo2 = 10.0**half
    sign = 1.0 if value >= 0.0 else -1.0
    integer = int(abs(value * expo1))
    remainder = abs(value * expo1) - integer
    rounded = float(int(remainder * expo2 + 0.5)) / expo2
    return sign * (integer + rounded) / expo1


def build_seccm_topology(
    coords,
    translations,
    *,
    length_unit: Literal["bohr", "angstrom"] | None = None,
    geometry_quantum: float | None = None,
) -> SECCMTopology:
    """Build a basis-invariant exact WS topology without Hamiltonian terms."""
    reference_coords = _validated_coords(coords)
    reference_translations = _validated_translations(translations)
    _validate_geometry_metadata(length_unit, geometry_quantum)

    natoms = len(reference_coords)
    reduced_lattice, unimodular_map = _reduce_cyclic_lattice(
        reference_translations
    )
    vecs, betrag = _first_shell(reduced_lattice)

    cells: list[list[SECCMImage]] = []
    classifications = _enumerate_candidate_classifications(
        reference_coords,
        reference_translations,
        reduced_lattice,
        unimodular_map,
        vecs,
        betrag,
        unit_scale=_bohr_per_length_unit(length_unit),
    )
    candidates_by_pair: dict[
        tuple[int, int], list[SECCMCandidateClassification]
    ] = {}
    for candidate in classifications:
        candidates_by_pair.setdefault(
            (candidate.central, candidate.origin), []
        ).append(candidate)
    for central in range(natoms):
        included: list[tuple[int, np.ndarray, ImageShellLabel]] = []
        for origin in range(natoms):
            for classification in candidates_by_pair[(central, origin)]:
                label = classification.image_shell_label
                if origin == central and label == (0, 0, 0):
                    continue
                if classification.legacy_inside:
                    translation = _translation_from_label(
                        label, reference_translations
                    )
                    included.append(
                        (
                            origin,
                            reference_coords[origin] + translation,
                            label,
                        )
                    )

        multiplicities: dict[int, int] = {}
        for origin, _, _ in included:
            multiplicities[origin] = multiplicities.get(origin, 0) + 1
        cells.append(
            [
                SECCMImage(
                    origin=origin,
                    weight=1.0 / multiplicities[origin],
                    disp=position - reference_coords[central],
                    image_shell_label=label,
                    ownership_multiplicity=multiplicities[origin],
                )
                for origin, position, label in included
            ]
        )

    reference_translation_copies = tuple(
        _readonly_copy(translation) for translation in reference_translations
    )
    return SECCMTopology(
        cells=cells,
        translations=list(reference_translation_copies),
        reference_coords=_readonly_copy(reference_coords),
        length_unit=length_unit,
        geometry_quantum=geometry_quantum,
        builder_provenance=_BUILDER_PROVENANCE,
        _reference_candidates=classifications,
        _reference_translations=reference_translation_copies,
    )


build_wigner_seitz = build_seccm_topology


def bind_finite_group(
    topology: SECCMTopology,
    *,
    primitive_vectors,
    replicas,
    geometry_tolerance: float,
    length_unit: str,
) -> SECCMTopology:
    """Bind validated finite BvK lattice metadata without requiring atom symmetry."""
    topology._require_builder_provenance()
    if topology.finite_group is not None:
        raise SECCMTopologyError("SECCM finite group is already bound")
    try:
        primitive_vectors = tuple(primitive_vectors)
    except TypeError as exc:
        _raise_topology_error(
            TopologyDiagnosticCode.INVALID_PRIMITIVE_VECTORS,
            "primitive vectors must contain one vector per active dimension",
            cause=exc,
        )
    dimensionality = len(primitive_vectors)
    if dimensionality != topology.dimensionality:
        _raise_topology_error(
            TopologyDiagnosticCode.FINITE_GROUP_DIMENSION_MISMATCH,
            "finite group and WS topology have different dimensionality",
        )
    group = SECCMFiniteGroup(
        primitive_vectors=primitive_vectors,
        replicas=replicas,
        geometry_tolerance=geometry_tolerance,
        length_unit=length_unit,
    )
    if topology.length_unit is not None and topology.length_unit != group.length_unit:
        _raise_topology_error(
            TopologyDiagnosticCode.CYCLIC_TRANSLATION_MISMATCH,
            "finite group and WS topology use different length units",
        )
    if not _cyclic_translations_match_group(topology.translations, group):
        _raise_topology_error(
            TopologyDiagnosticCode.CYCLIC_TRANSLATION_MISMATCH,
            "cyclic translations do not equal replica-scaled primitive vectors",
        )
    return replace(
        topology,
        finite_group=group,
        translation_action=None,
        _current_lattice_group_compatible=True,
        _current_atom_geometry_group_covariant=None,
    )


def bind_complete_translation_action(
    topology: SECCMTopology,
    *,
    atom_primitive_site_ids,
    atom_cell_labels,
    atom_equivalence_keys,
    equivalence_key_schema: str,
) -> SECCMTopology:
    """Bind a complete static-site action after geometric and chemical checks."""
    if topology.finite_group is None:
        _raise_topology_error(
            TopologyDiagnosticCode.ATOM_ACTION_NOT_BOUND,
            "complete translation action requires a bound finite group",
        )
    if topology.translation_action is not None:
        raise SECCMTopologyError("SECCM translation action is already bound")
    group = topology.finite_group
    if topology.reference_coords is None:
        raise SECCMTopologyProvenanceError(
            "complete translation action requires reference coordinates"
        )
    natoms = len(topology.reference_coords)
    site_ids = _validated_site_ids(
        atom_primitive_site_ids, expected_count=natoms
    )
    labels = _validated_atom_cell_labels(
        atom_cell_labels, group, expected_count=natoms
    )
    keys = _validated_equivalence_keys(
        atom_equivalence_keys, expected_count=natoms
    )
    schema = _validated_equivalence_key_schema(equivalence_key_schema)

    unique_site_ids = sorted(set(site_ids))
    if not unique_site_ids or unique_site_ids != list(range(len(unique_site_ids))):
        _raise_topology_error(
            TopologyDiagnosticCode.NONCONTIGUOUS_PRIMITIVE_SITE_IDS,
            "primitive site IDs must be contiguous from zero",
        )
    expected_count = len(unique_site_ids) * group.order
    if natoms != expected_count:
        _raise_topology_error(
            TopologyDiagnosticCode.MISSING_CELL_COPY,
            "complete action requires one copy of every primitive site per cell",
        )

    atom_by_site_and_cell: dict[tuple[int, FiniteTranslationLabel], int] = {}
    for atom_index, pair in enumerate(zip(site_ids, labels, strict=True)):
        if pair in atom_by_site_and_cell:
            _raise_topology_error(
                TopologyDiagnosticCode.DUPLICATE_CELL_COPY,
                "complete action contains a duplicate primitive-site/cell copy",
            )
        atom_by_site_and_cell[pair] = atom_index
    expected_pairs = {
        (site_id, label)
        for site_id in unique_site_ids
        for label in group.labels
    }
    if set(atom_by_site_and_cell) != expected_pairs:
        _raise_topology_error(
            TopologyDiagnosticCode.MISSING_CELL_COPY,
            "complete action is missing a primitive-site/cell copy",
        )

    zero_label = (0, 0, 0)
    coords = np.asarray(topology.reference_coords, dtype=float)
    for site_id in unique_site_ids:
        zero_index = atom_by_site_and_cell[(site_id, zero_label)]
        zero_coord = coords[zero_index]
        zero_key = keys[zero_index]
        for label in group.labels:
            atom_index = atom_by_site_and_cell[(site_id, label)]
            if keys[atom_index] != zero_key:
                _raise_topology_error(
                    TopologyDiagnosticCode.ATOM_EQUIVALENCE_KEY_MISMATCH,
                    "primitive-site copies have different equivalence keys",
                )
            expected_coord = zero_coord + _primitive_translation(group, label)
            if not np.allclose(
                coords[atom_index],
                expected_coord,
                rtol=0.0,
                atol=group.geometry_tolerance,
            ):
                _raise_topology_error(
                    TopologyDiagnosticCode.COORDINATE_COVARIANCE_MISMATCH,
                    "primitive-site copies are not translation covariant",
                )

    permutations = tuple(
        tuple(
            atom_by_site_and_cell[
                (site_ids[atom_index], group.add(labels[atom_index], shift))
            ]
            for atom_index in range(natoms)
        )
        for shift in group.labels
    )
    expected_indices = list(range(natoms))
    if any(sorted(permutation) != expected_indices for permutation in permutations):
        _raise_topology_error(
            TopologyDiagnosticCode.MISSING_CELL_COPY,
            "translation action did not produce bijective atom permutations",
        )
    action = SECCMTranslationAction(
        atom_primitive_site_ids=site_ids,
        atom_cell_labels=labels,
        atom_equivalence_keys=keys,
        atom_permutations=permutations,
        equivalence_key_schema=schema,
    )
    return replace(
        topology,
        translation_action=action,
        _current_lattice_group_compatible=True,
        _current_atom_geometry_group_covariant=True,
    )


def _first_shell(translations) -> tuple[np.ndarray, np.ndarray]:
    """Return the complete fixed Voronoi shell of a reduced D <= 3 basis."""
    translations = tuple(
        np.asarray(vector, dtype=float) for vector in translations
    )
    dimensionality = len(translations)
    vectors: list[np.ndarray] = []
    squared_lengths: list[float] = []
    for active_label in product((-1, 0, 1), repeat=dimensionality):
        if all(value == 0 for value in active_label):
            continue
        vector = sum(
            (
                active_label[axis] * translations[axis]
                for axis in range(dimensionality)
            ),
            start=np.zeros(3),
        )
        vectors.append(vector)
        squared_lengths.append(float(vector @ vector))
    return np.asarray(vectors), np.asarray(squared_lengths)


def _reduce_cyclic_lattice(
    translations: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Minkowski-reduce active vectors and retain the integer label map."""
    lattice = np.asarray(translations, dtype=float)
    dimensionality = len(translations)
    singular_values = np.linalg.svd(lattice, compute_uv=False)
    scale = max(1.0, float(singular_values[0]))
    rank_tolerance = max(
        1.0e-12,
        64.0 * np.finfo(float).eps * scale,
    )
    if float(singular_values[-1]) <= rank_tolerance:
        raise ValueError(
            "SECCM translations must be nonzero and linearly independent"
        )
    if dimensionality == 1:
        return lattice.copy(), np.eye(1, dtype=int)

    from ase.geometry import minkowski_reduce

    if dimensionality == 2:
        normal = np.cross(lattice[0], lattice[1])
        normal /= np.linalg.norm(normal)
        padded = np.vstack((lattice, normal))
        _, full_map = minkowski_reduce(
            padded, pbc=np.array([True, True, False])
        )
        unimodular_map = np.asarray(full_map, dtype=int)[:2, :2]
    else:
        _, full_map = minkowski_reduce(lattice)
        unimodular_map = np.asarray(full_map, dtype=int)

    determinant = round(float(np.linalg.det(unimodular_map)))
    if abs(determinant) != 1:
        raise ValueError(
            "SECCM Minkowski reduction did not return a unimodular label map"
        )
    reduced_lattice = unimodular_map @ lattice
    return np.asarray(reduced_lattice, dtype=float), unimodular_map


def _candidate_labels(
    displacement: np.ndarray,
    reduced_lattice: np.ndarray,
    unimodular_map: np.ndarray,
) -> tuple[tuple[ImageShellLabel, ...], frozenset[ImageShellLabel]]:
    """Return a complete fixed shell around the reduced closest cell."""
    gram = reduced_lattice @ reduced_lattice.T
    fractional = np.linalg.solve(gram, reduced_lattice @ (-displacement))
    base = np.rint(fractional).astype(int)
    labels: set[ImageShellLabel] = set()
    rim_labels: set[ImageShellLabel] = set()
    for shift in product(
        range(-_REDUCED_IMAGE_SEARCH, _REDUCED_IMAGE_SEARCH + 1),
        repeat=len(reduced_lattice),
    ):
        reduced_label = base + np.asarray(shift, dtype=int)
        active_label = reduced_label @ unimodular_map
        if np.any(active_label < np.iinfo(np.int32).min) or np.any(
            active_label > np.iinfo(np.int32).max
        ):
            raise SECCMTopologyError(
                "SECCM closest-image label exceeds the native integer range"
            )
        padded = tuple(
            int(active_label[axis]) if axis < len(active_label) else 0
            for axis in range(3)
        )
        labels.add(padded)
        if max(abs(value) for value in shift) == _REDUCED_IMAGE_SEARCH:
            rim_labels.add(padded)
    return tuple(sorted(labels)), frozenset(rim_labels)


def _bohr_per_length_unit(length_unit: str | None) -> float:
    """Bohr per caller length unit.

    ``None`` is the bohr convention of every vibe-qc route (the native builder
    in ``seccm/topology.hpp`` is bohr-only).  Mirrors
    ``_adapter_common.topology_length_unit_scale`` for a bound topology.
    """
    if length_unit is None or length_unit == "bohr":
        return 1.0
    if length_unit == "angstrom":
        return ANGSTROM_TO_BOHR
    raise ValueError(
        f"unsupported SECCM length unit {length_unit!r}; expected one of "
        f"{sorted(_SUPPORTED_LENGTH_UNITS)}"
    )


def _distance_tolerance(
    displacement: np.ndarray,
    translations: list[np.ndarray],
    *,
    unit_scale: float,
) -> float:
    """Tie window in the caller's length unit.

    The physical rule is MSINDO's 1.0e-5 bohr (``_EQUIDISTANT_IMAGE_TOLERANCE``),
    so it is divided by ``unit_scale`` (bohr per caller unit, GitLab #669).  The
    roundoff guard stays in caller units: it bounds the representation error of
    the numbers actually compared, ~5.7e-14 times their scale, and cannot reach
    the physical window in either unit.
    """
    scale = max(
        1.0,
        float(np.linalg.norm(displacement)),
        *(float(np.linalg.norm(vector)) for vector in translations),
    )
    return max(
        _EQUIDISTANT_IMAGE_TOLERANCE / float(unit_scale),
        256.0 * np.finfo(float).eps * scale,
    )


def _enumerate_candidate_classifications(
    coords: np.ndarray,
    translations: list[np.ndarray],
    reduced_lattice: np.ndarray,
    unimodular_map: np.ndarray,
    vecs: np.ndarray,
    betrag: np.ndarray,
    *,
    unit_scale: float,
) -> tuple[SECCMCandidateClassification, ...]:
    """Enumerate exact closest images, mapping reduced labels to input labels.

    ``unit_scale`` is bohr per caller length unit; it sizes the equidistant-image
    window so that identical physical coordinates classify identically whether
    they arrive in bohr or in angstrom (GitLab #669).

    Vectorised over the whole (central, origin) x candidate-label grid (issue
    #423).  The scalar path this replaces did one ``np.linalg.solve``, one
    ``norm`` and one small matrix-vector product per *candidate*, plus a
    dataclass allocation, which put ~99 % of a SECCM build in this function
    and made it the wall-clock bottleneck of the whole route.  The arithmetic
    per candidate is unchanged; only the loop order is.
    """
    natoms = len(coords)
    dimension = len(reduced_lattice)
    gram = reduced_lattice @ reduced_lattice.T

    # Pair displacements R_origin - R_central, flattened to (natoms^2, 3).
    pair_displacement = (
        coords[None, :, :] - coords[:, None, :]
    ).reshape(-1, 3)

    # Reduced-basis closest cell of every pair, the vector form of
    # _candidate_labels' solve + rint.
    fractional = np.linalg.solve(gram, reduced_lattice @ (-pair_displacement.T))
    base = np.rint(fractional.T).astype(np.int64)          # (npairs, dim)

    shifts = np.asarray(
        list(
            product(
                range(-_REDUCED_IMAGE_SEARCH, _REDUCED_IMAGE_SEARCH + 1),
                repeat=dimension,
            )
        ),
        dtype=np.int64,
    )                                                       # (nshift, dim)
    on_rim = np.abs(shifts).max(axis=1) == _REDUCED_IMAGE_SEARCH

    reduced_label = base[:, None, :] + shifts[None, :, :]   # (npairs, nshift, dim)
    active = reduced_label @ unimodular_map                 # (npairs, nshift, dim)
    if (
        active.min(initial=0) < np.iinfo(np.int32).min
        or active.max(initial=0) > np.iinfo(np.int32).max
    ):
        raise SECCMTopologyError(
            "SECCM closest-image label exceeds the native integer range"
        )
    padded = np.zeros(active.shape[:2] + (3,), dtype=np.int64)
    padded[..., :dimension] = active

    translation = np.zeros(padded.shape[:2] + (3,))
    for axis, vector in enumerate(translations):
        translation = translation + padded[..., axis, None] * vector

    # The scalar path used two different associations, and both are load
    # bearing at the last bit: the closest-image distance was taken on
    # `(R_origin - R_central) + translation`, while _classify_candidate
    # rebuilt the vector as `R_origin + translation - R_central`.  Keeping
    # one of them for both moves candidates whose Wigner-Seitz score sits
    # exactly on the 1.0 prefilter boundary by one ulp -- 70 of them on
    # MgO 2x2x2 -- which is a discrete reclassification, so both are kept.
    distance_displaced = pair_displacement[:, None, :] + translation
    distances = np.linalg.norm(distance_displaced, axis=2)  # (npairs, nshift)
    central_of_pair = np.repeat(np.arange(natoms), natoms)
    origin_of_pair = np.tile(np.arange(natoms), natoms)
    displaced = (
        coords[origin_of_pair][:, None, :] + translation
    ) - coords[central_of_pair][:, None, :]

    # _distance_tolerance, per pair.
    lattice_scale = max(
        [1.0] + [float(np.linalg.norm(vector)) for vector in translations]
    )
    pair_scale = np.maximum(
        lattice_scale, np.linalg.norm(pair_displacement, axis=1)
    )
    tolerance = np.maximum(
        _EQUIDISTANT_IMAGE_TOLERANCE / float(unit_scale),
        256.0 * np.finfo(float).eps * pair_scale,
    )                                                       # (npairs,)

    minimum = distances.min(axis=1)
    closest = distances <= (minimum + tolerance)[:, None]    # (npairs, nshift)
    if not closest.any(axis=1).all():
        raise SECCMTopologyError(
            "SECCM exact closest-image search returned no image"
        )
    # A label appearing twice in the shift box (possible when the unimodular
    # map folds two reduced shifts onto one active label) is one candidate,
    # exactly as the set-valued scalar path treated it.
    if ((closest & on_rim[None, :]).any(axis=1) & (minimum > tolerance)).any():
        raise SECCMTopologyError(
            "SECCM closest image reached the Minkowski-reduced search "
            "rim; exact topology cannot be certified"
        )

    # Classification arithmetic, identical per candidate to _classify_candidate.
    # Written as an explicit three-term sum rather than einsum/matmul: the
    # scalar path evaluated `vecs @ displacement` as a gemv, and a batched
    # contraction is free to reassociate.  Measured on MgO 2x2x2 that moved
    # 70 candidates whose score is exactly 1.0 by one ulp, flipping them
    # across the `> 1.0` prefilter boundary -- a discrete classification
    # change, not a rounding difference.
    projections = (
        vecs[None, None, :, 0] * displaced[:, :, None, 0]
        + vecs[None, None, :, 1] * displaced[:, :, None, 1]
        + vecs[None, None, :, 2] * displaced[:, :, None, 2]
    ) / betrag
    legacy_score = projections.max(axis=2)                   # (npairs, nshift)
    symmetric = np.abs(projections).max(axis=2)
    prefiltered = legacy_score > 1.0
    signed_margin = 0.5 - legacy_score
    absolute_margin = np.abs(signed_margin)
    # NOTE: the scalar path computed a score-based `legacy_inside` and
    # `reference_tie` inside _classify_candidate and then overwrote both with
    # the distance-based closest-image decision.  Only the distance-based
    # values ever reached a caller, so they are the only ones computed here.

    central_index = central_of_pair
    origin_index = origin_of_pair

    classifications: list[SECCMCandidateClassification] = []
    for pair in range(padded.shape[0]):
        central = int(central_index[pair])
        origin = int(origin_index[pair])
        # De-duplicate the label set per pair, preserving the sorted order the
        # scalar path produced from `tuple(sorted(labels))`.
        seen: dict[ImageShellLabel, int] = {}
        for shift_index in range(padded.shape[1]):
            label = (
                int(padded[pair, shift_index, 0]),
                int(padded[pair, shift_index, 1]),
                int(padded[pair, shift_index, 2]),
            )
            if label in seen:
                if closest[pair, shift_index]:
                    seen[label] = shift_index
                continue
            seen[label] = shift_index
        pair_closest_count = sum(
            1 for label in seen if closest[pair, seen[label]]
        )
        for label in sorted(seen):
            shift_index = seen[label]
            if central == origin and label == (0, 0, 0):
                continue
            inside = bool(closest[pair, shift_index])
            if prefiltered[pair, shift_index]:
                classifications.append(
                    SECCMCandidateClassification(
                        central=central,
                        origin=origin,
                        image_shell_label=label,
                        prefiltered=True,
                        legacy_membership_score=None,
                        rounded_legacy_score=None,
                        legacy_inside=inside,
                        signed_margin=None,
                        absolute_margin=None,
                        reference_tie=inside and pair_closest_count > 1,
                        symmetric_ws_score=None,
                    )
                )
                continue
            score = float(legacy_score[pair, shift_index])
            classifications.append(
                SECCMCandidateClassification(
                    central=central,
                    origin=origin,
                    image_shell_label=label,
                    prefiltered=False,
                    legacy_membership_score=score,
                    rounded_legacy_score=_rund(score, 4),
                    legacy_inside=inside,
                    signed_margin=float(signed_margin[pair, shift_index]),
                    absolute_margin=float(absolute_margin[pair, shift_index]),
                    reference_tie=inside and pair_closest_count > 1,
                    symmetric_ws_score=float(symmetric[pair, shift_index]),
                )
            )
    return tuple(classifications)


def _classify_candidate(
    central: int,
    origin: int,
    label: ImageShellLabel,
    coords: np.ndarray,
    translations: list[np.ndarray],
    vecs: np.ndarray,
    betrag: np.ndarray,
) -> SECCMCandidateClassification:
    translation = _translation_from_label(label, translations)
    displacement = coords[origin] + translation - coords[central]
    projections = (vecs @ displacement) / betrag
    legacy_score = float(np.max(projections))
    if legacy_score > 1.0:
        return SECCMCandidateClassification(
            central=central,
            origin=origin,
            image_shell_label=label,
            prefiltered=True,
            legacy_membership_score=None,
            rounded_legacy_score=None,
            legacy_inside=False,
            signed_margin=None,
            absolute_margin=None,
            reference_tie=False,
            symmetric_ws_score=None,
        )
    rounded_score = _rund(legacy_score, 4)
    signed_margin = 0.5 - legacy_score
    return SECCMCandidateClassification(
        central=central,
        origin=origin,
        image_shell_label=label,
        prefiltered=False,
        legacy_membership_score=legacy_score,
        rounded_legacy_score=rounded_score,
        legacy_inside=legacy_score <= 0.5 + _WS_SCORE_ATOL,
        signed_margin=signed_margin,
        absolute_margin=abs(signed_margin),
        reference_tie=abs(signed_margin) <= _WS_SCORE_ATOL,
        symmetric_ws_score=float(np.max(np.abs(projections))),
    )


def _translation_from_label(
    label: ImageShellLabel, translations: list[np.ndarray]
) -> np.ndarray:
    translation = np.zeros(3)
    for axis, vector in enumerate(translations):
        translation = translation + label[axis] * vector
    return translation


def _rebuild_from_labels(
    topology: SECCMTopology,
    coords: np.ndarray,
    translations: list[np.ndarray],
) -> SECCMTopology:
    cells = [
        [
            SECCMImage(
                origin=image.origin,
                weight=image.weight,
                disp=(
                    coords[image.origin]
                    + _translation_from_label(
                        image.image_shell_label, translations
                    )
                    - coords[central]
                ),
                image_shell_label=image.image_shell_label,
                ownership_multiplicity=image.ownership_multiplicity,
            )
            for image in cell
        ]
        for central, cell in enumerate(topology.cells)
    ]
    current_lattice_group_compatible = False
    current_atom_geometry_group_covariant = None
    if topology.finite_group is not None:
        (
            current_lattice_group_compatible,
            current_atom_geometry_group_covariant,
        ) = _current_group_compatibility(
            topology.finite_group,
            topology.translation_action,
            coords,
            translations,
        )
    return SECCMTopology(
        cells=cells,
        translations=[_readonly_copy(vector) for vector in translations],
        reference_coords=topology.reference_coords,
        length_unit=topology.length_unit,
        geometry_quantum=topology.geometry_quantum,
        builder_provenance=topology.builder_provenance,
        _reference_candidates=topology._reference_candidates,
        _reference_translations=topology._reference_translations,
        finite_group=topology.finite_group,
        translation_action=topology.translation_action,
        _current_lattice_group_compatible=(
            current_lattice_group_compatible
        ),
        _current_atom_geometry_group_covariant=(
            current_atom_geometry_group_covariant
        ),
    )


def _rebuild_displacements_unchecked(
    topology: SECCMTopology,
    coords,
    translations=None,
) -> SECCMTopology:
    # Extraction-parity helper only. Generic public rebuilds never call this.
    topology._require_builder_provenance()
    if topology.reference_coords is None:
        raise SECCMTopologyProvenanceError(
            "unchecked extraction rebuild requires reference coordinates"
        )
    new_coords = _validated_coords(coords, natoms=len(topology.reference_coords))
    current_translations = (
        topology.translations if translations is None else translations
    )
    new_translations = _validated_translations(current_translations)
    return _rebuild_from_labels(topology, new_coords, new_translations)


def _validated_coords(coords, natoms: int | None = None) -> np.ndarray:
    array = np.array(coords, dtype=float, copy=True)
    if array.ndim != 2 or array.shape[1:] != (3,):
        raise ValueError("SECCM coordinates must have shape (natoms, 3)")
    if natoms is not None and len(array) != natoms:
        raise ValueError("fixed-topology rebuild cannot change atom count")
    if not np.all(np.isfinite(array)):
        raise ValueError("SECCM coordinates must be finite")
    return array


def _validated_translations(translations) -> list[np.ndarray]:
    try:
        vectors = list(translations)
    except TypeError as exc:
        raise ValueError("SECCM translations must contain 1, 2, or 3 vectors") from exc
    if len(vectors) not in (1, 2, 3):
        raise ValueError("SECCM translations must contain 1, 2, or 3 vectors")
    result: list[np.ndarray] = []
    for vector in vectors:
        array = np.array(vector, dtype=float, copy=True)
        if array.shape != (3,):
            raise ValueError("each SECCM translation must have shape (3,)")
        if not np.all(np.isfinite(array)):
            raise ValueError("SECCM translations must be finite")
        result.append(array)
    return result


def _validated_primitive_vectors(
    primitive_vectors,
    dimensionality: int,
    geometry_tolerance: float,
) -> tuple[np.ndarray, ...]:
    if (
        not isinstance(geometry_tolerance, (int, float, np.integer, np.floating))
        or isinstance(geometry_tolerance, (bool, np.bool_))
        or not math.isfinite(float(geometry_tolerance))
        or float(geometry_tolerance) <= 0.0
    ):
        _raise_topology_error(
            TopologyDiagnosticCode.INVALID_PRIMITIVE_VECTORS,
            "finite-group geometry tolerance must be positive and finite",
        )
    try:
        vectors = tuple(primitive_vectors)
    except TypeError as exc:
        _raise_topology_error(
            TopologyDiagnosticCode.INVALID_PRIMITIVE_VECTORS,
            "primitive vectors must be an iterable of Cartesian vectors",
            cause=exc,
        )
    if dimensionality not in (1, 2, 3) or len(vectors) != dimensionality:
        _raise_topology_error(
            TopologyDiagnosticCode.INVALID_PRIMITIVE_VECTORS,
            "primitive vectors must contain one vector per active dimension",
        )
    arrays: list[np.ndarray] = []
    for vector in vectors:
        try:
            array = np.array(vector, dtype=float, copy=True)
        except (TypeError, ValueError) as exc:
            _raise_topology_error(
                TopologyDiagnosticCode.INVALID_PRIMITIVE_VECTORS,
                "each primitive vector must be a finite Cartesian vector",
                cause=exc,
            )
        if array.shape != (3,) or not np.all(np.isfinite(array)):
            _raise_topology_error(
                TopologyDiagnosticCode.INVALID_PRIMITIVE_VECTORS,
                "each primitive vector must be a finite Cartesian vector",
            )
        arrays.append(array)
    matrix = np.asarray(arrays)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    vector_scale = float(singular_values[0])
    rank_threshold = max(
        float(geometry_tolerance),
        np.finfo(float).eps * max(matrix.shape) * vector_scale,
    )
    if float(singular_values[-1]) <= rank_threshold:
        _raise_topology_error(
            TopologyDiagnosticCode.INVALID_PRIMITIVE_VECTORS,
            "primitive vectors are linearly dependent within the declared tolerance",
        )
    return tuple(_readonly_copy(array) for array in arrays)


def _validated_replicas(
    replicas, dimensionality: int
) -> tuple[int, int, int]:
    try:
        values = tuple(replicas)
    except TypeError as exc:
        _raise_topology_error(
            TopologyDiagnosticCode.INVALID_REPLICAS,
            "replicas must contain exactly three positive integers",
            cause=exc,
        )
    if len(values) != 3 or any(not _is_strict_integer(value) for value in values):
        _raise_topology_error(
            TopologyDiagnosticCode.INVALID_REPLICAS,
            "replicas must contain exactly three positive integers",
        )
    result = tuple(int(value) for value in values)
    if any(value <= 0 for value in result) or any(
        result[axis] != 1 for axis in range(dimensionality, 3)
    ):
        _raise_topology_error(
            TopologyDiagnosticCode.INVALID_REPLICAS,
            "active replicas must be positive and inactive replicas must equal one",
        )
    return result


def _validated_site_ids(site_ids, expected_count: int | None = None) -> tuple[int, ...]:
    try:
        values = tuple(site_ids)
    except TypeError as exc:
        _raise_topology_error(
            TopologyDiagnosticCode.NONCONTIGUOUS_PRIMITIVE_SITE_IDS,
            "primitive site IDs must be explicit nonnegative integers",
            cause=exc,
        )
    if expected_count is not None and len(values) != expected_count:
        _raise_topology_error(
            TopologyDiagnosticCode.MISSING_CELL_COPY,
            "primitive site ID count must equal the topology atom count",
        )
    if any(not _is_strict_integer(value) or int(value) < 0 for value in values):
        _raise_topology_error(
            TopologyDiagnosticCode.NONCONTIGUOUS_PRIMITIVE_SITE_IDS,
            "primitive site IDs must be explicit nonnegative integers",
        )
    return tuple(int(value) for value in values)


def _validated_atom_cell_labels(
    labels,
    group: SECCMFiniteGroup,
    *,
    expected_count: int,
) -> tuple[FiniteTranslationLabel, ...]:
    try:
        values = tuple(labels)
    except TypeError as exc:
        _raise_topology_error(
            TopologyDiagnosticCode.NONCANONICAL_CELL_LABEL,
            "atom cell labels must be explicit canonical group labels",
            cause=exc,
        )
    if len(values) != expected_count:
        _raise_topology_error(
            TopologyDiagnosticCode.MISSING_CELL_COPY,
            "atom cell-label count must equal the topology atom count",
        )
    return tuple(group.require_canonical_label(label) for label in values)


def _validated_equivalence_keys(
    keys, *, expected_count: int
) -> tuple[AtomEquivalenceKey, ...]:
    try:
        values = tuple(keys)
    except TypeError as exc:
        _raise_topology_error(
            TopologyDiagnosticCode.MISSING_ATOM_EQUIVALENCE_KEY,
            "every atom requires an explicit equivalence key",
            cause=exc,
        )
    if len(values) != expected_count:
        _raise_topology_error(
            TopologyDiagnosticCode.MISSING_ATOM_EQUIVALENCE_KEY,
            "every atom requires an explicit equivalence key",
        )
    normalized: list[AtomEquivalenceKey] = []
    for key in values:
        if (
            not isinstance(key, tuple)
            or not key
            or any(not isinstance(component, str) for component in key)
        ):
            _raise_topology_error(
                TopologyDiagnosticCode.INVALID_ATOM_EQUIVALENCE_KEY,
                "atom equivalence keys must be nonempty tuples of strings",
            )
        normalized_key = tuple(unicodedata.normalize("NFC", part) for part in key)
        if any(not component for component in normalized_key):
            _raise_topology_error(
                TopologyDiagnosticCode.INVALID_ATOM_EQUIVALENCE_KEY,
                "atom equivalence keys cannot contain empty components",
            )
        normalized.append(normalized_key)
    return tuple(normalized)


def _validated_equivalence_key_schema(schema: str) -> str:
    if not isinstance(schema, str):
        _raise_topology_error(
            TopologyDiagnosticCode.INVALID_EQUIVALENCE_KEY_SCHEMA,
            "equivalence-key schema must be a nonempty versioned string",
        )
    normalized = unicodedata.normalize("NFC", schema)
    if not _VERSIONED_SCHEMA_PATTERN.fullmatch(normalized):
        _raise_topology_error(
            TopologyDiagnosticCode.INVALID_EQUIVALENCE_KEY_SCHEMA,
            "equivalence-key schema must be a nonempty versioned string",
        )
    return normalized


def _normalized_length_unit(length_unit: str) -> str:
    if not isinstance(length_unit, str):
        raise ValueError("SECCM finite-group length_unit must be 'bohr' or 'angstrom'")
    normalized = length_unit.strip().lower()
    if normalized not in _SUPPORTED_LENGTH_UNITS:
        raise ValueError("SECCM finite-group length_unit must be 'bohr' or 'angstrom'")
    return normalized


def _primitive_translation(
    group: SECCMFiniteGroup, label: FiniteTranslationLabel
) -> np.ndarray:
    label = group.require_canonical_label(label)
    result = np.zeros(3)
    for axis, vector in enumerate(group.primitive_vectors):
        result = result + label[axis] * vector
    return result


def _cyclic_translations_match_group(
    translations, group: SECCMFiniteGroup
) -> bool:
    if len(translations) != group.dimensionality:
        return False
    return all(
        np.allclose(
            np.asarray(translations[axis], dtype=float),
            group.replicas[axis] * group.primitive_vectors[axis],
            rtol=0.0,
            atol=group.geometry_tolerance,
        )
        for axis in range(group.dimensionality)
    )


def _coordinates_match_action(
    coords: np.ndarray,
    group: SECCMFiniteGroup,
    action: SECCMTranslationAction,
) -> bool:
    if len(coords) != len(action.atom_primitive_site_ids):
        return False
    zero_label = (0, 0, 0)
    atom_by_site_and_cell = {
        (site_id, label): atom_index
        for atom_index, (site_id, label) in enumerate(
            zip(
                action.atom_primitive_site_ids,
                action.atom_cell_labels,
                strict=True,
            )
        )
    }
    for atom_index, (site_id, label) in enumerate(
        zip(
            action.atom_primitive_site_ids,
            action.atom_cell_labels,
            strict=True,
        )
    ):
        zero_index = atom_by_site_and_cell[(site_id, zero_label)]
        expected = coords[zero_index] + _primitive_translation(group, label)
        if not np.allclose(
            coords[atom_index],
            expected,
            rtol=0.0,
            atol=group.geometry_tolerance,
        ):
            return False
    return True


def _current_group_compatibility(
    group: SECCMFiniteGroup,
    action: SECCMTranslationAction | None,
    coords: np.ndarray,
    translations,
) -> tuple[bool, bool | None]:
    lattice_compatible = _cyclic_translations_match_group(translations, group)
    if action is None:
        return lattice_compatible, None
    if not lattice_compatible:
        return False, False
    return True, _coordinates_match_action(coords, group, action)


def _is_strict_integer(value) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


def _raise_topology_error(
    code: TopologyDiagnosticCode,
    message: str,
    *,
    cause: BaseException | None = None,
) -> Never:
    error = SECCMTopologyError(f"{code.value}: {message}")
    if cause is not None:
        raise error from cause
    raise error


def _validate_geometry_metadata(
    length_unit: str | None, geometry_quantum: float | None
) -> None:
    if length_unit is not None and length_unit not in _SUPPORTED_LENGTH_UNITS:
        raise ValueError("SECCM length_unit must be 'bohr' or 'angstrom'")
    if geometry_quantum is not None:
        if length_unit is None:
            raise ValueError("geometry_quantum requires an explicit length_unit")
        if not math.isfinite(geometry_quantum) or geometry_quantum <= 0.0:
            raise ValueError("geometry_quantum must be positive and finite")


def _readonly_copy(array) -> np.ndarray:
    copied = np.array(array, dtype=float, copy=True)
    copied.setflags(write=False)
    return copied


def _missing_provenance_diagnostic() -> TopologyDiagnostic:
    return TopologyDiagnostic(
        TopologyDiagnosticCode.MISSING_BUILDER_PROVENANCE,
        "operation requires labels and candidate state from a known builder",
    )


def _canonical_sha256(payload: dict) -> str:
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(serialized.encode("ascii")).hexdigest()


def _decimal_string(value: float) -> str:
    decimal = Decimal(str(value)).normalize()
    return format(decimal, "f")


def _quantized_array_payload(array: np.ndarray, quantum: float) -> dict:
    if not np.all(np.isfinite(array)):
        raise ValueError("reference geometry must be finite")
    scaled = array / quantum
    quantized = np.where(
        scaled >= 0.0,
        np.floor(scaled + 0.5),
        np.ceil(scaled - 0.5),
    ).astype(np.int64)
    quantized[quantized == 0] = 0
    return {
        "shape": list(array.shape),
        "values": [int(value) for value in quantized.reshape(-1)],
    }


__all__ = [
    "AtomEquivalenceKey",
    "FiniteTranslationLabel",
    "ImageShellLabel",
    "SECCMCandidateClassification",
    "SECCMFiniteGroup",
    "SECCMImage",
    "SECCMTopology",
    "SECCMTopologyChangedError",
    "SECCMTopologyError",
    "SECCMTopologyProvenanceError",
    "SECCMTranslationAction",
    "TopologyDiagnostic",
    "TopologyDiagnosticCode",
    "WSNeighbor",
    "WignerSeitzCells",
    "bind_complete_translation_action",
    "bind_finite_group",
    "build_seccm_topology",
    "build_wigner_seitz",
]
