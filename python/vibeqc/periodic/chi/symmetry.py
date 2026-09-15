"""Space-group analysis for the finite AICCM2026DEV-B torus.

The primitive-cell space group does not automatically act on an arbitrary
Born--von Karman cluster.  For a diagonal cluster matrix
``N = diag(N1, N2, N3)``, a point operation ``W`` descends to the finite
translation quotient exactly when ``N^-1 W N`` is an integer matrix.  This
module applies that test, builds the resulting atom/cell permutations, and
partitions the exact Gamma-centred reciprocal net into symmetry orbits.

SCF integration remains intentionally diagnostic.  It does not replace the
full k net used by SCF and does not skip libint shell pairs or quartets.  AO
matrix averaging is exposed only at Gamma, where nonsymmorphic Bloch phases
are unity.  D127 additionally supplies an immutable compact spatial action on
real-torus AO coefficient rows, without connecting it to an energy path.
General-k AO sewing matrices and petite-list integral scattering must be
validated before either optimization can be enabled.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
import json
from math import prod
from typing import TYPE_CHECKING, Sequence

import numpy as np

from ..._vibeqc_core import BasisSet, PeriodicSystem
from ...symmetry_ao import AtomPermutation, build_ao_permutation_matrix
from ...symmetry_lattice import lattice_to_cartesian_rotation

if TYPE_CHECKING:
    from ...symmetry_shared import SpaceIdentity

_REAL_TORUS_AO_ACTION_SCHEMA = (
    "vibeqc.aiccm2026dev-b.real-torus-ao-action/v1"
)
_REAL_TORUS_AO_ACTION_MAX_MAPPING_RESIDUAL_BOHR = 1.0e-5
_REAL_TORUS_AO_ACTION_BASIS_ORIGIN_TOLERANCE_BOHR = 1.0e-10

__all__ = [
    "AICCM2026DevBSymmetryMode",
    "AICCM2026DevBSymmetryDiagnostics",
    "AICCM2026DevBSymmetryOperation",
    "AICCM2026DevBSymmetryPlan",
    "AICCM2026DevBRealTorusAOAction",
    "AICCM2026DevBOccupiedSymmetryAction",
    "AICCM2026DevBTorusSymmetryGroup",
    "AICCM2026DevBOccupiedGroupWitness",
    "AICCM2026DevBRestrictedSnapshot",
    "build_aiccm2026dev_b_restricted_snapshot",
    "build_aiccm2026dev_b_torus_symmetry_group",
    "build_aiccm2026dev_b_occupied_group_witness",
    "build_aiccm2026dev_b_occupied_symmetry_action",
    "build_aiccm2026dev_b_real_torus_ao_action",
    "build_aiccm2026dev_b_symmetry_plan",
    "gamma_matrix_symmetry_residual",
    "shell_pair_orbits",
    "shell_quartet_orbits",
    "symmetrize_gamma_ao_matrix",
]


@dataclass(frozen=True, init=False, slots=True, eq=False)
class AICCM2026DevBRestrictedSnapshot:
    """Bounded numerical bridge to the shared native all-k RHF contract.

    ``state`` can be consumed by the internal native orbital-sewing leaf.
    It is NOT a native SCF capture or an authenticated integral source. The
    supplied F/S/C/epsilon/occupations are checked, never reconstructed from
    eigenvalues or repaired. The native state identity binds those numbers;
    its calculation identity additionally binds the caller's declaration and
    the chi convention. Neither digest authenticates the declaration.

    See D130 in the chi decision log. No production correlation admission,
    representative-only evaluation, or gradient gate consumes this object.
    """

    state: object
    finite_torus_convention: object
    declared_calculation_identity: str
    estimated_snapshot_bytes: int

    def __init__(self) -> None:
        raise TypeError("use build_aiccm2026dev_b_restricted_snapshot")

    @property
    def physical_source_symmetry_certified(self) -> bool:
        return False

    def __copy__(self) -> AICCM2026DevBRestrictedSnapshot:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> AICCM2026DevBRestrictedSnapshot:
        return self


def build_aiccm2026dev_b_restricted_snapshot(
    result: object,
    system: PeriodicSystem,
    basis: BasisSet,
    *,
    declared_calculation_identity: str,
    frozen_core_bands: Sequence[int],
    minimum_band_gap_hartree: float,
    max_snapshot_bytes: int = 256 << 20,
) -> AICCM2026DevBRestrictedSnapshot:
    """Audit a complete chi RHF character mesh using the native state gate.

    This small-system diagnostic admits four-center, all-electron 3D RHF
    only. Frozen bands are explicit zero-based occupied indices, identical
    at every k (not an energy cutoff). Non-Aufbau, fractional occupations,
    stale/nonstationary Fock blocks and missing points fail closed. Actual
    k coordinates and weights are forwarded, not replaced by a fresh mesh.
    The native gate validates recorded weights before canonicalizing accepted
    roundoff to exact uniform 1/Nk; no matrix payload is repaired.

    The pre-copy inventory bounds additional explicit numerical storage for
    input assembly, native state construction/validation and one-k conversion
    work. It excludes caller-owned inputs, Python/control/allocator overhead
    and BLAS workspaces; it is not a process RSS limit. The native sewing leaf
    has its own complete inventory and must be separately admitted. Fixed
    diagnostic limits are 512 k points and 256 AOs. No target-sized run is
    authorized by this bridge.
    """
    from ... import _vibeqc_core as core
    from .scf import (
        AICCM2026DevBDiagnostics,
        _finite_torus_convention,
        cyclic_lattice_extension,
    )

    cap = _symmetry_positive_int(max_snapshot_bytes, "max_snapshot_bytes")
    identity = declared_calculation_identity
    if not isinstance(identity, str) or len(identity) != 64 or any(
        c not in "0123456789abcdef" for c in identity
    ):
        raise ValueError("declared_calculation_identity must be lowercase SHA-256-shaped")
    if isinstance(minimum_band_gap_hartree, (bool, np.bool_)) or not isinstance(
        minimum_band_gap_hartree, (int, float, np.integer, np.floating)
    ):
        raise ValueError("minimum_band_gap_hartree must be finite and positive")
    try:
        gap = float(minimum_band_gap_hartree)
    except OverflowError as exc:
        raise ValueError("minimum_band_gap_hartree must be finite and positive") from exc
    if not np.isfinite(gap) or gap <= 0:
        raise ValueError("minimum_band_gap_hartree must be finite and positive")
    diag = getattr(result, "aiccm2026dev_b", None)
    if not isinstance(diag, AICCM2026DevBDiagnostics) or diag.electronic_method != "RHF":
        raise ValueError("snapshot requires a chi RHF result")
    if (
        not isinstance(getattr(result, "converged", None), (bool, np.bool_))
        or not result.converged
    ):
        raise ValueError("snapshot requires a converged chi RHF result")
    if diag.backend != "four_center":
        raise NotImplementedError("chi snapshot currently admits four_center RHF only")
    if int(system.dim) != 3 or diag.ecp_total_ncore != 0:
        raise NotImplementedError("chi snapshot requires all-electron 3D RHF")
    if getattr(result, "functional", None) is not None:
        raise ValueError("a functional-bearing result is not an RHF snapshot")
    if diag.smearing_temperature != 0.0:
        raise ValueError("snapshot requires zero executed smearing")
    mesh = tuple(diag.mesh)
    if len(mesh) != 3:
        raise ValueError("snapshot requires three mesh extents")
    mesh = tuple(_symmetry_positive_int(v, "mesh extent") for v in mesh)
    nk, n = prod(mesh), int(basis.nbasis)
    if nk > 512 or not 1 <= n <= 256:
        raise ValueError("chi snapshot exceeds the 512-k/256-AO diagnostic limit")
    if diag.n_kpoints != nk or diag.n_cyclic_cells != nk:
        raise ValueError("snapshot full-mesh counts disagree")
    electrons = int(system.n_electrons())
    if electrons <= 0 or electrons % 2 or diag.effective_electron_count != electrons:
        raise ValueError("snapshot requires matching positive even electron counts")
    nocc = electrons // 2
    blocks = {}
    for name in ("overlap", "fock", "mo_coeffs", "mo_energies", "occupations"):
        values = getattr(result, name, None)
        if not isinstance(values, (list, tuple)) or len(values) != nk:
            raise ValueError(f"snapshot {name} must contain the complete character mesh")
        if any(
            not isinstance(v, np.ndarray)
            or v.dtype.kind not in "fc"
            or v.dtype.itemsize > (8 if v.dtype.kind == "f" else 16)
            for v in values
        ):
            raise ValueError(
                f"snapshot {name} requires NumPy blocks representable in binary64"
            )
        blocks[name] = values
    first = blocks["mo_coeffs"][0]
    if first.ndim != 2 or first.shape[0] != n:
        raise ValueError("snapshot coefficient shape disagrees with basis")
    r = first.shape[1]
    if not nocc < r <= n:
        raise ValueError("snapshot requires a retained virtual space")
    # Count-only native estimate and generous explicit-array validation work;
    # no payload conversion, basis expansion or native input exists yet.
    resident = core._estimate_periodic_restricted_mean_field_resident_bytes(mesh, n, r)
    estimated = (
        3 * resident + 16 * (12 * n * n + 12 * n * r + 12 * r * r)
        + 64 * nk * (r + 4)
    )
    if estimated > cap:
        raise MemoryError(f"chi snapshot needs {estimated} explicit numerical bytes; cap is {cap}")
    if (
        not isinstance(frozen_core_bands, (list, tuple, np.ndarray))
        or (isinstance(frozen_core_bands, np.ndarray) and frozen_core_bands.ndim != 1)
        or len(frozen_core_bands) > nocc
    ):
        raise ValueError("frozen_core_bands must be an explicit occupied-index sequence")
    frozen = []
    for band in frozen_core_bands:
        if (
            isinstance(band, (bool, np.bool_))
            or not isinstance(band, (int, np.integer))
            or not 0 <= band < nocc
        ):
            raise ValueError("frozen_core_bands must contain occupied integer indices")
        frozen.append(int(band))
    if len(set(frozen)) != len(frozen) or len(frozen) == nocc:
        raise ValueError("frozen_core_bands must be unique and leave correlated occupied bands")
    frozen = sorted(frozen)
    # This also validates the supplied basis origins. Cartesian shells are
    # supported by the native sewing leaf; do not use D127's pure-only helper.
    for shell in basis.shells():
        a = int(shell.atom_index)
        if not 0 <= a < len(system.unit_cell) or not np.allclose(
            shell.origin, system.unit_cell[a].xyz, rtol=0, atol=1e-10,
        ):
            raise ValueError("snapshot basis origins disagree with system")
    convention = _finite_torus_convention(
        system, mesh, cyclic_lattice_extension(system, mesh),
        full_range_exchange_coefficient=1.0,
    )
    if (
        diag.finite_torus_convention != convention
        or getattr(result, "finite_torus_convention", None) != convention
    ):
        raise ValueError("snapshot chi finite-torus convention disagrees with system/mesh/RHF")
    coordinates = getattr(result, "kpoints_cart", None)
    weights = getattr(result, "kpoint_weights", None)
    for name, value, shape in (
        ("kpoints_cart", coordinates, (nk, 3)), ("kpoint_weights", weights, (nk,)),
    ):
        if (
            not isinstance(value, np.ndarray) or value.dtype.kind != "f"
            or value.dtype.itemsize > 8 or value.shape != shape
        ):
            raise ValueError(f"snapshot requires recorded {name} with shape {shape}")
    for name, values in blocks.items():
        if name == "mo_coeffs":
            shape = (n, r)
        elif name in ("mo_energies", "occupations"):
            shape = (r,)
        else:
            shape = (n, n)
        for block in values:
            if block.shape != shape or not np.all(np.isfinite(block)):
                raise ValueError(f"snapshot {name} has an invalid shape or non-finite payload")
            if name in ("mo_energies", "occupations") and block.dtype.kind != "f":
                raise ValueError(f"snapshot {name} must be real floating data")
    data = core._PeriodicRestrictedMeanFieldInput()
    data.calculation_identity = sha256(json.dumps({
        "schema": "vibeqc.chi.restricted-snapshot-declaration/v1",
        "declared_calculation_identity": identity,
        "finite_torus_convention": asdict(convention),
        "frozen_core_bands": frozen,
    }, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")).hexdigest()
    data.periodic_dimension = 3
    data.mesh = mesh
    data.is_shift = (0, 0, 0)
    data.reciprocal_lattice = system.reciprocal_lattice()
    data.converged = True  # numerical audit only, never a native production capture
    data.n_basis = n
    data.n_effective_orbitals = r
    data.electrons_per_cell = electrons
    data.reference_energy_per_cell = result.energy
    data.minimum_band_gap_hartree = gap
    frozen_mask = [int(b in frozen) for b in range(r)]
    active_mask = [int(b < nocc and b not in frozen) for b in range(r)]
    virtual_mask = [int(b >= nocc) for b in range(r)]
    for k in range(nk):
        data.add_kpoint(
            k_cartesian=coordinates[k], weight=weights[k],
            overlap=blocks["overlap"][k], fock=blocks["fock"][k],
            coefficients=blocks["mo_coeffs"][k], orbital_energies=blocks["mo_energies"][k],
            occupations=blocks["occupations"][k], frozen_core_mask=frozen_mask,
            correlated_occupied_mask=active_mask, virtual_mask=virtual_mask,
        )
    state = core._make_periodic_restricted_mean_field_state(data)
    snapshot = object.__new__(AICCM2026DevBRestrictedSnapshot)
    for name, value in (
        ("state", state), ("finite_torus_convention", convention),
        ("declared_calculation_identity", identity), ("estimated_snapshot_bytes", estimated),
    ):
        object.__setattr__(snapshot, name, value)
    return snapshot


class AICCM2026DevBSymmetryMode(str, Enum):
    """Supported B-stream symmetry behavior.

    ``DIAGNOSTIC`` constructs and verifies the symmetry plan but deliberately
    leaves the SCF's full k net and integral build unchanged.  ``INTEGRALS``
    names the requested future petite-list route and fails closed today.
    """

    OFF = "off"
    DIAGNOSTIC = "diagnostic"
    INTEGRALS = "integrals"


@dataclass(frozen=True)
class AICCM2026DevBSymmetryOperation:
    """One cluster-compatible ``{W|w}`` operation and its atom mapping.

    ``atom_lattice_shifts[a]`` is the integer vector ``q_a`` defined by
    ``W f_a + w = f_perm[a] + q_a``.  Thus a basis center on cell ``r``
    maps to cell ``W r + q_a`` modulo the cyclic mesh.
    """

    full_group_index: int
    rotation: np.ndarray
    translation: np.ndarray
    atom_permutation: np.ndarray
    atom_lattice_shifts: np.ndarray
    max_atom_mapping_residual_bohr: float
    has_fractional_translation: bool


@dataclass(frozen=True)
class AICCM2026DevBSymmetryPlan:
    """Verified symmetry metadata for one finite cyclic cluster."""

    mesh: tuple[int, int, int]
    space_group_number: int
    international_symbol: str
    hall_number: int
    point_group: str
    wyckoff_letters: tuple[str, ...]
    site_symmetry_symbols: tuple[str, ...]
    equivalent_atoms: tuple[int, ...]
    n_operations_full: int
    operations: tuple[AICCM2026DevBSymmetryOperation, ...]
    incompatible_operation_indices: tuple[int, ...]
    full_kpoints_frac: np.ndarray
    full_to_irreducible: np.ndarray
    irreducible_representative_indices: np.ndarray
    irreducible_weights: np.ndarray
    time_reversal: bool
    acceleration_applied: bool = False
    ao_acceleration_status: str = (
        "diagnostic only; general-k AO sewing and libint petite lists disabled"
    )

    @property
    def n_operations_compatible(self) -> int:
        return len(self.operations)

    @property
    def n_kpoints_full(self) -> int:
        return int(self.full_kpoints_frac.shape[0])

    @property
    def n_kpoints_irreducible(self) -> int:
        return int(self.irreducible_representative_indices.size)

    def map_cell(
        self,
        operation_index: int,
        atom_index: int,
        cell: Sequence[int],
    ) -> tuple[int, tuple[int, int, int]]:
        """Map ``(atom, cell)`` through one compatible operation."""

        operation = self.operations[operation_index]
        r = np.asarray(cell, dtype=np.int64).reshape(3)
        image = operation.rotation @ r + operation.atom_lattice_shifts[atom_index]
        residue = np.mod(image, np.asarray(self.mesh, dtype=np.int64))
        return (
            int(operation.atom_permutation[atom_index]),
            tuple(int(value) for value in residue),
        )


@dataclass(frozen=True)
class AICCM2026DevBSymmetryDiagnostics:
    """Post-SCF witness for the non-mutating diagnostic route."""

    plan: AICCM2026DevBSymmetryPlan
    gamma_fock_residual: float | None
    gamma_density_residual: float | None
    n_shell_pairs: int
    n_unique_shell_pairs: int
    n_shell_quartets: int | None
    n_unique_shell_quartets: int | None
    energy_change_hartree: float = 0.0


@dataclass(frozen=True, init=False, slots=True, eq=False)
class AICCM2026DevBRealTorusAOAction:
    """Compact action of one space-group operation on real-torus AOs.

    ``primitive_ao_action`` follows the repository convention
    ``P[destination AO, source AO]``.  ``cell_images[r, a]`` is the
    destination cell of source atom ``a`` in source cell ``r``.  Keeping
    those two factors separate avoids the dense
    ``(n_cells * nbf) x (n_cells * nbf)`` torus operator.

    ``atom_reference_cell_offsets`` records how the supplied atom
    representatives differ from the wrapped primitive-cell representatives;
    ``atom_lattice_shifts`` contains the corrected shifts that act on those
    supplied representatives.

    The object is factory-only and its arrays are backed by immutable Python
    ``bytes`` objects.  Copies therefore safely share the same immutable
    instance.  ``fingerprint`` identifies this compact action payload, not the
    complete system/basis state or a stable crystallographic operation ID.  It
    is a diagnostic representation primitive: it does not attest an
    occupied-orbital gauge, reduce a pair list, or enter an SCF or
    correlation-energy path.
    """

    mesh: tuple[int, int, int]
    operation_index: int
    full_group_index: int
    rotation: np.ndarray
    translation: np.ndarray
    atom_permutation: np.ndarray
    atom_lattice_shifts: np.ndarray
    atom_reference_cell_offsets: np.ndarray
    ao_atom_indices: np.ndarray
    primitive_ao_action: np.ndarray
    cell_images: np.ndarray
    schema: str = _REAL_TORUS_AO_ACTION_SCHEMA
    fingerprint: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError(
            "AICCM2026DevBRealTorusAOAction is factory-only; use "
            "build_aiccm2026dev_b_real_torus_ao_action(...)"
        )

    def __copy__(self) -> AICCM2026DevBRealTorusAOAction:
        return self

    def __deepcopy__(
        self,
        memo: dict[int, object],
    ) -> AICCM2026DevBRealTorusAOAction:
        memo[id(self)] = self
        return self

    @classmethod
    def _from_payload(
        cls,
        *,
        mesh: tuple[int, int, int],
        operation_index: int,
        full_group_index: int,
        rotation: np.ndarray,
        translation: np.ndarray,
        atom_permutation: np.ndarray,
        atom_lattice_shifts: np.ndarray,
        atom_reference_cell_offsets: np.ndarray,
        ao_atom_indices: np.ndarray,
        primitive_ao_action: np.ndarray,
        cell_images: np.ndarray,
    ) -> AICCM2026DevBRealTorusAOAction:
        instance = object.__new__(cls)
        object.__setattr__(instance, "mesh", tuple(int(value) for value in mesh))
        object.__setattr__(instance, "operation_index", operation_index)
        object.__setattr__(instance, "full_group_index", full_group_index)
        object.__setattr__(
            instance,
            "rotation",
            _immutable_numeric_array(rotation, np.dtype(np.int64)),
        )
        object.__setattr__(
            instance,
            "translation",
            _immutable_numeric_array(translation, np.dtype(np.float64)),
        )
        object.__setattr__(
            instance,
            "atom_permutation",
            _immutable_numeric_array(atom_permutation, np.dtype(np.int64)),
        )
        object.__setattr__(
            instance,
            "atom_lattice_shifts",
            _immutable_numeric_array(atom_lattice_shifts, np.dtype(np.int64)),
        )
        object.__setattr__(
            instance,
            "atom_reference_cell_offsets",
            _immutable_numeric_array(
                atom_reference_cell_offsets,
                np.dtype(np.int64),
            ),
        )
        object.__setattr__(
            instance,
            "ao_atom_indices",
            _immutable_numeric_array(ao_atom_indices, np.dtype(np.int64)),
        )
        object.__setattr__(
            instance,
            "primitive_ao_action",
            _immutable_numeric_array(primitive_ao_action, np.dtype(np.float64)),
        )
        object.__setattr__(
            instance,
            "cell_images",
            _immutable_numeric_array(cell_images, np.dtype(np.int64)),
        )
        object.__setattr__(instance, "schema", _REAL_TORUS_AO_ACTION_SCHEMA)
        object.__setattr__(instance, "fingerprint", instance._build_fingerprint())
        return instance

    def _build_fingerprint(self) -> str:
        digest = sha256()
        digest.update(self.schema.encode("ascii"))
        digest.update(b"\0")
        for value in (
            *self.mesh,
            self.operation_index,
            self.full_group_index,
            self.n_atoms,
            self.nbasis,
        ):
            digest.update(int(value).to_bytes(8, "big", signed=False))
        for array in (
            self.rotation,
            self.atom_permutation,
            self.atom_lattice_shifts,
            self.atom_reference_cell_offsets,
            self.ao_atom_indices,
            self.cell_images,
        ):
            digest.update(np.asarray(array, dtype=">i8").tobytes(order="C"))
        for array in (self.translation, self.primitive_ao_action):
            canonical = np.array(array, dtype=np.float64, order="C", copy=True)
            canonical[canonical == 0.0] = 0.0
            digest.update(np.asarray(canonical, dtype=">f8").tobytes(order="C"))
        return digest.hexdigest()

    @property
    def n_cells(self) -> int:
        return prod(self.mesh)

    @property
    def n_atoms(self) -> int:
        return int(self.atom_permutation.size)

    @property
    def nbasis(self) -> int:
        return int(self.ao_atom_indices.size)

    @property
    def torus_nbasis(self) -> int:
        return self.n_cells * self.nbasis

    @property
    def payload_nbytes(self) -> int:
        """Bytes retained by the compact numeric payload."""

        return sum(
            int(array.nbytes)
            for array in (
                self.rotation,
                self.translation,
                self.atom_permutation,
                self.atom_lattice_shifts,
                self.atom_reference_cell_offsets,
                self.ao_atom_indices,
                self.primitive_ao_action,
                self.cell_images,
            )
        )

    def apply(self, coefficients: np.ndarray) -> np.ndarray:
        """Apply the point operation to AO-row coefficient vectors.

        ``coefficients`` must have shape ``(n_cells * nbf, n_vectors)`` in
        deterministic C-order cell blocks.  The returned array has the same
        shape.  Apart from that result, only atom-block-sized temporaries are
        used; no dense torus action is materialized.
        """

        array = np.asarray(coefficients)
        if array.ndim != 2:
            raise ValueError("real-torus AO coefficients must be two-dimensional")
        expected_rows = self.torus_nbasis
        if array.shape[0] != expected_rows:
            raise ValueError(
                "real-torus AO coefficients must have "
                f"{expected_rows} rows; got {array.shape[0]}"
            )
        if array.dtype.kind not in "biufc":
            raise ValueError("real-torus AO coefficients must be numeric")

        n_vectors = int(array.shape[1])
        result = np.zeros(
            (expected_rows, n_vectors),
            dtype=np.result_type(array.dtype, np.float64),
        )
        for source_atom in range(self.n_atoms):
            source_aos = np.flatnonzero(self.ao_atom_indices == source_atom)
            destination_atom = int(self.atom_permutation[source_atom])
            destination_aos = np.flatnonzero(
                self.ao_atom_indices == destination_atom
            )
            block = self.primitive_ao_action[
                np.ix_(destination_aos, source_aos)
            ]
            for source_cell, destination_cell in enumerate(
                self.cell_images[:, source_atom]
            ):
                source_rows = source_cell * self.nbasis + source_aos
                destination_rows = (
                    int(destination_cell) * self.nbasis + destination_aos
                )
                result[destination_rows, :] = block @ array[source_rows, :]
        return result


@dataclass(frozen=True, slots=True, init=False, eq=False)
class AICCM2026DevBOccupiedSymmetryAction:
    """One spatial action in a supplied orthonormal occupied gauge.

    Rows are destination orbitals, columns source orbitals. A monomial
    action satisfies ``matrix[permutation[j], j] ~= phases[j]``. The full
    matrix is always retained, without rounding or replacing its entries.
    General mixing supplies neither a permutation nor phases. This is a
    numerical covariance witness for one operation, not a group seal or
    permission to omit pairs in a correlation calculation.
    """

    matrix: np.ndarray
    permutation: np.ndarray | None
    phases: np.ndarray | None
    ao_action_fingerprint: str
    input_fingerprint: str
    fingerprint: str
    tolerance: float
    metric_covariance_residual: float
    orthonormality_residual: float
    closure_residual: float
    unitarity_residual: float
    monomial_residual: float
    estimated_workspace_bytes: int

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError(
            "occupied symmetry actions are factory-only; use "
            "build_aiccm2026dev_b_occupied_symmetry_action(...)"
        )

    def __copy__(self) -> AICCM2026DevBOccupiedSymmetryAction:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> AICCM2026DevBOccupiedSymmetryAction:
        memo[id(self)] = self
        return self

    @property
    def is_monomial(self) -> bool:
        return self.permutation is not None

    @property
    def citation_numerics(self) -> tuple[str, ...]:
        """Tokens for the shared citation assembler used by diagnostic callers."""
        return ("chi_occupied_symmetry",)


def _occupied_action_array_fingerprint(*arrays: np.ndarray) -> str:
    digest = sha256(b"vibeqc.chi.occupied-action-input/v1\0")
    for array in arrays:
        digest.update(np.asarray(array.shape, dtype=">u8").tobytes())
        # Canonical complex wire: interleaved real/imag, normalized signed zero.
        wire = np.array(array, dtype=np.complex128, order="C", copy=True)
        components = wire.view(np.float64)
        components[components == 0.0] = 0.0
        digest.update(components.astype(">f8").tobytes())
    return digest.hexdigest()


def build_aiccm2026dev_b_occupied_symmetry_action(
    action: AICCM2026DevBRealTorusAOAction,
    localization: object,
    *,
    tolerance: float = 1.0e-8,
    max_workspace_bytes: int = 512 * 1024**2,
) -> AICCM2026DevBOccupiedSymmetryAction:
    """Project a D127 action into an actual localized occupied space.

    Uses the existing localization result's ``coefficients`` and ``overlap``;
    neither is repaired or symmetrized. Positive-definite overlap is required
    in this first diagnostic envelope. ``max_workspace_bytes`` bounds a
    conservative inventory of explicit dense NumPy arrays, excluding the
    caller-owned inputs and BLAS/LAPACK internal workspaces; it is not an RSS
    limit. No dense torus AO action is formed. The localizer's dense C/S and
    the occupied action itself still limit this diagnostic to small systems.

    The input digest binds the numerical C/S payload, not a full SCF/basis/run
    fingerprint. It does not certify stationarity, localization optimality,
    the entire group, or PAO/PNO/amplitude covariance. These remain necessary
    before D126's support quotient can enter representative energy execution.
    """
    from .localization import AICCM2026DevBLocalizationResult

    if not isinstance(action, AICCM2026DevBRealTorusAOAction):
        raise TypeError("occupied symmetry requires a D127 real-torus AO action")
    if not isinstance(localization, AICCM2026DevBLocalizationResult):
        raise TypeError("occupied symmetry requires a chi localization result")
    if isinstance(tolerance, (bool, np.bool_)) or not isinstance(
        tolerance, (int, float, np.integer, np.floating)
    ):
        raise ValueError("occupied symmetry tolerance must be a finite positive real")
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or not 0.0 < tolerance <= 1.0e-6:
        raise ValueError("occupied symmetry tolerance must lie in (0, 1e-6]")
    if isinstance(max_workspace_bytes, (bool, np.bool_)) or not isinstance(
        max_workspace_bytes, (int, np.integer)
    ) or max_workspace_bytes < 1:
        raise ValueError("max_workspace_bytes must be a positive integer")
    n = action.torus_nbasis
    for label, value in (
        ("n_cells", localization.n_cells),
        ("n_occ_per_cell", localization.n_occ_per_cell),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ) or value < 1:
            raise ValueError(f"localization {label} must be a positive integer")
    if localization.n_cells != action.n_cells:
        raise ValueError("localization cell count differs from the AO action")
    m = int(localization.n_occ_per_cell) * action.n_cells
    if m > n:
        raise ValueError("occupied dimension exceeds the AO dimension")
    # Inspect already-allocated arrays before converting/copying any payload.
    for label, array, shape in (
        ("coefficients", localization.coefficients, (n, m)),
        ("overlap", localization.overlap, (n, n)),
        ("translations", localization.translations, (action.n_cells, 3)),
    ):
        if not isinstance(array, np.ndarray) or array.shape != shape:
            raise ValueError(f"localization {label} must be a NumPy array of shape {shape}")
        allowed = "iu" if label == "translations" else "fc"
        if array.dtype.kind not in allowed:
            raise ValueError(f"localization {label} has an unsupported numeric dtype")
    # Includes snapshots, covariant-metric check, products, residuals and hash
    # serialization. Python integers avoid overflow in admission arithmetic.
    workspace = 16 * (8 * n * n + 12 * n * m + 12 * m * m)
    if workspace > max_workspace_bytes:
        raise MemoryError(
            f"occupied symmetry explicit-array estimate {workspace} bytes "
            f"exceeds max_workspace_bytes={max_workspace_bytes}"
        )
    for index, cell in enumerate(np.ndindex(action.mesh)):
        if tuple(localization.translations[index]) != cell:
            raise ValueError("localization translations must match AO-action C-order cells")
    c = np.array(localization.coefficients, dtype=np.complex128, copy=True)
    s = np.array(localization.overlap, dtype=np.complex128, copy=True)
    if not np.all(np.isfinite(c)) or not np.all(np.isfinite(s)):
        raise ValueError("occupied symmetry coefficients/overlap must be finite")

    def residual(array: np.ndarray, scale: float = 1.0) -> float:
        value = float(np.linalg.norm(array) / scale)
        if not np.isfinite(value):
            raise ValueError("occupied symmetry produced a nonfinite residual")
        return value

    s_norm = float(np.linalg.norm(s))
    if not np.isfinite(s_norm) or s_norm == 0.0:
        raise ValueError("occupied symmetry overlap must have a finite nonzero norm")
    if residual(s - s.conj().T, s_norm) > tolerance:
        raise ValueError("occupied symmetry overlap must be Hermitian")
    try:
        np.linalg.cholesky(s)
    except np.linalg.LinAlgError as exc:
        raise ValueError("occupied symmetry overlap must be positive definite") from exc
    # U is real orthogonal on AO coefficient rows. U S U^dagger = S is
    # equivalent to U^dagger S U = S, without building a dense AO U.
    moved_s = action.apply(action.apply(s).conj().T).conj().T
    metric_error = residual(moved_s - s, s_norm)
    if metric_error > tolerance:
        raise ValueError("AO action does not preserve the supplied overlap metric")
    del moved_s
    identity = np.eye(m)
    orth_error = residual(c.conj().T @ s @ c - identity)
    if orth_error > tolerance:
        raise ValueError("occupied coefficients are not overlap-orthonormal")
    moved_c = action.apply(c)
    # Casassa et al., TCA 116, 726 (2006), DOI 10.1007/s00214-006-0119-z,
    # Eq. (3): a symmetry may mix petals. In a finite orthonormal occupied
    # basis its coefficients are D_g = C^dagger S U_g C, destination/source.
    matrix = c.conj().T @ s @ moved_c
    closure_error = residual(moved_c - c @ matrix, max(float(np.linalg.norm(c)), 1.0))
    unitary_error = residual(matrix.conj().T @ matrix - identity)
    if closure_error > tolerance or unitary_error > tolerance:
        raise ValueError("AO action does not close in the supplied occupied space")
    # Eq. (6)'s single-petal +/- case generalizes to unit complex phases.
    # Keep the complete computed D even when this monomial diagnostic passes.
    permutation = np.argmax(np.abs(matrix), axis=0)
    selected = matrix[permutation, np.arange(m)]
    phases = selected / np.where(np.abs(selected) > 0.0, np.abs(selected), 1.0)
    monomial = np.zeros_like(matrix)
    monomial[permutation, np.arange(m)] = phases
    monomial_error = residual(matrix - monomial)
    is_monomial = len(np.unique(permutation)) == m and monomial_error <= tolerance
    input_fingerprint = _occupied_action_array_fingerprint(c, s)
    matrix_fingerprint = _occupied_action_array_fingerprint(matrix)
    digest = sha256(b"vibeqc.chi.occupied-symmetry-action/v1\0")
    for token in (action.fingerprint, input_fingerprint, matrix_fingerprint, tolerance.hex()):
        digest.update(token.encode("ascii") + b"\0")
    instance = object.__new__(AICCM2026DevBOccupiedSymmetryAction)
    payload = {
        "matrix": _immutable_numeric_array(matrix, np.dtype(np.complex128)),
        "permutation": (
            _immutable_numeric_array(permutation, np.dtype(np.int64))
            if is_monomial else None
        ),
        "phases": (
            _immutable_numeric_array(phases, np.dtype(np.complex128))
            if is_monomial else None
        ),
        "ao_action_fingerprint": action.fingerprint,
        "input_fingerprint": input_fingerprint,
        "fingerprint": digest.hexdigest(),
        "tolerance": tolerance,
        "metric_covariance_residual": metric_error,
        "orthonormality_residual": orth_error,
        "closure_residual": closure_error,
        "unitarity_residual": unitary_error,
        "monomial_residual": monomial_error,
        "estimated_workspace_bytes": workspace,
    }
    for name, value in payload.items():
        object.__setattr__(instance, name, value)
    return instance


@dataclass(frozen=True, slots=True, init=False, eq=False)
class AICCM2026DevBTorusSymmetryGroup:
    """Closed supplied spatial cosets, with their integer translation cocycle.

    ``products[g,h] = k`` and ``lattice_cocycle[g,h] = ell`` mean
    ``U_g U_h = T_ell U_k`` (h acts first). This validates the supplied
    quotient, not its completeness relative to the crystal's space group.
    Time reversal is not included. No orbital or Hamiltonian is certified.
    """

    actions: tuple[AICCM2026DevBRealTorusAOAction, ...]
    products: np.ndarray
    lattice_cocycle: np.ndarray
    inverses: np.ndarray
    identity_index: int
    tolerance: float
    maximum_seitz_residual_bohr: float
    maximum_ao_product_residual: float
    estimated_workspace_bytes: int
    group_products_checked: int
    fingerprint: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("torus symmetry groups are factory-only")

    def __copy__(self) -> AICCM2026DevBTorusSymmetryGroup:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> AICCM2026DevBTorusSymmetryGroup:
        memo[id(self)] = self
        return self

    @property
    def citation_numerics(self) -> tuple[str, ...]:
        return ("chi_torus_symmetry_group",)


def _symmetry_positive_int(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _symmetry_tolerance(value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError("symmetry tolerance must lie in (0, 1e-6]")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError("symmetry tolerance must lie in (0, 1e-6]") from exc
    if not np.isfinite(result) or not 0.0 < result <= 1e-6:
        raise ValueError("symmetry tolerance must lie in (0, 1e-6]")
    return result


def _symmetry_residual(array: np.ndarray) -> float:
    value = float(np.linalg.norm(array))
    if not np.isfinite(value):
        raise ValueError("symmetry algebra produced a nonfinite residual")
    return value


def build_aiccm2026dev_b_torus_symmetry_group(
    system: PeriodicSystem,
    basis: BasisSet,
    plan: AICCM2026DevBSymmetryPlan,
    *,
    tolerance: float = 1e-8,
    max_workspace_bytes: int = 256 * 1024**2,
    max_group_products: int = 10_000_000,
) -> AICCM2026DevBTorusSymmetryGroup:
    """Validate all spatial products, including nonsymmorphic translations.

    The finite translation subgroup is implicit: no (N_c |G|)^2 table is
    built. Integer atom-image arithmetic determines the cocycle, never a
    rounded Bloch phase. Numerical AO and Cartesian Seitz residuals use the
    declared tolerance (Frobenius and bohr respectively). Up to 192 supplied
    cosets are admitted. The byte estimate is a conservative explicit-array
    inventory, not an allocator/RSS or BLAS-workspace guarantee.
    """
    tolerance = _symmetry_tolerance(tolerance)
    cap = _symmetry_positive_int(max_workspace_bytes, "max_workspace_bytes")
    work_cap = _symmetry_positive_int(max_group_products, "max_group_products")
    if not isinstance(plan, AICCM2026DevBSymmetryPlan) or not isinstance(plan.operations, tuple):
        raise TypeError("torus symmetry group requires a plan with a tuple of operations")
    size = len(plan.operations)
    if not 1 <= size <= 192:
        raise ValueError("torus symmetry group requires 1..192 spatial cosets")
    if len(plan.mesh) != 3:
        raise ValueError("torus symmetry group requires a three-dimensional mesh")
    mesh = tuple(_symmetry_positive_int(v, "mesh extent") for v in plan.mesh)
    n_cells, n_atoms, b = prod(mesh), len(system.unit_cell), int(basis.nbasis)
    # Includes retained actions, product/cocycle snapshots and scratch, and
    # single-action construction. Python integers make admission overflow-free.
    workspace = 64 * (size * (b*b + n_cells*n_atoms + 32*n_atoms + 4*b + 128)
                      + 16*size*size + 6*n_cells + b*b)
    products_checked = size**2 + size**3
    if workspace > cap:
        raise MemoryError("torus symmetry group explicit-array estimate exceeds max_workspace_bytes")
    if products_checked > work_cap:
        raise ValueError("torus symmetry group exceeds max_group_products")
    actions = tuple(build_aiccm2026dev_b_real_torus_ao_action(system, basis, plan, i)
                    for i in range(size))
    # Exact Python integer intermediates avoid int64 wrap on malformed plans.
    rotations = [a.rotation.astype(object) for a in actions]
    shifts = [a.atom_lattice_shifts.astype(object) for a in actions]
    def key(rotation, permutation):
        return tuple(int(v) for v in rotation.flat), tuple(int(v) for v in permutation)
    lookup = {}
    for i, a in enumerate(actions):
        signature = key(a.rotation, a.atom_permutation)
        if signature in lookup:
            raise ValueError("duplicate spatial coset (modulo lattice translations)")
        lookup[signature] = i
    identity = lookup.get(key(np.eye(3, dtype=int), np.arange(n_atoms)))
    if identity is None or np.any(shifts[identity] != 0):
        raise ValueError("spatial group requires a normalized zero-translation identity")
    products = np.empty((size, size), dtype=np.int64)
    cocycle = np.empty((size, size, 3), dtype=np.int64)
    seitz_error = ao_error = 0.0
    lattice = np.asarray(system.lattice)
    for g, ag in enumerate(actions):
        for h, ah in enumerate(actions):
            permutation = ag.atom_permutation[ah.atom_permutation]
            k = lookup.get(key(rotations[g] @ rotations[h], permutation))
            if k is None:
                raise ValueError("supplied spatial cosets are not closed")
            # Active Seitz action: q_gh,a = W_g q_h,a + q_g,pi_h(a).
            # Hence g h = {I|ell(g,h)} k. Atom representative offsets cancel.
            delta = shifts[h] @ rotations[g].T + shifts[g][ah.atom_permutation] - shifts[k]
            if not np.all(delta == delta[0]):
                raise ValueError("spatial product has atom-dependent lattice cocycle")
            ell = [int(v) for v in delta[0]]
            if any(abs(v) > 2**52 for v in ell):
                raise ValueError("lattice cocycle exceeds exact binary64 integer range")
            products[g, h], cocycle[g, h] = k, ell
            fractional_error = (ag.rotation @ ah.translation + ag.translation
                                - actions[k].translation - np.asarray(ell, dtype=float))
            seitz_error = max(seitz_error, _symmetry_residual(lattice @ fractional_error))
            ao_error = max(ao_error, _symmetry_residual(
                ag.primitive_ao_action @ ah.primitive_ao_action - actions[k].primitive_ao_action))
    if seitz_error > tolerance or ao_error > tolerance:
        raise ValueError("spatial Seitz/AO product residual exceeds tolerance")
    # Shared exact group/extension validator. Existing Seitz/AO audits above
    # remain chi-owned. Existing table/scratch and per-action control
    # reservations cover the native tables and inverse snapshot; retain the
    # published workspace cap and product count.
    from ...symmetry_shared import Budget, FiniteGroup
    group = FiniteGroup.from_table(
        np.ascontiguousarray(products), np.zeros(size, dtype=np.uint8),
        identity=int(identity), identity_label="chi supplied spatial quotient",
        rotations=np.ascontiguousarray([a.rotation for a in actions], dtype=np.int64),
        cocycle=np.ascontiguousarray(cocycle),
        budget=Budget(min(cap, 2**63-1), 256*products_checked + 128*size*size),
    )
    inverses = np.asarray([group.inverse(g) for g in range(size)], dtype=np.int64)
    digest = sha256(b"vibeqc.chi.torus-symmetry-group/v1\0")
    for a in actions:
        digest.update(a.fingerprint.encode("ascii") + b"\0")
    digest.update(tolerance.hex().encode("ascii"))
    for array in (products, cocycle, inverses):
        digest.update(np.asarray(array, dtype=">i8").tobytes())
    result = object.__new__(AICCM2026DevBTorusSymmetryGroup)
    for name, value in dict(
        actions=actions, products=_immutable_numeric_array(products, np.dtype(np.int64)),
        lattice_cocycle=_immutable_numeric_array(cocycle, np.dtype(np.int64)),
        inverses=_immutable_numeric_array(inverses, np.dtype(np.int64)),
        identity_index=int(identity), tolerance=tolerance,
        maximum_seitz_residual_bohr=seitz_error, maximum_ao_product_residual=ao_error,
        estimated_workspace_bytes=workspace, group_products_checked=products_checked,
        fingerprint=digest.hexdigest(),
    ).items():
        object.__setattr__(result, name, value)
    return result


@dataclass(frozen=True, slots=True, init=False, eq=False)
class AICCM2026DevBOccupiedGroupWitness:
    """Numerical spatial/translation representation in a supplied occupied gauge.

    This is not a frozen/active/virtual, Fock, domain or amplitude certificate.
    It does not permit representative-only energy execution.
    """

    group: AICCM2026DevBTorusSymmetryGroup
    occupied_actions: tuple[AICCM2026DevBOccupiedSymmetryAction, ...]
    translation_generators: np.ndarray
    maximum_translation_residual: float
    maximum_group_residual: float
    tolerance: float
    estimated_workspace_bytes: int
    fingerprint: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("occupied group witnesses are factory-only")

    def __copy__(self) -> AICCM2026DevBOccupiedGroupWitness:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> AICCM2026DevBOccupiedGroupWitness:
        memo[id(self)] = self
        return self

    @property
    def citation_numerics(self) -> tuple[str, ...]:
        return ("chi_torus_symmetry_group", "chi_occupied_symmetry")


def build_aiccm2026dev_b_occupied_group_witness(
    group: AICCM2026DevBTorusSymmetryGroup,
    localization: object,
    *,
    tolerance: float = 1e-8,
    max_workspace_bytes: int = 512 * 1024**2,
    max_group_products: int = 100_000,
) -> AICCM2026DevBOccupiedGroupWitness:
    """Check all occupied spatial products and finite-translation generators.

    D128 supplies D_g in one common C/S gauge; T_ell is projected from actual
    AO cell translations, not guessed occupied permutations. Full matrices,
    including nonmonomial gauges, are retained. No dense all-AO U is formed,
    but dense C/S and |G| occupied matrices still restrict this to diagnostics.
    """
    from .localization import AICCM2026DevBLocalizationResult

    if not isinstance(group, AICCM2026DevBTorusSymmetryGroup):
        raise TypeError("occupied group witness requires a torus symmetry group")
    if not isinstance(localization, AICCM2026DevBLocalizationResult):
        raise TypeError("occupied group witness requires a chi localization result")
    tolerance = _symmetry_tolerance(tolerance)
    cap = _symmetry_positive_int(max_workspace_bytes, "max_workspace_bytes")
    work_cap = _symmetry_positive_int(max_group_products, "max_group_products")
    a = group.actions[0]
    n, size = a.torus_nbasis, len(group.actions)
    m = a.n_cells * _symmetry_positive_int(localization.n_occ_per_cell, "n_occ_per_cell")
    if m > n:
        raise ValueError("occupied dimension exceeds AO dimension")
    # Three matrix-equivalents per retained action also cover its optional
    # permutation/phase vectors, including the smallest m=1 case.
    workspace = group.estimated_workspace_bytes + 16 * (
        12*n*n + 20*n*m + (24+3*size)*m*m)
    if workspace > cap:
        raise MemoryError("occupied group explicit-array estimate exceeds max_workspace_bytes")
    if size**2 + 3*size + 12 > work_cap:
        raise ValueError("occupied group exceeds max_group_products")
    occupied = tuple(build_aiccm2026dev_b_occupied_symmetry_action(
        action, localization, tolerance=tolerance, max_workspace_bytes=cap -
        group.estimated_workspace_bytes - 16*(3*size+3)*m*m) for action in group.actions)
    if len({item.input_fingerprint for item in occupied}) != 1:
        raise ValueError("occupied group inputs changed during construction")
    c = np.array(localization.coefficients, dtype=complex, copy=True)
    s = np.array(localization.overlap, dtype=complex, copy=True)
    if _occupied_action_array_fingerprint(c, s) != occupied[0].input_fingerprint:
        raise ValueError("occupied group inputs changed during construction")
    identity = np.eye(m)
    translation_error = 0.0
    def translate(panel, shift):
        # T_ell maps source cell R to R+ell: +roll on destination rows.
        shifts = tuple(int(v) % extent for v, extent in zip(shift, a.mesh))
        return np.roll(panel.reshape((*a.mesh, a.nbasis, panel.shape[1])),
                       shifts, axis=(0, 1, 2)).reshape(panel.shape)
    def project_translation(shift):
        nonlocal translation_error
        moved = translate(c, shift)
        matrix = c.conj().T @ s @ moved
        moved_s = translate(translate(s, shift).conj().T, shift).conj().T
        error = max(_symmetry_residual(moved_s-s) / float(np.linalg.norm(s)),
                    _symmetry_residual(moved-c@matrix) / max(float(np.linalg.norm(c)), 1.0),
                    _symmetry_residual(matrix.conj().T@matrix-identity))
        translation_error = max(translation_error, error)
        if error > tolerance:
            raise ValueError("translation does not preserve the metric/occupied space")
        return matrix
    generators = np.asarray([project_translation(row) for row in np.eye(3, dtype=int)])
    algebra_error = _symmetry_residual(occupied[group.identity_index].matrix - identity)
    for axis in range(3):
        algebra_error = max(algebra_error, _symmetry_residual(
            np.linalg.matrix_power(generators[axis], a.mesh[axis]) - identity))
        for other in range(axis):
            algebra_error = max(algebra_error, _symmetry_residual(
                generators[axis] @ generators[other] - generators[other] @ generators[axis]))
    for g, dg in enumerate(occupied):
        for axis in range(3):
            transformed = project_translation(group.actions[g].rotation[:, axis])
            algebra_error = max(algebra_error, _symmetry_residual(
                dg.matrix @ generators[axis] - transformed @ dg.matrix))
        for h, dh in enumerate(occupied):
            k = int(group.products[g, h])
            ell = group.lattice_cocycle[g, h]
            translated = (occupied[k].matrix if not np.any(ell)
                          else project_translation(ell) @ occupied[k].matrix)
            algebra_error = max(algebra_error, _symmetry_residual(dg.matrix @ dh.matrix - translated))
    if algebra_error > tolerance:
        raise ValueError("occupied spatial/translation group law exceeds tolerance")
    digest = sha256(b"vibeqc.chi.occupied-group-witness/v1\0")
    for token in (group.fingerprint, tolerance.hex(), *(d.fingerprint for d in occupied),
                  _occupied_action_array_fingerprint(generators)):
        digest.update(token.encode("ascii") + b"\0")
    result = object.__new__(AICCM2026DevBOccupiedGroupWitness)
    for name, value in dict(
        group=group, occupied_actions=occupied,
        translation_generators=_immutable_numeric_array(generators, np.dtype(np.complex128)),
        maximum_translation_residual=translation_error, maximum_group_residual=algebra_error,
        tolerance=tolerance, estimated_workspace_bytes=workspace, fingerprint=digest.hexdigest(),
    ).items():
        object.__setattr__(result, name, value)
    return result


def _dataset_value(dataset: object, name: str):
    """Read a spglib dataset field across old/new Python APIs."""

    if hasattr(dataset, name):
        return getattr(dataset, name)
    return dataset[name]  # type: ignore[index]


def _normalise_mesh(
    system: PeriodicSystem,
    mesh: int | Sequence[int],
) -> tuple[int, int, int]:
    dim = int(system.dim)
    if isinstance(mesh, (int, np.integer)):
        values = [int(mesh)] * dim
    else:
        values = [int(value) for value in mesh]
    if len(values) == dim:
        values.extend([1] * (3 - dim))
    if len(values) != 3 or any(value < 1 for value in values):
        raise ValueError("AICCM2026DEV-B symmetry requires a positive 3-axis mesh")
    if any(values[axis] != 1 for axis in range(dim, 3)):
        raise ValueError("inactive periodic directions must have mesh size one")
    return tuple(values)  # type: ignore[return-value]


def _cluster_compatible(rotation: np.ndarray, mesh: tuple[int, int, int]) -> bool:
    """Return whether ``N^-1 W N`` is integral, using integer arithmetic."""

    W = np.asarray(rotation)
    # (N^-1 W N)_ij = W_ij N_j / N_i.  Avoid floating-point tests.
    return all(
        (int(W[i, j]) * int(mesh[j])) % int(mesh[i]) == 0
        for i in range(3)
        for j in range(3)
    )


def _fractional_positions(system: PeriodicSystem) -> tuple[np.ndarray, np.ndarray]:
    lattice = np.asarray(system.lattice, dtype=float)
    inv_lattice = np.linalg.inv(lattice)
    positions = np.asarray(
        [
            np.asarray(atom.xyz, dtype=float) @ inv_lattice.T
            for atom in system.unit_cell
        ]
    )
    species = np.asarray([int(atom.Z) for atom in system.unit_cell], dtype=np.int64)
    return positions % 1.0, species


def _system_to_spglib_cell(
    system: PeriodicSystem,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Convert repository column lattice vectors to spglib row vectors."""

    positions, species = _fractional_positions(system)
    return (
        np.asarray(system.lattice, dtype=float).T,
        positions,
        [int(value) for value in species],
    )


def _map_atoms(
    system: PeriodicSystem,
    rotation: np.ndarray,
    translation: np.ndarray,
    *,
    symprec: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Derive ``W f_a + w = f_b + q_a`` with a Cartesian residual gate."""

    lattice = np.asarray(system.lattice, dtype=float)
    positions, species = _fractional_positions(system)
    n_atoms = len(positions)
    permutation = np.full(n_atoms, -1, dtype=np.int64)
    shifts = np.zeros((n_atoms, 3), dtype=np.int64)
    max_residual = 0.0
    for atom_index, (position, atomic_number) in enumerate(zip(positions, species)):
        image = rotation @ position + translation
        matches: list[tuple[int, np.ndarray, float]] = []
        for candidate in np.flatnonzero(species == atomic_number):
            delta = image - positions[candidate]
            shift = np.rint(delta).astype(np.int64)
            residual = float(np.linalg.norm((delta - shift) @ lattice.T))
            if residual <= symprec:
                matches.append((int(candidate), shift, residual))
        if len(matches) != 1:
            raise ValueError(
                "AICCM2026DEV-B symmetry atom mapping is not unique for "
                f"atom {atom_index}: found {len(matches)} matches at "
                f"symprec={symprec:.3e} bohr"
            )
        candidate, shift, residual = matches[0]
        permutation[atom_index] = candidate
        shifts[atom_index] = shift
        max_residual = max(max_residual, residual)
    if len(set(int(value) for value in permutation)) != n_atoms:
        raise RuntimeError("space-group operation did not induce an atom permutation")
    return permutation, shifts, max_residual


def _gamma_mesh_fractional(mesh: tuple[int, int, int]) -> np.ndarray:
    """Full Gamma-centred character mesh in deterministic C order."""

    return np.asarray(
        [
            (i / mesh[0], j / mesh[1], k / mesh[2])
            for i in range(mesh[0])
            for j in range(mesh[1])
            for k in range(mesh[2])
        ],
        dtype=float,
    )


def _k_grid_index(kpoint: np.ndarray, mesh: tuple[int, int, int]) -> int:
    scaled = np.asarray(kpoint, dtype=float) * np.asarray(mesh, dtype=float)
    rounded = np.rint(scaled).astype(np.int64)
    if np.max(np.abs(scaled - rounded)) > 1.0e-8:
        raise RuntimeError("compatible symmetry operation left the cyclic k net")
    residue = np.mod(rounded, np.asarray(mesh, dtype=np.int64))
    return int((residue[0] * mesh[1] + residue[1]) * mesh[2] + residue[2])


def _k_orbits(
    mesh: tuple[int, int, int],
    operations: Sequence[AICCM2026DevBSymmetryOperation],
    *,
    time_reversal: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = _gamma_mesh_fractional(mesh)
    parent = np.arange(len(points), dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for source, kpoint in enumerate(points):
        for operation in operations:
            reciprocal_rotation = np.linalg.inv(operation.rotation).T
            target = reciprocal_rotation @ kpoint
            union(source, _k_grid_index(target, mesh))
        if time_reversal:
            union(source, _k_grid_index(-kpoint, mesh))

    roots = np.asarray([find(index) for index in range(len(points))], dtype=np.int64)
    representatives = np.asarray(
        sorted(set(int(root) for root in roots)), dtype=np.int64
    )
    root_to_ir = {int(root): ir for ir, root in enumerate(representatives)}
    mapping = np.asarray([root_to_ir[int(root)] for root in roots], dtype=np.int64)
    weights = np.bincount(mapping, minlength=len(representatives)).astype(float)
    weights /= len(points)
    return points, mapping, representatives, weights


def build_aiccm2026dev_b_symmetry_plan(
    system: PeriodicSystem,
    mesh: int | Sequence[int],
    *,
    symprec: float = 1.0e-5,
    time_reversal: bool = True,
    require_full_space_group: bool = False,
) -> AICCM2026DevBSymmetryPlan:
    """Analyze the space group and its exact action on a finite BvK torus.

    The current implementation is restricted to 3D.  Applying ordinary
    3D spglib to a slab or wire cell would treat the artificial vacuum as a
    genuine lattice direction and can create nonphysical operations; that
    case therefore fails closed pending a layer/rod-group implementation.
    """

    if int(system.dim) != 3:
        raise NotImplementedError(
            "AICCM2026DEV-B space-group diagnostics currently require dim=3; "
            "layer and rod groups for 2D/1D are not implemented"
        )
    if not np.isfinite(symprec) or symprec <= 0.0:
        raise ValueError("symprec must be a finite positive distance in bohr")
    mesh_tuple = _normalise_mesh(system, mesh)
    try:
        import spglib
    except ImportError as exc:  # pragma: no cover - core dependency
        raise ImportError("AICCM2026DEV-B symmetry diagnostics require spglib") from exc

    dataset = spglib.get_symmetry_dataset(
        _system_to_spglib_cell(system), symprec=float(symprec)
    )
    if dataset is None:
        raise RuntimeError("spglib failed to determine the crystal space group")
    rotations = np.asarray(_dataset_value(dataset, "rotations"), dtype=np.int64)
    translations = np.asarray(_dataset_value(dataset, "translations"), dtype=float)

    operations: list[AICCM2026DevBSymmetryOperation] = []
    incompatible: list[int] = []
    for operation_index, (rotation, translation) in enumerate(
        zip(rotations, translations)
    ):
        if not _cluster_compatible(rotation, mesh_tuple):
            incompatible.append(operation_index)
            continue
        permutation, shifts, residual = _map_atoms(
            system,
            rotation,
            translation,
            symprec=float(symprec),
        )
        operations.append(
            AICCM2026DevBSymmetryOperation(
                full_group_index=operation_index,
                rotation=rotation.copy(),
                translation=translation.copy(),
                atom_permutation=permutation,
                atom_lattice_shifts=shifts,
                max_atom_mapping_residual_bohr=residual,
                has_fractional_translation=bool(
                    np.max(np.abs(translation - np.rint(translation))) > 1.0e-10
                ),
            )
        )
    if not operations:
        raise RuntimeError("cyclic-cluster-compatible subgroup is empty")
    if require_full_space_group and incompatible:
        raise ValueError(
            "cyclic mesh is incompatible with the full crystal space group: "
            f"{len(incompatible)} of {len(rotations)} operations fail "
            "N^-1 W N integral; use an isotropic/symmetry-compatible mesh or "
            "set require_full_space_group=False to use the exact subgroup"
        )

    points, mapping, representatives, weights = _k_orbits(
        mesh_tuple, operations, time_reversal=bool(time_reversal)
    )
    return AICCM2026DevBSymmetryPlan(
        mesh=mesh_tuple,
        space_group_number=int(_dataset_value(dataset, "number")),
        international_symbol=str(_dataset_value(dataset, "international")),
        hall_number=int(_dataset_value(dataset, "hall_number")),
        point_group=str(_dataset_value(dataset, "pointgroup")),
        wyckoff_letters=tuple(
            str(value) for value in _dataset_value(dataset, "wyckoffs")
        ),
        site_symmetry_symbols=tuple(
            str(value) for value in _dataset_value(dataset, "site_symmetry_symbols")
        ),
        equivalent_atoms=tuple(
            int(value) for value in _dataset_value(dataset, "equivalent_atoms")
        ),
        n_operations_full=len(rotations),
        operations=tuple(operations),
        incompatible_operation_indices=tuple(incompatible),
        full_kpoints_frac=points,
        full_to_irreducible=mapping,
        irreducible_representative_indices=representatives,
        irreducible_weights=weights,
        time_reversal=bool(time_reversal),
    )


def _gamma_ao_actions(
    system: PeriodicSystem,
    basis: BasisSet,
    plan: AICCM2026DevBSymmetryPlan,
) -> tuple[np.ndarray, ...]:
    """Build Gamma AO actions; all nonsymmorphic Bloch phases equal one."""

    lattice = np.asarray(system.lattice, dtype=float)
    actions: list[np.ndarray] = []
    for operation in plan.operations:
        rotation_cart = lattice_to_cartesian_rotation(
            operation.rotation,
            lattice,
        )
        orthogonality_error = np.linalg.norm(
            rotation_cart.T @ rotation_cart - np.eye(3)
        )
        if orthogonality_error > 1.0e-8:
            raise RuntimeError(
                "space-group Cartesian rotation is not orthogonal; check lattice "
                f"conventions (residual {orthogonality_error:.3e})"
            )
        atom_mapping = AtomPermutation(
            operation.atom_permutation,
            operation.atom_lattice_shifts,
        )
        actions.append(
            build_ao_permutation_matrix(basis, rotation_cart, atom_mapping)
        )
    return tuple(actions)


def _immutable_numeric_array(array: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Detach an array onto bytes-backed storage that cannot be re-enabled."""

    canonical = np.ascontiguousarray(array, dtype=dtype)
    return np.frombuffer(
        canonical.tobytes(order="C"),
        dtype=dtype,
    ).reshape(canonical.shape)


def _validated_int64_array(
    raw: object,
    *,
    name: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    """Snapshot an integer payload without permitting narrowing wraparound."""

    array = np.asarray(raw)
    if array.shape != shape or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must have shape {shape} and integer dtype")
    if array.size:
        lower = int(np.min(array))
        upper = int(np.max(array))
        bounds = np.iinfo(np.int64)
        if lower < int(bounds.min) or upper > int(bounds.max):
            raise ValueError(f"{name} is outside the signed int64 range")
    return np.array(array, dtype=np.int64, order="C", copy=True)


def _rounded_int64_array(raw: np.ndarray, *, name: str) -> np.ndarray:
    """Round finite floats to an int64 snapshot after an explicit range gate."""

    array = np.array(raw, dtype=np.float64, order="C", copy=True)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    rounded = np.rint(array)
    int64_limit = float(2**63)
    if np.any(rounded < -int64_limit) or np.any(rounded >= int64_limit):
        raise ValueError(f"{name} is outside the signed int64 range")
    return rounded.astype(np.int64)


def _exact_int3_determinant(matrix: np.ndarray) -> int:
    """Return the exact determinant of a 3 x 3 integer matrix."""

    a, b, c = (int(value) for value in matrix[0])
    d, e, f = (int(value) for value in matrix[1])
    g, h, i = (int(value) for value in matrix[2])
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _validate_real_torus_mesh_bounds(
    mesh: tuple[int, int, int],
    *,
    n_atoms: int,
    nbasis: int,
) -> int:
    """Return the cell count after all retained-array address checks."""

    n_cells = prod(int(value) for value in mesh)
    int64_max = int(np.iinfo(np.int64).max)
    intp_max = int(np.iinfo(np.intp).max)
    if n_cells > min(int64_max, intp_max):
        raise ValueError("symmetry plan mesh exceeds addressable array bounds")
    largest_row_bytes = max(
        3 * np.dtype(np.int64).itemsize,
        3 * np.dtype(np.float64).itemsize,
    )
    if n_cells > intp_max // largest_row_bytes:
        raise ValueError("symmetry mesh payload exceeds addressable array bounds")
    cell_action_row_bytes = n_atoms * np.dtype(np.int64).itemsize
    if cell_action_row_bytes > 0 and n_cells > intp_max // cell_action_row_bytes:
        raise ValueError("symmetry cell-action table exceeds addressable array bounds")
    if nbasis > 0 and n_cells > intp_max // nbasis:
        raise ValueError("real-torus AO dimension exceeds addressable array bounds")
    return n_cells


def _canonical_mesh_cells_and_kpoints(
    mesh: tuple[int, int, int],
    *,
    n_atoms: int,
    nbasis: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize the canonical C-order finite mesh after address checks."""

    n_cells = _validate_real_torus_mesh_bounds(
        mesh,
        n_atoms=n_atoms,
        nbasis=nbasis,
    )

    flat = np.arange(n_cells, dtype=np.int64)
    plane = int(mesh[1]) * int(mesh[2])
    first = flat // plane
    second = (flat // int(mesh[2])) % int(mesh[1])
    third = flat % int(mesh[2])
    cells = np.column_stack((first, second, third))
    kpoints = np.empty((n_cells, 3), dtype=np.float64)
    kpoints[:, 0] = first / int(mesh[0])
    kpoints[:, 1] = second / int(mesh[1])
    kpoints[:, 2] = third / int(mesh[2])
    return cells, kpoints


def _basis_ao_atom_indices(basis: BasisSet, atoms: Sequence[object]) -> np.ndarray:
    n_atoms = len(atoms)
    indices: list[int] = []
    for shell in basis.shells():
        atom_index = int(shell.atom_index)
        if atom_index < 0 or atom_index >= n_atoms:
            raise ValueError(
                "basis shell atom index is incompatible with the periodic system"
            )
        origin = np.asarray(shell.origin, dtype=np.float64)
        atom_position = np.asarray(atoms[atom_index].xyz, dtype=np.float64)
        if (
            origin.shape != (3,)
            or not np.all(np.isfinite(origin))
            or float(np.linalg.norm(origin - atom_position))
            > _REAL_TORUS_AO_ACTION_BASIS_ORIGIN_TOLERANCE_BOHR
        ):
            raise ValueError(
                "basis shell origin does not match its periodic-system atom"
            )
        angular_momentum = int(shell.l)
        if not bool(shell.pure) and angular_momentum >= 1:
            raise NotImplementedError(
                "real-torus AO actions currently require spherical l >= 1 "
                "shells; Cartesian p-and-higher actions are not implemented"
            )
        indices.extend([atom_index] * (2 * angular_momentum + 1))
    if len(indices) != int(basis.nbasis):
        raise RuntimeError(
            "basis shell-size sum does not match the reported AO dimension"
        )
    result = np.asarray(indices, dtype=np.int64)
    if any(not np.any(result == atom) for atom in range(n_atoms)):
        raise ValueError("real-torus AO action requires basis functions on every atom")
    return result


def _validate_basis_symmetry_closure(
    basis: BasisSet,
    atom_permutation: np.ndarray,
) -> None:
    """Require identical ordered radial shells on symmetry-related atoms."""

    by_atom: dict[int, list[object]] = {}
    for shell in basis.shells():
        by_atom.setdefault(int(shell.atom_index), []).append(shell)
    for source_atom, destination_atom in enumerate(atom_permutation):
        source_shells = by_atom.get(source_atom, [])
        destination_shells = by_atom.get(int(destination_atom), [])
        if len(source_shells) != len(destination_shells):
            raise ValueError("symmetry-related atoms have different AO shell counts")
        for source, destination in zip(source_shells, destination_shells):
            source_exponents = np.asarray(source.exponents, dtype=np.float64)
            destination_exponents = np.asarray(
                destination.exponents,
                dtype=np.float64,
            )
            source_coefficients = np.asarray(source.coefficients, dtype=np.float64)
            destination_coefficients = np.asarray(
                destination.coefficients,
                dtype=np.float64,
            )
            if (
                int(source.l) != int(destination.l)
                or bool(source.pure) != bool(destination.pure)
                or source_exponents.shape != destination_exponents.shape
                or source_coefficients.shape != destination_coefficients.shape
                or not np.all(np.isfinite(source_exponents))
                or not np.all(np.isfinite(destination_exponents))
                or not np.all(np.isfinite(source_coefficients))
                or not np.all(np.isfinite(destination_coefficients))
                or not np.array_equal(source_exponents, destination_exponents)
                or not np.array_equal(source_coefficients, destination_coefficients)
            ):
                raise ValueError(
                    "symmetry-related atoms have different ordered radial AO shells"
                )


def build_aiccm2026dev_b_real_torus_ao_action(
    system: PeriodicSystem,
    basis: BasisSet,
    plan: AICCM2026DevBSymmetryPlan,
    operation_index: int,
) -> AICCM2026DevBRealTorusAOAction:
    """Build one compact finite-torus AO point action.

    The factory snapshots and validates the convention-critical operation,
    primitive AO rotation, and atom-dependent cell images.  It deliberately
    returns one operation at a time so a consumer need not retain the full
    space group.  Since the symmetry-plan payload does not retain its
    construction ``symprec``, this action factory independently requires an
    actual atom-mapping residual of at most ``1e-5`` bohr.  The result is not
    connected to SCF or post-HF execution.
    """

    if not isinstance(plan, AICCM2026DevBSymmetryPlan):
        raise ValueError(
            "real-torus AO action requires an AICCM2026DevBSymmetryPlan"
        )
    if int(system.dim) != 3:
        raise NotImplementedError("real-torus AO point actions currently require dim=3")
    try:
        raw_mesh = tuple(plan.mesh)
    except TypeError as exc:
        raise ValueError(
            "symmetry plan mesh must contain three positive integers"
        ) from exc
    if len(raw_mesh) != 3 or any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 1
        for value in raw_mesh
    ):
        raise ValueError("symmetry plan mesh must contain three positive integers")
    mesh = _normalise_mesh(system, raw_mesh)
    n_cells = prod(mesh)
    atoms = list(system.unit_cell)
    n_atoms = len(atoms)
    if n_atoms < 1:
        raise ValueError("real-torus AO action requires at least one atom")
    _validate_real_torus_mesh_bounds(
        mesh,
        n_atoms=n_atoms,
        nbasis=int(basis.nbasis),
    )
    raw_plan_kpoints = np.asarray(plan.full_kpoints_frac)
    if (
        raw_plan_kpoints.shape != (n_cells, 3)
        or raw_plan_kpoints.dtype.kind != "f"
    ):
        raise ValueError("symmetry plan k-mesh metadata is inconsistent")
    plan_kpoints = np.array(
        raw_plan_kpoints,
        dtype=np.float64,
        order="C",
        copy=True,
    )
    if not np.all(np.isfinite(plan_kpoints)):
        raise ValueError("symmetry plan k-mesh metadata is inconsistent")
    if (
        isinstance(operation_index, (bool, np.bool_))
        or not isinstance(operation_index, (int, np.integer))
    ):
        raise ValueError("operation_index must be an integer")
    selected_index = int(operation_index)
    if selected_index < 0 or selected_index >= len(plan.operations):
        raise ValueError("operation_index is outside the compatible symmetry plan")
    operation = plan.operations[selected_index]

    rotation = _validated_int64_array(
        operation.rotation,
        name="symmetry operation rotation",
        shape=(3, 3),
    )
    if abs(_exact_int3_determinant(rotation)) != 1:
        raise ValueError("symmetry operation rotation must be unimodular")
    if not _cluster_compatible(rotation, mesh):
        raise ValueError("symmetry operation is incompatible with the finite mesh")

    raw_translation = np.asarray(operation.translation)
    if raw_translation.shape != (3,) or raw_translation.dtype.kind != "f":
        raise ValueError(
            "symmetry operation translation must be a real-floating 3-vector"
        )
    translation = np.array(
        raw_translation,
        dtype=np.float64,
        order="C",
        copy=True,
    )
    if not np.all(np.isfinite(translation)):
        raise ValueError("symmetry operation translation must be a finite 3-vector")
    permutation = _validated_int64_array(
        operation.atom_permutation,
        name="symmetry atom mapping",
        shape=(n_atoms,),
    )
    shifts = _validated_int64_array(
        operation.atom_lattice_shifts,
        name="symmetry atom lattice shifts",
        shape=(n_atoms, 3),
    )
    if not np.array_equal(np.sort(permutation), np.arange(n_atoms)):
        raise ValueError("symmetry atom mapping is not a permutation")

    lattice = np.array(system.lattice, dtype=np.float64, order="C", copy=True)
    rotation_cart = lattice_to_cartesian_rotation(rotation, lattice)
    orthogonality_error = float(
        np.linalg.norm(rotation_cart.T @ rotation_cart - np.eye(3))
    )
    if orthogonality_error > 1.0e-8:
        raise ValueError(
            "symmetry Cartesian rotation is not orthogonal for the supplied lattice"
        )
    positions, species = _fractional_positions(system)
    inverse_lattice = np.linalg.inv(lattice)
    raw_positions = np.asarray(
        [
            inverse_lattice @ np.asarray(atom.xyz, dtype=np.float64)
            for atom in atoms
        ]
    )
    reference_offsets = _rounded_int64_array(
        raw_positions - positions,
        name="periodic atom reference-cell offsets",
    )
    reference_residual = raw_positions - positions - reference_offsets
    if np.max(np.linalg.norm(reference_residual @ lattice.T, axis=1)) > 1.0e-10:
        raise ValueError("periodic atom reference-cell offsets are not integral")
    action_shifts_object = (
        shifts.astype(object)
        + reference_offsets.astype(object) @ rotation.T.astype(object)
        - reference_offsets[permutation].astype(object)
    )
    action_shift_min = min(int(value) for value in action_shifts_object.flat)
    action_shift_max = max(int(value) for value in action_shifts_object.flat)
    int64_bounds = np.iinfo(np.int64)
    if (
        action_shift_min < int(int64_bounds.min)
        or action_shift_max > int(int64_bounds.max)
    ):
        raise ValueError("corrected atom lattice shifts exceed signed int64 range")
    action_shifts = np.asarray(action_shifts_object, dtype=np.int64)
    actual_max_residual = 0.0
    for source_atom, destination_atom in enumerate(permutation):
        if species[source_atom] != species[int(destination_atom)]:
            raise ValueError("symmetry atom mapping changes the chemical species")
        fractional_residual = (
            rotation @ raw_positions[source_atom]
            + translation
            - raw_positions[int(destination_atom)]
            - action_shifts[source_atom]
        )
        actual_max_residual = max(
            actual_max_residual,
            float(np.linalg.norm(lattice @ fractional_residual)),
        )
    raw_declared_residual = operation.max_atom_mapping_residual_bohr
    if (
        isinstance(raw_declared_residual, (bool, np.bool_))
        or not isinstance(
            raw_declared_residual,
            (int, float, np.integer, np.floating),
        )
    ):
        raise ValueError("symmetry atom-mapping residual must be a real scalar")
    declared_residual = float(raw_declared_residual)
    if not np.isfinite(declared_residual) or declared_residual < 0.0:
        raise ValueError(
            "symmetry atom-mapping residual must be finite and nonnegative"
        )
    if not np.isclose(
        actual_max_residual,
        declared_residual,
        rtol=1.0e-8,
        atol=1.0e-10,
    ):
        raise ValueError(
            "symmetry plan atom mapping does not match the supplied system"
        )
    has_fractional_translation = bool(
        np.max(np.abs(translation - np.rint(translation))) > 1.0e-10
    )
    if not isinstance(operation.has_fractional_translation, (bool, np.bool_)):
        raise ValueError("fractional-translation metadata must be boolean")
    if operation.has_fractional_translation != has_fractional_translation:
        raise ValueError("symmetry translation metadata is inconsistent")
    if (
        isinstance(operation.full_group_index, (bool, np.bool_))
        or not isinstance(operation.full_group_index, (int, np.integer))
        or int(operation.full_group_index) < 0
    ):
        raise ValueError("full_group_index must be a nonnegative integer")
    if (
        isinstance(plan.n_operations_full, (bool, np.bool_))
        or not isinstance(plan.n_operations_full, (int, np.integer))
        or int(plan.n_operations_full) < 1
        or int(operation.full_group_index) >= int(plan.n_operations_full)
    ):
        raise ValueError("full_group_index is outside the full space group")
    if (
        actual_max_residual
        > _REAL_TORUS_AO_ACTION_MAX_MAPPING_RESIDUAL_BOHR + 1.0e-12
    ):
        raise ValueError(
            "symmetry atom mapping exceeds the real-torus action admission "
            "tolerance of "
            f"{_REAL_TORUS_AO_ACTION_MAX_MAPPING_RESIDUAL_BOHR:.1e} bohr"
        )

    cells, expected_kpoints = _canonical_mesh_cells_and_kpoints(
        mesh,
        n_atoms=n_atoms,
        nbasis=int(basis.nbasis),
    )
    if (
        not np.allclose(
            plan_kpoints,
            expected_kpoints,
            atol=1.0e-12,
            rtol=0.0,
        )
    ):
        raise ValueError("symmetry plan k-mesh metadata is inconsistent")
    cell_images = np.empty((n_cells, n_atoms), dtype=np.int64)
    for atom_index in range(n_atoms):
        intermediate_bound = abs(int(action_shifts[atom_index, 0]))
        for row in range(3):
            intermediate_bound = max(
                intermediate_bound,
                abs(int(action_shifts[atom_index, row]))
                + sum(
                    (int(mesh[column]) - 1)
                    * abs(int(rotation[row, column]))
                    for column in range(3)
                ),
            )
        if intermediate_bound > int(np.iinfo(np.int64).max):
            raise ValueError("symmetry cell action exceeds signed int64 range")
        images = cells @ rotation.T + action_shifts[atom_index]
        residues = np.mod(images, np.asarray(mesh, dtype=np.int64))
        cell_images[:, atom_index] = (
            (residues[:, 0] * mesh[1] + residues[:, 1]) * mesh[2]
            + residues[:, 2]
        )
        if not np.array_equal(
            np.sort(cell_images[:, atom_index]),
            np.arange(n_cells, dtype=np.int64),
        ):
            raise ValueError("symmetry cell action is not a finite-torus permutation")

    ao_atom_indices = _basis_ao_atom_indices(basis, atoms)
    _validate_basis_symmetry_closure(basis, permutation)
    primitive_action = np.asarray(
        build_ao_permutation_matrix(
            basis,
            rotation_cart,
            permutation,
        ),
        dtype=np.float64,
    )
    nbasis = int(basis.nbasis)
    if primitive_action.shape != (nbasis, nbasis) or not np.all(
        np.isfinite(primitive_action)
    ):
        raise RuntimeError("primitive AO symmetry action has an invalid payload")
    if np.linalg.norm(primitive_action.T @ primitive_action - np.eye(nbasis)) > 1e-10:
        raise RuntimeError("primitive AO symmetry action is not orthogonal")
    for source_atom, destination_atom in enumerate(permutation):
        source_aos = np.flatnonzero(ao_atom_indices == source_atom)
        destination_aos = np.flatnonzero(
            ao_atom_indices == int(destination_atom)
        )
        if len(source_aos) != len(destination_aos):
            raise ValueError("symmetry atom mapping changes the AO shell structure")
        outside = np.ones((nbasis, len(source_aos)), dtype=bool)
        outside[destination_aos, :] = False
        if np.any(np.abs(primitive_action[:, source_aos][outside]) > 1.0e-12):
            raise RuntimeError("primitive AO action escaped the mapped atom block")

    return AICCM2026DevBRealTorusAOAction._from_payload(
        mesh=mesh,
        operation_index=selected_index,
        full_group_index=int(operation.full_group_index),
        rotation=rotation,
        translation=translation,
        atom_permutation=permutation,
        atom_lattice_shifts=action_shifts,
        atom_reference_cell_offsets=reference_offsets,
        ao_atom_indices=ao_atom_indices,
        primitive_ao_action=primitive_action,
        cell_images=cell_images,
    )


def _shell_permutations(
    basis: BasisSet,
    plan: AICCM2026DevBSymmetryPlan,
) -> tuple[np.ndarray, ...]:
    """Map primitive-cell shell indices under compatible operations."""

    shells = list(basis.shells())
    by_atom: dict[int, list[int]] = {}
    positions: dict[int, int] = {}
    for shell_index, shell in enumerate(shells):
        atom = int(shell.atom_index)
        positions[shell_index] = len(by_atom.setdefault(atom, []))
        by_atom[atom].append(shell_index)
    permutations: list[np.ndarray] = []
    for operation in plan.operations:
        permutation = np.empty(len(shells), dtype=int)
        for shell_index, shell in enumerate(shells):
            destination_atom = int(
                operation.atom_permutation[int(shell.atom_index)]
            )
            destination_shells = by_atom.get(destination_atom, [])
            slot = positions[shell_index]
            if slot >= len(destination_shells):
                raise RuntimeError(
                    "space-group atom permutation changed shell structure"
                )
            destination = destination_shells[slot]
            if int(shells[destination].l) != int(shell.l):
                raise RuntimeError(
                    "space-group shell permutation changed angular momentum"
                )
            permutation[shell_index] = destination
        if not np.array_equal(np.sort(permutation), np.arange(len(shells))):
            raise RuntimeError("space-group shell mapping is not a permutation")
        permutations.append(permutation)
    return tuple(permutations)


def _partition_discrete_orbits(
    objects: set[tuple],
    actions,
) -> tuple[tuple[tuple, ...], ...]:
    unassigned = set(objects)
    orbits: list[tuple[tuple, ...]] = []
    while unassigned:
        seed = min(unassigned)
        members = {seed}
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            for action in actions:
                image = action(current)
                if image not in members:
                    if image not in objects:
                        raise RuntimeError("shell symmetry orbit escaped its space")
                    members.add(image)
                    frontier.append(image)
        unassigned -= members
        orbits.append(tuple(sorted(members)))
    flattened = [member for orbit in orbits for member in orbit]
    if len(flattened) != len(objects) or set(flattened) != objects:
        raise RuntimeError("shell symmetry orbits do not form a partition")
    return tuple(sorted(orbits, key=lambda orbit: orbit[0]))


def _canonical_shell_pair(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left <= right else (right, left)


def shell_pair_orbits(
    basis: BasisSet,
    plan: AICCM2026DevBSymmetryPlan,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Partition intrinsic-unique primitive shell pairs by point symmetry."""

    n_shells = len(list(basis.shells()))
    pairs = {
        (left, right)
        for left in range(n_shells)
        for right in range(left, n_shells)
    }
    permutations = _shell_permutations(basis, plan)
    actions = tuple(
        lambda pair, permutation=permutation: _canonical_shell_pair(
            int(permutation[pair[0]]),
            int(permutation[pair[1]]),
        )
        for permutation in permutations
    )
    return _partition_discrete_orbits(pairs, actions)  # type: ignore[return-value]


def _canonical_shell_quartet(
    first: tuple[int, int],
    second: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    left = _canonical_shell_pair(*first)
    right = _canonical_shell_pair(*second)
    return (left, right) if left <= right else (right, left)


def shell_quartet_orbits(
    basis: BasisSet,
    plan: AICCM2026DevBSymmetryPlan,
) -> tuple[tuple[tuple[tuple[int, int], tuple[int, int]], ...], ...]:
    """Partition intrinsic 8-fold-unique primitive shell quartets."""

    n_shells = len(list(basis.shells()))
    pairs = tuple(
        (left, right)
        for left in range(n_shells)
        for right in range(left, n_shells)
    )
    quartets = {
        (left, right)
        for left_index, left in enumerate(pairs)
        for right in pairs[left_index:]
    }
    permutations = _shell_permutations(basis, plan)
    actions = tuple(
        lambda quartet, permutation=permutation: _canonical_shell_quartet(
            (
                int(permutation[quartet[0][0]]),
                int(permutation[quartet[0][1]]),
            ),
            (
                int(permutation[quartet[1][0]]),
                int(permutation[quartet[1][1]]),
            ),
        )
        for permutation in permutations
    )
    return _partition_discrete_orbits(  # type: ignore[return-value]
        quartets,
        actions,
    )


def symmetrize_gamma_ao_matrix(
    matrix: np.ndarray,
    system: PeriodicSystem,
    basis: BasisSet,
    plan: AICCM2026DevBSymmetryPlan,
) -> np.ndarray:
    """Project a Gamma AO matrix onto the compatible space-group invariant.

    This is the Reynolds operator ``|G|^-1 sum_g U_g M U_g^T``.  It is
    rigorous only at Gamma in this increment.  The routine is a diagnostic
    utility and is not inserted into DIIS or the B-stream SCF build.
    """

    array = np.asarray(matrix)
    if array.shape != (basis.nbasis, basis.nbasis):
        raise ValueError(
            f"Gamma AO matrix must have shape {(basis.nbasis, basis.nbasis)}; "
            f"got {array.shape}"
        )
    actions = _gamma_ao_actions(system, basis, plan)
    projected = np.zeros_like(array, dtype=np.result_type(array, np.float64))
    for action in actions:
        projected += action @ array @ action.T
    return projected / len(actions)


def gamma_matrix_symmetry_residual(
    matrix: np.ndarray,
    system: PeriodicSystem,
    basis: BasisSet,
    plan: AICCM2026DevBSymmetryPlan,
) -> float:
    """Maximum Frobenius violation ``||U_g M U_g^T - M||`` at Gamma."""

    array = np.asarray(matrix)
    actions = _gamma_ao_actions(system, basis, plan)
    return max(
        float(np.linalg.norm(action @ array @ action.T - array))
        for action in actions
    )


@dataclass(frozen=True, slots=True)
class ChiAOBlochSpaceAction:
    """Diagnostic shared SpaceAction adapter for native chi AO Bloch panels.

    Geometry, basis, operation and controls are borrowed descriptors, rechecked
    by native transport at each call. Space identities are declarations, not
    authenticated snapshots. Native tiny diagnostic limits remain in effect.
    """
    source: SpaceIdentity
    target: SpaceIdentity
    basis: object
    system: object
    operation: object
    mesh: object
    source_index: int
    antiunitary: bool
    options: object
    inventory: object
    caps: object

    @property
    def dimension(self) -> int:
        return int(self.basis.nbasis)

    def apply(self, coefficients: np.ndarray) -> np.ndarray:
        from ... import _vibeqc_core as core
        from ...symmetry_shared import SpaceIdentity
        if not isinstance(self.source, SpaceIdentity) or not isinstance(self.target, SpaceIdentity):
            raise TypeError("chi action requires named shared spaces")
        if type(self.antiunitary) is not bool:
            raise TypeError("chi antiunitary flag must be bool")
        return core._apply_periodic_ao_bloch_operation(
            self.basis, self.system, self.operation, self.mesh, self.source_index,
            self.antiunitary, coefficients, self.options, self.inventory, self.caps,
        ).coefficients_copy()
