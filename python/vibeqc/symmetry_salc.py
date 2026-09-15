"""Phase SYM6: symmetry-adapted linear combinations (SALC) and irrep projection.

Provides character tables for common point groups and a projection
operator that decomposes AO-basis matrices into irreducible
representations of the crystallographic point group.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ._vibeqc_core import PeriodicSystem, SymmetryOp
from .symmetry_ao import atom_permutation_under_op, build_ao_permutation_matrix
from .symmetry_integrals import symmorphic_operations
from .symmetry_lattice import lattice_to_cartesian_rotation

__all__ = [
    "character_table",
    "symmetry_project_matrix",
    "PointGroupCharacters",
]

# ---------------------------------------------------------------------------
# Character tables for common crystallographic point groups.
#
# Each entry maps the international point-group symbol to a list of
# (label, characters) tuples.  Characters are ordered by the operator
# list returned by spglib (same order as system.symmetry.operations).
# ---------------------------------------------------------------------------


@dataclass
class PointGroupCharacters:
    """Character table for a crystallographic point group."""

    symbol: str
    irreps: List[str]  # irrep labels (A1g, T1u, etc.)
    characters: np.ndarray  # (n_irreps, n_classes) character table
    class_indices: List[List[int]]  # which operator indices belong to each class

    @property
    def n_irreps(self) -> int:
        return len(self.irreps)


# Pre-computed character tables for the most common point groups.
# Characters follow the standard conventions (Mulliken notation).
_CHARACTER_TABLES: Dict[str, Tuple[List[str], List[List[float]], List[List[int]]]] = {
    # Oh -- cubic, order 48.  Rows are classes in the order of
    # _CLASS_SPECS["m-3m"]; verified against the standard Oh table
    # (e.g. Bishop, "Group Theory and Chemistry", Table 7.7) by the
    # orthogonality test in tests/test_symmetry_salc.py.
    "m-3m": (
        ["A1g", "A2g", "Eg", "T1g", "T2g", "A1u", "A2u", "Eu", "T1u", "T2u"],
        [
            [1, 1, 2, 3, 3, 1, 1, 2, 3, 3],  # E
            [1, 1, -1, 0, 0, 1, 1, -1, 0, 0],  # 8C3
            [1, 1, 2, -1, -1, 1, 1, 2, -1, -1],  # 3C2 (= C4^2)
            [1, -1, 0, -1, 1, 1, -1, 0, -1, 1],  # 6C2'
            [1, -1, 0, 1, -1, 1, -1, 0, 1, -1],  # 6C4
            [1, 1, 2, 3, 3, -1, -1, -2, -3, -3],  # i
            [1, 1, -1, 0, 0, -1, -1, 1, 0, 0],  # 8S6
            [1, 1, 2, -1, -1, -1, -1, -2, 1, 1],  # 3sh
            [1, -1, 0, -1, 1, -1, 1, 0, 1, -1],  # 6sd
            [1, -1, 0, 1, -1, -1, 1, 0, -1, 1],  # 6S4
        ],
        None,  # class indices depend on operator ordering
    ),
    # Td -- tetrahedral, order 24 (e.g. GaAs, ZnS)
    "-43m": (
        ["A1", "A2", "E", "T1", "T2"],
        [
            [1, 1, 2, 3, 3],  # E
            [1, 1, -1, 0, 0],  # 8C3
            [1, 1, 2, -1, -1],  # 3C2
            [1, -1, 0, -1, 1],  # 6sd
            [1, -1, 0, 1, -1],  # 6S4
        ],
        None,
    ),
    # D4h -- tetragonal, order 16 (e.g. TiO2 rutile)
    "4/mmm": (
        ["A1g", "A2g", "B1g", "B2g", "Eg", "A1u", "A2u", "B1u", "B2u", "Eu"],
        [
            [1, 1, 1, 1, 2, 1, 1, 1, 1, 2],  # E
            [1, 1, -1, -1, 0, 1, 1, -1, -1, 0],  # 2C4
            [1, 1, 1, 1, -2, 1, 1, 1, 1, -2],  # C2
            [1, -1, 1, -1, 0, 1, -1, 1, -1, 0],  # 2C2'
            [1, -1, -1, 1, 0, 1, -1, -1, 1, 0],  # 2C2''
            [1, 1, 1, 1, 2, -1, -1, -1, -1, -2],  # i
            [1, 1, -1, -1, 0, -1, -1, 1, 1, 0],  # 2S4
            [1, 1, 1, 1, -2, -1, -1, -1, -1, 2],  # sh
            [1, -1, 1, -1, 0, -1, 1, -1, 1, 0],  # 2sv
            [1, -1, -1, 1, 0, -1, 1, 1, -1, 0],  # 2sd
        ],
        None,
    ),
    # C6v -- hexagonal, order 12 (e.g. GaN wurtzite)
    "6mm": (
        ["A1", "A2", "B1", "B2", "E1", "E2"],
        [
            [1, 1, 1, 1, 2, 2],  # E
            [1, 1, -1, -1, 1, -1],  # 2C6
            [1, 1, 1, 1, -1, -1],  # 2C3
            [1, 1, -1, -1, -2, 2],  # C2
            [1, -1, 1, -1, 0, 0],  # 3sv
            [1, -1, -1, 1, 0, 0],  # 3sd
        ],
        None,
    ),
    # D3d -- trigonal, order 12 (corundum Al2O3, calcite CaCO3, Bi2Se3)
    "-3m": (
        ["A1g", "A2g", "Eg", "A1u", "A2u", "Eu"],
        [
            [1, 1, 2, 1, 1, 2],  # E
            [1, 1, -1, 1, 1, -1],  # 2C3
            [1, -1, 0, 1, -1, 0],  # 3C2'
            [1, 1, 2, -1, -1, -2],  # i
            [1, 1, -1, -1, -1, 1],  # 2S6
            [1, -1, 0, -1, 1, 0],  # 3sd
        ],
        None,
    ),
    # D6h -- hexagonal, order 24 (graphite, h-BN)
    "6/mmm": (
        [
            "A1g",
            "A2g",
            "B1g",
            "B2g",
            "E1g",
            "E2g",
            "A1u",
            "A2u",
            "B1u",
            "B2u",
            "E1u",
            "E2u",
        ],
        [
            [1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2],  # E
            [1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -1],  # 2C6
            [1, 1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1],  # 2C3
            [1, 1, -1, -1, -2, 2, 1, 1, -1, -1, -2, 2],  # C2
            [1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0],  # 3C2'
            [1, -1, -1, 1, 0, 0, 1, -1, -1, 1, 0, 0],  # 3C2''
            [1, 1, 1, 1, 2, 2, -1, -1, -1, -1, -2, -2],  # i
            [1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, 1],  # 2S3
            [1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1],  # 2S6
            [1, 1, -1, -1, -2, 2, -1, -1, 1, 1, 2, -2],  # sh
            [1, -1, 1, -1, 0, 0, -1, 1, -1, 1, 0, 0],  # 3sv
            [1, -1, -1, 1, 0, 0, -1, 1, 1, -1, 0, 0],  # 3sd
        ],
        None,
    ),
    # D2h -- orthorhombic, order 8 (olivine, polyethylene)
    "mmm": (
        ["Ag", "B1g", "B2g", "B3g", "Au", "B1u", "B2u", "B3u"],
        [
            [1, 1, 1, 1, 1, 1, 1, 1],  # E
            [1, 1, -1, -1, 1, 1, -1, -1],  # C2(z)
            [1, -1, 1, -1, 1, -1, 1, -1],  # C2(y)
            [1, -1, -1, 1, 1, -1, -1, 1],  # C2(x)
            [1, 1, 1, 1, -1, -1, -1, -1],  # i
            [1, 1, -1, -1, -1, -1, 1, 1],  # s(xy)
            [1, -1, 1, -1, -1, 1, -1, 1],  # s(xz)
            [1, -1, -1, 1, -1, 1, 1, -1],  # s(yz)
        ],
        None,
    ),
    # C2h -- monoclinic, order 4
    "2/m": (
        ["Ag", "Bg", "Au", "Bu"],
        [
            [1, 1, 1, 1],  # E
            [1, -1, 1, -1],  # C2
            [1, 1, -1, -1],  # i
            [1, -1, -1, 1],  # sh
        ],
        None,
    ),
    # Ci -- triclinic, order 2 (centrosymmetric)
    "-1": (
        ["Ag", "Au"],
        [
            [1, 1],  # E
            [1, -1],  # i
        ],
        None,
    ),
    # C3v -- trigonal, order 6 (wurtzite alternative, NH3)
    "3m": (
        ["A1", "A2", "E"],
        [
            [1, 1, 2],  # E
            [1, 1, -1],  # 2C3
            [1, -1, 0],  # 3sv
        ],
        None,
    ),
    # C4v -- tetragonal, order 8 (ferroelectric BaTiO3)
    "4mm": (
        ["A1", "A2", "B1", "B2", "E"],
        [
            [1, 1, 1, 1, 2],  # E
            [1, 1, -1, -1, 0],  # 2C4
            [1, 1, 1, 1, -2],  # C2
            [1, -1, 1, -1, 0],  # 2sv
            [1, -1, -1, 1, 0],  # 2sd
        ],
        None,
    ),
    # Th -- pyritohedral, order 24 (pyrite FeS2).  Real merged form:
    # Eg/Eu are the complex-conjugate 1-D pairs of T merged into real
    # 2-D rows (chi(C3) = w + w* = -1), and the 8C3 / 8S6 columns merge
    # the 4C3+4C3^2 / 4S6+4S6⁵ conjugacy-class pairs (the merged-pair
    # characters are constant across each pair).  The projector divides
    # by the character-row norm n_a (= 2 for Eg/Eu) so the merged rows
    # project without double counting.
    "m-3": (
        ["Ag", "Au", "Eg", "Eu", "Tg", "Tu"],
        [
            [1, 1, 2, 2, 3, 3],  # E
            [1, 1, -1, -1, 0, 0],  # 8C3
            [1, 1, 2, 2, -1, -1],  # 3C2
            [1, -1, 2, -2, 3, -3],  # i
            [1, -1, -1, 1, 0, 0],  # 8S6
            [1, -1, 2, -2, -1, 1],  # 3sh
        ],
        None,
    ),
}


# Operation type from the (det, trace) of the rotation matrix -- both are
# similarity-invariant, so the *fractional* (integer) rotation matrix can
# be classified directly.  Proper rotations have tr = 1 + 2costh; improper
# (rotoinversions S(th) = i.C(th+pi) == reflection-rotations) have
# tr = 2costh - 1:
#   E (1,3)  C6 (1,2)  C4 (1,1)  C3 (1,0)  C2 (1,-1)
#   i (-1,-3)  S3 (-1,-2)  S4 (-1,-1)  S6 (-1,0)  s (-1,1)
_OP_TYPE_BY_DET_TRACE: Dict[Tuple[int, int], str] = {
    (1, 3): "E",
    (1, 2): "C6",
    (1, 1): "C4",
    (1, 0): "C3",
    (1, -1): "C2",
    (-1, -3): "i",
    (-1, -2): "S3",
    (-1, -1): "S4",
    (-1, 0): "S6",
    (-1, 1): "sigma",
}

# Machine-readable class structure per table, in the same column order as
# the character rows above: (operation type, number of operators).  Where
# two columns share the same (type, count) -- e.g. D4h's 2C2′/2C2″ or
# 3sv/3sd pairs -- computed conjugacy classes are assigned to columns in
# encounter order, so the corresponding irrep labels (B1g vs B2g, ...) are
# fixed only up to that axis convention.  The projections themselves are
# exact either way; only the *labels* of such pairs may swap relative to
# another program's convention.
_CLASS_SPECS: Dict[str, List[Tuple[str, int]]] = {
    "m-3m": [("E", 1), ("C3", 8), ("C2", 3), ("C2", 6), ("C4", 6),
             ("i", 1), ("S6", 8), ("sigma", 3), ("sigma", 6), ("S4", 6)],
    "-43m": [("E", 1), ("C3", 8), ("C2", 3), ("sigma", 6), ("S4", 6)],
    "4/mmm": [("E", 1), ("C4", 2), ("C2", 1), ("C2", 2), ("C2", 2),
              ("i", 1), ("S4", 2), ("sigma", 1), ("sigma", 2), ("sigma", 2)],
    "6mm": [("E", 1), ("C6", 2), ("C3", 2), ("C2", 1),
            ("sigma", 3), ("sigma", 3)],
    "-3m": [("E", 1), ("C3", 2), ("C2", 3), ("i", 1), ("S6", 2),
            ("sigma", 3)],
    "6/mmm": [("E", 1), ("C6", 2), ("C3", 2), ("C2", 1), ("C2", 3),
              ("C2", 3), ("i", 1), ("S3", 2), ("S6", 2), ("sigma", 1),
              ("sigma", 3), ("sigma", 3)],
    "mmm": [("E", 1), ("C2", 1), ("C2", 1), ("C2", 1),
            ("i", 1), ("sigma", 1), ("sigma", 1), ("sigma", 1)],
    "2/m": [("E", 1), ("C2", 1), ("i", 1), ("sigma", 1)],
    "-1": [("E", 1), ("i", 1)],
    "3m": [("E", 1), ("C3", 2), ("sigma", 3)],
    "4mm": [("E", 1), ("C4", 2), ("C2", 1), ("sigma", 2), ("sigma", 2)],
    # Th (m-3): the real table merges the complex-conjugate C3/C3^2 (and
    # S6/S6⁵) conjugacy-class pairs into single 8-member columns, and the
    # complex-pair irreps into real 2-D "Eg"/"Eu" rows.  The class
    # matcher below merges the computed 4+4 classes accordingly, and the
    # projection normalisation handles the merged-pair prefactor.
    "m-3": [("E", 1), ("C3", 8), ("C2", 3), ("i", 1), ("S6", 8),
            ("sigma", 3)],
}


def _rotation_int(op: SymmetryOp) -> np.ndarray:
    """The fractional rotation matrix as an exact integer array."""
    return np.rint(np.asarray(op.rotation, dtype=float)).astype(int)


def _operation_type(R_int: np.ndarray) -> str:
    """Classify a (fractional, integer) rotation by similarity invariants."""
    det = int(round(float(np.linalg.det(R_int))))
    tr = int(R_int.trace())
    try:
        return _OP_TYPE_BY_DET_TRACE[(det, tr)]
    except KeyError:  # pragma: no cover - not a crystallographic rotation
        raise ValueError(
            f"rotation with det={det}, trace={tr} is not a "
            "crystallographic point-group operation"
        ) from None


def _conjugacy_classes(operations: Sequence[SymmetryOp]) -> List[List[int]]:
    """Conjugacy classes of the rotation parts, by exact integer algebra.

    Returns lists of operator indices.  Raises if the rotations do not
    close under conjugation (i.e. the input is not a group).
    """
    from .symmetry_shared import Budget, FiniteGroup

    # These tables cover crystallographic point groups of order at most 48.
    # Admit before copying operations; this is the rotation quotient only.
    if not 1 <= len(operations) <= 48:
        raise ValueError("character-table rotation group must contain 1 to 48 operations")
    budget = Budget(16 << 20, 100_000_000)
    rotations = np.asarray([_rotation_int(op) for op in operations], dtype=np.int64)
    group = FiniteGroup.from_integer_rotations(
        rotations, identity_label="SALC crystallographic rotation quotient", budget=budget,
    )
    return [list(members) for members in group.conjugacy_classes(budget=budget)]


def _match_classes_to_columns(
    classes: List[List[int]],
    op_types: List[str],
    specs: List[Tuple[str, int]],
) -> List[List[int]]:
    """Assign computed conjugacy classes to character-table columns.

    Matching is by (operation type, class size).  A single table column
    may absorb several computed classes of its type when the table uses
    a merged real form (Th's 8C3 = 4C3 + 4C3^2).  Columns sharing the
    same (type, size) -- D4h's 2C2′/2C2″ etc. -- are filled in encounter
    order (label-convention caveat documented on ``_CLASS_SPECS``).
    """
    from collections import defaultdict

    by_type: Dict[str, List[List[int]]] = defaultdict(list)
    for cls in classes:
        by_type[op_types[cls[0]]].append(cls)

    columns_by_type: Dict[str, List[int]] = defaultdict(list)
    for col, (t, _cnt) in enumerate(specs):
        columns_by_type[t].append(col)

    if set(by_type.keys()) != set(columns_by_type.keys()):
        raise ValueError(
            f"operation types {sorted(by_type)} do not match the "
            f"character table's classes {sorted(columns_by_type)}"
        )

    out: List[Optional[List[int]]] = [None] * len(specs)
    for t, cols in columns_by_type.items():
        computed = by_type[t]
        spec_sizes = [specs[c][1] for c in cols]
        comp_sizes = [len(c) for c in computed]
        if len(cols) == len(computed) and sorted(spec_sizes) == sorted(
            comp_sizes
        ):
            # one computed class per column; match sizes in order
            remaining = list(computed)
            for col in cols:
                want = specs[col][1]
                pick = next(
                    idx
                    for idx, c in enumerate(remaining)
                    if len(c) == want
                )
                out[col] = remaining.pop(pick)
        elif len(cols) == 1 and sum(comp_sizes) == spec_sizes[0]:
            # merged real-table column (e.g. Th 8C3 = 4C3 + 4C3^2)
            merged: List[int] = []
            for c in computed:
                merged.extend(c)
            out[cols[0]] = sorted(merged)
        else:
            raise ValueError(
                f"cannot match computed {t} classes of sizes {comp_sizes} "
                f"to table columns of sizes {spec_sizes}"
            )
    return [c for c in out if c is not None]


def character_table(
    point_group: str,
    operations: Sequence[SymmetryOp],
) -> Optional[PointGroupCharacters]:
    """Return the character table for a point group, with the
    operator->class mapping computed from ``operations``.

    Supported point groups: m-3m (Oh), -43m (Td), 4/mmm (D4h),
    6mm (C6v), -3m (D3d), 6/mmm (D6h), mmm (D2h), 2/m (C2h), -1 (Ci),
    3m (C3v), 4mm (C4v), m-3 (Th).  Returns ``None`` for groups without
    a pre-computed table.

    ``class_indices[c]`` lists the indices into ``operations`` whose
    rotations belong to the table's class column ``c`` (conjugacy
    classes computed by exact integer algebra on the fractional
    rotation matrices; merged real-form columns absorb their
    complex-conjugate class pairs).
    """
    if point_group not in _CHARACTER_TABLES:
        return None

    irreps, chars_table, _ = _CHARACTER_TABLES[point_group]
    chars = np.array(chars_table, dtype=float).T  # -> (n_irreps, n_classes)

    classes = _conjugacy_classes(operations)
    op_types = [_operation_type(_rotation_int(op)) for op in operations]
    class_indices = _match_classes_to_columns(
        classes,
        op_types,
        _CLASS_SPECS[point_group],
    )

    return PointGroupCharacters(
        symbol=point_group,
        irreps=list(irreps),
        characters=chars,
        class_indices=class_indices,
    )


def symmetry_project_matrix(
    M: np.ndarray,
    system: PeriodicSystem,
    basis,  # BasisSet
    *,
    operations: Optional[Sequence[SymmetryOp]] = None,
) -> Dict[str, np.ndarray]:
    """Project a matrix M onto the irreducible components of the group
    action ``r(R)[M] = P(R) . M . P(R)ᵀ`` (the adjoint action on AO
    matrices).

    For each irrep a with per-class character chi_a the projector is

        M_a = (d_a / (n_a . |G|)) S_R chi_a(R) . P(R) M P(R)ᵀ

    with d_a = chi_a(E), P(R) the AO representation matrix of R, and
    n_a = (1/|G|) S_R chi_a(R)^2 the Frobenius norm of the character row
    (1 for a genuine real irrep; 2 for the merged complex-conjugate
    pairs of the real Th table), so merged rows project without double
    counting.  The components are complete: S_a M_a = M exactly.

    A totally symmetric matrix (the overlap S, a converged Fock)
    projects entirely into the first irrep (A1g/Ag/A1); non-zero weight
    in other components measures symmetry breaking of M.

    Parameters
    ----------
    M
        (nbf, nbf) matrix to project (typically Fock or density).
    system
        PeriodicSystem with symmetry attached.
    basis
        BasisSet for AO representation.
    operations
        Symmetry operators. If None, uses system.symmetry filtered
        to symmorphic subset.

    Returns
    -------
    Dict mapping irrep label -> (nbf, nbf) projected component.
    Raises ValueError if the point group is not in the character table.
    """
    from .symmetry_scf import build_ao_permutation_cache as _bpc

    M = np.asarray(M, dtype=float)
    if operations is None:
        if system.symmetry is None:
            return {}
        operations = symmorphic_operations(system.symmetry.operations)
    if not operations:
        return {}

    pg = system.symmetry.international_symbol if system.symmetry else ""
    pg_short = ""
    # Extract point group (e.g., "Fm-3m" -> "m-3m")
    ct = character_table(pg, operations)
    if ct is None:
        # Try just the point group symbol
        pg_short = system.symmetry.point_group if system.symmetry else ""
        ct = character_table(pg_short, operations)
    if ct is None:
        raise ValueError(f"No character table for point group {pg} / {pg_short}")

    P_cache = _bpc(system, basis, operations)
    group_order = sum(len(cls) for cls in ct.class_indices)

    # Per-class transformed sums S_{Rinclass} P(R) M P(R)ᵀ -- the
    # chi-independent heavy lifting, shared by every irrep.
    class_sums = [
        sum(P_cache[j] @ M @ P_cache[j].T for j in cls)
        for cls in ct.class_indices
    ]

    result: Dict[str, np.ndarray] = {}
    for ir_idx, irrep in enumerate(ct.irreps):
        chi = ct.characters[ir_idx]  # per-class characters of irrep a
        d_alpha = chi[0]
        norm = (
            sum(
                len(cls) * chi[c] ** 2
                for c, cls in enumerate(ct.class_indices)
            )
            / group_order
        )
        prefactor = d_alpha / (norm * group_order)
        component = prefactor * sum(
            chi[c] * class_sums[c] for c in range(len(class_sums))
        )
        result[irrep] = component

    return result
