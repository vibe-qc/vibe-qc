"""Phase SYM2a: AO-basis representation of a symmetry operator.

Given a molecule or periodic system and a space-group operator
``(R, t)`` (3x3 orthogonal rotation plus a 3-vector translation), build:

1. The **atom permutation** pi such that atom ``a`` at position ``R_a``
   is mapped to atom ``pi(a)`` at position ``R.R_a + t``, plus (for
   periodic systems) the integer lattice shift needed to bring the
   image into the reference cell. Raises if ``(R, t)`` isn't a
   symmetry of the structure.

2. The **AO permutation matrix** ``P`` of shape ``(n_bf, n_bf)`` that
   encodes the operator's action on the AO basis. For each shell of
   angular momentum ``l`` the within-shell block is the real
   Wigner D-matrix ``D^l(R)`` from :mod:`vibeqc.symmetry_core`,
   mapped to the image atom.

Under this representation, the overlap matrix transforms as

    P . S . P^T  =  S

(the basis maps to itself under a symmetry of the nuclear framework),
and the Fock and density matrices inherit the same invariance when
the SCF state has that symmetry. This is the building block for
SYM2b (real-space matrix-block reduction for periodic
``LatticeMatrixSet`` objects) and everything downstream.

Convention
----------

Under ``P``, an AO coefficient vector ``v`` rotates as ``v' = P . v``.
The Wigner D-matrices from :func:`vibeqc.wigner_d_real` use the same
"columns are the old basis, rows are the new" convention, so the
within-shell block of ``P`` is literally ``D^l(R)`` placed at
(destination shell rows, source shell columns) with no transpose.

The existing real Wigner implementation also handles improper operations
through inversion parity. These pure-shell relations require matching radial
contractions and a compatible finite operator support; geometric atom matching
alone does not certify a truncated Fock builder or an electronic state.
The compact molecular adapter below makes those basis checks explicit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple, Union

import numpy as np

from ._vibeqc_core import BasisSet, Molecule, PeriodicSystem
from .symmetry_core import wigner_d_real

if TYPE_CHECKING:
    from .symmetry_shared import BlockSpaceAction, Budget, FiniteGroup, SpaceIdentity


__all__ = [
    "AtomPermutation",
    "atom_permutation_under_op",
    "build_ao_permutation_matrix",
    "build_molecular_space_action",
    "molecular_point_group",
]


# A structure may be either a molecule or a periodic cell. We key the
# permutation logic off duck-typed attributes (``atoms`` for molecules,
# ``unit_cell`` / ``lattice`` for periodic systems) so the function
# handles both without needing a wrapper.
Structure = Union[Molecule, PeriodicSystem]


class AtomPermutation:
    """Result of :func:`atom_permutation_under_op`.

    Attributes
    ----------
    perm
        1-D integer array of length ``n_atoms`` where ``perm[a] = b``
        means atom ``a`` is carried to atom ``b`` by the operator.
    lattice_shift
        ``(n_atoms, 3)`` integer array. For a periodic system,
        ``lattice_shift[a]`` holds the lattice vector (in fractional
        coordinates) to add to the image position so it coincides
        with atom ``perm[a]``'s reference-cell position. For a
        molecule this is always all zeros.
    """

    __slots__ = ("perm", "lattice_shift")

    def __init__(self, perm: np.ndarray, lattice_shift: np.ndarray):
        self.perm = np.asarray(perm, dtype=int)
        self.lattice_shift = np.asarray(lattice_shift, dtype=int)

    def __repr__(self) -> str:
        return (
            f"AtomPermutation(perm={self.perm.tolist()}, "
            f"has_shifts={np.any(self.lattice_shift != 0)})"
        )


def atom_permutation_under_op(
    structure: Structure,
    R: np.ndarray,
    t: np.ndarray = None,
    *,
    tolerance: float = 1e-6,
) -> AtomPermutation:
    """Compute the atom permutation induced by symmetry op ``(R, t)``.

    For each atom ``a`` at position ``r_a``, search for an atom ``b``
    such that ``R.r_a + t = r_b`` (molecule) or
    ``R.r_a + t == r_b + L`` for some lattice vector ``L`` (periodic).

    Raises ``ValueError`` if
    - ``(R, t)`` is not actually a symmetry of the structure (some
      atom has no image, or lands on an atom of a different element),
    - the image of some atom is ambiguous at the given tolerance
      (two atoms within tolerance of the image position -- a
      degenerate geometry).

    Parameters
    ----------
    structure
        :class:`vibeqc.Molecule` (non-periodic) or
        :class:`vibeqc.PeriodicSystem`.
    R
        ``(3, 3)`` orthogonal Cartesian rotation, including improper operations.
    t
        ``(3,)`` translation vector in bohr. Default zero.
    tolerance
        Position-match tolerance in bohr. Default ``1e-6`` is tight
        enough to distinguish symmetrically-inequivalent atoms in
        any reasonable geometry.
    """
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError(f"R must be 3x3, got {R.shape}")
    if t is None:
        t = np.zeros(3)
    else:
        t = np.asarray(t, dtype=float).reshape(3)

    is_periodic = isinstance(structure, PeriodicSystem)
    if is_periodic:
        atoms = list(structure.unit_cell)
        lattice = np.asarray(structure.lattice, dtype=float)
        # columns of `lattice` are a_1, a_2, a_3 so the inverse takes
        # a Cartesian vector to its fractional coordinates.
        inv_lattice = np.linalg.inv(lattice)
    else:
        atoms = list(structure.atoms)

    n_atoms = len(atoms)
    coords = np.array([list(a.xyz) for a in atoms], dtype=float)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 3)
    Zs = np.array([int(a.Z) for a in atoms], dtype=int)

    perm = np.full(n_atoms, -1, dtype=int)
    shifts = np.zeros((n_atoms, 3), dtype=int)

    for a in range(n_atoms):
        new_pos = R @ coords[a] + t
        matches: List[Tuple[int, np.ndarray]] = []
        for b in range(n_atoms):
            if Zs[b] != Zs[a]:
                continue
            if is_periodic:
                # Fractional difference (b_pos - new_pos) must be integer.
                diff = coords[b] - new_pos
                frac = inv_lattice @ diff
                rounded = np.round(frac)
                if np.all(np.abs(frac - rounded) < tolerance):
                    matches.append((b, rounded.astype(int)))
            else:
                if np.linalg.norm(coords[b] - new_pos) < tolerance:
                    matches.append((b, np.zeros(3, dtype=int)))

        if len(matches) == 0:
            raise ValueError(
                f"atom_permutation_under_op: not a symmetry -- atom "
                f"{a} (Z={Zs[a]}) at {coords[a]} has image "
                f"{new_pos} which matches no atom in the structure"
            )
        if len(matches) > 1:
            raise ValueError(
                f"atom_permutation_under_op: atom {a} image is "
                f"ambiguous -- {len(matches)} atoms within tolerance "
                f"{tolerance}. Tighten geometry or lower tolerance."
            )
        perm[a], shifts[a] = matches[0]

    return AtomPermutation(perm, shifts)


# ---------------------------------------------------------------------------
# AO permutation matrix
# ---------------------------------------------------------------------------

def build_ao_permutation_matrix(
    basis: BasisSet,
    R: np.ndarray,
    atom_perm: Union[np.ndarray, AtomPermutation],
) -> np.ndarray:
    """Full ``(n_bf, n_bf)`` AO-basis permutation matrix for the
    symmetry operator ``(R, t)``.

    Shell-by-shell, the within-shell block of ``P`` is the real
    Wigner D-matrix :func:`wigner_d_real(l, R) <wigner_d_real>`,
    placed at

        P[destination_shell_rows, source_shell_columns] = D^l(R)

    The destination shell must have the same angular momentum and exact
    radial contraction on the image atom. Repeated identical contractions
    match by occurrence, so reordered radial channels are supported.

    Parameters
    ----------
    basis
        :class:`vibeqc.BasisSet` whose shells we are rotating.
    R
        ``(3, 3)`` orthogonal Cartesian rotation, including improper operations.
    atom_perm
        Either a raw ``perm[a] = b`` integer array, or the full
        :class:`AtomPermutation` returned by
        :func:`atom_permutation_under_op`. The lattice shift (for
        periodic systems) doesn't enter this matrix -- it's bookkeeping
        for the outer caller who's tracking lattice-cell indices.

    Returns
    -------
    np.ndarray
        ``(n_bf, n_bf)`` real orthogonal matrix.

    Notes
    -----
    Raises ``ValueError`` for incompatible radial contractions, a nonbijective
    atom map, or non-pure shells beyond s. Atom-image phases and finite operator
    supports remain caller responsibilities. This legacy API materializes a
    dense matrix; use the explicitly budgeted compact adapter for panels.
    """
    if isinstance(atom_perm, AtomPermutation):
        perm = atom_perm.perm
    else:
        perm = np.asarray(atom_perm)

    offsets, destinations, rotations = _pure_shell_layout(
        list(basis.shells()), basis.nbasis, perm, R,
    )
    # Compatibility API explicitly requests a dense matrix. Compact callers
    # use the same layout with BlockSpaceAction and never materialize this.
    matrix = np.zeros((basis.nbasis, basis.nbasis), dtype=float)
    for source, destination in enumerate(destinations):
        matrix[offsets[destination]:offsets[destination+1],
               offsets[source]:offsets[source+1]] = rotations[source].real
    return matrix


def _pure_shell_layout(shells, dimension, permutation, rotation, *, maximum_l=None):
    """Shared radial-channel matching and small pure-shell rotations.

    Resource admission and atom/origin validation belong to the caller. The
    legacy dense API has already copied its metadata; the compact adapter
    admits that snapshot first. No atom-image shift is discarded here.
    """
    permutation = np.asarray(permutation)
    if (permutation.ndim != 1 or permutation.dtype.kind not in "iu"
            or sorted(permutation.tolist()) != list(range(len(permutation)))):
        raise ValueError("AO atom action must be a permutation")
    offsets = [0]
    profiles = []
    by_profile: dict[tuple, list[int]] = {}
    occurrences = []
    for i, sh in enumerate(shells):
        # The legacy dense API also accepts Cartesian s shells: their
        # one-dimensional scalar action is identical to the pure action.
        if (not sh.pure and sh.l != 0) or sh.l < 0:
            raise ValueError("AO symmetry requires pure shells beyond s")
        if maximum_l is not None and sh.l > maximum_l:
            raise ValueError("compact molecular action requires pure shells through l=6")
        atom = int(sh.atom_index)
        if not 0 <= atom < len(permutation):
            raise ValueError("AO shell atom index outside atom permutation")
        exponents, coefficients = tuple(sh.exponents), tuple(sh.coefficients)
        if not np.all(np.isfinite(exponents+coefficients)):
            raise ValueError("nonfinite AO radial profile")
        profile = (sh.l, sh.pure, exponents, coefficients)
        profiles.append(profile)
        offsets.append(offsets[-1]+2*sh.l+1)
        siblings = by_profile.setdefault((atom, profile), [])
        occurrences.append(len(siblings))
        siblings.append(i)
    if offsets[-1] != dimension:
        raise ValueError("AO shell layout disagrees with basis dimension")
    destinations, rotations = [], []
    rotation_cache: dict[int, np.ndarray] = {}
    for i, sh in enumerate(shells):
        siblings = by_profile[(sh.atom_index, profiles[i])]
        candidates = by_profile.get((int(permutation[sh.atom_index]), profiles[i]), [])
        if len(candidates) != len(siblings):
            raise ValueError("AO symmetry radial contraction mismatch")
        destinations.append(candidates[occurrences[i]])
        if sh.l not in rotation_cache:
            rotation_cache[sh.l] = np.asarray(wigner_d_real(sh.l, rotation))
        rotations.append(rotation_cache[sh.l])
    return offsets, destinations, rotations


def build_molecular_space_action(
    molecule: Molecule, basis: BasisSet, rotation: np.ndarray,
    translation: np.ndarray, *, source: SpaceIdentity, target: SpaceIdentity, budget: Budget,
    antiunitary: bool = False, tolerance: float = 1e-9,
) -> BlockSpaceAction:
    """Experimental molecular pure-shell adapter to shared compact transport.

    Admits native metadata before copying ShellInfo records, validates exact
    radial profiles and origins, then uses the existing Wigner rotations.
    Does not change molecular SCF routing or authorize integral skipping.
    """
    from .symmetry_shared import BlockSpaceAction, Budget, _array, _core, _tolerance

    if not isinstance(molecule, Molecule):
        raise TypeError("molecular space action requires a Molecule")
    if not isinstance(budget, Budget):
        raise TypeError("molecular space action requires a Budget")
    tolerance = _tolerance(tolerance)
    if not 0 < tolerance <= 1e-6:
        raise ValueError("molecular symmetry tolerance must lie in (0,1e-6]")
    _array(rotation, "float64", 2)
    _array(translation, "float64", 1)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("molecular operation shape mismatch")
    if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
        raise ValueError("nonfinite molecular operation")
    if np.max(np.abs(rotation.T @ rotation-np.eye(3))) > tolerance:
        raise ValueError("molecular rotation is not orthogonal")
    metadata_bytes = _core()._symmetry_basis_snapshot_bytes(basis, budget.native())
    # Bound atom search and block/matrix payload before copying descriptors.
    n = int(basis.nbasis)
    na = _core()._symmetry_molecule_count(molecule, budget.native())
    # At l<=6, packed shell rotations contain at most 13*n values.
    # Reserve radial/profile metadata, packed arrays and immutable snapshots,
    # plus fixed Wigner workspace. No n-by-n AO array is materialized here.
    budget.admit(metadata_bytes+131072+4096*n+1024*na,
                 256*na*na+512*n+7_000_000)
    shells = list(basis.shells())
    atoms = list(molecule.atoms)
    permutation = atom_permutation_under_op(
        molecule, rotation, translation, tolerance=tolerance,
    ).perm
    if len(set(permutation.tolist())) != na:
        raise ValueError("molecular atom action is not bijective")
    for sh in shells:
        if not sh.pure:
            raise ValueError("compact molecular action requires pure shells through l=6")
        atom = int(sh.atom_index)
        if (not 0 <= atom < na or not np.all(np.isfinite(sh.origin))
                or np.linalg.norm(np.asarray(sh.origin)-atoms[atom].xyz) > tolerance):
            raise ValueError("molecular basis origin/atom mismatch")
    offsets, destinations, rotations = _pure_shell_layout(
        shells, n, permutation, rotation, maximum_l=6,
    )
    return BlockSpaceAction(source, target, np.asarray(offsets, dtype=np.int64),
        np.asarray(destinations, dtype=np.int64),
        np.asarray(np.concatenate([block.ravel() for block in rotations]), dtype=np.complex128),
        antiunitary=antiunitary, budget=budget)


def molecular_point_group(molecule: Molecule, rotations: np.ndarray,
                          translations: np.ndarray, *, identity_label: str,
                          budget: Budget, tolerance: float = 1e-9) -> FiniteGroup:
    """Admit supplied Cartesian molecular operations through the shared group.

    Geometry and Seitz matching are numerical; group laws of the resulting
    table are exact. No search for the complete molecular point group occurs.
    """
    from .symmetry_shared import Budget, FiniteGroup, _array, _core, _tolerance
    if not isinstance(molecule, Molecule) or not isinstance(budget, Budget):
        raise TypeError("molecular point group requires a Molecule and Budget")
    _array(rotations, "float64", 3)
    _array(translations, "float64", 2)
    n = rotations.shape[0]
    if n < 1 or rotations.shape != (n, 3, 3) or translations.shape != (n, 3):
        raise ValueError("molecular group operation shape mismatch")
    tolerance = _tolerance(tolerance)
    if not 0 < tolerance <= 1e-6:
        raise ValueError("molecular group tolerance must lie in (0,1e-6]")
    na = _core()._symmetry_molecule_count(molecule, budget.native())
    budget.admit(4096+128*n*n+256*n+1024*na,
                 1024*n**3+256*n*na**2)
    if not np.all(np.isfinite(rotations)) or not np.all(np.isfinite(translations)):
        raise ValueError("nonfinite molecular group operation")
    for r, t in zip(rotations, translations):
        if np.max(np.abs(r.T @ r-np.eye(3))) > tolerance:
            raise ValueError("molecular group rotation is not orthogonal")
        atom_permutation_under_op(molecule, r, t, tolerance=tolerance)
    def match(r, t):
        candidates = [i for i in range(n)
                      if np.max(np.abs(rotations[i]-r)) <= tolerance
                      and np.linalg.norm(translations[i]-t) <= tolerance]
        if len(candidates) != 1:
            raise ValueError("molecular operations are not uniquely closed")
        return candidates[0]
    identity = match(np.eye(3), np.zeros(3))
    table = np.empty((n, n), dtype=np.int64)
    for g in range(n):
        for h in range(n):
            table[g, h] = match(rotations[g] @ rotations[h],
                                rotations[g] @ translations[h]+translations[g])
    return FiniteGroup.from_table(table, np.zeros(n, dtype=np.uint8),
        identity=identity, identity_label=identity_label, budget=budget)
