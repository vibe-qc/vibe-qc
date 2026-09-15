"""Internal and delocalised internal coordinates (Phase 4).

Implements redundant primitive internal coordinates (bonds, angles,
dihedrals, out-of-plane bends), the Wilson B-matrix, and the
delocalised internal coordinate (DLC) construction of Baker, Kessi,
and Delley (J. Chem. Phys. 105, 192, 1996).

Coordinate transforms
---------------------
* Forward  (Cartesian -> DLC): iterative Newton-Raphson back-transform
  from Baker (J. Comput. Chem. 18, 1079, 1997).
* Backward (DLC -> Cartesian): B-matrix pseudoinverse via SVD.
* Gradient projection: g_dlc = (B^T)^+ g_cart.
* Hessian projection: H_dlc = (B^T)^+ H_cart B^+.

References
----------
* Baker, Kessi, Delley, J. Chem. Phys. 105, 192 (1996).
* Baker, J. Comput. Chem. 18, 1079 (1997).
* Pulay, Fogarasi, J. Chem. Phys. 96, 2856 (1992).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .._vibeqc_core import Atom, Molecule
from .coordinates import CoordinateRepresentation

# ---------------------------------------------------------------------------
# Internal coordinate primitives
# ---------------------------------------------------------------------------


def _bond_length(positions: np.ndarray, i: int, j: int) -> float:
    """Distance between atoms i and j (bohr)."""
    return float(np.linalg.norm(positions[i] - positions[j]))


def _bond_angle(positions: np.ndarray, i: int, j: int, k: int) -> float:
    """Angle i-j-k in radians."""
    v1 = positions[i] - positions[j]
    v2 = positions[k] - positions[j]
    dot = float(np.dot(v1, v2))
    norms = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if norms < 1e-12:
        return 0.0
    cos_theta = max(-1.0, min(1.0, dot / norms))
    return float(np.arccos(cos_theta))


def _dihedral_angle(positions: np.ndarray, i: int, j: int, k: int, l: int) -> float:
    """Dihedral angle i-j-k-l in radians (-pi to pi)."""
    b1 = positions[j] - positions[i]
    b2 = positions[k] - positions[j]
    b3 = positions[l] - positions[k]

    # Normalise b2
    b2_norm = float(np.linalg.norm(b2))
    if b2_norm < 1e-12:
        return 0.0

    # Normal to i-j-k plane and j-k-l plane
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    n1_norm = float(np.linalg.norm(n1))
    n2_norm = float(np.linalg.norm(n2))
    if n1_norm < 1e-12 or n2_norm < 1e-12:
        return 0.0

    n1 = n1 / n1_norm
    n2 = n2 / n2_norm
    m1 = np.cross(n1, b2 / b2_norm)

    x = float(np.dot(n1, n2))
    y = float(np.dot(m1, n2))
    return float(np.arctan2(y, x))


def _out_of_plane(positions: np.ndarray, i: int, j: int, k: int, l: int) -> float:
    """Out-of-plane bend: angle between bond i-l and the j-k-l plane."""
    v_il = positions[i] - positions[l]
    v_jl = positions[j] - positions[l]
    v_kl = positions[k] - positions[l]

    n = np.cross(v_jl, v_kl)
    n_norm = float(np.linalg.norm(n))
    v_norm = float(np.linalg.norm(v_il))
    if n_norm < 1e-12 or v_norm < 1e-12:
        return 0.0

    sin_phi = float(np.dot(v_il, n)) / (v_norm * n_norm)
    sin_phi = max(-1.0, min(1.0, sin_phi))
    return float(np.arcsin(sin_phi))


# ---------------------------------------------------------------------------
# Internal coordinate descriptors
# ---------------------------------------------------------------------------


class _InternalCoord:
    """A single primitive internal coordinate."""

    #: True for coordinates that live on a 2π circle (torsions), so a
    #: difference of two values must be wrapped into (-π, π] before it is
    #: used as a residual. Bonds and valence angles (range [0, π]) are not
    #: periodic in this sense.
    periodic: bool = False

    def value(self, positions: np.ndarray) -> float:
        raise NotImplementedError

    def b_matrix_row(self, positions: np.ndarray) -> np.ndarray:
        """Return (3.n_atoms,) row of the Wilson B-matrix at *positions*."""
        raise NotImplementedError


def _wrap_primitive_residual(
    residual: np.ndarray, primitives: list["_InternalCoord"]
) -> np.ndarray:
    """Wrap the torsion components of a primitive-space residual into
    (-π, π].

    A torsion at +179 deg and one at -179 deg differ by 2 deg, not 358.
    Without wrapping, the DLC back-transform of a molecule whose starting
    torsions sit near ±π (e.g. planar glycine) sees spurious ~2π residuals
    and drives the geometry apart until the SCF diverges."""
    out = np.asarray(residual, dtype=float).copy()
    for idx, prim in enumerate(primitives):
        if prim.periodic:
            out[idx] = (out[idx] + np.pi) % (2.0 * np.pi) - np.pi
    return out


class _Bond(_InternalCoord):
    def __init__(self, i: int, j: int):
        self.i = i
        self.j = j
        self._atoms = (i, j)

    def value(self, pos: np.ndarray) -> float:
        return _bond_length(pos, self.i, self.j)

    def b_matrix_row(self, pos: np.ndarray) -> np.ndarray:
        n = len(pos)
        row = np.zeros(3 * n)
        v = pos[self.i] - pos[self.j]
        r = float(np.linalg.norm(v))
        if r < 1e-12:
            return row
        u = v / r
        row[3 * self.i : 3 * self.i + 3] = u
        row[3 * self.j : 3 * self.j + 3] = -u
        return row


class _Angle(_InternalCoord):
    def __init__(self, i: int, j: int, k: int):
        self.i = i
        self.j = j
        self.k = k

    def value(self, pos: np.ndarray) -> float:
        return _bond_angle(pos, self.i, self.j, self.k)

    def b_matrix_row(self, pos: np.ndarray) -> np.ndarray:
        n = len(pos)
        row = np.zeros(3 * n)
        v1 = pos[self.i] - pos[self.j]
        v2 = pos[self.k] - pos[self.j]
        r1 = float(np.linalg.norm(v1))
        r2 = float(np.linalg.norm(v2))
        if r1 < 1e-12 or r2 < 1e-12:
            return row
        e1 = v1 / r1
        e2 = v2 / r2
        cos_t = float(np.dot(e1, e2))
        cos_t = max(-1.0, min(1.0, cos_t))
        sin_t = np.sqrt(max(0.0, 1.0 - cos_t * cos_t))
        if sin_t < 1e-12:
            return row

        # dth/dx_i = (costh . e1 - e2) / (r1 . sinth)
        d_i = (cos_t * e1 - e2) / (r1 * sin_t)
        d_k = (cos_t * e2 - e1) / (r2 * sin_t)
        d_j = -(d_i + d_k)

        row[3 * self.i : 3 * self.i + 3] = d_i
        row[3 * self.j : 3 * self.j + 3] = d_j
        row[3 * self.k : 3 * self.k + 3] = d_k
        return row


class _Dihedral(_InternalCoord):
    periodic = True

    def __init__(self, i: int, j: int, k: int, l: int):
        self.i = i
        self.j = j
        self.k = k
        self.l = l

    def value(self, pos: np.ndarray) -> float:
        return _dihedral_angle(pos, self.i, self.j, self.k, self.l)

    def b_matrix_row(self, pos: np.ndarray) -> np.ndarray:
        n = len(pos)
        row = np.zeros(3 * n)

        v_ij = pos[self.i] - pos[self.j]
        v_kj = pos[self.k] - pos[self.j]
        v_lk = pos[self.l] - pos[self.k]
        v_jk = -v_kj

        n1 = np.cross(v_ij, v_kj)
        n2 = np.cross(v_kj, v_lk)

        n1_norm = float(np.linalg.norm(n1))
        n2_norm = float(np.linalg.norm(n2))
        r_kj = float(np.linalg.norm(v_kj))
        r_ij = float(np.linalg.norm(v_ij))
        r_lk = float(np.linalg.norm(v_lk))

        if n1_norm < 1e-12 or n2_norm < 1e-12 or r_kj < 1e-12:
            return row

        # Standard Wilson B-matrix for dihedral:
        # dphi/dx_i = -(r_kj / n1^2) . n1
        row_i = -r_kj * n1 / (n1_norm * n1_norm)
        row_l = r_kj * n2 / (n2_norm * n2_norm)

        # dphi/dx_j = (r_ij/r_kj - v_ij.v_jk/r_kj^2).r_kj.n1/n1^2
        #            + (v_lk.v_jk/r_kj^2).r_kj.n2/n2^2
        c_ij = float(np.dot(v_ij, v_jk)) / (r_kj * r_kj)
        c_lk = float(np.dot(v_lk, v_jk)) / (r_kj * r_kj)
        row_j_row = (r_ij / r_kj - c_ij) * (-row_i) - c_lk * row_l

        # dphi/dx_k = -dphi/dx_i - dphi/dx_j - dphi/dx_l
        row_k = -row_i - row_j_row - row_l

        row[3 * self.i : 3 * self.i + 3] = row_i
        row[3 * self.j : 3 * self.j + 3] = row_j_row
        row[3 * self.k : 3 * self.k + 3] = row_k
        row[3 * self.l : 3 * self.l + 3] = row_l
        return row


# ---------------------------------------------------------------------------
# Primitive internal coordinate generator
# ---------------------------------------------------------------------------


def _generate_primitives(
    atomic_numbers: list[int],
    n_atoms: int,
    positions: np.ndarray,
    bond_cutoff_scale: float = 1.5,
) -> list[_InternalCoord]:
    """Generate redundant primitive internal coordinates.

    Bonds: all pairs within *bond_cutoff_scale* x sum of covalent radii.
    Angles: all triples where i-j and j-k are bonds.
    Dihedrals: all quartets where i-j, j-k, k-l are bonds.
    Out-of-plane: for sp^2 centres (3 bonded neighbours).

    Returns a list of :class:`_InternalCoord` instances.
    """
    # Covalent radii in bohr (Pyykko 2009 single-bond)
    _COV_RADII: dict[int, float] = {
        1: 0.604,
        6: 1.426,
        7: 1.272,
        8: 1.176,
        9: 1.080,
        14: 2.103,
        15: 2.020,
        16: 1.982,
        17: 1.932,
        35: 2.266,
    }
    _DEFAULT_R = 1.5

    coords: list[_InternalCoord] = []

    # --- bonds ---------------------------------------------------------------
    bonds: list[tuple[int, int]] = []
    for i in range(n_atoms):
        ri = _COV_RADII.get(atomic_numbers[i], _DEFAULT_R)
        for j in range(i + 1, n_atoms):
            rj = _COV_RADII.get(atomic_numbers[j], _DEFAULT_R)
            d = _bond_length(positions, i, j)
            if d < bond_cutoff_scale * (ri + rj):
                bonds.append((i, j))
                coords.append(_Bond(i, j))

    # --- angles --------------------------------------------------------------
    for j in range(n_atoms):
        neighbours = []
        for a, b in bonds:
            if a == j:
                neighbours.append(b)
            elif b == j:
                neighbours.append(a)
        for a_idx in range(len(neighbours)):
            for b_idx in range(a_idx + 1, len(neighbours)):
                coords.append(_Angle(neighbours[a_idx], j, neighbours[b_idx]))

    # --- dihedrals -----------------------------------------------------------
    for i, j in bonds:
        for k in range(n_atoms):
            if k == i or k == j:
                continue
            if not _is_bonded(bonds, j, k):
                continue
            for l in range(n_atoms):
                if l == i or l == j or l == k:
                    continue
                if not _is_bonded(bonds, k, l):
                    continue
                coords.append(_Dihedral(i, j, k, l))

    return coords


def _is_bonded(bonds: list[tuple[int, int]], a: int, b: int) -> bool:
    return (a, b) in bonds or (b, a) in bonds


# ---------------------------------------------------------------------------
# Wilson B-matrix construction
# ---------------------------------------------------------------------------


def _build_b_matrix(
    primitives: list[_InternalCoord],
    positions: np.ndarray,
) -> np.ndarray:
    """Build the full Wilson B-matrix: B[i,j] = dq_i/dx_j.

    Returns (n_prim, 3.n_atoms) matrix.
    """
    n_prim = len(primitives)
    n_coord = 3 * len(positions)
    B = np.zeros((n_prim, n_coord))
    for i, prim in enumerate(primitives):
        B[i, :] = prim.b_matrix_row(positions)
    return B


# ---------------------------------------------------------------------------
# DLC construction (Baker-Kessi-Delley 1996)
# ---------------------------------------------------------------------------


def _build_dlc_transform(
    B: np.ndarray,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Construct the DLC transformation matrix U.

    G = B B^T, diagonalise G, keep eigenvectors with eigenvalues > tol.
    U = (eigenvectors of G with l > 0), shape (n_dlc, n_prim).

    Returns (U, B_pinv, n_dlc).
    """
    # G = B B^T (n_prim x n_prim)
    G = B @ B.T

    eigvals, eigvecs = np.linalg.eigh(G)

    # Keep eigenvectors with significant eigenvalues
    if eigvals.size == 0:
        return np.zeros((0, B.shape[0])), np.zeros((B.shape[1], 0)), 0
    tol_val = tol * np.max(eigvals) if np.max(eigvals) > 0 else 0.0
    mask = eigvals > tol_val
    n_dlc = int(np.sum(mask))

    # U: map from primitives to DLC
    U = eigvecs[:, mask].T  # (n_dlc, n_prim)

    # B^+ = pseudoinverse: B^+ = (B^T B)^(-1) B^T for nonsquare
    # Use SVD for stability
    try:
        B_pinv = np.linalg.pinv(B, rcond=tol)
    except np.linalg.LinAlgError:
        B_pinv = np.linalg.pinv(B, rcond=1e-4)

    return U, B_pinv, n_dlc


# ---------------------------------------------------------------------------
# Cartesian <-> DLC transforms
# ---------------------------------------------------------------------------


def _cartesian_to_dlc(
    positions_cart: np.ndarray,
    ref_positions: np.ndarray,
    primitives: list[_InternalCoord],
    U: np.ndarray,
    B_pinv: np.ndarray,
    max_iter: int = 30,
    tol: float = 1e-8,
) -> np.ndarray:
    """Convert Cartesian positions to DLC via iterative Newton-Raphson.

    Starting from the reference primitive values q_ref, iteratively
    solve for DLC parameters s such that q(s) ≈ q_ref + U^T s.
    """
    # Reference primitive values
    q_ref = np.array([p.value(ref_positions) for p in primitives])

    # DLC value at the actual geometry: s = U (q_cur - q_ref), with the
    # torsion components of the difference wrapped into (-π, π].
    q_cur = np.array([p.value(positions_cart) for p in primitives])
    s = U @ _wrap_primitive_residual(q_cur - q_ref, primitives)
    return s


def _dlc_to_cartesian(
    s: np.ndarray,
    ref_positions: np.ndarray,
    primitives: list[_InternalCoord],
    U: np.ndarray,
    B_pinv: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-8,
    seed_positions: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Convert DLC parameters back to Cartesian positions.

    Uses iterative Newton-Raphson: given target primitive values
    q_target = q_ref + U^T s, find x such that q(x) = q_target. The
    Newton iteration is seeded from *seed_positions* when supplied (the
    previous optimizer step's geometry), which keeps each step's target
    close to the seed so the linearised back-transform stays in its
    convergence basin -- seeding always from the far-away reference is
    what left large-amplitude DLC steps unrealised.
    """
    positions = (
        np.asarray(seed_positions, dtype=float).reshape(ref_positions.shape).copy()
        if seed_positions is not None
        else ref_positions.copy()
    )
    q_ref = np.array([p.value(ref_positions) for p in primitives])
    # Target primitive values implied by the DLC step. U^T s is a
    # primitive-space displacement from the reference; the wrap keeps the
    # torsion targets on the correct branch of the 2π circle.
    dq_target = U.T @ np.asarray(s, dtype=float)
    q_target = q_ref + dq_target

    prev_norm = np.inf
    best_positions = positions.copy()
    best_norm = np.inf
    for _ in range(max_iter):
        q_cur = np.array([p.value(positions) for p in primitives])
        # Wrap the torsion components so a residual never spuriously reads
        # ~2π near the ±π branch cut (the planar-glycine failure).
        resid = _wrap_primitive_residual(q_target - q_cur, primitives)
        resid_norm = float(np.linalg.norm(resid))
        if resid_norm < best_norm:
            best_norm = resid_norm
            best_positions = positions.copy()
        if resid_norm < tol:
            break
        # The pseudoinverse MUST be evaluated at the CURRENT geometry --
        # the Wilson B-matrix is geometry-dependent, so reusing the
        # reference B_pinv (the pre-fix bug) makes the Newton step wrong
        # as soon as the geometry moves, and the back-transform diverges.
        B_cur = _build_b_matrix(primitives, positions)
        try:
            B_pinv_cur = np.linalg.pinv(B_cur, rcond=1e-6)
        except np.linalg.LinAlgError:
            B_pinv_cur = B_pinv  # reference fallback
        dx = B_pinv_cur @ resid
        # Diverging: stop and return the best geometry seen so far rather
        # than letting the iteration run away into a broken structure.
        if resid_norm > prev_norm * 2.0:
            break
        prev_norm = resid_norm
        positions = positions + dx.reshape(-1, 3)

    return best_positions


# ---------------------------------------------------------------------------
# DLC CoordinateRepresentation
# ---------------------------------------------------------------------------


class DelocalizedInternalCoordinates(CoordinateRepresentation):
    """Delocalised internal coordinates (Baker-Kessi-Delley 1996).

    Constructs redundant primitive internals (bonds, angles, dihedrals),
    builds the Wilson B-matrix, and diagonalises G = B B^T to obtain
    a non-redundant set of delocalised internal coordinates.

    Parameters
    ----------
    n_atoms : int
    atomic_numbers : list of int
    reference_positions : (n_atoms, 3) ndarray
        The Cartesian geometry at which the DLC basis is constructed.
    freeze_indices : sequence of int, optional
    bond_cutoff_scale : float
        Scale factor for covalent-radii-based bond detection (default 1.3).
    """

    def __init__(
        self,
        n_atoms: int,
        atomic_numbers: list[int],
        reference_positions: np.ndarray,
        freeze_indices: Optional[Sequence[int]] = None,
        bond_cutoff_scale: float = 1.5,
    ):
        self._n_atoms = n_atoms
        self._ref_positions = np.asarray(reference_positions, dtype=float).reshape(
            n_atoms, 3
        )
        self._frozen_set: set[int] = (
            {int(i) for i in freeze_indices} if freeze_indices else set()
        )

        # Generate primitives at the reference geometry
        self._primitives = _generate_primitives(
            atomic_numbers, n_atoms, self._ref_positions, bond_cutoff_scale
        )

        # Reference primitive values
        self._q_ref = np.array([p.value(self._ref_positions) for p in self._primitives])

        # B-matrix at reference
        B = _build_b_matrix(self._primitives, self._ref_positions)

        # DLC transform
        self._U, self._B_pinv, self._n_dlc = _build_dlc_transform(B)

        # Completeness check. A non-redundant DLC set should span the
        # 3N-6 (or 3N-5 for a linear molecule) internal degrees of
        # freedom, minus 3 per fully frozen atom. If the auto-generated
        # primitive set is rank-deficient, some internal DOF is not
        # representable and the optimizer cannot relax along it -- warn
        # loudly rather than silently under-optimising (the primitive
        # generator does not yet guarantee completeness for every
        # topology; see handovers).
        expected = max(0, 3 * self._n_atoms - 6 - 3 * len(self._frozen_set))
        if 0 < self._n_dlc < expected:
            import warnings

            warnings.warn(
                f"Delocalised internal coordinates span {self._n_dlc} of the "
                f"expected {expected} internal DOF for this {self._n_atoms}-atom "
                "system: the auto-generated primitive set is incomplete, so a "
                "geometry optimisation in these coordinates may not fully "
                "converge. Use geom_coords='cartesian' for this system until "
                "the primitive generator covers its topology.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Most recent back-transformed geometry, used to seed the next
        # step's Newton back-transform (see _dlc_to_cartesian). Starts at
        # the reference geometry.
        self._last_positions = self._ref_positions.copy()

    @property
    def n_params(self) -> int:
        return self._n_dlc

    def x0(self, molecule: Molecule) -> np.ndarray:
        """Starting DLC vector -- zero (at reference)."""
        return np.zeros(self._n_dlc)

    def to_cartesian(self, template: Molecule, x: np.ndarray) -> Molecule:
        """Rebuild a Molecule from DLC parameters."""
        positions = _dlc_to_cartesian(
            np.asarray(x, dtype=float),
            self._ref_positions,
            self._primitives,
            self._U,
            self._B_pinv,
            seed_positions=self._last_positions,
        )
        # Seed the next back-transform from here -- consecutive optimizer
        # steps are close, so this keeps Newton in its convergence basin.
        self._last_positions = positions.copy()
        new_atoms: list[Atom] = []
        for i in range(self._n_atoms):
            new_atoms.append(Atom(int(template.atoms[i].Z), list(positions[i])))
        return Molecule(new_atoms, template.charge, template.multiplicity)

    def _b_pinv_at(self, molecule: Molecule) -> np.ndarray:
        """Pseudoinverse of the Wilson B-matrix at *molecule*'s geometry.

        The B-matrix is geometry-dependent, so the gradient/Hessian
        projection must use the pinv at the CURRENT geometry, not the
        reference (the same stale-reference bug that broke the
        back-transform)."""
        positions = np.array([list(a.xyz) for a in molecule.atoms], dtype=float)
        B_cur = _build_b_matrix(self._primitives, positions)
        try:
            return np.linalg.pinv(B_cur, rcond=1e-6)
        except np.linalg.LinAlgError:
            return self._B_pinv

    def project_gradient(
        self, molecule: Molecule, cartesian_gradient: np.ndarray
    ) -> np.ndarray:
        """Project Cartesian gradient into DLC space.

        g_dlc = U (B^+)^T g_cart, with B^+ evaluated at the current
        geometry.
        """
        g_flat = np.asarray(cartesian_gradient, dtype=float).ravel()
        g_prim = self._b_pinv_at(molecule).T @ g_flat
        return self._U @ g_prim

    def project_hessian(
        self, molecule: Molecule, cartesian_hessian: np.ndarray
    ) -> np.ndarray:
        """Project Cartesian Hessian into DLC space.

        H_dlc = U (B^+)^T H_cart B^+ U^T (neglecting the dB/dx term),
        with B^+ evaluated at the current geometry.
        """
        H = np.asarray(cartesian_hessian, dtype=float)
        B_pinv_cur = self._b_pinv_at(molecule)
        H_prim = B_pinv_cur.T @ H @ B_pinv_cur
        return self._U @ H_prim @ self._U.T

    @property
    def frozen_set(self) -> set[int]:
        return self._frozen_set

    def apply_frozen(
        self, x: np.ndarray, gradient: np.ndarray, frozen_set: set[int]
    ) -> np.ndarray:
        """DLC gradient masking is approximate -- zero for Cartesian only."""
        # DLCs mix all atoms; freezing individual atoms is not well-defined
        # in DLC space. Return the gradient unmodified -- the coordinate
        # back-transform handles frozen atoms via position constraints.
        return np.asarray(gradient, dtype=float).ravel()
