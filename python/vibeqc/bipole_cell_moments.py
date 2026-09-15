"""BIPOLE multipole-far-pair branch -- Phase 4a: cell-level multipole
moments from periodic density + shell-pair moments.

The full cell density per unit cell is

    r_cell(r) = S_{muν, g} P_muν(g) . phi_mu(r) . phi_ν(r - R_g)

(here ``P_muν(g) = <P>_{mu_0, ν_g}`` is the (mu_0, ν_g) block of the
real-space density matrix at lattice cell ``g``). Its multipole
moments around an expansion origin ``O`` are::

    M^cell_{i,j,k}(O) = ∫ r_cell(r) . (r_x - O_x)^i (r_y - O_y)^j
                                       (r_z - O_z)^k dr

  = S_{muν, g} P_muν(g) . < phi_mu | (r_x-O_x)^i (r_y-O_y)^j (r_z-O_z)^k | phi_ν(r-R_g) >

  = S_{muν, g} P_muν(g) . M^{muν}_{i,j,k}(g; O)                       (*)

The inner <...> matrix is exactly what ``compute_multipole_moments_lattice``
returns (Phase 1). This module performs the contraction (*) over the
density matrix and produces the cell-level Cartesian moment vector.

These complete-cell moments are useful diagnostics and building blocks, but
lattice-summing them does not by itself reproduce CRYSTAL's printed
``EXT EL-POLE``.  Saunders et al. (1992), Eqs. 116-130, instead use
shell-partitioned point multipoles of a chosen order together with a ``T2``
penetration-zone subtraction.  The constant-``Q`` term from the density
defect's spherical second moment is reported separately as
``EXT EL-SPHEROPOLE``.

Module status: Phase 4a complete-cell moments landed.  The CRYSTAL component
decomposition remains gated until the shell partition, multipole order, and
penetration criterion are explicit; the production total-energy route uses
the separate exact Ewald-J split.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ._vibeqc_core import (
    LatticeMatrixSet,
    LatticeMultipoleSet,
    cartesian_multipole_n_components,
)

__all__ = [
    "CellMultipoleMoments",
    "compute_cell_multipole_moments",
    "cartesian_component_label",
    "cartesian_component_indices",
]


# Cartesian component (i, j, k) order matches libint's emultipole{1,2,3}.
# l = i + j + k; layout is (S, x, y, z, xx, xy, xz, yy, yz, zz, xxx, ...).
_CARTESIAN_LAYOUT_BY_L_MAX = {
    0: [(0, 0, 0)],
    1: [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
    2: [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (2, 0, 0),
        (1, 1, 0),
        (1, 0, 1),
        (0, 2, 0),
        (0, 1, 1),
        (0, 0, 2),
    ],
    3: [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (2, 0, 0),
        (1, 1, 0),
        (1, 0, 1),
        (0, 2, 0),
        (0, 1, 1),
        (0, 0, 2),
        (3, 0, 0),
        (2, 1, 0),
        (2, 0, 1),
        (1, 2, 0),
        (1, 1, 1),
        (1, 0, 2),
        (0, 3, 0),
        (0, 2, 1),
        (0, 1, 2),
        (0, 0, 3),
    ],
    4: [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (2, 0, 0),
        (1, 1, 0),
        (1, 0, 1),
        (0, 2, 0),
        (0, 1, 1),
        (0, 0, 2),
        (3, 0, 0),
        (2, 1, 0),
        (2, 0, 1),
        (1, 2, 0),
        (1, 1, 1),
        (1, 0, 2),
        (0, 3, 0),
        (0, 2, 1),
        (0, 1, 2),
        (0, 0, 3),
        (4, 0, 0),
        (3, 1, 0),
        (3, 0, 1),
        (2, 2, 0),
        (2, 1, 1),
        (2, 0, 2),
        (1, 3, 0),
        (1, 2, 1),
        (1, 1, 2),
        (1, 0, 3),
        (0, 4, 0),
        (0, 3, 1),
        (0, 2, 2),
        (0, 1, 3),
        (0, 0, 4),
    ],
}


def cartesian_component_indices(L_max: int) -> List[tuple]:
    """Return the list of (i, j, k) tuples for Cartesian components.

    Ordered to match the libint emultipole buffer layout:
    for each L=0..L_max, iterate i=L..0, j=L-i..0, k=L-i-j.

    This matches the C++ multipole_cartesian_indices function in
    bipole_multipole.cpp and the actual libint emultipole{1,2,3}
    buffer convention (confirmed empirically).
    """
    result = []
    for L in range(L_max + 1):
        for i in range(L, -1, -1):
            for j in range(L - i, -1, -1):
                k = L - i - j
                result.append((i, j, k))
    return result


def cartesian_component_label(i: int, j: int, k: int) -> str:
    """Human-readable Cartesian-component label.

    ``(0,0,0) -> 'S'`` (overlap / monopole)
    ``(1,0,0) -> 'x'``  (dipole-x)
    ``(2,0,0) -> 'xx'`` (quadrupole-xx)
    ``(1,1,1) -> 'xyz'`` (octupole-xyz)
    """
    if i == 0 and j == 0 and k == 0:
        return "S"
    return "x" * i + "y" * j + "z" * k


@dataclass
class CellMultipoleMoments:
    """Cell-level multipole-moment vector around an expansion origin.

    Attributes
    ----------
    L_max : int
        Maximum angular momentum (matches the originating ``LatticeMultipoleSet``).
    n_components : int
        Number of Cartesian components, ``(L_max+1)(L_max+2)(L_max+3)/6``.
    origin : tuple of 3 floats
        Expansion origin in Cartesian bohr (matches the
        ``LatticeMultipoleSet.origin`` used in the construction).
    moments : np.ndarray of shape (n_components,)
        Cartesian moments in libint emultipole component order.
        Without nuclear compensation, ``moments[0]`` is the positive total
        cell electron count.  Passing ``system`` to
        :func:`compute_cell_multipole_moments` subtracts nuclear moments.

    Notes
    -----
    Sign convention: r_cell here is the **electron density**, i.e. a strictly
    positive distribution.  Its zero moment is
    ``sum_g,mn P_mn(g) S_mn(g) = N_electrons``.  With ``system`` supplied,
    the nuclear point moments are subtracted so a neutral cell has zero
    monopole.
    """

    L_max: int
    n_components: int
    origin: tuple
    moments: np.ndarray

    def __getitem__(self, idx_or_label) -> float:
        """``moments[3]`` or ``moments['xyz']`` accessors."""
        if isinstance(idx_or_label, str):
            indices = cartesian_component_indices(self.L_max)
            labels = [cartesian_component_label(i, j, k) for i, j, k in indices]
            try:
                k = labels.index(idx_or_label)
            except ValueError:
                raise KeyError(
                    f"no Cartesian component {idx_or_label!r} at L_max={self.L_max}; "
                    f"available labels: {labels}"
                )
            return float(self.moments[k])
        return float(self.moments[int(idx_or_label)])

    def get_spheropole_trace(self) -> Optional[float]:
        """``Q_xx + Q_yy + Q_zz`` -- the Cartesian "trace" of the
        quadrupole tensor. Required for L_max >= 2; returns ``None``
        otherwise.

        CRYSTAL prints this separately as ``::: EXT EL-SPHEROPOLE``
        because it's invariant under rotation (spherical-tensor s-wave
        radial-second-moment <r^2> weighted by r_cell).
        """
        if self.L_max < 2:
            return None
        return self["xx"] + self["yy"] + self["zz"]


def compute_cell_multipole_moments(
    P_real: LatticeMatrixSet,
    M_lattice: LatticeMultipoleSet,
    *,
    system: Optional["PeriodicSystem"] = None,
) -> CellMultipoleMoments:
    """Cell-level multipole moments via the contraction
    ``M^cell_lm = S_{muν, g} P_muν(g) . M^{muν}_lm(g)``.

    Parameters
    ----------
    P_real : LatticeMatrixSet
        Real-space density matrix per cell. Typically the output of
        ``real_space_density_from_kpoints`` -- one ``(nbf, nbf)`` block
        per lattice cell.
    M_lattice : LatticeMultipoleSet
        Per-cell shell-pair Cartesian multipole moments (Phase 1
        output of ``compute_multipole_moments_lattice``).
    system : PeriodicSystem, optional
        When provided, nuclear point-charge moments are subtracted from the
        positive electron-number moments.  The monopole becomes
        ``N_e - S_I Z_I`` (zero for neutral cells), eliminating the
        spurious self-interaction in multipole far-field J builds.

    Returns
    -------
    CellMultipoleMoments
        Cell-level Cartesian moments around ``M_lattice.origin``.

    Notes
    -----
    The two ``LatticeMatrixSet`` and ``LatticeMultipoleSet`` MUST share
    the same cell list (same ``cells`` attribute, in the same order).
    Phase 1's ``compute_multipole_moments_lattice`` uses
    ``direct_lattice_cells(system, opts.cutoff_bohr)``; the density
    matrix's cell list is set by ``real_space_density_from_kpoints``
    which in turn uses the integrals' cell list. Mismatch will raise.

    The cell-multipole computation is exact in the density:
    if ``P_real`` is the converged SCF density, the returned moments
    are the cell's true Cartesian moments around ``origin``. If
    ``P_real`` is the SAD initial guess, the moments characterise
    the atomic-density superposition.
    """
    if P_real.nbf != M_lattice.nbf:
        raise ValueError(
            f"P_real.nbf={P_real.nbf} differs from M_lattice.nbf={M_lattice.nbf}"
        )
    if len(P_real.cells) < len(M_lattice.cells):
        raise ValueError(
            f"cell-list length mismatch: P_real has {len(P_real.cells)} cells, "
            f"M_lattice has {len(M_lattice.cells)}. The density must cover at "
            f"least the moment cell list."
        )
    # The density may ride the one-electron cell list, which under
    # LatticeSumOptions.pair_complete_1e (#429) is longer than the plain
    # |g| ball the multipole moments deliberately keep (the position
    # operator grows with |g|; that sum is conditionally convergent and the
    # |g| bound does physical work there). direct_lattice_cells stable-sorts
    # by |r|, so the shorter list is an exact prefix of the longer one; the
    # contraction below runs over the moments' length. Verify the shared
    # prefix agrees cell by cell.
    for c_idx, (cp, cm) in enumerate(zip(P_real.cells, M_lattice.cells)):
        if not (np.asarray(cp.index) == np.asarray(cm.index)).all():
            raise ValueError(
                f"cell-list mismatch at index {c_idx}: P_real has "
                f"{tuple(cp.index)}, M_lattice has {tuple(cm.index)}"
            )

    n_components = cartesian_multipole_n_components(M_lattice.L_max)
    moments = np.zeros(n_components, dtype=float)

    n_cells = len(M_lattice.cells)
    for c in range(n_cells):
        P_block = np.asarray(P_real.blocks[c], dtype=float)
        if P_block.size == 0:
            continue
        for comp in range(n_components):
            M_block = np.asarray(M_lattice.blocks[c][comp], dtype=float)
            # Contract: S_muν P_muν(g) . M^muν_comp(g)
            moments[comp] += float(np.einsum("ij,ij->", P_block, M_block))

    # ---- Nuclear charge compensation ----------------------------------
    # The electronic moments above use libint's convention where the
    # density matrix P (positive-semidefinite) contracted with the
    # overlap M_S = S gives +Tr[P.S] ≈ +N_e.  To make the total
    # cell charge zero for a neutral cell, we SUBTRACT the nuclear
    # charge: Q_total = +N_e - S Z_I = 0.
    if system is not None:
        origin = np.array(
            [float(x) for x in M_lattice.origin],
            dtype=float,
        )
        _layouts = _CARTESIAN_LAYOUT_BY_L_MAX[M_lattice.L_max]
        for atom in system.unit_cell:
            Z = float(getattr(atom, "Z", 0))
            if Z == 0.0:
                continue
            r = np.array([float(x) for x in atom.xyz], dtype=float) - origin
            for comp_idx, (i, j, k) in enumerate(_layouts):
                if i == 0 and j == 0 and k == 0:
                    moments[comp_idx] -= Z  # subtract nuclear charge
                else:
                    moments[comp_idx] -= Z * (r[0] ** i) * (r[1] ** j) * (r[2] ** k)

    return CellMultipoleMoments(
        L_max=M_lattice.L_max,
        n_components=n_components,
        origin=tuple(M_lattice.origin),
        moments=moments,
    )
