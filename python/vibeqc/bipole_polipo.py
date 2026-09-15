"""POLIPO native multipole engine adapter for the BIPOLE infrastructure.

Provides :func:`compute_moments_via_polipo` as a drop-in replacement
for the libint ``compute_multipole_moments_lattice`` call.  When the
infrastructure builder is invoked with ``multipole_engine="polipo"``,
this module handles the basis-set conversion, native C++ engine call,
and result formatting back to the format expected by downstream code
(:func:`vibeqc.bipole_pair_moments.pair_center_moments`).

Spherical harmonic conversion uses the same real-solid-harmonic
convention as libint's ``sphemultipole`` (Stone/Schmidt-semi-normalised,
sqrt(4pi/(2l+1)) absorbed), implemented via the existing
:func:`vibeqc._cart_to_sph.cartesian_to_spherical_matrix` for L≤4
and extended numerically for L≤12.

Reference
---------
McMurchie & Davidson, J. Comput. Phys. 26, 218 (1978).
"""

from __future__ import annotations

from typing import Any, List
from types import SimpleNamespace

import numpy as np


def _pure_p_shell_permutation_matrix() -> np.ndarray:
    """Permutation that converts POLIPO Cartesian p-ordering (x,y,z)
    to libint's real-solid-harmonic ordering for pure p shells (y,z,x).

    Libint stores pure-shell AOs as m = -l, ..., 0, ..., +l, which for
    l=1 maps to the real combinations (py, pz, px) at indices 0,1,2.

    This ordering statement follows the libint pure-shell convention; it
    is not derived in Saunders et al. (1992).
    """
    # POLIPO cart: [0]=px, [1]=py, [2]=pz
    # libint pure: [0]=py, [1]=pz, [2]=px
    # So: libint[0] = polipo[1], libint[1] = polipo[2], libint[2] = polipo[0]
    P = np.zeros((3, 3), dtype=float)
    P[0, 1] = 1.0  # libint py = polipo py
    P[1, 2] = 1.0  # libint pz = polipo pz
    P[2, 0] = 1.0  # libint px = polipo px
    return P


def _apply_pure_shell_transformation(
    result: Any,
    shells: List[Any],
    L_max: int,
) -> None:
    """Apply Cartesian→spherical AO reordering to pure-shell blocks.

    This is a no-op kept for API compatibility; the actual transformation
    is applied in _apply_pure_shell_transformation_to_blocks on the
    Python-copied arrays.
    """
    pass


def _apply_pure_shell_transformation_to_blocks(
    blocks_out: List[List[np.ndarray]],
    shells: List[Any],
) -> None:
    """Apply Cartesian→spherical AO reordering to pure-shell blocks (in-place).

    Operates on Python numpy arrays (not the C++ Eigen wrappers).
    """
    offsets: List[int] = []
    n_bfs: List[int] = []
    pure_flags: List[bool] = []
    for sh in shells:
        offsets.append(sh.bf_offset)
        n_bfs.append(sh.n_bf)
        pure_flags.append(sh.pure)

    if not any(sh.pure and sh.l == 1 for sh in shells):
        return

    P_p = _pure_p_shell_permutation_matrix()

    for comps in blocks_out:
        for comp in range(len(comps)):
            block = comps[comp]
            for s1 in range(len(shells)):
                for s2 in range(len(shells)):
                    pure1 = pure_flags[s1] and shells[s1].l == 1
                    pure2 = pure_flags[s2] and shells[s2].l == 1
                    if not pure1 and not pure2:
                        continue
                    o1 = offsets[s1]
                    o2 = offsets[s2]
                    n1 = n_bfs[s1]
                    n2 = n_bfs[s2]
                    sub = block[o1:o1 + n1, o2:o2 + n2].copy()
                    if pure1 and pure2:
                        block[o1:o1 + n1, o2:o2 + n2] = P_p @ sub @ P_p.T
                    elif pure1:
                        block[o1:o1 + n1, o2:o2 + n2] = P_p @ sub
                    else:  # pure2
                        block[o1:o1 + n1, o2:o2 + n2] = sub @ P_p.T


def compute_moments_via_polipo(
    basis: Any,
    system: Any,
    lat_opts: Any,
    L_max: int,
    origin: tuple = (0.0, 0.0, 0.0),
    spherical: bool = False,
) -> SimpleNamespace:
    """Compute multipole moments using the POLIPO native engine.

    Parameters
    ----------
    basis : BasisSet
    system : PeriodicSystem
    lat_opts : LatticeSumOptions
    L_max : int
        Maximum multipole order (0-12).
    origin : tuple
        Expansion origin (default: (0, 0, 0)).
    spherical : bool
        If True, convert Cartesian to real spherical harmonics.

    Returns
    -------
    SimpleNamespace with ``nbf``, ``L_max``, ``spherical``, ``cells``,
    ``origin``, ``blocks`` attributes.
    """
    from ._vibeqc_core import (
        PolipoShellInfo,
        PolipoCellInfo,
        PolipoOptions,
        compute_polipo_moments_lattice,
        direct_lattice_cells,
    )

    # Build shell descriptors.
    shells: List[PolipoShellInfo] = []
    bf_offset = 0
    for sh in basis.shells():
        info = PolipoShellInfo()
        info.l = int(sh.l)
        info.pure = bool(getattr(sh, "pure", True))
        info.origin = (
            float(sh.origin[0]),
            float(sh.origin[1]),
            float(sh.origin[2]),
        )
        info.exponents = [float(e) for e in sh.exponents]
        info.coeffs = [float(c) for c in sh.coefficients]
        info.bf_offset = bf_offset
        # Number of basis functions: 2*l+1 for spherical, (l+1)(l+2)/2 for Cartesian
        if info.pure:
            info.n_bf = 2 * int(sh.l) + 1
        else:
            info.n_bf = (int(sh.l) + 1) * (int(sh.l) + 2) // 2
        bf_offset += info.n_bf
        shells.append(info)

    # Build cell descriptors.
    cells_lattice = direct_lattice_cells(system, lat_opts.cutoff_bohr)
    cells: List[PolipoCellInfo] = []
    for cell in cells_lattice:
        cinfo = PolipoCellInfo()
        cinfo.r_cart = (
            float(cell.r_cart[0]),
            float(cell.r_cart[1]),
            float(cell.r_cart[2]),
        )
        cinfo.index = (
            int(cell.index[0]),
            int(cell.index[1]),
            int(cell.index[2]),
        )
        cells.append(cinfo)

    # Set POLIPO options.
    opts = PolipoOptions()
    opts.cutoff_bohr = float(lat_opts.cutoff_bohr)
    opts.L_max = L_max
    opts.spherical = False  # Cartesian for compatibility with pair shift

    # Compute moments.
    result = compute_polipo_moments_lattice(shells, cells, opts)

    # Convert to Python numpy arrays first (C++ Eigen wrappers are
    # read-only through pybind11, so we must copy before mutating).
    n_comp = (
        (result.L_max + 1) * (result.L_max + 2) * (result.L_max + 3) // 6
    )
    blocks_out: List[List[np.ndarray]] = []
    for c in range(len(result.cells)):
        comps: List[np.ndarray] = []
        for comp in range(n_comp):
            if comp < len(result.blocks[c]):
                comps.append(np.asarray(result.blocks[c][comp], dtype=float))  # no copy: permutation in C++
            else:
                comps.append(
                    np.zeros((basis.nbasis, basis.nbasis), dtype=float)
                )
        blocks_out.append(comps)

    # Post-process: pure-shell AO permutation is now handled in C++
    # (compute_polipo_shell_pair_moments applies the l=1 permutation).
    # The Python _apply_pure_shell_transformation_to_blocks is kept
    # for backward compatibility but is no longer called here.

    # Optional spherical harmonic conversion.
    if spherical:
        from ._cart_to_sph import cartesian_to_spherical_matrix
        C = cartesian_to_spherical_matrix(result.L_max)
        n_sph = C.shape[0]
        sph_blocks = []
        for comps in blocks_out:
            sph_comps = []
            for s in range(n_sph):
                M_sph = np.zeros((basis.nbasis, basis.nbasis), dtype=float)
                for cart in range(n_comp):
                    if abs(C[s, cart]) > 1e-15:
                        M_sph += C[s, cart] * comps[cart]
                sph_comps.append(M_sph)
            sph_blocks.append(sph_comps)
        blocks_out = sph_blocks

    return SimpleNamespace(
        nbf=basis.nbasis,
        L_max=result.L_max,
        spherical=spherical,
        cells=list(cells_lattice),
        origin=origin,
        blocks=blocks_out,
    )
