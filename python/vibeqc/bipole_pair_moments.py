"""Per-shell-pair multipole moments about adjoined-Gaussian pair centers.

BIPOLE-EXACT-ZONE increment 2b. The quartet-level bipolar far field
(Pisani, Dovesi & Roetti, Lecture Notes in Chemistry 48 (1988),
Sec. II.4c, Eq. II.4.7) contracts multipole moments of the two product
distributions taken about *their own* expansion centers -- a global
origin makes the expansion parameter |r_pair - origin| and the series
useless. The periodic source identifies the expansion origins as the
centroids of the two product distributions. Pisani-Dovesi (1980), Sec. 4,
assigns each shell an adjoined diffuse s-Gaussian for overlap screening;
the standard Gaussian product theorem gives the pair centre used here:

    C = (a_mu * A + a_nu * (B + g)) / (a_mu + a_nu)

with ``a`` the smallest primitive exponent of each contracted shell. It is
an approximation to the full contracted product-distribution centroid.

The moments themselves are NOT recomputed per pair: the global-origin
moments (``compute_multipole_moments_lattice``, one libint pass) are
shifted analytically -- the polynomial-shift identity is exact because
the Cartesian moment integrand factorises,

    <(x - Cx)^i ...> = sum_p binom(i, p) (-Cx)^(i-p) <x^p ...>

per axis (the seam the C++ header documents deliberately). Shifts are
implemented through the requested ``L_target`` supported by the input
moment set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

__all__ = [
    "PairMultipoleMoments",
    "adjoined_pair_centers",
    "pair_center_moments",
]

# libint Cartesian component order for emultipole2 (see
# cpp/include/vibeqc/multipole_moments_lattice.hpp): S, x, y, z, xx, xy,
# xz, yy, yz, zz.
_COMP_S = 0
_COMP_D = {0: 1, 1: 2, 2: 3}
_COMP_Q = {(0, 0): 4, (0, 1): 5, (0, 2): 6, (1, 1): 7, (1, 2): 8, (2, 2): 9}
# emultipole3 adds 10 octupole components (i+j+k=3):
# xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz
_COMP_O = {
    (0, 0, 0): 10, (0, 0, 1): 11, (0, 0, 2): 12,
    (0, 1, 1): 13, (0, 1, 2): 14, (0, 2, 2): 15,
    (1, 1, 1): 16, (1, 1, 2): 17, (1, 2, 2): 18, (2, 2, 2): 19,
}
# emultipole4 adds 15 hexadecapole components (i+j+k+l=4) in libint's
# lexicographic order over axis indices (0=x, 1=y, 2=z):
_COMP_H = {
    (0,0,0,0): 20, (0,0,0,1): 21, (0,0,0,2): 22,
    (0,0,1,1): 23, (0,0,1,2): 24, (0,0,2,2): 25,
    (0,1,1,1): 26, (0,1,1,2): 27, (0,1,2,2): 28, (0,2,2,2): 29,
    (1,1,1,1): 30, (1,1,1,2): 31, (1,1,2,2): 32, (1,2,2,2): 33,
    (2,2,2,2): 34,
}


@dataclass
class PairMultipoleMoments:
    """Per-pair-center Cartesian moments on the lattice cell list.

    ``blocks[c][comp]`` is an (nbf, nbf) matrix whose (mu in s1, nu in
    s2) sub-block holds the moment about that pair's own adjoined-Gaussian
    center; ``centers[c]`` is an (n_sh, n_sh, 3) array of those centers.
    ``shell_slices`` maps shell index -> (first bf, n bf).
    """

    nbf: int
    L_max: int
    cells: list
    blocks: List[List[np.ndarray]]
    centers: List[np.ndarray]
    shell_slices: List[tuple]


def _shell_first_bf_and_sizes(basis) -> List[tuple]:
    out = []
    first = 0
    for sh in basis.shells():
        l = int(sh.l)
        # Solid-harmonic (pure) shells span 2l+1 functions, Cartesian
        # shells (l+1)(l+2)/2 -- ShellInfo carries the `pure` flag.
        n = (2 * l + 1) if bool(sh.pure) else (l + 1) * (l + 2) // 2
        out.append((first, n))
        first += n
    return out


def _shell_min_exponents(basis) -> np.ndarray:
    return np.array(
        [min(sh.exponents) for sh in basis.shells()], dtype=float
    )


def _shell_origins(basis) -> np.ndarray:
    return np.array(
        [np.asarray(sh.origin, dtype=float).reshape(3) for sh in basis.shells()],
        dtype=float,
    )


def adjoined_pair_centers(basis, cells) -> List[np.ndarray]:
    """Adjoined-Gaussian product centers per (shell, shell, cell).

    Returns one (n_sh, n_sh, 3) array per lattice cell:
    ``C = (a1*A + a2*(B + g)) / (a1 + a2)`` with ``a`` each shell's most
    diffuse primitive exponent. Pisani-Dovesi (1980), Sec. 4, defines the
    adjoined diffuse s-Gaussian convention; the centre follows from the
    standard Gaussian product theorem. It approximates the product-
    distribution centroid used in Pisani-Dovesi-Roetti (1988), Ch. II.4c.
    """
    a = _shell_min_exponents(basis)
    A = _shell_origins(basis)
    n_sh = len(a)
    denom = a[:, None] + a[None, :]
    w1 = (a[:, None] / denom)[:, :, None]
    w2 = (a[None, :] / denom)[:, :, None]
    out = []
    for cell in cells:
        g = np.asarray(cell.r_cart, dtype=float).reshape(3)
        B_g = A[None, :, :] + g[None, None, :]
        out.append(w1 * A[:, None, :] + w2 * B_g)
    return out


def pair_center_moments(M_lat, basis, *, L_target: int = 2) -> PairMultipoleMoments:
    """Shift global-origin lattice moments to adjoined pair centers.

    ``M_lat`` is a ``compute_multipole_moments_lattice`` result taken
    about origin O.  ``L_target`` controls the output order:
    * ``2`` — overlap + dipole + quadrupole (10 Cartesian components)
    * ``3`` — overlap + dipole + quadrupole + octupole (20 Cartesian)
    * ``4`` — through hexadecapole (35 Cartesian components).
      For L=4, if M_lat.spherical=True (sphemultipole), converts
      spherical→Cartesian via pseudoinverse before shifting.

    The per-axis polynomial shift is the standard binomial translation of
    Cartesian moments. The selected centre approximation is implementation-
    specific.
    """
    if getattr(M_lat, "spherical", False):
        # Review B4 (2026-08-12): raw sphemultipole blocks are SPHERICAL
        # components; treating them as the leading Cartesian components
        # silently mixes bases (measured rel-14 corruption of even the
        # l <= 2 pair moments). Exact Cartesian moments cannot be
        # reconstructed from spherical input alone (the trace content --
        # e.g. the r^2 moment -- is not in any traceless component), so
        # fail closed and point at the sound route.
        raise ValueError(
            "pair_center_moments: input carries SPHERICAL components "
            "(sphemultipole); the Cartesian polynomial shift cannot "
            "consume them directly and exact traces are not "
            "reconstructible from spherical moments. Merge with the "
            "exact emultipole3 Cartesian set first via "
            "vibeqc._sph_to_cart.spherical_to_cartesian_with_traces "
            "(the dormant bipole_far_field_infrastructure prototype path)."
        )
    if int(M_lat.L_max) < L_target:
        raise ValueError(
            f"pair_center_moments: input L_max={M_lat.L_max} < "
            f"target L_target={L_target}"
        )
    nbf = int(M_lat.nbf)
    cells = list(M_lat.cells)
    origin = np.asarray(M_lat.origin, dtype=float).reshape(3)
    slices = _shell_first_bf_and_sizes(basis)
    centers = adjoined_pair_centers(basis, cells)

    n_comp = {2: 10, 3: 20, 4: 35}.get(L_target, 35)
    blocks_out: List[List[np.ndarray]] = []
    for c in range(len(cells)):
        n_avail = len(M_lat.blocks[c])
        # Slice to available components; pad with zeros for missing ones
        # (e.g. when spherical sphemultipole supplies 25 but Cartesian
        # shift expects 35 — the L=4 spherical moments are handled
        # separately by merge_spherical_into_cartesian before calling).
        src = [np.asarray(M_lat.blocks[c][comp], dtype=float)
               for comp in range(min(n_comp, n_avail))]
        while len(src) < n_comp:
            src.append(np.zeros((nbf, nbf), dtype=float))
        dst = [m.copy() for m in src]
        C_cell = centers[c] - origin[None, None, :]
        for i1, (b1, n1) in enumerate(slices):
            for i2, (b2, n2) in enumerate(slices):
                D = C_cell[i1, i2]
                sl = (slice(b1, b1 + n1), slice(b2, b2 + n2))
                S = src[_COMP_S][sl]
                d = [src[_COMP_D[a]][sl] for a in range(3)]
                # Quadrupoles first (they consume unshifted dipoles).
                for (a, b), comp in _COMP_Q.items():
                    dst[comp][sl] = (
                        src[comp][sl]
                        - D[a] * d[b]
                        - D[b] * d[a]
                        + D[a] * D[b] * S
                    )
                for a in range(3):
                    dst[_COMP_D[a]][sl] = d[a] - D[a] * S
                # Octupoles (only if L_target >= 3).
                #
                # All shift formulas below expand the multi-index binomial
                # theorem about the new center,
                #   M'_p = sum_{q <= p} binom(p, q) (-D)^(p-q) M_q,
                # so every lower-order moment on the right-hand side is the
                # UNSHIFTED (src) moment about the original origin. Feeding
                # already-shifted lower moments into the same symmetric slot
                # pattern double-counts the translation (measured: O(50)
                # absolute octupole error at |D| ~ 2.6 bohr; review B4).
                if L_target >= 3:
                    q = [[None for _ in range(3)] for _ in range(3)]
                    for a in range(3):
                        for b in range(a, 3):
                            comp_q = _COMP_Q[(a, b)]
                            q[a][b] = src[comp_q][sl]
                            q[b][a] = q[a][b]
                    for (a, b, c), comp in _COMP_O.items():
                        dst[comp][sl] = (
                            src[comp][sl]
                            - D[a] * q[b][c]
                            - D[b] * q[a][c]
                            - D[c] * q[a][b]
                            + D[a] * D[b] * d[c]
                            + D[a] * D[c] * d[b]
                            + D[b] * D[c] * d[a]
                            - D[a] * D[b] * D[c] * S
                        )
                # Hexadecapoles (only if L_target >= 4).
                if L_target >= 4:
                    o = [[[None for _ in range(3)] for _ in range(3)] for _ in range(3)]
                    for a in range(3):
                        for b in range(a, 3):
                            for c in range(b, 3):
                                comp_o = _COMP_O[(a, b, c)]
                                val = src[comp_o][sl]
                                o[a][b][c] = val
                                o[a][c][b] = val
                                o[b][a][c] = val
                                o[b][c][a] = val
                                o[c][a][b] = val
                                o[c][b][a] = val
                    for (a, b, c, d_i), comp in _COMP_H.items():
                        dst[comp][sl] = src[comp][sl] \
                            - D[a] * o[b][c][d_i] - D[b] * o[a][c][d_i] \
                            - D[c] * o[a][b][d_i] - D[d_i] * o[a][b][c] \
                            + D[a]*D[b]*q[c][d_i] + D[a]*D[c]*q[b][d_i] \
                            + D[a]*D[d_i]*q[b][c] + D[b]*D[c]*q[a][d_i] \
                            + D[b]*D[d_i]*q[a][c] + D[c]*D[d_i]*q[a][b] \
                            - D[a]*D[b]*D[c]*d[d_i] - D[a]*D[b]*D[d_i]*d[c] \
                            - D[a]*D[c]*D[d_i]*d[b] - D[b]*D[c]*D[d_i]*d[a] \
                            + D[a] * D[b] * D[c] * D[d_i] * S
        blocks_out.append(dst)

    return PairMultipoleMoments(
        nbf=nbf,
        L_max=L_target,
        cells=cells,
        blocks=blocks_out,
        centers=centers,
        shell_slices=slices,
    )


def extend_pair_moments_to_L6(pair_mom: PairMultipoleMoments, basis) -> PairMultipoleMoments:
    """Extend pair-centre moments from L=4 to L=6 using isotropic formulas.

    L=5 (21 components): all zero (odd moments vanish for isotropic).
    L=6 (28 components): computed from overlap S and product width γ.

    The formula for even isotropic moments:
        M_{nx,ny,nz} = S · (nx-1)!! · (ny-1)!! · (nz-1)!! / (2γ)^(L/2)
    """
    n_cells = len(pair_mom.blocks)
    nbf = pair_mom.nbf

    # Double factorial helper
    def df(n):
        if n <= 0: return 1
        r = 1
        for k in range(1, n + 1, 2): r *= k
        return r

    # Generate monomials (i,j,k) with i+j+k = L
    def monomials_of_order(L):
        result = []
        for i in range(L, -1, -1):
            for j in range(L - i, -1, -1):
                k = L - i - j
                result.append((i, j, k))
        return result

    # Shell min exponents
    a = _shell_min_exponents(basis)
    slices = pair_mom.shell_slices
    n_sh = len(a)

    n_comp_L4 = 35
    n_comp_L5 = 21  # comps 35..55
    n_comp_L6 = 28  # comps 56..83

    blocks_out = []
    for c in range(n_cells):
        src = [np.asarray(pair_mom.blocks[c][comp], dtype=float) for comp in range(n_comp_L4)]
        # Append zero L=5 components
        for _ in range(n_comp_L5):
            src.append(np.zeros((nbf, nbf), dtype=float))
        # Compute L=6 components
        monos6 = monomials_of_order(6)
        for _ in range(n_comp_L6):
            src.append(np.zeros((nbf, nbf), dtype=float))

        for s1 in range(n_sh):
            b1, n1_s = slices[s1]
            for s2 in range(s1, n_sh):
                b2, n2_s = slices[s2]

                # Product width
                a1, a2 = a[s1], a[s2]
                gamma = a1 * a2 / (a1 + a2) if (a1 + a2) > 0 else 0.0
                if gamma < 1e-30: continue
                inv_2gamma = 0.5 / gamma

                # Overlap sub-block from component 0
                sl = (slice(b1, b1 + n1_s), slice(b2, b2 + n2_s))
                S_sub = src[0][sl]

                # Compute L=6 isotropic moments
                for idx, (nx, ny, nz) in enumerate(monos6):
                    if (nx % 2) or (ny % 2) or (nz % 2):
                        continue  # odd → zero
                    coeff = df(nx - 1) * df(ny - 1) * df(nz - 1) * (inv_2gamma ** 3)
                    if coeff != 0.0:
                        src[n_comp_L4 + n_comp_L5 + idx][sl] += S_sub * coeff

        blocks_out.append(src)

    return PairMultipoleMoments(
        nbf=nbf, L_max=6, cells=pair_mom.cells,
        blocks=blocks_out, centers=pair_mom.centers,
        shell_slices=slices,
    )
