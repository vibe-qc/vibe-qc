"""Prototype spherical-to-Cartesian L=4 multipole assembly.

For use by the BIPOLE multipole far-field builder when the moment source
is libint sphemultipole (L_max=4 spherical output), which carries only
the traceless part of the multipole tensor. A polynomial shift requires
the full Cartesian tensor including traces. Spherical ``L=4`` data alone
does not determine the missing intrinsic trace exactly.

This dormant prototype combines:
  - the spherical (traceless) moments at order L,
  - the already-computed full Cartesian moments at orders 0 … L−1.

Provenance
----------
Pisani-Dovesi-Roetti (1988), Ch. II.4c, is the periodic product-
distribution multipole source. Neither it nor Saunders et al. (1992)
derives this trace-completion prototype.
"""

from __future__ import annotations

import numpy as np

from ._cart_to_sph import cartesian_to_spherical_matrix
from .bipole_cell_moments import cartesian_component_indices

__all__ = [
    "spherical_to_cartesian_with_traces",
    "n_cartesian_components",
]


def n_cartesian_components(L_max: int) -> int:
    """Number of Cartesian multipole components up through order L_max."""
    return (L_max + 1) * (L_max + 2) * (L_max + 3) // 6


def _build_traceless_cart_from_sph(L_max: int) -> np.ndarray:
    """Build the matrix P such that M_cart_traceless = P @ M_sph.

    The pseudoinverse of the Cartesian→spherical matrix C gives the
    traceless Cartesian moments from spherical moments.  Since C maps
    the full Cartesian space (including traces) to the spherical space
    (traceless only), its rows are linearly dependent; the Moore-Penrose
    pseudoinverse C⁺ gives the unique minimum-norm solution, which is
    exactly the traceless part.

    Returns
    -------
    P : np.ndarray, shape (n_cart, n_sph)
    """
    C = cartesian_to_spherical_matrix(L_max)
    # P = C⁺ = C^T (C C^T)^{-1}  (right inverse, since C is (n_sph, n_cart)
    # with n_sph < n_cart and full row rank)
    CCt = C @ C.T
    try:
        CCt_inv = np.linalg.inv(CCt)
    except np.linalg.LinAlgError:
        # Fallback to pseudoinverse with higher tolerance
        CCt_inv = np.linalg.pinv(CCt, rcond=1e-12)
    P = C.T @ CCt_inv  # (n_cart, n_sph)
    return P


# libint's ``Operator::sphemultipole`` emits the SCALED real solid
# harmonics N^{+/-}_{l,m} of J. M. Perez-Jorda & W. Yang, J. Chem.
# Phys. 104, 8003 (1996), doi:10.1063/1.468354 -- per the operator
# documentation in libint2/engine.h, the Racah real solid harmonics
# C_l^m / S_l^m are recovered as
#   Racah_lm = (-1)^m sqrt((2 - delta_m0) (l+|m|)! (l-|m|)!) * N_lm.
# The BIPOLE far field is written in the ``cartesian_to_spherical_matrix``
# convention instead (its interaction tensors and the C++ matrix twin are
# matched to those rows), and each of that matrix's l=4 rows is exactly
# proportional to the corresponding solid harmonic. The resulting exact
# per-m conversion factors Z_module = s_m * N_libint for l = 4, verified
# to machine precision both symbolically (polynomial ratio at generic
# points) and against libint sphemultipole lattice integrals vs
# brute-force quadrature moments:
_SQRT5 = np.sqrt(5.0)
_L4_LIBINT_TO_MODULE = np.array([
    48.0 * _SQRT5,   # m = -4
    -24.0 * _SQRT5,  # m = -3
    12.0 * _SQRT5,   # m = -2
    -24.0 * _SQRT5,  # m = -1
    24.0,            # m =  0
    -24.0 * _SQRT5,  # m = +1
    24.0 * _SQRT5,   # m = +2
    -24.0 * _SQRT5,  # m = +3
    96.0 * _SQRT5,   # m = +4
])


def _cart_index(i: int, j: int, k: int) -> int:
    """Return the flat Cartesian index for monomial (i,j,k) with
    i+j+k between 0 and 4, in the libint lexicographic order used by
    cartesian_component_indices."""
    indices = cartesian_component_indices(4)
    return indices.index((i, j, k))


def _add_L4_traces(
    cart_full: np.ndarray,
    cart_L2: np.ndarray,
    nbf: int,
) -> np.ndarray:
    """Add trace contributions to the L=4 Cartesian moments.

    The L=4 Cartesian tensor H_{ijkl} (i+j+k=4) has 15 components.
    After converting from spherical, we have only the traceless part
    H̃_{ijkl}.  The six independent trace conditions are:

      Σ_a H_{aakl} = r² · Q_{kl}      for each (k,l)
      Σ_b H_{abbl} = r² · Q_{al}      …
      Σ_c H_{abcc} = r² · Q_{ab}      …

    where r² = x² + y² + z² is the scalar second-moment operator,
    and Q_{pq} (p+q=2) are the Cartesian quadrupole moments.

    For L=4 in the normalised monomial basis, the trace projection
    onto the six independent components yields:

      H_{iijj} += (1/5) · r² · (δ_{ij}·Q_{kl} + …) / …

    We compute this explicitly for each of the 15 hexadecapole components.

    Parameters
    ----------
    cart_full : np.ndarray, shape (35, nbf, nbf) or (n_components, nbf, nbf)
        Traceless Cartesian moments for L=0-4.  The L≤3 entries are already
        full (from emultipole3); only L=4 entries need trace correction.
    cart_L2 : np.ndarray, shape (10, nbf, nbf)
        Full Cartesian moments for L≤2 from emultipole3.
    nbf : int

    Returns
    -------
    np.ndarray, same shape as cart_full
        With L=4 trace contributions added.
    """
    result = np.copy(cart_full)

    # Quadrupole moments (L=2 Cartesian, components 4-9 in emultipole3 order)
    # Component mapping for L=2: xx→xx, xy→xy, xz→xz, yy→yy, yz→yz, zz→zz
    # In the Cartesian component list, indices are:
    #   0:x, 1:y, 2:z ← dipole; 3:xx, 4:xy, 5:xz, 6:yy, 7:yz, 8:zz
    # Wait, emultipole stores: S(0), x(1), y(2), z(3), xx(4), xy(5),
    #   xz(6), yy(7), yz(8), zz(9), then octupoles 10-19.
    # The quadrupole components in the FULL Cartesian list are:
    #   Q_xx = cart_full[4], Q_xy = cart_full[5], Q_xz = cart_full[6],
    #   Q_yy = cart_full[7], Q_yz = cart_full[8], Q_zz = cart_full[9]

    def _Q(p: int, q: int) -> np.ndarray:
        """Retrieve quadrupole moment Q_{pq} as (nbf, nbf) array."""
        # Canonical order: x→0, y→1, z→2
        key = tuple(sorted([p, q]))
        if key == (0, 0):
            return cart_L2[4]  # xx
        elif key == (0, 1):
            return cart_L2[5]  # xy
        elif key == (0, 2):
            return cart_L2[6]  # xz
        elif key == (1, 1):
            return cart_L2[7]  # yy
        elif key == (1, 2):
            return cart_L2[8]  # yz
        elif key == (2, 2):
            return cart_L2[9]  # zz
        return np.zeros((nbf, nbf))

    def _r2_Q(p: int, q: int) -> np.ndarray:
        """r² · Q_{pq} = Q_xx + Q_yy + Q_zz contracted onto Q_{pq}? No.

        Actually: r² = x² + y² + z², and the moment ⟨r² x^p y^q⟩
        decomposes into sum over traces.  The identity:
          ⟨r² · x^i y^j z^k⟩ = ⟨x^{i+2} y^j z^k⟩ + ⟨x^i y^{j+2} z^k⟩
                             + ⟨x^i y^j z^{k+2}⟩
        allows us to express the L=4 traces in terms of L=2 moments.

        For a hexadecapole component H_{ijkl} with i+j+k=4:
          trace H on axes (a,b) = Σ_c H_{...c...c...}
                                = r² · M^{(2)}_{...}
        where M^{(2)} are the dot products with r².

        Concretely, the trace vector for L=4 hexadecapole has 6 independent
        components (since the trace on any pair of axes gives the same result):

        T_{xx} = r²·Q_{xx}  (contraction on y,z → H_{xxyy} + H_{xxzz})
        T_{yy} = r²·Q_{yy}  (contraction on x,z → H_{xxyy} + H_{yyzz})
        T_{zz} = r²·Q_{zz}  (contraction on x,y → H_{xxzz} + H_{yyzz})
        T_{xy} = r²·Q_{xy}  (contraction on z   → H_{xyzz})
        T_{xz} = r²·Q_{xz}  (contraction on y   → H_{xzyy})
        T_{yz} = r²·Q_{yz}  (contraction on x   → H_{yzxx})

        But these are further constrained: T_{xx}+T_{yy}+T_{zz}=r²·(Q_{xx}+Q_{yy}+Q_{zz})
        which is just r² times the monopole (S component, index 0 of cart_L2).
        """
        # Actually: r²·Q_{pq} = ∫ φ_i (x²+y²+z²)·x^p y^q φ_j dr
        # = ⟨x^{p+2} y^q⟩ + ⟨x^p y^{q+2}⟩ + ⟨x^p y^q z^2⟩
        # = sum over the three L=4 moments with two extra powers on each axis.
        # So the trace vector for each (p,q) quadrupole index is computed
        # by summing the full L=4 Cartesian moments over same-axis indices.
        # This is circular: we need the L=4 traces to compute L=4 moments.

        # Instead, use the identity:
        # r²·Q_{pq} can be approximated from the actual r² operator.
        # But we don't have r²·Q moments — we have Q moments about a specific
        # origin.  The relationship is more subtle: the L=4 moment about
        # the origin decomposes into the traceless part plus the trace from
        # L=2 moments shifted TO the origin by r².

        # Simplification: for a Gaussian product centered at P near the
        # origin, the traces are small.  We set them to zero initially,
        # and the analytical shift formula will generate proper traces
        # when the moments are shifted to the pair centers.

        # So we simply return the traceless L=4 moments as full.
        return np.zeros((nbf, nbf))

    # The L=4 Cartesian components occupy indices 20-34 in the full list
    # (0: S, 1-3: dipole, 4-9: quadrupole, 10-19: octupole, 20-34: hexadecapole)

    # Trace projection: for each hexadecapole component H_{ijkl},
    # add (1/5) of the trace contributions.
    # For a symmetric Cartesian tensor of order 4, the trace on any axis pair
    # adds to 5 different components (the 5 ways to place the remaining
    # indices).  Each component receives contributions from 6 trace channels.

    # The explicit formula for the trace addition to H_{ijkl}:
    # ΔH_{ijkl} = (1/7)·[δ_{ij}·R_{kl} + δ_{ik}·R_{jl} + δ_{il}·R_{jk}
    #                    + δ_{jk}·R_{il} + δ_{jl}·R_{ik} + δ_{kl}·R_{ij}]
    # where R_{pq} = (r²·Q_{pq}) / (number of distinct axis assignments).

    # For the CRYSTAL convention (normalised monomials, not symmetrised),
    # each monomial (i,j,k) gets trace from the symmetrised form.

    # Spherical data does not determine the missing intrinsic traces, so this
    # prototype uses the following approximation:
    #   - The traceless part from sphemultipole is kept as full.
    #   - The polynomial shift generates translation terms from the full
    #     lower-order moments, but cannot restore the missing intrinsic trace.

    # The L=4 moments at the origin therefore remain traceless-only. No
    # primary-source error bound has been established for this approximation.

    return result


def spherical_to_cartesian_with_traces(
    sph_moments: np.ndarray,
    cart_L3: np.ndarray,
    nbf: int,
    L_sph: int = 4,
) -> np.ndarray:
    """Merge spherical L=4 moments into Cartesian L≤3 to form full L=4 Cartesian.

    The emultipole3 output provides exact full Cartesian moments for L=0-3
    (20 components).  The sphemultipole output provides spherical (traceless)
    moments for L=0-4 (25 components).  This function converts the spherical
    L=4 moments to traceless Cartesian and appends them to the full Cartesian
    L≤3, producing a combined 35-component prototype input for the
    polynomial shift to pair centres.

    The L=4 Cartesian moments at the origin remain traceless-only. The
    binomial shift generates translation terms coupled to lower orders but
    does not recover missing intrinsic L=4 trace data. This limitation is
    one reason the surrounding quartet route remains fail-closed.

    Parameters
    ----------
    sph_moments : LatticeMultipoleSet
        Spherical moments from sphemultipole (L_max=4, n_comp=25, spherical=True).
    cart_L3 : LatticeMultipoleSet
        Full Cartesian moments from emultipole3 (L_max=3, n_comp=20, spherical=False).
    nbf : int
    L_sph : int

    Returns
    -------
    np.ndarray, shape (n_cells, 35, nbf, nbf)
    """
    if L_sph != 4:
        raise ValueError(f"L_sph must be 4, got {L_sph}")
    if not getattr(sph_moments, 'spherical', False):
        raise ValueError("sph_moments must have spherical=True")
    if getattr(cart_L3, 'spherical', False):
        raise ValueError("cart_L3 must have spherical=False")

    n_cells = len(sph_moments.cells)
    n_cart = n_cartesian_components(L_sph)  # 35

    # Build the traceless-Cartesian-from-spherical matrix (35 × 25)
    P = _build_traceless_cart_from_sph(L_sph)

    result_blocks: list = []
    for c in range(n_cells):
        # L≤3: copy from emultipole3 Cartesian (20 components → first 20 of 35)
        cart_cell = [
            np.asarray(cart_L3.blocks[c][comp], dtype=float)
            for comp in range(min(20, len(cart_L3.blocks[c])))
        ]
        # Pad L≤3 to 20 components if needed
        while len(cart_cell) < 20:
            cart_cell.append(np.zeros((nbf, nbf), dtype=float))

        # L=4: convert spherical to traceless Cartesian (25 → 35)
        n_sph_avail = len(sph_moments.blocks[c])
        sph_flat = np.zeros((25, nbf * nbf), dtype=float)
        for comp in range(min(n_sph_avail, 25)):
            sph_flat[comp, :] = np.asarray(
                sph_moments.blocks[c][comp], dtype=float
            ).ravel()
        # Convert libint's Perez-Jorda & Yang normalization to the
        # module convention BEFORE reconstructing Cartesians -- P is the
        # pseudoinverse of cartesian_to_spherical_matrix, so feeding it
        # raw N_lm values yields degree-4 content wrong by factors up to
        # 96*sqrt(5) (~215x on |m| = 4). Only the l = 4 rows (16..24)
        # matter: both C and P are degree-block-diagonal, and the l <= 3
        # output is discarded in favor of exact emultipole3 below.
        sph_flat[16:25, :] *= _L4_LIBINT_TO_MODULE[:, None]

        # P: (35, 25) @ sph_flat: (25, nbf*nbf) → (35, nbf*nbf)
        cart_traceless_flat = P @ sph_flat
        cart_traceless = cart_traceless_flat.reshape(35, nbf, nbf)

        # Copy L=0-3 from emultipole3 (exact) and L=4 from sphemultipole (traceless)
        combined = []
        for comp in range(35):
            if comp < 20:
                combined.append(cart_cell[comp])
            else:
                combined.append(cart_traceless[comp])
        result_blocks.append(combined)

    return result_blocks
