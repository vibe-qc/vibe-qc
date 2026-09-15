"""Shell-pair spherical multipole moment buffer for the bipolar far field.

BIPOLE-EXACT-ZONE increment 3a.  Pre-computes per-shell-pair spherical
multipole moments at per-product-distribution centres, caches them across
SCF iterations, and serves them to the quartet far-field Fock contractor.

Architecture
------------
1. Compute global-origin Cartesian moments via
   ``compute_multipole_moments_lattice`` (C++, one libint pass per geometry).
2. Shift to adjoined-Gaussian pair centres via ``pair_center_moments``
   (analytic polynomial shift, exact).
3. Convert to spherical Stone-convention moments via
   ``cartesian_to_spherical_matrix`` pseudoinverse.
4. Store per-(shell1, shell2, cell_idx) as a compact buffer with
   shape ``(n1, n2, n_spherical_components)`` for each entry.
5. Serve on demand to the quartet far-field contractor.

The buffer is an implementation cache built once per geometry and reused
across SCF iterations. Saunders (1992) does not derive this cache design.

Reference
---------
Pisani, Dovesi, and Roetti (1988), Ch. II.4c,
doi:10.1007/978-3-642-93385-1 - product-distribution moments in the
periodic quartet expansion. The buffer layout itself is a prototype.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .bipole_multipole import n_components as _n_sph_components
from .bipole_pair_moments import PairMultipoleMoments, pair_center_moments
from ._cart_to_sph import cartesian_to_spherical_matrix as _cart_to_sph_mat


# ---------------------------------------------------------------------------
# Cartesian component index → (i, j, k) power tuple maps.
# Used for direct polynomial differentiation of shifted moments.
# ---------------------------------------------------------------------------

# L=2 Cartesian component (i,j,k) powers, libint order.
_CART_IDX_2 = [
    (0, 0, 0),  # 0: S
    (1, 0, 0),  # 1: x
    (0, 1, 0),  # 2: y
    (0, 0, 1),  # 3: z
    (2, 0, 0),  # 4: xx
    (1, 1, 0),  # 5: xy
    (1, 0, 1),  # 6: xz
    (0, 2, 0),  # 7: yy
    (0, 1, 1),  # 8: yz
    (0, 0, 2),  # 9: zz
]

# L=3 adds:
_CART_IDX_3 = _CART_IDX_2 + [
    (3, 0, 0),  # 10: xxx
    (2, 1, 0),  # 11: xxy
    (2, 0, 1),  # 12: xxz
    (1, 2, 0),  # 13: xyy
    (1, 1, 1),  # 14: xyz
    (1, 0, 2),  # 15: xzz
    (0, 3, 0),  # 16: yyy
    (0, 2, 1),  # 17: yyz
    (0, 1, 2),  # 18: yzz
    (0, 0, 3),  # 19: zzz
]

# L=4 adds (i+j+k=4):
_CART_IDX_4 = _CART_IDX_3 + [
    (4, 0, 0),  # 20: xxxx
    (3, 1, 0),  # 21: xxxy
    (3, 0, 1),  # 22: xxxz
    (2, 2, 0),  # 23: xxyy
    (2, 1, 1),  # 24: xxyz
    (2, 0, 2),  # 25: xxzz
    (1, 3, 0),  # 26: xyyy
    (1, 2, 1),  # 27: xyyz
    (1, 1, 2),  # 28: xyzz
    (1, 0, 3),  # 29: xzzz
    (0, 4, 0),  # 30: yyyy
    (0, 3, 1),  # 31: yyyz
    (0, 2, 2),  # 32: yyzz
    (0, 1, 3),  # 33: yzzz
    (0, 0, 4),  # 34: zzzz
]

_CART_IDX = {2: _CART_IDX_2, 3: _CART_IDX_3, 4: _CART_IDX_4}

# Inverse: (i,j,k) → Cartesian component index.
def _cart_tuple_to_idx(cart_L: int) -> Dict[Tuple[int, int, int], int]:
    """Map (i,j,k) power tuple → Cartesian component index for given L."""
    out = {}
    for idx, tup in enumerate(_CART_IDX[cart_L]):
        out[tup] = idx
    return out

__all__ = [
    "SphericalMomentBuffer",
    "build_spherical_moment_buffer",
    "spherical_pair_moments",
]


@dataclass
class SphericalMomentBuffer:
    """Pre-computed per-shell-pair spherical multipole moments.

    ``blocks[(s1, s2, cell_idx)]`` is an array of shape
    ``(n1, n2, n_sph)`` where ``n1, n2`` are the AO dimensions of
    shells ``s1`` and ``s2``, and ``n_sph = (L_max+1)^2`` is the
    number of real spherical harmonic components.

    ``centers[(s1, s2, cell_idx)]`` is the (3,) Cartesian product-
    distribution centre for this pair.

    ``cells`` is the lattice cell list (shared by all entries).
    """

    L_max: int
    n_sph: int
    cells: list
    shell_slices: List[Tuple[int, int]]  # (first_bf, n_bf) per shell
    blocks: Dict[Tuple[int, int, int], np.ndarray] = field(
        default_factory=dict
    )
    centers: Dict[Tuple[int, int, int], np.ndarray] = field(
        default_factory=dict
    )
    moment_grads: Dict[Tuple[int, int, int], np.ndarray] = field(
        default_factory=dict
    )  # (3, n1, n2, n_sph_deriv) per pair, empty if L_max <= 0

    def __len__(self) -> int:
        return len(self.blocks)

    def get_moments(
        self, s1: int, s2: int, cell_idx: int
    ) -> Optional[np.ndarray]:
        """Return (n1, n2, n_sph) spherical moments or None."""
        return self.blocks.get((s1, s2, cell_idx))

    def get_center(
        self, s1: int, s2: int, cell_idx: int
    ) -> Optional[np.ndarray]:
        """Return (3,) product-distribution centre or None."""
        return self.centers.get((s1, s2, cell_idx))

    def get_moment_gradient(
        self, s1: int, s2: int, cell_idx: int
    ) -> Optional[np.ndarray]:
        """Return (3, n1, n2, n_sph_deriv) derivative moments dM/dC or None."""
        return self.moment_grads.get((s1, s2, cell_idx))


def build_spherical_moment_buffer(
    pair_moments: PairMultipoleMoments,
    basis,
    *,
    L_max: int = 2,
) -> SphericalMomentBuffer:
    """Convert Cartesian per-pair moments to spherical and buffer them.

    Parameters
    ----------
    pair_moments : PairMultipoleMoments
        Cartesian moments from ``pair_center_moments``.  May have
        ``L_max`` 2 (10 components) or 3 (20 components).
    basis : BasisSet
    L_max : int
        Target spherical multipole order (0-4).  The input ``pair_moments``
        provides Cartesian components up to its own ``L_max``; if
        ``L_max > pair_moments.L_max``, higher spherical components
        are zero-filled.

    Returns
    -------
    SphericalMomentBuffer
    """
    if L_max > 6:
        raise ValueError(f"L_max <= 6 supported; got {L_max}")
    n_sph = _n_sph_components(L_max)
    # Conversion matrix for the Cartesian input order.
    # For L_max > 4, the Cartesian input only goes to L=4, so higher
    # spherical components are zero-filled (odd L=5 are identically zero
    # for the prototype isotropic-Gaussian model; L=6 requires the
    # extend_lattice_moments C++ function for isotropic moments).
    cart_L = min(int(pair_moments.L_max), 4)
    C_mat = _cart_to_sph_mat(cart_L)  # (n_sph_cart, n_cart)
    n_cart = C_mat.shape[1]  # 10 for L=2, 20 for L=3

    slices = pair_moments.shell_slices
    n_sh = len(slices)
    cells = list(pair_moments.cells)

    buffer = SphericalMomentBuffer(
        L_max=L_max,
        n_sph=n_sph,
        cells=cells,
        shell_slices=slices,
    )

    # ---- Try C++ OpenMP fast path first --------------------------------
    try:
        from ._vibeqc_core import (
            convert_cartesian_to_spherical_buffer,
            multipole_cartesian_to_spherical_matrix,
        )
        # Build C++ conversion matrix.
        C_cpp = np.asarray(multipole_cartesian_to_spherical_matrix(cart_L), dtype=float)
        # Build flat pair-centres array: list of (n_cells) of flat (n_sh*n_sh, 3).
        pair_centres_flat = []
        for cell_idx in range(len(cells)):
            ccell = pair_moments.centers[cell_idx]
            # ccell is (n_sh, n_sh, 3); flatten to (n_sh*n_sh, 3) as list of tuples.
            flat = []
            for s1 in range(n_sh):
                for s2 in range(n_sh):
                    flat.append((float(ccell[s1,s2,0]), float(ccell[s1,s2,1]), float(ccell[s1,s2,2])))
            pair_centres_flat.append(flat)
        # Convert Cartesian blocks to list-of-lists format.
        cart_blocks = []
        for cell_idx in range(len(cells)):
            comp_list = []
            for comp in range(n_cart):
                comp_list.append(np.asarray(pair_moments.blocks[cell_idx][comp], dtype=float))
            cart_blocks.append(comp_list)
        # Convert shell_slices to list of (int, int) pairs.
        shell_slices_cpp = [(int(s[0]), int(s[1])) for s in slices]
        # Call C++ converter.
        cpp_buf = convert_cartesian_to_spherical_buffer(
            cart_blocks, C_cpp, shell_slices_cpp, cells, pair_centres_flat,
        )
        # Convert C++ result back to Python SphericalMomentBuffer.
        for entry in cpp_buf.entries:
            s1 = int(entry.shell_1)
            s2 = int(entry.shell_2)
            ci = int(entry.cell_index)
            n1 = int(entry.bf_count_1)
            n2 = int(entry.bf_count_2)
            # Reshape flat moments: (n1*n2, n_sph) → (n1, n2, n_sph)
            sph_mat = np.asarray(entry.moments_flat, dtype=float).reshape(n1, n2, n_sph)
            buffer.blocks[(s1, s2, ci)] = sph_mat
            buffer.centers[(s1, s2, ci)] = np.array(
                [entry.centre_x, entry.centre_y, entry.centre_z], dtype=float)
            # Store transpose.
            if s1 != s2:
                buffer.blocks[(s2, s1, ci)] = np.transpose(sph_mat, (1, 0, 2))
                buffer.centers[(s2, s1, ci)] = buffer.centers[(s1, s2, ci)]
        # Compute moment derivatives for gradient module.
        if L_max >= 1:
            _build_moment_derivatives_for_buffer(
                buffer, pair_moments, C_mat, n_cart,
            )
        return buffer
    except (ImportError, TypeError, ValueError, RuntimeError):
        pass  # Fall back to Python loop below.

    # ---- Python fallback loop -----------------------------------------

    for cell_idx in range(len(cells)):
        centers_cell = pair_moments.centers[cell_idx]
        for s1 in range(n_sh):
            b1, n1 = slices[s1]
            for s2 in range(s1, n_sh):  # upper triangle + diagonal
                b2, n2 = slices[s2]
                # Cartesian moments for this shell pair: shape (n1, n2, n_cart)
                cart = np.stack(
                    [
                        np.asarray(
                            pair_moments.blocks[cell_idx][comp][
                                b1 : b1 + n1, b2 : b2 + n2
                            ],
                            dtype=float,
                        )
                        for comp in range(n_cart)
                    ],
                    axis=-1,
                )  # (n1, n2, n_cart)
                # Convert to spherical: M_sph[i,j,:] = C @ cart[i,j,:]
                # Vectorised: sph = cart @ C_mat^T
                # When L_max < cart_L, only keep the first n_sph rows of C_mat.
                sph = np.zeros((n1, n2, n_sph), dtype=float)
                n_sph_cart = min(C_mat.shape[0], n_sph)
                C_sub = C_mat[:n_sph, :]
                sph_result = np.tensordot(
                    cart, C_sub.T, axes=([2], [0])
                )
                sph[:, :, :sph_result.shape[2]] = sph_result
                buffer.blocks[(s1, s2, cell_idx)] = sph
                buffer.centers[(s1, s2, cell_idx)] = np.asarray(
                    centers_cell[s1, s2], dtype=float
                )

                # Also store the transpose (s2, s1) for efficient look-up
                # when the bra-pair shell order is reversed.
                if s1 != s2:
                    # Transpose the AO indices: (n2, n1, n_sph)
                    sph_t = np.transpose(sph, (1, 0, 2))
                    buffer.blocks[(s2, s1, cell_idx)] = sph_t
                    buffer.centers[(s2, s1, cell_idx)] = np.asarray(
                        centers_cell[s2, s1], dtype=float
                    )

    # Compute moment derivatives for gradient module.
    # Try C++ OpenMP path first, fall back to Python.
    if L_max >= 1 and n_cart > 0:
        _build_moment_derivatives_for_buffer_cpp(
            buffer, pair_moments, C_mat, n_cart,
        )

    return buffer


# ---------------------------------------------------------------------------
# Moment-derivative computation for the prototype far-field gradient.
# ---------------------------------------------------------------------------


def _build_moment_derivatives_for_buffer_cpp(
    buffer: SphericalMomentBuffer,
    pair_moments: PairMultipoleMoments,
    C_mat: np.ndarray,
    n_cart: int,
) -> None:
    """Try C++ OpenMP moment derivative path; fall back to Python."""
    try:
        from ._vibeqc_core import (
            compute_moment_derivatives_for_buffer,
            multipole_cartesian_to_spherical_matrix,
        )
        # Build C++ conversion matrix.
        _cart_max_L = {10: 2, 20: 3, 35: 4}
        max_cart_L = _cart_max_L.get(n_cart, 2)
        C_cpp = np.asarray(
            multipole_cartesian_to_spherical_matrix(max_cart_L),
            dtype=float,
        )
        # Build cartesian_blocks input.
        cart_blocks = []
        for cell_idx in range(len(pair_moments.cells)):
            comp_list = []
            for comp in range(n_cart):
                comp_list.append(
                    np.asarray(pair_moments.blocks[cell_idx][comp],
                              dtype=float))
            cart_blocks.append(comp_list)
        shell_slices_cpp = [
            (int(s[0]), int(s[1])) for s in pair_moments.shell_slices
        ]
        cells = list(pair_moments.cells)

        cpp_buf = compute_moment_derivatives_for_buffer(
            cart_blocks, C_cpp, shell_slices_cpp, cells, n_cart,
        )
        for entry in cpp_buf.entries:
            s1 = int(entry.shell_1)
            s2 = int(entry.shell_2)
            ci = int(entry.cell_index)
            n1 = int(entry.bf_count_1)
            n2 = int(entry.bf_count_2)
            n_sph_deriv = cpp_buf.n_sph_deriv
            # Reshape: flat array → (3, n1, n2, n_sph_deriv).
            flat = np.asarray(entry.gradient_flat, dtype=float)
            deriv_sph = np.zeros((3, n1, n2, n_sph_deriv), dtype=float)
            n_flat = n1 * n2 * n_sph_deriv
            for a in range(3):
                deriv_sph[a] = flat[a * n_flat : (a + 1) * n_flat].reshape(
                    n1, n2, n_sph_deriv)
            buffer.moment_grads[(s1, s2, ci)] = deriv_sph
        return  # Success: C++ path completed.
    except (ImportError, TypeError, ValueError, RuntimeError):
        pass  # Fall back to Python.

    _build_moment_derivatives_for_buffer(
        buffer, pair_moments, C_mat, n_cart,
    )


def _build_moment_derivatives_for_buffer(
    buffer: SphericalMomentBuffer,
    pair_moments: PairMultipoleMoments,
    C_mat: np.ndarray,
    n_cart: int,
) -> None:
    """Compute and store dM_sph/dC for each shell pair in the buffer.

    The derivative of the Cartesian moment M^{(i,j,k)}(C) w.r.t.
    the pair centre C_a (a = x, y, z) is:

        dM^{(i,j,k)}/dC_a = -i * M^{(i-1,j,k)}(C)   (if power_a > 0, else 0)

    This follows by differentiating the standard binomial moment-shift
    formula. The Cartesian
    derivative is then converted to spherical via the same
    Cartesian→spherical matrix.

    The spherical derivative moments ``dM_sph/dC_a`` have one lower
    maximum order (L-1), so they have ``L^2`` components vs the
    parent moments' ``(L+1)^2``.

    Parameters
    ----------
    buffer : SphericalMomentBuffer
        Modified in-place: ``buffer.moment_grads`` is populated.
    pair_moments : PairMultipoleMoments
        Source Cartesian moments at pair centres.
    C_mat : np.ndarray shape (n_sph_cart, n_cart)
        Cartesian→spherical conversion matrix.
    n_cart : int
        Number of Cartesian components in ``pair_moments``.
    """
    # Map (i,j,k) → Cartesian index.
    _cart_max_L = {10: 2, 20: 3, 35: 4}
    max_cart_L = _cart_max_L.get(n_cart, 2)
    cart_tuples = _CART_IDX[max_cart_L]
    idx_map = _cart_tuple_to_idx(max_cart_L)

    # Number of spherical derivative components: L^2 (one order less).
    n_sph_deriv = max_cart_L * max_cart_L

    slices = pair_moments.shell_slices
    n_sh = len(slices)
    cells = list(pair_moments.cells)

    for cell_idx in range(len(cells)):
        for s1 in range(n_sh):
            b1, n1 = slices[s1]
            for s2 in range(n_sh):  # all pairs, not just upper triangle
                b2, n2 = slices[s2]
                key = (s1, s2, cell_idx)
                if key not in buffer.blocks:
                    continue

                # Get Cartesian moments at pair centre.
                cart_c = np.zeros((n1, n2, n_cart), dtype=float)
                for comp in range(n_cart):
                    if comp < len(pair_moments.blocks[cell_idx]):
                        cart_c[:, :, comp] = np.asarray(
                            pair_moments.blocks[cell_idx][comp][
                                b1:b1+n1, b2:b2+n2
                            ],
                            dtype=float,
                        )

                # Build derivative Cartesian moments for each axis.
                # dM/dC_a has the same Cartesian component layout but
                # with values from one order lower.
                deriv_cart = np.zeros((3, n1, n2, n_cart), dtype=float)
                for idx, (i, j, k) in enumerate(cart_tuples):
                    if i > 0:
                        lk = (i - 1, j, k)
                        if lk in idx_map:
                            deriv_cart[0, :, :, idx] = -float(i) * cart_c[:, :, idx_map[lk]]
                    if j > 0:
                        lk = (i, j - 1, k)
                        if lk in idx_map:
                            deriv_cart[1, :, :, idx] = -float(j) * cart_c[:, :, idx_map[lk]]
                    if k > 0:
                        lk = (i, j, k - 1)
                        if lk in idx_map:
                            deriv_cart[2, :, :, idx] = -float(k) * cart_c[:, :, idx_map[lk]]

                # Convert to spherical: only first n_sph_deriv components
                # are meaningful (highest-order derivatives vanish).
                deriv_sph = np.zeros((3, n1, n2, n_sph_deriv), dtype=float)
                C_deriv = C_mat[:n_sph_deriv, :]
                for a in range(3):
                    dc_flat = deriv_cart[a].reshape(n1 * n2, n_cart)
                    ds_flat = dc_flat @ C_deriv.T
                    deriv_sph[a] = ds_flat.reshape(n1, n2, n_sph_deriv)

                buffer.moment_grads[key] = deriv_sph


def spherical_pair_moments(
    buffer: SphericalMomentBuffer,
    s1: int,
    s2: int,
    cell_idx: int,
    *,
    L_truncate: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Retrieve spherical moments for one shell pair, optionally truncated.

    Parameters
    ----------
    buffer : SphericalMomentBuffer
    s1, s2 : int
        Shell indices.
    cell_idx : int
        Index into ``buffer.cells``.
    L_truncate : int or None
        If given, return only the first ``(L_truncate+1)^2`` spherical
        components.  If None, return all ``(L_max+1)^2``.

    Returns
    -------
    ndarray shape (n1, n2, n_trunc) or None
    """
    mom = buffer.get_moments(s1, s2, cell_idx)
    if mom is None:
        return None
    if L_truncate is not None:
        n_keep = _n_sph_components(L_truncate)
        return mom[:, :, :n_keep]
    return mom
