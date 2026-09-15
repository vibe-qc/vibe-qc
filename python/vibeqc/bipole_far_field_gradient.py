"""Derivative prototype for the quartet-level bipolar far-field energy.

Direct differentiation of the implemented prototype has two
contributions:

1. **dT/dR** — gradient of the interaction tensor with respect to
   the separation vector R = C_ket - C_bra.  Dominant term,
   O(1/R^{L+1}).  Implemented in C++ OpenMP
   (``compute_bipolar_far_field_gradient_cpp``) with per-thread
   tensor caching inside the dormant prototype.

2. **dM/dA** — gradient of the shifted spherical moments with
   respect to atomic positions, via the chain rule through the
   pair centres.  Sub-dominant term, O(1/R^{L+2}).  Computed in
   Python using precomputed ``moment_grads`` in the
   ``SphericalMomentBuffer``.

The prototype implements bare Coulomb and erfc-screened kernels through
L=4. Public BIPOLE SCF routes fail closed before this quartet far-field
gradient can be selected.

Provenance
----------
Pisani, Dovesi, and Roetti (1988), Ch. II.4c,
doi:10.1007/978-3-642-93385-1, derives the periodic quartet expansion.
Saunders et al. (1992), Sec. 5.3, Eqs. (90)-(92), gives Cartesian
derivatives of a spherically symmetric radial kernel, but not this full
quartet gradient. The assembly below is a direct derivative of the
implementation prototype and is not route-certified.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .bipole_multipole import n_components as _n_sph
from .bipole_multipole import (
    multipole_interaction_tensor,
)

__all__ = [
    "multipole_interaction_tensor_gradient",
    "build_bipolar_far_field_gradient_contribution",
    "_add_moment_derivative_gradient",
]


# ---------------------------------------------------------------------------
# Gradient of the Cartesian interaction kernel
# ---------------------------------------------------------------------------


def _cartesian_derivative_inv_r_gradient(
    gamma: Tuple[int, int, int], R: np.ndarray,
) -> np.ndarray:
    """Cartesian gradient ∇_R d^gamma (1/|R|).

    ∂/∂R_a d^{(i,j,k)}(1/r) = d^{(i+δ_a)}(1/r)
    (no minus sign — the next-order derivative IS the gradient).

    Returns (3,) array [d/dx, d/dy, d/dz].
    """
    from .bipole_multipole import _cartesian_derivative_inv_r

    i, j, k = gamma
    dx = _cartesian_derivative_inv_r((i + 1, j, k), R)
    dy = _cartesian_derivative_inv_r((i, j + 1, k), R)
    dz = _cartesian_derivative_inv_r((i, j, k + 1), R)
    return np.array([dx, dy, dz], dtype=float)


def multipole_interaction_tensor_gradient(
    L_max_A: int,
    L_max_B: int,
    R: np.ndarray,
) -> np.ndarray:
    """Gradient ``∇_R T_{l1,m1;l2,m2}(R)``, shape ``(3, n_A, n_B)``.

    Differentiates the bipolar Taylor expansion of ``1/|R - r1 + r2|``
    with respect to the separation vector R.  The gradient of the
    Cartesian interaction tensor is:

        ∇_R T_cart[α,β] = (-1)^{|α|}/(α! β!) · ∇_R d^{α+β}(1/|R|)

    and the spherical gradient follows from the same pseudoinverse
    conversion as the tensor itself.

    Prefers the C++ implementation which supports L up to 4 and
    uses the McMurchie-Davidson Hermite ladder for screened kernels.

    Parameters
    ----------
    L_max_A, L_max_B : int
        Maximum multipole orders.
    R : ndarray shape (3,)
        Separation vector (must have |R| > 0).

    Returns
    -------
    ndarray shape (3, n_A, n_B)
        ``grad_T[a, i, j]`` = ∂/∂R_a of T_{i,j}(R).
    """
    R = np.asarray(R, dtype=float).reshape(3)
    L_max = max(L_max_A, L_max_B)

    # Try C++ native gradient first (supports L up to 4).
    try:
        from ._vibeqc_core import (
            multipole_interaction_tensor_gradient as _cpp_gradT,
        )
    except ImportError:
        _cpp_gradT = None

    if _cpp_gradT is not None:
        cpp_result = _cpp_gradT(
            L_max_A, L_max_B,
            float(R[0]), float(R[1]), float(R[2]),
        )
        return np.asarray(cpp_result, dtype=float)

    # Pure-Python fallback for L <= 3 (the old path).
    if L_max > 3:
        raise ValueError(
            "multipole_interaction_tensor_gradient: L_max > 4 not supported"
        )

    from math import factorial as _factorial
    from .bipole_cell_moments import cartesian_component_indices as _cart_idx
    from ._cart_to_sph import cartesian_to_spherical_matrix as _C_mat

    indices = _cart_idx(L_max)
    n_cart = len(indices)
    n_A = _n_sph(L_max_A)
    n_B = _n_sph(L_max_B)

    grad_T_cart = np.zeros((3, n_cart, n_cart), dtype=float)

    for ia, (i1, j1, k1) in enumerate(indices):
        for ib, (i2, j2, k2) in enumerate(indices):
            L = i1 + i2 + j1 + j2 + k1 + k2
            if L > L_max:
                continue
            gamma = (i1 + i2, j1 + j2, k1 + k2)
            sign = (-1.0) ** (i1 + j1 + k1)
            denom = (
                _factorial(i1) * _factorial(j1) * _factorial(k1)
                * _factorial(i2) * _factorial(j2) * _factorial(k2)
            )
            grad_D = _cartesian_derivative_inv_r_gradient(gamma, R)
            grad_T_cart[:, ia, ib] = sign * grad_D / denom

    C = _C_mat(L_max)
    C_pinv = np.linalg.pinv(C)
    grad_T_sph = np.zeros((3, n_A, n_B), dtype=float)
    for a in range(3):
        gt = C_pinv.T @ grad_T_cart[a] @ C_pinv
        grad_T_sph[a, :n_A, :n_B] = gt[:n_A, :n_B]
    return grad_T_sph


# ---------------------------------------------------------------------------
# Far-field nuclear gradient contribution
# ---------------------------------------------------------------------------


def build_bipolar_far_field_gradient_contribution(
    moment_buffer,
    dispatch: "QuartetBipolarDispatch",
    density_blocks: Dict[Tuple[int, int, int], np.ndarray],
    n_atoms: int,
    *,
    ewald_omega: float = 0.0,
    atom_to_shells: Optional[Dict[int, List[int]]] = None,
    include_moment_derivative: bool = True,
) -> np.ndarray:
    """Compute the far-field bipolar contribution to nuclear forces.

    For each far-field quartet, the gradient of the multipole interaction
    tensor T(R) with respect to the separation vector R = C_ket - C_bra
    is computed.  The force on each atom follows from the chain rule
    through the product-distribution centres.

    The dominant long-range term is dT/dR:  the force on atom A is

        F_A = - Sum_q D_bra * D_ket * (M_bra^T * grad_R T * M_ket)
              * (dC_ket/dA - dC_bra/dA)

    where C_bra = Sum_sh w_sh A_sh is the weighted average of atomic
    positions contributing to the bra product distribution.

    If ``include_moment_derivative`` is True, the sub-dominant
    dM/dA terms are also evaluated: the energy depends on the pair
    centre C through the shifted moments M(C), so

        dE/dA += D_bra * D_ket *
                 [(dM_bra/dC_bra * dC_bra/dA)^T * T * M_ket
                + M_bra^T * T * (dM_ket/dC_ket * dC_ket/dA)]

    These terms are O(1/R^{L+2}) versus O(1/R^{L+1}) for the dT/dR
    term by direct differentiation of the implemented expansion.

    Parameters
    ----------
    moment_buffer : SphericalMomentBuffer
        Must carry ``moment_grads`` if ``include_moment_derivative``.
    dispatch : QuartetBipolarDispatch
    density_blocks : dict of (ix,iy,iz) → (nbf,nbf) ndarray
    n_atoms : int
    ewald_omega : float
    atom_to_shells : dict mapping atom_index → list of shell indices
    include_moment_derivative : bool
        If True (default), include dM/dA terms for higher accuracy.
        Requires the buffer to have ``moment_grads`` populated
        (built automatically when L_max >= 1).

    Returns
    -------
    ndarray shape (n_atoms, 3)
        Far-field nuclear force contribution in Hartree/bohr.
    """
    # ---- Try C++ OpenMP gradient contractor first ---------------------
    try:
        from .bipole_far_field_gradient_native import (
            compute_bipolar_far_field_gradient_native,
        )
        return compute_bipolar_far_field_gradient_native(
            moment_buffer, dispatch, density_blocks, n_atoms,
            ewald_omega=ewald_omega,
            atom_to_shells=atom_to_shells,
            include_moment_derivative=include_moment_derivative,
        )
    except (ImportError, RuntimeError):
        pass  # Fall back to pure Python below.

    # ---- Python fallback path ---------------------------------------------
    grad = np.zeros((n_atoms, 3), dtype=float)
    slices = moment_buffer.shell_slices

    # Cache for interaction tensor gradients.
    tensor_cache: Dict[Tuple, np.ndarray] = {}

    # Try C++ native gradient for speed.
    _use_cpp_grad = False
    _use_cpp_erfc_grad = False
    try:
        from ._vibeqc_core import (
            multipole_interaction_tensor_gradient as _cpp_gradT,
            multipole_erfc_interaction_tensor_gradient as _cpp_erfc_gradT,
        )
        _use_cpp_grad = True
        _use_cpp_erfc_grad = True
    except ImportError:
        pass

    for q in range(len(dispatch)):
        s1, s2, bra_cell = dispatch.bra_pairs[q]
        s3, s4, ket_cell = dispatch.ket_pairs[q]
        L_order = dispatch.truncation_orders[q]
        if L_order <= 0:
            continue

        centre_bra = moment_buffer.get_center(s1, s2, bra_cell)
        centre_ket = moment_buffer.get_center(s3, s4, ket_cell)
        if centre_bra is None or centre_ket is None:
            continue
        R_sep = centre_ket - centre_bra
        r2 = float(np.dot(R_sep, R_sep))
        if r2 < 1e-30:
            continue

        # Gradient of the interaction tensor.
        cache_key = (
            round(float(R_sep[0]), 5),
            round(float(R_sep[1]), 5),
            round(float(R_sep[2]), 5),
            int(L_order),
        )
        if cache_key not in tensor_cache:
            if _use_cpp_erfc_grad and ewald_omega > 0:
                gamma_bra = dispatch.bra_widths[q]
                gamma_ket = dispatch.ket_widths[q]
                from .bipole_quartet_far_field import (
                    quartet_effective_screening_parameter,
                )
                mu_eff = quartet_effective_screening_parameter(
                    gamma_bra, gamma_ket, ewald_omega)
                grad_T_list = _cpp_erfc_gradT(
                    L_order, L_order,
                    float(R_sep[0]), float(R_sep[1]), float(R_sep[2]),
                    float(mu_eff),
                )
                grad_T = np.asarray(grad_T_list, dtype=float)
            elif _use_cpp_grad:
                grad_T_list = _cpp_gradT(
                    L_order, L_order,
                    float(R_sep[0]), float(R_sep[1]), float(R_sep[2]),
                )
                grad_T = np.asarray(grad_T_list, dtype=float)
            else:
                grad_T = multipole_interaction_tensor_gradient(
                    L_order, L_order, R_sep,
                )
            tensor_cache[cache_key] = grad_T
        grad_T = tensor_cache[cache_key]  # (3, n_comp, n_comp)

        # Get density sub-blocks.
        b1, n1 = slices[s1]
        b2, n2 = slices[s2]
        b3, n3 = slices[s3]
        b4, n4 = slices[s4]

        ket_cell_obj = moment_buffer.cells[ket_cell]
        ket_key = (
            ket_cell_obj.index[0],
            ket_cell_obj.index[1],
            ket_cell_obj.index[2],
        )
        bra_cell_obj = moment_buffer.cells[bra_cell]
        bra_key = (
            bra_cell_obj.index[0],
            bra_cell_obj.index[1],
            bra_cell_obj.index[2],
        )

        D_ket = density_blocks.get(ket_key)
        D_bra = density_blocks.get(bra_key)
        if D_ket is None or D_bra is None:
            continue
        D_ket_sub = np.asarray(D_ket[b3:b3+n3, b4:b4+n4], dtype=float)
        D_bra_sub = np.asarray(D_bra[b1:b1+n1, b2:b2+n2], dtype=float)

        # Get shell-pair moments.
        n_keep = _n_sph(L_order)
        mom_bra = moment_buffer.get_moments(s1, s2, bra_cell)
        mom_ket = moment_buffer.get_moments(s3, s4, ket_cell)
        if mom_bra is None or mom_ket is None:
            continue
        mom_bra_q = np.asarray(mom_bra[:, :, :n_keep], dtype=float)
        mom_ket_q = np.asarray(mom_ket[:, :, :n_keep], dtype=float)
        mom_bra_2d = mom_bra_q.reshape(n1 * n2, n_keep)
        mom_ket_2d = mom_ket_q.reshape(n3 * n4, n_keep)
        D_ket_flat = D_ket_sub.reshape(n3 * n4)
        D_bra_flat = D_bra_sub.reshape(n1 * n2)

        # For each Cartesian direction: scalar force.
        # F_R_a = D_bra · (M_bra · ∇_a T · M_ket^T) · D_ket
        f_R = np.zeros(3, dtype=float)
        for a in range(3):
            grad_E_ab = mom_bra_2d @ grad_T[a] @ mom_ket_2d.T
            f_R[a] = float(D_bra_flat @ grad_E_ab @ D_ket_flat)

        # Chain rule: dR/dA.
        # R = C_ket - C_bra, so dR/dC_bra = -I, dR/dC_ket = +I.
        # The centre is a weighted average of atomic positions.
        # We distribute the force to the atoms hosting the shells.
        # ---------------------------------------------------------------
        # Determine which atoms contribute to the bra/ket centres.
        bra_atoms = set()
        ket_atoms = set()
        if atom_to_shells is not None:
            for sh_idx, atom_map in atom_to_shells.items():
                if sh_idx == s1 or sh_idx == s2:
                    bra_atoms.update(atom_map)
                if sh_idx == s3 or sh_idx == s4:
                    ket_atoms.update(atom_map)

        # ---- dT/dR contribution (dominant, O(1/R^{L+1})) -------------
        _distribute_force(grad, bra_atoms, ket_atoms, -f_R, f_R)

        # ---- dM/dA contribution (sub-dominant, O(1/R^{L+2})) ---------
        if include_moment_derivative:
            mom_grad_bra = moment_buffer.get_moment_gradient(
                s1, s2, bra_cell)
            mom_grad_ket = moment_buffer.get_moment_gradient(
                s3, s4, ket_cell)
            if mom_grad_bra is not None and mom_grad_ket is not None:
                n_sph_deriv = mom_grad_bra.shape[3]
                # Slice interaction tensor to match derivative moments.
                # T has shape (n_keep, n_keep); the derivative
                # contraction uses only the first n_sph_deriv rows/cols.
                T_sub_bra = grad_T[:, :n_sph_deriv, :n_keep]
                T_sub_ket = grad_T[:, :n_keep, :n_sph_deriv]

                # Bra moment derivative: dE/dC_bra_a =
                #   D_bra^T * (dM_bra/dC_a) @ T @ M_ket^T * D_ket
                # Shapes: dM_bra_a (n1*n2, n_sph_deriv),
                #   T_sub (n_sph_deriv, n_keep),
                #   mom_ket_2d (n3*n4, n_keep)
                dE_dC = np.zeros(3, dtype=float)
                for a in range(3):
                    dM_bra_a = np.asarray(
                        mom_grad_bra[a, :, :, :n_sph_deriv],
                        dtype=float,
                    ).reshape(n1 * n2, n_sph_deriv)
                    # TM = T_sub @ M_ket^T: (n_sph_deriv, n_keep) @ (n_keep, n3*n4)
                    TM = grad_T[a][:n_sph_deriv, :n_keep] @ mom_ket_2d.T
                    # E_deriv = dM_bra @ TM: (n1*n2, n_sph_deriv) @ (n_sph_deriv, n3*n4)
                    E_deriv = dM_bra_a @ TM
                    dE_dC[a] += float(D_bra_flat @ E_deriv @ D_ket_flat)

                # Ket moment derivative: dE/dC_ket_a =
                #   D_bra^T * M_bra @ T @ (dM_ket/dC_a)^T * D_ket
                # Shapes: mom_bra_2d (n1*n2, n_keep),
                #   T_sub (n_keep, n_sph_deriv),
                #   dM_ket_a (n3*n4, n_sph_deriv)
                for a in range(3):
                    dM_ket_a = np.asarray(
                        mom_grad_ket[a, :, :, :n_sph_deriv],
                        dtype=float,
                    ).reshape(n3 * n4, n_sph_deriv)
                    # TM = M_bra @ T_sub: (n1*n2, n_keep) @ (n_keep, n_sph_deriv)
                    TM = mom_bra_2d @ grad_T[a][:n_keep, :n_sph_deriv]
                    # E_deriv = TM @ dM_ket^T: (n1*n2, n_sph_deriv) @ (n_sph_deriv, n3*n4)
                    E_deriv = TM @ dM_ket_a.T
                    dE_dC[a] += float(D_bra_flat @ E_deriv @ D_ket_flat)

                # Distribute moment-derivative forces: dE/dC_bra and
                # dE/dC_ket contribute to atomic forces via the chain
                # rule through the pair centres (positive convention).
                _distribute_force(
                    grad, bra_atoms, ket_atoms, dE_dC, dE_dC)

    return grad


def _add_moment_derivative_gradient(
    grad: np.ndarray,
    moment_buffer,
    dispatch: "QuartetBipolarDispatch",
    density_blocks: Dict[Tuple[int, int, int], np.ndarray],
    n_atoms: int,
    ewald_omega: float,
    atom_to_shells: Optional[Dict[int, List[int]]] = None,
) -> None:
    """Add dM/dA contribution to an existing gradient array in-place.

    This is the Python-side complement to the C++ dT/dR gradient
    contractor.  It uses the precomputed ``moment_buffer.moment_grads``
    to evaluate the sub-dominant moment-derivative force terms.
    """
    slices = moment_buffer.shell_slices
    # Try C++ gradient tensors, fall back to Python.
    _use_cpp_grad = False
    _use_cpp_erfc_grad = False
    try:
        from ._vibeqc_core import (
            multipole_interaction_tensor_gradient as _cpp_gradT,
            multipole_erfc_interaction_tensor_gradient as _cpp_erfc_gradT,
        )
        _use_cpp_grad = True
        _use_cpp_erfc_grad = True
    except ImportError:
        pass

    tensor_cache: Dict[Tuple, np.ndarray] = {}

    for q in range(len(dispatch)):
        s1, s2, bra_cell = dispatch.bra_pairs[q]
        s3, s4, ket_cell = dispatch.ket_pairs[q]
        L_order = dispatch.truncation_orders[q]
        if L_order <= 0:
            continue

        # Get moment derivatives.
        mom_grad_bra = moment_buffer.get_moment_gradient(s1, s2, bra_cell)
        mom_grad_ket = moment_buffer.get_moment_gradient(s3, s4, ket_cell)
        if mom_grad_bra is None or mom_grad_ket is None:
            continue

        centre_bra = moment_buffer.get_center(s1, s2, bra_cell)
        centre_ket = moment_buffer.get_center(s3, s4, ket_cell)
        if centre_bra is None or centre_ket is None:
            continue
        R_sep = centre_ket - centre_bra
        r2 = float(np.dot(R_sep, R_sep))
        if r2 < 1e-30:
            continue

        # Get gradient of interaction tensor.
        mu_eff = 0.0
        if ewald_omega > 0:
            from .bipole_quartet_far_field import (
                quartet_effective_screening_parameter,
            )
            gamma_bra = dispatch.bra_widths[q]
            gamma_ket = dispatch.ket_widths[q]
            mu_eff = quartet_effective_screening_parameter(
                gamma_bra, gamma_ket, ewald_omega)

        cache_key = (
            round(float(R_sep[0]), 5),
            round(float(R_sep[1]), 5),
            round(float(R_sep[2]), 5),
            int(L_order),
            round(float(mu_eff), 5),
        )
        if cache_key not in tensor_cache:
            if _use_cpp_erfc_grad and ewald_omega > 0:
                grad_T_list = _cpp_erfc_gradT(
                    L_order, L_order,
                    float(R_sep[0]), float(R_sep[1]), float(R_sep[2]),
                    float(mu_eff),
                )
                grad_T = np.asarray(grad_T_list, dtype=float)
            elif _use_cpp_grad:
                grad_T_list = _cpp_gradT(
                    L_order, L_order,
                    float(R_sep[0]), float(R_sep[1]), float(R_sep[2]),
                )
                grad_T = np.asarray(grad_T_list, dtype=float)
            else:
                grad_T = multipole_interaction_tensor_gradient(
                    L_order, L_order, R_sep,
                )
            tensor_cache[cache_key] = grad_T
        grad_T = tensor_cache[cache_key]

        n_keep = _n_sph(L_order)
        n_sph_deriv = mom_grad_bra.shape[3]
        # Safety: derivative moments must have at least 1 component.
        if n_sph_deriv < 1:
            continue
        # Use only the minimum of available derivative components
        # and required tensor dimensions.
        n_use = min(n_sph_deriv, n_keep)
        b1, n1 = slices[s1]; b2, n2 = slices[s2]
        b3, n3 = slices[s3]; b4, n4 = slices[s4]

        # Density sub-blocks.
        ket_cell_obj = moment_buffer.cells[ket_cell]
        ket_key = (ket_cell_obj.index[0], ket_cell_obj.index[1], ket_cell_obj.index[2])
        bra_cell_obj = moment_buffer.cells[bra_cell]
        bra_key = (bra_cell_obj.index[0], bra_cell_obj.index[1], bra_cell_obj.index[2])

        D_ket = density_blocks.get(ket_key)
        D_bra = density_blocks.get(bra_key)
        if D_ket is None or D_bra is None:
            continue
        D_ket_sub = np.asarray(D_ket[b3:b3+n3, b4:b4+n4], dtype=float)
        D_bra_sub = np.asarray(D_bra[b1:b1+n1, b2:b2+n2], dtype=float)
        D_ket_flat = D_ket_sub.reshape(n3 * n4)
        D_bra_flat = D_bra_sub.reshape(n1 * n2)

        # Moments.
        mom_bra = moment_buffer.get_moments(s1, s2, bra_cell)
        mom_ket = moment_buffer.get_moments(s3, s4, ket_cell)
        if mom_bra is None or mom_ket is None:
            continue
        mom_bra_2d = np.asarray(mom_bra[:, :, :n_keep], dtype=float).reshape(n1 * n2, n_keep)
        mom_ket_2d = np.asarray(mom_ket[:, :, :n_keep], dtype=float).reshape(n3 * n4, n_keep)

        # Atoms.
        bra_atoms = set()
        ket_atoms = set()
        if atom_to_shells is not None:
            for sh_idx, atom_map in atom_to_shells.items():
                if sh_idx == s1 or sh_idx == s2:
                    bra_atoms.update(atom_map)
                if sh_idx == s3 or sh_idx == s4:
                    ket_atoms.update(atom_map)

        dE_dC = np.zeros(3, dtype=float)
        for a in range(3):
            dM_bra_a = np.asarray(
                mom_grad_bra[a, :, :, :n_use], dtype=float,
            ).reshape(n1 * n2, n_use)
            TM = grad_T[a][:n_use, :n_keep] @ mom_ket_2d.T
            E_deriv = dM_bra_a @ TM
            dE_dC[a] += float(D_bra_flat @ E_deriv @ D_ket_flat)

        for a in range(3):
            dM_ket_a = np.asarray(
                mom_grad_ket[a, :, :, :n_use], dtype=float,
            ).reshape(n3 * n4, n_use)
            TM = mom_bra_2d @ grad_T[a][:n_keep, :n_use]
            E_deriv = TM @ dM_ket_a.T
            dE_dC[a] += float(D_bra_flat @ E_deriv @ D_ket_flat)

        _distribute_force(grad, bra_atoms, ket_atoms, dE_dC, dE_dC)


def _distribute_force(
    grad: np.ndarray,
    bra_atoms: set,
    ket_atoms: set,
    f_bra: np.ndarray,
    f_ket: np.ndarray,
) -> None:
    """Distribute bra/ket forces to atoms, weighted equally.

    grad is modified in-place: bra atoms get +f_bra (each 1/N_bra)
    and ket atoms get +f_ket (each 1/N_ket).
    """
    if bra_atoms:
        fp = f_bra / len(bra_atoms)
        for a_idx in bra_atoms:
            grad[a_idx] += fp
    if ket_atoms:
        fp = f_ket / len(ket_atoms)
        for a_idx in ket_atoms:
            grad[a_idx] += fp
