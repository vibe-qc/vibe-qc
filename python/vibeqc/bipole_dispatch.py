"""Prototype geometric dispatcher for a periodic bipolar expansion.

Pisani, Dovesi, and Roetti (1988), Ch. II.4c, Eqs. II.4.7-II.4.10,
derive the periodic four-centre Coulomb expansion between two product
distributions and its common lattice-translation sum. They do not
derive the classifier implemented here. The ``D^2 * gamma`` metric,
cell-length scaling, overlap bound, and logarithmic order map in this
module are implementation prototypes and are not a literature-certified
near/far partition.

The prototype dispatch is geometric: it depends on product centres,
their separation, and diffuse primitive exponents. It does not depend
on the density matrix, so its result can be cached for a fixed geometry.
The name "penetration" follows Saunders et al. (1992), Secs. 6.7-6.8,
Eqs. (123)-(128), where overlapping Gaussian tails limit a point-charge
electrostatic approximation. Those equations do not specify this
quartet classifier or its truncation-order map.

References
----------
Pisani, C.; Dovesi, R.; Roetti, C. *Hartree-Fock Ab Initio Treatment
of Crystalline Systems*, Lecture Notes in Chemistry 48 (1988),
Ch. II.4c, Eqs. II.4.7-II.4.10. doi:10.1007/978-3-642-93385-1

Saunders, V. R.; Freyria-Fava, C.; Dovesi, R.; Salasco, L.;
Roetti, C. "On the electrostatic potential in crystalline systems
where the charge density is expanded in Gaussian functions."
*Mol. Phys.* **77**, 629 (1992). doi:10.1080/00268979200102671
- Secs. 6.7-6.8, Eqs. (123)-(128): electrostatic tail and penetration
  estimates. These are context for terminology, not a derivation of
  the prototype classifier below.

Pisani, C.; Dovesi, R. *Int. J. Quantum Chem.* **17**, 501 (1980),
Sec. 4, doi:10.1002/qua.560170311: the adjoined diffuse s-Gaussian
overlap-screening convention. The pair centre then follows from the
standard Gaussian product theorem.

Module status
-------------
2026-08-06 -- adopted electrostatic-penetration terminology and added
the quartet-level prototype shell-loop dispatcher. Its numerical
defaults remain implementation choices rather than paper-derived values.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


__all__ = [
    "PenetrationDispatchParameters",
    "product_distribution_gaussian_center",
    "product_distribution_gaussian_width",
    "estimate_shell_pair_overlap_upper_bound",
    "compute_quartet_multipole_truncation_order",
    "build_shell_quartet_penetration_dispatch",
    "build_shell_quartet_penetration_dispatch_native",
]


@dataclass(frozen=True)
class PenetrationDispatchParameters:
    """Parameters for the prototype geometric quartet dispatcher.

    These values control an implementation-specific near/far decision.
    Neither Pisani-Dovesi-Roetti (1988), Ch. II.4c, nor Saunders et al.
    (1992) derives this particular decision rule.

    Attributes
    ----------
    maximum_multipole_order : int
        Prototype ceiling for the retained order. Pisani-Dovesi-Roetti
        (1988), p. 51, discusses a total-order cutoff, but does not
        prescribe the default used here. In this dispatcher, order 0
        is reserved as the exact-quartet sentinel.
    dispatch_slope : float
        Steepness of the penetration→far transition.
        Default is ``maximum_multipole_order + 1``. This is a prototype
        implementation choice, not a formula from Saunders (1992).
    cell_length_scale_inv_bohr : float
        ``(1 / V_cell)^{1/3}`` for 3D, ``(1 / A)^{1/2}`` for 2D,
        ``1/L`` for 1D. The use of this cell scale in the classifier is
        implementation-specific.
        Named ``cell_length_scale_inv_bohr`` to emphasise that it is
        an inverse length (bohr^-1), not a dimensionless ratio.
    per_shell_scaling_factor : float
        Prototype per-system prefactor. Default 1.0.
    overlap_drop_threshold : float
        Prototype diffuse-primitive overlap bound below which the quartet
        is dropped. This is not a one-to-one mapping to TOLINTEG.
    """
    maximum_multipole_order: int = 4
    dispatch_slope: float = 5.0
    cell_length_scale_inv_bohr: float = 1.0
    per_shell_scaling_factor: float = 1.0
    overlap_drop_threshold: float = 1e-7

    @classmethod
    def for_cell_volume(
        cls,
        volume_bohr3: float,
        *,
        maximum_multipole_order: int = 4,
        per_shell_scaling_factor: float = 1.0,
        overlap_drop_threshold: float = 1e-7,
        dimensionality: int = 3,
    ) -> "PenetrationDispatchParameters":
        """Construct dispatch parameters for a periodic cell.

        Derives ``cell_length_scale_inv_bohr`` from the cell volume
        and dimensionality:
        * 3D: (1/V)^{1/3}
        * 2D: (1/A)^{1/2}
        * 1D: 1/L

        Sets ``dispatch_slope = maximum_multipole_order + 1``.
        """
        if volume_bohr3 <= 0.0:
            raise ValueError(
                f"cell volume must be positive; got {volume_bohr3}"
            )
        dim = int(dimensionality)
        if dim == 1:
            cell_length_scale_inv_bohr = 1.0 / volume_bohr3
        elif dim == 2:
            cell_length_scale_inv_bohr = float(volume_bohr3) ** (-0.5)
        else:
            cell_length_scale_inv_bohr = float(volume_bohr3) ** (-1.0 / 3.0)
        return cls(
            maximum_multipole_order=int(maximum_multipole_order),
            dispatch_slope=float(maximum_multipole_order + 1),
            cell_length_scale_inv_bohr=float(cell_length_scale_inv_bohr),
            per_shell_scaling_factor=float(per_shell_scaling_factor),
            overlap_drop_threshold=float(overlap_drop_threshold),
        )


# ---------------------------------------------------------------------------
# Product-distribution geometry
# ---------------------------------------------------------------------------


def product_distribution_gaussian_center(
    shell_centre_1: np.ndarray,
    shell_centre_2: np.ndarray,
    smallest_exponent_1: float,
    smallest_exponent_2: float,
) -> np.ndarray:
    """Product-distribution centre for two contracted shells.

    The weighted average ``(a_1·R_1 + a_2·R_2) / (a_1 + a_2)`` where
    ``a`` are the *smallest* primitive exponents of each contracted
    shell. Pisani-Dovesi (1980), Sec. 4, assigns each shell an adjoined
    diffuse s-Gaussian for overlap screening; applying the standard Gaussian
    product theorem gives this centre. The periodic quartet expansion uses
    product-distribution centroids (Pisani-Dovesi-Roetti 1988, Ch. II.4c),
    so the adjoined-Gaussian centre is an approximation to that origin.

    This centre is the natural origin for the multipole expansion
    of the product density ``φ_μ(r)·φ_ν(r-g)`` because it makes the
    dipole moment of the leading Gaussian product identically zero.

    Parameters
    ----------
    shell_centre_1, shell_centre_2 : array_like shape (3,)
        Shell Cartesian centres (bohr).  For a displaced shell pair
        at lattice vector g, pass ``shell_centre_2 = B + g``.
    smallest_exponent_1, smallest_exponent_2 : float > 0
        Smallest primitive Gaussian exponent in each contracted shell.

    Returns
    -------
    ndarray shape (3,)
        Product-distribution centre.
    """
    if smallest_exponent_1 <= 0.0 or smallest_exponent_2 <= 0.0:
        raise ValueError(
            "exponents must be positive; got "
            f"{smallest_exponent_1}, {smallest_exponent_2}"
        )
    a1 = float(smallest_exponent_1)
    a2 = float(smallest_exponent_2)
    return (a1 * np.asarray(shell_centre_1, dtype=float)
            + a2 * np.asarray(shell_centre_2, dtype=float)) / (a1 + a2)


def product_distribution_gaussian_width(
    smallest_exponent_1: float,
    smallest_exponent_2: float,
) -> float:
    """Product-distribution width parameter ``gamma = a₁·a₂ / (a₁ + a₂)``.

    Controls the overlap decay of the product density:
    ``⟨μ|ν(g)⟩ ∝ exp(-gamma · |R_1 - R_2 - g|²)``.

    Larger gamma means faster decay in this standard Gaussian overlap
    estimate. Saunders (1992), Eq. (91), instead concerns radial
    derivatives of ``1/r`` and is not the source of this expression.
    """
    if smallest_exponent_1 <= 0.0 or smallest_exponent_2 <= 0.0:
        raise ValueError(
            "exponents must be positive; got "
            f"{smallest_exponent_1}, {smallest_exponent_2}"
        )
    a1 = float(smallest_exponent_1)
    a2 = float(smallest_exponent_2)
    return a1 * a2 / (a1 + a2)


def estimate_shell_pair_overlap_upper_bound(
    smallest_exponent_1: float,
    smallest_exponent_2: float,
    centre_1: np.ndarray,
    centre_2: np.ndarray,
) -> float:
    """Upper bound on the overlap of two contracted shell pairs.

    Uses the most diffuse primitive exponents (smallest a₁, a₂) to
    estimate the maximum possible overlap: S_max ≈ (π/(a₁+a₂))^{3/2}
    · exp(-a₁·a₂/(a₁+a₂) · |R₁-R₂|²).  If this upper bound is below
    ``overlap_drop_threshold``, the quartet can be skipped entirely.

    This prototype bound has no documented one-to-one TOLINTEG mapping.
    """
    gamma = smallest_exponent_1 * smallest_exponent_2 / (
        smallest_exponent_1 + smallest_exponent_2
    )
    R_sq = float(
        np.sum(
            (np.asarray(centre_1) - np.asarray(centre_2)) ** 2
        )
    )
    prefactor = (np.pi / (smallest_exponent_1 + smallest_exponent_2)) ** 1.5
    return float(prefactor * np.exp(-gamma * R_sq))


# ---------------------------------------------------------------------------
# Prototype quartet-level geometric dispatch
# ---------------------------------------------------------------------------


def compute_quartet_multipole_truncation_order(
    bra_product_centre: np.ndarray,
    ket_product_centre: np.ndarray,
    bra_product_width: float,
    params: PenetrationDispatchParameters,
) -> int:
    """Multipole truncation order for one shell quartet.

    Uses the implementation-specific dimensionless metric ``D^2 * gamma_bra``.
    Small metric selects the exact path; larger metric selects a
    prototype truncated expansion. No cited primary source supplies an
    accuracy guarantee for this classifier, which currently uses only
    the bra-pair width in its order decision.

    Return semantics
    ----------------
    * ``0`` — near / penetrating quartet: use exact ERI.
    * ``1..max_order`` - quartet assigned to the prototype far path.
    * Increasing separation reduces the retained order toward 1 under
      the current logarithmic map.

    Parameters
    ----------
    bra_product_centre, ket_product_centre : array_like shape (3,)
        Product-distribution centres (bohr).
    bra_product_width : float > 0
        Bra-pair width ``gamma = a₁·a₂/(a₁+a₂)``.
    params : PenetrationDispatchParameters
        System-wide dispatch parameters.

    Returns
    -------
    int
        Multipole truncation order ∈ [0, params.maximum_multipole_order].
    """
    bra = np.asarray(bra_product_centre, dtype=float).reshape(3)
    ket = np.asarray(ket_product_centre, dtype=float).reshape(3)
    separation = bra - ket
    separation_sq = float(np.dot(separation, separation))

    # Prototype dimensionless penetration metric: D^2 * gamma.
    metric = separation_sq * float(bra_product_width)

    # Implementation-specific near-field cutoff.
    L_inv = params.cell_length_scale_inv_bohr
    near_cutoff = 1.0 / (params.dispatch_slope * L_inv + 1.0)

    if metric < near_cutoff:
        return 0  # Near-field sentinel: use the exact ERI path.

    # Far-field: order decreases with distance.
    ratio = metric / near_cutoff
    order_raw = params.maximum_multipole_order - int(np.log2(ratio))
    # When max_order == 0, all quartets are near-field (no far-field).
    # Otherwise, far-field quartets always receive at least L=1 (dipole),
    # Order zero is already reserved as the exact-path sentinel.
    if params.maximum_multipole_order == 0:
        return 0
    return int(max(1, min(params.maximum_multipole_order, order_raw)))


# ---------------------------------------------------------------------------
# Quartet-level dispatch builder
# ---------------------------------------------------------------------------


def _shell_min_exponents_and_origins(basis) -> Tuple[np.ndarray, np.ndarray]:
    """Return (min_exponents, origins) per shell, each shape (n_sh,)."""
    exponents = np.array(
        [min(sh.exponents) for sh in basis.shells()], dtype=float
    )
    origins = np.array(
        [
            np.asarray(sh.origin, dtype=float).reshape(3)
            for sh in basis.shells()
        ],
        dtype=float,
    )
    return exponents, origins


def build_shell_quartet_penetration_dispatch(
    basis,
    lattice_cells: list,
    params: Optional[PenetrationDispatchParameters] = None,
    *,
    compute_exchange: bool = False,
) -> Tuple[
    "QuartetBipolarDispatch",
    "QuartetBipolarDispatch",
]:
    """Build far-field quartet dispatches for Coulomb (J) and Exchange (K).

    Walks all shell quartets (s1, s2_g, s3_lam, s4_sig) on the
    given lattice cells, applies the prototype geometric classifier,
    and collects quartets assigned an order greater than zero. That
    assignment is not, by itself, a paper-derived validity certificate.

    The Coulomb and Exchange dispatches differ in the handling of
    the ket-side shell-pair width: for Exchange the product width
    involves the (s3, s4) pair rather than only the bra side.

    Returns (j_dispatch, k_dispatch) as
    :class:`QuartetBipolarDispatch` objects.

    Parameters
    ----------
    basis : BasisSet
    lattice_cells : list of LatticeCell
        Cells over which to enumerate shell quartets.
    params : PenetrationDispatchParameters or None
        Dispatch parameters.  If None, constructed from the cell volume.
    compute_exchange : bool
        If True, also compute the Exchange far-field dispatch.

    Returns
    -------
    (j_dispatch, k_dispatch) : (QuartetBipolarDispatch, QuartetBipolarDispatch)
    """
    from .bipole_quartet_far_field import QuartetBipolarDispatch

    if params is None:
        raise ValueError(
            "build_shell_quartet_penetration_dispatch: params is required; "
            "construct via PenetrationDispatchParameters.for_cell_volume()."
        )

    min_exp, origins = _shell_min_exponents_and_origins(basis)
    n_sh = len(min_exp)
    j_dispatch = QuartetBipolarDispatch()
    k_dispatch = QuartetBipolarDispatch()

    for s1 in range(n_sh):
        for s2 in range(s1, n_sh):
            a_bra = min_exp[s1]
            b_bra = min_exp[s2]
            gamma_bra = product_distribution_gaussian_width(a_bra, b_bra)

            for g_idx, cell_g in enumerate(lattice_cells):
                g = np.asarray(cell_g.r_cart, dtype=float).reshape(3)
                A_bra = origins[s1]
                B_bra = origins[s2] + g
                centre_bra = product_distribution_gaussian_center(
                    A_bra, B_bra, a_bra, b_bra
                )

                for s3 in range(n_sh):
                    for s4 in range(s3, n_sh):
                        a_ket = min_exp[s3]
                        b_ket = min_exp[s4]
                        gamma_ket = product_distribution_gaussian_width(
                            a_ket, b_ket
                        )

                        for lam_idx, cell_lam in enumerate(lattice_cells):
                            lam = np.asarray(
                                cell_lam.r_cart, dtype=float
                            ).reshape(3)
                            A_ket = origins[s3]
                            B_ket = origins[s4] + lam
                            centre_ket = (
                                product_distribution_gaussian_center(
                                    A_ket, B_ket, a_ket, b_ket
                                )
                            )

                            # ---- Prototype overlap pre-screening --------
                            # Skip quartets where bra or ket overlap is
                            # below the implementation threshold.
                            if params.overlap_drop_threshold > 0:
                                S_bra = estimate_shell_pair_overlap_upper_bound(
                                    a_bra, b_bra, A_bra, B_bra,
                                )
                                S_ket = estimate_shell_pair_overlap_upper_bound(
                                    a_ket, b_ket, A_ket, B_ket,
                                )
                                if S_bra < params.overlap_drop_threshold:
                                    continue
                                if S_ket < params.overlap_drop_threshold:
                                    continue

                            order_j = (
                                compute_quartet_multipole_truncation_order(
                                    centre_bra, centre_ket,
                                    gamma_bra, params,
                                )
                            )
                            if order_j > 0:
                                j_dispatch.add_quartet(
                                    s1, s2, g_idx,
                                    s3, s4, lam_idx,
                                    order_j, gamma_bra, gamma_ket,
                                )

                            if compute_exchange:
                                order_k = (
                                    compute_quartet_multipole_truncation_order(
                                        centre_bra, centre_ket,
                                        gamma_bra, params,
                                    )
                                )
                                if order_k > 0:
                                    k_dispatch.add_quartet(
                                        s1, s2, g_idx,
                                        s3, s4, lam_idx,
                                        order_k, gamma_bra, gamma_ket,
                                    )

    return j_dispatch, k_dispatch


def build_penetration_dispatch_for_bipole_context(
    basis,
    system,
    lattice_cells: list,
    *,
    ewald_omega: Optional[float] = None,
    maximum_multipole_order: int = 4,
) -> "QuartetBipolarDispatch":
    """Convenience: build Coulomb (J) far-field dispatch for a BIPOLE SCF.

    Derives the penetration parameters from the periodic system's cell
    volume via :meth:`PenetrationDispatchParameters.for_cell_volume`,
    then enumerates shell quartets on the given lattice cells and
    returns the Coulomb dispatch.  The Exchange dispatch can be built
    by passing the result to the full ``build_shell_quartet_penetration_dispatch``
    with ``compute_exchange=True``.

    This is the integration point between the geometric dispatch and
    the BIPOLE SCF driver: call once before the SCF loop, store the
    result on ``BipoleFockContext``, and pass it to
    ``accumulate_bipolar_coulomb_fock_contribution`` inside the
    Fock builder.

    Parameters
    ----------
    basis : BasisSet
    system : PeriodicSystem
    lattice_cells : list of LatticeCell
        The SR real-space cell list used by the Fock builder.
    ewald_omega : float or None
        Ewald splitting parameter for per-quartet effective screening.
        Passed through to the far-field contract; does not affect dispatch.
    maximum_multipole_order : int
        Maximum bipolar order (default 4, hexadecapole).

    Returns
    -------
    QuartetBipolarDispatch
        The Coulomb far-field dispatch.  Pass to
        :func:`bipole_quartet_far_field.accumulate_bipolar_coulomb_fock_contribution`.
    """
    from .bipole_bravais_utils import cell_length_scale_inv_bohr, cell_dimensionality
    from .bipole_quartet_far_field import QuartetBipolarDispatch

    measure = cell_length_scale_inv_bohr(system)
    dim = cell_dimensionality(system)
    lattice = np.asarray(system.lattice, dtype=float)
    if dim == 1:
        vol = float(np.linalg.norm(lattice.reshape(-1)[:3]))
    elif dim == 2:
        a = lattice[0, :3]
        b = lattice[1, :3]
        vol = float(np.linalg.norm(np.cross(a, b)))
    else:
        vol = float(abs(np.linalg.det(lattice.reshape(3, 3))))
    params = PenetrationDispatchParameters.for_cell_volume(
        vol,
        maximum_multipole_order=maximum_multipole_order,
        dimensionality=dim,
    )
    # Prefer C++ OpenMP-parallel path; fall back to pure Python.
    try:
        j_dispatch, _k_dispatch = (
            build_shell_quartet_penetration_dispatch_native(
                basis,
                lattice_cells,
                params,
                compute_exchange=False,
            )
        )
    except (ImportError, RuntimeError, ValueError, TypeError):
        j_dispatch, _k_dispatch = build_shell_quartet_penetration_dispatch(
            basis,
            lattice_cells,
            params,
            compute_exchange=False,
        )
    return j_dispatch


# ---------------------------------------------------------------------------
# C++-backed implementation of the prototype dispatcher
# ---------------------------------------------------------------------------


def build_shell_quartet_penetration_dispatch_native(
    basis,
    lattice_cells: list,
    params: PenetrationDispatchParameters,
    *,
    compute_exchange: bool = False,
) -> Tuple["QuartetBipolarDispatch", "QuartetBipolarDispatch"]:
    """Build far-field quartet dispatches using the C++ OpenMP-parallel
    penetration enumeration.

    This is the high-performance path: calls
    :func:`_vibeqc_core.compute_bipolar_penetration_dispatch` which runs
    the shell-quartet loop in C++ with thread-safe local accumulation and
    OpenMP dynamic scheduling over cell pairs.

    Falls back to the pure-Python path if the native extension is not
    available or the call fails.

    Parameters
    ----------
    basis : BasisSet
    lattice_cells : list of LatticeCell
    params : PenetrationDispatchParameters
    compute_exchange : bool

    Returns
    -------
    (j_dispatch, k_dispatch) : (QuartetBipolarDispatch, QuartetBipolarDispatch)
    """
    from .bipole_quartet_far_field import QuartetBipolarDispatch

    try:
        from ._vibeqc_core import (
            BipolarQuartetEntry,
            BipoleShellInfo,
            BipoleCellInfo,
            BipoleDispatchParams,
            compute_bipolar_penetration_dispatch,
        )
    except ImportError:
        # C++ extension not available; use pure-Python path.
        return build_shell_quartet_penetration_dispatch(
            basis, lattice_cells, params,
            compute_exchange=compute_exchange,
        )

    # Build shell descriptors.
    cpp_shells = []
    for i_sh, sh in enumerate(basis.shells()):
        info = BipoleShellInfo()
        info.min_exponent = float(min(sh.exponents))
        info.origin = (float(sh.origin[0]),
                        float(sh.origin[1]),
                        float(sh.origin[2]))
        info.atom_index = int(getattr(sh, 'atom_index', 0))
        cpp_shells.append(info)

    # Build cell descriptors.
    cpp_cells = []
    for cell in lattice_cells:
        cinfo = BipoleCellInfo()
        cinfo.r_cart = (float(cell.r_cart[0]),
                         float(cell.r_cart[1]),
                         float(cell.r_cart[2]))
        cinfo.index = (int(cell.index[0]),
                        int(cell.index[1]),
                        int(cell.index[2]))
        cpp_cells.append(cinfo)

    # Build dispatch parameters.
    cpp_params = BipoleDispatchParams()
    cpp_params.cell_length_scale_inv = float(
        params.cell_length_scale_inv_bohr
    )
    cpp_params.dispatch_slope = float(params.dispatch_slope)
    cpp_params.overlap_threshold = float(params.overlap_drop_threshold)
    cpp_params.max_multipole_order = int(params.maximum_multipole_order)

    # Call C++ OpenMP-parallel dispatch.
    entries = compute_bipolar_penetration_dispatch(
        cpp_shells, cpp_cells, cpp_params,
    )

    # Convert C++ entries to QuartetBipolarDispatch.
    j_dispatch = QuartetBipolarDispatch()
    k_dispatch = QuartetBipolarDispatch()
    for e in entries:
        j_dispatch.add_quartet(
            e.s1, e.s2, e.c_bra,
            e.s3, e.s4, e.c_ket,
            e.truncation_order,
            e.bra_width, e.ket_width,
        )
        if compute_exchange:
            k_dispatch.add_quartet(
                e.s1, e.s2, e.c_bra,
                e.s3, e.s4, e.c_ket,
                e.truncation_order,
                e.bra_width, e.ket_width,
            )

    return j_dispatch, k_dispatch
