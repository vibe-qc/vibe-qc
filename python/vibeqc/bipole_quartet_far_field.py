"""Quartet-level bipolar far-field Fock contributions.

BIPOLE-EXACT-ZONE increment 3b.  Given a pre-computed
:class:`SphericalMomentBuffer` and a geometric penetration dispatch
(:class:`QuartetBipolarDispatch`), this module computes the Coulomb (J)
and Exchange (K) two-electron Fock matrix contributions from the
truncated multipole-multipole (bipolar) expansion for far-field
(non-penetrating) shell quartets.

Theory and provenance
---------------------
Pisani, Dovesi, and Roetti, *Hartree-Fock Ab Initio Treatment of
Crystalline Systems* (1988), Ch. II.4c, Eqs. II.4.7-II.4.10, derive
the periodic bipolar expansion between product distributions. The
quartet classifier, screening-parameter composition, and contraction
kernel in this module are implementation prototypes rather than
algorithms derived in that source.

In the Ewald-split gauge (Dovesi et al., *Phys. Rev. B* 28, 5781 (1983),
Eq. (24a)), the two-electron operator is partitioned as::

    1/r = erfc(w·r)/r  +  erf(w·r)/r
         (short-range)    (long-range)

The exact ``J_LR`` reciprocal channel carries the ``erf`` half exactly.
For non-penetrating shell quartets, the remaining ``erfc`` short-range
four-centre ERI is replaced by a multipole-multipole interaction through
the short-range tensor ``T_sr = T_bare - T_erf``, whose per-quartet
effective screening parameter follows::

    1/mu_eff = 1/gamma_bra + 1/gamma_ket + 1/w^2

This composition is implementation-specific. Saunders (1992), Eq. (91),
is a radial derivative of ``1/r`` and does not derive this expression.

Convention
----------
All multipole moments are in the Stone real-solid-harmonic convention
(Schmidt-semi-normalised, sqrt(4pi/(2l+1)) absorbed), identical to
the ``_cart_to_sph`` conversion and ``bipole_multipole`` tensors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .bipole_multipole import n_components as _n_sph
from .bipole_multipole import (
    multipole_interaction_tensor,
    sr_multipole_interaction_tensor,
)
from .bipole_spherical_moment_buffer import SphericalMomentBuffer

__all__ = [
    "QuartetBipolarDispatch",
    "QuartetMultipoleFarField",
    "quartet_effective_screening_parameter",
    "estimate_quartet_product_distribution_width",
    "build_bipolar_coulomb_far_field",
    "build_bipolar_exchange_far_field",
]


# ---------------------------------------------------------------------------
# Quartet dispatch data structure
# ---------------------------------------------------------------------------


@dataclass
class QuartetBipolarDispatch:
    """Per-shell-quartet flags from the prototype classifier.

    This record holds quartets assigned to the dormant far path and the
    metadata needed by its contractor. Membership is not a paper-derived
    certificate that a multipole approximation is valid.

    These are computed once per geometry and reused across SCF iterations.

    Attributes
    ----------
    bra_pairs : list of (s1, s2, cell_idx)
        Bra-side shell-pair identifiers, indexed into the moment buffer.
    ket_pairs : list of (s3, s4, cell_idx)
        Ket-side shell-pair identifiers.
    truncation_orders : list of int
        Per-quartet maximum multipole order (0 to 4).  Quartets with
        order 0 are not in this dispatch (they are near-field/exact).
    bra_widths : list of float
        Product-distribution width gamma_bra for each quartet.
    ket_widths : list of float
        Product-distribution width gamma_ket for each quartet.
    """

    bra_pairs: List[Tuple[int, int, int]] = field(default_factory=list)
    ket_pairs: List[Tuple[int, int, int]] = field(default_factory=list)
    truncation_orders: List[int] = field(default_factory=list)
    bra_widths: List[float] = field(default_factory=list)
    ket_widths: List[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.bra_pairs)

    def add_quartet(
        self,
        s1: int,
        s2: int,
        cell_bra_idx: int,
        s3: int,
        s4: int,
        cell_ket_idx: int,
        truncation_order: int,
        bra_width: float,
        ket_width: float,
    ) -> None:
        """Register one far-field quartet."""
        self.bra_pairs.append((s1, s2, cell_bra_idx))
        self.ket_pairs.append((s3, s4, cell_ket_idx))
        self.truncation_orders.append(truncation_order)
        self.bra_widths.append(bra_width)
        self.ket_widths.append(ket_width)


@dataclass
class QuartetMultipoleFarField:
    """Result of a quartet-level bipolar far-field Fock build.

    Attributes
    ----------
    fock_blocks : dict
        ``{(gx, gy, gz): ndarray(nbf, nbf)}`` -- the accumulated far-field
        Fock matrices.  These are ADDITIVE corrections to the exact
        short-range Fock.
    e_coulomb_far : float
        Far-field Coulomb multipole energy contribution.
    e_exchange_far : float
        Far-field Exchange multipole energy contribution.
    n_quartets : int
        Number of far-field quartets evaluated.
    """

    fock_blocks: Dict[Tuple[int, int, int], np.ndarray] = field(
        default_factory=dict
    )
    e_coulomb_far: float = 0.0
    e_exchange_far: float = 0.0
    n_quartets: int = 0


# ---------------------------------------------------------------------------
# Gaussian-width helpers (shared with bipole_dispatch)
# ---------------------------------------------------------------------------


def estimate_quartet_product_distribution_width(
    exponents_s1: np.ndarray,
    exponents_s2: np.ndarray,
    *,
    use_smallest: bool = True,
) -> float:
    """Product-distribution Gaussian width gamma = a1*a2/(a1+a2).

    This standard Gaussian overlap-decay parameter is evaluated with the
    adjoined diffuse s-Gaussians of Pisani-Dovesi (1980), Sec. 4. Its use
    inside the current quartet classifier remains implementation-specific.
    """
    if use_smallest:
        a1 = float(np.min(exponents_s1))
        a2 = float(np.min(exponents_s2))
    else:
        a1 = float(np.max(exponents_s1))
        a2 = float(np.max(exponents_s2))
    return a1 * a2 / (a1 + a2)


def quartet_effective_screening_parameter(
    gamma_bra: float,
    gamma_ket: float,
    ewald_omega: float,
) -> float:
    """Effective ``mu`` for the erfc-screened quartet interaction tensor.

    The implementation uses
    ``1/mu_eff = 1/gamma_bra + 1/gamma_ket + 1/omega^2``.
    No derivation for this quartet composition has been identified in the
    cited periodic bipolar source. Saunders (1992), Eq. (91), is instead
    the bare-Coulomb radial derivative ladder.

    Returns ``mu_eff`` (NOT its inverse).
    """
    if gamma_bra <= 0 or gamma_ket <= 0:
        raise ValueError("gamma values must be positive")
    inv_mu = 1.0 / gamma_bra + 1.0 / gamma_ket
    if ewald_omega > 0:
        inv_mu += 1.0 / (ewald_omega * ewald_omega)
    if inv_mu <= 0:
        raise ValueError(f"invalid effective screening: inv_mu={inv_mu}")
    return 1.0 / inv_mu


# ---------------------------------------------------------------------------
# Main far-field Fock builders
# ---------------------------------------------------------------------------


def build_bipolar_coulomb_far_field(
    moment_buffer: SphericalMomentBuffer,
    dispatch: QuartetBipolarDispatch,
    density_blocks: Dict[Tuple[int, int, int], np.ndarray],
    *,
    ewald_omega: float = 0.0,
    nbf: int = 0,
    tensor_cache: Optional["QuartetTensorCache"] = None,
) -> QuartetMultipoleFarField:
    """Build the Coulomb (J) far-field Fock contribution.

    For each far-field quartet flagged in ``dispatch``, this:
    1. Retrieves the pre-computed spherical moments for bra and ket pairs.
    2. Computes (or looks up from cache) the interaction tensor at the
       product-centre separation.
    3. Contracts moments × tensor × density sub-block and accumulates
       into the Fock.

    Parameters
    ----------
    moment_buffer : SphericalMomentBuffer
        Pre-computed per-pair spherical moments.
    dispatch : QuartetBipolarDispatch
        Far-field quartet flags from the geometric penetration criterion.
    density_blocks : dict
        Density matrix blocks ``{(gx, gy, gz): ndarray(nbf, nbf)}``.
    ewald_omega : float
        Ewald splitting parameter.  0 = bare Coulomb; >0 = erfc-screened
        short-range tensor (matching the BIPOLE Ewald-split gauge).
    nbf : int
        Total number of basis functions (for allocating the Fock blocks).

    Returns
    -------
    QuartetMultipoleFarField
    """
    result = QuartetMultipoleFarField()
    slices = moment_buffer.shell_slices
    # Use pre-computed cache if provided, else build on-the-fly.
    _local_tensor_cache: Dict[Tuple, np.ndarray] = {}

    # Allocate Fock blocks on demand.
    def _fock_block(cell_key):
        if cell_key not in result.fock_blocks:
            result.fock_blocks[cell_key] = np.zeros(
                (nbf, nbf), dtype=float
            )
        return result.fock_blocks[cell_key]

    for q in range(len(dispatch)):
        s1, s2, bra_cell = dispatch.bra_pairs[q]
        s3, s4, ket_cell = dispatch.ket_pairs[q]
        L_order = dispatch.truncation_orders[q]
        if L_order <= 0:
            continue

        # Retrieve pre-computed moments.
        mom_bra = moment_buffer.get_moments(s1, s2, bra_cell)
        mom_ket = moment_buffer.get_moments(s3, s4, ket_cell)
        if mom_bra is None or mom_ket is None:
            continue
        # Truncate to per-quartet order.
        n_keep = _n_sph(L_order)
        if n_keep > mom_bra.shape[2] or n_keep > mom_ket.shape[2]:
            continue  # insufficient moments in buffer for this truncation order
        mom_bra_q = mom_bra[:, :, :n_keep]  # (n1, n2, n_keep)
        mom_ket_q = mom_ket[:, :, :n_keep]  # (n3, n4, n_keep)

        # Interaction tensor.
        centre_bra = moment_buffer.get_center(s1, s2, bra_cell)
        centre_ket = moment_buffer.get_center(s3, s4, ket_cell)
        if centre_bra is None or centre_ket is None:
            continue
        R_sep = centre_ket - centre_bra
        r2 = float(np.dot(R_sep, R_sep))
        if r2 < 1e-30:
            continue
        # Compute effective screening (zero for bare).
        mu_eff = 0.0
        if ewald_omega > 0:
            gamma_bra = dispatch.bra_widths[q]
            gamma_ket = dispatch.ket_widths[q]
            mu_eff = quartet_effective_screening_parameter(
                gamma_bra, gamma_ket, ewald_omega
            )
        cache_key = (
            round(R_sep[0], 5),
            round(R_sep[1], 5),
            round(R_sep[2], 5),
            L_order,
            round(float(mu_eff), 5),
        )
        # Look up from pre-computed cache, else local on-the-fly cache.
        T_mat = None
        if tensor_cache is not None:
            T_mat = tensor_cache.get(R_sep, L_order, float(mu_eff))
        if T_mat is None:
            T_mat = _local_tensor_cache.get(cache_key)
        if T_mat is None:
            if ewald_omega > 0:
                T_mat = sr_multipole_interaction_tensor(
                    L_order, L_order, R_sep, mu_eff
                )
            else:
                T_mat = multipole_interaction_tensor(
                    L_order, L_order, R_sep
                )
            _local_tensor_cache[cache_key] = T_mat

        # AO index ranges.
        b1, n1 = slices[s1]
        b2, n2 = slices[s2]
        b3, n3 = slices[s3]
        b4, n4 = slices[s4]

        # Bra cell key.
        bra_cell_obj = moment_buffer.cells[bra_cell]
        bra_key = (
            bra_cell_obj.index[0],
            bra_cell_obj.index[1],
            bra_cell_obj.index[2],
        )

        # Ket cell key for density look-up.
        ket_cell_obj = moment_buffer.cells[ket_cell]
        ket_key = (
            ket_cell_obj.index[0],
            ket_cell_obj.index[1],
            ket_cell_obj.index[2],
        )

        D_ket = density_blocks.get(ket_key)
        if D_ket is None:
            continue
        D_sub = np.asarray(D_ket[b3 : b3 + n3, b4 : b4 + n4], dtype=float)
        if np.max(np.abs(D_sub)) < 1e-30:
            continue

        F_bra = _fock_block(bra_key)

        # ---- Vectorised contract: M_bra · T · M_ket^T × D ---------
        # Reshape moment tensors for matrix multiply:
        #   mom_bra_q: (n1, n2, n_comp) → (n1*n2, n_comp)
        #   mom_ket_q: (n3, n4, n_comp) → (n3*n4, n_comp)
        #   E_flat = mom_bra_flat @ T @ mom_ket_flat.T  → (n1*n2, n3*n4)
        #   F += E_flat @ D_flat (reshaped back to n1×n2)
        mom_bra_2d = mom_bra_q.reshape(n1 * n2, n_keep)
        mom_ket_2d = mom_ket_q.reshape(n3 * n4, n_keep)
        # Interaction energy matrix per AO pair:
        # (n1*n2, n_keep) @ (n_keep, n_keep) @ (n_keep, n3*n4) → (n1*n2, n3*n4)
        E_pair = mom_bra_2d @ T_mat @ mom_ket_2d.T
        D_flat = D_sub.reshape(n3 * n4)
        F_contrib = E_pair @ D_flat  # (n1*n2,)
        F_bra[b1 : b1 + n1, b2 : b2 + n2] += F_contrib.reshape(n1, n2)

        # ---- Track Coulomb energy: 0.5 * Tr(D · F) ------------
        D_bra_sub = np.asarray(
            density_blocks.get(bra_key, np.zeros((nbf, nbf), dtype=float))[
                b1 : b1 + n1, b2 : b2 + n2
            ],
            dtype=float,
        )
        result.e_coulomb_far += 0.5 * float(
            np.sum(D_bra_sub * F_contrib.reshape(n1, n2))
        )

        result.n_quartets += 1

    return result


def build_bipolar_exchange_far_field(
    moment_buffer: SphericalMomentBuffer,
    dispatch: QuartetBipolarDispatch,
    density_blocks: Dict[Tuple[int, int, int], np.ndarray],
    *,
    ewald_omega: float = 0.0,
    exchange_scale: float = 0.5,
    nbf: int = 0,
    tensor_cache: Optional["QuartetTensorCache"] = None,
) -> QuartetMultipoleFarField:
    """Build the Exchange (K) far-field Fock contribution.

    The K contraction pattern differs from J: for each far-field quartet,
    the multipole energy ``E_ab = M_bra^T T M_ket`` couples the exchange
    density pattern ``D(mu,lam) · D(nu,sig)``.

    Parameters
    ----------
    exchange_scale : float
        Pre-factor: 0.5 for RHF (alpha_HF/4), c_HF/2 for hybrid DFT, 0
        for pure DFT.

    See :func:`build_bipolar_coulomb_far_field` for other parameters.
    """
    result = QuartetMultipoleFarField()
    slices = moment_buffer.shell_slices
    _local_tensor_cache: Dict[Tuple, np.ndarray] = {}

    def _fock_block(cell_key):
        if cell_key not in result.fock_blocks:
            result.fock_blocks[cell_key] = np.zeros(
                (nbf, nbf), dtype=float
            )
        return result.fock_blocks[cell_key]

    for q in range(len(dispatch)):
        s1, s2, bra_cell = dispatch.bra_pairs[q]
        s3, s4, ket_cell = dispatch.ket_pairs[q]
        L_order = dispatch.truncation_orders[q]
        if L_order <= 0:
            continue

        mom_bra = moment_buffer.get_moments(s1, s2, bra_cell)
        mom_ket = moment_buffer.get_moments(s3, s4, ket_cell)
        if mom_bra is None or mom_ket is None:
            continue
        n_keep = _n_sph(L_order)
        mom_bra_q = mom_bra[:, :, :n_keep]
        mom_ket_q = mom_ket[:, :, :n_keep]

        centre_bra = moment_buffer.get_center(s1, s2, bra_cell)
        centre_ket = moment_buffer.get_center(s3, s4, ket_cell)
        if centre_bra is None or centre_ket is None:
            continue
        R_sep = centre_ket - centre_bra
        r2 = float(np.dot(R_sep, R_sep))
        if r2 < 1e-30:
            continue  # overlapping product centres: multipole diverges at R=0
        cache_key = (
            round(R_sep[0], 5),
            round(R_sep[1], 5),
            round(R_sep[2], 5),
            L_order,
        )
        # Look up from pre-computed cache, else local on-the-fly cache.
        T_mat = None
        if tensor_cache is not None:
            T_mat = tensor_cache.get(R_sep, L_order, float(mu_eff))
        if T_mat is None:
            T_mat = _local_tensor_cache.get(cache_key)
        if T_mat is None:
            if ewald_omega > 0:
                mu_eff = quartet_effective_screening_parameter(
                    dispatch.bra_widths[q],
                    dispatch.ket_widths[q],
                    ewald_omega,
                )
                T_mat = sr_multipole_interaction_tensor(
                    L_order, L_order, R_sep, mu_eff
                )
            else:
                T_mat = multipole_interaction_tensor(
                    L_order, L_order, R_sep
                )
            _local_tensor_cache[cache_key] = T_mat

        b1, n1 = slices[s1]
        b2, n2 = slices[s2]
        b3, n3 = slices[s3]
        b4, n4 = slices[s4]

        bra_cell_obj = moment_buffer.cells[bra_cell]
        bra_key = (
            bra_cell_obj.index[0],
            bra_cell_obj.index[1],
            bra_cell_obj.index[2],
        )
        ket_cell_obj = moment_buffer.cells[ket_cell]
        ket_key = (
            ket_cell_obj.index[0],
            ket_cell_obj.index[1],
            ket_cell_obj.index[2],
        )

        D_ml = density_blocks.get(ket_key)
        if D_ml is None:
            continue
        D_ml_sub = np.asarray(
            D_ml[b3 : b3 + n3, b1 : b1 + n1], dtype=float
        )

        # K exchange: sigma cell for D(nu, sig)
        sig_key = (
            ket_cell_obj.index[0] - bra_cell_obj.index[0],
            ket_cell_obj.index[1] - bra_cell_obj.index[1],
            ket_cell_obj.index[2] - bra_cell_obj.index[2],
        )
        D_ns = density_blocks.get(sig_key)
        if D_ns is None:
            continue
        D_ns_sub = np.asarray(
            D_ns[b4 : b4 + n4, b2 : b2 + n2], dtype=float
        )

        F_bra = _fock_block(bra_key)

        # ---- Vectorised K contract: D_ml · E · D_ns --------------------
        # E_pair = mom_bra @ T @ mom_ket^T: (n1*n2, n3*n4)
        mom_bra_2d = mom_bra_q.reshape(n1 * n2, n_keep)
        mom_ket_2d = mom_ket_q.reshape(n3 * n4, n_keep)
        E_pair = mom_bra_2d @ T_mat @ mom_ket_2d.T
        # D_ml_sub: (n3, n1), D_ns_sub: (n4, n2)
        # K[mu,la] -= scale * sum_{nu,sig} D_ns[sig,nu] * E[mu,nu; la,si] * D_ml[la,mu]
        # This is: K_contrib = scale * D_ml^T @ E_reshaped @ D_ns
        # Reshape E_pair from (n1*n2, n3*n4) to (n1, n2, n3, n4)
        E_4d = E_pair.reshape(n1, n2, n3, n4)
        # Contract: K[mu, la] = scale * sum_{nu,si} D_ns[si,nu] * E[mu,nu,la,si] * D_ml[la,mu]
        # = scale * sum_{nu,si} E[mu,nu,la,si] * D_ml[la,mu] * D_ns[si,nu]
        # Compute via tensordot: (E, D_ml, D_ns) → contracted along (nu,si) axes
        # For now, use einsum for clarity:
        F_bra[b1 : b1 + n1, b2 : b2 + n2] -= exchange_scale * np.einsum(
            'mnls,lm,sn->mn',
            E_4d,
            D_ml_sub,
            D_ns_sub,
        )
        # Energy: -0.25 * scale * Tr(D_ml^T @ E @ D_ns)
        result.e_exchange_far -= 0.25 * exchange_scale * float(
            np.sum(D_ml_sub.T @ E_pair.reshape(n1 * n2, n3 * n4) @ D_ns_sub)
        )

        result.n_quartets += 1

    return result
