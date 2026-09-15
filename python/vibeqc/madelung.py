"""Phase 12e-c-4c-iv: Madelung cancellation helpers for neutral crystals.

The Ewald-3D Hartree J matrix in :mod:`vibeqc.ewald_composed` and the
multi-cell long-range J in :mod:`vibeqc.periodic_density` both pin the
G = 0 Fourier mode of the electronic potential to zero (see
:mod:`vibeqc.ewald_j`). That gauge choice introduces a Makov-Payne-
like scalar-times-overlap shift

    J_periodic = J_isolated + a_e . S,    a_e = -a_M . Q_e / L

where ``Q_e = ∫ r_e dr = tr(D . S)`` is the cell electron count,
``a_M ≈ 2.837`` is the simple-cubic Madelung constant, and ``L`` is the
cubic cell edge. Symmetrically, any self-consistent treatment of the
electron-nuclear attraction that also pins its G = 0 mode contributes
``-a_n . S`` with ``a_n = -a_M . Q_n / L`` (Q_n = S_I Z_I is the
unit-cell nuclear charge, sign: positive point charges).

For a **neutral crystal** (``Q_e = Q_n``) the total Makov-Payne
correction ``a_net = a_e - a_n`` vanishes exactly: the electronic
G = 0 pinning cancels against the matching nuclear term -- the
**Madelung cancellation**.

For a **charged cell** (e.g., a defect, an ion in a box) the net
a_net != 0 and the correction ``a_net . S`` shifts the Fock matrix by a
constant overlap amount. Adding it back recovers the isolated-system
limit; leaving it in is appropriate for a true charged periodic cell
with a compensating uniform background.

This module exposes the cancellation explicitly: query the cell
charges, compute a, apply the ``a . S`` shift. Drivers that want the
correction applied can fold it in optionally -- the Γ-Ewald and multi-k
SCF drivers currently don't apply it automatically, leaving the
behavior to the caller.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np

from ._vibeqc_core import CoulombMethod, PeriodicSystem
from .ewald_composed import makov_payne_coefficient_cubic


__all__ = [
    "cell_electron_charge",
    "cell_nuclear_charge",
    "cell_net_charge",
    "cubic_cell_edge",
    "madelung_alpha",
    "madelung_correction_scalar",
    "apply_madelung_correction",
    "apply_madelung_correction_per_k",
    "madelung_energy_correction",
    "madelung_energy_correction_for_lat",
    # exxdiv='ewald' K-matrix shift (HF exchange G=0 self-image fix on
    # finite k-meshes; McClain, Sun, Chan, Berkelbach JCTC 13, 1209 (2017)).
    "madelung_constant_for_cell",
    "apply_exxdiv_ewald_to_K",
    "exxdiv_ewald_energy_shift",
    "MADELUNG_CITATION_KEYS",
    "method_citations",
]


# Citation keys for the monopole (Makov-Payne) finite-box correction, resolved
# against python/vibeqc/output/citations/database.toml. The correction
# (`a_M Q^2/(2L)`, `makov_payne_coefficient_cubic` / `madelung_energy_correction`)
# is an offline library capability a caller applies to a periodic total energy,
# not a `run_periodic_job` feature: every Coulomb method reachable through the
# runner zeroes it (EWALD_3D and SLAB return 0.0; the Γ-only DIRECT_TRUNCATED
# user route `FFT_POISSON` was retired in v0.13.0), and the default GDF path
# applies the *exchange-divergence* exxdiv='ewald' Madelung constant instead
# (cited separately via mcclain_berkelbach). So there is no `[routes.*]` row;
# the module names its provenance the way the BDIIS optimiser and the periodic
# stress tensor do (CLAUDE.md § 8), for callers of `madelung_energy_correction`
# / `makov_payne_coefficient_cubic` / `apply_madelung_correction`. The
# resolve-in-database guard lives in tests/test_madelung_citation.py.
MADELUNG_CITATION_KEYS: tuple[str, ...] = ("makov_payne_1995",)


def method_citations() -> tuple[str, ...]:
    """Citation-database keys for the Makov-Payne finite-box correction."""
    return MADELUNG_CITATION_KEYS


def madelung_energy_correction_for_lat(
    D: np.ndarray,
    S: np.ndarray,
    system: PeriodicSystem,
    lat_opts,
) -> float:
    """Convenience wrapper around :func:`madelung_energy_correction`
    that infers ``nuclear_uses_ewald`` from
    ``lat_opts.coulomb_method``. Used by all 8 Python Ewald-3D SCF
    drivers so each call site is a single one-line invocation.

    .. note:: **v0.7.0 deprecation in progress.**

        The original v0.6.1 Madelung correction compensated for the
        gauge mismatch between the bare-libint V_ne (``compute_nuclear_lattice``,
        no G = 0 omission) and the FFT-Poisson J build (G = 0 dropped).
        v0.7.0 ships a V_ne dispatch
        (``vibeqc.periodic_v_ne.compute_nuclear_lattice_dispatch``) that
        switches V_ne to ``compute_nuclear_lattice_ewald`` whenever
        ``coulomb_method == EWALD_3D``, putting V_ne and J in the same
        gauge. The correction is then **double-counted** and produces
        spuriously positive iter-1 totals on tight ionic crystals
        (LiH conventional rocksalt: +94 Ha over-correction
        empirically demonstrated in the v0.7 work).

        For EWALD_3D this function therefore returns ``0.0``.
        DIRECT_TRUNCATED users of the function still get the original
        Q_e^2-only correction since their V_ne path remains the bare
        libint sum.
    """
    if lat_opts.coulomb_method == CoulombMethod.EWALD_3D:
        # v0.7.0: V_ne is dispatched to the Ewald path which already
        # handles the G = 0 gauge. No further correction is needed.
        return 0.0
    return madelung_energy_correction(
        D, S, system, nuclear_uses_ewald=False,
    )


def cell_electron_charge(D: np.ndarray, S: np.ndarray) -> float:
    """Return ``Q_e = Re tr(D . S)`` -- the cell electron count.

    Accepts real Γ-point matrices or Hermitian complex S (in which case
    the imaginary part of the trace is numerical noise and is dropped).
    For the multi-k driver, pass ``D_real.blocks[g=0]`` (the home-cell
    real-space density block) and ``S_real.blocks[g=0]``, OR the
    Γ-point Bloch sums ``D(k=0)`` and ``S(k=0)`` -- both give the same
    ``Q_e`` for an insulator at Γ.
    """
    D_arr = np.asarray(D)
    S_arr = np.asarray(S)
    tr = np.trace(D_arr @ S_arr)
    return float(np.real(tr))


def cell_nuclear_charge(system: PeriodicSystem) -> float:
    """Return ``Q_n = S_I Z_I`` for the unit cell."""
    return float(sum(atom.Z for atom in system.unit_cell))


def cell_net_charge(
    D: np.ndarray, S: np.ndarray, system: PeriodicSystem,
) -> float:
    """Net cell charge ``Q_e - Q_n``. Zero for a neutral crystal to
    within SCF convergence of ``tr(D S)``."""
    return cell_electron_charge(D, S) - cell_nuclear_charge(system)


def cubic_cell_edge(system: PeriodicSystem) -> float:
    """Return ``L = V_cell^(1/3)`` -- the characteristic cubic edge.

    Exact for cubic cells; for general orthorhombic cells it's the
    geometric mean of the three edge lengths and the Madelung
    correction acquires aspect-ratio-dependent corrections on top of
    what :func:`makov_payne_coefficient_cubic` predicts. For strongly
    non-cubic cells treat the returned a with care -- validate with an
    w-invariance sweep before trusting it quantitatively.
    """
    lat = np.asarray(system.lattice, dtype=float)
    return float(abs(np.linalg.det(lat))) ** (1.0 / 3.0)


def madelung_alpha(Q: float, L: float) -> float:
    """Return the Makov-Payne scalar ``a = -a_M . Q / L`` for a cubic
    cell. Thin wrapper around
    :func:`vibeqc.makov_payne_coefficient_cubic` with a name that
    mirrors the periodic-HF literature's terminology.
    """
    return makov_payne_coefficient_cubic(float(Q), float(L))


def madelung_correction_scalar(
    D: np.ndarray,
    S: np.ndarray,
    system: PeriodicSystem,
) -> float:
    """Return ``a_net = a_e - a_n`` for the current (D, S, system).

    Zero (to SCF precision) for a neutral crystal, demonstrating the
    Madelung cancellation. Non-zero for charged cells; applying
    ``a_net . S`` to the Fock matrix restores the isolated-molecule
    limit.
    """
    Q_e = cell_electron_charge(D, S)
    Q_n = cell_nuclear_charge(system)
    L = cubic_cell_edge(system)
    alpha_e = madelung_alpha(Q_e, L)
    alpha_n = madelung_alpha(Q_n, L)
    return alpha_e - alpha_n


def apply_madelung_correction(
    F: np.ndarray, S: np.ndarray, alpha: float,
) -> np.ndarray:
    """Return ``F + a . S``. Works for real or complex matrices; the
    output dtype follows the inputs. ``a`` is always a real scalar.

    When ``a`` is the :func:`madelung_correction_scalar` output for a
    neutral crystal, the correction is numerically zero. For a charged
    cell with ``a != 0`` this is the scalar-times-overlap shift that
    un-does the G = 0 gauge of the FFT Poisson solver.
    """
    return np.asarray(F) + float(alpha) * np.asarray(S)


def madelung_energy_correction(
    D: np.ndarray,
    S: np.ndarray,
    system: PeriodicSystem,
    *,
    nuclear_uses_ewald: bool = False,
) -> float:
    """Madelung leak fix for the SCF total energy (v0.6.1).

    Returns the additive correction to the SCF total energy that
    cancels the spurious Madelung-like leak from the Ewald-3D split:

      - Always present: the Hartree J build (``build_j_ewald_3d``)
        pins the G=0 Fourier mode of the electronic potential to
        zero, leaking ``-a_M Q_e^2/(2L)`` into the SCF energy via the
        ``(1/2) tr(D.J)`` term.

      - Conditional: when ``nuclear_uses_ewald=True``, the nuclear
        repulsion path additionally carries the matching
        ``-a_M Q_n^2/(2L)`` Madelung self-image. With
        ``DIRECT_TRUNCATED`` nuclear (the default), the nuclear
        side is the bare lattice sum and contributes no leak.

    Total correction:

      - Default (DIRECT_TRUNCATED nuclear): ``+a_M Q_e^2 / (2L)``.
      - With EWALD_3D nuclear: ``+a_M (Q_n^2 + Q_e^2) / (2L)``.

    Where ``a_M ≈ 2.837`` is the simple-cubic Madelung constant,
    ``Q_n = S_I Z_I`` the unit-cell nuclear charge,
    ``Q_e = tr(D . S)`` the cell electron count, and ``L = V^(1/3)``.

    Validated on H2 / He in 30-bohr boxes: drops the diff vs
    molecular reference from ~189 mHa (DIRECT nuclear path) or
    ~375 mHa (EWALD nuclear path) to ~3 mHa each (the residual is
    finite-density-extent vs point-charge in the formula).

    Used by all 8 Python Ewald-3D SCF drivers (RHF / UHF / RKS / UKS,
    Γ-only + multi-k). Cubic-formula approximation; for aspect-ratio
    orthorhombic cells the Madelung constant picks up small
    corrections -- quantify via w-invariance check (which tracks the
    same physics).
    """
    Q_n = cell_nuclear_charge(system)
    Q_e = cell_electron_charge(D, S)
    L = cubic_cell_edge(system)
    from .ewald_composed import _SIMPLE_CUBIC_MADELUNG
    nuclear_term = (Q_n * Q_n) if nuclear_uses_ewald else 0.0
    return _SIMPLE_CUBIC_MADELUNG * (nuclear_term + Q_e * Q_e) / (2.0 * L)


def apply_madelung_correction_per_k(
    F_k_list: Sequence[np.ndarray],
    S_k_list: Sequence[np.ndarray],
    alpha: float,
) -> List[np.ndarray]:
    """Per-k variant of :func:`apply_madelung_correction` -- applies the
    same scalar ``a`` to every ``(F(k), S(k))`` pair.

    The scalar is k-independent because the a . S correction is a
    real-space shift of the potential whose Bloch sum at any k is
    simply ``a . S(k)``.
    """
    if len(F_k_list) != len(S_k_list):
        raise ValueError(
            f"F_k_list and S_k_list must match length; got "
            f"{len(F_k_list)} vs {len(S_k_list)}"
        )
    return [
        apply_madelung_correction(F_k, S_k, alpha)
        for F_k, S_k in zip(F_k_list, S_k_list)
    ]


# ============================================================
# exxdiv='ewald' K-matrix shift
# ============================================================
#
# The G = 0 self-image divergence in the HF exchange kernel on a finite
# k-mesh produces an O(1/N_k) error that doesn't cancel as N_k -> inf for
# a periodic insulator. PySCF (default ``exxdiv='ewald'``) restores
# correct behaviour by adding
#
#     K(k) -> K(k) + ξ . S(k) . D(k) . S(k)
#
# per k-point, where ξ is the cell Ewald-Madelung constant a_M / L
# (cubic). The shift is equivalent to using a probe-charge G = 0 limit
# of the Coulomb kernel rather than the naive zero-Fourier-mode value;
# derivation in McClain, Sun, Chan, Berkelbach,
# *J. Chem. Theory Comput.* **13**, 1209 (2017),
# DOI 10.1021/acs.jctc.6b01184.

def madelung_constant_for_cell(
    system: PeriodicSystem,
    *,
    precision: float = 1e-12,
    eta: Optional[float] = None,
) -> float:
    """Return the Ewald-Madelung constant ``ξ`` for the cell.

    Computed via a proper Ewald sum of the self-energy of a unit point
    charge in the periodic cell with a neutralising background:

    ::

        ξ = -[ -2η/√pi
              + S_{T!=0} erfc(η.|T|)/|T|
              + (4pi/V) S_{G!=0} exp(-G^2/(4η^2))/G^2
              - pi/(η^2.V) ]

    The bracketed expression is the standard Ewald self-energy
    ``v_self`` (negative for a positive point charge bound by its
    neutralising background); ``ξ = -v_self`` is reported as a
    positive magnitude, matching PySCF's
    ``pbc.tools.madelung(cell, kpts)`` convention.

    Geometry-correct for any Bravais lattice. Previously this routine
    used the cubic Wigner shortcut ``a_M ≈ 2.837297, L = V^(1/3)``,
    which gave the right answer for conventional cubic cells but was
    off by ~2% for primitive FCC cells (e.g., LiH primitive: 0.5836
    vs proper 0.5941, propagating to ~18 mHa SCF error via the
    K-shift). The Ewald-sum implementation here is η-invariant at
    convergence (verified on cubic and FCC primitive cells; bit-exact
    PySCF to 6 sig figs).

    Parameters
    ----------
    system
        Periodic system (3D Bravais lattice).
    precision
        Target precision for the real- and reciprocal-space sum
        truncation (default ``1e-12``).
    eta
        Ewald damping parameter (bohr⁻¹). If ``None``, default to
        ``1 / |a_smallest|`` which gives well-balanced real- and
        reciprocal-space convergence on typical cells. The answer is
        η-invariant at convergence.

    References
    ----------
    McClain, Sun, Chan, Berkelbach, *J. Chem. Theory Comput.* **13**,
    1209 (2017), DOI 10.1021/acs.jctc.6b01184; PySCF
    ``pyscf.pbc.tools.pbc.madelung`` (read for understanding; no code
    copied per CLAUDE.md Sec.10).
    """
    from scipy.special import erfc as _erfc

    # The Ewald self-energy + neutralising-background form below is a 3D
    # electrostatic model (both the -pi/(η^2.V) background term and the
    # 4pi/G^2 reciprocal kernel are 3D). For a 1D/2D cell it would silently
    # return a meaningless (negative) 3D number that then propagates into
    # the exxdiv='ewald' K-shift. Refuse rather than mislead -- mirrors the
    # dim!=3 guard in run_pbc_gdf_rhf and the C++
    # compute_nuclear_lattice_ewald. (Audit 2026-05-30.)
    if int(getattr(system, "dim", 3)) != 3:
        raise NotImplementedError(
            "madelung_constant_for_cell: only dim=3 is implemented (the "
            "Ewald self-energy uses a 3D neutralising background); got "
            f"system.dim={int(system.dim)}. A true 1D/2D Madelung/exxdiv "
            "treatment is not implemented -- use exxdiv='none' for low-dim "
            "cells."
        )

    a = np.asarray(system.lattice, dtype=float)
    if a.shape != (3, 3):
        raise RuntimeError(
            f"madelung_constant_for_cell: lattice must be 3x3, got {a.shape}"
        )
    V = float(abs(np.linalg.det(a)))
    if V <= 0:
        raise RuntimeError(
            f"madelung_constant_for_cell: degenerate lattice (V = {V})"
        )
    # PeriodicSystem stores direct-lattice vectors as the columns of ``a``.
    # The reciprocal vectors are therefore the columns of 2pi.A^{-T}.
    b = 2.0 * np.pi * np.linalg.inv(a).T
    a_norms = np.linalg.norm(a, axis=0)
    b_norms = np.linalg.norm(b, axis=0)

    if eta is None:
        eta = 1.0 / float(a_norms.min())

    # Real-space cutoff: erfc(η.rcut) < precision . |a_min| -> rcut sized so
    # that erfc(η.r)/r < precision at r = rcut.
    rcut = float(np.sqrt(-np.log(precision * float(a_norms.min()))) / eta) + 10.0
    # Reciprocal-space cutoff: exp(-G^2/(4η^2)) < precision at G = gcut.
    gcut = 2.0 * float(eta) * float(np.sqrt(-np.log(precision)))

    # Real-space sum.
    # If T=A.n lies inside the real-space sphere, duality gives
    # |n_i|=|b_i.T|/(2pi) <= rcut.|b_i|/(2pi).  These component-wise
    # bounds remain complete for skew cells without relying on the shortest
    # direct-vector norm (which is not a coefficient bound).
    n_max_r = np.ceil(rcut * b_norms / (2.0 * np.pi)).astype(int)
    idx_r = [np.arange(-n, n + 1) for n in n_max_r]
    n1, n2, n3 = np.meshgrid(*idx_r, indexing="ij")
    integers_r = np.stack([n1.ravel(), n2.ravel(), n3.ravel()], axis=-1).astype(float)
    T = integers_r @ a.T
    T_norms = np.linalg.norm(T, axis=1)
    keep_r = (T_norms > 1e-12) & (T_norms <= rcut)
    real_term = float(np.sum(_erfc(eta * T_norms[keep_r]) / T_norms[keep_r]))

    # Reciprocal-space sum.
    # Likewise, for G=B.m inside the reciprocal-space sphere,
    # |m_i|=|a_i.G|/(2pi) <= gcut.|a_i|/(2pi).
    n_max_g = np.ceil(gcut * a_norms / (2.0 * np.pi)).astype(int)
    idx_g = [np.arange(-n, n + 1) for n in n_max_g]
    n1, n2, n3 = np.meshgrid(*idx_g, indexing="ij")
    integers_g = np.stack([n1.ravel(), n2.ravel(), n3.ravel()], axis=-1).astype(float)
    G = integers_g @ b.T
    G2 = (G**2).sum(axis=1)
    keep_g = (G2 > 1e-12) & (G2 <= gcut**2)
    recip_term = float(
        (4.0 * np.pi / V) * np.sum(np.exp(-G2[keep_g] / (4.0 * eta**2)) / G2[keep_g])
    )

    # Self + background.
    self_term = -2.0 * float(eta) / float(np.sqrt(np.pi))
    bg_term = -float(np.pi) / (float(eta) ** 2 * V)

    v_self = real_term + recip_term + self_term + bg_term
    # Return positive magnitude (vibe-qc / PySCF convention).
    return -v_self


def apply_exxdiv_ewald_to_K(
    K_k_list: Sequence[np.ndarray],
    S_k_list: Sequence[np.ndarray],
    D_k_list: Sequence[np.ndarray],
    madelung: float,
) -> List[np.ndarray]:
    """Apply the ``exxdiv='ewald'`` Madelung shift per k-point to K.

    For each k, returns ``K(k) + ξ . S(k) . D(k) . S(k)`` where
    ``ξ = madelung`` is the cell Ewald-Madelung constant
    (:func:`madelung_constant_for_cell`).

    The shift handles the G = 0 self-image divergence of HF exchange
    on a finite k-mesh; it is the standard PySCF / Quantum ESPRESSO
    HF-exchange convergence accelerator. Without it, K(k) has an
    O(1/N_k) bias that doesn't vanish as N_k -> inf.

    Drivers compute K either with the convention ``D_k = C_occ . C_occ.T``
    (HF/RHF) or ``D_k = 2 . C_occ . C_occ.T`` (closed-shell
    pre-multiplied). This helper does NOT touch D_k internally -- it
    just applies ``S.D.S`` to whatever ``D_k_list`` it is given.
    Caller is responsible for matching the convention K is built with.

    Notes
    -----
    The shift is unitless in the spectral sense: ξ has units of
    1/bohr, S has units of 1, K has units of Hartree -- so S.D.S has
    units of 1 and ξ.S.D.S has units of 1/bohr. That doesn't match K's
    Hartree. The resolution is that ξ is properly a
    *Hartree-per-electron* charge-screening constant (in atomic units,
    1/bohr is energy/charge for q in units of e), so ξ.S.D.S is in
    Hartree. The legacy literature carries this convention silently.
    """
    if not (len(K_k_list) == len(S_k_list) == len(D_k_list)):
        raise ValueError(
            "apply_exxdiv_ewald_to_K: K/S/D lists must have matching "
            f"length; got K={len(K_k_list)}, S={len(S_k_list)}, "
            f"D={len(D_k_list)}"
        )
    xi = float(madelung)
    out = []
    for K, S, D in zip(K_k_list, S_k_list, D_k_list):
        K_arr = np.asarray(K)
        S_arr = np.asarray(S)
        D_arr = np.asarray(D)
        shift = S_arr @ D_arr @ S_arr
        # Match output dtype to the input K so callers don't pick up
        # spurious imaginary parts at Γ when D / S are real-only.
        if np.iscomplexobj(K_arr) or np.iscomplexobj(D_arr) or np.iscomplexobj(S_arr):
            out.append(K_arr + xi * shift)
        else:
            out.append(K_arr + xi * np.real(shift))
    return out


def exxdiv_ewald_energy_shift(
    D_k_list: Sequence[np.ndarray],
    S_k_list: Sequence[np.ndarray],
    madelung: float,
    *,
    hf_exchange_fraction: float = 1.0,
    weights: Optional[Sequence[float]] = None,
) -> float:
    """Energy contribution from the exxdiv='ewald' K-matrix shift.

    The shift ``K(k) += ξ . S.D.S`` modifies the closed-shell HF
    exchange energy by

        ΔE_xx = -1/4 . a_HF . ξ . S_k w_k . tr(D(k) . S(k) . D(k) . S(k))

    (sign + factor match the closed-shell SCF convention
    ``E_HF_xx = -1/4 a_HF S_k w_k tr(D K)`` where D = 2.D_occ is the
    spin-summed density and K is built from D under the same
    convention). This is the additive correction to the total energy
    when transitioning from exxdiv='none' to exxdiv='ewald'.

    Parameters
    ----------
    D_k_list, S_k_list
        Per-k density / overlap matrices. Must have matching length.
    madelung
        Cell Ewald-Madelung constant (:func:`madelung_constant_for_cell`).
    hf_exchange_fraction
        ``a_HF`` -- fraction of HF exchange in the functional (1.0 for
        pure HF; e.g. 0.20 for B3LYP). Default 1.0 (pure HF).
    weights
        Per-k weights (Brillouin-zone Monkhorst-Pack weights). If
        ``None``, uniform ``1/N_k``.

    Returns
    -------
    float
        The energy shift (Hartree). Add to the total energy when
        switching to exxdiv='ewald'.
    """
    if len(D_k_list) != len(S_k_list):
        raise ValueError(
            f"exxdiv_ewald_energy_shift: D and S lists must match "
            f"length; got D={len(D_k_list)}, S={len(S_k_list)}"
        )
    nk = len(D_k_list)
    if weights is None:
        wk = np.full(nk, 1.0 / nk)
    else:
        wk = np.asarray(weights, dtype=float)
        if wk.shape != (nk,):
            raise ValueError(
                f"exxdiv_ewald_energy_shift: weights must have length "
                f"{nk}; got shape {wk.shape}"
            )
    total = 0.0
    for w, D, S in zip(wk, D_k_list, S_k_list):
        D_arr = np.asarray(D)
        S_arr = np.asarray(S)
        # tr(D.S.D.S) -- real for Hermitian D,S whose product is itself
        # Hermitian. Imaginary part is numerical noise; drop it.
        sds = S_arr @ D_arr @ S_arr
        tr = np.trace(D_arr @ sds)
        total += float(w) * float(np.real(tr))
    return -0.25 * float(hf_exchange_fraction) * float(madelung) * total
