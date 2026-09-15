"""BIPOLE multipole far-pair Fock builder.

Replaces far direct ERIs with truncated multipole expansions in the
periodic Fock build. For each cell pair (g, h), the Coulomb interaction
between bra-pair density at cell g and ket-pair density at cell h is
approximated using cell-level multipole moments and the multipole-
multipole interaction tensor.

This module computes the far-field J contributions via multipole
moments. The near-field J is still built via the C++ real-space builder
(build_fock_2e_real_space). The split is controlled by a geometric
cutoff radius R_bipole -- cell pairs with |R_g-R_h| <= R_bipole
use direct ERIs; those beyond use the multipole approximation.

STATUS -- RETIRED as a production candidate (G1, 2026-06-20)
----------------------------------------------------------
This branch is a **gated research artifact**: it never auto-enables and
is OFF on the production path. It is broken in general -- Ha-scale errors
on *both* dipolar / low-symmetry cells (~2.8 Ha on LiH/STO-3G, not
converging with multipole order) and, on current ``main``, even neutral
dipole-free cells (H₂ ~0.5 Ha off). The historical "dipole-free cells are
accurate" claim reflected an older code state and no longer holds.

The bare interaction tensor is *not* at fault -- ``multipole_pair_energy``
/ ``multipole_interaction_tensor`` are unit-test-correct vs exact Coulomb
(``tests/test_bipole_multipole.py``). The failure is in the periodic
far-field *composition* under the Ewald-J split:

* **(A) charge-incomplete moments.** At Γ the driver keeps only the home
  density block (``_zero_cross_cell_density``), so the cell-moment
  contraction ``S_g P(g).M(g)`` collapses to ``Tr[P(0).M(0)] != N_e``
  instead of the Γ-fold ``Tr[P(0).S_g M(g)]`` -- a spurious per-cell net
  charge (≈ -0.63 e on LiH).
* **(B) additive-vs-replacement + conditional convergence.**
  ``build_j_far_field_multipole`` returns the far-field *potential* (an
  additive Fock term), but ``apply_multipole_far_field`` *replaces* every
  cell's ``J_SR`` block with it (all cells, incl. the home cell, are
  flagged "far") and drops the reciprocal ``J_LR`` -- a bare-Coulomb
  real-space lattice sum that is only conditionally convergent for a
  dipolar 3D cell (the Madelung/Ewald problem; CLAUDE.md Sec.7).
* **(C) J_SR double-count at the call site.** In ``pbc_bipole`` the driver
  does ``f2e_real = F_J_SR_lat`` (an alias) and then overwrites the blocks
  with ``j_sr - 1/2K + F_LR`` *before* passing ``F_J_SR_lat.blocks`` as the
  ``J_SR_blocks`` argument to ``apply_multipole_far_field``. Its "near"
  branch then adds ``F_LR`` and ``K`` a second time, so even a neutral,
  dipole-free cell with *no* far pairs is wrong (this is why H₂ is ~0.5 Ha
  off regardless of geometry). Confined to the enabled-multipole path; the
  default (non-multipole) production path is unaffected.

The exact Ewald-J split already handles the far field optimally in
reciprocal space, so there is neither an accuracy nor a speed case for
this branch on the production path. See ``docs/bipole_status.md`` (G1) and
the gitignored ``references/g1_multipole_probe*.py`` /
``references/g1_diag*.py`` for the reproduction + layer-by-layer
diagnosis.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ._cart_to_sph import cartesian_to_spherical_matrix
from ._vibeqc_core import (
    BasisSet,
    LatticeMatrixSet,
    LatticeMultipoleSet,
    LatticeSumOptions,
    PeriodicSystem,
    compute_multipole_moments_lattice,
)
from .bipole_cell_moments import (
    compute_cell_multipole_moments,
)
from .bipole_multipole import (
    multipole_interaction_tensor,
    n_components,
)

__all__ = [
    "BipoleFarFieldJ",
    "BipoleMultipoleConfig",
    "MultipoleMomentCache",
    "build_j_far_field_multipole",
    "estimate_bipole_radius",
    "resolve_multipole_config",
    "apply_multipole_far_field",
]


def _shell_pair_exponent_ratio(alpha_1: float, alpha_2: float) -> float:
    """Bra-pair exponent-ratio factor ``F34 = a_1.a_2 / (a_1 + a_2)``.

    Controls the width of the bra-pair Gaussian -- the product
    overlap S_{12}(g) ∝ exp(-F34 . |R_1 - R_2 - g|^2). Larger F34 ->
    tighter pair (faster overlap decay -> more aggressive multipole
    truncation possible).

    Private copy for this retired, gated module: the live dispatch
    path (``bipole_dispatch.py``) adopted electrostatic-penetration
    terminology in 63cdc0fd0; restored here unchanged
    since this module still needs it and is not part of that refactor.
    """
    if alpha_1 <= 0.0 or alpha_2 <= 0.0:
        raise ValueError(f"exponents must be positive; got {alpha_1}, {alpha_2}")
    a1 = float(alpha_1)
    a2 = float(alpha_2)
    return a1 * a2 / (a1 + a2)


@dataclass
class _RetiredBipoleDispatchParams:
    """Retired prototype BIPOLE dispatch parameters.

    Private copy for this retired, gated module: the live dispatch
    path (``bipole_dispatch.py``) replaced this class with
    ``PenetrationDispatchParameters`` in the penetration-terminology
    rename (63cdc0fd0); restored here unchanged since
    this module still needs the old ``from_cell_volume`` construction
    and is not part of that refactor.

    Attributes
    ----------
    IDICOU : int
        Retired prototype maximum multipole order (default 4).
        Order 0 = monopole only, 4 = up through hexadecapole.
    ADICOU : float
        Retired prototype dispatch slope (default 100). Larger values give
        more aggressive multipole truncation (more quartets dispatched
        to lower IDIPC).
    BILBO : float
        Length-scale parameter, set to ``(1 / V_cell)^(1/3)`` for 3D.
        Used to non-dimensionalise the bra-ket distance squared.
    CCCFAJ : float
        Retired prototype per-system scaling factor. Default 1.0.
    schwarz_threshold : float
        Prototype bra-pair overlap bound below which work is dropped.
        This has no documented one-to-one TOLINTEG mapping.
    """

    IDICOU: int = 4
    ADICOU: float = 100.0
    BILBO: float = 1.0
    CCCFAJ: float = 1.0
    schwarz_threshold: float = 1e-7

    @classmethod
    def from_cell_volume(
        cls,
        V_cell_bohr3: float,
        IDICOU: int = 4,
        ADICOU: float = 100.0,
        CCCFAJ: float = 1.0,
        schwarz_threshold: float = 1e-7,
    ) -> "_RetiredBipoleDispatchParams":
        """Construct dispatch params with ``BILBO = (1/V)^(1/3)``."""
        if V_cell_bohr3 <= 0.0:
            raise ValueError(f"V_cell must be positive; got {V_cell_bohr3}")
        BILBO = float(V_cell_bohr3) ** (-1.0 / 3.0)
        return cls(
            IDICOU=int(IDICOU),
            ADICOU=float(ADICOU),
            BILBO=float(BILBO),
            CCCFAJ=float(CCCFAJ),
            schwarz_threshold=float(schwarz_threshold),
        )


@dataclass
class MultipoleMomentCache:
    """Pre-computed multipole moment data, reusable across SCF iterations.

    The libint ``compute_multipole_moments_lattice`` call is the
    bottleneck of the far-field J build (O(N_cells . N_shells^2)).
    This cache stores the result once per geometry and reuses it
    across density updates.

    Attributes
    ----------
    M_lat : LatticeMultipoleSet
        Raw per-cell, per-component shell-pair multipole moments.
    M_per_cell_comp : list of list of ndarray
        Pre-extracted per-cell per-component (nbf, nbf) blocks.
    cells : list
        Lattice cells matching the moment data.
    L_max : int
        Maximum multipole order.
    """

    M_lat: LatticeMultipoleSet
    M_per_cell_comp: list
    cells: list
    L_max: int

    @classmethod
    def build(
        cls,
        basis: BasisSet,
        system: PeriodicSystem,
        lat_opts: LatticeSumOptions,
        L_max: int = 2,
    ) -> "MultipoleMomentCache":
        """Build the cache from scratch (call once per geometry)."""
        M_lat = compute_multipole_moments_lattice(
            basis,
            system,
            lat_opts,
            L_max,
            (0.0, 0.0, 0.0),
        )
        n_comp = len(M_lat.blocks[0]) if M_lat.blocks else 0
        n_cells = len(M_lat.cells)
        M_per_cell_comp = [
            [np.asarray(M_lat.blocks[c][comp], dtype=float) for comp in range(n_comp)]
            for c in range(n_cells)
        ]
        return cls(
            M_lat=M_lat,
            M_per_cell_comp=M_per_cell_comp,
            cells=list(M_lat.cells),
            L_max=L_max,
        )


@dataclass
class BipoleFarFieldJ:
    """Result of build_j_far_field_multipole."""

    J_blocks: List[np.ndarray]
    cells: list
    e_j_far: float
    n_cell_pairs: int
    L_max: int
    R_bipole: float


def estimate_bipole_radius(
    system: PeriodicSystem,
    basis: BasisSet,
    *,
    L_max: int = 2,
    IDIPCOU: int = 4,
    ADIPCOU: float = 100.0,
) -> float:
    """Estimate a reasonable R_bipole cutoff radius.

    Uses the IDIPC dispatch formula in reverse: find the distance D
    at which the most compact shell pair (largest F34) would yield
    IDIPC = 1 (borderline near/far).
    """
    max_F34 = 0.0
    for shell_a in basis.shells():
        alpha_a = min(shell_a.exponents) if shell_a.exponents else 1.0
        for shell_b in basis.shells():
            alpha_b = min(shell_b.exponents) if shell_b.exponents else 1.0
            F34 = _shell_pair_exponent_ratio(alpha_a, alpha_b)
            if F34 > max_F34:
                max_F34 = F34

    if max_F34 <= 0.0:
        return 10.0

    V_cell = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
    params = _RetiredBipoleDispatchParams.from_cell_volume(
        V_cell,
        IDICOU=IDIPCOU,
        ADICOU=ADIPCOU,
    )

    target_VDID = max(0.0, (params.IDICOU - 1.0) / params.ADICOU)
    inner = params.BILBO - target_VDID / params.CCCFAJ
    if inner <= 0.0:
        return 3.0
    D = float(np.sqrt(inner / max_F34))
    return max(3.0, min(D, 30.0))


def build_j_far_field_multipole(
    P_real: LatticeMatrixSet,
    basis: BasisSet,
    system: PeriodicSystem,
    lat_opts: LatticeSumOptions,
    *,
    L_max: int = 2,
    R_bipole: Optional[float] = None,
    omega: float = 0.0,
    cache: Optional[MultipoleMomentCache] = None,
) -> BipoleFarFieldJ:
    """Build the far-field Hartree J via multipole-multipole interactions.

    For cell pairs with |R_g-R_h| > R_bipole, the Coulomb interaction
    is approximated using cell-level multipole moments up to L_max.
    Near cell pairs are skipped (handled by the direct ERI builder).

    When ``omega > 0``, the interaction tensor is multiplied by
    ``erfc(omega * R)`` -- this gives the short-range Ewald-split
    piece, consistent with the erfc-screened J^SR in the BIPOLE
    Ewald-J split. For ``omega = 0``, the bare Coulomb is used.
    """
    if system.dim != 3:
        raise ValueError("build_j_far_field_multipole requires dim=3")
    if P_real.nbf != basis.nbasis:
        raise ValueError(f"P_real.nbf={P_real.nbf} != basis.nbasis={basis.nbasis}")

    nbf = basis.nbasis
    cells = list(P_real.cells)
    n_cells = len(cells)
    R_g_arr = np.array(
        [np.asarray(c.r_cart, dtype=float) for c in cells],
        dtype=float,
    )

    if R_bipole is None:
        R_bipole = estimate_bipole_radius(system, basis, L_max=L_max)
    R_bipole_sq = R_bipole * R_bipole

    # ---- shell-pair multipole moments (reusable across SCF iters) ---
    if cache is not None:
        M_lat = cache.M_lat
        M_per_cell_comp = cache.M_per_cell_comp
        if cache.L_max != L_max:
            raise ValueError(f"Cache L_max={cache.L_max} != requested L_max={L_max}")
    else:
        M_lat = compute_multipole_moments_lattice(
            basis,
            system,
            lat_opts,
            L_max,
            (0.0, 0.0, 0.0),
        )
        n_comp_full = len(M_lat.blocks[0]) if M_lat.blocks else 0
        M_per_cell_comp = [
            [
                np.asarray(M_lat.blocks[c][comp], dtype=float)
                for comp in range(n_comp_full)
            ]
            for c in range(len(M_lat.cells))
        ]

    # ---- cell-level multipole moments from density ------------------
    cell_moments = compute_cell_multipole_moments(P_real, M_lat, system=system)
    M_cell = np.asarray(cell_moments.moments, dtype=float)
    n_comp = len(M_cell)

    # ---- cell-pair multipole interactions ---------------------------
    J_blocks = [np.zeros((nbf, nbf), dtype=float) for _ in range(n_cells)]
    e_j_far = 0.0
    n_cell_pairs = 0

    # For L_max >= 2, convert cell moments to spherical convention
    # (Cartesian: 10 comps for L=2, Spherical: 9 comps for L=2).
    if L_max >= 2:
        C = cartesian_to_spherical_matrix(L_max)
        CT = C.T
        M_cell_sph = C @ M_cell
        n_sph = M_cell_sph.shape[0]
    else:
        C = None
        CT = None
        M_cell_sph = M_cell
        n_sph = len(M_cell)

    for g in range(n_cells):
        R_g = R_g_arr[g]
        for h in range(g, n_cells):
            dR = R_g_arr[h] - R_g
            dR_sq = float(np.dot(dR, dR))

            if dR_sq <= R_bipole_sq:
                continue

            n_cell_pairs += 1

            T_sph = multipole_interaction_tensor(L_max, L_max, dR)
            if omega > 0.0:
                dR_norm = float(np.sqrt(dR_sq))
                erfc_factor = float(math.erfc(omega * dR_norm))
                T_sph = T_sph * erfc_factor

            # T_M = T @ M_sph  (in spherical convention)
            T_Mh = T_sph @ M_cell_sph

            # Convert back to Cartesian for Fock contribution.
            # M_cell_use is Cartesian; T_Mh_use is Cartesian.
            M_cell_use = M_cell
            T_Mh_use = (CT @ T_Mh) if CT is not None else T_Mh

            n_use = len(M_cell_use)
            # Fock contribution for cell g:
            for comp in range(n_use):
                coeff = float(T_Mh_use[comp])
                if abs(coeff) < 1e-16:
                    continue
                J_blocks[g] += M_per_cell_comp[g][comp] * coeff

            # Fock contribution for cell h (symmetric):
            Mg_T_sph = M_cell_sph @ T_sph
            Mg_T_use = (CT @ Mg_T_sph) if CT is not None else Mg_T_sph
            if g != h:
                for comp in range(n_use):
                    coeff = float(Mg_T_use[comp])
                    if abs(coeff) < 1e-16:
                        continue
                    J_blocks[h] += M_per_cell_comp[h][comp] * coeff

    # ---- energy: 0.5 * sum_g tr[D(g) J_far(g)] --------------------
    for g in range(n_cells):
        P_g = np.asarray(P_real.blocks[g], dtype=float)
        if P_g.size == 0:
            continue
        e_j_far += 0.5 * float(np.sum(P_g * J_blocks[g]))

    return BipoleFarFieldJ(
        J_blocks=J_blocks,
        cells=cells,
        e_j_far=e_j_far,
        n_cell_pairs=n_cell_pairs,
        L_max=L_max,
        R_bipole=R_bipole,
    )


# ---------------------------------------------------------------------------
# Production integration helpers for the BIPOLE drivers
# ---------------------------------------------------------------------------


@dataclass
class BipoleMultipoleConfig:
    """Resolved multipole far-field configuration for a BIPOLE run.

    Attributes
    ----------
    enabled : bool
        Whether multipole far-field acceleration is active.
    L_max : int
        Multipole truncation order (default 2 = up to quadrupole).
    R_bipole : float
        Cell-centre cutoff radius in bohr. Cell pairs beyond this
        distance use multipole J instead of exact ERIs.
    cache : MultipoleMomentCache or None
        Pre-built moment cache (shared across SCF iterations).
    """

    enabled: bool
    L_max: int
    R_bipole: float
    cache: Optional[MultipoleMomentCache] = None


def resolve_multipole_config(
    system: PeriodicSystem,
    basis: BasisSet,
    lat_opts: LatticeSumOptions,
    *,
    user_enable: Optional[bool] = None,
    multipole_l_max: int = 2,
) -> BipoleMultipoleConfig:
    """Resolve whether and how to use the (retired) multipole far-field.

    G1 RETIRED 2026-06-20 (see the module docstring + ``docs/bipole_status.md``):
    the branch **never auto-enables** -- it is a gated research artifact that
    is correct only on dipole-free cells. There is no auto-decision path.

    * Only for 3D periodic systems (else disabled).
    * ``user_enable=None`` -> **disabled** (no auto-enable).
    * ``user_enable=False`` -> disabled.
    * ``user_enable=True`` -> enabled as a diagnostic / parity probe; emits a
      ``UserWarning`` and is NOT correct on dipolar / low-symmetry cells.
    """
    if system.dim != 3:
        return BipoleMultipoleConfig(enabled=False, L_max=0, R_bipole=0.0)

    R_mp = estimate_bipole_radius(system, basis, L_max=multipole_l_max)

    # Safety margin: the multipole expansion needs the far-field cells
    # to be well-separated from the unit cell. Require R_bipole >= L_max
    # times the largest lattice vector -- for L_max=3 this gives expansion
    # parameter (a/R) <= 1/3, ensuring ~1% octupole convergence.
    max_lat = float(
        max(
            np.linalg.norm(np.asarray(system.lattice, dtype=float)[:, d])
            for d in range(3)
        )
    )
    R_mp = max(R_mp, float(multipole_l_max) * max_lat)

    # G1 RETIRED (2026-06-20): the branch NEVER auto-enables. The bare
    # interaction tensor is unit-test-correct vs exact Coulomb
    # (tests/test_bipole_multipole.py), but the far-field *composition*
    # under the Ewald-J split is structurally wrong on dipolar /
    # low-symmetry cells -- Ha-scale errors (~2.8 Ha on LiH/STO-3G) that do
    # not converge with multipole order:
    #   (A) cell moments built from the P0-localized Γ density carry a
    #       spurious net charge -- the S_g P(g).M(g) contraction collapses
    #       to Tr[P(0).M(0)] != N_e instead of the Γ-fold Tr[P(0).S_g M(g)];
    #   (B) the far-field *potential* REPLACES (rather than adds to) J_SR
    #       for every cell while dropping the reciprocal J_LR -- a bare-
    #       Coulomb real-space lattice sum that is only conditionally
    #       convergent for a dipolar 3D cell (Madelung/Ewald, CLAUDE.md Sec.7).
    # The exact Ewald-J split already handles the far field optimally in
    # reciprocal space (no real-space far cost to accelerate, no speed
    # win), so the branch stays a gated research artifact: it activates
    # ONLY on an explicit user_enable=True, never by auto-decision.
    # See docs/bipole_status.md (G1) and references/g1_multipole_probe*.py.
    enabled = bool(user_enable) if user_enable is not None else False

    if not enabled:
        return BipoleMultipoleConfig(enabled=False, L_max=0, R_bipole=R_mp)

    warnings.warn(
        "BIPOLE multipole far-field (G1 cell-level path) is RETIRED "
        "(2026-06-20). This low-level builder is an experimental research "
        "artifact and is not reachable from a BIPOLE SCF driver. The "
        "quartet-level replacement is also unavailable pending redesign "
        "of its periodic translation domain. See docs/bipole_status.md.",
        UserWarning,
        stacklevel=2,
    )
    # G1 path is retired and must not receive L=4 (libint emultipole max is 3).
    g1_L_max = min(int(multipole_l_max), 3)
    cache = MultipoleMomentCache.build(basis, system, lat_opts, L_max=g1_L_max)
    return BipoleMultipoleConfig(
        enabled=True,
        L_max=g1_L_max,
        R_bipole=R_mp,
        cache=cache,
    )


@dataclass
class _MultipoleFarFieldResult:
    """Internal result from apply_multipole_far_field."""

    f2e_blocks: List[np.ndarray]
    e_j_multipole: float
    n_far_cells: int
    n_total_cells: int


def apply_multipole_far_field(
    density: LatticeMatrixSet,
    basis: BasisSet,
    system: PeriodicSystem,
    lat_opts: LatticeSumOptions,
    config: BipoleMultipoleConfig,
    *,
    J_SR_blocks: List[np.ndarray],
    K_blocks: Optional[List[np.ndarray]] = None,
    F_LR_blocks: Optional[List[np.ndarray]] = None,
    exchange_scale: float = 0.5,
) -> _MultipoleFarFieldResult:
    """Apply multipole far-field J replacement to a BIPOLE Fock build.

    For each cell whose centre lies beyond ``config.R_bipole`` from the
    origin, the Coulomb J is replaced with a multipole approximation.
    Exchange (K) is preserved unchanged.

    **Composition** (Ewald-J split path)::

        Near cells:  F^2e = J_SR + J_LR - a.K
        Far cells:   F^2e = J_mp - a.K      (J_mp ≈ J_SR + J_LR)

    where a = ``exchange_scale`` (0.5 for RHF, 0.5.a_HF for RKS,
    1.0 per spin for UHF/UKS).

    Parameters
    ----------
    density : LatticeMatrixSet
        Real-space density matrix.
    basis, system, lat_opts :
        Standard BIPOLE inputs.
    config : BipoleMultipoleConfig
        Resolved multipole configuration (from ``resolve_multipole_config``).
    J_SR_blocks : list of ndarray
        Short-range (erfc-screened) Coulomb J per cell.
    K_blocks : list of ndarray or None
        Full exchange K per cell. If None, exchange is omitted.
    F_LR_blocks : list of ndarray or None
        Long-range reciprocal J per cell (with V_bg.S already included).
        If None, J_LR is omitted (direct-only path).
    exchange_scale : float
        Prefactor for K: 0.5 for RHF, 0.5.a_HF for RKS, 1.0 per spin
        for UHF/UKS.

    Returns
    -------
    _MultipoleFarFieldResult
        ``f2e_blocks`` -- corrected F^2e blocks per cell.
        ``e_j_multipole`` -- far-field J energy contribution.
        ``n_far_cells`` -- number of cells replaced.
        ``n_total_cells`` -- total cell count.
    """
    n_cells = len(J_SR_blocks)

    if not config.enabled:
        blocks: List[np.ndarray] = []
        for c in range(n_cells):
            blk = np.asarray(J_SR_blocks[c], dtype=float)
            if F_LR_blocks is not None:
                blk = blk + np.asarray(F_LR_blocks[c], dtype=float)
            if K_blocks is not None and exchange_scale != 0.0:
                blk = blk - exchange_scale * np.asarray(K_blocks[c], dtype=float)
            blocks.append(blk)
        return _MultipoleFarFieldResult(
            f2e_blocks=blocks,
            e_j_multipole=0.0,
            n_far_cells=0,
            n_total_cells=n_cells,
        )

    # Build the multipole far-field J (full Coulomb, w=0).
    far_j = build_j_far_field_multipole(
        density,
        basis,
        system,
        lat_opts,
        L_max=config.L_max,
        R_bipole=config.R_bipole,
        omega=0.0,
        cache=config.cache,
    )

    blocks: List[np.ndarray] = []
    n_far = 0

    for c in range(n_cells):
        j_mp = (
            np.asarray(far_j.J_blocks[c], dtype=float)
            if c < len(far_j.J_blocks)
            else np.zeros((basis.nbasis, basis.nbasis), dtype=float)
        )
        if np.max(np.abs(j_mp)) < 1e-16:
            # Near cell: J_SR + J_LR - a.K
            blk = np.asarray(J_SR_blocks[c], dtype=float)
            if F_LR_blocks is not None:
                blk = blk + np.asarray(F_LR_blocks[c], dtype=float)
            if K_blocks is not None and exchange_scale != 0.0:
                blk = blk - exchange_scale * np.asarray(K_blocks[c], dtype=float)
            blocks.append(blk)
        else:
            # Far cell: J_mp - a.K  (multipole J subsumes both J_SR and J_LR)
            blk = np.asarray(j_mp, dtype=float)
            if K_blocks is not None and exchange_scale != 0.0:
                blk = blk - exchange_scale * np.asarray(K_blocks[c], dtype=float)
            blocks.append(blk)
            n_far += 1

    return _MultipoleFarFieldResult(
        f2e_blocks=blocks,
        e_j_multipole=far_j.e_j_far,
        n_far_cells=n_far,
        n_total_cells=n_cells,
    )
