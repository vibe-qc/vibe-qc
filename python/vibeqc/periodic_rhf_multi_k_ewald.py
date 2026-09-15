"""Phase 12e-c-4c-iii-b: multi-k closed-shell RHF SCF driver with
composed EWALD_3D Coulomb dispatch.

Wraps :func:`vibeqc.build_periodic_fock_ewald3d_k` (Phase
12e-c-4c-iii-a) in a proper k-sampled SCF loop:

    1. Build real-space one-electron integrals S_lat, T_lat, V_lat at
       the requested ``lattice_opts`` cutoff; Bloch-sum to S(k),
       Hcore(k) at every k-point in ``kmesh.kpoints``.
    2. Canonical-orthogonalize each S(k) independently (per-k
       linear-dependence threshold).
    3. Initial guess: diagonalize Hcore(k); assemble D_real via
       ``real_space_density_from_kpoints`` using the k-mesh's
       weights.
    4. SCF iteration:
       a. ``F_k_list = build_periodic_fock_ewald3d_k(..., D_real,
          Hcore_k=Hcore_k_list, ...)``
       b. Per-cell electronic energy
          ``E_elec = S_k w_k . 1/2 Re tr[D(k) (Hcore(k) + F(k))]``
          with ``D(k) = 2 . C_occ(k) . C_occ(k)^+``.
       c. Convergence on ``|ΔE| < conv_tol_energy`` AND
          ``S_k w_k ‖F(k) D(k) S(k) - S(k) D(k) F(k)‖_F <
          conv_tol_grad``.
       d. Diagonalize each F(k) in its per-k canonical-orthogonalized
          basis -> C(k), e(k).
       e. Rebuild D_real via inverse-Bloch fold, optionally damped
          against the previous D_real block-by-block.

    5. Add the Ewald per-cell nuclear repulsion to produce the total
       energy.

**Occupation.** By default this driver uses closed-shell Aufbau
occupations at every k-point.  For metallic k-meshes, set
``options.smearing_temperature`` (``k_B T`` in Hartree) to enable
Fermi-Dirac occupations, bisection on the chemical potential, and
free-energy convergence.

**DIIS.** Pulay DIIS extrapolation on the per-k Fock matrices, with
the error vector ``e(k) = F(k) D(k) S(k) - S(k) D(k) F(k)`` evaluated
at every k-point and the Pulay B matrix built from the k-weighted
Frobenius inner product ``B_ij = S_k w_k Re tr(e_i(k)^+ e_j(k))``.
Activated by ``options.use_diis = True`` (the default), kicking in
at iteration ``options.diis_start_iter``. Once DIIS is active, the
density damping is skipped (matches the Γ-Ewald driver convention
in :mod:`vibeqc.periodic_rhf_ewald`). On tight bulk cells where
plain damping plateaus, multi-k DIIS converges in O(10-20)
iterations.

**Density convention.** Identical to the multi-k Fock builder: the
long-range J piece is built from the proper periodic density
``r(r) = S_g S_{muν} D(g)_{muν} chi_mu(r) chi_ν(r - R_g)``. At a [1,1,1]
k-mesh the inverse Bloch fold replicates the single-k density into
every cell block, matching the molecular-limit convention used by
``run_rhf_periodic_gamma_ewald3d``. At genuine multi-k meshes the
density acquires k-space structure and the SCF converges to a
different fixed point (bulk RHF) -- that's the point.
"""

from __future__ import annotations


from .guess import periodic_result_selection

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from ._vibeqc_core import (
    BasisSet,
    BlochKMesh,
    CoulombMethod,
    InitialGuess,
    LatticeMatrixSet,
    LatticeSumOptions,
    PeriodicSystem,
    SCFIteration,
    bloch_sum,
    build_jk_gamma_molecular_limit,
    compute_kinetic_lattice,
    compute_nuclear_lattice,
    compute_overlap_lattice,
    nuclear_repulsion_per_cell,
    real_space_density_from_kpoints,
    real_space_density_from_kpoints_fractional,
)
from .ewald_composed import build_j_ewald_3d, make_ewald_3d_gamma_j_builder
from .ewald_j import auto_grid
from .guess import (
    initial_density_closed_shell,
    periodic_fock_guess_k,
)
from .level_shift_schedule import LevelShiftSchedule
from .madelung import (
    madelung_energy_correction_for_lat as _madelung_energy_correction_for_lat,
)
from .mom import select_occupied_by_max_overlap as _mom_select
from .oda import compute_oda_lambda as _compute_oda_lambda
from .oda import oda_mix_densities as _oda_mix
from .perf import PerfScope
from .periodic_corrected_exchange import (
    CorrectedEwaldExchange,
    corrected_exchange_unavailable_reason,
    is_unsupported_gauge,
)
from .periodic_fock_multi_k import (
    build_periodic_fock_ewald3d_k,
    build_periodic_fock_slab_ewald2d_k,
    make_ewald_3d_lattice_j_cache,
    make_slab_ewald_2d_lattice_j_cache,
)
from .periodic_k_density import density_matrices_per_k as _density_matrices_per_k
from .periodic_scf_accelerators import (
    DynamicDamping,
    MultiKPeriodicSCFAccelerator,
)
from .progress import ProgressLogger, resolve_progress
from .scf_divergence import check_scf_divergence
from .smearing import (
    closed_shell_periodic_occupations as _closed_shell_periodic_occupations,
    occupations_are_per_k_integer_aufbau as _occupations_are_per_k_integer_aufbau,
)


def _g0_block(lms):
    """Return the (numpy) g=0 block of a LatticeMatrixSet. Used by
    Madelung correction to extract the unit-cell density / overlap
    from a real-space LatticeMatrixSet."""
    for idx, cell in enumerate(lms.cells):
        if (cell.index == np.array([0, 0, 0])).all():
            return np.asarray(lms.blocks[idx])
    raise RuntimeError("LatticeMatrixSet missing g=0 cell")


__all__ = [
    "PeriodicRHFMultiKEwaldResult",
    "run_rhf_periodic_multi_k_ewald3d",
]


@dataclass
class PeriodicRHFMultiKEwaldResult:
    """Result of :func:`run_rhf_periodic_multi_k_ewald3d`.

    Per-cell quantities (``energy``, ``e_electronic``, ``e_nuclear``)
    and per-k matrices (``mo_energies``, ``mo_coeffs``, ``fock``,
    ``overlap``, ``hcore``) alongside the converged real-space density
    ``D_real``.
    """

    energy: float
    e_electronic: float
    e_nuclear: float
    n_iter: int
    converged: bool

    # Per-k arrays -- lists of length ``len(kmesh.kpoints)``.
    mo_energies: List[np.ndarray]
    mo_coeffs: List[np.ndarray]
    fock: List[np.ndarray]
    overlap: List[np.ndarray]
    hcore: List[np.ndarray]

    # Real-space converged density.
    density: LatticeMatrixSet

    # Diagnostic trace: (iter, E_total, dE, grad_norm_sum_k).
    scf_trace: List[SCFIteration] = field(default_factory=list)

    # Ewald-specific.
    omega: float = 0.0
    grid_shape: Tuple[int, int, int] = (0, 0, 0)

    # Phase C1b -- Fermi-Dirac smearing diagnostics. ``temperature`` is
    # the value passed in via options.smearing_temperature; the rest are
    # filled in by the SCF when smearing is active. ``free_energy`` =
    # ``energy - temperature . entropy`` and is the variational quantity
    # under smearing. ``occupations`` is a list of per-k occupation
    # arrays (each in [0, 2] for closed-shell RHF). When smearing is
    # off (temperature == 0), ``free_energy == energy``, ``entropy == 0``,
    # and ``occupations`` lists hard-Aufbau {0, 2} arrays.
    smearing_temperature: float = 0.0
    fermi_level: float = 0.0
    entropy: float = 0.0
    free_energy: float = 0.0
    occupations: List[np.ndarray] = field(default_factory=list)
    # True when full-range exact exchange ran on a symmetry-reduced
    # Monkhorst-Pack mesh via the Pisani/Dovesi star unfolding; the
    # runner turns this into the kpoint_symmetry_unfolding citation.
    used_kpoint_symmetry_unfolding: bool = False
    # Snapshot of the lattice support actually used after truncation
    # optimization. Analytic derivatives must reconstruct S/T/V/J/K on this
    # support, not on the caller's pre-optimization options object.
    effective_lattice_opts: Optional[LatticeSumOptions] = None

    restart_basis: object = None
    restart_lattice: object = None
    guess_selection: object = None

    restart_mesh: object = None
    restart_kpoints: object = None
    restart_weights: object = None

    # Preserve the physical per-k state independently of lattice cutoff.
    density_k: Optional[List[np.ndarray]] = None


class _MultiKPulayDIIS:
    """Pulay DIIS extrapolation on a per-k Fock-matrix list.

    Mirrors the Γ-Ewald Pulay DIIS (the C++ ``vibeqc::DIIS`` binding)
    but with per-k error vectors and a k-weighted Frobenius inner
    product on the Pulay B matrix:

        e_i(k)  =  F_i(k) D_i(k) S(k) - S(k) D_i(k) F_i(k)
        B_{ij}  =  S_k w_k . Re tr( e_i(k)^+ e_j(k) )

    Returns an extrapolated Fock list ``F_ex(k) = S_i c_i F_i(k)``
    with the coefficients ``c_i`` that minimize the augmented Pulay
    quadratic form. Numerically unstable B matrices (near-singular,
    coefficient blow-up) trigger dropping the oldest history entry
    and resolving -- same fallback as the Γ-Ewald driver.
    """

    def __init__(self, max_subspace: int = 8):
        self.max_subspace = int(max_subspace)
        # Each history entry is a list[k] of complex (n_bf, n_bf) arrays.
        self.fock_histories: List[List[np.ndarray]] = []
        self.error_histories: List[List[np.ndarray]] = []

    @property
    def subspace_size(self) -> int:
        return len(self.fock_histories)

    def extrapolate(
        self,
        F_k_list: Sequence[np.ndarray],
        error_k_list: Sequence[np.ndarray],
        weights: Sequence[float],
    ) -> List[np.ndarray]:
        # Snapshot the current per-k F and error.
        self.fock_histories.append([np.asarray(F).copy() for F in F_k_list])
        self.error_histories.append([np.asarray(e).copy() for e in error_k_list])
        while len(self.fock_histories) > self.max_subspace:
            self.fock_histories.pop(0)
            self.error_histories.pop(0)

        n = len(self.fock_histories)
        if n < 2:
            return [np.asarray(F).copy() for F in F_k_list]

        w = np.asarray(weights, dtype=float)
        n_k = len(F_k_list)
        assert w.shape == (n_k,)

        while True:
            B = np.zeros((n + 1, n + 1))
            for i in range(n):
                for j in range(i, n):
                    bij = 0.0
                    for k_idx in range(n_k):
                        ei = self.error_histories[i][k_idx]
                        ej = self.error_histories[j][k_idx]
                        # Re tr(ei^+ ej) = Re S ei.conj() * ej
                        bij += float(w[k_idx]) * float(
                            np.real(np.vdot(ei.ravel(), ej.ravel()))
                        )
                    B[i, j] = B[j, i] = bij
            B[n, :n] = -1.0
            B[:n, n] = -1.0
            rhs = np.zeros(n + 1)
            rhs[n] = -1.0
            try:
                coeffs = np.linalg.solve(B, rhs)
                if np.any(np.abs(coeffs[:n]) > 1e6):
                    raise np.linalg.LinAlgError("coeff blow-up")
                break
            except np.linalg.LinAlgError:
                if n <= 2:
                    return [np.asarray(F).copy() for F in F_k_list]
                self.fock_histories.pop(0)
                self.error_histories.pop(0)
                n -= 1

        F_ex_list: List[np.ndarray] = []
        for k_idx in range(n_k):
            F_ex = np.zeros_like(self.fock_histories[0][k_idx], dtype=complex)
            for i, c in enumerate(coeffs[:n]):
                F_ex = F_ex + c * self.fock_histories[i][k_idx]
            F_ex_list.append(F_ex)
        return F_ex_list


def _canonical_orthogonalizer_complex(
    S: np.ndarray,
    threshold: float = 1e-7,
    *,
    normalize_diag_first: bool = True,
) -> Tuple[np.ndarray, int]:
    """Canonical orthogonaliser X(k) for a Hermitian S(k).

    Returns the rectangular ``(n_bf, n_kept)`` complex X with
    ``X^+ S X ≈ I_{n_kept}``. Directions with S-eigenvalue below
    ``threshold`` are projected out.

    See :func:`vibeqc.periodic_rhf_ewald._canonical_orthogonalizer`
    for the rationale of ``normalize_diag_first`` (default True;
    matches PySCF's ``canonical_orth_``).
    """
    if normalize_diag_first:
        # S(k) is Hermitian; the diagonal is real. Use np.real to
        # cast away the imaginary noise (the diagonal of a Hermitian
        # matrix is exactly real, but Bloch-summation can leave
        # 1e-15-level imaginary drift).
        d = np.real(np.asarray(S).diagonal())
        if np.any(d <= 0):
            normalize_diag_first = False
        else:
            scale = 1.0 / np.sqrt(d)
            Snorm = S * np.outer(scale, scale)
            eigvals, eigvecs = np.linalg.eigh(Snorm)
            mask = eigvals >= threshold
            n_kept = int(mask.sum())
            if n_kept == 0:
                raise RuntimeError(
                    "run_rhf_periodic_multi_k_ewald3d: no S(k) eigenvalue "
                    f"above threshold {threshold:.1e} (after sqrt-diag "
                    "normalisation); basis is fully linearly dependent"
                )
            kept_vals = eigvals[mask]
            kept_vecs = eigvecs[:, mask]
            X_norm = kept_vecs / np.sqrt(kept_vals)
            X = scale[:, None] * X_norm
            return X, n_kept

    # Legacy raw-S path.
    eigvals, eigvecs = np.linalg.eigh(S)
    mask = eigvals >= threshold
    n_kept = int(mask.sum())
    if n_kept == 0:
        raise RuntimeError(
            "run_rhf_periodic_multi_k_ewald3d: no S(k) eigenvalue above "
            f"threshold {threshold:.1e}; basis is fully linearly dependent"
        )
    kept_vals = eigvals[mask]
    kept_vecs = eigvecs[:, mask]
    X = kept_vecs / np.sqrt(kept_vals)
    return X, n_kept


def _diag_in_orth_basis(
    F: np.ndarray,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Diagonalize F(k) in the per-k canonical-orthogonalized basis
    X(k). Returns ``C(k)`` shaped ``(n_bf, n_kept)`` and real
    ``e(k)`` shaped ``(n_kept,)``."""
    Fp = X.conj().T @ F @ X
    Fp = 0.5 * (Fp + Fp.conj().T)  # Hermitise numerical drift
    eps, Cp = np.linalg.eigh(Fp)
    C = X @ Cp
    return C, eps


def _damp_lattice_matrix(
    D_new: LatticeMatrixSet,
    D_prev: Optional[LatticeMatrixSet],
    alpha: float,
) -> LatticeMatrixSet:
    """Block-wise ``D <- a . D_prev + (1 - a) . D_new``.

    With ``D_prev is None`` (first iteration) or ``alpha == 0``,
    returns ``D_new`` unchanged. Uses ``LatticeMatrixSet.set_block``
    to mutate the underlying C++ vector in place -- assignment via
    ``D_new.blocks[g] = ...`` writes to a transient Python-list copy
    that silently doesn't persist.
    """
    if D_prev is None or alpha == 0.0:
        return D_new
    n_cells = len(D_new.cells)
    assert len(D_prev.cells) == n_cells
    for g_idx in range(n_cells):
        blk_new = np.asarray(D_new.blocks[g_idx])
        blk_prev = np.asarray(D_prev.blocks[g_idx])
        D_new.set_block(g_idx, alpha * blk_prev + (1.0 - alpha) * blk_new)
    return D_new


def _localize_single_gamma_density(
    D_real: LatticeMatrixSet,
    C_gamma: np.ndarray,
    occupations: np.ndarray,
) -> LatticeMatrixSet:
    """Use the Γ-only direct-space convention P(g!=0)=0.

    ``real_space_density_from_kpoints`` is the mathematically correct
    inverse Bloch fold for a sampled k-mesh. For a single Γ point it
    cannot recover any cell decay and therefore fills every real-space
    block with P(Γ). The dedicated Γ Ewald driver uses the direct-space
    molecular-limit convention instead: only the home-cell density is
    known and all image-density blocks are zero. The single-Γ branch of
    this multi-k driver must use that same convention to avoid building
    a homogeneous-density Fock from a one-point mesh.
    """
    C = np.asarray(C_gamma, dtype=complex)
    occ = np.asarray(occupations, dtype=float).reshape(-1)
    D_home_c = (C * occ[None, :].astype(complex)) @ C.conj().T
    D_home_c = 0.5 * (D_home_c + D_home_c.conj().T)
    D_home = np.real_if_close(D_home_c, tol=1000).real
    D_home = 0.5 * (D_home + D_home.T)
    zero = np.zeros_like(D_home)
    for g_idx, cell in enumerate(D_real.cells):
        if (cell.index == np.array([0, 0, 0])).all():
            D_real.set_block(g_idx, D_home)
        else:
            D_real.set_block(g_idx, zero)
    return D_real


def _build_single_gamma_molecular_limit_fock(
    basis: BasisSet,
    system: PeriodicSystem,
    D_real: LatticeMatrixSet,
    Hcore_gamma: np.ndarray,
    *,
    omega: float,
    lattice_opts: LatticeSumOptions,
    grid_shape: Tuple[int, int, int],
    origin: Optional[Sequence[float]],
    spacing_bohr: float,
    j_build=None,
) -> List[np.ndarray]:
    """Single-Γ Fock matching ``run_rhf_periodic_gamma_ewald3d``.

    ``j_build`` is an optional cached Γ Hartree-J closure from
    :func:`make_ewald_3d_gamma_j_builder` (reuses the analytic-FT machinery
    across SCF iterations); ``None`` rebuilds via :func:`build_j_ewald_3d`
    each call (unchanged behaviour).
    """
    D_home = np.asarray(_g0_block(D_real))
    if j_build is not None:
        J = j_build(D_home)
    else:
        J = build_j_ewald_3d(
            basis,
            system,
            D_home,
            omega=float(omega),
            lattice_opts=lattice_opts,
            grid_shape=grid_shape,
            origin=origin,
            spacing_bohr=spacing_bohr,
        )
    jk_full = build_jk_gamma_molecular_limit(
        basis,
        system,
        lattice_opts,
        D_home,
        0.0,
    )
    K = np.asarray(jk_full.K)
    F = np.asarray(Hcore_gamma, dtype=complex) + J - 0.5 * K
    F = 0.5 * (F + F.conj().T)
    return [F]


def _hermitize_k_matrices(matrices: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Return Hermitian per-k matrices, removing builder-level noise."""
    out: List[np.ndarray] = []
    for mat in matrices:
        arr = np.asarray(mat, dtype=complex)
        out.append(0.5 * (arr + arr.conj().T))
    return out


def run_rhf_periodic_multi_k_ewald3d(
    system: PeriodicSystem,
    basis: BasisSet,
    kmesh: BlochKMesh,
    options=None,  # PeriodicRHFOptions-like
    *,
    omega: float = 0.0,
    grid_shape: Optional[Union[Tuple[int, int, int], int]] = None,
    origin: Optional[Sequence[float]] = None,
    spacing_bohr: float = 0.3,
    linear_dep_threshold: float = 1e-7,
    canonical_orth_normalize_diag_first: bool = True,
    auto_optimize_truncation: bool = True,
    level_shift_schedule: Optional["LevelShiftSchedule"] = None,
    use_mom: bool = False,
    use_oda: bool = False,
    oda_trust_lambda_max: float = 1.0,
    progress: Union[bool, ProgressLogger, None] = None,
    initial_density_k: Optional[Sequence] = None,
    verbose: Optional[int] = None,
    bz_integration: Optional[str] = None,
    exchange_exxdiv: Optional[str] = None,
) -> PeriodicRHFMultiKEwaldResult:
    """Multi-k closed-shell periodic RHF SCF with EWALD_3D Coulomb.

    Parameters
    ----------
    system, basis
        Periodic system and AO basis.
    kmesh
        :class:`BlochKMesh` (from :func:`vibeqc.monkhorst_pack`)
        defining the k-point sampling. ``kmesh.weights`` must sum to 1.
    options
        Optional :class:`PeriodicRHFOptions`; uses defaults when None.
        ``max_iter``, ``damping``, ``conv_tol_energy``,
        ``conv_tol_grad``, and ``lattice_opts`` are honored.
        ``use_diis`` is currently ignored (multi-k DIIS is a follow-up).
    omega
        Ewald splitting parameter.
    grid_shape, origin, spacing_bohr
        FFT-Poisson grid controls forwarded to the long-range J build.
    linear_dep_threshold
        Per-k S(k) eigenvalue floor for canonical orthogonalisation.
    exchange_exxdiv
        Which ``q -> 0`` gauge the exact-exchange arm uses. ``None``
        (default) keeps the historical per-mesh choice: the
        molecular-limit kernel at a single Gamma point, the corrected
        Ewald split on every other mesh. ``'ewald'`` takes the corrected
        split at **every** mesh, including a single Gamma point, and
        raises :class:`NotImplementedError` when the split is unavailable
        rather than falling back. ``'none'`` takes the molecular-limit
        kernel and is accepted only at a single Gamma point, where that
        sum has exactly one term.

        The two are different finite-size conventions with the same
        infinite-k limit (Carrier, Rohra & Goerling, Phys. Rev. B 75,
        205126 (2007), Sec. IV), so a Gamma-supercell energy band-folds
        onto a multi-k energy only when both sides name the same one --
        see ``docs/user_guide/periodic_methods.md`` and
        ``tests/test_periodic_exchange_gauge_band_folding.py``.
    progress
        Live progress logging. ``True`` prints flushed per-stage and
        per-iteration lines to stdout (the ``tail -f`` workflow);
        ``False`` / ``None`` is silent; a :class:`ProgressLogger`
        instance is used as-is. See :mod:`vibeqc.progress`.

    Returns
    -------
    :class:`PeriodicRHFMultiKEwaldResult`.
    """
    # Late import to avoid a circular dependency via __init__.
    from ._vibeqc_core import PeriodicRHFOptions

    opts = options if options is not None else PeriodicRHFOptions()
    from .guess import select_periodic_driver_guess
    guess = select_periodic_driver_guess(
        system, opts, route="ewald", method="RHF", driver="run_rhf_periodic_multi_k_ewald3d",
        multi_k=True, restart_supplied=initial_density_k is not None,
    ).transport
    if initial_density_k is not None:
        if len(kmesh.kpoints) != int(np.prod(kmesh.mesh)):
            raise NotImplementedError("READ requires a complete full k mesh on the Ewald lattice adapter")
        if getattr(opts, "atomic_spins", None):
            raise ValueError("atomic_spins requires SAD without a restart density")
    if (getattr(opts, "initial_guess", None) == InitialGuess.READ) and initial_density_k is None:
        raise NotImplementedError(
            "periodic READ requires initial_density_k with a complete per-k "
            "density payload. Use run_periodic_job(read_from=...) to load "
            "a compatible result or archive; see docs/user_guide/initial_guess.md."
        )
    lat_opts: LatticeSumOptions = opts.lattice_opts
    slab_mode = lat_opts.coulomb_method == CoulombMethod.SLAB_EWALD_2D
    if slab_mode and system.dim != 2:
        raise ValueError(
            "run_rhf_periodic_multi_k_ewald3d: SLAB_EWALD_2D requires "
            f"dim == 2; got dim = {system.dim}"
        )
    plog = resolve_progress(progress, verbose=verbose)

    # ---- Force EWALD_3D gauge (gauge consistency; audit F1 2026-05-31) --
    # This driver hard-codes the Hartree J to the Ewald-3D builder, so
    # V_ne (compute_nuclear_lattice_dispatch) and e_nuc
    # (nuclear_repulsion_per_cell) MUST share that gauge or the Madelung
    # / G=0 self-energy fails to cancel and the SCF converges to a
    # non-physical energy with no warning (CLAUDE.md Sec.7). The Γ-only
    # sibling run_rhf_periodic_gamma_ewald3d forces this; the multi-k
    # driver did not, so a default PeriodicRHFOptions()
    # (coulomb_method=DIRECT_TRUNCATED) produced converged garbage
    # (LiH FCC primitive at Γ: -264.85 Ha vs the gauge-consistent
    # -7.52 Ha).
    # Gate on dim == 3: the EWALD_3D V_ne dispatch
    # (compute_nuclear_lattice_dispatch) is implemented only for dim == 3 -- the
    # 1D/2D variants raise (periodic_v_ne.py). The original F1 force was
    # unconditional, which started raising on 1D chains once the analytic-FT
    # V_ne landed its dim==3 guard (commit 00f321e6); gating here restores the
    # DIRECT_TRUNCATED gauge for low-dim cells (the pre-existing behaviour the
    # dim=1 accelerator-uniformity tests rely on) while keeping the dim==3
    # gauge fix. See handover F4 (2026-06-01).
    if system.dim == 3 and lat_opts.coulomb_method != CoulombMethod.EWALD_3D:
        plog.info(
            "coulomb_method forced to EWALD_3D for gauge consistency "
            f"(was {lat_opts.coulomb_method!r}); this driver's Hartree J "
            "is Ewald-3D and V_ne / e_nuc must match"
        )
        lat_opts.coulomb_method = CoulombMethod.EWALD_3D

    # w must match the nuclear Ewald a so the jellium background
    # terms cancel exactly.  When not specified explicitly, use
    # CRYSTAL's default a = 2.8/V^{1/3} (matching the BIPOLE driver).
    _ewald_tol = getattr(opts, "ewald_tolerance", 1e-12)
    _cutoff = getattr(opts, "ewald_cutoff_bohr", lat_opts.nuclear_cutoff_bohr)
    if slab_mode:
        if omega <= 0.0:
            omega = float(getattr(lat_opts, "slab_ewald_alpha", 0.4))
        if omega <= 0.0:
            omega = 0.4
        lat_opts.slab_ewald_alpha = float(omega)
    elif omega <= 0.0:
        _user_omega = getattr(opts, "ewald_omega", None)
        if _user_omega is not None and float(_user_omega) > 0.0:
            omega = float(_user_omega)
        else:
            from .pbc_bipole_common import default_ewald_alpha

            V_cell = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
            # GitLab #651: CRYSTAL's 2.8/V^(1/3), bounded below so that
            # erfc(omega r)/r has decayed to opts.ewald_tolerance at this
            # route's fixed real-space image cutoff (lat_opts.cutoff_bohr).
            omega = default_ewald_alpha(
                V_cell,
                real_cutoff_bohr=float(lat_opts.cutoff_bohr),
                tolerance=float(getattr(opts, "ewald_tolerance", 1e-12)),
            )

    lat = np.asarray(system.lattice, dtype=float)

    if slab_mode:
        grid_shape_t = (0, 0, 0)
    elif grid_shape is None:
        grid_shape_t = auto_grid(lat, spacing_bohr)
    elif isinstance(grid_shape, int):
        grid_shape_t = (grid_shape, grid_shape, grid_shape)
    else:
        grid_shape_t = tuple(int(x) for x in grid_shape)
    if slab_mode:
        plog.info(f"RHF multi-k SLAB_EWALD_2D / alpha = {float(omega):.3f}")
    else:
        plog.info(
            f"RHF multi-k EWALD_3D / omega = {float(omega):.3f}, "
            f"FFT grid {grid_shape_t[0]}x{grid_shape_t[1]}x{grid_shape_t[2]}"
        )
    plog.info(f"basis: {basis.name}  ({basis.nbasis} BFs / {basis.nshells} shells)")
    from .options_dump import dump_active_settings

    dump_active_settings(
        plog,
        [
            ("PeriodicRHFOptions", opts),
            ("LatticeSumOptions", lat_opts),
            (
                "Driver kwargs",
                {
                    "omega": float(omega),
                    "grid_shape": grid_shape_t,
                    "origin": origin,
                    "spacing_bohr": float(spacing_bohr),
                    "linear_dep_threshold": float(linear_dep_threshold),
                    "canonical_orth_normalize_diag_first": canonical_orth_normalize_diag_first,
                    "auto_optimize_truncation": auto_optimize_truncation,
                },
            ),
        ],
    )
    if plog.level >= 5:
        from .scf_log import format_basis_summary

        plog.write_raw(format_basis_summary(basis))

    # Closed-shell sanity.
    n_elec = system.n_electrons()
    if n_elec % 2 != 0:
        raise ValueError(
            "run_rhf_periodic_multi_k_ewald3d: closed-shell RHF requires "
            f"even electron count; got {n_elec}"
        )
    if system.multiplicity != 1:
        raise ValueError(
            "run_rhf_periodic_multi_k_ewald3d: closed-shell RHF requires "
            f"multiplicity=1; got {system.multiplicity}"
        )
    n_occ = n_elec // 2
    smearing_T = float(getattr(opts, "smearing_temperature", 0.0))
    if smearing_T < 0.0:
        raise ValueError(
            "run_rhf_periodic_multi_k_ewald3d: smearing_temperature must be >= 0"
        )

    # BZ-integration backend (opt-in, experimental). None / "smearing" keeps
    # the established temperature path (Aufbau at T=0, Fermi-Dirac at T>0);
    # "gilat" selects the parameter-free Gilat-Raubenheimer net occupations
    # (the CRYSTAL SHRINK-ISP analogue) on the full regular mesh. GR is a T=0
    # method, so it cannot be combined with finite smearing.
    if bz_integration not in (None, "smearing", "gilat"):
        raise ValueError(
            "run_rhf_periodic_multi_k_ewald3d: bz_integration must be None, "
            f"'smearing', or 'gilat'; got {bz_integration!r}"
        )
    use_gilat = bz_integration == "gilat"
    if use_gilat and smearing_T > 0.0:
        raise ValueError(
            "run_rhf_periodic_multi_k_ewald3d: bz_integration='gilat' is a "
            "T=0 integrator; do not combine it with smearing_temperature > 0"
        )
    # Both finite-T smearing and the Gilat-Raubenheimer net produce fractional
    # occupations, so they share the fractional-density build path (vs the
    # integer Aufbau path).
    use_fractional_density = (smearing_T > 0.0) or use_gilat

    k_points = list(kmesh.kpoints)
    weights = np.asarray(kmesh.weights, dtype=float)
    n_k = len(k_points)
    if n_k == 0:
        raise ValueError("kmesh has no k-points")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError(f"kmesh.weights must sum to 1; got {weights.sum():.6f}")
    single_gamma_mesh = (
        n_k == 1
        and np.isclose(float(weights[0]), 1.0)
        and np.allclose(np.asarray(k_points[0], dtype=float), 0.0, atol=1e-12)
    )
    # ---- Exact-exchange q -> 0 gauge (#560) ------------------------------
    # Two conventions, one infinite-k limit, different at every finite mesh:
    # the corrected Ewald split carries the probe-charge q + G = 0 term and
    # the molecular-limit kernel carries none. Per Carrier, Rohra & Goerling,
    # Phys. Rev. B 75, 205126 (2007), doi:10.1103/PhysRevB.75.205126, Sec. IV,
    # the singularity correction and the uncorrected exchange energy "vary
    # oppositely with increasing number of k points", the corrected one
    # converging far faster -- so the two are not interchangeable at a fixed
    # mesh and a caller comparing a Gamma-supercell energy with a multi-k
    # energy has to put both on one of them. Measured on H2/STO-3G (a = 8
    # bohr, d = 1.4 bohr) they are 21.7 mHa apart at a single Gamma point.
    if exchange_exxdiv not in (None, "ewald", "none"):
        raise ValueError(
            "run_rhf_periodic_multi_k_ewald3d: exchange_exxdiv must be "
            "'ewald', 'none', or None for the per-mesh default; got "
            f"{exchange_exxdiv!r}"
        )
    if exchange_exxdiv == "none" and not single_gamma_mesh:
        raise NotImplementedError(
            "run_rhf_periodic_multi_k_ewald3d: exchange_exxdiv='none' is "
            "available only at a single Gamma point, where the exchange sum "
            "has exactly one term. On a real k mesh the SCF density is "
            "Born-von-Karman-torus periodic, D(g) does not decay, and the "
            "uncorrected image sum does not converge -- H2/STO-3G in a "
            "12-bohr cube on a (2,1,1) mesh moved from -1.4505 to -1.6863 Ha "
            "over a 12-to-20-bohr lattice cutoff sweep. Use "
            "exchange_exxdiv='ewald' or the default."
        )
    if exchange_exxdiv == "none" and lat_opts.pair_complete_1e:
        raise NotImplementedError(
            "Physical quartet support uses the periodic density at Gamma "
            "and requires corrected exchange; use exchange_exxdiv='ewald' "
            "or the default."
        )
    # This flag is the gauge, not the mesh: everything keyed on it below --
    # the density localisation, the Gamma J closure, the Fock build, the ODA
    # rebuild and the final self-consistency pass -- selects the
    # molecular-limit kernel together, and asking for 'ewald' means the
    # single-Gamma mesh is served by the ordinary multi-k machinery.
    single_gamma_kmesh = (
        single_gamma_mesh and exchange_exxdiv != "ewald"
        and not lat_opts.pair_complete_1e
    )
    plog.info(
        f"k-mesh: {n_k} k-point{'s' if n_k != 1 else ''}, "
        f"weights sum = {weights.sum():.4f}"
    )
    if single_gamma_mesh:
        plog.info(
            "exact-exchange q->0 gauge: "
            + (
                "corrected Ewald split (exxdiv='ewald')"
                if not single_gamma_kmesh
                else "molecular-limit kernel, no q+G=0 term "
                "(exchange_exxdiv='ewald' selects the corrected split)"
            )
        )

    # ---- Auto-optimise lattice truncation (default ON) -------------------
    # Multi-k version: optimise against the worst k-point in the mesh.
    if auto_optimize_truncation and lat_opts.coulomb_method == CoulombMethod.EWALD_3D:
        from .eigs_preflight import (
            format_truncation_optimization_report,
            optimize_truncation,
        )

        k_arr = [np.asarray(k, dtype=float) for k in k_points]
        opt_rep = optimize_truncation(
            system,
            basis,
            lattice_opts=lat_opts,
            k_points_cart=k_arr,
        )
        if (
            opt_rep.n_evaluations > 1
            or opt_rep.optimized_lattice_opts.cutoff_bohr != lat_opts.cutoff_bohr
        ):
            plog.write_raw(format_truncation_optimization_report(opt_rep))
            if not opt_rep.converged:
                plog.warn(
                    "auto_optimize_truncation did not converge; SCF will run "
                    "with the last evaluated settings."
                )
            lat_opts = opt_rep.optimized_lattice_opts

    # ---- Real-space one-electron integrals -------------------------------
    with (
        plog.stage(
            "integrals_lattice",
            detail=f"S/T/V at cutoff {lat_opts.cutoff_bohr:.2f} bohr",
        ),
        PerfScope("periodic.integrals_lattice"),
    ):
        with PerfScope("periodic.compute_overlap_lattice"):
            S_lat = compute_overlap_lattice(basis, system, lat_opts)
        with PerfScope("periodic.compute_kinetic_lattice"):
            T_lat = compute_kinetic_lattice(basis, system, lat_opts)
        with PerfScope("periodic.compute_nuclear_lattice"):
            if slab_mode:
                from .periodic_v_ne_slab import (
                    build_v_ne_slab_ewald_2d_k_cache,
                )

                slab_vne_cache = build_v_ne_slab_ewald_2d_k_cache(
                    basis,
                    system,
                    lat_opts,
                    alpha=float(omega),
                )
                V_lat = None
            else:
                from .periodic_v_ne import compute_nuclear_lattice_dispatch

                V_lat = compute_nuclear_lattice_dispatch(basis, system, lat_opts)
    from .lattice_screening import on_physical_eri_cells

    S_lat = on_physical_eri_cells(S_lat, basis, system, lat_opts)
    T_lat = on_physical_eri_cells(T_lat, basis, system, lat_opts)
    V_lat = on_physical_eri_cells(V_lat, basis, system, lat_opts)
    cells = list(S_lat.cells)

    # Per-k S(k), Hcore(k), orthogonaliser X(k).
    S_k_list: List[np.ndarray] = []
    Hcore_k_list: List[np.ndarray] = []
    X_k_list: List[np.ndarray] = []
    n_kept_list: List[int] = []
    # Linear-dependence preflight runs per k-point. The CRYSTAL
    # implementation flagged this in 2017 (Searle, Bernasconi, Harrison,
    # ARCHER eCSE04-16): a basis that's well-conditioned at Γ can fail
    # at zone-boundary k-points where Bloch phases do not constructively
    # add. Each S(k) is checked independently; severity is taken as the
    # worst over all k-points and the report emits to the SCF banner.
    from .linear_dependence import (
        PERIODIC_OVERLAP_HINT,
        scf_preflight_overlap_check,
    )

    for k_idx, k in enumerate(k_points):
        k_arr = np.asarray(k, dtype=float).reshape(3)
        S_k = np.asarray(bloch_sum(S_lat, k_arr))
        T_k = np.asarray(bloch_sum(T_lat, k_arr))
        if slab_mode:
            from .periodic_v_ne_slab import compute_v_ne_slab_ewald_2d_k_matrix

            V_k = compute_v_ne_slab_ewald_2d_k_matrix(
                basis,
                system,
                lat_opts,
                k_arr,
                alpha=float(omega),
                cache=slab_vne_cache,
            )
        else:
            V_k = np.asarray(bloch_sum(V_lat, k_arr))
        H_k = T_k + V_k
        S_k = 0.5 * (S_k + S_k.conj().T)
        H_k = 0.5 * (H_k + H_k.conj().T)
        scf_preflight_overlap_check(
            S_k,
            plog=plog,
            label=f"S(k={k_idx}, k_cart={k_arr.round(4).tolist()})",
            basis=basis,
            remediation_hint=PERIODIC_OVERLAP_HINT,
        )
        X_k, n_kept = _canonical_orthogonalizer_complex(
            S_k,
            linear_dep_threshold,
            normalize_diag_first=canonical_orth_normalize_diag_first,
        )
        if n_occ > n_kept:
            raise RuntimeError(
                "run_rhf_periodic_multi_k_ewald3d: canonical "
                "orthogonalisation at k = "
                f"{k_arr} dropped too many directions (n_occ = {n_occ}, "
                f"n_kept = {n_kept}); loosen linear_dep_threshold or "
                "pick a less redundant basis."
            )
        S_k_list.append(S_k)
        Hcore_k_list.append(H_k)
        X_k_list.append(X_k)
        n_kept_list.append(n_kept)

    fock_guess_per_k = guess in (InitialGuess.SAP, InitialGuess.HUECKEL)
    guess_fock_k = Hcore_k_list
    if fock_guess_per_k:
        guess_fock_k = list(
            periodic_fock_guess_k(
                system,
                basis,
                k_points,
                guess,
                lattice_opts=lat_opts,
                kinetic_lattice=T_lat,
                overlap_lattice=S_lat,
            )
        )

    # T_lat / V_lat are folded into Hcore(k), and SAP has consumed T_lat when
    # requested. Free the per-cell one-electron lattice integrals before the
    # SCF iterations (S_lat is still needed for its g0 block).
    del T_lat, V_lat

    # ---- Nuclear repulsion per cell --------------------------------------
    e_nuc = float(nuclear_repulsion_per_cell(system, lat_opts))

    # ---- Initial guess: diagonalize Hcore(k) or F_SAP(k) -----------------
    C_per_k: List[np.ndarray] = []
    eps_per_k: List[np.ndarray] = []
    for F_guess_k, X_k in zip(guess_fock_k, X_k_list):
        C_k, eps_k = _diag_in_orth_basis(F_guess_k, X_k)
        C_per_k.append(C_k.astype(complex))
        eps_per_k.append(eps_k)

    # Smearing setup (C1b). When ``temperature == 0`` this path
    # collapses to hard Aufbau: occupations are 2.0 for the lowest
    # n_occ orbitals, 0 elsewhere -- bit-identical to the integer-
    # occupation density build.
    n_electrons_per_cell = float(n_elec)

    def _occupations_from_eps(
        eps_per_k_local: Sequence[np.ndarray],
    ) -> Tuple[List[np.ndarray], float, float]:
        if use_gilat:
            # Gilat-Raubenheimer net occupations + E_F (CRYSTAL SHRINK-ISP
            # analogue). Sharp Fermi surface, so the electronic entropy is zero.
            # Handles a full Monkhorst-Pack mesh or a symmetry-reduced (IBZ)
            # mesh -- the IBZ case is expanded to the full BZ internally
            # (eps_n(R k) = eps_n(k)) and the occupations gathered back onto the
            # IBZ k-points the density build uses.
            from .bz_integration import gilat_occupations_for_kmesh

            occ_gr, ef_gr = gilat_occupations_for_kmesh(
                system,
                kmesh,
                eps_per_k_local,
                n_electrons_per_cell,
                spin_degeneracy=2.0,
            )
            return occ_gr, float(ef_gr), 0.0
        # MOM permutes C(k) / eps(k) so its selected occupied states lead;
        # the T = 0 fill must occupy that block positionally rather than
        # re-pick by energy across the mesh (GitLab #725).
        return _closed_shell_periodic_occupations(
            eps_per_k_local,
            weights,
            n_electrons_per_cell,
            n_occ,
            smearing_T,
            fixed_occupied_subspace=bool(use_mom),
        )

    occ_per_k, fermi_level, entropy = _occupations_from_eps(eps_per_k)

    def _fractional_density_needed(
        occ_per_k_local: Sequence[np.ndarray],
    ) -> bool:
        """Whether these occupations need the occupation-driven builder.

        The fixed ``C[:, :n_occ]`` slice builder is exact only for the legacy
        per-k integer Aufbau pattern. The T = 0 fill is one global Fermi level
        over the mesh (#85), so a band-overlap mesh can occupy a different
        number of bands per k (or a shared fractional group at mu), which no
        fixed slice reproduces; those occupations must be built from
        themselves, as the native multi-k GDF route does (#509).
        """
        if _occupations_are_per_k_integer_aufbau(occ_per_k_local, n_occ):
            return False
        if use_mom or use_oda:
            raise NotImplementedError(
                "run_rhf_periodic_multi_k_ewald3d: the zero-temperature "
                "Brillouin-zone fill is not per-k-integer-aufbau on this "
                "mesh (the bands overlap the Fermi level), so there is no "
                "sharp occupied subspace for use_mom/use_oda. Disable "
                "use_mom/use_oda or add finite smearing."
            )
        return True

    if _fractional_density_needed(occ_per_k):
        D_real = real_space_density_from_kpoints_fractional(
            C_per_k,
            occ_per_k,
            kmesh,
            cells,
        )
    else:
        # Use the integer-occupation density builder for the Aufbau
        # path so existing tests / numerics see no drift.
        n_occ_per_k = [n_occ] * n_k
        D_real = real_space_density_from_kpoints(
            C_per_k,
            n_occ_per_k,
            kmesh,
            cells,
        )
    if single_gamma_kmesh:
        D_real = _localize_single_gamma_density(
            D_real,
            C_per_k[0],
            occ_per_k[0],
        )

    # If SAD was requested, overwrite the Hcore-derived D_real with
    # the Superposition-of-Atomic-Densities density on the unit cell.
    # We place it in the g=0 cell only (Γ-only-style real-space
    # density); all other cells start at zero. The first SCF iter
    # builds F(D_SAD) and continues from there. This is the
    # essential fix for ionic insulators (NaCl, MgO) where the Hcore
    # initial guess underbinds deep core states so badly that DIIS
    # amplifies the swing into nonsense before it ever recovers.
    # Multi-k convention: density-mode guesses replace D(g=0) with the
    # engine output and zero the other cells; non-density guesses leave
    # the Hcore-derived multi-cell density alone.
    if initial_density_k is not None:
        D_engine = None
        plog.info("initial guess: READ (complete per-k density)")
    elif fock_guess_per_k:
        # Fock-mode guesses use the route's normal global-k Aufbau/smearing
        # policy to build D(k) from the per-k one-particle orbitals above.
        D_engine = None
        plog.info(
            f"initial guess: {guess.name} "
            "(one-particle Hamiltonian diagonalise at each k)"
        )
    else:
        D_engine = initial_density_closed_shell(
            system.unit_cell_molecule(),
            basis,
            n_occ,
            InitialGuess.SAD if guess == InitialGuess.PATOM else guess,
            is_periodic=True,
            periodic_system=system,
            lattice_opts=lat_opts,
            overlap=S_k_list, weights=weights,
        )
    if D_engine is not None:
        plog.info(f"initial guess: {guess.name} (density via GuessEngine)")
        for g_idx in range(len(D_real.cells)):
            if (D_real.cells[g_idx].index == np.array([0, 0, 0])).all():
                D_real.set_block(g_idx, D_engine)
            else:
                D_real.set_block(g_idx, np.zeros_like(D_engine, dtype=float))
    elif not fock_guess_per_k and initial_density_k is None:
        plog.info(f"initial guess: {guess.name} (Hcore-diagonalise at each k)")

    D_per_k = _density_matrices_per_k(C_per_k, occ_per_k)
    if D_engine is not None:
        D_engine_k = np.asarray(D_engine, dtype=complex)
        D_engine_k = 0.5 * (D_engine_k + D_engine_k.conj().T)
        D_per_k = [D_engine_k.copy() for _ in range(n_k)]

    if initial_density_k is not None:
        from .guess import periodic_restart_lattice_density, normalize_density_k_guess
        D_per_k = normalize_density_k_guess(initial_density_k, S_k_list, weights, n_elec)
        D_real = periodic_restart_lattice_density(
            D_per_k, S_k_list, weights, n_elec, kmesh, D_real.cells, retained_k_system=system,
        )
    D_real_prev: Optional[LatticeMatrixSet] = None
    D_per_k_prev: Optional[List[np.ndarray]] = None

    damping = float(opts.damping)
    if not (0.0 <= damping < 1.0):
        raise ValueError(
            f"run_rhf_periodic_multi_k_ewald3d: damping must be in "
            f"[0, 1); got {damping}"
        )

    # SCF accelerator -- dispatch on opts.scf_accelerator. DIIS / KDIIS
    # run on per-k complex Hermitian matrices; EDIIS / ADIIS /
    # EDIIS_DIIS bridge through the per-cell real-block representation
    # and the C++ block-vector kernels. Activation at
    # ``diis_start_iter`` matches the Γ-Ewald convention.
    use_diis = bool(opts.use_diis)
    diis_start_iter = int(opts.diis_start_iter)
    accel: Optional[MultiKPeriodicSCFAccelerator] = (
        MultiKPeriodicSCFAccelerator(opts) if use_diis else None
    )

    # Dynamic damping (Zerner-Hehenberger 1979).
    damper: Optional[DynamicDamping] = None
    if bool(getattr(opts, "dynamic_damping", False)):
        damper = DynamicDamping(
            initial_alpha=damping,
            alpha_min=float(getattr(opts, "dynamic_damping_min", 0.0)),
            alpha_max=float(getattr(opts, "dynamic_damping_max", 0.95)),
        )

    # Saunders-Hillier level shift on each F(k). 0.0 = off (default);
    # b > 0 raises virtual MO eigenvalues by b before diagonalisation
    # at every k-point. The shift is "inert" at the converged density
    # so the SCF fixed point is preserved.
    #
    # If `level_shift_schedule` is passed, it supersedes
    # `opts.level_shift`: per-iter shift = schedule.at(iter_idx). Use
    # `LevelShiftSchedule.crystal_default()` for tight ionic crystals
    # (Hcore guess -> first Fock build wants a large orbital rotation;
    # iter-decreasing shift bounds the rotation rate so the SCF stays
    # in the physical basin rather than collapsing to an over-bound
    # one -- see `examples/regression/crystal_parity/diag_lih_multik_iter3_*`).
    level_shift_static = float(getattr(opts, "level_shift", 0.0))
    if level_shift_schedule is not None and not isinstance(
        level_shift_schedule,
        LevelShiftSchedule,
    ):
        raise TypeError(
            "run_rhf_periodic_multi_k_ewald3d: level_shift_schedule must "
            "be a LevelShiftSchedule or None; "
            f"got {type(level_shift_schedule).__name__}"
        )
    # Fall back to the options-struct schedule field (the unified
    # `opts.level_shift_schedule` vector shared with the molecular
    # drivers) when no explicit schedule argument is given.
    if level_shift_schedule is None:
        _opt_sched = list(getattr(opts, "level_shift_schedule", None) or [])
        if _opt_sched:
            level_shift_schedule = LevelShiftSchedule(_opt_sched)
    if level_shift_schedule is not None:
        plog.info(
            f"level_shift_schedule: {level_shift_schedule.as_list()} "
            "(supersedes opts.level_shift)"
        )

    # Maximum-Overlap Method (MOM). After diagonalising F(k) at iter
    # k > 1, pick the occupied subspace by maximum S(k)-overlap with
    # iter (k-1)'s occupied MOs rather than by Aufbau (lowest n_occ
    # energies). Prevents orbital flipping that can carry the SCF into
    # a stable but unphysical basin (see
    # `examples/regression/crystal_parity/diag_lih_multik_iter3_*` --
    # LiH primitive at kmesh=(2,2,2) over-binds to -228 Ha with
    # Aufbau even from a SAD guess because iter-2 diagonalisation
    # rotates the occupied subspace away from SAD's). MOM is
    # incompatible with finite-T smearing (fractional occupations
    # don't have a sharp "occupied" subspace); refuse with a clear
    # error rather than silently producing a wrong answer.
    if use_mom and use_fractional_density:
        raise ValueError(
            "run_rhf_periodic_multi_k_ewald3d: use_mom is incompatible "
            "with fractional occupations (smearing_temperature > 0 or "
            "bz_integration='gilat') -- no sharp occupied subspace"
        )
    if use_mom:
        plog.info("MOM (Maximum Overlap Method): ON")
    # Tracked across iters; populated at end of each iter from the
    # final C_per_k state.
    C_prev_occ_per_k: Optional[List[np.ndarray]] = None

    # Optimal Damping Algorithm (Cancès & Le Bris 2000). Per-iter
    # line search between D_n (input density) and D_naive (post-diag
    # density), minimising a quadratic energy model along the
    # interpolation. Costs one EXTRA Fock build per iter (at
    # D_naive). Mutually exclusive with DIIS -- both want to control
    # the density update path; ODA does it via per-iter optimisation,
    # DIIS via multi-iter F extrapolation; combining them masks the
    # signal each is trying to use.
    if use_oda and use_diis:
        raise ValueError(
            "run_rhf_periodic_multi_k_ewald3d: use_oda and use_diis "
            "are mutually exclusive (both control the density update "
            "path; combining them is unsupported)"
        )
    if use_oda and use_fractional_density:
        raise ValueError(
            "run_rhf_periodic_multi_k_ewald3d: use_oda is incompatible "
            "with fractional occupations (smearing_temperature > 0 or "
            "bz_integration='gilat'); the quadratic energy model uses "
            "tr(D F) for the integer-occupation path -- generalisation is "
            "a follow-up"
        )
    if use_oda:
        if not (0.0 < oda_trust_lambda_max <= 1.0):
            raise ValueError(
                "run_rhf_periodic_multi_k_ewald3d: "
                f"oda_trust_lambda_max must be in (0, 1]; "
                f"got {oda_trust_lambda_max}"
            )
        plog.info(
            f"ODA (Optimal Damping Algorithm): ON (+1 Fock build/iter, "
            f"trust l_max = {oda_trust_lambda_max})"
        )

    # Phase C1c -- quadratic SCF fallback (per-k Newton step).
    quadratic_fallback_iter = int(getattr(opts, "quadratic_fallback_iter", 0))
    quadratic_fallback_shift = float(getattr(opts, "quadratic_fallback_shift", 0.1))
    quadratic_fallback_max_step = float(
        getattr(opts, "quadratic_fallback_max_step", 0.1)
    )

    # ---- SCF loop --------------------------------------------------------
    scf_label = "SLAB_EWALD_2D" if slab_mode else "EWALD_3D"
    plog.banner(f"SCF (RHF multi-k, {scf_label})")
    plog.info("  iter         energy (Ha)            dE          ||[F,DS]||   DIIS")

    scf_trace: List[SCFIteration] = []
    E_prev = 0.0
    F_k_list: List[np.ndarray] = [np.zeros_like(H) for H in Hcore_k_list]

    converged = False
    iter_idx = 0
    t_scf_start_perf = time.perf_counter()

    # Cache the iteration-invariant analytic-FT Hartree-J machinery once so
    # the SCF loop redoes only the density contraction (E2,
    # docs/pbc_audit_2026-06.md): a Γ closure for the single-Γ (1,1,1) mesh,
    # else the per-cell lattice cache keyed on the fixed D_real.cells. Both
    # are bit-identical to the per-iteration rebuild; grid backend -> None.
    if single_gamma_kmesh:
        if slab_mode:
            from .ewald_composed_slab import make_slab_ewald_2d_gamma_j_builder

            gamma_j_build = make_slab_ewald_2d_gamma_j_builder(
                basis, system, lattice_opts=lat_opts, alpha=float(omega)
            )
        else:
            gamma_j_build = make_ewald_3d_gamma_j_builder(
                basis, system, omega=float(omega), lattice_opts=lat_opts,
                grid_shape=grid_shape_t, origin=origin, spacing_bohr=spacing_bohr,
            )
        j_cache = None
    elif slab_mode:
        gamma_j_build = None
        j_cache = make_slab_ewald_2d_lattice_j_cache(
            basis, system, D_real.cells, lattice_opts=lat_opts, alpha=float(omega),
        )
    else:
        gamma_j_build = None
        j_cache = make_ewald_3d_lattice_j_cache(
            basis, system, D_real.cells, lattice_opts=lat_opts,
        )

    # ---- Corrected-Ewald exact exchange ----------------------------------
    # Multi-k RHF is pure HF: the full-range arm is always present
    # (c_full = 1). Summed as a bare 1/r kernel over a radial image ball it
    # does not converge, because the SCF density on a finite k mesh is
    # BvK-torus periodic and D(g) does not decay -- this cell fell from
    # -1.4505 to -1.6863 Ha going from a 12- to a 20-bohr cutoff. Use the
    # convergent Ewald split instead (erfc K_SR in real space, reciprocal
    # K_LR per q = k-k', probe-charge q+G=0 term); see
    # vibeqc.periodic_corrected_exchange.
    #
    # Not applied on two paths, both deliberately:
    #   * single_gamma_kmesh -- _build_single_gamma_molecular_limit_fock
    #     zeroes every image block first, so its bare sum has exactly one
    #     term and IS the molecular-limit K. Nothing to correct, and this
    #     is what keeps (1,1,1) RHF in bit-parity with the Gamma driver.
    #   * slab_mode -- the reciprocal q-channel cache is 3D-only and the
    #     2D block builder carries no full_range_alpha seam.
    corrected_exchange = None
    if slab_mode and exchange_exxdiv == "ewald":
        raise NotImplementedError(
            "run_rhf_periodic_multi_k_ewald3d: exchange_exxdiv='ewald' is "
            "unavailable under the SLAB_EWALD_2D gauge (the reciprocal "
            "q-channel cache is 3D-only). Drop the argument to keep the "
            "historical kernel."
        )
    if not slab_mode and not single_gamma_kmesh:
        _unavailable = corrected_exchange_unavailable_reason(
            system, kmesh, slab_mode=slab_mode
        )
        if _unavailable is not None and exchange_exxdiv == "ewald":
            # An explicit request never degrades silently, whatever the
            # failure class -- including the 2D/dim != 3 gauges that the
            # default path is allowed to keep the historical kernel for.
            raise NotImplementedError(
                "run_rhf_periodic_multi_k_ewald3d: exchange_exxdiv='ewald' "
                f"was requested, but {_unavailable}."
            )
        if _unavailable is None:
            corrected_exchange = CorrectedEwaldExchange.build(
                basis,
                system,
                np.asarray(
                    [np.asarray(c.r_cart, dtype=float) for c in cells]
                ),
                kmesh,
                float(omega),
                where="run_rhf_periodic_multi_k_ewald3d",
                lattice_opts=lat_opts,
            )
        elif is_unsupported_gauge(system, slab_mode=slab_mode):
            # A gauge the split does not exist for (2D slab, dim != 3): the
            # reciprocal cache is 3D-only. These routes pre-date the fix and
            # are live, so they keep the historical kernel and say so rather
            # than being deleted. Remaining gap, recorded with a maintainer
            # ask in handovers/HANDOVER_OPEN_BUGS_V015.md.
            plog.info(
                "exact exchange stays on the uncorrected real-space image "
                f"sum: {_unavailable}. Energies on this route are not "
                "lattice-cutoff converged."
            )
        else:
            # An explicit k list with no star metadata. Fail closed: there
            # is no structure to unfold from, and leaving it on the bare
            # kernel while a full mesh takes the corrected split makes the
            # two disagree (measured at 4.6e-5 Ha on H2/(2,2,2)).
            # Symmetry-reduced Monkhorst-Pack meshes ARE served since
            # 2026-07-29 via the wedge-native star unfolding.
            raise NotImplementedError(
                "run_rhf_periodic_multi_k_ewald3d: multi-k Hartree-Fock "
                f"needs full-range exact exchange, but {_unavailable}. "
                "Use a Monkhorst-Pack mesh (full, or symmetry-reduced "
                "with vq.attach_symmetry + use_symmetry=True)."
            )


    # SYM3b orbit reduction for the real-space exchange arms, wedge path
    # only. Profiled at 94 % of a wedge SCF iteration (LiH rock-salt,
    # cutoff 12: 45 s of 45.2 s), and the k-mesh-independent cost is why a
    # wedge run could be SLOWER than the full mesh. The reduced build
    # equals the *symmetrized* full build, which differs from the raw one
    # by the operator truncation asymmetry on shift-bearing cells -- the
    # wedge is already the symmetrized-answer gauge (its density is the
    # orbit-symmetrized object), so the reduction is only enabled there;
    # full-mesh numbers stay byte-identical. Mapping build is one-time
    # pure Python on the SAME radial cell list the K builder uses, so
    # positional block indexing is unchanged.
    # The legacy radial orbit map does not index the physical K enclosure.
    # Physical mode retains the full build until a complete orbit map is used.
    exchange_symmetry_reduction = None
    if (
        corrected_exchange is not None
        and corrected_exchange.unfolding is not None
        and not lat_opts.pair_complete_1e
    ):
        from ._vibeqc_core import direct_lattice_cells as _radial_cells
        from .bipole_symmetry_fock import (
            cell_orbit_mapping,
            representative_cell_indices,
        )

        # The mapping must index the K builder's OWN radial cell list
        # (direct_lattice_cells at the 2e cutoff), not the overlap
        # template ``cells``: under LatticeSumOptions.pair_complete_1e
        # (#429) the template is longer. Identical lists with the switch
        # off, so byte-identical there.
        _sym_mapping = cell_orbit_mapping(
            system, basis, list(_radial_cells(system, lat_opts.cutoff_bohr))
        )
        if _sym_mapping is not None:
            exchange_symmetry_reduction = (
                _sym_mapping,
                representative_cell_indices(_sym_mapping),
            )

    def _build_multi_k_fock(
        D_input: LatticeMatrixSet,
        D_input_per_k: Optional[List[np.ndarray]] = None,
    ) -> List[np.ndarray]:
        if slab_mode:
            return build_periodic_fock_slab_ewald2d_k(
                basis,
                system,
                D_input,
                alpha=float(omega),
                k_points_cart=[np.asarray(k) for k in k_points],
                Hcore_k=Hcore_k_list,
                lattice_opts=lat_opts,
                j_cache=j_cache,
            )
        if (
            corrected_exchange is not None
            and corrected_exchange.unfolding is not None
        ):
            # Symmetry-reduced wedge: the weighted wedge fold behind
            # D_input uses each representative in place of its orbit
            # average and is wrong at multi-cell cutoffs. J and K_SR are
            # first order in that error on the exchange path, so rebuild
            # the real-space density exactly -- unfold the wedge per-k
            # densities to the full BZ (Pisani, Dovesi & Roetti 1988,
            # Eq. II.7.21) and fold over the full mesh (Eq. II.7.48).
            if D_input_per_k is None:
                raise RuntimeError(
                    "run_rhf_periodic_multi_k_ewald3d: the IBZ exchange "
                    "path needs the per-k density list at every Fock "
                    "build"
                )
            D_input = corrected_exchange.fold_density(
                D_input_per_k, lat_opts
            )
        F_k = build_periodic_fock_ewald3d_k(
            basis,
            system,
            D_input,
            omega=float(omega),
            k_points_cart=[np.asarray(k) for k in k_points],
            Hcore_k=Hcore_k_list,
            lattice_opts=lat_opts,
            grid_shape=grid_shape_t,
            origin=origin,
            spacing_bohr=spacing_bohr,
            # Puts the full-range arm on erfc(alpha r)/r; the reciprocal
            # and q+G=0 arms of the same split follow below.
            full_range_alpha=(
                None if corrected_exchange is None else float(omega)
            ),
            symmetry_reduction=exchange_symmetry_reduction,
            j_cache=j_cache,
        )
        if corrected_exchange is None:
            return F_k
        # 0.5 matches the closed-shell J - 0.5 K fold in
        # build_fock_2e_ewald3d_blocks (c_full is 1.0 here). D(k) comes
        # from the driver rather than an inverse fold of D_input: the
        # per-k density is the primary object, and the fold would demand
        # the lattice cell list cover the BvK torus, which standard
        # fixtures do not.
        D_k_list = (
            list(D_input_per_k)
            if D_input_per_k is not None
            else corrected_exchange.torus_densities(D_input)
        )
        terms = corrected_exchange.k_space_terms_all_k(S_k_list, D_k_list)
        return [f - 0.5 * t for f, t in zip(F_k, terms)]

    if guess == InitialGuess.PATOM:
        seed_focks = _build_multi_k_fock(D_real, D_per_k)
        C_per_k, eps_per_k = [], []
        for fock, x in zip(seed_focks, X_k_list):
            c, e = _diag_in_orth_basis(fock, x)
            C_per_k.append(c.astype(complex))
            eps_per_k.append(e)
        occ_per_k, fermi_level, entropy = _occupations_from_eps(eps_per_k)
        D_per_k = _density_matrices_per_k(C_per_k, occ_per_k)
        D_real = real_space_density_from_kpoints_fractional(
            C_per_k, occ_per_k, kmesh, cells,
        )
        D_real_prev = D_per_k_prev = None

    for iter_idx in range(1, int(opts.max_iter) + 1):
        if damper is not None:
            damping = damper.alpha
        diis_active = use_diis and iter_idx >= diis_start_iter
        # Density damping on D_real (block-wise) when available -- but
        # skip it once DIIS is driving convergence.
        D_used = D_real
        D_used_per_k = [D.copy() for D in D_per_k]
        if iter_idx > 1 and damping > 0.0 and not diis_active:
            D_used = _damp_lattice_matrix(D_real, D_real_prev, damping)
            if D_per_k_prev is not None:
                D_used_per_k = [
                    damping * D_prev + (1.0 - damping) * D_cur
                    for D_cur, D_prev in zip(D_per_k, D_per_k_prev)
                ]

        # --- Fock at every k
        if single_gamma_kmesh:
            F_k_list = _build_single_gamma_molecular_limit_fock(
                basis,
                system,
                D_used,
                Hcore_k_list[0],
                omega=float(omega),
                lattice_opts=lat_opts,
                grid_shape=grid_shape_t,
                origin=origin,
                spacing_bohr=spacing_bohr,
                j_build=gamma_j_build,
            )
        else:
            F_k_list = _build_multi_k_fock(D_used, D_used_per_k)
        F_k_list = _hermitize_k_matrices(F_k_list)

        # --- Per-cell electronic energy + per-k error vectors.
        # When smearing is active, D(k) is built from fractional
        # occupations; otherwise it's the standard 2.C_occ.C_occ+.
        E_elec = 0.0
        grad_norm_sum = 0.0
        error_k_list: List[np.ndarray] = []
        for idx in range(n_k):
            D_k = D_used_per_k[idx]
            H_k = Hcore_k_list[idx]
            F_k = F_k_list[idx]
            w = float(weights[idx])
            E_elec += w * 0.5 * np.real(np.trace(D_k @ (H_k + F_k)))
            S_k = S_k_list[idx]
            FDS = F_k @ D_k @ S_k
            grad = FDS - FDS.conj().T
            error_k_list.append(grad)
            grad_norm_sum += w * float(np.linalg.norm(grad))
        # Madelung-leak correction (v0.6.1). Uses the g=0 real-space
        # density block and overlap; both are the unit-cell quantities.
        D_g0 = np.asarray(_g0_block(D_used))
        S_g0 = np.asarray(_g0_block(S_lat))
        E_madelung_fix = (
            0.0
            if slab_mode
            or (
                single_gamma_kmesh
                and lat_opts.coulomb_method == CoulombMethod.EWALD_3D
            )
            else _madelung_energy_correction_for_lat(D_g0, S_g0, system, lat_opts)
        )
        E_total = float(E_elec) + e_nuc + E_madelung_fix

        # Free energy A = E - T.S is the variational quantity under
        # smearing. Convergence is checked on A (which equals E when
        # T = 0 since entropy = 0).
        free_energy = E_total - smearing_T * entropy

        dE = free_energy - E_prev
        # Divergence detection (v0.6.2).
        check_scf_divergence(
            "run_rhf_periodic_multi_k_ewald3d",
            iter_idx,
            free_energy,
            grad_norm_sum,
            dE,
        )
        scf_trace.append(
            SCFIteration(
                iter=iter_idx,
                energy=float(free_energy),
                delta_e=float(dE if iter_idx > 1 else 0.0),
                grad_norm=float(grad_norm_sum),
                diis_subspace=(accel.subspace_size if accel is not None else 0),
            )
        )
        plog.iteration(
            iter_idx,
            energy=float(free_energy),
            dE=float(dE if iter_idx > 1 else 0.0),
            grad=float(grad_norm_sum),
            diis=(accel.subspace_size if accel is not None else 0),
        )
        # Per-iter perf row -- populated when a perf_log() context is
        # active (no-op otherwise via the active_tracker() check).
        from .perf import active_tracker as _active_tracker

        _pt = _active_tracker()
        if _pt is not None:
            _pt.add_scf_iter(
                iter=iter_idx,
                energy=float(free_energy),
                dE=float(dE if iter_idx > 1 else 0.0),
                grad=float(grad_norm_sum),
                diis=int(accel.subspace_size if accel is not None else 0),
                wall_s=time.perf_counter() - t_scf_start_perf,
            )

        converged = (
            iter_idx > 1
            and abs(dE) < float(opts.conv_tol_energy)
            and grad_norm_sum < float(opts.conv_tol_grad)
        )

        # Phase C1c gate. When the quadratic fallback is active, take
        # a per-k Newton step in MO space -- DIIS and level shift are
        # bypassed because the trust-region cap on κ is already a
        # complete update mechanism.
        in_quadratic_phase = (
            quadratic_fallback_iter > 0 and iter_idx > quadratic_fallback_iter
        )

        if in_quadratic_phase:
            from .quadratic_scf import quadratic_step

            new_C_per_k: List[np.ndarray] = []
            new_eps_per_k: List[np.ndarray] = []
            for idx in range(n_k):
                C_k, eps_k = quadratic_step(
                    F_k_list[idx],
                    C_per_k[idx],
                    eps_per_k[idx],
                    n_occ,
                    shift=quadratic_fallback_shift,
                    max_step=quadratic_fallback_max_step,
                )
                new_C_per_k.append(C_k)
                new_eps_per_k.append(eps_k)
            C_per_k = new_C_per_k
            eps_per_k = new_eps_per_k
        else:
            # --- SCF-accelerator extrapolation: feed the per-k iterate
            #     and replace F_k_list with the extrapolated Fock when
            #     active. Convergence criteria above use the *pre-
            #     extrapolation* pair so dE / grad_norm_sum still report
            #     raw SCF progress. C_per_k / eps_per_k from the
            #     previous diagonalisation give KDIIS its per-k
            #     canonical-MO basis.
            if accel is not None:
                F_ex_list = accel.extrapolate_rhf(
                    F_k_list,
                    error_k_list=error_k_list,
                    density_k_list=D_used_per_k,
                    energy=E_total,
                    mo_coeffs_k_list=C_per_k,
                    n_occ=n_occ,
                    weights=list(weights),
                    cells=cells,
                    kpoints=list(k_points),
                )
                if diis_active:
                    F_k_list = F_ex_list

            # --- Apply Saunders-Hillier level shift per k:
            #     F'(k) = F(k) + b . S(k) - (b/2) . S(k) . D(k) . S(k)
            # Per-iter shift: schedule.at(iter) when supplied, else
            # the static opts.level_shift.
            if level_shift_schedule is not None:
                level_shift_b = level_shift_schedule.at(iter_idx)
            else:
                level_shift_b = level_shift_static
            if level_shift_b != 0.0:
                F_for_diag: List[np.ndarray] = []
                for idx in range(n_k):
                    D_k = D_used_per_k[idx]
                    S_k = S_k_list[idx]
                    F_shift = (
                        F_k_list[idx]
                        + level_shift_b * S_k
                        - (level_shift_b / 2.0) * (S_k @ D_k @ S_k)
                    )
                    F_shift = 0.5 * (F_shift + F_shift.conj().T)
                    F_for_diag.append(F_shift)
            else:
                F_for_diag = F_k_list

            # --- Diagonalize F(k) at every k -> new C(k), e(k)
            new_C_per_k = []
            new_eps_per_k = []
            for idx in range(n_k):
                C_k, eps_k = _diag_in_orth_basis(F_for_diag[idx], X_k_list[idx])
                new_C_per_k.append(C_k)
                new_eps_per_k.append(eps_k)
            # --- MOM: reorder so MOM-selected occupied columns sit
            # first, otherwise `[:, :n_occ]` slicing downstream picks
            # the wrong subspace. At iter 1 there's no previous to
            # compare against, so fall through to Aufbau (default).
            if use_mom and C_prev_occ_per_k is not None:
                for idx in range(n_k):
                    C_k = new_C_per_k[idx]
                    eps_k = new_eps_per_k[idx]
                    S_k = S_k_list[idx]
                    sel = _mom_select(
                        C_k,
                        S_k,
                        C_prev_occ_per_k[idx],
                        n_occ,
                        eps_new=eps_k,
                    )
                    n_kept_idx = C_k.shape[1]
                    virt_mask = np.ones(n_kept_idx, dtype=bool)
                    virt_mask[sel] = False
                    virt_sel = np.where(virt_mask)[0]
                    # Energy-sort the virtual block too (cosmetic).
                    virt_sel = virt_sel[np.argsort(np.real(eps_k[virt_sel]))]
                    order = np.concatenate([sel, virt_sel])
                    new_C_per_k[idx] = C_k[:, order]
                    new_eps_per_k[idx] = eps_k[order]
            C_per_k = new_C_per_k
            eps_per_k = new_eps_per_k

        # --- Re-determine occupations on the new eigenvalues. Needed
        # at every iteration when smearing is active because mu moves
        # with the band structure; for the Aufbau (T=0) path this is
        # cheap and bit-identical to before.
        occ_per_k, fermi_level, entropy = _occupations_from_eps(eps_per_k)

        # --- Rebuild D_real from the new MOs and occupations.
        if _fractional_density_needed(occ_per_k):
            D_real_new = real_space_density_from_kpoints_fractional(
                C_per_k,
                occ_per_k,
                kmesh,
                cells,
            )
        else:
            D_real_new = real_space_density_from_kpoints(
                C_per_k,
                [n_occ] * n_k,
                kmesh,
                cells,
            )
        if single_gamma_kmesh:
            D_real_new = _localize_single_gamma_density(
                D_real_new,
                C_per_k[0],
                occ_per_k[0],
            )
        D_new_per_k = _density_matrices_per_k(C_per_k, occ_per_k)

        # --- ODA: build F at the post-diag density (D_naive) and
        # do a 1-D line search to pick the optimal mix between D_used
        # (the F-input density this iter) and D_real_new.
        if use_oda and not in_quadratic_phase:
            if single_gamma_kmesh:
                F_naive_k_list = _build_single_gamma_molecular_limit_fock(
                    basis,
                    system,
                    D_real_new,
                    Hcore_k_list[0],
                    omega=float(omega),
                    lattice_opts=lat_opts,
                    grid_shape=grid_shape_t,
                    origin=origin,
                    spacing_bohr=spacing_bohr,
                    j_build=gamma_j_build,
                )
            else:
                F_naive_k_list = _build_multi_k_fock(
                    D_real_new, D_new_per_k
                )
            F_naive_k_list = _hermitize_k_matrices(F_naive_k_list)
            oda_step = _compute_oda_lambda(
                D_used,
                D_real_new,
                F_k_list,
                F_naive_k_list,
                [np.asarray(k) for k in k_points],
                weights,
                trust_lambda_max=oda_trust_lambda_max,
            )
            # Mix D_used (in place) with D_real_new by l.
            _oda_mix(D_used, D_real_new, oda_step.lam)
            D_real_prev = D_real  # the prior unmixed density
            D_per_k_prev = [D.copy() for D in D_per_k]
            D_per_k = [
                (1.0 - oda_step.lam) * D_old + oda_step.lam * D_new
                for D_old, D_new in zip(D_used_per_k, D_new_per_k)
            ]
            D_real = D_used  # the ODA-mixed result is the new D
            plog.info(
                f"  ODA: l = {oda_step.lam:.4f}  "
                f"(g0 = {oda_step.g0:+.3e}, g1 = {oda_step.g1:+.3e}, "
                f"{'interior' if oda_step.quadratic_minimum else 'clamped'})"
            )
        else:
            D_real_prev = D_used
            D_per_k_prev = [D.copy() for D in D_used_per_k]
            D_real = D_real_new
            D_per_k = D_new_per_k

        # --- Snapshot per-k occupied subspace for MOM at the next iter.
        if use_mom:
            C_prev_occ_per_k = [
                np.asarray(C_per_k[idx][:, :n_occ]).copy() for idx in range(n_k)
            ]

        if damper is not None:
            damper.update(free_energy)
        E_prev = free_energy
        if converged:
            break

    # ---- Final self-consistency pass on the converged D ------------------
    if converged:
        if single_gamma_kmesh:
            F_k_list = _build_single_gamma_molecular_limit_fock(
                basis,
                system,
                D_real,
                Hcore_k_list[0],
                omega=float(omega),
                lattice_opts=lat_opts,
                grid_shape=grid_shape_t,
                origin=origin,
                spacing_bohr=spacing_bohr,
                j_build=gamma_j_build,
            )
        else:
            F_k_list = _build_multi_k_fock(D_real, D_per_k)
        F_k_list = _hermitize_k_matrices(F_k_list)
        E_elec = 0.0
        final_C_per_k: List[np.ndarray] = []
        final_eps_per_k: List[np.ndarray] = []
        for idx in range(n_k):
            C_k, eps_k = _diag_in_orth_basis(F_k_list[idx], X_k_list[idx])
            final_C_per_k.append(C_k)
            final_eps_per_k.append(eps_k)
        C_per_k = final_C_per_k
        eps_per_k = final_eps_per_k
        # Re-determine occupations + Fermi level on the final spectrum.
        occ_per_k, fermi_level, entropy = _occupations_from_eps(eps_per_k)
        _fractional_final = _fractional_density_needed(occ_per_k)
        if _fractional_final:
            D_real_final = real_space_density_from_kpoints_fractional(
                C_per_k,
                occ_per_k,
                kmesh,
                cells,
            )
        else:
            D_real_final = real_space_density_from_kpoints(
                C_per_k,
                [n_occ] * n_k,
                kmesh,
                cells,
            )
        if single_gamma_kmesh:
            D_real_final = _localize_single_gamma_density(
                D_real_final,
                C_per_k[0],
                occ_per_k[0],
            )
        D_real = D_real_final
        for idx in range(n_k):
            C_k = C_per_k[idx]
            if _fractional_final:
                D_k = (C_k * occ_per_k[idx][None, :].astype(complex)) @ C_k.conj().T
            else:
                C_occ = C_k[:, :n_occ]
                D_k = 2.0 * (C_occ @ C_occ.conj().T)
            w = float(weights[idx])
            E_elec += (
                w * 0.5 * np.real(np.trace(D_k @ (Hcore_k_list[idx] + F_k_list[idx])))
            )
        # Madelung-leak correction (v0.6.1).
        D_g0 = np.asarray(_g0_block(D_real))
        S_g0 = np.asarray(_g0_block(S_lat))
        E_madelung_fix = (
            0.0
            if slab_mode
            or (
                single_gamma_kmesh
                and lat_opts.coulomb_method == CoulombMethod.EWALD_3D
            )
            else _madelung_energy_correction_for_lat(D_g0, S_g0, system, lat_opts)
        )
        E_total = float(E_elec) + e_nuc + E_madelung_fix

    free_energy = E_total - smearing_T * entropy
    plog.converged(n_iter=iter_idx, energy=E_total, converged=converged)

    from .eigs_preflight import _copy_lattice_opts

    return PeriodicRHFMultiKEwaldResult(
               restart_mesh=tuple(kmesh.mesh),
               restart_kpoints=np.asarray(kmesh.kpoints).copy(), restart_weights=np.asarray(kmesh.weights).copy(),
               guess_selection=periodic_result_selection(system, opts.initial_guess, restarted=initial_density_k is not None),
               restart_basis=basis, restart_lattice=np.asarray(system.lattice).copy(),
        used_kpoint_symmetry_unfolding=(
            corrected_exchange is not None
            and corrected_exchange.unfolding is not None
        ),
        energy=E_total,
        e_electronic=float(E_elec),
        e_nuclear=e_nuc,
        n_iter=iter_idx,
        converged=converged,
        mo_energies=eps_per_k,
        mo_coeffs=C_per_k,
        fock=F_k_list,
        overlap=S_k_list,
        hcore=Hcore_k_list,
        density=D_real,
        density_k=(_density_matrices_per_k(C_per_k, occ_per_k) if converged
                   else [d.copy() for d in D_per_k]),
        scf_trace=scf_trace,
        omega=float(omega),
        grid_shape=grid_shape_t,
        smearing_temperature=smearing_T,
        fermi_level=float(fermi_level),
        entropy=float(entropy),
        free_energy=float(free_energy),
        occupations=[np.asarray(o, dtype=float) for o in occ_per_k],
        effective_lattice_opts=_copy_lattice_opts(lat_opts),
    )
