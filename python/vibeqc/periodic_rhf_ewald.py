"""Phase 12e-c-4b: Γ-point periodic RHF SCF driver using the
composed Ewald-3D Coulomb dispatch.

Replaces the ``DIRECT_TRUNCATED`` J build used by the existing
C++ ``run_rhf_periodic_gamma`` driver with the w-invariant composed
J from :func:`vibeqc.build_j_ewald_3d` (Phase 12e-c-4a). Exchange K
stays on the full-range real-space builder
(:func:`vibeqc.build_jk_gamma_molecular_limit` with w = 0) because
K inherits its locality from the density matrix rather than the
Coulomb operator -- standard periodic-HF practice.

What this driver does
---------------------

At every SCF iteration:

    F  =  H_core  +  J_ewald(w, D)  -  1/2 K(D)

with H_core = T + V (both periodic one-electron integrals, Bloch-
summed at Γ), S the Γ-point overlap, and canonical-orthogonalisation
replacing symmetric orthogonalisation so that linearly-dependent
bases converge rather than crash. Convergence check is identical to
the C++ driver: ``|ΔE| < conv_tol_energy`` and
``‖F D S - S D F‖_F < conv_tol_grad``.

Scope (this commit)
-------------------

* **Γ-only** -- single k-point at (0, 0, 0). Multi-k EWALD_3D is a
  follow-up (12e-c-4c) that needs the long-range J FFT generalised
  across Bloch-summed densities.
* **Molecular-limit cells** -- the cell has to be big enough that
  the density doesn't overlap its periodic image through the box
  boundary. For real bulk cells (tight unit cells with cell-to-cell
  density overlap) the FFT density grid needs a multi-cell sum, also
  deferred to 12e-c-4c.
* **DIIS supported** via ``options.use_diis = True`` (the default on
  :class:`PeriodicRHFOptions`). Pure-Python Pulay-DIIS matching the
  C++ molecular implementation (same subspace size, same error-vector
  ``F D S - S D F``); typically halves iteration counts on
  molecular-limit SCF cases vs plain damping.

Validation targets
------------------

* **w-invariance**: total SCF energy must be the same (to ~µHa) for
  w in [0.3, 2.0] when applied to the same system.
* **Cross-check against DIRECT_TRUNCATED at molecular limit**: for a
  molecule in a ~30 bohr vacuum box, the EWALD_3D and DIRECT_TRUNCATED
  drivers must agree to within Makov-Payne finite-box corrections
  (~O(1/L^3), sub-mHa at this box size).

These are exercised in ``tests/test_periodic_rhf_ewald.py``.
"""

from __future__ import annotations

from .guess import periodic_result_selection

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from ._vibeqc_core import (
    apply_level_shift,
    BasisSet,
    bloch_sum,
    build_jk_gamma_molecular_limit,
    compute_kinetic_lattice,
    compute_nuclear_lattice,
    compute_overlap_lattice,
    CoulombMethod,
    EwaldOptions,
    InitialGuess,
    LatticeSumOptions,
    level_shift_at_iter,
    LevelShiftDensity,
    nuclear_repulsion_per_cell,
    PeriodicRHFOptions,
    PeriodicSystem,
    SCFIteration,
)
from .ewald_composed import make_ewald_3d_gamma_j_builder
from .ewald_composed_slab import make_slab_ewald_2d_gamma_j_builder
from .ewald_j import auto_grid
from .guess import initial_density_closed_shell
from .linear_dependence import scf_preflight_overlap_check
from .options_dump import dump_active_settings
from .periodic_scf_accelerators import DynamicDamping, PeriodicSCFAccelerator
from .progress import ProgressLogger, resolve_progress
from .scf_divergence import check_scf_divergence

__all__ = [
    "PeriodicRHFEwaldResult",
    "run_rhf_periodic_gamma_ewald2d",
    "run_rhf_periodic_gamma_ewald3d",
]


@dataclass
class PeriodicRHFEwaldResult:
    """Result of :func:`run_rhf_periodic_gamma_ewald3d`.

    Structurally parallel to :class:`PeriodicRHFResult` but carries
    Ewald-specific bookkeeping (``omega``, ``grid_shape``) so callers
    can reproduce the calculation or sweep parameters. All matrices
    are at Γ-point (no Bloch sums).
    """

    energy: float
    e_electronic: float
    e_nuclear: float
    n_iter: int
    converged: bool
    mo_energies: np.ndarray
    mo_coeffs: np.ndarray
    density: np.ndarray
    fock: np.ndarray
    overlap: np.ndarray
    scf_trace: List[SCFIteration] = field(default_factory=list)
    # Ewald-specific:
    omega: float = 0.0
    grid_shape: Tuple[int, int, int] = (0, 0, 0)

    restart_basis: object = None
    restart_lattice: object = None
    guess_selection: object = None

    restart_kpoints: object = None
    restart_weights: object = None


def _canonical_orthogonalizer(
    S: np.ndarray,
    threshold: float = 1e-7,
    *,
    normalize_diag_first: bool = True,
) -> Tuple[np.ndarray, int]:
    """Return the (n_bf, n_kept) rectangular orthogonaliser X such
    that ``X^T S X ≈ I_{n_kept}``, with any S-eigenvalue below
    ``threshold`` dropped (Löwdin canonical orthogonalisation).

    Parameters
    ----------
    S
        ``(n, n)`` symmetric overlap matrix. Real-valued; for the
        complex Hermitian (multi-k) case use
        :func:`_canonical_orthogonalizer_complex` (in
        ``periodic_rhf_multi_k_ewald``).
    threshold
        Eigenvalues below this are projected out. ``1e-7`` matches
        the PySCF default.
    normalize_diag_first
        When ``True`` (default, PySCF-aligned), pre-conditioning
        scales rows / columns of S by ``1 / sqrt(diag(S))`` before
        the eigendecomposition; the eigenvalues then live on a
        unit-diagonal matrix where the threshold has its intended
        meaning. After diagonalisation the normalisation is undone
        so X still satisfies ``X^T S X = I``. Mirrors PySCF's
        ``pyscf.scf.addons.canonical_orth_`` (and the comment
        "Ensure the basis functions are normalized -- symmetry-adapted
        ones are not!").

        When ``False`` (vibe-qc legacy <= v0.6), eigh runs on the
        bare S. Cheaper, but the threshold is harder to interpret on
        bases whose diagonal entries vary by orders of magnitude.
        Kept available for cross-checking against pre-v0.7 numbers.

    Returns
    -------
    X : np.ndarray of shape ``(n_bf, n_kept)``
        Rectangular orthogonaliser.
    n_kept : int
        Number of S-eigenvectors retained.

    References
    ----------
    Löwdin, P.-O. *Adv. Quantum Chem.* **5**, 185 (1970) --
    canonical orthogonalisation in molecular quantum chemistry.

    Lykos, P. G., Schmeising, H. N. *J. Chem. Phys.* **35**, 288
    (1961) -- early analysis of the AO overlap eigenvalue spectrum.
    """
    if normalize_diag_first:
        # PySCF-aligned path. Diagonal scaling: D = diag(1/sqrt(S_ii)).
        # Snorm = D . S . D has unit diagonal so the threshold operates
        # on a "shape-only" overlap and is robust to per-AO scale.
        d = np.asarray(S).diagonal()
        # Guard against zero or negative diagonals (shouldn't happen
        # for a real basis, but the preflight check might let through
        # numerical near-zero entries).
        if np.any(d <= 0):
            # Fall back to raw-S path for the affected rows; canonical
            # orth on the remainder.
            normalize_diag_first = False
        else:
            scale = 1.0 / np.sqrt(d)
            Snorm = S * np.outer(scale, scale)
            eigvals, eigvecs = np.linalg.eigh(Snorm)
            mask = eigvals >= threshold
            n_kept = int(mask.sum())
            if n_kept == 0:
                raise RuntimeError(
                    "run_rhf_periodic_gamma_ewald3d: no overlap eigenvalue "
                    f"above threshold {threshold:.1e} (after sqrt-diag "
                    "normalisation); basis is fully linearly dependent"
                )
            kept_vals = eigvals[mask]
            kept_vecs = eigvecs[:, mask]
            X_norm = kept_vecs / np.sqrt(kept_vals)
            # Plug normalisation back in: X = D . X_norm so X^T S X = I.
            X = scale[:, None] * X_norm
            return X, n_kept

    # Legacy raw-S path -- kept for cross-checking pre-v0.7 numbers.
    eigvals, eigvecs = np.linalg.eigh(S)
    mask = eigvals >= threshold
    n_kept = int(mask.sum())
    if n_kept == 0:
        raise RuntimeError(
            "run_rhf_periodic_gamma_ewald3d: no overlap eigenvalue "
            f"above threshold {threshold:.1e}; basis is fully "
            "linearly dependent"
        )
    kept_vals = eigvals[mask]
    kept_vecs = eigvecs[:, mask]
    X = kept_vecs / np.sqrt(kept_vals)
    return X, n_kept


# Calibrated 2026-06-13 to catch the basis-overlap collapse mode, using the
# largest cross-cell AO overlap |S_muν(L!=0)| as the molecular-limit-breakdown
# signal:
#   LiH  rocksalt (Li 2s/2p a=0.048, very diffuse): overlap 0.41 ->
#        |ΔE| = 1.97 Ha                                          (buggy)
#   MgO  rocksalt: overlap 0.19 -> below this overlap-specific guard
#   NaCl rocksalt: overlap 0.12 -> below this overlap-specific guard
#   H2 / dilute boxes: overlap <= 0.006 -> exact                 (fine)
# The Γ-only driver drops the cross-cell density-matrix blocks (the
# molecular-limit convention D(g!=0)=0); that is invalid precisely when a
# basis function overlaps its own periodic image. The criterion is
# BASIS-dependent, not geometric: LiH and MgO have near-identical nearest-
# image contact (3.86 vs 3.98 bohr) but opposite overlap status because Li's
# STO-3G valence is far more diffuse. We refuse above the mid-gap overlap.
# This is not a claim that compact condensed cells are absolute-energy parity
# cases for the molecular-limit EWALD bridge: the regression-suite PySCF GDF
# contract must use the public GDF routes. The user-facing EWALD_3D route is
# retired (jk_method drops 'fft_poisson'); this guard keeps the known
# overlap-collapse regime unreachable through the internal driver too
# (CLAUDE.md Sec.7).
_DENSE_IONIC_MAX_XCELL_OVERLAP = 0.3


def _max_cross_cell_ao_overlap(
    system: PeriodicSystem, basis: BasisSet, lat_opts: LatticeSumOptions
) -> float:
    """Largest absolute AO overlap between a home-cell basis function and a
    basis function in any periodic image cell (L != 0).

    This is the direct measure of molecular-limit breakdown: ~0 for an
    isolated molecule in a vacuum-padded box, growing as periodic images of
    the basis overlap. Evaluated at the lattice-sum cutoff the SCF will use;
    the home cell (L = 0) is excluded -- the molecular limit treats it exactly.
    """
    overlaps = compute_overlap_lattice(basis, system, lat_opts)
    worst = 0.0
    for cell, block in zip(overlaps.cells, overlaps.blocks):
        if np.array_equal(np.asarray(cell.index), (0, 0, 0)):
            continue
        worst = max(worst, float(np.abs(np.asarray(block)).max()))
    return worst


def _refuse_if_dense_ionic(
    system: PeriodicSystem,
    basis: BasisSet,
    lat_opts: LatticeSumOptions,
    allow_dense_ionic: bool,
) -> None:
    """Fail closed when periodic images of the basis overlap enough to
    invalidate the molecular-limit density convention (CLAUDE.md Sec.7).

    The Γ-only EWALD_3D drivers return a ~2 Ha-wrong energy in that regime
    (e.g. LiH rocksalt / STO-3G). Rather than hand back a wrong number, raise
    and name the correct routes.

    ``allow_dense_ionic=True`` is an internal escape hatch for exercising the
    driver's *mechanics* (truncation, basis filtering, grid) on such a cell;
    the energy it then returns is NOT reliable. It is deliberately not wired
    to the user-facing dispatch, which has retired EWALD_3D entirely.
    """
    if allow_dense_ionic or getattr(system, "dim", 3) != 3:
        return
    overlap = _max_cross_cell_ao_overlap(system, basis, lat_opts)
    if overlap > _DENSE_IONIC_MAX_XCELL_OVERLAP:
        raise ValueError(
            "Γ-only EWALD_3D is retired for dense ionic crystals whose "
            "periodic images overlap: the largest cross-cell AO overlap is "
            f"{overlap:.2f} (> {_DENSE_IONIC_MAX_XCELL_OVERLAP}), where the "
            "molecular-limit density convention D(g!=0)=0 gives a ~2 Ha-wrong "
            "energy (e.g. LiH rocksalt / STO-3G vs GDF). Use jk_method='gdf' "
            "(default), 'bipole', or 'gpw'. See "
            "docs/user_guide/periodic_methods.md."
        )


def run_rhf_periodic_gamma_ewald3d(
    system: PeriodicSystem,
    basis: BasisSet,
    options: Optional[PeriodicRHFOptions] = None,
    *,
    omega: Optional[float] = None,
    grid_shape: Optional[Union[Tuple[int, int, int], int]] = None,
    origin: Optional[Sequence[float]] = None,
    spacing_bohr: float = 0.3,
    linear_dep_threshold: float = 1e-7,
    canonical_orth_normalize_diag_first: bool = True,
    auto_optimize_truncation: bool = True,
    allow_dense_ionic: bool = False,
    progress: Union[bool, ProgressLogger, None] = None,
    verbose: Optional[int] = None,
) -> PeriodicRHFEwaldResult:
    """Γ-point closed-shell periodic RHF SCF with Ewald-3D Coulomb.

    The Coulomb matrix is built via the composed Ewald split
    ``J_ewald(w, D) = J_SR(w) + J_LR(w)`` from
    :func:`build_j_ewald_3d`. Exchange K uses the full-range
    real-space builder (``build_jk_gamma_molecular_limit`` at w = 0);
    in periodic HF, K's real-space decay comes from the density
    matrix, not the Coulomb operator, so an Ewald split on K is
    neither needed nor used by standard periodic codes.

    Parameters
    ----------
    system
        :class:`PeriodicSystem`.
    basis
        AO basis for the unit cell.
    options
        Optional :class:`PeriodicRHFOptions` controlling max_iter,
        damping, convergence tolerances. ``use_diis`` is currently
        ignored -- plain damping only; DIIS is a planned follow-up.
    omega
        Ewald splitting parameter. Result is w-independent at
        convergence to ~ µHa across w in [0.3, 2.0].
    grid_shape, origin, spacing_bohr
        FFT-Poisson grid controls forwarded to :func:`build_j_ewald_3d`.
    linear_dep_threshold
        Overlap-eigenvalue threshold below which AO directions are
        projected out of the SCF. Default 1e-7 matches the molecular
        SCF drivers.
    allow_dense_ionic
        Internal escape hatch. The driver fails closed when periodic images
        of the basis overlap (max cross-cell AO overlap above ~0.3), where
        the molecular-limit density convention gives a ~2 Ha-wrong energy
        (CLAUDE.md Sec.7); use ``jk_method='gdf'``/``'bipole'``/``'gpw'``
        instead. Set ``True`` only to exercise the driver's mechanics on
        such a cell -- the returned energy is then NOT reliable.

    Returns
    -------
    :class:`PeriodicRHFEwaldResult`.
    """
    opts = options if options is not None else PeriodicRHFOptions()
    from .guess import select_periodic_driver_guess
    selection = select_periodic_driver_guess(
        system, opts, route="ewald", method="RHF", driver="run_rhf_periodic_gamma_ewald3d",
    )
    lat_opts = opts.lattice_opts

    # ---- Force EWALD_3D gauge ----
    if lat_opts.coulomb_method != CoulombMethod.EWALD_3D:
        lat_opts.coulomb_method = CoulombMethod.EWALD_3D
    # Tight-cell cutoff floor: small 3D cells need a large real-space cutoff
    # for the EWALD_3D lattice sums to converge (and for S(Γ) to be PSD).
    # Gated on auto_optimize_truncation -- a user who explicitly opts out
    # gets their cutoff verbatim, so the linear-dependence preflight below
    # can refuse a non-PSD overlap rather than have it silently inflated
    # away (CLAUDE.md Sec.7).
    if auto_optimize_truncation and lat_opts.cutoff_bohr < 18.0 and system.dim == 3:
        V_cell = float(abs(np.linalg.det(np.asarray(system.lattice))))
        if V_cell < 1000.0:
            lat_opts.cutoff_bohr = max(lat_opts.cutoff_bohr, 18.0)
            lat_opts.nuclear_cutoff_bohr = max(lat_opts.nuclear_cutoff_bohr, 25.0)

    # Fail closed if periodic images of the basis overlap enough to make the
    # molecular-limit energy wrong (CLAUDE.md Sec.7); evaluated at the resolved
    # lattice cutoff. GDF/BIPOLE/GPW are correct there.
    _refuse_if_dense_ionic(system, basis, lat_opts, allow_dense_ionic)

    plog = resolve_progress(progress, verbose=verbose)

    # ---- Ewald splitting parameter w -- must match nuclear a ----
    # When not specified explicitly, use the default Ewald splitting
    # a = 2.8.V^{-1/3} (matching the BIPOLE driver; CRYSTAL convention).  The old
    # geometry-based formula √(-log tol)/cutoff had no relationship to
    # the nuclear Ewald a and broke gauge consistency.
    _ewald_tol = getattr(opts, "ewald_tolerance", 1e-12)
    _cutoff = getattr(opts, "ewald_cutoff_bohr", lat_opts.nuclear_cutoff_bohr)
    if omega is None:
        # omega not explicitly set by caller; check options, then CRYSTAL default.
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

    if grid_shape is None:
        grid_shape_t = auto_grid(lat, spacing_bohr)
    elif isinstance(grid_shape, int):
        grid_shape_t = (grid_shape, grid_shape, grid_shape)
    else:
        grid_shape_t = tuple(int(x) for x in grid_shape)
    plog.info(
        f"RHF Gamma EWALD_3D / omega = {float(omega):.3f}, "
        f"FFT grid {grid_shape_t[0]}x{grid_shape_t[1]}x{grid_shape_t[2]}"
    )
    plog.info(f"basis: {basis.name}  ({basis.nbasis} BFs / {basis.nshells} shells)")
    # Active-settings dump -- every overridable knob gets printed so
    # the SCF log self-documents its inputs. See vibeqc.options_dump.
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
                    "canonical_orth_normalize_diag_first": True,
                },
            ),
        ],
    )
    if plog.level >= 5:
        from .scf_log import format_basis_summary

        plog.write_raw(format_basis_summary(basis))

    # Closed-shell check
    n_elec = system.n_electrons()
    if n_elec % 2 != 0:
        raise ValueError(
            "run_rhf_periodic_gamma_ewald3d: closed-shell RHF "
            f"requires even electron count; got {n_elec}"
        )
    if system.multiplicity != 1:
        raise ValueError(
            "run_rhf_periodic_gamma_ewald3d: closed-shell RHF requires "
            f"multiplicity=1; got {system.multiplicity}"
        )
    n_occ = n_elec // 2

    # ---- Auto-optimise lattice truncation (default ON) -------------------
    # See periodic_rks_ewald.py for the rationale.
    if auto_optimize_truncation and lat_opts.coulomb_method == CoulombMethod.EWALD_3D:
        from .eigs_preflight import (
            format_truncation_optimization_report,
            optimize_truncation,
        )

        opt_rep = optimize_truncation(system, basis, lattice_opts=lat_opts)
        if (
            opt_rep.n_evaluations > 1
            or opt_rep.optimized_lattice_opts.cutoff_bohr != lat_opts.cutoff_bohr
        ):
            plog.write_raw(format_truncation_optimization_report(opt_rep))
            if not opt_rep.converged:
                plog.warn(
                    "auto_optimize_truncation did not converge; SCF will run "
                    "with the last evaluated settings. Check the report and "
                    "consider vq.disambiguate_critical_overlap and / or "
                    "vq.make_basis(..., exp_to_discard=...)."
                )
            lat_opts = opt_rep.optimized_lattice_opts

    # ---- One-electron integrals at Γ --------------------------------------
    with plog.stage(
        "integrals_lattice", detail=f"S/T/V at cutoff {lat_opts.cutoff_bohr:.2f} bohr"
    ):
        S_lat = compute_overlap_lattice(basis, system, lat_opts)
        T_lat = compute_kinetic_lattice(basis, system, lat_opts)
        from .periodic_v_ne import compute_nuclear_lattice_dispatch

        V_lat = compute_nuclear_lattice_dispatch(basis, system, lat_opts)
    k_gamma = np.zeros(3)
    S = np.real(bloch_sum(S_lat, k_gamma))
    T = np.real(bloch_sum(T_lat, k_gamma))
    V = np.real(bloch_sum(V_lat, k_gamma))
    Hcore = T + V
    # Symmetrize against tiny imaginary drift from the Bloch sum.
    S = 0.5 * (S + S.T)
    Hcore = 0.5 * (Hcore + Hcore.T)

    # ---- Linear-dependence preflight on S(Γ) ------------------------------
    # PySCF only logs this at DEBUG verbosity; we surface it at INFO
    # because near-singular / non-PSD S in periodic SCF is much more
    # common than in molecular SCF (especially after the v0.7 periodic
    # AO image-summing) and silently truncating the basis via canonical
    # orthogonalisation can give plausible-but-wrong total energies.
    # Negative S eigenvalues (severity="critical") abort by default.
    scf_preflight_overlap_check(
        S,
        plog=plog,
        label="S(Γ)",
        basis=basis,
    )

    # ---- Canonical orthogonalisation --------------------------------------
    X, n_kept = _canonical_orthogonalizer(
        S,
        linear_dep_threshold,
        normalize_diag_first=canonical_orth_normalize_diag_first,
    )
    if n_occ > n_kept:
        raise RuntimeError(
            "run_rhf_periodic_gamma_ewald3d: canonical orthogonalisation "
            f"dropped too many directions (n_occ = {n_occ}, "
            f"n_kept = {n_kept}); loosen linear_dep_threshold or pick "
            "a less redundant basis."
        )

    use_davidson = getattr(opts, "use_davidson", False)
    dav_opts = getattr(opts, "davidson", None)
    dav_dim = getattr(opts, "davidson_min_dim", 100)
    use_dav = use_davidson and S.shape[0] >= dav_dim
    if use_dav and dav_opts is None:
        from vibeqc._vibeqc_core import DavidsonOptions

        dav_opts = DavidsonOptions()

    # ---- Nuclear repulsion per cell ---------------------------------------
    e_nuc = nuclear_repulsion_per_cell(system, lat_opts)

    # ---- Madelung-leak correction (v0.6.1 fix) ---------------------------
    # See vibeqc.madelung.madelung_energy_correction for the rationale.
    from .madelung import madelung_energy_correction

    # ---- Initial guess via the unified engine.
    # SAD is the right default for ionic / covalent insulators (NaCl,
    # MgO, Si) where Hcore underbinds the deep core states so badly
    # that DIIS can amplify the swing into +30 000 / -16 000 Ha
    # territory before recovering.
    def diagonalise(F):
        Fp = X.T @ F @ X
        Fp = 0.5 * (Fp + Fp.T)
        if use_dav and dav_opts is not None:
            from vibeqc._vibeqc_core import davidson_solve

            if dav_opts.n_eig == 0:
                dav_opts.n_eig = Fp.shape[0]
            if dav_opts.guess_vectors is not None:
                pass  # already set from previous iteration
            dres = davidson_solve(Fp, dav_opts)
            if not dres.converged:
                raise RuntimeError(
                    f"Davidson did not converge after {dres.n_iter} iters"
                )
            eps, Cp = dres.eigenvalues, dres.eigenvectors
            dav_opts.guess_vectors = Cp
        else:
            eps, Cp = np.linalg.eigh(Fp)
        C = X @ Cp
        return C, eps

    j_build = make_ewald_3d_gamma_j_builder(
        basis,
        system,
        omega=float(omega),
        lattice_opts=lat_opts,
        grid_shape=grid_shape_t,
        origin=origin,
        spacing_bohr=spacing_bohr,
    )

    guess = selection.effective
    seed_guess = InitialGuess.SAD if guess == InitialGuess.PATOM else guess
    D_engine = initial_density_closed_shell(
        system.unit_cell_molecule(),
        basis,
        n_occ,
        seed_guess,
        is_periodic=True,
        periodic_system=system,
        lattice_opts=lat_opts,
        # READ restart (Γ-only): prior g=0 cell density (pre-resolved from
        # read_from, or read + projected from read_path). Ignored unless READ.
        read_density=getattr(opts, "read_density", None),
        read_path=getattr(opts, "read_path", ""),
        overlap=S,
    )
    if D_engine is not None:
        plog.info(f"initial guess: {guess.name} (density via GuessEngine)")
        D = D_engine
        C0, eps0 = diagonalise(Hcore)  # MO trace for log symmetry
    else:
        plog.info(f"initial guess: {guess.name} (Hcore-diagonalise)")
        C0, eps0 = diagonalise(Hcore)
        D = 2.0 * C0[:, :n_occ] @ C0[:, :n_occ].T
        D = 0.5 * (D + D.T)
    if guess == InitialGuess.PATOM:
        # PATOM is one full-HF in-field step, independent of the target XC.
        J_seed = j_build(D)
        K_seed = np.asarray(build_jk_gamma_molecular_limit(
            basis, system, lat_opts, D, 0.0).K)
        C0, eps0 = diagonalise(Hcore + J_seed - 0.5 * K_seed)
        D = 2.0 * C0[:, :n_occ] @ C0[:, :n_occ].T
        D = 0.5 * (D + D.T)
    D_prev = D.copy()

    damping = float(opts.damping)
    if not (0.0 <= damping < 1.0):
        raise ValueError(
            f"run_rhf_periodic_gamma_ewald3d: damping must be in [0, 1); got {damping}"
        )

    # Dynamic damping (Zerner-Hehenberger 1979) -- adjusts a between
    # ``[dynamic_damping_min, dynamic_damping_max]`` based on the
    # iteration-to-iteration energy decrease. Static damping is the
    # ``opts.dynamic_damping = False`` special case.
    damper: Optional[DynamicDamping] = None
    if bool(getattr(opts, "dynamic_damping", False)):
        damper = DynamicDamping(
            initial_alpha=damping,
            alpha_min=float(getattr(opts, "dynamic_damping_min", 0.0)),
            alpha_max=float(getattr(opts, "dynamic_damping_max", 0.95)),
        )

    # SCF accelerator -- dispatches on opts.scf_accelerator. DIIS /
    # KDIIS / EDIIS / EDIIS_DIIS / ADIIS all share the
    # ``diis_start_iter`` activation gate; damping is suppressed once
    # the accelerator is active because damping + Fock-extrapolation
    # mixes badly.
    use_diis = bool(opts.use_diis)
    diis_start_iter = int(opts.diis_start_iter)
    accel: Optional[PeriodicSCFAccelerator] = (
        PeriodicSCFAccelerator(opts) if use_diis else None
    )

    # Saunders-Hillier level shift -- opts.level_shift is 0.0 by default
    # (no shift). ``b > 0`` raises virtual MO eigenvalues by ``b``
    # before each diagonalisation, suppressing occupied / virtual swaps
    # on small-gap insulators.
    level_shift = float(getattr(opts, "level_shift", 0.0))
    # Explicit per-iteration schedule (unified with the molecular
    # drivers). Empty ⇒ the constant/persistent `level_shift` above;
    # non-empty ⇒ resolved per iteration by the shared C++ helper.
    _ls_schedule = list(getattr(opts, "level_shift_schedule", None) or [])
    _ls_max_iter = int(opts.max_iter)

    # CRYSTAL-style FMIXING: linear blend with the previous iteration's
    # Fock matrix.  ``fock_mixing_value`` is the weight of the previous
    # Fock; 0.0 disables the mix, 0.3 reproduces CRYSTAL14's default
    # FMIXING 30, 0.5-0.7 helps tough ionic SCF (LiH, NaCl, MgO
    # primitive).  Applied AFTER DIIS extrapolation and BEFORE the
    # level-shift / diagonalisation step -- same order as the gdf
    # driver (see periodic_rhf_gdf.py).
    fock_mixing_value = float(getattr(opts, "fock_mixing", 0.0))
    if not (0.0 <= fock_mixing_value < 1.0):
        raise ValueError(
            "run_rhf_periodic_gamma_ewald3d: fock_mixing must be in [0, 1); "
            f"got {fock_mixing_value}"
        )
    if fock_mixing_value != 0.0:
        plog.info(
            "fock mixing: CRYSTAL FMIXING "
            f"{100.0 * fock_mixing_value:.1f}% "
            "(previous Fock matrix weight)"
        )
    F_prev_mixed: Optional[np.ndarray] = None

    # Phase C1c -- second-order ("quadratic") SCF fallback. Inactive
    # by default. When opts.quadratic_fallback_iter > 0 and the SCF
    # has run that many iterations without converging, switch from
    # diagonalizing F to a Newton step in MO space (κ_{ai} =
    # -F_{ai}^MO / (e_a - e_i + l); C_new = C_prev . exp(κ)). DIIS
    # and level shift are skipped during the quadratic phase.
    quadratic_fallback_iter = int(getattr(opts, "quadratic_fallback_iter", 0))
    quadratic_fallback_shift = float(getattr(opts, "quadratic_fallback_shift", 0.1))
    quadratic_fallback_max_step = float(
        getattr(opts, "quadratic_fallback_max_step", 0.1)
    )

    # ---- SCF loop ---------------------------------------------------------
    plog.banner("SCF (RHF Gamma, EWALD_3D)")
    plog.info("  iter         energy (Ha)            dE          ||[F,DS]||   DIIS")

    scf_trace: List[SCFIteration] = []
    result = PeriodicRHFEwaldResult(
                 restart_kpoints=np.zeros((1, 3)), restart_weights=np.ones(1),
                 guess_selection=periodic_result_selection(system, opts.initial_guess, restarted=False),
                 restart_basis=basis, restart_lattice=np.asarray(system.lattice).copy(),
        energy=0.0,
        e_electronic=0.0,
        e_nuclear=float(e_nuc),
        n_iter=0,
        converged=False,
        mo_energies=np.empty(0),
        mo_coeffs=np.empty((0, 0)),
        density=D.copy(),
        fock=np.empty((0, 0)),
        overlap=S,
        scf_trace=scf_trace,
        omega=float(omega),
        grid_shape=grid_shape_t,
    )

    E_prev = 0.0
    F_final = Hcore.copy()
    C_final = C0
    eps_final = eps0

    # Cache the iteration-invariant analytic-FT Hartree-J machinery once
    # (Bloch AO-pair FT, dense G-mesh, 4pi/G^2 kernel) so the SCF loop redoes
    # only the density contraction -- the GDF "build Lpq once" pattern; E2 in
    # docs/pbc_audit_2026-06.md. ``j_build(D)`` is bit-identical to a
    # per-iteration build_j_ewald_3d call (grid backend: uncached, unchanged).

    for iter_idx in range(1, int(opts.max_iter) + 1):
        # Dynamic-damping read: ``damper.alpha`` carries the a
        # selected by the previous iteration's energy step. On iter 1
        # this equals the initial a (the previous-iteration energy
        # has not been seen yet).
        if damper is not None:
            damping = damper.alpha
        diis_active = use_diis and iter_idx >= diis_start_iter
        # When DIIS is driving convergence, density damping is
        # unnecessary and actually slows things down by blurring
        # the density DIIS is trying to iterate. Match the C++
        # driver and skip damping once DIIS is active.
        D_used = (
            D
            if (iter_idx == 1 or damping == 0.0 or diis_active)
            else damping * D_prev + (1.0 - damping) * D
        )

        # J via composed Ewald-3D analytic FT, reusing the cached
        # iteration-invariant machinery (see j_build above).
        J = j_build(D_used)
        # K via full-range real-space builder (omega=0).  At Γ-only with
        # N_k = 1 this is the inverse-Bloch convention P(g) = P_Γ.d_{g,0}
        # -- density localized to the home cell, cross-cell P(h!=0) set
        # to zero (no information available without multi-k sampling).
        # The 2026-05-14 handover originally framed this as a builder
        # bug; the cutoff-convergence sweep on 2026-05-15 showed the
        # alternative "homogeneous-D" build diverges and the
        # molecular-limit kernel is in fact the correct Γ-only build.
        # Tight-ionic-crystal accuracy needs multi-k sampling.
        jk_full = build_jk_gamma_molecular_limit(
            basis,
            system,
            lat_opts,
            D_used,
            0.0,
        )
        K = np.asarray(jk_full.K)

        F = Hcore + J - 0.5 * K
        F = 0.5 * (F + F.T)

        # Closed-shell electronic energy: 1/2 tr[D (Hcore + F)]
        E_elec = 0.5 * float(np.einsum("ij,ij->", D_used, Hcore + F))

        # Madelung-leak correction (v0.6.1) -- disabled for EWALD_3D in
        # v0.7.0 since V_ne now dispatches to the Ewald path that's
        # gauge-aligned with J. See ``madelung_energy_correction_for_lat``
        # docstring + LiH dense-ionic decomposition in
        # ``examples/debug/lih_energy_decomposition.py``.
        if lat_opts.coulomb_method == CoulombMethod.EWALD_3D:
            E_madelung_fix = 0.0
        else:
            E_madelung_fix = madelung_energy_correction(
                D_used,
                S,
                system,
                nuclear_uses_ewald=False,
            )
        E_total = E_elec + float(e_nuc) + E_madelung_fix

        # Orbital-gradient norm
        FDS = F @ D_used @ S
        grad = FDS - FDS.T
        grad_norm = float(np.linalg.norm(grad))

        dE = E_total - E_prev
        # Divergence detection (v0.6.2 generalisation of v0.5.6 hook).
        check_scf_divergence(
            "run_rhf_periodic_gamma_ewald3d",
            iter_idx,
            E_total,
            grad_norm,
            dE,
        )
        converged = (
            iter_idx > 1
            and abs(dE) < float(opts.conv_tol_energy)
            and grad_norm < float(opts.conv_tol_grad)
        )
        scf_trace.append(
            SCFIteration(
                iter=iter_idx,
                energy=float(E_total),
                delta_e=float(dE if iter_idx > 1 else 0.0),
                grad_norm=float(grad_norm),
                diis_subspace=(accel.subspace_size if accel is not None else 0),
            )
        )
        plog.iteration(
            iter_idx,
            energy=float(E_total),
            dE=float(dE if iter_idx > 1 else 0.0),
            grad=float(grad_norm),
            diis=(accel.subspace_size if accel is not None else 0),
        )
        # Per-iter energy decomposition for cross-code parity comparison.
        E_kin_iter = float(np.einsum("ij,ij->", D_used, T))
        E_ne_iter = float(np.einsum("ij,ij->", D_used, V))
        E_J_iter = 0.5 * float(np.einsum("ij,ij->", D_used, J))
        E_K_iter = -0.5 * 0.5 * float(np.einsum("ij,ij->", D_used, K))
        plog.energy_decomposition(
            iter_idx,
            E_kin=E_kin_iter,
            E_ne=E_ne_iter,
            E_J=E_J_iter,
            E_K=E_K_iter,
            E_nuc=float(e_nuc),
            E_madelung=float(E_madelung_fix),
        )

        # Phase C1c gate: when the quadratic fallback is enabled and
        # we've passed its activation iteration, take a Newton step
        # in MO space instead of diagonalizing. DIIS extrapolation
        # and level shift are both skipped -- the Newton step is its
        # own update mechanism and mixing with DIIS undoes the
        # trust-region cap.
        in_quadratic_phase = (
            quadratic_fallback_iter > 0 and iter_idx > quadratic_fallback_iter
        )

        if in_quadratic_phase:
            from .quadratic_scf import quadratic_step

            C_new, eps_new = quadratic_step(
                F,
                C_final,
                eps_final,
                n_occ,
                shift=quadratic_fallback_shift,
                max_step=quadratic_fallback_max_step,
            )
        else:
            # SCF-accelerator extrapolation: feed the current
            # iterate's matrices into the active accelerator and
            # replace F with the extrapolated Fock for the
            # diagonalisation step. The convergence check above uses
            # the *pre-extrapolation* F so dE and grad_norm still
            # reflect the raw SCF progress. Pass C_final / eps_final
            # from the previous iteration (the C e pair that produced
            # the current density D from which F was built) -- these
            # are the canonical-MO inputs KDIIS needs to project the
            # orbital-rotation gradient.
            if accel is not None:
                F_ex = accel.extrapolate_rhf(
                    F,
                    error=grad,
                    density=D_used,
                    energy=E_total,
                    mo_coeffs=C_final,
                    mo_energies=eps_final,
                    n_occ=n_occ,
                )
                if diis_active:
                    F = F_ex

            # CRYSTAL-style FMIXING -- applied after DIIS, before level-
            # shift / diag.  Smooths Fock-matrix oscillations on tough
            # ionic SCF where the proper periodic J/K has two stationary
            # points (physical, over-bound).  Matches the gdf driver's
            # FMIXING order.
            if fock_mixing_value != 0.0:
                if F_prev_mixed is not None:
                    F_mixed = (
                        1.0 - fock_mixing_value
                    ) * F + fock_mixing_value * F_prev_mixed
                    F = 0.5 * (F_mixed + F_mixed.T)
                F_prev_mixed = F.copy()

            # Apply Saunders-Hillier level shift to F before diagonalisation:
            #   F_shifted = F + b . S - (b/2) . S . D . S
            # Raises virtual eigenvalues by b in the new MO basis. The shift
            # leaves the SCF fixed point unchanged but suppresses occupied /
            # virtual mixing during iteration. We use the *un-extrapolated*
            # density D_used (pre-update) so the shift respects the current
            # iterate, not the next.
            b = (
                level_shift_at_iter(
                    level_shift, 0, _ls_schedule, _ls_max_iter, iter_idx
                )
                if _ls_schedule
                else level_shift
            )
            # Shared Saunders-Hillier operator: the weight on S·D·S is
            # fixed by the density convention, not by taste. ``D_used`` is
            # the closed-shell total density (occupations in {0, 2}), hence
            # TOTAL. Returns F untouched when b == 0.
            # Guarded: skip the pybind conversion of S and D entirely on an
            # unshifted cycle, which is the default and the common case.
            F_diag = (
                F if b == 0.0
                else apply_level_shift(F, S, D_used, b, LevelShiftDensity.TOTAL)
            )

            C_new, eps_new = diagonalise(F_diag)
        D_prev = D_used
        D = 2.0 * C_new[:, :n_occ] @ C_new[:, :n_occ].T
        D = 0.5 * (D + D.T)

        C_final = C_new
        eps_final = eps_new
        F_final = F
        # Dynamic-damping update -- feed the just-computed E_total so
        # the *next* iteration reads an a adapted to this iteration's
        # energy step. No-op when ``damper is None`` (static damping).
        if damper is not None:
            damper.update(E_total)
        E_prev = E_total

        # Stash the per-iter state on the result for inspection even
        # if we don't converge.
        result.energy = E_total
        result.e_electronic = E_elec
        result.n_iter = iter_idx
        result.mo_energies = eps_new
        result.mo_coeffs = C_new
        result.density = D_used
        result.fock = F

        if converged:
            # Final self-consistency pass on the fresh D (same
            # convention as the C++ driver).
            J_f = j_build(D)
            jk_f = build_jk_gamma_molecular_limit(
                basis,
                system,
                lat_opts,
                D,
                0.0,
            )
            K_f = np.asarray(jk_f.K)
            F_f = Hcore + J_f - 0.5 * K_f
            F_f = 0.5 * (F_f + F_f.T)
            C_f, eps_f = diagonalise(F_f)
            E_elec_f = 0.5 * float(np.einsum("ij,ij->", D, Hcore + F_f))
            if lat_opts.coulomb_method == CoulombMethod.EWALD_3D:
                E_madelung_fix_f = 0.0
            else:
                E_madelung_fix_f = madelung_energy_correction(
                    D,
                    S,
                    system,
                    nuclear_uses_ewald=False,
                )
            result.energy = E_elec_f + float(e_nuc) + E_madelung_fix_f
            result.e_electronic = E_elec_f
            result.mo_energies = eps_f
            result.mo_coeffs = C_f
            result.density = D
            result.fock = F_f
            result.converged = True
            plog.converged(
                n_iter=result.n_iter,
                energy=result.energy,
                converged=True,
            )
            return result

    result.converged = False
    plog.converged(
        n_iter=result.n_iter,
        energy=result.energy,
        converged=False,
    )
    return result


def run_rhf_periodic_gamma_ewald2d(
    system: PeriodicSystem,
    basis: BasisSet,
    options: Optional[PeriodicRHFOptions] = None,
    *,
    alpha: Optional[float] = None,
    omega: Optional[float] = None,
    grid_shape: Optional[Union[Tuple[int, int, int], int]] = None,
    origin: Optional[Sequence[float]] = None,
    spacing_bohr: float = 0.3,
    linear_dep_threshold: float = 1e-7,
    canonical_orth_normalize_diag_first: bool = True,
    progress: Union[bool, ProgressLogger, None] = None,
    verbose: Optional[int] = None,
) -> PeriodicRHFEwaldResult:
    """Gamma-point closed-shell periodic RHF with rigorous 2D slab Ewald.

    This is the ``SLAB_EWALD_2D`` analogue of
    :func:`run_rhf_periodic_gamma_ewald3d`: one-electron ``V_ne``, nuclear
    repulsion, and Hartree ``J`` all use the same Parry/de Leeuw 2D-Ewald
    split parameter. Exchange remains the full-range Gamma molecular-limit
    real-space build. Slab analytic gradients remain a separate workstream.

    ``alpha`` controls the slab Ewald split. For compatibility with the
    dispatcher, a positive ``omega`` is accepted as an alias for ``alpha``.
    FFT grid arguments are accepted but unused; the slab ``J`` is analytic-FT
    and z-profile based rather than FFT-Poisson based.
    """
    opts = options if options is not None else PeriodicRHFOptions()
    from .guess import select_periodic_driver_guess
    selection = select_periodic_driver_guess(
        system, opts, route="ewald", method="RHF", driver="run_rhf_periodic_gamma_ewald2d",
    )
    lat_opts = opts.lattice_opts
    if system.dim != 2:
        raise ValueError(
            "run_rhf_periodic_gamma_ewald2d: SLAB_EWALD_2D requires "
            f"dim == 2; got dim = {system.dim}"
        )
    if float(getattr(opts, "smearing_temperature", 0.0)) > 0.0:
        raise NotImplementedError(
            "run_rhf_periodic_gamma_ewald2d: Fermi-Dirac smearing requires "
            "the multi-k Ewald machinery, which is not yet available for "
            "SLAB_EWALD_2D"
        )

    if lat_opts.coulomb_method != CoulombMethod.SLAB_EWALD_2D:
        lat_opts.coulomb_method = CoulombMethod.SLAB_EWALD_2D

    omega_alpha = None
    if omega is not None and float(omega) > 0.0:
        omega_alpha = float(omega)
    if alpha is None:
        alpha = omega_alpha
    elif omega_alpha is not None and abs(float(alpha) - omega_alpha) > 1e-14:
        raise ValueError(
            "run_rhf_periodic_gamma_ewald2d: alpha and positive omega alias "
            "disagree"
        )
    if alpha is None:
        alpha = float(getattr(lat_opts, "slab_ewald_alpha", 0.4))
    alpha = float(alpha)
    if alpha <= 0.0:
        alpha = 0.4
    lat_opts.slab_ewald_alpha = alpha

    plog = resolve_progress(progress, verbose=verbose)
    plog.info(f"RHF Gamma SLAB_EWALD_2D / alpha = {alpha:.3f}")
    plog.info(f"basis: {basis.name}  ({basis.nbasis} BFs / {basis.nshells} shells)")
    dump_active_settings(
        plog,
        [
            ("PeriodicRHFOptions", opts),
            ("LatticeSumOptions", lat_opts),
            (
                "Driver kwargs",
                {
                    "alpha": alpha,
                    "omega_alias": omega,
                    "grid_shape": grid_shape,
                    "origin": origin,
                    "spacing_bohr": float(spacing_bohr),
                    "linear_dep_threshold": float(linear_dep_threshold),
                    "canonical_orth_normalize_diag_first": (
                        canonical_orth_normalize_diag_first
                    ),
                },
            ),
        ],
    )
    if plog.level >= 5:
        from .scf_log import format_basis_summary

        plog.write_raw(format_basis_summary(basis))

    n_elec = system.n_electrons()
    if n_elec % 2 != 0:
        raise ValueError(
            "run_rhf_periodic_gamma_ewald2d: closed-shell RHF requires even "
            f"electron count; got {n_elec}"
        )
    if system.multiplicity != 1:
        raise ValueError(
            "run_rhf_periodic_gamma_ewald2d: closed-shell RHF requires "
            f"multiplicity=1; got {system.multiplicity}"
        )
    n_occ = n_elec // 2

    eopts = EwaldOptions()
    eopts.alpha = alpha
    eopts.real_cutoff_bohr = lat_opts.nuclear_cutoff_bohr

    with plog.stage(
        "integrals_lattice", detail=f"S/T/V at cutoff {lat_opts.cutoff_bohr:.2f} bohr"
    ):
        S_lat = compute_overlap_lattice(basis, system, lat_opts)
        T_lat = compute_kinetic_lattice(basis, system, lat_opts)
        from .periodic_v_ne import compute_nuclear_lattice_dispatch

        V_lat = compute_nuclear_lattice_dispatch(
            basis, system, lat_opts, ewald_options=eopts
        )

    k_gamma = np.zeros(3)
    S = np.real(bloch_sum(S_lat, k_gamma))
    T = np.real(bloch_sum(T_lat, k_gamma))
    V = np.real(bloch_sum(V_lat, k_gamma))
    Hcore = 0.5 * (T + V + (T + V).T)
    S = 0.5 * (S + S.T)

    scf_preflight_overlap_check(S, plog=plog, label="S(Gamma)", basis=basis)
    X, n_kept = _canonical_orthogonalizer(
        S,
        linear_dep_threshold,
        normalize_diag_first=canonical_orth_normalize_diag_first,
    )
    if n_occ > n_kept:
        raise RuntimeError(
            "run_rhf_periodic_gamma_ewald2d: canonical orthogonalisation "
            f"dropped too many directions (n_occ = {n_occ}, n_kept = {n_kept})"
        )

    use_davidson = getattr(opts, "use_davidson", False)
    dav_opts = getattr(opts, "davidson", None)
    dav_dim = getattr(opts, "davidson_min_dim", 100)
    use_dav = use_davidson and S.shape[0] >= dav_dim
    if use_dav and dav_opts is None:
        from vibeqc._vibeqc_core import DavidsonOptions

        dav_opts = DavidsonOptions()

    e_nuc = nuclear_repulsion_per_cell(system, lat_opts)

    def diagonalise(F):
        Fp = X.T @ F @ X
        Fp = 0.5 * (Fp + Fp.T)
        if use_dav and dav_opts is not None:
            from vibeqc._vibeqc_core import davidson_solve

            if dav_opts.n_eig == 0:
                dav_opts.n_eig = Fp.shape[0]
            dres = davidson_solve(Fp, dav_opts)
            if not dres.converged:
                raise RuntimeError(
                    f"Davidson did not converge after {dres.n_iter} iters"
                )
            eps, Cp = dres.eigenvalues, dres.eigenvectors
            dav_opts.guess_vectors = Cp
        else:
            eps, Cp = np.linalg.eigh(Fp)
        return X @ Cp, eps

    j_build = make_slab_ewald_2d_gamma_j_builder(
        basis, system, lattice_opts=lat_opts, alpha=alpha
    )

    guess = selection.effective
    seed_guess = InitialGuess.SAD if guess == InitialGuess.PATOM else guess
    D_engine = initial_density_closed_shell(
        system.unit_cell_molecule(),
        basis,
        n_occ,
        seed_guess,
        is_periodic=True,
        periodic_system=system,
        lattice_opts=lat_opts,
        read_density=getattr(opts, "read_density", None),
        read_path=getattr(opts, "read_path", ""),
        overlap=S,
    )
    if D_engine is not None:
        plog.info(f"initial guess: {guess.name} (density via GuessEngine)")
        D = D_engine
        C0, eps0 = diagonalise(Hcore)
    else:
        plog.info(f"initial guess: {guess.name} (Hcore-diagonalise)")
        C0, eps0 = diagonalise(Hcore)
        D = 2.0 * C0[:, :n_occ] @ C0[:, :n_occ].T
        D = 0.5 * (D + D.T)
    if guess == InitialGuess.PATOM:
        # PATOM is one full-HF in-field step, independent of the target XC.
        J_seed = j_build(D)
        K_seed = np.asarray(build_jk_gamma_molecular_limit(
            basis, system, lat_opts, D, 0.0).K)
        C0, eps0 = diagonalise(Hcore + J_seed - 0.5 * K_seed)
        D = 2.0 * C0[:, :n_occ] @ C0[:, :n_occ].T
        D = 0.5 * (D + D.T)
    D_prev = D.copy()

    damping = float(opts.damping)
    if not (0.0 <= damping < 1.0):
        raise ValueError(
            "run_rhf_periodic_gamma_ewald2d: damping must be in [0, 1); "
            f"got {damping}"
        )
    damper: Optional[DynamicDamping] = None
    if bool(getattr(opts, "dynamic_damping", False)):
        damper = DynamicDamping(
            initial_alpha=damping,
            alpha_min=float(getattr(opts, "dynamic_damping_min", 0.0)),
            alpha_max=float(getattr(opts, "dynamic_damping_max", 0.95)),
        )

    use_diis = bool(opts.use_diis)
    diis_start_iter = int(opts.diis_start_iter)
    accel: Optional[PeriodicSCFAccelerator] = (
        PeriodicSCFAccelerator(opts) if use_diis else None
    )
    level_shift = float(getattr(opts, "level_shift", 0.0))
    # Explicit per-iteration schedule (unified with the molecular
    # drivers). Empty ⇒ the constant/persistent `level_shift` above;
    # non-empty ⇒ resolved per iteration by the shared C++ helper.
    _ls_schedule = list(getattr(opts, "level_shift_schedule", None) or [])
    _ls_max_iter = int(opts.max_iter)
    fock_mixing_value = float(getattr(opts, "fock_mixing", 0.0))
    if not (0.0 <= fock_mixing_value < 1.0):
        raise ValueError(
            "run_rhf_periodic_gamma_ewald2d: fock_mixing must be in [0, 1); "
            f"got {fock_mixing_value}"
        )
    F_prev_mixed: Optional[np.ndarray] = None

    quadratic_fallback_iter = int(getattr(opts, "quadratic_fallback_iter", 0))
    quadratic_fallback_shift = float(getattr(opts, "quadratic_fallback_shift", 0.1))
    quadratic_fallback_max_step = float(
        getattr(opts, "quadratic_fallback_max_step", 0.1)
    )

    plog.banner("SCF (RHF Gamma, SLAB_EWALD_2D)")
    plog.info("  iter         energy (Ha)            dE          ||[F,DS]||   DIIS")

    scf_trace: List[SCFIteration] = []
    result = PeriodicRHFEwaldResult(
                 restart_kpoints=np.zeros((1, 3)), restart_weights=np.ones(1),
                 guess_selection=periodic_result_selection(system, opts.initial_guess, restarted=False),
                 restart_basis=basis, restart_lattice=np.asarray(system.lattice).copy(),
        energy=0.0,
        e_electronic=0.0,
        e_nuclear=float(e_nuc),
        n_iter=0,
        converged=False,
        mo_energies=np.empty(0),
        mo_coeffs=np.empty((0, 0)),
        density=D.copy(),
        fock=np.empty((0, 0)),
        overlap=S,
        scf_trace=scf_trace,
        omega=alpha,
        grid_shape=(0, 0, 0),
    )


    E_prev = 0.0
    C_final = C0
    eps_final = eps0
    F_final = Hcore.copy()

    for iter_idx in range(1, int(opts.max_iter) + 1):
        if damper is not None:
            damping = damper.alpha
        diis_active = use_diis and iter_idx >= diis_start_iter
        D_used = (
            D
            if (iter_idx == 1 or damping == 0.0 or diis_active)
            else damping * D_prev + (1.0 - damping) * D
        )

        J = j_build(D_used)
        jk_full = build_jk_gamma_molecular_limit(
            basis, system, lat_opts, D_used, 0.0
        )
        K = np.asarray(jk_full.K)
        F = Hcore + J - 0.5 * K
        F = 0.5 * (F + F.T)
        E_elec = 0.5 * float(np.einsum("ij,ij->", D_used, Hcore + F))
        E_total = E_elec + float(e_nuc)

        FDS = F @ D_used @ S
        grad = FDS - FDS.T
        grad_norm = float(np.linalg.norm(grad))
        dE = E_total - E_prev
        check_scf_divergence(
            "run_rhf_periodic_gamma_ewald2d",
            iter_idx,
            E_total,
            grad_norm,
            dE,
        )
        converged = (
            iter_idx > 1
            and abs(dE) < float(opts.conv_tol_energy)
            and grad_norm < float(opts.conv_tol_grad)
        )
        scf_trace.append(
            SCFIteration(
                iter=iter_idx,
                energy=float(E_total),
                delta_e=float(dE if iter_idx > 1 else 0.0),
                grad_norm=float(grad_norm),
                diis_subspace=(accel.subspace_size if accel is not None else 0),
            )
        )
        plog.iteration(
            iter_idx,
            energy=float(E_total),
            dE=float(dE if iter_idx > 1 else 0.0),
            grad=float(grad_norm),
            diis=(accel.subspace_size if accel is not None else 0),
        )
        plog.energy_decomposition(
            iter_idx,
            E_kin=float(np.einsum("ij,ij->", D_used, T)),
            E_ne=float(np.einsum("ij,ij->", D_used, V)),
            E_J=0.5 * float(np.einsum("ij,ij->", D_used, J)),
            E_K=-0.25 * float(np.einsum("ij,ij->", D_used, K)),
            E_nuc=float(e_nuc),
            E_madelung=0.0,
        )

        in_quadratic_phase = (
            quadratic_fallback_iter > 0 and iter_idx > quadratic_fallback_iter
        )
        if in_quadratic_phase:
            from .quadratic_scf import quadratic_step

            C_new, eps_new = quadratic_step(
                F,
                C_final,
                eps_final,
                n_occ,
                shift=quadratic_fallback_shift,
                max_step=quadratic_fallback_max_step,
            )
        else:
            if accel is not None:
                F_ex = accel.extrapolate_rhf(
                    F,
                    error=grad,
                    density=D_used,
                    energy=E_total,
                    mo_coeffs=C_final,
                    mo_energies=eps_final,
                    n_occ=n_occ,
                )
                if diis_active:
                    F = F_ex
            if fock_mixing_value != 0.0:
                if F_prev_mixed is not None:
                    F_mixed = (
                        (1.0 - fock_mixing_value) * F
                        + fock_mixing_value * F_prev_mixed
                    )
                    F = 0.5 * (F_mixed + F_mixed.T)
                F_prev_mixed = F.copy()
            b = (
                level_shift_at_iter(
                    level_shift, 0, _ls_schedule, _ls_max_iter, iter_idx
                )
                if _ls_schedule
                else level_shift
            )
            # Shared Saunders-Hillier operator: the weight on S·D·S is
            # fixed by the density convention, not by taste. ``D_used`` is
            # the closed-shell total density (occupations in {0, 2}), hence
            # TOTAL. Returns F untouched when b == 0.
            # Guarded: skip the pybind conversion of S and D entirely on an
            # unshifted cycle, which is the default and the common case.
            F_diag = (
                F if b == 0.0
                else apply_level_shift(F, S, D_used, b, LevelShiftDensity.TOTAL)
            )
            C_new, eps_new = diagonalise(F_diag)

        D_prev = D_used
        D = 2.0 * C_new[:, :n_occ] @ C_new[:, :n_occ].T
        D = 0.5 * (D + D.T)
        C_final = C_new
        eps_final = eps_new
        F_final = F
        if damper is not None:
            damper.update(E_total)
        E_prev = E_total

        result.energy = E_total
        result.e_electronic = E_elec
        result.n_iter = iter_idx
        result.mo_energies = eps_new
        result.mo_coeffs = C_new
        result.density = D_used
        result.fock = F

        if converged:
            J_f = j_build(D)
            jk_f = build_jk_gamma_molecular_limit(
                basis, system, lat_opts, D, 0.0
            )
            K_f = np.asarray(jk_f.K)
            F_f = Hcore + J_f - 0.5 * K_f
            F_f = 0.5 * (F_f + F_f.T)
            C_f, eps_f = diagonalise(F_f)
            E_elec_f = 0.5 * float(np.einsum("ij,ij->", D, Hcore + F_f))
            result.energy = E_elec_f + float(e_nuc)
            result.e_electronic = E_elec_f
            result.mo_energies = eps_f
            result.mo_coeffs = C_f
            result.density = D
            result.fock = F_f
            result.converged = True
            plog.converged(
                n_iter=result.n_iter,
                energy=result.energy,
                converged=True,
            )
            return result

    result.fock = F_final
    result.converged = False
    plog.converged(
        n_iter=result.n_iter,
        energy=result.energy,
        converged=False,
    )
    return result
