"""Shared internal helpers for the BIPOLE periodic SCF drivers.

This module holds the pieces common to the four BIPOLE drivers
(:mod:`vibeqc.pbc_bipole` RHF, :mod:`vibeqc.pbc_bipole_rks` RKS,
:mod:`vibeqc.pbc_bipole_uhf` UHF, :mod:`vibeqc.pbc_bipole_uks` UKS):
the CRYSTAL-gauge Ewald option builder, the analytic 3D ``V_ne``
reciprocal-FT path, the real-space lattice contractions, the Γ-only
density-locality fix, the IBZ->full k-mesh expansion for Ewald-J, the
spin-occupation split, and the lattice block copy/combine helpers.

It is deliberately BIPOLE-only and must not grow into a general
periodic-SCF framework. The four drivers import everything they share
from here; the helpers also remain importable from their historical
modules (``pbc_bipole`` / ``pbc_bipole_uhf``) as compatibility
re-exports for ``bipole_gradient`` and the test-suite.

Hard invariants preserved here (see the BIPOLE unification handover):

* ``V_ne``, ``E_nn``, and the optional reciprocal ``J_LR`` share one
  Ewald a / K-envelope via :func:`_crystal_ewald_options` -- the Ewald
  state is never recomputed independently per variant.
* Γ-only density locality (``P(g!=0) = 0``) is applied through
  :func:`_zero_cross_cell_density` after every density rebuild.
"""

from __future__ import annotations

import contextlib
import math
import time
import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ._vibeqc_core import (
    BasisSet,
    BlochKMesh,
    CoulombMethod,
    EwaldOptions,
    GridOptions,
    LatticeMatrixSet,
    LatticeSumOptions,
    PeriodicSystem,
    SCFIteration,
    compute_nuclear_erfc_lattice,
    compute_overlap_lattice,
    make_lattice_matrix_set,
    real_space_density_from_kpoints_fractional,
)
from ._vibeqc_core import (
    monkhorst_pack as _native_monkhorst_pack,
)
from .progress import ProgressLogger


def _emit_bipole_semantic_diagnostic(
    message: str,
    *,
    warning: bool,
    once_key: Optional[str] = None,
    **fields: object,
) -> None:
    """Route a BIPOLE diagnostic through the shared semantic logger."""
    if warning:
        from .output import warn as output_warn

        output_warn(message, once_key=once_key, **fields)
    else:
        from .output import note as output_note

        output_note(message, once_key=once_key, **fields)


__all__ = [
    "PBCBipoleEnergyComponents",
    "BipoleFoldUnreliableError",
    "reject_bipole_ecp_options",
    "reject_bipole_quartet_far_field",
    "validate_bipole_kmesh",
    "reject_bipole_lone_non_gamma_kpoint",
    "reject_bipole_solver_options",
    "reject_bipole_unsupported_ks_functional",
    "restricted_bipole_commutator_norm",
    "unrestricted_bipole_commutator_norm",
    "refresh_bipole_terminal_trace",
    "_zero_cross_cell_density",
    "home_cell_block",
    "bvk_torus_density_matrices",
    "s_fold_truncation_drift",
    "validate_bipole_fold_drift",
    "smearing_basin_warning",
    "_bloch_sum_blocks",
    "_bloch_sum_blocks_multi_k",
    "_cell_key",
    "_lattice_contract",
    "_lattice_contract_blocks",
    "_crystal_ewald_options",
    "clamp_bipole_nuclear_cutoff",
    "ewald_real_cutoff_for_alpha",
    "ewald_erfc_lattice_options",
    "raise_bipole_ewald_real_cutoff",
    "prepare_bipole_lattice_options",
    "resolve_fock_mixing",
    "resolve_auto_fock_mixing",
    "reject_bipole_fractional_legacy_gauge",
    "resolve_bipole_fock_symmetry",
    "resolve_incremental_jk",
    "warn_bipole_legacy_multik_gauge",
    "warn_bipole_charged_cell",
    "warn_bipole_exact_j_core_tail",
    "bipole_sr_image_extent",
    "resolve_bipole_sr_image_extent",
    "_expand_ibz_kmesh_for_ewald_j",
    "_default_bipole_v_ne_grid_options",
    "_compute_nuclear_lattice_ewald_reciprocal_ft",
    "_spin_occupations",
    "_combine_density_sets",
    "_copy_lattice_with_blocks",
    "_density_set_gamma_or_lattice",
]


def reject_bipole_quartet_far_field(
    requested: Optional[bool],
    *,
    driver: str,
) -> None:
    """Fail closed on the incomplete quartet-level bipolar approximation.

    The exact periodic Fock contraction carries three lattice translations
    and contracts the density at their relative translation.  The current
    experimental dispatch and add-back carry only two translations, so they
    do not replace the same quartet domain.  Keep the exact route as the only
    supported driver behavior until that domain identity has an end-to-end
    equation-level regression.
    """
    if requested is None:
        return
    if not isinstance(requested, (bool, np.bool_)):
        raise TypeError(
            f"{driver}: use_multipole_far_field must be bool or None; "
            f"got {type(requested).__name__}."
        )
    if bool(requested):
        raise NotImplementedError(
            f"{driver}: use_multipole_far_field=True is unavailable because "
            "the experimental quartet dispatch does not yet preserve the "
            "three-translation periodic Fock domain. Use "
            "use_multipole_far_field=False for the supported exact route."
        )


def reject_bipole_solver_options(options: object, *, driver: str) -> None:
    """Reject iterative eigensolvers that direct BIPOLE does not execute."""
    if bool(getattr(options, "use_davidson", False)):
        raise NotImplementedError(
            f"{driver}: iterative diagonalization is not implemented by the "
            "BIPOLE SCF loop. Use the dense solver instead."
        )


def reject_bipole_fractional_legacy_gauge(
    *,
    use_ewald_j_split: bool,
    exchange_split_active: bool,
    fractional_occupations: bool,
    driver: str,
) -> None:
    """Reject fractional occupations on the legacy Ewald-J gauge.

    The corrected gauge forms the reciprocal long-range density directly
    from the supplied real-space density. The legacy gauge reconstructs it
    from occupied orbitals and therefore assumes integer occupations. Mixing
    that reconstruction with a smeared or Gilat-Raubenheimer density produces
    a Hartree operator for a different electronic state.
    """
    if (
        bool(use_ewald_j_split)
        and not bool(exchange_split_active)
        and bool(fractional_occupations)
    ):
        raise NotImplementedError(
            f"{driver}: fractional occupations require the corrected "
            "Ewald J/exchange-split gauge. The legacy gauge reconstructs "
            "the long-range density with integer occupations; leave "
            "use_exchange_ewald_split enabled."
        )


def reject_bipole_unsupported_ks_functional(
    functional: object,
    *,
    driver: str,
) -> None:
    """Reject KS components absent from the BIPOLE energy/gradient route."""
    if bool(getattr(functional, "is_double_hybrid", False)):
        raise NotImplementedError(
            f"{driver}: double-hybrid functionals are unavailable because "
            "the periodic BIPOLE driver does not add the required MP2 "
            "correlation energy."
        )
    if bool(getattr(functional, "needs_vv10", False)):
        raise NotImplementedError(
            f"{driver}: VV10/nonlocal-correlation functionals are unavailable "
            "because the periodic BIPOLE XC builder does not evaluate the "
            "nonlocal correlation energy or potential."
        )


def validate_bipole_kmesh(
    kmesh: object,
    *,
    driver: str,
    require_complete: bool = False,
) -> Tuple[int, int, bool, bool]:
    """Validate MP/IBZ metadata and weights shared by BIPOLE workflows.

    Returns ``(stored_n, full_n, true_multik, is_ibz)``. Direct SCF drivers
    retain ad-hoc multi-k lists for diagnostics, while geometry/NEB callers
    set ``require_complete=True`` so only a full MP mesh or a valid IBZ
    reduction can define a production objective.
    """
    points = np.asarray(getattr(kmesh, "kpoints", []), dtype=float).reshape(-1, 3)
    stored_n = int(points.shape[0])
    if stored_n < 1:
        raise ValueError(f"{driver}: kmesh has no k-points")

    weights = np.asarray(getattr(kmesh, "weights", []), dtype=float).reshape(-1)
    if weights.shape != (stored_n,):
        raise ValueError(
            f"{driver}: kmesh has {stored_n} points but {weights.size} weights"
        )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError(f"{driver}: kmesh weights must be finite and non-negative")
    if not np.isclose(float(weights.sum()), 1.0):
        raise ValueError(
            f"{driver}: kmesh weights must sum to 1; got {weights.sum():.6f}"
        )

    mesh_raw = np.asarray(getattr(kmesh, "mesh", (1, 1, 1))).reshape(-1)
    if mesh_raw.shape != (3,):
        raise ValueError(f"{driver}: kmesh.mesh must contain three integers")
    try:
        mesh_float = np.asarray(mesh_raw, dtype=float)
        mesh = np.asarray(mesh_raw, dtype=int)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{driver}: kmesh.mesh must contain three positive integers"
        ) from exc
    if (
        not np.all(np.isfinite(mesh_float))
        or not np.all(mesh_float == mesh)
        or np.any(mesh < 1)
    ):
        raise ValueError(
            f"{driver}: kmesh.mesh must contain three positive integers"
        )
    full_n = int(np.prod(mesh))

    ir_mapping = np.asarray(
        getattr(kmesh, "ir_mapping", []), dtype=int
    ).reshape(-1)
    is_ibz = False
    if ir_mapping.size:
        if ir_mapping.size != full_n:
            raise ValueError(
                f"{driver}: ir_mapping has {ir_mapping.size} entries; "
                f"expected the full mesh size {full_n}"
            )
        if np.any(ir_mapping < 0) or np.any(ir_mapping >= stored_n):
            raise ValueError(
                f"{driver}: ir_mapping indices must lie in [0, {stored_n})"
            )
        if not np.array_equal(np.unique(ir_mapping), np.arange(stored_n)):
            raise ValueError(
                f"{driver}: every stored IBZ representative must be used"
            )
        expected_weights = np.bincount(
            ir_mapping, minlength=stored_n
        ).astype(float) / float(full_n)
        if not np.allclose(weights, expected_weights, atol=1.0e-12, rtol=0.0):
            raise ValueError(
                f"{driver}: IBZ weights do not match ir_mapping star counts"
            )
        is_ibz = stored_n < full_n
    elif stored_n == full_n:
        expected = np.full(stored_n, 1.0 / float(stored_n))
        if not np.allclose(weights, expected, atol=1.0e-12, rtol=0.0):
            raise ValueError(
                f"{driver}: a full Monkhorst-Pack mesh requires uniform weights"
            )

    true_multik = stored_n > 1 or full_n > 1 or ir_mapping.size > 1
    complete = stored_n == full_n or is_ibz
    if require_complete and true_multik and not complete:
        raise NotImplementedError(
            f"{driver}: multi-k BIPOLE work requires a complete "
            "Monkhorst-Pack mesh or its valid IBZ reduction. Ad-hoc k-point "
            "lists use the diagnostic nonstationary legacy gauge."
        )
    return stored_n, full_n, true_multik, is_ibz


def reject_bipole_lone_non_gamma_kpoint(kmesh: object, *, driver: str) -> None:
    """Reject a single twisted k point that the BIPOLE loops treat as Gamma.

    The current one-point fast paths use the home-cell density/Fock folds and
    do not construct the q-channel/BvK machinery needed for a nonzero twist.
    A complete multi-point Monkhorst-Pack mesh remains supported.
    """
    points = np.asarray(getattr(kmesh, "kpoints", []), dtype=float).reshape(-1, 3)
    stored_n, _, _, is_ibz = validate_bipole_kmesh(
        kmesh,
        driver=driver,
        require_complete=False,
    )
    if (
        stored_n == 1
        and not is_ibz
        and float(np.linalg.norm(points[0])) > 1.0e-10
    ):
        raise NotImplementedError(
            f"{driver}: a lone non-Gamma k point is not supported. Use the "
            "Gamma point or a complete Monkhorst-Pack mesh."
        )


def refresh_bipole_terminal_trace(
    trace: List[SCFIteration],
    objective: float,
    *,
    grad_norm: Optional[float] = None,
) -> Tuple[float, float]:
    """Align the terminal trace row with the density returned to callers.

    Returns the refreshed ``(delta_e, grad_norm)`` pair so the driver can
    verify that a provisional convergence decision still holds after the
    mandatory terminal Fock rebuild.
    """
    if not trace:
        return 0.0, float(0.0 if grad_norm is None else grad_norm)
    last = trace[-1]
    # The rebuild is one additional physical evaluation after the loop's
    # terminal row: compare against that row, not against trace[-2].  The
    # latter would skip the density that was just diagonalised and can hide
    # (or invent) a final fixed-point displacement.
    delta_e = float(objective) - float(last.energy)
    final_grad = (
        float(last.grad_norm) if grad_norm is None else float(grad_norm)
    )
    trace[-1] = SCFIteration(
        iter=int(last.iter),
        energy=float(objective),
        delta_e=delta_e,
        grad_norm=final_grad,
        diis_subspace=int(last.diis_subspace),
        newton_cg_iter=int(last.newton_cg_iter),
        trah_level_shift=float(last.trah_level_shift),
        wall_s=float(last.wall_s),
    )
    return delta_e, final_grad


def bipole_terminal_check(
    plog,
    *,
    phase: str,
    iter_idx: int,
    loop_objective: float,
    exact_objective: float,
    loop_grad_norm: float,
    exact_grad_norm: float,
    conv_tol_energy: float,
    conv_tol_grad: float,
) -> tuple[bool, float]:
    """Judge a provisional BIPOLE convergence on the exact operator (#116).

    The SCF loop converges on the incremental ``J_SR``/``K_SR`` chain
    (:class:`~vibeqc.bipole_fock_ewald.IncrementalJK`), whose accumulated
    screening error makes its energy differ from a non-incremental build of
    the same density by a small, structural amount: 2.1e-8 Ha on LiH/STO-3G
    (2,2,2) at cutoff 25 (recorded regression case), against
    ``conv_tol_energy = 1e-8``. Until #116 the loop was judged after the
    fact on that non-incremental rebuild (``final_delta = E_exact - E_loop``)
    and the disagreement revoked a converged SCF with no way back: 22 h of
    correct computation on that LiH row and 750 core-hours on an Al2O3
    (2,2,2) cell (the gradient leg there) returned no energy at all.

    This is the ONE criterion both call sites now apply, so the loop cannot
    exit on a state the post-loop refuses: the in-loop confirmation
    (``phase="in_loop"``) evaluates the exact rebuild's objective and
    commutator norm at the committed density, and the post-loop refresh
    (``phase="post_loop"``) re-applies the same test to the same rebuild.
    ``delta = exact_objective - loop_objective`` is exactly what
    :func:`refresh_bipole_terminal_trace` writes into the terminal trace
    row. Both values are logged and emitted as a ``scf_terminal_check``
    structured event whether the check passes or not, so a withdrawal is
    auditable from the artifact (the number that failed was previously
    printed nowhere).

    Returns ``(passed, delta)``. On an in-loop failure the driver reseeds
    its incremental accumulator from the exact rebuild and keeps iterating,
    so it converges on the operator it is judged by rather than on one path
    and scored on another; no tolerance is widened.
    """
    from .structured_log import emit as _emit_structured

    delta = float(exact_objective) - float(loop_objective)
    tol_e = float(conv_tol_energy)
    tol_g = float(conv_tol_grad)
    passed = bool(abs(delta) < tol_e and float(exact_grad_norm) < tol_g)
    verdict = "confirmed" if passed else "NOT confirmed"
    where = "terminal" if phase == "post_loop" else "in-loop"
    plog.info(
        f"  {where} check on the exact (non-incremental) operator at "
        f"iteration {int(iter_idx)}: {verdict}; "
        f"E_exact - E_loop = {delta:+.3e} Ha (conv_tol_energy {tol_e:.0e}), "
        f"||[F,DS]|| = {float(exact_grad_norm):.3e} exact vs "
        f"{float(loop_grad_norm):.3e} in-loop (conv_tol_grad {tol_g:.0e})"
        + (
            "" if passed or phase == "post_loop"
            else " -> continuing from the exact Fock"
        )
    )
    _emit_structured(
        "scf_terminal_check",
        phase=str(phase),
        n_iter=int(iter_idx),
        energy_loop=float(loop_objective),
        energy_exact=float(exact_objective),
        delta_e=delta,
        grad_norm_loop=float(loop_grad_norm),
        grad_norm_exact=float(exact_grad_norm),
        conv_tol_energy=tol_e,
        conv_tol_grad=tol_g,
        passed=passed,
    )
    return passed, delta


def restricted_bipole_commutator_norm(
    fock_k: Sequence[np.ndarray],
    density_k: Sequence[np.ndarray],
    overlap_k: Sequence[np.ndarray],
    weights: Sequence[float],
) -> float:
    """Return the BZ-weighted restricted ``||FDS-(FDS)^H||`` norm."""
    return float(
        sum(
            float(weight)
            * float(
                np.linalg.norm(
                    np.asarray(fock) @ np.asarray(density) @ np.asarray(overlap)
                    - (
                        np.asarray(fock)
                        @ np.asarray(density)
                        @ np.asarray(overlap)
                    ).conj().T
                )
            )
            for fock, density, overlap, weight in zip(
                fock_k, density_k, overlap_k, weights
            )
        )
    )


def unrestricted_bipole_commutator_norm(
    fock_alpha_k: Sequence[np.ndarray],
    fock_beta_k: Sequence[np.ndarray],
    density_alpha_k: Sequence[np.ndarray],
    density_beta_k: Sequence[np.ndarray],
    overlap_k: Sequence[np.ndarray],
    weights: Sequence[float],
) -> float:
    """Return the BZ-weighted two-spin BIPOLE commutator norm."""
    total = 0.0
    for f_a, f_b, d_a, d_b, overlap, weight in zip(
        fock_alpha_k,
        fock_beta_k,
        density_alpha_k,
        density_beta_k,
        overlap_k,
        weights,
    ):
        fds_a = np.asarray(f_a) @ np.asarray(d_a) @ np.asarray(overlap)
        fds_b = np.asarray(f_b) @ np.asarray(d_b) @ np.asarray(overlap)
        err_a = fds_a - fds_a.conj().T
        err_b = fds_b - fds_b.conj().T
        total += float(weight) * float(
            np.sqrt(np.linalg.norm(err_a) ** 2 + np.linalg.norm(err_b) ** 2)
        )
    return float(total)


def reject_bipole_ecp_options(
    options: object,
    *,
    driver: str,
    basis: object | None = None,
    system: object | None = None,
) -> None:
    """Fail before SCF when direct BIPOLE receives an ECP request.

    The current drivers use the all-electron nuclear operator, electron count,
    and ionic repulsion. Merely carrying ECP metadata, or passing a basis
    family designed to accompany an ECP, is therefore not enough to define a
    consistent pseudopotential Hamiltonian. Charge-only records matching
    the known parent's physical nuclear charges are all-electron metadata.
    """
    from ._vibeqc_core import Molecule
    from .guess import _has_guess_ecp_metadata

    if isinstance(system, PeriodicSystem):
        molecule = system.unit_cell_molecule()
    elif isinstance(system, Molecule):
        molecule = system
    else:
        molecule = None
    active = _has_guess_ecp_metadata(options, molecule=molecule)
    basis_name = str(getattr(basis, "name", "") or "")
    if basis_name:
        # Decide per element through the registry: an ECP-bearing sidecar
        # block for an atom present, or a valence-only family beyond its
        # threshold (def2 / def2-m* / pob-TZVP-rev2 beyond Kr, cc-pVnZ-PP
        # from Cu on), makes the request an ECP request. A light molecule in
        # LANL2DZ or an all-electron relativistic basis is admitted.
        from .basis_registry import basis_replaces_core

        atoms = list(
            getattr(system, "unit_cell", None)
            or getattr(system, "atoms", None)
            or []
        )
        atomic_numbers = {int(atom.Z) for atom in atoms}
        if atomic_numbers and basis_replaces_core(basis_name, atomic_numbers):
            active = True
    if active:
        raise NotImplementedError(
            f"{driver}: ECP metadata is not supported by the BIPOLE "
            "Hamiltonian. Use an all-electron basis or another periodic "
            "J/K route with complete ECP support."
        )


@dataclass
class PBCBipoleEnergyComponents:
    """Per-iteration energy components in CRYSTAL ``ENECYCLE`` terms."""

    iter: int
    e_total: float
    e_electronic: float
    e_kinetic: float
    e_nuclear_attraction: float
    e_two_electron: float
    e_nuclear_repulsion: float
    e_bielet_zone_ee: Optional[float] = None
    e_ext_el_pole: Optional[float] = None
    e_ext_el_spheropole: Optional[float] = None
    e_j_short_range: Optional[float] = None
    e_j_long_range: Optional[float] = None
    e_exchange: Optional[float] = None
    # The probe-charge-Madelung (exxdiv='ewald') finite-size share of
    # e_exchange, reported separately so a cross-code comparison can see
    # the gauge: this is exactly what the run gains from treating the
    # q -> 0 exchange singularity, so a code that treats it differently
    # (CRYSTAL truncates exchange in real space instead) differs from
    # vibe-qc by roughly this much at the same k-mesh. ``None`` when no
    # corrected exchange split ran (pure functionals, legacy gauge);
    # EXACTLY 0.0 under exchange_exxdiv='none'. See #82: -2.88 Ha on
    # MgO/STO-3G (2,2,2) RHF, against a 0.64 Ha filed cross-code offset.
    e_exchange_finite_size: Optional[float] = None
    e_j_multipole: Optional[float] = None
    e_dft_plus_u: Optional[float] = None


def _zero_cross_cell_density(density: LatticeMatrixSet, nbf: int, n_k: int) -> None:
    """Zero out P(g!=0) blocks for Γ-only BIPOLE density locality.

    At Γ-only the inverse Bloch transform
    :func:`real_space_density_from_kpoints` produces ``P(g) = P(Γ)`` at
    *every* lattice cell, which is mathematically correct as a Bloch sum
    but destroys the density-locality convention that CRYSTAL-style BIPOLE
    Fock builds assume (``P(g!=0) = 0``).  Without this fix, the Fock build
    will include exchange contributions from all ``n_cells^2`` cell pairs
    instead of just the home cell, overcounting K by a factor of
    ``n_cells``.

    Only active when ``n_k == 1`` (Γ-only).  Multi-k naturally recovers
    ``P(g!=0) -> 0`` as ``|g| -> inf`` via the k-averaging in the Bloch
    transform.

    See ``cpp/src/periodic_scf.cpp``
    ``real_space_density_from_kpoints`` Γ-only caveat.
    """
    if n_k != 1:
        return
    zero_block = np.zeros((nbf, nbf), dtype=float)
    for g_idx in range(len(density.cells)):
        if not (density.cells[g_idx].index == np.array([0, 0, 0])).all():
            density.set_block(g_idx, zero_block)


@contextlib.contextmanager
def _bipole_exact_build_phase(
    plog,
    *,
    event: str,
    what: str,
    iter_idx: int,
    energy: float,
    converged: bool,
    n_k: int,
    nbf: int,
    reused: bool,
):
    """Announce, make durable, time and truthfully close one exact rebuild.

    Shared body of :func:`bipole_confirmation_phase` (the in-loop exact
    confirmation) and :func:`bipole_final_density_phase` (the post-loop
    evaluation of the density returned to callers). The ``<event>_begin``
    record carries the loop's energy, iteration and convergence flag and is
    written BEFORE the work, so a wall-kill inside the build preserves the
    loop's provisional result (#115). The ``<event>_end`` record carries the
    wall time and a ``status`` of ``"done"``, ``"reused"`` or ``"raised"``; a raising
    build is reported as such and re-raised, never as a successful end.
    The ambient output also receives and flushes the provisional energy
    before work, including runs without an enabled structured log.
    """
    from .output import flush, write
    from .output.document import OutputDocument, Quantity
    from .structured_log import emit as _emit_structured

    document = OutputDocument().section(
        "Exact density evaluation pending", unit_for="energy"
    )
    document.scalar("Phase", what)
    document.scalar("Iteration", str(int(iter_idx)))
    document.scalar("Provisional SCF energy", Quantity(float(energy), "energy"))
    document.scalar(
        "Build", "reuse exact confirmation" if reused else "cold full density"
    )
    write(document.blank().render())
    flush()

    def record_outcome(status: str, wall_s: float) -> None:
        document = OutputDocument().section(
            f"Exact density evaluation {status}", unit_for="duration"
        )
        document.scalar("Phase", what)
        document.scalar("Wall time", Quantity(wall_s, "duration"))
        write(document.blank().render())
        flush()

    if reused:
        plog.info(
            f"  {what}: reusing the exact confirmation rebuild of iteration "
            f"{int(iter_idx)} (no extra Fock build)"
        )
    else:
        plog.info(
            f"  {what}: one non-incremental Fock build "
            f"({int(n_k)} k-point(s) x {int(nbf)} BFs) over the committed "
            "density. This cold build screens the full-density image "
            "support. Its cost is timed separately and can exceed the "
            "initial-guess build."
        )
    _emit_structured(
        f"{event}_begin",
        n_iter=int(iter_idx),
        energy=float(energy),
        converged=bool(converged),
        n_k=int(n_k),
        nbf=int(nbf),
        reused=bool(reused),
    )
    t0 = time.perf_counter()
    try:
        yield
    except BaseException as exc:
        dt = time.perf_counter() - t0
        record_outcome("raised", dt)
        plog.info(
            f"  {what} FAILED after {dt:.1f} s: {type(exc).__name__}: {exc}"
        )
        _emit_structured(
            f"{event}_end",
            wall_s=float(dt),
            status="raised",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    dt = time.perf_counter() - t0
    record_outcome("reused" if reused else "done", dt)
    plog.info(f"  {what} done ({dt:.1f} s)")
    _emit_structured(
        f"{event}_end",
        wall_s=float(dt),
        status="reused" if reused else "done",
    )


@contextlib.contextmanager
def bipole_confirmation_phase(
    plog,
    *,
    iter_idx: int,
    energy: float,
    n_k: int,
    nbf: int,
):
    """The in-loop exact confirmation rebuild, announced before it runs (#115).

    Since #514 the expensive cold rebuild of a converged run happens INSIDE
    the loop (the #116 terminal check judges it), before the post-loop phase
    that used to announce it, so the durable ``scf_final_density_begin``
    record came after the work it was written to protect: a job killed
    during the rebuild still lost its provisional energy. This phase wraps
    that build: ``scf_exact_confirmation_begin`` carries the loop's
    provisional energy and iteration first, the text says what is running
    and why it is expensive, and the end record reports the wall time or the
    exception. The post-loop phase then reuses the build at no cost.
    """
    with _bipole_exact_build_phase(
        plog,
        event="scf_exact_confirmation",
        what=(
            f"exact-operator confirmation of the provisional convergence at "
            f"iteration {int(iter_idx)}"
        ),
        iter_idx=iter_idx,
        energy=energy,
        converged=True,
        n_k=n_k,
        nbf=nbf,
        reused=False,
    ):
        yield


@contextlib.contextmanager
def bipole_final_density_phase(
    plog,
    *,
    iter_idx: int,
    energy: float,
    converged: bool,
    n_k: int,
    nbf: int,
    reused: bool = False,
):
    """Announce, time and record the post-loop final-density evaluation.

    Every BIPOLE driver ends its SCF loop with one **non-incremental** Fock
    build over the committed density (the loop diagonalises and commits a new
    density *after* evaluating the previous one, so returning the loop's Fock
    would mix two SCF states). Because it starts cold it forfeits the
    accumulated ``ΔD`` density-envelope screening the SCF built up, and
    therefore costs about a *first* iteration rather than a last one --
    measured at ~22,000 s on a six-basis-function LiH cell whose final
    iterations cost 819 s (#115).

    Before this wrapper the phase emitted **nothing**. A 24 h canary
    converged at wall 67,593 s and then spent >= 5.22 h here in silence
    before the wall killed it, which destroyed a converged result and was
    indistinguishable from a hang. Two things follow, and this context
    manager does both:

    * the phase announces itself **before** it starts, with what it is doing
      and why it is expensive, so a long silence is attributable;
    * the loop's converged energy is emitted to the structured log **before**
      the expensive work begins, so a wall-kill during finalisation can no
      longer lose it.

    On a converged run the build already happened inside the loop under
    :func:`bipole_confirmation_phase` (the #116 terminal check); pass
    ``reused=True`` and the phase says so and costs nothing. The bound on
    the post-SCF cost is therefore one cold build per exact confirmation
    the loop needed (usually one) and none here. A max-iteration exit builds
    once here unless confirmation already evaluated that final density;
    reuse does not turn a failed confirmation into convergence.
    A build that raises ends with ``status="raised"``
    and re-raises; it is never reported as done.

    The text goes through the ordinary :class:`vibeqc.progress.ProgressLogger`
    (``info``, verbose >= 2) and the events through
    :func:`vibeqc.structured_log.emit`; neither adds a new output surface
    (CLAUDE.md § 16).
    """
    status = "converged" if converged else "NOT converged"
    plog.info(
        f"SCF loop finished at iteration {int(iter_idx)} ({status}); "
        f"E = {float(energy):.10f} Ha"
    )
    with _bipole_exact_build_phase(
        plog,
        event="scf_final_density",
        what="final density evaluation",
        iter_idx=iter_idx,
        energy=energy,
        converged=converged,
        n_k=n_k,
        nbf=nbf,
        reused=reused,
    ):
        yield


def s_fold_truncation_drift(
    basis,
    system,
    lat_opts: LatticeSumOptions,
    *,
    extend_factor: float = 1.5,
    k_points: Optional[Sequence[np.ndarray]] = None,
) -> float:
    """Max-element drift of the S(k) Bloch fold vs an extended cutoff.

    Measures how converged the overlap lattice fold is at
    ``lat_opts.cutoff_bohr`` by comparing against a fold at
    ``extend_factor x`` the cutoff (one-electron integrals -- cheap).
    The corrected (Ewald-exchange-split) BIPOLE gauge contracts full
    Bloch folds, so diffuse AO tails demand fold-converged cutoffs:
    on MgO/STO-3G the Γ drift is 3.9e-1 at 8 bohr (spurious SCF
    states possible), 1.5e-2 at 10, 5.0e-3 at 12 and 2.3e-6 at 16
    (2026-06-10 diagnosis). The legacy projected gauge never
    contracted cross-cell operator blocks and is insensitive to this.

    ``k_points``: optional mesh of crystal momenta. When given, the
    drift is the max over all requested k of the per-k fold drift
    ``|S_g e^{+ik.R_g}.ΔS(g)|_max`` -- the multi-k split contracts
    S(k) at every mesh point, and the alternating phases truncate
    differently from the Γ fold (zone-boundary folds can be better
    OR worse converged). Default (``None``) keeps the historical
    Γ-only measure.
    """
    from ._vibeqc_core import compute_overlap_lattice

    if k_points is None or len(k_points) == 0:
        k_list = [np.zeros(3)]
    else:
        k_list = [np.asarray(k, dtype=float).reshape(3) for k in k_points]

    def _folds(lat):
        S_lat = compute_overlap_lattice(basis, system, lat)
        blocks = [
            np.asarray(S_lat.blocks[c], dtype=float) for c in range(len(S_lat.cells))
        ]
        return [_bloch_sum_blocks(blocks, S_lat.cells, k) for k in k_list]

    lat_ext = _lattice_options_passthrough(lat_opts)
    lat_ext.cutoff_bohr = float(extend_factor) * float(lat_opts.cutoff_bohr)
    lat_ext.nuclear_cutoff_bohr = lat_ext.cutoff_bohr
    base = _folds(lat_opts)
    ext = _folds(lat_ext)
    return float(max(np.abs(b - e).max() for b, e in zip(base, ext)))


def validate_bipole_fold_drift(value: object) -> float:
    """Return one finite nonnegative overlap-fold diagnostic or fail closed."""

    if isinstance(value, (bool, np.bool_)):
        raise RuntimeError(
            "BIPOLE overlap-fold preflight returned a boolean instead of a "
            "finite nonnegative drift"
        )
    try:
        drift = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "BIPOLE overlap-fold preflight did not return a finite "
            "nonnegative drift"
        ) from exc
    if not np.isfinite(drift) or drift < 0.0:
        raise RuntimeError(
            "BIPOLE overlap-fold preflight did not return a finite "
            f"nonnegative drift: {drift!r}"
        )
    return drift


#: Finite energy (Ha) an optimizer returns for a *trial* geometry whose SCF
#: the BIPOLE stack refused (:class:`BipoleFoldUnreliableError`) or that did
#: not converge. One shared value for every periodic optimizer, so the
#: recovery contract below cannot drift between routes: it must exceed any
#: physical energy by orders of magnitude so an Armijo / sufficient-decrease
#: line search rejects the point and shrinks the step, and it is returned
#: with a zero gradient so a curvature update sees no false slope. It is
#: never a result -- a refusal at the *initial* geometry still aborts,
#: because there is nothing to back off to (GitLab IID 536).
BIPOLE_TRIAL_PENALTY_HA = 1.0e6


class BipoleFoldUnreliableError(RuntimeError):
    """The BIPOLE fold-support preflight refused to enter SCF.

    A distinct type (still a ``RuntimeError`` for existing callers) so
    optimizers can treat a refusal at a wild *trial* geometry -- a line
    search overstepping into a configuration whose lattice fold is
    genuinely unreliable -- as a recoverable penalty-barrier evaluation
    rather than a fatal job error. A refusal at the user's own geometry
    still aborts: there is nothing to back off to.
    """


def raise_if_bipole_fold_unreliable(
    label: str,
    drift: float,
    cutoff_bohr: float,
) -> None:
    """Refuse corrected-gauge SCF on critically truncated AO support.

    The direct BIPOLE drivers already classify fold drift above ``1e-2`` as
    numerically unreliable and tell users to converge it below ``1e-4``.
    Continuing after that diagnosis can spend hours building a Fock operator
    on a metric that admits spurious SCF states. Moderate drift keeps the
    existing note-only behavior; callers invoke this helper only in the
    established unreliable branch.
    """
    raise BipoleFoldUnreliableError(
        f"{label} {float(drift):.1e} at cutoff "
        f"{float(cutoff_bohr):.1f} bohr is in the unreliable numerical-"
        "support regime; refusing to enter SCF. Increase the BIPOLE lattice "
        "cutoff (`bipole_cutoff_bohr` in `run_periodic_job`, or "
        "`options.lattice_opts.cutoff_bohr` in a direct driver) until the "
        "fold drift falls below 1e-4."
    )


_HA2EV = 27.211386245988


def smearing_basin_warning(
    smearing_T: float,
    spin_channels: Sequence[
        Tuple[Sequence[np.ndarray], Sequence[np.ndarray], int, float]
    ],
    entropy: float,
    driver: str,
) -> Optional[str]:
    """Surface a near-metallic-basin risk when KS smearing straddles the gap.

    On minimal / under-converged bases the HOMO-LUMO gap can be far
    smaller than the physical gap (MgO/STO-3G: ~0.3 eV vs the ~7.8 eV
    experimental gap -- the minimal basis crushes it). A finite-T
    smearing whose window is comparable to that (artificially small)
    gap fractionally occupies states across it, settling the SCF into a
    near-metallic basin whose total energy can sit *hundreds of mHa*
    from the physical (gapped) solution -- the documented ionic-Γ basin
    trap (MgO RKS Γ +330 mHa at T = 0.01 Ha vs the gap, 2026-06-13
    diagnosis in handovers/HANDOVER_BIPOLE_PRODUCTION.md Sec.0a).

    This is a **diagnostic only** -- it changes no energy and applies no
    convergence aid (CLAUDE.md Sec.7: surface a basin problem, don't bury
    it). It returns a warning string (the driver logs + emits it) when:

    * smearing is on (``smearing_T > 0``), AND
    * there is a genuine gap (``gap_min > 0.02 eV`` -- avoids warning on
      true (near-)metals where fractional occupation is physical), AND
    * the converged state actually carries fractional occupations, AND
    * the smearing window is a large fraction of the gap
      (``k_B.T ≳ 1/2.gap``), i.e. the smearing is washing the gap out.

    ``spin_channels``: one ``(eps_per_k, occ_per_k, n_occ, full_occ)``
    tuple per spin channel -- RKS passes one (``full_occ = 2``), UKS
    passes alpha + beta (``full_occ = 1`` each). ``gap_min`` is the
    minimum per-k HOMO-LUMO gap across all channels (the most-straddled
    one). Returns ``None`` when no straddle is detected.
    """
    if smearing_T <= 0.0:
        return None
    gap_min_ha: Optional[float] = None
    n_frac = 0
    for eps_per_k, occ_per_k, n_occ, full_occ in spin_channels:
        if n_occ < 1:
            continue
        for eps in eps_per_k:
            e = np.sort(np.real(np.asarray(eps, dtype=float)))
            if e.size > n_occ:
                g = float(e[n_occ] - e[n_occ - 1])
                gap_min_ha = g if gap_min_ha is None else min(gap_min_ha, g)
        for occ in occ_per_k:
            o = np.asarray(occ, dtype=float)
            n_frac += int(((o > 1e-4) & (o < float(full_occ) - 1e-4)).sum())
    if gap_min_ha is None:
        return None
    gap_eV = gap_min_ha * _HA2EV
    T_eV = float(smearing_T) * _HA2EV
    if gap_eV > 0.02 and n_frac > 0 and T_eV > 0.5 * gap_eV:
        return (
            f"{driver}: smearing_temperature {T_eV:.3f} eV "
            f"({float(smearing_T):.4f} Ha) is comparable to the minimum "
            f"HOMO-LUMO gap {gap_eV:.3f} eV -- the converged state carries "
            f"{n_frac} fractional occupation(s) (entropy {float(entropy):.3f}), "
            f"a near-metallic basin. On a gapped insulator this can settle "
            f"a wrong basin hundreds of mHa from the physical (gapped) "
            f"solution; reduce smearing_temperature well below the gap "
            f"(e.g. <= {0.2 * gap_eV:.3f} eV) to converge to it. See the "
            f"ionic-Γ basin note in docs/troubleshooting.md."
        )
    return None


def home_cell_block(density: LatticeMatrixSet) -> np.ndarray:
    """Return a copy of the (0,0,0)-cell block of a lattice matrix set.

    At Γ-only (n_k = 1) the home-cell block *is* the BvK representative
    D(Γ) for every density representation the BIPOLE drivers produce:
    the full Bloch fold stores ``P(g) = D(Γ)`` at every cell, local
    (SAD-style) guesses store ``P(0) = D`` with zeros elsewhere, and
    damped/ODA mixes of either preserve the property. Forward Bloch
    summation ``S_g P(g)`` over a cutoff cell list is *not* equivalent
    for the unprojected Bloch fold (it overcounts by n_cells), so the
    Ewald-exchange-split Γ path reads D(Γ) through this helper instead.
    """
    for g_idx in range(len(density.cells)):
        if not np.asarray(density.cells[g_idx].index, dtype=int).any():
            return np.asarray(density.blocks[g_idx], dtype=float).copy()
    raise RuntimeError("home_cell_block: lattice set has no (0,0,0) cell")


def bvk_torus_density_matrices(
    density: LatticeMatrixSet,
    k_points: Sequence[np.ndarray],
    mesh: Sequence[int],
) -> list[np.ndarray]:
    """Recover per-k density matrices from a real-space density set.

    Folds over exactly ONE complete residue system of lattice cells --
    a Born-von-Kármán torus of the Monkhorst-Pack mesh
    ``(n_1, n_2, n_3)``, using the minimal-``|R|`` representative of
    each residue class ``m mod n`` present in the density list::

        D(k) = S_{g in torus} exp(+i k.R_g) . D(g)

    This inverts :func:`real_space_density_from_kpoints` **exactly**
    for every density representation the BIPOLE drivers produce:

    * The stored fold ``D(g) = S_k w_k Re[e^{-ik.R_g} D(k)]`` is
      BvK-periodic in ``g``, so the torus geometric sums collapse to
      ``N_k.d`` exactly (no truncated-sphere phase leakage -- a fold
      over the full cutoff cell list would overcount by ~n_cells/N_k
      and leak between k-points). The ``Re[]`` is lossless because
      Monkhorst-Pack meshes (any 0/1 shift) are time-reversal
      symmetric mod G and the BIPOLE Fock blocks are real, giving
      ``D(-k) = D(k)*``: the fold returns ``1/2[D(k) + D(-k)*] = D(k)``.
    * Linear mixtures (damping, ODA) of such folds are BvK-periodic
      too -- the fold of the mixture is the mixture of the ``D(k)``.
    * SAD/PATOM-local guesses (``D(g=0)`` only) give the intended
      ``D(k) = D(0)`` at every k -- only the home cell contributes.

    At mesh = (1,1,1) the torus is the home cell: this reduces to
    :func:`home_cell_block` (the Γ BvK representative).

    Requires uniform weights ``w_k = 1/N_k`` (the multi-k Ewald-J
    split already enforces this) and a density list covering the
    torus cells -- the Ewald-exchange-split density list (2x the
    operator cutoff) covers it whenever the cutoff spans half the
    BvK diameter, which fold-converged cutoffs always do. Raises
    with the missing cell index otherwise.
    """
    n = [int(x) for x in mesh]
    if len(n) != 3 or any(x < 1 for x in n):
        raise ValueError(
            f"bvk_torus_density_matrices: mesh must be three positive "
            f"integers; got {list(mesh)}"
        )
    blocks_by_cell = {
        _cell_key(cell): np.asarray(block, dtype=float)
        for cell, block in zip(density.cells, density.blocks)
    }
    r_cart_by_cell = {
        _cell_key(cell): np.asarray(cell.r_cart, dtype=float) for cell in density.cells
    }
    # One representative per residue class (m mod n) -- ANY complete
    # residue system inverts the BvK-periodic fold exactly, so pick
    # the minimal-|R| representative available in the density list.
    # On non-orthogonal cells this matters: the naive index box
    # [-n//2, n-n//2)^3 puts the fcc (-1,-1,-1) corner at |a₁+a₂+a₃| =
    # a√3 ≈ 13.8 bohr for MgO while the same class is covered by
    # (-1,1,1) <-> (a,0,0) at 7.96 bohr.
    rep_by_class: dict = {}
    for key, r_cart in r_cart_by_cell.items():
        cls = (key[0] % n[0], key[1] % n[1], key[2] % n[2])
        norm = float(np.dot(r_cart, r_cart))
        held = rep_by_class.get(cls)
        if held is None or norm < held[0]:
            rep_by_class[cls] = (norm, key)
    classes = [
        (c1, c2, c3) for c1 in range(n[0]) for c2 in range(n[1]) for c3 in range(n[2])
    ]
    missing = [cls for cls in classes if cls not in rep_by_class]
    if missing:
        raise ValueError(
            f"bvk_torus_density_matrices: density cell list does not "
            f"cover the BvK torus of mesh {tuple(n)} -- no cell in "
            f"residue class(es) {missing[:4]}"
            f"{'...' if len(missing) > 4 else ''} (mod mesh). Increase "
            f"lattice_opts.cutoff_bohr so the (2x-cutoff) density list "
            f"holds one representative per class."
        )
    torus_keys = [rep_by_class[cls][1] for cls in classes]
    torus_blocks = [blocks_by_cell[key] for key in torus_keys]
    torus_cell_r = [r_cart_by_cell[key] for key in torus_keys]

    # Use the fused C++ multi-k inverse Bloch when available.
    k_list = [np.asarray(k, dtype=float).reshape(3) for k in k_points]
    try:
        from .._vibeqc_core import inverse_bloch_multi_k as _inv_mk
    except ImportError:
        pass
    else:
        result = _inv_mk(torus_blocks, torus_cell_r, k_list)
        return [np.asarray(r) for r in result]

    # Python fallback: per-k Hermitised Bloch sum.
    out: list[np.ndarray] = []
    for k in k_points:
        k_arr = np.asarray(k, dtype=float).reshape(3)
        D_k = np.zeros_like(blocks_by_cell[(0, 0, 0)], dtype=complex)
        for key in torus_keys:
            phase = np.exp(1j * float(np.dot(k_arr, r_cart_by_cell[key])))
            D_k = D_k + phase * blocks_by_cell[key]
        out.append(0.5 * (D_k + D_k.conj().T))
    return out


def _bloch_sum_blocks(
    blocks: Sequence[np.ndarray],
    cells,
    k_cart: np.ndarray,
) -> np.ndarray:
    """F(k) = S_g exp(+i k.R_g) F(g). Real -> complex result."""
    k = np.asarray(k_cart, dtype=float).reshape(3)
    F_k = np.zeros_like(blocks[0], dtype=complex)
    for g_idx, block in enumerate(blocks):
        R_g = np.asarray(cells[g_idx].r_cart, dtype=float)
        phase = np.exp(1j * float(np.dot(k, R_g)))
        F_k = F_k + phase * np.asarray(block, dtype=float)
    return F_k


def _bloch_sum_blocks_multi_k(
    blocks: Sequence[np.ndarray],
    cells,
    k_points: Sequence[np.ndarray],
) -> list[np.ndarray]:
    """Per-k Bloch folds ``F(k) = S_g exp(+i k.R_g) F(g)`` for a k list.

    Uses the fused OpenMP C++ kernel (``bloch_sum_multi_k``,
    k-parallel, one pass over the real-space blocks) when available;
    the serial per-k :func:`_bloch_sum_blocks` fold is the ImportError
    fallback. Both accumulate the cells in list order, and the kernel's
    cos/sin phase agrees with ``np.exp`` to one ulp.
    """
    k_list = [np.asarray(k, dtype=float).reshape(3) for k in k_points]
    try:
        from ._vibeqc_core import bloch_sum_multi_k as _bloch_mk
    except ImportError:
        return [_bloch_sum_blocks(blocks, cells, k) for k in k_list]
    blocks_f = [np.asarray(b, dtype=float) for b in blocks]
    cell_r = [np.asarray(c.r_cart, dtype=float) for c in cells]
    return [np.asarray(m) for m in _bloch_mk(blocks_f, cell_r, k_list)]


def _cell_key(cell) -> Tuple[int, int, int]:
    return tuple(int(x) for x in np.asarray(cell.index, dtype=int).reshape(3))


def _lattice_contract(
    density: LatticeMatrixSet,
    operator: LatticeMatrixSet,
    *,
    operator_name: str,
) -> float:
    return _lattice_contract_blocks(
        density,
        operator.cells,
        operator.blocks,
        operator_name=operator_name,
    )


def _lattice_contract_blocks(
    density: LatticeMatrixSet,
    operator_cells,
    operator_blocks,
    *,
    operator_name: str,
) -> float:
    """Real-space periodic trace ``S_g S_muν D_muν(g) M_muν(g)``.

    CRYSTAL's BIPOLE energy path contracts the current real-space
    density against real-space operator blocks, not against a Γ-folded
    Bloch sum. The elementwise form mirrors ``cpp/src/periodic_scf.cpp``
    and avoids accidentally adding cross-cell one-electron terms when
    the initial SAD density lives only at ``g = 0``.

    Iterates the OPERATOR cells and looks the density up by cell key:
    the operator's lattice support is the truncation boundary, and the
    density list may legitimately be wider (the Ewald-exchange-split Γ
    path stores the Bloch density out to 2x the operator cutoff so the
    C++ builder can resolve every P(b-a) difference its traversal
    forms). A density cell missing from the operator's list contributes
    nothing -- exactly as in the equal-list case, where every density
    cell has an operator partner. A density that fails to cover the
    operator support is a bookkeeping error and raises. (For SAD-local
    guesses both directions agree: the guess stores explicit zero
    blocks at every cell of the template.)
    """
    d_blocks = {
        _cell_key(cell): np.asarray(block)
        for cell, block in zip(density.cells, density.blocks)
    }
    total = 0.0
    for cell, m_block_raw in zip(operator_cells, operator_blocks):
        key = _cell_key(cell)
        if key not in d_blocks:
            raise ValueError(
                f"_lattice_contract: density is missing cell {key} "
                f"required by {operator_name}"
            )
        m_block = np.asarray(m_block_raw)
        d_block = d_blocks[key]
        if d_block.shape != m_block.shape:
            raise ValueError(
                f"_lattice_contract: shape mismatch for {operator_name} "
                f"at cell {key}: D{d_block.shape} vs M{m_block.shape}"
            )
        total += float(np.real(np.sum(d_block * m_block)))
    return total


def clamp_bipole_nuclear_cutoff(
    system,
    lat_opts: LatticeSumOptions,
    use_ewald_j_split: bool,
    plog,
) -> None:
    """Neutral-cell cutoff coherence under the 3-D Ewald-J split.

    With the Ewald-J split active on a 3-D cell, the nuclear/Ewald
    real-space cell set must not exceed the electronic J/K cell set:
    the library defaults (electronic ``cutoff_bohr=15.0`` vs nuclear
    ``nuclear_cutoff_bohr=25.0``) otherwise put nuclear point-charge
    tails on cells whose compensating electronic charge is truncated
    away, over-binding the molecular limit by mHa (water/PBE0/6-31G
    was the reproducer). ``run_periodic_job`` applies the same clamp
    method-agnostically; this driver-level copy protects direct
    callers (the FD gradient path, validation scripts). Landed for RKS
    in ``9ea2651d``; centralised here so all four BIPOLE drivers share
    one clamp instead of diverging per driver (the pre-centralisation
    state left RHF/UHF/UKS direct callers over-binding).

    **Consequence worth knowing before you reason about the 1e Ewald sum**
    (#478): because this clamps DOWN and never up, ``nuclear_cutoff_bohr``
    is inert as a convergence knob on the Ewald-J split -- sweeping it over
    6, 12.5, 18.5, 25, 40 and 60 bohr on H2/STO-3G/SVWN in a 12-bohr box
    returns a bit-identical energy.  #478 was filed reading that invariance
    as evidence that its residual was structural rather than a truncation
    artefact; it was in fact both, and the knob that reaches it is
    ``ewald_precision`` via :func:`raise_bipole_ewald_real_cutoff`, which
    runs after this clamp and raises the cutoff back to the radius the
    pinned alpha requires.  The clamp's own invariant is unaffected: it
    bounds *unscreened* nuclear tails, and what gets widened is the
    erfc-screened 1e sum on a private ``lat_opts_1e`` copy.

    Mutates ``lat_opts.nuclear_cutoff_bohr`` in place.
    """
    if (
        use_ewald_j_split
        and system.dim == 3
        and lat_opts.nuclear_cutoff_bohr > lat_opts.cutoff_bohr
    ):
        plog.info(
            "  BIPOLE neutral-cell cutoff: clamping nuclear/Ewald real "
            f"cutoff from {lat_opts.nuclear_cutoff_bohr:.2f} to "
            f"{lat_opts.cutoff_bohr:.2f} bohr to match the electronic "
            "J/K cell set"
        )
        lat_opts.nuclear_cutoff_bohr = lat_opts.cutoff_bohr


def ewald_real_cutoff_for_alpha(alpha_bohr_inv: float, tolerance: float) -> float:
    """Real-space radius at which ``erfc(alpha.R)`` has decayed to ``tolerance``.

    The exact inverse of the ``auto_alpha`` convention in
    ``cpp/src/ewald.cpp`` -- ``erfc(alpha.R) ~ tol  <=>  alpha.R =
    sqrt(-ln tol)`` (Allen-Tildesley / DL_POLY / LAMMPS) -- and the same
    expression :func:`vibeqc.bipole_fock_ewald.probe_charge_madelung`
    already uses for its own (alpha-invariant, PySCF-pinned) real-space
    envelope. ``auto_alpha`` derives alpha FROM a cutoff; this derives the
    cutoff FROM an alpha, which is the direction a caller that pins alpha
    explicitly needs.
    """
    alpha = float(alpha_bohr_inv)
    tol = float(tolerance)
    if not alpha > 0.0:
        raise ValueError(
            f"ewald_real_cutoff_for_alpha: alpha must be > 0; got {alpha!r}"
        )
    if not 0.0 < tol < 1.0:
        raise ValueError(
            f"ewald_real_cutoff_for_alpha: tolerance must be in (0, 1); "
            f"got {tol!r}"
        )
    return math.sqrt(-math.log(tol)) / alpha


def ewald_alpha_lower_bound(real_cutoff_bohr: float, tolerance: float) -> float:
    """Smallest Ewald ``alpha`` whose real-space kernel has decayed to
    ``tolerance`` at ``real_cutoff_bohr``.

    The exact inverse of :func:`ewald_real_cutoff_for_alpha` and the
    ``auto_alpha`` convention in ``cpp/src/ewald.cpp``: with
    ``erfc(x) <= exp(-x^2)`` for ``x >= 0``, requiring
    ``erfc(alpha R_cut) <= tol`` gives ``alpha >= sqrt(-ln tol) / R_cut``
    (Allen-Tildesley section 6.3; DL_POLY and LAMMPS use the same rule).
    """
    r_cut = float(real_cutoff_bohr)
    tol = float(tolerance)
    if not r_cut > 0.0:
        raise ValueError(
            f"ewald_alpha_lower_bound: real_cutoff_bohr must be > 0; got {r_cut!r}"
        )
    if not 0.0 < tol < 1.0:
        raise ValueError(
            f"ewald_alpha_lower_bound: tolerance must be in (0, 1); got {tol!r}"
        )
    return math.sqrt(-math.log(tol)) / r_cut


def default_ewald_alpha(
    V_cell_bohr3: float, *, real_cutoff_bohr: float, tolerance: float
) -> float:
    """Default Ewald splitting ``alpha`` for a route whose erfc-screened
    real-space sums are truncated at a FIXED image cutoff (GitLab #651).

    Ewald's split trades the two sums against each other: the reciprocal
    series converges faster the smaller the splitting parameter and the
    real-space series faster the larger it is, so "ein mittlerer Wert" is
    chosen for the numerical work (Ewald, Ann. Phys. 369, 253 (1921), the
    discussion of Eq. (32), p. 273, DOI 10.1002/andp.19213690304).
    CRYSTAL's ``alpha = 2.8 / V^(1/3)`` (:func:`crystal_default_ewald_alpha`)
    is such a middle value, and vibe-qc keeps it for term-by-term parity --
    but it scales DOWN with the cell volume while the EWALD_3D drivers cut
    the erfc(alpha r)/r arm of their corrected exchange split (the ``K_SR``
    of ``exchange_exxdiv='ewald'``, the default on every mesh other than a
    lone Gamma point and opt-in there) at the fixed
    ``LatticeSumOptions.cutoff_bohr`` (15 bohr). The other pieces are not
    exposed: the Hartree J is analytic-FT and alpha-invariant, and V_ne /
    E_nn size their own alpha from ``nuclear_cutoff_bohr``. Past
    ``V^(1/3) = 2.8 R_cut / sqrt(-ln tol)`` (8.0 bohr at the defaults) the
    truncated exchange sum is silently short: on H2 in a 16-bohr cube at a
    (2,1,1) mesh, ``alpha = 0.175`` at 15 bohr is 4.29e-5 Ha above the
    30-bohr value, while ``alpha = 0.35`` at 15 bohr is within 1.1e-8 Ha of
    it; the (2,2,2) H2 supercell at Gamma with ``exchange_exxdiv='ewald'``
    (tests/test_ccm_periodic_3d.py) is 6.1e-5 Ha/cell short at 0.175.

    This function is the one place that pairs the two halves for those
    routes: it returns CRYSTAL's value wherever that already satisfies the
    pointwise bound of :func:`ewald_alpha_lower_bound`, and the bound
    otherwise. Raising ``alpha`` moves work into the reciprocal sums, so
    every consumer must derive its reciprocal envelope from the ``alpha`` in
    use (``auto_recip_cutoff`` in ``cpp/src/ewald.cpp``;
    ``corrected_ewald_reciprocal_cutoff`` in ``periodic_corrected_exchange``;
    the FFT grid of the analytic-FT J). The ROHF and ROKS multi-k drivers
    cut their K_LR arm at CRYSTAL's volume-only ``K_max`` until #651 and
    were moved onto ``corrected_ewald_reciprocal_cutoff(V, alpha)`` in the
    same change: with the volume-only envelope, H2 in a 30-bohr box at
    ``alpha = 0.438`` read -1.0790 Ha against the split-invariant
    -1.1170858 Ha, while CRYSTAL's 0.0933 at the 12-bohr cutoff was 1.9e-5
    Ha short of it. The BIPOLE corrected exchange split takes this same
    bound through :func:`resolve_bipole_ewald_alpha` and resolves its
    envelope with ``bipole_ewald_reciprocal_cutoff`` (GitLab #674);
    BIPOLE's one-electron terms take the other branch of the same
    contract and widen their nuclear image ball from ``alpha``
    (:func:`ewald_erfc_lattice_options`), which is affordable there because
    only the one-electron sum is widened.

    The closed-form RMS truncation-error estimates (Kolafa and Perram, Mol.
    Simul. 9, 351 (1992)) are not in the paper library yet; they are
    requested in ``library/maintenance/manual-dois.txt`` and this pointwise
    bound should be upgraded to them once filed.
    """
    from .bipole_ext_el_pole import crystal_default_ewald_alpha

    return max(
        crystal_default_ewald_alpha(V_cell_bohr3),
        ewald_alpha_lower_bound(real_cutoff_bohr, tolerance),
    )


def resolve_bipole_ewald_alpha(
    V_cell_bohr3: float,
    ewald_omega: float | None,
    lat_opts_2e: LatticeSumOptions,
    tolerance: float,
    *,
    erfc_exchange_arm_active: bool,
    plog=None,
) -> float:
    """The single Ewald ``alpha`` of a 3D BIPOLE run (GitLab #674).

    An explicit ``ewald_omega`` wins. Otherwise the default is CRYSTAL's
    ``2.8 / V^(1/3)`` (:func:`~vibeqc.bipole_ext_el_pole.crystal_default_ewald_alpha`),
    which the legacy CRYSTAL-gauge scaffold needs for term-by-term parity,
    EXCEPT when the run carries an erfc-screened exchange arm, i.e. the
    corrected exchange split with a nonzero full-range exact-exchange
    coefficient. That ``K_SR`` is the erfc exchange summed over the Fock
    output cells ``|g| <= lat_opts_2e.cutoff_bohr``, an AO-overlap length
    that does not grow with the cell, while CRYSTAL's ``alpha`` shrinks
    with it: past ``V^(1/3) = 2.8 R_cut / sqrt(-ln tol)`` (7.8 bohr at the
    12-bohr cutoff and ``tol = 1e-8``) ``erfc(alpha R)/R`` has not decayed
    at the cutoff and the image exchange beyond it, for one doubly
    occupied orbital ``-(N_e/2) sum_{|g|>R_cut} erfc(alpha g)/g``, is
    silently dropped. H2/STO-3G in a 30-bohr box at (1,1,1) and cutoff 12
    read -1.1170669756 Ha against the split-invariant -1.1170858285 Ha
    (1.9e-5 Ha short) and the one-electron H atom -0.4667239569 against
    -0.4667330007 (9.0e-6 Ha short). The J arm is not exposed (its
    ket-image ball is padded from ``alpha`` by
    :func:`resolve_bipole_sr_image_extent`) and the 1e Ewald terms size
    their own ball (:func:`raise_bipole_ewald_real_cutoff`); a pure
    functional has no exchange arm and keeps CRYSTAL's value.

    On the exposed route the default is :func:`default_ewald_alpha`:
    CRYSTAL's value bounded below by ``sqrt(-ln tol) / cutoff_bohr`` with
    ``tol = ewald_precision``, the tolerance BIPOLE already uses for its
    erfc image balls. Ewald's potential is invariant to the screening
    parameter once both series are converged (Saunders, Freyria-Fava,
    Dovesi and Roetti, Mol. Phys. 77, 629 (1992), Property 11; Ewald, Ann.
    Phys. 369, 253 (1921)), and CRYSTAL's ``2.8 / V^(1/3)`` (Eq. (55) of
    the 1992 paper) is a cost optimum for fully summed series, not an
    accuracy requirement. Raising ``alpha`` moves work into the reciprocal
    sums, so the caller derives its ``K_max`` from the returned value
    (:func:`~vibeqc.bipole_ext_el_pole.bipole_ewald_reciprocal_cutoff`).
    """
    from .bipole_ext_el_pole import crystal_default_ewald_alpha

    if ewald_omega is not None:
        return float(ewald_omega)
    alpha_crystal = crystal_default_ewald_alpha(V_cell_bohr3)
    if not erfc_exchange_arm_active:
        return alpha_crystal
    r_cut = float(lat_opts_2e.cutoff_bohr)
    alpha = default_ewald_alpha(
        V_cell_bohr3, real_cutoff_bohr=r_cut, tolerance=float(tolerance)
    )
    if plog is not None and alpha > alpha_crystal:
        plog.info(
            f"  Ewald alpha: CRYSTAL's {alpha_crystal:.4f} bohr⁻¹ leaves "
            f"erfc(alpha·R) above {float(tolerance):.0e} at the "
            f"{r_cut:.2f}-bohr exchange cutoff -> bounded to "
            f"{alpha:.4f} bohr⁻¹ (GitLab #674)"
        )
    return alpha


def ewald_erfc_lattice_options(
    lat_opts: LatticeSumOptions,
    alpha_bohr_inv: float,
    tolerance: float,
    *,
    basis: BasisSet,
    system: PeriodicSystem,
) -> LatticeSumOptions:
    """Copy with nuclear images covering the displaced Gaussian products.

    The single source of truth for the erfc half of the 1e Ewald sum (#478).
    Every consumer that walks nuclear images under a *pinned* alpha must go
    through this: the energy's ``compute_nuclear_erfc_lattice`` and its
    derivative partner ``nuclear_erfc_lattice_gradient_contribution`` alike.
    Widening one and not the other desyncs the analytic gradient from the
    energy, which is how the legacy-gauge multi-k UHF fixture went to
    ``max|dg| = 1.3e-3`` against its 1e-3 analytic-vs-FD gate.

    Only ``nuclear_cutoff_bohr`` moves, and only upward. ``cutoff_bohr`` --
    which fixes the returned *cell list*, and with it the
    ``len(V_short.cells) == len(S_lat.cells)`` contract -- is copied
    unchanged, as is every other passthrough field.

    #704: an alpha-only point-charge radius is insufficient for V_ne.
    A primitive AO product at P has exponent p = a + b. Its interaction
    with a point nucleus through erfc(alpha*r)/r decays as
    [erf(sqrt(p)*d) - erf(sqrt(q)*d)]/d, q^-1 = p^-1 + alpha^-2
    (Saunders et al., Mol. Phys. 77, 629 (1992), Eq. 66). Thus the
    slowest s-product erfc factor uses p >= 2*min(a), not alpha alone.
    Moreover P is in the convex hull of the two translated AO centres.
    In the legacy cell-origin domain, for output |g| <= R_out and reference nuclear position C,
    |P-C| <= R_out + max_{shell,C}|O_shell-C|. Add both this displacement
    and the smeared range to the nuclear *cell-origin* ball. Using
    differences of positions makes the radius invariant to a common
    origin shift, including basis centres not on nuclei.

    In the physical pair domain, nuclear images are selected about the
    arithmetic AO-pair midpoint M. Positive primitive exponents imply
    |P-M| <= |A-B|/2 <= cutoff_bohr/2. The source radius therefore needs
    only this half-pair offset plus the smeared range. Cell-origin padding
    would needlessly enlarge the physical source ball when an atom is
    relabelled by a lattice vector.

    This bounds the prototype erfc factor, not the summed integral error:
    angular polynomials, contraction amplitudes and lattice multiplicities
    still require convergence checks. It is not a source-symmetry
    certificate. Energy and derivative callers must use the same helper;
    no derivative of the discrete image-set selection is implied.
    """
    if not math.isfinite(float(alpha_bohr_inv)):
        raise ValueError("ewald_erfc_lattice_options: finite alpha required")
    point_radius = ewald_real_cutoff_for_alpha(alpha_bohr_inv, tolerance)
    shells = list(basis.shells())
    nuclei = [np.asarray(atom.xyz, dtype=float) for atom in system.unit_cell]
    if not shells or not nuclei:
        raise ValueError("ewald_erfc_lattice_options: basis and nuclei are required")
    if any(not np.all(np.isfinite(nucleus)) for nucleus in nuclei):
        raise ValueError("ewald_erfc_lattice_options: invalid nuclear position")
    gamma_min = math.inf
    displacement = 0.0
    for shell in shells:
        exponents = np.asarray(shell.exponents, dtype=float)
        origin = np.asarray(shell.origin, dtype=float)
        if (not exponents.size or not np.all(np.isfinite(exponents))
                or np.any(exponents <= 0.0) or not np.all(np.isfinite(origin))):
            raise ValueError("ewald_erfc_lattice_options: invalid Gaussian shell")
        gamma_min = min(gamma_min, float(exponents.min()))
        if not lat_opts.pair_complete_1e:
            for nucleus in nuclei:
                displacement = max(displacement, float(np.linalg.norm(origin - nucleus)))
    cutoff = float(lat_opts.cutoff_bohr)
    nuclear_cutoff = float(lat_opts.nuclear_cutoff_bohr)
    if (not math.isfinite(cutoff) or cutoff < 0.0
            or not math.isfinite(nuclear_cutoff) or nuclear_cutoff < 0.0
            or not math.isfinite(point_radius)):
        raise ValueError("ewald_erfc_lattice_options: finite nonnegative radii required")
    from scipy.special import erfcinv

    smeared_range = float(erfcinv(tolerance)) * math.hypot(
        1.0 / math.sqrt(2.0 * gamma_min), 1.0 / float(alpha_bohr_inv),
    )
    product_offset = (
        0.5 * cutoff if lat_opts.pair_complete_1e else cutoff + displacement
    )
    gaussian_radius = product_offset + smeared_range
    if not math.isfinite(gaussian_radius):
        raise ValueError("ewald_erfc_lattice_options: nonfinite Gaussian image radius")
    out = _lattice_options_passthrough(lat_opts)
    out.nuclear_cutoff_bohr = max(
        nuclear_cutoff,
        point_radius,
        gaussian_radius,
    )
    return out


def raise_bipole_ewald_real_cutoff(
    system,
    lat_opts_1e: LatticeSumOptions,
    alpha_bohr_inv: float,
    tolerance: float,
    plog,
) -> float:
    """Make the 1e Ewald real-space cutoff consistent with the pinned alpha.

    BIPOLE pins ``alpha = 2.8 / V^(1/3)``
    (:func:`~vibeqc.bipole_ext_el_pole.crystal_default_ewald_alpha`), so
    the erfc kernel at the *nearest image* is ``erfc(alpha.L) =
    erfc(2.8) = 7.5e-5`` for a cubic cell of ANY side ``L`` -- the
    screened kernel is scale-invariant and never becomes negligible at
    the image distance.  Truncating that sum at a fixed
    ``nuclear_cutoff_bohr`` (an AO-overlap length, unrelated to alpha)
    therefore drops a term that decays only as ``1/L``.

    Measured on a single point charge in a cubic box (#478,
    ``ewald_nuclear_repulsion`` vs the simple-cubic Madelung constant
    ``xi.L = 2.837297479481``): at ``rcut = 15`` bohr the error is
    ``-2.251e-4 / L`` Ha for L = 20, 50, 80 and 140 bohr -- constant in
    ``err.L`` to four digits, i.e. exactly the truncated image tail
    ``-(1/2) sum_{R!=0} erfc(alpha.R)/R``.  At the alpha-consistent
    radius returned by :func:`ewald_real_cutoff_for_alpha` the same
    quantity is ``-4e-10 / L``.

    In the SCF this leaks into the Gamma molecular limit as
    ``+ (1/2) N_e^2 sum_{R!=0} erfc(alpha.R)/R = +0.225 N_e^2 / L`` mHa:
    the electron-electron erfc sum is padded to the smeared-kernel range
    by :func:`resolve_bipole_sr_image_extent` and so IS converged, while
    the electron-nuclear (``-2 N_e Z``) and nuclear-nuclear (``+Z^2``)
    channels are truncated, leaving ``-2 N_e Z + Z^2 = -N_e^2`` of the
    monopole triple uncancelled for a neutral cell.

    This does NOT reopen the :func:`clamp_bipole_nuclear_cutoff`
    failure mode.  That clamp keeps the *unscreened* nuclear
    point-charge tail inside the electronic J/K cell set.  Here the
    nuclear sum is erfc-screened and only ``lat_opts_1e`` -- a private
    copy consumed by V_ne and E_nn -- is widened. The electronic
    ``lat_opts_2e`` output set is untouched. This step supplies the
    point-charge radius; :func:`ewald_erfc_lattice_options` additionally
    accounts for displaced Gaussian products in V_ne (#704). The ERI
    sum has its own smeared range and precision. These independently
    chosen image balls need not contain one another; each channel still
    requires a convergence check rather than a charge-counting argument
    based on a common radial cell list.

    Mutates ``lat_opts_1e.nuclear_cutoff_bohr`` in place (never
    shrinks it) and returns the resolved radius.
    """
    if system.dim != 3:
        return float(lat_opts_1e.nuclear_cutoff_bohr)
    required = ewald_real_cutoff_for_alpha(alpha_bohr_inv, tolerance)
    current = float(lat_opts_1e.nuclear_cutoff_bohr)
    if required > current:
        plog.info(
            "  BIPOLE Ewald real cutoff: raising the 1e nuclear/Ewald "
            f"real-space cutoff from {current:.2f} to {required:.2f} bohr "
            f"for alpha={float(alpha_bohr_inv):.5f} bohr^-1 at tolerance "
            f"{float(tolerance):.1e} (erfc(alpha.R) <= tolerance)"
        )
        lat_opts_1e.nuclear_cutoff_bohr = required
        return required
    return current


def resolve_fock_mixing(
    options: object,
    override: Optional[float],
    *,
    where: str,
) -> float:
    """Resolve one requested previous-Fock weight without rewriting options.

    The keyword override wins when it is not ``None``; otherwise the options
    field supplies the request.  Backends may apply an additional documented
    automatic rule to this validated request, as the BIPOLE KS drivers do.
    """

    value = override
    if value is None:
        value = getattr(options, "fock_mixing", 0.0)
    if value is None:
        value = 0.0
    resolved = float(value)
    if not (0.0 <= resolved < 1.0):
        raise ValueError(
            f"{where}: fock_mixing must be in [0, 1); got {resolved}"
        )
    return resolved


def resolve_auto_fock_mixing(
    fock_mixing: float,
    *,
    alpha_hf: float,
    use_diis: bool,
    where: str,
) -> float:
    """Resolve the KS drivers' CRYSTAL-style auto-FMIXING default.

    An explicit ``fock_mixing > 0`` is always honored. When it is left
    at 0, pure/hybrid-DFT runs (``alpha_hf < 1``) *without DIIS* get
    the CRYSTAL-style 30% previous-Fock damping: the bare Roothaan
    iteration on tight ionic cells wobbles and stalls (MgO primitive
    [2,2,2]/SVWN/c6: ~1 mHa oscillation, unconverged at 25 iterations;
    FMIXING-30 converges it in 18 to the same fixed point). Under DIIS
    the mixing is redundant damping applied on top of the extrapolated
    Fock and only slows convergence (same fixture: 14 iterations with
    FMIXING vs 10 without, converged energies identical to 1e-9 Ha),
    so no auto default fires -- Gap-B validation 2026-07-13,
    HANDOVER_BIPOLE_PRODUCTION.md Sec. 0a. Always 0 for HF
    (``alpha_hf == 1``), matching CRYSTAL practice.
    """
    value = float(fock_mixing)
    if value == 0.0 and alpha_hf < 1.0 and not use_diis:
        value = 0.30
    if not (0.0 <= value < 1.0):
        raise ValueError(
            f"{where}: fock_mixing must be in [0, 1); got {value}"
        )
    return value


# User-settable ``LatticeSumOptions`` fields that must survive into the
# derived 1e/2e option sets. ``coulomb_method`` is deliberately absent:
# the CRYSTAL gauge overrides it per channel (2e always DIRECT_TRUNCATED,
# 1e EWALD_3D for 3-D cells). Everything else -- most importantly the
# Schwarz/screening thresholds -- carries the user's value: the derived
# sets used to be fresh ``LatticeSumOptions()`` copying only the two
# cutoffs, so a user-set ``schwarz_threshold`` silently never reached the
# BIPOLE 2-e build (2026-07-13 truncation audit,
# HANDOVER_BIPOLE_PRODUCTION.md Sec. 0a).
_LATTICE_PASSTHROUGH_FIELDS = (
    "cutoff_bohr",
    "nuclear_cutoff_bohr",
    "eri_interaction_cutoff_bohr",
    "becke_image_radius_bohr",
    "slab_ewald_alpha",
    "screening_overlap_threshold",
    "screening_exchange_threshold",
    "schwarz_threshold",
    "schwarz_threshold_forces",
    # The separation-aware charge-pair Schwarz SR
    # screening flag is a user-settable LatticeSumOptions field too --
    # run_periodic_job(sr_range_screening=True) sets it on
    # options.lattice_opts, so it must survive into the derived 2-e set
    # or the build silently runs unscreened while the run still cites
    # the charge-pair bound.
    "sr_range_screening",
    "sr_sparse_traversal",
    # #429 stage 2: the one-electron pair-complete switch must survive into
    # every derived option set or the 1e and 2e term sets silently disagree.
    "pair_complete_1e",
)


def _lattice_options_passthrough(lat_opts: LatticeSumOptions) -> LatticeSumOptions:
    """Fresh ``LatticeSumOptions`` with the user's passthrough fields copied.

    ``hasattr`` guards keep this callable against an older compiled core
    whose ``LatticeSumOptions`` predates a newer field.
    """
    out = LatticeSumOptions()
    for name in _LATTICE_PASSTHROUGH_FIELDS:
        if hasattr(lat_opts, name):
            setattr(out, name, getattr(lat_opts, name))
    return out


def prepare_bipole_lattice_options(
    system,
    lat_opts: LatticeSumOptions,
    use_ewald_j_split: Optional[bool],
    plog,
) -> Tuple[bool, bool, LatticeSumOptions, LatticeSumOptions]:
    """Resolve the Ewald-J split and build the 1e/2e lattice options.

    The block every BIPOLE driver used to inline (M3a extraction):

    1. Resolve ``use_ewald_j_split`` (``None`` -> auto: on for 3-D).
    2. Apply the neutral-cell nuclear-cutoff clamp (in place on
       ``lat_opts`` -- see :func:`clamp_bipole_nuclear_cutoff`).
    3. Build the CRYSTAL-gauge pair as passthrough copies of the user's
       options (cutoffs + screening thresholds; see
       ``_LATTICE_PASSTHROUGH_FIELDS``), overriding only
       ``coulomb_method``: ``lat_opts_2e`` always DIRECT_TRUNCATED
       (F^2e uses the direct cell list for J_SR/K; the optional
       reciprocal J_LR shares the one-electron Ewald alpha),
       ``lat_opts_1e`` EWALD_3D for 3-D cells and DIRECT_TRUNCATED for
       dim < 3 (no Ewald_3D path there).

    Returns ``(use_ewald_j_split, use_ewald_j_split_auto,
    lat_opts_2e, lat_opts_1e)``.
    """
    use_ewald_j_split_auto = use_ewald_j_split is None
    resolved_j_split = (
        system.dim == 3 if use_ewald_j_split_auto else bool(use_ewald_j_split)
    )
    clamp_bipole_nuclear_cutoff(system, lat_opts, resolved_j_split, plog)

    lat_opts_2e = _lattice_options_passthrough(lat_opts)
    lat_opts_2e.coulomb_method = CoulombMethod.DIRECT_TRUNCATED

    lat_opts_1e = _lattice_options_passthrough(lat_opts)
    lat_opts_1e.coulomb_method = (
        CoulombMethod.EWALD_3D if system.dim == 3 else CoulombMethod.DIRECT_TRUNCATED
    )
    return resolved_j_split, use_ewald_j_split_auto, lat_opts_2e, lat_opts_1e


def resolve_bipole_fock_symmetry(
    system,
    basis,
    lat_opts_2e: LatticeSumOptions,
    use_fock_symmetry: Optional[bool],
    use_fock_symmetry_reduce: Optional[bool],
    plog,
    *,
    per_spin: bool = False,
    exchange_split_active: Optional[bool] = None,
    auto_reduce_safe: bool = True,
):
    """Resolve the SYM3b Fock-symmetry map + reduced cell set.

    ``use_fock_symmetry_reduce=None`` auto-enables the pair-resolved,
    orbit-representative build when the crystal carries attached symmetry
    and the corrected Ewald exchange split is active, provided the caller
    confirms the internal build domain is group-invariant through
    ``auto_reduce_safe``. Pass ``False`` to disable that default. Enforcement
    without reduction remains explicit
    (``use_fock_symmetry=True, use_fock_symmetry_reduce=False``).

    ROOT CAUSE of the historical ~4e-2 Ha
    enforcement shift (2026-07-12 probe investigation, see
    HANDOVER_BIPOLE_PRODUCTION.md Sec.0a): the radial cell list is
    closed under the bare point-group action, but the machinery acts
    on atom-pair triples ``h -> R.h + s_a - s_b`` whose per-atom
    shifts (|L.s| up to 13.78 bohr on MgO) push partners of EVERY
    cell outside the radial output/internal/density domains; with
    group-invariant pair-resolved (|r_b + L.h - r_a| <= cutoff)
    domains the truncated blocks are orbit-symmetric to 1e-15 and
    enforcement is an exact no-op. Cell-list expansion provably never
    closes the orbits — do not "fix" this by raising the cutoff.

    M2-M5 of the pair-resolved workstream: when the caller reports the
    Ewald exchange split active (``exchange_split_active=True`` — the
    production default path), the mapping is built on the
    PAIR-RESOLVED domains (``pair_resolved_fock_mapping``): the output
    triple set is group-invariant, the orbit partition runs with
    ``require_closed=True`` (no silently skipped partners), and the
    J/K build routes through the M1 full-domain binding. The truncation
    SET therefore changes when this machinery is active - energies move at
    truncation-error level relative to the radial-domain build. M3 supplies
    the pair-resolved density support and the default M5 padded M4b internal
    domain makes enforcement an exact no-op. Legacy non-split callers
    (``exchange_split_active=False``/``None``) keep the radial-domain
    mapping unchanged. The symmetry-reduced S/T integral path
    (bit-exact) is independent of this flag and stays auto-enabled
    with attached symmetry. See
    tests/test_bipole_symmetry_fock.py::test_symmetry_fock_mgo_energy_invariance.

    ``use_fock_symmetry_reduce`` implies the symmetry machinery -- it
    reduces the direct-ERI build to orbit representatives and
    reconstructs, which requires the same orbit mapping.

    Returns ``(fock_sym_map, rep_cell_indices)``; both ``None`` when
    symmetry is off or unusable for this cell. ``per_spin=True`` only
    changes the log line (the unrestricted drivers build J/K once per
    spin channel).
    """
    attached_symmetry = bool(
        getattr(getattr(system, "symmetry", None), "operations", None)
    )
    if lat_opts_2e.pair_complete_1e:
        if use_fock_symmetry or use_fock_symmetry_reduce:
            raise NotImplementedError(
                "Physical quartet exchange has a wider output support than "
                "the legacy Fock symmetry projector; use the full direct "
                "build with use_fock_symmetry=False and "
                "use_fock_symmetry_reduce=False."
            )
        if attached_symmetry and use_fock_symmetry_reduce is None:
            plog.info(
                "  Physical quartet support: full direct Fock build "
                "(legacy symmetry projector does not cover exchange outputs)"
            )
        return None, None
    reduce_auto = use_fock_symmetry_reduce is None
    reduce_active = (
        attached_symmetry and bool(exchange_split_active) and auto_reduce_safe
        if reduce_auto
        else bool(use_fock_symmetry_reduce)
    )

    if (
        reduce_auto
        and attached_symmetry
        and bool(exchange_split_active)
        and not auto_reduce_safe
    ):
        plog.info(
            "  SYM3b reduction auto-disabled: caller has not certified "
            "the Fock build domain and electronic state as symmetry-covariant"
        )

    fock_sym_map = None
    rep_cell_indices = None
    if not (use_fock_symmetry or reduce_active):
        return fock_sym_map, rep_cell_indices

    from ._vibeqc_core import direct_lattice_cells
    from .bipole_symmetry_fock import (
        cell_orbit_mapping,
        pair_resolved_fock_mapping,
    )

    cells_for_sym = list(direct_lattice_cells(system, lat_opts_2e.cutoff_bohr))
    if exchange_split_active:
        fock_sym_map = pair_resolved_fock_mapping(system, basis, lat_opts_2e)
        if fock_sym_map is not None:
            plog.info(
                "SYM3b Fock symmetry enforcement: ENABLED (pair-resolved "
                "domains)"
            )
            plog.info(
                "  pair-resolved truncation: "
                f"{fock_sym_map.domain.n_triples} (a,b,h) triples on "
                f"{len(fock_sym_map.cells)} cells "
                f"(radial list: {len(cells_for_sym)} cells), "
                f"orbit closure exact"
            )
    else:
        fock_sym_map = cell_orbit_mapping(system, basis, cells_for_sym)
        if fock_sym_map is not None:
            plog.info("SYM3b Fock symmetry enforcement: ENABLED")
    if fock_sym_map is not None:
        if reduce_active:
            from .bipole_symmetry_fock import representative_cell_indices

            rep_cell_indices = representative_cell_indices(fock_sym_map)
            n_out = len(fock_sym_map.cells)
            suffix = " (per spin)" if per_spin else ""
            plog.info(
                "  SYM3b reduced direct-ERI build: "
                f"{len(rep_cell_indices)}/{n_out} "
                f"orbit-representative cells{suffix} "
                f"({n_out / max(len(rep_cell_indices), 1):.1f}x "
                "fewer output blocks)"
            )
            if reduce_auto:
                plog.info(
                    "  SYM3b reduction auto-enabled by attached crystal "
                    "symmetry"
                )
    elif reduce_active:
        plog.info(
            "  SYM3b reduced build requested but no usable symmetry "
            "(falling back to the full direct-ERI build)"
        )
    return fock_sym_map, rep_cell_indices


def resolve_incremental_jk(
    *,
    use_incremental_fock: bool,
    exchange_split_active: bool,
    use_oda: bool,
    plog,
):
    """Resolve the opt-in incremental/differential J_SR(+K_SR) accumulator.

    Active only in the corrected gauge -- where J_SR+K_SR is one fused erfc
    traversal, the linear-in-density piece the accumulator telescopes -- and
    with DIIS: ODA's interleaved naive build would break the per-iter ΔD
    chain.  That single direct traversal is ~99% of the corrected-gauge
    Fock-build wall (2026-06-14 profile).

    Returns the ``IncrementalJK`` accumulator, or ``None`` when the request
    is absent or inactive.
    """

    incremental_jk = None
    if use_incremental_fock:
        if exchange_split_active and not use_oda:
            from .bipole_fock_ewald import IncrementalJK

            incremental_jk = IncrementalJK()
            plog.info(
                "  incremental Fock (differential J_SR/K_SR via ΔD "
                "density-envelope screening): ON"
            )
        else:
            plog.info(
                "  incremental Fock requested but inactive "
                "(needs the corrected gauge + DIIS, not ODA)"
            )
    return incremental_jk


def warn_bipole_exact_j_core_tail(
    basis,
    ke_cutoff: float,
    plog,
    *,
    precision: float = 1e-6,
) -> Optional[float]:
    """Surface the exact-FT Hartree core reciprocal-tail undercoverage.

    The 2026-07-12 split-gap triage (HANDOVER_BIPOLE_PRODUCTION.md
    Sec. 0a) identified the mechanism: the slowest pair decay
    ``e^{-G^2/(4*g_max)}`` needs ``ke >~ 2*ln(1/eps)*g_max`` (~11000 Ha
    on Mg) to converge by brute mesh. The 2026-07-13 Gap-B validation
    measured the magnitude at the production ``VIBEQC_J_EWALD3D_KE =
    200`` on MgO/STO-3G [2,2,2]: ~1.03 Ha of density-independent core
    reciprocal content missing from *absolute* totals (+0.997 Ha ke
    200->800 at fixed SAD density, +0.987 Ha between converged fixed
    points, +0.043 Ha 800->2000; the c12 no-aids fixed point plus this
    tail reproduces PySCF KRKS to ~0.4 mHa -- the earlier ~0.49 Ha
    figure was the ke300-referenced triage number). The tail cancels
    in energy differences, does not change the SCF landscape, but
    breaks absolute CRYSTAL/PySCF parity. The structural fix is the
    M4-repaired SR+LR composition (pair-resolved truncation
    workstream); until that lands, this diagnostic makes the caveat
    visible instead of silent.

    Returns the estimated ke needed for ``precision`` (Ha), or ``None``
    when the current cutoff already covers it (no warning emitted).
    """
    gamma_max = 0.0
    for shell in basis.shells():
        exps = np.asarray(shell.exponents, dtype=float)
        gamma_max = max(gamma_max, float(exps.max()))
    if gamma_max <= 0.0:
        return None
    ke_needed = 2.0 * float(np.log(1.0 / float(precision))) * gamma_max
    if float(ke_cutoff) >= ke_needed:
        return None
    msg = (
        "BIPOLE exact-FT Hartree J: ke_cutoff = "
        f"{float(ke_cutoff):g} Ha does not cover the core reciprocal "
        f"tails of this basis (most compact primitive {gamma_max:g}; "
        f"~{ke_needed:.0f} Ha needed for {precision:g} relative "
        "coverage). ABSOLUTE totals undercount a density-independent "
        "core term (~1.0 Ha on MgO/STO-3G at ke=200, measured "
        "2026-07-13); energy "
        "DIFFERENCES cancel it. For absolute cross-code parity use the "
        "SR+LR split with a converged SR image sum (sr_image_extent_bohr) "
        "or raise VIBEQC_J_EWALD3D_KE."
    )
    warnings.warn(msg, RuntimeWarning, stacklevel=3)
    output_message = (
        f"exact-FT J ke_cutoff {float(ke_cutoff):g} Ha < "
        f"~{ke_needed:.0f} Ha core-tail coverage -- absolute totals "
        "undercount; differences unaffected"
    )
    _emit_bipole_semantic_diagnostic(
        output_message,
        warning=True,
        warning_kind="exact_j_core_tail",
        ke_cutoff_ha=float(ke_cutoff),
        ke_needed_ha=float(ke_needed),
        precision=float(precision),
    )
    plog.info(f"  WARNING: {output_message}")
    return ke_needed


def bipole_sr_image_extent(
    basis,
    system,
    omega: float,
    *,
    precision: float = 1e-8,
    include_atom_offsets: bool = True,
) -> float:
    """Internal (c_λ, c_σ) ket-image ball radius for the SR erfc traversal.

    The 2026-07-12 split-gap triage (HANDOVER_BIPOLE_PRODUCTION.md
    Sec. 0a) root-caused a −515 mHa J_SR undercount on MgO/STO-3G at an
    8-bohr traversal ball: the erfc(ω·r) interaction between *smeared*
    diffuse AO-pair charge distributions decays as
    ``erf(√m·r) − erf(√m_ω·r)`` with ``1/m_ω = 1/γ_bra + 1/γ_ket +
    1/ω²``, so the ket-image sum needs a ball set by the *smeared*
    kernel range — not by the bare cutoff. This helper returns the
    safe absolute radius

        R = cutoff-independent kernel range erfc⁻¹(ε) / √m_ω
            + max_a |r_a|   (atom offsets shift pair centroids)

    evaluated at the slowest possible pair decay. Callers use it as
    ``sr_image_extent_bohr = lat_opts.cutoff_bohr +
    bipole_sr_image_extent(...)``.

    Physical product-centered traversal sets ``include_atom_offsets=False``.
    Its interaction radius is measured between pair midpoints, so absolute
    atom coordinates are irrelevant. Each Gaussian product center differs
    from its pair midpoint by at most half the pair separation; adding
    ``cutoff_bohr`` covers the two displacements. This radial decay estimate
    alone does not bound the total lattice error or angular amplitudes.

    Derivation of the kernel range (the erfc tail inequality, exactly):

    * ``γ_bra`` / ``γ_ket`` in the triage formula are *pair* (Gaussian
      product) exponents. A pair exponent is the SUM of its two
      primitive exponents, so ``γ_pair >= 2·γ_min`` and the slowest
      decay over all quartets is

          1/m_ω <= 1/(2γ_min) + 1/(2γ_min) + 1/ω² = 1/γ_min + 1/ω².

      (This radial-kernel estimate uses the same Gaussian-product exponent
      convention as C++ charge-pair Schwarz screening in
      ``build_jk_2e_real_space_impl``. Angular polynomial factors are
      bounded separately by the quartet screen. The pre-2026-08-05 revision of
      this helper set γ_bra = γ_ket = γ_min — conflating a primitive
      exponent with a pair exponent — which inflated ``1/m_ω`` to
      ``2/γ_min + 1/ω²`` and over-padded every default SR ball by
      ~√2 in the diffuse-basis limit: the measured ~2x over-radius of
      BIPOLE-SR-PAD-OVERCONSERVATIVE, ~40x traversal overhead.)
    * The smallest radius with ``erfc(√m_ω·R) <= ε`` is exactly
      ``R = erfc⁻¹(ε)/√m_ω``. (The pre-fix revision inverted the loose
      Chernoff-style bound ``erfc(x) <= exp(-x²)`` instead:
      ``√ln(1/ε) = 3.717`` vs ``erfc⁻¹(1e-6) = 3.459`` at the
      production ε, another ~7% radius.)

    The ``+ max_a |r_a|`` term keeps the shipped composition: pair
    centroids sit within the convex hull of their two atom positions,
    so a ket-image cell at origin distance R can host centroids up to
    ``max_a |r_a|`` closer to the bra region. The remaining
    conservatism is deliberate and rigorous in the erfc factor: the
    bound ignores the additional 1/R Coulomb decay, the Schwarz
    amplitude of the pairs, and the density envelope — measured
    home-contraction bit-stability radii on the four rocksalt probes
    (LiH 26, LiF 26, MgO 20, NaCl 22-26 bohr absolute at cutoff 6;
    2026-08-05 investigation) all sit BELOW the radii this bound
    derives, with margin.
    """
    if not (omega and float(omega) > 0.0):
        raise ValueError(
            "bipole_sr_image_extent: needs the erfc screening omega > 0 "
            f"(got {omega!r}); the pad is specific to the SR split kernel"
        )
    if not (0.0 < precision < 1.0):
        raise ValueError(
            f"bipole_sr_image_extent: precision must be in (0, 1); "
            f"got {precision!r}"
        )
    gamma_min = None
    for shell in basis.shells():
        exps = np.asarray(shell.exponents, dtype=float)
        smallest = float(exps.min())
        if gamma_min is None or smallest < gamma_min:
            gamma_min = smallest
    if gamma_min is None or gamma_min <= 0.0:
        raise ValueError("bipole_sr_image_extent: basis has no primitives")
    # Slowest smeared-kernel decay over quartets: both pair exponents at
    # their 2·γ_min floor (see the derivation in the docstring).
    inv_m_w = 1.0 / gamma_min + 1.0 / (float(omega) ** 2)
    # Exact inverse of the erfc tail: erfc(√m_ω R) <= ε  <=>
    # R >= erfc⁻¹(ε)·√(1/m_ω).
    from scipy.special import erfcinv

    r_kernel = float(erfcinv(float(precision))) * float(np.sqrt(inv_m_w))
    if not include_atom_offsets:
        return r_kernel
    max_atom = 0.0
    for atom in system.unit_cell_molecule().atoms:
        max_atom = max(max_atom, float(np.linalg.norm(np.asarray(atom.xyz))))
    return r_kernel + max_atom


def resolve_bipole_sr_image_extent(
    basis,
    system,
    lat_opts: LatticeSumOptions,
    lat_opts_2e: LatticeSumOptions,
    omega: Optional[float],
    *,
    use_ewald_j_split: bool,
    sr_image_precision: Optional[float],
    sr_image_extent_bohr: Optional[float],
    erfc_sr_build_active: bool = True,
    plog=None,
) -> Optional[float]:
    """Resolve the BIPOLE SR image or physical pair-interaction radius.

    An explicit ``sr_image_extent_bohr`` keeps the M4a oracle contract and
    wins over the precision policy. Otherwise ``sr_image_precision`` selects
    the production M5 radius

    ``cutoff_bohr + bipole_sr_image_extent(..., precision=...)``.

    Physical pair support measures the interaction radius between pair
    midpoints and omits absolute atom-origin padding from this policy.

    The precision path also enables charge-pair Schwarz screening on
    both the user and derived lattice options, so the setting reaches the C++
    two-electron build and the citation manifest. ``None`` restores the
    historical unpadded internal ball. Routes that do not build an erfc SR
    term (lower-dimensional direct-only runs and pure-RKS exact-FT J) leave
    the radius and screening policy untouched.
    """
    if use_ewald_j_split and system.dim != 3:
        raise ValueError(
            "use_ewald_j_split requires dim=3 (3D periodic). "
            f"Got dim={system.dim}."
        )
    cutoff = float(lat_opts_2e.cutoff_bohr)
    if not (use_ewald_j_split and erfc_sr_build_active):
        return None
    radius_label = ("SR pair-interaction radius" if lat_opts_2e.pair_complete_1e
                    else "SR ket-image radius")

    if sr_image_extent_bohr is not None:
        extent = float(sr_image_extent_bohr)
        if extent <= cutoff:
            raise ValueError(
                "sr_image_extent_bohr must exceed the electronic "
                f"cutoff_bohr ({extent!r} vs {cutoff!r})"
            )
        if plog is not None:
            plog.info(
                f"  {radius_label}: {extent:.2f} bohr "
                "(explicit sr_image_extent_bohr)"
            )
        return extent

    if sr_image_precision is None:
        return None
    precision = float(sr_image_precision)
    if not (0.0 < precision < 1.0):
        raise ValueError(
            f"sr_image_precision must be in (0, 1) or None; got "
            f"{sr_image_precision!r}"
        )
    if omega is None or float(omega) <= 0.0:
        raise RuntimeError(
            "BIPOLE SR image-domain resolution needs the active Ewald "
            "screening omega"
        )

    extent = cutoff + bipole_sr_image_extent(
        basis,
        system,
        float(omega),
        precision=precision,
        include_atom_offsets=not lat_opts_2e.pair_complete_1e,
    )
    lat_opts.sr_range_screening = True
    lat_opts_2e.sr_range_screening = True
    if plog is not None:
        plog.info(
            f"  {radius_label}: {extent:.2f} bohr "
            f"(sr_image_precision={precision:.0e}; "
            "charge-pair Schwarz screening on)"
        )
    return extent


def enforce_bipole_fold_support(
    basis,
    system,
    lat_opts_2e,
    *,
    method: str,
    n_k: int,
    k_points=None,
    exchange_split_active: bool,
    plog,
) -> float:
    """Measure the S(k) lattice-fold truncation drift and fail closed.

    One shared, GAUGE-INDEPENDENT guard for all four BIPOLE drivers
    (maintainer-approved hoist, 2026-08-06; previously each driver
    carried a drifting inline copy behind ``if exchange_split_active:``,
    so legacy-gauge runs skipped the reliability guard entirely --
    HANDOVER_OPEN_BUGS_V015.md, the fold-guard-hole entry).

    Why the guard applies in BOTH gauges: every BIPOLE SCF diagonalizes
    Bloch-folded ``F(k) - e S(k)`` regardless of exchange convention.
    An S(k) assembled from a truncated cell list is not the Gram matrix
    of any orbital set (the GPW non-PSD lesson: LiF min eig -1.5), and
    the eigenproblem is corrupted at first order in the missing overlap
    tail. The corrected gauge additionally contracts the full Bloch
    folds of S/T/V/F2e -- diffuse AO tails (e.g. STO-3G Mg 3sp,
    outermost exponent ~ 0.046) keep cross-cell overlaps alive far
    beyond kernel-driven cutoffs: on MgO/STO-3G the S(Gamma) fold
    truncation is 3.9e-1 at cutoff 8, 1.5e-2 at 10, 5.0e-3 at 12 and
    2.3e-6 at 16 bohr, and an SCF on a grossly under-converged fold can
    descend into spurious states (c8: converged 0.70 Ha below the PySCF
    reference with the electron count off by 0.43; 2026-06-10
    diagnosis).

    Thresholds (established 2026-08-01, unchanged here): drift > 1e-2
    -> semantic warning + ``BipoleFoldUnreliableError`` before Fock
    construction; > 1e-4 -> truncation note; else converged log. The
    measurement recomputes S over a 1.5x-extended cutoff (per-k at
    multi-k) -- one-electron only, cheap, and bracketed in a progress
    stage. Returns the validated drift for result provenance.
    """
    with plog.stage(
        "s_fold_drift",
        detail=f"S recompute at extended cutoff (n_k={n_k})",
    ):
        drift = validate_bipole_fold_drift(
            s_fold_truncation_drift(
                basis,
                system,
                lat_opts_2e,
                k_points=(k_points if n_k > 1 else None),
            )
        )
    label = (
        "S(k) fold truncation (max over mesh)"
        if n_k > 1
        else "S(Γ) fold truncation"
    )
    gauge_suffix = (
        ""
        if exchange_split_active
        else (
            " (legacy gauge: the k-fold eigenproblem consumes S(k) "
            "regardless of the exchange convention)"
        )
    )
    if drift > 1e-2:
        output_message = (
            f"{label} {drift:.1e} at cutoff "
            f"{lat_opts_2e.cutoff_bohr:.1f} bohr -- the lattice sums are "
            f"badly under-converged for this basis's AO tails; "
            f"absolute energies are UNRELIABLE (spurious SCF states "
            f"possible). Increase lattice_opts.cutoff_bohr until the "
            f"drift falls below 1e-4.{gauge_suffix}"
        )
        _emit_bipole_semantic_diagnostic(
            output_message,
            warning=True,
            once_key=f"bipole.fold_truncation.{method}.warning",
            warning_kind="fold_truncation",
            method=method,
            drift=float(drift),
            cutoff_bohr=float(lat_opts_2e.cutoff_bohr),
            n_k=int(n_k),
        )
        plog.info(f"  WARNING: {output_message}")
        raise_if_bipole_fold_unreliable(
            label,
            drift,
            lat_opts_2e.cutoff_bohr,
        )
    elif drift > 1e-4:
        note_message = (
            f"{label} {drift:.1e} at cutoff "
            f"{lat_opts_2e.cutoff_bohr:.1f} bohr -- expect "
            f"~{drift:.0e}-scale absolute-energy truncation; "
            f"increase cutoff_bohr for tighter work{gauge_suffix}"
        )
        _emit_bipole_semantic_diagnostic(
            note_message,
            warning=False,
            once_key=f"bipole.fold_truncation.{method}.note",
            note_kind="fold_truncation",
            method=method,
            drift=float(drift),
            cutoff_bohr=float(lat_opts_2e.cutoff_bohr),
            n_k=int(n_k),
        )
        plog.info(f"  note: {note_message}")
    else:
        plog.info(f"  {label}: {drift:.1e} (converged)")
    return float(drift)


def resolve_bipole_exact_zone(
    system,
    lat_opts_2e: LatticeSumOptions,
    exact_zone_bohr: Optional[float],
    *,
    exchange_split_active: bool,
    sr_image_extent: Optional[float],
    pair_mode: bool,
    rep_cell_indices,
    plog=None,
) -> Optional[float]:
    """Validate + resolve the restricted exact-ERI output zone.

    BIPOLE-EXACT-ZONE increment 1 (2026-08-06): the exact erfc J_SR/K_SR
    traversal's OUTPUT zone may be bounded below the operator/fold
    cutoff. The operator cell template (S/T/V_ne/J_LR blocks, the S(k)
    fold gate, the density support) stays at ``cutoff_bohr``; only
    output cells with ``|r_cell| <= exact_zone_bohr`` receive exact
    erfc quartets, and farther cells carry the reciprocal J_LR channel
    plus the neutralising background alone. This mirrors the
    bielectronic-zone / monoelectronic-zone partition of the CRYSTAL
    Coulomb machinery -- Dovesi, Pisani, Roetti & Saunders,
    Phys. Rev. B 28, 5781 (1983), Eq. (24a) (exact zone + infinite
    multipolar/Ewald model series); Pisani, Dovesi & Roetti, Lecture
    Notes in Chemistry 48 (1988), Sec. II.4b/II.4d -- expressed in the
    Ewald-split gauge, where the reciprocal channel *is* the far-field
    model of the density. It decouples the (cheap, one-electron)
    fold-convergence range from the (expensive) exact bielectronic
    range: a diffuse basis can carry a fold-converged ``cutoff_bohr``
    without paying exact-ERI cost over the whole fold ball.

    Fail-closed preconditions (increment 1): corrected exchange split
    active; the padded SR path in effect (``sr_image_extent`` resolved);
    no SYM3b symmetry reduction or pair-resolved domain (their
    group-invariant output sets are not yet zone-composable).
    """
    if exact_zone_bohr is None:
        return None
    zone = float(exact_zone_bohr)
    cutoff = float(lat_opts_2e.cutoff_bohr)
    if not exchange_split_active:
        raise ValueError(
            "exact_zone_bohr requires the corrected Ewald exchange split "
            "(use_exchange_ewald_split); the legacy gauge has no "
            "reciprocal far-field channel to carry the far cells"
        )
    if sr_image_extent is None:
        raise ValueError(
            "exact_zone_bohr requires the padded SR image path; keep the "
            "default sr_image_precision (or pass sr_image_extent_bohr) "
            "instead of opting out with sr_image_precision=None"
        )
    if pair_mode or rep_cell_indices is not None:
        raise NotImplementedError(
            "exact_zone_bohr is not wired for SYM3b symmetry-reduced or "
            "pair-resolved Fock domains yet; pass "
            "use_fock_symmetry_reduce=False (and disable pair mode) to "
            "combine a restricted exact zone with this run"
        )
    if not zone > 0.0:
        raise ValueError(f"exact_zone_bohr must be positive; got {zone!r}")
    if not zone < cutoff:
        raise ValueError(
            "exact_zone_bohr must be strictly below the operator "
            f"cutoff_bohr ({zone!r} vs {cutoff!r}); at or above the "
            "cutoff the restriction is a no-op -- omit it instead"
        )
    if plog is not None:
        from ._vibeqc_core import direct_lattice_cells

        n_zone = len(list(direct_lattice_cells(system, zone)))
        n_all = len(list(direct_lattice_cells(system, cutoff)))
        plog.info(
            f"  Exact bielectronic zone: {zone:.2f} bohr ({n_zone} of "
            f"{n_all} operator cells); far cells carry J_LR + background "
            "(Ewald far-field model)"
        )
    return zone


def warn_bipole_legacy_multik_gauge(
    system,
    exchange_split_active: bool,
    n_k: int,
    plog,
) -> None:
    """Loud warning: 3-D legacy-gauge multi-k totals are not stationary.

    The legacy (non-exchange-split) 3-D gauge adds the EXT EL-SPHEROPOLE
    term to the reported total energy only, never to the Fock operator,
    so the SCF minimises a functional without dE_sph/dD and the converged
    multi-k total is NOT a stationary value of the reported expression
    (documented Ha-scale wrong-sign k-mesh dependence on MgO: Γ->[2,2,2]
    RAISES E by +1.53 Ha where the reference lowers it -- see
    HANDOVER_BIPOLE_PRODUCTION.md). The configuration stays reachable on
    purpose: ad-hoc k-lists (band paths) auto-fall back to it, and
    diagnostic/parity scripts compare against it. Any ABSOLUTE multi-k
    energy from it must not be trusted; use a Monkhorst-Pack mesh with
    the (default) corrected exchange-split gauge for production numbers.
    """
    if system.dim == 3 and not exchange_split_active and int(n_k) > 1:
        msg = (
            "BIPOLE legacy gauge at multi-k: the spheropole term enters the "
            "energy but not the Fock operator, so the converged total is not "
            "a stationary value and carries a documented Ha-scale k-mesh "
            "dependence. Use a Monkhorst-Pack mesh with the default "
            "corrected exchange-split gauge for trustworthy absolute "
            "energies; this configuration is for band paths and diagnostics."
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
        output_message = (
            "legacy-gauge multi-k total energies are "
            "non-stationary (spheropole in E, not in F) -- diagnostics only"
        )
        _emit_bipole_semantic_diagnostic(
            output_message,
            warning=True,
            warning_kind="legacy_multik_gauge",
            n_k=int(n_k),
        )
        plog.info(f"  WARNING: {output_message}")


def warn_bipole_charged_cell(system, plog) -> None:
    """Warn that a net-charged cell's energy is jellium-background-defined.

    BIPOLE's Ewald pieces neutralise a charged cell with an implicit
    uniform background (jellium). That is the standard charged-defect
    convention (CRYSTAL does the same), but the absolute total energy
    then depends on the background convention and the cell volume, and
    is NOT comparable across cell sizes or codes without a correction
    scheme (e.g. Makov-Payne / Freysoldt). Runs used to proceed with no
    notice; production discipline is to say so once per run.
    """
    net_charge = int(getattr(system, "charge", 0))
    if net_charge != 0:
        msg = (
            f"BIPOLE: net cell charge {net_charge:+d} is neutralised by an "
            "implicit uniform (jellium) background; the absolute total "
            "energy is background-convention- and volume-dependent. "
            "Compare charged-cell energies only via a finite-size "
            "correction scheme (Makov-Payne / Freysoldt)."
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
        output_message = (
            f"net cell charge {net_charge:+d} -> jellium "
            "background; absolute E is convention-dependent"
        )
        _emit_bipole_semantic_diagnostic(
            output_message,
            warning=True,
            warning_kind="charged_cell",
            charge=net_charge,
        )
        plog.info(f"  WARNING: {output_message}")


def _crystal_ewald_options(
    lat_opts: LatticeSumOptions,
    *,
    alpha_bohr_inv: Optional[float],
    tolerance: float,
    recip_cutoff_bohr_inv: Optional[float] = None,
) -> EwaldOptions:
    """Build the single Ewald state used by BIPOLE V_ne / E_nn / J_LR.

    Sharing one Ewald state across V_ne / E_nn / J_LR matters less
    because the final Ewald sum depends on alpha (it should not, in the
    complete limit) and more because finite cutoffs / quadrature do.
    This helper makes that shared state explicit on the Python side.

    When ``recip_cutoff_bohr_inv`` is provided (positive), the C++
    nuclear Ewald will use this K_max instead of auto-computing it
    from a and tolerance.  This guarantees that nuclear and electronic
    Ewald sums use the same reciprocal lattice envelope -- essential
    for G=0 cancellation at finite cutoffs.

    The real-space envelope is held to the same standard (#478): when
    an explicit ``alpha`` is pinned, ``real_cutoff_bohr`` is floored at
    :func:`ewald_real_cutoff_for_alpha`.  ``cpp/src/ewald.cpp`` takes
    ``opts.real_cutoff_bohr`` at face value and only auto-derives alpha
    when alpha is unset, so a pinned alpha paired with an
    AO-overlap-sized cutoff silently truncates the erfc sum -- see
    :func:`raise_bipole_ewald_real_cutoff` for the measured 1/L leak.
    """
    opts = EwaldOptions()
    opts.real_cutoff_bohr = lat_opts.nuclear_cutoff_bohr
    opts.tolerance = tolerance
    if alpha_bohr_inv is not None:
        opts.alpha = float(alpha_bohr_inv)
        opts.real_cutoff_bohr = max(
            float(opts.real_cutoff_bohr),
            ewald_real_cutoff_for_alpha(alpha_bohr_inv, tolerance),
        )
    if recip_cutoff_bohr_inv is not None and recip_cutoff_bohr_inv > 0.0:
        opts.recip_cutoff_bohr_inv = float(recip_cutoff_bohr_inv)
    return opts


def _expand_ibz_kmesh_for_ewald_j(
    system: PeriodicSystem,
    kmesh: BlochKMesh,
    plog: ProgressLogger,
) -> BlochKMesh:
    """Expand an IBZ-reduced MP mesh to the full mesh for Ewald-J.

    The current Ewald-J long-range density transform needs the full
    Bloch-summed AO-pair FT over the whole Monkhorst-Pack mesh. A
    symmetry-reduced ``BlochKMesh`` is fine as user input as long as it
    carries ``ir_mapping`` metadata; we reconstruct the corresponding
    full mesh here so the rest of the SCF loop sees uniform weights.
    """
    ir_mapping = np.asarray(
        getattr(kmesh, "ir_mapping", []),
        dtype=int,
    ).reshape(-1)
    if ir_mapping.size == 0:
        return kmesh

    mesh = tuple(int(x) for x in getattr(kmesh, "mesh", (1, 1, 1)))
    shift = tuple(int(x) for x in getattr(kmesh, "is_shift", (0, 0, 0)))
    full_n = int(np.prod(mesh))
    current_n = len(list(kmesh.kpoints))
    if full_n <= current_n:
        return kmesh

    expanded = _native_monkhorst_pack(
        system,
        list(mesh),
        list(shift),
        False,
    )
    plog.info(
        "  k-mesh symmetry expansion: "
        f"{current_n} IBZ point{'s' if current_n != 1 else ''} "
        f"-> {len(list(expanded.kpoints))} full MP points for Ewald-J "
        f"(mesh={mesh}, shift={shift})"
    )
    return expanded


def _default_bipole_v_ne_grid_options() -> GridOptions:
    """Fallback grid for the smooth long-range Ewald V_ne matrix.

    CRYSTAL evaluates the one-electron Ewald potential analytically. The
    BIPOLE driver now does the same by default via AO-pair Fourier
    transforms. If a caller passes ``v_ne_grid_options`` explicitly, the
    driver falls back to the older quadrature path; use a tighter Lebedev
    grid there than the generic XC default so off-diagonal Hcore elements
    remain stable for CRYSTAL cycle-parity diagnostics.
    """
    opts = GridOptions()
    opts.n_radial = 99
    opts.angular = "lebedev"
    opts.lebedev_order = 41
    opts.angular_pruning = "none"
    opts.partition = "becke"
    return opts


def _compute_nuclear_lattice_ewald_reciprocal_ft(
    basis: BasisSet,
    system: PeriodicSystem,
    lat_opts: LatticeSumOptions,
    ewald_options: EwaldOptions,
    S_lat: LatticeMatrixSet,
    *,
    cache=None,
    precision: float = 1e-8,
    K_max: Optional[float] = None,
):
    """Analytic 3D Ewald V_ne blocks via AO-pair Fourier transforms.

    The generic ``compute_nuclear_lattice_ewald`` path evaluates the
    smooth long-range potential on a molecular quadrature grid. That is
    fine for total energies, but BIPOLE CRYSTAL parity is sensitive to
    off-diagonal Hcore elements after the first Fock diagonalisation.
    Here we evaluate the reciprocal-space part analytically:

    ``V_lr(g) = -S_G kernel(G) r_nuc(G) FT_g(G)^* + pi Q_n/(a^2V) S(g)``.

    The short-range erfc piece remains libint-analytic.

    Parameters
    ----------
    K_max : float, optional
        Explicit reciprocal-space cutoff in bohr⁻¹ for the V_ne
        reciprocal-sum cache.  When provided together with the shared
        Ewald a, guarantees that V_ne and J_LR use the same envelope.
    """
    if system.dim != 3:
        raise ValueError("analytic Ewald V_ne requires dim=3")
    alpha = float(ewald_options.alpha)
    if alpha <= 0.0:
        raise ValueError("analytic Ewald V_ne requires explicit alpha")

    # #478: the erfc half of V_ne is sized from THIS alpha, not from the
    # caller's AO-overlap cutoff. ``compute_nuclear_erfc_lattice`` takes its
    # nuclear image set from ``nuclear_cutoff_bohr`` while its cell list comes
    # from ``cutoff_bohr``, so widening the former leaves the returned cell
    # list -- and the ``len(V_short.cells) == len(S_lat.cells)`` contract
    # below -- untouched. Enforced here rather than only in the drivers so the
    # gradient helpers in ``bipole_gradient.py``, which pass their own
    # unwidened ``lattice_opts`` beside a cell-scaled alpha, stay the exact
    # derivative partner of the energy.
    lat_opts_erfc = ewald_erfc_lattice_options(
        lat_opts, alpha, precision, basis=basis, system=system,
    )

    V_short = compute_nuclear_erfc_lattice(
        basis,
        system,
        alpha,
        lat_opts_erfc,
    )
    if len(V_short.cells) != len(S_lat.cells):
        raise RuntimeError("analytic Ewald V_ne: V_short and S cell lists differ")
    if any(
        _cell_key(v) != _cell_key(s)
        or not np.array_equal(np.asarray(v.r_cart), np.asarray(s.r_cart))
        for v, s in zip(V_short.cells, S_lat.cells)
    ):
        raise RuntimeError("analytic Ewald V_ne: V_short and S cell ordering differs")

    cells_r_cart_arr = np.array(
        [np.asarray(c.r_cart, dtype=float) for c in V_short.cells], dtype=float,
    )
    if cache is not None and not np.array_equal(cache.cells_r_cart, cells_r_cart_arr):
        raise RuntimeError("analytic Ewald V_ne: reciprocal cache cell list differs")
    pair_cutoff = float(lat_opts.cutoff_bohr) if lat_opts.pair_complete_1e else None
    if cache is not None and getattr(cache, "pair_cutoff_bohr", None) != pair_cutoff:
        raise RuntimeError("analytic Ewald V_ne: reciprocal cache pair support differs")

    if cache is None:
        from .bipole_fock_ewald import _build_j_long_range_cache

        cache = _build_j_long_range_cache(
            basis,
            system,
            cells_r_cart_arr,
            alpha,
            precision,
            K_max=K_max, lattice_opts=lat_opts)

    atom_pos = np.array(
        [[float(x) for x in atom.xyz] for atom in system.unit_cell],
        dtype=float,
    )
    atom_z = np.array(
        [float(atom.Z) for atom in system.unit_cell],
        dtype=float,
    )
    phases = np.exp(-1j * (atom_pos @ cache.K_vectors.T))
    rho_nuc = atom_z @ phases
    weighted = cache.kernel * rho_nuc

    a = np.asarray(system.lattice, dtype=float)
    V_cell = float(abs(np.linalg.det(a)))
    q_nuc = float(atom_z.sum())
    background = np.pi * q_nuc / (alpha * alpha * V_cell)

    for c, cell in enumerate(V_short.cells):
        if _cell_key(cell) != _cell_key(S_lat.cells[c]):
            raise RuntimeError(
                "analytic Ewald V_ne: cell ordering differs between V_short and S"
            )
        v_lr = -np.einsum(
            "k,mnk->mn",
            weighted,
            cache.ft_per_cell[c].conj(),
        )
        block = (
            np.asarray(V_short.blocks[c], dtype=float)
            + np.real(v_lr)
            + background * np.asarray(S_lat.blocks[c], dtype=float)
        )
        if lat_opts.pair_complete_1e:
            from .lattice_screening import ao_pair_support_mask

            block[~ao_pair_support_mask(basis, cell.r_cart, lat_opts.cutoff_bohr)] = 0.0
        V_short.set_block(c, block)
    return V_short, cache


def _spin_occupations(system: PeriodicSystem) -> Tuple[int, int]:
    n_elec = int(system.n_electrons())
    mult = int(system.multiplicity)
    if mult < 1:
        raise ValueError(f"run_pbc_bipole_uhf: multiplicity must be >= 1; got {mult}")
    if mult == 1 and n_elec % 2 == 1:
        # 7991ae58: default-multiplicity odd-electron cells (Al, Cu, H
        # primitive cells) promote to the minimum open shell instead of
        # failing. Say so -- a user who MEANT a singlet must not learn
        # about the promotion from a wrong magnetic moment.
        warnings.warn(
            f"BIPOLE open-shell driver: {n_elec} electrons/cell cannot "
            "form a multiplicity=1 singlet; promoting to the minimum "
            "open shell (one unpaired electron, multiplicity 2). Pass an "
            "explicit multiplicity to make the spin state intentional.",
            RuntimeWarning,
            stacklevel=3,
        )
        return (n_elec + 1) // 2, n_elec // 2
    if (n_elec + mult - 1) % 2 != 0 or (n_elec - mult + 1) % 2 != 0:
        raise ValueError(
            f"run_pbc_bipole_uhf: (n_electrons={n_elec}, "
            f"multiplicity={mult}) cannot be split into integer alpha/beta."
        )
    n_alpha = (n_elec + mult - 1) // 2
    n_beta = (n_elec - mult + 1) // 2
    if n_beta < 0:
        raise ValueError(
            f"run_pbc_bipole_uhf: multiplicity={mult} is too large for "
            f"{n_elec} electrons"
        )
    return n_alpha, n_beta


def _combine_density_sets(
    basis: BasisSet,
    system: PeriodicSystem,
    lat_opts: LatticeSumOptions,
    D_alpha: LatticeMatrixSet,
    D_beta: LatticeMatrixSet,
) -> LatticeMatrixSet:
    """Return a fresh ``D_alpha + D_beta`` lattice set."""

    out = compute_overlap_lattice(basis, system, lat_opts)
    alpha_blocks = {
        _cell_key(cell): np.asarray(block, dtype=float)
        for cell, block in zip(D_alpha.cells, D_alpha.blocks)
    }
    beta_blocks = {
        _cell_key(cell): np.asarray(block, dtype=float)
        for cell, block in zip(D_beta.cells, D_beta.blocks)
    }
    for idx, cell in enumerate(out.cells):
        key = _cell_key(cell)
        if key not in alpha_blocks or key not in beta_blocks:
            raise ValueError(
                f"_combine_density_sets: missing spin density block for cell {key}"
            )
        out.set_block(idx, alpha_blocks[key] + beta_blocks[key])
    return out


def _copy_lattice_with_blocks(
    basis: BasisSet,
    system: PeriodicSystem,
    lat_opts: LatticeSumOptions,
    cells,
    blocks: Sequence[np.ndarray],
    *,
    fill_missing: bool = False,
) -> LatticeMatrixSet:
    """Build a fresh lattice matrix set with the given cell-indexed blocks.

    When *fill_missing* is True, cells in the output template that have no
    corresponding block are filled with zeros instead of raising.
    This handles the case where the block source (e.g. exchange K from
    ``build_jk_2e_real_space``) is on a different cell list than the target
    template (e.g. J from ``build_fock_2e_real_space``).
    """

    out = compute_overlap_lattice(basis, system, lat_opts)
    block_by_cell = {
        _cell_key(cell): np.asarray(block, dtype=float)
        for cell, block in zip(cells, blocks)
    }
    for idx, cell in enumerate(out.cells):
        key = _cell_key(cell)
        if key not in block_by_cell:
            if fill_missing:
                out.set_block(idx, np.zeros((basis.nbasis, basis.nbasis), dtype=float))
            else:
                raise ValueError(
                    f"_copy_lattice_with_blocks: missing block for cell {key}"
                )
        else:
            out.set_block(idx, block_by_cell[key])
    return out


def _density_set_gamma_or_lattice(
    template: LatticeMatrixSet,
    D_real: LatticeMatrixSet,
) -> LatticeMatrixSet:
    """Return a ``LatticeMatrixSet`` suitable for XC evaluation.

    If ``D_real`` carries per-cell blocks (multi-k), use it directly.
    Otherwise wrap a Γ-folded matrix in a fresh degenerate set.  The
    operator ``template`` is read-only: the RKS/UKS callers pass ``S_lat``,
    which remains live in the generalized eigenproblem and in the Ewald
    background contraction after the XC build returns.
    """
    if len(D_real.cells) > 1 and D_real.blocks:
        return D_real
    nbf = template.nbf
    D = (
        np.asarray(D_real.blocks[0], dtype=float)
        if D_real.blocks
        else np.zeros((nbf, nbf))
    )
    blocks = [
        D.copy()
        if np.array_equal(np.asarray(cell.index), np.zeros(3, dtype=int))
        else np.zeros_like(D)
        for cell in template.cells
    ]
    return make_lattice_matrix_set(
        nbf,
        list(template.cells),
        blocks,
    )


def build_bipole_one_electron_lattice(
    basis,
    system,
    lat_opts_1e: LatticeSumOptions,
    lat_opts_2e: LatticeSumOptions,
    ewald_options_1e: Optional[EwaldOptions],
    ewald_precision: float,
    ewald_k_max: Optional[float],
    v_ne_grid_options: Optional[GridOptions],
    plog,
    *,
    sym_log_style: str = "rich",
    log_v_ne_ft: bool = False,
):
    """Build the S/T lattice sets and V_ne (+ optional LR cache).

    The body every BIPOLE driver used to inline under its
    ``integrals_lattice`` stage (M3 slice-1 extraction; byte-identical
    operations in the drivers' shared order):

    1. S/T on the two-electron cell list -- symmetry-reduced
       (:func:`compute_overlap_lattice_reduced` /
       :func:`compute_kinetic_lattice_reduced`) when the system carries
       attached symmetry operations, dense otherwise.
    2. V_ne: the analytic AO-pair-FT Ewald path (3-D with an Ewald state
       and no explicit grid override), which also yields the shared
       long-range cache reused by J^LR; otherwise the grid/dispatch
       fallback.

    ``sym_log_style`` preserves each driver's historical live-log wording
    on the symmetry-reduced path (``"rich"`` = RHF/UHF/UKS space-group
    form, ``"plain"`` = RKS operator-count form); ``log_v_ne_ft`` keeps
    RHF's extra "V_ne Ewald long range" line. Log text is the only
    per-driver variation -- numerics are identical.

    Returns ``(S_lat, T_lat, V_lat, v_ne_lr_cache)``.
    """
    from ._vibeqc_core import (
        compute_kinetic_lattice,
        make_lattice_matrix_set,
    )
    from .periodic_v_ne import compute_nuclear_lattice_dispatch
    from .symmetry_integrals_reduced import (
        compute_kinetic_lattice_reduced,
        compute_overlap_lattice_reduced,
        one_electron_lattice_cells,
    )

    _use_sym = bool(
        getattr(getattr(system, "symmetry", None), "operations", None)
    )
    if _use_sym:
        ops = system.symmetry.operations
        # The template the reduced S/T blocks are reconstructed on: the
        # plain ball, or the pair-complete list under pair_complete_1e
        # (#429). Taken from the one place the reduced builders use.
        cells = one_electron_lattice_cells(basis, system, lat_opts_2e)
        if sym_log_style == "plain":
            plog.info(
                f"symmetry-reduced integrals: {len(ops)} operators, "
                f"{len(cells)} lattice cells"
            )
        else:
            plog.info(
                f"S/T integrals: symmetry-reduced path "
                f"(SG {system.symmetry.international_symbol}, "
                f"{system.symmetry.order} ops, "
                f"{len(cells)} lattice cells)"
            )
        _, S_blocks = compute_overlap_lattice_reduced(
            basis,
            system,
            lat_opts_2e,
            ops,
        )
        S_lat = make_lattice_matrix_set(
            basis.nbasis, cells, [np.asarray(b, dtype=float) for b in S_blocks]
        )

        _, T_blocks = compute_kinetic_lattice_reduced(
            basis,
            system,
            lat_opts_2e,
            ops,
        )
        T_lat = make_lattice_matrix_set(
            basis.nbasis, cells, [np.asarray(b, dtype=float) for b in T_blocks]
        )
    else:
        S_lat = compute_overlap_lattice(basis, system, lat_opts_2e)
        T_lat = compute_kinetic_lattice(basis, system, lat_opts_2e)
    v_ne_lr_cache = None
    if (
        system.dim == 3
        and ewald_options_1e is not None
        and v_ne_grid_options is None
    ):
        if log_v_ne_ft:
            plog.info(
                "  V_ne Ewald long range: analytic AO-pair FT "
                "(shared with J^LR cache)"
            )
        V_lat, v_ne_lr_cache = _compute_nuclear_lattice_ewald_reciprocal_ft(
            basis,
            system,
            lat_opts_1e,
            ewald_options_1e,
            S_lat,
            precision=ewald_precision,
            K_max=ewald_k_max,
        )
    else:
        v_ne_grid = (
            v_ne_grid_options
            if v_ne_grid_options is not None
            else (
                _default_bipole_v_ne_grid_options()
                if system.dim == 3
                else None
            )
        )
        V_lat = compute_nuclear_lattice_dispatch(
            basis,
            system,
            lat_opts_1e,
            grid_options=v_ne_grid,
            ewald_options=ewald_options_1e,
        )
    return S_lat, T_lat, V_lat, v_ne_lr_cache


def bloch_h_core_and_orthogonalizer(
    S_lat: LatticeMatrixSet,
    T_lat: LatticeMatrixSet,
    V_lat: LatticeMatrixSet,
    k_arr: np.ndarray,
    linear_dep_threshold: float,
    canonical_orth_normalize_diag_first: bool,
):
    from .periodic_rhf_multi_k_ewald import _canonical_orthogonalizer_complex

    """Bloch-sum S/T/V at one k-point and build H_k plus its orthogonalizer.

    The numerical core every BIPOLE driver's per-k loop used to inline
    (M3 slice-2 extraction): Bloch-sum the three real-space lattice sets,
    Hermitize T_k and V_k INDIVIDUALLY before summing into H_k (not H_k
    after summing), Hermitize S_k, then re-Hermitize H_k. Hermitization
    commutes exactly with addition mathematically, but NOT bit-for-bit
    under floating-point reassociation -- this is the RHF-reference
    order (M2's documented protocol: the shared builder reproduces the
    reference variant bit-identically; DFT variants drift at the
    existing ~1e-13 XC/BLAS noise floor, not from this function).

    Deliberately NOT unified here: the overlap-diagnostic dispatch
    (``scf_preflight_overlap_check`` vs ``check_overlap_matrix`` by
    ``n_k`` threshold, the per-k severity log, and the ``n_k > 16``
    summary line) differs in a way that is not just log wording --
    RHF's summary block and `+inf`-safe condition-number formatting are
    NOT present in RKS/UHF/UKS today. Silently adding them would be a
    behavior change, not a mechanical extraction; it is an open M3
    decision candidate (see HANDOVER_BIPOLE_VARIANT_UNIFICATION.md),
    parallel to the existing UHF SYM3b-reduce candidate.

    Returns ``(S_k, T_k, V_k, H_k, X_k, n_kept)``.
    """
    from ._vibeqc_core import bloch_sum

    S_k = np.asarray(bloch_sum(S_lat, k_arr))
    T_k = np.asarray(bloch_sum(T_lat, k_arr))
    V_k = np.asarray(bloch_sum(V_lat, k_arr))
    T_k = 0.5 * (T_k + T_k.conj().T)
    V_k = 0.5 * (V_k + V_k.conj().T)
    H_k = T_k + V_k
    S_k = 0.5 * (S_k + S_k.conj().T)
    H_k = 0.5 * (H_k + H_k.conj().T)
    X_k, n_kept = _canonical_orthogonalizer_complex(
        S_k,
        linear_dep_threshold,
        normalize_diag_first=canonical_orth_normalize_diag_first,
    )
    return S_k, T_k, V_k, H_k, X_k, n_kept


# =====================================================================
# Unrestricted (UHF/UKS) spin helpers — M4 slice 1
# =====================================================================
# These three were nested closures duplicated in `pbc_bipole_uhf` and
# `pbc_bipole_uks`. They are the M4 scaffold's "alpha/beta occupation
# splitting" and "spin-density construction" bullets
# (handovers/HANDOVER_BIPOLE_VARIANT_UNIFICATION.md).
#
# The two copies differed only in type annotations, docstring wording and
# line wrapping; the single code difference was a dead `nbf = ...`
# assignment in the UKS copy's zero-smearing branch, whose loop uses
# `eps.shape[0]` instead. Richest form adopted, per the M3a/M3b
# precedent. Closure state is passed explicitly — 2-4 driver locals each,
# which is what makes this extraction cheap where M3's was not.


def unrestricted_split_k_density_list(
    density: LatticeMatrixSet,
    *,
    n_k: int,
    k_points,
    bvk_mesh,
) -> List[np.ndarray]:
    """Per-k (single-spin) density matrices for the split paths.

    Home block at Γ; exact BvK-torus fold at multi-k (see
    :func:`bvk_torus_density_matrices`) -- exact for orbital, SAD-local,
    damped, ODA-mixed, and warm-start spin densities alike.
    """
    if n_k == 1:
        return [home_cell_block(density).astype(complex)]
    assert bvk_mesh is not None
    return bvk_torus_density_matrices(density, k_points, bvk_mesh)


def unrestricted_spin_density(
    C_per_k_local: Sequence[np.ndarray],
    n_occ_each: int,
    *,
    n_k: int,
    kmesh,
    cells_density,
    exchange_split_active: bool,
) -> LatticeMatrixSet:
    """Real-space single-spin density from per-k MO coefficients.

    Integer occupations (1.0 per occupied band, single-spin convention);
    the cross-cell blocks are zeroed unless an exchange split is active,
    matching the Γ-locality invariant the drivers rely on.
    """
    nbf = C_per_k_local[0].shape[1]
    occ_per_k = []
    for _ in range(n_k):
        occ = np.zeros(nbf, dtype=float)
        occ[:n_occ_each] = 1.0
        occ_per_k.append(occ)
    result = real_space_density_from_kpoints_fractional(
        C_per_k_local,
        occ_per_k,
        kmesh,
        cells_density,
    )
    if not exchange_split_active:
        _zero_cross_cell_density(result, nbf, n_k)
    return result


def unrestricted_occupations_per_spin(
    eps_spin_per_k: Sequence[np.ndarray],
    n_spin: int,
    *,
    smearing_T: float,
    weights,
    system: Optional[PeriodicSystem] = None,
    kmesh: Optional[BlochKMesh] = None,
    bz_integration: Optional[str] = None,
):
    """Per-spin Aufbau, Fermi-Dirac, or Gilat occupations.

    Returns ``(occ_per_k, mu, entropy_spin)`` where occupations are in
    ``[0, 1]`` per band (single-spin), ``mu`` is the spin chemical
    potential in Hartree, and ``entropy_spin`` is ``S/k_B`` for that spin.
    ``bz_integration='gilat'`` applies the T=0 microcell integral separately
    to this fixed spin population and returns zero entropy. Otherwise
    ``smearing_T <= 0`` (or an empty spin channel) returns exact integer
    Aufbau occupations.
    """
    from .smearing.fermi_dirac import fermi_dirac_occupations_per_k as _fd_per_k

    if bz_integration not in (None, "smearing", "gilat"):
        raise ValueError(
            "unrestricted_occupations_per_spin: bz_integration must be None, "
            f"'smearing', or 'gilat'; got {bz_integration!r}"
        )
    use_gilat = bz_integration == "gilat"
    if use_gilat and smearing_T > 0.0:
        raise ValueError(
            "unrestricted_occupations_per_spin: bz_integration='gilat' is a "
            "T=0 integrator; do not combine it with smearing_temperature > 0"
        )
    if n_spin == 0:
        occ = []
        for eps in eps_spin_per_k:
            o = np.zeros(eps.shape[0], dtype=float)
            occ.append(o)
        return occ, 0.0, 0.0
    if use_gilat:
        if system is None or kmesh is None:
            raise ValueError(
                "unrestricted_occupations_per_spin: system and kmesh are "
                "required for bz_integration='gilat'"
            )
        # Gilat & Raubenheimer, Phys. Rev. 144, 390 (1966), Sec. II,
        # Eqs. (9)-(21): linearly integrate each band over its reciprocal-space
        # microcell. A single UHF/UKS spin channel has state degeneracy one.
        from .bz_integration import gilat_occupations_for_kmesh

        occ, e_fermi = gilat_occupations_for_kmesh(
            system,
            kmesh,
            eps_spin_per_k,
            float(n_spin),
            spin_degeneracy=1.0,
        )
        return occ, float(e_fermi), 0.0
    if smearing_T <= 0.0:
        occ = []
        for eps in eps_spin_per_k:
            o = np.zeros(eps.shape[0], dtype=float)
            o[:n_spin] = 1.0
            occ.append(o)
        return occ, 0.0, 0.0
    # The shared closed-shell helper returns occupations in [0, 2];
    # divide by 2 for the single-spin convention.
    occ_double, mu, entropy_double = _fd_per_k(
        eps_spin_per_k,
        weights,
        float(2 * n_spin),
        smearing_T,
    )
    occ = [np.asarray(o, dtype=float) * 0.5 for o in occ_double]
    return occ, mu, entropy_double * 0.5
