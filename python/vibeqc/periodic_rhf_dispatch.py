"""Phase 12e-c-4: end-to-end ``CoulombMethod`` dispatch for periodic RHF.

Single public entry point that routes an RHF calculation to the right
SCF backend based on ``options.lattice_opts.coulomb_method``:

  CoulombMethod.DIRECT_TRUNCATED ->
      existing C++ driver (vibeqc.run_rhf_periodic / run_rhf_periodic_gamma).
      Uses direct lattice-sum ERIs truncated at ``cutoff_bohr``.
      Extra Ewald keyword arguments (``omega``, ``grid_shape``,
      ``origin``, ``spacing_bohr``) are silently ignored.

  CoulombMethod.EWALD_3D ->
      the Ewald-composed Python driver
      (vibeqc.run_rhf_periodic_multi_k_ewald3d /
      run_rhf_periodic_gamma_ewald3d) from Phase 12e-c-4b / 4c.
      Honours ``omega``, ``grid_shape``, ``origin``, ``spacing_bohr``.

  CoulombMethod.SLAB_EWALD_2D ->
      Gamma-only RHF slab Ewald via run_rhf_periodic_gamma_ewald2d;
      dense meshes route through the multi-k slab Ewald path.

  CoulombMethod.NEUTRALIZED_1D ->
      not implemented yet; raises ``NotImplementedError``.

The dispatcher returns the native result type of whichever backend
ran:

  - DIRECT -> :class:`PeriodicRHFResult` (C++-side dataclass).
  - EWALD_3D multi-k -> :class:`PeriodicRHFMultiKEwaldResult`.
  - EWALD_3D Γ-only -> :class:`PeriodicRHFEwaldResult`.

Callers that want a uniform return type should post-process -- the
native types carry slightly different bookkeeping (Ewald-side tracks
``omega``, ``grid_shape``; direct-side tracks nothing Coulomb-specific)
and collapsing them to a common shape would drop useful provenance
information. Shared fields (``energy``, ``e_electronic``,
``e_nuclear``, ``n_iter``, ``converged``, ``scf_trace``) are present
on every result type.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from ._vibeqc_core import (
    BasisSet,
    BlochKMesh,
    CoulombMethod,
    PeriodicRHFOptions,
    PeriodicSCFOptions,
    PeriodicSystem,
    monkhorst_pack,
    run_rhf_periodic,
    run_rhf_periodic_gamma,
)
from .banner import VIBEQC_VERSION, codename_for_version
from .periodic_rhf_ewald import (
    run_rhf_periodic_gamma_ewald2d,
    run_rhf_periodic_gamma_ewald3d,
)
from .periodic_rhf_multi_k_ewald import run_rhf_periodic_multi_k_ewald3d
from .progress import ProgressLogger, resolve_progress


def _version_banner_line() -> str:
    """One-line ``vibe-qc <version> [codename] (PID <n>)`` banner for SCF
    entries.

    Pulls the live version from ``vibeqc.banner.VIBEQC_VERSION`` and the
    codename via the patch-inherits-minor lookup. Truncated to a single
    line so it fits next to the SCF header without extra vertical
    space; the multi-line ``print_banner()`` is still available for
    callers that want the full splash.

    Includes the process PID so users tailing the live ``.out`` can
    immediately attach ``py-spy --pid <n>`` / ``htop -p <n>`` /
    ``/proc/<n>/status`` without waiting for the run to finish (the
    same PID lands in the post-mortem ``.perf`` header and the
    ``.system`` ``[run]`` block).
    """
    cn = codename_for_version(VIBEQC_VERSION)
    suffix = f' "{cn}"' if cn else ""
    return f"vibe-qc {VIBEQC_VERSION}{suffix} (PID {os.getpid()})"


__all__ = [
    "run_rhf_periodic_scf",
    "run_rhf_periodic_gamma_scf",
]


def _smearing_temperature(options, driver: str) -> float:
    T = float(getattr(options, "smearing_temperature", 0.0))
    if T < 0.0:
        raise ValueError(f"{driver}: smearing_temperature must be >= 0")
    return T


def _raise_direct_smearing_unavailable(driver: str) -> None:
    raise NotImplementedError(
        f"{driver}: Fermi-Dirac smearing requires the native multi-k "
        "EWALD_3D driver today. DIRECT_TRUNCATED would ignore "
        "smearing_temperature, so vibe-qc refuses this route instead "
        "of silently running an integer-Aufbau calculation."
    )


def _copy_options_to_rhf(options) -> PeriodicRHFOptions:
    """Translate a ``PeriodicSCFOptions`` or ``PeriodicRHFOptions`` into
    a fresh ``PeriodicRHFOptions`` (the shape the Ewald driver expects).

    The two classes are structurally identical on the public field
    surface; this helper copies field-by-field and preserves the
    ``lattice_opts`` reference (its ``coulomb_method`` stays whatever
    the caller set).
    """
    out = PeriodicRHFOptions()
    out.conv_tol_energy = options.conv_tol_energy
    out.conv_tol_grad = options.conv_tol_grad
    out.damping = options.damping
    if hasattr(out, "dynamic_damping"):
        out.dynamic_damping = bool(getattr(options, "dynamic_damping", False))
        out.dynamic_damping_min = float(getattr(options, "dynamic_damping_min", 0.0))
        out.dynamic_damping_max = float(getattr(options, "dynamic_damping_max", 0.95))
    out.fock_mixing = float(getattr(options, "fock_mixing", 0.0))
    out.diis_start_iter = options.diis_start_iter
    out.diis_subspace_size = options.diis_subspace_size
    if hasattr(out, "scf_accelerator") and hasattr(options, "scf_accelerator"):
        out.scf_accelerator = options.scf_accelerator
    if hasattr(out, "ediis_diis_switch_threshold"):
        out.ediis_diis_switch_threshold = float(
            getattr(options, "ediis_diis_switch_threshold", 1e-1)
        )
    out.lattice_opts = options.lattice_opts
    out.max_iter = options.max_iter
    out.use_diis = options.use_diis
    # C1a Saunders-Hillier level shift; defaults to 0.0 if the caller
    # doesn't set it. ``getattr`` keeps us forward-compatible with older
    # PeriodicSCFOptions / PeriodicRHFOptions builds that don't yet
    # expose the field.
    out.level_shift = float(getattr(options, "level_shift", 0.0))
    if hasattr(out, "level_shift_warmup_cycles"):
        out.level_shift_warmup_cycles = int(
            getattr(options, "level_shift_warmup_cycles", -1)
        )
    out.smearing_temperature = float(getattr(options, "smearing_temperature", 0.0))
    out.quadratic_fallback_iter = int(getattr(options, "quadratic_fallback_iter", 0))
    out.quadratic_fallback_shift = float(
        getattr(options, "quadratic_fallback_shift", 0.1)
    )
    out.quadratic_fallback_max_step = float(
        getattr(options, "quadratic_fallback_max_step", 0.1)
    )
    from .guess import copy_initial_guess_options
    return copy_initial_guess_options(options, out)


def _copy_options_to_scf(options) -> PeriodicSCFOptions:
    """Translate a ``PeriodicRHFOptions`` or ``PeriodicSCFOptions`` into
    a fresh ``PeriodicSCFOptions`` (the shape the C++ driver expects).
    """
    out = PeriodicSCFOptions()
    out.conv_tol_energy = options.conv_tol_energy
    out.conv_tol_grad = options.conv_tol_grad
    out.damping = options.damping
    if hasattr(out, "dynamic_damping"):
        out.dynamic_damping = bool(getattr(options, "dynamic_damping", False))
        out.dynamic_damping_min = float(getattr(options, "dynamic_damping_min", 0.0))
        out.dynamic_damping_max = float(getattr(options, "dynamic_damping_max", 0.95))
    out.fock_mixing = float(getattr(options, "fock_mixing", 0.0))
    out.diis_start_iter = options.diis_start_iter
    out.diis_subspace_size = options.diis_subspace_size
    if hasattr(out, "scf_accelerator") and hasattr(options, "scf_accelerator"):
        out.scf_accelerator = options.scf_accelerator
    if hasattr(out, "ediis_diis_switch_threshold"):
        out.ediis_diis_switch_threshold = float(
            getattr(options, "ediis_diis_switch_threshold", 1e-1)
        )
    out.lattice_opts = options.lattice_opts
    out.max_iter = options.max_iter
    out.use_diis = options.use_diis
    # C1a Saunders-Hillier level shift; defaults to 0.0 if absent.
    out.level_shift = float(getattr(options, "level_shift", 0.0))
    if hasattr(out, "level_shift_warmup_cycles"):
        out.level_shift_warmup_cycles = int(
            getattr(options, "level_shift_warmup_cycles", -1)
        )
    out.smearing_temperature = float(getattr(options, "smearing_temperature", 0.0))
    out.quadratic_fallback_iter = int(getattr(options, "quadratic_fallback_iter", 0))
    out.quadratic_fallback_shift = float(
        getattr(options, "quadratic_fallback_shift", 0.1)
    )
    out.quadratic_fallback_max_step = float(
        getattr(options, "quadratic_fallback_max_step", 0.1)
    )
    from .guess import copy_initial_guess_options
    return copy_initial_guess_options(options, out)


def run_rhf_periodic_scf(
    system: PeriodicSystem,
    basis: BasisSet,
    kmesh,  # BlochKMesh | KPoints -- boundary helper accepts both
    options=None,
    *,
    initial_density_k: Optional[Sequence] = None,
    omega: float = 0.0,
    grid_shape: Optional[Union[Tuple[int, int, int], int]] = None,
    origin: Optional[Sequence[float]] = None,
    spacing_bohr: float = 0.3,
    linear_dep_threshold: float = 1e-7,
    progress: Union[bool, ProgressLogger, None] = None,
    bz_integration: Optional[str] = None,
):
    """Multi-k periodic closed-shell RHF with Coulomb-method dispatch.

    Inspects ``options.lattice_opts.coulomb_method`` and routes to:

      - :func:`vibeqc.run_rhf_periodic` for
        ``CoulombMethod.DIRECT_TRUNCATED``.
      - :func:`vibeqc.run_rhf_periodic_multi_k_ewald3d` for
        ``CoulombMethod.EWALD_3D``.

    Parameters
    ----------
    system, basis, kmesh
        Periodic system, AO basis, k-point sampling. ``kmesh`` may be
        either a native :class:`BlochKMesh` (legacy path) or the
        public :class:`vibeqc.KPoints` builder result (K7, v0.6) --
        normalized at the boundary via
        :func:`vibeqc.as_bloch_kmesh`.
    options
        :class:`PeriodicSCFOptions` or :class:`PeriodicRHFOptions`.
        If ``None``, a default ``PeriodicSCFOptions`` is used whose
        ``coulomb_method`` is ``DIRECT_TRUNCATED``.
    omega, grid_shape, origin, spacing_bohr, linear_dep_threshold
        Ewald-specific parameters. Forwarded to
        :func:`run_rhf_periodic_multi_k_ewald3d` when ``EWALD_3D`` is
        selected; ignored otherwise.

    Returns
    -------
    The native result of whichever backend ran -- see module docstring.

    Raises
    ------
    NotImplementedError
        If ``coulomb_method`` is ``NEUTRALIZED_1D`` or another backend that
        has not been wired into SCF dispatch yet.
    """
    if options is None:
        options = PeriodicSCFOptions()
    method = options.lattice_opts.coulomb_method
    from .guess import select_periodic_driver_guess
    selection = select_periodic_driver_guess(
        system, options, route=("native" if method == CoulombMethod.DIRECT_TRUNCATED else "ewald"),
        method="RHF", driver="run_rhf_periodic_scf", multi_k=True,
        restart_supplied=initial_density_k is not None,
    )
    plog = resolve_progress(progress)

    # K7 boundary: accept either a KPoints (preferred, new API) or a
    # raw BlochKMesh (legacy / direct-from-monkhorst_pack). The helper
    # passes BlochKMesh through unchanged; KPoints flavors are
    # materialised on demand.
    from .kpoints import as_bloch_kmesh

    bm = as_bloch_kmesh(kmesh)
    plog.banner(_version_banner_line())
    plog.banner(
        f"run_rhf_periodic_scf  coulomb={method.name if hasattr(method, 'name') else method}"
    )
    smearing_T = _smearing_temperature(options, "run_rhf_periodic_scf")

    if method == CoulombMethod.DIRECT_TRUNCATED:
        if smearing_T > 0.0:
            _raise_direct_smearing_unavailable("run_rhf_periodic_scf")
        # The DIRECT_TRUNCATED path runs in compiled C++ with no
        # Python-level progress callback. Live per-iteration logging
        # isn't available there -- we still emit the start banner and
        # post-hoc trace through the converged() summary.
        scf_opts = _copy_options_to_scf(options)
        plog.info(
            "backend: C++ run_rhf_periodic (DIRECT_TRUNCATED) "
            "-- per-iteration progress not available"
        )
        if initial_density_k is not None:
            from ._vibeqc_core import InitialGuess
            scf_opts.initial_guess = InitialGuess.READ
            scf_opts.read_density_k = list(initial_density_k)
        result = run_rhf_periodic(system, basis, bm, scf_opts)
        if initial_density_k is not None:
            from .guess import GuessSelection
            result.guess_selection = GuessSelection(selection.requested, InitialGuess.READ, InitialGuess.READ)
        plog.converged(
            n_iter=int(getattr(result, "n_iter", 0)),
            energy=float(getattr(result, "energy", 0.0)),
            converged=bool(getattr(result, "converged", False)),
        )
        return result

    if method == CoulombMethod.EWALD_3D:
        rhf_opts = _copy_options_to_rhf(options)
        return run_rhf_periodic_multi_k_ewald3d(
            system,
            basis,
            bm,
            rhf_opts,
            omega=omega,
            grid_shape=grid_shape,
            origin=origin,
            spacing_bohr=spacing_bohr,
            linear_dep_threshold=linear_dep_threshold,
            progress=plog,
            bz_integration=bz_integration,
            initial_density_k=initial_density_k,
        )

    if method == CoulombMethod.SLAB_EWALD_2D:
        rhf_opts = _copy_options_to_rhf(options)
        n_kpts = len(bm.kpoints)
        if n_kpts == 1 and np.allclose(bm.kpoints[0], 0.0) and smearing_T == 0.0 and initial_density_k is None:
            return run_rhf_periodic_gamma_ewald2d(
                system,
                basis,
                rhf_opts,
                omega=omega,
                grid_shape=grid_shape,
                origin=origin,
                spacing_bohr=spacing_bohr,
                linear_dep_threshold=linear_dep_threshold,
                progress=plog,
            )
        return run_rhf_periodic_multi_k_ewald3d(
            system,
            basis,
            bm,
            rhf_opts,
            omega=omega,
            grid_shape=grid_shape,
            origin=origin,
            spacing_bohr=spacing_bohr,
            linear_dep_threshold=linear_dep_threshold,
            progress=plog,
            bz_integration=bz_integration,
            initial_density_k=initial_density_k,
        )

    raise NotImplementedError(
        f"run_rhf_periodic_scf: coulomb_method {method!r} is not wired "
        "into the SCF dispatch. Supported: DIRECT_TRUNCATED, EWALD_3D, "
        "SLAB_EWALD_2D. NEUTRALIZED_1D wire Ewald is not yet implemented."
    )


def run_rhf_periodic_gamma_scf(
    system: PeriodicSystem,
    basis: BasisSet,
    options=None,
    *,
    omega: float = 0.0,
    grid_shape: Optional[Union[Tuple[int, int, int], int]] = None,
    origin: Optional[Sequence[float]] = None,
    spacing_bohr: float = 0.3,
    linear_dep_threshold: float = 1e-7,
    progress: Union[bool, ProgressLogger, None] = None,
):
    """Γ-only periodic closed-shell RHF with Coulomb-method dispatch.

    Mirror of :func:`run_rhf_periodic_scf` for a single-k (Γ-point)
    calculation. Routes to:

      - :func:`vibeqc.run_rhf_periodic_gamma` for
        ``CoulombMethod.DIRECT_TRUNCATED``.
      - :func:`vibeqc.run_rhf_periodic_gamma_ewald3d` for
        ``CoulombMethod.EWALD_3D``.
      - :func:`vibeqc.run_rhf_periodic_gamma_ewald2d` for
        ``CoulombMethod.SLAB_EWALD_2D``.
    """
    if options is None:
        options = PeriodicRHFOptions()
    method = options.lattice_opts.coulomb_method
    from .guess import select_periodic_driver_guess
    select_periodic_driver_guess(
        system, options, route=("native" if method == CoulombMethod.DIRECT_TRUNCATED else "ewald"),
        method="RHF", driver="run_rhf_periodic_gamma_scf",
    )
    plog = resolve_progress(progress)
    plog.banner(
        f"run_rhf_periodic_gamma_scf  "
        f"coulomb={method.name if hasattr(method, 'name') else method}"
    )
    smearing_T = _smearing_temperature(options, "run_rhf_periodic_gamma_scf")

    if method == CoulombMethod.DIRECT_TRUNCATED:
        if smearing_T > 0.0:
            _raise_direct_smearing_unavailable("run_rhf_periodic_gamma_scf")
        rhf_opts = _copy_options_to_rhf(options)
        plog.info(
            "backend: C++ run_rhf_periodic_gamma (DIRECT_TRUNCATED) "
            "-- per-iteration progress not available"
        )
        result = run_rhf_periodic_gamma(system, basis, rhf_opts)
        plog.converged(
            n_iter=int(getattr(result, "n_iter", 0)),
            energy=float(getattr(result, "energy", 0.0)),
            converged=bool(getattr(result, "converged", False)),
        )
        return result

    if method == CoulombMethod.EWALD_3D:
        rhf_opts = _copy_options_to_rhf(options)
        if smearing_T > 0.0:
            plog.info("backend: multi-k Gamma EWALD_3D (honours Fermi-Dirac smearing)")
            gamma_mesh = monkhorst_pack(system, [1, 1, 1])
            return run_rhf_periodic_multi_k_ewald3d(
                system,
                basis,
                gamma_mesh,
                rhf_opts,
                omega=omega,
                grid_shape=grid_shape,
                origin=origin,
                spacing_bohr=spacing_bohr,
                linear_dep_threshold=linear_dep_threshold,
                progress=plog,
            )
        return run_rhf_periodic_gamma_ewald3d(
            system,
            basis,
            rhf_opts,
            omega=omega,
            grid_shape=grid_shape,
            origin=origin,
            spacing_bohr=spacing_bohr,
            linear_dep_threshold=linear_dep_threshold,
            progress=plog,
        )

    if method == CoulombMethod.SLAB_EWALD_2D:
        rhf_opts = _copy_options_to_rhf(options)
        return run_rhf_periodic_gamma_ewald2d(
            system,
            basis,
            rhf_opts,
            omega=omega,
            grid_shape=grid_shape,
            origin=origin,
            spacing_bohr=spacing_bohr,
            linear_dep_threshold=linear_dep_threshold,
            progress=plog,
        )

    raise NotImplementedError(
        f"run_rhf_periodic_gamma_scf: coulomb_method {method!r} is not "
        "wired into the SCF dispatch. Supported: DIRECT_TRUNCATED, EWALD_3D. "
        "SLAB_EWALD_2D. NEUTRALIZED_1D wire Ewald is not yet implemented."
    )
