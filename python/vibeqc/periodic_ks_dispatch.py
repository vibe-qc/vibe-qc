"""KS dispatcher: ``CoulombMethod`` routing for periodic Kohn-Sham
DFT. Mirrors :mod:`vibeqc.periodic_rhf_dispatch` for the closed-shell
DFT side.

  CoulombMethod.DIRECT_TRUNCATED ->
      :func:`vibeqc.run_rks_periodic` / :func:`run_rks_periodic_gamma`
      (existing C++ DIRECT_TRUNCATED drivers; Γ-only is a
      conventional ``[1,1,1]``-mesh call).

  CoulombMethod.EWALD_3D ->
      Γ-only: :func:`vibeqc.run_rks_periodic_gamma_ewald3d`
      (Phase 15c-1 -- composed Ewald J + libxc V_xc on the periodic
      Becke grid). Multi-k:
      :func:`vibeqc.run_rks_periodic_multi_k_ewald3d` (Phase 15c-2 --
      same building blocks at every k, k-weighted DIIS, Fermi-Dirac
      smearing, optional level shift). UKS multi-k Ewald lands in
      Phase 15c-3.

  CoulombMethod.SLAB_EWALD_2D ->
      Gamma-only RKS slab Ewald via
      :func:`vibeqc.run_rks_periodic_gamma_ewald2d`; dense meshes route
      through the multi-k slab Ewald path.

  CoulombMethod.NEUTRALIZED_1D ->
      not implemented; raises ``NotImplementedError``.

This dispatcher exists today for the same reason the RHF dispatcher
does: tutorial / example code can use a single uniform entry point
without knowing whether the backend is the C++ direct-truncation
driver or the Python Ewald driver. User code stays the same when
the multi-k KS Ewald backend lands.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

from ._vibeqc_core import (
    BasisSet,
    BlochKMesh,
    CoulombMethod,
    PeriodicKSOptions,
    PeriodicSystem,
    monkhorst_pack,
    run_rks_periodic,
)
from .periodic_rhf_dispatch import _version_banner_line
from .periodic_rks_ewald import (
    run_rks_periodic_gamma_ewald2d,
    run_rks_periodic_gamma_ewald3d,
)
from .periodic_rks_multi_k_ewald import run_rks_periodic_multi_k_ewald3d
from .progress import ProgressLogger, resolve_progress

__all__ = [
    "run_rks_periodic_scf",
    "run_rks_periodic_gamma_scf",
]


def _smearing_temperature(options: PeriodicKSOptions, driver: str) -> float:
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


def _copy_ks_options(options: PeriodicKSOptions) -> PeriodicKSOptions:
    """Return a fresh ``PeriodicKSOptions`` field-mirroring the input.

    We construct a new object rather than passing the user's options
    through unchanged so that internal mutations (e.g. forcing the
    coulomb_method on a routing branch) don't affect the caller's
    state.
    """
    out = PeriodicKSOptions()
    out.functional = options.functional
    out.grid = options.grid
    out.max_iter = options.max_iter
    out.conv_tol_energy = options.conv_tol_energy
    out.conv_tol_grad = options.conv_tol_grad
    out.damping = options.damping
    if hasattr(out, "dynamic_damping"):
        out.dynamic_damping = bool(getattr(options, "dynamic_damping", False))
        out.dynamic_damping_min = float(getattr(options, "dynamic_damping_min", 0.0))
        out.dynamic_damping_max = float(getattr(options, "dynamic_damping_max", 0.95))
    out.fock_mixing = float(getattr(options, "fock_mixing", 0.0))
    out.use_diis = options.use_diis
    out.diis_start_iter = options.diis_start_iter
    out.diis_subspace_size = options.diis_subspace_size
    if hasattr(out, "scf_accelerator") and hasattr(options, "scf_accelerator"):
        out.scf_accelerator = options.scf_accelerator
    if hasattr(out, "ediis_diis_switch_threshold"):
        out.ediis_diis_switch_threshold = float(
            getattr(options, "ediis_diis_switch_threshold", 1e-1)
        )
    out.lattice_opts = options.lattice_opts
    # Forward-compatible field copy: any future field added to
    # PeriodicKSOptions in C++ that isn't here yet falls through with
    # its default -- explicit getattr lets older Python pin to newer
    # C++ builds and vice versa.
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
    out.use_periodic_becke = bool(getattr(options, "use_periodic_becke", False))
    out.becke_image_radius_bohr = float(
        getattr(options, "becke_image_radius_bohr", 10.0)
    )
    from .guess import copy_initial_guess_options
    return copy_initial_guess_options(options, out)


def run_rks_periodic_scf(
    system: PeriodicSystem,
    basis: BasisSet,
    kmesh,  # BlochKMesh | KPoints -- boundary helper accepts both
    options: Optional[PeriodicKSOptions] = None,
    *,
    initial_density_k: Optional[Sequence] = None,
    omega: float = 0.0,
    grid_shape: Optional[Union[Tuple[int, int, int], int]] = None,
    origin: Optional[Sequence[float]] = None,
    spacing_bohr: float = 0.3,
    linear_dep_threshold: float = 1e-7,
    progress: Union[bool, ProgressLogger, None] = None,
    bz_integration: Optional[str] = None,
    density_mixer: Optional[str] = None,
    density_mixer_depth: int = 8,
    density_mixer_beta: float = 0.5,
    density_mixer_kerker: bool = False,
    kerker_k0: float = 1.5,
    kerker_strength: float = 1.0,
    kerker_cutoff_ha: float = 120.0,
):
    """Multi-k closed-shell periodic Kohn-Sham SCF with Coulomb-method
    dispatch.

    Routes on ``options.lattice_opts.coulomb_method``:

      - ``DIRECT_TRUNCATED`` -> :func:`vibeqc.run_rks_periodic`.
      - ``EWALD_3D`` -> :func:`vibeqc.run_rks_periodic_gamma_ewald3d`
        for a [1,1,1] mesh, otherwise
        :func:`vibeqc.run_rks_periodic_multi_k_ewald3d`.

    The ``omega``, ``grid_shape``, ``origin``, ``spacing_bohr``,
    ``linear_dep_threshold`` keywords are forwarded to the EWALD_3D
    backends; the DIRECT_TRUNCATED path silently ignores them.

    The ``kmesh`` argument may be either a native :class:`BlochKMesh`
    (legacy) or the public :class:`vibeqc.KPoints` builder result
    (K7, v0.6) -- normalized at the boundary via
    :func:`vibeqc.as_bloch_kmesh`.

    ``density_mixer`` (``None`` / ``"anderson"`` / ``"broyden"``) selects a
    real-space density-matrix mixer (D4 metal-mixing program), and
    ``density_mixer_kerker`` adds the Kerker preconditioner (D4b); both are
    honoured only on the multi-k EWALD_3D route and fail closed elsewhere. See
    :func:`run_rks_periodic_multi_k_ewald3d` for ``density_mixer_depth`` /
    ``density_mixer_beta`` / ``kerker_k0`` / ``kerker_strength`` /
    ``kerker_cutoff_ha``.
    """
    if options is None:
        options = PeriodicKSOptions()
    method = options.lattice_opts.coulomb_method
    from .guess import select_periodic_driver_guess
    selection = select_periodic_driver_guess(
        system, options, route=("native" if method == CoulombMethod.DIRECT_TRUNCATED else "ewald"),
        method="RKS", driver="run_rks_periodic_scf", multi_k=True,
        restart_supplied=initial_density_k is not None,
    )
    plog = resolve_progress(progress)

    # K7 boundary: KPoints -> BlochKMesh, BlochKMesh passes through.
    from .kpoints import as_bloch_kmesh

    bm = as_bloch_kmesh(kmesh)
    plog.banner(_version_banner_line())
    plog.banner(
        f"run_rks_periodic_scf  functional={options.functional!r}  "
        f"coulomb={method.name if hasattr(method, 'name') else method}"
    )
    smearing_T = _smearing_temperature(options, "run_rks_periodic_scf")

    # Density-space mixers are wired only on the multi-k EWALD_3D driver. Rather
    # than silently ignore a requested mixer on a route that can't honour it
    # (a Sec.7-class lie), fail closed with the supported route named.
    _multi_k_ewald = (
        method == CoulombMethod.EWALD_3D
        and not (len(bm.kpoints) == 1 and np.allclose(bm.kpoints[0], 0.0)
                 and smearing_T == 0.0 and initial_density_k is None)
    )
    if (density_mixer is not None or density_mixer_kerker) and not _multi_k_ewald:
        requested = density_mixer if density_mixer is not None else "kerker"
        raise NotImplementedError(
            f"run_rks_periodic_scf: density_mixer/kerker ({requested!r}) is wired "
            "only on the multi-k EWALD_3D driver "
            "(coulomb_method=EWALD_3D with a >1 k-point or smeared mesh). The "
            f"current route is {method.name if hasattr(method, 'name') else method}"
            f"{' (Γ-only EWALD_3D)' if method == CoulombMethod.EWALD_3D else ''}; "
            "use a multi-k Ewald run, or DIIS/EDIIS/ADIIS via scf_accelerator."
        )

    if method == CoulombMethod.DIRECT_TRUNCATED:
        if smearing_T > 0.0:
            _raise_direct_smearing_unavailable("run_rks_periodic_scf")
        plog.info(
            "backend: C++ run_rks_periodic (DIRECT_TRUNCATED) "
            "-- per-iteration progress not available"
        )
        scf_opts = _copy_ks_options(options)
        if initial_density_k is not None:
            from ._vibeqc_core import InitialGuess
            scf_opts.initial_guess = InitialGuess.READ
            scf_opts.read_density_k = list(initial_density_k)
        result = run_rks_periodic(system, basis, bm, scf_opts)
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
        # Γ-only KS Ewald (15c-1) for [1,1,1] meshes; full multi-k
        # driver (15c-2) for everything else. Both share the same
        # building blocks (composed Ewald J + libxc V_xc + optional
        # hybrid HF exchange); the multi-k path adds Bloch-summed
        # per-k Fock matrices, k-weighted DIIS and Fermi-Dirac
        # smearing.
        n_kpts = len(bm.kpoints)
        if n_kpts == 1 and np.allclose(bm.kpoints[0], 0.0) and smearing_T == 0.0 and initial_density_k is None:
            return run_rks_periodic_gamma_ewald3d(
                system,
                basis,
                _copy_ks_options(options),
                omega=omega,
                grid_shape=grid_shape,
                origin=origin,
                spacing_bohr=spacing_bohr,
                linear_dep_threshold=linear_dep_threshold,
                progress=plog,
            )
        return run_rks_periodic_multi_k_ewald3d(
            system,
            basis,
            bm,
            _copy_ks_options(options),
            omega=omega,
            grid_shape=grid_shape,
            origin=origin,
            spacing_bohr=spacing_bohr,
            linear_dep_threshold=linear_dep_threshold,
            progress=plog,
            bz_integration=bz_integration,
            initial_density_k=initial_density_k,
            density_mixer=density_mixer,
            density_mixer_depth=density_mixer_depth,
            density_mixer_beta=density_mixer_beta,
            density_mixer_kerker=density_mixer_kerker,
            kerker_k0=kerker_k0,
            kerker_strength=kerker_strength,
            kerker_cutoff_ha=kerker_cutoff_ha,
        )

    if method == CoulombMethod.SLAB_EWALD_2D:
        n_kpts = len(bm.kpoints)
        if n_kpts == 1 and np.allclose(bm.kpoints[0], 0.0) and smearing_T == 0.0 and initial_density_k is None:
            return run_rks_periodic_gamma_ewald2d(
                system,
                basis,
                _copy_ks_options(options),
                omega=omega,
                grid_shape=grid_shape,
                origin=origin,
                spacing_bohr=spacing_bohr,
                linear_dep_threshold=linear_dep_threshold,
                progress=plog,
            )
        return run_rks_periodic_multi_k_ewald3d(
            system,
            basis,
            bm,
            _copy_ks_options(options),
            omega=omega,
            grid_shape=grid_shape,
            origin=origin,
            spacing_bohr=spacing_bohr,
            linear_dep_threshold=linear_dep_threshold,
            progress=plog,
            bz_integration=bz_integration,
            initial_density_k=initial_density_k,
            density_mixer=density_mixer,
            density_mixer_depth=density_mixer_depth,
            density_mixer_beta=density_mixer_beta,
            density_mixer_kerker=density_mixer_kerker,
            kerker_k0=kerker_k0,
            kerker_strength=kerker_strength,
            kerker_cutoff_ha=kerker_cutoff_ha,
        )

    raise NotImplementedError(
        f"run_rks_periodic_scf: coulomb_method {method!r} is not wired "
        "into the SCF dispatch. Supported: DIRECT_TRUNCATED, EWALD_3D, "
        "SLAB_EWALD_2D. NEUTRALIZED_1D wire Ewald is not yet "
        "implemented."
    )


def run_rks_periodic_gamma_scf(
    system: PeriodicSystem,
    basis: BasisSet,
    options: Optional[PeriodicKSOptions] = None,
    *,
    omega: float = 0.0,
    grid_shape: Optional[Union[Tuple[int, int, int], int]] = None,
    origin: Optional[Sequence[float]] = None,
    spacing_bohr: float = 0.3,
    linear_dep_threshold: float = 1e-7,
    progress: Union[bool, ProgressLogger, None] = None,
):
    """Γ-only periodic closed-shell Kohn-Sham SCF with Coulomb-method
    dispatch.

    A thin wrapper that dispatches the Γ-only KS calculation as the
    multi-k driver at a ``[1,1,1]`` mesh -- vibe-qc's C++ stack doesn't
    ship a separate Γ-only KS entry point (unlike the HF side, where
    ``run_rhf_periodic_gamma`` is its own driver). For the user the
    interface is identical to :func:`run_rhf_periodic_gamma_scf`.
    """
    if options is None:
        options = PeriodicKSOptions()
    method = options.lattice_opts.coulomb_method
    from .guess import select_periodic_driver_guess
    select_periodic_driver_guess(
        system, options, route=("native" if method == CoulombMethod.DIRECT_TRUNCATED else "ewald"),
        method="RKS", driver="run_rks_periodic_gamma_scf",
    )
    plog = resolve_progress(progress)
    plog.banner(_version_banner_line())
    plog.banner(
        f"run_rks_periodic_gamma_scf  functional={options.functional!r}  "
        f"coulomb={method.name if hasattr(method, 'name') else method}"
    )
    smearing_T = _smearing_temperature(options, "run_rks_periodic_gamma_scf")

    if method == CoulombMethod.DIRECT_TRUNCATED:
        if smearing_T > 0.0:
            _raise_direct_smearing_unavailable("run_rks_periodic_gamma_scf")
        kmesh = monkhorst_pack(system, [1, 1, 1])
        plog.info(
            "backend: C++ run_rks_periodic (DIRECT_TRUNCATED) "
            "-- per-iteration progress not available"
        )
        result = run_rks_periodic(
            system,
            basis,
            kmesh,
            _copy_ks_options(options),
        )
        plog.converged(
            n_iter=int(getattr(result, "n_iter", 0)),
            energy=float(getattr(result, "energy", 0.0)),
            converged=bool(getattr(result, "converged", False)),
        )
        return result

    if method == CoulombMethod.EWALD_3D:
        if smearing_T > 0.0:
            plog.info("backend: multi-k Gamma EWALD_3D (honours Fermi-Dirac smearing)")
            kmesh = monkhorst_pack(system, [1, 1, 1])
            return run_rks_periodic_multi_k_ewald3d(
                system,
                basis,
                kmesh,
                _copy_ks_options(options),
                omega=omega,
                grid_shape=grid_shape,
                origin=origin,
                spacing_bohr=spacing_bohr,
                linear_dep_threshold=linear_dep_threshold,
                progress=plog,
            )
        return run_rks_periodic_gamma_ewald3d(
            system,
            basis,
            _copy_ks_options(options),
            omega=omega,
            grid_shape=grid_shape,
            origin=origin,
            spacing_bohr=spacing_bohr,
            linear_dep_threshold=linear_dep_threshold,
            progress=plog,
        )

    if method == CoulombMethod.SLAB_EWALD_2D:
        return run_rks_periodic_gamma_ewald2d(
            system,
            basis,
            _copy_ks_options(options),
            omega=omega,
            grid_shape=grid_shape,
            origin=origin,
            spacing_bohr=spacing_bohr,
            linear_dep_threshold=linear_dep_threshold,
            progress=plog,
        )

    raise NotImplementedError(
        f"run_rks_periodic_gamma_scf: coulomb_method {method!r} is not "
        "wired into the SCF dispatch. Supported: DIRECT_TRUNCATED, EWALD_3D, "
        "SLAB_EWALD_2D. NEUTRALIZED_1D wire Ewald is not yet implemented."
    )
