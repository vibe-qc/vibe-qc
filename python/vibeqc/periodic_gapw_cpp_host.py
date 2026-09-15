"""C++ SCF host for the GPW Γ-only periodic RHF route (v0.12 R3).

This module wires the Python :class:`vibeqc.periodic_gapw_j.GpwJBuilder`
into the C++ SCF driver ``run_rhf_scf_with_jk`` via the
:class:`vibeqc._vibeqc_core.JKBuilder` pybind11 trampoline added in
v0.12 R3. The result is that the GPW Hartree-J kernel -- implemented in
Python as an FFT-Poisson solve on the smooth grid -- can drive the
production C++ SCF loop with **DIIS, dynamic damping, level-shift,
Newton/SOSCF, and the standard convergence diagnostics** instead of
the simple Python loop in ``run_periodic_rhf_gpw``.

Architecture
------------

* :class:`PyGpwJKBuilder` -- Python subclass of the C++ ``JKBuilder``
  abstract base. ``build_J(D)`` delegates to a pre-built
  :class:`vibeqc.periodic_gapw_j.GpwJBuilder` (smooth-grid FFT
  Hartree); ``build_K(D)`` delegates to
  :func:`vibeqc._vibeqc_core.build_jk_gamma_molecular_limit` (Γ-only
  K via direct AO-image sum, same kernel the Python GPW driver uses).
* :func:`run_rhf_scf_gpw_cpp` -- public entry point. Builds Hcore, S,
  E_nn from the existing Γ-folded lattice helpers; constructs the
  ``PyGpwJKBuilder``; hands off to the C++
  ``run_rhf_scf_with_jk``; returns a
  :class:`vibeqc.periodic_gapw_j.GpwScfResult`.

The kernel + gauge are identical to ``run_periodic_rhf_gpw`` --
T / S / V_ne / J / K all built the same way -- so the converged total
energy matches to machine precision. The only difference is the SCF
*driver*: C++ EDIIS+DIIS replaces the simple Python DIIS, which
typically converges in fewer iterations on the same system.
"""

from __future__ import annotations

from .guess import periodic_guess_capabilities

import warnings
from typing import Optional

import numpy as np

from . import _vibeqc_core as _core
from .periodic_gapw_grid import GAPWExperimentalWarning, PlaneWaveGrid
from .periodic_gapw_j import (
    GpwJBuilder,
    GpwScfResult,
    _build_v_ne,
    _kinetic_lattice_gamma,
    _overlap_lattice_gamma,
    evaluate_gpw_energy,
)

__all__ = ["PyGpwJKBuilder", "run_rhf_scf_gpw_cpp"]


class PyGpwJKBuilder(_core.JKBuilder):
    """Python ``JKBuilder`` subclass that fuses the GPW Hartree-J and
    Γ-only molecular-limit K into a single object the C++ SCF driver
    can consume.

    Parameters
    ----------
    basis
        :class:`vibeqc.BasisSet` for the system.
    periodic_system
        :class:`vibeqc._vibeqc_core.PeriodicSystem`.
    grid
        :class:`PlaneWaveGrid` for the FFT-Poisson Hartree build.
    jk_method
        Reserved for future open-shell / range-separated routes;
        currently only ``"gpw"`` is supported.
    omega
        Range-separation parameter. Currently unused -- pure HF
        / hybrid only. Reserved for wB97-class routes.

    Notes
    -----
    The K builder uses
    :func:`vibeqc._vibeqc_core.build_jk_gamma_molecular_limit` with a
    fixed 25 bohr lattice cutoff, matching the convention used by
    :func:`vibeqc.periodic_gapw_j.run_periodic_rhf_gpw`. ``build_J``
    delegates to :class:`GpwJBuilder` so the FFT plan + AO-collocation
    tables are reused across SCF iterations.
    """

    def __init__(
        self,
        basis,
        periodic_system,
        grid: PlaneWaveGrid,
        *,
        jk_method: str = "gpw",
        omega: float = 0.0,
    ) -> None:
        super().__init__()
        if jk_method != "gpw":
            raise ValueError(
                f"PyGpwJKBuilder: only jk_method='gpw' is supported "
                f"at v0.12 R3 (got {jk_method!r})"
            )
        if float(omega) != 0.0:
            raise ValueError(
                f"PyGpwJKBuilder: range-separated hybrids (omega != 0) "
                f"not yet wired on the GPW route (got omega={omega})"
            )
        self._basis = basis
        self._system = periodic_system
        self._grid = grid
        self._gpw = GpwJBuilder(basis, grid)
        self._lat_opts = _core.LatticeSumOptions()
        self._lat_opts.cutoff_bohr = 25.0

    def build_J(self, D):
        """FFT-Poisson Hartree-J on the smooth plane-wave grid."""
        D_arr = np.asarray(D, dtype=float)
        return np.asarray(self._gpw.build_J(D_arr))

    def build_K(self, D):
        """Γ-only exchange via direct AO-image sum at the
        ``LatticeSumOptions.cutoff_bohr = 25`` truncation."""
        D_arr = np.asarray(D, dtype=float)
        jk = _core.build_jk_gamma_molecular_limit(
            self._basis, self._system, self._lat_opts, D_arr,
        )
        return np.asarray(jk.K)


def run_rhf_scf_gpw_cpp(
    system,
    basis,
    *,
    grid: Optional[PlaneWaveGrid] = None,
    cutoff_ha: float = 300.0,
    max_iter: int = 50,
    conv_tol_energy: float = 1e-9,
    conv_tol_grad: float = 1e-7,
    damping: float = 0.0,
    use_diis: bool = True,
    diis_subspace_size: int = 8,
    diis_start_iter: int = 2,
    level_shift: float = 0.0,
    dynamic_damping: bool = False,
    v_ne_convention: str = "ewald",
    smearing_alpha: Optional[float] = None,
    initial_density: Optional[np.ndarray] = None,
    initial_guess: object = "AUTO",
    quiet: bool = False,
) -> GpwScfResult:
    """GPW Γ-only RHF driven by the C++ SCF host.

    Functionally equivalent to
    :func:`vibeqc.periodic_gapw_j.run_periodic_rhf_gpw` (closed-shell
    HF, no functional, no DFT+U, no smearing) but uses the C++
    ``run_rhf_scf_with_jk`` driver so DIIS / dynamic damping /
    level-shift run in C++. Converges in fewer iterations than the
    Python SCF on the same system.

    Parameters
    ----------
    system, basis
        Periodic system + AO basis. Same convention as
        :func:`run_periodic_rhf_gpw`.
    grid, cutoff_ha
        Pre-built :class:`PlaneWaveGrid` or, if ``None``, build one
        from ``cutoff_ha`` (default 300 Ha).
    max_iter, conv_tol_energy, conv_tol_grad
        SCF convergence knobs forwarded to ``RHFOptions``.
    damping, dynamic_damping, use_diis, diis_subspace_size,
    diis_start_iter, level_shift
        Forwarded to ``RHFOptions`` -- see the C++ docstrings for
        semantics. EDIIS+DIIS is the default accelerator (matches
        the molecular RHF driver).
    v_ne_convention, smearing_alpha
        Same as :func:`run_periodic_rhf_gpw` -- ``"ewald"`` by default;
        ``"smeared_erfc"`` requires ``smearing_alpha``.
    initial_density
        Optional ``(n_basis, n_basis)`` initial AO density. ``None``
        constructs the selected guess. AUTO chooses SAD on this route.
    quiet
        Suppress the :class:`GAPWExperimentalWarning`.

    Returns
    -------
    result
        :class:`GpwScfResult` evaluated at the converged density.
        ``n_iter`` reflects the C++ SCF iteration count.
    """
    from .guess import initial_density_closed_shell, select_initial_guess

    selection = select_initial_guess(
        system.unit_cell_molecule(), initial_guess, is_periodic=True,
        supported=periodic_guess_capabilities('gpw-native', 'RHF', dim=getattr(system, "dim", 3), multi_k=False, transport='k'),
        restart_supplied=initial_density is not None,
        driver="run_rhf_scf_gpw_cpp",
    )
    if not quiet:
        warnings.warn(
            "run_rhf_scf_gpw_cpp: C++ SCF host for the GPW Γ-only "
            "route is experimental -- gauge + parity match "
            "run_periodic_rhf_gpw to machine precision but the "
            "driver surface is new in v0.12 R3.",
            category=GAPWExperimentalWarning,
            stacklevel=2,
        )

    if system.dim != 3:
        raise ValueError(
            f"run_rhf_scf_gpw_cpp: only dim == 3 is supported "
            f"(got dim={system.dim})"
        )

    n_elec = int(system.n_electrons())
    if n_elec % 2 != 0:
        raise ValueError(
            f"run_rhf_scf_gpw_cpp: cell has {n_elec} electrons (odd); "
            f"closed-shell RHF needs an even count."
        )

    if grid is None:
        from .periodic_gapw_grid import make_grid as _make_grid
        grid = _make_grid(
            np.asarray(system.lattice, dtype=float),
            cutoff_ha=cutoff_ha,
        )

    # One-electron integrals -- identical recipe to
    # run_periodic_rhf_gpw so the converged energy matches to machine
    # precision once both SCFs reach their respective fixed points.
    T = _kinetic_lattice_gamma(basis, system)
    V_ne = _build_v_ne(
        basis, system, v_ne_convention, smearing_alpha, grid,
    )
    S = _overlap_lattice_gamma(basis, system)
    Hcore = T + V_ne

    # Ewald nuclear repulsion (Γ).
    E_nn = float(
        _core.ewald_nuclear_repulsion(system, _core.EwaldOptions())
    )

    # JK builder via the trampoline.
    jk_builder = PyGpwJKBuilder(basis, system, grid)

    # C++ SCF options.
    opts = _core.RHFOptions()
    opts.initial_guess = selection.requested
    opts.max_iter = int(max_iter)
    opts.conv_tol_energy = float(conv_tol_energy)
    opts.conv_tol_grad = float(conv_tol_grad)
    opts.damping = float(damping)
    opts.dynamic_damping = bool(dynamic_damping)
    opts.use_diis = bool(use_diis)
    opts.diis_subspace_size = int(diis_subspace_size)
    opts.diis_start_iter = int(diis_start_iter)
    opts.level_shift = float(level_shift)
    # Disable Newton + SOSCF + TRAH on the GPW path -- the Python
    # ``build_J`` is too expensive per call to make second-order
    # methods competitive at v0.12 R3, and Newton's CG inner loop
    # would call build_J many times per outer iteration.
    opts.newton_threshold = 0.0
    opts.soscf_threshold = 0.0
    opts.trah_threshold = 0.0
    opts.quadratic_fallback_iter = 0
    # The shared native constructor receives this route's Gamma metric,
    # Hamiltonian and GPW JK operator, plus explicit selection provenance.
    if initial_density is None:
        init_D = np.zeros((0, 0))
        if selection.effective in (
            _core.InitialGuess.SAP, _core.InitialGuess.HUECKEL,
            _core.InitialGuess.MINAO,
        ):
            # These constructions need the periodic potential/projection,
            # not the molecular constructor inside the native SCF host.
            # The supplied density is the selected physical construction;
            # it is not a READ restart or an internal retry.
            init_D = initial_density_closed_shell(
                system.unit_cell_molecule(), basis, n_elec // 2,
                selection.effective, periodic_system=system, overlap=S,
            )
    else:
        init_D = np.asarray(initial_density)
        if init_D.shape != (basis.nbasis, basis.nbasis):
            raise ValueError(
                f"initial_density shape {init_D.shape} doesn't "
                f"match basis ({basis.nbasis} functions)"
            )
    if np.iscomplexobj(init_D):
        if np.max(np.abs(init_D.imag), initial=0.0) > 1e-10:
            raise ValueError("GPW Gamma native host requires a real initial density")
        init_D = init_D.real
    init_D = np.asarray(init_D, dtype=float)

    rhf = _core.run_rhf_scf_with_jk(
        basis,
        n_elec,
        S,
        Hcore,
        E_nn,
        jk_builder,
        opts,
        init_D,
        molecule=system.unit_cell_molecule(), guess_selection=selection,
    )

    # Wrap the C++ result as a GpwScfResult so downstream code
    # (writers, ASE Calculator, restart) sees the same shape as
    # the Python SCF.
    D_conv = np.asarray(rhf.density, dtype=float)
    breakdown = evaluate_gpw_energy(
        system, basis, D_conv,
        grid=grid,
        v_ne_convention=v_ne_convention,
        smearing_alpha=smearing_alpha,
        quiet=True,
    )

    # Re-pack SCF trace to the dict shape the Python GPW path uses
    # -- keep keys consistent across the two drivers so writers /
    # ASE Calculator can index either uniformly.
    scf_trace = tuple(
        {
            "iter": int(it.iter),
            "energy": float(it.energy),
            "delta_e": float(it.delta_e),
            "grad_norm": float(it.grad_norm),
            "e_xc": 0.0,
        }
        for it in rhf.scf_trace
    )

    n_occ = n_elec // 2
    occupations = np.zeros(basis.nbasis, dtype=float)
    occupations[:n_occ] = 2.0

    return GpwScfResult(
        restart_basis=basis, restart_lattice=np.asarray(system.lattice).copy(),
        guess_selection=rhf.guess_selection,
        energy=float(rhf.energy),
        breakdown=breakdown,
        density=D_conv,
        mo_coeffs=np.asarray(rhf.mo_coeffs, dtype=float),
        mo_energies=np.asarray(rhf.mo_energies, dtype=float),
        converged=bool(rhf.converged),
        n_iter=int(rhf.n_iter),
        grid=grid,
        scf_trace=scf_trace,
        e_dispersion=0.0,
        e_dft_plus_u=0.0,
        smearing_temperature=0.0,
        smearing_entropy=0.0,
        fermi_level=0.0,
        occupations=tuple(occupations.tolist()),
        fock=np.asarray(rhf.fock, dtype=float),
        overlap=S,
    )
