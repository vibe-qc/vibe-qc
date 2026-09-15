"""Periodic PM6 model -- Gamma-point NDDO for 1D/2D/3D crystals.

Provides a Python wrapper around the C++ ``run_pm6_gamma`` driver
with finite-difference gradients and stress tensor.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from vibeqc._vibeqc_core import PeriodicSystem
from vibeqc._vibeqc_core.semiempirical import nddo as _nddo
from vibeqc.semiempirical.routes import plan_periodic_semiempirical_route


def _get_params(system: PeriodicSystem, params=None):
    """Load PM6 params, auto-selecting MOPAC if elements beyond H,C,N,O,F."""
    if params is None:
        from vibeqc.semiempirical.methods.pm6_params import load_pm6_params_auto

        zs = [a.Z for a in system.unit_cell]
        params = load_pm6_params_auto(zs)
    if params.method_name() != "pm6":
        raise ValueError(
            "periodic PM6 requires PM6 parameters, not "
            f"{params.method_name()}"
        )
    return params


def run_pm6_gamma(
    system: PeriodicSystem,
    params=None,
    *,
    cutoff_bohr: float = 15.0,
    max_iter: int = 100,
    conv_tol: float = 1e-7,
):
    """Run a periodic PM6 SCF calculation at the Gamma point.

    Parameters
    ----------
    system : PeriodicSystem
        Periodic system (unit cell atoms, lattice vectors, dim).
    params : PM6ParameterSet
        PM6 parameter set (from ``load_pm6_params()`` or
        ``load_pm6_mopac_params()``).
    cutoff_bohr : float
        Real-space lattice sum cutoff (bohr).
    max_iter : int
        Maximum SCF iterations.
    conv_tol : float
        SCF convergence tolerance for both max |ΔD| and max |[F, D]|.

    Returns
    -------
    PeriodicPM6Result
        C++ result struct with energy, density, MO coefficients, etc.
    """
    plan_periodic_semiempirical_route("pm6", system)
    params = _get_params(system, params)

    opts = _nddo.PeriodicPM6Options()
    opts.cutoff_bohr = cutoff_bohr
    opts.max_iter = max_iter
    opts.conv_tol = conv_tol

    return _nddo.run_pm6_gamma(system, params, opts)


def compute_pm6_gamma_gradient_fd(
    system: PeriodicSystem,
    params=None,
    *,
    h: float = 0.001,
    cutoff_bohr: float = 15.0,
    max_iter: int = 100,
    conv_tol: float = 1e-7,
) -> np.ndarray:
    """Finite-difference gradient for periodic PM6.

    Parameters
    ----------
    system : PeriodicSystem
        Periodic system at reference geometry.
    h : float
        Finite-difference step (bohr).
    cutoff_bohr, max_iter, conv_tol :
        Passed through to ``run_pm6_gamma``.

    Returns
    -------
    grad : ndarray of shape (n_atoms, 3)
        Energy gradient dE/dR in Hartree/bohr.
    """
    result = compute_pm6_gamma_derivatives_fd(
        system,
        params,
        h=h,
        cutoff_bohr=cutoff_bohr,
        max_iter=max_iter,
        conv_tol=conv_tol,
        compute_stress=False,
    )
    return np.asarray(result.gradient)


def compute_pm6_gamma_stress_fd(
    system: PeriodicSystem,
    params=None,
    *,
    h: float = 0.001,
    cutoff_bohr: float = 15.0,
    max_iter: int = 100,
    conv_tol: float = 1e-7,
) -> np.ndarray:
    """Finite-difference stress tensor for periodic PM6.

    Parameters
    ----------
    system : PeriodicSystem
        Periodic system at reference lattice.
    h : float
        Finite-difference strain step (dimensionless).
    cutoff_bohr, max_iter, conv_tol :
        Passed through to ``run_pm6_gamma``.

    Returns
    -------
    stress : ndarray of shape (3, 3)
        Stress tensor s_{ij} = (1/V) . dE/de_{ij} in Hartree/bohr^3.
    """
    result = compute_pm6_gamma_derivatives_fd(
        system,
        params,
        h=h,
        cutoff_bohr=cutoff_bohr,
        max_iter=max_iter,
        conv_tol=conv_tol,
        compute_gradient=False,
    )
    return np.asarray(result.stress)


def compute_pm6_gamma_derivatives_fd(
    system: PeriodicSystem,
    params=None,
    *,
    h: float = 0.001,
    cutoff_bohr: float = 15.0,
    max_iter: int = 100,
    conv_tol: float = 1e-7,
    compute_gradient: bool = True,
    compute_stress: bool = True,
) -> Any:
    """Compute periodic PM6 derivatives in one native FD batch."""
    if not compute_gradient and not compute_stress:
        raise ValueError("periodic PM6 FD must request gradient or stress")
    properties = ["energy"]
    if compute_gradient:
        properties.append("gradient")
    if compute_stress:
        properties.append("stress")
    plan_periodic_semiempirical_route(
        "pm6",
        system,
        properties=properties,
    )
    params = _get_params(system, params)

    scf_options = _nddo.PeriodicPM6Options()
    scf_options.cutoff_bohr = cutoff_bohr
    scf_options.max_iter = max_iter
    scf_options.conv_tol = conv_tol
    fd_options = _nddo.PeriodicNDDOFDBatchOptions()
    fd_options.coordinate_step = h
    fd_options.strain_step = h
    fd_options.compute_gradient = compute_gradient
    fd_options.compute_stress = compute_stress
    fd_options.strain_positions = True
    return _nddo.compute_pm6_gamma_fd_batch(
        system,
        params,
        scf_options,
        fd_options,
    )


class PeriodicPM6Model:
    """Periodic PM6 at Gamma-point with FD gradients and stress.

    Parameters
    ----------
    system : PeriodicSystem
        The periodic system (unit cell atoms and lattice vectors).
    params : PM6ParameterSet, optional
        PM6 parameter set. Loaded from defaults if not given.
    cutoff_bohr : float
        Lattice-sum cutoff.
    """

    def __init__(
        self,
        system: PeriodicSystem,
        params=None,
        *,
        cutoff_bohr: float = 15.0,
        max_iter: int = 100,
        conv_tol: float = 1e-7,
    ):
        self._route_plan = plan_periodic_semiempirical_route(
            "pm6",
            system,
            properties=("energy", "gradient", "stress"),
        )
        self._system = system
        self._params = _get_params(system, params)
        self._cutoff = cutoff_bohr
        self._max_iter = max_iter
        self._conv_tol = conv_tol
        self._last_result = None

    @property
    def system(self) -> PeriodicSystem:
        return self._system

    @property
    def params(self):
        return self._params

    @property
    def parameter_identity(self) -> Optional[str]:
        identity = getattr(self._last_result, "parameter_identity", None)
        return None if identity is None else str(identity)

    @property
    def parameter_sha256(self) -> Optional[str]:
        sha256 = getattr(self._last_result, "parameter_sha256", None)
        return None if sha256 is None else str(sha256)

    def _retain_derivatives(self, result):
        previous_sha256 = getattr(self._last_result, "parameter_sha256", None)
        current_sha256 = getattr(result, "parameter_sha256", None)
        if (
            previous_sha256 is not None
            and current_sha256 is not None
            and str(previous_sha256) != str(current_sha256)
        ):
            raise RuntimeError(
                "periodic PM6 derivatives require the same immutable "
                "parameter snapshot as the retained energy result"
            )
        self._last_result = result
        return result

    def _require_retained_parameter_snapshot(self) -> None:
        retained_sha256 = getattr(self._last_result, "parameter_sha256", None)
        if retained_sha256 is not None and (
            str(self._params.content_sha256()) != str(retained_sha256)
        ):
            raise RuntimeError(
                "periodic PM6 derivatives require the same immutable "
                "parameter snapshot as the retained energy result"
            )

    def energy(self, system: PeriodicSystem | None = None) -> float:
        """Total energy per unit cell (Hartree)."""
        target = self._system if system is None else system
        self._last_result = run_pm6_gamma(
            target,
            self._params,
            cutoff_bohr=self._cutoff,
            max_iter=self._max_iter,
            conv_tol=self._conv_tol,
        )
        return float(self._last_result.energy)

    def gradient(self, system: PeriodicSystem | None = None) -> np.ndarray:
        """Nuclear gradient dE/dR (Hartree/bohr), shape (n_atoms, 3)."""
        target = self._system if system is None else system
        self._require_retained_parameter_snapshot()
        result = self._retain_derivatives(
            compute_pm6_gamma_derivatives_fd(
                target,
                self._params,
                cutoff_bohr=self._cutoff,
                max_iter=self._max_iter,
                conv_tol=self._conv_tol,
                compute_stress=False,
            )
        )
        return np.asarray(result.gradient)

    def stress(self, system: PeriodicSystem | None = None) -> np.ndarray:
        """Stress tensor s_{ij} (Hartree/bohr^3), shape (3, 3)."""
        target = self._system if system is None else system
        self._require_retained_parameter_snapshot()
        result = self._retain_derivatives(
            compute_pm6_gamma_derivatives_fd(
                target,
                self._params,
                cutoff_bohr=self._cutoff,
                max_iter=self._max_iter,
                conv_tol=self._conv_tol,
                compute_gradient=False,
            )
        )
        return np.asarray(result.stress)

    def derivatives(self, system: PeriodicSystem | None = None) -> Any:
        """Return gradient and stress from one native FD batch."""
        target = self._system if system is None else system
        self._require_retained_parameter_snapshot()
        return self._retain_derivatives(
            compute_pm6_gamma_derivatives_fd(
                target,
                self._params,
                cutoff_bohr=self._cutoff,
                max_iter=self._max_iter,
                conv_tol=self._conv_tol,
            )
        )
