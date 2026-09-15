"""Fail-closed compatibility surface for the retired periodic OMx prototype.

The old Gamma-point driver did not implement the published OMx ORT, ECP, and
penetration Hamiltonian.  Every public entry point in this module is therefore
gated by the canonical route planner; topology-bound OMx-SECCM is a separate
experimental route.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from vibeqc._vibeqc_core import PeriodicSystem
from vibeqc._vibeqc_core.semiempirical import nddo as _nddo
from vibeqc.semiempirical.routes import plan_periodic_semiempirical_route

_VALID_OMX_VARIANTS = ("om1", "om2", "om3")


def _validate_omx_variant(variant: str) -> None:
    if variant not in _VALID_OMX_VARIANTS:
        raise ValueError(
            f"variant must be one of {_VALID_OMX_VARIANTS}, got {variant!r}"
        )


def _get_omx_params(system: PeriodicSystem, variant: str = "om2", params=None):
    """Load OMx params for the given variant."""
    _validate_omx_variant(variant)
    if params is None:
        from vibeqc.semiempirical.methods.omx_params import load_omx_params

        params = load_omx_params(variant)
    if params.method_name() != variant:
        raise ValueError(
            f"periodic {variant.upper()} requires matching parameters, "
            f"not {params.method_name()}"
        )
    return params


def run_omx_gamma(
    system: PeriodicSystem,
    variant: str = "om2",
    params=None,
    *,
    cutoff_bohr: float = 15.0,
    max_iter: int = 100,
    conv_tol: float = 1e-7,
    warmup_iters: int = 5,
    density_mixing: float = 0.2,
    eval_floor: float = 1e-5,
):
    """Reject the retired periodic OMx prototype through the route gate.

    Parameters
    ----------
    system : PeriodicSystem
    variant : str
        One of "om1", "om2", "om3".
    params : OMxParameterSet, optional
    cutoff_bohr : float
    max_iter : int
    conv_tol : float
    warmup_iters : int
        Orthogonal-basis iterations before DIIS starts.
    density_mixing : float
        Linear mixing fraction (0-1) during the pre-DIIS warm-up.
    eval_floor : float
        Numerical eigenvalue floor for Loewdin S^(-1/2).

    Returns
    -------
    PeriodicOMxResult
    """
    _validate_omx_variant(variant)
    plan_periodic_semiempirical_route(variant, system)
    params = _get_omx_params(system, variant, params)
    opts = _nddo.PeriodicOMxOptions()
    opts.cutoff_bohr = cutoff_bohr
    opts.max_iter = max_iter
    opts.conv_tol = conv_tol
    opts.warmup_iters = warmup_iters
    opts.density_mixing = density_mixing
    opts.eval_floor = eval_floor
    return _nddo.run_omx_gamma(system, params, opts)


class PeriodicOMxModel:
    """Compatibility wrapper that fails closed for periodic OM1/OM2/OM3.

    Parameters
    ----------
    system : PeriodicSystem
    variant : str
        "om1", "om2", or "om3".
    params : OMxParameterSet, optional
    cutoff_bohr : float
    """

    def __init__(
        self,
        system: PeriodicSystem,
        variant: str = "om2",
        params=None,
        *,
        cutoff_bohr: float = 15.0,
        max_iter: int = 200,
        conv_tol: float = 1e-7,
        warmup_iters: int = 3,
        density_mixing: float = 0.3,
        eval_floor: float = 1e-5,
    ):
        _validate_omx_variant(variant)
        self._route_plan = plan_periodic_semiempirical_route(
            variant,
            system,
            properties=("energy", "gradient", "stress"),
        )
        self._system = system
        self._variant = variant
        self._params = _get_omx_params(system, variant, params)
        self._cutoff = cutoff_bohr
        self._max_iter = max_iter
        self._conv_tol = conv_tol
        self._warmup = warmup_iters
        self._mix = density_mixing
        self._eval_floor = eval_floor

    @property
    def system(self) -> PeriodicSystem:
        return self._system

    def energy(self, system: PeriodicSystem | None = None) -> float:
        target = self._system if system is None else system
        result = run_omx_gamma(
            target,
            self._variant,
            self._params,
            cutoff_bohr=self._cutoff,
            max_iter=self._max_iter,
            conv_tol=self._conv_tol,
            warmup_iters=self._warmup,
            density_mixing=self._mix,
            eval_floor=self._eval_floor,
        )
        if not result.converged:
            raise RuntimeError(
                f"periodic {self._variant.upper()} did not converge "
                f"after {int(result.n_iter)} iterations"
            )
        return result.energy

    def gradient(self, system: PeriodicSystem | None = None) -> np.ndarray:
        """FD nuclear gradient (Hartree/bohr), shape (n_atoms, 3)."""
        target = self._system if system is None else system
        return compute_omx_gamma_gradient_fd(
            target,
            self._variant,
            self._params,
            cutoff_bohr=self._cutoff,
            max_iter=self._max_iter,
            conv_tol=self._conv_tol,
            warmup_iters=self._warmup,
            density_mixing=self._mix,
            eval_floor=self._eval_floor,
        )

    def stress(self, system: PeriodicSystem | None = None) -> np.ndarray:
        """FD stress tensor (Hartree/bohr^3), shape (3, 3)."""
        target = self._system if system is None else system
        return compute_omx_gamma_stress_fd(
            target,
            self._variant,
            self._params,
            cutoff_bohr=self._cutoff,
            max_iter=self._max_iter,
            conv_tol=self._conv_tol,
            warmup_iters=self._warmup,
            density_mixing=self._mix,
            eval_floor=self._eval_floor,
        )

    def derivatives(self, system: PeriodicSystem | None = None) -> Any:
        """Return gradient and stress from one native FD batch."""
        target = self._system if system is None else system
        return compute_omx_gamma_derivatives_fd(
            target,
            self._variant,
            self._params,
            cutoff_bohr=self._cutoff,
            max_iter=self._max_iter,
            conv_tol=self._conv_tol,
            warmup_iters=self._warmup,
            density_mixing=self._mix,
            eval_floor=self._eval_floor,
        )


# ---------------------------------------------------------------------------
# FD gradient and stress
# ---------------------------------------------------------------------------


def compute_omx_gamma_gradient_fd(
    system: PeriodicSystem,
    variant: str = "om2",
    params=None,
    *,
    h: float = 0.001,
    cutoff_bohr: float = 15.0,
    max_iter: int = 200,
    conv_tol: float = 1e-7,
    warmup_iters: int = 3,
    density_mixing: float = 0.3,
    eval_floor: float = 1e-5,
) -> np.ndarray:
    """Finite-difference gradient for periodic OMx."""
    result = compute_omx_gamma_derivatives_fd(
        system,
        variant,
        params,
        h=h,
        cutoff_bohr=cutoff_bohr,
        max_iter=max_iter,
        conv_tol=conv_tol,
        warmup_iters=warmup_iters,
        density_mixing=density_mixing,
        eval_floor=eval_floor,
        compute_stress=False,
    )
    return np.asarray(result.gradient)


def compute_omx_gamma_stress_fd(
    system: PeriodicSystem,
    variant: str = "om2",
    params=None,
    *,
    h: float = 0.001,
    cutoff_bohr: float = 15.0,
    max_iter: int = 200,
    conv_tol: float = 1e-7,
    warmup_iters: int = 3,
    density_mixing: float = 0.3,
    eval_floor: float = 1e-5,
) -> np.ndarray:
    """Finite-difference stress tensor for periodic OMx."""
    result = compute_omx_gamma_derivatives_fd(
        system,
        variant,
        params,
        h=h,
        cutoff_bohr=cutoff_bohr,
        max_iter=max_iter,
        conv_tol=conv_tol,
        warmup_iters=warmup_iters,
        density_mixing=density_mixing,
        eval_floor=eval_floor,
        compute_gradient=False,
    )
    return np.asarray(result.stress)


def compute_omx_gamma_derivatives_fd(
    system: PeriodicSystem,
    variant: str = "om2",
    params=None,
    *,
    h: float = 0.001,
    cutoff_bohr: float = 15.0,
    max_iter: int = 200,
    conv_tol: float = 1e-7,
    warmup_iters: int = 3,
    density_mixing: float = 0.3,
    eval_floor: float = 1e-5,
    compute_gradient: bool = True,
    compute_stress: bool = True,
) -> Any:
    """Compute periodic OMx derivatives in one native FD batch."""
    _validate_omx_variant(variant)
    if not compute_gradient and not compute_stress:
        raise ValueError("periodic OMx FD must request gradient or stress")
    properties = ["energy"]
    if compute_gradient:
        properties.append("gradient")
    if compute_stress:
        properties.append("stress")
    plan_periodic_semiempirical_route(
        variant,
        system,
        properties=properties,
    )
    params = _get_omx_params(system, variant, params)

    scf_options = _nddo.PeriodicOMxOptions()
    scf_options.cutoff_bohr = cutoff_bohr
    scf_options.max_iter = max_iter
    scf_options.conv_tol = conv_tol
    scf_options.warmup_iters = warmup_iters
    scf_options.density_mixing = density_mixing
    scf_options.eval_floor = eval_floor
    fd_options = _nddo.PeriodicNDDOFDBatchOptions()
    fd_options.coordinate_step = h
    fd_options.strain_step = h
    fd_options.compute_gradient = compute_gradient
    fd_options.compute_stress = compute_stress
    fd_options.strain_positions = True
    return _nddo.compute_omx_gamma_fd_batch(
        system,
        params,
        scf_options,
        fd_options,
    )
