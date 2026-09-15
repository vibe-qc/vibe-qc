"""One-call native molecular basis optimisation via the analytic-gradient BDIIS.

Turns the completed molecular analytic energy gradient (RHF / UHF / RKS / UKS x
exponents + coefficients, see
[`energy_gradient.py`](../energy_gradient.py)) into a single user-facing call.
Unlike :mod:`vibeqc.basis_optimization.recipes.production` (multi-crystal,
derivative-free NLopt via the external-program Calculator), this is the
vibe-qc-native *molecular* path: it builds the consistent in-process
``(objective, analytic-gradient)`` pair and drives the robust
:func:`vibeqc.basis_optimization.optimize_bdiis` loop with the analytic
gradient -- no per-parameter re-SCF finite differences.

Use::

    from vibeqc.basis_optimization.parametrise import (
        BasisParametrisation, FreeSpec, Transform)
    from vibeqc.basis_optimization.recipes.molecular import optimize_molecular_basis
    import vibeqc as vq

    par = BasisParametrisation(atoms={...}, free=[FreeSpec(...), ...])
    res = optimize_molecular_basis(
        par, lambda: vq.Molecule([vq.Atom(6, [0, 0, 0])], multiplicity=3),
        functional="PBE", open_shell=True)          # UKS
    print(res.summary())
    optimised_basis = res.optimized_atoms           # {symbol: CrystalAtomBasis}

The reference is selected exactly as in
:func:`vibeqc.basis_optimization.recipes.objective.make_native_objective_gradient`:
``functional=None`` -> Hartree-Fock (RHF / UHF by ``open_shell``);
``functional="PBE"``/... -> Kohn-Sham (RKS / UKS by ``open_shell``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

import numpy as np


@dataclass
class MolecularOptResult:
    """Outcome of a native molecular basis optimisation."""

    optimized_atoms: dict          # {symbol: CrystalAtomBasis} at the optimum
    x_optimal: np.ndarray          # optimiser-space parameter vector
    labels: list[str]
    starting_objective: float
    optimal_objective: float
    n_evaluations: int
    wall_seconds: float
    converged: bool
    message: str
    method: str
    history: list = field(default_factory=list)
    citations: tuple = ()          # method-citation keys (database.toml) to cite

    @property
    def improvement_mha(self) -> float:
        return (self.starting_objective - self.optimal_objective) * 1000.0

    def summary(self) -> str:
        lines = [
            f"molecular basis optimisation ({self.method})",
            f"  objective: {self.starting_objective:.8f} -> "
            f"{self.optimal_objective:.8f} Ha",
            f"  improvement: {self.improvement_mha:+.4f} mHa  (lower is better)",
            f"  evals: {self.n_evaluations}   wall: {self.wall_seconds:.1f} s",
            f"  converged: {self.converged}  ({self.message})",
        ]
        for lab, xs, xo in zip(self.labels, self._x_start_phys(), self._x_opt_phys()):
            lines.append(f"    {lab:<28} {xs:>12.6f} -> {xo:>12.6f}")
        if self.citations:
            lines.append(f"  cite (method): {', '.join(self.citations)}")
            lines.append("  + the source basis set + functional (per the SCF "
                         "references block)")
        return "\n".join(lines)

    # physical-space parameter values (best-effort; populated by the recipe)
    _start_phys: list = field(default_factory=list)
    _opt_phys: list = field(default_factory=list)

    def _x_start_phys(self):
        return self._start_phys or [float("nan")] * len(self.labels)

    def _x_opt_phys(self):
        return self._opt_phys or [float("nan")] * len(self.labels)


def _method_label(functional: Optional[str], open_shell: bool) -> str:
    if functional is None:
        return "UHF" if open_shell else "RHF"
    tag = "UKS" if open_shell else "RKS"
    return f"{tag}/{functional}"


def optimize_molecular_basis(
    parametrisation: Any,
    molecule_factory: Callable[[], Any],
    *,
    functional: Optional[str] = None,
    open_shell: bool = False,
    analytic: bool = True,
    use_cond_penalty: bool = False,
    cond_gamma: float = 1e-3,
    use_ld_penalty: bool = False,
    ld_lambda: float = 1e3,
    gated_symbols: Optional[Iterable[str]] = None,
    max_iter: int = 200,
    tol_energy: float = 1e-7,
    tol_grad: Optional[float] = None,
    diis_size: int = 8,
    trust_radius: float = 0.5,
) -> MolecularOptResult:
    """Optimise a molecular basis in-process with the analytic-gradient BDIIS.

    Parameters
    ----------
    parametrisation, molecule_factory
        The free parameters and the molecule, exactly as consumed by
        :func:`make_native_objective_gradient`.
    functional, open_shell
        Reference selector (HF vs KS, restricted vs unrestricted).
    analytic
        Drive BDIIS with the analytic gradient (default). ``False`` falls back
        to the driver's finite-difference gradient -- needed for SP
        ``coeff_s``/``coeff_p`` parameters, which have no analytic derivative.
    use_cond_penalty / cond_gamma / use_ld_penalty / ld_lambda / gated_symbols
        Conditioning / linear-dependence penalty, as in
        :func:`make_native_objective_gradient`.
    max_iter, tol_energy, tol_grad, diis_size, trust_radius
        Passed to :func:`optimize_bdiis`. ``tol_grad`` defaults to 1e-5 for HF
        and 1e-4 for KS (the grid-based KS gradient is accurate to ~1e-5).

    Returns
    -------
    MolecularOptResult
        Carries the optimised ``{symbol: CrystalAtomBasis}`` dict.
    """
    from ..io import TempBasisLibrary
    from .objective import make_native_objective_gradient

    if tol_grad is None:
        tol_grad = 1e-4 if functional is not None else 1e-5

    with TempBasisLibrary() as lib:
        objective, gradient = make_native_objective_gradient(
            parametrisation, molecule_factory, lib,
            open_shell=open_shell, functional=functional,
            use_cond_penalty=use_cond_penalty, cond_gamma=cond_gamma,
            use_ld_penalty=use_ld_penalty, ld_lambda=ld_lambda,
            gated_symbols=gated_symbols,
        )
        return _drive_bdiis(
            parametrisation, objective, gradient,
            method=_method_label(functional, open_shell), analytic=analytic,
            max_iter=max_iter, tol_energy=tol_energy, tol_grad=tol_grad,
            diis_size=diis_size, trust_radius=trust_radius,
        )


def optimize_molecular_basis_multi(
    parametrisation: Any,
    molecule_factories,
    *,
    weights=None,
    functional: Optional[str] = None,
    open_shell: bool = False,
    analytic: bool = True,
    use_cond_penalty: bool = False,
    cond_gamma: float = 1e-3,
    use_ld_penalty: bool = False,
    ld_lambda: float = 1e3,
    gated_symbols: Optional[Iterable[str]] = None,
    max_iter: int = 200,
    tol_energy: float = 1e-7,
    tol_grad: Optional[float] = None,
    diis_size: int = 8,
    trust_radius: float = 0.5,
) -> MolecularOptResult:
    """Optimise one basis jointly against a *set* of molecules (calibration set).

    The multi-system counterpart of :func:`optimize_molecular_basis` -- the way
    real basis sets are calibrated (objective = S_s w_s E(system_s), one shared
    basis). All systems use the same reference (``functional``/``open_shell``);
    ``weights`` defaults to 1.0 per system. The conditioning/LD penalty is
    applied once at the joint level. Returns a :class:`MolecularOptResult` whose
    ``optimal_objective`` is the (weighted) total over the set.
    """
    from ..io import TempBasisLibrary
    from .objective import make_multi_native_objective_gradient

    if tol_grad is None:
        tol_grad = 1e-4 if functional is not None else 1e-5

    n_sys = len(list(molecule_factories))
    method = f"{_method_label(functional, open_shell)} ({n_sys} systems)"

    with TempBasisLibrary() as lib:
        objective, gradient = make_multi_native_objective_gradient(
            parametrisation, molecule_factories, lib,
            weights=weights, open_shell=open_shell, functional=functional,
            use_cond_penalty=use_cond_penalty, cond_gamma=cond_gamma,
            use_ld_penalty=use_ld_penalty, ld_lambda=ld_lambda,
            gated_symbols=gated_symbols,
        )
        return _drive_bdiis(
            parametrisation, objective, gradient,
            method=method, analytic=analytic,
            max_iter=max_iter, tol_energy=tol_energy, tol_grad=tol_grad,
            diis_size=diis_size, trust_radius=trust_radius,
        )


def _drive_bdiis(parametrisation, objective, gradient, *, method, analytic,
                 max_iter, tol_energy, tol_grad, diis_size, trust_radius):
    """Run optimize_bdiis on a prebuilt (objective, gradient) and package the
    :class:`MolecularOptResult`. The caller owns the basis-library lifetime."""
    from ..bdiis import method_citations, optimize_bdiis

    x0 = parametrisation.pack()
    start_phys = [spec.transform.from_optim(float(xi))
                  for spec, xi in zip(parametrisation.free, x0)]
    f_start = objective(x0)
    res = optimize_bdiis(
        objective, x0,
        bounds=parametrisation.optim_bounds(),
        grad=gradient if analytic else None,
        max_iter=max_iter, tol_energy=tol_energy, tol_grad=tol_grad,
        diis_size=diis_size, trust_radius=trust_radius,
    )
    opt_phys = [spec.transform.from_optim(float(xi))
                for spec, xi in zip(parametrisation.free, res.x)]
    return MolecularOptResult(
        optimized_atoms=parametrisation.unpack(res.x),
        x_optimal=res.x,
        labels=parametrisation.labels(),
        starting_objective=float(f_start),
        optimal_objective=float(res.fun),
        n_evaluations=res.n_evaluations,
        wall_seconds=res.wall_seconds,
        converged=res.success,
        message=res.message,
        method=method,
        history=res.history,
        citations=tuple(method_citations()),
        _start_phys=start_phys,
        _opt_phys=opt_phys,
    )
