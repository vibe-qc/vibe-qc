"""Objective factories using the Calculator abstraction.

:func:`make_objective` -- single-structure energy evaluation.
:func:`make_multi_objective` -- joint minimization across multiple
    structures (how pob-TZVP was created).

Both accept a :class:`Calculator` so the SCF engine (CRYSTAL14,
vibe-qc, ORCA, ...) can be swapped without changing the optimization
logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

from ...basis_crystal import emit_crystal
from ..ld_diagnostics import EPS_LD, cond_penalty_from_atom, ld_penalty_from_atom
from ..parametrise import BasisParametrisation


def _summed_ld_penalty(
    atoms: dict,
    gated: Optional[Iterable[str]],
    lambda_ld: float,
    epsilon: float,
) -> float:
    """Sum the in-memory LD penalty over each gated element symbol.

    Any failure to build / diagonalise the per-element overlap pushes the
    objective to +inf -- the optimizer treats that iterate as forbidden,
    which is the correct semantics for a degenerate basis.
    """
    targets = list(gated) if gated is not None else list(atoms.keys())
    total = 0.0
    for sym in targets:
        try:
            total += ld_penalty_from_atom(
                atoms[sym], lambda_ld=lambda_ld, epsilon=epsilon
            )
        except Exception:
            return float("inf")
    return total


def _summed_cond_penalty(
    atoms: dict,
    gated: Optional[Iterable[str]],
    gamma: float,
    epsilon: float,
) -> float:
    """Sum the in-memory g.ln κ(S) penalty over each gated element symbol.

    The condition-number counterpart of :func:`_summed_ld_penalty`; same
    +inf-on-failure semantics. This is the CRYSTAL OPTBASIS / VandeVondele
    objective term (see
    :func:`vibeqc.basis_optimization.ld_diagnostics.condition_number_penalty`).
    """
    targets = list(gated) if gated is not None else list(atoms.keys())
    total = 0.0
    for sym in targets:
        try:
            total += cond_penalty_from_atom(atoms[sym], gamma=gamma, epsilon=epsilon)
        except Exception:
            return float("inf")
    return total


def make_native_objective_gradient(
    parametrisation: BasisParametrisation,
    molecule_factory,
    library,
    *,
    open_shell: bool = False,
    functional: Optional[str] = None,
    scf_runner=None,
    use_cond_penalty: bool = False,
    cond_gamma: float = 1e-3,
    use_ld_penalty: bool = False,
    ld_lambda: float = 1e3,
    ld_epsilon: float = EPS_LD,
    gated_symbols: Optional[Iterable[str]] = None,
    basis_name_prefix: str = "bopt",
) -> tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]:
    """In-process HF/KS objective + matching *analytic* gradient for BDIIS.

    Unlike :func:`make_objective` (which drives an external Calculator and has
    no gradient), this returns a consistent ``(objective, gradient)`` pair for
    a vibe-qc-native molecular optimisation -- drop straight into
    :func:`vibeqc.basis_optimization.optimize_bdiis` as
    ``optimize_bdiis(objective, x0, grad=gradient, ...)`` for a fully analytic,
    fully in-process loop (no per-parameter re-SCF finite differences).

    The reference is selected by ``functional`` / ``open_shell``:

    * ``functional=None`` (default) -- Hartree-Fock. ``open_shell=False`` -> RHF
      (``vq.run_rhf`` + :func:`energy_gradient_analytic`); ``open_shell=True`` ->
      UHF (``vq.run_uhf`` + :func:`energy_gradient_analytic_uhf`), for the
      open-shell atoms (C, N, O, ...) the pob recipe needs.
    * ``functional="PBE"`` / ``"B3LYP"`` / ... -- closed-shell RKS (``vq.run_rks``
      + :func:`energy_gradient_analytic_rks`, LDA/GGA/hybrid). ``open_shell=True``
      with a functional (UKS) is not yet supported and raises.

    The :func:`make_rhf_/make_uhf_/make_rks_native_objective_gradient` wrappers
    pin the choice.

    objective(x)
        E(molecule, x) + S penalties; ``np.inf`` if the SCF fails to converge
        or a penalty overlap is degenerate.
    gradient(x)
        the matching analytic energy gradient (frozen-density Pulay assembly +,
        for KS, the explicit grid XC term) + :func:`penalty_gradient`.

    Both share the same ``parametrisation`` / ``molecule_factory`` / ``library``,
    so the gradient is the exact derivative of the objective. Exponent and
    (non-SP) coefficient free parameters are both analytic; SP
    ``coeff_s``/``coeff_p`` raise -- use the FD path (default ``grad=None`` in
    ``optimize_bdiis``) for those.

    The penalty terms (``g.ln κ`` and/or the l_min LD hinge) are the per-element
    atomic-overlap terms shared with :func:`make_objective`; ``gated_symbols``
    selects which elements they cover (default: all in the parametrisation).
    """
    from ..energy_gradient import (
        energy_gradient_analytic,
        energy_gradient_analytic_rks,
        energy_gradient_analytic_uhf,
        energy_gradient_analytic_uks,
    )
    from ..gradients import penalty_gradient

    is_ks = functional is not None

    def _runner():
        if scf_runner is not None:
            return scf_runner
        import vibeqc as vq

        if is_ks and open_shell:
            def _uks(mol, basis):
                opts = vq.UKSOptions()
                opts.functional = functional
                return vq.run_uks(mol, basis, opts)
            return _uks
        if is_ks:
            def _rks(mol, basis):
                opts = vq.RKSOptions()
                opts.functional = functional
                return vq.run_rks(mol, basis, opts)
            return _rks
        return vq.run_uhf if open_shell else vq.run_rhf

    def _add_penalties(atoms, value: float) -> float:
        if use_cond_penalty:
            pc = _summed_cond_penalty(atoms, gated_symbols, cond_gamma, ld_epsilon)
            if not np.isfinite(pc):
                return float("inf")
            value += pc
        if use_ld_penalty:
            p = _summed_ld_penalty(atoms, gated_symbols, ld_lambda, ld_epsilon)
            if not np.isfinite(p):
                return float("inf")
            value += p
        return value

    n_calls = [0]

    def objective(x: np.ndarray) -> float:
        import vibeqc as vq

        atoms = parametrisation.unpack(np.asarray(x, dtype=float))
        n_calls[0] += 1
        name = library.write_g94(atoms, basis_name=f"{basis_name_prefix}-o{n_calls[0]:06d}")
        mol = molecule_factory()
        try:
            res = _runner()(mol, vq.BasisSet(mol, name))
        except Exception:
            return float("inf")
        if res is None or not getattr(res, "converged", False):
            return float("inf")
        return _add_penalties(atoms, float(res.energy))

    def gradient(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if is_ks and open_shell:
            g = energy_gradient_analytic_uks(
                parametrisation, molecule_factory, library, x,
                functional=functional, basis_name_prefix=f"{basis_name_prefix}-g",
            )
        elif is_ks:
            g = energy_gradient_analytic_rks(
                parametrisation, molecule_factory, library, x,
                functional=functional, basis_name_prefix=f"{basis_name_prefix}-g",
            )
        else:
            grad_fn = energy_gradient_analytic_uhf if open_shell else energy_gradient_analytic
            g = grad_fn(
                parametrisation, molecule_factory, library, _runner(), x,
                basis_name_prefix=f"{basis_name_prefix}-g",
            )
        if use_cond_penalty or use_ld_penalty:
            g = g + penalty_gradient(
                parametrisation, x,
                use_cond_penalty=use_cond_penalty, gamma=cond_gamma,
                use_ld_penalty=use_ld_penalty, lambda_ld=ld_lambda,
                epsilon=ld_epsilon,
                gated_symbols=list(gated_symbols) if gated_symbols is not None else None,
            )
        return g

    return objective, gradient


def make_rhf_native_objective_gradient(parametrisation, molecule_factory, library,
                                       **kwargs):
    """Closed-shell (RHF) :func:`make_native_objective_gradient`."""
    return make_native_objective_gradient(
        parametrisation, molecule_factory, library, open_shell=False, **kwargs
    )


def make_uhf_native_objective_gradient(parametrisation, molecule_factory, library,
                                       **kwargs):
    """Open-shell (UHF) :func:`make_native_objective_gradient` -- for the
    open-shell atoms (C, N, O, ...) the pob recipe optimises."""
    return make_native_objective_gradient(
        parametrisation, molecule_factory, library, open_shell=True, **kwargs
    )


def make_rks_native_objective_gradient(parametrisation, molecule_factory, library,
                                       *, functional, **kwargs):
    """Closed-shell RKS (DFT) :func:`make_native_objective_gradient` -- LDA/GGA/
    hybrid basis optimisation against ``functional``."""
    return make_native_objective_gradient(
        parametrisation, molecule_factory, library,
        functional=functional, open_shell=False, **kwargs
    )


def make_uks_native_objective_gradient(parametrisation, molecule_factory, library,
                                       *, functional, **kwargs):
    """Open-shell UKS (DFT) :func:`make_native_objective_gradient` -- LDA/GGA/
    hybrid basis optimisation of the open-shell atoms (C, N, O, ...) at DFT
    level, against ``functional``."""
    return make_native_objective_gradient(
        parametrisation, molecule_factory, library,
        functional=functional, open_shell=True, **kwargs
    )


def make_multi_native_objective_gradient(
    parametrisation: BasisParametrisation,
    molecule_factories,
    library,
    *,
    weights=None,
    open_shell: bool = False,
    functional: Optional[str] = None,
    use_cond_penalty: bool = False,
    cond_gamma: float = 1e-3,
    use_ld_penalty: bool = False,
    ld_lambda: float = 1e3,
    ld_epsilon: float = EPS_LD,
    gated_symbols: Optional[Iterable[str]] = None,
    basis_name_prefix: str = "bopt-multi",
) -> tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]:
    """Joint ``(objective, analytic-gradient)`` over a *set* of molecules.

    Real basis-set calibration fits one basis to a whole reference set, not a
    single system (this is how the pob sets were made). This is the molecular,
    analytic-gradient analogue of :func:`make_multi_objective`:

        objective(x) = S_s w_s . E(system_s, x) + S penalties
        gradient(x)  = S_s w_s . gradE(system_s, x) + grad(penalties)

    Each per-system ``E``/``gradE`` is the in-process
    :func:`make_native_objective_gradient` pair (penalty *off*); the
    conditioning / linear-dependence penalty is applied **once** at the joint
    level. All systems share one ``parametrisation`` (one basis) and the same
    reference (``functional`` / ``open_shell``) -- the pob convention. ``weights``
    defaults to 1.0 per system. ``np.inf`` propagates if any system's SCF fails.
    """
    from ..gradients import penalty_gradient

    factories = list(molecule_factories)
    if not factories:
        raise ValueError("at least one molecule_factory required")
    ws = list(weights) if weights is not None else [1.0] * len(factories)
    if len(ws) != len(factories):
        raise ValueError(
            f"weights length {len(ws)} != molecule_factories length {len(factories)}"
        )

    # Per-system (objective, gradient) pairs, penalty OFF (added once below).
    pairs = [
        make_native_objective_gradient(
            parametrisation, mf, library,
            open_shell=open_shell, functional=functional,
            basis_name_prefix=f"{basis_name_prefix}-s{i}",
        )
        for i, mf in enumerate(factories)
    ]

    def _add_penalties(atoms, value: float) -> float:
        if use_cond_penalty:
            pc = _summed_cond_penalty(atoms, gated_symbols, cond_gamma, ld_epsilon)
            if not np.isfinite(pc):
                return float("inf")
            value += pc
        if use_ld_penalty:
            p = _summed_ld_penalty(atoms, gated_symbols, ld_lambda, ld_epsilon)
            if not np.isfinite(p):
                return float("inf")
            value += p
        return value

    def objective(x: np.ndarray) -> float:
        total = 0.0
        for w, (obj_s, _grad_s) in zip(ws, pairs):
            e = obj_s(x)
            if not np.isfinite(e):
                return float("inf")
            total += w * e
        atoms = parametrisation.unpack(np.asarray(x, dtype=float))
        return _add_penalties(atoms, total)

    def gradient(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        g = np.zeros(len(parametrisation.free))
        for w, (_obj_s, grad_s) in zip(ws, pairs):
            g = g + w * grad_s(x)
        if use_cond_penalty or use_ld_penalty:
            g = g + penalty_gradient(
                parametrisation, x,
                use_cond_penalty=use_cond_penalty, gamma=cond_gamma,
                use_ld_penalty=use_ld_penalty, lambda_ld=ld_lambda,
                epsilon=ld_epsilon,
                gated_symbols=list(gated_symbols) if gated_symbols is not None else None,
            )
        return g

    return objective, gradient


def make_objective(
    parametrisation: BasisParametrisation,
    calculator,
    *,
    structure=None,
    Z: int | None = None,
    method: str = "rhf",
    use_ld_penalty: bool = False,
    ld_lambda: float = 1e3,
    ld_epsilon: float = EPS_LD,
    ld_gated_symbols: Optional[Iterable[str]] = None,
    use_cond_penalty: bool = False,
    cond_gamma: float = 1e-3,
    cond_gated_symbols: Optional[Iterable[str]] = None,
) -> Callable[[np.ndarray], float]:
    """Build a single-structure objective using a Calculator.

    On each call: unpack basis -> emit_crystal -> calculator.evaluate_* -> energy
    (+ optional penalty terms). Returns ``np.inf`` on failure or on
    overlap-construction failure.

    Parameters
    ----------
    use_ld_penalty
        If True, each evaluation adds ``l_ld . max(0, e - l_min)^2`` for every
        gated element's atomic overlap, computed in-memory (no file round-trip).
    ld_lambda, ld_epsilon
        Penalty strength and LD threshold.
    ld_gated_symbols
        Element symbols whose per-atom overlap is checked; defaults to all
        elements present in the parametrisation.
    use_cond_penalty
        If True, each evaluation adds the VandeVondele / OPTBASIS
        condition-number term ``g . ln κ(S)`` per gated element -- the
        smooth conditioning penalty BDIIS is designed to optimise against.
    cond_gamma
        Condition-number penalty weight g (Hartree).
    cond_gated_symbols
        Element symbols whose κ is penalised; defaults to all elements.
    """
    if structure is not None:
        kind = "crystal"
    elif Z is not None:
        kind = "atom"
    else:
        raise ValueError("structure or Z required")

    n_calls = [0]

    def objective(x: np.ndarray) -> float:
        n_calls[0] += 1

        atoms = parametrisation.unpack(x)
        basis_list = [atoms[k] for k in parametrisation.atoms_keys]
        basis_text = emit_crystal(basis_list)

        if kind == "atom":
            e = calculator.evaluate_atom(basis_text, Z, method=method)
        else:
            e = calculator.evaluate_crystal(basis_text, structure, method=method)

        if e is None:
            return float("inf")

        if use_ld_penalty:
            p = _summed_ld_penalty(atoms, ld_gated_symbols, ld_lambda, ld_epsilon)
            if not np.isfinite(p):
                return float("inf")
            e += p
        if use_cond_penalty:
            pc = _summed_cond_penalty(atoms, cond_gated_symbols, cond_gamma, ld_epsilon)
            if not np.isfinite(pc):
                return float("inf")
            e += pc
        return e

    return objective


def make_multi_objective(
    parametrisation: BasisParametrisation,
    calculator,
    structures: list,
    *,
    method: str = "rhf",
    weights: list[float] | None = None,
    use_ld_penalty: bool = False,
    ld_lambda: float = 1e3,
    ld_epsilon: float = EPS_LD,
    ld_gated_symbols: Optional[Iterable[str]] = None,
    use_cond_penalty: bool = False,
    cond_gamma: float = 1e-3,
    cond_gated_symbols: Optional[Iterable[str]] = None,
) -> Callable[[np.ndarray], float]:
    """Build a multi-system joint objective.

    The same basis is evaluated on every compound.  Returns the
    weighted sum S wᵢ . Eᵢ (+ optional penalty terms applied once
    per evaluation), or ``np.inf`` if any compound fails or the gated
    atomic overlap can't be built.

    See :func:`make_objective` for the penalty parameter semantics --
    they are identical here.
    """
    if not structures:
        raise ValueError("at least one structure required")

    ws = weights or [1.0] * len(structures)
    if len(ws) != len(structures):
        raise ValueError(
            f"weights length {len(ws)} != structures length {len(structures)}"
        )

    n_calls = [0]

    def objective(x: np.ndarray) -> float:
        n_calls[0] += 1

        atoms = parametrisation.unpack(x)
        basis_list = [atoms[k] for k in parametrisation.atoms_keys]
        basis_text = emit_crystal(basis_list)

        total = 0.0
        for s, w in zip(structures, ws):
            e = calculator.evaluate_crystal(basis_text, s, method=method)
            if e is None:
                return float("inf")
            total += w * e

        if use_ld_penalty:
            p = _summed_ld_penalty(atoms, ld_gated_symbols, ld_lambda, ld_epsilon)
            if not np.isfinite(p):
                return float("inf")
            total += p
        if use_cond_penalty:
            pc = _summed_cond_penalty(atoms, cond_gated_symbols, cond_gamma, ld_epsilon)
            if not np.isfinite(pc):
                return float("inf")
            total += pc

        return total

    return objective
