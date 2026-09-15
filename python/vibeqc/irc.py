"""IRC -- intrinsic reaction coordinate (mass-weighted reaction-path following).

The intrinsic reaction coordinate (Fukui, *Acc. Chem. Res.* 14, 363
(1981)) is the steepest-descent path from a transition state in
mass-weighted Cartesian coordinates. :func:`run_irc` starts at a saddle
point, steps off along the imaginary-frequency mode in both directions,
and follows the path downhill to the two connected minima -- confirming
*which* reactant and product a transition state actually connects (the
natural follow-up to a :func:`vibeqc.run_dimer` or :func:`vibeqc.run_neb`
saddle search).

The integrator is a stabilized mass-weighted steepest descent: each step
moves along the unit mass-weighted gradient; a step that would raise the
energy (an overshoot past the path or the minimum) is rejected and the
step size halved, which keeps the energy monotonically decreasing along
the path and converges cleanly at the minimum (the damped steepest
descent of Ishida, Morokuma & Komornicki, *J. Chem. Phys.* 66, 2153
(1977)). The more elaborate Gonzalez-Schlegel constrained integrator
(*J. Chem. Phys.* 90, 2154 (1989)) is a future refinement.

This version covers **molecular** systems (RHF / UHF / ROHF / RKS / UKS /
ROKS and the MACE potential), reusing the per-geometry gradient evaluators
of :func:`vibeqc.run_neb`. Periodic IRC is a follow-up. ROKS requires an
explicit transition mode until an analytic ROKS gradient is available for
the finite-difference Hessian.

References
=========
* Fukui, K. *The path of chemical reactions - the IRC approach.*
  Acc. Chem. Res. 14, 363 (1981). doi:10.1021/ar00072a001.
* Ishida, K.; Morokuma, K.; Komornicki, A. *The intrinsic reaction
  coordinate. An ab initio calculation...* J. Chem. Phys. 66, 2153
  (1977). doi:10.1063/1.434152.
* Gonzalez, C.; Schlegel, H. B. *An improved algorithm for reaction path
  following.* J. Chem. Phys. 90, 2154 (1989). doi:10.1063/1.456010.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from ._vibeqc_core import Molecule
from .dimer import _build_force_fn
from .neb import _positions_of, _rebuild_with_positions

System = Molecule
GradFn = Callable[[np.ndarray], "tuple[float, np.ndarray]"]


@dataclass
class IRCResult:
    """Outcome of a :func:`run_irc` reaction-path trace.

    The two branches are ordered **starting at the transition state**:
    ``forward_path[0]`` / ``reverse_path[0]`` are the first points off the
    TS (along ``+mode`` and ``-mode``), and the last entry of each is the
    converged minimum on that side.
    """

    transition_state: System
    ts_energy: float
    forward_path: list[System]
    forward_energies: np.ndarray
    reverse_path: list[System]
    reverse_energies: np.ndarray
    forward_converged: bool
    reverse_converged: bool
    method: Optional[str] = None
    basis: Optional[str] = None
    functional: Optional[str] = None
    mace_model_citation: Optional[str] = None

    @property
    def forward_minimum(self) -> Optional[System]:
        return self.forward_path[-1] if self.forward_path else None

    @property
    def reverse_minimum(self) -> Optional[System]:
        return self.reverse_path[-1] if self.reverse_path else None

    @property
    def barrier_forward(self) -> Optional[float]:
        """TS energy minus the forward-minimum energy (Hartree)."""
        if len(self.forward_energies):
            return float(self.ts_energy - self.forward_energies[-1])
        return None

    @property
    def barrier_reverse(self) -> Optional[float]:
        if len(self.reverse_energies):
            return float(self.ts_energy - self.reverse_energies[-1])
        return None

    def write_citations(self, stem: Any) -> "tuple[Any, Any]":
        """Write ``{stem}.bibtex`` + ``{stem}.references``.

        Fires the IRC papers (Fukui 1981 + the reaction-path-following
        algorithm) via ``routes.drivers.irc``, plus the per-geometry
        method / basis / functional -- or, for MACE, the model papers.
        """
        from .output.citations import (
            load_default_database,
            write_bibtex,
            write_references,
        )

        is_mace = (self.method or "").lower() == "mace"
        asm: dict[str, Any] = {
            "method": self.method or "RHF",
            "basis": "mace" if is_mace else (self.basis or "sto-3g"),
            "functional": None if is_mace else self.functional,
            "uses_irc": True,
        }
        if is_mace:
            asm["uses_integrals"] = False
            asm["uses_scf"] = False
            if self.mace_model_citation:
                asm["extra_entries"] = [self.mace_model_citation]
        refs = load_default_database().assemble(**asm)
        return write_bibtex(stem, refs), write_references(stem, refs)


# ---------------------------------------------------------------------------
# Core integrator (grad_fn only)
# ---------------------------------------------------------------------------


def _irc_branch(
    x_ts: np.ndarray,
    grad_fn: GradFn,
    masses_amu: np.ndarray,
    mode_cart: np.ndarray,
    sign: float,
    *,
    step_size: float,
    max_points: int,
    conv_tol_grad: float,
) -> "tuple[list[np.ndarray], list[float], bool]":
    """Trace one IRC branch by stabilized mass-weighted steepest descent.

    ``x_ts`` is the TS (n_atoms, 3) bohr; ``grad_fn(x) -> (E, grad)`` the
    Cartesian gradient; ``mode_cart`` the unit Cartesian transition vector;
    ``sign`` selects the branch (+1 / -1). Returns ``(positions, energies,
    converged)``.
    """
    n_atoms = x_ts.shape[0]
    m3 = np.repeat(np.asarray(masses_amu, dtype=float), 3)
    sqrt_m = np.sqrt(m3)

    mc = mode_cart.reshape(-1).astype(float)
    mc = mc / float(np.linalg.norm(mc))
    # Initial displacement off the saddle along ±mode (one step).
    x = x_ts.reshape(-1) + sign * step_size * mc
    energy, grad = grad_fn(x.reshape(n_atoms, 3))
    energy = float(energy)
    grad = np.asarray(grad, dtype=float).reshape(-1)

    positions = [x.reshape(n_atoms, 3).copy()]
    energies = [energy]
    s = float(step_size)
    converged = False

    for _ in range(max_points):
        if float(np.max(np.abs(grad))) < conv_tol_grad:
            converged = True
            break
        g_mw = grad / sqrt_m                       # mass-weighted gradient
        gn = float(np.linalg.norm(g_mw))
        # Step along the unit mass-weighted gradient, mapped back to
        # Cartesian: dx_i = -s (g_i/m_i) / |g_mw|.
        dx = -s * (grad / m3) / gn
        e_try, g_try = grad_fn((x + dx).reshape(n_atoms, 3))
        e_try = float(e_try)
        if e_try > energy + 1e-12:
            # Overshot the path / the minimum -- reject and shrink the step.
            s *= 0.5
            if s < 1e-5 * step_size:
                converged = True
                break
            continue
        x = x + dx
        energy = e_try
        grad = np.asarray(g_try, dtype=float).reshape(-1)
        positions.append(x.reshape(n_atoms, 3).copy())
        energies.append(energy)
        s = min(s * 1.2, step_size)

    return positions, energies, converged


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _transition_mode_from_hessian(
    ts: Molecule, basis: str, method: str, *,
    functional: Optional[str], rhf_options, uhf_options, rks_options,
    uks_options, rohf_options, gradient_options, grid_options,
    hessian_step_bohr: float,
    grid_level: str = "orca-defgrid3",
) -> "tuple[np.ndarray, np.ndarray]":
    """Compute the FD Hessian at the TS and return ``(mode_cart, masses_amu)``.

    The transition vector is the lowest normal mode (the imaginary one at a
    genuine first-order saddle); a warning is emitted if it isn't imaginary.

    ``gradient_options`` is the caller's (possibly ``None``) object, forwarded
    to :func:`vibeqc.hessian.compute_hessian_fd`, which derives the
    gradient from the SCF options when it is ``None`` -- JK backend and ECP
    fields alike -- so the TS Hessian differentiates the same Hamiltonian
    as the IRC path (#576).
    """
    import warnings

    from .hessian import HessianFDOptions, compute_hessian_fd

    scf_opts = {
        "rhf": rhf_options,
        "uhf": uhf_options,
        "rks": rks_options,
        "uks": uks_options,
        "rohf": rohf_options,
    }[method]
    hres = compute_hessian_fd(
        ts, basis, method=method.upper(), scf_options=scf_opts,
        grid_options=grid_options, gradient_options=gradient_options,
        grid_level=grid_level,
        hessian_options=HessianFDOptions(step_bohr=hessian_step_bohr))
    if float(hres.frequencies_cm1[0]) > -1.0:
        warnings.warn(
            "run_irc: the lowest Hessian frequency at the start geometry is "
            f"{hres.frequencies_cm1[0]:.1f} cm^-1 (not clearly imaginary) -- "
            "is this really a transition state?", RuntimeWarning, stacklevel=3)
    masses = np.asarray(hres.masses_amu, dtype=float)
    mode_mw = np.asarray(hres.normal_modes[:, 0], dtype=float)
    # Un-mass-weight the (mass-weighted) eigenvector to a Cartesian direction.
    mode_cart = mode_mw / np.repeat(np.sqrt(masses), 3)
    return mode_cart, masses


def run_irc(
    transition_state: Molecule,
    basis: Optional[str] = None,
    *,
    method: str = "RKS",
    functional: Optional[str] = "pbe",
    transition_mode: Optional[np.ndarray] = None,
    direction: str = "both",
    step_size: float = 0.1,
    max_points: int = 100,
    conv_tol_grad: float = 1e-3,
    rhf_options: Any = None,
    uhf_options: Any = None,
    rks_options: Any = None,
    uks_options: Any = None,
    rohf_options: Any = None,
    roks_options: Any = None,
    gradient_options: Any = None,
    grid_options: Any = None,
    grid_level: str = "orca-defgrid3",
    dispersion_params: Any = None,
    mlip_options: Any = None,
    hessian_step_bohr: float = 0.005,
    fd_step_bohr: float = 1e-3,
) -> IRCResult:
    """Trace the intrinsic reaction coordinate from a transition state.

    Parameters
    ----------
    transition_state
        A :class:`Molecule` at (or very near) a first-order saddle point --
        e.g. the output of :func:`vibeqc.run_dimer` or the climbing image of
        :func:`vibeqc.run_neb`.
    basis
        Basis-set name. Required for the SCF methods; ignored for
        ``method="mace"``.
    method
        ``"RHF"`` / ``"UHF"`` / ``"ROHF"`` / ``"RKS"`` / ``"UKS"`` /
        ``"ROKS"`` or ``"MACE"``. ROKS requires ``transition_mode``.
    transition_mode
        ``(n_atoms, 3)`` Cartesian transition vector (the imaginary mode).
        ``None`` => compute it from an FD Hessian at ``transition_state``
        (SCF methods only; for ``method="mace"`` you must pass it -- e.g.
        a ``DimerResult.mode``).
    direction
        ``"both"`` (default), ``"forward"`` (``+mode``) or ``"reverse"``.
    step_size
        Mass-weighted steepest-descent step (bohr.amu^1/2).
    max_points, conv_tol_grad
        Cap on points per branch and the max-force convergence threshold
        (Ha/bohr) at which a branch is declared to have reached a minimum.

    Returns
    -------
    :class:`IRCResult`
    """
    if not isinstance(transition_state, Molecule):
        raise NotImplementedError(
            "run_irc: only molecular transition states are supported in this "
            "version (periodic IRC is a follow-up).")
    method_lower = method.lower()
    if method_lower not in (
        "rhf", "uhf", "rohf", "rks", "uks", "roks", "mace"
    ):
        raise ValueError(
            f"run_irc: unsupported method {method!r}. "
            "Use one of 'RHF', 'UHF', 'ROHF', 'RKS', 'UKS', 'ROKS', 'MACE'.")
    is_mace = method_lower == "mace"
    if not is_mace and basis is None:
        raise ValueError(
            f"run_irc: a basis set is required for method={method!r} "
            "(basis is only optional for method='mace').")
    if direction not in ("both", "forward", "reverse"):
        raise ValueError(
            "run_irc: direction must be 'both', 'forward', or 'reverse'.")
    if is_mace and transition_mode is None:
        # Checked before loading the model so the message is actionable even
        # without the MACE stack installed.
        raise ValueError(
            "run_irc: pass transition_mode for method='mace' (e.g. a "
            "DimerResult.mode) -- the FD-Hessian auto-mode is SCF-only.")
    if method_lower == "roks" and transition_mode is None:
        raise NotImplementedError(
            "run_irc: method='ROKS' requires an explicit transition_mode; "
            "the FD-Hessian auto-mode requires an analytic ROKS gradient, "
            "which is not yet available.")

    ts_positions = _positions_of(transition_state)
    if transition_mode is not None:
        try:
            raw_mode = np.asarray(transition_mode)
            if np.iscomplexobj(raw_mode):
                raise ValueError("complex transition vectors are not supported")
            mode_cart = np.asarray(raw_mode, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "run_irc: transition_mode must be a finite real numeric array "
                f"with shape {ts_positions.shape}."
            ) from exc
        if mode_cart.shape != ts_positions.shape:
            raise ValueError(
                "run_irc: transition_mode must match the transition-state "
                f"coordinate shape {ts_positions.shape}; got "
                f"{mode_cart.shape}."
            )
        if not np.all(np.isfinite(mode_cart)):
            raise ValueError(
                "run_irc: transition_mode must contain only finite values."
            )
        mode_scale = (
            float(np.max(np.abs(mode_cart))) if mode_cart.size else 0.0
        )
        if mode_scale == 0.0:
            raise ValueError(
                "run_irc: transition_mode must have a finite, nonzero norm."
            )
        scaled_mode = mode_cart / mode_scale
        mode_norm = float(np.linalg.norm(scaled_mode))
        if not np.isfinite(mode_norm) or mode_norm == 0.0:
            raise ValueError(
                "run_irc: transition_mode must have a finite, nonzero norm."
            )
        # The IRC only consumes the direction. Normalizing through the largest
        # component keeps finite extreme-scale inputs from overflowing or
        # underflowing before _irc_branch applies its ordinary unit-vector step.
        transition_mode = scaled_mode / mode_norm

    # Cartesian gradient evaluator (reuses run_neb's per-geometry evaluators
    # via the dimer's force builder; force = -gradient).
    force_fn, mace_citation = _build_force_fn(
        transition_state, basis, method_lower,
        is_periodic=False, is_mace=is_mace, functional=functional,
        rhf_options=rhf_options, uhf_options=uhf_options,
        rks_options=rks_options, uks_options=uks_options,
        rohf_options=rohf_options, roks_options=roks_options,
        gradient_options=gradient_options, grid_options=grid_options,
        dispersion_params=dispersion_params, mlip_options=mlip_options,
        kmesh=None, fd_step_bohr=fd_step_bohr, dft_plus_u=None,
        grid_level=grid_level)

    def grad_fn(positions: np.ndarray) -> "tuple[float, np.ndarray]":
        e, f = force_fn(positions)
        return e, -np.asarray(f, dtype=float)

    ts_energy, _ = grad_fn(ts_positions)

    # Transition vector + masses.
    if transition_mode is not None:
        mode_cart = np.asarray(transition_mode, dtype=float)
        from .hessian import _resolve_masses
        masses = _resolve_masses(transition_state, None)
    else:
        # SCF only (the mace+None case was rejected above).
        mode_cart, masses = _transition_mode_from_hessian(
            transition_state, basis, method_lower, functional=functional,
            rhf_options=rhf_options, uhf_options=uhf_options,
            rks_options=rks_options, uks_options=uks_options,
            rohf_options=rohf_options, gradient_options=gradient_options,
            grid_options=grid_options, hessian_step_bohr=hessian_step_bohr,
            grid_level=grid_level)

    def _branch(sign: float):
        pos, en, conv = _irc_branch(
            ts_positions, grad_fn, masses, mode_cart, sign,
            step_size=step_size, max_points=max_points,
            conv_tol_grad=conv_tol_grad)
        systems = [_rebuild_with_positions(transition_state, p) for p in pos]
        return systems, np.asarray(en, dtype=float), conv

    if direction in ("both", "forward"):
        fwd_sys, fwd_e, fwd_c = _branch(+1.0)
    else:
        fwd_sys, fwd_e, fwd_c = [], np.asarray([], float), True
    if direction in ("both", "reverse"):
        rev_sys, rev_e, rev_c = _branch(-1.0)
    else:
        rev_sys, rev_e, rev_c = [], np.asarray([], float), True

    return IRCResult(
        transition_state=transition_state, ts_energy=float(ts_energy),
        forward_path=fwd_sys, forward_energies=fwd_e,
        reverse_path=rev_sys, reverse_energies=rev_e,
        forward_converged=fwd_c, reverse_converged=rev_c,
        method=method, basis=basis,
        functional=(
            functional if method_lower in ("rks", "uks", "roks") else None
        ),
        mace_model_citation=(mace_citation or None),
    )


__all__ = ["IRCResult", "run_irc"]
