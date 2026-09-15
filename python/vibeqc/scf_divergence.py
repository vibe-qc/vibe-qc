"""SCF divergence-detection heuristic shared across periodic SCF drivers.

A single source of truth for the "is this SCF run going off the rails"
test that every periodic Python Ewald driver runs at every iteration.
Originally landed in v0.5.6 as inline code inside
``run_rks_periodic_gamma_ewald3d``; v0.6.2 lifts it into a helper so
the same heuristic applies uniformly across all 8 periodic SCF entry
points (Γ-only + multi-k for RHF / UHF / RKS / UKS).

Why: ionic insulators with deep core states (NaCl / MgO with STO-3G --
Na 1s e ≈ -40 Ha, Cl 1s e ≈ -100 Ha) make the Hcore initial guess so
far from physical that DIIS amplifies the swing into +30k / -16k Ha
territory before recovering. The user's wall-clock cost of running
those iterations to ``max_iter`` and then crashing some downstream
property calc with an index error / NaN is much worse than an early
RuntimeError with a remediation hint.

Heuristics (each fires its own RuntimeError):
  - NaN / Inf in E or grad:  arithmetic blowup, unrecoverable.
  - |E| > 1e6 Ha:              unphysical magnitude, density is broken.
  - |dE| > 1e4 Ha after iter 5: still oscillating wildly.

The hint string in each error is identical so users see the same
remediation suggestion regardless of which driver they hit. The hint
points at SAD guess + stronger damping + tighter Ewald -- the three
levers that actually move the needle on Hcore-bombing systems.
"""

from __future__ import annotations

import math
from typing import Optional


__all__ = ["check_scf_divergence", "REMEDIATION_HINT"]


REMEDIATION_HINT = (
    "Try a SAD initial guess (opts.initial_guess = vq.InitialGuess.SAD) "
    "for ionic / covalent insulators, stronger damping "
    "(opts.damping = 0.85+), or a tighter Ewald (omega = 0.3). "
    "If the system has deep core states (Na, Mg, Cl, transition metals) "
    "Hcore is the wrong starting point -- see docs/roadmap.md "
    "'SCF guess + convergence program'."
)


def check_scf_divergence(
    driver_name: str,
    iter_idx: int,
    energy: float,
    grad_norm: float,
    delta_e: float,
    *,
    energy_blowup_threshold: float = 1.0e6,
    oscillation_threshold: float = 1.0e4,
    oscillation_min_iter: int = 5,
) -> None:
    """Raise ``RuntimeError`` with a remediation hint if the SCF
    iteration looks divergent. Otherwise return ``None``.

    Parameters
    ----------
    driver_name
        Stamped into the error message so users see which driver
        bailed (e.g. ``"run_rks_periodic_gamma_ewald3d"``).
    iter_idx, energy, grad_norm, delta_e
        Current iteration's quantities.
    energy_blowup_threshold
        |E| > this value (Ha) is unphysical for any periodic SCF on
        an integer-charge cell. Default 1e6 Ha.
    oscillation_threshold
        |dE| > this value (Ha) after ``oscillation_min_iter``
        iterations is "still oscillating wildly". Default 1e4 Ha.
    oscillation_min_iter
        Skip the |dE| check before this iteration so the very first
        SCF step (which can have a huge dE if the guess is far from
        the converged density) doesn't trigger.

    Each heuristic raises its own ``RuntimeError`` with a
    driver-stamped message + the shared :data:`REMEDIATION_HINT`.
    """
    if not (math.isfinite(energy) and math.isfinite(grad_norm)):
        raise RuntimeError(
            f"{driver_name}: SCF diverged at iter {iter_idx} "
            f"(E = {energy!r}, grad = {grad_norm!r}) -- arithmetic "
            f"blowup. {REMEDIATION_HINT}"
        )
    if abs(energy) > energy_blowup_threshold:
        raise RuntimeError(
            f"{driver_name}: SCF energy |{energy:.3e}| > "
            f"{energy_blowup_threshold:.0e} Ha at iter {iter_idx} -- "
            f"the Hcore initial guess is unphysical for this system. "
            f"{REMEDIATION_HINT}"
        )
    if iter_idx >= oscillation_min_iter and abs(delta_e) > oscillation_threshold:
        raise RuntimeError(
            f"{driver_name}: SCF oscillating at iter {iter_idx} with "
            f"|dE| = {abs(delta_e):.3e} Ha -- DIIS / damping not "
            f"recovering. {REMEDIATION_HINT}"
        )
