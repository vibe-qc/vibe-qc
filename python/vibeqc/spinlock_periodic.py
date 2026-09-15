"""SPINLOCK two-phase (SPIN_SCHEDULE) orchestration for periodic SCF drivers.

CRYSTAL's ``SPINLOCK n nstep`` (the SPIN_SCHEDULE mode) converges a *locked*
spin state for the first ``nstep`` cycles, then restarts at the multiplicity
target from that density. The molecular path (cpp/src/uhf.cpp, uks.cpp) runs
two sequential SCFs for this; the periodic Python drivers mirror it here.

Phase 1 converges (for at most ``spinlock_iterations`` cycles) a cell whose
multiplicity is locked to ``spinlock_value + 1`` (so n_alpha - n_beta =
spinlock_value); phase 2 runs the user's target multiplicity restarting from
phase 1's converged per-spin density via the READ machinery
(``opts.read_density_{alpha,beta}``). The locked phase polarises (e.g.
high-spin) before the SCF relaxes to the target -- useful for hard
broken-symmetry magnetic cases that a single-phase SCF drives to the wrong
basin.

The options struct (``PeriodicRHFOptions`` / ``PeriodicKSOptions``) is a C++
binding that is not copyable, so the two phases mutate the *same* opts object
and restore it in a ``finally`` (no observable side effect on the caller's
object). Negative ``spinlock_value`` (lock a beta-majority state) is not
representable through the cell multiplicity (which fixes n_alpha >= n_beta);
v1 supports spinlock_value >= 0 and fails closed otherwise.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from ._vibeqc_core import (
    Atom,
    InitialGuess,
    PeriodicSystem,
    SpinlockMode,
)


def rebuild_system_with_multiplicity(
    system: PeriodicSystem, multiplicity: int
) -> PeriodicSystem:
    """A copy of ``system`` with a different spin multiplicity (same cell,
    atoms, and charge). Used to lock the spin state of the SPIN_SCHEDULE
    phase-1 SCF."""
    mol = system.unit_cell_molecule()
    atoms = [Atom(int(a.Z), list(a.xyz)) for a in mol.atoms]
    return PeriodicSystem(
        int(system.dim),
        np.asarray(system.lattice, dtype=float),
        atoms,
        charge=int(system.charge),
        multiplicity=int(multiplicity),
    )


def _g0_density(d) -> np.ndarray:
    """The g=0 cell density as a plain real ``(nbf, nbf)`` array, for seeding the
    phase-2 READ restart. Accepts a plain array (Γ Ewald / GDF driver results)
    or a ``LatticeMatrixSet`` (BIPOLE / multi-k results), from which the
    ``[0,0,0]`` block is taken -- it Bloch-sums to a broken-symmetry ``D(k)`` for
    the restart guess (a warm start, not an exact continuation)."""
    if hasattr(d, "blocks") and hasattr(d, "cells"):
        for i, cell in enumerate(d.cells):
            if tuple(int(x) for x in cell.index) == (0, 0, 0):
                return np.real(np.asarray(d.blocks[i])).astype(float)
        return np.real(np.asarray(d.blocks[0])).astype(float)
    return np.real(np.asarray(d)).astype(float)


def run_spin_schedule(
    phase_runner: Callable,
    system: PeriodicSystem,
    opts,
):
    """Two-phase SPIN_SCHEDULE SCF: lock n_alpha-n_beta = spinlock_value for
    spinlock_iterations cycles, then release to the target multiplicity,
    restarting from the locked density.

    ``phase_runner(system, opts) -> result`` runs one single-phase SCF on the
    given system + (mutated) options. It is a thin closure over the concrete
    periodic open-shell driver that binds the driver's own ``basis`` / ``kmesh``
    / keyword arguments, so this helper is agnostic to the driver's call
    signature (the Γ Ewald drivers take ``(system, basis, opts)``; BIPOLE takes
    ``(system, basis, kmesh, opts)``; GDF takes ``(system, basis, opts, **kw)``).
    ``opts.spinlock_mode`` is forced to ``OFF`` for each phase, so the driver's
    own SPIN_SCHEDULE delegation does not recurse. The driver result must expose
    ``density_alpha`` / ``density_beta`` (used to seed the phase-2 READ restart).
    """
    n_elec = int(system.n_electrons())
    spinlock_value = int(getattr(opts, "spinlock_value", 0))
    spinlock_iterations = int(getattr(opts, "spinlock_iterations", 0))
    if spinlock_value < 0:
        raise NotImplementedError(
            "periodic SPINLOCK SPIN_SCHEDULE supports spinlock_value >= 0 "
            "(lock a high-spin state via the cell multiplicity); a "
            "beta-majority lock is not representable through the cell "
            "multiplicity. See docs/roadmap.md Sec.G2."
        )
    if spinlock_value > n_elec or ((n_elec + spinlock_value) % 2) != 0:
        raise ValueError(
            f"periodic SPINLOCK: spinlock_value={spinlock_value} is "
            f"incompatible with {n_elec} electrons (need 0 <= value <= n_elec "
            "and matching parity)."
        )

    mult_lock = spinlock_value + 1
    system_lock = rebuild_system_with_multiplicity(system, mult_lock)

    saved_mode = opts.spinlock_mode
    saved_max_iter = int(opts.max_iter)
    saved_guess = opts.initial_guess
    saved_atomic_spins = list(opts.atomic_spins)
    saved_rda = np.asarray(opts.read_density_alpha, dtype=float).copy()
    saved_rdb = np.asarray(opts.read_density_beta, dtype=float).copy()
    try:
        # Phase 1 -- locked spin, OFF, capped at spinlock_iterations cycles.
        opts.spinlock_mode = SpinlockMode.OFF
        opts.max_iter = max(1, spinlock_iterations)
        locked = phase_runner(system_lock, opts)

        # Phase 2 -- release to the target multiplicity, restarting from the
        # locked per-spin density via READ.
        opts.max_iter = saved_max_iter
        opts.initial_guess = InitialGuess.READ
        opts.atomic_spins = []  # Construction tags have already seeded phase 1.
        opts.read_density_alpha = _g0_density(locked.density_alpha)
        opts.read_density_beta = _g0_density(locked.density_beta)
        released = phase_runner(system, opts)
        from .guess import GuessSelection, resolve_initial_guess
        released.guess_selection = GuessSelection(
            saved_guess,
            resolve_initial_guess(system.unit_cell_molecule(), saved_guess,
                                  is_periodic=True, is_open_shell=True),
            InitialGuess.READ,
        )
        return released
    finally:
        opts.spinlock_mode = saved_mode
        opts.max_iter = saved_max_iter
        opts.initial_guess = saved_guess
        opts.atomic_spins = saved_atomic_spins
        opts.read_density_alpha = saved_rda
        opts.read_density_beta = saved_rdb


def check_spinlock_support(opts, supported, driver_name: str) -> None:
    """Fail closed when a SPINLOCK mode is requested on a driver that does not
    implement it (rather than silently ignoring the request). ``supported`` is
    the set of ``SpinlockMode`` values the driver handles (OFF is always ok)."""
    mode = getattr(opts, "spinlock_mode", SpinlockMode.OFF)
    if mode == SpinlockMode.OFF or mode in supported:
        return
    names = ", ".join(sorted(m.name for m in supported)) or "none"
    raise NotImplementedError(
        f"SPINLOCK {mode.name} is not implemented for {driver_name} "
        f"(supported here: {names}). PATTERN_HOLD protects an ATOMSPIN seed on "
        "the open-shell Ewald (Γ + multi-k UKS), GDF and BIPOLE drivers; "
        "SPIN_SCHEDULE (two-phase) is on the Γ-Ewald, Γ-GDF and "
        "Γ-BIPOLE drivers. "
        "See docs/roadmap.md Sec.G2."
    )


__all__ = [
    "rebuild_system_with_multiplicity",
    "run_spin_schedule",
    "check_spinlock_support",
]
