"""Iter-indexed Saunders-Hillier level-shift schedule.

CRYSTAL14's ``LEVSHIFT B IRESET`` is a per-iter shift schedule --
start with a large shift (e.g. ``B=5`` -> 0.5 Ha) at iter 1, decrease as the
SCF settles. vibe-qc's existing ``opts.level_shift`` is a single
constant; this helper is the ergonomic front-end for the per-iter
schedule that supersedes it when supplied.

The schedule lowers to the C++ ``std::vector<double>`` field
``options.level_shift_schedule`` (available on every molecular and
periodic options struct) via :meth:`as_list` / :meth:`apply_to`. The
single source of truth for *how much* shift applies at a given SCF
iteration is the C++ ``level_shift_at_iter`` helper
(``cpp/include/vibeqc/level_shift.hpp``, exposed to Python as
``vibeqc.level_shift_at_iter``); this dataclass just builds the curve
it consumes. Both the molecular C++ drivers and the periodic Python
drivers resolve the per-iteration shift through that one helper, so the
schedule behaves identically everywhere.

Use case: tight ionic crystals (LiH, MgO, NaCl, ...) where the
Hcore initial guess is far from the converged density and the
first Fock build demands a large orbital rotation. Without
iter-decreasing LEVSHIFT the multi-k SCF can stably converge to
the wrong basin (see
`examples/regression/crystal_parity/diag_lih_multik_iter3_instrumented.py`
on this branch -- iter 3 over-binds to E_total = -228 Ha on LiH
primitive at kmesh=(2,2,2) with a constant shift).

CRYSTAL14 reference: Dovesi, Pisani, Roetti, Saunders,
*Phys. Rev. B* **28**, 5781 (1983); Pisani, Dovesi, Roetti,
*Hartree-Fock Ab Initio Treatment of Crystalline Systems*,
Lecture Notes in Chemistry vol 48 (Springer, 1988), Sec.3.7
("Convergence accelerators").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class LevelShiftSchedule:
    """Per-iter Saunders-Hillier shift schedule.

    ``shifts[i]`` is the shift in Hartree applied at SCF iteration
    ``i + 1`` (1-indexed iters; CRYSTAL convention). For iterations
    past ``len(shifts)``, the last entry is reused -- set the last
    entry to 0.0 to fully release the shift once the SCF is in the
    basin (the standard convention).
    """
    shifts: Sequence[float]

    def __post_init__(self) -> None:
        if len(self.shifts) == 0:
            raise ValueError(
                "LevelShiftSchedule: shifts must be non-empty"
            )
        for i, s in enumerate(self.shifts):
            if s < 0:
                raise ValueError(
                    "LevelShiftSchedule: shifts must be non-negative; "
                    f"got shifts[{i}] = {s}"
                )

    def at(self, iter_idx: int) -> float:
        """Return the shift for SCF iteration ``iter_idx`` (1-indexed).

        Iterations past the schedule length clamp to the last entry.
        """
        if iter_idx < 1:
            raise ValueError(
                f"LevelShiftSchedule.at: iter_idx must be >= 1; "
                f"got {iter_idx}"
            )
        idx = min(iter_idx - 1, len(self.shifts) - 1)
        return float(self.shifts[idx])

    @classmethod
    def crystal_default(cls) -> "LevelShiftSchedule":
        """CRYSTAL14 ``LEVSHIFT 5 1`` analogue: start at 0.5 Ha,
        decrease step-by-step to 0 over seven iters.

        Recommended for tight ionic crystals (LiH, MgO, NaCl, LiF)
        where the Hcore initial guess overshoots into the over-bound
        basin without iter-1 stabilisation.
        """
        return cls([0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.0])

    @classmethod
    def crystal_aggressive(cls) -> "LevelShiftSchedule":
        """Heavier opening shift for very hard ionic SCFs
        (e.g. NaCl, LiF -- wide-gap insulators with very deep core
        states): CRYSTAL ``LEVSHIFT 10`` / 1.0 Ha first iter,
        slower release."""
        return cls([1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05, 0.0])

    @classmethod
    def constant(cls, shift: float, n_iters: int = 1) -> "LevelShiftSchedule":
        """Schedule that holds ``shift`` Ha for all iters -- equivalent
        to the legacy ``opts.level_shift`` constant-shift convention."""
        return cls([float(shift)] * max(1, int(n_iters)))

    def as_list(self) -> List[float]:
        return [float(s) for s in self.shifts]

    def apply_to(self, options):
        """Lower this schedule onto ``options.level_shift_schedule``.

        Ergonomic bridge from the dataclass to the C++
        ``std::vector<double>`` field consumed by every molecular and
        periodic SCF driver. Returns ``options`` for chaining::

            opts = LevelShiftSchedule.crystal_default().apply_to(vq.RHFOptions())
            vq.run_rhf(mol, basis, opts)

        The non-empty vector supersedes ``options.level_shift`` and
        ``options.level_shift_warmup_cycles`` (an explicit schedule takes
        precedence in ``level_shift_at_iter``).
        """
        options.level_shift_schedule = self.as_list()
        return options
