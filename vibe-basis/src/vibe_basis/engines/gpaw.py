"""GPAW energy engine — the plane-wave fallback. **Declared, not built.**

GPAW is today's plane-wave-limit oracle: 28 systems at r2SCAN with
converged cutoff, k-mesh and box, and with the free-atom convention
problem solved (aspherical, spin-polarised, Hund where magmom > 0, plus
damped-mixer branches for K / Ca / Br). That work is validated and
recorded in ``handovers/HANDOVER_GPAW_PW_REFERENCE.md`` and
``studies/pw_limit_atomization/CONSOLIDATED.md``.

Why this is a stub
------------------
The engine roster is declared at M-0 so that no downstream code is
written CRYSTAL-shaped and retrofitted later. The *implementation* is
M5's, where the cross-validation ladder actually needs it.

Deferring it is the point, not an omission. The validated GPAW worker
lives at ``examples/regression/core/runner_gpaw.py`` and
``examples/regression/pw_limit_atomization/`` in the vibe-qc repository,
and it belongs to the **vibeqc-gpaw** chat, not to this one. Two things
follow:

* Re-implementing PW-limit atomization here would fork a validated
  oracle -- the exact duplication that produced the K / Ca / Br
  convergence work in the first place.
* vibe-basis cannot import it: tier 1 may not depend on vibe-qc
  (``tests/test_no_vibeqc_dependency.py``), and that worker sits in the
  vibe-qc tree.

So M5 wires this class to the existing worker across a subprocess
boundary -- the same out-of-process discipline the worker itself already
uses for GPAW (CLAUDE.md § 10) -- rather than either importing it or
rewriting it. Until then, constructing this engine is allowed (so a
roster or a config can name it) and *calling* it is a loud error.

See ``vibe-basis/ROADMAP.md`` § 4 (M5) and § 5 (BOC-4, BOC-7).
"""

from __future__ import annotations

from typing import Any, Optional

from ..engine import EnergyEngine, EngineEnergy

_NOT_BUILT = (
    "GpawEngine is declared but not implemented (vibe-basis M-0 declares "
    "the engine roster; M5 implements this one). The validated GPAW "
    "PW-limit worker lives in the vibe-qc repository at "
    "examples/regression/pw_limit_atomization/ and is owned by the "
    "vibeqc-gpaw chat; M5 drives it out-of-process rather than forking "
    "or importing it. Use Crystal23Engine or VibeQcEngine today."
)


class GpawEngine(EnergyEngine):
    """Plane-wave energy engine backed by external GPAW. Not yet built.

    Constructing this is fine -- a campaign config may legitimately name
    the engine it intends to use before that engine exists. Evaluating
    raises :class:`NotImplementedError`.

    Note this is a genuine programming error rather than an evaluation
    failure, so it raises instead of returning ``ok=False``. The
    ``ok=False`` contract covers *infeasible points* -- a collapsed
    exponent, a dropped queue job -- which an optimizer routes around.
    Returning ``ok=False`` here would let a campaign silently score
    every plane-wave point as infeasible and "converge" on nothing.
    """

    name: str = "gpaw"

    def version(self) -> Optional[str]:
        return None

    def crystal_energy(
        self,
        basis_text: str,
        structure: Any,
        method: str = "rhf",
    ) -> EngineEnergy:
        raise NotImplementedError(_NOT_BUILT)

    def atom_energy(
        self,
        basis_text: str,
        Z: int,
        method: str = "rhf",
    ) -> EngineEnergy:
        raise NotImplementedError(_NOT_BUILT)
