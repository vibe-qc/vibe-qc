"""Convergence policy -- composable geometry-optimisation stopping criteria.

Every optimiser checks the same criteria via the same
:class:`ConvergencePolicy` object.  Criteria are AND-ed: all active
gates must pass for the run to be declared converged.

Built-in gates
--------------
* ``gmax`` -- max absolute gradient component (inf-norm), Ha/bohr
* ``grms`` -- root-mean-square gradient, Ha/bohr
* ``dmax`` -- max atomic displacement between consecutive steps, bohr
* ``drms`` -- RMS displacement between consecutive steps, bohr
* ``ediff`` -- absolute energy change between consecutive steps, Ha

The Phase 1 default is ``gmax`` only (mirrors the existing
``_gradient_converged`` convention shared by molecular and BIPOLE
optimisers).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..output.document import CriteriaTable, Quantity, active_policy


_GATE_QUANTITY_KINDS = {
    "gmax": "gradient",
    "grms": "gradient",
    "dmax": "length",
    "drms": "length",
    "ediff": "energy_delta",
}


@dataclass
class ConvergencePolicy:
    """AND-ed convergence gates for geometry optimisation.

    All non-``None`` values are active.  Set a gate to ``None`` to
    disable it.

    Parameters
    ----------
    gmax : float or None
        Max absolute gradient component tolerance (Ha/bohr).
        Default 4.5e-4 (about 0.023 eV/A).
    grms : float or None
        RMS gradient tolerance.  Disabled by default (set when the
        system is large or the PES is shallow).
    dmax : float or None
        Max atomic displacement tolerance (bohr).  Disabled by default.
    drms : float or None
        RMS displacement tolerance.  Disabled by default.
    ediff : float or None
        Absolute energy change tolerance (Ha).  Disabled by default.
    """

    gmax: Optional[float] = 4.5e-4
    grms: Optional[float] = None
    dmax: Optional[float] = None
    drms: Optional[float] = None
    ediff: Optional[float] = None

    @classmethod
    def tight(cls) -> "ConvergencePolicy":
        """Preset for high-accuracy work (e.g. frequencies)."""
        return cls(gmax=1e-5, grms=1e-5, dmax=1e-4, drms=1e-4, ediff=1e-8)

    @classmethod
    def loose(cls) -> "ConvergencePolicy":
        """Preset for rough pre-optimisation or scan steps."""
        return cls(gmax=1e-2)

    @classmethod
    def default(cls) -> "ConvergencePolicy":
        """The vibe-qc standard: max-component gradient only."""
        return cls(gmax=4.5e-4)

    def check(
        self,
        gradient: Optional[np.ndarray] = None,
        x_old: Optional[np.ndarray] = None,
        x_new: Optional[np.ndarray] = None,
        e_old: Optional[float] = None,
        e_new: Optional[float] = None,
    ) -> "ConvergenceReport":
        """Evaluate all active gates.

        Returns a :class:`ConvergenceReport` with per-gate results.
        """
        passed = True
        details: dict[str, bool] = {}
        values: dict[str, float] = {}
        thresholds = {
            name: float(threshold)
            for name, threshold in (
                ("gmax", self.gmax),
                ("grms", self.grms),
                ("dmax", self.dmax),
                ("drms", self.drms),
                ("ediff", self.ediff),
            )
            if threshold is not None
        }

        # --- gradient gates -------------------------------------------------
        if self.gmax is not None:
            if gradient is not None and gradient.size > 0:
                v = float(np.max(np.abs(gradient)))
            else:
                v = float("inf")
            values["gmax"] = v
            ok = v <= self.gmax
            details["gmax"] = ok
            passed = passed and ok

        if self.grms is not None:
            if gradient is not None and gradient.size > 0:
                v = float(np.sqrt(np.mean(np.square(gradient))))
            else:
                v = float("inf")
            values["grms"] = v
            ok = v <= self.grms
            details["grms"] = ok
            passed = passed and ok

        # --- displacement gates ---------------------------------------------
        if self.dmax is not None:
            if x_old is not None and x_new is not None and len(x_old) == len(x_new):
                g = np.asarray(x_new, dtype=float) - np.asarray(x_old, dtype=float)
                v = (
                    float(np.max(np.abs(g.reshape(-1, 3))))
                    if g.size >= 3
                    else float(np.max(np.abs(g)))
                )
            else:
                v = float("inf")
            values["dmax"] = v
            ok = v <= self.dmax
            details["dmax"] = ok
            passed = passed and ok

        if self.drms is not None:
            if x_old is not None and x_new is not None and len(x_old) == len(x_new):
                g = np.asarray(x_new, dtype=float) - np.asarray(x_old, dtype=float)
                v = float(np.sqrt(np.mean(np.square(g))))
            else:
                v = float("inf")
            values["drms"] = v
            ok = v <= self.drms
            details["drms"] = ok
            passed = passed and ok

        # --- energy gate ----------------------------------------------------
        if self.ediff is not None:
            if e_old is not None and e_new is not None:
                v = float(abs(e_new - e_old))
            else:
                v = float("inf")
            values["ediff"] = v
            ok = v <= self.ediff
            details["ediff"] = ok
            passed = passed and ok

        return ConvergenceReport(
            passed=passed,
            details=details,
            values=values,
            thresholds=thresholds,
        )


@dataclass
class ConvergenceReport:
    """Result of a :meth:`ConvergencePolicy.check` call.

    Attributes
    ----------
    passed : bool
        ``True`` iff all active gates passed.
    details : dict[str, bool]
        Per-gate pass/fail.
    values : dict[str, float]
        Per-gate measured value (``inf`` when not computable).
    thresholds : dict[str, float]
        Per-gate upper threshold. Empty only for a legacy caller-constructed
        report that did not retain its policy.
    """

    passed: bool
    details: dict[str, bool]
    values: dict[str, float]
    thresholds: dict[str, float] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.passed

    def criteria_table(self) -> CriteriaTable:
        """Return the semantic value/threshold/verdict representation."""
        table = CriteriaTable(indent=0)
        for gate, ok in self.details.items():
            kind = _GATE_QUANTITY_KINDS.get(gate, "dimensionless")
            threshold = self.thresholds.get(gate)
            table.add(
                gate,
                Quantity(self.values.get(gate, float("nan")), kind),
                None if threshold is None else Quantity(threshold, kind),
                ok,
            )
        return table

    def summary(self) -> str:
        """Render the compact form used by the optimizer progress table."""
        return self.criteria_table().render_inline(active_policy())
