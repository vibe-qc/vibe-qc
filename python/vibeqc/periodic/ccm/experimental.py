"""Experimental-status marker for the Γ-CCM (``aiccm2026dev-a``) line.

Every public Γ-CCM SCF driver fires :func:`_warn_experimental` once per call
site, mirroring the ``AICCM2026DevBExperimentalWarning`` convention of the
sibling χ-CCM line (``periodic/chi/scf.py``) and the GAPW / embedded-
surface experimental warnings. Correlation, gradient, and property drivers
inherit the warning transitively through the SCF they run (or were handed).

Filter via::

    warnings.filterwarnings(
        "ignore", category=vibeqc.AICCM2026DevAExperimentalWarning
    )

The catalog of experimental features (and this line's caveats) is
``docs/experimental/catalog.md``; the Γ-CCM reference page is
``docs/aiccm2026dev_a.md``.
"""

from __future__ import annotations

import warnings

__all__ = ["AICCM2026DevAExperimentalWarning"]


class AICCM2026DevAExperimentalWarning(UserWarning):
    """The Γ-CCM (aiccm2026dev-a) line is experimental research code."""


def _warn_experimental() -> None:
    warnings.warn(
        "aiccm2026dev-a (Γ-CCM) is experimental: research-grade "
        "union-and-weight/Wigner-Seitz torus energies; check the documented "
        "parity gates and invariants before using an energy quantitatively",
        category=AICCM2026DevAExperimentalWarning,
        stacklevel=3,
    )


def _reject_open_shell_cluster(ccm, *, who, sibling):
    """Refuse a cluster whose spin state is not closed-shell.

    The closed-shell Γ-CCM drivers historically checked electron **parity**
    only, so an even-electron cluster with ``multiplicity > 1`` (a triplet)
    was silently solved as a closed shell and returned a wrong number --
    measured 2026-08-23 on H2/sto-3g (2 e-, mult 3, 8-bohr box):
    ``run_ccm_rhf`` -1.0491709020 and ``run_ccm_rks(pbe)`` -1.1023577337,
    where the shell-aware ``run_ccm_scf`` dispatches to UHF and gives
    -0.7749672190. The route dispatcher grew this guard in 2026-07-15/16;
    the per-method entry points did not, and this closes that gap.
    """
    n_elec = int(ccm.supercell.n_electrons())
    mult = int(ccm.supercell.multiplicity)
    if n_elec % 2 != 0:
        raise ValueError(
            f"{who} is closed-shell but the cluster has {n_elec} electrons; "
            f"use {sibling} for open shells."
        )
    if mult != 1:
        raise ValueError(
            f"{who} is closed-shell but the cluster declares multiplicity "
            f"{mult} ({n_elec} electrons). An even-electron cluster can "
            f"still be open-shell -- solving it closed-shell would silently "
            f"return the wrong state. Use {sibling}, or "
            f"vibeqc.periodic.ccm.run_ccm_scf, which dispatches by shell."
        )


def _reject_double_hybrid(functional, *, who):
    """Refuse a double-hybrid functional on an SCF-only KS driver.

    A double hybrid is its hybrid-KS SCF **plus** a scaled-MP2 correction.
    A driver that runs only the SCF half and returns its energy under the
    double hybrid's name reports a number that is not that functional --
    measured 2026-08-23: ``run_ccm_rks(ccm, "b2plyp")`` returned
    -1.1446213910 on compact H2/sto-3g, exactly the hybrid-SCF half of the
    true -1.1490167438. Fail closed instead (the real-Γ route's
    ``run_ccm_rks_direct`` has always done so).
    """
    if functional is None:
        return
    from vibeqc._vibeqc_core import Functional

    if bool(getattr(Functional(str(functional), 1), "is_double_hybrid", False)):
        raise NotImplementedError(
            f"{who}: {functional!r} is a double hybrid -- its energy is the "
            f"hybrid-KS SCF plus a scaled-MP2 correction, and this driver "
            f"computes only the SCF half. Returning it under the "
            f"double-hybrid name would be a wrong answer. Use a plain "
            f"global hybrid here, or "
            f"vibeqc.periodic.ccm.direct.run_ccm_double_hybrid_direct for "
            f"the composed double-hybrid energy on the real-Γ route."
        )
