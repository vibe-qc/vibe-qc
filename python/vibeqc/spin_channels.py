"""Value-based open-shell detection for SCF result objects.

``hasattr(result, "density_alpha")`` is *not* a reliable open-shell test.
Several runner-contract result types -- notably
:class:`~vibeqc.periodic.ccm.real_gamma_runner.CCMRealGammaResult`,
:class:`~vibeqc.periodic.ccm.four_center_runner.CCMFourCentreResult` and
:class:`~vibeqc.periodic.ccm.dft.CCMDFTResult` -- declare
``density_alpha`` / ``density_beta`` as dataclass *fields* that are ``None``
on a closed-shell run.  ``hasattr`` is then always true, so an open-shell
branch is taken for a closed-shell result and ``None + None`` raises
``TypeError``.  Because the population/property consumers wrap each section
in ``except Exception``, the failure surfaces as a plausible-looking
``N/A -- TypeError`` row rather than a crash, which is how it has hidden
(GitLab #679, and again in the real-Gamma population sidecar).

Ask for the *values* instead of the attributes.  This module has no
``vibeqc`` imports of its own, so it is safe to import from anywhere in the
package without creating a cycle.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

__all__ = ["spin_densities", "is_open_shell_result"]


def spin_densities(result: Any) -> Tuple[Optional[Any], Optional[Any]]:
    """Return ``(density_alpha, density_beta)`` for a genuinely open-shell result.

    Returns ``(None, None)`` when either channel is absent *or* present but
    ``None`` -- i.e. whenever the caller should fall back to the combined
    ``result.density``.  A half-populated pair is treated as closed-shell
    rather than raising: the combined density is still the right answer, and
    the consumers of this helper are diagnostics, not the SCF itself.
    """
    alpha = getattr(result, "density_alpha", None)
    beta = getattr(result, "density_beta", None)
    if alpha is None or beta is None:
        return None, None
    return alpha, beta


def is_open_shell_result(result: Any) -> bool:
    """True iff *result* carries both spin densities with non-``None`` values."""
    alpha, _ = spin_densities(result)
    return alpha is not None
