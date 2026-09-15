"""Which analytic nuclear-gradient terms the molecular RKS / UKS kernel lacks.

GitLab #571: ``cpp/src/gradient.cpp`` differentiates the exact-exchange
energy through the global fraction ``Functional.hf_exchange_fraction()``
only.  For a range-separated (CAM / RSH) hybrid that fraction is
``cam_alpha``; the erf-attenuated long-range arm ``cam_beta * K_erf(omega)``
that ``cpp/src/rks.cpp`` / ``uks.cpp`` build for the *energy* has no
gradient counterpart.  The VV10 nonlocal correlation that the SCF adds
self-consistently for ``needs_vv10`` functionals (Vydrov and Van Voorhis,
J. Chem. Phys. 133, 244103 (2010), doi:10.1063/1.3521275) has no gradient
counterpart either.  Measured on a bent O/H/H molecule, STO-3G, against
the optimizer's own full-energy central finite differences (step 0.005
bohr, OMP_NUM_THREADS=1): pbe 2.1e-6 Ha/bohr (noise floor), hse06 5.6e-3,
vv10 7.0e-4, wb97x-v 5.6e-2, with max |g| about 0.1 Ha/bohr.

Until those terms are implemented, every consumer of the analytic RKS /
UKS gradient asks this module first: the molecular optimizers fall back to
full-energy central finite differences, and the public
:func:`vibeqc.compute_gradient_rks` / :func:`vibeqc.compute_gradient_uks`
wrappers refuse unless the caller opts in with ``allow_incomplete=True``.
Nothing here adds gradient physics; the detection reads the predicates the
C++ :class:`vibeqc.Functional` already exposes (``is_range_separated``,
``needs_vv10``).
"""

from __future__ import annotations

from typing import Any

from ._vibeqc_core import Functional

__all__ = [
    "functional_gradient_terms_missing",
    "missing_gradient_terms_message",
    "require_complete_analytic_gradient",
    "resolve_functional_object",
]

#: Reference for the VV10 kernel whose nuclear derivative is missing.
VV10_GRADIENT_REFERENCE = (
    "Vydrov and Van Voorhis, J. Chem. Phys. 133, 244103 (2010), "
    "doi:10.1063/1.3521275"
)


def resolve_functional_object(
    functional: Any, *, spin: int = 1
) -> Functional | None:
    """Return the :class:`Functional` behind *functional*, or ``None``.

    Accepts a functional name, a :class:`Functional`, or any object with a
    string ``functional`` attribute (``RKSOptions``, ``RKSResult``, their
    UKS siblings).  ``None`` and the empty string resolve to ``None``.  A
    name libxc does not know also resolves to ``None``: the SCF that would
    consume it raises the descriptive error, and a capability query must
    not pre-empt that with a different one.
    """
    if functional is None:
        return None
    if isinstance(functional, Functional):
        return functional
    name = functional
    if not isinstance(name, str):
        name = getattr(functional, "functional", None)
    if name is None:
        return None
    name = str(name).strip()
    if not name:
        return None
    try:
        return Functional(name, int(spin))
    except (ValueError, RuntimeError):
        # Functional raises ValueError("unknown name ...") for a name libxc
        # does not know; RuntimeError covers a libxc construction failure.
        return None


def functional_gradient_terms_missing(
    functional: Any, *, spin: int = 1
) -> list[str]:
    """Analytic nuclear-gradient terms the RKS / UKS kernel omits for *functional*.

    Returns an empty list when the analytic gradient is complete (LDA, GGA,
    meta-GGA, global hybrids such as PBE0 / B3LYP, and ``None``).  Otherwise
    each entry names one missing term:

    * a range-separated hybrid (``Functional.is_range_separated``): the
      long-range exact-exchange gradient ``cam_beta * dK_erf(omega)/dR``;
    * a VV10-paired functional (``Functional.needs_vv10``): the VV10
      nonlocal correlation gradient.

    *functional* may be a name, a :class:`Functional`, or an options /
    result object carrying ``.functional``; *spin* selects the polarised
    (2) or unpolarised (1) libxc construction when a name is resolved.
    """
    func = resolve_functional_object(functional, spin=spin)
    if func is None:
        return []
    missing: list[str] = []
    if bool(getattr(func, "is_range_separated", False)):
        beta = float(getattr(func, "cam_beta", 0.0))
        omega = float(getattr(func, "rsh_omega", 0.0))
        missing.append(
            "the long-range exact-exchange gradient of the range-separated "
            f"hybrid (cam_beta*K_erf(omega) with beta={beta:.4f}, "
            f"omega={omega:.4f} bohr^-1; the kernel differentiates only the "
            "global fraction cam_alpha)"
        )
    if bool(getattr(func, "needs_vv10", False)):
        b = float(getattr(func, "vv10_b", 0.0))
        c = float(getattr(func, "vv10_C", 0.0))
        missing.append(
            f"the VV10 nonlocal correlation gradient (b={b:g}, C={c:g}; "
            f"{VV10_GRADIENT_REFERENCE})"
        )
    return missing


def missing_gradient_terms_message(
    functional_name: str, missing: list[str], *, route: str
) -> str:
    """One paragraph naming the omitted terms and the finite-difference route."""
    terms = "; ".join(f"({i + 1}) {t}" for i, t in enumerate(missing))
    return (
        f"{route}: the analytic nuclear gradient for functional "
        f"{functional_name!r} is incomplete. It omits {terms}. Measured on "
        "O/H/H STO-3G (GitLab #571): hse06 5.6e-3, vv10 7.0e-4, wb97x-v "
        "5.6e-2 Ha/bohr off the full-energy finite-difference gradient, "
        "against a 2e-6 noise floor for pbe. run_job(optimize=True), "
        "optimize_molecule, optimize_molecule_brent, the geomopt provider "
        "and the ASE calculator fall back to full-energy central finite "
        "differences automatically; for a standalone gradient use "
        "vibeqc.molecular_optimize._gradient_via_central_difference. Pass "
        "allow_incomplete=True to receive the partial analytic gradient "
        "knowingly (it is not the derivative of the reported energy)."
    )


def require_complete_analytic_gradient(
    functional: Any, *, route: str, spin: int = 1
) -> list[str]:
    """Raise :class:`NotImplementedError` when *functional* has missing terms.

    Returns the (empty) list of missing terms on success so callers can use
    it as a one-line guard.
    """
    missing = functional_gradient_terms_missing(functional, spin=spin)
    if missing:
        func = resolve_functional_object(functional, spin=spin)
        name = str(getattr(functional, "functional", functional))
        if isinstance(functional, Functional) or func is functional:
            name = str(getattr(func, "name", name))
        raise NotImplementedError(
            missing_gradient_terms_message(name, missing, route=route)
        )
    return missing
