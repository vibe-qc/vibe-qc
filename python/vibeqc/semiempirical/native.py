"""Python-owned adapter for the unified molecular native facade."""

from __future__ import annotations

from typing import Any

from .routes import BOUNDARY_MOLECULE, SemiempiricalRoutePlan


def native_route_descriptor(
    plan: SemiempiricalRoutePlan,
    *,
    max_iter: int | None = None,
    conv_tol: float | None = None,
    charge_mixing: float | None = None,
) -> Any:
    """Translate a validated molecular energy plan to the native descriptor."""
    if plan.boundary != BOUNDARY_MOLECULE:
        raise NotImplementedError(
            "the unified native facade currently accepts molecular routes only."
        )
    if plan.properties != ("energy",):
        raise NotImplementedError(
            "the first unified native facade slice accepts energy plans only."
        )
    if plan.variant == "cosmo":
        raise NotImplementedError(
            "MSINDO COSMO remains Python-orchestrated and does not traverse "
            "the molecular native facade."
        )
    if plan.variant == "om1":
        import warnings

        from vibeqc.semiempirical import NDDOExperimentalWarning

        warnings.warn(
            "OM1's analytic core-valence ECP (Kolb & Thiel 1993) is not "
            "implemented; heavy-atom bonds can be too short and close "
            "contacts can variationally collapse. Prefer OM2/OM3 for "
            "molecular prescreening.",
            category=NDDOExperimentalWarning,
            stacklevel=2,
        )

    from vibeqc._vibeqc_core import semiempirical as native

    method_names = {
        "dftb0": "DFTB0",
        "scc_dftb": "SCCDFTB",
        "gfn2_xtb": "GFN2XTB",
        "pm6": "PM6",
        "upm6": "PM6",
        "om1": "OM1",
        "om2": "OM2",
        "om3": "OM3",
        "indo": "MSINDO",
        "nddo": "MSINDONDDO",
    }
    try:
        method_name = method_names[plan.variant]
    except KeyError as exc:
        raise NotImplementedError(
            f"no unified native facade adapter exists for variant={plan.variant!r}."
        ) from exc

    descriptor = native.NativeRoute()
    descriptor.method = getattr(native.NativeMethod, method_name)
    descriptor.spin = (
        native.NativeSpin.Unrestricted
        if plan.spin == "unrestricted"
        else native.NativeSpin.ClosedShell
    )
    if max_iter is not None:
        descriptor.max_iter = int(max_iter)
    if conv_tol is not None:
        descriptor.conv_tol = float(conv_tol)
    if charge_mixing is not None:
        descriptor.charge_mixing = float(charge_mixing)
    return descriptor


def run_native_energy(
    plan: SemiempiricalRoutePlan,
    molecule: Any,
    parameters: Any,
    *,
    max_iter: int | None = None,
    conv_tol: float | None = None,
    charge_mixing: float | None = None,
) -> Any:
    """Run one planned molecular energy through the common native facade."""
    from vibeqc._vibeqc_core import semiempirical as native

    descriptor = native_route_descriptor(
        plan,
        max_iter=max_iter,
        conv_tol=conv_tol,
        charge_mixing=charge_mixing,
    )
    return native.run_native(descriptor, molecule, parameters)


__all__ = ["native_route_descriptor", "run_native_energy"]
