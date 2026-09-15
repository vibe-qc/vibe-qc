"""Small validation helpers for backend smearing gates."""

from __future__ import annotations


def reject_unsupported_smearing_temperature(
    options,
    driver: str,
    *,
    detail: str = "",
) -> float:
    """Return legacy ``smearing_temperature`` or reject unsupported use."""
    temperature = float(getattr(options, "smearing_temperature", 0.0))
    if temperature < 0.0:
        raise ValueError(f"{driver}: smearing_temperature must be >= 0")
    if temperature > 0.0:
        suffix = f" {detail}" if detail else ""
        raise NotImplementedError(
            f"{driver}: smearing_temperature > 0 is not implemented for "
            f"this backend yet.{suffix}"
        )
    return temperature
