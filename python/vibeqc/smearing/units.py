"""Smearing-temperature unit conversions.

The canonical electronic temperature stored on the SCF options
surface is ``k_B T`` in **Hartree**. These helpers exist so user
code can stay readable (Kelvin, eV, Ry) and the resolver in
``vibeqc.smearing.resolution`` can parse strings like ``"1000 K"``
or ``"0.1 eV"`` without spreading unit-conversion logic.
"""

from __future__ import annotations

from ._constants import (
    EV_PER_HARTREE,
    HARTREE_PER_RYDBERG,
    KB_HARTREE_PER_K,
)


def kelvin_to_hartree_temperature(temperature_kelvin: float) -> float:
    """Return ``k_B T`` in Hartree for an electronic temperature in K."""
    temperature = float(temperature_kelvin)
    if temperature < 0.0:
        raise ValueError(
            "kelvin_to_hartree_temperature: temperature must be >= 0 K"
        )
    return temperature * KB_HARTREE_PER_K


def electronvolt_to_hartree_temperature(width_ev: float) -> float:
    """Return a smearing width in Hartree from an eV ``k_B T`` value."""
    width = float(width_ev)
    if width < 0.0:
        raise ValueError(
            "electronvolt_to_hartree_temperature: width must be >= 0 eV"
        )
    return width / EV_PER_HARTREE


def rydberg_to_hartree_temperature(width_ry: float) -> float:
    """Return a smearing width in Hartree from a Rydberg ``k_B T`` value."""
    width = float(width_ry)
    if width < 0.0:
        raise ValueError(
            "rydberg_to_hartree_temperature: width must be >= 0 Ry"
        )
    return width * HARTREE_PER_RYDBERG


def hartree_to_kelvin_temperature(kbt_hartree: float) -> float:
    """Return the Kelvin temperature corresponding to ``k_B T`` in Ha."""
    kbt = float(kbt_hartree)
    if kbt < 0.0:
        raise ValueError(
            "hartree_to_kelvin_temperature: k_B T must be >= 0 Ha"
        )
    return kbt / KB_HARTREE_PER_K


def temperature_in_hartree(value: float, unit: str) -> float:
    """Convert a numeric width in ``unit`` to Hartree ``k_B T``.

    Accepted units: ``hartree`` / ``ha`` / ``au``; ``kelvin`` / ``k``;
    ``ev`` / ``electronvolt``; ``rydberg`` / ``ry``. Unknown units
    raise ``ValueError`` rather than silently mis-interpreting input.
    """
    u = str(unit).strip().lower().replace("_", "-").replace(" ", "-")
    if u in ("ha", "hartree", "hartrees", "au", "a.u."):
        temperature = float(value)
        if temperature < 0.0:
            raise ValueError("smearing_temperature must be >= 0 Ha")
        return temperature
    if u in ("k", "kelvin"):
        return kelvin_to_hartree_temperature(float(value))
    if u in ("ev", "electronvolt", "electronvolts", "electron-volt",
             "electron-volts"):
        return electronvolt_to_hartree_temperature(float(value))
    if u in ("ry", "rydberg", "rydbergs"):
        return rydberg_to_hartree_temperature(float(value))
    raise ValueError(
        "unknown smearing temperature unit "
        f"{unit!r}; expected hartree, kelvin, eV, or Ry"
    )
