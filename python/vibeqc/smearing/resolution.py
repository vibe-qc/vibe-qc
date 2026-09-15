"""User-input -> canonical smearing-temperature resolution.

Accepts numeric widths (with optional unit suffix in a string),
named presets, ``"auto"``, or ``None``/``"none"``/``"off"`` and
returns a :class:`SmearingResolution` carrying the canonical
Hartree ``k_B T`` plus provenance metadata that user-facing logs
can print so it's obvious how a temperature was chosen.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Union

from .units import temperature_in_hartree


@dataclass(frozen=True)
class SmearingResolution:
    """Canonical finite-temperature smearing choice.

    ``temperature`` is the electronic energy ``k_B T`` in Hartree,
    matching what the SCF options surface stores. ``method`` is the
    canonical flavour name (``"fermi-dirac"``, ``"mermin"``,
    ``"methfessel-paxton"``, ``"marzari-vanderbilt"``); explicit
    metadata lets user-facing wrappers explain how an automatic or
    preset value was chosen.
    """

    temperature: float
    method: str = "fermi-dirac"
    source: str = "explicit"
    reason: str = ""


SMEARING_PRESETS = {
    "none": 0.0,
    "off": 0.0,
    "insulator": 0.0,
    "semiconductor": 0.0,
    "small-gap": 0.002,
    "semimetal": 0.002,
    "metal": 0.005,
    "metallic": 0.005,
    "debug": 0.010,
}


_NUMERIC_WITH_OPTIONAL_UNIT = re.compile(
    r"^\s*"
    r"([+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?)"
    r"(?:\s*([A-Za-z. _-]+))?"
    r"\s*$"
)


def resolve_smearing_temperature(
    value: Union[None, str, float] = 0.0,
    *,
    unit: str = "hartree",
    method: str = "fermi-dirac",
    metallic: Optional[bool] = None,
    band_gap_hartree: Optional[float] = None,
    n_electrons: Optional[float] = None,
) -> SmearingResolution:
    """Resolve user smearing input to canonical Hartree ``k_B T``.

    ``value`` may be a numeric width, a string with units such as
    ``"1000 K"`` or ``"0.1 eV"``, ``None``/``"none"``/``"off"``, a
    named preset (``"metal"``, ``"small-gap"``, ``"debug"``), or
    ``"auto"``.  ``method`` selects the smearing flavour
    (``"fermi-dirac"``, ``"mermin"``, ``"methfessel-paxton"``,
    ``"marzari-vanderbilt"``); the full flavour configuration lives on
    :class:`vibeqc.smearing.options.SmearingOptions`.
    """
    meth = str(method).strip().lower().replace("_", "-")
    # Normalise spelling aliases to canonical flavours.
    _METHOD_ALIASES = {"fermi": "fermi-dirac", "fd": "fermi-dirac",
                       "meremin": "mermin"}
    meth = _METHOD_ALIASES.get(meth, meth)
    if meth not in ("fermi-dirac", "mermin", "methfessel-paxton",
                    "marzari-vanderbilt"):
        raise NotImplementedError(
            "Smearing method not implemented; "
            f"got method={method!r}. Supported: fermi-dirac, mermin, "
            "methfessel-paxton, marzari-vanderbilt"
        )

    if value is None:
        return SmearingResolution(
            0.0, method=meth, source="explicit", reason="smearing disabled"
        )

    if isinstance(value, str):
        token = value.strip().lower().replace("_", "-")
        if token == "auto":
            # Local import to avoid a circular module load: auto.py
            # imports SmearingResolution from here.
            from .auto import guess_smearing_temperature
            return guess_smearing_temperature(
                metallic=metallic,
                band_gap_hartree=band_gap_hartree,
                n_electrons=n_electrons,
            )
        if token in SMEARING_PRESETS:
            return SmearingResolution(
                float(SMEARING_PRESETS[token]),
                method=meth,
                source=f"preset:{token}",
                reason=f"named smearing preset {token!r}",
            )
        match = _NUMERIC_WITH_OPTIONAL_UNIT.match(value)
        if match is None:
            valid = ", ".join(sorted(SMEARING_PRESETS)) + ", auto"
            raise ValueError(
                f"unknown smearing preset {value!r}; valid presets: {valid}"
            )
        numeric = float(match.group(1))
        embedded_unit = match.group(2)
        active_unit = embedded_unit if embedded_unit is not None else unit
        return SmearingResolution(
            temperature_in_hartree(numeric, active_unit),
            method=meth,
            source=f"explicit:{active_unit}",
            reason=f"numeric string in {active_unit}",
        )

    return SmearingResolution(
        temperature_in_hartree(float(value), unit),
        method=meth,
        source=f"explicit:{unit}",
        reason=f"numeric value in {unit}",
    )
