"""User-facing smearing-options dataclass.

At v0.10.x this lives Python-side only -- the C++-bound
``PeriodicRHFOptions`` / ``PeriodicSCFOptions`` /
``PeriodicKSOptions`` structs keep their existing
``smearing_temperature: float`` field as the on-options-surface
single source of truth (see `docs/design_smearing.md` Sec. Q4 for
the staging rationale). Drivers normalise the legacy float at
the top of their SCF loop via
``SmearingOptions.from_legacy_kwarg(...)`` and feed the
resulting dataclass to ``vibeqc.smearing.apply_smearing(...)``.

At v0.11.0 this dataclass becomes a C++ struct embedded on every
periodic options surface, and the legacy ``smearing_temperature``
float is removed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from .resolution import resolve_smearing_temperature


VALID_FLAVORS = ("fermi-dirac", "mermin", "methfessel-paxton", "marzari-vanderbilt")

# Spelling aliases mapped onto a canonical flavor.
_FLAVOR_ALIASES = {
    "fermi": "fermi-dirac",
    "fd": "fermi-dirac",
    "meremin": "mermin",
    "mp": "methfessel-paxton",
    "cold": "marzari-vanderbilt",
}


@dataclass(frozen=True)
class SmearingOptions:
    """Canonical user-facing smearing configuration.

    ``temperature`` is the electronic energy ``k_B T`` in Hartree
    (matching the v0.4.0 contract on the C++ options surface).
    ``temperature == 0.0`` disables smearing -- drivers fast-path
    on it.

    ``flavor`` selects the occupation function: ``"fermi-dirac"``
    (default), ``"mermin"`` (Mermin finite-temperature free-energy
    functional -- same occupation shape as Fermi-Dirac, reports
    ``A = E - T·S``), ``"methfessel-paxton"`` (M6, Hermite-polynomial
    cold smearing of order ``mp_order``) and ``"marzari-vanderbilt"``
    (M7, cold smearing) are all implemented -- ``apply_smearing``
    dispatches each. An unrecognised flavor raises
    ``NotImplementedError``.

    ``mp_order`` is the Hermite-polynomial order for
    Methfessel-Paxton; ignored for other flavors.

    ``source`` and ``reason`` mirror :class:`SmearingResolution`
    so SCF logs can say "preset:metal" or "explicit:kelvin"
    rather than just a number.
    """

    temperature: float = 0.0
    flavor: str = "fermi-dirac"
    mp_order: int = 1
    source: str = "explicit"
    reason: str = ""

    def __post_init__(self) -> None:
        if float(self.temperature) < 0.0:
            raise ValueError(
                "SmearingOptions.temperature must be >= 0 (Hartree); "
                f"got {self.temperature!r}"
            )
        flav = str(self.flavor).strip().lower().replace("_", "-")
        flav = _FLAVOR_ALIASES.get(flav, flav)
        if flav not in VALID_FLAVORS:
            raise ValueError(
                f"SmearingOptions.flavor must be one of {VALID_FLAVORS} "
                "(aliases: 'fermi'/'fd' -> fermi-dirac, 'meremin' -> mermin, "
                f"'mp' -> methfessel-paxton, 'cold' -> marzari-vanderbilt); "
                f"got {self.flavor!r}"
            )
        if flav == "methfessel-paxton":
            n = int(self.mp_order)
            if n not in (1, 2):
                raise ValueError(
                    "SmearingOptions.mp_order must be 1 or 2 for "
                    f"Methfessel-Paxton; got {self.mp_order!r}"
                )
        # Normalise alias spelling on the stored field so downstream
        # comparisons never see "fermi_dirac" vs "fermi-dirac".
        object.__setattr__(self, "flavor", flav)

    @property
    def enabled(self) -> bool:
        """``True`` when smearing is active (``temperature > 0``)."""
        return float(self.temperature) > 0.0

    @classmethod
    def from_legacy_kwarg(
        cls,
        smearing_temperature: float,
        *,
        flavor: str = "fermi-dirac",
        mp_order: int = 1,
    ) -> "SmearingOptions":
        """Build from the v0.4.0 ``smearing_temperature: float`` kwarg.

        This is the internal driver-side constructor that bridges
        the legacy C++ option field to the new Python dataclass
        without involving the user. No deprecation warning here --
        the warning fires at the user-facing kwarg boundary.
        """
        T = float(smearing_temperature)
        return cls(
            temperature=T,
            flavor=flavor,
            mp_order=int(mp_order),
            source="legacy_kwarg" if T > 0.0 else "explicit",
            reason=(
                "via PeriodicRHFOptions.smearing_temperature "
                "(deprecated in v0.10.x)"
                if T > 0.0
                else "smearing disabled"
            ),
        )

    @classmethod
    def from_user(
        cls,
        value: Union[None, str, float] = 0.0,
        *,
        unit: str = "hartree",
        flavor: str = "fermi-dirac",
        mp_order: int = 1,
        metallic: Optional[bool] = None,
        band_gap_hartree: Optional[float] = None,
        n_electrons: Optional[float] = None,
    ) -> "SmearingOptions":
        """Resolve user-facing temperature input to canonical options.

        Accepts everything ``resolve_smearing_temperature`` accepts
        (numeric width, ``"1000 K"``, ``"0.1 eV"``, ``"metal"``,
        ``"auto"``, etc.).
        """
        res = resolve_smearing_temperature(
            value,
            unit=unit,
            metallic=metallic,
            band_gap_hartree=band_gap_hartree,
            n_electrons=n_electrons,
        )
        return cls(
            temperature=res.temperature,
            flavor=flavor,
            mp_order=int(mp_order),
            source=res.source,
            reason=res.reason,
        )
