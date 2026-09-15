"""Back-compat shim -- moved to :mod:`vibeqc.smearing` in v0.10.0.

Every public symbol that lived in ``vibeqc.occupations`` continues
to import from here, so existing user code and downstream tests
keep working without edits. The shim emits no warning on import
(too noisy for a module that's loaded eagerly by the periodic SCF
drivers) -- the user-visible deprecation surface lives at the
``smearing_temperature=`` kwarg level on the high-level wrappers
and on the C++-bound options structs.

Removal target: v0.11.0, alongside removal of the legacy
``smearing_temperature: float`` field on the periodic options
structs. See `docs/design_smearing.md` Sec. Q4 for the schedule.
"""

from __future__ import annotations

from .smearing import (
    EV_PER_HARTREE,
    HARTREE_PER_RYDBERG,
    KB_HARTREE_PER_K,
    SMEARING_PRESETS,
    SmearingResolution,
    aufbau_occupations_per_k,
    electronvolt_to_hartree_temperature,
    fermi_dirac_occupations_per_k,
    guess_smearing_temperature,
    hartree_to_kelvin_temperature,
    kelvin_to_hartree_temperature,
    resolve_smearing_temperature,
    rydberg_to_hartree_temperature,
)


__all__ = [
    "EV_PER_HARTREE",
    "HARTREE_PER_RYDBERG",
    "KB_HARTREE_PER_K",
    "SMEARING_PRESETS",
    "SmearingResolution",
    "aufbau_occupations_per_k",
    "electronvolt_to_hartree_temperature",
    "fermi_dirac_occupations_per_k",
    "guess_smearing_temperature",
    "hartree_to_kelvin_temperature",
    "kelvin_to_hartree_temperature",
    "resolve_smearing_temperature",
    "rydberg_to_hartree_temperature",
]
