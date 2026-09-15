"""Backend-agnostic smearing utility for periodic and molecular SCF.

The single shared home for fractional-occupation smearing
machinery consumed by every SCF backend x spin x k-mesh
combination vibe-qc ships.  Fermi-Dirac, Mermin (finite-T free
energy), Methfessel-Paxton (M6) and Marzari-Vanderbilt cold (M7)
plug in via the same :func:`apply_smearing` dispatcher without
touching any backend driver.

See `docs/design_smearing.md` for the design contract and
`docs/user_guide/smearing.md` for the user-facing story.
"""

from __future__ import annotations

from ._constants import (
    EV_PER_HARTREE,
    HARTREE_PER_RYDBERG,
    KB_HARTREE_PER_K,
)
from .units import (
    electronvolt_to_hartree_temperature,
    hartree_to_kelvin_temperature,
    kelvin_to_hartree_temperature,
    rydberg_to_hartree_temperature,
    temperature_in_hartree,
)
from .resolution import (
    SMEARING_PRESETS,
    SmearingResolution,
    resolve_smearing_temperature,
)
from .auto import guess_smearing_temperature
from .options import VALID_FLAVORS, SmearingOptions
from .aufbau import aufbau_occupations_per_k
from .fermi_dirac import fermi_dirac_occupations_per_k
from .mermin import mermin_occupations_per_k
from .apply import (
    DEGENERATE_GROUP_SPAN_FACTOR,
    MIN_RESOLVABLE_FRONTIER_GAP,
    ZERO_TEMPERATURE_DEGENERACY_ULPS,
    SmearingResult,
    apply_smearing,
    apply_smearing_open_shell,
    closed_shell_periodic_occupations,
    occupations_are_per_k_integer_aufbau,
    require_fixed_occupied_subspace,
    smeared_occupation_selfconsistency_tolerance,
    unresolved_frontier_cut,
)


__all__ = [
    "EV_PER_HARTREE",
    "HARTREE_PER_RYDBERG",
    "KB_HARTREE_PER_K",
    "SMEARING_PRESETS",
    "SmearingOptions",
    "SmearingResolution",
    "SmearingResult",
    "DEGENERATE_GROUP_SPAN_FACTOR",
    "MIN_RESOLVABLE_FRONTIER_GAP",
    "ZERO_TEMPERATURE_DEGENERACY_ULPS",
    "VALID_FLAVORS",
    "apply_smearing",
    "apply_smearing_open_shell",
    "aufbau_occupations_per_k",
    "closed_shell_periodic_occupations",
    "electronvolt_to_hartree_temperature",
    "fermi_dirac_occupations_per_k",
    "mermin_occupations_per_k",
    "occupations_are_per_k_integer_aufbau",
    "require_fixed_occupied_subspace",
    "guess_smearing_temperature",
    "hartree_to_kelvin_temperature",
    "kelvin_to_hartree_temperature",
    "resolve_smearing_temperature",
    "rydberg_to_hartree_temperature",
    "unresolved_frontier_cut",
    "smeared_occupation_selfconsistency_tolerance",
    "temperature_in_hartree",
]
