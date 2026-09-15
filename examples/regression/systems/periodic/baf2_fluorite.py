"""BaF₂ fluorite — conventional cubic 12-atom cell, A·B₂ stoichiometry.

Third member of the CaF₂ / SrF₂ / BaF₂ alkaline-earth fluorite
series. Same space group Fm-3m (no. 225), same Wyckoff positions,
lattice constant a = 6.200 Å (room-temperature experimental, Wyckoff
*Crystal Structures* Vol. 1, 2nd ed., 1963, p. 240). Ba is a heavy
post-Xe cation; this spec exercises the per-element ECP / large-basis
routing far past the STO-3G coverage horizon. Use only with
pob-TZVP-rev2 (which covers Ba) or another large-element-capable
basis.

The CaF₂ / SrF₂ / BaF₂ triple is a useful pob-basis test set in its
own right because the three differ only in the cation (Z = 20, 38,
56) — energy differences across the series probe basis-set
transferability cleanly.

Wyckoff positions identical to CaF₂ / SrF₂:
  Ba at 4a: (0, 0, 0) and Fm-3m equivalents
  F  at 8c: (1/4, 1/4, 1/4) and Fm-3m equivalents
"""
from __future__ import annotations

from ...core.spec import AtomFrac, PeriodicSpec

_A_ANG = 6.200

_BA_FRAC = (
    (0.0, 0.0, 0.0),
    (0.0, 0.5, 0.5),
    (0.5, 0.0, 0.5),
    (0.5, 0.5, 0.0),
)
_F_FRAC = (
    (0.25, 0.25, 0.25),
    (0.25, 0.25, 0.75),
    (0.25, 0.75, 0.25),
    (0.75, 0.25, 0.25),
    (0.75, 0.75, 0.75),
    (0.75, 0.75, 0.25),
    (0.75, 0.25, 0.75),
    (0.25, 0.75, 0.75),
)

SPEC = PeriodicSpec(
    id="baf2_fluorite",
    family="fluorite",
    lattice_ang=(
        (_A_ANG, 0.0, 0.0),
        (0.0, _A_ANG, 0.0),
        (0.0, 0.0, _A_ANG),
    ),
    space_group="Fm-3m",
    atoms=tuple(
        [AtomFrac(symbol="Ba", z=56, frac=p) for p in _BA_FRAC]
        + [AtomFrac(symbol="F",  z=9,  frac=p) for p in _F_FRAC]
    ),
    default_kmesh=(1, 1, 1),
    default_spacing_bohr=0.5,
    default_cutoff_bohr=12.0,
    default_nuclear_cutoff_bohr=25.0,
    default_omega=0.5,
    default_conv_tol_energy=1e-7,
    default_max_iter=60,
    default_initial_guess="SAD",
    default_damping=0.7,
    notes=(
        "BaF₂ fluorite at the experimental room-temperature lattice "
        "constant (6.200 Å, Wyckoff 1963). Heaviest member of the "
        "alkaline-earth fluorite triple (Z=20/38/56). Ba is past the "
        "STO-3G coverage horizon; use only with pob-TZVP-rev2 or "
        "another large-basis / ECP-aware setup. The CaF₂/SrF₂/BaF₂ "
        "series is a clean basis-transferability stress test — "
        "isostructural, cation-only variation."
    ),
    citation=(
        "Wyckoff, *Crystal Structures* Vol. 1, 2nd ed., Wiley **1963**, "
        "p. 240 (BaF₂ fluorite, lattice constant 6.200 Å). Peintinger, "
        "Vilela Oliveira, Bredow, *J. Comput. Chem.* **2013**, 34, 451 "
        "(DOI 10.1002/jcc.23153) — pob-TZVP basis paper, SI compound "
        "list includes BaF₂."
    ),
)
