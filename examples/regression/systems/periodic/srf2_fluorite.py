"""SrF₂ fluorite — conventional cubic 12-atom cell, A·B₂ stoichiometry.

Sibling spec to ``caf2_fluorite.py``; same space group (Fm-3m, no.
225) and Wyckoff positions, scaled lattice constant a = 5.800 Å
(room-temperature experimental, Wyckoff *Crystal Structures* Vol. 1,
2nd ed., 1963, p. 240). Sr is heavier than Ca and not in the bundled
STO-3G basis — this spec is intended for pob-TZVP-rev2 / pob-DZVP-
rev2 runs only.

Useful as a paired comparison with CaF₂ + BaF₂ — same structure,
different cation mass / radius → tests the per-element lattice-sum
behaviour and the pob-* basis for second-row alkaline-earth cations.

Wyckoff positions identical to CaF₂:
  Sr at 4a: (0, 0, 0) and Fm-3m equivalents
  F  at 8c: (1/4, 1/4, 1/4) and Fm-3m equivalents
"""
from __future__ import annotations

from ...core.spec import AtomFrac, PeriodicSpec

_A_ANG = 5.800

_SR_FRAC = (
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
    id="srf2_fluorite",
    family="fluorite",
    lattice_ang=(
        (_A_ANG, 0.0, 0.0),
        (0.0, _A_ANG, 0.0),
        (0.0, 0.0, _A_ANG),
    ),
    space_group="Fm-3m",
    atoms=tuple(
        [AtomFrac(symbol="Sr", z=38, frac=p) for p in _SR_FRAC]
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
        "SrF₂ fluorite at the experimental room-temperature lattice "
        "constant (5.800 Å, Wyckoff 1963). Same Fm-3m structure as "
        "CaF₂ with the cation row swapped (Z=20 → Z=38). Sr is NOT "
        "in the bundled STO-3G basis; this spec is for pob-TZVP-rev2 "
        "/ pob-DZVP-rev2 runs only. Per-element ECP routing for Sr "
        "(post-Kr core) is also exercised by this spec when used "
        "with pob bases."
    ),
    citation=(
        "Wyckoff, *Crystal Structures* Vol. 1, 2nd ed., Wiley **1963**, "
        "p. 240 (SrF₂ fluorite, lattice constant 5.800 Å). Peintinger, "
        "Vilela Oliveira, Bredow, *J. Comput. Chem.* **2013**, 34, 451 "
        "(DOI 10.1002/jcc.23153) — pob-TZVP basis paper, SI compound "
        "list includes SrF₂."
    ),
)
