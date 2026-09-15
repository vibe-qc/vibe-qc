"""CaF₂ fluorite — conventional cubic 12-atom cell, A·B₂ stoichiometry.

Fluorite is the canonical A·B₂ ionic-crystal structure (space group
Fm-3m, no. 225). The conventional cubic cell contains 4 Ca cations
on the fcc sublattice and 8 F anions filling all tetrahedral holes.
Lattice constant a = 5.463 Å is the room-temperature experimental
value (Wyckoff, *Crystal Structures* Vol. 1, 2nd ed., Wiley 1963,
p. 240). Peintinger 2013 (JCC, the pob-TZVP basis paper) covers CaF₂
in the SI compound list; published PW1PW + HF totals at the pob-TZVP
optimised geometry can be substituted into ``expected/*.json`` as
``published_energy_ha`` / ``published_energy_ha_hf`` once read off
the SI (deferred — the structure is the ground-truth contribution
here).

Wyckoff positions:
  Ca at 4a: (0, 0, 0) and Fm-3m equivalents
  F  at 8c: (1/4, 1/4, 1/4) and Fm-3m equivalents

For the cross-code parity suite this is a single-determinant SCF on
the 12-atom conventional cell — a higher per-cell electron count
than the 8-atom rocksalts (40 electrons vs 20-28 for LiH/NaCl/MgO
sto-3g), tests the lattice-sum machinery on a denser cell.
"""
from __future__ import annotations

from ...core.spec import AtomFrac, PeriodicSpec

_A_ANG = 5.463

_CA_FRAC = (
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
    id="caf2_fluorite",
    family="fluorite",
    lattice_ang=(
        (_A_ANG, 0.0, 0.0),
        (0.0, _A_ANG, 0.0),
        (0.0, 0.0, _A_ANG),
    ),
    space_group="Fm-3m",
    atoms=tuple(
        [AtomFrac(symbol="Ca", z=20, frac=p) for p in _CA_FRAC]
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
        "CaF₂ fluorite at the experimental room-temperature lattice "
        "constant (5.463 Å, Wyckoff 1963). Canonical A·B₂ structure: "
        "4 Ca on fcc sublattice + 8 F filling tetrahedral holes = 12 "
        "atoms / cell. Strong ionic Ca²⁺(F⁻)₂ binding, ~12 eV optical "
        "gap (LDA underestimates). Higher per-cell electron count "
        "than the 8-atom rocksalts → useful stress test for the "
        "lattice-sum machinery."
    ),
    citation=(
        "Wyckoff, *Crystal Structures* Vol. 1, 2nd ed., Wiley **1963**, "
        "p. 240 (CaF₂ fluorite structure + room-temperature lattice "
        "constant 5.463 Å). Peintinger, Vilela Oliveira, Bredow, "
        "*J. Comput. Chem.* **2013**, 34, 451 (DOI 10.1002/jcc.23153) "
        "— pob-TZVP basis paper, SI compound list includes CaF₂; "
        "published PW1PW / HF total energies can be backfilled into "
        "expected/*.json once looked up."
    ),
)
