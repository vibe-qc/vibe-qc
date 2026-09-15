"""NaH rocksalt — conventional cubic 8-atom cell.

a = 4.800 Å (pob-TZVP/PW1PW-optimised; experimental 4.890).
PW1PW total **−162.895781 Ha**; HF total **−162.454844 Ha**.
"""
from __future__ import annotations

from ...core.spec import AtomFrac, PeriodicSpec

_A_ANG = 4.800

_NA_FRAC = (
    (0.0, 0.0, 0.0),
    (0.0, 0.5, 0.5),
    (0.5, 0.0, 0.5),
    (0.5, 0.5, 0.0),
)
_H_FRAC = (
    (0.5, 0.5, 0.5),
    (0.5, 0.0, 0.0),
    (0.0, 0.5, 0.0),
    (0.0, 0.0, 0.5),
)

SPEC = PeriodicSpec(
    id="nah_rocksalt",
    family="rocksalt",
    lattice_ang=(
        (_A_ANG, 0.0, 0.0),
        (0.0, _A_ANG, 0.0),
        (0.0, 0.0, _A_ANG),
    ),
    space_group="Fm-3m",
    atoms=tuple(
        [AtomFrac(symbol="Na", z=11, frac=p) for p in _NA_FRAC]
        + [AtomFrac(symbol="H",  z=1,  frac=p) for p in _H_FRAC]
    ),
    default_kmesh=(1, 1, 1),
    default_spacing_bohr=0.5,
    default_cutoff_bohr=12.0,
    default_nuclear_cutoff_bohr=25.0,
    default_omega=0.3,
    default_conv_tol_energy=1e-7,
    default_max_iter=60,
    default_initial_guess="SAD",
    default_damping=0.85,
    notes=(
        "NaH rocksalt at pob-TZVP/PW1PW-optimised geometry. Hybrid "
        "ionic / covalent character (Na cation, H anion). Smaller "
        "Madelung magnitude than NaCl due to lighter anion."
    ),
    citation=(
        "Peintinger, Vilela Oliveira, Bredow, *J. Comput. Chem.* "
        "**2013**, 34, 451 (DOI 10.1002/jcc.23153) — Table 4 + Tables "
        "1-2."
    ),
)
