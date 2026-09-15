"""NaF rocksalt — conventional cubic 8-atom cell.

a = 4.627 Å (pob-TZVP/PW1PW-optimised, Peintinger 2013 Table 4;
experimental 4.632 Å). PW1PW total **−262.264337 Ha** (Table 1);
HF total **−261.473781 Ha** (Table 2).
"""
from __future__ import annotations

from ...core.spec import AtomFrac, PeriodicSpec

_A_ANG = 4.627

_NA_FRAC = (
    (0.0, 0.0, 0.0),
    (0.0, 0.5, 0.5),
    (0.5, 0.0, 0.5),
    (0.5, 0.5, 0.0),
)
_F_FRAC = (
    (0.5, 0.5, 0.5),
    (0.5, 0.0, 0.0),
    (0.0, 0.5, 0.0),
    (0.0, 0.0, 0.5),
)

SPEC = PeriodicSpec(
    id="naf_rocksalt",
    family="rocksalt",
    lattice_ang=(
        (_A_ANG, 0.0, 0.0),
        (0.0, _A_ANG, 0.0),
        (0.0, 0.0, _A_ANG),
    ),
    space_group="Fm-3m",
    atoms=tuple(
        [AtomFrac(symbol="Na", z=11, frac=p) for p in _NA_FRAC]
        + [AtomFrac(symbol="F",  z=9,  frac=p) for p in _F_FRAC]
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
        "NaF rocksalt at pob-TZVP/PW1PW-optimised geometry. Deep-core "
        "Na cation — same SAD + damping=0.85 + ω=0.3 recipe as NaCl "
        "applies for converging the bare RKS-LDA on this minimal-basis "
        "test case."
    ),
    citation=(
        "Peintinger, Vilela Oliveira, Bredow, *J. Comput. Chem.* "
        "**2013**, 34, 451 (DOI 10.1002/jcc.23153) — Table 4 + Tables "
        "1-2."
    ),
)
