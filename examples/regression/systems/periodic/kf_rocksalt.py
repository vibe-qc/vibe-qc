"""KF rocksalt — conventional cubic 8-atom cell.

a = 5.364 Å (pob-TZVP/PW1PW-optimised; experimental 5.347).
PW1PW total **−699.904607 Ha**; HF total **−698.756554 Ha**.
"""
from __future__ import annotations

from ...core.spec import AtomFrac, PeriodicSpec

_A_ANG = 5.364

_K_FRAC = (
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
    id="kf_rocksalt",
    family="rocksalt",
    lattice_ang=(
        (_A_ANG, 0.0, 0.0),
        (0.0, _A_ANG, 0.0),
        (0.0, 0.0, _A_ANG),
    ),
    space_group="Fm-3m",
    atoms=tuple(
        [AtomFrac(symbol="K", z=19, frac=p) for p in _K_FRAC]
        + [AtomFrac(symbol="F", z=9,  frac=p) for p in _F_FRAC]
    ),
    default_kmesh=(1, 1, 1),
    default_spacing_bohr=0.6,
    default_cutoff_bohr=12.0,
    default_nuclear_cutoff_bohr=25.0,
    default_omega=0.3,
    default_conv_tol_energy=1e-7,
    default_max_iter=60,
    default_initial_guess="SAD",
    default_damping=0.85,
    notes=(
        "KF rocksalt at pob-TZVP/PW1PW-optimised geometry. Heavy "
        "alkali metal K (Z=19, 18 core electrons) plus F — extreme "
        "deep-core ionic case, slightly larger lattice (5.36 Å) "
        "than NaF."
    ),
    citation=(
        "Peintinger, Vilela Oliveira, Bredow, *J. Comput. Chem.* "
        "**2013**, 34, 451 (DOI 10.1002/jcc.23153) — Table 4 + Tables "
        "1-2."
    ),
)
