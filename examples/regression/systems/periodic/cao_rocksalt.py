"""CaO rocksalt — conventional cubic 8-atom cell.

a = 4.770 Å (pob-TZVP/PW1PW-optimised; experimental 4.811).
PW1PW total **−752.994745 Ha**; HF total **−751.822979 Ha**.
"""
from __future__ import annotations

from ...core.spec import AtomFrac, PeriodicSpec

_A_ANG = 4.770

_CA_FRAC = (
    (0.0, 0.0, 0.0),
    (0.0, 0.5, 0.5),
    (0.5, 0.0, 0.5),
    (0.5, 0.5, 0.0),
)
_O_FRAC = (
    (0.5, 0.5, 0.5),
    (0.5, 0.0, 0.0),
    (0.0, 0.5, 0.0),
    (0.0, 0.0, 0.5),
)

SPEC = PeriodicSpec(
    id="cao_rocksalt",
    family="rocksalt",
    lattice_ang=(
        (_A_ANG, 0.0, 0.0),
        (0.0, _A_ANG, 0.0),
        (0.0, 0.0, _A_ANG),
    ),
    space_group="Fm-3m",
    atoms=tuple(
        [AtomFrac(symbol="Ca", z=20, frac=p) for p in _CA_FRAC]
        + [AtomFrac(symbol="O",  z=8,  frac=p) for p in _O_FRAC]
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
        "CaO rocksalt at pob-TZVP/PW1PW-optimised geometry. Divalent "
        "alkaline-earth oxide; sister case to MgO with deeper Ca core. "
        "Standard rock-salt benchmark for ionic / covalent-crossover "
        "DFT."
    ),
    citation=(
        "Peintinger, Vilela Oliveira, Bredow, *J. Comput. Chem.* "
        "**2013**, 34, 451 (DOI 10.1002/jcc.23153) — Table 4 + Tables "
        "1-2."
    ),
)
