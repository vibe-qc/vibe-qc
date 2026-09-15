"""KH rocksalt — conventional cubic 8-atom cell.

a = 5.633 Å (pob-TZVP/PW1PW-optimised; experimental 5.704).
PW1PW total **−600.538439 Ha**; HF total **−599.744158 Ha**.
"""
from __future__ import annotations

from ...core.spec import AtomFrac, PeriodicSpec

_A_ANG = 5.633

_K_FRAC = (
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
    id="kh_rocksalt",
    family="rocksalt",
    lattice_ang=(
        (_A_ANG, 0.0, 0.0),
        (0.0, _A_ANG, 0.0),
        (0.0, 0.0, _A_ANG),
    ),
    space_group="Fm-3m",
    atoms=tuple(
        [AtomFrac(symbol="K", z=19, frac=p) for p in _K_FRAC]
        + [AtomFrac(symbol="H", z=1,  frac=p) for p in _H_FRAC]
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
        "KH rocksalt at pob-TZVP/PW1PW-optimised geometry. Heaviest "
        "alkali hydride in the curated set; deep K core (Z=19) plus "
        "H anion. Stress test for periodic SCF on systems with very "
        "different electron counts per sublattice."
    ),
    citation=(
        "Peintinger, Vilela Oliveira, Bredow, *J. Comput. Chem.* "
        "**2013**, 34, 451 (DOI 10.1002/jcc.23153) — Table 4 + Tables "
        "1-2."
    ),
)
