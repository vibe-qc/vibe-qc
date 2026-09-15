"""LiH rocksalt — conventional cubic 8-atom cell. The 'easy' rocksalt."""
from __future__ import annotations

from ...core.spec import AtomFrac, PeriodicSpec

_A_ANG = 4.084                                    # LiH experimental a (RT)

_LI_FRAC = (
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
    id="lih_rocksalt",
    family="rocksalt",
    lattice_ang=(
        (_A_ANG, 0.0, 0.0),
        (0.0, _A_ANG, 0.0),
        (0.0, 0.0, _A_ANG),
    ),
    space_group="Fm-3m",
    atoms=tuple(
        [AtomFrac(symbol="Li", z=3, frac=p) for p in _LI_FRAC]
        + [AtomFrac(symbol="H",  z=1, frac=p) for p in _H_FRAC]
    ),
    default_kmesh=(1, 1, 1),
    default_spacing_bohr=0.5,
    default_cutoff_bohr=12.0,
    default_nuclear_cutoff_bohr=25.0,
    default_omega=0.5,
    default_conv_tol_energy=1e-7,
    default_max_iter=40,
    default_initial_guess="SAD",
    default_damping=0.7,
    notes=(
        "Light-element ionic insulator — Li and H are both shallow-core; "
        "easier to converge than NaCl/MgO, but SAD guess still recommended."
    ),
    citation=(
        "LiH experimental low-temperature lattice constant 4.084 Å "
        "(Smith & Leider, JACS 1968). Same geometry as the xfail "
        "regression test in tests/test_periodic_atomic_limit_bug.py."
    ),
)
