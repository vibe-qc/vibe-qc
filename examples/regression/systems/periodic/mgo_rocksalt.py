"""MgO rocksalt — conventional cubic 8-atom cell."""
from __future__ import annotations

from ...core.spec import AtomFrac, PeriodicSpec

_A_ANG = 4.211                                    # MgO experimental a (RT)

_MG_FRAC = (
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
    id="mgo_rocksalt",
    family="rocksalt",
    lattice_ang=(
        (_A_ANG, 0.0, 0.0),
        (0.0, _A_ANG, 0.0),
        (0.0, 0.0, _A_ANG),
    ),
    space_group="Fm-3m",
    atoms=tuple(
        [AtomFrac(symbol="Mg", z=12, frac=p) for p in _MG_FRAC]
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
        "Wider-gap (~7 eV LDA) ionic rocksalt. Smaller lattice (4.21 Å) "
        "than NaCl means tighter overlap; same SAD + damping recipe "
        "applies."
    ),
    citation=(
        "MgO experimental room-temperature lattice constant 4.211 Å "
        "(Sasaki et al. 1979)."
    ),
)
