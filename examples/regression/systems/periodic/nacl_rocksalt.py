"""NaCl rocksalt — conventional cubic 8-atom cell."""
from __future__ import annotations

from ...core.spec import AtomFrac, PeriodicSpec

_A_ANG = 5.640                                    # NaCl experimental a (RT)

_NA_FRAC = (
    (0.0, 0.0, 0.0),
    (0.0, 0.5, 0.5),
    (0.5, 0.0, 0.5),
    (0.5, 0.5, 0.0),
)
_CL_FRAC = (
    (0.5, 0.5, 0.5),
    (0.5, 0.0, 0.0),
    (0.0, 0.5, 0.0),
    (0.0, 0.0, 0.5),
)

SPEC = PeriodicSpec(
    id="nacl_rocksalt",
    family="rocksalt",
    lattice_ang=(
        (_A_ANG, 0.0, 0.0),
        (0.0, _A_ANG, 0.0),
        (0.0, 0.0, _A_ANG),
    ),
    space_group="Fm-3m",
    atoms=tuple(
        [AtomFrac(symbol="Na", z=11, frac=p) for p in _NA_FRAC]
        + [AtomFrac(symbol="Cl", z=17, frac=p) for p in _CL_FRAC]
    ),
    default_kmesh=(1, 1, 1),
    default_spacing_bohr=0.6,
    default_cutoff_bohr=12.0,
    default_nuclear_cutoff_bohr=25.0,
    default_omega=0.3,                            # tighter Ewald for ionic
    default_conv_tol_energy=1e-7,
    default_max_iter=60,
    default_initial_guess="SAD",                  # required for deep-core ionic
    default_damping=0.85,                         # required for deep-core ionic
    notes=(
        "Standard rocksalt parity benchmark. LDA underestimates the NaCl "
        "gap (experimental optical ~8.7 eV). HCORE + default DIIS diverges "
        "on this ionic / deep-core system; SAD guess + damping=0.85 + "
        "omega=0.3 are the v0.7 SCF driver's recommended fixes."
    ),
    citation=(
        "Lattice constant: NaCl experimental room-temperature value 5.640 Å "
        "(Hull, ICSD)."
    ),
)
