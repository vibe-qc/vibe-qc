"""Silicon — cubic Si (Fd-3m), conventional 8-atom cell.

a = 5.391 Å (pob-TZVP/PW1PW-optimised; experimental 5.431).
PW1PW total **−578.997391 Ha** (Peintinger 2013 SI Table 1).

Silicon shares diamond's structure (Fd-3m, two FCC sublattices
offset by (1/4, 1/4, 1/4)) but with much weaker σ bonds and a
narrower gap (~1.1 eV indirect, experimental). The baseline
semiconductor for everything in microelectronics.
"""
from __future__ import annotations

from ...core.spec import AtomFrac, PeriodicSpec

_A_ANG = 5.391

# Diamond-cubic Si lattice (same skeleton as diamond_c, scaled up)
_SI_FRAC = (
    (0.0, 0.0, 0.0),
    (0.0, 0.5, 0.5),
    (0.5, 0.0, 0.5),
    (0.5, 0.5, 0.0),
    (0.25, 0.25, 0.25),
    (0.25, 0.75, 0.75),
    (0.75, 0.25, 0.75),
    (0.75, 0.75, 0.25),
)

SPEC = PeriodicSpec(
    id="silicon",
    family="diamond_cubic",
    lattice_ang=(
        (_A_ANG, 0.0, 0.0),
        (0.0, _A_ANG, 0.0),
        (0.0, 0.0, _A_ANG),
    ),
    space_group="Fd-3m",
    atoms=tuple(AtomFrac(symbol="Si", z=14, frac=p) for p in _SI_FRAC),
    default_kmesh=(1, 1, 1),
    default_spacing_bohr=0.5,
    default_cutoff_bohr=10.0,
    default_nuclear_cutoff_bohr=18.0,
    default_omega=0.5,
    default_conv_tol_energy=1e-7,
    default_max_iter=40,
    default_initial_guess="HCORE",
    default_damping=0.5,
    notes=(
        "Silicon at pob-TZVP/PW1PW-optimised geometry. Standard "
        "covalent semiconductor reference. Larger cell than diamond "
        "(5.39 Å vs 3.55 Å) — useful for testing SCF behaviour at "
        "moderate cell sizes with the same structural family."
    ),
    citation=(
        "Peintinger, Vilela Oliveira, Bredow, *J. Comput. Chem.* "
        "**2013**, 34, 451 (DOI 10.1002/jcc.23153) — Table 8 (lattice "
        "constant) + Table 1 (PW1PW total energy)."
    ),
)
