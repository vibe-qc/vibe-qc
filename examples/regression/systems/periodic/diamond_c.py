"""Diamond — cubic carbon (Fd-3m), conventional 8-atom cell.

a = 3.547 Å (pob-TZVP/PW1PW-optimised; experimental 3.567).
PW1PW total **−76.224192 Ha** (Peintinger 2013 SI Table 1).

Diamond cubic structure (Fd-3m): two interpenetrating FCC carbon
sublattices offset by (1/4, 1/4, 1/4). Same skeleton as zincblende
but both sublattices are the same element. The textbook covalent
solid.
"""
from __future__ import annotations

from ...core.spec import AtomFrac, PeriodicSpec

_A_ANG = 3.547

# Diamond cubic: C at FCC + C at FCC + (1/4, 1/4, 1/4)
_C_FRAC = (
    # Sublattice 1
    (0.0, 0.0, 0.0),
    (0.0, 0.5, 0.5),
    (0.5, 0.0, 0.5),
    (0.5, 0.5, 0.0),
    # Sublattice 2 (offset by (1/4, 1/4, 1/4))
    (0.25, 0.25, 0.25),
    (0.25, 0.75, 0.75),
    (0.75, 0.25, 0.75),
    (0.75, 0.75, 0.25),
)

SPEC = PeriodicSpec(
    id="diamond_c",
    family="diamond_cubic",
    lattice_ang=(
        (_A_ANG, 0.0, 0.0),
        (0.0, _A_ANG, 0.0),
        (0.0, 0.0, _A_ANG),
    ),
    space_group="Fd-3m",
    atoms=tuple(AtomFrac(symbol="C", z=6, frac=p) for p in _C_FRAC),
    default_kmesh=(1, 1, 1),
    default_spacing_bohr=0.4,
    default_cutoff_bohr=10.0,
    default_nuclear_cutoff_bohr=18.0,
    default_omega=0.5,
    default_conv_tol_energy=1e-7,
    default_max_iter=40,
    default_initial_guess="HCORE",
    default_damping=0.5,
    notes=(
        "Diamond at pob-TZVP/PW1PW-optimised geometry. The reference "
        "covalent solid — closed-shell, wide-gap (~5.5 eV indirect "
        "LDA), no symmetry-breaking. Easy SCF convergence; useful as "
        "a covalent counterpart to the rare-gas Ne FCC sanity case."
    ),
    citation=(
        "Peintinger, Vilela Oliveira, Bredow, *J. Comput. Chem.* "
        "**2013**, 34, 451 (DOI 10.1002/jcc.23153) — Table 8 (lattice "
        "constant) + Table 1 (PW1PW total energy)."
    ),
)
