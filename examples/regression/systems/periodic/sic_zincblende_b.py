"""β-SiC zincblende — conventional cubic 8-atom cell, F-43m.

a = 4.344 Å (pob-TZVP/PW1PW-optimised; experimental 4.358).
PW1PW total **−327.641042 Ha** (Peintinger 2013 SI Table 1).

β-SiC is the cubic polytype of silicon carbide; α-SiC (hexagonal
6H polytype, P6_3mc) is the more common form. Industrial substrate
for high-power and high-temperature electronics.
"""
from __future__ import annotations

from ...core.spec import AtomFrac, PeriodicSpec

_A_ANG = 4.344

# Zincblende: Si at FCC, C at FCC + (1/4, 1/4, 1/4)
_SI_FRAC = (
    (0.0, 0.0, 0.0),
    (0.0, 0.5, 0.5),
    (0.5, 0.0, 0.5),
    (0.5, 0.5, 0.0),
)
_C_FRAC = (
    (0.25, 0.25, 0.25),
    (0.25, 0.75, 0.75),
    (0.75, 0.25, 0.75),
    (0.75, 0.75, 0.25),
)

SPEC = PeriodicSpec(
    id="sic_zincblende_b",
    family="zincblende",
    lattice_ang=(
        (_A_ANG, 0.0, 0.0),
        (0.0, _A_ANG, 0.0),
        (0.0, 0.0, _A_ANG),
    ),
    space_group="F-43m",
    atoms=tuple(
        [AtomFrac(symbol="Si", z=14, frac=p) for p in _SI_FRAC]
        + [AtomFrac(symbol="C",  z=6,  frac=p) for p in _C_FRAC]
    ),
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
        "β-SiC zincblende at pob-TZVP/PW1PW-optimised geometry. "
        "Wide-gap covalent semiconductor (~2.4 eV indirect gap LDA, "
        "experimental 2.36 eV). The classic 'between Si and diamond' "
        "test case — covalent character with electronegativity "
        "asymmetry."
    ),
    citation=(
        "Peintinger, Vilela Oliveira, Bredow, *J. Comput. Chem.* "
        "**2013**, 34, 451 (DOI 10.1002/jcc.23153) — Table 8 (lattice "
        "constant) + Table 1 (PW1PW total energy)."
    ),
)
