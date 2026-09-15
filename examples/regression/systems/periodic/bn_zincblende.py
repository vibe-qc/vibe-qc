"""β-BN zincblende — conventional cubic 8-atom cell, F-43m.

a = 3.606 Å (pob-TZVP/PW1PW-optimised; experimental 3.625, Table 8).
PW1PW total **−79.719012 Ha** (Peintinger 2013 SI Table 1).

Zincblende structure (F-43m): two interpenetrating FCC sublattices
offset by (1/4, 1/4, 1/4) — same skeleton as diamond cubic but
with two distinct elements. β-BN is the cubic polymorph of boron
nitride; α-BN (hexagonal, P6_3/mmc) is the more common graphite-like
form.
"""
from __future__ import annotations

from ...core.spec import AtomFrac, PeriodicSpec

_A_ANG = 3.606

# Zincblende: B at FCC, N at FCC + (1/4, 1/4, 1/4)
_B_FRAC = (
    (0.0, 0.0, 0.0),
    (0.0, 0.5, 0.5),
    (0.5, 0.0, 0.5),
    (0.5, 0.5, 0.0),
)
_N_FRAC = (
    (0.25, 0.25, 0.25),
    (0.25, 0.75, 0.75),
    (0.75, 0.25, 0.75),
    (0.75, 0.75, 0.25),
)

SPEC = PeriodicSpec(
    id="bn_zincblende",
    family="zincblende",
    lattice_ang=(
        (_A_ANG, 0.0, 0.0),
        (0.0, _A_ANG, 0.0),
        (0.0, 0.0, _A_ANG),
    ),
    space_group="F-43m",
    atoms=tuple(
        [AtomFrac(symbol="B", z=5, frac=p) for p in _B_FRAC]
        + [AtomFrac(symbol="N", z=7, frac=p) for p in _N_FRAC]
    ),
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
        "β-BN zincblende at pob-TZVP/PW1PW-optimised geometry. "
        "Wide-gap covalent semiconductor (~6 eV indirect gap LDA). "
        "Light-element zincblende — easier convergence than the "
        "α-BN hexagonal polymorph (which has a complex layered "
        "structure)."
    ),
    citation=(
        "Peintinger, Vilela Oliveira, Bredow, *J. Comput. Chem.* "
        "**2013**, 34, 451 (DOI 10.1002/jcc.23153) — Table 8 (lattice "
        "constant) + Table 1 (PW1PW total energy)."
    ),
)
