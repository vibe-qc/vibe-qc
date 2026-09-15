"""Ne FCC — closed-shell, wide-gap, converges trivially.

Used as the first-cut sanity-check anchor: 4 atoms, 20 BFs at sto-3g,
~14 eV gap, no linear-dependence, no SCF stiffness. If the regression
plumbing (verbose log + CSV + ΔE comparator) doesn't work on Ne FCC,
nothing else will.
"""
from __future__ import annotations

from ...core.spec import AtomFrac, PeriodicSpec

_A_ANG = 4.43                                     # Ne FCC experimental a

_FCC = (
    (0.0, 0.0, 0.0),
    (0.0, 0.5, 0.5),
    (0.5, 0.0, 0.5),
    (0.5, 0.5, 0.0),
)

SPEC = PeriodicSpec(
    id="ne_fcc",
    family="rare_gas_fcc",
    lattice_ang=(
        (_A_ANG, 0.0, 0.0),
        (0.0, _A_ANG, 0.0),
        (0.0, 0.0, _A_ANG),
    ),
    space_group="Fm-3m",
    atoms=tuple(AtomFrac(symbol="Ne", z=10, frac=p) for p in _FCC),
    default_kmesh=(1, 1, 1),
    default_spacing_bohr=0.4,
    default_cutoff_bohr=10.0,
    default_nuclear_cutoff_bohr=18.0,
    default_omega=0.5,
    default_conv_tol_energy=1e-7,
    default_max_iter=30,
    notes=(
        "Closed-shell, wide-gap, no linear-dependence. The first-cut "
        "sanity-check anchor — if anything goes wrong here it's the "
        "regression plumbing, not the SCF / basis."
    ),
    citation="Ne FCC lattice constant 4.43 Å (low-T experimental).",
)
