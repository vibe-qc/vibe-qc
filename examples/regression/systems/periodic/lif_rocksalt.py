"""LiF rocksalt — conventional cubic 8-atom cell.

Lattice constant a = 4.027 Å is the pob-TZVP/PW1PW-optimised value
from Peintinger 2013 Table 4 (matches the experimental 4.027 Å exactly
at this level of theory). The Peintinger 2013 SI Table 1 lists the
PW1PW total energy per cell at this geometry as **−107.519730 Ha**;
Table 2 gives the HF total as **−107.084927 Ha**. Both are recorded
in `expected/lif_rocksalt__*.json` as advisory reference values until
vibe-qc gets PW1PW + multi-k for direct numerical match.
"""
from __future__ import annotations

from ...core.spec import AtomFrac, PeriodicSpec

_A_ANG = 4.027

_LI_FRAC = (
    (0.0, 0.0, 0.0),
    (0.0, 0.5, 0.5),
    (0.5, 0.0, 0.5),
    (0.5, 0.5, 0.0),
)
_F_FRAC = (
    (0.5, 0.5, 0.5),
    (0.5, 0.0, 0.0),
    (0.0, 0.5, 0.0),
    (0.0, 0.0, 0.5),
)

SPEC = PeriodicSpec(
    id="lif_rocksalt",
    family="rocksalt",
    lattice_ang=(
        (_A_ANG, 0.0, 0.0),
        (0.0, _A_ANG, 0.0),
        (0.0, 0.0, _A_ANG),
    ),
    space_group="Fm-3m",
    atoms=tuple(
        [AtomFrac(symbol="Li", z=3, frac=p) for p in _LI_FRAC]
        + [AtomFrac(symbol="F",  z=9, frac=p) for p in _F_FRAC]
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
        "LiF rocksalt at the pob-TZVP/PW1PW-optimised lattice constant "
        "(4.027 Å, identical to experimental at this level). Light-"
        "element ionic crystal — easier convergence than NaCl/MgO. "
        "Strong ionic Li⁺F⁻ binding, ~14 eV optical gap (LDA "
        "underestimates)."
    ),
    citation=(
        "Peintinger, Vilela Oliveira, Bredow, *J. Comput. Chem.* "
        "**2013**, 34, 451 (DOI 10.1002/jcc.23153) — pob-TZVP basis-set "
        "paper, Table 4 (lattice constant) + Tables 1-2 (PW1PW + HF "
        "reference total energies). Vilela Oliveira et al. *J. Comput. "
        "Chem.* **2019**, 40, 2364 (DOI 10.1002/jcc.26013) — pob-TZVP-"
        "rev2 / pob-DZVP-rev2 update Table 2."
    ),
)
