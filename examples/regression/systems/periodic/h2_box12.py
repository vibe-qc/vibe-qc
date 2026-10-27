"""H2 in a 12-bohr simple-cubic box -- the open-shell state witness (#647).

One cell carries either the closed-shell singlet or, with an explicit
``multiplicity=3`` matrix request, the uniform triplet lattice: two alpha
and no beta electrons in every cell of the Born-von Karman supercell. It is
the cell of the BIPOLE parity references
(``bipole_parity/multik_exchange_split_refs_pyscf.py``), whose module notes
explain why PySCF needs the per-cell ``nelec`` at multi-k.
"""
from __future__ import annotations

from ...core.spec import AtomFrac, PeriodicSpec

_BOHR_ANG = 0.529177210903
_A_BOHR = 12.0
_HALF_BOND = 0.7 / _A_BOHR  # exactly 1.4 bohr; the triplet curve is steep

SPEC = PeriodicSpec(
    id="h2_box12",
    family="molecular_box",
    lattice_ang=(
        (_A_BOHR * _BOHR_ANG, 0.0, 0.0),
        (0.0, _A_BOHR * _BOHR_ANG, 0.0),
        (0.0, 0.0, _A_BOHR * _BOHR_ANG),
    ),
    space_group="P4/mmm",
    atoms=(
        AtomFrac(symbol="H", z=1, frac=(0.5, 0.5, 0.5 - _HALF_BOND)),
        AtomFrac(symbol="H", z=1, frac=(0.5, 0.5, 0.5 + _HALF_BOND)),
    ),
    default_kmesh=(1, 1, 1),
    default_cutoff_bohr=12.0,
    default_nuclear_cutoff_bohr=25.0,
    default_conv_tol_energy=1e-9,
    default_max_iter=60,
    notes=(
        "Closed-shell by default; matrix rows request the per-cell triplet "
        "with multiplicity=3. Not a magnetic material, a state-transport "
        "control with a cheap, well-separated cell."
    ),
    citation="Geometry of the vibe-qc BIPOLE parity fixtures (H-H 1.4 bohr).",
)
