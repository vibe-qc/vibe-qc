"""S22 #04 — formamide dimer (HCONH2)2. Strong double-H-bond benchmark.

Geometry from the GMTKN55 distribution of the S22 test set (Jurečka,
Šponer, Černý, Hobza 2006), in Ångström. Revised S22A reference
interaction energy is -16.12 kcal/mol (Marshall et al. 2011) — one of
the strongest noncovalent interactions in S22.
"""
from __future__ import annotations

from ...core.spec import AtomCart, MoleculeSpec

SPEC = MoleculeSpec(
    id="s22_formamide_dimer",
    family="molecule_noncovalent_hb",
    atoms=(
        AtomCart(symbol="C", z=6, xyz_ang=(-2.01864960,   0.05288301,   0.00000000)),
        AtomCart(symbol="C", z=6, xyz_ang=( 2.01864960,  -0.05288301,   0.00000000)),
        AtomCart(symbol="N", z=7, xyz_ang=(-1.40777040,  -1.14248435,   0.00000000)),
        AtomCart(symbol="N", z=7, xyz_ang=( 1.40777040,   1.14248435,   0.00000000)),
        AtomCart(symbol="H", z=1, xyz_ang=(-0.38724410,  -1.20778240,   0.00000000)),
        AtomCart(symbol="H", z=1, xyz_ang=(-3.11706187,  -0.01370100,   0.00000000)),
        AtomCart(symbol="H", z=1, xyz_ang=(-1.96459656,  -1.97703662,   0.00000000)),
        AtomCart(symbol="H", z=1, xyz_ang=( 1.96459656,   1.97703662,   0.00000000)),
        AtomCart(symbol="H", z=1, xyz_ang=( 0.38724410,   1.20778240,   0.00000000)),
        AtomCart(symbol="H", z=1, xyz_ang=( 3.11706187,   0.01370100,   0.00000000)),
        AtomCart(symbol="O", z=8, xyz_ang=(-1.45220040,   1.14363435,   0.00000000)),
        AtomCart(symbol="O", z=8, xyz_ang=( 1.45220040,  -1.14363435,   0.00000000)),
    ),
    charge=0,
    multiplicity=1,
    default_conv_tol_energy=1e-9,
    default_max_iter=80,
    notes=(
        "S22 system #04 — formamide dimer with two cooperative N-H···O=C "
        "hydrogen bonds. Tests cross-code agreement on a doubly-bound "
        "amide-amide motif relevant to peptide / protein structure."
    ),
    citation=(
        "Jurečka, Šponer, Černý, Hobza, *Phys. Chem. Chem. Phys.* "
        "**2006**, 8, 1985 (S22 origin, DOI 10.1039/B600027D); "
        "Marshall, Burns, Sherrill, *J. Chem. Phys.* **2011**, 135, "
        "194102 (S22A revised, DOI 10.1063/1.3659142). Geometry from "
        "grimme-lab/GMTKN55 S22/04/struc.xyz."
    ),
)
