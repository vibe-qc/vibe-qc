"""S22 #08 — methane dimer (CH4)2. Pure-dispersion benchmark.

Geometry from the GMTKN55 distribution of the S22 test set (Jurečka,
Šponer, Černý, Hobza 2006), in Ångström. Revised S22A reference
interaction energy is -0.53 kcal/mol (Marshall et al. 2011) — among the
weakest noncovalent interactions in S22, almost entirely London
dispersion.
"""
from __future__ import annotations

from ...core.spec import AtomCart, MoleculeSpec

SPEC = MoleculeSpec(
    id="s22_methane_dimer",
    family="molecule_noncovalent_dispersion",
    atoms=(
        AtomCart(symbol="C", z=6, xyz_ang=( 0.00000000,   0.00000000,  -1.85916155)),
        AtomCart(symbol="C", z=6, xyz_ang=( 0.00000000,   0.00000000,   1.85916155)),
        AtomCart(symbol="H", z=1, xyz_ang=( 0.88862133,  -0.51304576,  -1.49459314)),
        AtomCart(symbol="H", z=1, xyz_ang=( 0.00000000,   1.02609153,  -1.49459314)),
        AtomCart(symbol="H", z=1, xyz_ang=( 0.00000000,   0.00000000,  -2.94828473)),
        AtomCart(symbol="H", z=1, xyz_ang=(-0.88862133,  -0.51304576,  -1.49459314)),
        AtomCart(symbol="H", z=1, xyz_ang=( 0.00000000,   0.00000000,   2.94828473)),
        AtomCart(symbol="H", z=1, xyz_ang=(-0.88862133,   0.51304576,   1.49459314)),
        AtomCart(symbol="H", z=1, xyz_ang=( 0.88862133,   0.51304576,   1.49459314)),
        AtomCart(symbol="H", z=1, xyz_ang=( 0.00000000,  -1.02609153,   1.49459314)),
    ),
    charge=0,
    multiplicity=1,
    default_conv_tol_energy=1e-9,
    default_max_iter=80,
    notes=(
        "S22 system #08 — (CH4)2 weakly-bound aliphatic dimer at the "
        "minimum of the dispersion well. Pure London-dispersion test; "
        "RHF predicts a repulsive PES here, so cross-code RHF parity "
        "tests the *energy* without requiring the binding to be right."
    ),
    citation=(
        "Jurečka, Šponer, Černý, Hobza, *Phys. Chem. Chem. Phys.* "
        "**2006**, 8, 1985 (S22 origin, DOI 10.1039/B600027D); "
        "Marshall, Burns, Sherrill, *J. Chem. Phys.* **2011**, 135, "
        "194102 (S22A revised, DOI 10.1063/1.3659142). Geometry from "
        "grimme-lab/GMTKN55 S22/08/struc.xyz."
    ),
)
