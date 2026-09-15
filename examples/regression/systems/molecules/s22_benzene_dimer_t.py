"""S22 #20 — benzene dimer T-shape. π-system mixed dispersion / quadrupole.

Geometry from the GMTKN55 distribution of the S22 test set (Jurečka,
Šponer, Černý, Hobza 2006), in Ångström. Revised S22A reference
interaction energy is -2.82 kcal/mol (Marshall et al. 2011); the
T-shape is the global minimum of the benzene dimer PES.
"""
from __future__ import annotations

from ...core.spec import AtomCart, MoleculeSpec

SPEC = MoleculeSpec(
    id="s22_benzene_dimer_t",
    family="molecule_noncovalent_dispersion",
    atoms=(
        AtomCart(symbol="C", z=6, xyz_ang=( 0.00000000,   0.00000000,   1.05872819)),
        AtomCart(symbol="C", z=6, xyz_ang=( 0.00000000,  -1.20600877,   1.75736735)),
        AtomCart(symbol="C", z=6, xyz_ang=( 0.00000000,  -1.20717706,   3.15128400)),
        AtomCart(symbol="C", z=6, xyz_ang=( 0.00000000,   0.00000000,   3.84826888)),
        AtomCart(symbol="C", z=6, xyz_ang=( 0.00000000,   1.20717706,   3.15128400)),
        AtomCart(symbol="C", z=6, xyz_ang=( 0.00000000,   1.20600877,   1.75736735)),
        AtomCart(symbol="C", z=6, xyz_ang=(-1.39406376,   0.00000000,  -2.45446048)),
        AtomCart(symbol="C", z=6, xyz_ang=(-0.69704702,   1.20723812,  -2.45493588)),
        AtomCart(symbol="C", z=6, xyz_ang=( 0.69704702,   1.20723812,  -2.45493588)),
        AtomCart(symbol="C", z=6, xyz_ang=( 1.39406376,   0.00000000,  -2.45446048)),
        AtomCart(symbol="C", z=6, xyz_ang=( 0.69704702,  -1.20723812,  -2.45493588)),
        AtomCart(symbol="C", z=6, xyz_ang=(-0.69704702,  -1.20723812,  -2.45493588)),
        AtomCart(symbol="H", z=1, xyz_ang=( 0.00000000,   0.00000000,  -0.02188791)),
        AtomCart(symbol="H", z=1, xyz_ang=( 0.00000000,  -2.14163943,   1.21411459)),
        AtomCart(symbol="H", z=1, xyz_ang=( 0.00000000,  -2.14356624,   3.69268894)),
        AtomCart(symbol="H", z=1, xyz_ang=( 0.00000000,   0.00000000,   4.92984410)),
        AtomCart(symbol="H", z=1, xyz_ang=( 0.00000000,   2.14356624,   3.69268894)),
        AtomCart(symbol="H", z=1, xyz_ang=( 0.00000000,   2.14163943,   1.21411459)),
        AtomCart(symbol="H", z=1, xyz_ang=(-2.47540014,   0.00000000,  -2.45063009)),
        AtomCart(symbol="H", z=1, xyz_ang=(-1.23823249,   2.14356624,  -2.45398459)),
        AtomCart(symbol="H", z=1, xyz_ang=( 1.23823249,   2.14356624,  -2.45398459)),
        AtomCart(symbol="H", z=1, xyz_ang=( 2.47540014,   0.00000000,  -2.45063009)),
        AtomCart(symbol="H", z=1, xyz_ang=( 1.23823249,  -2.14356624,  -2.45398459)),
        AtomCart(symbol="H", z=1, xyz_ang=(-1.23823249,  -2.14356624,  -2.45398459)),
    ),
    charge=0,
    multiplicity=1,
    default_conv_tol_energy=1e-9,
    default_max_iter=80,
    notes=(
        "S22 system #20 — benzene dimer in T-shaped configuration "
        "(global minimum on the (C6H6)2 PES). Mixed CH···π / "
        "dispersion / quadrupole-quadrupole test case; the largest "
        "noncovalent S22 entry in the present curated subset."
    ),
    citation=(
        "Jurečka, Šponer, Černý, Hobza, *Phys. Chem. Chem. Phys.* "
        "**2006**, 8, 1985 (S22 origin, DOI 10.1039/B600027D); "
        "Marshall, Burns, Sherrill, *J. Chem. Phys.* **2011**, 135, "
        "194102 (S22A revised, DOI 10.1063/1.3659142). Geometry from "
        "grimme-lab/GMTKN55 S22/20/struc.xyz."
    ),
)
