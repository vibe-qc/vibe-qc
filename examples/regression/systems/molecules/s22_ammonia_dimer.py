"""S22 #01 — ammonia dimer (NH3)2. Weak hydrogen-bond benchmark.

Geometry from the GMTKN55 distribution of the S22 test set (Jurečka,
Šponer, Černý, Hobza 2006), in Ångström. Revised S22A reference
interaction energy is -3.13 kcal/mol (Marshall et al. 2011).
"""
from __future__ import annotations

from ...core.spec import AtomCart, MoleculeSpec

SPEC = MoleculeSpec(
    id="s22_ammonia_dimer",
    family="molecule_noncovalent_hb",
    atoms=(
        AtomCart(symbol="N", z=7, xyz_ang=(-1.57871846,  -0.04661102,   0.00000000)),
        AtomCart(symbol="N", z=7, xyz_ang=( 1.57871846,   0.04661102,   0.00000000)),
        AtomCart(symbol="H", z=1, xyz_ang=(-2.15862159,   0.13639604,   0.80956523)),
        AtomCart(symbol="H", z=1, xyz_ang=(-0.84947124,   0.65819316,   0.00000000)),
        AtomCart(symbol="H", z=1, xyz_ang=(-2.15862159,   0.13639604,  -0.80956523)),
        AtomCart(symbol="H", z=1, xyz_ang=( 2.15862159,  -0.13639604,  -0.80956523)),
        AtomCart(symbol="H", z=1, xyz_ang=( 0.84947124,  -0.65819316,   0.00000000)),
        AtomCart(symbol="H", z=1, xyz_ang=( 2.15862159,  -0.13639604,   0.80956523)),
    ),
    charge=0,
    multiplicity=1,
    default_conv_tol_energy=1e-9,
    default_max_iter=80,
    notes=(
        "S22 system #01 — (NH3)2 weak hydrogen-bonded dimer. Smallest "
        "S22 entry beyond the methane dimer; NH3 lone pair as donor / "
        "N-H as acceptor."
    ),
    citation=(
        "Jurečka, Šponer, Černý, Hobza, *Phys. Chem. Chem. Phys.* "
        "**2006**, 8, 1985 (S22 origin, DOI 10.1039/B600027D); "
        "Marshall, Burns, Sherrill, *J. Chem. Phys.* **2011**, 135, "
        "194102 (S22A revised CCSD(T)/CBS references, DOI 10.1063/"
        "1.3659142). Geometry from grimme-lab/GMTKN55 S22/01/struc.xyz."
    ),
)
