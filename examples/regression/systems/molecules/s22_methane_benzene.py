"""S22 #10 — methane⋯benzene — CH⋯π dispersion test.

Geometry from the GMTKN55 distribution of the S22 test set (Jurečka,
Šponer, Černý, Hobza 2006), in Ångström. Revised S22A reference
interaction energies in Marshall, Burns, Sherrill 2011 (the canonical
post-2011 CCSD(T)/CBS reference).

For the cross-code parity suite this is a single-determinant SCF on
the dimer geometry — interaction-energy benchmarking (E_AB − E_A −
E_B + counterpoise) is wave-2 multi-component-machinery work.
"""
from __future__ import annotations

from ...core.spec import AtomCart, MoleculeSpec

SPEC = MoleculeSpec(
    id="s22_methane_benzene",
    family="molecule_noncovalent_dispersion",
    atoms=(
        AtomCart(symbol="C", z=6, xyz_ang=( 1.39321820,  0.03629132, -1.09376784)),
        AtomCart(symbol="C", z=6, xyz_ang=( 0.72803660, -1.18840183, -1.09378923)),
        AtomCart(symbol="C", z=6, xyz_ang=(-0.66517989, -1.22470802, -1.09376784)),
        AtomCart(symbol="C", z=6, xyz_ang=(-1.39320447, -0.03629727, -1.09378923)),
        AtomCart(symbol="C", z=6, xyz_ang=(-0.72803831,  1.18841669, -1.09376784)),
        AtomCart(symbol="C", z=6, xyz_ang=( 0.66516787,  1.22469910, -1.09378923)),
        AtomCart(symbol="C", z=6, xyz_ang=( 0.00000000,  0.00000000,  2.62213305)),
        AtomCart(symbol="H", z=1, xyz_ang=( 1.29295913, -2.11054147, -1.09222761)),
        AtomCart(symbol="H", z=1, xyz_ang=(-1.18132318, -2.17500873, -1.09221154)),
        AtomCart(symbol="H", z=1, xyz_ang=(-2.47426210, -0.06446472, -1.09222761)),
        AtomCart(symbol="H", z=1, xyz_ang=(-1.29295122,  2.11056025, -1.09221154)),
        AtomCart(symbol="H", z=1, xyz_ang=( 1.18130297,  2.17500619, -1.09222761)),
        AtomCart(symbol="H", z=1, xyz_ang=( 2.47427440,  0.06444848, -1.09221154)),
        AtomCart(symbol="H", z=1, xyz_ang=( 0.58687774,  0.83817445,  2.98589091)),
        AtomCart(symbol="H", z=1, xyz_ang=(-1.01931923,  0.08916380,  2.98589091)),
        AtomCart(symbol="H", z=1, xyz_ang=( 0.00000000,  0.00000000,  1.53618291)),
        AtomCart(symbol="H", z=1, xyz_ang=( 0.43244150, -0.92733825,  2.98589091)),
    ),
    charge=0,
    multiplicity=1,
    default_conv_tol_energy=1e-9,
    default_max_iter=80,
    notes=(
        "S22 system #10 — methane⋯benzene — CH⋯π dispersion test. Geometry from grimme-lab/GMTKN55 "
        "S22/10/struc.xyz."
    ),
    citation=(
        "Jurečka, Šponer, Černý, Hobza, *Phys. Chem. Chem. Phys.* "
        "**2006**, 8, 1985 (S22 origin, DOI 10.1039/B600027D); "
        "Marshall, Burns, Sherrill, *J. Chem. Phys.* **2011**, 135, "
        "194102 (S22A revised CCSD(T)/CBS, DOI 10.1063/1.3659142). "
        "Geometry: grimme-lab/GMTKN55 S22/10/struc.xyz."
    ),
)
