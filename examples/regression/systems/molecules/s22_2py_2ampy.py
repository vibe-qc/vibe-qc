"""S22 #06 — 2-pyridoxine ⋯ 2-aminopyridine — hydrogen-bonded heterocyclic pair.

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
    id="s22_2py_2ampy",
    family="molecule_noncovalent_hb",
    atoms=(
        AtomCart(symbol="O", z=8, xyz_ang=(-1.47861893, -2.01917716, -0.38424748)),
        AtomCart(symbol="N", z=7, xyz_ang=(-1.54525258,  0.23084314,  0.00228882)),
        AtomCart(symbol="N", z=7, xyz_ang=( 1.35176487,  0.23063064, -0.03289209)),
        AtomCart(symbol="N", z=7, xyz_ang=( 1.30731557, -2.04164411,  0.40287375)),
        AtomCart(symbol="C", z=6, xyz_ang=(-2.19894800,  1.39736545,  0.21689709)),
        AtomCart(symbol="C", z=6, xyz_ang=(-2.15838106, -0.99708925, -0.20688273)),
        AtomCart(symbol="C", z=6, xyz_ang=(-3.59660129, -0.93853500, -0.19269983)),
        AtomCart(symbol="C", z=6, xyz_ang=(-4.26673812,  0.23632724,  0.01915473)),
        AtomCart(symbol="C", z=6, xyz_ang=( 2.03442339, -0.91368497,  0.15116867)),
        AtomCart(symbol="C", z=6, xyz_ang=( 3.44276254, -0.93494962,  0.13756146)),
        AtomCart(symbol="C", z=6, xyz_ang=( 4.13759377,  0.24023863, -0.06953420)),
        AtomCart(symbol="C", z=6, xyz_ang=( 3.42897465,  1.42816211, -0.26191765)),
        AtomCart(symbol="C", z=6, xyz_ang=( 2.04701712,  1.36199301, -0.23447875)),
        AtomCart(symbol="C", z=6, xyz_ang=(-3.56425797,  1.44497180,  0.23313400)),
        AtomCart(symbol="H", z=1, xyz_ang=(-1.57391717,  2.26507054,  0.37156063)),
        AtomCart(symbol="H", z=1, xyz_ang=(-4.12112097, -1.86818548, -0.35486829)),
        AtomCart(symbol="H", z=1, xyz_ang=(-0.50752391,  0.22787304, -0.00958748)),
        AtomCart(symbol="H", z=1, xyz_ang=(-5.34880336,  0.23740312,  0.02420063)),
        AtomCart(symbol="H", z=1, xyz_ang=( 3.96492468, -1.86947590,  0.29074710)),
        AtomCart(symbol="H", z=1, xyz_ang=( 5.21894693,  0.23326123, -0.08327620)),
        AtomCart(symbol="H", z=1, xyz_ang=( 3.93009614,  2.36909212, -0.42994659)),
        AtomCart(symbol="H", z=1, xyz_ang=( 1.45299100,  2.25604464, -0.38399790)),
        AtomCart(symbol="H", z=1, xyz_ang=(-4.07303166,  2.37941702,  0.40450025)),
        AtomCart(symbol="H", z=1, xyz_ang=( 1.78847475, -2.91461775,  0.27709730)),
        AtomCart(symbol="H", z=1, xyz_ang=( 0.32790963, -2.04133450,  0.11314476)),
    ),
    charge=0,
    multiplicity=1,
    default_conv_tol_energy=1e-9,
    default_max_iter=80,
    notes=(
        "S22 system #06 — 2-pyridoxine ⋯ 2-aminopyridine — hydrogen-bonded heterocyclic pair. Geometry from grimme-lab/GMTKN55 "
        "S22/06/struc.xyz."
    ),
    citation=(
        "Jurečka, Šponer, Černý, Hobza, *Phys. Chem. Chem. Phys.* "
        "**2006**, 8, 1985 (S22 origin, DOI 10.1039/B600027D); "
        "Marshall, Burns, Sherrill, *J. Chem. Phys.* **2011**, 135, "
        "194102 (S22A revised CCSD(T)/CBS, DOI 10.1063/1.3659142). "
        "Geometry: grimme-lab/GMTKN55 S22/06/struc.xyz."
    ),
)
