"""S22 #03 — formic acid dimer (HCOOH)₂ — doubly H-bonded carboxylic-acid pair.

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
    id="s22_formic_acid_dimer",
    family="molecule_noncovalent_hb",
    atoms=(
        AtomCart(symbol="C", z=6, xyz_ang=(-1.88889652, -0.17969205,  0.00000000)),
        AtomCart(symbol="C", z=6, xyz_ang=( 1.88889652,  0.17969205,  0.00000000)),
        AtomCart(symbol="O", z=8, xyz_ang=(-1.17043534, -1.16659032,  0.00000000)),
        AtomCart(symbol="O", z=8, xyz_ang=(-1.49328045,  1.07368936,  0.00000000)),
        AtomCart(symbol="O", z=8, xyz_ang=( 1.49328045, -1.07368936,  0.00000000)),
        AtomCart(symbol="O", z=8, xyz_ang=( 1.17043534,  1.16659032,  0.00000000)),
        AtomCart(symbol="H", z=1, xyz_ang=(-0.49883314,  1.10719530,  0.00000000)),
        AtomCart(symbol="H", z=1, xyz_ang=(-2.97948880, -0.25882908,  0.00000000)),
        AtomCart(symbol="H", z=1, xyz_ang=( 2.97948880,  0.25882908,  0.00000000)),
        AtomCart(symbol="H", z=1, xyz_ang=( 0.49883314, -1.10719530,  0.00000000)),
    ),
    charge=0,
    multiplicity=1,
    default_conv_tol_energy=1e-9,
    default_max_iter=80,
    notes=(
        "S22 system #03 — formic acid dimer (HCOOH)₂ — doubly H-bonded carboxylic-acid pair. Geometry from grimme-lab/GMTKN55 "
        "S22/03/struc.xyz."
    ),
    citation=(
        "Jurečka, Šponer, Černý, Hobza, *Phys. Chem. Chem. Phys.* "
        "**2006**, 8, 1985 (S22 origin, DOI 10.1039/B600027D); "
        "Marshall, Burns, Sherrill, *J. Chem. Phys.* **2011**, 135, "
        "194102 (S22A revised CCSD(T)/CBS, DOI 10.1063/1.3659142). "
        "Geometry: grimme-lab/GMTKN55 S22/03/struc.xyz."
    ),
)
