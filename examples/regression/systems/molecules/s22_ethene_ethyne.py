"""S22 #16 — ethene⋯ethyne — mixed π/H-bond character.

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
    id="s22_ethene_ethyne",
    family="molecule_noncovalent_mixed",
    atoms=(
        AtomCart(symbol="C", z=6, xyz_ang=( 0.00000000, -0.66757818, -1.76775602)),
        AtomCart(symbol="C", z=6, xyz_ang=( 0.00000000,  0.66757818, -1.76775602)),
        AtomCart(symbol="C", z=6, xyz_ang=( 0.00000000,  0.00000000,  3.25740731)),
        AtomCart(symbol="C", z=6, xyz_ang=( 0.00000000,  0.00000000,  2.05014396)),
        AtomCart(symbol="H", z=1, xyz_ang=(-0.92362127,  1.23225337, -1.76928212)),
        AtomCart(symbol="H", z=1, xyz_ang=( 0.92362127,  1.23225337, -1.76928212)),
        AtomCart(symbol="H", z=1, xyz_ang=( 0.92362127, -1.23225337, -1.76928212)),
        AtomCart(symbol="H", z=1, xyz_ang=(-0.92362127, -1.23225337, -1.76928212)),
        AtomCart(symbol="H", z=1, xyz_ang=( 0.00000000,  0.00000000,  0.98425568)),
        AtomCart(symbol="H", z=1, xyz_ang=( 0.00000000,  0.00000000,  4.32083360)),
    ),
    charge=0,
    multiplicity=1,
    default_conv_tol_energy=1e-9,
    default_max_iter=80,
    notes=(
        "S22 system #16 — ethene⋯ethyne — mixed π/H-bond character. Geometry from grimme-lab/GMTKN55 "
        "S22/16/struc.xyz."
    ),
    citation=(
        "Jurečka, Šponer, Černý, Hobza, *Phys. Chem. Chem. Phys.* "
        "**2006**, 8, 1985 (S22 origin, DOI 10.1039/B600027D); "
        "Marshall, Burns, Sherrill, *J. Chem. Phys.* **2011**, 135, "
        "194102 (S22A revised CCSD(T)/CBS, DOI 10.1063/1.3659142). "
        "Geometry: grimme-lab/GMTKN55 S22/16/struc.xyz."
    ),
)
