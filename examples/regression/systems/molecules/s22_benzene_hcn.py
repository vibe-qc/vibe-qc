"""S22 #19 — benzene⋯HCN — CH⋯π / mixed.

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
    id="s22_benzene_hcn",
    family="molecule_noncovalent_mixed",
    atoms=(
        AtomCart(symbol="C", z=6, xyz_ang=(-0.71414688, -0.65965751,  1.20770216)),
        AtomCart(symbol="C", z=6, xyz_ang=(-1.41090694, -0.63458738,  0.00000000)),
        AtomCart(symbol="C", z=6, xyz_ang=(-0.71414688, -0.65965751, -1.20770216)),
        AtomCart(symbol="C", z=6, xyz_ang=( 0.67959269, -0.70974503, -1.20786555)),
        AtomCart(symbol="C", z=6, xyz_ang=( 1.37660577, -0.73478678,  0.00000000)),
        AtomCart(symbol="C", z=6, xyz_ang=( 0.67959269, -0.70974503,  1.20786555)),
        AtomCart(symbol="C", z=6, xyz_ang=( 0.07082373,  2.70147035,  0.00000000)),
        AtomCart(symbol="H", z=1, xyz_ang=(-2.49129292, -0.59294047,  0.00000000)),
        AtomCart(symbol="H", z=1, xyz_ang=(-1.25432120, -0.63786252, -2.14405122)),
        AtomCart(symbol="H", z=1, xyz_ang=( 1.21991594, -0.72730982, -2.14425687)),
        AtomCart(symbol="H", z=1, xyz_ang=( 2.45721670, -0.77221635,  0.00000000)),
        AtomCart(symbol="H", z=1, xyz_ang=( 1.21991594, -0.72730982,  2.14425687)),
        AtomCart(symbol="H", z=1, xyz_ang=(-1.25432120, -0.63786252,  2.14405122)),
        AtomCart(symbol="H", z=1, xyz_ang=( 0.14325695,  1.63605091,  0.00000000)),
        AtomCart(symbol="N", z=7, xyz_ang=(-0.00778439,  3.86615948,  0.00000000)),
    ),
    charge=0,
    multiplicity=1,
    default_conv_tol_energy=1e-9,
    default_max_iter=80,
    notes=(
        "S22 system #19 — benzene⋯HCN — CH⋯π / mixed. Geometry from grimme-lab/GMTKN55 "
        "S22/19/struc.xyz."
    ),
    citation=(
        "Jurečka, Šponer, Černý, Hobza, *Phys. Chem. Chem. Phys.* "
        "**2006**, 8, 1985 (S22 origin, DOI 10.1039/B600027D); "
        "Marshall, Burns, Sherrill, *J. Chem. Phys.* **2011**, 135, "
        "194102 (S22A revised CCSD(T)/CBS, DOI 10.1063/1.3659142). "
        "Geometry: grimme-lab/GMTKN55 S22/19/struc.xyz."
    ),
)
