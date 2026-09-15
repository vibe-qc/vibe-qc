"""S22 #18 — benzene⋯ammonia — NH⋯π / mixed.

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
    id="s22_benzene_ammonia",
    family="molecule_noncovalent_mixed",
    atoms=(
        AtomCart(symbol="C", z=6, xyz_ang=(-0.80893998,  0.73922310, -1.20710830)),
        AtomCart(symbol="C", z=6, xyz_ang=(-1.49580341,  0.61989005,  0.00000000)),
        AtomCart(symbol="C", z=6, xyz_ang=(-0.80893998,  0.73922310,  1.20710830)),
        AtomCart(symbol="C", z=6, xyz_ang=( 0.56456836,  0.97798446,  1.20707386)),
        AtomCart(symbol="C", z=6, xyz_ang=( 1.25138498,  1.09710132,  0.00000000)),
        AtomCart(symbol="C", z=6, xyz_ang=( 0.56456836,  0.97798446, -1.20707386)),
        AtomCart(symbol="H", z=1, xyz_ang=(-1.34160864,  0.64397615, -2.14328993)),
        AtomCart(symbol="H", z=1, xyz_ang=(-2.55988007,  0.42858260,  0.00000000)),
        AtomCart(symbol="H", z=1, xyz_ang=(-1.34160864,  0.64397615,  2.14328993)),
        AtomCart(symbol="H", z=1, xyz_ang=( 1.09714214,  1.07083319,  2.14369568)),
        AtomCart(symbol="H", z=1, xyz_ang=( 2.31670047,  1.28297592,  0.00000000)),
        AtomCart(symbol="H", z=1, xyz_ang=( 1.09714214,  1.07083319, -2.14369568)),
        AtomCart(symbol="H", z=1, xyz_ang=( 0.68989100, -2.92260407, -0.80607315)),
        AtomCart(symbol="H", z=1, xyz_ang=( 0.68989100, -2.92260407,  0.80607315)),
        AtomCart(symbol="H", z=1, xyz_ang=(-0.02524203, -1.72159601,  0.00000000)),
        AtomCart(symbol="N", z=7, xyz_ang=( 0.11073430, -2.72577955,  0.00000000)),
    ),
    charge=0,
    multiplicity=1,
    default_conv_tol_energy=1e-9,
    default_max_iter=80,
    notes=(
        "S22 system #18 — benzene⋯ammonia — NH⋯π / mixed. Geometry from grimme-lab/GMTKN55 "
        "S22/18/struc.xyz."
    ),
    citation=(
        "Jurečka, Šponer, Černý, Hobza, *Phys. Chem. Chem. Phys.* "
        "**2006**, 8, 1985 (S22 origin, DOI 10.1039/B600027D); "
        "Marshall, Burns, Sherrill, *J. Chem. Phys.* **2011**, 135, "
        "194102 (S22A revised CCSD(T)/CBS, DOI 10.1063/1.3659142). "
        "Geometry: grimme-lab/GMTKN55 S22/18/struc.xyz."
    ),
)
