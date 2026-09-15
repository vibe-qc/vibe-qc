"""S22 #21 — indole⋯benzene, T-shaped — protein-aromatic mixed.

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
    id="s22_indole_benzene_t",
    family="molecule_noncovalent_mixed",
    atoms=(
        AtomCart(symbol="C", z=6, xyz_ang=( 2.40026833,  1.58653755,  0.00000000)),
        AtomCart(symbol="C", z=6, xyz_ang=( 2.60137819,  0.91937635, -1.20829211)),
        AtomCart(symbol="C", z=6, xyz_ang=( 3.00615100, -0.41522136, -1.20836504)),
        AtomCart(symbol="C", z=6, xyz_ang=( 3.20975364, -1.08220865,  0.00000000)),
        AtomCart(symbol="C", z=6, xyz_ang=( 3.00615100, -0.41522136,  1.20836504)),
        AtomCart(symbol="C", z=6, xyz_ang=( 2.60137819,  0.91937635,  1.20829211)),
        AtomCart(symbol="C", z=6, xyz_ang=(-0.62774439, -2.12780021,  0.00000000)),
        AtomCart(symbol="C", z=6, xyz_ang=(-2.00150812, -2.21992766,  0.00000000)),
        AtomCart(symbol="C", z=6, xyz_ang=(-2.50486451, -0.88556089,  0.00000000)),
        AtomCart(symbol="C", z=6, xyz_ang=(-1.37569771, -0.01888894,  0.00000000)),
        AtomCart(symbol="C", z=6, xyz_ang=(-1.50123281,  1.37328955,  0.00000000)),
        AtomCart(symbol="C", z=6, xyz_ang=(-2.78428288,  1.89816791,  0.00000000)),
        AtomCart(symbol="C", z=6, xyz_ang=(-3.91708414,  1.05900163,  0.00000000)),
        AtomCart(symbol="C", z=6, xyz_ang=(-3.79144987, -0.32019864,  0.00000000)),
        AtomCart(symbol="H", z=1, xyz_ang=( 2.09077232,  2.62265881,  0.00000000)),
        AtomCart(symbol="H", z=1, xyz_ang=( 2.43954472,  1.43521355, -2.14459070)),
        AtomCart(symbol="H", z=1, xyz_ang=( 3.15866890, -0.93361854, -2.14483849)),
        AtomCart(symbol="H", z=1, xyz_ang=( 3.52518297, -2.11663039,  0.00000000)),
        AtomCart(symbol="H", z=1, xyz_ang=( 3.15866890, -0.93361854,  2.14483849)),
        AtomCart(symbol="H", z=1, xyz_ang=( 2.43954472,  1.43521355,  2.14459070)),
        AtomCart(symbol="H", z=1, xyz_ang=( 0.69489266, -0.47436436,  0.00000000)),
        AtomCart(symbol="H", z=1, xyz_ang=( 0.11937043, -2.90379585,  0.00000000)),
        AtomCart(symbol="H", z=1, xyz_ang=(-2.57020874, -3.13408379,  0.00000000)),
        AtomCart(symbol="H", z=1, xyz_ang=(-0.63050549,  2.01547500,  0.00000000)),
        AtomCart(symbol="H", z=1, xyz_ang=(-2.91938992,  2.97130908,  0.00000000)),
        AtomCart(symbol="H", z=1, xyz_ang=(-4.90223269,  1.50545996,  0.00000000)),
        AtomCart(symbol="H", z=1, xyz_ang=(-4.66965181, -0.95276954,  0.00000000)),
        AtomCart(symbol="N", z=7, xyz_ang=(-0.25587288, -0.80717056,  0.00000000)),
    ),
    charge=0,
    multiplicity=1,
    default_conv_tol_energy=1e-9,
    default_max_iter=80,
    notes=(
        "S22 system #21 — indole⋯benzene, T-shaped — protein-aromatic mixed. Geometry from grimme-lab/GMTKN55 "
        "S22/21/struc.xyz."
    ),
    citation=(
        "Jurečka, Šponer, Černý, Hobza, *Phys. Chem. Chem. Phys.* "
        "**2006**, 8, 1985 (S22 origin, DOI 10.1039/B600027D); "
        "Marshall, Burns, Sherrill, *J. Chem. Phys.* **2011**, 135, "
        "194102 (S22A revised CCSD(T)/CBS, DOI 10.1063/1.3659142). "
        "Geometry: grimme-lab/GMTKN55 S22/21/struc.xyz."
    ),
)
