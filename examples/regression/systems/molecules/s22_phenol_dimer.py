"""S22 #22 — phenol dimer — OH⋯π plus dispersion.

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
    id="s22_phenol_dimer",
    family="molecule_noncovalent_mixed",
    atoms=(
        AtomCart(symbol="C", z=6, xyz_ang=(-2.04861867,  0.94853107, -0.14411943)),
        AtomCart(symbol="C", z=6, xyz_ang=(-1.50459365,  0.03277290,  0.75922472)),
        AtomCart(symbol="C", z=6, xyz_ang=(-2.18909204, -1.14482488,  1.05259948)),
        AtomCart(symbol="C", z=6, xyz_ang=(-3.41583424, -1.41845818,  0.45381805)),
        AtomCart(symbol="C", z=6, xyz_ang=(-3.95588618, -0.49916975, -0.44487141)),
        AtomCart(symbol="C", z=6, xyz_ang=(-3.27856295,  0.67764598, -0.74538130)),
        AtomCart(symbol="C", z=6, xyz_ang=( 1.99546282,  0.97118948,  0.11378065)),
        AtomCart(symbol="C", z=6, xyz_ang=( 1.54889056,  0.25437096, -0.99318407)),
        AtomCart(symbol="C", z=6, xyz_ang=( 2.20022479, -0.92229184, -1.34857992)),
        AtomCart(symbol="C", z=6, xyz_ang=( 3.29005578, -1.38187581, -0.61063226)),
        AtomCart(symbol="C", z=6, xyz_ang=( 3.72817241, -0.65500537,  0.49287557)),
        AtomCart(symbol="C", z=6, xyz_ang=( 3.08097200,  0.52303483,  0.86028084)),
        AtomCart(symbol="H", z=1, xyz_ang=(-3.68467968,  1.39811982, -1.44152794)),
        AtomCart(symbol="H", z=1, xyz_ang=(-0.56532471,  2.14933739, -0.04222940)),
        AtomCart(symbol="H", z=1, xyz_ang=( 1.74277264,  2.53343518,  1.19394289)),
        AtomCart(symbol="H", z=1, xyz_ang=(-0.55216361,  0.24134186,  1.22848818)),
        AtomCart(symbol="H", z=1, xyz_ang=(-1.75662650, -1.84746085,  1.75207373)),
        AtomCart(symbol="H", z=1, xyz_ang=(-3.94397992, -2.33270238,  0.68402638)),
        AtomCart(symbol="H", z=1, xyz_ang=(-4.90858668, -0.69760919, -0.91690064)),
        AtomCart(symbol="H", z=1, xyz_ang=( 0.70303895,  0.62148336, -1.55762724)),
        AtomCart(symbol="H", z=1, xyz_ang=( 1.85063433, -1.48028820, -2.20595342)),
        AtomCart(symbol="H", z=1, xyz_ang=( 3.79151137, -2.29646936, -0.89243535)),
        AtomCart(symbol="H", z=1, xyz_ang=( 4.57225184, -1.00032544,  1.07349529)),
        AtomCart(symbol="H", z=1, xyz_ang=( 3.41837385,  1.08772279,  1.72118093)),
        AtomCart(symbol="O", z=8, xyz_ang=(-1.43001724,  2.11453775, -0.47888924)),
        AtomCart(symbol="O", z=8, xyz_ang=( 1.31160472,  2.12295792,  0.43654492)),
    ),
    charge=0,
    multiplicity=1,
    default_conv_tol_energy=1e-9,
    default_max_iter=80,
    notes=(
        "S22 system #22 — phenol dimer — OH⋯π plus dispersion. Geometry from grimme-lab/GMTKN55 "
        "S22/22/struc.xyz."
    ),
    citation=(
        "Jurečka, Šponer, Černý, Hobza, *Phys. Chem. Chem. Phys.* "
        "**2006**, 8, 1985 (S22 origin, DOI 10.1039/B600027D); "
        "Marshall, Burns, Sherrill, *J. Chem. Phys.* **2011**, 135, "
        "194102 (S22A revised CCSD(T)/CBS, DOI 10.1063/1.3659142). "
        "Geometry: grimme-lab/GMTKN55 S22/22/struc.xyz."
    ),
)
