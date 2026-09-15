"""S22 #17 — benzene⋯water — OH⋯π / mixed dispersion + electrostatic.

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
    id="s22_benzene_water",
    family="molecule_noncovalent_mixed",
    atoms=(
        AtomCart(symbol="C", z=6, xyz_ang=( 0.76502022, -0.57333610, -1.20754294)),
        AtomCart(symbol="C", z=6, xyz_ang=( 0.46281235,  0.78759239, -1.20790440)),
        AtomCart(symbol="C", z=6, xyz_ang=( 0.31206761,  1.46840926,  0.00000000)),
        AtomCart(symbol="C", z=6, xyz_ang=( 0.46281235,  0.78759239,  1.20790440)),
        AtomCart(symbol="C", z=6, xyz_ang=( 0.76502022, -0.57333610,  1.20754294)),
        AtomCart(symbol="C", z=6, xyz_ang=( 0.91655958, -1.25341023,  0.00000000)),
        AtomCart(symbol="H", z=1, xyz_ang=( 0.88107738, -1.10105382, -2.14414887)),
        AtomCart(symbol="H", z=1, xyz_ang=( 0.34179792,  1.31476104, -2.14405525)),
        AtomCart(symbol="H", z=1, xyz_ang=( 0.07626763,  2.52369294,  0.00000000)),
        AtomCart(symbol="H", z=1, xyz_ang=( 0.34179792,  1.31476104,  2.14405525)),
        AtomCart(symbol="H", z=1, xyz_ang=( 0.88107738, -1.10105382,  2.14414887)),
        AtomCart(symbol="H", z=1, xyz_ang=( 1.15341508, -2.30861581,  0.00000000)),
        AtomCart(symbol="H", z=1, xyz_ang=(-2.63850373, -1.18253187,  0.00000000)),
        AtomCart(symbol="H", z=1, xyz_ang=(-1.91710251,  0.13446260,  0.00000000)),
        AtomCart(symbol="O", z=8, xyz_ang=(-2.80411940, -0.23793391,  0.00000000)),
    ),
    charge=0,
    multiplicity=1,
    default_conv_tol_energy=1e-9,
    default_max_iter=80,
    notes=(
        "S22 system #17 — benzene⋯water — OH⋯π / mixed dispersion + electrostatic. Geometry from grimme-lab/GMTKN55 "
        "S22/17/struc.xyz."
    ),
    citation=(
        "Jurečka, Šponer, Černý, Hobza, *Phys. Chem. Chem. Phys.* "
        "**2006**, 8, 1985 (S22 origin, DOI 10.1039/B600027D); "
        "Marshall, Burns, Sherrill, *J. Chem. Phys.* **2011**, 135, "
        "194102 (S22A revised CCSD(T)/CBS, DOI 10.1063/1.3659142). "
        "Geometry: grimme-lab/GMTKN55 S22/17/struc.xyz."
    ),
)
