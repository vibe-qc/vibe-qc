"""S22 #05 — uracil dimer, hydrogen-bonded — two cooperative N-H···O=C bonds.

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
    id="s22_uracil_dimer_hb",
    family="molecule_noncovalent_hb",
    atoms=(
        AtomCart(symbol="O", z=8, xyz_ang=(-1.46633196,  1.01216958,  0.00000000)),
        AtomCart(symbol="O", z=8, xyz_ang=(-0.59722309,  5.48640841,  0.00000000)),
        AtomCart(symbol="O", z=8, xyz_ang=( 1.46633196, -1.01216958,  0.00000000)),
        AtomCart(symbol="O", z=8, xyz_ang=( 0.59722309, -5.48640841,  0.00000000)),
        AtomCart(symbol="C", z=6, xyz_ang=( 1.27690391,  4.00617740,  0.00000000)),
        AtomCart(symbol="C", z=6, xyz_ang=(-0.12860053,  4.36215614,  0.00000000)),
        AtomCart(symbol="C", z=6, xyz_ang=(-0.62814655,  1.91426832,  0.00000000)),
        AtomCart(symbol="C", z=6, xyz_ang=( 0.62814655, -1.91426832,  0.00000000)),
        AtomCart(symbol="C", z=6, xyz_ang=(-1.63672943, -2.70527729,  0.00000000)),
        AtomCart(symbol="C", z=6, xyz_ang=(-1.27690391, -4.00617740,  0.00000000)),
        AtomCart(symbol="C", z=6, xyz_ang=( 0.12860053, -4.36215614,  0.00000000)),
        AtomCart(symbol="C", z=6, xyz_ang=( 1.63672943,  2.70527729,  0.00000000)),
        AtomCart(symbol="N", z=7, xyz_ang=( 0.72050948,  1.68826925,  0.00000000)),
        AtomCart(symbol="N", z=7, xyz_ang=(-0.97772326,  3.23964420,  0.00000000)),
        AtomCart(symbol="N", z=7, xyz_ang=(-0.72050948, -1.68826925,  0.00000000)),
        AtomCart(symbol="N", z=7, xyz_ang=( 0.97772326, -3.23964420,  0.00000000)),
        AtomCart(symbol="H", z=1, xyz_ang=( 1.02325179,  0.70618217,  0.00000000)),
        AtomCart(symbol="H", z=1, xyz_ang=(-1.97002736,  3.43238599,  0.00000000)),
        AtomCart(symbol="H", z=1, xyz_ang=( 2.01035092,  4.79386582,  0.00000000)),
        AtomCart(symbol="H", z=1, xyz_ang=( 2.66906269,  2.38834229,  0.00000000)),
        AtomCart(symbol="H", z=1, xyz_ang=(-2.01035092, -4.79386582,  0.00000000)),
        AtomCart(symbol="H", z=1, xyz_ang=(-1.02325179, -0.70618217,  0.00000000)),
        AtomCart(symbol="H", z=1, xyz_ang=( 1.97002736, -3.43238599,  0.00000000)),
        AtomCart(symbol="H", z=1, xyz_ang=(-2.66906269, -2.38834229,  0.00000000)),
    ),
    charge=0,
    multiplicity=1,
    default_conv_tol_energy=1e-9,
    default_max_iter=80,
    notes=(
        "S22 system #05 — uracil dimer, hydrogen-bonded — two cooperative N-H···O=C bonds. Geometry from grimme-lab/GMTKN55 "
        "S22/05/struc.xyz."
    ),
    citation=(
        "Jurečka, Šponer, Černý, Hobza, *Phys. Chem. Chem. Phys.* "
        "**2006**, 8, 1985 (S22 origin, DOI 10.1039/B600027D); "
        "Marshall, Burns, Sherrill, *J. Chem. Phys.* **2011**, 135, "
        "194102 (S22A revised CCSD(T)/CBS, DOI 10.1063/1.3659142). "
        "Geometry: grimme-lab/GMTKN55 S22/05/struc.xyz."
    ),
)
