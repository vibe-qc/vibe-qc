"""S22 #15 — adenine⋯thymine, stacked — DNA base-stacking reference.

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
    id="s22_adenine_thymine_stack",
    family="molecule_noncovalent_dispersion",
    atoms=(
        AtomCart(symbol="N", z=7, xyz_ang=( 0.17587859,  2.64192566, -0.74008019)),
        AtomCart(symbol="N", z=7, xyz_ang=(-1.70113508,  1.95307392,  0.29442611)),
        AtomCart(symbol="N", z=7, xyz_ang=(-1.52213235,  0.06730897,  2.67581664)),
        AtomCart(symbol="N", z=7, xyz_ang=( 0.78234036,  0.16500940,  2.35762175)),
        AtomCart(symbol="N", z=7, xyz_ang=( 1.83755521,  1.45928795,  0.60589211)),
        AtomCart(symbol="N", z=7, xyz_ang=( 1.17203807, -0.41281378, -2.11223925)),
        AtomCart(symbol="N", z=7, xyz_ang=(-1.09911479, -0.40330145, -1.80637075)),
        AtomCart(symbol="H", z=1, xyz_ang=(-1.34229819, -0.72439474,  3.27043043)),
        AtomCart(symbol="H", z=1, xyz_ang=(-2.39529688,  0.05626337,  2.17303430)),
        AtomCart(symbol="H", z=1, xyz_ang=( 2.80261100,  0.31392714,  2.01149050)),
        AtomCart(symbol="H", z=1, xyz_ang=( 0.75758463,  3.06489105, -1.44477891)),
        AtomCart(symbol="H", z=1, xyz_ang=( 2.32245470, -1.63199285, -0.88120628)),
        AtomCart(symbol="H", z=1, xyz_ang=( 0.07234767, -2.34052151,  1.66430034)),
        AtomCart(symbol="H", z=1, xyz_ang=(-0.26359203, -3.64195657,  0.52962168)),
        AtomCart(symbol="H", z=1, xyz_ang=( 1.40782179, -3.12219183,  0.81703788)),
        AtomCart(symbol="H", z=1, xyz_ang=(-1.76286366,  3.25811606, -1.39041916)),
        AtomCart(symbol="H", z=1, xyz_ang=(-2.00482911, -0.01508636, -2.03290489)),
        AtomCart(symbol="H", z=1, xyz_ang=( 1.98190666, -0.04093195, -2.57978680)),
        AtomCart(symbol="C", z=6, xyz_ang=( 0.25417481, -1.78886473, -0.38738587)),
        AtomCart(symbol="C", z=6, xyz_ang=( 0.37870645, -2.78286444,  0.71779406)),
        AtomCart(symbol="C", z=6, xyz_ang=(-0.59314853,  1.40652181,  0.89586298)),
        AtomCart(symbol="C", z=6, xyz_ang=( 0.59179593,  1.81307191,  0.27207021)),
        AtomCart(symbol="C", z=6, xyz_ang=( 1.83181253,  0.64237363,  1.66247447)),
        AtomCart(symbol="C", z=6, xyz_ang=(-1.07189429, -1.29472588, -0.72830767)),
        AtomCart(symbol="C", z=6, xyz_ang=(-0.44955959,  0.52655679,  1.98290665)),
        AtomCart(symbol="C", z=6, xyz_ang=( 1.30963082, -1.31859970, -1.08939530)),
        AtomCart(symbol="C", z=6, xyz_ang=(-0.03495267,  0.11590948, -2.51070478)),
        AtomCart(symbol="C", z=6, xyz_ang=(-1.18828017,  2.68083263, -0.68548925)),
        AtomCart(symbol="O", z=8, xyz_ang=(-0.14321039,  0.95778651, -3.38743740)),
        AtomCart(symbol="O", z=8, xyz_ang=(-2.10635148, -1.60461048, -0.15427361)),
    ),
    charge=0,
    multiplicity=1,
    default_conv_tol_energy=1e-9,
    default_max_iter=80,
    notes=(
        "S22 system #15 — adenine⋯thymine, stacked — DNA base-stacking reference. Geometry from grimme-lab/GMTKN55 "
        "S22/15/struc.xyz."
    ),
    citation=(
        "Jurečka, Šponer, Černý, Hobza, *Phys. Chem. Chem. Phys.* "
        "**2006**, 8, 1985 (S22 origin, DOI 10.1039/B600027D); "
        "Marshall, Burns, Sherrill, *J. Chem. Phys.* **2011**, 135, "
        "194102 (S22A revised CCSD(T)/CBS, DOI 10.1063/1.3659142). "
        "Geometry: grimme-lab/GMTKN55 S22/15/struc.xyz."
    ),
)
