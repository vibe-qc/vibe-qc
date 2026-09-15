"""S22 #02 — water dimer (H2O)2. Hydrogen-bond benchmark.

Geometry from the GMTKN55 distribution of the S22 test set (Jurečka,
Šponer, Černý, Hobza 2006), in Ångström. Reference interaction energy
for the canonical S22A revised values is -5.02 kcal/mol (Marshall,
Burns, Sherrill 2011).

For the cross-code parity suite this is a single-determinant ``RHF`` /
``RKS-LDA`` / ``MP2`` test on the dimer geometry — interaction energy
benchmarking (which would require the two monomer single-points and
counterpoise correction) is wave-2 work.
"""
from __future__ import annotations

from ...core.spec import AtomCart, MoleculeSpec

SPEC = MoleculeSpec(
    id="s22_water_dimer",
    family="molecule_noncovalent_hb",
    atoms=(
        AtomCart(symbol="O", z=8, xyz_ang=(-1.65542049,  -0.12330038,   0.00000000)),
        AtomCart(symbol="O", z=8, xyz_ang=( 1.24621235,   0.10268869,   0.00000000)),
        AtomCart(symbol="H", z=1, xyz_ang=(-0.70409021,   0.03193167,   0.00000000)),
        AtomCart(symbol="H", z=1, xyz_ang=(-2.03867259,   0.75372288,   0.00000000)),
        AtomCart(symbol="H", z=1, xyz_ang=( 1.57598546,  -0.38252144,  -0.75856124)),
        AtomCart(symbol="H", z=1, xyz_ang=( 1.57598546,  -0.38252144,   0.75856124)),
    ),
    charge=0,
    multiplicity=1,
    default_conv_tol_energy=1e-9,
    default_max_iter=80,
    notes=(
        "S22 system #02 — (H2O)2 hydrogen-bonded dimer. The textbook "
        "noncovalent benchmark; agreement at sub-µHa (RHF) and "
        "sub-mHa (LDA) tracks the molecular-SCF + ERI stack for a "
        "system where dispersion is the secondary contribution and "
        "the H-bond geometry exercises the long-range Coulomb tail."
    ),
    citation=(
        "Jurečka, Šponer, Černý, Hobza, *Phys. Chem. Chem. Phys.* "
        "**2006**, 8, 1985 (S22 origin, DOI 10.1039/B600027D); "
        "Marshall, Burns, Sherrill, *J. Chem. Phys.* **2011**, 135, "
        "194102 (S22A revised CCSD(T)/CBS references, DOI 10.1063/"
        "1.3659142). Geometry from the GMTKN55 distribution "
        "(grimme-lab/GMTKN55 S22/02/struc.xyz)."
    ),
)
