"""S22 #07 — adenine⋯thymine Watson-Crick pair — biological H-bond reference.

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
    id="s22_adenine_thymine_wc",
    family="molecule_noncovalent_hb",
    atoms=(
        AtomCart(symbol="N", z=7, xyz_ang=( 1.28120341, -0.03370390, -0.59373940)),
        AtomCart(symbol="N", z=7, xyz_ang=( 3.19933947,  0.02007932, -2.05580781)),
        AtomCart(symbol="N", z=7, xyz_ang=( 4.43477120, -0.01116669,  1.31413134)),
        AtomCart(symbol="N", z=7, xyz_ang=( 5.27567605,  0.03551662, -0.77157526)),
        AtomCart(symbol="N", z=7, xyz_ang=( 1.41780564, -0.08226041,  1.72429192)),
        AtomCart(symbol="N", z=7, xyz_ang=(-3.57498636, -0.00668839, -1.73121405)),
        AtomCart(symbol="N", z=7, xyz_ang=(-1.57946129, -0.01678309, -0.57874259)),
        AtomCart(symbol="C", z=6, xyz_ang=( 5.52918125,  0.01967332,  0.57237013)),
        AtomCart(symbol="C", z=6, xyz_ang=( 3.91079959,  0.01382081, -0.92083506)),
        AtomCart(symbol="C", z=6, xyz_ang=( 2.02015201, -0.04150040,  0.52758415)),
        AtomCart(symbol="C", z=6, xyz_ang=( 1.89526414, -0.00446689, -1.79564901)),
        AtomCart(symbol="C", z=6, xyz_ang=(-4.26749690,  0.01118131, -0.54849979)),
        AtomCart(symbol="C", z=6, xyz_ang=(-3.64555208,  0.01621101,  0.65148636)),
        AtomCart(symbol="C", z=6, xyz_ang=(-2.18994963,  0.00174131,  0.66182494)),
        AtomCart(symbol="C", z=6, xyz_ang=( 3.42098405, -0.01517179,  0.38460866)),
        AtomCart(symbol="C", z=6, xyz_ang=(-2.19340278, -0.02067120, -1.81108387)),
        AtomCart(symbol="C", z=6, xyz_ang=(-4.36442696,  0.03561352,  1.95901673)),
        AtomCart(symbol="H", z=1, xyz_ang=( 5.94972628,  0.05915173, -1.51852913)),
        AtomCart(symbol="H", z=1, xyz_ang=( 0.40487919, -0.04810030,  1.78907093)),
        AtomCart(symbol="H", z=1, xyz_ang=( 1.99056780, -0.04046330,  2.54706891)),
        AtomCart(symbol="H", z=1, xyz_ang=( 1.22563141, -0.00069779, -2.64641938)),
        AtomCart(symbol="H", z=1, xyz_ang=( 6.53444869,  0.03183042,  0.95903501)),
        AtomCart(symbol="H", z=1, xyz_ang=(-4.05553099, -0.00933159, -2.61534074)),
        AtomCart(symbol="H", z=1, xyz_ang=(-0.53763810, -0.02734060, -0.59327470)),
        AtomCart(symbol="H", z=1, xyz_ang=(-5.34473580,  0.02121092, -0.63756611)),
        AtomCart(symbol="H", z=1, xyz_ang=(-4.09774199, -0.83598133,  2.55471858)),
        AtomCart(symbol="H", z=1, xyz_ang=(-4.08051936,  0.91289426,  2.53817877)),
        AtomCart(symbol="H", z=1, xyz_ang=(-5.44221090,  0.04482922,  1.80988095)),
        AtomCart(symbol="O", z=8, xyz_ang=(-1.52128591,  0.00548551,  1.69723620)),
        AtomCart(symbol="O", z=8, xyz_ang=(-1.59549114, -0.03491160, -2.87222668)),
    ),
    charge=0,
    multiplicity=1,
    default_conv_tol_energy=1e-9,
    default_max_iter=80,
    notes=(
        "S22 system #07 — adenine⋯thymine Watson-Crick pair — biological H-bond reference. Geometry from grimme-lab/GMTKN55 "
        "S22/07/struc.xyz."
    ),
    citation=(
        "Jurečka, Šponer, Černý, Hobza, *Phys. Chem. Chem. Phys.* "
        "**2006**, 8, 1985 (S22 origin, DOI 10.1039/B600027D); "
        "Marshall, Burns, Sherrill, *J. Chem. Phys.* **2011**, 135, "
        "194102 (S22A revised CCSD(T)/CBS, DOI 10.1063/1.3659142). "
        "Geometry: grimme-lab/GMTKN55 S22/07/struc.xyz."
    ),
)
