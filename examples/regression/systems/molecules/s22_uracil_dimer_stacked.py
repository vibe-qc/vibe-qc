"""S22 #13 — uracil dimer, stacked — base-stacking analog.

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
    id="s22_uracil_dimer_stacked",
    family="molecule_noncovalent_dispersion",
    atoms=(
        AtomCart(symbol="N", z=7, xyz_ang=( 2.01135924, -1.21320765,  0.19492378)),
        AtomCart(symbol="N", z=7, xyz_ang=( 1.37688895,  0.83974564,  1.02762692)),
        AtomCart(symbol="N", z=7, xyz_ang=(-2.01135924,  1.21320765,  0.19492378)),
        AtomCart(symbol="N", z=7, xyz_ang=(-1.37688895, -0.83974564,  1.02762692)),
        AtomCart(symbol="H", z=1, xyz_ang=( 1.72728775,  0.99084689, -2.31901467)),
        AtomCart(symbol="H", z=1, xyz_ang=( 1.05180428,  1.38622332,  1.81636257)),
        AtomCart(symbol="H", z=1, xyz_ang=( 2.12946419, -2.20150530,  0.34980452)),
        AtomCart(symbol="H", z=1, xyz_ang=( 2.29752142, -1.39105962, -1.85265535)),
        AtomCart(symbol="H", z=1, xyz_ang=(-2.29752142,  1.39105962, -1.85265535)),
        AtomCart(symbol="H", z=1, xyz_ang=(-1.72728775, -0.99084689, -2.31901467)),
        AtomCart(symbol="H", z=1, xyz_ang=(-1.05180428, -1.38622332,  1.81636257)),
        AtomCart(symbol="H", z=1, xyz_ang=(-2.12946419,  2.20150530,  0.34980452)),
        AtomCart(symbol="C", z=6, xyz_ang=( 2.02570819, -0.69717987, -1.07141224)),
        AtomCart(symbol="C", z=6, xyz_ang=(-2.02570819,  0.69717987, -1.07141224)),
        AtomCart(symbol="C", z=6, xyz_ang=( 1.30896084,  1.45753443, -0.22759804)),
        AtomCart(symbol="C", z=6, xyz_ang=(-1.71452307, -0.59196526, -1.31949850)),
        AtomCart(symbol="C", z=6, xyz_ang=( 1.64599138, -0.48521144,  1.31171809)),
        AtomCart(symbol="C", z=6, xyz_ang=(-1.30896084, -1.45753443, -0.22759804)),
        AtomCart(symbol="C", z=6, xyz_ang=( 1.71452307,  0.59196526, -1.31949850)),
        AtomCart(symbol="C", z=6, xyz_ang=(-1.64599138,  0.48521144,  1.31171809)),
        AtomCart(symbol="O", z=8, xyz_ang=( 1.56110939, -0.97180641,  2.42279771)),
        AtomCart(symbol="O", z=8, xyz_ang=(-0.92059286, -2.61108704, -0.33305479)),
        AtomCart(symbol="O", z=8, xyz_ang=(-1.56110939,  0.97180641,  2.42279771)),
        AtomCart(symbol="O", z=8, xyz_ang=( 0.92059286,  2.61108704, -0.33305479)),
    ),
    charge=0,
    multiplicity=1,
    default_conv_tol_energy=1e-9,
    default_max_iter=80,
    notes=(
        "S22 system #13 — uracil dimer, stacked — base-stacking analog. Geometry from grimme-lab/GMTKN55 "
        "S22/13/struc.xyz."
    ),
    citation=(
        "Jurečka, Šponer, Černý, Hobza, *Phys. Chem. Chem. Phys.* "
        "**2006**, 8, 1985 (S22 origin, DOI 10.1039/B600027D); "
        "Marshall, Burns, Sherrill, *J. Chem. Phys.* **2011**, 135, "
        "194102 (S22A revised CCSD(T)/CBS, DOI 10.1063/1.3659142). "
        "Geometry: grimme-lab/GMTKN55 S22/13/struc.xyz."
    ),
)
