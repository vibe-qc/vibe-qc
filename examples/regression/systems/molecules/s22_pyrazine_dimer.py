"""S22 #12 — pyrazine dimer — heteroaromatic π-stacking.

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
    id="s22_pyrazine_dimer",
    family="molecule_noncovalent_dispersion",
    atoms=(
        AtomCart(symbol="C", z=6, xyz_ang=(-1.24680950, -1.17150405, -0.69613900)),
        AtomCart(symbol="C", z=6, xyz_ang=(-1.24680950, -1.17150405,  0.69613900)),
        AtomCart(symbol="C", z=6, xyz_ang=( 0.73191314, -2.26490528,  0.69672902)),
        AtomCart(symbol="C", z=6, xyz_ang=( 0.73191314, -2.26490528, -0.69672902)),
        AtomCart(symbol="C", z=6, xyz_ang=(-0.33762293,  2.08037885,  1.13004555)),
        AtomCart(symbol="C", z=6, xyz_ang=( 0.85440589,  1.35966498,  1.13063108)),
        AtomCart(symbol="C", z=6, xyz_ang=( 0.85440589,  1.35966498, -1.13063108)),
        AtomCart(symbol="C", z=6, xyz_ang=(-0.33762293,  2.08037885, -1.13004555)),
        AtomCart(symbol="H", z=1, xyz_ang=( 1.54918104, -2.71251159,  1.24756080)),
        AtomCart(symbol="H", z=1, xyz_ang=( 1.54918104, -2.71251159, -1.24756080)),
        AtomCart(symbol="H", z=1, xyz_ang=(-2.06305665, -0.72200261, -1.24728008)),
        AtomCart(symbol="H", z=1, xyz_ang=(-2.06305665, -0.72200261,  1.24728008)),
        AtomCart(symbol="H", z=1, xyz_ang=(-0.80999578,  2.36462142,  2.06186488)),
        AtomCart(symbol="H", z=1, xyz_ang=( 1.32123889,  1.06737872,  2.06239919)),
        AtomCart(symbol="H", z=1, xyz_ang=( 1.32123889,  1.06737872, -2.06239919)),
        AtomCart(symbol="H", z=1, xyz_ang=(-0.80999578,  2.36462142, -2.06186488)),
        AtomCart(symbol="N", z=7, xyz_ang=( 1.47055938,  0.99107756,  0.00000000)),
        AtomCart(symbol="N", z=7, xyz_ang=(-0.25857083, -1.72326012,  1.41448001)),
        AtomCart(symbol="N", z=7, xyz_ang=(-0.25857083, -1.72326012, -1.41448001)),
        AtomCart(symbol="N", z=7, xyz_ang=(-0.95192593,  2.45320183,  0.00000000)),
    ),
    charge=0,
    multiplicity=1,
    default_conv_tol_energy=1e-9,
    default_max_iter=80,
    notes=(
        "S22 system #12 — pyrazine dimer — heteroaromatic π-stacking. Geometry from grimme-lab/GMTKN55 "
        "S22/12/struc.xyz."
    ),
    citation=(
        "Jurečka, Šponer, Černý, Hobza, *Phys. Chem. Chem. Phys.* "
        "**2006**, 8, 1985 (S22 origin, DOI 10.1039/B600027D); "
        "Marshall, Burns, Sherrill, *J. Chem. Phys.* **2011**, 135, "
        "194102 (S22A revised CCSD(T)/CBS, DOI 10.1063/1.3659142). "
        "Geometry: grimme-lab/GMTKN55 S22/12/struc.xyz."
    ),
)
