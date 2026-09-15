"""S22 #11 — benzene dimer, parallel-displaced — π-stacking minimum.

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
    id="s22_benzene_dimer_pd",
    family="molecule_noncovalent_dispersion",
    atoms=(
        AtomCart(symbol="C", z=6, xyz_ang=(-1.04782552, -1.42167398,  0.00000000)),
        AtomCart(symbol="C", z=6, xyz_ang=(-1.45450380, -0.85544613,  1.20620521)),
        AtomCart(symbol="C", z=6, xyz_ang=(-1.45450380, -0.85544613, -1.20620521)),
        AtomCart(symbol="C", z=6, xyz_ang=(-2.26679755,  0.27716108,  1.20695425)),
        AtomCart(symbol="C", z=6, xyz_ang=(-2.67147876,  0.84502132,  0.00000000)),
        AtomCart(symbol="C", z=6, xyz_ang=(-2.26679755,  0.27716108, -1.20695425)),
        AtomCart(symbol="C", z=6, xyz_ang=( 1.04782552,  1.42167398,  0.00000000)),
        AtomCart(symbol="C", z=6, xyz_ang=( 1.45450380,  0.85544613, -1.20620521)),
        AtomCart(symbol="C", z=6, xyz_ang=( 1.45450380,  0.85544613,  1.20620521)),
        AtomCart(symbol="C", z=6, xyz_ang=( 2.26679755, -0.27716108, -1.20695425)),
        AtomCart(symbol="C", z=6, xyz_ang=( 2.67147876, -0.84502132,  0.00000000)),
        AtomCart(symbol="C", z=6, xyz_ang=( 2.26679755, -0.27716108,  1.20695425)),
        AtomCart(symbol="H", z=1, xyz_ang=(-1.13385373, -1.29205969, -2.14231568)),
        AtomCart(symbol="H", z=1, xyz_ang=(-2.58249512,  0.71630678, -2.14379838)),
        AtomCart(symbol="H", z=1, xyz_ang=(-3.30304321,  1.72327051,  0.00000000)),
        AtomCart(symbol="H", z=1, xyz_ang=(-2.58249512,  0.71630678,  2.14379838)),
        AtomCart(symbol="H", z=1, xyz_ang=(-1.13385373, -1.29205969,  2.14231568)),
        AtomCart(symbol="H", z=1, xyz_ang=(-0.40602541, -2.29190553,  0.00000000)),
        AtomCart(symbol="H", z=1, xyz_ang=( 0.40602541,  2.29190553,  0.00000000)),
        AtomCart(symbol="H", z=1, xyz_ang=( 1.13385373,  1.29205969,  2.14231568)),
        AtomCart(symbol="H", z=1, xyz_ang=( 2.58249512, -0.71630678,  2.14379838)),
        AtomCart(symbol="H", z=1, xyz_ang=( 3.30304321, -1.72327051,  0.00000000)),
        AtomCart(symbol="H", z=1, xyz_ang=( 2.58249512, -0.71630678, -2.14379838)),
        AtomCart(symbol="H", z=1, xyz_ang=( 1.13385373,  1.29205969, -2.14231568)),
    ),
    charge=0,
    multiplicity=1,
    default_conv_tol_energy=1e-9,
    default_max_iter=80,
    notes=(
        "S22 system #11 — benzene dimer, parallel-displaced — π-stacking minimum. Geometry from grimme-lab/GMTKN55 "
        "S22/11/struc.xyz."
    ),
    citation=(
        "Jurečka, Šponer, Černý, Hobza, *Phys. Chem. Chem. Phys.* "
        "**2006**, 8, 1985 (S22 origin, DOI 10.1039/B600027D); "
        "Marshall, Burns, Sherrill, *J. Chem. Phys.* **2011**, 135, "
        "194102 (S22A revised CCSD(T)/CBS, DOI 10.1063/1.3659142). "
        "Geometry: grimme-lab/GMTKN55 S22/11/struc.xyz."
    ),
)
