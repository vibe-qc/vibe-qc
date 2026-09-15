"""S22 #09 — ethene dimer (C₂H₄)₂ — π-dispersion model.

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
    id="s22_ethene_dimer",
    family="molecule_noncovalent_dispersion",
    atoms=(
        AtomCart(symbol="C", z=6, xyz_ang=(-0.47192515, -0.47192515, -1.85911158)),
        AtomCart(symbol="C", z=6, xyz_ang=( 0.47192515,  0.47192515, -1.85911158)),
        AtomCart(symbol="C", z=6, xyz_ang=(-0.47192515,  0.47192515,  1.85911158)),
        AtomCart(symbol="C", z=6, xyz_ang=( 0.47192515, -0.47192515,  1.85911158)),
        AtomCart(symbol="H", z=1, xyz_ang=(-0.87046427, -0.87046427, -2.78330874)),
        AtomCart(symbol="H", z=1, xyz_ang=( 0.87046427,  0.87046427, -2.78330874)),
        AtomCart(symbol="H", z=1, xyz_ang=(-0.87242223, -0.87242223, -0.93612524)),
        AtomCart(symbol="H", z=1, xyz_ang=( 0.87242223,  0.87242223, -0.93612524)),
        AtomCart(symbol="H", z=1, xyz_ang=(-0.87242223,  0.87242223,  0.93612524)),
        AtomCart(symbol="H", z=1, xyz_ang=( 0.87242223, -0.87242223,  0.93612524)),
        AtomCart(symbol="H", z=1, xyz_ang=(-0.87046427,  0.87046427,  2.78330874)),
        AtomCart(symbol="H", z=1, xyz_ang=( 0.87046427, -0.87046427,  2.78330874)),
    ),
    charge=0,
    multiplicity=1,
    default_conv_tol_energy=1e-9,
    default_max_iter=80,
    notes=(
        "S22 system #09 — ethene dimer (C₂H₄)₂ — π-dispersion model. Geometry from grimme-lab/GMTKN55 "
        "S22/09/struc.xyz."
    ),
    citation=(
        "Jurečka, Šponer, Černý, Hobza, *Phys. Chem. Chem. Phys.* "
        "**2006**, 8, 1985 (S22 origin, DOI 10.1039/B600027D); "
        "Marshall, Burns, Sherrill, *J. Chem. Phys.* **2011**, 135, "
        "194102 (S22A revised CCSD(T)/CBS, DOI 10.1063/1.3659142). "
        "Geometry: grimme-lab/GMTKN55 S22/09/struc.xyz."
    ),
)
