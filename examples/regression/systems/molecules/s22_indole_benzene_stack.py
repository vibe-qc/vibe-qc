"""S22 #14 — indole⋯benzene, stacked — Trp-Phe model.

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
    id="s22_indole_benzene_stack",
    family="molecule_noncovalent_dispersion",
    atoms=(
        AtomCart(symbol="C", z=6, xyz_ang=( 0.02063558,  1.50889305, -1.27016119)),
        AtomCart(symbol="C", z=6, xyz_ang=(-1.23296995,  0.95113439, -1.51363647)),
        AtomCart(symbol="C", z=6, xyz_ang=(-1.33659613, -0.24866702, -2.21464238)),
        AtomCart(symbol="C", z=6, xyz_ang=(-0.18723288, -0.88937438, -2.67502145)),
        AtomCart(symbol="C", z=6, xyz_ang=( 1.06649827, -0.32648605, -2.43746813)),
        AtomCart(symbol="C", z=6, xyz_ang=( 1.17070967,  0.87371011, -1.73620976)),
        AtomCart(symbol="C", z=6, xyz_ang=(-1.98035814,  0.40288525,  1.99512908)),
        AtomCart(symbol="C", z=6, xyz_ang=(-0.77323224,  1.05107679,  2.20047256)),
        AtomCart(symbol="C", z=6, xyz_ang=( 0.41213850,  0.42631646,  1.77852002)),
        AtomCart(symbol="C", z=6, xyz_ang=( 1.79257223,  0.78092484,  1.81317458)),
        AtomCart(symbol="C", z=6, xyz_ang=( 2.48684631, -0.25404313,  1.22910533)),
        AtomCart(symbol="C", z=6, xyz_ang=( 0.32783128, -0.84991770,  1.15564915)),
        AtomCart(symbol="C", z=6, xyz_ang=(-0.88675719, -1.50828144,  0.95446763)),
        AtomCart(symbol="C", z=6, xyz_ang=(-2.03757567, -0.86473592,  1.38141834)),
        AtomCart(symbol="H", z=1, xyz_ang=(-2.12339091,  1.44248360, -1.14679424)),
        AtomCart(symbol="H", z=1, xyz_ang=(-0.74344333,  2.02141298,  2.67938309)),
        AtomCart(symbol="H", z=1, xyz_ang=(-2.30926450, -0.68458124, -2.39889676)),
        AtomCart(symbol="H", z=1, xyz_ang=(-0.26863220, -1.81854555, -3.22349772)),
        AtomCart(symbol="H", z=1, xyz_ang=( 2.22872117,  1.67685969,  2.22136458)),
        AtomCart(symbol="H", z=1, xyz_ang=( 1.95829501, -0.81705356, -2.80562135)),
        AtomCart(symbol="H", z=1, xyz_ang=( 2.14174508,  1.30970722, -1.54626872)),
        AtomCart(symbol="H", z=1, xyz_ang=(-2.90005573,  0.87241477,  2.31767971)),
        AtomCart(symbol="H", z=1, xyz_ang=(-0.93121047, -2.47845425,  0.47717511)),
        AtomCart(symbol="H", z=1, xyz_ang=( 0.10178380,  2.43359449, -0.71562222)),
        AtomCart(symbol="H", z=1, xyz_ang=(-2.99728859, -1.34335388,  1.24061404)),
        AtomCart(symbol="H", z=1, xyz_ang=( 1.84928446, -2.05966570,  0.32707757)),
        AtomCart(symbol="H", z=1, xyz_ang=( 3.54459009, -0.37150335,  1.06329726)),
        AtomCart(symbol="N", z=7, xyz_ang=( 1.60635648, -1.23675044,  0.84931232)),
    ),
    charge=0,
    multiplicity=1,
    default_conv_tol_energy=1e-9,
    default_max_iter=80,
    notes=(
        "S22 system #14 — indole⋯benzene, stacked — Trp-Phe model. Geometry from grimme-lab/GMTKN55 "
        "S22/14/struc.xyz."
    ),
    citation=(
        "Jurečka, Šponer, Černý, Hobza, *Phys. Chem. Chem. Phys.* "
        "**2006**, 8, 1985 (S22 origin, DOI 10.1039/B600027D); "
        "Marshall, Burns, Sherrill, *J. Chem. Phys.* **2011**, 135, "
        "194102 (S22A revised CCSD(T)/CBS, DOI 10.1063/1.3659142). "
        "Geometry: grimme-lab/GMTKN55 S22/14/struc.xyz."
    ),
)
