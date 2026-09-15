"""H2CO — formaldehyde. Mixed C=O / C-H bonding, planar Cs/C2v."""
from __future__ import annotations

import math

from ...core.spec import AtomCart, MoleculeSpec

# Planar H2CO, C2v. R(C=O) = 1.207, R(C-H) = 1.116, ∠H-C-H = 116.5°
# (NIST CCCBDB). Place C at origin, O along +x, H atoms in xy-plane.
_R_CO = 1.207
_R_CH = 1.116
_HCH_HALF = math.radians(116.5 / 2.0)
_Hx = -_R_CH * math.cos(_HCH_HALF)               # H is on the opposite side of O
_Hy =  _R_CH * math.sin(_HCH_HALF)

SPEC = MoleculeSpec(
    id="h2co",
    family="molecule_polyatomic",
    atoms=(
        AtomCart(symbol="C", z=6, xyz_ang=(0.0,    0.0,  0.0)),
        AtomCart(symbol="O", z=8, xyz_ang=(_R_CO,  0.0,  0.0)),
        AtomCart(symbol="H", z=1, xyz_ang=(_Hx,    _Hy,  0.0)),
        AtomCart(symbol="H", z=1, xyz_ang=(_Hx,   -_Hy,  0.0)),
    ),
    charge=0,
    multiplicity=1,
    default_conv_tol_energy=1e-10,
    default_max_iter=80,
    notes=(
        "Formaldehyde planar C2v: R(C=O) = 1.207 Å, R(C-H) = 1.116 Å, "
        "∠H-C-H = 116.5°. Mixed σ/π bonding test."
    ),
    citation="H2CO ground-state geometry (NIST CCCBDB).",
)
