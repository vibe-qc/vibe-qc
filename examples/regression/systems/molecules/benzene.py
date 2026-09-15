"""Benzene C6H6 — aromatic π system. The DF showcase."""
from __future__ import annotations

import math

from ...core.spec import AtomCart, MoleculeSpec

# Planar D6h benzene. Standard reference geometry: R(C-C) = 1.397 Å,
# R(C-H) = 1.084 Å. Carbons at vertices of a regular hexagon in z=0;
# each H along the radial direction from the centre.
_R_CC = 1.397
_R_CH = 1.084
_R_C  = _R_CC                                     # C distance from centre = R(C-C)
_R_H  = _R_CC + _R_CH                             # H distance from centre

_C, _H = [], []
for k in range(6):
    phi = math.radians(60.0 * k)
    _C.append((_R_C * math.cos(phi), _R_C * math.sin(phi), 0.0))
    _H.append((_R_H * math.cos(phi), _R_H * math.sin(phi), 0.0))

SPEC = MoleculeSpec(
    id="benzene",
    family="molecule_aromatic",
    atoms=tuple(
        [AtomCart(symbol="C", z=6, xyz_ang=p) for p in _C]
        + [AtomCart(symbol="H", z=1, xyz_ang=p) for p in _H]
    ),
    charge=0,
    multiplicity=1,
    default_conv_tol_energy=1e-9,
    default_max_iter=120,
    notes=(
        "Benzene D6h, R(C-C) = 1.397 Å, R(C-H) = 1.084 Å. The standard "
        "molecular DF showcase — large enough that DF actually saves "
        "wall time vs direct 4-index ERIs at def2-SVP / def2-TZVP."
    ),
    citation="Benzene experimental geometry (Tamagawa et al. 1976).",
)
