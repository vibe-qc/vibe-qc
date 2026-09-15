"""NH3 — ammonia, pyramidal. Lone-pair test case."""
from __future__ import annotations

import math

from ...core.spec import AtomCart, MoleculeSpec

# NH3 C3v: R(N-H) = 1.012 Å, H-N-H = 106.7°. Place N at origin, three
# H on a circle below the xy-plane.
_R = 1.012
_THETA = math.radians(106.7 / 2.0)               # half H-N-H angle
_RHO = _R * math.sin(_THETA) * 2.0 / math.sqrt(3.0)   # H radius from C3 axis
_Z = -math.sqrt(max(_R * _R - _RHO * _RHO, 0.0))      # H height (below N)

_H_POS = []
for k in range(3):
    phi = 2.0 * math.pi * k / 3.0
    _H_POS.append((_RHO * math.cos(phi), _RHO * math.sin(phi), _Z))

SPEC = MoleculeSpec(
    id="nh3",
    family="molecule_polyatomic",
    atoms=(
        AtomCart(symbol="N", z=7, xyz_ang=(0.0, 0.0, 0.0)),
        AtomCart(symbol="H", z=1, xyz_ang=_H_POS[0]),
        AtomCart(symbol="H", z=1, xyz_ang=_H_POS[1]),
        AtomCart(symbol="H", z=1, xyz_ang=_H_POS[2]),
    ),
    charge=0,
    multiplicity=1,
    default_conv_tol_energy=1e-10,
    default_max_iter=80,
    notes="NH3 pyramidal C3v: R(N-H) = 1.012 Å, ∠H-N-H = 106.7°.",
    citation="NH3 R_e and θ_e (NIST CCCBDB).",
)
