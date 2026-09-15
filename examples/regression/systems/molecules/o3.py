"""O3 — ozone, bent C2v singlet ground state with biradical character."""
from __future__ import annotations

import math

from ...core.spec import AtomCart, MoleculeSpec

# Ozone ¹A1 ground state, R(O-O) = 1.278 Å, ∠O-O-O = 116.8°. Place
# central O at origin, terminal O atoms in the xy-plane mirrored about
# the y-axis (apex pointing -y).
_R = 1.278
_THETA = math.radians(116.8 / 2.0)               # half O-O-O angle
_X = _R * math.sin(_THETA)
_Y = -_R * math.cos(_THETA)

SPEC = MoleculeSpec(
    id="o3",
    family="molecule_polyatomic",
    atoms=(
        AtomCart(symbol="O", z=8, xyz_ang=(0.0,  0.0, 0.0)),     # central
        AtomCart(symbol="O", z=8, xyz_ang=(+_X, _Y,  0.0)),
        AtomCart(symbol="O", z=8, xyz_ang=(-_X, _Y,  0.0)),
    ),
    charge=0,
    multiplicity=1,
    default_conv_tol_energy=1e-9,
    default_max_iter=120,
    notes=(
        "Ozone bent C2v ¹A1 at R(O-O) = 1.278 Å, ∠O-O-O = 116.8°. "
        "Closed-shell singlet ground state, but notoriously hard for "
        "single-reference SCF because of substantial biradical / "
        "multireference character — RHF/RKS at minimal basis often miss "
        "the right energy. Useful as a 'codes converge to the same "
        "wrong place' parity test."
    ),
    citation="Ozone experimental geometry (Tanaka & Morino 1970).",
)
