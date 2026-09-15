"""CH4 — methane, tetrahedral. Smallest C-containing closed-shell."""
from __future__ import annotations

from ...core.spec import AtomCart, MoleculeSpec

# Tetrahedral CH4, R(C-H) = 1.090 Å. Place H at the four corners of an
# inscribed tetrahedron of edge a = 4 R / sqrt(3) ≈ 2.518 Å so each H
# is at distance R from C.
_R = 1.090
_d = _R / 3 ** 0.5                                # Cartesian magnitude per axis

SPEC = MoleculeSpec(
    id="ch4",
    family="molecule_polyatomic",
    atoms=(
        AtomCart(symbol="C", z=6, xyz_ang=(0.0,  0.0,  0.0)),
        AtomCart(symbol="H", z=1, xyz_ang=( _d,  _d,  _d)),
        AtomCart(symbol="H", z=1, xyz_ang=( _d, -_d, -_d)),
        AtomCart(symbol="H", z=1, xyz_ang=(-_d,  _d, -_d)),
        AtomCart(symbol="H", z=1, xyz_ang=(-_d, -_d,  _d)),
    ),
    charge=0,
    multiplicity=1,
    default_conv_tol_energy=1e-10,
    default_max_iter=80,
    notes="CH4 at R(C-H) = 1.090 Å (NIST experimental).",
    citation="CH4 R_e = 1.0858 Å (NIST CCCBDB).",
)
