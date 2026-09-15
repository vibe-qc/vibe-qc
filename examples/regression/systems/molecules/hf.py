"""HF — hydrogen fluoride, polar diatomic. Tests highest-Z 1st-row atom."""
from __future__ import annotations

from ...core.spec import AtomCart, MoleculeSpec

_R = 0.917                                        # NIST experimental R_e

SPEC = MoleculeSpec(
    id="hf",
    family="molecule_diatomic",
    atoms=(
        AtomCart(symbol="F", z=9, xyz_ang=(0.0, 0.0, 0.0)),
        AtomCart(symbol="H", z=1, xyz_ang=(_R,  0.0, 0.0)),
    ),
    charge=0,
    multiplicity=1,
    default_conv_tol_energy=1e-10,
    default_max_iter=80,
    notes="HF at R(H-F) = 0.917 Å (NIST). Polar 1Σ+ ground state.",
    citation="HF R_e = 0.9168 Å (NIST CCCBDB).",
)
