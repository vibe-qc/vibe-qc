"""H2O — workhorse molecular SCF benchmark."""
from __future__ import annotations

from ...core.spec import AtomCart, MoleculeSpec

# Szabo & Ostlund standard geometry (used in tests/conftest.py for the
# vibe-qc molecular RHF reference).
SPEC = MoleculeSpec(
    id="h2o",
    family="molecule_polyatomic",
    atoms=(
        AtomCart(symbol="O", z=8, xyz_ang=(0.0,         0.0,        0.0)),
        AtomCart(symbol="H", z=1, xyz_ang=(0.7569503,   0.5858823,  0.0)),
        AtomCart(symbol="H", z=1, xyz_ang=(-0.7569503,  0.5858823,  0.0)),
    ),
    charge=0,
    multiplicity=1,
    default_conv_tol_energy=1e-10,
    default_max_iter=80,
    notes=(
        "H2O at the Szabo & Ostlund reference geometry (R(O-H) = 1.0 Å, "
        "θ(H-O-H) = 104.5°). Same geometry tests/conftest.py uses as the "
        "vibe-qc molecular reference, so results are directly comparable "
        "to the unit-test fixtures."
    ),
    citation="Szabo & Ostlund, 'Modern Quantum Chemistry', §3.5 H2O example.",
)
