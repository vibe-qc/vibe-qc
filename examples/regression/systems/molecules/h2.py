"""H2 — smallest closed-shell molecule. Smoke-test anchor."""
from __future__ import annotations

from ...core.spec import AtomCart, MoleculeSpec

# H2 equilibrium bond length (experimental, 0.741 Å) — close enough
# to the LDA / sto-3g minimum that no relaxation is needed for parity.
_R_HALF_ANG = 0.741 / 2.0

SPEC = MoleculeSpec(
    id="h2",
    family="molecule_diatomic",
    atoms=(
        AtomCart(symbol="H", z=1, xyz_ang=(-_R_HALF_ANG, 0.0, 0.0)),
        AtomCart(symbol="H", z=1, xyz_ang=(+_R_HALF_ANG, 0.0, 0.0)),
    ),
    charge=0,
    multiplicity=1,
    default_conv_tol_energy=1e-10,
    default_max_iter=50,
    notes=(
        "H2 at equilibrium bond length 0.741 Å. The classical first "
        "molecular SCF case — anything that breaks here is broken at "
        "the level of two-electron integrals, not chemistry."
    ),
    citation="H2 R_e = 0.741 Å (NIST, ground-state X1Σg+).",
)
