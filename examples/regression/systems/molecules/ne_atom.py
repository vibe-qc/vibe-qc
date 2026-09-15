"""Ne atom — closed-shell, no nuclear motion, no symmetry-breaking risk."""
from __future__ import annotations

from ...core.spec import AtomCart, MoleculeSpec

SPEC = MoleculeSpec(
    id="ne_atom",
    family="molecule_atom",
    atoms=(AtomCart(symbol="Ne", z=10, xyz_ang=(0.0, 0.0, 0.0)),),
    charge=0,
    multiplicity=1,
    default_conv_tol_energy=1e-10,
    default_max_iter=50,
    notes=(
        "Single Ne atom. Closed-shell, no SCF instability, atomic energy "
        "well-known across codes. Useful as a 'is the basis-set library "
        "wired up?' check — if Ne disagrees by more than µHa, the basis "
        "is being parsed differently between codes."
    ),
    citation="Ne 1s²2s²2p⁶ ground state.",
)
