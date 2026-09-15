"""O2 — molecular oxygen, ³Σg⁻ triplet ground state. Open-shell diradical."""
from __future__ import annotations

from ...core.spec import AtomCart, MoleculeSpec

_R = 1.208                                        # NIST experimental R_e (³Σg⁻)
_HALF = _R / 2.0

SPEC = MoleculeSpec(
    id="o2",
    family="molecule_diradical",
    atoms=(
        AtomCart(symbol="O", z=8, xyz_ang=(-_HALF, 0.0, 0.0)),
        AtomCart(symbol="O", z=8, xyz_ang=(+_HALF, 0.0, 0.0)),
    ),
    charge=0,
    multiplicity=3,                               # triplet — two unpaired πg* electrons
    default_conv_tol_energy=1e-9,                 # open-shell DIIS less aggressive
    default_max_iter=120,
    notes=(
        "O2 ³Σg⁻ diradical at R(O-O) = 1.208 Å. Two unpaired electrons "
        "in degenerate πg* orbitals. Open-shell SCF stress test — UHF / "
        "UKS pick up spin-symmetry breaking the molecular ground state "
        "needs."
    ),
    citation="O2 ³Σg⁻ R_e = 1.2075 Å (Huber & Herzberg).",
)
