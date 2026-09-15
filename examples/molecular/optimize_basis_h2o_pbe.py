"""Re-optimise a molecular basis at the DFT level with the analytic BDIIS.

Demonstrates ``vibeqc.basis_optimization.recipes.molecular.optimize_molecular_basis``:
the one-call, vibe-qc-native molecular basis optimiser. It builds a consistent
in-process (energy objective, *analytic* energy gradient) pair and drives the
robust BDIIS loop — the basis-set analogue of CRYSTAL23 OPTBASIS, but with the
analytic gradient instead of finite differences, and for any of RHF / UHF /
RKS / UKS.

Here we take the valence / polarisation exponents of pob-TZVP for an isolated
H2O molecule and let RKS/PBE pull them to the values that minimise the
molecular energy in that fixed contraction pattern, then write the improved O
basis back out as a Gaussian-94 deck.

Run:  python examples/molecular/optimize_basis_h2o_pbe.py

Measured (pob-TZVP, RKS/PBE, H2O at ~104.5 deg, r(OH)=0.958 A):

    molecular basis optimisation (RKS/PBE)
      objective: -76.36859599 -> -76.37085698 Ha
      improvement: +2.2610 mHa  (lower is better)
      evals: 14   wall: ~55 s
      converged: True
        O.shell3.prim0.exponent      0.356587 ->     0.346200   (diffuse s)
        O.shell7.prim0.exponent      0.253462 ->     0.513548   (d polarisation)
        H.shell3.prim0.exponent      0.800000 ->     1.194617   (p polarisation)

The molecular optimum tightens the polarisation functions (the pob-TZVP set
is calibrated for solids, not a single gas-phase H2O), so the landing point
differs from the published pob-TZVP values — the point is the machinery, not
the numbers.

Cite (CLAUDE.md § 8): the optimiser method (BDIIS — Daga, Civalleri, Maschio,
J. Chem. Theory Comput. 16, 2192 (2020); see ``result.citations``), the
starting basis set (pob-TZVP — Peintinger, Vilela Oliveira, Bredow,
J. Comput. Chem. 34, 451 (2013)), and the functional (PBE, via libxc).
"""

from __future__ import annotations

from pathlib import Path

import vibeqc as vq
from vibeqc.basis_crystal import emit_g94, parse_crystal_atom_basis_file
from vibeqc.basis_optimization.parametrise import (
    BasisParametrisation,
    FreeSpec,
    Transform,
)
from vibeqc.basis_optimization.recipes.molecular import optimize_molecular_basis

SRC = Path(vq.__file__).parent / "basis_library" / "sources" / "pob-TZVP"


def main() -> None:
    atoms = {
        "O": parse_crystal_atom_basis_file(SRC / "08_O"),
        "H": parse_crystal_atom_basis_file(SRC / "01_H"),
    }

    # Free: O diffuse-s (shell 3), O d-polarisation (shell 7), H p-polarisation
    # (shell 3). All single-primitive valence/polarisation exponents — a
    # well-conditioned subset (core exponents are left fixed). LOG transform +
    # a lower bound guard against collapse toward linear dependence.
    parametrisation = BasisParametrisation(
        atoms=atoms,
        free=[
            FreeSpec("O", 3, 0, "exponent", transform=Transform.LOG, bounds=(0.05, None)),
            FreeSpec("O", 7, 0, "exponent", transform=Transform.LOG, bounds=(0.05, None)),
            FreeSpec("H", 3, 0, "exponent", transform=Transform.LOG, bounds=(0.05, None)),
        ],
    )

    # Isolated H2O (singlet) -> RKS. Geometry in bohr.
    def molecule_factory():
        return vq.Molecule(
            [
                vq.Atom(8, [0.0, 0.0, 0.0]),
                vq.Atom(1, [0.0, 1.4304, 1.1071]),
                vq.Atom(1, [0.0, -1.4304, 1.1071]),
            ],
            multiplicity=1,
        )

    result = optimize_molecular_basis(
        parametrisation,
        molecule_factory,
        functional="PBE",        # closed-shell RKS; open_shell=True -> UKS
    )

    print(result.summary())
    print("\nmethod citations:", ", ".join(result.citations))

    # The optimised basis is returned as {symbol: CrystalAtomBasis} — write the
    # improved O basis back out as a Gaussian-94 deck (or emit_crystal for a
    # CRYSTAL inline deck).
    improved_o = result.optimized_atoms["O"]
    print("\n--- improved O basis (Gaussian-94) ---")
    print(emit_g94([improved_o], header_comment="pob-TZVP O, PBE-reoptimised for H2O"))


if __name__ == "__main__":
    main()
