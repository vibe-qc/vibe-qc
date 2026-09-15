"""Periodic MACE on bulk silicon: compact strain probe and optional relaxation.

Requires the optional ``[mace]`` extra (PyTorch + e3nn), ASE, and Python
<= 3.13:

    pip install 'vibe-qc[mace]'

Run:

    python examples/mlip/02_periodic_mace_silicon.py
    python examples/mlip/02_periodic_mace_silicon.py --relax

The default run evaluates two nearby Si cells with one retained MACE
calculator and prints energy, gradient, stress, and model provenance.
This is a single-point/strain smoke probe, not an equation-of-state
validation. ``--relax`` additionally runs a variable-cell relaxation.

``method="mace"`` is not wired into
``run_periodic_job`` (which is SCF-only); periodic MACE uses the direct
drivers below — following the periodic-semiempirical precedent
(``run_pm6_gamma``). The default model is the MIT-licensed MACE-MPA-0
(materials, 89 elements), built for exactly this kind of crystal.
"""

from __future__ import annotations

import argparse

import numpy as np
from ase.build import bulk

from vibeqc.ase_periodic import atoms_to_periodic_system
from vibeqc.mlip.mace import PeriodicMACEEvaluator, optimize_periodic_mace_cell

_HA_BOHR3_TO_GPA = 29421.0  # 1 Ha/bohr^3 in GPa
_BOHR_TO_A = 0.529177


def silicon(a_angstrom: float):
    """Return a two-atom primitive diamond-Si cell."""
    return atoms_to_periodic_system(bulk("Si", "diamond", a=a_angstrom))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--relax",
        action="store_true",
        help="also run a variable-cell relaxation after the compact probe",
    )
    args = parser.parse_args()

    lattice_constants = (5.43, 5.47)
    evaluator = PeriodicMACEEvaluator()
    results = [
        evaluator.run(silicon(lattice_constant))
        for lattice_constant in lattice_constants
    ]
    reference_energy = min(float(result.energy) for result in results)
    info = evaluator.model_info

    print(
        f"model={info.key} loader={info.loader} "
        f"license={info.license} citation={info.citation}"
    )
    print("a_conventional_A,delta_energy_meV,max_gradient_Ha_per_bohr,stress_diag_GPa")
    for lattice_constant, result in zip(
        lattice_constants,
        results,
        strict=True,
    ):
        delta_mev = (float(result.energy) - reference_energy) * 27211.386245988
        max_gradient = float(np.abs(result.gradient()).max())
        stress_gpa = np.diag(result.stress()) * _HA_BOHR3_TO_GPA
        stress_text = ";".join(f"{value:.6f}" for value in stress_gpa)
        print(
            f"{lattice_constant:.4f},{delta_mev:.8f},"
            f"{max_gradient:.8e},{stress_text}"
        )

    print(
        "This two-point relative probe confirms execution and model reuse; "
        "it is not an EOS or reference-accuracy validation."
    )
    if not args.relax:
        return

    initial = silicon(lattice_constants[0])
    relaxed = optimize_periodic_mace_cell(initial, fmax=0.01)
    after = evaluator.run(relaxed)
    a0 = np.linalg.norm(np.asarray(initial.lattice)[0]) * _BOHR_TO_A
    a1 = np.linalg.norm(np.asarray(relaxed.lattice)[0]) * _BOHR_TO_A
    print("\nafter variable-cell relaxation:")
    print(f"  stress (diag): {np.diag(after.stress()) * _HA_BOHR3_TO_GPA} GPa")
    print(f"  primitive |a|: {a0:.4f} -> {a1:.4f} A")
    print("  This is a model-internal relaxation, not an external accuracy claim.")


if __name__ == "__main__":
    main()
