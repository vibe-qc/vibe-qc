"""Formaldehyde — CPCM dielectric scan across the bundled solvent table.

Loops through every solvent in ``vibeqc.SOLVENT_PRESETS`` and reports
the polarisation energy E_solv = ½ q · V_tot. Demonstrates the
characteristic Cossi-Scalmani conductor-screening curve:
E_solv ∝ (ε − 1)/ε — saturates at the conductor limit by ε ~ 20,
so water (78.4) and acetonitrile (35.9) give nearly identical
solvation energies from electrostatics alone.

Matches Section 3 of docs/tutorial/solvation_water.md.

Run:
    .venv/bin/python examples/molecular/solvation/input-ch2o-dielectric-scan.py

Outputs (next to this script):
    input-ch2o-dielectric-scan.<solvent>.out   (one per solvent)
    Plus a text summary table on stdout.
"""

from pathlib import Path

import vibeqc as vq

HERE = Path(__file__).resolve().parent
STEM = Path(__file__).stem

ANG = 1.8897261339213

mol = vq.Molecule([
    vq.Atom(6, [0.0,  0.000 * ANG,  0.0000 * ANG]),
    vq.Atom(8, [0.0,  0.000 * ANG,  1.2055 * ANG]),
    vq.Atom(1, [0.0,  0.940 * ANG, -0.5870 * ANG]),
    vq.Atom(1, [0.0, -0.940 * ANG, -0.5870 * ANG]),
], 0, 1)

# Apolar → polar order so the saturation effect is visible.
ordered_solvents = [
    "vacuum",         # ε = 1 (gas-phase short-circuit)
    "n-hexane",       # ε = 1.88
    "toluene",        # ε = 2.37
    "chloroform",     # ε = 4.71
    "thf",            # ε = 7.43
    "dichloromethane",  # ε = 8.93
    "acetone",        # ε = 20.49
    "ethanol",        # ε = 24.85
    "methanol",       # ε = 32.61
    "acetonitrile",   # ε = 35.94
    "dmso",           # ε = 46.83
    "water",          # ε = 78.39
]

kcal_per_Ha = 627.5094740631

print(f"\n{'solvent':<20} {'ε':>8} {'E_solv (kcal/mol)':>18}")
print("-" * 48)
for name in ordered_solvents:
    result = vq.run_job(
        mol, basis="def2-svp", method="rks", functional="pbe0",
        solvent=name,
        output=HERE / f"{STEM}.{name}",
        progress=False,
    )
    sol = result.solvent_result
    e_kcal = sol.e_solv * kcal_per_Ha
    print(f"{name:<20} {sol.epsilon:>8.3f} {e_kcal:>18.3f}")
