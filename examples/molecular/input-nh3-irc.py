"""Ammonia inversion: intrinsic reaction coordinate from the planar D3h saddle.

System: NH3 at RHF/STO-3G. The script builds planar NH3 (N at the origin,
three H at 1.00 angstrom in the xy-plane, 120 degrees apart) and relaxes it
with ``vibeqc.optimize_molecule``; the out-of-plane gradient vanishes by
symmetry, so the optimiser stays in the plane and lands on the D3h
first-order saddle point of the umbrella inversion (N-H = 1.0055 angstrom).
Method: ``vibeqc.run_irc(ts, "sto-3g", method="RHF", direction="both")``.
With no ``transition_mode`` given, run_irc takes a finite-difference
Hessian at the saddle, picks the imaginary mode, and follows the
mass-weighted steepest-descent path in both directions
(step 0.1 bohr amu^1/2) until the maximum force drops below 1e-3 Ha/bohr.

Files (next to this script):
    input-nh3-irc.out          text summary: both branches, endpoint check, barrier
    input-nh3-irc.bibtex       citations for the IRC algorithm and the method
    input-nh3-irc.references   the same list in plain text
(run_irc itself writes no files; the summary is written by this script.)

Success: both branches converge after 26 points each to the same energy,
-55.455417 Ha, 11.14 kcal/mol below the saddle (E_TS = -55.437665 Ha), and
the two minima are mirror images: nitrogen sits 0.81 bohr above the H3
plane on one side and 0.81 bohr below it on the other. The forward and
reverse endpoint energies agree to 1e-6 Ha and the barrier matches the
climbing-image NEB value in examples/workflows/input-nh3-umbrella-neb.py
(about 11 kcal/mol; experiment 5.8 kcal/mol, the usual HF/STO-3G
overshoot). Tolerance: 0.05 kcal/mol on the barrier (vibe-qc 0.17.x
development build, Apple M5 Max, about 20 s).
"""

from pathlib import Path

import numpy as np

import vibeqc as vq

HERE = Path(__file__).parent
STEM = Path(__file__).stem
ANG = 1.8897259886
KCAL = 627.5094740631


def planar_nh3(r_nh_angstrom: float = 1.00) -> vq.Molecule:
    r = r_nh_angstrom * ANG
    atoms = [vq.Atom(7, [0.0, 0.0, 0.0])]
    for k in range(3):
        phi = 2.0 * np.pi * k / 3.0
        atoms.append(vq.Atom(1, [r * np.cos(phi), r * np.sin(phi), 0.0]))
    return vq.Molecule(atoms, 0, 1)


def n_height(system) -> float:
    """Signed distance of N from the plane through the three H atoms (bohr)."""
    xyz = np.array([list(a.xyz) for a in system.atoms], dtype=float)
    h = xyz[1:]
    normal = np.cross(h[1] - h[0], h[2] - h[0])
    normal /= np.linalg.norm(normal)
    return float(np.dot(xyz[0] - h.mean(axis=0), normal))


# 1. The planar saddle: relax in-plane, symmetry keeps it planar.
opt = vq.optimize_molecule(planar_nh3(), "sto-3g", method="rhf")
ts = opt.system
r_nh = np.linalg.norm(np.array(ts.atoms[1].xyz) - np.array(ts.atoms[0].xyz))
print(f"\n  planar NH3 relaxed: E = {opt.energy:.8f} Ha, N-H = {r_nh / ANG:.4f} angstrom, "
      f"N height = {n_height(ts):+.4f} bohr")

# 2. Follow the IRC in both directions.
irc = vq.run_irc(ts, "sto-3g", method="RHF", functional=None,
                 direction="both", step_size=0.1, max_points=60)

lines = [f"IRC from the planar NH3 saddle, RHF/STO-3G; E_TS = {irc.ts_energy:.8f} Ha", ""]
for name, path, energies, conv in (
    ("forward", irc.forward_path, irc.forward_energies, irc.forward_converged),
    ("reverse", irc.reverse_path, irc.reverse_energies, irc.reverse_converged),
):
    lines.append(f"{name} branch: {len(path)} points, converged={conv}")
    lines.append(f"  {'point':>5s} {'E - E_TS (kcal/mol)':>20s} {'N height (bohr)':>16s}")
    for i, (sys_i, e_i) in enumerate(zip(path, energies), start=1):
        if i <= 5 or i == len(path) or i % 5 == 0:
            lines.append(f"  {i:5d} {(e_i - irc.ts_energy) * KCAL:20.4f} {n_height(sys_i):16.4f}")
    lines.append("")

e_fwd, e_rev = irc.forward_energies[-1], irc.reverse_energies[-1]
h_fwd, h_rev = n_height(irc.forward_minimum), n_height(irc.reverse_minimum)
lines += [
    f"endpoints: E_forward = {e_fwd:.8f} Ha, E_reverse = {e_rev:.8f} Ha, "
    f"|difference| = {abs(e_fwd - e_rev):.2e} Ha",
    f"N height at the endpoints: {h_fwd:+.4f} / {h_rev:+.4f} bohr (mirror images)",
    f"inversion barrier: forward {irc.barrier_forward * KCAL:.3f} kcal/mol, "
    f"reverse {irc.barrier_reverse * KCAL:.3f} kcal/mol",
]
text = "\n".join(lines)
print("\n" + "\n".join("  " + ln for ln in lines))
(HERE / f"{STEM}.out").write_text(text + "\n")
irc.write_citations(HERE / STEM)

both_pyramidal = h_fwd * h_rev < 0 and abs(abs(h_fwd) - abs(h_rev)) < 1e-3
print(f"\n  both ends pyramidal and mirror-related: {both_pyramidal}; "
      f"same energy to {abs(e_fwd - e_rev):.1e} Ha")
