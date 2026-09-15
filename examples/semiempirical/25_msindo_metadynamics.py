"""Well-tempered metadynamics — free-energy reconstruction (analytic + MSINDO).

vibe-qc's metadynamics driver (:mod:`vibeqc.metadynamics`) layers a
history-dependent Gaussian bias on a few collective variables (distances /
angles) on top of the general MD driver. It is method-agnostic: the
underlying potential is any ``force_fn(coords) -> (energy, gradient)``.

Part 1 (fast, analytic) is the headline validation: a 1-D double well, filled
by the bias, with the free-energy profile reconstructed from the accumulated
hills via F(s) = −γ/(γ−1)·V_bias(s) (well-tempered, Barducci-Bussi-Parrinello
2008) — compared against the known potential.

Part 2 wires MSINDO in (``run_msindo_metadynamics``) and biases an O-H
distance of H2O. Filling a real chemical basin needs many MD steps and each
costs 6N+1 SCFs (the FD gradient), so this part is a short API demonstration,
not a converged free-energy surface.

Run:  .venv/bin/python examples/semiempirical/25_msindo_metadynamics.py
"""

from __future__ import annotations

import numpy as np

from vibeqc.metadynamics import DistanceCV, run_metadynamics
from vibeqc.semiempirical.methods.msindo import run_msindo_metadynamics

_KB = 1.380649e-23 / 4.3597447222071e-18  # Ha/K


def part1_analytic_double_well() -> None:
    """Fill a 1-D double well and recover its free-energy profile."""
    A, dc, w, kperp = 0.02, 3.0, 0.8, 0.6  # minima at dc±w, barrier A·w⁴

    def force_fn(R):
        R = np.asarray(R, float)
        dx = R[1, 0] - R[0, 0]
        s = (dx - dc) ** 2 - w ** 2
        e = A * s * s
        g = np.zeros_like(R)
        d = A * 4.0 * s * (dx - dc)
        g[1, 0] += d
        g[0, 0] -= d
        for a in range(2):           # perpendicular confinement → clean 1-D
            for c in (1, 2):
                e += 0.5 * kperp * R[a, c] ** 2
                g[a, c] += kperp * R[a, c]
        return e, g

    x0 = np.array([[0.0, 0, 0], [2.2, 0, 0]])   # start in the left well
    res = run_metadynamics(
        force_fn, x0, [DistanceCV(0, 1)], masses_amu=np.full(2, 12.0),
        bias_factor=8.0, hill_height=8e-4, hill_sigma=0.15,
        deposition_stride=10, timestep_fs=0.5, n_steps=16000,
        temperature_K=300.0, thermostat="berendsen", thermostat_tau_fs=30.0,
        remove_com=False, seed=1, record_stride=20)

    cv = res.cv_trajectory[:, 0]
    grid = np.linspace(1.9, 4.1, 221)
    f = res.free_energy(grid)
    u = A * ((grid - dc) ** 2 - w ** 2) ** 2
    u -= u.min()

    print("Part 1 — analytic 1-D double well")
    print(f"  hills deposited : {res.n_hills}")
    print(f"  CV explored     : {cv.min():.2f} .. {cv.max():.2f} bohr "
          f"(wells at {dc - w:.1f} / {dc + w:.1f})")
    print(f"  barrier (recovered/true) : {f[(grid > 2.85) & (grid < 3.15)].max():.4f}"
          f" / {A * w ** 4:.4f} Ha")
    rmsd = np.sqrt(np.mean((f - u)[(grid > 2.0) & (grid < 4.0)] ** 2))
    print(f"  RMSD(F_recovered − U)    : {rmsd:.4f} Ha ({rmsd / (_KB * 300):.1f} k_B T)")


def part2_msindo_h2o() -> None:
    """Short metadynamics on the MSINDO surface biasing an O-H distance."""
    Z = [8, 1, 1]
    xyz = np.array([[0.0, 0.0, 0.0],
                    [0.0, 0.7572, 0.5865],
                    [0.0, -0.7572, 0.5865]])
    res = run_msindo_metadynamics(
        Z, xyz, [DistanceCV(0, 1)], hill_height=5e-4, hill_sigma=0.12,
        deposition_stride=4, timestep_fs=0.3, n_steps=40, temperature_K=400.0,
        thermostat="berendsen", seed=1, output="msindo_metad_h2o")
    cv = res.cv_trajectory[:, 0]
    print("\nPart 2 — MSINDO H2O (O-H distance CV, short demo)")
    print(f"  hills deposited : {res.n_hills}")
    print(f"  O-H CV explored : {cv.min():.3f} .. {cv.max():.3f} bohr")
    print("  Wrote msindo_metad_h2o.bibtex / .references")


def main() -> None:
    print(__doc__)
    part1_analytic_double_well()
    part2_msindo_h2o()


if __name__ == "__main__":
    main()
