"""Water / def2-SVP: DFT integration-grid convergence for PBE and r2SCAN.

Runs the named grid levels, an angular series of Lebedev rules (orders 23
to 95, 194 to 3074 points per shell, no pruning, 99 radial shells) and a
radial series (50 to 200 Treutler-Ahlrichs shells at order 59), and prints
each energy against the largest grid. See docs/tutorial/dft_grid_convergence.md.

Measured with the vibe-qc 0.17.17 development build (main, 2026-10-14),
errors in microhartree against order 59 with 200 radial shells:

    grid                 PBE      r2SCAN
    coarse              +6.31    +119.36
    orca-defgrid3       +1.75      +8.26
    fine                +0.33      +1.28
    L41, 99 radial      -0.03      +1.02
    L95, 99 radial      +0.04      +1.07
    L59, 75 radial      -0.10      +6.11
    L59, 150 radial     +0.00      +0.04

The angular rule is converged by order 41 for both functionals; the
remaining r2SCAN error is radial. Orders 59, 65, 77 and 95 need a build of
main from 2026-10-14 or later (the memory estimate refused them before).

Run from an empty scratch directory (vibe-qc writes its outputs into the
current directory); the whole series takes about a minute:

    python /path/to/examples/molecular/input-h2o-grid-convergence.py
"""

import numpy as np

import vibeqc as vq


def water() -> vq.Molecule:
    """Water at the experimental geometry (O-H 0.9572 A, H-O-H 104.5 deg)."""
    th = np.deg2rad(104.5 / 2)
    r = 0.9572 / 0.529177210903
    return vq.Molecule([
        vq.Atom(8, [0.0, 0.0, 0.0]),
        vq.Atom(1, [r * np.sin(th), r * np.cos(th), 0.0]),
        vq.Atom(1, [-r * np.sin(th), r * np.cos(th), 0.0]),
    ])


def energy(functional: str, tag: str, grid_level: str | None = None, **grid) -> float:
    kw = dict(basis="def2-svp", method="rks", functional=functional, output=tag,
              write_molden_file=False, write_population_file=False)
    if grid:
        opts = vq.RKSOptions()
        for name, value in grid.items():
            setattr(opts.grid, name, value)
        opts.conv_tol_energy = 1e-10
        kw["rks_options"] = opts
    else:
        kw["grid_level"] = grid_level
    return vq.run_job(water(), **kw).energy


def main() -> None:
    for functional in ("pbe", "r2scan"):
        series = {}
        for level in ("coarse", "orca-defgrid3", "fine"):
            series[level] = energy(functional, f"{functional}_{level}", grid_level=level)
        for order in (23, 29, 35, 41, 47, 53, 59, 65, 77, 95):
            series[f"L{order} R99"] = energy(
                functional, f"{functional}_L{order}_R99", n_radial=99,
                angular="lebedev", lebedev_order=order, angular_pruning="none",
                partition="stratmann")
        for n_radial in (50, 75, 150, 200):
            series[f"L59 R{n_radial}"] = energy(
                functional, f"{functional}_L59_R{n_radial}", n_radial=n_radial,
                angular="lebedev", lebedev_order=59, angular_pruning="none",
                partition="stratmann")
        reference = series["L59 R200"]
        print(f"\n{functional}: reference {reference:.10f} Ha")
        for label, value in series.items():
            print(f"  {label:16s} {value:.10f}  {(value - reference) * 1e6:+9.2f} uHa")


if __name__ == "__main__":
    main()
