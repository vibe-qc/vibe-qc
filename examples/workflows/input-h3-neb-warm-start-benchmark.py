"""H + H₂ → H₂ + H climbing-image NEB — warm-start vs. cold-start benchmark.

Run:

    .venv/bin/python input-h3-neb-warm-start-benchmark.py

Produces:

    output-h3-neb-warm-start-benchmark.out   — timing + energies summary
    output-h3-neb-warm-start.qvf             — warm-start NEB animation
    output-h3-neb-cold-start.qvf             — cold-start NEB animation
                                               (emitted so the user can compare
                                               the two converged paths in
                                               vibe-view)

The textbook H + H₂ → H₂ + H collinear hydrogen-exchange reaction at
STO-3G UHF — three H atoms along a line, the middle one shuttling
between the two outer ones. The symmetric collinear saddle has the
central atom equidistant from the two outer atoms; CI-NEB pins it
exactly there.

This script runs the same NEB twice — once with ``warm_start=False``
and once with ``warm_start=True`` (the production default). On this
single-basin textbook case they should agree within tight numerical
tolerances, but warm and cold calculations are not guaranteed to be
bit-exact: roundoff and guess-dependent SCF basins can change a general
path. What should differ here is wall time and per-iteration SCF work.
Warm-start feeds the converged density from each image's previous outer
iteration as the SCF guess for that image's next outer iteration, so the
SCF converges in fewer iterations than from a SAD/Hcore cold start.

For this small benchmark the speedup is modest (open-shell UHF SCFs at
STO-3G converge fast even cold). The headline win is on UKS — see
the periodic NEB warm-start milestone + the NEB user-guide warm-start section
for the full speedup matrix.

Algorithm: improved-tangent NEB (Henkelman+Jónsson 2000) with the
climbing image variant (Henkelman+Uberuaga+Jónsson 2000). 5 intermediate
images, linear initial path, quick-min outer loop (default max_iter=80).
Converges to ≤ 1e-4 bohr saddle-symmetry in <50 iters.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from vibeqc import Atom, Molecule, UHFOptions, run_neb

HERE = Path(__file__).parent
TEXT_OUT = HERE / "output-h3-neb-warm-start-benchmark.out"
QVF_WARM = HERE / "output-h3-neb-warm-start"
QVF_COLD = HERE / "output-h3-neb-cold-start"


def _h3_doublet(z_positions):
    """Collinear H₃ doublet along z (charge=0, multiplicity=2)."""
    return Molecule(
        [Atom(1, [0.0, 0.0, float(z)]) for z in z_positions], 0, 2,
    )


def run_once(*, warm_start: bool) -> tuple:
    """Run the H + H₂ → H₂ + H NEB once, return (wall_time, NEBResult)."""
    reactant = _h3_doublet([0.0, 1.4, 4.4])   # A-B bonded, C far
    product = _h3_doublet([0.0, 3.0, 4.4])    # A far, B-C bonded
    uhf = UHFOptions()
    uhf.max_iter = 200

    t0 = time.time()
    result = run_neb(
        reactant, product,
        basis="sto-3g",
        n_images=5,
        method="UHF",
        uhf_options=uhf,
        spring_constant=0.1,
        interpolation="linear",
        max_iter=80,
        conv_tol_force=2e-3,
        n_jobs=1,
        initial_step=0.05,
        climbing_image=True,
        climbing_image_start_fraction=0.3,
        warm_start=warm_start,
    )
    return time.time() - t0, result


def main() -> None:
    print("Running H + H₂ → H₂ + H CI-NEB (warm-start vs cold-start)")
    print("=" * 70)

    print("\nCold-start (warm_start=False):")
    t_cold, r_cold = run_once(warm_start=False)
    print(
        f"  converged={r_cold.converged}  n_iter={r_cold.n_iter}  "
        f"max_force={r_cold.max_force:.4e} Ha/bohr  wall={t_cold:.2f}s"
    )

    print("\nWarm-start (warm_start=True, default):")
    t_warm, r_warm = run_once(warm_start=True)
    print(
        f"  converged={r_warm.converged}  n_iter={r_warm.n_iter}  "
        f"max_force={r_warm.max_force:.4e} Ha/bohr  wall={t_warm:.2f}s"
    )

    print("\nSpeedup:")
    if t_cold > 0:
        speedup = t_cold / t_warm
        print(f"  {speedup:.2f}× (cold {t_cold:.2f}s → warm {t_warm:.2f}s)")

    # Measure numerical agreement with the independently cold-started path.
    print("\nWarm/cold numerical agreement:")
    energy_delta = np.max(np.abs(r_warm.energies - r_cold.energies))
    print(f"  max |ΔE| across path = {energy_delta:.3e} Ha")
    ts_cold = np.array(
        [a.xyz for a in r_cold.path.images[r_cold.transition_state_index].system.atoms]
    )
    ts_warm = np.array(
        [a.xyz for a in r_warm.path.images[r_warm.transition_state_index].system.atoms]
    )
    pos_delta = float(np.max(np.abs(ts_warm - ts_cold)))
    print(f"  max |Δr_TS|         = {pos_delta:.3e} bohr")

    # Verify the climbing image lands at the symmetric saddle.
    z_ts = ts_warm[:, 2]
    midpoint = 0.5 * (z_ts[0] + z_ts[2])
    print("\nCI-NEB saddle symmetry (warm path):")
    print(f"  outer H z       = ({z_ts[0]:.6f}, {z_ts[2]:.6f}) bohr")
    print(f"  central H z     = {z_ts[1]:.6f} bohr")
    print(f"  midpoint of outer = {midpoint:.6f} bohr")
    print(f"  |z_B - midpoint|  = {abs(z_ts[1] - midpoint):.3e} bohr")

    # Write QVFs so the trajectories animate in vibe-view.
    QVF_WARM.unlink(missing_ok=True)
    QVF_COLD.unlink(missing_ok=True)
    r_warm.write_qvf(QVF_WARM)
    r_cold.write_qvf(QVF_COLD)
    print(f"\nWrote {QVF_WARM.with_suffix('.qvf').name}")
    print(f"Wrote {QVF_COLD.with_suffix('.qvf').name}")

    # Text summary.
    with open(TEXT_OUT, "w", encoding="utf-8") as f:
        f.write("H + H₂ → H₂ + H climbing-image NEB benchmark\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Cold-start: {t_cold:.3f}s, n_iter={r_cold.n_iter}\n")
        f.write(f"Warm-start: {t_warm:.3f}s, n_iter={r_warm.n_iter}\n")
        if t_cold > 0:
            f.write(f"Speedup:    {t_cold / t_warm:.2f}×\n")
        f.write(f"\nNumerical agreement (warm vs cold):\n")
        f.write(f"  max |ΔE| = {energy_delta:.3e} Ha\n")
        f.write(f"  max |Δr| at TS = {pos_delta:.3e} bohr\n")
        f.write(f"\nCI-NEB saddle (warm):\n")
        f.write(f"  outer H z = ({z_ts[0]:.6f}, {z_ts[2]:.6f}) bohr\n")
        f.write(f"  central H z = {z_ts[1]:.6f} bohr\n")
        f.write(f"  |z_B - midpoint| = {abs(z_ts[1] - midpoint):.3e} bohr\n")
        f.write(f"\nE_TS = {r_warm.energies[r_warm.transition_state_index]:.10f} Ha\n")
    print(f"Wrote {TEXT_OUT.name}")


if __name__ == "__main__":
    main()
