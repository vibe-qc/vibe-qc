"""Level-shift SCF trace — tutorial 24 figure.

Runs three Γ-point Ewald SCFs on a tight 3D LiH cubic cell
(a = 4.5 bohr, sto-3g) with level_shift ∈ {0.0, 0.3, 0.7} Ha and
plots the orbital-gradient norm |∇L| against iteration. All three
runs converge to the same total energy (the inertness-at-convergence
property of the Saunders–Hillier shift); the *trajectory* visibly
differs:

  level_shift = 0.0  → fastest per step but most aggressive — risks
                       oscillating away from the SCF fixed point on
                       harder cases;
  level_shift = 0.3  → moderate — typical hard-case recommendation;
  level_shift = 0.7  → very stable, slowest convergence — only
                       needed when 0.3 still oscillates.

DIIS is intentionally disabled (use_diis = False) so the level-shift
effect is visible. With DIIS on, both Pulay extrapolation and the
level shift act as damping mechanisms — the level shift is most
visible at startup before DIIS has built a useful subspace.

Output: ``docs/_static/plots/level-shift-scf-trace.png``.

Run:
    .venv/bin/python examples/plots/level-shift-scf-trace.py
"""

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

import vibeqc as vq

HERE = Path(__file__).resolve().parent
PLOT_OUT = HERE.parent.parent / "docs" / "_static" / "plots" / "level-shift-scf-trace.png"

A = 4.5  # bohr — tight 3D LiH lattice parameter


def grad_norms(scf_trace) -> list[float]:
    """Return |grad| for each SCF iteration. As of vibe-qc v0.4 (commit
    347ea71) every SCF backend returns ``scf_trace`` items as
    :class:`vibeqc.SCFIteration` dataclasses with a ``.grad_norm``
    attribute — the older tuple-vs-dataclass split has been unified."""
    return [float(it.grad_norm) for it in scf_trace]


def main() -> None:
    t_total = time.perf_counter()

    # Tight LiH cubic cell. sto-3g for speed; off-FFT-grid shift
    # (0.05 bohr) so corner atoms don't inflate the omega-residual.
    sysp = vq.PeriodicSystem(
        3, np.eye(3) * A,
        [vq.Atom(3, [0.05,         0.05,         0.05]),
         vq.Atom(1, [0.5*A + 0.05, 0.5*A + 0.05, 0.5*A + 0.05])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")

    opts = vq.PeriodicSCFOptions()
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.lattice_opts.cutoff_bohr = 10.0
    opts.lattice_opts.nuclear_cutoff_bohr = 15.0
    # Disable DIIS so the level-shift effect is visible. (In production,
    # use_diis = True is fine — level_shift then accelerates DIIS by
    # giving it cleaner per-step inputs to extrapolate.)
    opts.use_diis = False
    opts.damping = 0.1
    opts.max_iter = 30
    opts.conv_tol_energy = 1e-7
    opts.conv_tol_grad = 1e-5

    runs: list[tuple[float, vq.PeriodicRHFEwaldResult, list[float], float]] = []
    for ls in (0.0, 0.3, 0.7):
        opts.level_shift = ls
        t0 = time.perf_counter()
        r = vq.run_rhf_periodic_gamma_scf(
            sysp, basis, opts, omega=0.5, spacing_bohr=0.5,
        )
        dt = time.perf_counter() - t0
        gn = grad_norms(r.scf_trace)
        runs.append((ls, r, gn, dt))
        print(f"  level_shift = {ls:>3.1f}  iters = {r.n_iter:>3d}  "
              f"E = {r.energy:.6f} Ha   ({dt:.0f} s wall)")

    # Energy-inertness check: every level_shift converges to the same
    # total energy.
    energies = np.array([r.energy for (_, r, _, _) in runs])
    spread = float(np.max(energies) - np.min(energies))
    print(f"  energy spread across runs: {spread:.2e} Ha (target < 1e-6)")
    assert spread < 1e-5, (
        f"level_shift broke inertness-at-convergence: spread = {spread:.2e}"
    )

    PLOT_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)

    colours = {0.0: "#1f77b4", 0.3: "#2ca02c", 0.7: "#d62728"}
    markers = {0.0: "o",       0.3: "s",       0.7: "^"}

    for (ls, r, gn, _) in runs:
        ax.semilogy(np.arange(1, len(gn) + 1), gn,
                    "-" + markers[ls],
                    color=colours[ls], markersize=7, linewidth=1.6,
                    label=fr"level_shift $b = {ls}$ Ha "
                          fr"({r.n_iter} iters)")
    ax.axhline(opts.conv_tol_grad, color="grey", linestyle=":",
               linewidth=1.0, label=fr"conv_tol_grad = {opts.conv_tol_grad}")

    ax.set_xlabel("SCF iteration")
    ax.set_ylabel(r"$\|\nabla L\|$ (orbital-gradient norm, Ha)")
    ax.set_title(
        "Level shift trades wall-clock for stability\n"
        "Tight 3D LiH cubic cell (a = 4.5 bohr, sto-3g, EWALD_3D, "
        "no DIIS)",
        fontsize=10.5,
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3, which="both", linestyle=":")

    fig.tight_layout()
    fig.savefig(PLOT_OUT, dpi=300, bbox_inches="tight")

    print(f"\nWrote {PLOT_OUT.relative_to(HERE.parent.parent)}  "
          f"(total {time.perf_counter()-t_total:.0f} s)")


if __name__ == "__main__":
    main()
