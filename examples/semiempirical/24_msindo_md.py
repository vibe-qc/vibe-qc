"""Born-Oppenheimer molecular dynamics on the MSINDO surface (H2O).

vibe-qc's MD driver (:mod:`vibeqc.md`) is **method-agnostic**: it integrates
Newton's equations for any ``force_fn(coords) -> (energy, gradient)`` callable,
the same energy+gradient seam ``run_neb`` uses. Here MSINDO supplies the
energy (``run_msindo``) and a finite-difference nuclear gradient, wired in by
``run_msindo_md`` / ``msindo_force_provider``.

What this script demonstrates / validates:
  1. **NVE** (microcanonical): the symplectic velocity-Verlet integrator
     conserves the total energy with no secular drift.
  2. **NVT** (canonical): the Berendsen and Nosé-Hoover thermostats both drive
     the mean temperature to the target; the Nosé-Hoover extended-system
     energy H_NH is itself conserved.

Cost note: MSINDO exposes a finite-difference gradient, so each MD step costs
6N+1 ``run_msindo`` SCFs. H2O is tiny (N = 3), keeping the whole run to a few
seconds — but this is a short-trajectory demonstration, not production MD.

Run:  .venv/bin/python examples/semiempirical/24_msindo_md.py
"""

from __future__ import annotations

import numpy as np

from vibeqc.semiempirical.methods.msindo import run_msindo_md

# H2O near its MSINDO minimum (Ångström).
Z = [8, 1, 1]
XYZ = np.array(
    [[0.0, 0.0, 0.0],
     [0.0, 0.7572, 0.5865],
     [0.0, -0.7572, 0.5865]]
)


def main() -> None:
    print(__doc__)

    # --- NVE: energy conservation -------------------------------------
    nve = run_msindo_md(Z, XYZ, timestep_fs=0.3, n_steps=40,
                        temperature_K=300.0, thermostat=None, seed=1)
    e = nve.total_energy
    print("NVE (microcanonical)")
    print(f"  E0            = {e[0]:.8f} Ha")
    print(f"  energy drift  = {nve.energy_drift():.2e} Ha "
          f"(rel {nve.energy_drift() / abs(e[0]):.1e})")
    print(f"  T range       = {nve.temperature.min():.0f}..{nve.temperature.max():.0f} K")

    # --- NVT: Berendsen + Nosé-Hoover temperature control -------------
    for thermostat in ("berendsen", "nose_hoover"):
        nvt = run_msindo_md(Z, XYZ, timestep_fs=0.3, n_steps=120,
                            temperature_K=350.0, thermostat=thermostat,
                            thermostat_tau_fs=25.0, seed=2)
        label = thermostat.replace("_", "-").title()
        line = (f"NVT [{label}]  mean T (2nd half) = "
                f"{nvt.mean_temperature():.0f} K  (target 350 K)")
        if thermostat == "nose_hoover":
            h = nvt.conserved_energy
            line += f"   H_NH drift = {np.max(np.abs(h - h[0])):.2e} Ha"
        print(line)

    # Emit the citation sidecars (MSINDO + velocity-Verlet + thermostat).
    run_msindo_md(Z, XYZ, timestep_fs=0.3, n_steps=5, thermostat="nose_hoover",
                  seed=1, output="msindo_md_h2o")
    print("\nWrote msindo_md_h2o.bibtex / .references")


if __name__ == "__main__":
    main()
