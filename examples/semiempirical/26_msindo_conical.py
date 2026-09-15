"""CIS excited-state gradients + conical-intersection search on the MSINDO surface (H2O).

vibe-qc's CIS excited-state gradient (:mod:`vibeqc.excited_gradient`) and the
penalty-function conical-intersection optimizer (:mod:`vibeqc.conical`) are
**reference-agnostic**: they consume CIS state energies + gradients from any
reference (HF or MSINDO) over the shared ``ERIProvider`` seam. A CIS
excited-state *total* energy is ``E(R) = E_SCF(R) + ω_state(R)``, so its nuclear
gradient is a central difference of that total energy, with the tracked root
followed by CIS-amplitude overlap (the root-flip guard). MSINDO wires this in
through ``msindo_cis_gradient_fd`` and ``msindo_meci``.

What this script demonstrates / validates:
  1. **Excited-state gradient**: the S1 gradient is finite at the ground-state
     geometry (S1 prefers a different structure), while state 0 reproduces the
     plain MSINDO ground-state gradient to machine precision.
  2. **Minimum-energy conical intersection (MECI)**: the Levine-Coe-Martinez
     penalty optimizer drives the S1/S0 gap from its ~7 eV vertical value down
     to a near-degeneracy — using only the two state gradients, with *no*
     nonadiabatic derivative coupling vector (which is exactly what the
     finite-difference CIS gradients supply).

This exercises the optimizer *mechanics* on INDO states, not reference-MSINDO
CIS parity: vibe-qc ships the rigorous (unscaled) CIS, so the absolute
excitation energies differ from the reference program's empirically scaled CIS.

Cost note: the FD CIS gradient costs 6N+1 ``run_msindo`` SCF+CIS evaluations per
geometry; H2O (N = 3) keeps the whole run to a few seconds — a demonstration,
not production excited-state dynamics.

Run:  .venv/bin/python examples/semiempirical/26_msindo_conical.py
"""

from __future__ import annotations

import numpy as np

from vibeqc.semiempirical.methods.msindo import (
    msindo_cis,
    msindo_cis_gradient_fd,
    msindo_gradient_fd,
    msindo_meci,
)

# H2O near its MSINDO ground-state minimum (Ångström).
Z = [8, 1, 1]
XYZ = np.array(
    [[0.0, 0.0, 0.0],
     [0.0, 0.7572, 0.5865],
     [0.0, -0.7572, 0.5865]]
)


def main() -> None:
    print(__doc__)

    # --- 1. CIS excited-state gradient --------------------------------
    s1 = msindo_cis(Z, XYZ, spin="singlet", n_states=4)
    g_s1 = msindo_cis_gradient_fd(Z, XYZ, state=1, spin="singlet")
    g_s0 = msindo_cis_gradient_fd(Z, XYZ, state=0)          # ground state via CIS path
    g_gs = msindo_gradient_fd(Z, XYZ)                       # plain MSINDO gradient
    print("CIS excited-state gradient (H2O, singlet)")
    print(f"  vertical S1            = {s1.excitation_energies_ev[0]:.3f} eV")
    print(f"  |grad S1|              = {np.linalg.norm(g_s1):.5f} Ha/bohr "
          f"(S1 is not at its minimum)")
    print(f"  max|grad S0 - ground|  = {np.max(np.abs(g_s0 - g_gs)):.1e} "
          f"(state 0 == msindo_gradient_fd)")

    # --- 2. Minimum-energy conical intersection (S1/S0) ---------------
    meci = msindo_meci(Z, XYZ, lower_state=0, upper_state=1, spin="singlet",
                       gap_tol=1e-3, max_macro=6)
    print("\nS1/S0 minimum-energy conical intersection (Levine-Coe-Martinez penalty)")
    print(f"  start gap (vertical)   = {s1.excitation_energies_ev[0]:.3f} eV")
    print(f"  optimized gap          = {meci.gap_ev:.4f} eV   "
          f"(converged={meci.converged}, {meci.n_macro} sigma cycles, "
          f"{meci.n_energy_evals} evals)")

    # Emit the citation sidecars (MSINDO + CIS + conical-intersection method).
    msindo_meci(Z, XYZ, lower_state=0, upper_state=1, spin="singlet",
                gap_tol=1e-3, max_macro=1, output="msindo_conical_h2o")
    print("\nWrote msindo_conical_h2o.bibtex / .references")


if __name__ == "__main__":
    main()
