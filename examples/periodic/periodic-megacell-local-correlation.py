"""Periodic local correlation via the megacell route — MP2 + DLPNO-CCSD(T).

Demonstrates vibe-qc's **megacell** (finite-supercell) route to periodic
correlation (Nejad et al. 2025, *J. Chem. Phys.* **163**, 214108 — Paper II):
a 1-D chain of H2 molecules (one per cell) is replicated into growing
supercells, and the **per-cell** MP2 correlation energy converges to the bulk
(thermodynamic) limit. The same route reaches periodic **DLPNO-CCSD(T)** by
dispatching the supercell through the molecular ``run_job``.

Why this works without any periodic-exchange correction: a finite supercell is
an ordinary *molecular* calculation, so there is no periodic ``exxdiv`` / G=0
divergence to subtract — the MP2 denominators are clean and the per-cell energy
converges as the megacell grows. (The alternative Γ-Bloch route *does* need a
finite-size correction; see handovers/HANDOVER_PERIODIC_LOCAL_CORRELATION.md § 6.5.)

Run with:
    python examples/periodic/periodic-megacell-local-correlation.py

Sample output (this machine; H2 chain / STO-3G, cell = 6 bohr along z):
    Megacell MP2 — per-cell correlation energy vs supercell size
      ncell=1   E_corr/cell = -0.01315787 Ha
      ncell=2   E_corr/cell = -0.01321478 Ha   (Δ vs prev = -5.69e-05)
      ncell=3   E_corr/cell = -0.01323465 Ha   (Δ vs prev = -1.99e-05)
      ncell=4   E_corr/cell = -0.01324464 Ha   (Δ vs prev = -9.99e-06)
      ncell=5   E_corr/cell = -0.01325065 Ha   (Δ vs prev = -6.00e-06)
    Shrinking increments → converging to the bulk per-cell correlation energy.

    Periodic DLPNO-CCSD(T) via the megacell route (1x1x1, def2-svp):
      E_total = -1.16334798 Ha   E_corr(CCSD) = -0.03444628 Ha   E_(T) = -2.5e-36
    ((T) ≈ 0 for the 2-electron cell; the (1x1x1) megacell reproduces molecular
     DLPNO-CCSD(T) on the unit cell.)
"""

from __future__ import annotations

import numpy as np

import vibeqc as vq
from vibeqc.periodic_megacell_mp2 import megacell_mp2, megacell_run_job


def h2_chain_cell() -> vq.PeriodicSystem:
    """One H2 per cell; periodic along z (6 bohr), vacuum in x, y (20 bohr)."""
    return vq.PeriodicSystem(
        3,
        np.diag([20.0, 20.0, 6.0]),
        [vq.Atom(1, [10.0, 10.0, 2.3]), vq.Atom(1, [10.0, 10.0, 3.7])],
    )


def main() -> None:
    system = h2_chain_cell()

    print("Megacell MP2 — per-cell correlation energy vs supercell size")
    prev = None
    for n in (1, 2, 3, 4, 5):
        r = megacell_mp2(system, "sto-3g", (1, 1, n))
        pc = r.e_corr_per_cell
        delta = "" if prev is None else f"   (Δ vs prev = {pc - prev:+.2e})"
        print(f"  ncell={n}   E_corr/cell = {pc:.8f} Ha{delta}")
        prev = pc
    print("Shrinking increments → converging to the bulk per-cell "
          "correlation energy.\n")

    print("Periodic DLPNO-CCSD(T) via the megacell route (1x1x1, def2-svp):")
    mc = megacell_run_job(system, "def2-svp", (1, 1, 1), method="dlpno-ccsd(t)")
    cc = mc.result.dlpno_ccsd
    print(
        f"  E_total = {mc.e_total:.8f} Ha   "
        f"E_corr(CCSD) = {cc.e_corr:.8f} Ha   E_(T) = {cc.e_t:.3g}"
    )
    print("  (the (1x1x1) megacell reproduces molecular DLPNO-CCSD(T) on the "
          "unit cell.)")


if __name__ == "__main__":
    main()
