"""Local parity test: LiH primitive RHF/STO-3G at kmesh=(2,2,2) vs
CRYSTAL14 reference -7.93817 Ha/FU.

Uses the production driver
:func:`vibeqc.periodic_rhf_multi_k_ewald.run_rhf_periodic_multi_k_ewald3d`
with the fix recipe established by the iter-3 diagnostic on this
branch (see `diag_lih_multik_iter3_instrumented.py`):

  - Initial guess:  SAD (closer to physical basin than HCORE).
  - LEVSHIFT:       CRYSTAL14-style schedule [5,4,3,2,1,0.5,0].
  - MOM:            track occupied subspace by max S(k)-overlap.
  - ODA:            Optimal Damping Algorithm (Cancès–Le Bris 2000)
                    — per-iter line search picks the optimal mixing
                    fraction between D_n and D_naive via a quadratic
                    energy model. Costs one EXTRA Fock build per iter.
                    Mutually exclusive with DIIS in this implementation.
  - DIIS:           OFF (ODA replaces it).
  - damping:        OFF (ODA replaces fixed-α mixing).

Earlier runs (no ODA) drifted away from the physical basin:
  - DIIS@4 + damp=0.5  → -21.4 Ha after 40 iters
  - damp=0.5, no DIIS  → -113 Ha after 25 iters
  - quadratic@7, no aids → -25 Ha after 25 iters
  - standalone (no damp, no DIIS) → violent osc -10 ↔ -80 by iter 22
ODA is the principled fix: each iter picks the mixing fraction that
minimises a model of the energy along the line D_n → D_naive.

Runs locally on the M5 Max (per
`reference_local_crystal14` user memory — no need to go to compute-reference
for a 4-electron system). Expected wall time: ~5 min per iter at
cutoff 12 bohr.

Goal: E_total/cell within 1 mHa of CRYSTAL14 -7.93817. Stretch
goal: <100 µHa.
"""
from __future__ import annotations

import sys
import time

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import vibeqc as vq
from vibeqc import (
    CoulombMethod,
    InitialGuess,
    LatticeSumOptions,
    monkhorst_pack,
)
from vibeqc._vibeqc_core import PeriodicRHFOptions
from vibeqc.level_shift_schedule import LevelShiftSchedule
from vibeqc.periodic_rhf_multi_k_ewald import (
    run_rhf_periodic_multi_k_ewald3d,
)

ANG2BOHR = 1.0 / 0.529177210903

# CRYSTAL14 reference (sealed v0.8.0 parity baseline; see
# examples/regression/crystal_parity/baseline_sto3g/PARITY_TABLE.md).
CRYSTAL14_E_PER_FU = -7.93817        # Ha
TARGET_MILLIHARTREE = 1.0


def build_lih_primitive():
    """FCC primitive: 2 atoms / 1 FU; matches CRYSTAL14
    lih-rhf-sto3g.d12 (SG 225, a=4.084 Å, Li(0,0,0), H(0.5,0.5,0.5)
    in conventional fractional coords)."""
    a = 4.084 * ANG2BOHR
    lattice = (a / 2.0) * np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ])
    atoms = [
        vq.Atom(3, [0.0, 0.0, 0.0]),                # Li
        vq.Atom(1, [a / 2.0, a / 2.0, a / 2.0]),     # H
    ]
    system = vq.PeriodicSystem(3, lattice, atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def main() -> int:
    print("=== LiH primitive parity test (vs CRYSTAL14) ===")
    system, basis = build_lih_primitive()
    print(f"  basis: STO-3G ({basis.nbasis} BFs / {basis.nshells} shells)")
    print(f"  n_electrons = {system.n_electrons()}")

    kmesh = monkhorst_pack(system, [2, 2, 2])
    print(f"  kmesh: (2,2,2), {len(list(kmesh.kpoints))} k-points")

    opts = PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 12.0
    opts.lattice_opts.coulomb_method = CoulombMethod.EWALD_3D
    opts.initial_guess = InitialGuess.SAD
    opts.max_iter = 30
    opts.use_diis = False             # ODA replaces DIIS
    opts.damping = 0.0                # ODA replaces fixed-α damping
    opts.quadratic_fallback_iter = 0  # not needed with ODA
    opts.conv_tol_energy = 1e-7
    opts.conv_tol_grad = 1e-4

    schedule = LevelShiftSchedule.crystal_default()
    print(f"  LEVSHIFT schedule: {schedule.as_list()}")
    print(f"  MOM: ON")
    print(f"  ODA: ON")
    print(f"  DIIS: OFF (mutually exclusive with ODA)")
    print(f"  damping: OFF")
    print()

    t0 = time.time()
    result = run_rhf_periodic_multi_k_ewald3d(
        system, basis, kmesh, opts,
        level_shift_schedule=schedule,
        use_mom=True,
        use_oda=True,
        oda_trust_lambda_max=0.3,   # bound D step to avoid basin
                                     # jumps across the long path
                                     # the unbounded model permits
        auto_optimize_truncation=False,
        progress=True, verbose=5,
    )
    wall = time.time() - t0

    print()
    print(f"=== Result ===")
    print(f"  converged:  {result.converged}")
    print(f"  iterations: {result.n_iter}")
    print(f"  wall time:  {wall:.1f}s")
    print(f"  E_total:    {result.energy:.8f} Ha/cell  (= 1 FU)")
    print(f"  E_elec:     {result.e_electronic:.8f} Ha")
    print(f"  E_nuc:      {result.e_nuclear:.8f} Ha")
    print()
    delta_mha = (result.energy - CRYSTAL14_E_PER_FU) * 1000
    print(f"  CRYSTAL14:  {CRYSTAL14_E_PER_FU:.8f} Ha/FU")
    print(f"  Δ:          {delta_mha:+.4f} mHa")
    print(f"  target:     ±{TARGET_MILLIHARTREE} mHa")

    if not result.converged:
        print("\n  FAIL: SCF did not converge")
        return 2
    if abs(delta_mha) > TARGET_MILLIHARTREE:
        print(f"\n  FAIL: Δ exceeds ±{TARGET_MILLIHARTREE} mHa target")
        return 1
    print("\n  PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
