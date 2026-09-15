"""Parity test: LiH primitive RHF/STO-3G at kmesh=(8,8,8) vs CRYSTAL14
SHRINK 8 8 -7.9381812655 Ha.

This is the correct parity comparison after the 2026-05-17 finding
that the LiH/STO-3G primitive RHF SCF is intrinsically unstable at
small kmeshes (SHRINK 2 2 and 4 4) in CRYSTAL14 itself. Only
SHRINK 8 8 gives a smooth, well-defined SCF convergence.

Uses CRYSTAL-defaults-equivalent SCF aids:
  - Initial guess: SAD
  - Density mixing (CRYSTAL's FMIXING 30%): opts.damping = 0.3
  - DIIS: ON, starts iter 2 (vibe-qc default)
  - LEVSHIFT: OFF (CRYSTAL default)
  - MOM:      OFF (not in CRYSTAL's stack)
  - ODA:      OFF

The K-matrix Madelung shift (PySCF's `exxdiv='ewald'` analogue) is
NOT applied here — per handover §A.7 ("K — SHELLEXCH — no monopole
issue (antisymmetric)") CRYSTAL has no analogue and vibe-qc's K
from `build_fock_2e_real_space` should share the antisymmetric-no-
monopole property. The K-Madelung helpers live on a hold-out
branch pending empirical verification (whether this script lands
within ±1 mHa of CRYSTAL's -7.93818 is the test).

Uses symmetry reduction (FCC SG 225): 512 → 29 irreducible k-points.

Reference: CRYSTAL14 local run on the M5 Max (see
`lih-primitive-shrink-8-8.d12`), converged in 12 cycles to
-7.9381812655 Ha. Matches the sealed PARITY_TABLE.md baseline.
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
    attach_symmetry,
    monkhorst_pack,
)
from vibeqc._vibeqc_core import PeriodicRHFOptions
from vibeqc.periodic_rhf_multi_k_ewald import (
    run_rhf_periodic_multi_k_ewald3d,
)

ANG2BOHR = 1.0 / 0.529177210903

CRYSTAL14_REF = -7.93818127      # SHRINK 8 8 converged, sealed
TARGET_MILLIHARTREE = 1.0


def build_lih_primitive():
    a = 4.084 * ANG2BOHR
    lattice = (a / 2.0) * np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ])
    atoms = [
        vq.Atom(3, [0.0, 0.0, 0.0]),
        vq.Atom(1, [a / 2.0, a / 2.0, a / 2.0]),
    ]
    system = vq.PeriodicSystem(3, lattice, atoms)
    attach_symmetry(system)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def main() -> int:
    print("=== LiH primitive parity vs CRYSTAL14 SHRINK 8 8 ===")
    system, basis = build_lih_primitive()
    print(f"  basis: STO-3G ({basis.nbasis} BFs / {basis.nshells} shells)")
    print(f"  n_electrons = {system.n_electrons()}")

    kmesh = monkhorst_pack(system, [8, 8, 8], use_symmetry=True)
    n_k = len(list(kmesh.kpoints))
    print(f"  kmesh=(8,8,8), {n_k} irreducible k-points "
          f"(SG 225 FCC reduction)")

    opts = PeriodicRHFOptions()
    # cutoff 12 was sufficient at kmesh=(2,2,2) but at (8,8,8) the
    # preflight catches negative-eigval S(k) at zone-boundary k=3
    # (Bloch sum mis-converged for the denser sampling). Bump to 18.
    opts.lattice_opts.cutoff_bohr = 18.0
    opts.lattice_opts.nuclear_cutoff_bohr = 18.0
    opts.lattice_opts.coulomb_method = CoulombMethod.EWALD_3D
    opts.initial_guess = InitialGuess.SAD
    opts.max_iter = 30
    opts.use_diis = True
    opts.diis_start_iter = 2
    opts.damping = 0.3              # ≈ CRYSTAL's FMIXING 30%
    opts.conv_tol_energy = 1e-7
    opts.conv_tol_grad = 1e-4

    print(f"  SCF aids (CRYSTAL-defaults-equivalent):")
    print(f"    initial guess = SAD")
    print(f"    damping       = {opts.damping} (~ CRYSTAL FMIXING 30%)")
    print(f"    use_diis      = {opts.use_diis}, diis_start_iter = {opts.diis_start_iter}")
    print(f"    LEVSHIFT      = OFF")
    print(f"    MOM           = OFF")
    print(f"    ODA           = OFF")
    print()

    t0 = time.time()
    result = run_rhf_periodic_multi_k_ewald3d(
        system, basis, kmesh, opts,
        # All "new" features OFF — pure CRYSTAL-defaults SCF
        level_shift_schedule=None,
        use_mom=False,
        use_oda=False,
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
    delta_mha = (result.energy - CRYSTAL14_REF) * 1000
    print(f"  CRYSTAL14 SHRINK 8 8:  {CRYSTAL14_REF:.8f} Ha/FU")
    print(f"  Δ:                      {delta_mha:+.4f} mHa")
    print(f"  target:                ±{TARGET_MILLIHARTREE} mHa")

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
