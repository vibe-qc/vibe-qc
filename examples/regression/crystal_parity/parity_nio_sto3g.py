"""NiO bulk UHF/STO-3G BIPOLE smoke runner.

CRYSTAL14 reference input (``crystal_demos/nio_sto3g.d12``):
  - SG 225 rocksalt, a=4.164 Å, Ni(0,0,0) + O(1/2,1/2,1/2), STO-3G
  - UHF, SPINLOCK 2 15 (M_s = 1 -> multiplicity = 3), LEVSHIFT 3 1,
    FMIXING 30, SHRINK 8 8

The UHF BIPOLE driver now exists, so this script is no longer a hard
blocker stub. It is still a smoke runner rather than a sealed parity
check: the committed CRYSTAL14 reference converges to -1563.9384656162
Ha at SHRINK 8, but CRYSTAL's direct-lattice generator reports RMAX
58.88485 bohr for this input. Reaching that case efficiently requires
the native SDR multipole far-pair branch, not the current exact-ERI
Ewald-J scaffold.
"""
from __future__ import annotations

import math
import sys

from vibeqc import InitialGuess, monkhorst_pack
from vibeqc._vibeqc_core import PeriodicRHFOptions
from vibeqc.pbc_bipole_uhf import run_pbc_bipole_uhf

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, "examples/regression/crystal_parity")
from crystal_demos.builders import build_nio_sto3g


CRYSTAL14_REF_HA = -1563.9384656162


def main() -> int:
    system, basis = build_nio_sto3g()
    print("=== NiO STO-3G UHF BIPOLE smoke ===")
    print(f"  dim={system.dim}, n_atoms={len(system.unit_cell)}, "
          f"n_elec={system.n_electrons()}, "
          f"multiplicity={system.multiplicity}")
    print(f"  basis: STO-3G ({basis.nbasis} BFs / {basis.nshells} shells)")

    kmesh = monkhorst_pack(system, [1, 1, 1], use_symmetry=False)
    opts = PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 5.0
    opts.lattice_opts.nuclear_cutoff_bohr = 10.0
    opts.initial_guess = InitialGuess.HCORE
    opts.max_iter = 1
    opts.use_diis = False

    result = run_pbc_bipole_uhf(
        system, basis, kmesh, opts,
        ewald_precision=1e-6,
        progress=True,
        verbose=3,
    )

    print()
    print("=== Result ===")
    print(f"  iterations: {result.n_iter}")
    print(f"  E_total:    {result.energy:.10f} Ha")
    print(f"  E_elec:     {result.e_electronic:.10f} Ha")
    print(f"  E_nuc:      {result.e_nuclear:.10f} Ha")
    print(f"  <S^2>:      {result.s_squared:.10f} "
          f"(ideal {result.s_squared_ideal:.10f})")
    print(f"  CRYSTAL14 SHRINK 8 reference: {CRYSTAL14_REF_HA:.10f} Ha")
    print(f"  non-comparable smoke delta:   "
          f"{(result.energy - CRYSTAL14_REF_HA) * 1000.0:+.3f} mHa")
    print()
    print("  NOTE: smoke only; the committed CRYSTAL reference uses a "
          "58.88485 bohr effective direct-lattice radius. Use this runner "
          "to validate the UHF plumbing, not final NiO parity.")

    if not math.isfinite(result.energy):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
