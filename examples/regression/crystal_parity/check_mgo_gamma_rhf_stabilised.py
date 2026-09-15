"""Diagnostic: vibe-qc gamma RHF GDF on MgO/sto-3g WITH stabilizers.

The "no FMIXING / no LEVSHIFT" version diverges (consistent with
CRYSTAL14 also failing with its default 30% FMIXING). MgO HF on
tight ionic crystals is genuinely a hard SCF problem, not a
vibe-qc bug.

The question for v0.8.0: does vibe-qc's native gamma RHF GDF
converge MgO HF/sto-3g with proper stabilizers (matching CRYSTAL's
ionic-HF recipe of FMIXING + LEVSHIFT)?

If yes: the 5% metric error in the native Lpq is not blocking;
v0.8.0 deliverable is "restricted RHF/RKS works on all Bravais
with stabilizers" and we have parity with CRYSTAL14.

If no: the native Lpq metric error is structural; slice 3d
(compcell + Ewald-split) is critical-path.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("PYSCF_TMPDIR", "/tmp/pyscf_vibeqc")

import numpy as np
import vibeqc as vq


def main() -> int:
    a_ang = 4.211
    ang2bohr = 1.0 / 0.529177210903
    a = a_ang * ang2bohr
    mg_frac = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    o_frac = [(0.5, 0.5, 0.5), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)]
    atoms = []
    for fx, fy, fz in mg_frac:
        atoms.append(vq.Atom(12, [fx * a, fy * a, fz * a]))
    for fx, fy, fz in o_frac:
        atoms.append(vq.Atom(8, [fx * a, fy * a, fz * a]))
    system = vq.PeriodicSystem(3, np.diag([a, a, a]), atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    print(f"vibeqc {vq.__version__} — MgO RHF/sto-3g/Γ native GDF")
    print(f"  with FMIXING 30%, LEVSHIFT 0.5 Ha (5-cycle warm-up)")
    print()

    opts = vq.PeriodicRHFOptions()
    opts.use_diis = True
    opts.damping = 0.0
    opts.fock_mixing = 0.30
    opts.level_shift = 0.5
    opts.level_shift_warmup_cycles = 5
    opts.max_iter = 60
    opts.conv_tol_energy = 1e-8

    result = vq.run_rhf_periodic_gamma_gdf(
        system, basis, opts, progress=True, verbose=2,
    )
    print()
    print(f"FINAL E    = {result.energy:.10f} Ha")
    print(f"  n_iter   = {result.n_iter}")
    print(f"  converged = {result.converged}")
    delta = float(result.energy) - (-1085.2315464583)
    print(f"  delta vs committed (PySCF-Lpq era): {delta:.3e} Ha")
    if result.converged and abs(delta) < 1e-3:
        print()
        print("VERDICT: native GDF + stabilisers reproduces the PySCF-Lpq "
              "era result. 5% metric error is tolerable with stabilisers.")
        return 0
    if result.converged:
        print()
        print(f"VERDICT: converged but to a different energy. Likely the "
              f"5% Lpq metric error shifts the answer by ~{delta:.1f} Ha. "
              f"Need to fix the metric (slice 3d) before v0.8.0.")
        return 2
    print()
    print("VERDICT: even with stabilisers the native GDF doesn't converge. "
          "Critical: native Lpq metric error is structural; slice 3d "
          "(compcell + Ewald-split) is v0.8.0 critical-path.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
