#!/usr/bin/env python
"""DLPNO local-correlation demo on the AICCM-2026 test set (the `aiccm2026dev-a` line).

Runs the full local-correlation stack on a 1-D test-set crystal, built on the
**neutral** four-center reference (the RI-consistent `ccm_neutral_cderi`):

  neutral SCF  →  canonical CCM MP2 / CCSD(T)
               →  DLPNO-MP2     (no-truncation == canonical, + PNO truncation)
               →  DLPNO-CCSD(T) (no-truncation == canonical, + PNO truncation)

The no-truncation limit reproducing the canonical CCM correlation is the hard
correctness gate (also pinned in tests/test_ccm_dlpno_{mp2,ccsd}.py). The neutral
reference is required for CCM correlation (the bare-1/r Madelung background shifts
the occ–virt denominators — see docs/aiccm2026dev_a_followon.md).

Usage:  python demo_dlpno.py [--system h-chain] [--nrep N] [--out FILE]

Reference: Riplinger & Neese, J. Chem. Phys. 138, 034106 (2013); Riplinger et al.,
J. Chem. Phys. 139, 134101 (2013) (DLPNO-CCSD(T)).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import testset  # noqa: E402

from vibeqc.periodic.ccm import CCMSystem, ccm_dlpno_ccsd, ccm_dlpno_mp2  # noqa: E402
from vibeqc.periodic.ccm.ccsd import run_ccm_ccsd  # noqa: E402
from vibeqc.periodic.ccm.mp2 import run_ccm_mp2  # noqa: E402
from vibeqc.periodic.ccm.neutral import ccm_eri_neutral, ccm_neutral_cderi  # noqa: E402
from vibeqc.periodic.ccm.scf import run_ccm_rhf  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--system", default="h-chain", help="test-set system (1-D recommended)")
    ap.add_argument("--nrep", type=int, default=2, help="cluster size along the chain")
    ap.add_argument("--ke-cutoff", type=float, default=120.0, help="neutral-cderi PW cutoff (Ha)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    meta = testset.SYSTEMS[args.system]
    unit = testset.build(args.system)
    basis = meta.get("basis", "sto-3g")
    ccm = CCMSystem(unit, (args.nrep, 1, 1), basis)
    print(f"system={args.system}  nrep=({args.nrep},1,1)  n_atoms={ccm.n_atoms}  "
          f"nbf={ccm.nbf}  basis={basis}")

    # Neutral reference: SCF + canonical correlation on g_eff = Σ_P L⊗L.
    g = ccm_eri_neutral(ccm, ke_cutoff=args.ke_cutoff)
    L = ccm_neutral_cderi(ccm, ke_cutoff=args.ke_cutoff)
    scf = run_ccm_rhf(ccm, eri=g)
    mp2_can = run_ccm_mp2(ccm, scf, eri=g).e_correlation
    cc_can = run_ccm_ccsd(ccm, scf, eri=g)
    print(f"\nneutral SCF E/atom = {scf.energy_per_atom:.8f}")
    print(f"canonical CCM   MP2 Ecorr = {mp2_can:.8f}   "
          f"CCSD(T) Ecorr = {cc_can.e_correlation:.8f}  [(T)={cc_can.e_t:.2e}]")

    out = {"system": args.system, "nrep": [args.nrep, 1, 1], "n_atoms": ccm.n_atoms,
           "basis": basis, "neutral_scf_e_per_atom": scf.energy_per_atom,
           "canonical": {"mp2": mp2_can, "ccsd_t": cc_can.e_correlation, "t": cc_can.e_t},
           "dlpno_mp2": [], "dlpno_ccsd_t": []}

    print(f"\n{'method':14s} {'tcut_pno':>9} {'Ecorr':>13} {'Δ vs canonical':>16} {'detail':>14}")
    for tc in (0.0, 1e-6, 1e-4):
        d = ccm_dlpno_mp2(ccm, scf, cderi=L, localize="pm", tcut_pno=tc)
        print(f"{'DLPNO-MP2':14s} {tc:9.0e} {d.e_corr:13.8f} {d.e_corr-mp2_can:+16.2e} "
              f"{'pno/pair=' + str(max(d.pno_per_pair.values())):>14}")
        out["dlpno_mp2"].append({"tcut_pno": tc, "e_corr": d.e_corr,
                                 "delta_vs_canonical": d.e_corr - mp2_can})
    for tc in (0.0, 1e-6, 1e-4):
        d = ccm_dlpno_ccsd(ccm, scf, cderi=L, g_neutral=g, localize="pm", tcut_pno=tc)
        print(f"{'DLPNO-CCSD(T)':14s} {tc:9.0e} {d.e_correlation:13.8f} "
              f"{d.e_correlation-cc_can.e_correlation:+16.2e} "
              f"{'n_vpno=' + str(d.n_virtual_pno) + '/' + str(d.n_virtual_full):>14}")
        out["dlpno_ccsd_t"].append({"tcut_pno": tc, "e_corr": d.e_correlation,
                                    "delta_vs_canonical": d.e_correlation - cc_can.e_correlation,
                                    "n_virtual_pno": d.n_virtual_pno})

    print("\nGate: at tcut_pno=0 the DLPNO correlation == canonical CCM to machine ε.")
    if args.out:
        args.out.write_text(json.dumps(out, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
