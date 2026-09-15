"""BIPOLE MgO/STO-3G SHRINK 2 2 vs CRYSTAL — the exchange-convention finding.

WHAT THIS SHOWS (2026-06-16 finding; see docs/periodic_jk_routes.md
§ Current status):

A fold-converged BIPOLE multi-k SCF (corrected exchange gauge,
``exchange_exxdiv='ewald'``) at MgO/STO-3G SHRINK 2 2 does **NOT** match the
matched-k CRYSTAL SHRINK-2-2 reference (-271.85640 Ha/FU). It is +641 mHa
away — and that is **not a bug**. It is a finite-k exchange finite-size
*convention* difference:

  * BIPOLE applies the probe-charge Ewald (Madelung) exxdiv correction
    ``(ξ_M − π/(V_sc·ω²))·S(k)D(k)S(k)`` (PySCF-equivalent ``exxdiv='ewald'``),
    which removes the leading 1/N_k^{1/3} exchange finite-size error. So
    BIPOLE's SHRINK-2-2 energy is *already k-converged*.
  * CRYSTAL applies no such correction; its SHRINK-2-2 exchange over-binds its
    own converged SHRINK-8-8 value by 638 mHa.

So the physically meaningful cross-family comparison is BIPOLE SHRINK 2 2
(exxdiv-corrected) ↔ CRYSTAL **SHRINK 8 8** (k-converged): they agree to a few
mHa (STO-3G truncation scale). The matched-k SHRINK-2-2-vs-SHRINK-2-2 premise
compares a k-converged value (BIPOLE) against an under-converged value
(CRYSTAL) and is therefore invalid.

CRYSTAL references (local CRYSTAL23 oracle, §10; sealed values):

  * SHRINK 2 2 : -271.85640 Ha/FU   (coarse, uncorrected — the WRONG target)
  * SHRINK 8 8 : -271.21814 Ha/FU   (k-converged — the valid cross-family target)

Run (laptop, ~12 min at c12; the per-iter cost is the C++ direct-ERI build):

    python examples/regression/crystal_parity/parity_mgo_shrink22_crystal_convention.py
"""
from __future__ import annotations

import sys

import numpy as np

import vibeqc as vq
from vibeqc._vibeqc_core import InitialGuess, PeriodicRHFOptions, monkhorst_pack
from vibeqc.pbc_bipole import run_pbc_bipole_rhf

# CRYSTAL23 oracle (§10, out-of-process), Ha per formula unit.
E_CRYSTAL_SHRINK22 = -271.85639652   # coarse, uncorrected exchange
E_CRYSTAL_SHRINK88 = -271.21814375   # k-converged  (the valid target)
# PySCF KRHF GDF exxdiv='ewald' [2,2,2] (same exxdiv convention as BIPOLE).
E_PYSCF_KRHF_222 = -271.213356

CUTOFF = 12.0          # S(k)-fold ~5e-3; metric ΣwTr[DS]=20.000 (physical basin)
# Cross-family agreement tolerance vs the k-converged CRYSTAL value. The c12
# lattice-sum truncation (~5 mHa) dominates; loosen for a faster/coarser run,
# tighten (and raise CUTOFF to ~16) for a sub-mHa pin.
TOL_VS_CRYSTAL_SHRINK88_MHA = 10.0


def build_mgo():
    ang2bohr = 1.0 / 0.529177210903
    a = 4.21 * ang2bohr
    lattice = (a / 2.0) * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    sysp = vq.PeriodicSystem(
        3, lattice, [vq.Atom(12, [0, 0, 0]), vq.Atom(8, [a / 2] * 3)]
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def main() -> int:
    sysp, basis = build_mgo()
    kmesh = monkhorst_pack(sysp, [2, 2, 2])

    opts = PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = CUTOFF
    opts.lattice_opts.nuclear_cutoff_bohr = CUTOFF
    opts.max_iter = 60
    opts.use_diis = True
    opts.initial_guess = InitialGuess.SAD
    opts.conv_tol_energy = 1e-8
    opts.conv_tol_grad = 1e-6

    r = run_pbc_bipole_rhf(
        sysp,
        basis,
        kmesh,
        opts,
        use_exchange_ewald_split=True,   # corrected gauge (exxdiv='ewald')
        use_incremental_fock=True,
        ewald_precision=1e-8,
        progress=True,
    )
    E = float(r.energy)

    # Metric validity: Σ_k w_k Tr[D(k)S(k)] must equal N_elec (=20). A value
    # near 19.57 would be the cutoff-8 metric-invalid (spurious) state.
    n_occ = int(sysp.n_electrons()) // 2
    weights = [1.0 / len(r.mo_coeffs)] * len(r.mo_coeffs)
    metric = 0.0
    for C, S, w in zip(r.mo_coeffs, r.overlap, weights):
        C = np.asarray(C)
        D = 2.0 * (C[:, :n_occ] @ C[:, :n_occ].conj().T)
        metric += w * np.real(np.trace(D @ np.asarray(S)))

    d_s22 = (E - E_CRYSTAL_SHRINK22) * 1e3
    d_s88 = (E - E_CRYSTAL_SHRINK88) * 1e3
    d_pyscf = (E - E_PYSCF_KRHF_222) * 1e3

    print(f"\nBIPOLE MgO/STO-3G SHRINK 2 2, exxdiv='ewald', cutoff={CUTOFF}")
    print(f"  E_total = {E:.8f} Ha/FU   (converged={r.converged}, n_iter={r.n_iter})")
    print(f"  metric Σ_k w_k Tr[D(k)S(k)] = {metric:.6f}  (target {sysp.n_electrons()})")
    print(f"  Δ vs CRYSTAL SHRINK 2 2 (coarse, WRONG target) = {d_s22:+9.2f} mHa")
    print(f"  Δ vs CRYSTAL SHRINK 8 8 (k-converged, VALID)   = {d_s88:+9.2f} mHa")
    print(f"  Δ vs PySCF KRHF [2,2,2] exxdiv='ewald'         = {d_pyscf:+9.2f} mHa")

    ok = (
        r.converged
        and abs(metric - float(sysp.n_electrons())) < 1e-3
        and abs(d_s88) < TOL_VS_CRYSTAL_SHRINK88_MHA
    )
    if not ok:
        print("\nFAIL: did not reproduce the k-converged CRYSTAL value within tolerance")
        return 1
    print(
        f"\nPASS: BIPOLE (coarse, exxdiv-corrected) reproduces CRYSTAL SHRINK 8 8 "
        f"to {abs(d_s88):.1f} mHa — cross-family validated. "
        f"The +{d_s22:.0f} mHa vs CRYSTAL SHRINK 2 2 is the exxdiv convention gap."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
