"""MgO [2,2,2] multi-k corrected-gauge parity vs PySCF (option (b) Phase 3 M3c).

Runs MgO fcc primitive / STO-3G at the [2,2,2] Monkhorst-Pack mesh
with the corrected exchange gauge (use_exchange_ewald_split=True) at
a fold-converged cutoff, from SAD:

* RHF      — PySCF KRHF GDF exxdiv='ewald' reference: −271.213356
* RKS/SVWN — PySCF KRKS (slater,vwn5) reference:      −270.499374
  (with Fermi-Dirac smearing 0.01 Ha; the FMIXING auto default no
  longer fires under DIIS since the 2026-07-13 Gap-B validation)

Expected agreement: tens-of-mHa truncation scale at cutoff 12 (the Γ
c12 run landed +21 mHa from PySCF). The two documented traps apply:
heed the S(Γ) drift guard line in the log, and remember the
ionic-Γ/multi-k basin multiplicity — a discrepancy beyond ~50 mHa
warrants a warm-start basin check before declaring a gauge bug.

Self-contained vq payload (CLAUDE.md §15): runs against the host's
DEPLOYED vibeqc-dev package; writes JSON + logs to $VQ_WORKDIR when
the queue provides it, otherwise to VIBEQC_RUNS_DIR or ~/vibeqc-runs.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from examples.regression.core.output_paths import make_run_id, resolve_output_root


def _payload_workdir() -> Path:
    vq_workdir = os.environ.get("VQ_WORKDIR")
    if vq_workdir:
        out = Path(vq_workdir).expanduser().resolve(strict=False)
    else:
        run_id = os.environ.get("VIBEQC_REGRESSION_RUN_ID") or make_run_id(
            "bipole-mgo-multik"
        )
        out = resolve_output_root(create=True) / run_id
    out.mkdir(parents=True, exist_ok=True)
    return out


workdir = _payload_workdir()
out: dict = {"phase": "bipole-phase3-mgo-multik-parity"}

import vibeqc as vq  # noqa: E402
from vibeqc._vibeqc_core import (  # noqa: E402
    InitialGuess,
    PeriodicKSOptions,
    PeriodicRHFOptions,
    monkhorst_pack,
)

# Gate: deployed package must carry the Phase 3 multi-k API.
try:
    from vibeqc.bipole_fock_ewald import (  # noqa: F401
        compute_K_long_range_at_k,
        probe_charge_madelung_supercell,
    )
    from vibeqc.pbc_bipole_common import (  # noqa: F401
        bvk_torus_density_matrices,
    )

    out["phase3_api"] = True
except Exception as exc:  # noqa: BLE001
    out["phase3_api"] = False
    out["gate_error"] = repr(exc)
    (workdir / "mgo_multik_parity.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out))
    raise SystemExit(78)

ANG2BOHR = 1.0 / 0.529177210903
a = 4.21 * ANG2BOHR
lattice = (a / 2.0) * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
sysp = vq.PeriodicSystem(
    3, lattice, [vq.Atom(12, [0, 0, 0]), vq.Atom(8, [a / 2] * 3)]
)
basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
kmesh = monkhorst_pack(sysp, [2, 2, 2])

xi_sc = probe_charge_madelung_supercell(sysp, (2, 2, 2))
out["xi_supercell"] = xi_sc
assert abs(xi_sc - 0.288147805800) < 1e-9, f"supercell madelung pin: {xi_sc}"

CUTOFF = 12.0
E_PYSCF_KRHF = -271.213356
E_PYSCF_KRKS_SVWN = -270.499374

# ---- RHF [2,2,2], corrected gauge, from SAD ---------------------------
from vibeqc.pbc_bipole import run_pbc_bipole_rhf  # noqa: E402

opts = PeriodicRHFOptions()
opts.lattice_opts.cutoff_bohr = CUTOFF
opts.lattice_opts.nuclear_cutoff_bohr = CUTOFF
opts.max_iter = 80
opts.use_diis = True
opts.initial_guess = InitialGuess.SAD
opts.conv_tol_energy = 1e-8
opts.conv_tol_grad = 1e-6

t0 = time.time()
r = run_pbc_bipole_rhf(
    sysp,
    basis,
    kmesh,
    opts,
    use_exchange_ewald_split=True,
    ewald_precision=1e-8,
    progress=True,
)
out["rhf"] = {
    "energy": float(r.energy),
    "converged": bool(r.converged),
    "n_iter": int(r.n_iter),
    "pyscf_ref": E_PYSCF_KRHF,
    "diff_mHa": (float(r.energy) - E_PYSCF_KRHF) * 1e3,
    "wall_s": time.time() - t0,
}
print("RHF [2,2,2] c12:", json.dumps(out["rhf"]))

# ---- RKS/SVWN [2,2,2], corrected gauge, from SAD ----------------------
# BONUS check, NOT the gauge certification (RHF above is). MgO/STO-3G
# RKS is a pathological minimal-basis basin: the STO-3G gap is only
# ~0.3 eV (real MgO ~7.8 eV), so the SCF basin is near-degenerate and
# smearing-sensitive — at Γ, at multi-k, AND in PySCF (which lands in
# different basins from different starts; 2026-06-10 Gap-B entry).
# The 2026-06-11 phase-4a script's T=0.01 Ha (~0.27 eV ≈ the gap)
# forced a fractionally-occupied near-metallic basin → +330 mHa,
# 42 mHa entropy (2026-06-13 compute-small round). Here we use a SUB-GAP
# smearing (T=0.002 Ha ≈ 0.05 eV, ~1/5 the gap) and REPORT basin
# health (entropy + fractional-occupation count) so the result is
# self-documenting: low entropy + integer occ ⇒ physical basin ⇒
# the diff is a real gauge+truncation number; otherwise it is
# Gap-B-confounded and the diff is not a gauge verdict.
from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks  # noqa: E402

ks = PeriodicKSOptions()
ks.functional = "svwn"
ks.lattice_opts.cutoff_bohr = CUTOFF
ks.lattice_opts.nuclear_cutoff_bohr = CUTOFF
ks.max_iter = 200
ks.use_diis = True
ks.initial_guess = InitialGuess.SAD
ks.conv_tol_energy = 1e-7
ks.conv_tol_grad = 1e-5
ks.smearing_temperature = 0.002  # SUB-GAP (gap ~0.3 eV); see note above

t0 = time.time()
rk = run_pbc_bipole_rks(
    sysp,
    basis,
    kmesh,
    ks,
    use_exchange_ewald_split=True,
    ewald_precision=1e-8,
    progress=True,
)
# Basin-health metrics: a wide-gap insulator in the physical basin
# has ~integer occupations (entropy → 0). Fractional occupations near
# the Fermi level signal the near-metallic wrong basin.
occ0 = np.asarray(rk.occupations[0]) if len(rk.occupations) else np.array([])
n_frac = int(((occ0 > 1e-4) & (occ0 < 2.0 - 1e-4)).sum()) if occ0.size else -1
eps0 = np.sort(np.real(np.asarray(rk.mo_energies[0])))
n_occ = int(sysp.n_electrons()) // 2
gap_eV = float(eps0[n_occ] - eps0[n_occ - 1]) * 27.211386 if eps0.size > n_occ else None
basin_healthy = bool(rk.entropy < 1e-2 and n_frac in (0,))
out["rks_svwn"] = {
    "energy": float(rk.energy),
    "free_energy": float(rk.free_energy),
    "entropy": float(rk.entropy),
    "n_fractional_occ": n_frac,
    "homo_lumo_gap_eV": gap_eV,
    "basin_healthy": basin_healthy,
    "converged": bool(rk.converged),
    "n_iter": int(rk.n_iter),
    "pyscf_ref": E_PYSCF_KRKS_SVWN,
    "diff_mHa": (float(rk.energy) - E_PYSCF_KRKS_SVWN) * 1e3,
    "verdict": (
        "gauge+truncation diff (physical basin)" if basin_healthy
        else "Gap-B-confounded (near-degenerate basin; NOT a gauge verdict)"
    ),
    "wall_s": time.time() - t0,
}
print("RKS/SVWN [2,2,2] c12:", json.dumps(out["rks_svwn"]))

(workdir / "mgo_multik_parity.json").write_text(json.dumps(out, indent=2))
print(json.dumps({k: out[k] for k in ("phase3_api", "xi_supercell")}))
# Exit status keys on the HEADLINE RHF certification only; the RKS leg
# is diagnostic (a near-degenerate-basin non-convergence must not red
# the gauge certification).
ok = out["rhf"]["converged"]
raise SystemExit(0 if ok else 1)
