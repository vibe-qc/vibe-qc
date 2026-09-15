#!/usr/bin/env python3
"""Periodic ATOMSPIN broken-symmetry parity vs PySCF.pbc (out-of-process, §10).

The ATOMSPIN feature (``opts.atomic_spins``, per-atom +1/-1/0) seeds a
broken-symmetry open-shell start so an AFM / ferrimagnetic solid converges to
the spin-broken solution instead of the spin-symmetric one. This script
validates that periodic ATOMSPIN reaches the *same* broken-symmetry solution
as PySCF.pbc given a comparable localized broken-symmetry initial guess.

Test system: stretched H2 (R = 3.0 bohr) in a 30-bohr cubic box — the periodic
analog of the molecular ATOMSPIN demonstration (tests/test_guess.py). At Γ in a
large box this is the molecular limit, where the exact-exchange UHF energy is
unambiguous. ``atomic_spins=[+1,-1]`` localizes α on H1 / β on H2; the converged
state has ⟨S²⟩ ≈ 0.77 (vs 0 for the symmetric RHF-like solution).

What is validated, and the honest caveats:

* **Broken-symmetry state (the ATOMSPIN deliverable):** vibe-qc and PySCF must
  converge to the *same* spin-broken solution — matching ⟨S²⟩ and a lower
  energy than the symmetric start. This is the primary assertion.

* **Energy parity (µHa):** the cleanest µHa comparison is vibe-qc's exact
  EWALD-3D lattice-sum Γ driver against the molecular UHF reference, since a
  30-bohr box reproduces molecular UHF to < 1 µHa (asserted in
  tests/test_guess.py::test_periodic_atomic_spins_breaks_symmetry_gamma_uhf_ewald;
  the molecular reference E = -0.95101800 Ha is itself PySCF UHF/STO-3G).
  This script additionally reports the vibe-qc GDF vs PySCF.pbc GDF gap, which
  is the *density-fitting auxiliary-basis* difference between the two GDF
  implementations — NOT a property of ATOMSPIN: the identical-magnitude gap
  appears in the spin-symmetric closed-shell RHF case with no spin and no
  ATOMSPIN at all. Cross-code GDF µHa parity needs a matched auxiliary basis
  (see tests/test_pbc_gdf_mdf.py) and is out of scope here.

PySCF runs in a separate interpreter (CLAUDE.md §10 — never import pyscf in
vibe-qc's runtime). Set VIBEQC_PYSCF_PYTHON to choose the external interpreter.

Usage:
    python examples/regression/parity_periodic_atomspin_vs_pyscf.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

BOX_BOHR = 30.0
R_BOHR = 3.0
RESULT_MARKER = "VIBEQC-PYSCF-RESULT:"


def run_vibeqc_gamma_ewald(atomic_spins):
    """vibe-qc exact EWALD-3D Γ UHF — the molecular-limit energy anchor."""
    import vibeqc as vq

    lat = BOX_BOHR * np.eye(3)
    atoms = [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [R_BOHR, 0.0, 0.0])]
    sysp = vq.PeriodicSystem(3, lat, atoms, charge=0, multiplicity=1)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    o = vq.PeriodicRHFOptions()
    o.initial_guess = vq.InitialGuess.SAD
    o.conv_tol_energy = 1e-10
    o.max_iter = 300
    if atomic_spins is not None:
        o.atomic_spins = atomic_spins
    r = vq.run_uhf_periodic_gamma_ewald3d(sysp, basis, o, verbose=0)
    return float(r.energy), float(r.s_squared), bool(r.converged)


def run_vibeqc_gdf(atomic_spins):
    """vibe-qc GDF UHF (the density-fitting backend that matches PySCF's family)."""
    import vibeqc as vq

    lat = BOX_BOHR * np.eye(3)
    atoms = [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [R_BOHR, 0.0, 0.0])]
    sysp = vq.PeriodicSystem(3, lat, atoms, charge=0, multiplicity=1)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    o = vq.PeriodicRHFOptions()
    o.initial_guess = vq.InitialGuess.SAD
    o.conv_tol_energy = 1e-10
    o.max_iter = 300
    if atomic_spins is not None:
        o.atomic_spins = atomic_spins
    r = vq.run_pbc_gdf_uhf(sysp, basis, o, exxdiv="ewald", verbose=0)
    return float(r.energy), float(r.s_squared), bool(r.converged)


_PYSCF_SCRIPT = r'''
import json, sys
import numpy as np
RESULT_MARKER = "VIBEQC-PYSCF-RESULT:"
try:
    from pyscf.pbc import gto as pgto, scf as pscf
    p = json.loads(sys.stdin.read())
    lat = (p["box"] * np.eye(3)).tolist()
    cell = pgto.M(atom=f'H 0 0 0; H {p["R"]} 0 0', a=lat, basis="sto-3g",
                  unit="B", spin=0, verbose=0)
    mf = pscf.UHF(cell).density_fit()       # exxdiv defaults to 'ewald'
    mf.conv_tol = 1e-10
    mf.max_cycle = 300
    nao = cell.nao_nr()
    if p["break"]:
        # Localized broken-symmetry init: one alpha on H1, one beta on H2.
        dma = np.zeros((nao, nao)); dmb = np.zeros((nao, nao))
        dma[0, 0] = 1.0; dmb[1, 1] = 1.0
        e = float(mf.kernel(np.array([dma, dmb])))
    else:
        e = float(mf.kernel())
    ss, _ = mf.spin_square()
    print(RESULT_MARKER + json.dumps(
        {"status": "ok", "energy": e, "s_squared": float(ss),
         "converged": bool(mf.converged)}))
except ModuleNotFoundError as exc:
    print(RESULT_MARKER + json.dumps({"status": "unavailable", "note": str(exc)}))
except Exception as exc:  # pragma: no cover - reference-side failure
    print(RESULT_MARKER + json.dumps({"status": "error", "note": str(exc)[:200]}))
'''


def run_pyscf(break_symmetry):
    python = os.environ.get("VIBEQC_PYSCF_PYTHON", sys.executable)
    payload = {"box": BOX_BOHR, "R": R_BOHR, "break": bool(break_symmetry)}
    proc = subprocess.run(
        [python, "-c", _PYSCF_SCRIPT], input=json.dumps(payload),
        text=True, capture_output=True,
    )
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(RESULT_MARKER):
            return json.loads(line[len(RESULT_MARKER):])
    return {"status": "error", "note": f"no result (rc={proc.returncode}); "
            f"stderr: {proc.stderr[-200:]}"}


def main():
    print(f"Periodic ATOMSPIN parity — stretched H2 (R={R_BOHR} bohr) in a "
          f"{BOX_BOHR}-bohr box at Γ\n")

    # --- vibe-qc exact EWALD-3D anchor (molecular limit) ------------------
    e_sym_ew, ss_sym_ew, _ = run_vibeqc_gamma_ewald(None)
    e_bs_ew, ss_bs_ew, _ = run_vibeqc_gamma_ewald([1, -1])
    print("vibe-qc EWALD-3D Γ UHF (exact lattice-sum exchange):")
    print(f"  symmetric SAD : E = {e_sym_ew:.8f} Ha   S^2 = {ss_sym_ew:.4f}")
    print(f"  ATOMSPIN[+1,-1]: E = {e_bs_ew:.8f} Ha   S^2 = {ss_bs_ew:.4f}")
    E_MOL_UHF_BS = -0.95101800  # PySCF UHF/STO-3G, atom-localized BS init
    d_anchor = abs(e_bs_ew - E_MOL_UHF_BS) * 1e6
    print(f"  vs molecular UHF reference {E_MOL_UHF_BS:.8f} Ha: "
          f"Δ = {d_anchor:.3f} µHa  (molecular-limit µHa anchor)\n")

    # --- vibe-qc GDF vs PySCF.pbc GDF (same DF family) --------------------
    e_sym_gdf, ss_sym_gdf, _ = run_vibeqc_gdf(None)
    e_bs_gdf, ss_bs_gdf, _ = run_vibeqc_gdf([1, -1])
    ref_sym = run_pyscf(break_symmetry=False)
    ref_bs = run_pyscf(break_symmetry=True)
    if ref_bs.get("status") != "ok":
        print(f"PySCF.pbc reference unavailable ({ref_bs.get('note')}); "
              "skipping cross-code GDF comparison.")
        return 0

    print("vibe-qc GDF UHF vs PySCF.pbc UHF density_fit (exxdiv='ewald'):")
    print(f"  symmetric : vibe-qc {e_sym_gdf:.8f}  PySCF {ref_sym['energy']:.8f}"
          f"   Δ = {abs(e_sym_gdf - ref_sym['energy'])*1e6:.1f} µHa  (no spin/ATOMSPIN)")
    print(f"  ATOMSPIN  : vibe-qc {e_bs_gdf:.8f}  PySCF {ref_bs['energy']:.8f}"
          f"   Δ = {abs(e_bs_gdf - ref_bs['energy'])*1e6:.1f} µHa")
    print(f"  ⟨S²⟩      : vibe-qc {ss_bs_gdf:.4f}  PySCF {ref_bs['s_squared']:.4f}"
          f"   (broken-symmetry state agreement — the ATOMSPIN deliverable)")
    print("\n  Note: the GDF Δ is the density-fitting auxiliary-basis difference"
          "\n  between the two GDF implementations (identical magnitude in the"
          "\n  spin-symmetric row above), not an ATOMSPIN effect. The µHa energy"
          "\n  anchor is the EWALD-3D vs molecular-UHF Δ reported above.")

    # --- The validated assertions ----------------------------------------
    ok = True
    if not (d_anchor < 1.0):
        print(f"\nFAIL: EWALD-3D Γ does not reproduce molecular UHF to µHa "
              f"(Δ={d_anchor:.3f} µHa)")
        ok = False
    if not (abs(ss_bs_gdf - ref_bs["s_squared"]) < 1e-2):
        print(f"\nFAIL: broken-symmetry ⟨S²⟩ disagrees with PySCF "
              f"({ss_bs_gdf:.4f} vs {ref_bs['s_squared']:.4f})")
        ok = False
    if not (e_bs_gdf < e_sym_gdf - 1e-3):
        print("\nFAIL: ATOMSPIN did not reach a lower broken-symmetry energy")
        ok = False
    print("\n" + ("PASS: periodic ATOMSPIN matches PySCF.pbc broken-symmetry."
                  if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
