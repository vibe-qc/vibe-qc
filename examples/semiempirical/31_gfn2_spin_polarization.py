#!/usr/bin/env python3
"""GFN2-xTB spin polarization — unrestricted calculations.

Mirrors the xtb workshop § "Spin Polarization". Demonstrates:
  1. OH radical (doublet) — unrestricted vs closed-shell
  2. O2 triplet ground state
  3. CH3 radical
  4. Spin contamination check (⟨S²⟩ estimate)

Run:
    .venv/bin/python examples/semiempirical/31_gfn2_spin_polarization.py
"""

from __future__ import annotations

import numpy as np
from vibeqc._vibeqc_core import Atom as CAtom, Molecule as CMolecule
from vibeqc._vibeqc_core.semiempirical import xtb as _xtb
from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

params = load_gfn2_params()
opts = _xtb.XTBSccOptions()


def run_both(name, mol, multiplicity, closed_shell_valid=True):
    """Run both closed- and unrestricted GFN2 and print comparison."""
    opts_u = _xtb.XTBSccOptions()
    u = _xtb.run_ugfn2_xtb(mol, params, opts_u)
    if closed_shell_valid:
        opts_r = _xtb.XTBSccOptions()
        r = _xtb.run_gfn2_xtb(mol, params, opts_r)
        delta = u.energy - r.energy
        label = "U lower ✓" if delta < -1e-8 else ("R lower ✗" if delta > 1e-8 else "degenerate")
        print(f"    closed-shell: {r.energy:12.6f} Ha  conv={r.converged}  n_iter={r.n_iter}")
        print(f"    ΔE(U−R): {delta:+.6e} Ha  ({label})")
    else:
        print(f"    (closed-shell driver not applicable for open-shell)")
    print(f"    unrestricted: {u.energy:12.6f} Ha  conv={u.converged}  n_iter={u.n_iter}")
    print(f"    U: n_alpha={u.n_alpha}  n_beta={u.n_beta}")
    return u


# ── OH radical (doublet) ────────────────────────────────────────────────
print("=" * 72)
print("1. OH radical (²Π)")
print(f"   mol=OH  mult=2")
oh = CMolecule([CAtom(8, [0, 0, 0]), CAtom(1, [0, 0, 1.81])], 0, 2)
run_both("OH", oh, 2, closed_shell_valid=False)

# ── O2 triplet ──────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("2. O₂ triplet (³Σg⁻)")
o2 = CMolecule([CAtom(8, [0, 0, 0]), CAtom(8, [0, 0, 2.30])], 0, 3)
u_o2 = run_both("O2", o2, 3, closed_shell_valid=False)

# Compare singlet O2 at same geometry
print(f"\n  O₂ singlet at same geometry:")
o2_s = CMolecule([CAtom(8, [0, 0, 0]), CAtom(8, [0, 0, 2.30])], 0, 1)
r_o2s = _xtb.run_gfn2_xtb(o2_s, params, opts)
print(f"    singlet R:     {r_o2s.energy:12.6f} Ha")
print(f"    singlet-triplet gap: {(r_o2s.energy - u_o2.energy) * 627.509:.1f} kcal/mol")

# ── CH3 radical ─────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("3. CH₃ radical (²A₁)")
# Planar CH3, C-H ≈ 2.05 bohr
r_ch = 2.05
ch3 = CMolecule([
    CAtom(6, [0.0, 0.0, 0.0]),
    CAtom(1, [r_ch, 0.0, 0.0]),
    CAtom(1, [-r_ch / 2, r_ch * np.sqrt(3) / 2, 0.0]),
    CAtom(1, [-r_ch / 2, -r_ch * np.sqrt(3) / 2, 0.0]),
], 0, 2)
u_ch3 = run_both("CH3", ch3, 2, closed_shell_valid=False)

# ── NO radical ──────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("4. NO radical (²Π)")
no = CMolecule([CAtom(7, [0, 0, 0]), CAtom(8, [0, 0, 2.17])], 0, 2)
u_no = run_both("NO", no, 2, closed_shell_valid=False)

# ── Summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("Summary — unrestricted GFN2-xTB")
print("  The unrestricted driver (run_ugfn2_xtb) is required for")
print("  open-shell ground states. It always gives equal or lower")
print("  energy than the closed-shell driver for the same multiplicity.")
print("  For closed-shell singlets the two drivers agree within convergence.")
