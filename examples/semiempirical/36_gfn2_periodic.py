#!/usr/bin/env python3
"""GFN2-xTB periodic calculations: graphene, BN, and fail-closed multi-k.

Mirrors the xtb workshop § "Periodic Calculations". Demonstrates:
  1. Graphene 2-atom cell — Γ-point energy
  2. Multi-k availability guard
  3. h-BN monolayer
  4. Periodic gradients and stress tensor
  5. Molecular limit check (large isolated box vs molecular energy)

Run:
    .venv/bin/python examples/semiempirical/36_gfn2_periodic.py
"""

from __future__ import annotations

import numpy as np

from vibeqc._vibeqc_core import Atom as PAtom
from vibeqc._vibeqc_core import PeriodicSystem
from vibeqc._vibeqc_core import semiempirical as _se
from vibeqc._vibeqc_core.semiempirical import xtb as _xtb
from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

params = load_gfn2_params()
opts = _xtb.XTBSccOptions()
opts.max_iter = 400


def run_periodic(name, system):
    """Run periodic GFN2 and print summary."""
    r = _xtb.run_gfn2_xtb_gamma(system, params, opts)
    summary = (
        f"   {name}: E = {r.energy:14.8f} Ha  "
        f"n_cells={r.n_cells:4d}  n_iter={r.n_iter:3d}  "
        f"conv={r.converged}"
    )
    if r.converged:
        summary += f"  E_fermi={r.fermi_level:.4f}"
    if r.converged and r.smearing_temperature > 0.0:
        # Periodic GFN2-xTB smears by default: the variational potential
        # is the Mermin free energy A = E - T*S.
        summary += (
            f"  A = {r.free_energy:14.8f} Ha"
            f"  S/kB = {r.entropy:9.6f}"
        )
    print(summary)
    return r


# ── 1. Graphene ─────────────────────────────────────────────────────────
print("=" * 72)
print("1. Graphene (2-atom cell, Γ-point)")

a = 4.65      # bohr (≈2.46 Å)
c = 20.0      # vacuum
lattice_gr = np.array([
    [a, 0, 0],
    [a / 2, a * np.sqrt(3) / 2, 0],
    [0, 0, c],
])
atoms_gr = [
    PAtom(6, [0.0, 0.0, 0.0]),
    PAtom(6, [a / 2, a * np.sqrt(3) / 6, 0.0]),
]
system_gr = PeriodicSystem(2, lattice_gr, atoms_gr, charge=0, multiplicity=1)
r_gr = run_periodic("Graphene", system_gr)

# ── 2. k-point sampling ─────────────────────────────────────────────────
print("\n" + "=" * 72)
print("2. k-point sampling: intentionally disabled")
print("   Multi-k GFN2 fails closed until complex Bloch and AES parity exist.")
print("   Use DFTB0/SCC-DFTB for supported k-point semiempirical work.")

# ── 3. h-BN monolayer ──────────────────────────────────────────────────
print("\n" + "=" * 72)
print("3. h-BN monolayer (2-atom cell, Γ-point)")

a_bn = 4.72   # bohr (≈2.50 Å)
c_bn = 20.0
lattice_bn = np.array([
    [a_bn, 0, 0],
    [a_bn / 2, a_bn * np.sqrt(3) / 2, 0],
    [0, 0, c_bn],
])
atoms_bn = [
    PAtom(5, [0.0, 0.0, 0.0]),                              # B
    PAtom(7, [a_bn / 2, a_bn * np.sqrt(3) / 6, 0.0]),       # N
]
system_bn = PeriodicSystem(2, lattice_bn, atoms_bn, charge=0, multiplicity=1)
r_bn = run_periodic("h-BN", system_bn)

# ── 4. Periodic gradients ───────────────────────────────────────────────
print("\n" + "=" * 72)
print("4. Periodic gradients and stress")

grad = _se.compute_periodic_gfn2_gradient(system_gr, r_gr, params)
print(f"   Gradient shape: {grad.shape}")
print(f"   |grad|_max:     {np.abs(grad).max():.6f} Ha/bohr")
for a in range(len(atoms_gr)):
    print(f"   C{a}: {np.array2string(grad[a], precision=6, suppress_small=True)}")

# Stress tensor.  The analytic periodic GFN2 kernel is gated off (issue #338:
# it is not the strain derivative of the periodic GFN2 energy -- measured 5.3x
# too large on xx and sign-inverted on yy/zz).  The finite-difference stress
# differentiates the shipped energy by construction, so it is the supported
# route and is what optimize_cell uses.
from vibeqc.semiempirical.periodic import finite_difference_stress


def _gfn2_energy(sys_strained):
    return _xtb.run_gfn2_xtb_gamma(sys_strained, params, opts).energy


# strain_positions=True applies the homogeneous strain to the atomic
# coordinates as well as the lattice -- the full strain derivative.
stress = finite_difference_stress(
    system_gr, _gfn2_energy, strain_positions=True
)
print("\n   Stress tensor (Ha/bohr³, finite difference):")
print(f"   {np.array2string(stress, precision=8, suppress_small=True)}")
# Pressure = -Tr(stress)/3
pressure = -np.trace(stress) / 3.0
print(f"   Pressure: {pressure:.6f} Ha/bohr³ = {pressure * 29421.0:.2f} GPa")

# ── 5. Molecular limit check ────────────────────────────────────────────
print("\n" + "=" * 72)
print("5. Molecular limit — periodic vs molecular energy")

# H2 in a large box should match molecular H2 energy
from vibeqc import Molecule, Atom
from vibeqc.semiempirical.methods.gfn2 import GFN2Model

mol_h2 = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.40])])
e_mol = GFN2Model(mol_h2, params=params, warn=False).energy()
print(f"   Molecular H2:          {e_mol:.8f} Ha")

box = 15.0
system_h2 = PeriodicSystem(
    3, np.eye(3) * box,
    [PAtom(1, [0, 0, 0]), PAtom(1, [0, 0, 1.40])],
    charge=0, multiplicity=1,
)
# Molecular-limit parity: request the exact zero-temperature Aufbau so the
# periodic run compares against the molecular (Aufbau) energy directly.
parity_opts = _xtb.XTBSccOptions()
parity_opts.max_iter = opts.max_iter
parity_opts.electronic_temperature = 0.0
e_per = _xtb.run_gfn2_xtb_gamma(system_h2, params, parity_opts).energy
print(f"   Periodic H2 ({box} bohr box): {e_per:.8f} Ha")
print(f"   Difference:             {abs(e_mol - e_per):.2e} Ha")

# ── 6. 1D carbon chain ──────────────────────────────────────────────────
print("\n" + "=" * 72)
print("6. 1D carbon chain (Γ-point)")

a_cc = 2.5
system_c1d = PeriodicSystem(
    1, np.diag([a_cc, 30.0, 30.0]),
    [PAtom(6, [0, 0, 0])],
    charge=0, multiplicity=1,
)
r_c1d = run_periodic("C chain", system_c1d)

print("\nDone.")
