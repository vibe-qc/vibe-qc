#!/usr/bin/env python3
"""Periodic TDDFT: H2 molecule in a box -- GPW SCF + TDA excited states.

Demonstrates the end-to-end workflow:
1. Build a periodic system (H2 in a vacuum-padded cell).
2. Run a GPW-RHF ground-state SCF.
3. Compute vertical excitation energies via TDA/CIS.
4. Report excitation energies, oscillator strengths, and
   dominant orbital transitions.

This is a Gamma-point calculation; the vacuum padding makes
periodic-image interactions negligible, so the TDA spectrum
matches the molecular CIS result.

Expected output (H2 / STO-3G, R=1.4 bohr):
  RHF energy:      -1.116714 Ha
  S1 (sigma->sigma*): ~0.947 Ha (~25.8 eV)
"""

import warnings

import numpy as np
import vibeqc as vq
from vibeqc import _vibeqc_core as core

# ---------------------------------------------------------------------------
# 1. Build the system — H2 in a 16-bohr cubic cell
# ---------------------------------------------------------------------------
L = 16.0
d = 1.4  # H2 bond length in bohr
half = d / 2.0

system = core.PeriodicSystem()
system.dim = 3
system.lattice = np.eye(3) * L
system.unit_cell = [
    core.Atom(1, [L / 2, L / 2, L / 2 - half]),
    core.Atom(1, [L / 2, L / 2, L / 2 + half]),
]

mol = vq.Molecule(list(system.unit_cell), 0, 1)
basis = vq.BasisSet(mol, "sto-3g")

print(f"Cell: {L:.0f} bohr cube, {basis.nbasis} basis functions")
print(f"Atoms: {len(system.unit_cell)}, electrons: {mol.n_electrons()}")

# ---------------------------------------------------------------------------
# 2. Ground-state GPW-RHF SCF
# ---------------------------------------------------------------------------
warnings.simplefilter("ignore", vq.GAPWExperimentalWarning)

rhf_result = vq.run_periodic_rhf_gpw(
    system,
    basis,
    cutoff_ha=300.0,
    max_iter=50,
    conv_tol_energy=1e-9,
    quiet=True,
)

print(f"\nRHF converged: {rhf_result.converged} in {rhf_result.n_iter} iterations")
print(f"RHF total energy: {rhf_result.energy:.10f} Ha")

n_occ = mol.n_electrons() // 2
print(f"Occupied orbitals: {n_occ}, virtual: {basis.nbasis - n_occ}")

# ---------------------------------------------------------------------------
# 3. TDDFT — TDA excited states
# ---------------------------------------------------------------------------
n_states = 3

tddft = vq.run_tddft_tda_periodic(
    rhf_result,
    basis,
    n_occ=n_occ,
    n_states=n_states,
)

print(
    f"\n{'State':>6s}  {'E_exc (Ha)':>12s}  {'E_exc (eV)':>12s}  "
    f"{'lambda (nm)':>12s}  {'f_osc':>8s}  Dominant transition"
)
print("-" * 80)

for state in tddft.states:
    dom = state.dominant_amplitudes
    if dom:
        occ, virt, amp = dom[0]
        # 'virt' is 1-based within the virtual block; convert to
        # overall MO index: virt_mo = n_occ + virt.
        virt_mo = n_occ + virt
        lumo_idx = virt  # 1 = LUMO, 2 = LUMO+1, ...
        if lumo_idx == 1:
            desc = f"HOMO -> LUMO (MO {virt_mo})"
        else:
            desc = f"HOMO -> LUMO+{lumo_idx - 1} (MO {virt_mo})"
    else:
        desc = "—"
    print(
        f"S{state.index:>5d}  {state.excitation_energy:>12.6f}  "
        f"{state.excitation_energy_ev:>12.4f}  "
        f"{state.wavelength_nm:>12.1f}  {state.oscillator_strength:>8.4f}  "
        f"{desc}"
    )
