"""CASSCF example: H2O CAS(2,2) and CAS(4,4).

Demonstrates CASCI, CASSCF (single-state + state-averaged), and NEVPT2/CASPT2
on the CASSCF reference.  Run with:

    python examples/wavefunction/01_casscf_h2o.py
"""

from __future__ import annotations

from vibeqc import Atom, Molecule
from vibeqc.runner import run_job
import numpy as np

# Water molecule in Bohr (equilibrium geometry)
h2o = Molecule(
    [
        Atom(8, [0.0, 0.0, 0.0]),
        Atom(1, [0.0, 1.43, -0.93]),
        Atom(1, [0.0, -1.43, -0.93]),
    ]
)

print("=" * 60)
print("H2O / STO-3G — CASSCF + multireference PT2")
print("=" * 60)

# ── CASCI — single-point CI in the active space ──
print("\n1. CASCI(2,2) — CI in the HF active space")
casci_22 = run_job(
    h2o,
    basis="sto-3g",
    method="casci",
    active_space=(2, 2),
    write_xyz_file=False,
    write_molden_file=False,
    write_population_file=False,
    citations=False,
)
print(f"   E_CASCI(2,2) = {casci_22.energy:.8f} Ha")

# ── CASSCF — orbital-optimized ──
print("\n2. CASSCF(2,2) — orbital-optimized (auto: Super-CI then NR)")
casscf_22 = run_job(
    h2o,
    basis="sto-3g",
    method="casscf",
    active_space=(2, 2),
    write_xyz_file=False,
    write_molden_file=False,
    write_population_file=False,
    citations=False,
)
print(f"   E_CASSCF(2,2) = {casscf_22.energy:.8f} Ha")

# ── CASSCF improvements over CASCI ──
delta = casci_22.energy - casscf_22.energy
print(f"   Δ = CASCI - CASSCF = {delta:.6f} Ha = {delta * 627.5:.2f} kcal/mol")
print(f"   (CASSCF is {delta:.6f} Ha lower — orbital optimization recovers")
print(f"    static correlation the frozen HF orbitals miss)")

# ── NEVPT2 on CASSCF ──
print("\n3. NEVPT2(2,2) on CASSCF reference — dynamic correlation")
nevpt2_22 = run_job(
    h2o,
    basis="sto-3g",
    method="nevpt2",
    active_space=(2, 2),
    write_xyz_file=False,
    write_molden_file=False,
    write_population_file=False,
    citations=False,
)
print(f"   E_NEVPT2(2,2) = {nevpt2_22.energy:.8f} Ha")
print(f"   E_corr(PT2) = {nevpt2_22.energy - casscf_22.energy:.6f} Ha")

# ── CASCI(4,4) — larger active space ──
print("\n4. CASCI(4,4) — 4 electrons in 4 active orbitals")
casci_44 = run_job(
    h2o,
    basis="sto-3g",
    method="casci",
    active_space=(4, 4),
    write_xyz_file=False,
    write_molden_file=False,
    write_population_file=False,
    citations=False,
)
print(f"   E_CASCI(4,4) = {casci_44.energy:.8f} Ha")

# ── CASSCF(4,4) ──
print("\n5. CASSCF(4,4)")
casscf_44 = run_job(
    h2o,
    basis="sto-3g",
    method="casscf",
    active_space=(4, 4),
    write_xyz_file=False,
    write_molden_file=False,
    write_population_file=False,
    citations=False,
)
print(f"   E_CASSCF(4,4) = {casscf_44.energy:.8f} Ha")

# ── CASPT2(4,4) on CASSCF ──
print("\n6. CASPT2(4,4) on CASSCF reference (internally-contracted)")
caspt2_44 = run_job(
    h2o,
    basis="sto-3g",
    method="caspt2",
    active_space=(4, 4),
    write_xyz_file=False,
    write_molden_file=False,
    write_population_file=False,
    citations=False,
)
print(f"   E_CASPT2(4,4) = {caspt2_44.energy:.8f} Ha")

# CASSCF incomplete analytic nuclear-gradient preview.
print("\n8. CASSCF(4,4) analytic-gradient preview - not FD-tight")
grad = casscf_44.gradient
if grad is not None:
    print(f"   Gradient shape: {grad.shape}")
    print(f"   Gradient norm: {np.linalg.norm(grad):.6f} Ha/bohr")
    print(f"   Net force (translational invariance check):")
    net = np.sum(grad, axis=0)
    print(f"     {net[0]:.2e} {net[1]:.2e} {net[2]:.2e}")
    print(f"   Forces (F = -dE/dR, Ha/bohr):")
    for i, row in enumerate(-grad):
        print(f"     atom {i}: {row[0]:10.6f} {row[1]:10.6f} {row[2]:10.6f}")
    print("\n   The gradient captures ~87% of the full CP-MCSCF gradient.")
    print("   Missing 13% = W^z (CI+orbital relaxation, Handy-Schaefer z-vector).")
else:
    print("   (gradient not computed — SA-CASSCF or non-converged)")

print("\n" + "=" * 60)
print("For geometry optimization: result.gradient gives forces directly.")
print("See docs/user_guide/non_hf_solvers.md for the validation table.")
print("=" * 60)


print("\n" + "=" * 60)
print("Energy ladder: HF < CASCI < CASSCF < NEVPT2/CASPT2 < FCI")
print("CASSCF recovers static correlation; NEVPT2/CASPT2 add dynamic.")
print("=" * 60)

# ── State-averaged CASSCF ──
print("\n\n7. SA-CASSCF(2,2) — 2 roots, equal weights")
# State averaging is spin-pure by default: the two averaged roots are
# the two lowest SINGLETS (the M_s = 0 determinant sector interleaves a
# triplet between them, which a CSF-based code never averages; pass
# spin_pure=False for the raw-sector mixed-spin ensemble).
# This requires the low-level API since run_job doesn't expose nroots/weights yet.
from vibeqc import BasisSet
from vibeqc.solvers import build_hamiltonian_mo, casci, casscf, get_hf_orbital_provider

basis = BasisSet(h2o, "sto-3g")
C = get_hf_orbital_provider(h2o, basis)
H = build_hamiltonian_mo(h2o, basis, C)
n_elec_total = h2o.n_electrons()
n_core = (n_elec_total - 2) // 2  # 2 active electrons -> n_core = 4

sa = casscf(
    H.h1e,
    H.h2e,
    n_active_elec=2,
    n_active_orb=2,
    n_core=n_core,
    nuclear_repulsion=H.nuclear_repulsion,
    nroots=2,  # 2 states in the average
    verbose=1,
)
print(f"   E_SA = {sa.e_total:.8f} Ha")
print(f"   E_0 = {sa.e_totals[0]:.8f} Ha (ground state)")
print(f"   E_1 = {sa.e_totals[1]:.8f} Ha (excited state)")
print(
    f"   ΔE = {sa.e_totals[1] - sa.e_totals[0]:.6f} Ha = "
    f"{(sa.e_totals[1] - sa.e_totals[0]) * 627.5:.2f} kcal/mol"
)
print(f"   |grad| = {sa.grad_norm:.2e}")
