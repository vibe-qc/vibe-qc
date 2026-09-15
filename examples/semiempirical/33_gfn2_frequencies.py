#!/usr/bin/env python3
"""GFN2-xTB vibrational frequencies via finite-difference Hessian.

Mirrors the xtb workshop § "Frequencies". Demonstrates:
  1. Hessian construction by central-differencing analytic gradients
  2. Harmonic frequency calculation (cm⁻¹)
  3. Zero-point vibrational energy (ZPVE)
  4. Thermochemical corrections (enthalpy, entropy at 298 K)

Run:
    .venv/bin/python examples/semiempirical/33_gfn2_frequencies.py
"""

from __future__ import annotations

import math
import numpy as np
from vibeqc import Molecule, Atom
from vibeqc.semiempirical.methods.gfn2 import GFN2Model
from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

params = load_gfn2_params()

# Physical constants
HA_TO_J = 4.359744722e-18
BOHR_TO_M = 5.291772109e-11
ME = 9.1093837e-31
C_CM = 2.99792458e10           # speed of light in cm/s
KB = 3.1668114e-6              # kB in Ha/K
H_PLANCK = 2 * np.pi           # ħ in a.u.

# Hessian eigenvalue (Ha/me/bohr²) → cm⁻¹
_AMU_TO_ME = 1822.888486
_HESS_TO_CM1 = math.sqrt(HA_TO_J / (BOHR_TO_M**2 * ME)) / (2 * np.pi * C_CM)


def hessian_fd(model, h=0.005):
    """Central-difference Hessian from GFN2Model analytic gradients."""
    mol = model.molecule
    n = len(mol.atoms)
    H = np.zeros((3 * n, 3 * n))
    for a in range(n):
        for c in range(3):
            xyz_p = list(mol.atoms[a].xyz)
            xyz_p[c] += h
            mol_p = Molecule(
                [Atom(at.Z, xyz_p if i == a else list(at.xyz))
                 for i, at in enumerate(mol.atoms)],
                mol.charge, mol.multiplicity,
            )
            g_p = np.asarray(
                GFN2Model(mol_p, params=model.params, warn=False).gradient()
            ).flatten()

            xyz_m = list(mol.atoms[a].xyz)
            xyz_m[c] -= h
            mol_m = Molecule(
                [Atom(at.Z, xyz_m if i == a else list(at.xyz))
                 for i, at in enumerate(mol.atoms)],
                mol.charge, mol.multiplicity,
            )
            g_m = np.asarray(
                GFN2Model(mol_m, params=model.params, warn=False).gradient()
            ).flatten()

            H[3 * a + c, :] = (g_p - g_m) / (2.0 * h)
    return (H + H.T) / 2.0


def compute_frequencies(mol, params):
    """Compute harmonic frequencies (cm⁻¹) and ZPVE (kcal/mol)."""
    model = GFN2Model(mol, params=params, warn=False)

    # Check gradient magnitude (should be near zero at stationary point)
    g = np.asarray(model.gradient())
    g_rms = np.sqrt(np.mean(g**2))
    print(f"   Gradient RMS: {g_rms:.6f} Ha/bohr")

    # Build and mass-weight Hessian
    H = hessian_fd(model)
    atom_masses_amu = np.array([
        {1: 1.007825, 6: 12.0, 7: 14.003074, 8: 15.994915, 9: 18.998403}.get(at.Z, at.Z * 2.0)
        for at in mol.atoms
    ])
    masses_me = atom_masses_amu * _AMU_TO_ME
    M_inv_sqrt = np.diag(1.0 / np.sqrt(np.repeat(masses_me, 3)))
    H_mw = M_inv_sqrt @ H @ M_inv_sqrt

    # Diagonalise
    eigvals = np.linalg.eigvalsh(H_mw)

    # Keep positive eigenvalues, compute frequencies
    freqs_cm1 = np.sqrt(np.maximum(eigvals, 0.0)) * _HESS_TO_CM1

    # Sort ascending
    order = np.argsort(freqs_cm1)
    freqs_cm1 = freqs_cm1[order]

    # ZPVE: ½ Σ ħω
    zpve_ha = 0.5 * np.sum(np.maximum(eigvals, 0.0) ** 0.5) / _HESS_TO_CM1 * (
        _HESS_TO_CM1 / (C_CM * 2 * np.pi)
    )
    # Simpler: ZPVE = ½ Σ νᵢ (in cm⁻¹) · hc
    zpve_cm1 = 0.5 * np.sum(freqs_cm1)
    zpve_kcal = zpve_cm1 * 2.859144e-3  # cm⁻¹ → kcal/mol

    return freqs_cm1, zpve_kcal


def thermo_corrections(freqs_cm1, temperature=298.15):
    """Compute enthalpy and entropy corrections at given T."""
    T = temperature
    beta = 1.0 / (KB * T)  # 1/Ha

    ZPVE_cm1 = 0.5 * np.sum(freqs_cm1)
    ZPVE_kcal = ZPVE_cm1 * 2.859144e-3

    # Filter real frequencies only
    real_freqs = freqs_cm1[freqs_cm1 > 1.0]
    if len(real_freqs) == 0:
        return ZPVE_kcal, 0.0, 0.0

    # Vibrational partition function per mode
    # ν (cm⁻¹) → Ha: ν_Ha = ν_cm1 * c_cm * h / Ha_to_J
    nu_ha = real_freqs * C_CM * 2.0 * np.pi / (HA_TO_J / H_PLANCK)
    # Actually: ν_cm⁻¹ * c(cm/s) * h(J·s) / (Ha→J)
    # E = hν = h * c * ν̃
    # hc in Ha·cm: (h in J·s) * (c in cm/s) / (Ha in J)
    HC_HA_CM = 4.556335e-6  # hc in Ha·cm
    nu_ha = real_freqs * HC_HA_CM

    # Harmonic oscillator partition function contributions
    x = beta * nu_ha
    exp_x = np.exp(x)
    # U_vib = Σ ħωᵢ / (exp(βħωᵢ) - 1)
    U_vib_ha = np.sum(nu_ha / (exp_x - 1.0))
    U_vib_kcal = U_vib_ha * 627.509

    # S_vib/kB = Σ [βħωᵢ/(exp(βħωᵢ)-1) - ln(1 - exp(-βħωᵢ))]
    S_vib_kb = np.sum(x / (exp_x - 1.0) - np.log(1.0 - np.exp(-x)))

    # Translational + rotational (rigid rotor / ideal gas)
    # For 3N-6 modes, we already have the vibrational part.
    # Add standard trans/rot contributions:
    #  H_trans = (3/2)RT, H_rot = (3/2)RT (nonlinear)
    #  S_trans/R = ... Stirling, S_rot/R = ...
    # Simplified: use 3RT translational + 1.5RT rotational for nonlinear
    n_atoms = len(freqs_cm1) // 3
    RT = KB * T * 627.509  # kcal/mol
    if n_atoms == 1:
        H_trans_rot = 1.5 * RT
    elif n_atoms == 2:
        H_trans_rot = 2.5 * RT  # linear
    else:
        H_trans_rot = 4.0 * RT  # nonlinear: (3/2+3/2+1)RT = 4RT

    enthalpy_kcal = ZPVE_kcal + U_vib_kcal + H_trans_rot
    # Entropy (simplified ideal gas)
    # S_trans/R ≈ ln[(2πmkT/h²)^(3/2) V] + 5/2
    # We'll skip the full Sackur-Tetrode here and note the simplification.
    entropy_cal_mol_K = S_vib_kb * 1.987  # R in cal/(mol·K)

    return ZPVE_kcal, enthalpy_kcal, entropy_cal_mol_K


# ── Water ───────────────────────────────────────────────────────────────
print("=" * 72)
print("1. Water (H2O) — vibrational frequencies")
mol_h2o = Molecule([
    Atom(8, [ 0.00,  0.00,  0.00]),
    Atom(1, [ 1.43,  0.98,  0.00]),
    Atom(1, [-1.43,  0.98,  0.00]),
])

freqs, zpve = compute_frequencies(mol_h2o, params)
print(f"\n   Harmonic frequencies (cm⁻¹):")
zmode_labels = {}
for i, f in enumerate(freqs):
    label = ""
    if f < 1.0:
        label = "(trans/rot)"
    elif len(freqs) - i <= 3:
        pass
    print(f"      mode {i+1:2d}: {f:8.1f}  {label}")

real_freqs = freqs[freqs > 1.0]
if len(real_freqs) >= 3:
    print(f"\n   Bend:                {real_freqs[0]:8.1f} cm⁻¹")
    print(f"   Symmetric stretch:   {real_freqs[1]:8.1f} cm⁻¹")
    print(f"   Asymmetric stretch:  {real_freqs[2]:8.1f} cm⁻¹")
print(f"   ZPVE: {zpve:.2f} kcal/mol")

zpve_kcal, H_kcal, S_cal = thermo_corrections(freqs)
print(f"\n   Thermodynamics at 298.15 K:")
print(f"      ZPVE:               {zpve_kcal:8.2f} kcal/mol")
print(f"      Enthalpy correction: {H_kcal:8.2f} kcal/mol")
print(f"      Entropy (vib only):  {S_cal:8.2f} cal/(mol·K)")

# ── H₂ ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("2. Hydrogen (H₂) — diatomic frequency")
mol_h2 = Molecule([
    Atom(1, [0.0, 0.0, 0.0]),
    Atom(1, [0.0, 0.0, 1.40]),
])
freqs_h2, zpve_h2 = compute_frequencies(mol_h2, params)
real_h2 = freqs_h2[freqs_h2 > 1.0]
print(f"   Stretch frequency: {real_h2[0]:8.1f} cm⁻¹")
print(f"   ZPVE: {zpve_h2:.2f} kcal/mol")

# ── CH₄ ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("3. Methane (CH4) — polyatomic")
d = 2.05 / np.sqrt(3)
mol_ch4 = Molecule([
    Atom(6, [ 0.00,  0.00,  0.00]),
    Atom(1, [    d,     d,     d]),
    Atom(1, [   -d,    -d,     d]),
    Atom(1, [   -d,     d,    -d]),
    Atom(1, [    d,    -d,    -d]),
])
freqs_ch4, zpve_ch4 = compute_frequencies(mol_ch4, params)
real_ch4 = freqs_ch4[freqs_ch4 > 1.0]
print(f"   Nonzero modes: {len(real_ch4)}")
print(f"   Lowest freq:   {real_ch4[0]:8.1f} cm⁻¹")
print(f"   Highest freq:  {real_ch4[-1]:8.1f} cm⁻¹")
print(f"   ZPVE: {zpve_ch4:.2f} kcal/mol")

print("\nNote: These are harmonic frequencies at approximate geometries.")
print("For quantitative results, optimise the geometry first (example 32).")
