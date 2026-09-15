"""Vibrational frequency analysis for DFTB0 (analytic Hessian via FD of gradients).

Computes harmonic vibrational frequencies, zero-point energy, and
thermodynamic corrections (ZPVE, enthalpy, entropy at 298 K) for
molecules at a stationary point.

Only DFTB0 is supported (correct PES + analytic gradients).
GFN2-xTB and PM6 produce unphysical PES (see PES gate xfails).

Run:  .venv/bin/python examples/semiempirical/21_frequencies.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from vibeqc import Atom, Molecule
from vibeqc.semiempirical.dftb0 import DFTB0Model

HERE = Path(__file__).resolve().parent

# Physical constants
HA_TO_J = 4.359744722e-18
BOHR_TO_M = 5.291772109e-11
ME = 9.1093837e-31
C_CM = 2.99792458e10  # speed of light in cm/s
AMU_TO_ME = 1822.888486  # 1 amu in electron masses
KB = 3.1668114e-6  # Boltzmann constant in Ha/K
H_PLANCK = 2 * np.pi  # h-bar in atomic units

# Conversion factor: Hessian eigenvalue (Ha/me/bohr^2) -> cm^-1
HESS_TO_CM1 = math.sqrt(HA_TO_J / (BOHR_TO_M**2 * ME)) / (2 * np.pi * C_CM)


def optimize_geometry(
    atoms: list[Atom], charge: int = 0, mult: int = 1, max_steps: int = 100
) -> tuple[list[Atom], float]:
    """Simple steepest-descent optimization to a stationary point."""
    alpha = 0.05
    for step in range(max_steps):
        mol = Molecule(list(atoms), charge, mult)
        m = DFTB0Model(mol)
        e = m.energy()
        g = np.asarray(m.gradient())
        gmax = float(np.max(np.abs(g)))
        if gmax < 3e-4:
            break
        for a in range(len(atoms)):
            atoms[a] = Atom(atoms[a].Z, list(np.array(atoms[a].xyz) - alpha * g[a]))
    mol = Molecule(list(atoms), charge, mult)
    return atoms, DFTB0Model(mol).energy()


def compute_hessian_fd(
    atoms: list[Atom], charge: int = 0, mult: int = 1, h: float = 0.005
) -> np.ndarray:
    """Finite-difference Hessian from analytic gradients (central diff)."""
    n = len(atoms)
    H = np.zeros((3 * n, 3 * n))
    for a in range(n):
        for c in range(3):
            xyz_p = list(atoms[a].xyz)
            xyz_p[c] += h
            ap = [
                Atom(at.Z, xyz_p if i == a else list(at.xyz))
                for i, at in enumerate(atoms)
            ]
            g_p = np.asarray(
                DFTB0Model(Molecule(ap, charge, mult)).gradient()
            ).flatten()

            xyz_m = list(atoms[a].xyz)
            xyz_m[c] -= h
            am = [
                Atom(at.Z, xyz_m if i == a else list(at.xyz))
                for i, at in enumerate(atoms)
            ]
            g_m = np.asarray(
                DFTB0Model(Molecule(am, charge, mult)).gradient()
            ).flatten()

            H[:, a * 3 + c] = (g_p - g_m) / (2 * h)
    return 0.5 * (H + H.T)


def compute_frequencies(H: np.ndarray, atoms: list[Atom]) -> dict:
    """Mass-weight Hessian, diagonalize, return frequencies and modes."""
    n = len(atoms)
    masses = np.array([at.Z * 1.0 for at in atoms])  # approximate atomic masses in amu
    # Better: use actual atomic masses
    mass_amu = {1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998}
    m = np.array([mass_amu.get(at.Z, at.Z) for at in atoms])
    M_inv_sqrt = np.diag(1.0 / np.sqrt(np.repeat(m * AMU_TO_ME, 3)))
    H_mw = M_inv_sqrt @ H @ M_inv_sqrt

    evals, evecs = np.linalg.eigh(H_mw)
    # Convert to cm^-1
    omegas = np.sign(evals) * np.sqrt(np.abs(evals)) * HESS_TO_CM1
    return {"freqs": omegas, "modes": evecs, "hessian": H}


def thermodynamics(freqs: np.ndarray, temperature: float = 298.15) -> dict:
    """Compute ZPVE, enthalpy, entropy from harmonic frequencies.

    Only real frequencies (positive) contribute.
    """
    real_freqs = freqs[freqs > 1.0]  # discard translations/rotations and imaginary
    n_real = len(real_freqs)

    # ZPVE = ½ Σ hν
    zpve_ha = 0.5 * sum(real_freqs) / 219474.63  # cm^-1 -> Ha

    kT_ha = KB * temperature
    beta = 1.0 / kT_ha if kT_ha > 0 else 0.0

    # Vibrational partition function contributions
    e_vib = 0.0
    s_vib = 0.0
    for nu in real_freqs:
        nu_ha = nu / 219474.63
        x = beta * nu_ha
        if x > 50:
            continue
        e_vib += nu_ha / (math.exp(x) - 1.0)
        if x > 1e-10:
            s_vib += x / (math.exp(x) - 1.0) - math.log(1.0 - math.exp(-x))

    # Translational + rotational (rigid rotor, ideal gas)
    # Approximate: 3/2 RT translation + 3/2 RT rotation (nonlinear)
    e_trans_rot = 3.0 * kT_ha
    s_trans_rot = 3.0  # approximate

    return {
        "zpve_ha": zpve_ha,
        "zpve_kcal": zpve_ha * 627.509,
        "e_thermal_ha": e_vib + e_trans_rot,
        "s_vib": s_vib * KB,  # in Ha/K
        "n_real": n_real,
        "n_imag": int(np.sum(freqs < -1.0)),
    }


def main():
    # H2O
    print("=" * 60)
    print(" DFTB0 Vibrational Frequencies — H2O")
    print("=" * 60)

    theta = np.deg2rad(52.25)
    r0 = 1.81
    atoms = [
        Atom(8, [0, 0, 0]),
        Atom(1, [0, r0 * np.sin(theta), r0 * np.cos(theta)]),
        Atom(1, [0, -r0 * np.sin(theta), r0 * np.cos(theta)]),
    ]

    print("Optimizing geometry...")
    atoms_opt, e_opt = optimize_geometry(atoms)
    mol = Molecule(atoms_opt, 0, 1)
    g = np.asarray(DFTB0Model(mol).gradient())
    r1 = np.linalg.norm(np.array(atoms_opt[1].xyz) - np.array(atoms_opt[0].xyz))
    v1 = np.array(atoms_opt[1].xyz) - np.array(atoms_opt[0].xyz)
    v2 = np.array(atoms_opt[2].xyz) - np.array(atoms_opt[0].xyz)
    angle = np.degrees(np.arccos(np.dot(v1, v2) / (r1 * r1)))

    print(f"  E_opt = {e_opt:.6f} Ha")
    print(f"  max|g| = {np.max(np.abs(g)):.6f} Ha/bohr")
    print(f"  O-H = {r1:.4f} bohr = {r1 * 0.529:.3f} Å")
    print(f"  H-O-H = {angle:.1f}°")
    print()

    print("Computing Hessian...")
    H = compute_hessian_fd(atoms_opt)
    result = compute_frequencies(H, atoms_opt)
    freqs = result["freqs"]

    print("Frequencies (cm^-1):")
    for i, nu in enumerate(sorted(freqs)):
        tag = "i" if nu < -1 else " "
        print(f"  mode {i + 1:2d}: {tag} {abs(nu):8.1f}")

    thermo = thermodynamics(freqs)
    print()
    print("Thermodynamics at 298.15 K:")
    print(f"  ZPVE:      {thermo['zpve_kcal']:.3f} kcal/mol")
    print(f"  E_thermal: {thermo['e_thermal_ha'] * 627.509:.3f} kcal/mol")
    print(f"  Imaginary: {thermo['n_imag']} (warnings if > 0)")
    print(f"  Real freq: {thermo['n_real']}")

    print()
    print("Reference H2O frequencies (DFTB/mio-1-1): ~1600, ~3700, ~3800 cm^-1")
    print("(Exact values depend on the repulsive potential parameterization)")


if __name__ == "__main__":
    main()
