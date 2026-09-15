"""CASSCF analytic-gradient component audit vs ORCA and OpenMolcas.

Out-of-process comparison (CLAUDE.md §10 — never import external QC programs).
Characterizes the incomplete z-vector-free gradient against two independent
references. It does not validate the total derivative; use
``casscf_gradient_fd_reproducer.py`` as the correctness gate.

Usage:
    python examples/regression/parity_casscf_gradient.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from vibeqc import Atom, BasisSet, Molecule
from vibeqc.gradient import compute_casscf_gradient
from vibeqc.solvers import build_hamiltonian_mo, get_hf_orbital_provider
from vibeqc.solvers._casscf import casscf
from vibeqc.solvers._rdm import make_rdm12

# --- Config -----------------------------------------------------------
ORCA_CMD = os.path.expanduser("~/bin/orca/orca")
MOLCAS_WORKDIR = os.path.expanduser("~/gitlab/OpenMolcas")
MOLCAS_CMD = os.path.join(MOLCAS_WORKDIR, "build", "pymolcas")


def _find_orca() -> str | None:
    if os.path.isfile(ORCA_CMD) and os.access(ORCA_CMD, os.X_OK):
        return ORCA_CMD
    return None


def _find_openmolcas() -> str | None:
    if os.path.isfile(MOLCAS_CMD) and os.access(MOLCAS_CMD, os.X_OK):
        env = os.environ.copy()
        env["MOLCAS"] = os.path.join(MOLCAS_WORKDIR, "build")
        # Test that pymolcas runs
        try:
            r = subprocess.run(
                [MOLCAS_CMD, "--version"],
                capture_output=True,
                text=True,
                env=env,
                timeout=10,
            )
            if r.returncode == 0 or "version" in (r.stdout + r.stderr).lower():
                return MOLCAS_CMD
        except Exception:
            pass
        # Try anyway
        return MOLCAS_CMD
    return None


# --- System definitions ----------------------------------------------


def h2o_sto3g():
    """H2O/STO-3G CAS(4,4) n_core=1 — standard test case."""
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.117]),
            Atom(1, [0.0, 0.757, -0.469]),
            Atom(1, [0.0, -0.757, -0.469]),
        ]
    )
    return (
        mol,
        "sto-3g",
        1,
        4,
        4,
        1,
    )  # n_core, n_active_orb, n_active_elec, multiplicity


# --- vibe-qc gradient ------------------------------------------------


def compute_vibeqc_grad(mol, basis_name, n_core, n_act_orb, n_act_elec):
    """Compute vibe-qc's incomplete analytic-gradient preview."""
    basis = BasisSet(mol, basis_name)
    C_hf = get_hf_orbital_provider(mol, basis)
    H = build_hamiltonian_mo(mol, basis, C_hf)
    sc = casscf(
        H.h1e,
        H.h2e,
        n_act_elec,
        n_act_orb,
        n_core=n_core,
        nuclear_repulsion=H.nuclear_repulsion,
        max_macro=50,
    )
    assert sc.converged, f"vibe-qc CASSCF did not converge (|g|={sc.grad_norm:.2e})"
    C_conv = C_hf @ sc.mo_rotation
    rdm1, rdm2 = make_rdm12(sc.cas.ci_coeffs, sc.cas.determinants, n_act_orb)
    grad = compute_casscf_gradient(
        mol,
        basis,
        C_conv,
        sc.h1e_cas,
        sc.h2e_cas,
        n_core=n_core,
        n_active_orb=n_act_orb,
        rdm1=rdm1,
        rdm2=rdm2,
    )
    return sc.e_total, grad


# --- ORCA ------------------------------------------------------------


def _orca_xyz_block(mol):
    """Return an ORCA xyz block (Angstrom)."""
    BOHR_TO_ANG = 0.529177210903
    lines = ["* xyz 0 1"]
    for atom in mol.atoms:
        x, y, z = atom.xyz
        lines.append(
            f"  {atom.symbol}  {x * BOHR_TO_ANG:.10f}  {y * BOHR_TO_ANG:.10f}  {z * BOHR_TO_ANG:.10f}"
        )
    lines.append("*")
    return "\n".join(lines)


def compute_orca_grad(mol, basis_name, n_core, n_act_orb, n_act_elec, orca_cmd):
    """Run ORCA CASSCF + EnGrad and parse the gradient."""
    with tempfile.TemporaryDirectory(prefix="vq_orca_casgrad_") as tmpdir:
        tmpdir = Path(tmpdir)
        inp = tmpdir / "job.inp"
        xyz_block = _orca_xyz_block(mol)
        inp.write_text(f"""! {basis_name} TightSCF EnGrad NoKeepMolden
%casscf
  nel {n_act_elec}
  norb {n_act_orb}
  mult {mol.multiplicity}
  nroots 1
end
{xyz_block}
""")
        env = os.environ.copy()
        env["PATH"] = str(Path(orca_cmd).parent) + ":" + env.get("PATH", "")
        try:
            result = subprocess.run(
                [orca_cmd, str(inp)],
                cwd=str(tmpdir),
                capture_output=True,
                text=True,
                timeout=300,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return None, None, "ORCA timed out"
        out = result.stdout + result.stderr
        if result.returncode != 0:
            return None, None, f"ORCA exit {result.returncode}"

        # Parse energy
        e_match = re.search(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", out)
        energy = float(e_match.group(1)) if e_match else None

        # Parse gradient
        grad_match = re.search(
            r"CARTESIAN GRADIENT\s*\n\s*\n(.*?)(?:\n\s*\n|\n\*\*\*)", out, re.DOTALL
        )
        if not grad_match:
            # Try alternative: the gradient block after "The final MP2 gradient"
            # For CASSCF, look for gradient after energy
            lines = out.splitlines()
            grad_lines = []
            in_grad = False
            for line in lines:
                if "CARTESIAN GRADIENT" in line:
                    in_grad = True
                    continue
                if in_grad:
                    if line.strip() == "" or "***" in line:
                        break
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            grad_lines.append([float(x) for x in parts[1:4]])
                        except ValueError:
                            continue
            if grad_lines:
                grad = np.array(grad_lines, dtype=float)
            else:
                grad = None
        else:
            block = grad_match.group(1)
            nums = []
            for line in block.strip().splitlines():
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        nums.extend([float(x) for x in parts[1:4]])
                    except ValueError:
                        pass
            if nums:
                grad = np.array(nums, dtype=float).reshape(-1, 3)
            else:
                grad = None

        return energy, grad, None


# --- OpenMolcas ------------------------------------------------------


def _molcas_xyz(mol):
    """Return OpenMolcas xyz lines (Bohr)."""
    lines = [f"{len(mol.atoms)}\nBohr"]
    for atom in mol.atoms:
        x, y, z = atom.xyz
        lines.append(f"{atom.symbol}  {x:.10f}  {y:.10f}  {z:.10f}")
    return "\n".join(lines)


def compute_openmolcas_grad(mol, basis_name, n_core, n_act_orb, n_act_elec, molcas_cmd):
    """Run OpenMolcas CASSCF + ALASKA and parse energy + gradient."""
    with tempfile.TemporaryDirectory(prefix="vq_om_casgrad_") as tmpdir:
        tmpdir = Path(tmpdir)
        xyz_file = tmpdir / "job.xyz"
        xyz_file.write_text(_molcas_xyz(mol))

        # RASSCF for H2O/STO-3G CAS(4,4): 1 inactive, 4 RAS2, 2 active electrons per spin
        # Since n_act_elec=4 and mult=1, n_elec_per_spin = 2
        inp = tmpdir / "job.input"
        inp.write_text(f"""&GATEWAY
  Coord = job.xyz
  Basis = {basis_name.upper()}
  Group = C1
&SEWARD
&SCF
&RASSCF
  Spin = {mol.multiplicity}
  nActEl = {n_act_elec} 0 0
  Inactive = {n_core}
  RAS2 = {n_act_orb}
  CIRoot = 1 1 1
&ALASKA
""")

        env = os.environ.copy()
        env["MOLCAS"] = os.path.join(MOLCAS_WORKDIR, "build")
        env["MOLCAS_PRINT"] = "3"
        try:
            result = subprocess.run(
                [molcas_cmd, "-f", str(inp)],
                cwd=str(tmpdir),
                capture_output=True,
                text=True,
                timeout=600,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return None, None, "OpenMolcas timed out"

        out = result.stdout + result.stderr

        # Parse RASSCF energy (root 1)
        e_match = re.search(
            r"RASSCF root number\s+1\s+Total energy:\s+(-?\d+\.\d+)", out
        )
        energy = float(e_match.group(1)) if e_match else None

        # Parse ALASKA gradient.  Look for "Molecular gradients" block.
        # Format varies; try the standard ALASKA output.
        grad = None
        lines = out.splitlines()
        for i, line in enumerate(lines):
            if "Molecular gradients" in line or "GRADIENT IN HARTREE/BOHR" in line:
                grad_vals = []
                for j in range(i + 1, min(i + len(mol.atoms) + 5, len(lines))):
                    parts = lines[j].split()
                    if len(parts) >= 4:
                        try:
                            grad_vals.append([float(x) for x in parts[1:4]])
                        except ValueError:
                            if grad_vals and len(grad_vals) >= len(mol.atoms):
                                break
                            continue
                if len(grad_vals) >= len(mol.atoms):
                    grad = np.array(grad_vals[: len(mol.atoms)], dtype=float)
                break

        return energy, grad, None


# --- Main ------------------------------------------------------------


def main():
    print("=" * 70)
    print("  CASSCF Analytic Gradient Parity: vibe-qc vs ORCA vs OpenMolcas")
    print("=" * 70)

    orca_cmd = _find_orca()
    molcas_cmd = _find_openmolcas()

    print(f"\nORCA:       {'found' if orca_cmd else 'NOT FOUND'}")
    print(f"OpenMolcas: {'found' if molcas_cmd else 'NOT FOUND'}")

    mol, basis_name, n_core, n_act_orb, n_act_elec, mult = h2o_sto3g()
    print(f"\nSystem: H2O/STO-3G CAS({n_act_elec},{n_act_orb}) n_core={n_core}")

    # 1. vibe-qc
    print("\n--- vibe-qc ---")
    e_vq, g_vq = compute_vibeqc_grad(mol, basis_name, n_core, n_act_orb, n_act_elec)
    print(f"  E_total = {e_vq:.10f} Ha")
    print(f"  Gradient (Ha/bohr):")
    for i, row in enumerate(g_vq):
        print(f"    atom {i}: {row[0]:12.8f} {row[1]:12.8f} {row[2]:12.8f}")
    g_vq_norm = np.linalg.norm(g_vq)

    # 2. ORCA
    if orca_cmd:
        print("\n--- ORCA ---")
        e_o, g_o, err_o = compute_orca_grad(
            mol, basis_name, n_core, n_act_orb, n_act_elec, orca_cmd
        )
        if err_o:
            print(f"  ERROR: {err_o}")
        else:
            print(f"  E_total = {e_o:.10f} Ha")
            if g_o is not None:
                for i, row in enumerate(g_o):
                    print(f"    atom {i}: {row[0]:12.8f} {row[1]:12.8f} {row[2]:12.8f}")
                g_o_norm = np.linalg.norm(g_o)
                delta = np.max(np.abs(g_vq - g_o))
                print(f"  max |vibe-qc - ORCA| = {delta:.2e} Ha/bohr")
    else:
        print("\n--- ORCA: skipped (not found) ---")

    # 3. OpenMolcas
    if molcas_cmd:
        print("\n--- OpenMolcas ---")
        e_om, g_om, err_om = compute_openmolcas_grad(
            mol, basis_name, n_core, n_act_orb, n_act_elec, molcas_cmd
        )
        if err_om:
            print(f"  ERROR: {err_om}")
        else:
            print(f"  E_total = {e_om:.10f} Ha")
            if g_om is not None:
                for i, row in enumerate(g_om):
                    print(f"    atom {i}: {row[0]:12.8f} {row[1]:12.8f} {row[2]:12.8f}")
                g_om_norm = np.linalg.norm(g_om)
                delta = np.max(np.abs(g_vq - g_om))
                print(f"  max |vibe-qc - OpenMolcas| = {delta:.2e} Ha/bohr")
    else:
        print("\n--- OpenMolcas: skipped (not found) ---")

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
