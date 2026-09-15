"""X23 molecular-crystal lattice energies via BIPOLE.

Computes lattice energies E_latt = E_crystal / Z - E_molecule for the
X23 benchmark set (Reilly & Tkatchenko 2013, Dolgonos-Hoja-Boese 2019
revision) using the supported exact Ewald-J BIPOLE periodic SCF route.
The fail-closed quartet multipole prototype is not used.

Usage:
    .venv/bin/python examples/regression/x23_lattice_energy.py urea
    .venv/bin/python examples/regression/x23_lattice_energy.py --all

References
----------
- Reilly & Tkatchenko, J. Chem. Phys. 139, 024705 (2013).
- Dolgonos, Hoja & Boese, Phys. Chem. Chem. Phys. 21, 24333 (2019).
- Pisani, Dovesi & Roetti, Hartree-Fock Ab Initio Treatment of
  Crystalline Systems, Lecture Notes in Chemistry 48 (1988),
  doi:10.1007/978-3-642-93385-1.
- Saunders et al., Mol. Phys. 77, 629 (1992),
  doi:10.1080/00268979200102671, for Gaussian-density electrostatics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# X23 reference data (revised, Dolgonos-Hoja-Boese 2019, Table 2)
# ---------------------------------------------------------------------------
X23_REFERENCE = {
    "urea":        {"lattice_kjmol": 102.5, "Z": 2, "formula": "CH4N2O"},
    "benzene":     {"lattice_kjmol":  53.7, "Z": 4, "formula": "C6H6"},
    "ice_ih":      {"lattice_kjmol":  63.7, "Z": 4, "formula": "H2O"},
}

KJMOL_TO_HA = 1.0 / 2625.499638  # 1 kJ/mol in Hartree


def run_bipole_scf(system, basis, kmesh, method="RHF", functional=None,
                    cutoff=12.0, ewald_precision=1e-6, progress=False):
    """Run a BIPOLE SCF and return the total energy."""
    import vibeqc as vq
    from vibeqc._vibeqc_core import (
        PeriodicRHFOptions, PeriodicKSOptions, monkhorst_pack,
    )

    if kmesh is None:
        kmesh = monkhorst_pack(system, [1, 1, 1])

    if functional:
        opts = PeriodicKSOptions()
        opts.functional = functional
    else:
        opts = PeriodicRHFOptions()

    opts.lattice_opts.cutoff_bohr = cutoff
    opts.lattice_opts.nuclear_cutoff_bohr = cutoff * 2.0
    opts.max_iter = 60
    opts.use_diis = True
    opts.conv_tol_energy = 1e-7
    opts.initial_guess = vq.InitialGuess.SAD
    opts.damping = 0.7

    if functional:
        from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks
        result = run_pbc_bipole_rks(
            system, basis, kmesh, opts,
            use_ewald_j_split=True,
            ewald_precision=ewald_precision,
            progress=progress,
        )
    else:
        from vibeqc.pbc_bipole import run_pbc_bipole_rhf
        result = run_pbc_bipole_rhf(
            system, basis, kmesh, opts,
            use_ewald_j_split=True,
            ewald_precision=ewald_precision,
            progress=progress,
        )
    return result


def compute_lattice_energy(name: str, method="RHF", functional=None,
                            cutoff=12.0, verbose=True):
    """Compute X23 lattice energy for one member.

    Returns dict with keys: e_crystal, e_molecule, e_latt_ha, e_latt_kjmol,
    ref_kjmol, delta_kjmol.
    """
    import vibeqc as vq
    from vibeqc._vibeqc_core import monkhorst_pack

    ref = X23_REFERENCE.get(name)
    if ref is None:
        raise ValueError(f"Unknown X23 member: {name}.  Known: {list(X23_REFERENCE)}")

    # --- Load geometry ---
    spec_path = Path(__file__).resolve().parent / "systems" / "periodic" / f"x23_{name}.py"
    if not spec_path.exists():
        raise FileNotFoundError(f"X23 geometry not found: {spec_path}")

    # Import the spec module.
    import importlib.util
    spec_mod = importlib.util.spec_from_file_location(f"x23_{name}", spec_path)
    mod = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(mod)
    spec = mod.SPEC

    # --- Build periodic system ---
    lattice_ang = np.array(spec.lattice_ang, dtype=float)
    lat_bohr = lattice_ang * (1.0 / 0.529177210903)

    atoms_crystal = []
    for a in spec.atoms:
        frac = np.array(a.frac, dtype=float)
        cart = lat_bohr.T @ frac
        atoms_crystal.append(vq.Atom(int(a.z), cart.tolist()))

    system_crystal = vq.PeriodicSystem(3, lat_bohr, atoms_crystal)
    basis_crystal = vq.BasisSet(system_crystal.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(system_crystal, spec.default_kmesh)

    # --- Crystal SCF ---
    if verbose:
        print(f"  Crystal SCF ({name}, {method}/{functional or 'HF'}, "
              f"Z={ref['Z']})...", end=" ", flush=True)
    result_crystal = run_bipole_scf(
        system_crystal, basis_crystal, kmesh,
        method=method, functional=functional,
        cutoff=cutoff, progress=False,
    )
    e_crystal = result_crystal.energy
    if verbose:
        print(f"E = {e_crystal:.6f} Ha  ({result_crystal.n_iter} iters, "
              f"converged={result_crystal.converged})")

    # --- Isolated molecule SCF ---
    if verbose:
        print(f"  Molecule SCF...", end=" ", flush=True)
    mol = system_crystal.unit_cell_molecule()
    # Extract one formula unit (first Z atoms).
    # For urea Z=2, there are 16 atoms. One molecule has 8 atoms.
    n_atoms_per_mol = len(atoms_crystal) // ref["Z"]
    mol_atoms = atoms_crystal[:n_atoms_per_mol]
    molecule = vq.Molecule(mol_atoms)
    basis_mol = vq.BasisSet(molecule, "sto-3g")

    if functional:
        from vibeqc import RKSJob, run_rks
        opts_mol = vq.RKSOptions()
        opts_mol.functional = functional
        opts_mol.max_iter = 100
        opts_mol.conv_tol_energy = 1e-9
        opts_mol.initial_guess = vq.InitialGuess.SAD
        result_mol = run_rks(molecule, basis_mol, opts_mol)
    else:
        from vibeqc import RHFJob, run_rhf
        opts_mol = vq.RHFOptions()
        opts_mol.max_iter = 100
        opts_mol.conv_tol_energy = 1e-9
        opts_mol.initial_guess = vq.InitialGuess.SAD
        result_mol = run_rhf(molecule, basis_mol, opts_mol)
    e_molecule = result_mol.energy
    if verbose:
        print(f"E = {e_molecule:.6f} Ha  "
              f"(converged={result_mol.converged})")

    # --- Lattice energy ---
    e_latt_ha = e_crystal / ref["Z"] - e_molecule
    e_latt_kjmol = e_latt_ha / KJMOL_TO_HA
    delta = e_latt_kjmol - ref["lattice_kjmol"]

    if verbose:
        print(f"  E_latt = {e_latt_kjmol:.2f} kJ/mol "
              f"(ref: {ref['lattice_kjmol']:.1f}, "
              f"Δ = {delta:+.1f} kJ/mol)")
        print()

    return {
        "name": name,
        "e_crystal": e_crystal,
        "e_molecule": e_molecule,
        "e_latt_ha": e_latt_ha,
        "e_latt_kjmol": e_latt_kjmol,
        "ref_kjmol": ref["lattice_kjmol"],
        "delta_kjmol": delta,
        "converged": result_crystal.converged and result_mol.converged,
    }


def main():
    parser = argparse.ArgumentParser(
        description="X23 lattice energies via BIPOLE periodic SCF")
    parser.add_argument("name", nargs="?", default=None,
                        help="X23 member name (urea, benzene, ice_ih) or --all")
    parser.add_argument("--all", action="store_true",
                        help="Compute all available X23 members")
    parser.add_argument("--method", default="RHF",
                        help="SCF method (RHF or RKS)")
    parser.add_argument("--functional", default=None,
                        help="DFT functional for RKS (e.g., PBE, LDA)")
    parser.add_argument("--cutoff", type=float, default=12.0,
                        help="BIPOLE lattice cutoff (bohr)")
    args = parser.parse_args()

    if args.all:
        names = list(X23_REFERENCE)
    elif args.name:
        names = [args.name]
    else:
        parser.print_help()
        return

    print(f"X23 lattice energies via BIPOLE ({args.method}"
          + (f"/{args.functional}" if args.functional else "")
          + f", cutoff {args.cutoff} bohr)")
    print("=" * 60)

    results = []
    for name in names:
        try:
            r = compute_lattice_energy(
                name, method=args.method, functional=args.functional,
                cutoff=args.cutoff,
            )
            results.append(r)
        except Exception as exc:
            print(f"  FAILED: {name} — {exc}")
            print()

    if results:
        print("Summary:")
        print(f"  {'Name':<12} {'E_latt':>8} {'Ref':>8} {'Δ':>8} {'Conv'}")
        print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*5}")
        for r in results:
            print(f"  {r['name']:<12} {r['e_latt_kjmol']:>8.1f} "
                  f"{r['ref_kjmol']:>8.1f} {r['delta_kjmol']:>+8.1f} "
                  f"{'OK' if r['converged'] else 'FAIL'}")


if __name__ == "__main__":
    main()
