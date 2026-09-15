"""Reproduce MgO RKS BIPOLE convergence oscillation.

Per handovers/HANDOVER_BIPOLE_PRODUCTION.md Gap B: MgO primitive cell oscillates
~0.5 Ha in RKS BIPOLE. First rule out a gauge/physics bug (§7 CLAUDE.md),
then decide if it's a convergence-aid gap.

Usage:
    VIBEQC_AOPAIR_FT_BACKEND=python python \
        examples/debug/bipole_debug/mgo_rks_convergence.py

Run with --pyscf to also attempt an out-of-process PySCF reference.
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from typing import Optional

import numpy as np

# Must be set before any vibeqc import
os.environ.setdefault("VIBEQC_AOPAIR_FT_BACKEND", "python")

import vibeqc as vq
from vibeqc.pbc_bipole_rks import PeriodicKSOptions, run_pbc_bipole_rks

ANG2BOHR = 1.0 / 0.529177210903


def build_mgo_primitive(a_ang: float = 4.21):
    """MgO FCC primitive cell (2 atoms: Mg at 0,0,0; O at ½,½,½)."""
    a = a_ang * ANG2BOHR
    lattice = (a / 2.0) * np.array(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    atoms = [
        vq.Atom(12, [0.0, 0.0, 0.0]),
        vq.Atom(8, [0.25 * a, 0.25 * a, 0.25 * a]),
    ]
    return vq.PeriodicSystem(3, lattice, atoms)


def run_rks_trajectory(
    system,
    basis_name="sto-3g",
    kmesh=(1, 1, 1),
    functional="svwn",
    max_iter=100,
    scf_accelerator="kdiis",
    level_shift: Optional[float] = None,
    fock_mixing: Optional[float] = None,
    use_ewald=True,
    verbose=True,
):
    """Run RKS BIPOLE SCF and return convergence trajectory."""
    basis = vq.BasisSet(system.unit_cell_molecule(), basis_name)
    kmesh_obj = vq.monkhorst_pack(system, list(kmesh))

    opts = PeriodicKSOptions()
    opts.lattice_opts.cutoff_bohr = 10.0  # default CRYSTAL-like
    opts.lattice_opts.nuclear_cutoff_bohr = 10.0
    opts.max_iter = max_iter
    opts.conv_tol_energy = 1e-8
    opts.conv_tol_grad = 1e-5
    opts.initial_guess = vq.InitialGuess.SAD
    opts.functional = functional
    opts.use_diis = True
    accel_map = {
        "diis": vq.SCFAccelerator.DIIS,
        "kdiis": vq.SCFAccelerator.KDIIS,
        "ediis": vq.SCFAccelerator.EDIIS,
    }
    opts.scf_accelerator = accel_map.get(scf_accelerator, vq.SCFAccelerator.KDIIS)
    opts.diis_start_iter = 2
    # Exact EDIIS enumerates simplex faces and supports at most 12 stored
    # iterates; DIIS/KDIIS can retain the deeper diagnostic history.
    opts.diis_subspace_size = 12 if scf_accelerator == "ediis" else 15

    if level_shift is not None:
        opts.level_shift = float(level_shift)
    if fock_mixing is not None:
        opts.fock_mixing = float(fock_mixing)

    if verbose:
        print(
            f"  Running {functional} RKS BIPOLE on MgO primitive "
            f"(SHRINK {kmesh[0]} {kmesh[1]} {kmesh[2]}), "
            f"max_iter={max_iter}, accel={scf_accelerator}"
        )
        print(f"  level_shift={level_shift}, fock_mixing={fock_mixing}")

    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = run_pbc_bipole_rks(
                system,
                basis,
                kmesh_obj,
                opts,
                use_ewald_j_split=use_ewald,
                ewald_precision=1e-8,
                progress=verbose,
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            return None, None, None

    elapsed = time.time() - t0
    if verbose:
        print(
            f"  Converged: {result.converged}, n_iter={result.n_iter}, "
            f"E={result.energy:.8f} Ha, time={elapsed:.1f}s"
        )

    # Check for energy trace
    energies = None
    if hasattr(result, "energy_trace"):
        energies = list(result.energy_trace)
    elif hasattr(result, "energy_history"):
        energies = list(result.energy_history)
    elif hasattr(result, "_energy_history"):
        energies = list(result._energy_history)
    else:
        energies = [result.energy]

    return result, energies, elapsed


def run_rhf_reference(system, basis_name="sto-3g", verbose=True):
    """Run RHF BIPOLE SCF as a reference (known-stable)."""
    from vibeqc.pbc_bipole import PeriodicRHFOptions, run_pbc_bipole_rhf

    basis = vq.BasisSet(system.unit_cell_molecule(), basis_name)
    kmesh_obj = vq.monkhorst_pack(system, [1, 1, 1])

    opts = PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 10.0
    opts.lattice_opts.nuclear_cutoff_bohr = 10.0
    opts.max_iter = 100
    opts.use_diis = True
    opts.conv_tol_energy = 1e-8
    opts.conv_tol_grad = 1e-5

    t0 = time.time()
    result = run_pbc_bipole_rhf(
        system,
        basis,
        kmesh_obj,
        opts,
        use_ewald_j_split=True,
        ewald_precision=1e-8,
        progress=verbose,
    )
    elapsed = time.time() - t0
    if verbose:
        print(
            f"  RHF converged: {result.converged}, n_iter={result.n_iter}, "
            f"E={result.energy:.8f} Ha, time={elapsed:.1f}s"
        )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="MgO RKS BIPOLE convergence diagnostic"
    )
    parser.add_argument("--a", type=float, default=4.21, help="Lattice constant in Å")
    parser.add_argument(
        "--func", default="svwn", help="XC functional (svwn, pbe, b3lyp)"
    )
    parser.add_argument("--max-iter", type=int, default=150)
    parser.add_argument("--level-shift", type=float, default=None)
    parser.add_argument("--fock-mixing", type=float, default=None)
    parser.add_argument("--accel", default="kdiis", choices=["diis", "kdiis", "ediis"])
    parser.add_argument("--kmesh", type=int, nargs=3, default=[1, 1, 1])
    parser.add_argument("--basis", default="sto-3g", help="Basis set name")
    parser.add_argument("--no-rhf", action="store_true", help="Skip RHF reference")
    args = parser.parse_args()

    print("=" * 72)
    print("MgO RKS BIPOLE convergence diagnostic")
    print("=" * 72)
    print(f"  System: MgO primitive, a={args.a} Å ({args.a * ANG2BOHR:.3f} bohr)")
    print(f"  Basis: {args.basis}, Functional: {args.func}")
    print(f"  kmesh: {args.kmesh}, accel: {args.accel}")
    if args.level_shift:
        print(f"  level_shift: {args.level_shift}")
    if args.fock_mixing:
        print(f"  fock_mixing: {args.fock_mixing}")

    system = build_mgo_primitive(args.a)

    # ---- RKS run ----
    print("\n--- RKS BIPOLE ---")
    result_ks, energies_ks, t_ks = run_rks_trajectory(
        system,
        basis_name=args.basis,
        kmesh=args.kmesh,
        functional=args.func,
        max_iter=args.max_iter,
        scf_accelerator=args.accel,
        level_shift=args.level_shift,
        fock_mixing=args.fock_mixing,
    )

    if energies_ks is not None and len(energies_ks) > 1:
        print("\n  Energy trajectory (first 10):")
        for i, e in enumerate(energies_ks[:10]):
            print(f"    iter {i:3d}:  E={e:.8f} Ha")
        if len(energies_ks) > 10:
            print(f"    ... ({len(energies_ks) - 10} more iters)")
            print("    last 5:")
            for i, e in enumerate(energies_ks[-5:]):
                print(f"    iter {len(energies_ks) - 5 + i:3d}:  E={e:.8f} Ha")
        e_min = min(energies_ks)
        e_max = max(energies_ks)
        print(
            f"\n  Energy range: [{e_min:.6f}, {e_max:.6f}] Ha, "
            f"Δ = {e_max - e_min:.6f} Ha"
        )

    # ---- RHF reference ----
    if not args.no_rhf:
        print("\n--- RHF BIPOLE reference ---")
        result_hf = run_rhf_reference(system, args.basis)
        if result_hf is not None and result_hf.converged:
            print(f"  RHF reference E = {result_hf.energy:.8f} Ha")

    print("\nDone.")


if __name__ == "__main__":
    main()
