"""Shared parity-runner glue for the CRYSTAL-demo systems.

Wraps ``run_pbc_bipole_rhf`` with CRYSTAL-defaults-equivalent SCF
aids and returns a pass/fail exit code against a sealed CRYSTAL14
reference energy. Each parity_*.py demo script is essentially a
single :func:`run_demo_parity` call.

Per CLAUDE.md §10: no CRYSTAL import. The reference energies in
each demo script come from local CRYSTAL14 / CRYSTAL23-demo runs
of the matching ``.d12`` in ``crystal_demos/``; generated `.out`
logs are kept out of the repo.
"""
from __future__ import annotations

import sys
import time
from typing import Callable, Optional, Tuple

import numpy as np

import vibeqc as vq
from vibeqc import CoulombMethod, InitialGuess, monkhorst_pack
from vibeqc._vibeqc_core import PeriodicKSOptions, PeriodicRHFOptions
from vibeqc.pbc_bipole import run_pbc_bipole_rhf
from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks


def run_demo_parity(
    *,
    label: str,
    build_fn: Callable[[], Tuple[vq.PeriodicSystem, vq.BasisSet]],
    kmesh_size: Tuple[int, int, int],
    crystal14_ref_ha_per_fu: float,
    cutoff_bohr: float = 18.0,
    nuclear_cutoff_bohr: Optional[float] = None,
    initial_guess: InitialGuess = InitialGuess.SAD,
    use_diis: bool = True,
    diis_start_iter: int = 2,
    damping: float = 0.3,
    max_iter: int = 50,
    use_symmetry: bool = True,
    target_millihartree: float = 1.0,
    conv_tol_energy: float = 1e-7,
    conv_tol_grad: float = 1e-4,
    verbose: int = 5,
    use_ewald_j_split: Optional[bool] = None,
    ewald_omega: Optional[float] = None,
    ewald_precision: float = 1e-8,
) -> int:
    """Build the system + basis, run BIPOLE SCF, compare to CRYSTAL14.

    ``use_ewald_j_split=None`` mirrors :func:`run_pbc_bipole_rhf`:
    the CRYSTAL-gauge Ewald-J path is used automatically for 3D
    systems, while lower-dimensional diagnostic demos remain on the
    direct-only branch until the low-dim BIPOLE work lands.

    Returns an int exit code: 0 = PASS, 1 = converged but off-target,
    2 = SCF failed to converge.
    """
    print(f"=== {label} parity vs CRYSTAL14 ===")
    system, basis = build_fn()
    use_ewald_j_split_resolved = (
        system.dim == 3
        if use_ewald_j_split is None
        else bool(use_ewald_j_split)
    )
    use_symmetry_resolved = bool(use_symmetry)
    if (
        use_ewald_j_split_resolved
        and system.dim == 3
        and np.prod(kmesh_size) > 1
        and use_symmetry_resolved
    ):
        use_symmetry_resolved = False
        print(
            "  use_symmetry auto-disabled: Ewald-J split needs the full "
            "Monkhorst-Pack mesh until IBZ orbit expansion lands."
        )
    print(f"  basis: {basis.name}  ({basis.nbasis} BFs / "
          f"{basis.nshells} shells)")
    print(f"  n_electrons = {system.n_electrons()}")
    print(f"  dim = {system.dim}")
    if system.dim != 3:
        print(f"  WARNING: run_pbc_bipole_rhf is 3D-tested; dim={system.dim} "
              f"may need a low-dim driver.")

    kmesh = monkhorst_pack(
        system, list(kmesh_size), use_symmetry=use_symmetry_resolved,
    )
    n_k = len(list(kmesh.kpoints))
    print(f"  kmesh={kmesh_size}, {n_k} irreducible k-points "
          f"(use_symmetry={use_symmetry_resolved})")

    opts = PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = cutoff_bohr
    opts.lattice_opts.nuclear_cutoff_bohr = (
        nuclear_cutoff_bohr if nuclear_cutoff_bohr is not None else cutoff_bohr
    )
    # coulomb_method is driver-internal in run_pbc_bipole_rhf — V_ne / E_nn
    # always use EWALD_3D for 3D systems; F^2e always uses direct ERIs.
    opts.initial_guess = initial_guess
    opts.max_iter = int(max_iter)
    opts.use_diis = bool(use_diis)
    opts.diis_start_iter = int(diis_start_iter)
    opts.damping = float(damping)
    opts.conv_tol_energy = float(conv_tol_energy)
    opts.conv_tol_grad = float(conv_tol_grad)

    print(f"  cutoff_bohr           = {cutoff_bohr}")
    print(f"  initial guess         = {initial_guess.name}")
    print(f"  use_diis              = {use_diis} (start at iter {diis_start_iter})")
    print(f"  damping               = {damping}")
    print(f"  LEVSHIFT/MOM/ODA      = OFF (CRYSTAL-defaults-equivalent)")
    if use_ewald_j_split_resolved:
        print(f"  use_ewald_j_split     = ON (CRYSTAL-gauge F^2e)")
        print(f"  ewald_precision       = {ewald_precision:.0e}")
    print()

    t0 = time.time()
    result = run_pbc_bipole_rhf(
        system, basis, kmesh, opts,
        level_shift_schedule=None,
        use_mom=False,
        use_oda=False,
        use_ewald_j_split=use_ewald_j_split_resolved,
        ewald_omega=ewald_omega,
        ewald_precision=ewald_precision,
        progress=True, verbose=verbose,
    )
    wall = time.time() - t0

    print()
    print("=== Result ===")
    print(f"  converged:  {result.converged}")
    print(f"  iterations: {result.n_iter}")
    print(f"  wall time:  {wall:.1f}s")
    print(f"  E_total:    {result.energy:.8f} Ha/FU")
    print(f"  E_elec:     {result.e_electronic:.8f} Ha")
    print(f"  E_nuc:      {result.e_nuclear:.8f} Ha")
    print()
    delta_mha = (result.energy - crystal14_ref_ha_per_fu) * 1000.0
    print(f"  CRYSTAL14 ref:  {crystal14_ref_ha_per_fu:.8f} Ha/FU")
    print(f"  Δ:              {delta_mha:+.4f} mHa")
    print(f"  target:        ±{target_millihartree:.4f} mHa")

    if not result.converged:
        print("\n  FAIL: SCF did not converge")
        return 2
    if abs(delta_mha) > target_millihartree:
        print(f"\n  FAIL: Δ exceeds ±{target_millihartree} mHa target")
        return 1
    print("\n  PASS")
    return 0


def run_demo_parity_rks(
    *,
    label: str,
    build_fn: Callable[[], Tuple[vq.PeriodicSystem, vq.BasisSet]],
    functional: str,
    kmesh_size: Tuple[int, int, int],
    crystal_ref_ha_per_fu: float,
    cutoff_bohr: float = 14.0,
    nuclear_cutoff_bohr: Optional[float] = None,
    initial_guess: InitialGuess = InitialGuess.SAD,
    use_diis: bool = True,
    diis_start_iter: int = 2,
    damping: float = 0.3,
    max_iter: int = 50,
    use_symmetry: bool = False,
    target_millihartree: float = 2.0,
    conv_tol_energy: float = 1e-7,
    conv_tol_grad: float = 1e-4,
    verbose: int = 5,
    ewald_omega: Optional[float] = None,
    ewald_precision: float = 1e-8,
) -> int:
    """RKS (DFT) sibling of :func:`run_demo_parity`.

    Wires a builder + a sealed CRYSTAL DFT reference energy + the
    CRYSTAL-gauge ``run_pbc_bipole_rks`` driver into a pass/fail check.
    Used for the meta-GGA parity demos (e.g. MgO r2SCAN/STO-3G against
    the sealed CRYSTAL23 reference). The 3D Ewald-J split is always on
    for the closed-shell DFT route here; smearing is **off** (integer
    occupations) — finite-T smearing on a minimal-basis ionic insulator
    settles a near-metallic basin hundreds of mHa above the physical
    gapped state (the ionic-Γ basin trap; see ``smearing_basin_warning``
    and ``docs/troubleshooting.md``). The RHF parity recipe (damping +
    DIIS, no smearing) is what reproduces CRYSTAL here.

    Per CLAUDE.md §10 the CRYSTAL reference is sealed out-of-process;
    nothing in this path imports a QC program. Returns an int exit code:
    0 = PASS, 1 = converged but off-target, 2 = SCF failed to converge.
    """
    print(f"=== {label} parity vs CRYSTAL ({functional}) ===")
    system, basis = build_fn()
    print(f"  basis: {basis.name}  ({basis.nbasis} BFs / "
          f"{basis.nshells} shells)")
    print(f"  n_electrons = {system.n_electrons()}")
    print(f"  dim = {system.dim}")

    kmesh = monkhorst_pack(
        system, list(kmesh_size), use_symmetry=bool(use_symmetry),
    )
    n_k = len(list(kmesh.kpoints))
    print(f"  kmesh={kmesh_size}, {n_k} k-points "
          f"(use_symmetry={use_symmetry})")

    opts = PeriodicKSOptions()
    opts.functional = str(functional)
    opts.lattice_opts.cutoff_bohr = cutoff_bohr
    opts.lattice_opts.nuclear_cutoff_bohr = (
        nuclear_cutoff_bohr if nuclear_cutoff_bohr is not None else cutoff_bohr
    )
    opts.initial_guess = initial_guess
    opts.max_iter = int(max_iter)
    opts.use_diis = bool(use_diis)
    opts.diis_start_iter = int(diis_start_iter)
    opts.damping = float(damping)
    opts.conv_tol_energy = float(conv_tol_energy)
    opts.conv_tol_grad = float(conv_tol_grad)
    # smearing_temperature stays 0.0 — integer occupations only.

    print(f"  functional            = {functional}")
    print(f"  cutoff_bohr           = {cutoff_bohr}")
    print(f"  initial guess         = {initial_guess.name}")
    print(f"  use_diis              = {use_diis} (start at iter {diis_start_iter})")
    print(f"  damping               = {damping}")
    print(f"  smearing              = OFF (integer occupations)")
    print(f"  use_ewald_j_split     = ON (CRYSTAL-gauge F^2e)")
    print(f"  ewald_precision       = {ewald_precision:.0e}")
    print()

    t0 = time.time()
    result = run_pbc_bipole_rks(
        system, basis, kmesh, opts,
        level_shift_schedule=None,
        use_mom=False,
        use_oda=False,
        use_ewald_j_split=True,
        ewald_omega=ewald_omega,
        ewald_precision=ewald_precision,
        progress=True, verbose=verbose,
    )
    wall = time.time() - t0

    print()
    print("=== Result ===")
    print(f"  converged:  {result.converged}")
    print(f"  iterations: {result.n_iter}")
    print(f"  wall time:  {wall:.1f}s")
    print(f"  E_total:    {result.energy:.8f} Ha/FU")
    print(f"  E_elec:     {result.e_electronic:.8f} Ha")
    print(f"  E_xc:       {result.e_xc:.8f} Ha")
    print(f"  E_nuc:      {result.e_nuclear:.8f} Ha")
    if result.basin_warning:
        print(f"  basin_warning: {result.basin_warning}")
    print()
    delta_mha = (result.energy - crystal_ref_ha_per_fu) * 1000.0
    print(f"  CRYSTAL ref:    {crystal_ref_ha_per_fu:.8f} Ha/FU")
    print(f"  Δ:              {delta_mha:+.4f} mHa")
    print(f"  target:        ±{target_millihartree:.4f} mHa")

    if not result.converged:
        print("\n  FAIL: SCF did not converge")
        return 2
    if abs(delta_mha) > target_millihartree:
        print(f"\n  FAIL: Δ exceeds ±{target_millihartree} mHa target")
        return 1
    print("\n  PASS")
    return 0
