"""BIPOLE far-field research stress probe for dense periodic fixtures.

Exercises the low-level quartet prototype under dense k-meshes, ionic
crystals, and tight screening thresholds. It does not validate an SCF route:
the driver path is fail-closed, and the reported exact and prototype scalars
do not cover the same quartet domain. The quartet expansion is from Pisani,
Dovesi, and Roetti (1988), Chapter II.4c.

Usage:
    .venv/bin/python examples/regression/bipole_parity/bipole_stress_test.py
"""

from __future__ import annotations

import time, sys, numpy as np

def _header(msg):
    print(f"\n{'='*70}\n  {msg}\n{'='*70}")

def build_far_field_diagnostic(system, basis, cutoff, omega, k_mesh, multipole_l_max, overlap_threshold):
    """Build the full bipolar far-field infrastructure and run one Fock build."""
    from vibeqc._vibeqc_core import (
        LatticeSumOptions,
        build_jk_2e_real_space,
        compute_multipole_moments_lattice,
        direct_lattice_cells,
        make_lattice_matrix_set,
        monkhorst_pack,
        real_space_density_from_kpoints,
    )
    from vibeqc.bipole_pair_moments import pair_center_moments
    from vibeqc.bipole_spherical_moment_buffer import build_spherical_moment_buffer
    from vibeqc.bipole_dispatch import (
        PenetrationDispatchParameters,
        build_shell_quartet_penetration_dispatch,
    )
    from vibeqc.bipole_quartet_far_field import build_bipolar_coulomb_far_field
    from vibeqc.bipole_quartet_tensor_cache import build_quartet_tensor_cache

    lo = LatticeSumOptions()
    lo.cutoff_bohr = cutoff
    nbf = basis.nbasis
    n_sh = len(list(basis.shells()))

    # ---- k-mesh ---------------------------------------------------------
    km = monkhorst_pack(system, k_mesh)
    kpts = list(km.kpoints)
    wts = np.asarray(km.weights, dtype=float)
    n_k = len(kpts)
    print(f"  k-mesh: {k_mesh} → {n_k} k-points, cutoff={cutoff} bohr")

    # ---- Trial density (Gamma-local identity guess) --------------------
    cells = direct_lattice_cells(system, cutoff)
    n_cells = len(cells)
    blocks = [
        np.eye(nbf) if np.allclose(c.index, 0)
        else np.zeros((nbf, nbf)) for c in cells
    ]
    P = make_lattice_matrix_set(nbf, cells, blocks)
    print(f"  Lattice cells: {n_cells}")

    # ---- Step 1: Cartesian moments (libint) ---------------------------
    t0 = time.perf_counter()
    cart = compute_multipole_moments_lattice(basis, system, lo, 2, (0.0, 0.0, 0.0))
    dt_cart = time.perf_counter() - t0
    print(f"  Cartesian moments: {len(cart.cells)} cells, {dt_cart:.2f}s")

    # ---- Step 2: Per-pair centre moments ------------------------------
    t0 = time.perf_counter()
    pair_mom = pair_center_moments(cart, basis)
    dt_pair = time.perf_counter() - t0
    print(f"  Pair-centre moments: L_max={pair_mom.L_max}, {dt_pair:.2f}s")

    # ---- Step 3: Spherical moment buffer ------------------------------
    t0 = time.perf_counter()
    buf = build_spherical_moment_buffer(pair_mom, basis, L_max=multipole_l_max)
    dt_buf = time.perf_counter() - t0
    print(f"  Spherical buffer: {len(buf)} entries, L_max={multipole_l_max}, {dt_buf:.2f}s")

    # ---- Step 4: Penetration dispatch ---------------------------------
    t0 = time.perf_counter()
    from vibeqc.bipole_bravais_utils import cell_volume_bohr, cell_dimensionality
    vol = cell_volume_bohr(system)
    dim = cell_dimensionality(system)
    params = PenetrationDispatchParameters.for_cell_volume(
        vol, maximum_multipole_order=multipole_l_max,
        overlap_drop_threshold=overlap_threshold, dimensionality=dim,
    )
    j_disp, _k = build_shell_quartet_penetration_dispatch(
        basis, list(cart.cells), params, compute_exchange=False,
    )
    dt_disp = time.perf_counter() - t0
    n_full = n_sh**4 * n_cells**2
    print(f"  Penetration dispatch: {len(j_disp)} far quartets / {n_full} full")
    print(f"    threshold={overlap_threshold}, {dt_disp:.2f}s")

    # ---- Step 5: Tensor cache -----------------------------------------
    t0 = time.perf_counter()
    tcache = build_quartet_tensor_cache(buf, j_disp, ewald_omega=omega)
    dt_cache = time.perf_counter() - t0
    print(f"  Tensor cache: {len(tcache)} unique, {dt_cache:.2f}s")

    # ---- Step 6: Exact J_SR (reference) -------------------------------
    t0 = time.perf_counter()
    jk_exact = build_jk_2e_real_space(basis, system, lo, P, omega)
    dt_exact = time.perf_counter() - t0
    e_exact = sum(
        0.5 * np.sum(np.asarray(P.blocks[c]) * np.asarray(jk_exact.J.blocks[c]))
        for c in range(n_cells)
    )
    print(f"  Exact J_SR: {e_exact:+.8f} Ha, {dt_exact:.2f}s")

    # ---- Step 7: Bipolar far-field J ----------------------------------
    density_dict = {
        (c.index[0], c.index[1], c.index[2]): np.asarray(P.blocks[i])
        for i, c in enumerate(cells)
    }
    t0 = time.perf_counter()
    ff = build_bipolar_coulomb_far_field(
        buf, j_disp, density_dict,
        ewald_omega=omega, nbf=nbf, tensor_cache=tcache,
    )
    dt_ff = time.perf_counter() - t0
    diff = abs(e_exact - ff.e_coulomb_far)
    print(f"  Bipolar J:   {ff.e_coulomb_far:+.8f} Ha, {dt_ff:.2f}s")
    print(f"  |dE| = {diff:.2e} Ha, quartets={ff.n_quartets}")

    return {
        "system": system, "n_k": n_k, "n_cells": n_cells, "n_sh": n_sh,
        "n_quartets": ff.n_quartets, "n_full": n_full,
        "e_exact": e_exact, "e_bipolar": ff.e_coulomb_far,
        "abs_diff": diff,
        "t_cart": dt_cart, "t_pair": dt_pair, "t_buf": dt_buf,
        "t_disp": dt_disp, "t_cache": dt_cache, "t_exact": dt_exact,
        "t_ff": dt_ff,
    }


def main():
    from vibeqc import Atom, PeriodicSystem, BasisSet
    ANG2BOHR = 1.0 / 0.529177210903

    results = []

    # ------ MgO, rocksalt, pob-TZVP (or STO-3G for quick test) -------
    _header("MgO — rocksalt, STO-3G, k=8×8×8, cutoff=8, omega=0.5")
    a = 4.212 * ANG2BOHR
    lattice = np.diag([a, a, a])
    system = PeriodicSystem(3, lattice, [
        Atom(12, [0.0, 0.0, 0.0]),
        Atom(8,  [a/2, a/2, a/2]),
    ])
    basis = BasisSet(system.unit_cell_molecule(), "sto-3g")
    print(f"  MgO: {basis.nbasis} AOs, {len(list(basis.shells()))} shells")
    r = build_far_field_diagnostic(
        system, basis, cutoff=8.0, omega=0.5,
        k_mesh=(8, 8, 8), multipole_l_max=2, overlap_threshold=1e-6,
    )
    results.append(("MgO 8×8×8", r))

    # ------ Al2O3, corundum, STO-3G, k=6×6×6 --------------------------
    _header("Al2O3 — corundum, STO-3G, k=6×6×6, cutoff=8, omega=0.5")
    a_hex = 4.76 * ANG2BOHR
    c_hex = 12.99 * ANG2BOHR
    lattice = np.array([
        [a_hex, 0.0, 0.0],
        [-a_hex/2, a_hex * np.sqrt(3)/2, 0.0],
        [0.0, 0.0, c_hex],
    ])
    # Corundum Al2O3: 12 Al + 18 O in the hexagonal cell
    # Simplified: 2 formula units = 4 Al + 6 O at high-symmetry positions
    atoms = [
        Atom(13, [0.0, 0.0, 0.3522 * c_hex]),
        Atom(13, [0.0, 0.0, 0.6478 * c_hex]),
        Atom(8,  [0.306, 0.0, 0.25]),
        Atom(8,  [0.0, 0.306, 0.25]),
        Atom(8,  [0.694, 0.694, 0.25]),
    ]
    system_al = PeriodicSystem(3, lattice, atoms)
    basis_al = BasisSet(system_al.unit_cell_molecule(), "sto-3g")
    print(f"  Al2O3: {basis_al.nbasis} AOs, {len(list(basis_al.shells()))} shells")
    r2 = build_far_field_diagnostic(
        system_al, basis_al, cutoff=8.0, omega=0.5,
        k_mesh=(6, 6, 6), multipole_l_max=2, overlap_threshold=1e-6,
    )
    results.append(("Al2O3 6×6×6", r2))

    # ------ Diamond, FCC + basis, STO-3G, k=8×8×8 --------------------
    _header("Diamond — FCC, STO-3G, k=8×8×8, cutoff=8, omega=0.5")
    a_dia = 3.567 * ANG2BOHR  # diamond lattice constant
    lattice_dia = (a_dia / 2.0) * np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ])
    # Diamond: 2-atom basis at (0,0,0) and (1/4,1/4,1/4)
    tau = 0.25 * a_dia
    system_dia = PeriodicSystem(3, lattice_dia, [
        Atom(6, [0.0, 0.0, 0.0]),
        Atom(6, [tau, tau, tau]),
    ])
    basis_dia = BasisSet(system_dia.unit_cell_molecule(), "sto-3g")
    print(f"  Diamond: {basis_dia.nbasis} AOs, {len(list(basis_dia.shells()))} shells")
    r3 = build_far_field_diagnostic(
        system_dia, basis_dia, cutoff=8.0, omega=0.5,
        k_mesh=(8, 8, 8), multipole_l_max=2, overlap_threshold=1e-6,
    )
    results.append(("Diamond 8×8×8", r3))

    # ------ Summary table ------------------------------------------------
    _header("Summary")
    hdr = f"{'System':<16s} {'k':>5s} {'cells':>6s} {'sh':>4s} {'quartets':>10s} {'full':>10s} {'|dE|/Ha':>12s} {'t_infra':>8s} {'t_exact':>8s} {'t_ff':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for name, r in results:
        t_infra = r["t_cart"] + r["t_pair"] + r["t_buf"] + r["t_disp"] + r["t_cache"]
        print(
            f"{name:<16s} {r['n_k']:>5d} {r['n_cells']:>6d} {r['n_sh']:>4d} "
            f"{r['n_quartets']:>10d} {r['n_full']:>10d} "
            f"{r['abs_diff']:>12.2e} {t_infra:>8.2f} {r['t_exact']:>8.2f} "
            f"{r['t_ff']:>8.2f}"
        )

if __name__ == "__main__":
    main()
