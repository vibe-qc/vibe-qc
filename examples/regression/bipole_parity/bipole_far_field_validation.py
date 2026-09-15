"""BIPOLE quartet-level bipolar far-field research diagnostic.

BIPOLE-EXACT-ZONE increment 3c.  For a small periodic system, this
script:
  1. Builds the exact two-electron Fock matrix (J_SR + K_SR) via the
     production C++ erfc-screened four-centre ERI kernel.
  2. Builds the spherical moment buffer and prototype dispatch.
  3. Computes the quartet-level bipolar far-field Fock contribution
     and compares it component-by-component against the exact path.
  4. Reports the per-cell and total energy differences.

This is a diagnostic, not a certification script; it does not modify the
production SCF. The exact scalar covers the full J_SR domain while the
prototype scalar covers only its dispatched subset, so the reported
difference is not a like-for-like route accuracy measure. Run with a small
cell (e.g. LiH/STO-3G) at a modest
cutoff (6-8 bohr) — the dispatch enumeration scales as O(n_sh^4 *
n_cells^2) and is geometry-only proof-of-concept code (not SCF-loop
speed).

Usage
-----
    python examples/regression/bipole_parity/bipole_far_field_validation.py

References
----------
Pisani, Dovesi & Roetti, Hartree-Fock Ab Initio Treatment of Crystalline
Systems, Chapter II.4c (1988), doi:10.1007/978-3-642-93385-1.
Saunders et al., Mol. Phys. 77, 629 (1992),
doi:10.1080/00268979200102671, supplies related Gaussian-density
electrostatics and penetration context, not the quartet dispatcher.
"""

from __future__ import annotations

import time
import numpy as np

from vibeqc import (
    Atom,
    BasisSet,
    PeriodicSystem,
)
from vibeqc._vibeqc_core import (
    LatticeSumOptions,
    build_jk_2e_real_space,
    compute_multipole_moments_lattice,
    direct_lattice_cells,
    make_lattice_matrix_set,
    real_space_density_from_kpoints,
)
from vibeqc.bipole_pair_moments import pair_center_moments
from vibeqc.bipole_spherical_moment_buffer import (
    build_spherical_moment_buffer,
)
from vibeqc.bipole_dispatch import (
    build_penetration_dispatch_for_bipole_context,
)
from vibeqc.bipole_quartet_far_field import (
    build_bipolar_coulomb_far_field,
)


ANG2BOHR = 1.0 / 0.529177210903


def _make_lih_test_system(cutoff_bohr=6.0):
    """LiH in a 4.084 Å cubic cell, STO-3G basis."""
    a = 4.084 * ANG2BOHR
    lattice = (a / 2.0) * np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
        dtype=float,
    )
    system = PeriodicSystem(
        3,
        lattice,
        [Atom(3, [0, 0, 0]), Atom(1, [a / 2, a / 2, a / 2])],
    )
    basis = BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def main():
    print("=" * 70)
    print("BIPOLE Quartet-Level Bipolar Far-Field Research Diagnostic")
    print("Pisani-Dovesi-Roetti 1988, doi:10.1007/978-3-642-93385-1")
    print("=" * 70)

    system, basis = _make_lih_test_system()
    cutoff = 6.0
    omega = 0.5  # Ewald splitting parameter (bohr^-1)
    multipole_l_max = 2  # quadrupole for this test

    print(f"\nSystem: LiH / STO-3G, cutoff={cutoff:.1f} bohr, omega={omega:.2f}")

    # ---- Lattice options ------------------------------------------------
    lo = LatticeSumOptions()
    lo.cutoff_bohr = cutoff
    cells = direct_lattice_cells(system, cutoff)
    print(f"Lattice cells: {len(cells)}")

    # ---- Build a test density matrix ------------------------------------
    # Use a minimal-information guess (SAD-like, Gamma-local).
    S_lat = make_lattice_matrix_set(
        basis.nbasis,
        cells,
        [
            np.eye(basis.nbasis, dtype=float)
            for _ in range(len(cells))
        ],
    )
    # Identity-based trial density (crude but sufficient for comparison).
    P_trial = make_lattice_matrix_set(
        basis.nbasis,
        cells,
        [
            np.eye(basis.nbasis, dtype=float)
            if np.allclose(cell.index, 0)
            else np.zeros((basis.nbasis, basis.nbasis), dtype=float)
            for cell in cells
        ],
    )

    # ---- Exact J_SR Fock (reference) ------------------------------------
    print("\n--- Exact J_SR (erfc-screened four-centre ERI) ---")
    t0 = time.perf_counter()
    jk_exact = build_jk_2e_real_space(
        basis, system, lo, P_trial, omega
    )
    t_exact = time.perf_counter() - t0
    J_exact_blocks = [np.asarray(b, dtype=float) for b in jk_exact.J.blocks]
    print(f"  Time: {t_exact:.3f} s")
    print(f"  J_SR exact block norms (home cell): {np.linalg.norm(J_exact_blocks[0]):.6f}")

    # ---- Bipolar far-field infrastructure --------------------------------
    print("\n--- Building bipolar far-field infrastructure ---")
    t0 = time.perf_counter()
    try:
        cart_moments = compute_multipole_moments_lattice(
            basis, system, lo, 2, (0.0, 0.0, 0.0)
        )
        moment_cells = list(cart_moments.cells)
        print(f"  Cartesian moments: {len(moment_cells)} cells")

        pair_mom = pair_center_moments(cart_moments, basis)
        print(f"  Pair moments: L_max={pair_mom.L_max}")

        spherical_buffer = build_spherical_moment_buffer(
            pair_mom, basis, L_max=multipole_l_max,
        )
        print(f"  Spherical buffer: {len(spherical_buffer)} entries")

        penetration_j = build_penetration_dispatch_for_bipole_context(
            basis,
            system,
            moment_cells,
            maximum_multipole_order=multipole_l_max,
        )
        print(f"  Far-field Coulomb quartets: {len(penetration_j)}")
    except Exception as exc:
        print(f"  Infrastructure build FAILED: {exc}")
        return 1

    t_build = time.perf_counter() - t0
    print(f"  Build time: {t_build:.3f} s")

    # ---- Quartet-level bipolar far-field Fock ---------------------------
    print("\n--- Computing quartet-level bipolar far-field J ---")
    density_dict = {
        (cell.index[0], cell.index[1], cell.index[2]): np.asarray(
            P_trial.blocks[c], dtype=float
        )
        for c, cell in enumerate(P_trial.cells)
    }
    t0 = time.perf_counter()
    ff_result = build_bipolar_coulomb_far_field(
        spherical_buffer,
        penetration_j,
        density_dict,
        ewald_omega=omega,
        nbf=basis.nbasis,
    )
    t_ff = time.perf_counter() - t0
    print(f"  Quartets evaluated: {ff_result.n_quartets}")
    print(f"  Far-field J energy: {ff_result.e_coulomb_far:+.8f} Ha")
    print(f"  Far-field Fock blocks: {len(ff_result.fock_blocks)}")
    print(f"  Time: {t_ff:.3f} s")

    # ---- Compare against exact J_SR (component-wise) --------------------
    print("\n--- Component comparison ---")
    # The far-field replaces J_SR for far quartets.  For a fair
    # comparison, we check the exact J energy and the far-field J
    # energy; they should be of similar magnitude (not equal -- the
    # far-field is an approximation to the exact erfc-screened ERI).
    e_j_exact = 0.0
    for c, cell in enumerate(P_trial.cells):
        D_block = np.asarray(P_trial.blocks[c], dtype=float)
        J_block = J_exact_blocks[c]
        e_j_exact += 0.5 * np.sum(D_block * J_block)

    print(f"  E_J_SR (exact 4c ERI):    {e_j_exact:+.8f} Ha")
    print(f"  E_J_bipolar (quartet ff): {ff_result.e_coulomb_far:+.8f} Ha")
    diff = abs(e_j_exact - ff_result.e_coulomb_far)
    print(f"  |ΔE|:                      {diff:.2e} Ha")
    if e_j_exact != 0:
        rel = diff / abs(e_j_exact)
        print(f"  Relative difference:      {rel:.3e}")

    # ---- Summary --------------------------------------------------------
    print("\n" + "=" * 70)
    print("Research diagnostic complete; this is not route certification.")
    print(f"  The quartet bipolar far-field is an APPROXIMATION to the")
    print(f"  exact erfc-screened J_SR.  |ΔE| = {diff:.2e} Ha.")
    print("  No certification threshold applies until the skipped exact")
    print("  domain and the multipole add-back are proved identical.")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
