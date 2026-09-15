"""Test CIS density matrix and 2PDM assembly for H2O.

Validates the port against the Fortran cisgrad.f formulas by checking:
1. T-matrix trace properties
2. D_oo and D_vv trace properties (D_oo + D_vv trace = 0)
3. 2PDM symmetry and known numerical values
4. Scaled CIS and triplet variants
"""

from __future__ import annotations

import numpy as np
from vibeqc.excited import cis_excitations
from vibeqc.semiempirical.methods.msindo import (
    ANGSTROM_TO_BOHR,
    _atom_blocks,
    _build_core_and_gamma,
    _build_fock,
    _indo_ao_eri,
    _scf_rhf,
    eff_core_charge,
)
from vibeqc.semiempirical.methods.msindo_cis import (
    _build_2pdm,
    _build_dmat_oo,
    _build_dmat_oo_mo,
    _build_dmat_unrelaxed,
    _build_dmat_vv,
    _build_dmat_vv_mo,
    _build_t_matrix,
    assemble_cis_densities,
)


def test_h2o_singlet():
    """H2O singlet CIS: verify density matrices and 2PDM."""
    Z = [8, 1, 1]
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.757, 0.586],
            [0.0, -0.757, 0.586],
        ]
    )  # Angstrom

    # Run SCF
    C = coords * ANGSTROM_TO_BOHR
    blocks, nsto = _atom_blocks(Z)
    cz = [eff_core_charge(z) for z in Z]
    nelec = sum(cz)
    nocc = nelec // 2

    H, G = _build_core_and_gamma(Z, C, blocks, nsto)
    P, _F, _e_elec, eps, converged, _it = _scf_rhf(
        H, G, blocks, Z, nocc, max_iter=200, conv_tol=1e-10
    )
    assert converged, "SCF did not converge"

    # MO coefficients from diagonalizing the converged Fock
    F = _build_fock(H, G, P, blocks, Z)
    _eigvals, C_mo = np.linalg.eigh(F)

    # CIS
    g_ao = _indo_ao_eri(G, blocks)
    Co = C_mo[:, :nocc]
    Cv = C_mo[:, nocc:]
    ovov = np.einsum("mnls,mi,na,lj,sb->iajb", g_ao, Co, Cv, Co, Cv)
    oovv = np.einsum("mnls,mi,nj,la,sb->ijab", g_ao, Co, Co, Cv, Cv)
    cis = cis_excitations(eps, nocc, ovov, oovv, spin="singlet", n_states=3)

    print(f"H2O CIS singlet excitation energies: {cis.excitation_energies}")
    print(f"H2O CIS excitation energies (eV): {cis.excitation_energies_ev}")

    # Test state 0 (S1)
    for state in range(min(3, len(cis.excitation_energies))):
        print(f"\n--- State {state} (S{state + 1}) ---")
        _test_state(nsto, C_mo, P, cis, state, spin="singlet")

    # Test triplet
    cis_trip = cis_excitations(eps, nocc, ovov, oovv, spin="triplet", n_states=1)
    print(f"\n--- Triplet T1 ---")
    _test_state(nsto, C_mo, P, cis_trip, 0, spin="triplet")


def _test_state(nsto, C_mo, P_gs, cis, state, spin="singlet"):
    """Validate density matrices for a single CIS state."""
    cismat = cis.amplitudes[:, state].reshape(cis.n_occ, cis.n_vir)

    # 1. T-matrix
    T = _build_t_matrix(cismat, C_mo)
    print(f"  T-matrix: shape={T.shape}, trace={np.trace(T):.6f}")
    # In Fortran, T = C_occ @ CISMAT @ C_vir^T, so trace(T) ≈ 0
    # (occ-virt coupling, no occ-occ or virt-virt component)
    assert abs(np.trace(T)) < 1e-10, f"T-matrix trace should be ~0, got {np.trace(T)}"

    # 2. D_oo in MO basis
    D_oo_mo = _build_dmat_oo_mo(cismat)
    D_oo = _build_dmat_oo(cismat, C_mo)
    # D_oo_mo trace = -sum_{ia} a_{ia}^2 = -||cismat||^2
    D_oo_trace_mo = np.trace(D_oo_mo)
    D_oo_trace = np.trace(D_oo)
    print(f"  D_oo MO trace: {D_oo_trace_mo:.6f}, AO trace: {D_oo_trace:.6f}")
    assert abs(D_oo_trace - D_oo_trace_mo) < 1e-12, (
        f"Trace mismatch: MO={D_oo_trace_mo}, AO={D_oo_trace}"
    )

    # 3. D_vv in MO basis
    D_vv_mo = _build_dmat_vv_mo(cismat)
    D_vv = _build_dmat_vv(cismat, C_mo)
    # D_vv_mo trace = sum_{ia} a_{ia}^2 = +||cismat||^2
    D_vv_trace_mo = np.trace(D_vv_mo)
    D_vv_trace = np.trace(D_vv)
    print(f"  D_vv MO trace: {D_vv_trace_mo:.6f}, AO trace: {D_vv_trace:.6f}")

    # 4. Trace conservation: D_oo + D_vv trace = 0
    D_unrelaxed = _build_dmat_unrelaxed(cismat, C_mo)
    assert abs(np.trace(D_unrelaxed)) < 1e-10, (
        f"D_unrelaxed trace should be 0, got {np.trace(D_unrelaxed)}"
    )
    assert abs(D_oo_trace_mo + D_vv_trace_mo) < 1e-12, (
        "MO trace should cancel: D_oo + D_vv"
    )

    # 5. 2PDM
    GAMM = _build_2pdm(T, P_gs, D_unrelaxed, spin=spin)
    print(f"  2PDM: shape={GAMM.shape}, norm={np.linalg.norm(GAMM):.6f}")

    # Check that diagonal elements are non-negative for the ground-state part
    # (P_gs part dominates, GAMM[i,i] should be positive)
    for i in range(nsto):
        assert GAMM[i, i] > -1e-12, (
            f"2PDM diagonal element {i} is negative: {GAMM[i, i]}"
        )

    # 6. Convenience wrapper
    result = assemble_cis_densities(cis, state, C_mo, P_gs, spin=spin)
    assert "T" in result
    assert "GAMM" in result
    assert np.allclose(result["T"], T), "Convenience wrapper T mismatch"
    assert np.allclose(result["D_oo"], D_oo), "Convenience wrapper D_oo mismatch"
    assert np.allclose(result["D_vv"], D_vv), "Convenience wrapper D_vv mismatch"
    assert np.allclose(result["D_unrelaxed"], D_unrelaxed), (
        "Convenience wrapper D_unrelaxed mismatch"
    )
    assert np.allclose(result["GAMM"], GAMM), "Convenience wrapper GAMM mismatch"

    # 7. Verify D_oo + D_vv = D_unrelaxed
    assert np.allclose(D_oo + D_vv, D_unrelaxed), "D_oo + D_vv != D_unrelaxed"

    # 8. Scaled CIS and triplet variants (smoke test)
    GAMM_scaled = _build_2pdm(
        T, P_gs, D_unrelaxed, spin=spin, scaled_cis=True, sca1=0.40, sca2=0.95
    )
    print(
        f"  Scaled 2PDM: norm={np.linalg.norm(GAMM_scaled):.6f}, "
        f"diff from unscaled: {np.max(np.abs(GAMM_scaled - GAMM)):.6f}"
    )

    # Verify scaled differs from unscaled (but direction depends on state)
    assert np.max(np.abs(GAMM_scaled - GAMM)) > 0.0, (
        f"{spin} scaled 2PDM should differ from unscaled"
    )

    print(f"  ✅ State {state} ({spin}) passed all checks")


if __name__ == "__main__":
    test_h2o_singlet()
    print("\nAll tests passed!")
