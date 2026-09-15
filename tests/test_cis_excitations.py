"""Test CIS excitation energies for H2O — MSINDO/INDO.

Validates:
1. CIS singlet/triplet excitation energies are positive and reasonable.
2. CIS amplitude vectors are normalized.
3. Consistency with the general vibeqc.excited.cis_excitations kernel
   (same integrals must give identical results).
"""

from __future__ import annotations

import numpy as np
import pytest
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
    CISResult,
    _build_cis_hamiltonian,
    _build_mo_integrals,
    run_cis,
)

# --------------------------------------------------------------------------- #
# H2O geometry and helpers                                                    #
# --------------------------------------------------------------------------- #

H2O_Z = [8, 1, 1]
H2O_COORDS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.757, 0.586],
        [0.0, -0.757, 0.586],
    ]
)  # Angstrom


def _run_reference_scf():
    """Return (C_mo, mo_eps, G0, blocks, nocc) for the H2O ground state."""
    C_bohr = H2O_COORDS * ANGSTROM_TO_BOHR
    blocks, nsto = _atom_blocks(H2O_Z)
    cz = [eff_core_charge(z) for z in H2O_Z]
    nelec = sum(cz)
    nocc = nelec // 2

    H, G = _build_core_and_gamma(H2O_Z, C_bohr, blocks, nsto)
    P, _F, _e_elec, _eps, converged, _it = _scf_rhf(
        H, G, blocks, H2O_Z, nocc, max_iter=200, conv_tol=1e-10
    )
    assert converged, "SCF did not converge"

    F = _build_fock(H, G, P, blocks, H2O_Z)
    mo_eps, C_mo = np.linalg.eigh(F)
    return C_mo, mo_eps, G, [(lo, hi) for lo, hi in blocks], nocc


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #


class TestCISHamiltonian:
    """Tests for _build_cis_hamiltonian."""

    def test_symmetry(self):
        """CIS Hamiltonian must be symmetric."""
        C_mo, mo_eps, G0, blocks, nocc = _run_reference_scf()
        ovov, oovv = _build_mo_integrals(C_mo, nocc, G0, blocks)
        H = _build_cis_hamiltonian(mo_eps, nocc, ovov, oovv, "singlet")
        assert np.allclose(H, H.T), "CIS Hamiltonian is not symmetric"
        Ht = _build_cis_hamiltonian(mo_eps, nocc, ovov, oovv, "triplet")
        assert np.allclose(Ht, Ht.T), "Triplet CIS Hamiltonian is not symmetric"

    def test_singlet_vs_triplet(self):
        """Lowest triplet <= lowest singlet (Hund's rule)."""
        C_mo, mo_eps, G0, blocks, nocc = _run_reference_scf()
        ovov, oovv = _build_mo_integrals(C_mo, nocc, G0, blocks)
        Hs = _build_cis_hamiltonian(mo_eps, nocc, ovov, oovv, "singlet")
        Ht = _build_cis_hamiltonian(mo_eps, nocc, ovov, oovv, "triplet")
        ws = np.sort(np.linalg.eigh(Hs)[0])
        wt = np.sort(np.linalg.eigh(Ht)[0])
        # Lowest triplet <= lowest singlet (Hund's rule)
        assert wt[0] <= ws[0] + 1e-10, f"T1 {wt[0]:.6f} > S1 {ws[0]:.6f}"


class TestMOIntegrals:
    """Tests for _build_mo_integrals."""

    def test_consistency_with_einsum(self):
        """_build_mo_integrals must match the dense _indo_ao_eri + einsum path."""
        C_mo, mo_eps, G0, blocks, nocc = _run_reference_scf()

        # Reference: dense _indo_ao_eri tensor contracted by einsum.
        g_ao = _indo_ao_eri(G0, blocks)
        Co = C_mo[:, :nocc]
        Cv = C_mo[:, nocc:]
        ref_ovov_4d = np.einsum("mnls,mi,na,lj,sb->iajb", g_ao, Co, Cv, Co, Cv)
        ref_oovv_4d = np.einsum("mnls,mi,nj,la,sb->ijab", g_ao, Co, Co, Cv, Cv)

        # Sparse implementation must reproduce both dense tensors exactly.
        # ovov[i,a,j,b] = (ia|jb);  oovv[i,j,a,b] = (ij|ab).
        ovov, oovv = _build_mo_integrals(C_mo, nocc, G0, blocks)
        assert np.allclose(ovov, ref_ovov_4d, atol=1e-12), "ovov mismatch"
        assert np.allclose(oovv, ref_oovv_4d, atol=1e-12), "oovv mismatch"

    def test_cis_energies_consistency(self):
        """CIS energies from sparse path == energies from vibeqc.excited."""
        from vibeqc.excited import cis_excitations

        C_mo, mo_eps, G0, blocks, nocc = _run_reference_scf()

        # Reference path
        g_ao = _indo_ao_eri(G0, blocks)
        Co = C_mo[:, :nocc]
        Cv = C_mo[:, nocc:]
        ref_ovov = np.einsum("mnls,mi,na,lj,sb->iajb", g_ao, Co, Cv, Co, Cv)
        ref_oovv = np.einsum("mnls,mi,nj,la,sb->ijab", g_ao, Co, Co, Cv, Cv)
        ref_cis = cis_excitations(
            mo_eps, nocc, ref_ovov, ref_oovv, spin="singlet", n_states=3
        )

        # Sparse path
        ovov, oovv = _build_mo_integrals(C_mo, nocc, G0, blocks)
        H = _build_cis_hamiltonian(mo_eps, nocc, ovov, oovv, "singlet")
        w, _V = np.linalg.eigh(H)
        w = w[:3]

        assert np.allclose(w, ref_cis.excitation_energies, atol=1e-10), (
            f"CIS energy mismatch: {w} vs {ref_cis.excitation_energies}"
        )

        # Also check triplet
        ref_cis_t = cis_excitations(
            mo_eps, nocc, ref_ovov, ref_oovv, spin="triplet", n_states=3
        )
        Ht = _build_cis_hamiltonian(mo_eps, nocc, ovov, oovv, "triplet")
        wt, _Vt = np.linalg.eigh(Ht)
        wt = wt[:3]
        assert np.allclose(wt, ref_cis_t.excitation_energies, atol=1e-10), (
            f"Triplet CIS energy mismatch: {wt} vs {ref_cis_t.excitation_energies}"
        )


class TestRunCIS:
    """End-to-end tests for run_cis()."""

    def test_h2o_singlet(self):
        """H2O singlet CIS: positive energies, reasonable magnitudes."""
        result = run_cis(H2O_Z, H2O_COORDS, n_states=3, spin="singlet")

        assert isinstance(result, CISResult)
        assert result.spin == "singlet"
        assert result.converged

        # H2O: O(4) + 2×H(1) = 6 AOs, 8 valence e⁻ → 4 occupied, 2 virtual
        assert result.nocc == 4
        assert result.nvir == 2

        # 3 states, energies in Hartree
        assert len(result.excitation_energies) == 3
        assert len(result.coefficients) == 3

        # Energies should be positive and < ~2 Ha
        for w in result.excitation_energies:
            assert 0.0 < w < 2.0, f"Excitation energy {w:.6f} out of range"

        # Energies in eV
        assert np.allclose(
            result.excitation_energies_ev,
            result.excitation_energies * 27.211386245988,
        )

        # Amplitudes should be normalized
        for s, coeff in enumerate(result.coefficients):
            assert len(coeff) == result.nocc * result.nvir + 1
            # HF coefficient is 0 in TDA
            assert abs(coeff[0]) < 1e-14, f"State {s}: HF coeff ≠ 0"
            # CIS amplitudes normalized: ||V|| = 1
            assert abs(np.linalg.norm(coeff[1:]) - 1.0) < 1e-12, (
                f"State {s}: amplitudes not normalized"
            )

        # First excitation energy should be reasonable for H2O
        # (H2O first singlet is around 6-8 eV in INDO)
        w1_ev = result.excitation_energies_ev[0]
        assert 4.0 < w1_ev < 15.0, (
            f"H2O first singlet excitation {w1_ev:.2f} eV out of expected range"
        )

    def test_h2o_triplet(self):
        """H2O triplet CIS."""
        result = run_cis(H2O_Z, H2O_COORDS, n_states=3, spin="triplet")

        assert result.spin == "triplet"
        assert result.converged

        for w in result.excitation_energies:
            assert 0.0 < w < 2.0, f"Triplet excitation energy {w:.6f} out of range"

        for coeff in result.coefficients:
            assert abs(coeff[0]) < 1e-14
            assert abs(np.linalg.norm(coeff[1:]) - 1.0) < 1e-12

    def test_singlet_vs_triplet_ordering(self):
        """Triplet S1 ≤ singlet S1."""
        sing = run_cis(H2O_Z, H2O_COORDS, n_states=1, spin="singlet")
        trip = run_cis(H2O_Z, H2O_COORDS, n_states=1, spin="triplet")
        assert trip.excitation_energies[0] <= sing.excitation_energies[0] + 1e-10, (
            f"Triplet T1 ({trip.excitation_energies_ev[0]:.3f} eV) "
            f"> singlet S1 ({sing.excitation_energies_ev[0]:.3f} eV)"
        )

    def test_n_states_clipping(self):
        """n_states larger than NDET should be clipped."""
        result = run_cis(H2O_Z, H2O_COORDS, n_states=20, spin="singlet")
        # H2O has 4 occ × 2 vir = 8 determinants max
        assert len(result.excitation_energies) == 8
        assert len(result.coefficients) == 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
