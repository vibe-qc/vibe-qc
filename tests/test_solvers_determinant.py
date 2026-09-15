"""Tests for determinant data structures and Slater-Condon rules."""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc.solvers import (
    Det,
    build_hamiltonian_matrix,
    determinant_string,
    diagonal_matrix_element,
    double_excitation_matrix_element,
    excitation_rank,
    generate_closed_shell_determinants,
    generate_doubles,
    generate_singles,
    build_hamiltonian_matrix_unrestricted,
    hamiltonian_matrix_element,
    is_connected,
    pair_excitation_matrix_element,
    reference_determinant,
    single_excitation_matrix_element,
)


class TestDeterminant:
    def test_determinant_string(self):
        assert determinant_string((0, 1, 3)) == "1101"

    def test_excitation_rank(self):
        ref = (0, 1, 2)
        single = (0, 1, 5)  # 2→5
        double = (0, 3, 4)  # 1→3, 2→4
        assert excitation_rank(ref, single) == 1
        assert excitation_rank(ref, double) == 2
        assert excitation_rank(ref, ref) == 0

    def test_is_connected(self):
        ref = (0, 1, 2)
        assert is_connected(ref, (0, 1, 5))  # single
        assert is_connected(ref, (0, 3, 4))  # double
        assert not is_connected(ref, (3, 4, 5))  # triple → not connected

    def test_generate_closed_shell(self):
        dets = generate_closed_shell_determinants(4, 2)
        assert len(dets) == 6  # C(4,2)
        assert (0, 1) in dets
        assert (2, 3) in dets

    def test_generate_singles(self):
        ref = (0, 1)
        singles = generate_singles(ref, norb=4)
        # From (0,1): 0→2, 0→3, 1→2, 1→3
        assert len(singles) == 4
        assert (1, 2) in singles
        assert (0, 3) in singles

    def test_generate_doubles(self):
        ref = (0, 1)
        doubles = generate_doubles(ref, norb=4)
        # Only double: (0,1)→(2,3)
        assert (2, 3) in doubles

    def test_reference_determinant(self):
        ref = reference_determinant(2, 2, spin_restricted=True)
        assert ref == ((0, 1), (0, 1))


class TestSlaterCondon:
    def _make_h1e_h2e(self, norb: int):
        h1e = np.diag(np.arange(1, norb + 1, dtype=float))
        h2e = np.zeros((norb, norb, norb, norb))
        return h1e, h2e

    def test_diagonal_element_hf(self):
        """For the aufbau determinant, the diagonal energy equals the
        closed-shell HF energy expression: 2 Σ_i h_{ii} + Σ_{ij} (2J_{ij} − K_{ij}).
        With zero 2e integrals and unit h1e, E = 2 Σ_i ε_i."""
        norb = 4
        h1e = np.eye(norb)
        h2e = np.zeros((norb, norb, norb, norb))
        det = (0, 1)  # 2 electrons in 4 orbitals
        E = diagonal_matrix_element(det, h1e, h2e)
        assert E == pytest.approx(4.0)  # 2 * (h_{00} + h_{11})

    def test_diagonal_with_eri(self):
        """With J-type only 2e integrals, E_diag = 2 Σ_i h_{ii} + Σ_{ij} 2 J_{ij}.
        Set g_{ijij} = 1 for i≠j.  For 2 occupied: 2*(1+1) + 2*(2*1) = 4 + 4 = 8."""
        norb = 4
        h1e = np.eye(norb)
        h2e = np.zeros((norb, norb, norb, norb))
        # g_{0101} = 1, g_{1010} = 1
        h2e[0, 1, 0, 1] = 1.0
        h2e[1, 0, 1, 0] = 1.0
        det = (0, 1)
        E = diagonal_matrix_element(det, h1e, h2e)
        # 2*(h00+h11) + 2*g_{0101} + 2*g_{1010} - g_{0110} - g_{1001}
        # = 2*(1+1) + 2*1 + 2*1 - 0 - 0 = 4 + 4 = 8
        assert E == pytest.approx(8.0)

    def test_single_excitation_phase(self):
        """Single excitation (0→2) from |0,1⟩ gives phase (−1)^(occupied between 0 and 2).
        Occupied between 0 and 2: orbital 1 → sign = −1."""
        norb = 4
        h1e = np.zeros((norb, norb))
        h1e[0, 2] = 0.5
        h2e = np.zeros((norb, norb, norb, norb))
        det = (0, 1)
        val = single_excitation_matrix_element(det, 0, 2, h1e, h2e)
        # sign = -1, val = h1e[0,2] = 0.5
        assert val == pytest.approx(-0.5)

    def test_double_excitation(self):
        """Double excitation (0,1→2,3): g_{0123} − g_{0132}."""
        norb = 4
        h2e = np.zeros((norb, norb, norb, norb))
        h2e[0, 1, 2, 3] = 0.7
        h2e[0, 1, 3, 2] = 0.2
        det = (0, 1)
        val = double_excitation_matrix_element(det, 0, 1, 2, 3, h2e)
        # Phase trace: occ={0,1}, first (0→2): occupied between 0,2 = {1} → sign = -1
        # occ becomes {1,2}. Second (1→3): occupied between 1,3 = {2} → sign *= -1 → +1
        # value = g_{01,23} - g_{01,32} = 0.7 - 0.2 = 0.5
        # sign = +1, so val = +0.5
        assert val == pytest.approx(0.5)

    def test_hamiltonian_matrix_element_dispatch(self):
        norb = 4
        h1e = np.eye(norb)
        h2e = np.zeros((norb, norb, norb, norb))
        det = (0, 1)

        # Diagonal
        assert hamiltonian_matrix_element(det, det, h1e, h2e) == pytest.approx(4.0)

        # Two pairs moved (0,1 -> 2,3): a four-electron excitation of the
        # closed-shell determinant, zero for a two-body Hamiltonian (#639).
        assert hamiltonian_matrix_element(det, (2, 3), h1e, h2e) == pytest.approx(0.0)

    # ── Seniority-zero (closed-shell Det) rules, GitLab #639 ──────────────

    def _symmetric_integrals(self, norb, seed=0):
        """Random real integrals with the permutational symmetry of a real
        two-electron tensor in physicist notation, g_pqrs = g_qpsr = g_rspq."""
        rng = np.random.default_rng(seed)
        h1 = rng.standard_normal((norb, norb))
        h1 = h1 + h1.T
        g = rng.standard_normal((norb, norb, norb, norb))
        g = g + g.transpose(1, 0, 3, 2)
        g = g + g.transpose(2, 3, 0, 1)
        return h1, g

    def test_pair_excitation_element_is_g_iiaa_with_positive_phase(self):
        """One pair moved i^2 -> a^2 couples through g_{iiaa} = <ii|aa>, phase +1
        for every spectator configuration (Helgaker, Jorgensen & Olsen Sec.1.4:
        the alpha and beta single-sector phases are equal, so their product is
        +1).  Pre-#639 this element was scored with the one-electron single
        rule, which is the Brillouin term and vanishes for HF orbitals."""
        norb = 5
        h1, g = self._symmetric_integrals(norb)
        assert pair_excitation_matrix_element(0, 3, g) == g[0, 0, 3, 3]
        # dispatch: bra (1,3) vs ket (0,1) differ by the pair 0 -> 3; the
        # spectator pair (orbital 1) sits *between* hole and particle, which
        # would flip the sign of a one-electron excitation -- not of a pair.
        assert hamiltonian_matrix_element((1, 3), (0, 1), h1, g) == g[0, 0, 3, 3]
        assert hamiltonian_matrix_element((0, 1), (1, 3), h1, g) == g[3, 3, 0, 0]
        assert g[0, 0, 3, 3] == pytest.approx(g[3, 3, 0, 0])  # Hermitian

    def test_two_pair_move_has_zero_coupling(self):
        norb = 5
        h1, g = self._symmetric_integrals(norb)
        assert hamiltonian_matrix_element((0, 1), (2, 3), h1, g) == 0.0
        assert hamiltonian_matrix_element((0, 1, 2), (0, 3, 4), h1, g) == 0.0

    def test_closed_shell_builder_matches_unrestricted_oracle(self):
        """build_hamiltonian_matrix over closed-shell Dets is the seniority-zero
        Hamiltonian: element by element it equals the spin-orbital builder over
        the same determinants written as ((occ), (occ)) SpinDets -- the exact
        oracle, since the unrestricted rules are the general Slater-Condon
        rules.  Every off-diagonal in the closed-shell space is a single pair
        move; every two-pair move is zero."""
        norb, nocc = 5, 2
        h1, g = self._symmetric_integrals(norb, seed=1)
        dets = generate_closed_shell_determinants(norb, nocc)  # C(5,2) = 10
        H_cs = build_hamiltonian_matrix(dets, h1, g)
        H_u = build_hamiltonian_matrix_unrestricted([(d, d) for d in dets], h1, g)
        assert H_cs.shape == (10, 10)
        np.testing.assert_allclose(H_cs, H_u, rtol=0, atol=1e-12)
        np.testing.assert_allclose(H_cs, H_cs.T, rtol=0, atol=0)
        for I, dI in enumerate(dets):
            for J, dJ in enumerate(dets):
                rank = excitation_rank(dI, dJ)
                if rank >= 2:
                    assert H_cs[I, J] == 0.0
                elif rank == 1:
                    (i,) = set(dJ) - set(dI)
                    (a,) = set(dI) - set(dJ)
                    assert H_cs[I, J] == g[i, i, a, a]

    def test_build_hamiltonian_matrix_fci(self):
        """Full CI for 2e in 2 orbitals should give a 1×1 matrix."""
        norb = 2
        h1e = np.eye(norb)
        h2e = np.zeros((norb, norb, norb, norb))
        h2e[0, 1, 0, 1] = 1.0
        h2e[1, 0, 1, 0] = 1.0

        det_list = [(0, 1)]
        H = build_hamiltonian_matrix(det_list, h1e, h2e)
        assert H.shape == (1, 1)
        expected = 2 * (1.0 + 1.0) + 2 * 1.0 + 2 * 1.0  # = 8.0
        assert H[0, 0] == pytest.approx(expected)
