"""Tests for bond analysis methods: Wiberg, EDA, entanglement, NBO.

Covers both molecular and periodic pathways. COHP/COOP is NOT here --
the canonical periodic COHP/COOP implementation is
:mod:`vibeqc.coop_cohp` (tests in ``test_coop_cohp.py``).
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _h2o() -> vq.Molecule:
    return vq.Molecule(
        [
            vq.Atom(8, [0.0, 0.0, 0.0]),
            vq.Atom(1, [0.0, 1.43, -0.98]),
            vq.Atom(1, [0.0, -1.43, -0.98]),
        ]
    )


def _co() -> vq.Molecule:
    return vq.Molecule(
        [
            vq.Atom(6, [0.0, 0.0, 0.0]),
            vq.Atom(8, [0.0, 0.0, 2.132]),
        ]
    )


def _h2() -> vq.Molecule:
    return vq.Molecule(
        [
            vq.Atom(1, [0.0, 0.0, 0.0]),
            vq.Atom(1, [0.0, 0.0, 1.4]),
        ]
    )


# ---------------------------------------------------------------------------
# Wiberg bond orders
# ---------------------------------------------------------------------------


class TestWibergBondOrders:
    """Wiberg bond index from density matrix."""

    def test_wiberg_water_sto3g(self):
        """Wiberg bond orders for H2O / STO-3G RHF."""
        from vibeqc.bond_analysis import wiberg_bond_orders

        mol = _h2o()
        basis = vq.BasisSet(mol, "sto-3g")
        result = vq.run_rhf(mol, basis)
        bo = wiberg_bond_orders(result, basis, mol)

        assert bo.shape == (3, 3)
        assert bo[0, 1] > 0.3
        assert bo[0, 2] > 0.3
        assert bo[1, 2] < 0.15
        assert np.all(np.diag(bo) >= 0)

    def test_wiberg_symmetric(self):
        """Wiberg matrix must be symmetric."""
        from vibeqc.bond_analysis import wiberg_bond_orders

        mol = _h2()
        basis = vq.BasisSet(mol, "sto-3g")
        result = vq.run_rhf(mol, basis)
        bo = wiberg_bond_orders(result, basis, mol)

        assert bo.shape == (2, 2)
        assert abs(bo[0, 1] - bo[1, 0]) < 1e-14

    def test_wiberg_uhf(self):
        """Wiberg for open-shell (UHF) system."""
        from vibeqc.bond_analysis import wiberg_bond_orders

        mol = vq.Molecule(
            [vq.Atom(8, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.832])],
            charge=0,
            multiplicity=2,
        )
        basis = vq.BasisSet(mol, "sto-3g")
        result = vq.run_uhf(mol, basis)
        bo = wiberg_bond_orders(result, basis, mol)

        assert bo.shape == (2, 2)
        assert bo[0, 1] > 0.2


# ---------------------------------------------------------------------------
# Bond order summary
# ---------------------------------------------------------------------------


class TestBondOrderSummary:
    def test_summary_all_metrics(self):
        """bond_order_summary computes Mayer + Wiberg without error."""
        from vibeqc.bond_analysis import bond_order_summary

        mol = _co()
        basis = vq.BasisSet(mol, "sto-3g")
        result = vq.run_rhf(mol, basis)

        summary = bond_order_summary(result, basis, mol)
        assert summary.mayer is not None
        assert summary.wiberg is not None
        assert summary.mayer.shape == (2, 2)
        assert summary.wiberg.shape == (2, 2)

    def test_top_bonds(self):
        """Top bonds listing returns ordered entries."""
        from vibeqc.bond_analysis import bond_order_summary

        mol = _co()
        basis = vq.BasisSet(mol, "sto-3g")
        result = vq.run_rhf(mol, basis)

        summary = bond_order_summary(result, basis, mol)
        top = summary.top_bonds(metric="mayer", molecule=mol, threshold=0.05)
        assert len(top) >= 1
        assert top[0][0] == 0
        assert top[0][1] == 1
        assert top[0][2] > 0.5


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------


class TestEDA:
    """Energy Decomposition Analysis."""

    def test_eda_lmo_basic(self):
        """LMO-EDA returns physically plausible components."""
        from vibeqc.eda import eda_lmo

        n_ao = 4
        S = np.eye(n_ao)
        P1 = np.diag([1.0, 1.0, 0.0, 0.0])
        P2 = np.diag([0.0, 0.0, 1.0, 1.0])
        F1 = np.diag([-0.5, -0.5, 0.0, 0.0])
        F2 = np.diag([0.0, 0.0, -0.5, -0.5])

        result = eda_lmo(
            e_total=-1.8,
            e_frag1=-1.0,
            e_frag2=-1.0,
            fock_total=np.diag([-0.6, -0.6, -0.4, -0.4]),
            fock_frag1=F1,
            fock_frag2=F2,
            density_total=P1 + P2,
            density_frag1=P1,
            density_frag2=P2,
            overlap=S,
        )
        assert abs(result.e_int - 0.2) < 0.5
        assert abs(result.e_total - (-1.8)) < 1e-10
        assert result.method == "lmo"

    def test_eda_summary(self):
        """EDA summary string is non-empty."""
        from vibeqc.eda import eda_lmo

        S = np.eye(2)
        P1 = np.diag([1.0, 0.0])
        P2 = np.diag([0.0, 1.0])

        result = eda_lmo(
            e_total=-0.9,
            e_frag1=-0.5,
            e_frag2=-0.5,
            fock_total=np.diag([-0.5, -0.5]),
            fock_frag1=np.diag([-0.5, 0.0]),
            fock_frag2=np.diag([0.0, -0.5]),
            density_total=P1 + P2,
            density_frag1=P1,
            density_frag2=P2,
            overlap=S,
        )
        s = result.summary()
        assert "Electrostatic" in s
        assert "Exchange" in s
        assert "Polarisation" in s

    def test_eda_morokuma(self):
        """Morokuma EDA wraps LMO-EDA."""
        from vibeqc.eda import eda_morokuma

        S = np.eye(2)
        P1 = np.diag([1.0, 0.0])
        P2 = np.diag([0.0, 1.0])

        result = eda_morokuma(
            e_total=-0.9,
            e_frag1=-0.5,
            e_frag2=-0.5,
            fock_total=np.diag([-0.5, -0.5]),
            fock_frag1=np.diag([-0.5, 0.0]),
            fock_frag2=np.diag([0.0, -0.5]),
            density_total=P1 + P2,
            density_frag1=P1,
            density_frag2=P2,
            overlap=S,
        )
        assert result.method == "morokuma"
        assert "morokuma_components" in result.extra


# ---------------------------------------------------------------------------
# Entanglement
# ---------------------------------------------------------------------------


class TestEntanglement:
    """Orbital entanglement measures."""

    def test_single_orbital_entropy_closed_shell(self):
        """Single-orbital entropy for fully occupied / empty orbitals."""
        from vibeqc.entanglement import single_orbital_entropy

        rho_occ = np.diag([0.0, 0.0, 0.0, 1.0])
        s = single_orbital_entropy(np.array([rho_occ]))
        assert s[0] == 0.0

        rho_empty = np.diag([1.0, 0.0, 0.0, 0.0])
        s = single_orbital_entropy(np.array([rho_empty]))
        assert s[0] == 0.0

        # Pure one-/two-orbital RDMs must not acquire entropy from a
        # probability floor, especially before bond-order normalisation.
        from vibeqc.entanglement import entanglement_bond_order, mutual_information

        rho = np.array([rho_empty, rho_occ])
        rho2 = np.zeros((2, 2, 16, 16))
        rho2[0, 1] = np.kron(rho_empty, rho_occ)
        for two_orbital_densities in (None, rho2):
            mi = mutual_information(rho, two_orbital_densities)
            np.testing.assert_array_equal(mi, 0.0)
            np.testing.assert_array_equal(entanglement_bond_order(mi), 0.0)

    def test_entanglement_from_density(self):
        """Entanglement from single-determinant density."""
        from vibeqc.entanglement import entanglement_from_density

        n_ao = 4
        P = np.zeros((n_ao, n_ao), dtype=np.float64)
        P[0, 0] = 2.0
        P[1, 1] = 2.0
        S = np.eye(n_ao)

        result = entanglement_from_density(P, S)
        assert result.n_orbitals == n_ao
        np.testing.assert_array_equal(result.single_orbital_entropies, 0.0)
        np.testing.assert_array_equal(result.mutual_information, 0.0)
        np.testing.assert_array_equal(result.entanglement_bond_orders, 0.0)
        assert result.total_information == 0.0

    def test_entanglement_paired_orbital_rdms(self):
        """An entangled empty/pair superposition has known RDM entropies."""
        from vibeqc.entanglement import (
            entanglement_bond_order,
            mutual_information,
            single_orbital_entropy,
        )

        rho = np.array([np.diag([0.5, 0.0, 0.0, 0.5])] * 2)
        psi = np.zeros(16)
        psi[[3, 12]] = 1.0 / np.sqrt(2.0)
        rho2 = np.zeros((2, 2, 16, 16))
        rho2[0, 1] = np.outer(psi, psi)
        np.testing.assert_allclose(
            single_orbital_entropy(rho), np.log(2.0), atol=1e-14
        )
        mi = mutual_information(rho, rho2)
        np.testing.assert_allclose(mi, 2.0 * np.log(2.0), atol=1e-14)
        np.testing.assert_allclose(entanglement_bond_order(mi), 1.0, atol=1e-14)

    def test_entanglement_rhf_water(self):
        """A nonorthogonal RHF AO density has only 0/2 occupations."""
        from vibeqc.entanglement import entanglement_from_density

        mol = _h2o()
        basis = vq.BasisSet(mol, "sto-3g")
        rhf = vq.run_rhf(mol, basis)
        assert rhf.converged
        S = np.asarray(vq.compute_overlap(basis))
        P = np.asarray(rhf.density)
        # Independent occupation oracle: PS is similar to the orthogonal
        # density, without using the production square-root transform.
        occ = np.sort(np.linalg.eigvals(P @ S).real)
        np.testing.assert_allclose(occ, [0, 0, 2, 2, 2, 2, 2], atol=1e-12)
        result = entanglement_from_density(P, S)
        np.testing.assert_array_equal(result.single_orbital_entropies, 0.0)
        np.testing.assert_array_equal(result.mutual_information, 0.0)
        np.testing.assert_array_equal(result.entanglement_bond_orders, 0.0)
        assert result.total_information == 0.0

    @pytest.mark.parametrize("n_active", [None, 2])
    def test_entanglement_fractional_occupations(self, n_active):
        """All density-based measures use the same per-spin occupations."""
        from vibeqc.entanglement import (
            entanglement_bond_order,
            entanglement_from_density,
            single_orbital_entropy,
        )

        # S = A.T A and C = A^-1 give C.T S C = I, so these are known
        # natural occupations in a nonorthogonal basis without an eigh oracle.
        A = np.array([[1.2, 0.3, -0.1], [0.0, 0.9, 0.2], [0.0, 0.0, 1.1]])
        C = np.linalg.inv(A)
        occ = np.array([0.4, 0.8, 1.6])
        P = (C * occ) @ C.T
        result = entanglement_from_density(P, A.T @ A, n_active=n_active)
        n = occ[:n_active] / 2.0
        expected_s = -n * np.log(n) - (1.0 - n) * np.log1p(-n)
        expected_mi = 4.0 * np.outer(n * (1.0 - n), n * (1.0 - n))
        np.fill_diagonal(expected_mi, 0.0)
        np.testing.assert_allclose(
            result.single_orbital_entropies, expected_s, atol=1e-14
        )
        np.testing.assert_allclose(result.mutual_information, expected_mi, atol=1e-14)
        np.testing.assert_allclose(
            result.entanglement_bond_orders,
            entanglement_bond_order(expected_mi),
            atol=1e-14,
        )
        assert result.total_information == pytest.approx(expected_s.sum(), abs=1e-14)
        # A paired empty/doubly-occupied orbital has this same binary entropy.
        rho = np.array([np.diag([1.0 - p, 0.0, 0.0, p]) for p in n])
        np.testing.assert_allclose(single_orbital_entropy(rho), expected_s, atol=1e-14)

    def test_entanglement_casci_stretched_h2(self):
        """A correlated CAS(2,2) density retains positive entropy and MI."""
        from vibeqc.entanglement import entanglement_from_density
        from vibeqc.solvers import build_hamiltonian_mo, casci, get_hf_orbital_provider
        from vibeqc.solvers._rdm import make_rdm1

        mol = vq.Molecule(
            [
                vq.Atom(1, [0.0, 0.0, 0.0]),
                vq.Atom(1, [0.0, 0.0, 4.0]),
            ]
        )
        basis = vq.BasisSet(mol, "sto-3g")
        C = get_hf_orbital_provider(mol, basis)
        H = build_hamiltonian_mo(mol, basis, C)
        cas = casci(
            H.h1e, H.h2e, n_active_elec=2, n_active_orb=2,
            n_core=0, nuclear_repulsion=H.nuclear_repulsion,
        )
        D = make_rdm1(cas.ci_coeffs, cas.determinants, 2)
        n = np.linalg.eigvalsh(D) / 2.0
        assert np.all((n > 0.1) & (n < 0.9))
        expected_s = -n * np.log(n) - (1.0 - n) * np.log1p(-n)
        result = entanglement_from_density(C @ D @ C.T, vq.compute_overlap(basis))
        np.testing.assert_allclose(
            result.single_orbital_entropies, expected_s, atol=1e-12
        )
        assert result.total_information == pytest.approx(expected_s.sum(), abs=1e-12)
        assert result.total_information > 1.0
        assert result.mutual_information[0, 1] == pytest.approx(
            4.0 * n[0] * (1.0 - n[0]) * n[1] * (1.0 - n[1]), abs=1e-12
        )
        np.testing.assert_allclose(result.entanglement_bond_orders, [[0, 1], [1, 0]])

    def test_correlation_clusters_empty(self):
        """No clusters when MI is zero."""
        from vibeqc.entanglement import correlation_clusters

        mi = np.zeros((4, 4), dtype=np.float64)
        clusters = correlation_clusters(mi, threshold=0.1)
        assert clusters == []

    def test_correlation_clusters_connected(self):
        """Find connected clusters above threshold."""
        from vibeqc.entanglement import correlation_clusters

        mi = np.zeros((4, 4), dtype=np.float64)
        mi[0, 1] = 0.5
        mi[1, 0] = 0.5
        mi[1, 2] = 0.5
        mi[2, 1] = 0.5
        clusters = correlation_clusters(mi, threshold=0.1)
        assert len(clusters) == 1
        assert clusters[0] == {0, 1, 2}


# ---------------------------------------------------------------------------
# NBO
# ---------------------------------------------------------------------------


class TestNBO:
    """Natural Bond Orbital analysis."""

    def test_npa_charges_fail_closed_without_full_naos(self):
        """A Löwdin population must not be returned under the NPA label."""
        from vibeqc._vibeqc_core import compute_overlap
        from vibeqc.nbo import npa_charges

        mol = _h2o()
        basis = vq.BasisSet(mol, "sto-3g")
        result = vq.run_rhf(mol, basis)

        S = np.asarray(compute_overlap(basis))
        P = np.asarray(result.density)
        with pytest.raises(NotImplementedError, match="Natural Atomic Orbital"):
            npa_charges(P, S, basis, mol)

    def test_nbo_search_water(self):
        """NBO search finds bonding and lone-pair orbitals."""
        from vibeqc._vibeqc_core import compute_overlap
        from vibeqc.nbo import nbo_search

        mol = _h2o()
        basis = vq.BasisSet(mol, "sto-3g")
        result = vq.run_rhf(mol, basis)

        S = np.asarray(compute_overlap(basis))
        P = np.asarray(result.density)
        nbo_result = nbo_search(P, S, basis, mol)

        assert len(nbo_result.nbo_orbitals) > 0
        assert nbo_result.npa_charges.size == 0
        types = [n["type"] for n in nbo_result.nbo_orbitals]
        assert "BD" in types or "LP" in types

    def test_nbo_summary_nonempty(self):
        """NBO summary string is non-empty."""
        from vibeqc._vibeqc_core import compute_overlap
        from vibeqc.nbo import nbo_search

        mol = _h2o()
        basis = vq.BasisSet(mol, "sto-3g")
        result = vq.run_rhf(mol, basis)

        S = np.asarray(compute_overlap(basis))
        P = np.asarray(result.density)
        nbo_result = nbo_search(P, S, basis, mol)
        s = nbo_result.summary()
        assert "NBO" in s
        assert "NPA Atomic Charges" not in s
        assert len(s) > 50


# ---------------------------------------------------------------------------
# Periodic bond analysis
# ---------------------------------------------------------------------------


class TestPeriodicBondAnalysis:
    """Periodic bond analysis: Wiberg for Gamma-point SCF."""

    def test_periodic_wiberg_trivial(self):
        """periodic_wiberg_bond_orders delegates to wiberg_bond_orders."""
        from vibeqc.bond_analysis import periodic_wiberg_bond_orders

        mol = _h2()
        basis = vq.BasisSet(mol, "sto-3g")
        result = vq.run_rhf(mol, basis)

        bo = periodic_wiberg_bond_orders(result, basis, mol)
        assert bo.shape == (2, 2)
        assert bo[0, 1] > 0.1
