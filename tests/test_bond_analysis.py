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
        assert abs(s[0]) < 1e-12

        rho_empty = np.diag([1.0, 0.0, 0.0, 0.0])
        s = single_orbital_entropy(np.array([rho_empty]))
        assert abs(s[0]) < 1e-12

    def test_entanglement_from_density(self):
        """Entanglement from single-determinant density."""
        from vibeqc.entanglement import entanglement_from_density

        n_ao = 4
        P = np.zeros((n_ao, n_ao), dtype=np.float64)
        P[0, 0] = 1.0
        P[1, 1] = 1.0
        S = np.eye(n_ao)

        result = entanglement_from_density(P, S)
        assert result.n_orbitals == n_ao
        assert abs(result.single_orbital_entropies[0]) < 1e-12
        assert abs(result.single_orbital_entropies[1]) < 1e-12

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
