"""Selected-CI convergence: verify systematic improvability toward FCI."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import eigh
from vibeqc import Atom, Molecule
from vibeqc._vibeqc_core import BasisSet
from vibeqc.solvers import (
    SelectedCIOptions,
    build_hamiltonian_matrix_unrestricted,
    build_hamiltonian_mo,
    generate_determinants,
    get_hf_orbital_provider,
    solve_selected_ci,
)


def _fci_energy(ham):
    """Compute exact FCI energy."""
    nalpha = (ham.nelec + ham.ms2) // 2
    nbeta = (ham.nelec - ham.ms2) // 2
    dets = generate_determinants(ham.norb, nalpha, nbeta)
    H = build_hamiltonian_matrix_unrestricted(dets, ham.h1e, ham.h2e)
    evals, _ = eigh(H)
    return evals[0] + ham.nuclear_repulsion


def _selected_ci(ham, target_size, spin_restricted=True):
    opts = SelectedCIOptions(
        target_size=target_size,
        max_iter=30,
        pt2_threshold=1e-8,
        spin_restricted=spin_restricted,
        do_pt2_correction=True,
        verbose=0,
    )
    return solve_selected_ci(ham, opts)


@pytest.fixture
def h2_ham():
    mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])])
    b = BasisSet(mol, "sto-3g")
    return build_hamiltonian_mo(mol, b, get_hf_orbital_provider(mol, b))


@pytest.fixture
def lih_ham():
    mol = Molecule([Atom(3, [0, 0, 0]), Atom(1, [0, 0, 3.015])])
    b = BasisSet(mol, "sto-3g")
    return build_hamiltonian_mo(mol, b, get_hf_orbital_provider(mol, b))


class TestSelectedCIConvergence:
    def test_h2_monotonic_energy(self, h2_ham):
        """Energy should decrease monotonically as target_size grows (variational)."""
        e_fci = _fci_energy(h2_ham)
        prev = float("inf")
        for t in [1, 3, 5, 10, 50]:
            # Use variational only (no PT2) for monotonic comparison
            r = solve_selected_ci(
                h2_ham,
                SelectedCIOptions(
                    target_size=t,
                    max_iter=30,
                    spin_restricted=False,
                    do_pt2_correction=False,
                    verbose=0,
                ),
            )
            e_var = r.energy  # variational energy
            assert e_var <= prev + 1e-10
            prev = e_var
        assert abs(r.energy - e_fci) < 1e-8

    def test_lih_approaches_fci(self, lih_ham):
        """LiH unrestricted Selected-CI should approach FCI within 1 mHa."""
        e_fci = _fci_energy(lih_ham)
        r = _selected_ci(lih_ham, 300, spin_restricted=False)
        assert r.converged
        assert len(r.ci_labels) >= 50
        assert abs(r.energy - e_fci) < 1e-3

    def test_lih_pt2_improves_energy(self, lih_ham):
        """PT2 correction should lower the variational energy."""
        r_var = _selected_ci(lih_ham, 100, spin_restricted=False)
        r_var_no_pt2 = solve_selected_ci(
            lih_ham,
            SelectedCIOptions(
                target_size=100,
                max_iter=30,
                spin_restricted=False,
                do_pt2_correction=False,
                verbose=0,
            ),
        )
        e_var = r_var_no_pt2.energy
        e_pt2 = r_var.energy
        # PT2 should never raise the energy (for ground state)
        assert e_pt2 <= e_var + 1e-10

    def test_restricted_vs_unrestricted(self, h2_ham):
        """Unrestricted should give lower energy than restricted for same target."""
        r_r = _selected_ci(h2_ham, 10, spin_restricted=True)
        r_u = _selected_ci(h2_ham, 10, spin_restricted=False)
        # Unrestricted includes more determinants → lower (or equal) energy
        assert r_u.energy <= r_r.energy + 1e-10
