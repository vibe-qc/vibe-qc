"""End-to-end tests for v2RDM and transcorrelated solvers."""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc import Atom, BasisSet, Molecule
from vibeqc.solvers import (
    Hamiltonian,
    SelectedCIOptions,
    TranscorrelatedOptions,
    V2RDMOptions,
    build_hamiltonian_matrix_unrestricted,
    build_hamiltonian_mo,
    build_transcorrelated_hamiltonian,
    generate_determinants,
    get_hf_orbital_provider,
    solve_selected_ci,
    solve_v2rdm,
)
from vibeqc.solvers._v2rdm import V2RDMSolver


@pytest.fixture
def h2_sto3g_ham() -> Hamiltonian:
    mol = Molecule(
        [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])],
        charge=0,
        multiplicity=1,
    )
    basis = BasisSet(mol, "sto-3g")
    C = get_hf_orbital_provider(mol, basis)
    return build_hamiltonian_mo(mol, basis, C)


class TestV2RDM:
    def test_h2_v2rdm_matches_fci(self, h2_sto3g_ham):
        """Exact N-representable v2RDM should match H2/STO-3G FCI."""
        ham = h2_sto3g_ham
        nalpha = (ham.nelec + ham.ms2) // 2
        nbeta = (ham.nelec - ham.ms2) // 2
        determinants = generate_determinants(ham.norb, nalpha, nbeta)
        H_fci = build_hamiltonian_matrix_unrestricted(
            determinants, ham.h1e, ham.h2e
        )
        e_fci = float(np.linalg.eigvalsh(H_fci)[0] + ham.nuclear_repulsion)

        opts = V2RDMOptions(
            constraints="p",
            outer_max_iter=1,
            conv_tol_primal=1e-12,
            verbose=0,
        )
        result = solve_v2rdm(ham, opts)

        assert result.converged
        assert result.energy == pytest.approx(e_fci, abs=1e-9)
        assert result.constraint_residual is not None
        assert result.constraint_residual < 1e-10

    def test_h2_energy_reasonable(self, h2_sto3g_ham):
        """v2RDM on H2 should give a finite, negative energy."""
        opts = V2RDMOptions(
            outer_max_iter=200,
            mu=10.0,
            mu_factor=1.2,
            conv_tol_primal=1e-3,
            verbose=0,
        )
        result = solve_v2rdm(h2_sto3g_ham, opts)
        assert np.isfinite(result.energy)
        assert result.energy < 0.0
        assert result.rdm1 is not None

    def test_rdm1_trace(self, h2_sto3g_ham):
        """Trace constraint residual should be small."""
        opts = V2RDMOptions(
            outer_max_iter=200,
            mu=10.0,
            conv_tol_primal=1e-3,
            verbose=0,
        )
        result = solve_v2rdm(h2_sto3g_ham, opts)
        if result.constraint_residual is not None:
            assert result.constraint_residual < 0.1

    def test_result_structure(self, h2_sto3g_ham):
        opts = V2RDMOptions(
            outer_max_iter=5,
            conv_tol_primal=1e-3,
            verbose=0,
        )
        result = solve_v2rdm(h2_sto3g_ham, opts)
        assert result.method.startswith("v2rdm")
        assert result.rdm1 is not None
        assert result.rdm2 is not None
        assert result.constraint_residual is not None

    @pytest.mark.parametrize("constraints", ["q", "g", "pqg"])
    def test_qg_constraints_raise_instead_of_silent_non_enforcement(
        self, constraints
    ):
        """Q/G requests must not return a result unless those blocks are enforced."""
        ham = Hamiltonian(
            h1e=np.zeros((2, 2)),
            h2e=np.zeros((2, 2, 2, 2)),
            nuclear_repulsion=0.0,
            norb=2,
            nelec=2,
            ms2=0,
        )
        opts = V2RDMOptions(
            constraints=constraints,
            outer_max_iter=1,
            conv_tol_primal=1e-3,
            verbose=0,
        )

        with pytest.raises(NotImplementedError, match="Q/G constraint feedback"):
            solve_v2rdm(ham, opts)

    def test_residual_depends_on_requested_q_and_g_blocks(self):
        """Residual accounting must change when Q/G constraints are requested."""
        solver = V2RDMSolver()
        rdm1 = np.diag([3.0, -1.0])
        rdm2 = np.zeros((2, 2, 2, 2))

        p_only = solver._compute_residual(rdm2, rdm1, 2, "p")
        with_q = solver._compute_residual(rdm2, rdm1, 2, "pq")
        with_g = solver._compute_residual(rdm2, rdm1, 2, "pg")
        with_qg = solver._compute_residual(rdm2, rdm1, 2, "pqg")

        assert p_only == pytest.approx(0.0)
        assert with_q > p_only
        assert with_g > p_only
        assert with_qg == pytest.approx(with_q + with_g)


class TestTranscorrelated:
    def test_simple_gaussian(self, h2_sto3g_ham):
        """Simple Gaussian correlator produces a modified Hamiltonian."""
        tc_opts = TranscorrelatedOptions(
            form="simple_gaussian",
            gamma=0.3,
            no2b=True,
            symmetrize=True,
            report_diagnostics=True,
        )
        H_tc = build_transcorrelated_hamiltonian(h2_sto3g_ham, tc_opts)
        assert H_tc.norb == h2_sto3g_ham.norb
        assert H_tc.nelec == h2_sto3g_ham.nelec

    def test_tc_plus_ci(self, h2_sto3g_ham):
        """Transcorrelated + Selected-CI workflow."""
        tc_opts = TranscorrelatedOptions(
            form="simple_gaussian",
            gamma=0.2,
            no2b=True,
            symmetrize=True,
            report_diagnostics=True,
        )
        H_tc = build_transcorrelated_hamiltonian(h2_sto3g_ham, tc_opts)

        ci_opts = SelectedCIOptions(
            target_size=10, max_iter=5, conv_tol_energy=1e-6, verbose=0
        )
        result = solve_selected_ci(H_tc, ci_opts)
        assert result.converged
        assert np.isfinite(result.energy)

    def test_exponential_jastrow(self, h2_sto3g_ham):
        """Exponential Jastrow correlator."""
        tc_opts = TranscorrelatedOptions(
            form="exponential_jastrow",
            gamma=0.3,
            no2b=True,
            symmetrize=True,
            report_diagnostics=True,
        )
        H_tc = build_transcorrelated_hamiltonian(h2_sto3g_ham, tc_opts)
        assert H_tc.norb == h2_sto3g_ham.norb
        assert not np.allclose(H_tc.h2e, h2_sto3g_ham.h2e)

    def test_custom_form(self, h2_sto3g_ham):
        """Custom form passes through unchanged."""
        tc_opts = TranscorrelatedOptions(form="custom", report_diagnostics=False)
        H_tc = build_transcorrelated_hamiltonian(h2_sto3g_ham, tc_opts)
        np.testing.assert_allclose(H_tc.h1e, h2_sto3g_ham.h1e)
        np.testing.assert_allclose(H_tc.h2e, h2_sto3g_ham.h2e)
